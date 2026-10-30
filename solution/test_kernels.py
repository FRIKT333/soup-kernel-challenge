"""Correctness of your kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`. Add a test like this for every kernel you write.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_matches_pytorch(dtype, shape):
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (2, 64, 1536)])
def test_rmsnorm_matches_pytorch(dtype, shape):
    from solution.rmsnorm import fused_rmsnorm

    torch.manual_seed(42)
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(shape[-1], device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)
    eps = 1e-6

    # PyTorch reference
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    var = x_ref.float().pow(2).mean(-1, keepdim=True)
    out_ref = (x_ref.float() * torch.rsqrt(var + eps)) * w_ref.float()
    out_ref.backward(grad.float())
    ref_dx, ref_dw = x_ref.grad.float(), w_ref.grad.float()

    # Triton kernel
    out = fused_rmsnorm(x, w, eps)
    out.backward(grad)

    tol = 3e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(out.float(), out_ref, rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad.float(), ref_dx, rtol=tol, atol=tol)
    if w.grad is not None and ref_dw is not None:
        torch.testing.assert_close(w.grad.float(), ref_dw, rtol=tol, atol=tol)


@pytest.mark.parametrize("chunk", [64, 128])
def test_chunked_cross_entropy_matches_pytorch(chunk):
    from solution.cross_entropy import FusedChunkedHeadLossFunction

    torch.manual_seed(123)
    B, S, H, V = 2, 128, 64, 1024
    hidden = torch.randn((B, S, H), device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn((V, H), device="cuda", dtype=torch.float16)
    target = torch.randint(-1, V, (B, S), device="cuda", dtype=torch.long)
    target[target == -1] = -100

    denominator = float((target != -100).sum().clamp(min=1).item())

    # Reference PyTorch
    hidden_ref = hidden.detach().clone().requires_grad_(True)
    logits_ref = torch.matmul(hidden_ref, weight.t())
    loss_ref = torch.nn.functional.cross_entropy(
        logits_ref.view(-1, V).float(),
        target.view(-1),
        ignore_index=-100,
        reduction="sum",
    ) / denominator
    loss_ref.backward()
    ref_grad = hidden_ref.grad.clone()

    # Triton Fused Chunked
    loss_sol, entropy_sol, token_loss_sol, correct_sol = FusedChunkedHeadLossFunction.apply(
        hidden, weight, target, chunk, denominator, True
    )
    loss_sol.backward()

    torch.testing.assert_close(loss_sol, loss_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(hidden.grad, ref_grad, rtol=5e-3, atol=5e-3)
