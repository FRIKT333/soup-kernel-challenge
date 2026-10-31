import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(2, 33, 1536), (4, 512, 1536)])
def test_frozen_rmsnorm(dtype, shape):
    from solution.rmsnorm import FrozenRMSNorm

    torch.manual_seed(19)
    ref = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    actual_x = ref.detach().clone().requires_grad_()
    weight = torch.randn((1536,), device="cuda", dtype=torch.float32)
    x32 = ref.float()
    expected = weight * (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + 1e-6)).to(dtype)
    actual = FrozenRMSNorm.apply(actual_x, weight, 1e-6)
    grad = torch.randn_like(expected)
    actual.backward(grad)
    expected.backward(grad)
    assert actual.dtype == expected.dtype
    tol = 2e-3 if dtype == torch.float16 else 3e-6
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    torch.testing.assert_close(actual_x.grad, ref.grad, rtol=tol, atol=tol)


def test_rmsnorm_patch_falls_back_for_trainable_weights():
    from solution.rmsnorm import patch_qwen2_rmsnorm
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    patch_qwen2_rmsnorm()
    first = Qwen2RMSNorm.forward
    patch_qwen2_rmsnorm()
    assert first is Qwen2RMSNorm.forward
    layer = Qwen2RMSNorm(1536)
    x = torch.randn((2, 3, 1536), requires_grad=True)
    layer(x).sum().backward()
    assert layer.weight.grad is not None and x.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", ["full", "packed", "both"])
def test_packed_rmsnorm_outputs_and_combined_backward(mode):
    from solution.rmsnorm import PackedRMSNorm

    torch.manual_seed(103)
    x = torch.randn((2, 33, 1536), device="cuda", requires_grad=True)
    copy = x.detach().clone().requires_grad_()
    w = torch.randn((1536,), device="cuda")
    mask = torch.rand((66,), device="cuda") > 0.35
    indices = mask.nonzero(as_tuple=True)[0]
    inverse = torch.where(mask, mask.cumsum(0, dtype=torch.int32) - 1, -1)
    expected = w * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6))
    expected_packed = expected.reshape(-1, 1536).index_select(0, indices)
    full, packed = PackedRMSNorm.apply(copy, w, 1e-6, indices, inverse)
    g_full, g_packed = torch.randn_like(full), torch.randn_like(packed)
    if mode == "full":
        full.backward(g_full)
        expected.backward(g_full)
    elif mode == "packed":
        packed.backward(g_packed)
        expected_packed.backward(g_packed)
    else:
        torch.autograd.backward((full, packed), (g_full, g_packed))
        torch.autograd.backward((expected, expected_packed), (g_full, g_packed))
    torch.testing.assert_close(full, expected, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(packed, expected_packed, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(copy.grad, x.grad, rtol=5e-6, atol=5e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_norm_hook_supplies_compact_output_without_extra_copy(monkeypatch):
    from solution import compact
    from solution.rmsnorm import patch_qwen2_rmsnorm
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    patch_qwen2_rmsnorm()
    layer = Qwen2RMSNorm(1536).cuda().requires_grad_(False)
    x = torch.randn((2, 33, 1536), device="cuda", requires_grad=True)
    mask = torch.rand((66,), device="cuda") > 0.3
    indices = mask.nonzero(as_tuple=True)[0]
    inverse = torch.where(mask, mask.cumsum(0, dtype=torch.int32) - 1, -1)
    state = ((2, 33), indices, inverse)
    token = compact._tokens.set(state)
    try:
        output = layer(x)
        assert compact._pack(output, state) is output._packed_normalization[1]
    finally:
        compact._tokens.reset(token)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", ["full", "packed", "both"])
def test_half_packed_norm_preserves_output_and_gradient_rounding(mode):
    from solution.rmsnorm import PackedRMSNorm
    from solution.precision import half_activations
    torch.manual_seed(417)
    original = torch.randn((2, 33, 1536), device="cuda", dtype=torch.float16)
    ref = original.clone().requires_grad_()
    got = original.clone().requires_grad_()
    w = torch.randn(1536, device="cuda", dtype=torch.float32)
    mask = torch.arange(66, device="cuda") % 3 != 0
    indices = mask.nonzero(as_tuple=True)[0]
    inverse = torch.where(mask, mask.cumsum(0, dtype=torch.int32)-1, -1)
    xf = ref.float()
    normalized = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True)+1e-6)).half()
    expected = (w * normalized).half()
    selected = expected.reshape(-1,1536).index_select(0,indices)
    token = half_activations.set(True)
    try:
        full, packed = PackedRMSNorm.apply(got,w,1e-6,indices,inverse)
    finally:
        half_activations.reset(token)
    gf, gp = torch.randn_like(full), torch.randn_like(packed)
    if mode == "full":
        full.backward(gf); expected.backward(gf)
    elif mode == "packed":
        packed.backward(gp); selected.backward(gp)
    else:
        torch.autograd.backward((full,packed),(gf,gp))
        torch.autograd.backward((expected,selected),(gf,gp))
    torch.testing.assert_close(full,expected,rtol=2e-3,atol=2e-3)
    torch.testing.assert_close(packed,selected,rtol=2e-3,atol=2e-3)
    torch.testing.assert_close(got.grad,ref.grad,rtol=3e-3,atol=3e-3)
    assert (got.grad.float()-ref.grad.float()).norm() <= ref.grad.float().norm()*1e-3
