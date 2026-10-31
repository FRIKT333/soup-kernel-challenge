import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("heads", [2, 12])
@pytest.mark.parametrize("tokens", [33, 2048])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_packed_rotary_and_backward(heads, tokens, dtype):
    from solution.attention import PackedRotary

    torch.manual_seed(83)
    x = torch.randn((tokens, heads, 128), device="cuda", dtype=dtype, requires_grad=True)
    copy = x.detach().clone().requires_grad_()
    cos = torch.randn((tokens, 128), device="cuda", dtype=dtype)
    sin = torch.randn_like(cos)
    rotated = torch.cat((-x[..., 64:], x[..., :64]), dim=-1)
    expected = x * cos[:, None, :] + rotated * sin[:, None, :]
    actual = PackedRotary.apply(copy, cos, sin)
    grad = torch.randn_like(expected)
    expected.backward(grad)
    actual.backward(grad)
    tolerance = 0 if dtype == torch.float16 else 2e-6
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(copy.grad, x.grad, rtol=tolerance, atol=tolerance)
