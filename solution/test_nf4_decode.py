import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("shape", [(256, 1536), (1536, 1536), (8960, 1536)])
def test_double_nf4_decode_is_bit_exact(shape):
    import bitsandbytes as bnb
    from solution.nf4_decode import decode_nf4
    torch.manual_seed(19)
    weight = torch.randn(shape, device="cuda", dtype=torch.float16)
    packed, state = bnb.functional.quantize_4bit(weight, quant_type="nf4", compress_statistics=True)
    expected = bnb.functional.dequantize_4bit(packed, state)
    actual = decode_nf4(packed, state)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fused_nf4_linear_backward_matches_bitsandbytes():
    import bitsandbytes as bnb
    from solution.nf4_cache import FusedNF4Linear
    torch.manual_seed(27)
    layer = bnb.nn.Linear4bit(1536, 256, bias=True, compute_dtype=torch.float16,
        quant_type="nf4", compress_statistics=True).half().cuda().requires_grad_(False)
    x = torch.randn(29, 1536, device="cuda", dtype=torch.float16, requires_grad=True)
    ref = x.detach().clone().requires_grad_()
    expected = layer(ref)
    actual = FusedNF4Linear.apply(x, layer.weight, layer.weight.quant_state, layer.bias)
    grad = torch.randn_like(actual)
    expected.backward(grad)
    actual.backward(grad)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x.grad, ref.grad, rtol=2e-3, atol=2e-3)


def test_decoded_cache_budget_is_deterministic_and_clears_uncached_values():
    import bitsandbytes as bnb
    from solution.nf4_cache import install_nf4_cache
    from solution.schedule import choose_nf4_cache
    small = bnb.nn.Linear4bit(128,128,bias=False,compute_dtype=torch.float16,
        quant_type="nf4",compress_statistics=True).half().cuda().requires_grad_(False)
    large = bnb.nn.Linear4bit(128,256,bias=False,compute_dtype=torch.float16,
        quant_type="nf4",compress_statistics=True).half().cuda().requires_grad_(False)
    model = torch.nn.ModuleDict({'large':large,'small':small})
    install_nf4_cache(model)
    state = large.weight.quant_state
    state._frozen_dense_cache = ('old',torch.empty(1,device="cuda"))
    budget = 128*128*2
    assert choose_nf4_cache(model,budget) == budget
    assert small.weight.quant_state._permit_frozen_cache
    assert not state._permit_frozen_cache
    assert state._permit_fused_decode
    assert not hasattr(state,'_frozen_dense_cache')
    assert choose_nf4_cache(model,budget) == budget
    assert choose_nf4_cache(model,0) == 0
    assert not small.weight.quant_state._permit_frozen_cache
