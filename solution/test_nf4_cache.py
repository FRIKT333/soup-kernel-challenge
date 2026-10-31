import copy

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("out_features", [256, 1536])
def test_cached_nf4_matches_bnb_and_invalidates(out_features):
    import bitsandbytes as bnb
    from solution.nf4_cache import install_nf4_cache, patch_nf4_cache

    torch.manual_seed(37)
    ref = bnb.nn.Linear4bit(1536, out_features, bias=True, compute_dtype=torch.float16,
                            quant_type="nf4", compress_statistics=True).half().cuda()
    ref.requires_grad_(False)
    model = copy.deepcopy(ref)
    install_nf4_cache(model)
    patch_nf4_cache()
    x = torch.randn((2, 33, 1536), device="cuda", requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected = ref(x)
    actual = model(y)
    g = torch.randn_like(expected)
    expected.backward(g)
    actual.backward(g)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(y.grad, x.grad, rtol=3e-3, atol=3e-3)
    state = model.weight.quant_state
    first = state._frozen_dense_cache[1]
    model(y)
    assert state._frozen_dense_cache[1] is first
    with torch.no_grad():
        model.weight.quant_state.absmax.add_(0)  # version change must invalidate
    model(y)
    assert state._frozen_dense_cache[1] is not first
