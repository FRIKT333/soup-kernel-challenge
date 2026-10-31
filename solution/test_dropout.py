import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')


@pytest.mark.parametrize('shape',[(2,17,129),(4,512,1536)])
@pytest.mark.parametrize('probability,training',[(0.05,True),(0.0,True),(1.0,True),(0.05,False)])
def test_packed_dropout_matches_native_full_fp32_values_gradients_rng(shape,probability,training):
    from solution.dropout import packed_dropout
    torch.manual_seed(728)
    full = torch.randn(shape,device='cuda',dtype=torch.float16)
    active = torch.arange(shape[1],device='cuda')[None,:] < torch.tensor(
        [shape[1]-i*3 for i in range(shape[0])],device='cuda')[:,None]
    indices = active.reshape(-1).nonzero(as_tuple=True)[0]
    state = (shape[:2],indices,None,None,{})
    # Run two sequential masks with one cached ones buffer and verify complete
    # generator state after each independent native dropout call.
    initial_rng = torch.cuda.get_rng_state()
    expected = []
    for _ in range(2):
        x = full.clone().requires_grad_()
        out = torch.nn.functional.dropout(x.float(),p=probability,training=training).half()
        selected = out.reshape(-1,shape[-1]).index_select(0,indices)
        rng = torch.cuda.get_rng_state()
        grad = torch.arange(selected.numel(),device='cuda',dtype=torch.float32).reshape_as(selected)
        grad = (grad.remainder(31)/31).half()
        selected.backward(grad)
        expected.append((selected.detach(),x.grad.reshape(-1,shape[-1])[indices],rng,grad))
    torch.cuda.set_rng_state(initial_rng)
    for out, gradient, rng, incoming in expected:
        x = full.reshape(-1,shape[-1])[indices].clone().requires_grad_()
        actual = packed_dropout(x,state,probability,training)
        assert torch.equal(torch.cuda.get_rng_state(),rng)
        actual.backward(incoming)
        torch.testing.assert_close(actual,out,rtol=0,atol=0)
        torch.testing.assert_close(x.grad,gradient,rtol=0,atol=0)
