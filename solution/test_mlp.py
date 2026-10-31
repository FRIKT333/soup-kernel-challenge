import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


def make_mlp(h=256, width=512):
    import bitsandbytes as bnb
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    model = Qwen2MLP(Qwen2Config(hidden_size=h,intermediate_size=width))
    for name, inp, out in [('gate_proj',h,width),('up_proj',h,width),('down_proj',width,h)]:
        setattr(model, name, bnb.nn.Linear4bit(inp,out,bias=False,compute_dtype=torch.float16,
            quant_type='nf4',compress_statistics=True).half().cuda().requires_grad_(False))
    return model


def test_three_matrix_decode_is_bit_exact():
    import bitsandbytes as bnb
    from solution.nf4_decode import decode_mlp
    model = make_mlp(1536, 8960)
    weights = tuple(getattr(model,n).weight for n in ('gate_proj','up_proj','down_proj'))
    states = tuple(w.quant_state for w in weights)
    result = decode_mlp(weights, states)
    size = 1536 * 8960
    for i, (w, s) in enumerate(zip(weights,states)):
        expected = bnb.functional.dequantize_4bit(w, s)
        torch.testing.assert_close(result[i*size:(i+1)*size].view(s.shape), expected, rtol=0, atol=0)


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
@pytest.mark.parametrize('rows',[19,129])
def test_joined_mlp_output_gradient_and_rng(dtype, rows):
    from solution.mlp import frozen_mlp, eligible
    from solution.swiglu import swiglu
    torch.manual_seed(43)
    model = make_mlp()
    x = torch.randn(rows,256,device='cuda',dtype=dtype,requires_grad=True)
    ref = x.detach().clone().requires_grad_()
    assert eligible(model,x)
    rng = torch.cuda.get_rng_state()
    with torch.autocast('cuda',dtype=torch.float16):
        expected = model.down_proj(swiglu(model.gate_proj(ref.half()),model.up_proj(ref.half()))).to(dtype)
        actual = frozen_mlp(model,x)
    assert torch.equal(torch.cuda.get_rng_state(),rng)
    grad = torch.randn_like(actual)
    expected.backward(grad)
    actual.backward(grad)
    assert torch.isfinite(actual).all() and torch.isfinite(x.grad).all()
    assert (actual-expected).float().norm() <= expected.float().norm()*.003 + 1e-6
    assert (x.grad-ref.grad).float().norm() <= ref.grad.float().norm()*.005 + 1e-6
    assert all(p.grad is None for p in model.parameters())


def test_fused_mlp_guards_preserve_other_consumers():
    from solution.mlp import eligible
    model = make_mlp()
    x = torch.randn(3,256,device='cuda',dtype=torch.float16)
    assert eligible(model,x)
    handle = model.up_proj.register_forward_hook(lambda *args: None)
    assert not eligible(model,x)
    handle.remove()
    model.gate_proj.weight.quant_state._permit_frozen_cache = True
    assert not eligible(model,x)
    model.gate_proj.weight.quant_state._permit_frozen_cache = False
    model.down_proj.compute_dtype = torch.float32
    assert not eligible(model,x)
    model.down_proj.compute_dtype = torch.float16
    model.act_fn = torch.nn.SiLU(inplace=True)
    assert not eligible(model,x)
    model.act_fn = torch.nn.GELU()
    assert not eligible(model,x)
