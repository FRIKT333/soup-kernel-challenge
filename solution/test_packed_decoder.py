import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')


def make_model():
    import bitsandbytes as bnb
    from peft import LoraConfig,get_peft_model
    from transformers import Qwen2Config,Qwen2ForCausalLM
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=129,hidden_size=1536,intermediate_size=128,
        num_hidden_layers=2,num_attention_heads=12,num_key_value_heads=2,use_cache=False,
        attn_implementation='sdpa'))
    for parent in list(model.modules()):
        for name,child in list(parent.named_children()):
            if type(child) is torch.nn.Linear and name != 'lm_head':
                p = bnb.nn.Linear4bit(child.in_features,child.out_features,bias=child.bias is not None,
                    compute_dtype=torch.float16,quant_type='nf4',compress_statistics=True)
                p.weight = bnb.nn.Params4bit(child.weight.detach().half(),requires_grad=False,
                    quant_type='nf4',compress_statistics=True)
                if child.bias is not None:
                    p.bias = torch.nn.Parameter(child.bias.detach().half(),requires_grad=False)
                setattr(parent,name,p)
    model = model.cuda().train().requires_grad_(False)
    model.is_loaded_in_4bit = True
    model = get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,
        target_modules=['q_proj','v_proj']))
    with torch.no_grad():
        for name,p in model.named_parameters():
            if 'lora_B' in name:
                p.normal_(std=.01)
    return model


@pytest.mark.parametrize('left_pad',[False,True])
@pytest.mark.parametrize('seed',[157,715])
def test_packed_layers_keep_native_math_adapter_gradients_rng_and_public_fallback(monkeypatch,left_pad,seed):
    from functools import wraps
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model,Qwen2RMSNorm
    from solution.swiglu import patch_qwen2_mlp
    from solution.rmsnorm import patch_qwen2_rmsnorm
    from solution.compact import patch_compaction,install_compaction
    from solution.attention import patch_attention_compaction
    from solution.lora import patch_model
    from solution.precision import install_activation_precision
    from solution.nf4_cache import install_nf4_cache,patch_nf4_cache
    from solution.schedule import choose_nf4_cache
    from solution.stream_head import patch_stream_head,_stream_request
    from solution.packed_decoder import patch_packed_decoder,_packed_decoder
    torch.manual_seed(57)
    model = make_model()
    patch_qwen2_mlp()
    patch_qwen2_rmsnorm()
    patch_compaction()
    patch_attention_compaction()
    patch_stream_head()
    patch_nf4_cache()
    patch_model(model)
    install_compaction(model)
    install_nf4_cache(model)
    choose_nf4_cache(model,0)
    install_activation_precision(model)
    root = model.get_base_model()
    # Exercise the private streamed-consumer context with a small test vocab.
    # The actual projection stays dense so valid-token logits can be compared.
    root._streamed_head = True
    root.enable_input_require_grads()
    patch_packed_decoder()
    packed_forward = Qwen2Model.forward
    native_forward = packed_forward.__wrapped__
    norm_forward = Qwen2RMSNorm.forward
    shapes = []
    @wraps(norm_forward)
    def norm(self,x):
        shapes.append((x.ndim,_packed_decoder.get() is not None))
        return norm_forward(self,x)
    monkeypatch.setattr(Qwen2RMSNorm,'forward',norm)
    ids = torch.randint(129,(2,17),device='cuda')
    mask = torch.ones_like(ids)
    mask[0,:6] = 0 if left_pad else 1
    if not left_pad:
        mask[0,11:] = 0
    labels = ids.clone()
    labels[mask == 0] = -100
    for row in range(2):
        labels[row,mask[row].nonzero(as_tuple=True)[0][:3]] = -100
    results = []
    for packed,private in [(False,True),(True,True),(True,False)]:
        monkeypatch.setattr(Qwen2Model,'forward',packed_forward if packed else native_forward)
        model.zero_grad(set_to_none=True)
        shapes.clear()
        torch.manual_seed(seed)
        token = _stream_request.set(private)
        try:
            with torch.autocast('cuda',dtype=torch.float16):
                output = model(input_ids=ids,attention_mask=mask,labels=labels)
            rng = torch.cuda.get_rng_state()
            (output.loss*128).backward()
        finally:
            _stream_request.reset(token)
        grads = {n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad}
        results.append((output.loss.detach(),output.logits.detach(),rng,grads))
        assert len(shapes) == 5
        assert all(ndim == (2 if packed and private else 3) for ndim,_ in shapes)
        assert all(active == (packed and private) for _,active in shapes)
        assert _packed_decoder.get() is None
    ref,actual,public = results
    for compared in (actual,public):
        torch.testing.assert_close(compared[0],ref[0],rtol=2e-3,atol=2e-3)
        torch.testing.assert_close(compared[1][mask.bool()],ref[1][mask.bool()],rtol=1e-2,atol=3e-3)
        assert torch.equal(compared[2],ref[2])
        for name,g in ref[3].items():
            assert torch.isfinite(compared[3][name]).all()
            assert (compared[3][name]-g).norm() <= g.norm()*.02 + 1e-6
    # The public route retains the complete original padded logits too.
    torch.testing.assert_close(public[1],ref[1],rtol=0,atol=0)
