import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_half_activation_storage_keeps_dropout_rng_and_adapter_gradients(monkeypatch):
    import inspect
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from solution.precision import install_activation_precision
    from solution.rmsnorm import patch_qwen2_rmsnorm
    from solution.compact import patch_compaction, install_compaction
    from solution.attention import patch_attention_compaction
    from solution.lora import patch_model
    from solution.swiglu import patch_qwen2_mlp
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    torch.manual_seed(57)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=129, hidden_size=1536,
        intermediate_size=128, num_hidden_layers=2, num_attention_heads=12,
        num_key_value_heads=2, use_cache=False, attn_implementation="sdpa"))
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if type(child) is torch.nn.Linear and name != "lm_head":
                p = bnb.nn.Linear4bit(child.in_features, child.out_features,
                    bias=child.bias is not None, compute_dtype=torch.float16,
                    quant_type="nf4", compress_statistics=True)
                p.weight = bnb.nn.Params4bit(child.weight.detach().half(), requires_grad=False,
                                             quant_type="nf4", compress_statistics=True)
                if child.bias is not None:
                    p.bias = torch.nn.Parameter(child.bias.detach().half(), requires_grad=False)
                setattr(parent, name, p)
    model = model.cuda().train()
    model.is_loaded_in_4bit = True
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                                            target_modules=["q_proj", "v_proj"]))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(std=.01)
    patch_qwen2_mlp()
    patch_qwen2_rmsnorm()
    patch_compaction()
    patch_attention_compaction()
    packed_attention = Qwen2Attention.forward
    native_attention = inspect.unwrap(packed_attention)
    patch_model(model)
    install_compaction(model)
    ids = torch.randint(129, (2, 17), device="cuda")
    mask = torch.ones_like(ids)
    mask[0, 11:] = 0
    labels = ids.clone()
    labels[mask == 0] = -100
    labels[:, :3] = -100
    results = []
    sdpa_shapes = []
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    def observed_sdpa(q, *args, **kwargs):
        sdpa_shapes.append(tuple(q.shape))
        return original_sdpa(q, *args, **kwargs)
    monkeypatch.setattr(torch.nn.functional, 'scaled_dot_product_attention', observed_sdpa)
    for index, (precision, compact_attention) in enumerate(((False,False),(True,False),(True,True))):
        monkeypatch.setattr(Qwen2Attention, 'forward', packed_attention if compact_attention else native_attention)
        if precision:
            assert install_activation_precision(model) == (1 if index == 1 else 0)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(715)
        sdpa_shapes.clear()
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(input_ids=ids, attention_mask=mask, labels=labels)
        rng = torch.cuda.get_rng_state()
        (output.loss * 128.).backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
        results.append((output.loss.detach(), grads, rng))
        if compact_attention:
            assert len(sdpa_shapes) == 4
            assert all(shape[0] == 1 and shape[2] in (11,17) for shape in sdpa_shapes)
        else:
            assert len(sdpa_shapes) == 2 and all(shape[0] == 2 for shape in sdpa_shapes)
    for reference_result, actual_result in ((results[0],results[1]), (results[1],results[2])):
        torch.testing.assert_close(reference_result[0],actual_result[0],rtol=2e-3,atol=2e-3)
        assert torch.equal(reference_result[2],actual_result[2])
        for name, reference in reference_result[1].items():
            actual = actual_result[1][name]
            assert torch.isfinite(actual).all()
            assert (actual-reference).norm() <= reference.norm()*.02 + 1e-6
