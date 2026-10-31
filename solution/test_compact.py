import copy

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("left_pad", [False, True])
def test_compact_qwen_keeps_loss_gradients_and_dropout_rng(left_pad, monkeypatch):
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from solution.compact import install_compaction, patch_compaction
    from solution import compact
    from solution.attention import patch_attention_compaction
    from solution.lora import patch_model
    from solution.swiglu import patch_qwen2_mlp

    torch.manual_seed(93)
    config = Qwen2Config(vocab_size=129, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         attn_implementation="sdpa", use_cache=False)
    ref = Qwen2ForCausalLM(config)
    for parent in list(ref.modules()):
        for name, child in list(parent.named_children()):
            if type(child) is torch.nn.Linear and name != "lm_head":
                quant = bnb.nn.Linear4bit(child.in_features, child.out_features,
                                         bias=child.bias is not None, compute_dtype=torch.float16,
                                         quant_type="nf4", compress_statistics=True)
                quant.weight = bnb.nn.Params4bit(child.weight.detach(), requires_grad=False,
                                                quant_type="nf4", compress_statistics=True)
                if child.bias is not None:
                    quant.bias = torch.nn.Parameter(child.bias.detach(), requires_grad=False)
                setattr(parent, name, quant)
    ref = ref.cuda()
    ref.requires_grad_(False)
    ref.is_loaded_in_4bit = True
    ref = get_peft_model(ref, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                        target_modules=["q_proj", "v_proj"]))
    with torch.no_grad():
        for name, param in ref.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.02)
    model = copy.deepcopy(ref)
    patch_qwen2_mlp()
    patch_compaction()
    patch_attention_compaction()
    patch_model(model)
    install_compaction(model)
    ids = torch.randint(0, 129, (3, 33), device="cuda")
    mask = torch.zeros_like(ids)
    for i, length in enumerate([9, 17, 33]):
        mask[i, -length:] = 1 if left_pad else 0
        if not left_pad:
            mask[i, :length] = 1
    labels = ids.clone()
    labels[mask == 0] = -100
    for i in range(3):
        labels[i, mask[i].nonzero(as_tuple=True)[0][:2]] = -100
    calls = []
    original_pack = compact._pack

    def counted(x, state):
        calls.append((x.shape, state[1].numel()))
        return original_pack(x, state)

    monkeypatch.setattr(compact, "_pack", counted)
    attention_calls = []
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def counted_sdpa(q, *args, **kwargs):
        attention_calls.append(tuple(q.shape))
        return original_sdpa(q, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", counted_sdpa)
    results = []
    for current in (ref, model):
        current.train()
        torch.manual_seed(157)
        with torch.autocast("cuda", dtype=torch.float16):
            output = current(input_ids=ids, attention_mask=mask, labels=labels)
        output.loss.backward()
        results.append((output.loss, output.logits.detach(), torch.cuda.get_rng_state(),
                        {n: p.grad for n, p in current.named_parameters() if p.requires_grad}))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=2e-3, atol=2e-3)
    active = mask.bool()
    torch.testing.assert_close(results[0][1][active], results[1][1][active], rtol=1e-2, atol=2e-3)
    assert torch.equal(results[0][2], results[1][2])
    assert calls and all(n == 59 for _, n in calls)
    assert (1, 4, 9, 16) in attention_calls
    for name in results[0][3]:
        torch.testing.assert_close(results[0][3][name], results[1][3][name], rtol=2e-2, atol=2e-3)
        error = (results[0][3][name].float() - results[1][3][name].float()).norm()
        assert error <= 0.01 * results[0][3][name].float().norm() + 1e-7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("width", [256, 1536, 151936])
def test_token_copy_kernels_and_backward_are_exact(dtype, width):
    from solution.compact import TokenPack, TokenRestore

    torch.manual_seed(25)
    ref = torch.randn((2, width, 13), device="cuda", dtype=dtype).transpose(1, 2).requires_grad_()
    actual_x = ref.detach().clone().requires_grad_()
    mask = torch.rand((2, 13), device="cuda") > 0.4
    indices = mask.reshape(-1).nonzero(as_tuple=True)[0]
    inverse = torch.where(mask.reshape(-1), mask.reshape(-1).cumsum(0, dtype=torch.int32) - 1, -1)
    source = ref.reshape(-1, width).index_select(0, indices)
    expected = ref.new_zeros((26, width)).index_copy(0, indices, source).reshape(2, 13, width)
    packed = TokenPack.apply(actual_x, indices, inverse, (2, 13))
    actual = TokenRestore.apply(packed, indices, inverse, (2, 13))
    grad = torch.randn_like(expected)
    expected.backward(grad)
    actual.backward(grad)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_x.grad, ref.grad, rtol=0, atol=0)
