import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _pair():
    embedding = torch.nn.Embedding(129, 32).cuda().requires_grad_(False)
    with torch.no_grad():
        embedding.weight.copy_(embedding.weight.half().float())
    head = torch.nn.Linear(32, 129, bias=False).cuda().requires_grad_(False)
    head.weight = embedding.weight
    return embedding, head


@pytest.mark.parametrize("amp", [False, True])
def test_lossless_tied_storage_preserves_outputs_input_gradients_and_identity(amp):
    from solution.frozen_storage import _compress_pair
    from solution.precision import half_activations

    torch.manual_seed(241)
    embedding, head = _pair()
    parameter = embedding.weight
    reference = parameter.detach().clone()
    assert _compress_pair(embedding, head) == 1
    assert head.weight is embedding.weight is parameter
    assert parameter.dtype == torch.float16 and not parameter.requires_grad
    assert _compress_pair(embedding, head) == 0
    torch.testing.assert_close(parameter.float(), reference, rtol=0, atol=0)
    ids = torch.tensor([[3, 4, 8], [5, 0, 11]], device="cuda")
    for active in (False, True):
        token = half_activations.set(active)
        try:
            actual = embedding(ids)
        finally:
            half_activations.reset(token)
        expected = F.embedding(ids, reference)
        if active:
            expected = expected.half()
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    x = torch.randn(2, 7, 32, device="cuda", requires_grad=True)
    copy = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        expected = F.linear(x, reference)
        actual = head(copy)
    g = torch.randn_like(expected)
    expected.backward(g)
    actual.backward(g)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(copy.grad, x.grad, rtol=0, atol=0)
    assert parameter.grad is None


def test_nonrepresentable_or_trainable_weights_are_untouched():
    from solution.frozen_storage import _compress_pair

    embedding, head = _pair()
    with torch.no_grad():
        embedding.weight[0, 0] = 0.1234567
    ptr = embedding.weight.data_ptr()
    assert _compress_pair(embedding, head) == 0
    assert embedding.weight.dtype == torch.float32 and embedding.weight.data_ptr() == ptr
    with torch.no_grad():
        embedding.weight.copy_(embedding.weight.half().float())
    embedding.weight.requires_grad_(True)
    assert _compress_pair(embedding, head) == 0
    assert embedding.weight.dtype == torch.float32


def test_untied_weights_and_module_hooks_keep_original_path():
    from solution.frozen_storage import _compress_pair

    embedding, head = _pair()
    head.weight = torch.nn.Parameter(head.weight.detach().clone(), requires_grad=False)
    assert _compress_pair(embedding, head) == 0
    head.weight = embedding.weight
    handle = embedding.register_forward_hook(lambda module, args, output: output)
    try:
        assert _compress_pair(embedding, head) == 0
    finally:
        handle.remove()
    assert embedding.weight.dtype == torch.float32


def test_hf_input_gradient_hook_is_retained():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from solution.frozen_storage import _compress_pair

    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=129, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True,
        use_cache=False)).cuda().requires_grad_(False)
    embedding, head = model.get_input_embeddings(), model.get_output_embeddings()
    with torch.no_grad():
        embedding.weight.copy_(embedding.weight.half().float())
    model.enable_input_require_grads()
    hooks = dict(embedding._forward_hooks)
    assert hooks, "The real HF input-gradient hook must be exercised"
    try:
        assert _compress_pair(embedding, head) == 1
        assert dict(embedding._forward_hooks) == hooks
        value = embedding(torch.tensor([[1, 2, 3]], device="cuda"))
        assert value.dtype == torch.float32 and value.requires_grad
        head(value).sum().backward()
        assert torch.isfinite(value.grad).all()
        assert embedding.weight.grad is None
    finally:
        model.disable_input_require_grads()
