import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("vocab,chunk", [(129, 16), (151936, 16)])
def test_streamed_projection_loss_metrics_and_gradients(vocab, chunk):
    from solution.stream_head import streamed_projection
    torch.manual_seed(125)
    x = (torch.randn(2, 19, 64, device="cuda") * .1).requires_grad_()
    ref = x.detach().clone().requires_grad_()
    weight = torch.randn(vocab, 64, device="cuda") * .1
    labels = torch.randint(vocab, (2, 19), device="cuda")
    labels[:, :3] = -100
    labels[1, 12:] = -100
    indices = torch.cat((torch.arange(19, device="cuda"), torch.arange(19, 31, device="cuda")))
    with torch.autocast("cuda", dtype=torch.float16):
        dense = F.linear(ref, weight)
        ce, entropy, pred = streamed_projection(x, weight, labels, indices, chunk)
    target = F.pad(labels[:, 1:], (0, 1), value=-100)
    expected = F.cross_entropy(dense.float().flatten(0, 1), target.flatten(), reduction="none").reshape_as(labels)
    logp = dense.detach().float().log_softmax(-1)
    active = torch.zeros(38, device="cuda", dtype=torch.bool)
    active[indices] = True
    active = active.reshape_as(labels)
    torch.testing.assert_close(ce, expected, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(entropy[active], -(logp.exp() * logp).sum(-1)[active], rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(pred[active], dense.argmax(-1)[active], rtol=0, atol=0)
    g = torch.randn_like(ce)
    ce.backward(g)
    expected.backward(g)
    torch.testing.assert_close(x.grad, ref.grad, rtol=5e-3, atol=3e-4)


def test_no_supervised_tokens_have_zero_projection_gradient():
    from solution.stream_head import streamed_projection
    x = torch.randn(2, 5, 64, device="cuda", requires_grad=True)
    weight = torch.randn(129, 64, device="cuda")
    labels = torch.full((2, 5), -100, dtype=torch.int64, device="cuda")
    ce, entropy, pred = streamed_projection(x, weight, labels, chunk=4)
    ce.sum().backward()
    assert torch.count_nonzero(x.grad).item() == 0
    assert torch.isfinite(entropy).all()


def test_qwen_streaming_integration_keeps_public_logits_and_loss(monkeypatch):
    import bitsandbytes as bnb
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from peft import LoraConfig, get_peft_model
    from solution.stream_head import install_stream_head, patch_stream_head, _stream_request
    from solution.vocab import patch_vocabulary
    torch.manual_seed(211)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=151936, hidden_size=64,
        intermediate_size=128, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, use_cache=False))
    # Exercise the task's NF4 PEFT dispatcher. Dense test projections instead
    # probe every optional backend, including Kaggle's incompatible torchao.
    for parent in list(model.modules()):
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
    model = model.cuda().train()
    model.is_loaded_in_4bit = True
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                                            target_modules=["q_proj", "v_proj"]))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=.02)
    ids = torch.randint(151936, (2, 9), device="cuda")
    labels = ids.clone()
    labels[:, :3] = -100
    x = torch.randn(2, 9, 64, device="cuda", requires_grad=True)
    torch.manual_seed(411)
    with torch.autocast("cuda", dtype=torch.float16):
        reference = model(inputs_embeds=x, labels=labels)
    expected_rng = torch.cuda.get_rng_state()
    reference.loss.backward()
    expected_grad = x.grad.clone()
    expected_adapters = {name: p.grad.clone() for name, p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    x.grad = None
    patch_vocabulary()
    patch_stream_head()
    install_stream_head(model)
    token = _stream_request.set(True)
    try:
        torch.manual_seed(411)
        with torch.autocast("cuda", dtype=torch.float16):
            result = model(inputs_embeds=x, labels=labels)
    finally:
        _stream_request.reset(token)
    assert hasattr(result.logits, "_streamed_ce")
    assert result.logits.shape == (2, 9, 1)
    assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
    result.loss.backward()
    torch.testing.assert_close(result.loss, reference.loss, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(x.grad, expected_grad, rtol=1e-2, atol=3e-5)
    for name, param in model.named_parameters():
        if param.requires_grad:
            torch.testing.assert_close(param.grad, expected_adapters[name], rtol=1e-2, atol=3e-5)
    with torch.autocast("cuda", dtype=torch.float16):
        public = model(inputs_embeds=x, labels=labels)
    assert public.logits.shape == (2, 9, 151936)


def test_rewind_trainer_requests_streaming_and_keeps_row_records(monkeypatch):
    from collections import defaultdict
    from types import SimpleNamespace
    from transformers import Trainer
    from trl import SFTTrainer
    from soup_cli.trainer import rewind_hf
    from solution.stream_head import _stream_request
    from solution.vocab import patch_vocabulary
    patch_vocabulary()
    labels = torch.tensor([[-100, -100, 2, 4], [-100, 3, 4, 5]], device="cuda")
    ce = torch.tensor([[0., 2., 4., 0.], [1., 2., 3., 0.]], device="cuda")
    records = []
    class Sink:
        def record_batch(self, **record):
            records.append(record)
    trainer = object.__new__(rewind_hf.make_rewind_trainer_class(SFTTrainer))
    trainer._soup_rewind_state = rewind_hf.RewindState(Sink())
    for index in [27, 11]:
        trainer._soup_rewind_state.on_index(index)
    trainer.state = SimpleNamespace(global_step=0)
    trainer.num_virtual_tokens = 0
    trainer.aux_loss_enabled = False
    trainer.args = SimpleNamespace(use_liger_kernel=False, loss_type="nll")
    trainer.accelerator = SimpleNamespace(gather_for_metrics=lambda x: x)
    trainer._metrics = {"train": defaultdict(list)}
    trainer._total_train_tokens = 0
    def model_loss(self, model, inputs, **kwargs):
        import weakref
        assert _stream_request.get(), "rewind requests return_outputs=True internally"
        carrier = torch.empty(2, 4, 1, device="cuda", dtype=torch.float16)
        carrier._streamed_ce = (weakref.ref(labels), labels._version, ce)
        carrier._vocabulary_forward_stats = (weakref.ref(labels), labels._version,
                                             ce, torch.ones_like(ce), torch.zeros_like(labels))
        return ce.sum(), SimpleNamespace(logits=carrier)
    monkeypatch.setattr(Trainer, "compute_loss", model_loss)
    trainer.compute_loss(torch.nn.Identity(), {"labels": labels, "attention_mask": torch.ones_like(labels)})
    assert not trainer._soup_rewind_state.failed
    assert records[0]["rows"] == [27, 11]
    assert records[0]["row_tokens"] == [2, 3]
    assert records[0]["row_loss"] == [3., 2.]
