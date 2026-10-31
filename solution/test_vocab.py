import pytest
import torch
import torch.nn.functional as F
from collections import defaultdict
from types import SimpleNamespace

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("vocab", [129, 151936])
def test_vocabulary_loss_metrics_and_backward(dtype, vocab):
    from solution.vocab import VocabularyCE, vocabulary_stats

    torch.manual_seed(49)
    x = torch.randn((2, 17, vocab), device="cuda", dtype=dtype, requires_grad=True)
    labels = torch.randint(0, vocab, (2, 17), device="cuda")
    labels[:, :3] = -100
    labels[1] = -100
    with torch.no_grad():
        x[0, 0, :2] = 30  # exact tie: first vocabulary index must win
    copy = x.detach().clone().requires_grad_()
    shifted = F.pad(labels, (0, 1), value=-100)[:, 1:]
    ref = F.cross_entropy(copy.float().reshape(-1, vocab), shifted.reshape(-1), reduction="none").reshape(2, 17)
    actual = VocabularyCE.apply(x, labels)
    stats_ce, _, entropy, predictions = vocabulary_stats(x.detach(), labels)
    logp = copy.detach().float().log_softmax(-1)
    expected_entropy = -(logp.exp() * logp).sum(-1)
    torch.testing.assert_close(actual, ref, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(stats_ce, ref, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(entropy, expected_entropy, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(predictions, copy.detach().argmax(-1), rtol=0, atol=0)
    grad = torch.randn_like(ref)
    actual.backward(grad)
    ref.backward(grad)
    tol = 2e-3 if dtype == torch.float16 else 3e-6
    torch.testing.assert_close(x.grad, copy.grad, rtol=tol, atol=tol)


def test_trainer_metrics_and_rewind_integration(monkeypatch):
    from solution.vocab import patch_vocabulary
    from transformers import Trainer
    from transformers.loss.loss_utils import LOSS_MAPPING
    from trl.trainer.sft_trainer import SFTTrainer
    from soup_cli.trainer import rewind_hf
    from accelerate.utils import operations

    # Register undo before patching so this test does not change later tests.
    monkeypatch.setitem(LOSS_MAPPING, "ForCausalLM", LOSS_MAPPING["ForCausalLM"])
    monkeypatch.setattr(SFTTrainer, "compute_loss", SFTTrainer.compute_loss)
    monkeypatch.setattr(rewind_hf, "row_losses", rewind_hf.row_losses)
    monkeypatch.setattr(operations, "convert_to_fp32", operations.convert_to_fp32)
    original_loss = LOSS_MAPPING["ForCausalLM"]
    torch.manual_seed(72)
    logits = torch.randn((2, 7, 151936), device="cuda")
    labels = torch.randint(0, 151936, (2, 7), device="cuda")
    labels[:, :2] = -100
    mask = torch.ones_like(labels)
    expected_rows = rewind_hf.row_losses(logits, labels)
    expected_entropy = -(logits.log_softmax(-1).exp() * logits.log_softmax(-1)).sum(-1).mean()
    expected_accuracy = ((logits[:, :-1].argmax(-1) == labels[:, 1:]) & (labels[:, 1:] != -100)).sum() / (labels[:, 1:] != -100).sum()
    outputs = SimpleNamespace(logits=logits)
    sentinel_loss = torch.tensor(3.0, device="cuda")
    monkeypatch.setattr(Trainer, "compute_loss", lambda *args, **kwargs: (sentinel_loss, outputs))
    patch_vocabulary()
    for count in (None, torch.tensor(23, device="cuda")):
        operand = logits.half().requires_grad_()
        expected = original_loss(operand, labels, 151936, num_items_in_batch=count, output_attentions=False)
        got = LOSS_MAPPING["ForCausalLM"](operand, labels, 151936, num_items_in_batch=count, output_attentions=False)
        torch.testing.assert_close(got, expected, rtol=3e-5, atol=3e-5)
        pending, nodes = [got.grad_fn], []
        while pending:
            node = pending.pop()
            if node is not None:
                nodes.append(type(node).__name__)
                pending.extend(child for child, _ in node.next_functions)
        assert "VocabularyCEBackward" in nodes
    trainer = object.__new__(SFTTrainer)
    trainer.num_virtual_tokens = 0
    trainer.aux_loss_enabled = False
    trainer.args = SimpleNamespace(use_liger_kernel=False, loss_type="nll")
    trainer.accelerator = SimpleNamespace(gather_for_metrics=lambda x: x)
    trainer._metrics = {"train": defaultdict(list)}
    trainer._total_train_tokens = 10
    actual_loss, actual_outputs = trainer.compute_loss(
        torch.nn.Identity(), {"labels": labels, "attention_mask": mask}, return_outputs=True)
    assert actual_loss is sentinel_loss and actual_outputs is outputs
    assert trainer._total_train_tokens == 24
    assert trainer._metrics["train"]["entropy"][0] == pytest.approx(expected_entropy.item(), abs=3e-5)
    assert trainer._metrics["train"]["mean_token_accuracy"][0] == expected_accuracy.item()
    rows = rewind_hf.row_losses(logits, labels)
    assert rows[0] == pytest.approx(expected_rows[0], abs=3e-5)
    assert rows[1] == expected_rows[1]
    # A different label tensor cannot reuse stale row losses.
    new_labels = labels.clone()
    assert rewind_hf.row_losses(logits, new_labels)[1] == expected_rows[1]


def test_loss_stats_are_shared_through_accelerate(monkeypatch):
    from solution import vocab as optimized
    from transformers import Trainer
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.loss.loss_utils import LOSS_MAPPING
    from trl.trainer.sft_trainer import SFTTrainer
    from soup_cli.trainer import rewind_hf
    from accelerate.utils import operations

    monkeypatch.setitem(LOSS_MAPPING, "ForCausalLM", LOSS_MAPPING["ForCausalLM"])
    monkeypatch.setattr(SFTTrainer, "compute_loss", SFTTrainer.compute_loss)
    monkeypatch.setattr(rewind_hf, "row_losses", rewind_hf.row_losses)
    monkeypatch.setattr(operations, "convert_to_fp32", operations.convert_to_fp32)
    optimized.patch_vocabulary()
    calls = []
    original_stats = optimized.vocabulary_stats

    def counted(*args, **kwargs):
        calls.append(1)
        return original_stats(*args, **kwargs)

    monkeypatch.setattr(optimized, "vocabulary_stats", counted)
    torch.manual_seed(84)
    x = torch.randn((2, 7, 151936), device="cuda", dtype=torch.float16, requires_grad=True)
    labels = torch.randint(0, 151936, (2, 7), device="cuda")
    labels[:, :2] = -100
    loss = LOSS_MAPPING["ForCausalLM"](x, labels, 151936)
    assert hasattr(x, "_vocabulary_forward_stats")
    raw = CausalLMOutputWithPast(loss=loss, logits=x, hidden_states=(x[:, :, :2],))
    output = operations.ConvertOutputsToFp32(lambda: raw)()
    assert output.logits is x and output.loss.dtype == torch.float32
    assert output.hidden_states[0].dtype == torch.float32
    ordinary = operations.convert_to_fp32(CausalLMOutputWithPast(logits=x.detach()))
    assert ordinary.logits.dtype == torch.float32
    monkeypatch.setattr(Trainer, "compute_loss", lambda *args, **kwargs: (loss, output))
    trainer = object.__new__(SFTTrainer)
    trainer.num_virtual_tokens, trainer.aux_loss_enabled = 0, False
    trainer.args = SimpleNamespace(use_liger_kernel=False, loss_type="nll")
    trainer.accelerator = SimpleNamespace(gather_for_metrics=lambda x: x)
    trainer._metrics, trainer._total_train_tokens = {"train": defaultdict(list)}, 0
    trainer.compute_loss(torch.nn.Identity(), {"labels": labels, "attention_mask": torch.ones_like(labels)})
    actual_rows = rewind_hf.row_losses(output.logits, labels)
    assert calls == [1]
    logp = x.detach().float().log_softmax(-1)
    entropy = -(logp.exp() * logp).sum(-1).mean().item()
    assert trainer._metrics["train"]["entropy"][0] == pytest.approx(entropy, abs=3e-5)
    targets = F.pad(labels, (0, 1), value=-100)[:, 1:]
    ce = F.cross_entropy(x.float().reshape(-1, 151936), targets.reshape(-1), reduction="none").reshape(2, 7)
    expected_rows = ce.sum(-1) / (labels[:, 1:] != -100).sum(-1)
    assert actual_rows[0] == pytest.approx(expected_rows.tolist(), abs=3e-5)
