"""Streaming vocabulary reductions shared by loss, metrics and rewind outputs.

Only tile statistics cross the kernel boundary, never a vocabulary-sized softmax.
The training CE uses its own autograd boundary; metrics remain detached.
"""

from functools import wraps
from collections.abc import Mapping
import weakref

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def vocab_partial_kernel(X, MAX, SUM, MOM, ARG, V: tl.constexpr,
                          PARTS: tl.constexpr, BLOCK: tl.constexpr, METRICS: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * V + col, col < V, -float("inf")).to(tl.float32)
    maximum = tl.max(x, axis=0)
    z = x - maximum
    exp = tl.exp(z)
    total = tl.sum(exp, axis=0)
    index = row * PARTS + part
    tl.store(MAX + index, maximum)
    tl.store(SUM + index, total)
    if METRICS:
        moment = tl.sum(exp * tl.where(col < V, z, 0.0), axis=0)
        arg = tl.min(tl.where(x == maximum, col, 2147483647), axis=0)
        tl.store(MOM + index, moment)
        tl.store(ARG + index, arg)


@triton.jit(do_not_specialize=["T"])
def vocab_finalize_kernel(X, LABELS, MAX, SUM, MOM, ARG, LSE, ENTROPY, PRED, CE,
                           V: tl.constexpr, T, PARTS: tl.constexpr, BLOCK: tl.constexpr,
                           METRICS: tl.constexpr, ALIGNED: tl.constexpr = False):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    index = row * PARTS + part
    maximum = tl.load(MAX + index, part < PARTS, -float("inf"))
    total = tl.load(SUM + index, part < PARTS, 0)
    global_max = tl.max(maximum, axis=0)
    shift = maximum - global_max
    factor = tl.exp(shift)
    global_sum = tl.sum(total * factor, axis=0)
    lse = global_max + tl.log(global_sum)
    if METRICS:
        moment = tl.load(MOM + index, part < PARTS, 0)
        arg = tl.load(ARG + index, part < PARTS, 2147483647)
        global_moment = tl.sum((moment + total * tl.where(part < PARTS, shift, 0.0)) * factor, axis=0)
        entropy = tl.log(global_sum) - global_moment / global_sum
        prediction = tl.min(tl.where(maximum == global_max, arg, 2147483647), axis=0)
        tl.store(ENTROPY + row, entropy)
        tl.store(PRED + row, prediction)
    if ALIGNED:
        has_next = True
        target = tl.load(LABELS + row)
    else:
        has_next = row % T < T - 1
        target = tl.load(LABELS + row + 1, has_next, -100)
    valid = has_next & (target != -100)
    selected = tl.load(X + row * V + target, valid, 0).to(tl.float32)
    tl.store(LSE + row, lse)
    tl.store(CE + row, tl.where(valid, lse - selected, 0.0))


@triton.jit(do_not_specialize=["T"])
def vocab_ce_backward_kernel(X, LABELS, LSE, G, DX, V: tl.constexpr,
                              T, BLOCK: tl.constexpr, ALIGNED: tl.constexpr = False):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    if ALIGNED:
        has_next = True
        target = tl.load(LABELS + row)
    else:
        has_next = row % T < T - 1
        target = tl.load(LABELS + row + 1, has_next, -100)
    valid = has_next & (target != -100)
    x = tl.load(X + row * V + col, (col < V) & valid, 0).to(tl.float32)
    lse = tl.load(LSE + row)
    grad = tl.load(G + row)
    dx = (tl.exp(x - lse) - (col == target).to(tl.float32)) * grad
    tl.store(DX + row * V + col, tl.where(valid, dx, 0.0), col < V)


def vocabulary_stats(logits, labels, metrics=True, aligned=False):
    """Contiguous [B,T,V] logits -> per-position CE, entropy and first argmax."""
    b, t, v = logits.shape
    rows = b * t
    parts = triton.cdiv(v, 4096)
    partial = [torch.empty((rows, parts), device=logits.device, dtype=dtype)
               for dtype in (torch.float32, torch.float32, torch.float32, torch.int32)]
    lse, entropy, ce = [torch.empty((b, t), device=logits.device, dtype=torch.float32)
                       for _ in range(3)]
    pred = torch.empty((b, t), device=logits.device, dtype=torch.int64)
    vocab_partial_kernel[(rows, parts)](logits, *partial, v, parts, 4096, metrics, num_warps=4)
    vocab_finalize_kernel[(rows,)](logits, labels, *partial, lse, entropy, pred, ce,
                                   v, t, parts, triton.next_power_of_2(parts), metrics, aligned, num_warps=4)
    return ce, lse, entropy, pred


class VocabularyCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        ce, lse, entropy, predictions = vocabulary_stats(logits, labels)
        ctx.save_for_backward(logits, labels, lse)
        logits._vocabulary_forward_stats = (weakref.ref(labels), labels._version,
                                            ce.detach(), entropy, predictions)
        return ce

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        logits, labels, lse = ctx.saved_tensors
        grad = grad.contiguous()
        b, t, v = logits.shape
        dx = torch.empty_like(logits)
        vocab_ce_backward_kernel[(b * t, triton.cdiv(v, 4096))](
            logits, labels, lse, grad, dx, v, t, 4096, num_warps=4)
        return dx, None


def _supported(logits, labels):
    return (isinstance(logits, torch.Tensor) and logits.is_cuda and logits.ndim == 3
            and logits.is_contiguous() and logits.dtype in (torch.float16, torch.float32)
            and logits.shape[-1] == 151936 and logits.numel() > 0
            and isinstance(labels, torch.Tensor) and labels.shape == logits.shape[:2]
            and labels.device == logits.device and labels.dtype == torch.int64
            and labels.is_contiguous())


def patch_vocabulary():
    from transformers.loss.loss_utils import LOSS_MAPPING
    from trl.trainer.sft_trainer import SFTTrainer
    from soup_cli.trainer import rewind_hf

    original_loss = LOSS_MAPPING["ForCausalLM"]
    if getattr(original_loss, "_vocabulary_patched", False):
        return

    @wraps(original_loss)
    def loss(logits, labels, vocab_size, num_items_in_batch=None, ignore_index=-100,
             shift_labels=None, **kwargs):
        streamed = getattr(logits, "_streamed_ce", None)
        if (streamed is not None and streamed[0]() is labels
                and streamed[1] == labels._version and ignore_index == -100
                and shift_labels is None and vocab_size == 151936):
            count = (labels[:, 1:] != -100).sum() if num_items_in_batch is None else num_items_in_batch
            if torch.is_tensor(count):
                count = count.to(logits.device)
            return streamed[2].sum() / count
        # fixed_cross_entropy in the pinned transformers version ignores kwargs.
        # Model-forward metadata must not send supported training back to stock CE.
        if (ignore_index != -100 or shift_labels is not None
                or vocab_size != 151936 or not _supported(logits, labels)
                or not torch.is_grad_enabled() or not logits.requires_grad):
            return original_loss(logits, labels, vocab_size, num_items_in_batch,
                                 ignore_index, shift_labels, **kwargs)
        ce = VocabularyCE.apply(logits, labels)
        count = (labels[:, 1:] != -100).sum() if num_items_in_batch is None else num_items_in_batch
        if torch.is_tensor(count):
            count = count.to(logits.device)
        return ce.sum() / count

    loss._vocabulary_patched = True
    LOSS_MAPPING["ForCausalLM"] = loss
    original_compute = SFTTrainer.compute_loss

    @wraps(original_compute)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        attention = inputs.get("attention_mask")
        if (not getattr(model, "training", False) or not isinstance(labels, torch.Tensor)
                or not labels.is_cuda or labels.ndim != 2 or not isinstance(attention, torch.Tensor)
                or "shift_labels" in inputs or self.num_virtual_tokens != 0
                or self.aux_loss_enabled or self.args.use_liger_kernel or self.args.loss_type != "nll"):
            return original_compute(self, model, inputs, return_outputs, num_items_in_batch)
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        from .stream_head import _stream_request
        rewind_state = getattr(self, "_soup_rewind_state", None)
        internal_rewind = rewind_state is not None and not rewind_state.failed
        stream_token = _stream_request.set(not return_outputs or internal_rewind)
        try:
            value, outputs = super(SFTTrainer, self).compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        finally:
            _stream_request.reset(stream_token)
        logits = outputs.logits
        with torch.no_grad():
            if _supported(logits, labels) or hasattr(logits, "_streamed_ce"):
                saved = getattr(logits, "_vocabulary_forward_stats", None)
                if saved is not None and saved[0]() is labels and saved[1] == labels._version:
                    ce, entropy_rows, predictions = saved[2:]
                else:
                    ce, _, entropy_rows, predictions = vocabulary_stats(logits, labels)
                logits._vocabulary_rewind = (weakref.ref(labels), labels._version, ce)
            else:
                from trl.trainer.utils import entropy_from_logits
                entropy_rows = entropy_from_logits(logits)
                predictions = logits.argmax(dim=-1)
            entropy = (entropy_rows * attention).sum() / attention.sum()
            entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
            self._metrics["train"]["entropy"].append(entropy)
            n_tokens = self.accelerator.gather_for_metrics(attention.sum()).sum().item()
            self._total_train_tokens += n_tokens
            self._metrics["train"]["num_tokens"] = [self._total_train_tokens]
            supervised = labels[:, 1:] != -100
            correct = ((predictions[:, :-1] == labels[:, 1:]) & supervised).sum()
            correct = self.accelerator.gather_for_metrics(correct).sum()
            count = self.accelerator.gather_for_metrics(supervised.sum()).sum()
            accuracy = (correct / count).item() if count > 0 else 0.0
            self._metrics["train"]["mean_token_accuracy"].append(accuracy)
        return (value, outputs) if return_outputs else value

    SFTTrainer.compute_loss = compute_loss
    original_rows = rewind_hf.row_losses

    @wraps(original_rows)
    def row_losses(logits, labels):
        cached = getattr(logits, "_vocabulary_rewind", None)
        if cached is None or cached[0]() is not labels or cached[1] != labels._version:
            return original_rows(logits, labels)
        with torch.no_grad():
            count = (labels[:, 1:] != -100).sum(dim=1)
            means = cached[2].sum(dim=1) / count.clamp(min=1)
        return means.tolist(), count.tolist()

    rewind_hf.row_losses = row_losses
    # Accelerate normally widens all model outputs. The tagged logits are already
    # FP16 GEMM values, and every consumer below uses FP32 tile statistics. Keep
    # their identity so loss, metrics and rewind can share those exact statistics.
    from accelerate.utils import operations
    original_convert = operations.convert_to_fp32
    if not getattr(original_convert, "_vocabulary_patched", False):
        @wraps(original_convert)
        def convert(output):
            if isinstance(output, Mapping):
                logits = output.get("logits")
                if (isinstance(logits, torch.Tensor) and logits.dtype == torch.float16
                        and hasattr(logits, "_vocabulary_forward_stats")):
                    return type(output)(**{key: value if key == "logits" else original_convert(value)
                                           for key, value in output.items()})
            return original_convert(output)

        convert._vocabulary_patched = True
        operations.convert_to_fp32 = convert
