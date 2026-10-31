"""Exact token-chunked frozen classifier with shared loss/metric statistics.

The temporary logits buffer is O(chunk * vocabulary), not O(batch * time *
vocabulary). Backward recomputes each chunk and differentiates every supervised
token. Inspired by the memory scheduling idea in Cut Your Losses (2411.09009);
this is an independent implementation using cuBLAS and our vocabulary kernels.
No probability/gradient filtering or vocabulary truncation is used.
"""

from contextvars import ContextVar
from functools import wraps
from types import MethodType
import weakref

import torch
import torch.nn.functional as F
import triton
from torch.autograd.function import once_differentiable

from .vocab import vocabulary_stats, vocab_ce_backward_kernel

_labels = ContextVar("stream_head_labels", default=None)
_stream_request = ContextVar("stream_head_trainer_request", default=False)


class StreamedHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, indices, chunk):
        # Explicit casting matches the FP16 classifier GEMM under task AMP.
        x = hidden.reshape(-1, hidden.shape[-1]).index_select(0, indices).half()
        w = weight.half()
        rows, vocab = x.shape[0], w.shape[0]
        ce = torch.empty(rows, device=x.device, dtype=torch.float32)
        entropy = torch.empty_like(ce)
        lse = torch.empty_like(ce)
        pred = torch.empty(rows, device=x.device, dtype=torch.int64)
        selected = targets.index_select(0, indices).contiguous()
        for start in range(0, rows, chunk):
            stop = min(start + chunk, rows)
            logits = F.linear(x[start:stop], w).unsqueeze(0)
            stats = vocabulary_stats(logits, selected[start:stop].unsqueeze(0), aligned=True)
            ce[start:stop], lse[start:stop] = stats[0].flatten(), stats[1].flatten()
            entropy[start:stop], pred[start:stop] = stats[2].flatten(), stats[3].flatten()
            del logits, stats
        supervised = (selected != -100).nonzero(as_tuple=True)[0]
        ctx.save_for_backward(x.index_select(0, supervised), w,
                             selected.index_select(0, supervised),
                             indices.index_select(0, supervised),
                             lse.index_select(0, supervised), supervised)
        ctx.shape, ctx.dtype, ctx.chunk = hidden.shape, hidden.dtype, chunk
        ctx.mark_non_differentiable(entropy, pred)
        return ce, entropy, pred

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_ce, grad_entropy, grad_pred):
        x, w, targets, indices, lse, supervised = ctx.saved_tensors
        rows, vocab = x.shape[0], w.shape[0]
        dx = torch.empty_like(x, dtype=ctx.dtype)
        grad_ce = grad_ce.index_select(0, supervised).contiguous()
        # No model forward, dropout, RNG consumption or dataset access here.
        for start in range(0, rows, ctx.chunk):
            stop = min(start + ctx.chunk, rows)
            logits = F.linear(x[start:stop], w)
            dlogits = torch.empty_like(logits)
            vocab_ce_backward_kernel[(stop - start, triton.cdiv(vocab, 4096))](
                logits, targets[start:stop], lse[start:stop], grad_ce[start:stop],
                dlogits, vocab, stop - start, 4096, True, num_warps=4)
            dx[start:stop] = torch.mm(dlogits, w).to(ctx.dtype)
            del logits, dlogits
        full = torch.zeros((ctx.shape[0] * ctx.shape[1], ctx.shape[-1]),
                           device=x.device, dtype=ctx.dtype)
        full.index_copy_(0, indices, dx)
        return full.reshape(ctx.shape), None, None, None, None


def streamed_projection(hidden, weight, labels, indices=None, chunk=128):
    b, t, _ = hidden.shape
    if indices is None:
        indices = torch.arange(b * t, device=hidden.device)
    targets = F.pad(labels[:, 1:], (0, 1), value=-100).reshape(-1)
    ce, entropy, pred = StreamedHead.apply(hidden, weight, targets, indices, chunk)
    def restore(value):
        return value.new_zeros(b * t).index_copy(0, indices, value).reshape(b, t)
    return restore(ce), restore(entropy), restore(pred)


def install_stream_head(model):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    if not isinstance(model, torch.nn.Module):
        return
    for module in model.modules():
        if not isinstance(module, Qwen2ForCausalLM):
            continue
        head = module.lm_head
        if (type(head) is not torch.nn.Linear or head.weight.requires_grad
                or head.out_features != 151936 or getattr(head, "_streamed_head", False)):
            continue
        original = head.forward

        @wraps(original)
        def projection(self, hidden, _original=original):
            labels = _labels.get()
            if (labels is None or hidden.ndim != 3 or hidden.shape[:2] != labels.shape
                    or not hidden.is_cuda or not self.training or not torch.is_grad_enabled()
                    or not torch.is_autocast_enabled("cuda")
                    or torch.get_autocast_dtype("cuda") != torch.float16):
                return _original(hidden)
            from .compact import _tokens
            from .schedule import choose_head_chunk
            state = _tokens.get()
            indices = state[1] if state is not None else None
            ce, entropy, pred = streamed_projection(hidden, self.weight, labels, indices,
                                                   choose_head_chunk(self.out_features))
            # Internal carrier replaces dense training-only logits. The loss,
            # TRL metrics, Accelerate conversion and rewind all consume its stats.
            # Evaluation/inference and unsupported calls still return real logits.
            carrier = hidden.new_empty((*labels.shape, 1), dtype=torch.float16)
            carrier._vocabulary_forward_stats = (weakref.ref(labels), labels._version,
                                                 ce.detach(), entropy, pred)
            carrier._streamed_ce = (weakref.ref(labels), labels._version, ce)
            return carrier

        head.forward = MethodType(projection, head)
        head._streamed_head = True
        module._streamed_head = True


def patch_stream_head():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    original = Qwen2ForCausalLM.forward
    if getattr(original, "_streamed_context", False):
        return

    @wraps(original)
    def forward(self, *args, **kwargs):
        labels = kwargs.get("labels")
        supported = (_stream_request.get() and getattr(self, "_streamed_head", False) and self.training
                     and torch.is_grad_enabled() and isinstance(labels, torch.Tensor)
                     and labels.is_cuda and labels.ndim == 2 and labels.dtype == torch.int64
                     and not args and kwargs.get("logits_to_keep", 0) == 0
                     and kwargs.get("shift_labels") is None
                     and kwargs.get("ignore_index", -100) == -100
                     and not kwargs.get("output_attentions", False)
                     and not kwargs.get("output_hidden_states", False))
        token = _labels.set(labels if supported else None)
        try:
            return original(self, *args, **kwargs)
        finally:
            _labels.reset(token)

    forward._streamed_context = True
    Qwen2ForCausalLM.forward = forward
