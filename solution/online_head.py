"""Online vocabulary statistics and supervised gradient in one kernel.

All active forward rows retain CE, entropy and argmax. Only ignored backward
rows are omitted. No vocabulary entries or nonzero gradients are filtered.
"""
from functools import wraps
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

ENABLED = True
BLOCK = 2048
COLUMN_BACKWARD = False


def column_weight(weight):
    from .nf4_cache import _tensor_key
    key = (_tensor_key(weight), weight.stride())
    cache = getattr(weight, '_classifier_column_cache', None)
    if cache is None or cache[0] != key:
        cache = key, weight.t().contiguous().t()
        weight._classifier_column_cache = cache
    return cache[1]


@triton.jit(do_not_specialize=['ROWS'])
def online_vocabulary_gradient_kernel(LOGITS, LABELS, CE, ENTROPY, PRED, GRAD, ROWS,
        V: tl.constexpr, RANK_BLOCK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    maximum = tl.full((), -float('inf'), tl.float32)
    total = tl.full((), 0, tl.float32)
    moment = tl.full((), 0, tl.float32)
    arg = tl.full((), 2147483647, tl.int32)
    for part in range(tl.cdiv(V, BLOCK)):
        col = part * BLOCK + columns
        value = tl.load(LOGITS + row * V + col, col < V, other=-float('inf')).to(tl.float32)
        new_maximum = tl.maximum(maximum, tl.max(value, 0))
        shift = tl.where(total > 0, maximum - new_maximum, 0.0)
        factor = tl.exp(shift)
        centered = tl.where(col < V, value - new_maximum, 0.0)
        exp = tl.where(col < V, tl.exp(centered), 0.0)
        moment = (moment + total * shift) * factor + tl.sum(exp * centered, 0)
        total = total * factor + tl.sum(exp, 0)
        candidate = tl.min(tl.where((value == new_maximum) & (col < V), col, 2147483647), 0)
        arg = tl.minimum(tl.where(new_maximum == maximum, arg, 2147483647), candidate)
        maximum = new_maximum
    log_total = tl.log(total)
    lse = maximum + log_total
    target = tl.load(LABELS + row)
    valid = target != -100
    selected = tl.load(LOGITS + row * V + target, valid, other=0).to(tl.float32)
    tl.store(CE + row, tl.where(valid, lse - selected, 0.0))
    tl.store(ENTROPY + row, log_total - moment / total)
    tl.store(PRED + row, arg.to(tl.int64))
    if valid:
        positions = tl.arange(0, RANK_BLOCK)
        preceding = tl.load(LABELS + positions, positions < ROWS, other=-100)
        rank = tl.sum(((positions < row) & (preceding != -100)).to(tl.int32), 0)
        for part in range(tl.cdiv(V, BLOCK)):
            col = part * BLOCK + columns
            value = tl.load(LOGITS + row * V + col, col < V, other=0).to(tl.float32)
            grad = (tl.exp(value - lse) - (col == target).to(tl.float32)) * 128.0
            tl.store(GRAD + rank * V + col, grad, col < V)


class OnlineGradientHead(torch.autograd.Function):
    """Compute token Jacobian while logits are live; save only O(tokens * hidden).

    Online one-pass forward computes CE, entropy, argmax, and logits gradients.
    Immediately projects hidden state gradient dx = dlogits @ w in forward pass.
    Backward directly returns pre-accumulated dx with ZERO matrix multiplications.
    """
    @staticmethod
    def forward(ctx, hidden, weight, targets, indices, chunk):
        x = hidden.reshape(-1, hidden.shape[-1]).index_select(0, indices).half()
        w = weight.half()
        rows, vocab = x.shape[0], w.shape[0]
        ce = torch.empty(rows, device=x.device, dtype=torch.float32)
        entropy = torch.empty_like(ce)
        pred = torch.empty(rows, device=x.device, dtype=torch.int64)
        dx = torch.zeros_like(x, dtype=torch.float32)
        labels = targets.index_select(0, indices).contiguous()
        supervised = (labels != -100).nonzero(as_tuple=True)[0]
        boundaries = torch.searchsorted(supervised, torch.arange(0, rows + chunk, chunk, device=x.device)).tolist()
        logits = torch.empty((min(rows, chunk), vocab), device=x.device, dtype=torch.float16)
        gradients = torch.empty_like(logits)
        for start in range(0, rows, chunk):
            stop = min(start + chunk, rows)
            count = stop - start
            torch.mm(x[start:stop], w.t(), out=logits[:count])
            online_vocabulary_gradient_kernel[(count,)](
                logits, labels[start:stop], ce[start:stop], entropy[start:stop], pred[start:stop],
                gradients, count, vocab, triton.next_power_of_2(chunk), BLOCK,
                num_warps=4, enable_fp_fusion=False)
            left, right = boundaries[start // chunk:start // chunk + 2]
            if right > left:
                backward_weight = column_weight(w) if COLUMN_BACKWARD and right - left >= 256 else w
                value = torch.mm(gradients[:right - left], backward_weight, out_dtype=torch.float32)
                dx.index_copy_(0, supervised[left:right], value)
        ctx.save_for_backward(dx, indices)
        ctx.shape, ctx.dtype = hidden.shape, hidden.dtype
        ctx.mark_non_differentiable(entropy, pred)
        return ce, entropy, pred

    @staticmethod
    @once_differentiable
    def backward(ctx, grad, _entropy, _pred):
        dx, indices = ctx.saved_tensors
        packed = (dx * (grad.float().reshape(-1, 1) / 128.0)).to(ctx.dtype)
        full = torch.zeros((ctx.shape[0] * ctx.shape[1], ctx.shape[-1]),
                           device=dx.device, dtype=ctx.dtype)
        full.index_copy_(0, indices, packed)
        return full.reshape(ctx.shape), None, None, None, None


def patch_online_head():
    """Hook OnlineGradientHead into stream_head.streamed_projection."""
    global ENABLED
    ENABLED = True
    import solution.stream_head as sh
    original_streamed_projection = getattr(sh, 'streamed_projection', None)
    if original_streamed_projection is None or getattr(original_streamed_projection, '_online_head_patched', False):
        return

    @wraps(original_streamed_projection)
    def online_streamed_projection(hidden, weight, labels, indices=None, chunk=128):
        if not ENABLED:
            return original_streamed_projection(hidden, weight, labels, indices=indices, chunk=chunk)
        b, t, _ = hidden.shape
        if indices is None:
            indices = torch.arange(b * t, device=hidden.device)
        targets = F.pad(labels[:, 1:], (0, 1), value=-100).reshape(-1)
        ce, entropy, pred = OnlineGradientHead.apply(hidden, weight, targets, indices, chunk)
        def restore(value):
            return value.new_zeros(b * t).index_copy(0, indices, value).reshape(b, t)
        return restore(ce), restore(entropy), restore(pred)

    online_streamed_projection._online_head_patched = True
    sh.streamed_projection = online_streamed_projection
