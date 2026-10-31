"""One-pass RMSNorm forward/backward for Qwen's frozen norm weights."""

from functools import wraps

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def frozen_rms_forward_kernel(X, W, Y, INV, MAP, PACKED, N: tl.constexpr,
                              EPS: tl.constexpr, BLOCK: tl.constexpr, PACK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + cols, cols < N, 0).to(tl.float32)
    w = tl.load(W + cols, cols < N, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / N + EPS)
    # Qwen casts the normalized value back BEFORE multiplying by its FP32 weight.
    normalized = (x * inv).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y + row * N + cols, w * normalized, cols < N)
    if PACK:
        target = tl.load(MAP + row)
        tl.store(PACKED + target * N + cols, w * normalized, (cols < N) & (target >= 0))
    tl.store(INV + row, inv)


@triton.jit
def frozen_rms_backward_kernel(G, GP, X, W, INV, MAP, DX, N: tl.constexpr,
                               BLOCK: tl.constexpr, FULL: tl.constexpr, PACK: tl.constexpr,
                               ROUND_COMBINED: tl.constexpr = False):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + cols, cols < N, 0).to(tl.float32)
    g = tl.full((BLOCK,), 0, tl.float32)
    if FULL:
        g += tl.load(G + row * N + cols, cols < N, 0).to(tl.float32)
    if PACK:
        source = tl.load(MAP + row)
        g += tl.load(GP + source * N + cols, (cols < N) & (source >= 0), 0).to(tl.float32)
    # Autograd sums gradients in the forward output dtype before the norm.
    if ROUND_COMBINED and FULL and PACK:
        g = g.to(X.dtype.element_ty).to(tl.float32)
    w = tl.load(W + cols, cols < N, 0).to(tl.float32)
    inv = tl.load(INV + row)
    gw = (g * w).to(X.dtype.element_ty).to(tl.float32)
    correction = tl.sum(gw * x, axis=0) / N
    dx = gw * inv - x * (inv * inv * inv * correction)
    tl.store(DX + row * N + cols, dx, cols < N)


class FrozenRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        x = x.contiguous()
        n = x.shape[-1]
        inv = torch.empty((x.numel() // n,), device=x.device, dtype=torch.float32)
        from .precision import half_activations
        dtype = x.dtype if half_activations.get() else torch.promote_types(x.dtype, weight.dtype)
        y = torch.empty_like(x, dtype=dtype)
        frozen_rms_forward_kernel[(inv.numel(),)](
            x, weight, y, inv, x, x, n, float(eps), triton.next_power_of_2(n), False,
            num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(x, weight, inv)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, weight, inv = ctx.saved_tensors
        grad = grad.contiguous()
        dx = torch.empty_like(x)
        frozen_rms_backward_kernel[(inv.numel(),)](
            grad, grad, x, weight, inv, x, dx, x.shape[-1], triton.next_power_of_2(x.shape[-1]), True, False,
            num_warps=4, enable_fp_fusion=False)
        return dx, None, None


class PackedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, indices, inverse):
        x = x.contiguous()
        n = x.shape[-1]
        inv = torch.empty((x.numel() // n,), device=x.device, dtype=torch.float32)
        from .precision import half_activations
        dtype = x.dtype if half_activations.get() else torch.promote_types(x.dtype, weight.dtype)
        full = torch.empty_like(x, dtype=dtype)
        packed = torch.empty((indices.numel(), n), device=x.device, dtype=dtype)
        frozen_rms_forward_kernel[(inv.numel(),)](
            x, weight, full, inv, inverse, packed, n, float(eps), triton.next_power_of_2(n), True,
            num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(x, weight, inv, inverse)
        ctx.round_combined = dtype == torch.float16
        ctx.set_materialize_grads(False)
        return full, packed

    @staticmethod
    @once_differentiable
    def backward(ctx, full_grad, packed_grad):
        x, weight, inv, inverse = ctx.saved_tensors
        has_full, has_packed = full_grad is not None, packed_grad is not None
        g = full_grad.contiguous() if has_full else x
        gp = packed_grad.contiguous() if has_packed else x
        dx = torch.empty_like(x)
        frozen_rms_backward_kernel[(inv.numel(),)](
            g, gp, x, weight, inv, inverse, dx, x.shape[-1], triton.next_power_of_2(x.shape[-1]),
            has_full, has_packed, ctx.round_combined, num_warps=4, enable_fp_fusion=False)
        return dx, None, None, None, None


def patch_qwen2_rmsnorm():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    original = Qwen2RMSNorm.forward
    if getattr(original, "_frozen_rms_patched", False):
        return

    @wraps(original)
    def forward(self, hidden_states):
        if (not hidden_states.is_cuda or hidden_states.numel() == 0
                or hidden_states.shape[-1] != 1536 or self.weight.requires_grad
                or self.weight.dtype != torch.float32 or not self.weight.is_contiguous()
                or hidden_states.dtype not in (torch.float16, torch.float32)):
            return original(self, hidden_states)
        from .compact import _tokens
        state = _tokens.get()
        if state is not None and hidden_states.ndim == 3 and hidden_states.shape[:2] == state[0]:
            full, packed = PackedRMSNorm.apply(hidden_states, self.weight, self.variance_epsilon,
                                               state[1], state[2])
            full._packed_normalization = (state[1], packed)
            return full
        return FrozenRMSNorm.apply(hidden_states, self.weight, self.variance_epsilon)

    forward._frozen_rms_patched = True
    Qwen2RMSNorm.forward = forward
