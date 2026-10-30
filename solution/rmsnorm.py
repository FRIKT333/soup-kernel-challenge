"""Fused Qwen2RMSNorm forward and backward in Triton.

Eliminates multiple elementwise PyTorch kernel launches per normalization layer,
performing mean-square reduction, rsqrt, scaling, and backward gradient computation
entirely within registers in float32 precision.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_rmsnorm_fwd_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    r_ptr,
    stride_x,
    stride_y,
    D: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    x = tl.load(x_ptr + row_idx * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / D
    r = tl.rsqrt(var + eps)
    tl.store(r_ptr + row_idx, r)

    y = (x * r) * w
    tl.store(y_ptr + row_idx * stride_y + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def fused_rmsnorm_bwd_kernel(
    dy_ptr,
    x_ptr,
    w_ptr,
    r_ptr,
    dx_ptr,
    stride_dy,
    stride_x,
    stride_dx,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    dy = tl.load(dy_ptr + row_idx * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row_idx * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row_idx)

    dy_w = dy * w
    c = tl.sum(dy_w * x, axis=0)
    dx = r * (dy_w - (r * r / D) * x * c)

    tl.store(dx_ptr + row_idx * stride_dx + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


class FusedRMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        M, D = x_2d.shape
        y_2d = torch.empty_like(x_2d)
        r = torch.empty(M, device=x.device, dtype=torch.float32)

        BLOCK_SIZE = triton.next_power_of_2(D)
        num_warps = 8 if D >= 1024 else 4
        fused_rmsnorm_fwd_kernel[(M,)](
            x_2d,
            weight,
            y_2d,
            r,
            x_2d.stride(0),
            y_2d.stride(0),
            D=D,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.save_for_backward(x_2d, weight, r)
        ctx.orig_shape = orig_shape
        ctx.D = D
        ctx.eps = eps
        return y_2d.reshape(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x_2d, weight, r = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        dy_2d = dy.reshape(-1, orig_shape[-1]).contiguous()
        M, D = x_2d.shape
        dx_2d = torch.empty_like(x_2d)

        BLOCK_SIZE = triton.next_power_of_2(D)
        num_warps = 8 if D >= 1024 else 4
        fused_rmsnorm_bwd_kernel[(M,)](
            dy_2d,
            x_2d,
            weight,
            r,
            dx_2d,
            dy_2d.stride(0),
            x_2d.stride(0),
            dx_2d.stride(0),
            D=D,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        dw = None
        if ctx.needs_input_grad[1]:
            dw = (dy_2d.float() * (x_2d.float() * r.unsqueeze(1))).sum(dim=0).to(weight.dtype)
        return dx_2d.reshape(orig_shape), dw, None


def fused_rmsnorm(x, weight, eps=1e-6):
    return FusedRMSNormFunc.apply(x, weight, eps)


def patch_qwen2_rmsnorm():
    """Patch Qwen2RMSNorm.forward to use the fused Triton kernel on CUDA."""
    from transformers.models.qwen2 import modeling_qwen2

    original_forward = modeling_qwen2.Qwen2RMSNorm.forward

    def forward(self, hidden_states):
        if not getattr(hidden_states, "is_cuda", False) or hidden_states.shape[-1] > 2048:
            return original_forward(self, hidden_states)
        return FusedRMSNormFunc.apply(hidden_states, self.weight, self.variance_epsilon)

    modeling_qwen2.Qwen2RMSNorm.forward = forward
