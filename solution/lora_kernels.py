"""LoRA fusion at the rank boundary; GEMMs stay in cuBLAS.

Idea credit: LoRAFusion, section 5.1, https://arxiv.org/abs/2510.00206.
All kernels below are independently implemented here.
"""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit(do_not_specialize=["M"])
def lora_up_add_kernel(H, B, BASE, OUT, M, N: tl.constexpr,
                       RANK: tl.constexpr, SCALE: tl.constexpr,
                       CONVERT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    rank = tl.arange(0, RANK)
    h = tl.load(H + rows[:, None] * RANK + rank[None, :], rows[:, None] < M, 0)
    b = tl.load(B + cols[None, :] * RANK + rank[:, None], cols[None, :] < N, 0)
    up = tl.dot(h, b, input_precision="ieee")
    up = up.to(H.dtype.element_ty).to(tl.float32)
    scaled = (up * SCALE).to(H.dtype.element_ty)
    if CONVERT:
        scaled = scaled.to(BASE.dtype.element_ty)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    base = tl.load(BASE + rows[:, None] * N + cols[None, :], mask, 0)
    tl.store(OUT + rows[:, None] * N + cols[None, :],
             base.to(tl.float32) + scaled.to(tl.float32), mask)


@triton.jit
def lora_cast_scale_gradient_kernel(G, OUT, SIZE, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(G + offsets, offsets < SIZE, 0).to(OUT.dtype.element_ty)
    # Match backward through the output cast, then the scalar multiplication.
    scaled = g.to(tl.float32) * SCALE
    tl.store(OUT + offsets, scaled, offsets < SIZE)


class RankSplitLoRA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, u, a, b, scale, convert):
        shape = u.shape
        u = u.reshape(-1, a.shape[1]).contiguous()
        base_flat = base.reshape(-1, b.shape[0]).contiguous()
        a, b = a.contiguous(), b.contiguous()
        h = torch.mm(u, a.t())
        # Allocate fresh output buffer with torch.empty; avoid redundant base.clone()
        dtype = base.dtype if convert else torch.promote_types(base.dtype, h.dtype)
        out = torch.empty(base.shape, device=base.device, dtype=dtype)
        lora_up_add_kernel[(triton.cdiv(u.shape[0], 32), triton.cdiv(b.shape[0], 128))](
            h, b, base_flat, out, u.shape[0], b.shape[0], a.shape[0],
            float(scale), convert, 32, 128, num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(u, a, b, h)
        ctx.scale, ctx.input_shape, ctx.base_dtype = float(scale), shape, base.dtype
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        u, a, b, h = ctx.saved_tensors
        source = grad.reshape(-1, b.shape[0]).contiguous()
        g = torch.empty(source.shape, device=source.device, dtype=h.dtype)
        lora_cast_scale_gradient_kernel[(triton.cdiv(g.numel(), 1024),)](
            source, g, g.numel(), ctx.scale, 1024)
        need_base, need_u, need_a, need_b = ctx.needs_input_grad[:4]
        # These are compute-bound GEMMs: cuBLAS avoids the extra partial tensors
        # and reduction launches of the initial all-Triton implementation.
        db = torch.mm(g.t(), h) if need_b else None
        du = da = None
        if need_u or need_a:
            dh = torch.mm(g, b)
            if need_a:
                da = torch.mm(dh.t(), u)
            if need_u:
                du = torch.mm(dh, a).reshape(ctx.input_shape)
        return grad.to(ctx.base_dtype) if need_base else None, du, da, db, None, None


def rank_split_lora(base, u, a, b, scale, convert=True):
    """Inputs use the actual Linear compute dtype; casts stay in the PyTorch graph."""
    return RankSplitLoRA.apply(base, u, a, b, scale, convert)
