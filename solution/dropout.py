"""Packed LoRA dropout retaining the original full FP32 generator layout.

Native dropout still generates every original padded-shape mask. Its output
on ones supplies the exact FP32 multiplier; only active-token values and masks
are stored for backward. No random generator or seed is substituted.
"""
import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit(do_not_specialize=['N'])
def packed_dropout_forward_kernel(X, INDEX, SCALE, MASK, OUT, SAVED_MASK,
                                   N, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    row, col = i // WIDTH, i % WIDTH
    source = tl.load(INDEX + row, i < N, 0)
    position = source * WIDTH + col
    value = tl.load(X + i, i < N, 0).to(tl.float32)
    scale = tl.load(SCALE + position, i < N, 0)
    mask = tl.load(MASK + position, i < N, 0)
    tl.store(OUT + i, value * scale, i < N)
    tl.store(SAVED_MASK + i, mask, i < N)


@triton.jit(do_not_specialize=['N', 'SCALE'])
def packed_dropout_backward_kernel(GRAD, MASK, OUT, N, SCALE,
                                    BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    grad = tl.load(GRAD + i, i < N, 0).to(tl.float32)
    mask = tl.load(MASK + i, i < N, 0).to(tl.float32)
    tl.store(OUT + i, grad * mask * SCALE, i < N)


class PackedDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, indices, ones, probability):
        scales, mask = torch.ops.aten.native_dropout(ones, probability, True)
        out = torch.empty_like(x)
        saved = torch.empty(x.shape, device=x.device, dtype=torch.bool)
        packed_dropout_forward_kernel[(triton.cdiv(x.numel(),1024),)](
            x, indices, scales, mask, out, saved, x.numel(), x.shape[-1], 1024,
            enable_fp_fusion=False)
        ctx.save_for_backward(saved)
        ctx.scale = 1.0 / (1.0-probability)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (mask,) = ctx.saved_tensors
        grad = grad.contiguous()
        out = torch.empty_like(grad)
        packed_dropout_backward_kernel[(triton.cdiv(grad.numel(),1024),)](
            grad,mask,out,grad.numel(),ctx.scale,1024,enable_fp_fusion=False)
        return out,None,None,None


def packed_dropout(x, state, probability, training):
    if not training or probability == 0:
        return x
    if probability == 1:
        return x * 0
    shape = (*state[0], x.shape[-1])
    key = ('native_dropout_ones',shape,x.device)
    ones = state[4].get(key)
    if ones is None:
        ones = torch.ones(shape,device=x.device,dtype=torch.float32)
        state[4][key] = ones
    return PackedDropout.apply(x.contiguous(),state[1],ones,float(probability))
