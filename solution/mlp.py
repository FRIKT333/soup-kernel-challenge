"""Frozen NF4 MLP: one decode launch per direction, joined gate/up GEMM.

The SwiGLU kernels below derive from the shipped example and do not count as
independent work. The NF4 decoder is independent and restores exact values.
"""
import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable
from .nf4_decode import decode_mlp, supported


@triton.jit(do_not_specialize=['N'])
def example_joint_swiglu_fwd_kernel(GU, OUT, N, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos = i // WIDTH * (2 * WIDTH) + i % WIDTH
    gate = tl.load(GU + pos, i < N, 0).to(tl.float32)
    up = tl.load(GU + pos + WIDTH, i < N, 0).to(tl.float32)
    tl.store(OUT + i, gate * tl.sigmoid(gate) * up, i < N)


@triton.jit(do_not_specialize=['N'])
def example_joint_swiglu_bwd_kernel(DH, GU, DGU, N, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos = i // WIDTH * (2 * WIDTH) + i % WIDTH
    gate = tl.load(GU + pos, i < N, 0).to(tl.float32)
    up = tl.load(GU + pos + WIDTH, i < N, 0).to(tl.float32)
    grad = tl.load(DH + i, i < N, 0).to(tl.float32)
    sig = tl.sigmoid(gate)
    tl.store(DGU + pos, grad * up * sig * (1.0 + gate * (1.0 - sig)), i < N)
    tl.store(DGU + pos + WIDTH, grad * gate * sig, i < N)


class FrozenMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, up, down, states):
        h, width = states[0].shape[1], states[0].shape[0]
        dense = decode_mlp((gate, up, down), states)
        size = h * width
        gu = torch.nn.functional.linear(x.reshape(-1, h).half(), dense[:2*size].view(2*width, h))
        activated = torch.empty((gu.shape[0], width), device=x.device, dtype=torch.float16)
        n = activated.numel()
        example_joint_swiglu_fwd_kernel[(triton.cdiv(n, 1024),)](gu, activated, n, width, 1024)
        result = torch.nn.functional.linear(activated, dense[2*size:].view(h, width))
        ctx.save_for_backward(gu, gate, up, down)
        ctx.states, ctx.shape, ctx.dtype = states, x.shape, x.dtype
        return result.reshape(x.shape).to(x.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        gu, gate, up, down = ctx.saved_tensors
        h, width = ctx.states[0].shape[1], ctx.states[0].shape[0]
        size = h * width
        dense = decode_mlp((gate, up, down), ctx.states)
        dh = torch.mm(grad.reshape(-1, h).half(), dense[2*size:].view(h, width))
        dgu = torch.empty_like(gu)
        n = dh.numel()
        example_joint_swiglu_bwd_kernel[(triton.cdiv(n, 1024),)](dh, gu, dgu, n, width, 1024)
        dx = torch.mm(dgu, dense[:2*size].view(2*width, h)).to(ctx.dtype)
        return dx.reshape(ctx.shape), None, None, None, None


def eligible(module, x):
    import bitsandbytes as bnb
    from transformers.activations import SiLUActivation
    if (not x.is_cuda or x.dtype not in (torch.float16, torch.float32)
            or x.numel() == 0 or type(module.act_fn) not in (torch.nn.SiLU,SiLUActivation)
            or getattr(module.act_fn,'inplace',False) or 'forward' in module.act_fn.__dict__):
        return False
    if (module.act_fn._forward_hooks or module.act_fn._forward_pre_hooks
            or module.act_fn._backward_hooks or module.act_fn._backward_pre_hooks):
        return False
    if torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda') != torch.float16:
        return False
    projections = (module.gate_proj, module.up_proj, module.down_proj)
    for p in projections:
        if (type(p) is not bnb.nn.Linear4bit or p.weight.requires_grad or p.bias is not None
                or p.compute_dtype != torch.float16 or getattr(p, '_hf_hook', None) is not None
                or p.weight.device != x.device
                or ('forward' in p.__dict__ and not getattr(p, '_compact_forward', False))
                or p._forward_hooks or p._forward_pre_hooks or p._backward_hooks or p._backward_pre_hooks):
            return False
        state = p.weight.quant_state
        if not supported(state) or getattr(state, '_permit_frozen_cache', False):
            return False
    h, width = projections[0].weight.quant_state.shape[1], projections[0].weight.quant_state.shape[0]
    return (x.shape[-1] == h and tuple(projections[1].weight.quant_state.shape) == (width, h)
            and tuple(projections[2].weight.quant_state.shape) == (h, width))


def frozen_mlp(module, x):
    projections = (module.gate_proj, module.up_proj, module.down_proj)
    for p in projections:
        if not p.compute_type_is_set:
            p.set_compute_type(x.half())
            p.compute_type_is_set = True
    return FrozenMLP.apply(x, *(p.weight for p in projections),
                           tuple(p.weight.quant_state for p in projections))
