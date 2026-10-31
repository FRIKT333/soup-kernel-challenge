"""Reuse exact decoded NF4 values of frozen weights within the training process.

The original NF4 parameters and double-quantization state remain authoritative.
This trades measured VRAM for fewer dequantization launches; no FP16 source model
or prebuilt cache is used. Every decoded value comes from bitsandbytes at runtime.
"""

from functools import wraps

import torch
from torch.autograd.function import once_differentiable
from .nf4_decode import decode_nf4, supported


def _tensor_key(x):
    if not isinstance(x, torch.Tensor):
        return x
    return x.data_ptr(), x._version, x.dtype, tuple(x.shape)


def _decoded(a, packed, state, decode):
    nested = state.state2 if state.nested else None
    key = (_tensor_key(packed), _tensor_key(state.absmax), _tensor_key(state.offset),
           _tensor_key(nested.absmax) if nested is not None else None,
           _tensor_key(nested.code) if nested is not None else None,
           nested.blocksize if nested is not None else None,
           state.blocksize, state.quant_type, state.dtype, tuple(state.shape), a.dtype)
    cached = getattr(state, "_frozen_dense_cache", None)
    if cached is None or cached[0] != key:
        weight = decode(packed.view(-1, 1), state).to(a.dtype)
        state._frozen_dense_cache = (key, weight)
        return weight
    return cached[1]


class FrozenDecodedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(weight)
        ctx.shape = x.shape
        return torch.nn.functional.linear(x, weight, bias)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (weight,) = ctx.saved_tensors
        dx = torch.mm(grad.reshape(-1, grad.shape[-1]), weight).reshape(ctx.shape)
        return dx, None, None


class FusedNF4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, state, bias):
        result = torch.nn.functional.linear(x, decode_nf4(packed, state), bias)
        ctx.save_for_backward(packed)
        ctx.state, ctx.shape = state, x.shape
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (packed,) = ctx.saved_tensors
        weight = decode_nf4(packed, ctx.state)
        dx = torch.mm(grad.reshape(-1, grad.shape[-1]), weight).reshape(ctx.shape)
        return dx, None, None, None


def install_nf4_cache(model):
    if not isinstance(model, torch.nn.Module):
        return
    import bitsandbytes as bnb

    for module in model.modules():
        if type(module) is not bnb.nn.Linear4bit or module.weight.requires_grad:
            continue
        state = module.weight.quant_state
        if state is not None and state.quant_type == "nf4" and state.dtype == torch.float16:
            state._permit_frozen_cache = True
            state._permit_fused_decode = supported(state)


def patch_nf4_cache():
    import bitsandbytes as bnb

    original = bnb.matmul_4bit
    if getattr(original, "_frozen_cache_patched", False):
        return
    decode = bnb.functional.dequantize_4bit

    @wraps(original)
    def matmul(a, b, quant_state, out=None, bias=None):
        cached = getattr(quant_state, "_permit_frozen_cache", False)
        fused = getattr(quant_state, "_permit_fused_decode", False)
        if (not (cached or fused) or not a.is_cuda
                or a.dtype != torch.float16 or a.numel() == 0 or b.requires_grad
                or out is not None or (bias is not None and bias.requires_grad)
                or a.shape[-1] != quant_state.shape[1]):
            return original(a, b, quant_state, out=out, bias=bias)
        if not cached:
            return FusedNF4Linear.apply(a, b, quant_state, bias)
        weight = _decoded(a, b, quant_state, decode)
        return FrozenDecodedLinear.apply(a, weight, bias)

    matmul._frozen_cache_patched = True
    bnb.matmul_4bit = matmul
