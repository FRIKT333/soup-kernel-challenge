"""Remove padding inside each batch's pointwise/linear blocks, never across batches.

Attention still receives its original padded tensors and mask. LoRA dropout still
sees the full tensor so the PyTorch generator consumes exactly the original masks.
"""

from contextvars import ContextVar
from functools import wraps
from types import MethodType

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


_tokens = ContextVar("active_batch_tokens", default=None)
_adapter_full_input = ContextVar("adapter_full_dropout_input", default=None)


@triton.jit
def token_gather_kernel(X, INDEX, OUT, D, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source = tl.load(INDEX + row)
    value = tl.load(X + source * D + col, col < D, 0)
    tl.store(OUT + row * D + col, value, col < D)


@triton.jit
def token_restore_kernel(X, INVERSE, OUT, D, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source = tl.load(INVERSE + row)
    value = tl.load(X + source * D + col, (col < D) & (source >= 0), 0)
    tl.store(OUT + row * D + col, value, col < D)


def _gather(x, indices):
    x = x.reshape(-1, x.shape[-1]).contiguous()
    out = torch.empty((indices.numel(), x.shape[-1]), device=x.device, dtype=x.dtype)
    token_gather_kernel[(indices.numel(), triton.cdiv(x.shape[-1], 1024))](
        x, indices, out, x.shape[-1], 1024)
    return out


def _scatter(x, inverse, shape):
    x = x.contiguous()
    out = torch.empty((*shape, x.shape[-1]), device=x.device, dtype=x.dtype)
    token_restore_kernel[(inverse.numel(), triton.cdiv(x.shape[-1], 1024))](
        x, inverse, out, x.shape[-1], 1024)
    return out


class TokenPack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, indices, inverse, shape):
        ctx.save_for_backward(inverse)
        ctx.shape = shape
        return _gather(x, indices)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (inverse,) = ctx.saved_tensors
        return _scatter(grad, inverse, ctx.shape), None, None, None


class TokenRestore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, indices, inverse, shape):
        ctx.save_for_backward(indices)
        return _scatter(x, inverse, shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (indices,) = ctx.saved_tensors
        return _gather(grad, indices), None, None, None


def _can_compact(x, state):
    return (state is not None and x.ndim == 3 and x.shape[:2] == state[0]
            and x.device == state[1].device)


def _pack(x, state):
    normalized = getattr(x, "_packed_normalization", None)
    if normalized is not None and normalized[0] is state[1]:
        return normalized[1]
    return TokenPack.apply(x, state[1], state[2], state[0])


def _restore(x, state):
    return TokenRestore.apply(x, state[1], state[2], state[0])


def _linear_forward(original):
    original = original.__func__

    @wraps(original)
    def forward(self, x, *args, **kwargs):
        state = _tokens.get()
        if not _can_compact(x, state) or args or kwargs or self.weight.requires_grad:
            return original(self, x, *args, **kwargs)
        return _restore(original(self, _pack(x, state)), state)

    return forward


def install_compaction(model):
    """Register only real Qwen modules and frozen base projections after setup."""
    if not isinstance(model, torch.nn.Module):
        return
    import bitsandbytes as bnb
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    names = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if any(hasattr(module, "lora_A") and name.rsplit(".", 1)[-1] not in ("q_proj", "v_proj")
           for name, module in model.named_modules()):
        return
    for name, module in model.named_modules():
        if isinstance(module, Qwen2ForCausalLM):
            module._compact_batch_tokens = True
        pieces = name.split(".")
        leaf = pieces[-2] if pieces[-1] == "base_layer" and len(pieces) > 1 else pieces[-1]
        is_base = type(module) is bnb.nn.Linear4bit and leaf in names
        is_head = type(module) is torch.nn.Linear and leaf == "lm_head"
        if not (is_base or is_head) or module.weight.requires_grad:
            continue
        if getattr(module, "_compact_forward", False) or "forward" in module.__dict__:
            continue
        module.forward = MethodType(_linear_forward(module.forward), module)
        module._compact_forward = True


def patch_compaction():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2MLP

    original = Qwen2ForCausalLM.forward
    if getattr(original, "_compact_forward", False):
        return

    @wraps(original)
    def forward(self, *args, **kwargs):
        mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
        labels = kwargs.get("labels")
        if (not getattr(self, "_compact_batch_tokens", False) or not self.training
                or not isinstance(mask, torch.Tensor) or not mask.is_cuda or mask.ndim != 2
                or not isinstance(labels, torch.Tensor) or labels.shape != mask.shape):
            return original(self, *args, **kwargs)
        active = mask != 0
        # Masked positions must not have a supervised target. Keep the original
        # route for any caller whose labels intentionally supervise masked tokens.
        unsafe_target = ((~active) & (labels != -100)).any()
        unsafe_query = ((~active[:, :-1]) & (labels[:, 1:] != -100)).any()
        if (unsafe_target | unsafe_query).item():
            return original(self, *args, **kwargs)
        indices = active.reshape(-1).nonzero(as_tuple=True)[0]
        if indices.numel() == 0 or indices.numel() == active.numel():
            return original(self, *args, **kwargs)
        ranks = active.reshape(-1).cumsum(0, dtype=torch.int32) - 1
        inverse = torch.where(active.reshape(-1), ranks, -1).contiguous()
        counts = active.sum(dim=1).tolist()
        local_indices, start = [], 0
        for batch, count in enumerate(counts):
            local_indices.append(indices[start : start + count] - batch * mask.shape[1])
            start += count
        token = _tokens.set((tuple(mask.shape), indices, inverse, local_indices, {}))
        try:
            return original(self, *args, **kwargs)
        finally:
            _tokens.reset(token)

    forward._compact_forward = True
    Qwen2ForCausalLM.forward = forward
    original_mlp = Qwen2MLP.forward

    @wraps(original_mlp)
    def mlp(self, x):
        state = _tokens.get()
        from .packed_decoder import _packed_decoder
        if state is not None and _packed_decoder.get() is state and x.ndim == 2:
            from .mlp import eligible, frozen_mlp
            return frozen_mlp(self,x) if eligible(self,x) else original_mlp(self,x)
        if not _can_compact(x, state):
            return original_mlp(self, x)
        # Pack once around the whole MLP, keeping gate/up/SwiGLU/down compact.
        packed = _pack(x, state)
        from .mlp import eligible, frozen_mlp
        if eligible(self, packed):
            return _restore(frozen_mlp(self, packed), state)
        import bitsandbytes as bnb
        projections = (self.gate_proj, self.up_proj, self.down_proj)
        if (x.dtype in (torch.float16, torch.float32) and isinstance(self.act_fn, torch.nn.SiLU)
                and all(type(p) is bnb.nn.Linear4bit and p.compute_dtype == torch.float16
                        and not p.weight.requires_grad for p in projections)):
            from .swiglu import swiglu
            # These casts have separate autograd edges: the two input gradients
            # are promoted to FP32 before being added, as in the original graph.
            gate = self.gate_proj(packed.to(torch.float16))
            up = self.up_proj(packed.to(torch.float16))
            # Original gate/up outputs are FP16 GEMMs promoted to FP32. The
            # example computes SwiGLU in FP32 and down immediately casts to FP16.
            # Store that final FP16 value directly and restore the small output.
            result = self.down_proj(swiglu(gate, up)).to(x.dtype)
        else:
            result = original_mlp(self, packed)
        return _restore(result, state)

    Qwen2MLP.forward = mlp
