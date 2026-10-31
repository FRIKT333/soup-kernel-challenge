"""Keep q/k/v and rotary compact, preserve full-tensor LoRA dropout, use stock SDPA."""

from functools import wraps

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from .compact import _tokens, _restore, _pack, _adapter_full_input


@triton.jit(do_not_specialize=['TOTAL'])
def packed_rotary_kernel(X, COS, SIN, OUT, TOTAL, H: tl.constexpr,
                          D: tl.constexpr, BACKWARD: tl.constexpr, BLOCK: tl.constexpr,
                          ROUND_PRODUCTS: tl.constexpr = False):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = off // (H * D)
    dim = off % D
    other_dim = (dim + D // 2) % D
    other_off = (off // D) * D + other_dim
    x = tl.load(X + off, off < TOTAL, 0).to(tl.float32)
    rotated = tl.load(X + other_off, off < TOTAL, 0).to(tl.float32)
    c = tl.load(COS + token * D + dim, off < TOTAL, 0)
    if BACKWARD:
        s = tl.load(SIN + token * D + other_dim, off < TOTAL, 0)
        rotated = tl.where(dim < D // 2, rotated, -rotated)
    else:
        s = tl.load(SIN + token * D + dim, off < TOTAL, 0)
        rotated = tl.where(dim < D // 2, -rotated, rotated)
    left, right = x * c, rotated * s
    if ROUND_PRODUCTS:
        # Torch evaluates each Half multiplication before the Half sum.
        # Both casts are required in backward as well as in forward.
        left = left.to(tl.float16).to(tl.float32)
        right = right.to(tl.float16).to(tl.float32)
    tl.store(OUT + off, left + right, off < TOTAL)


class PackedRotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        x = x.contiguous()
        out = torch.empty_like(x)
        packed_rotary_kernel[(triton.cdiv(x.numel(), 1024),)](
            x, cos, sin, out, x.numel(), x.shape[1], x.shape[2], False, 1024,
            ROUND_PRODUCTS=x.dtype == torch.float16 and cos.dtype == torch.float16,
            enable_fp_fusion=False)
        ctx.save_for_backward(cos, sin)
        ctx.h, ctx.d = x.shape[1:]
        ctx.round_products = x.dtype == torch.float16 and cos.dtype == torch.float16
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        cos, sin = ctx.saved_tensors
        grad = grad.contiguous()
        out = torch.empty_like(grad)
        packed_rotary_kernel[(triton.cdiv(grad.numel(), 1024),)](
            grad, cos, sin, out, grad.numel(), ctx.h, ctx.d, True, 1024,
            ROUND_PRODUCTS=ctx.round_products,
            enable_fp_fusion=False)
        return out, None, None


def patch_attention_compaction():
    import bitsandbytes as bnb
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    from .lora import _configuration

    original = Qwen2Attention.forward
    if getattr(original, "_token_attention_patched", False):
        return

    @wraps(original)
    def attention(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        state = _tokens.get()
        from .packed_decoder import _packed_decoder
        across_layers = state is not None and _packed_decoder.get() is state
        cos, sin = position_embeddings
        supported = (state is not None and self.training and self.config._attn_implementation == "sdpa"
                     and self.attention_dropout == 0.0 and past_key_values is None
                     and not kwargs.get("output_attentions", False)
                     and hidden_states.dtype in (torch.float16, torch.float32)
                     and cos.dtype == hidden_states.dtype and sin.dtype == cos.dtype
                     and not cos.requires_grad and not sin.requires_grad
                     and (self.sliding_window is None or self.sliding_window >=
                          (state[0][1] if across_layers else hidden_states.shape[1])))
        if supported:
            for projection in (self.q_proj, self.v_proj):
                if (not getattr(projection, "_rank_split_patched", False)
                        or _configuration(projection, hidden_states, (), {}) is None):
                    supported = False
            for projection in (self.k_proj, self.o_proj):
                if type(projection) is not bnb.nn.Linear4bit or projection.weight.requires_grad:
                    supported = False
        if not supported:
            return original(self, hidden_states, position_embeddings, attention_mask,
                            past_key_values=past_key_values, **kwargs)
        packed = hidden_states if across_layers else _pack(hidden_states, state)
        token = _adapter_full_input.set(None if across_layers else hidden_states)
        try:
            q, k, v = self.q_proj(packed), self.k_proj(packed), self.v_proj(packed)
        finally:
            _adapter_full_input.reset(token)
        n = packed.shape[0]
        q = q.reshape(n, -1, self.head_dim)
        k = k.reshape(n, -1, self.head_dim)
        v = v.reshape(n, -1, self.head_dim)
        cache = state[4]
        key = (cos.data_ptr(), cos._version, sin.data_ptr(), sin._version)
        if cache.get("rotary_key") != key:
            b, t = state[0]
            cache["rotary"] = (_pack(cos.expand(b, t, -1), state),
                               _pack(sin.expand(b, t, -1), state))
            cache["rotary_key"] = key
        packed_cos, packed_sin = cache["rotary"]
        q = PackedRotary.apply(q, packed_cos, packed_sin)
        k = PackedRotary.apply(k, packed_cos, packed_sin)
        pieces, start = [], 0
        for indices in state[3]:
            count = indices.numel()
            if count == 0:
                continue
            qi = q[start : start + count].transpose(0, 1).unsqueeze(0)
            ki = k[start : start + count].transpose(0, 1).unsqueeze(0)
            vi = v[start : start + count].transpose(0, 1).unsqueeze(0)
            if self.num_key_value_groups != 1:
                ki = ki.repeat_interleave(self.num_key_value_groups, dim=1)
                vi = vi.repeat_interleave(self.num_key_value_groups, dim=1)
            result = torch.nn.functional.scaled_dot_product_attention(
                qi, ki, vi, dropout_p=0.0, is_causal=True, scale=self.scaling)
            pieces.append(result.transpose(1, 2).reshape(count, -1))
            start += count
        result = self.o_proj(torch.cat(pieces, dim=0))
        return (result if across_layers else _restore(result, state)), None

    attention._token_attention_patched = True
    Qwen2Attention.forward = attention
