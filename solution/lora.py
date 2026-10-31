"""Narrow per-instance PEFT hook, installed after Soup finishes model setup."""

from functools import wraps
from types import MethodType

import torch


def _configuration(module, x, args, kwargs):
    """Return an adapter only when its ordinary Linear semantics are supported."""
    if args or kwargs or not x.is_cuda or x.numel() == 0:
        return None
    if module.disable_adapters or module.merged or len(module.active_adapters) != 1:
        return None
    adapter = module.active_adapters[0]
    if adapter not in module.lora_A or adapter in module.lora_variant:
        return None
    a, b = module.lora_A[adapter], module.lora_B[adapter]
    dropout = module.lora_dropout[adapter]
    if type(a) is not torch.nn.Linear or type(b) is not torch.nn.Linear:
        return None
    if a.bias is not None or b.bias is not None:
        return None
    if type(dropout) not in (torch.nn.Dropout, torch.nn.Identity):
        return None
    if getattr(dropout, "inplace", False):
        return None
    width = x.shape[-1]
    if a.weight.shape != (16, width) or b.weight.shape[1] != 16 or b.weight.shape[0] > 1536:
        return None
    if width > 1536 or x.numel() // width > 2048:
        return None
    if x.dtype not in (torch.float16, torch.float32):
        return None
    if any(w.device != x.device or w.dtype not in (torch.float16, torch.float32)
           for w in (a.weight, b.weight)):
        return None
    weight = module.get_base_layer().weight
    if getattr(weight, "quant_type", None) != "nf4":
        return None
    # Module hooks/custom Linear subclasses need the original calls to fire.
    if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks or m._backward_pre_hooks
           for m in (a, b)):
        return None
    if torch.is_autocast_enabled("cuda"):
        if torch.get_autocast_dtype("cuda") != torch.float16:
            return None
    elif a.weight.dtype != torch.float16 or b.weight.dtype != torch.float16:
        return None
    if not getattr(module, "cast_input_dtype_enabled", True) and x.dtype != a.weight.dtype:
        return None
    return adapter


def _make_forward(original):
    # Keep the unbound method so copying a patched model cannot call the old
    # instance's weights on a fallback path.
    original = original.__func__

    @wraps(original)
    def forward(self, x, *args, **kwargs):
        adapter = _configuration(self, x, args, kwargs)
        if adapter is None:
            return original(self, x, *args, **kwargs)
        self._check_forward_args(x)
        a, b = self.lora_A[adapter], self.lora_B[adapter]
        # Keep the full input-gradient path through the frozen NF4 base branch.
        base = self.base_layer(x).clone()
        autocast = torch.is_autocast_enabled("cuda")
        if not autocast:
            x = self._cast_input_dtype(x, a.weight.dtype)
        # Dropout precedes the AMP casts, exactly as in PEFT. Each module consumes
        # its own fresh mask from the ordinary PyTorch generator.
        from .compact import _adapter_full_input, _tokens, _pack
        full = _adapter_full_input.get()
        state = _tokens.get()
        from .precision import half_activations
        dropout = self.lora_dropout[adapter]
        from .packed_decoder import _packed_decoder
        packed_native = (half_activations.get() and state is not None
                         and (full is not None or _packed_decoder.get() is state)
                         and x.ndim == 2 and x.dtype == torch.float16
                         and type(dropout) in (torch.nn.Dropout, torch.nn.Identity)
                         and not (dropout._forward_hooks or dropout._forward_pre_hooks
                                  or dropout._backward_hooks or dropout._backward_pre_hooks))
        if packed_native:
            from .dropout import packed_dropout
            u = packed_dropout(x,state,getattr(dropout,'p',0.0),dropout.training)
        else:
            if half_activations.get():
                # Keep FP32 dropout RNG layout despite compact storage.
                if full is not None:
                    full = full.float()
                else:
                    x = x.float()
            if full is not None and state is not None and x.ndim == 2:
                if not autocast:
                    full = self._cast_input_dtype(full, a.weight.dtype)
                u = _pack(dropout(full), state)
            else:
                u = dropout(x)
        dtype = torch.float16 if autocast else a.weight.dtype
        from .lora_kernels import rank_split_lora

        return rank_split_lora(base, u.to(dtype), a.weight.to(dtype), b.weight.to(dtype),
                               self.scaling[adapter], convert=not autocast)

    return forward


def patch_model(model):
    if not isinstance(model, torch.nn.Module):
        return 0
    from peft.tuners.lora.bnb import Linear4bit

    count = 0
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] not in ("q_proj", "v_proj"):
            continue
        if type(module) is not Linear4bit or getattr(module, "_rank_split_patched", False):
            continue
        if "forward" in module.__dict__:
            continue
        module.forward = MethodType(_make_forward(module.forward), module)
        module._rank_split_patched = True
        count += 1
    return count


def patch_soup_lora():
    from soup_cli.trainer.sft import SFTTrainerWrapper

    original = SFTTrainerWrapper._setup_transformers
    if getattr(original, "_rank_split_patched", False):
        return

    @wraps(original)
    def setup(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        patch_model(getattr(self, "model", None))
        from .compact import install_compaction
        install_compaction(getattr(self, "model", None))
        from .nf4_cache import install_nf4_cache
        install_nf4_cache(getattr(self, "model", None))
        from .precision import install_activation_precision
        if isinstance(getattr(self, "model", None), torch.nn.Module):
            install_activation_precision(self.model)
            from .stream_head import install_stream_head
            install_stream_head(self.model)
            from .frozen_storage import install_frozen_embedding_storage
            install_frozen_embedding_storage(self.model)
            from .schedule import choose_nf4_cache
            choose_nf4_cache(self.model)
        return result

    setup._rank_split_patched = True
    SFTTrainerWrapper._setup_transformers = setup
