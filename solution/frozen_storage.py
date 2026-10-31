"""Lossless storage of a frozen, tied embedding/classifier matrix.

Only matrices whose every FP32 value round-trips through FP16 exactly qualify.
FP32 embedding outputs and non-AMP classifier calls reconstruct their original
values/dtype. Parameter identity, tying, NF4 weights and LoRA remain unchanged.
Install once during setup, before any training forward.
"""

from functools import wraps
from types import MethodType

import torch
import torch.nn.functional as F


def _compress_pair(embedding, head):
    if (type(embedding) is not torch.nn.Embedding or type(head) is not torch.nn.Linear
            or head.bias is not None or embedding.weight is not head.weight):
        return 0
    weight = embedding.weight
    if (weight.requires_grad or weight.dtype != torch.float32 or not weight.is_cuda
            or not weight.is_contiguous()):
        return 0
    if any(m._forward_pre_hooks or m._backward_hooks or m._backward_pre_hooks
           for m in (embedding, head)) or head._forward_hooks:
        return 0
    # HF may require gradients on the embedding OUTPUT for k-bit training.
    # That outer Module hook still fires once, with the same output dtype.
    if any(getattr(hook, "__module__", "") != "transformers.modeling_utils"
           or "enable_input_require_grads" not in getattr(hook, "__qualname__", "")
           for hook in embedding._forward_hooks.values()):
        return 0
    original_embedding = getattr(embedding.forward, "__func__", None)
    original_head = getattr(head.forward, "__func__", None)
    if original_embedding is None or original_head is None:
        return 0
    with torch.no_grad():
        half = weight.half()
        if not torch.equal(weight, half.float()):
            return 0
        # Both modules retain the same Parameter object, with exact values.
        weight.data = half

    @wraps(original_embedding)
    def embed(self, *args, **kwargs):
        from .precision import half_activations
        value = original_embedding(self, *args, **kwargs)
        return value.half() if half_activations.get() else value.float()

    @wraps(original_head)
    def project(self, *args, **kwargs):
        # Outside AMP, restore the original FP32 weight contract, including
        # the original error for a mismatched input dtype.
        if not torch.is_autocast_enabled("cuda"):
            if len(args) == 1 and not kwargs:
                return F.linear(args[0], self.weight.float())
            if not args and len(kwargs) == 1:
                value = kwargs.get("input", kwargs.get("hidden"))
                if isinstance(value, torch.Tensor):
                    return F.linear(value, self.weight.float())
        return original_head(self, *args, **kwargs)

    embedding.forward = MethodType(embed, embedding)
    head.forward = MethodType(project, head)
    embedding._lossless_half_storage = head._lossless_half_storage = True
    return 1


def install_frozen_embedding_storage(model):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    changed = 0
    for module in model.modules():
        if (not isinstance(module, Qwen2ForCausalLM)
                or not getattr(module, "_half_activation_storage", False)
                or not getattr(module, "_streamed_head", False)
                or module.config.hidden_size != 1536 or module.config.vocab_size != 151936):
            continue
        changed += _compress_pair(module.get_input_embeddings(), module.get_output_embeddings())
    return changed
