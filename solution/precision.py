"""FP16 activation storage with FP32 norm arithmetic, frozen parameters intact.

Only the task's training AMP path uses compact activation storage. Inference and
evaluation retain normal outputs. No model/adapter weight dtype is changed.
"""
from contextvars import ContextVar
from functools import wraps
from types import MethodType
import torch

half_activations = ContextVar("qwen_half_activation_storage", default=False)


def install_activation_precision(model):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    installed = 0
    for module in model.modules():
        if (not isinstance(module, Qwen2ForCausalLM) or module.config.hidden_size != 1536
                or not getattr(module, "is_loaded_in_4bit", False)
                or getattr(module, "_half_activation_storage", False)):
            continue
        original = module.forward
        @wraps(original)
        def forward(self, *args, _original=original, **kwargs):
            labels = kwargs.get("labels")
            supported = (self.training and torch.is_grad_enabled()
                         and isinstance(labels, torch.Tensor) and labels.is_cuda
                         and torch.is_autocast_enabled("cuda")
                         and torch.get_autocast_dtype("cuda") == torch.float16)
            token = half_activations.set(supported)
            try:
                return _original(*args, **kwargs)
            finally:
                half_activations.reset(token)
        module.forward = MethodType(forward, module)
        embedding = module.model.embed_tokens
        embedding_forward = embedding.forward
        @wraps(embedding_forward)
        def embed(self, *args, _original=embedding_forward, **kwargs):
            value = _original(*args, **kwargs)
            return value.half() if half_activations.get() else value
        embedding.forward = MethodType(embed, embedding)
        module._half_activation_storage = True
        installed += 1
    return installed
