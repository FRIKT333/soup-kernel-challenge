"""Your solution.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. Patch whatever you need (transformers, peft,
bitsandbytes, Soup itself) so the training run uses your Triton kernels.
"""

from .cross_entropy import FusedChunkedHeadLoss
from .rmsnorm import patch_qwen2_rmsnorm
from .swiglu import patch_qwen2_mlp


def patch_soup_chunked_loss():
    """Hook FusedChunkedHeadLoss into Soup's training pipeline for Qwen causal LM."""
    try:
        from soup_cli.trainer import sft

        original_setup = sft.SFTTrainerWrapper._setup_transformers

        def patched_setup(self, cfg, tcfg):
            original_setup(self, cfg, tcfg)
            model = getattr(self, "model", None)
            if model is not None and hasattr(model, "parameters"):
                first_param = next(model.parameters(), None)
                if first_param is not None and getattr(first_param, "is_cuda", False):
                    get_base = getattr(model, "get_base_model", None)
                    causal_lm = get_base() if callable(get_base) else model
                    hook = FusedChunkedHeadLoss(causal_lm, chunk=64)
                    self._stream_loss_chunk_hook = hook

        sft.SFTTrainerWrapper._setup_transformers = patched_setup
    except (ImportError, AttributeError):
        pass


def apply():
    patch_qwen2_mlp()
    patch_qwen2_rmsnorm()
    patch_soup_chunked_loss()
