"""Compile independent public-shape kernels concurrently inside the timed run.

warmup=True compiles only: no kernels execute and no random numbers are drawn.
The Trainer barrier joins every worker before training starts. No external
cache, additional process/GPU, or excluded warm-up time is used.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import os
import time
import torch

ENABLED = True
_ticket = None
_patched_trainer_cls = None


def _jobs(buffers):
    h, f, i, u, b = (buffers[key] for key in ('half', 'float', 'index', 'byte', 'bool'))
    from .compact import token_gather_kernel, token_restore_kernel
    from .rmsnorm import frozen_rms_forward_kernel, frozen_rms_backward_kernel
    from .nf4_decode import nf4_double_decode_kernel, nf4_mlp_decode_kernel
    from .lora_kernels import lora_up_add_kernel, lora_cast_scale_gradient_kernel
    from .attention import packed_rotary_kernel
    from .dropout import packed_dropout_forward_kernel, packed_dropout_backward_kernel
    from .online_head import online_vocabulary_gradient_kernel, BLOCK
    from .swiglu import example_swiglu_fwd_kernel, example_swiglu_bwd_kernel
    from .mlp import example_joint_swiglu_fwd_kernel, example_joint_swiglu_bwd_kernel

    # Token gather / restore
    yield token_gather_kernel, (h, i, h, 1536, 1024), {}
    yield token_gather_kernel, (f, i, f, 1536, 1024), {}
    yield token_restore_kernel, (h, i, h, 1536, 1024), {}

    # RMSNorm forward / backward
    yield frozen_rms_forward_kernel, (h, f, h, f, h, h, 1536, 1.e-6, 2048, False), dict(num_warps=4, enable_fp_fusion=False)
    yield frozen_rms_backward_kernel, (h, h, h, f, f, i, h, 1536, 2048, True, False, False), dict(num_warps=4, enable_fp_fusion=False)

    # NF4 double decode / MLP decode
    yield nf4_double_decode_kernel, (u, u, f, f, f, f, h, 1536 * 1536, 64, 256, 1024), dict(num_warps=4, enable_fp_fusion=False)
    yield nf4_mlp_decode_kernel, (u, u, f, f, f, f, u, u, f, f, f, f, u, u, f, f, f, f, h, 1536 * 8960, 1024), dict(num_warps=4, enable_fp_fusion=False)

    # LoRA kernels
    yield lora_up_add_kernel, (h, h, h, h, 701, 1536, 16, 2.0, False, 32, 128), dict(num_warps=4, enable_fp_fusion=False)
    yield lora_cast_scale_gradient_kernel, (h, h, 701 * 1536, 2.0, 1024), {}

    # Attention rotary
    yield packed_rotary_kernel, (h, h, h, h, 701 * 12 * 128, 12, 128, False, 1024), dict(enable_fp_fusion=False)
    yield packed_rotary_kernel, (h, h, h, h, 701 * 12 * 128, 12, 128, True, 1024), dict(enable_fp_fusion=False)

    # Dropout forward / backward
    yield packed_dropout_forward_kernel, (h, i, f, b, h, b, 701 * 1536, 1536, 1024), dict(enable_fp_fusion=False)
    yield packed_dropout_backward_kernel, (h, b, h, 701 * 1536, 1.0 / 0.95, 1024), dict(enable_fp_fusion=False)

    # Online vocabulary gradient
    yield online_vocabulary_gradient_kernel, (h, i, f, f, i, h, 256, 151936, 256, BLOCK), dict(num_warps=4, enable_fp_fusion=False)

    # SwiGLU forward / backward
    yield example_swiglu_fwd_kernel, (h, h, h, 701 * 8960, 1024), {}
    yield example_swiglu_bwd_kernel, (h, h, h, h, h, 701 * 8960, 1024), {}
    yield example_joint_swiglu_fwd_kernel, (h, h, 701 * 8960, 8960, 1024), {}
    yield example_joint_swiglu_bwd_kernel, (h, h, h, h, 701 * 8960, 8960, 1024), {}


class Compilation:
    def __init__(self, options=None):
        global _ticket
        _ticket = self
        self.options = options if isinstance(options, dict) else {}
        self.started = time.perf_counter()
        self.results = None

        if not torch.cuda.is_available():
            self.device = None
            self.buffers = {}
            self.workers = 0
            self.executor = None
            self.futures = []
            return

        self.device = torch.cuda.current_device()
        self.buffers = {
            name: torch.empty(16, device=self.device, dtype=dtype)
            for name, dtype in [
                ('half', torch.float16),
                ('float', torch.float32),
                ('index', torch.int64),
                ('byte', torch.uint8),
                ('bool', torch.bool),
            ]
        }
        worker_cfg = self.options.get('workers', self.options.get('max_workers', 4))
        self.workers = min(int(worker_cfg), os.cpu_count() or 1)
        self.executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='kernel-compile')
        self.futures = [
            self.executor.submit(self.compile, kernel, args, compile_options)
            for kernel, args, compile_options in _jobs(self.buffers)
        ]

    def compile(self, kernel, args, options):
        began = time.perf_counter()
        with torch.cuda.device(self.device):
            kernel.warmup(*args, grid=(1, 1), **options)
        return dict(kernel=kernel.fn.__name__, seconds=round(time.perf_counter() - began, 4))

    def wait(self):
        if self.results is None:
            self.results = []
            if self.executor is not None:
                for future in self.futures:
                    try:
                        self.results.append(future.result())
                    except Exception as exc:
                        self.results.append(dict(error=type(exc).__name__))
                self.executor.shutdown(wait=True)
            self.buffers.clear()
        return self.results

    @staticmethod
    def install_barrier(trainer_cls=None):
        return install_barrier(trainer_cls=trainer_cls)

    @staticmethod
    def uninstall_barrier(trainer_cls=None):
        return uninstall_barrier(trainer_cls=trainer_cls)


def start(options=None):
    global _ticket
    if ENABLED and _ticket is None and torch.cuda.is_available():
        _ticket = Compilation(options=options)
    return _ticket


def wait():
    if _ticket is not None:
        return _ticket.wait()
    return []


def install_barrier(trainer_cls=None):
    global _patched_trainer_cls
    if _ticket is None:
        return None
    if trainer_cls is None:
        try:
            from transformers import Trainer
            trainer_cls = Trainer
        except ImportError:
            trainer_cls = None
    if trainer_cls is None:
        _ticket.wait()
        return None

    original = getattr(trainer_cls, 'train', None)
    if not callable(original):
        _ticket.wait()
        return None
    if getattr(original, '_compile_barrier', False):
        return original

    @wraps(original)
    def train(self, *args, **kwargs):
        if _ticket is not None:
            _ticket.wait()
        return original(self, *args, **kwargs)

    train._compile_barrier = True
    train._original_train = original
    trainer_cls.train = train
    _patched_trainer_cls = trainer_cls
    return train


def uninstall_barrier(trainer_cls=None):
    global _patched_trainer_cls
    target = trainer_cls or _patched_trainer_cls
    if target is None:
        try:
            from transformers import Trainer
            target = Trainer
        except ImportError:
            return None
    current = getattr(target, 'train', None)
    if current is not None and getattr(current, '_compile_barrier', False):
        original = getattr(current, '_original_train', getattr(current, '__wrapped__', None))
        if original is not None:
            target.train = original
    if target is _patched_trainer_cls:
        _patched_trainer_cls = None
    return getattr(target, 'train', None)
