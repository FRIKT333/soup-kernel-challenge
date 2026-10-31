"""Reuse dependency metadata already scanned in this process."""

from contextvars import ContextVar
from functools import wraps
import inspect
import uuid
import importlib
import importlib.metadata
import sys


def prime_dependency_metadata():
    """Reuse the same process's already-built, unmodified distribution mapping."""
    if "trl.import_utils" in sys.modules:
        return
    from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING

    original = importlib.metadata.packages_distributions

    def existing_mapping():
        return {name: list(distributions) for name, distributions in PACKAGE_DISTRIBUTION_MAPPING.items()}

    importlib.metadata.packages_distributions = existing_mapping
    try:
        importlib.import_module("trl.import_utils")
    finally:
        importlib.metadata.packages_distributions = original


_preparing = ContextVar("preparing_sft_dataset", default=False)


def patch_preprocessing():
    from datasets import Dataset
    from trl.trainer.sft_trainer import SFTTrainer

    original_map = Dataset.map
    if getattr(original_map, "_fresh_sft_map", False):
        return
    signature = inspect.signature(original_map)

    @wraps(original_map)
    def mapped(self, *args, **kwargs):
        if not _preparing.get():
            return original_map(self, *args, **kwargs)
        bound = signature.bind_partial(self, *args, **kwargs)
        if bound.arguments.get("new_fingerprint") is None:
            # A fresh identity prevents cache reuse and avoids pickling the
            # tokenizer/trainer closure solely to hash its mapping function.
            # Dataset.map still executes the actual transformation on every row.
            bound.arguments["new_fingerprint"] = uuid.uuid4().hex
        return original_map(*bound.args, **bound.kwargs)

    mapped._fresh_sft_map = True
    Dataset.map = mapped
    original_prepare = SFTTrainer._prepare_dataset

    @wraps(original_prepare)
    def prepare(self, *args, **kwargs):
        token = _preparing.set(True)
        try:
            return original_prepare(self, *args, **kwargs)
        finally:
            _preparing.reset(token)

    SFTTrainer._prepare_dataset = prepare
