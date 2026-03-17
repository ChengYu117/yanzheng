"""Top-level package for the SAE-RE project.

The package keeps imports lazy so lightweight utilities such as the AI judge
CLI can be used without importing the full torch/transformers stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "load_model_config",
    "dataset_summary",
    "load_jsonl",
    "load_local_model_and_tokenizer",
    "SparseAutoencoder",
    "load_sae_from_hub",
    "extract_and_process_streaming",
    "aggregate_to_utterance",
]

_LAZY_IMPORTS = {
    "load_model_config": (".config", "load_model_config"),
    "dataset_summary": (".data", "dataset_summary"),
    "load_jsonl": (".data", "load_jsonl"),
    "load_local_model_and_tokenizer": (".model", "load_local_model_and_tokenizer"),
    "SparseAutoencoder": (".sae", "SparseAutoencoder"),
    "load_sae_from_hub": (".sae", "load_sae_from_hub"),
    "extract_and_process_streaming": (".activations", "extract_and_process_streaming"),
    "aggregate_to_utterance": (".activations", "aggregate_to_utterance"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
