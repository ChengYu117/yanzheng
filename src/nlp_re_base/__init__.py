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
    "load_stage2_dataset",
    "extract_hidden_activations_by_layer",
    "save_activations_bundle",
    "load_activations_bundle",
    "train_linear_probes",
    "summarize_probe_results",
    "load_misc_annotation_records",
    "run_misc_label_mapping",
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
    "load_stage2_dataset": (".stage2_activation_extraction", "load_stage2_dataset"),
    "extract_hidden_activations_by_layer": (
        ".stage2_activation_extraction",
        "extract_hidden_activations_by_layer",
    ),
    "save_activations_bundle": (".stage2_activation_extraction", "save_activations_bundle"),
    "load_activations_bundle": (".stage2_activation_extraction", "load_activations_bundle"),
    "train_linear_probes": (".stage2_activation_extraction", "train_linear_probes"),
    "summarize_probe_results": (".stage2_activation_extraction", "summarize_probe_results"),
    "load_misc_annotation_records": (".misc_label_mapping", "load_misc_annotation_records"),
    "run_misc_label_mapping": (".misc_label_mapping", "run_misc_label_mapping"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
