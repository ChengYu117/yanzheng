"""Shared helpers for SAE structural diagnostics and parity checks."""

from __future__ import annotations

import importlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from safetensors.torch import load_file as load_safetensors
import zstandard as zstd

from .activations import _parse_hook_point, extract_and_process_streaming
from .config import PROJECT_ROOT
from .data import load_cactus_dataset
from .eval_structural import (
    BatchEnergyDebugCollector,
    OfficialMetricsAccumulator,
    OnlineStructuralAccumulator,
    compute_ce_kl_with_intervention,
    run_structural_evaluation,
)
from .model import is_transformer_lens_model


LLAMA_SCOPE_SOURCES = {
    "model_page": "https://huggingface.co/OpenMOSS-Team/Llama-Scope",
    "checkpoint_page": (
        "https://huggingface.co/OpenMOSS-Team/"
        "Llama3_1-8B-Base-LXR-8x/tree/main/Llama3_1-8B-Base-L19R-8x"
    ),
    "official_repo": "https://github.com/OpenMOSS/Language-Model-SAEs",
    "paper": "https://arxiv.org/abs/2410.20526",
    "dataset_official": "https://huggingface.co/datasets/cerebras/SlimPajama-627B",
    "dataset_mirror": "https://hf-mirror.com/datasets/cerebras/SlimPajama-627B",
    "dataset_proxy": "https://huggingface.co/datasets/venketh/SlimPajama-62B",
}


def build_variant_evidence_matrix(
    *,
    checkpoint_hyperparams: dict[str, Any],
) -> dict[str, Any]:
    """Build evidence for the Llama Scope TopK variant semantics."""
    act_fn = checkpoint_hyperparams.get("act_fn")
    top_k = checkpoint_hyperparams.get("top_k")
    sparsity_include_decoder_norm = checkpoint_hyperparams.get("sparsity_include_decoder_norm")
    norm_activation = checkpoint_hyperparams.get("norm_activation")
    return {
        "checkpoint": {
            "act_fn": act_fn,
            "top_k": top_k,
            "sparsity_include_decoder_norm": sparsity_include_decoder_norm,
            "norm_activation": norm_activation,
            "jump_relu_threshold": checkpoint_hyperparams.get("jump_relu_threshold"),
            "hook_point_in": checkpoint_hyperparams.get("hook_point_in"),
            "hook_point_out": checkpoint_hyperparams.get("hook_point_out"),
        },
        "components": {
            "R1_norm_activation": {
                "status": "aligned",
                "current_code": "official_runtime_datasetwise_then_inference",
                "paper_direct": "dataset-wise norm is part of the published checkpoint/runtime assumptions",
                "checkpoint_direct": f"norm_activation={norm_activation}",
                "official_code_direct": (
                    "standardize_parameters_of_dataset_norm folds dataset-wise scaling into parameters "
                    "and switches runtime to inference mode"
                ),
            },
            "R2_decoder_norm_gating": {
                "status": "aligned",
                "current_code": "delegated_to_official_encode",
                "paper_direct": "decoder norm is part of the improved TopK family",
                "checkpoint_direct": f"sparsity_include_decoder_norm={sparsity_include_decoder_norm}",
                "official_code_direct": (
                    "SparseAutoEncoder.encode multiplies hidden_pre by decoder_norm before activation "
                    "and divides feature_acts after activation when sparsity_include_decoder_norm=True"
                ),
            },
            "R3_jumprelu_threshold_semantics": {
                "status": "aligned",
                "current_code": "official_jumprelu_threshold_loaded_from_checkpoint",
                "paper_direct": "published checkpoints are exposed as JumpReLU variants",
                "checkpoint_direct": f"act_fn={act_fn}, jump_relu_threshold={checkpoint_hyperparams.get('jump_relu_threshold')}",
                "official_code_direct": (
                    "official JumpReLU forward is input * 1[input > threshold], not max(0, input-threshold)"
                ),
            },
            "R4_topk_runtime_semantics": {
                "status": "paper_not_fully_specified",
                "current_code": "configurable_audit_required",
                "paper_direct": (
                    "paper states TopK SAEs are post-processed into JumpReLU variants, "
                    "which suggests inference is no longer exact-K per input"
                ),
                "checkpoint_direct": f"top_k={top_k} is still present in hyperparams",
                "official_code_direct": (
                    "official vanilla SAE JumpReLU path does not apply hard top-k; "
                    "generic topk_to_jumprelu_conversion utility is only explicit for CLT"
                ),
            },
            "R5_k_annealing_runtime_role": {
                "status": "aligned",
                "current_code": "training_only",
                "paper_direct": "K-annealing is a training schedule",
                "checkpoint_direct": "public checkpoint exposes final top_k metadata, not a runtime schedule",
                "official_code_direct": "current_k is driven by trainer; inference path does not anneal K",
            },
        },
        "overall_current_alignment": "partially_aligned",
        "sources": LLAMA_SCOPE_SOURCES,
    }


def summarize_variant_compare(
    *,
    semantics: str,
    paper_result: dict[str, Any],
    mire_result: dict[str, Any],
    parity_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_topk_semantics": semantics,
        "paper_distribution": {
            "mse": paper_result["structural_metrics"].get("mse"),
            "cosine_similarity": paper_result["structural_metrics"].get("cosine_similarity"),
            "ev_openmoss_aligned": paper_result["structural_metrics"].get("ev_openmoss_aligned"),
            "ev_llamascope_paper": paper_result["structural_metrics"].get("ev_llamascope_paper"),
            "dead_ratio": paper_result["structural_metrics"].get("dead_ratio"),
            "l0_mean": paper_result["structural_metrics"].get("l0_mean"),
        },
        "mi_re": {
            "mse": mire_result["structural_metrics"].get("mse"),
            "cosine_similarity": mire_result["structural_metrics"].get("cosine_similarity"),
            "ev_openmoss_aligned": mire_result["structural_metrics"].get("ev_openmoss_aligned"),
            "ev_llamascope_paper": mire_result["structural_metrics"].get("ev_llamascope_paper"),
            "dead_ratio": mire_result["structural_metrics"].get("dead_ratio"),
            "l0_mean": mire_result["structural_metrics"].get("l0_mean"),
        },
        "parity": {
            "parity_passed": parity_report.get("parity_passed"),
            "reconstruction_max_abs_diff": parity_report.get("reconstruction_max_abs_diff"),
            "latent_l0_local": parity_report.get("latent_l0_local"),
            "latent_l0_official": parity_report.get("latent_l0_official"),
            "topk_exact_match": parity_report.get("topk_exact_match"),
            "reference_backend": parity_report.get("reference_backend"),
        },
    }


def _to_builtin(value: Any) -> Any:
    """Convert numpy / torch scalars and containers to JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _to_builtin(value.item())
        return [_to_builtin(v) for v in value.detach().cpu().tolist()]
    if isinstance(value, np.ndarray):
        return [_to_builtin(v) for v in value.tolist()]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Write a JSON file with UTF-8 encoding."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(_to_builtin(payload), f, indent=2, ensure_ascii=False)
    return target


def build_llamascope_evidence_table(
    *,
    dataset_access: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact evidence table for provenance and dataset access."""
    return {
        "sources": LLAMA_SCOPE_SOURCES,
        "claims": [
            {
                "claim": "Llama Scope is an improved TopK SAE family.",
                "source": LLAMA_SCOPE_SOURCES["model_page"],
                "evidence_type": "direct_evidence",
            },
            {
                "claim": (
                    "Llama3_1-8B-Base-L19R-8x is a layer-19 residual SAE checkpoint "
                    "under Llama3_1-8B-Base-LXR-8x."
                ),
                "source": LLAMA_SCOPE_SOURCES["checkpoint_page"],
                "evidence_type": "direct_evidence",
            },
            {
                "claim": "The SAE training data is SlimPajama.",
                "source": LLAMA_SCOPE_SOURCES["model_page"],
                "evidence_type": "direct_evidence",
            },
            {
                "claim": (
                    "venketh/SlimPajama-62B preserves 100% of the original "
                    "validation/test splits from cerebras/SlimPajama-627B."
                ),
                "source": LLAMA_SCOPE_SOURCES["dataset_proxy"],
                "evidence_type": "direct_evidence",
            },
            {
                "claim": (
                    "SlimPajama validation is the best public proxy for evaluating "
                    "whether the SAE behaves better near its training distribution."
                ),
                "source": LLAMA_SCOPE_SOURCES["paper"],
                "evidence_type": "inference",
            },
        ],
        "dataset_access": dataset_access,
    }


def _mean_of_debug(entries: list[dict[str, Any]], key: str) -> float | None:
    values = [float(entry[key]) for entry in entries if entry.get(key) is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def infer_space_resolution(
    *,
    sae: Any,
    raw_debug_entries: list[dict[str, Any]],
    normalized_debug_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    backend_sae = getattr(sae, "backend_sae", None)
    cfg = getattr(backend_sae, "cfg", None)
    runtime_norm_activation = getattr(cfg, "norm_activation", None)
    raw_scale_mean = _mean_of_debug(raw_debug_entries, "input_scale_factor_mean")
    raw_scale_std = _mean_of_debug(raw_debug_entries, "input_scale_factor_std")
    mean_abs_x_minus_x_norm = _mean_of_debug(raw_debug_entries, "mean_abs_x_minus_x_norm")

    normalized_effective_space = "sae_input_space"
    normalized_space_matches_raw = False
    note = "normalized activations are distinct from raw residuals"
    if runtime_norm_activation == "inference":
        normalized_effective_space = "post_standardization_inference_space"
    if (
        raw_scale_mean is not None
        and abs(raw_scale_mean - 1.0) < 1e-6
        and (raw_scale_std is None or raw_scale_std < 1e-6)
        and mean_abs_x_minus_x_norm is not None
        and mean_abs_x_minus_x_norm < 1e-3
    ):
        normalized_effective_space = "post_standardization_inference_space"
        normalized_space_matches_raw = True
        note = (
            "standardize_parameters_of_dataset_norm has folded dataset-wise scaling "
            "into the weights, so runtime normalized activations are effectively raw-space "
            "activations in inference mode"
        )

    return {
        "runtime_backend": type(sae).__name__,
        "runtime_norm_activation": runtime_norm_activation,
        "raw_space_id": "raw_residual",
        "normalized_space_reported_id": "normalized",
        "normalized_space_effective_id": normalized_effective_space,
        "pre_denorm_reconstruction_space_id": "sae_reconstruction_space_pre_denorm",
        "normalized_space_matches_raw": normalized_space_matches_raw,
        "debug_summary": {
            "raw_input_scale_factor_mean": raw_scale_mean,
            "raw_input_scale_factor_std": raw_scale_std,
            "raw_mean_abs_x_minus_x_norm": mean_abs_x_minus_x_norm,
            "raw_batches": len(raw_debug_entries),
            "normalized_batches": len(normalized_debug_entries),
        },
        "note": note,
    }


def build_metric_provenance(
    *,
    sae: Any,
    hook_point: str,
    space_resolution: dict[str, Any],
) -> dict[str, Any]:
    return {
        "alignment_basis": "openmoss_official_implementation",
        "hook_point": hook_point,
        "paper_reference": {
            "source_ref": "llama_scope_arxiv_2410.20526_section_4.1",
            "metric_name": "ev_openmoss_legacy",
            "formula": "mean(1 - per_token_l2_loss / total_variance)",
            "status": (
                "paper-facing primary metric is temporarily aligned to the official legacy EV "
                "because it matches the reported ~0.7 scale; negative EV variants are retained "
                "only as auxiliary diagnostics"
            ),
        },
        "official_openmoss_reference": {
            "source_ref": "lm_saes.metrics.ExplainedVarianceMetric",
            "metric_name": "ev_openmoss_aligned",
            "formula": "1 - mean(per_token_l2_loss) / mean(total_variance)",
            "legacy_formula": "mean(1 - per_token_l2_loss / total_variance)",
            "space_protocol": (
                "Computed on label/reconstructed tensors after sae.normalize_activations(batch). "
                "If norm_activation='inference', this is post_standardization_inference_space."
            ),
        },
        "metric_family_comparison": {
            "project_legacy_structural_metrics": [
                "mse",
                "cosine_similarity",
                "explained_variance_centered_raw",
                "fvu_centered_raw",
                "dead_ratio",
                "l0_mean",
                "ce_loss_delta",
                "kl_divergence",
            ],
            "official_openmoss_metrics_added": [
                "ev_openmoss_aligned",
                "ev_openmoss_legacy",
                "metrics/l2_norm_error",
                "metrics/l2_norm_error_ratio",
                "metrics/mean_feature_act",
                "metrics/l0",
                "sparsity/above_1e-1",
                "sparsity/above_1e-2",
                "sparsity/below_1e-5",
                "sparsity/below_1e-6",
                "sparsity/below_1e-7",
                "delta_lm_loss",
                "downstream_loss_ratio",
            ],
        },
        "project_mapping": {
            "ev_openmoss_aligned": {
                "field": "ev_openmoss_aligned",
                "derived_from": "official_metrics.metrics/explained_variance",
                "space_id": space_resolution["normalized_space_effective_id"],
                "status": "auxiliary_negative_variant_temporarily_not_used_as_primary",
            },
            "ev_openmoss_legacy": {
                "field": "ev_openmoss_legacy",
                "derived_from": "official_metrics.metrics/explained_variance_legacy",
                "space_id": space_resolution["normalized_space_effective_id"],
                "status": "primary",
            },
            "ev_llamascope_paper": {
                "field": "ev_llamascope_paper",
                "derived_from": "explained_variance_paper_raw",
                "space_id": "raw_residual",
                "status": "auxiliary_negative_variant_temporarily_not_used_as_primary",
            },
            "ev_centered_legacy": {
                "field": "ev_centered_legacy",
                "derived_from": "explained_variance_centered_raw",
                "space_id": "raw_residual",
                "note": "Project-defined global centered-SST EV kept for backward compatibility.",
                "status": "auxiliary_negative_variant_temporarily_not_used_as_primary",
            },
            "project_global_openmoss_style_ev": {
                "field": "explained_variance_openmoss_raw",
                "derived_from": "raw_full_metrics.explained_variance_openmoss",
                "space_id": "raw_residual",
                "note": (
                    "This is the project's token-weighted global ratio-of-means EV, not the official "
                    "batch-aggregated OpenMOSS evaluator output."
                ),
            },
        },
        "space_resolution": space_resolution,
        "sources": LLAMA_SCOPE_SOURCES,
    }


def build_ev_alignment_report(
    *,
    structural_metrics: dict[str, Any],
    space_resolution: dict[str, Any],
) -> dict[str, Any]:
    ev_openmoss = structural_metrics.get("ev_openmoss_aligned")
    ev_openmoss_legacy = structural_metrics.get("ev_openmoss_legacy")
    ev_paper = structural_metrics.get("ev_llamascope_paper")
    ev_centered = structural_metrics.get("ev_centered_legacy")

    equivalence_delta = None
    if ev_openmoss is not None and ev_centered is not None:
        equivalence_delta = float(ev_openmoss) - float(ev_centered)

    diagnosis = []
    if space_resolution.get("normalized_space_matches_raw"):
        diagnosis.append("normalized_fields_collapse_to_inference_space")
    if ev_openmoss is not None and ev_openmoss < 0:
        diagnosis.append("openmoss_aligned_ev_negative_in_current_runtime_space")
    if ev_paper is not None and ev_paper < 0:
        diagnosis.append("paper_ev_negative_in_raw_residual_space")
    if not diagnosis:
        diagnosis.append("no_space_mismatch_detected")

    return {
        "alignment_basis": "openmoss_official_implementation",
        "space_resolution": space_resolution,
        "metrics": {
            "ev_openmoss_aligned": ev_openmoss,
            "ev_openmoss_aligned_normalized": structural_metrics.get(
                "explained_variance_openmoss_normalized"
            ),
            "ev_openmoss_legacy": ev_openmoss_legacy,
            "ev_llamascope_paper": ev_paper,
            "ev_llamascope_paper_normalized": structural_metrics.get(
                "explained_variance_paper_normalized"
            ),
            "ev_centered_legacy": ev_centered,
        },
        "equivalence_checks": {
            "openmoss_minus_centered": equivalence_delta,
        },
        "diagnosis": diagnosis,
    }


def _sequence_summary(values: list[int]) -> dict[str, float | int] | None:
    """Summarize a numeric sequence with compact, stable statistics."""
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": int(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": int(arr.max()),
    }


def compute_text_statistics(
    texts: list[str],
    *,
    tokenizer: Any | None = None,
    max_seq_len: int | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Compute lightweight distribution summaries for a text collection."""
    normalized_texts = [text if isinstance(text, str) else str(text) for text in texts]
    char_lengths = [len(text) for text in normalized_texts]
    word_lengths = [len(text.split()) for text in normalized_texts]
    stats: dict[str, Any] = {
        "count": len(normalized_texts),
        "empty_count": sum(1 for text in normalized_texts if not text.strip()),
        "char_length": _sequence_summary(char_lengths),
        "word_length": _sequence_summary(word_lengths),
    }

    if tokenizer is not None and normalized_texts:
        token_lengths: list[int] = []
        for start in range(0, len(normalized_texts), batch_size):
            batch = normalized_texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=False,
                truncation=max_seq_len is not None,
                max_length=max_seq_len,
            )
            token_lengths.extend(len(ids) for ids in encoded["input_ids"])
        stats["token_length"] = _sequence_summary(token_lengths)

    return stats


def load_hf_texts(
    *,
    dataset_id: str,
    split: str,
    text_field: str,
    max_docs: int,
    config_name: str | None = None,
    streaming: bool = True,
) -> list[str]:
    """Load a bounded number of texts from a HuggingFace dataset."""
    bundle = load_hf_texts_with_metadata(
        dataset_id=dataset_id,
        split=split,
        text_field=text_field,
        max_docs=max_docs,
        config_name=config_name,
        streaming=streaming,
    )
    return bundle["texts"]


def load_hf_texts_with_metadata(
    *,
    dataset_id: str,
    split: str,
    text_field: str,
    max_docs: int,
    config_name: str | None = None,
    streaming: bool = True,
) -> dict[str, Any]:
    """Load texts plus access metadata for provenance/debugging."""
    from datasets import load_dataset

    load_kwargs: dict[str, Any] = {
        "path": dataset_id,
        "split": split,
        "streaming": streaming,
    }
    if config_name is not None:
        load_kwargs["name"] = config_name

    try:
        dataset = load_dataset(**load_kwargs)
        texts: list[str] = []

        if streaming:
            iterator = iter(dataset)
            for record in iterator:
                if text_field not in record:
                    raise KeyError(
                        f"Text field {text_field!r} not found in dataset {dataset_id!r}."
                    )
                text = record[text_field]
                if not isinstance(text, str):
                    text = str(text)
                if not text.strip():
                    continue
                texts.append(text)
                if len(texts) >= max_docs:
                    break
            return {
                "texts": texts,
                "dataset_access": {
                    "requested_dataset_id": dataset_id,
                    "resolved_dataset_id": dataset_id,
                    "dataset_config_name": config_name,
                    "split": split,
                    "text_field": text_field,
                    "streaming": streaming,
                    "access_mode": "datasets_api",
                    "resolved_files": [],
                },
            }

        if text_field not in dataset.column_names:
            raise KeyError(
                f"Text field {text_field!r} not found in dataset {dataset_id!r}."
            )
        selected = dataset.select(range(min(max_docs, len(dataset))))
        for text in selected[text_field]:
            normalized = text if isinstance(text, str) else str(text)
            if normalized.strip():
                texts.append(normalized)
        return {
            "texts": texts[:max_docs],
            "dataset_access": {
                "requested_dataset_id": dataset_id,
                "resolved_dataset_id": dataset_id,
                "dataset_config_name": config_name,
                "split": split,
                "text_field": text_field,
                "streaming": streaming,
                "access_mode": "datasets_api",
                "resolved_files": [],
            },
        }
    except Exception as exc:
        return _load_hf_texts_via_repo_files(
            dataset_id=dataset_id,
            split=split,
            text_field=text_field,
            max_docs=max_docs,
            fallback_error=exc,
        )


def _load_hf_texts_via_repo_files(
    *,
    dataset_id: str,
    split: str,
    text_field: str,
    max_docs: int,
    fallback_error: Exception | None = None,
) -> dict[str, Any]:
    """Fallback loader for simple JSONL(.zst) dataset repos on the Hub."""
    candidate_repo_ids = [dataset_id]
    if dataset_id == "cerebras/SlimPajama-627B" and split in {"validation", "test"}:
        # `venketh/SlimPajama-62B` preserves the full validation/test splits.
        candidate_repo_ids.append("venketh/SlimPajama-62B")

    last_error: Exception | None = fallback_error
    for repo_id in candidate_repo_ids:
        try:
            files = list_repo_files(repo_id, repo_type="dataset", token=False)
            split_files = sorted(
                file
                for file in files
                if file.startswith(f"{split}/") and file.endswith(".jsonl.zst")
            )
            if not split_files:
                split_files = sorted(
                    file
                    for file in files
                    if file.startswith(f"{split}/") and file.endswith(".jsonl")
                )
            if not split_files:
                raise FileNotFoundError(
                    f"No JSONL files found for split {split!r} in dataset {repo_id!r}."
                )

            texts: list[str] = []
            resolved_files: list[str] = []
            for file_name in split_files:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_name,
                    repo_type="dataset",
                    token=False,
                )
                resolved_files.append(file_name)
                if file_name.endswith(".zst"):
                    texts.extend(
                        _read_zstd_jsonl_texts(
                            local_path,
                            text_field=text_field,
                            limit=max_docs - len(texts),
                        )
                    )
                else:
                    texts.extend(
                        _read_jsonl_texts(
                            local_path,
                            text_field=text_field,
                            limit=max_docs - len(texts),
                        )
                    )
                if len(texts) >= max_docs:
                    break

            if texts:
                return {
                    "texts": texts[:max_docs],
                    "dataset_access": {
                        "requested_dataset_id": dataset_id,
                        "resolved_dataset_id": repo_id,
                        "dataset_config_name": None,
                        "split": split,
                        "text_field": text_field,
                        "streaming": True,
                        "access_mode": "repo_file_fallback",
                        "resolved_files": resolved_files,
                    },
                }
        except Exception as exc:  # pragma: no cover - exercised in live debugging
            last_error = exc

    raise RuntimeError(
        f"Unable to load dataset texts for {dataset_id!r} split={split!r} "
        f"via datasets or repo-file fallback. Last error: {last_error}"
    ) from last_error


def _read_jsonl_texts(
    path: str | Path,
    *,
    text_field: str,
    limit: int,
) -> list[str]:
    texts: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if len(texts) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get(text_field, "")
            if not isinstance(text, str):
                text = str(text)
            if text.strip():
                texts.append(text)
    return texts


def _read_zstd_jsonl_texts(
    path: str | Path,
    *,
    text_field: str,
    limit: int,
) -> list[str]:
    texts: list[str] = []
    with Path(path).open("rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8") as text_stream:
                for line in text_stream:
                    if len(texts) >= limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    text = record.get(text_field, "")
                    if not isinstance(text, str):
                        text = str(text)
                    if text.strip():
                        texts.append(text)
    return texts


def load_mi_re_texts(
    data_dir: str | Path = "data/mi_re",
) -> tuple[list[str], dict[str, Any]]:
    """Load the legacy MI-RE split and return plain texts plus dataset metadata."""
    root = Path(data_dir)
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    re_records, nonre_records, all_records = load_cactus_dataset(root)
    texts = [record.get("unit_text", "") for record in all_records]

    re_subcodes = sorted(
        {
            str(record.get("predicted_subcode", "")).strip()
            for record in re_records
            if record.get("predicted_subcode")
        }
    )
    nonre_codes = sorted(
        {
            str(record.get("predicted_code", "")).strip()
            for record in nonre_records
            if record.get("predicted_code")
        }
    )
    metadata = {
        "data_dir": str(root),
        "re_count": len(re_records),
        "nonre_count": len(nonre_records),
        "total_count": len(all_records),
        "re_subcodes": re_subcodes,
        "nonre_codes": nonre_codes,
    }
    return texts, metadata


def apply_full_structural_metrics(
    structural_metrics: dict[str, Any],
    *,
    raw_full_metrics: dict[str, Any],
    norm_full_metrics: dict[str, Any],
    official_runtime_metrics: dict[str, Any] | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Overwrite sample-batch structural metrics with full-dataset metrics."""
    structural_metrics = dict(structural_metrics)
    structural_metrics.update(raw_full_metrics)
    structural_metrics["metric_definition_version"] = max(
        int(structural_metrics.get("metric_definition_version", 0)),
        3,
    )
    structural_metrics["structural_scope"] = "full_dataset"
    structural_metrics["explained_variance_centered_raw"] = raw_full_metrics.get(
        "explained_variance"
    )
    structural_metrics["explained_variance_openmoss_raw"] = raw_full_metrics.get(
        "explained_variance_openmoss"
    )
    structural_metrics["explained_variance_openmoss_legacy_raw"] = raw_full_metrics.get(
        "explained_variance_openmoss_legacy"
    )
    structural_metrics["fvu_centered_raw"] = raw_full_metrics.get("fvu")
    structural_metrics["explained_variance_paper_raw"] = raw_full_metrics.get(
        "explained_variance_paper"
    )
    structural_metrics["paper_ev_denominator_energy_raw"] = raw_full_metrics.get(
        "paper_ev_denominator_energy"
    )
    structural_metrics["metric_primary"] = "ev_openmoss_legacy"
    structural_metrics["metric_primary_status"] = "preferred"
    structural_metrics["metric_primary_note"] = (
        "Temporarily use official legacy EV as the primary literature-facing metric. "
        "Negative EV variants remain available but are not used as the main conclusion metric."
    )
    structural_metrics["paper_metric_primary"] = "ev_openmoss_legacy"
    structural_metrics["paper_metric_formula"] = "mean(1 - per_token_l2_loss / total_variance)"
    structural_metrics["metric_source_ref"] = "lm_saes.metrics.ExplainedVarianceMetric.legacy"
    if official_runtime_metrics:
        structural_metrics["official_metrics"] = official_runtime_metrics
        structural_metrics["ev_openmoss_official_batch_aggregated"] = official_runtime_metrics.get(
            "metrics/explained_variance"
        )
        structural_metrics["ev_openmoss_official_legacy_batch_aggregated"] = official_runtime_metrics.get(
            "metrics/explained_variance_legacy"
        )
        structural_metrics["l2_norm_error_official"] = official_runtime_metrics.get(
            "metrics/l2_norm_error"
        )
        structural_metrics["l2_norm_error_ratio_official"] = official_runtime_metrics.get(
            "metrics/l2_norm_error_ratio"
        )
        structural_metrics["mean_feature_act_official"] = official_runtime_metrics.get(
            "metrics/mean_feature_act"
        )
        structural_metrics["l0_official"] = official_runtime_metrics.get("metrics/l0")
        structural_metrics["ev_openmoss_aligned"] = official_runtime_metrics.get(
            "metrics/explained_variance"
        )
        structural_metrics["ev_openmoss_legacy"] = official_runtime_metrics.get(
            "metrics/explained_variance_legacy"
        )
    else:
        structural_metrics["ev_openmoss_aligned"] = raw_full_metrics.get("explained_variance_openmoss")
        structural_metrics["ev_openmoss_legacy"] = raw_full_metrics.get(
            "explained_variance_openmoss_legacy"
        )
    structural_metrics["ev_llamascope_paper"] = raw_full_metrics.get("explained_variance_paper")
    structural_metrics["ev_centered_legacy"] = raw_full_metrics.get("explained_variance")
    if norm_full_metrics:
        structural_metrics["explained_variance_centered_normalized"] = norm_full_metrics.get(
            "explained_variance"
        )
        structural_metrics["explained_variance_openmoss_normalized"] = norm_full_metrics.get(
            "explained_variance_openmoss"
        )
        structural_metrics["explained_variance_openmoss_legacy_normalized"] = norm_full_metrics.get(
            "explained_variance_openmoss_legacy"
        )
        structural_metrics["fvu_centered_normalized"] = norm_full_metrics.get("fvu")
        structural_metrics["explained_variance_paper_normalized"] = norm_full_metrics.get(
            "explained_variance_paper"
        )
        structural_metrics["paper_ev_denominator_energy_normalized"] = norm_full_metrics.get(
            "paper_ev_denominator_energy"
        )
    structural_metrics.setdefault("space_metrics", {})
    structural_metrics["space_metrics"]["raw"] = {
        key: value
        for key, value in raw_full_metrics.items()
        if key
        in {
            "mse",
            "cosine_similarity",
            "explained_variance",
            "explained_variance_openmoss",
            "explained_variance_openmoss_legacy",
            "fvu",
            "explained_variance_paper",
            "paper_ev_denominator_energy",
            "n_tokens",
        }
    }
    structural_metrics["space_metrics"]["raw"]["explained_variance_centered"] = raw_full_metrics.get(
        "explained_variance"
    )
    structural_metrics["space_metrics"]["raw"]["explained_variance_openmoss"] = raw_full_metrics.get(
        "explained_variance_openmoss"
    )
    structural_metrics["space_metrics"]["raw"][
        "explained_variance_openmoss_legacy"
    ] = raw_full_metrics.get("explained_variance_openmoss_legacy")
    structural_metrics["space_metrics"]["raw"]["fvu_centered"] = raw_full_metrics.get("fvu")
    structural_metrics["space_metrics"]["normalized"] = {
        key: value
        for key, value in norm_full_metrics.items()
        if key
        in {
            "mse",
            "cosine_similarity",
            "explained_variance",
            "explained_variance_openmoss",
            "explained_variance_openmoss_legacy",
            "fvu",
            "explained_variance_paper",
            "paper_ev_denominator_energy",
            "n_tokens",
        }
    }
    structural_metrics["space_metrics"]["normalized"][
        "explained_variance_centered"
    ] = norm_full_metrics.get("explained_variance")
    structural_metrics["space_metrics"]["normalized"][
        "explained_variance_openmoss"
    ] = norm_full_metrics.get("explained_variance_openmoss")
    structural_metrics["space_metrics"]["normalized"][
        "explained_variance_openmoss_legacy"
    ] = norm_full_metrics.get("explained_variance_openmoss_legacy")
    structural_metrics["space_metrics"]["normalized"]["fvu_centered"] = norm_full_metrics.get(
        "fvu"
    )
    save_json(structural_metrics, Path(output_dir) / "metrics_structural.json")
    return structural_metrics


def run_structural_diagnostic(
    *,
    texts: list[str],
    output_dir: str | Path,
    dataset_label: str,
    tokenizer: Any,
    model: Any,
    sae: Any,
    hook_point: str,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
    ce_kl_batch_size: int | None = None,
    ce_kl_max_texts: int | None = None,
) -> dict[str, Any]:
    """Run the structural-only SAE diagnostic on an arbitrary text list."""
    if not texts:
        raise ValueError(f"No texts were provided for structural diagnostic: {dataset_label}")
    target_output_dir = Path(output_dir)
    target_output_dir.mkdir(parents=True, exist_ok=True)

    sample_stats = compute_text_statistics(
        texts,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )
    sample_stats["dataset_label"] = dataset_label
    save_json(sample_stats, target_output_dir / "sample_stats.json")

    official_runtime_accumulator = OfficialMetricsAccumulator()
    accumulator = {
        "raw": OnlineStructuralAccumulator(),
        "normalized": OnlineStructuralAccumulator(),
        "official": official_runtime_accumulator,
    }
    debug_collectors = {
        "raw": BatchEnergyDebugCollector(space_id="raw_residual"),
        "normalized": BatchEnergyDebugCollector(space_id="normalized"),
    }
    result = extract_and_process_streaming(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        texts=texts,
        hook_point=hook_point,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        aggregation="max",
        device=device,
        collect_structural_samples=5,
        structural_accumulator=accumulator,
        debug_collectors=debug_collectors,
    )

    ce_kl_results = compute_ce_kl_with_intervention(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        sae=sae,
        hook_point=hook_point,
        max_seq_len=max_seq_len,
        batch_size=ce_kl_batch_size or max(1, batch_size // 2),
        max_texts=ce_kl_max_texts,
    )

    structural_metrics = run_structural_evaluation(
        activations=result["sample_activations"],
        reconstructed=result["sample_reconstructed"],
        normalized_activations=result["sample_activations_normalized"],
        normalized_reconstructed=result["sample_reconstructed_normalized"],
        latents=result["sample_latents"],
        attention_mask=result["sample_mask"],
        ce_kl_results=ce_kl_results,
        output_dir=target_output_dir,
    )
    structural_metrics = apply_full_structural_metrics(
        structural_metrics,
        raw_full_metrics=accumulator["raw"].result(),
        norm_full_metrics=accumulator["normalized"].result(),
        official_runtime_metrics=official_runtime_accumulator.result(),
        output_dir=target_output_dir,
    )
    raw_debug_entries = debug_collectors["raw"].result()
    normalized_debug_entries = debug_collectors["normalized"].result()
    save_json(
        {
            "raw": raw_debug_entries,
            "normalized": normalized_debug_entries,
        },
        target_output_dir / "batch_energy_debug.json",
    )

    space_resolution = infer_space_resolution(
        sae=sae,
        raw_debug_entries=raw_debug_entries,
        normalized_debug_entries=normalized_debug_entries,
    )
    structural_metrics["space_resolution"] = space_resolution
    metric_provenance = build_metric_provenance(
        sae=sae,
        hook_point=hook_point,
        space_resolution=space_resolution,
    )
    save_json(metric_provenance, target_output_dir / "metric_provenance.json")
    ev_alignment_report = build_ev_alignment_report(
        structural_metrics=structural_metrics,
        space_resolution=space_resolution,
    )
    save_json(ev_alignment_report, target_output_dir / "ev_alignment_report.json")
    save_json(structural_metrics, target_output_dir / "metrics_structural.json")

    return {
        "dataset_label": dataset_label,
        "sample_stats": sample_stats,
        "structural_metrics": structural_metrics,
        "metric_provenance": metric_provenance,
        "ev_alignment_report": ev_alignment_report,
        "output_dir": str(target_output_dir),
    }


def _metric_delta(
    paper_metrics: dict[str, Any],
    mire_metrics: dict[str, Any],
    *,
    key: str,
) -> float | None:
    if key not in paper_metrics or key not in mire_metrics:
        return None
    if paper_metrics[key] is None or mire_metrics[key] is None:
        return None
    return float(paper_metrics[key]) - float(mire_metrics[key])


def _get_primary_paper_ev(metrics: dict[str, Any]) -> float | None:
    value = metrics.get("ev_openmoss_legacy")
    if value is None:
        official = metrics.get("official_metrics", {})
        if isinstance(official, dict):
            value = official.get("metrics/explained_variance_legacy")
    if value is None:
        value = metrics.get("explained_variance_paper_raw", metrics.get("explained_variance_paper"))
    return None if value is None else float(value)


def _get_normalized_paper_ev(metrics: dict[str, Any]) -> float | None:
    value = metrics.get("explained_variance_paper_normalized")
    if value is not None:
        return float(value)
    normalized = metrics.get("space_metrics", {}).get("normalized")
    if isinstance(normalized, dict) and normalized.get("explained_variance_paper") is not None:
        return float(normalized["explained_variance_paper"])
    return None


def _build_literature_alignment(
    *,
    paper_metrics: dict[str, Any],
    mire_metrics: dict[str, Any],
) -> dict[str, Any]:
    expected_band = [0.6, 0.7]
    paper_ev_raw = _get_primary_paper_ev(paper_metrics)
    paper_ev_normalized = _get_normalized_paper_ev(paper_metrics)
    mire_ev_raw = _get_primary_paper_ev(mire_metrics)
    legacy_centered_raw = float(
        paper_metrics.get("explained_variance_centered_raw", paper_metrics.get("explained_variance"))
    )

    if paper_ev_raw is not None and expected_band[0] <= paper_ev_raw <= expected_band[1]:
        alignment_status = (
            "metric_mismatch_supported" if legacy_centered_raw < 0.0 else "aligned"
        )
    elif paper_ev_raw is not None and paper_ev_raw > expected_band[1]:
        alignment_status = "aligned"
    else:
        alignment_status = "below_band"

    return {
        "paper_proxy_ev_raw": paper_ev_raw,
        "paper_proxy_ev_normalized": paper_ev_normalized,
        "mi_re_ev_raw": mire_ev_raw,
        "legacy_centered_ev_raw": legacy_centered_raw,
        "expected_literature_band": expected_band,
        "alignment_status": alignment_status,
    }


def _diagnose_distribution_shift(
    *,
    paper_metrics: dict[str, Any],
    mire_metrics: dict[str, Any],
) -> str:
    score = 0
    if paper_metrics.get("cosine_similarity", float("-inf")) > mire_metrics.get(
        "cosine_similarity", float("-inf")
    ):
        score += 1
    paper_ev_raw = _get_primary_paper_ev(paper_metrics)
    mire_ev_raw = _get_primary_paper_ev(mire_metrics)
    if (
        paper_ev_raw is not None
        and mire_ev_raw is not None
        and paper_ev_raw > mire_ev_raw
    ):
        score += 1
    paper_ev_norm = _get_normalized_paper_ev(paper_metrics)
    mire_ev_norm = _get_normalized_paper_ev(mire_metrics)
    if (
        paper_ev_norm is not None
        and mire_ev_norm is not None
        and paper_ev_norm > mire_ev_norm
    ):
        score += 1
    if paper_metrics.get("dead_ratio", float("inf")) < mire_metrics.get("dead_ratio", float("inf")):
        score += 1
    if paper_metrics.get("ce_loss_delta", float("inf")) < mire_metrics.get(
        "ce_loss_delta", float("inf")
    ):
        score += 1
    if paper_metrics.get("kl_divergence", float("inf")) < mire_metrics.get(
        "kl_divergence", float("inf")
    ):
        score += 1

    if score >= 4:
        return "paper_distribution_structurally_healthier_than_mi_re"
    if score <= 2:
        return "paper_distribution_not_clearly_healthier_than_mi_re"
    return "mixed_signal_check_parity_and_metric_semantics"


def build_comparison_payload(
    *,
    paper_result: dict[str, Any],
    mire_result: dict[str, Any],
) -> dict[str, Any]:
    """Build the fixed comparison payload for paper-distribution vs MI-RE."""
    paper_metrics = paper_result["structural_metrics"]
    mire_metrics = mire_result["structural_metrics"]
    delta_summary = {
        "cosine_similarity": _metric_delta(paper_metrics, mire_metrics, key="cosine_similarity"),
        "explained_variance": _metric_delta(
            paper_metrics, mire_metrics, key="explained_variance"
        ),
        "ev_openmoss_aligned": _metric_delta(
            paper_metrics, mire_metrics, key="ev_openmoss_aligned"
        ),
        "explained_variance_paper_raw": _metric_delta(
            paper_metrics, mire_metrics, key="explained_variance_paper_raw"
        ),
        "explained_variance_paper_normalized": _metric_delta(
            paper_metrics, mire_metrics, key="explained_variance_paper_normalized"
        ),
        "fvu": _metric_delta(paper_metrics, mire_metrics, key="fvu"),
        "ce_loss_delta": _metric_delta(paper_metrics, mire_metrics, key="ce_loss_delta"),
        "kl_divergence": _metric_delta(paper_metrics, mire_metrics, key="kl_divergence"),
        "dead_ratio": _metric_delta(paper_metrics, mire_metrics, key="dead_ratio"),
    }
    return {
        "paper_distribution": paper_result,
        "mi_re_baseline": mire_result,
        "delta_summary": delta_summary,
        "literature_alignment": _build_literature_alignment(
            paper_metrics=paper_metrics,
            mire_metrics=mire_metrics,
        ),
        "diagnosis_hint": _diagnose_distribution_shift(
            paper_metrics=paper_metrics,
            mire_metrics=mire_metrics,
        ),
    }


def collect_residual_batches(
    *,
    model: Any,
    tokenizer: Any,
    texts: list[str],
    hook_point: str,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Capture hook-point residual activations for a bounded text list."""
    batches: list[dict[str, Any]] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        if is_transformer_lens_model(model):
            with torch.inference_mode():
                _, cache = model.run_with_cache(
                    encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    return_type="logits",
                    return_cache_object=False,
                    names_filter=lambda name: name == hook_point,
                )
            residual = cache[hook_point].detach().cpu()
        else:
            layer_idx = _parse_hook_point(hook_point)
            target_layer = model.model.layers[layer_idx]
            captured: dict[str, torch.Tensor] = {}

            def _hook_fn(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured["hidden"] = hidden.detach()

            handle = target_layer.register_forward_hook(_hook_fn)
            with torch.inference_mode():
                model(**encoded)
            handle.remove()
            residual = captured["hidden"].detach().cpu()

        batches.append(
            {
                "texts": batch_texts,
                "attention_mask": encoded["attention_mask"].detach().cpu(),
                "residual": residual,
            }
        )
    return batches


class ReferenceCheckpointAdapter:
    """Independent SAE forward pass derived directly from checkpoint metadata."""

    def __init__(
        self,
        *,
        d_model: int,
        d_sae: int,
        act_fn: str,
        jump_relu_threshold: float,
        norm_scale: float | None,
        top_k: int | None,
        checkpoint_topk_semantics: str,
        use_decoder_bias: bool,
        b_pre: torch.Tensor,
        b_enc: torch.Tensor,
        b_dec: torch.Tensor,
        W_enc: torch.Tensor,
        W_dec: torch.Tensor,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.d_model = d_model
        self.d_sae = d_sae
        self.act_fn = act_fn.lower()
        self.jump_relu_threshold = float(jump_relu_threshold)
        self.norm_scale = norm_scale
        self.top_k = top_k
        self.checkpoint_topk_semantics = checkpoint_topk_semantics
        self.use_decoder_bias = use_decoder_bias
        self.device = torch.device(device)
        self.dtype = dtype
        self.b_pre = b_pre.to(device=self.device, dtype=self.dtype)
        self.b_enc = b_enc.to(device=self.device, dtype=self.dtype)
        self.b_dec = b_dec.to(device=self.device, dtype=self.dtype)
        self.W_enc = W_enc.to(device=self.device, dtype=self.dtype)
        self.W_dec = W_dec.to(device=self.device, dtype=self.dtype)

    @classmethod
    def from_state_dict(
        cls,
        *,
        hyperparams: dict[str, Any],
        state_dict: dict[str, torch.Tensor],
        device: str | torch.device,
        dtype: torch.dtype,
        checkpoint_topk_semantics: str = "disabled",
    ) -> "ReferenceCheckpointAdapter":
        d_model = int(hyperparams["d_model"])
        d_sae = int(hyperparams["d_sae"])
        act_fn = str(hyperparams.get("act_fn", "jumprelu"))
        threshold = float(hyperparams.get("jump_relu_threshold", 0.0))
        norm_info = hyperparams.get("dataset_average_activation_norm", {})
        if isinstance(norm_info, dict):
            norm_scale = norm_info.get("in")
        else:
            norm_scale = norm_info
        top_k = hyperparams.get("top_k")
        use_decoder_bias = bool(hyperparams.get("use_decoder_bias", True))

        weight_map = {
            "encoder.weight": "W_enc",
            "encoder.bias": "b_enc",
            "decoder.weight": "W_dec",
            "decoder.bias": "b_dec",
            "pre_bias": "b_pre",
            "W_enc": "W_enc",
            "b_enc": "b_enc",
            "W_dec": "W_dec",
            "b_dec": "b_dec",
            "b_pre": "b_pre",
            "encoder.W": "W_enc",
            "encoder.b": "b_enc",
            "decoder.W": "W_dec",
            "decoder.b": "b_dec",
            "encoder_bias": "b_enc",
            "decoder_bias": "b_dec",
            "pre_encoder_bias": "b_pre",
            "b_dec_out": "b_dec",
        }
        prefixes = ("sae.", "model.", "sparse_autoencoder.", "ae.", "autoencoder.")

        mapped: dict[str, torch.Tensor] = {}
        for raw_key, tensor in state_dict.items():
            clean_key = raw_key
            for prefix in prefixes:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    break
            if clean_key in weight_map:
                mapped[weight_map[clean_key]] = tensor

        if "W_enc" not in mapped or "W_dec" not in mapped or "b_enc" not in mapped:
            raise RuntimeError(
                "Reference adapter could not find the critical checkpoint keys "
                f"(available={sorted(mapped.keys())})."
            )

        W_enc = mapped["W_enc"]
        W_dec = mapped["W_dec"]
        if tuple(W_enc.shape) == (d_model, d_sae):
            W_enc = W_enc.T
        if tuple(W_dec.shape) == (d_sae, d_model):
            W_dec = W_dec.T
        if tuple(W_enc.shape) != (d_sae, d_model):
            raise RuntimeError(f"Unexpected W_enc shape for reference adapter: {tuple(W_enc.shape)}")
        if tuple(W_dec.shape) != (d_model, d_sae):
            raise RuntimeError(f"Unexpected W_dec shape for reference adapter: {tuple(W_dec.shape)}")

        b_pre = mapped.get("b_pre", torch.zeros(d_model))
        b_dec = mapped.get("b_dec", torch.zeros(d_model))

        return cls(
            d_model=d_model,
            d_sae=d_sae,
            act_fn=act_fn,
            jump_relu_threshold=threshold,
            norm_scale=float(norm_scale) if norm_scale is not None else None,
            top_k=int(top_k) if top_k is not None else None,
            checkpoint_topk_semantics=checkpoint_topk_semantics,
            use_decoder_bias=use_decoder_bias,
            b_pre=b_pre,
            b_enc=mapped["b_enc"],
            b_dec=b_dec,
            W_enc=W_enc,
            W_dec=W_dec,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_hub(
        cls,
        *,
        repo_id: str,
        subfolder: str,
        device: str | torch.device,
        dtype: torch.dtype,
        checkpoint_topk_semantics: str = "disabled",
    ) -> "ReferenceCheckpointAdapter":
        hp_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{subfolder}/hyperparams.json",
        )
        with open(hp_path, "r", encoding="utf-8") as f:
            hyperparams = json.load(f)

        snapshot_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=f"{subfolder}/checkpoints/*",
        )
        ckpt_dir = Path(snapshot_dir) / subfolder / "checkpoints"
        safetensor_files = sorted(ckpt_dir.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {ckpt_dir}")

        raw_state_dict: dict[str, torch.Tensor] = {}
        for sf_path in safetensor_files:
            raw_state_dict.update(load_safetensors(str(sf_path)))

        return cls.from_state_dict(
            hyperparams=hyperparams,
            state_dict=raw_state_dict,
            device=device,
            dtype=dtype,
            checkpoint_topk_semantics=checkpoint_topk_semantics,
        )

    def _apply_sparse_activation(self, pre_activation: torch.Tensor) -> torch.Tensor:
        if self.act_fn in {"relu", "topk", "topk_relu", "topk-relu"}:
            latents = torch.relu(pre_activation)
        else:
            latents = pre_activation * (pre_activation > self.jump_relu_threshold).to(
                pre_activation.dtype
            )

        if self.checkpoint_topk_semantics != "hard":
            return latents
        if self.top_k is None or self.top_k <= 0 or self.top_k >= latents.shape[-1]:
            return latents

        topk_indices = latents.topk(self.top_k, dim=-1).indices
        keep_mask = torch.zeros_like(latents, dtype=torch.bool)
        keep_mask.scatter_(-1, topk_indices, True)
        return latents * keep_mask.to(latents.dtype)

    def forward_with_details(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(device=self.device, dtype=self.dtype)
        if self.norm_scale is None:
            x_normalized = x
            input_norm = None
        else:
            input_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
            x_normalized = x * (self.norm_scale / input_norm)

        pre_activation = (x_normalized - self.b_pre) @ self.W_enc.T + self.b_enc
        latents = self._apply_sparse_activation(pre_activation)
        reconstructed_normalized = latents @ self.W_dec.T + self.b_dec
        if self.norm_scale is None or input_norm is None:
            reconstructed_raw = reconstructed_normalized
        else:
            reconstructed_raw = reconstructed_normalized * (input_norm / self.norm_scale)
        return {
            "input_normalized": x_normalized,
            "reconstructed_normalized": reconstructed_normalized,
            "reconstructed_raw": reconstructed_raw,
            "latents": latents,
        }


def load_reference_sae_backend(
    *,
    repo_id: str,
    subfolder: str,
    device: str | torch.device,
    dtype: torch.dtype,
    official_loader: str | None = None,
    official_repo_dir: str | Path | None = None,
    checkpoint_topk_semantics: str = "disabled",
) -> tuple[Any, str]:
    """Load an official backend when available, else fall back to checkpoint formulas."""
    if official_loader:
        inserted_paths: list[str] = []
        try:
            if official_repo_dir is not None:
                repo_root = Path(official_repo_dir)
                for candidate in (repo_root, repo_root / "src"):
                    if candidate.exists():
                        sys.path.insert(0, str(candidate))
                        inserted_paths.append(str(candidate))

            module_name, callable_name = official_loader.split(":", 1)
            module = importlib.import_module(module_name)
            factory = getattr(module, callable_name)
            backend = factory(
                repo_id=repo_id,
                subfolder=subfolder,
                device=device,
                dtype=dtype,
                checkpoint_topk_semantics=checkpoint_topk_semantics,
            )
            return backend, "official_loader"
        finally:
            for inserted in reversed(inserted_paths):
                if inserted in sys.path:
                    sys.path.remove(inserted)

    backend = ReferenceCheckpointAdapter.from_hub(
        repo_id=repo_id,
        subfolder=subfolder,
        device=device,
        dtype=dtype,
        checkpoint_topk_semantics=checkpoint_topk_semantics,
    )
    return backend, "checkpoint_formula_fallback"


def build_parity_report(
    *,
    local_details_list: list[dict[str, torch.Tensor]],
    reference_details_list: list[dict[str, torch.Tensor]],
    attention_masks: list[torch.Tensor],
    topk_compare: int = 10,
) -> dict[str, Any]:
    """Aggregate per-batch SAE outputs into a stable parity report."""
    if not local_details_list or not reference_details_list or not attention_masks:
        raise ValueError("Parity report requires at least one evaluated batch.")
    local_recon_parts: list[torch.Tensor] = []
    ref_recon_parts: list[torch.Tensor] = []
    local_latent_parts: list[torch.Tensor] = []
    ref_latent_parts: list[torch.Tensor] = []

    for local_details, ref_details, mask in zip(
        local_details_list, reference_details_list, attention_masks
    ):
        flat_mask = mask.bool().reshape(-1)
        local_recon_parts.append(
            local_details["reconstructed_raw"].detach().cpu().float().reshape(
                -1, local_details["reconstructed_raw"].shape[-1]
            )[flat_mask]
        )
        ref_recon_parts.append(
            ref_details["reconstructed_raw"].detach().cpu().float().reshape(
                -1, ref_details["reconstructed_raw"].shape[-1]
            )[flat_mask]
        )
        local_latent_parts.append(
            local_details["latents"].detach().cpu().float().reshape(
                -1, local_details["latents"].shape[-1]
            )[flat_mask]
        )
        ref_latent_parts.append(
            ref_details["latents"].detach().cpu().float().reshape(
                -1, ref_details["latents"].shape[-1]
            )[flat_mask]
        )

    local_recon = torch.cat(local_recon_parts, dim=0)
    ref_recon = torch.cat(ref_recon_parts, dim=0)
    local_latents = torch.cat(local_latent_parts, dim=0)
    ref_latents = torch.cat(ref_latent_parts, dim=0)

    max_abs_diff = float((local_recon - ref_recon).abs().max().item())
    reconstruction_cosine = float(
        F.cosine_similarity(local_recon, ref_recon, dim=-1).mean().item()
    )
    latent_l0_local = float((local_latents.abs() > 1e-8).float().sum(dim=-1).mean().item())
    latent_l0_official = float((ref_latents.abs() > 1e-8).float().sum(dim=-1).mean().item())

    k = min(topk_compare, local_latents.shape[-1])
    topk_latent_ids_local = (
        local_latents.max(dim=0).values.topk(k).indices.detach().cpu().tolist()
    )
    topk_latent_ids_official = (
        ref_latents.max(dim=0).values.topk(k).indices.detach().cpu().tolist()
    )
    topk_exact_match = topk_latent_ids_local == topk_latent_ids_official
    parity_passed = (
        max_abs_diff <= 1e-4
        and reconstruction_cosine >= 0.9999
        and abs(latent_l0_local - latent_l0_official) <= 1e-4
        and topk_exact_match
    )

    return {
        "reconstruction_max_abs_diff": max_abs_diff,
        "reconstruction_cosine": reconstruction_cosine,
        "latent_l0_local": latent_l0_local,
        "latent_l0_official": latent_l0_official,
        "topk_latent_ids_local": topk_latent_ids_local,
        "topk_latent_ids_official": topk_latent_ids_official,
        "topk_exact_match": topk_exact_match,
        "parity_passed": parity_passed,
        "n_eval_tokens": int(local_recon.shape[0]),
        "topk_compare": int(k),
    }
