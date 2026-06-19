"""Gemma3 + GemmaScope SAE evaluation pipeline for MISC labels."""

from __future__ import annotations

import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import PREFERRED_LABEL_ORDER
from .gemma_scope_sae import (
    DEFAULT_GEMMA_SCOPE_LAYER_IDX,
    DEFAULT_GEMMA_SCOPE_REPO_ID,
    GemmaScopeJumpReLUSAE,
    load_gemma_scope_sae,
)
from .layer_probe import discover_decoder_layers
from .misc_label_mapping import (
    load_misc_annotation_records,
    run_misc_label_mapping,
    select_labels,
    write_json,
    write_jsonl,
    write_label_indicator_csv,
)


DEFAULT_GEMMA_MODEL_ID = "google/gemma-3-4b-pt"
DEFAULT_LOCAL_GEMMA_DIR = "models/gemma-3-4b-pt"
DEFAULT_OUTPUT_DIR = "outputs/gemma3_l18_gemmascope_sae_eval"
DEFAULT_LABELS = tuple(label for label in PREFERRED_LABEL_ORDER if label != "OTHER")


@dataclass(frozen=True)
class GemmaScopeEvaluationConfig:
    model_source: str
    sae_repo_id: str = DEFAULT_GEMMA_SCOPE_REPO_ID
    sae_subfolder: str | None = None
    layer_idx: int = DEFAULT_GEMMA_SCOPE_LAYER_IDX
    data_dir: str = "data/mi_quality_counseling_misc"
    output_dir: str = DEFAULT_OUTPUT_DIR
    labels: tuple[str, ...] = DEFAULT_LABELS
    max_seq_len: int = 128
    batch_size: int = 2
    aggregation: str = "max"
    model_dtype: str = "bfloat16"
    sae_dtype: str = "float32"
    device: str | None = None
    device_map: str = "auto"
    cache_dir: str | None = None
    limit_records: int | None = None
    fdr_alpha: float = 0.05
    association_chunk_size: int = 512
    min_positive: int = 10
    min_negative: int = 10
    skip_label_mapping: bool = False


class StructuralAccumulator:
    """Online structural metrics over valid token activations."""

    def __init__(self) -> None:
        self.total_elements = 0
        self.total_tokens = 0
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        self.sum_sse = 0.0
        self.sum_abs_err = 0.0
        self.sum_cosine = 0.0
        self.sum_l0 = 0.0
        self.sum_l0_sq = 0.0
        self.invalid_tokens = 0

    def update(
        self,
        *,
        activations: torch.Tensor,
        reconstructed: torch.Tensor,
        latents: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        mask = attention_mask.to(device=activations.device, dtype=torch.bool)
        if not bool(mask.any()):
            return
        x = activations.detach().float()[mask]
        x_hat = reconstructed.detach().float()[mask]
        z = latents.detach()[mask]
        finite_rows = torch.isfinite(x).all(dim=-1) & torch.isfinite(x_hat).all(dim=-1)
        self.invalid_tokens += int((~finite_rows).sum().item())
        if not bool(finite_rows.any()):
            return
        x = x[finite_rows]
        x_hat = x_hat[finite_rows]
        z = z[finite_rows]
        err = x - x_hat
        self.total_tokens += int(x.shape[0])
        self.total_elements += int(x.numel())
        self.sum_x += float(x.sum().item())
        self.sum_x2 += float((x * x).sum().item())
        self.sum_sse += float((err * err).sum().item())
        self.sum_abs_err += float(err.abs().sum().item())
        self.sum_cosine += float(F.cosine_similarity(x, x_hat, dim=-1).sum().item())
        l0 = (z > 0).sum(dim=-1).float()
        self.sum_l0 += float(l0.sum().item())
        self.sum_l0_sq += float((l0 * l0).sum().item())

    def to_dict(self) -> dict[str, Any]:
        if self.total_elements == 0:
            return {}
        variance = self.sum_x2 - (self.sum_x * self.sum_x / max(self.total_elements, 1))
        mse = self.sum_sse / self.total_elements
        l0_mean = self.sum_l0 / max(self.total_tokens, 1)
        l0_var = self.sum_l0_sq / max(self.total_tokens, 1) - l0_mean * l0_mean
        return {
            "n_valid_tokens": int(self.total_tokens),
            "n_invalid_tokens_skipped": int(self.invalid_tokens),
            "n_activation_elements": int(self.total_elements),
            "mse": float(mse),
            "mae": float(self.sum_abs_err / self.total_elements),
            "explained_variance": float(1.0 - self.sum_sse / variance) if variance > 0 else 0.0,
            "cosine_similarity": float(self.sum_cosine / max(self.total_tokens, 1)),
            "l0_mean": float(l0_mean),
            "l0_std": float(math.sqrt(max(l0_var, 0.0))),
        }


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[str(name).lower()]


def resolve_gemma_model_source(model_source: str | None = None) -> str:
    if model_source:
        return model_source
    local = Path(DEFAULT_LOCAL_GEMMA_DIR)
    return str(local) if local.exists() else DEFAULT_GEMMA_MODEL_ID


def _configure_hf_endpoint(hf_endpoint: str | None) -> None:
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        os.environ["HF_HUB_ENDPOINT"] = hf_endpoint


def load_gemma_tokenizer(source: str, *, trust_remote_code: bool = False, token: str | None = None) -> Any:
    from transformers import AutoProcessor, AutoTokenizer

    kwargs = {"trust_remote_code": trust_remote_code}
    if token:
        kwargs["token"] = token
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, **kwargs)
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(source, use_fast=False, **kwargs)
        except Exception:
            processor = AutoProcessor.from_pretrained(source, **kwargs)
            tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_gemma_model(
    source: str,
    *,
    torch_dtype: torch.dtype,
    device_map: str = "auto",
    device: str | None = None,
    cache_dir: str | None = None,
    trust_remote_code: bool = False,
    token: str | None = None,
) -> torch.nn.Module:
    import transformers

    base_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if cache_dir:
        base_kwargs["cache_dir"] = cache_dir
    if token:
        base_kwargs["token"] = token
    if str(device_map).lower() != "none":
        base_kwargs["device_map"] = device_map

    model_classes: list[type[Any]] = []
    for class_name in (
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ):
        cls = getattr(transformers, class_name, None)
        if cls is not None:
            model_classes.append(cls)

    errors: list[str] = []
    model: torch.nn.Module | None = None
    for model_cls in model_classes:
        for dtype_key in ("dtype", "torch_dtype"):
            kwargs = dict(base_kwargs)
            kwargs[dtype_key] = torch_dtype
            try:
                model = model_cls.from_pretrained(source, **kwargs)
                break
            except TypeError as exc:
                errors.append(f"{model_cls.__name__}({dtype_key}): {exc}")
                continue
            except Exception as exc:
                errors.append(f"{model_cls.__name__}({dtype_key}): {exc}")
                model = None
                break
        if model is not None:
            break
    if model is None:
        joined = "\n".join(errors[-8:])
        raise RuntimeError(f"Could not load Gemma model from {source!r}.\n{joined}")

    if str(device_map).lower() == "none":
        target = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(target)
    model.eval()
    return model


def _first_device(model: torch.nn.Module) -> torch.device:
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tokenize(tokenizer: Any, texts: list[str], *, max_seq_len: int, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in encoded.items() if isinstance(v, torch.Tensor)}


def aggregate_tokens(values: torch.Tensor, attention_mask: torch.Tensor, method: str) -> torch.Tensor:
    mask = attention_mask.to(device=values.device, dtype=torch.bool)
    if method == "mean":
        mask_f = mask.to(dtype=values.dtype).unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1)
        return (values * mask_f).sum(dim=1) / denom
    if method == "max":
        masked = values.masked_fill(~mask.unsqueeze(-1), -torch.inf)
        out = masked.max(dim=1).values
        return torch.nan_to_num(out, neginf=0.0, posinf=0.0)
    raise ValueError("aggregation must be one of: max, mean")


def _texts_from_records(records: list[dict[str, Any]]) -> list[str]:
    texts = []
    for record in records:
        text = str(record.get("text") or record.get("unit_text") or "").strip()
        if not text:
            sample_id = record.get("sample_id") or record.get("record_id") or "<unknown>"
            raise ValueError(f"Record {sample_id} has no text/unit_text.")
        texts.append(text)
    return texts


def extract_gemma_scope_feature_store(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    sae: GemmaScopeJumpReLUSAE,
    texts: list[str],
    layer_idx: int,
    output_dir: str | Path,
    max_seq_len: int = 128,
    batch_size: int = 2,
    aggregation: str = "max",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    feature_dir = output_path / "feature_store"
    feature_dir.mkdir(parents=True, exist_ok=True)

    layers, layer_path = discover_decoder_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"layer_idx={layer_idx} outside discovered layer range 0..{len(layers)-1}")

    device = _first_device(model)
    captured: dict[str, torch.Tensor] = {}

    def _hook_fn(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        captured["hidden_states"] = output[0].detach() if isinstance(output, tuple) else output.detach()

    handle = layers[layer_idx].register_forward_hook(_hook_fn)
    feature_batches: list[torch.Tensor] = []
    activation_batches: list[torch.Tensor] = []
    structural = StructuralAccumulator()

    try:
        for start in tqdm(range(0, len(texts), batch_size), desc="GemmaScope extract", unit="batch"):
            batch_texts = texts[start : start + batch_size]
            encoded = _tokenize(tokenizer, batch_texts, max_seq_len=max_seq_len, device=device)
            mask = encoded["attention_mask"]
            captured.clear()
            with torch.inference_mode():
                try:
                    _ = model(**encoded, use_cache=False)
                except TypeError:
                    _ = model(**encoded)
            hidden = captured.get("hidden_states")
            if hidden is None:
                raise RuntimeError(f"Layer hook did not capture activations for layer {layer_idx}.")
            hidden = hidden.to(device=sae.device, dtype=sae.sae_dtype)
            mask = mask.to(sae.device)
            with torch.inference_mode():
                sae_out = sae.forward_with_details(hidden)
            latents = sae_out["latents"]
            reconstructed = sae_out["reconstructed_raw"]
            structural.update(
                activations=hidden,
                reconstructed=reconstructed,
                latents=latents,
                attention_mask=mask,
            )
            feature_batches.append(aggregate_tokens(latents, mask, aggregation).detach().cpu().float())
            activation_batches.append(aggregate_tokens(hidden, mask, aggregation).detach().cpu().float())
            del encoded, hidden, mask, latents, reconstructed, sae_out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        handle.remove()

    utterance_features = torch.cat(feature_batches, dim=0)
    utterance_activations = torch.cat(activation_batches, dim=0)
    features_path = feature_dir / "utterance_features.pt"
    activations_path = feature_dir / "utterance_activations.pt"
    metadata_path = feature_dir / "feature_metadata.json"
    torch.save(
        {
            "utterance_features": utterance_features,
            "aggregation": aggregation,
            "layer_idx": layer_idx,
            "layer_path": layer_path,
            "sae_config": sae.config.to_dict(),
        },
        features_path,
    )
    torch.save(
        {
            "utterance_activations": utterance_activations,
            "aggregation": aggregation,
            "layer_idx": layer_idx,
            "layer_path": layer_path,
        },
        activations_path,
    )
    metadata = {
        "n_records": int(len(texts)),
        "feature_shape": list(utterance_features.shape),
        "activation_shape": list(utterance_activations.shape),
        "aggregation": aggregation,
        "max_seq_len": int(max_seq_len),
        "batch_size": int(batch_size),
        "layer_idx": int(layer_idx),
        "layer_path": layer_path,
        "sae_config": sae.config.to_dict(),
        "files": {
            "utterance_features": str(features_path),
            "utterance_activations": str(activations_path),
        },
    }
    write_json(metadata_path, metadata)
    return {
        "utterance_features": utterance_features,
        "utterance_activations": utterance_activations,
        "structural_metrics": structural.to_dict(),
        "metadata": metadata,
    }


def write_gemma_scope_report(output_dir: str | Path, summary: dict[str, Any]) -> Path:
    output_path = Path(output_dir)
    report_path = output_path / "gemma_scope_sae_report.md"
    metrics = summary.get("structural_metrics", {})
    files = summary.get("files", {})
    lines = [
        "# Gemma3-4B GemmaScope Layer-18 SAE Evaluation",
        "",
        "## Summary",
        "",
        f"- Model source: `{summary.get('model_source')}`",
        f"- SAE checkpoint: `{summary.get('sae_repo_id')}/{summary.get('sae_subfolder')}`",
        f"- Layer index: `{summary.get('layer_idx')}` (0-based)",
        f"- Records: `{summary.get('n_records')}`",
        f"- Feature shape: `{summary.get('feature_shape')}`",
        f"- Aggregation: `{summary.get('aggregation')}`",
        "",
        "## Structural Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in ("explained_variance", "mse", "mae", "cosine_similarity", "l0_mean", "l0_std"):
        value = metrics.get(key)
        if value is not None:
            lines.append(f"| `{key}` | {float(value):.6f} |")
    lines.extend(
        [
            "",
            "## Generated Files",
            "",
        ]
    )
    for key, value in files.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- This run provides correlational and structural evidence over GemmaScope SAE latents.",
            "- Layer 18 is selected because the full-MISC Gemma layer probe found `RE` best at `layer_idx=18`.",
            "- `QU` had a different best layer (`layer_idx=10`), so QU findings here should be read as layer-18 robustness checks.",
            "- These outputs do not establish causal sufficiency without downstream intervention experiments.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_gemma_scope_sae_evaluation(
    *,
    config: GemmaScopeEvaluationConfig,
    hf_endpoint: str | None = None,
    trust_remote_code: bool = False,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Run Gemma3 layer-18 GemmaScope SAE extraction and MISC mapping."""

    started = time.time()
    _configure_hf_endpoint(hf_endpoint)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_misc_annotation_records(config.data_dir, limit=config.limit_records)
    texts = _texts_from_records(records)
    labels, label_indicators = select_labels(records, labels=list(config.labels))

    tokenizer = load_gemma_tokenizer(
        config.model_source,
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )
    model = load_gemma_model(
        config.model_source,
        torch_dtype=dtype_from_name(config.model_dtype),
        device_map=config.device_map,
        device=config.device,
        cache_dir=config.cache_dir,
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )
    sae_device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sae = load_gemma_scope_sae(
        repo_id=config.sae_repo_id,
        subfolder=config.sae_subfolder,
        layer_idx=config.layer_idx,
        cache_dir=config.cache_dir,
        device=sae_device,
        dtype=dtype_from_name(config.sae_dtype),
    )

    feature_payload = extract_gemma_scope_feature_store(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        texts=texts,
        layer_idx=config.layer_idx,
        output_dir=output_dir,
        max_seq_len=config.max_seq_len,
        batch_size=config.batch_size,
        aggregation=config.aggregation,
    )
    del model, sae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    write_jsonl(output_dir / "records.jsonl", records)
    write_label_indicator_csv(
        output_dir / "label_matrix.csv",
        records,
        labels,
        label_indicators,
    )

    mapping_summary: dict[str, Any] | None = None
    if not config.skip_label_mapping:
        mapping_summary = run_misc_label_mapping(
            records=records,
            features=feature_payload["utterance_features"],
            output_dir=output_dir / "functional" / "misc_label_mapping",
            labels=list(config.labels),
            fdr_alpha=config.fdr_alpha,
            precision_k_values=[10, 50, 100],
            min_positive=config.min_positive,
            min_negative=config.min_negative,
            chunk_size=config.association_chunk_size,
            top_k_per_label=50,
        )

    feature_shape = list(feature_payload["utterance_features"].shape)
    activation_shape = list(feature_payload["utterance_activations"].shape)
    summary = {
        "model_source": config.model_source,
        "sae_repo_id": config.sae_repo_id,
        "sae_subfolder": feature_payload["metadata"]["sae_config"]["subfolder"],
        "layer_idx": int(config.layer_idx),
        "n_records": int(len(records)),
        "labels": list(labels),
        "label_counts": {
            label: int(label_indicators[:, idx].sum())
            for idx, label in enumerate(labels)
        },
        "feature_shape": feature_shape,
        "activation_shape": activation_shape,
        "aggregation": config.aggregation,
        "structural_metrics": feature_payload["structural_metrics"],
        "runtime_seconds": round(time.time() - started, 3),
        "config": asdict(config),
        "files": {
            "records": str(output_dir / "records.jsonl"),
            "label_matrix": str(output_dir / "label_matrix.csv"),
            "feature_store": str(output_dir / "feature_store" / "utterance_features.pt"),
            "raw_hidden": str(output_dir / "feature_store" / "utterance_activations.pt"),
            "metrics_structural": str(output_dir / "metrics_structural.json"),
            "mapping_matrix": str(
                output_dir / "functional" / "misc_label_mapping" / "latent_label_matrix.csv"
            ),
        },
        "mapping_summary": mapping_summary,
    }
    write_json(output_dir / "metrics_structural.json", feature_payload["structural_metrics"])
    write_json(output_dir / "run_summary.json", summary)
    report_path = write_gemma_scope_report(output_dir, summary)
    summary["files"]["report"] = str(report_path)
    write_json(output_dir / "run_summary.json", summary)
    return summary
