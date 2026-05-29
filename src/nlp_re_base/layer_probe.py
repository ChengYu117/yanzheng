"""Layer-wise probing utilities for MISC concepts.

This module probes raw transformer hidden states instead of SAE latents. It is
designed for Gemma 3 style Hugging Face models, but the layer discovery is
intentionally generic enough for simple causal LM wrappers used in tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from .eval_functional import (
    _extract_probe_weights,
    _fit_torch_probe,
    _predict_torch_probe,
)


DEFAULT_CORE_LABELS = ["RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF", "OTHER"]
DEFAULT_POOLING = ["mean", "last"]


@dataclass
class LayerFeatureStore:
    """Paths and metadata for cached layer features."""

    output_dir: Path
    feature_dir: Path
    layer_path: str
    n_layers: int
    n_records: int
    hidden_size: int
    pooling: list[str]
    feature_dtype: str
    max_seq_len: int

    def feature_path(self, layer_idx: int, pooling: str) -> Path:
        return self.feature_dir / f"layer_{layer_idx:02d}_{pooling}.npy"


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def _resolve_attr(root: object, dotted_path: str) -> Any:
    current = root
    for part in dotted_path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def discover_decoder_layers(model: object) -> tuple[list[torch.nn.Module], str]:
    """Return decoder layers and their attribute path.

    Supported paths cover Gemma3ForCausalLM, Gemma3ForConditionalGeneration,
    language-model wrappers, and toy models used in tests.
    """

    candidate_paths = [
        "language_model.layers",
        "model.layers",
        "model.model.layers",
        "model.language_model.layers",
        "language_model.model.layers",
        "model.language_model.model.layers",
        "model.model.language_model.layers",
        "model.model.language_model.model.layers",
    ]
    for path in candidate_paths:
        layers = _resolve_attr(model, path)
        if layers is None:
            continue
        try:
            layer_list = list(layers)
        except TypeError:
            continue
        if layer_list and all(isinstance(layer, torch.nn.Module) for layer in layer_list):
            return layer_list, path

    raise ValueError(
        "Could not discover decoder layers. Tried: "
        + ", ".join(candidate_paths)
    )


def _first_parameter_device(model: torch.nn.Module) -> torch.device:
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tokenize_batch(
    tokenizer: Any,
    texts: list[str],
    max_seq_len: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    return {
        key: value.to(device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }


def _pool_hidden_states(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    mask = attention_mask.to(device=hidden.device, dtype=torch.float32)
    if pooling == "mean":
        mask_3d = mask.unsqueeze(-1)
        token_counts = mask_3d.sum(dim=1).clamp(min=1.0)
        return (hidden.float() * mask_3d).sum(dim=1) / token_counts

    if pooling == "last":
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        last_indices = (positions * mask.long()).max(dim=1).values
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden.float()[batch_indices, last_indices, :]

    raise ValueError(f"Unsupported pooling method: {pooling}")


def _feature_dtype_numpy(dtype_name: str) -> np.dtype:
    if dtype_name == "float16":
        return np.dtype("float16")
    if dtype_name == "float32":
        return np.dtype("float32")
    raise ValueError("feature_dtype must be one of: float16, float32")


def _metadata_path(output_dir: Path) -> Path:
    return output_dir / "feature_store" / "layer_feature_metadata.json"


def _can_reuse_feature_store(
    output_dir: Path,
    *,
    n_records: int,
    n_layers: int,
    pooling: list[str],
    max_seq_len: int,
) -> LayerFeatureStore | None:
    metadata_file = _metadata_path(output_dir)
    if not metadata_file.exists():
        return None
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    if int(metadata.get("n_records", -1)) != n_records:
        return None
    if int(metadata.get("n_layers", -1)) != n_layers:
        return None
    if int(metadata.get("max_seq_len", -1)) != max_seq_len:
        return None
    if not set(pooling).issubset(set(metadata.get("pooling", []))):
        return None

    store = LayerFeatureStore(
        output_dir=output_dir,
        feature_dir=output_dir / "feature_store",
        layer_path=str(metadata["layer_path"]),
        n_layers=n_layers,
        n_records=n_records,
        hidden_size=int(metadata["hidden_size"]),
        pooling=list(pooling),
        feature_dtype=str(metadata.get("feature_dtype", "float16")),
        max_seq_len=max_seq_len,
    )
    for pool in pooling:
        for layer_idx in range(n_layers):
            if not store.feature_path(layer_idx, pool).exists():
                return None
    return store


def extract_layer_features(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    output_dir: str | Path,
    pooling: list[str] | None = None,
    max_seq_len: int = 128,
    batch_size: int = 2,
    device: str | torch.device | None = None,
    feature_dtype: str = "float16",
    reuse_cache: bool = True,
) -> LayerFeatureStore:
    """Extract utterance-level features for every decoder layer.

    Features are written as per-layer .npy files to avoid keeping the full
    [layers, records, hidden] tensor in RAM.
    """

    pooling = list(pooling or DEFAULT_POOLING)
    invalid = sorted(set(pooling).difference(DEFAULT_POOLING))
    if invalid:
        raise ValueError(f"Unsupported pooling values: {invalid}")

    output_path = Path(output_dir)
    feature_dir = output_path / "feature_store"
    feature_dir.mkdir(parents=True, exist_ok=True)

    layers, layer_path = discover_decoder_layers(model)
    n_layers = len(layers)
    n_records = len(texts)
    if n_records == 0:
        raise ValueError("No texts provided for layer probing.")

    cached = _can_reuse_feature_store(
        output_path,
        n_records=n_records,
        n_layers=n_layers,
        pooling=pooling,
        max_seq_len=max_seq_len,
    )
    if reuse_cache and cached is not None:
        return cached

    target_device = torch.device(device) if device is not None else _first_parameter_device(model)
    model.eval()
    np_dtype = _feature_dtype_numpy(feature_dtype)

    captured: dict[int, torch.Tensor] = {}
    handles = []

    def _make_hook(layer_idx: int):
        def _hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden.detach()

        return _hook

    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(_make_hook(idx)))

    memmaps: dict[tuple[int, str], np.memmap] = {}
    hidden_size: int | None = None

    try:
        row_start = 0
        for start in tqdm(
            range(0, n_records, batch_size),
            desc="Gemma layer feature extraction",
            unit="batch",
        ):
            batch_texts = texts[start : start + batch_size]
            encoded = _tokenize_batch(tokenizer, batch_texts, max_seq_len, target_device)
            attention_mask = encoded["attention_mask"]
            captured.clear()

            with torch.inference_mode():
                try:
                    _ = model(**encoded, use_cache=False)
                except TypeError:
                    _ = model(**encoded)

            missing = [idx for idx in range(n_layers) if idx not in captured]
            if missing:
                raise RuntimeError(f"Missing hidden states for layers: {missing[:8]}")

            batch_rows = len(batch_texts)
            for layer_idx in range(n_layers):
                hidden = captured[layer_idx]
                if hidden_size is None:
                    hidden_size = int(hidden.shape[-1])
                    for pool in pooling:
                        for init_idx in range(n_layers):
                            path = feature_dir / f"layer_{init_idx:02d}_{pool}.npy"
                            memmaps[(init_idx, pool)] = np.lib.format.open_memmap(
                                path,
                                mode="w+",
                                dtype=np_dtype,
                                shape=(n_records, hidden_size),
                            )

                for pool in pooling:
                    pooled = _pool_hidden_states(hidden, attention_mask, pool)
                    arr = pooled.detach().cpu().numpy().astype(np_dtype, copy=False)
                    memmaps[(layer_idx, pool)][row_start : row_start + batch_rows] = arr

            row_start += batch_rows
            captured.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()

    if hidden_size is None:
        raise RuntimeError("Feature extraction did not produce any hidden states.")

    for mmap in memmaps.values():
        mmap.flush()

    metadata = {
        "layer_path": layer_path,
        "n_layers": n_layers,
        "n_records": n_records,
        "hidden_size": hidden_size,
        "pooling": pooling,
        "feature_dtype": feature_dtype,
        "max_seq_len": max_seq_len,
        "feature_files": {
            f"layer_{idx:02d}_{pool}": str(feature_dir / f"layer_{idx:02d}_{pool}.npy")
            for idx in range(n_layers)
            for pool in pooling
        },
    }
    write_json(_metadata_path(output_path), metadata)

    return LayerFeatureStore(
        output_dir=output_path,
        feature_dir=feature_dir,
        layer_path=layer_path,
        n_layers=n_layers,
        n_records=n_records,
        hidden_size=hidden_size,
        pooling=pooling,
        feature_dtype=feature_dtype,
        max_seq_len=max_seq_len,
    )


def _effective_cv_folds(y: np.ndarray, requested_folds: int) -> int:
    y = np.asarray(y, dtype=np.int64)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    return max(0, min(int(requested_folds), n_pos, n_neg))


def _probe_cv_metrics(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
) -> dict[str, float | int]:
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.int64)
    folds = _effective_cv_folds(y, cv_folds)
    if folds < 2:
        return {
            "n_folds": folds,
            "auc_mean": np.nan,
            "auc_std": np.nan,
            "accuracy_mean": np.nan,
            "f1_mean": np.nan,
            "balanced_accuracy_mean": np.nan,
        }

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    aucs: list[float] = []
    accs: list[float] = []
    f1s: list[float] = []
    baccs: list[float] = []

    for train_idx, test_idx in skf.split(X, y):
        probe_state = _fit_torch_probe(X[train_idx], y[train_idx])
        y_pred, y_prob = _predict_torch_probe(probe_state, X[test_idx])
        y_test = y[test_idx]
        try:
            auc = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            auc = 0.5
        aucs.append(auc)
        accs.append(float(accuracy_score(y_test, y_pred)))
        f1s.append(float(f1_score(y_test, y_pred, zero_division=0)))
        baccs.append(float(balanced_accuracy_score(y_test, y_pred)))

    return {
        "n_folds": folds,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=0)),
        "accuracy_mean": float(np.mean(accs)),
        "f1_mean": float(np.mean(f1s)),
        "balanced_accuracy_mean": float(np.mean(baccs)),
    }


def evaluate_layer_probes(
    *,
    feature_store: LayerFeatureStore,
    label_indicators: np.ndarray,
    labels: list[str],
    output_dir: str | Path,
    cv_folds: int = 5,
    recognition_auc_threshold: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one-vs-rest probes for each label, layer, and pooling method."""

    output_path = Path(output_dir)
    rows: list[dict[str, Any]] = []
    label_indicators = np.asarray(label_indicators, dtype=bool)

    for label_idx, label in enumerate(labels):
        y = label_indicators[:, label_idx].astype(np.int64)
        n_positive = int(y.sum())
        n_negative = int(len(y) - n_positive)
        for pool in feature_store.pooling:
            for layer_idx in tqdm(
                range(feature_store.n_layers),
                desc=f"Probe {label}/{pool}",
                unit="layer",
            ):
                feature_path = feature_store.feature_path(layer_idx, pool)
                X = np.load(feature_path, mmap_mode="r")
                metrics = _probe_cv_metrics(X, y, cv_folds=cv_folds)
                auc_value = metrics["auc_mean"]
                recognized = bool(
                    not pd.isna(auc_value)
                    and float(auc_value) >= recognition_auc_threshold
                )
                rows.append(
                    {
                        "label": label,
                        "layer_idx": layer_idx,
                        "pooling": pool,
                        "n_positive": n_positive,
                        "n_negative": n_negative,
                        "recognized": recognized,
                        **metrics,
                    }
                )

    metrics_df = pd.DataFrame(rows)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(
            ["label", "auc_mean", "f1_mean", "balanced_accuracy_mean", "layer_idx"],
            ascending=[True, False, False, False, True],
        ).reset_index(drop=True)
        metrics_df["rank_within_label"] = (
            metrics_df.groupby("label").cumcount() + 1
        )

    best_df = summarize_best_layers(
        metrics_df,
        recognition_auc_threshold=recognition_auc_threshold,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path / "layer_probe_metrics.csv", index=False)
    best_df.to_csv(output_path / "best_layers_by_label.csv", index=False)
    return metrics_df, best_df


def summarize_best_layers(
    metrics_df: pd.DataFrame,
    *,
    recognition_auc_threshold: float = 0.70,
) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    best_rows: list[dict[str, Any]] = []
    for label, group in metrics_df.groupby("label", sort=False):
        sorted_group = group.sort_values(
            ["auc_mean", "f1_mean", "balanced_accuracy_mean", "layer_idx"],
            ascending=[False, False, False, True],
        )
        best = sorted_group.iloc[0].to_dict()
        recognized = bool(
            not pd.isna(best.get("auc_mean"))
            and float(best["auc_mean"]) >= recognition_auc_threshold
        )
        best_rows.append(
            {
                "label": label,
                "best_layer_idx": int(best["layer_idx"]),
                "best_pooling": str(best["pooling"]),
                "best_auc_mean": float(best["auc_mean"]),
                "best_auc_std": float(best["auc_std"]),
                "best_accuracy_mean": float(best["accuracy_mean"]),
                "best_f1_mean": float(best["f1_mean"]),
                "best_balanced_accuracy_mean": float(best["balanced_accuracy_mean"]),
                "n_positive": int(best["n_positive"]),
                "n_negative": int(best["n_negative"]),
                "recognized": recognized,
                "recommended_for_ablation": recognized,
            }
        )
    return pd.DataFrame(best_rows)


def export_best_probe_artifacts(
    *,
    feature_store: LayerFeatureStore,
    label_indicators: np.ndarray,
    best_df: pd.DataFrame,
    labels: list[str],
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    """Fit full-data probes for best rows and save lightweight artifacts."""

    artifact_dir = Path(output_dir) / "probe_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    artifacts: list[dict[str, Any]] = []

    for _, row in best_df.iterrows():
        label = str(row["label"])
        label_idx = label_to_idx[label]
        layer_idx = int(row["best_layer_idx"])
        pool = str(row["best_pooling"])
        y = label_indicators[:, label_idx].astype(np.int64)
        X = np.load(feature_store.feature_path(layer_idx, pool), mmap_mode="r")
        probe_state = _fit_torch_probe(np.asarray(X, dtype=np.float32), y)
        weight = _extract_probe_weights(probe_state)
        bias = float(probe_state["model"].bias.detach().cpu().numpy().reshape(-1)[0])
        artifact = {
            "label": label,
            "layer_idx": layer_idx,
            "pooling": pool,
            "recognized": bool(row["recognized"]),
            "recommended_for_ablation": bool(row["recommended_for_ablation"]),
            "auc_mean": float(row["best_auc_mean"]),
            "weight": weight,
            "bias": bias,
            "mean": probe_state["mean"].astype(np.float32).reshape(-1),
            "std": probe_state["std"].astype(np.float32).reshape(-1),
        }
        artifact_path = artifact_dir / f"{label}_layer{layer_idx:02d}_{pool}_probe.pt"
        torch.save(artifact, artifact_path)
        meta = {
            key: value
            for key, value in artifact.items()
            if key not in {"weight", "mean", "std"}
        }
        meta["artifact_path"] = str(artifact_path)
        artifacts.append(meta)

    write_json(artifact_dir / "manifest.json", {"artifacts": artifacts})
    return artifacts


def write_probe_report(
    *,
    output_dir: str | Path,
    best_df: pd.DataFrame,
    model_source: str,
    recognition_auc_threshold: float,
    report_title: str = "Gemma3 MISC Layer Probe Report",
) -> Path:
    output_path = Path(output_dir)
    report_path = output_path / "layer_probe_report.md"
    recognized_count = int(best_df["recognized"].sum()) if not best_df.empty else 0
    lines = [
        f"# {report_title}",
        "",
        f"- Model source: `{model_source}`",
        f"- Recognition threshold: `auc_mean >= {recognition_auc_threshold:.2f}`",
        f"- Labels recognized: `{recognized_count} / {len(best_df)}`",
        "",
        "| Label | Best layer | Pooling | AUC | F1 | Balanced acc. | Recognized |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for _, row in best_df.iterrows():
        lines.append(
            "| {label} | {layer} | {pooling} | {auc:.4f} | {f1:.4f} | {bacc:.4f} | {rec} |".format(
                label=row["label"],
                layer=int(row["best_layer_idx"]),
                pooling=row["best_pooling"],
                auc=float(row["best_auc_mean"]),
                f1=float(row["best_f1_mean"]),
                bacc=float(row["best_balanced_accuracy_mean"]),
                rec="yes" if bool(row["recognized"]) else "no",
            )
        )
    label_set = set(best_df["label"].astype(str)) if not best_df.empty else set()
    if label_set == {"RE"}:
        label_note = "- Binary screening used `RE` positives against NonRE negatives."
    elif "OTHER" in label_set:
        label_note = "- `OTHER` is kept for completeness, but it remains a heterogeneous fallback label."
    else:
        label_note = "- `OTHER` was excluded from this layer-selection run; the report focuses on the core MISC labels."

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `recognized=True` means the label crosses the configured linear-probe AUC threshold.",
            label_note,
            "- Recommended ablation layers are the best recognized layer per label.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def make_dataset_summary(
    *,
    records: list[dict[str, Any]],
    labels: list[str],
    label_indicators: np.ndarray,
    model_source: str,
    feature_store: LayerFeatureStore,
    cv_folds: int,
    recognition_auc_threshold: float,
) -> dict[str, Any]:
    label_counts = {
        label: int(label_indicators[:, idx].sum())
        for idx, label in enumerate(labels)
    }
    return {
        "model_source": model_source,
        "n_records": len(records),
        "labels": labels,
        "label_counts": label_counts,
        "layer_path": feature_store.layer_path,
        "n_layers": feature_store.n_layers,
        "hidden_size": feature_store.hidden_size,
        "pooling": feature_store.pooling,
        "max_seq_len": feature_store.max_seq_len,
        "cv_folds": cv_folds,
        "recognition_auc_threshold": recognition_auc_threshold,
    }


def simple_namespace_model(layers: torch.nn.ModuleList) -> SimpleNamespace:
    """Small helper used by tests to mimic nested HF wrappers."""

    return SimpleNamespace(model=SimpleNamespace(layers=layers))
