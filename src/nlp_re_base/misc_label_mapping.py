"""MISC label-to-SAE latent mapping utilities.

This module extends the binary RE/NonRE analysis into a multi-label
Latent x Label association matrix for MISC annotations.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

from .eval_functional import benjamini_hochberg


PREFERRED_LABEL_ORDER = [
    "RE",
    "RES",
    "REC",
    "QU",
    "QUO",
    "QUC",
    "GI",
    "SU",
    "AF",
    "OTHER",
]


def sanitize_label(label: str) -> str:
    """Return a filesystem-safe label name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return cleaned.strip("_") or "label"


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out):
        return default
    return out


def _annotation_root(data_dir: str | Path) -> Path:
    root = Path(data_dir)
    if root.name == "misc_annotations":
        return root
    if (root / "misc_annotations").exists():
        return root / "misc_annotations"
    return root


def iter_misc_annotation_files(data_dir: str | Path) -> list[Path]:
    """Find MISC annotation JSONL files under a dataset root."""
    root = _annotation_root(data_dir)
    files = list(root.glob("*.jsonl")) + list(root.glob("*/*.jsonl"))
    unique = sorted({path.resolve(): path for path in files}.values())
    if not unique:
        raise FileNotFoundError(
            f"No MISC annotation JSONL files found under {root}. "
            "Expected misc_annotations/high/*.jsonl and misc_annotations/low/*.jsonl."
        )
    return unique


def record_label_set(record: dict[str, Any]) -> set[str]:
    """Build hierarchical labels for one MISC annotation record."""
    raw_code = str(record.get("predicted_code", "") or "").strip().upper()
    raw_subcode = str(record.get("predicted_subcode", "") or "").strip().upper()

    invalid = {"", "NONE", "NULL", "N/A", "NA", "NAN"}
    labels: set[str] = set()

    if raw_code in invalid:
        labels.add("OTHER")
    elif raw_code == "RE":
        labels.add("RE")
    elif raw_code == "QU":
        labels.add("QU")
    else:
        labels.add(raw_code)

    if raw_subcode not in invalid and raw_subcode not in labels:
        labels.add(raw_subcode)

    return labels or {"OTHER"}


def load_misc_annotation_records(
    data_dir: str | Path,
    *,
    confidence_threshold: float | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load full MISC annotation records from high/low JSONL files."""
    records: list[dict[str, Any]] = []
    for path in iter_misc_annotation_files(data_dir):
        split = path.parent.name if path.parent != _annotation_root(data_dir) else ""
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = str(
                    rec.get("unit_text")
                    or rec.get("utterance")
                    or rec.get("formatted_text")
                    or ""
                ).strip()
                if not text:
                    continue

                confidence = _safe_float(rec.get("confidence"), default=None)
                if (
                    confidence_threshold is not None
                    and (confidence is None or confidence < confidence_threshold)
                ):
                    continue

                enriched = dict(rec)
                enriched["unit_text"] = text
                enriched["confidence"] = confidence
                enriched["source_split"] = split
                enriched["source_file"] = path.name
                enriched["source_path"] = str(path)
                enriched["source_line"] = line_no
                enriched["labels"] = sorted(record_label_set(enriched))
                enriched.setdefault("record_id", f"{path.stem}:{line_no}")
                records.append(enriched)

                if limit is not None and len(records) >= limit:
                    return records

    return records


def select_labels(
    records: Iterable[dict[str, Any]],
    *,
    labels: list[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Return ordered labels and a boolean [N, L] indicator matrix."""
    label_sets = [set(record.get("labels") or record_label_set(record)) for record in records]
    if labels:
        ordered = [label.strip().upper() for label in labels if label.strip()]
    else:
        all_labels = set().union(*label_sets) if label_sets else set()
        ordered = [label for label in PREFERRED_LABEL_ORDER if label in all_labels]
        ordered.extend(sorted(all_labels.difference(ordered)))

    indicators = np.zeros((len(label_sets), len(ordered)), dtype=bool)
    for i, label_set in enumerate(label_sets):
        for j, label in enumerate(ordered):
            indicators[i, j] = label in label_set

    return ordered, indicators


def load_feature_matrix(path: str | Path) -> np.ndarray:
    """Load an utterance-level feature matrix from .pt, .npy, or .npz."""
    path = Path(path)
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, torch.Tensor):
            features = payload
        elif isinstance(payload, dict):
            for key in ("utterance_features", "features", "X"):
                if key in payload:
                    features = payload[key]
                    break
            else:
                raise KeyError(
                    f"{path} does not contain one of: utterance_features, features, X"
                )
        else:
            raise TypeError(f"Unsupported .pt payload type: {type(payload).__name__}")
        if isinstance(features, torch.Tensor):
            return features.detach().cpu().float().numpy()
        return np.asarray(features, dtype=np.float32)

    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)

    if path.suffix == ".npz":
        payload = np.load(path)
        for key in ("utterance_features", "features", "X"):
            if key in payload:
                return np.asarray(payload[key], dtype=np.float32)
        raise KeyError(f"{path} does not contain one of: utterance_features, features, X")

    raise ValueError(f"Unsupported feature file extension: {path.suffix}")


def _as_numpy_features(features: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().float().numpy()
    return np.ascontiguousarray(features, dtype=np.float32)


def _chunked_auc_by_rank(
    features: np.ndarray,
    positive_mask: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    n_samples, d_sae = features.shape
    n_pos = int(positive_mask.sum())
    n_neg = n_samples - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.full(d_sae, 0.5, dtype=np.float32)

    auc = np.empty(d_sae, dtype=np.float32)
    baseline = n_pos * (n_pos + 1) / 2.0
    denom = float(n_pos * n_neg)

    for start in range(0, d_sae, chunk_size):
        end = min(start + chunk_size, d_sae)
        ranks = stats.rankdata(features[:, start:end], axis=0, method="average")
        pos_rank_sum = ranks[positive_mask, :].sum(axis=0)
        auc[start:end] = ((pos_rank_sum - baseline) / denom).astype(np.float32)

    return np.nan_to_num(auc, nan=0.5, posinf=0.5, neginf=0.5)


def _chunked_precision_at_k(
    features: np.ndarray,
    positive_mask: np.ndarray,
    *,
    k: int,
    chunk_size: int,
) -> np.ndarray:
    n_samples, d_sae = features.shape
    if n_samples == 0:
        return np.zeros(d_sae, dtype=np.float32)
    k = min(max(int(k), 1), n_samples)
    if k == n_samples:
        return np.full(d_sae, float(positive_mask.mean()), dtype=np.float32)

    precision = np.empty(d_sae, dtype=np.float32)
    kth = n_samples - k
    for start in range(0, d_sae, chunk_size):
        end = min(start + chunk_size, d_sae)
        chunk = features[:, start:end]
        top_idx = np.argpartition(chunk, kth=kth, axis=0)[kth:, :]
        precision[start:end] = positive_mask[top_idx].mean(axis=0).astype(np.float32)
    return precision


def compute_latent_label_associations(
    features: torch.Tensor | np.ndarray,
    label_indicators: np.ndarray,
    labels: list[str],
    *,
    fdr_alpha: float = 0.05,
    precision_k_values: list[int] | None = None,
    min_positive: int = 10,
    min_negative: int = 10,
    chunk_size: int = 512,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Compute Latent x Label association rows in long format."""
    features_np = _as_numpy_features(features)
    if label_indicators.shape[0] != features_np.shape[0]:
        raise ValueError(
            "Feature row count must match label indicator row count: "
            f"{features_np.shape[0]} != {label_indicators.shape[0]}"
        )
    if label_indicators.shape[1] != len(labels):
        raise ValueError("label_indicators column count must match labels length.")

    precision_k_values = precision_k_values or [10, 50]
    rows: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    d_sae = features_np.shape[1]

    for label_idx, label in enumerate(tqdm(labels, desc="Labels", unit="label")):
        positive_mask = np.asarray(label_indicators[:, label_idx], dtype=bool)
        n_positive = int(positive_mask.sum())
        n_negative = int(len(positive_mask) - n_positive)
        prevalence = float(n_positive / len(positive_mask)) if len(positive_mask) else 0.0

        if n_positive < min_positive or n_negative < min_negative:
            skipped.append(
                {
                    "label": label,
                    "n_positive": n_positive,
                    "n_negative": n_negative,
                    "reason": "below min_positive/min_negative",
                }
            )
            continue

        pos = features_np[positive_mask]
        neg = features_np[~positive_mask]
        pos_mean = pos.mean(axis=0)
        neg_mean = neg.mean(axis=0)
        mean_diff = pos_mean - neg_mean

        pos_var = pos.var(axis=0, ddof=1) if n_positive > 1 else np.zeros(d_sae)
        neg_var = neg.var(axis=0, ddof=1) if n_negative > 1 else np.zeros(d_sae)
        denom = max(n_positive + n_negative - 2, 1)
        pooled_var = ((n_positive - 1) * pos_var + (n_negative - 1) * neg_var) / denom
        pooled_std = np.sqrt(np.maximum(pooled_var, 1e-24))
        cohens_d = mean_diff / pooled_std
        cohens_d = np.nan_to_num(cohens_d, nan=0.0, posinf=0.0, neginf=0.0)

        if pos.shape[0] > 1 and neg.shape[0] > 1:
            _, p_values = stats.ttest_ind(pos, neg, axis=0, equal_var=False)
            p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0)
        else:
            p_values = np.ones(d_sae, dtype=np.float32)

        auc = _chunked_auc_by_rank(
            features_np,
            positive_mask,
            chunk_size=max(1, int(chunk_size)),
        )

        payload: dict[str, Any] = {
            "label": label,
            "latent_idx": np.arange(d_sae, dtype=np.int32),
            "n_positive": n_positive,
            "n_negative": n_negative,
            "prevalence": prevalence,
            "pos_mean": pos_mean,
            "neg_mean": neg_mean,
            "mean_diff": mean_diff,
            "cohens_d": cohens_d,
            "abs_cohens_d": np.abs(cohens_d),
            "auc": auc,
            "directional_auc": np.where(cohens_d >= 0, auc, 1.0 - auc),
            "auc_effect": np.abs(auc - 0.5) * 2.0,
            "p_value": p_values,
            "significant_fdr": benjamini_hochberg(p_values, alpha=fdr_alpha),
        }

        for k in precision_k_values:
            precision = _chunked_precision_at_k(
                features_np,
                positive_mask,
                k=k,
                chunk_size=max(1, int(chunk_size)),
            )
            payload[f"precision_at_{k}"] = precision
            payload[f"precision_lift_at_{k}"] = precision - prevalence

        label_df = pd.DataFrame(payload)
        rows.append(label_df)

    if rows:
        matrix = pd.concat(rows, ignore_index=True)
        matrix = matrix.sort_values(
            ["label", "abs_cohens_d", "auc_effect"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
    else:
        matrix = pd.DataFrame()

    return matrix, skipped


def build_label_summary(
    labels: list[str],
    label_indicators: np.ndarray,
    *,
    skipped_labels: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = {
        "n_records": int(label_indicators.shape[0]),
        "n_labels": int(len(labels)),
        "labels": [],
        "skipped_labels": skipped_labels or [],
    }
    for idx, label in enumerate(labels):
        n_positive = int(label_indicators[:, idx].sum())
        summary["labels"].append(
            {
                "label": label,
                "n_positive": n_positive,
                "n_negative": int(label_indicators.shape[0] - n_positive),
                "prevalence": (
                    float(n_positive / label_indicators.shape[0])
                    if label_indicators.shape[0]
                    else 0.0
                ),
            }
        )
    return summary


def build_label_fragmentation(
    association_df: pd.DataFrame,
    *,
    top_k_per_label: int = 50,
) -> dict[str, Any]:
    labels: list[dict[str, Any]] = []
    if association_df.empty:
        return {"labels": labels}

    for label, group in association_df.groupby("label", sort=False):
        sig = group[group["significant_fdr"].astype(bool)]
        top = group.sort_values("abs_cohens_d", ascending=False).head(top_k_per_label)
        labels.append(
            {
                "label": label,
                "n_tested_latents": int(group.shape[0]),
                "n_significant_latents": int(sig.shape[0]),
                "n_positive_effect_significant": int((sig["cohens_d"] > 0).sum()),
                "n_negative_effect_significant": int((sig["cohens_d"] < 0).sum()),
                "top_abs_cohens_d": float(top["abs_cohens_d"].iloc[0]) if not top.empty else 0.0,
                "top_directional_auc": float(top["directional_auc"].iloc[0]) if not top.empty else 0.5,
                "top_latents": top[
                    [
                        "latent_idx",
                        "cohens_d",
                        "auc",
                        "directional_auc",
                        "p_value",
                        "significant_fdr",
                    ]
                ].to_dict("records"),
            }
        )

    return {"labels": labels}


def build_latent_overlap(association_df: pd.DataFrame) -> dict[str, Any]:
    if association_df.empty:
        return {
            "n_significant_latent_label_edges": 0,
            "n_latents_with_any_significant_label": 0,
            "single_label_latents": 0,
            "multi_label_latents": 0,
            "label_count_distribution": {},
            "top_overlapping_latents": [],
        }

    sig = association_df[association_df["significant_fdr"].astype(bool)].copy()
    if sig.empty:
        return {
            "n_significant_latent_label_edges": 0,
            "n_latents_with_any_significant_label": 0,
            "single_label_latents": 0,
            "multi_label_latents": 0,
            "label_count_distribution": {},
            "top_overlapping_latents": [],
        }

    grouped = []
    for latent_idx, group in sig.groupby("latent_idx"):
        group_sorted = group.sort_values("abs_cohens_d", ascending=False)
        grouped.append(
            {
                "latent_idx": int(latent_idx),
                "n_labels": int(group.shape[0]),
                "labels": group_sorted["label"].tolist(),
                "max_abs_cohens_d": float(group_sorted["abs_cohens_d"].iloc[0]),
                "max_directional_auc": float(group_sorted["directional_auc"].max()),
            }
        )

    grouped.sort(key=lambda row: (row["n_labels"], row["max_abs_cohens_d"]), reverse=True)
    counts = pd.Series([row["n_labels"] for row in grouped]).value_counts().sort_index()
    return {
        "n_significant_latent_label_edges": int(sig.shape[0]),
        "n_latents_with_any_significant_label": int(len(grouped)),
        "single_label_latents": int(sum(row["n_labels"] == 1 for row in grouped)),
        "multi_label_latents": int(sum(row["n_labels"] > 1 for row in grouped)),
        "label_count_distribution": {str(int(k)): int(v) for k, v in counts.items()},
        "top_overlapping_latents": grouped[:50],
    }


def build_topk_jaccard(
    association_df: pd.DataFrame,
    *,
    top_k: int = 50,
) -> list[dict[str, Any]]:
    if association_df.empty:
        return []

    label_sets: dict[str, set[int]] = {}
    for label, group in association_df.groupby("label", sort=False):
        top = group.sort_values("abs_cohens_d", ascending=False).head(top_k)
        label_sets[label] = set(top["latent_idx"].astype(int).tolist())

    rows: list[dict[str, Any]] = []
    labels = list(label_sets)
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            union = label_sets[left] | label_sets[right]
            inter = label_sets[left] & label_sets[right]
            rows.append(
                {
                    "label_a": left,
                    "label_b": right,
                    "top_k": int(top_k),
                    "intersection": int(len(inter)),
                    "union": int(len(union)),
                    "jaccard": float(len(inter) / len(union)) if union else 0.0,
                }
            )
    rows.sort(key=lambda row: row["jaccard"], reverse=True)
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def write_label_indicator_csv(
    path: str | Path,
    records: list[dict[str, Any]],
    labels: list[str],
    label_indicators: np.ndarray,
) -> None:
    rows = []
    for idx, record in enumerate(records):
        row = {
            "row_idx": idx,
            "record_id": record.get("record_id"),
            "file_id": record.get("file_id"),
            "source_split": record.get("source_split"),
            "source_file": record.get("source_file"),
            "predicted_code": record.get("predicted_code"),
            "predicted_subcode": record.get("predicted_subcode"),
            "confidence": record.get("confidence"),
            "unit_text": record.get("unit_text"),
        }
        for j, label in enumerate(labels):
            row[label] = int(label_indicators[idx, j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_top_latents_by_label(
    association_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    top_k_per_label: int,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if association_df.empty:
        return
    for label, group in association_df.groupby("label", sort=False):
        safe = sanitize_label(label)
        top = group.sort_values("abs_cohens_d", ascending=False).head(top_k_per_label)
        top.to_csv(out / f"{safe}.csv", index=False)


def write_top_example_cards(
    association_df: pd.DataFrame,
    features: torch.Tensor | np.ndarray,
    records: list[dict[str, Any]],
    label_indicators: np.ndarray,
    labels: list[str],
    output_dir: str | Path,
    *,
    top_latents_per_label: int = 5,
    top_examples_per_latent: int = 10,
) -> None:
    if association_df.empty or top_latents_per_label <= 0 or top_examples_per_latent <= 0:
        return

    features_np = _as_numpy_features(features)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    for label, group in association_df.groupby("label", sort=False):
        safe = sanitize_label(label)
        label_dir = out / safe
        label_dir.mkdir(parents=True, exist_ok=True)
        label_idx = label_to_idx[label]
        positives = label_indicators[:, label_idx]
        top_latents = group.sort_values("abs_cohens_d", ascending=False).head(top_latents_per_label)

        for _, row in top_latents.iterrows():
            latent_idx = int(row["latent_idx"])
            acts = features_np[:, latent_idx]
            top_indices = np.argsort(acts)[::-1][:top_examples_per_latent]

            lines = [
                f"# {label} / Latent {latent_idx}",
                "",
                f"- Cohen's d: {float(row['cohens_d']):.4f}",
                f"- AUC: {float(row['auc']):.4f}",
                f"- Directional AUC: {float(row['directional_auc']):.4f}",
                f"- FDR significant: {bool(row['significant_fdr'])}",
                "",
                "| Rank | Act. | Match | Labels | Text |",
                "|---:|---:|---|---|---|",
            ]
            for rank, rec_idx in enumerate(top_indices, start=1):
                rec = records[int(rec_idx)]
                text = str(rec.get("unit_text", "")).replace("|", "\\|")
                rec_labels = ",".join(rec.get("labels", []))
                lines.append(
                    f"| {rank} | {float(acts[rec_idx]):.4f} | "
                    f"{'yes' if positives[rec_idx] else 'no'} | {rec_labels} | {text} |"
                )
            (label_dir / f"latent_{latent_idx:05d}.md").write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )


def write_behavior_asymmetry_report(
    path: str | Path,
    label_summary: dict[str, Any],
    fragmentation: dict[str, Any],
    overlap: dict[str, Any],
    jaccard_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# MISC Label Mapping Structure Report",
        "",
        "## Label Distribution",
        "",
        "| Label | Positive | Negative | Prevalence |",
        "|---|---:|---:|---:|",
    ]
    for row in label_summary.get("labels", []):
        lines.append(
            f"| {row['label']} | {row['n_positive']} | {row['n_negative']} | "
            f"{row['prevalence']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Fragmentation By Label",
            "",
            "| Label | Significant Latents | Positive Effect | Negative Effect | Top | Top Directional AUC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fragmentation.get("labels", []):
        lines.append(
            f"| {row['label']} | {row['n_significant_latents']} | "
            f"{row['n_positive_effect_significant']} | "
            f"{row['n_negative_effect_significant']} | "
            f"{row['top_abs_cohens_d']:.3f} | {row['top_directional_auc']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Latent Overlap",
            "",
            f"- Significant latent-label edges: {overlap['n_significant_latent_label_edges']}",
            f"- Latents with any significant label: {overlap['n_latents_with_any_significant_label']}",
            f"- Single-label latents: {overlap['single_label_latents']}",
            f"- Multi-label latents: {overlap['multi_label_latents']}",
            "",
            "Top overlapping latents:",
            "",
            "| Latent | Label Count | Labels | Max |",
            "|---:|---:|---|---:|",
        ]
    )
    for row in overlap.get("top_overlapping_latents", [])[:20]:
        lines.append(
            f"| {row['latent_idx']} | {row['n_labels']} | "
            f"{', '.join(row['labels'])} | {row['max_abs_cohens_d']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Top-k Label Similarity",
            "",
            "| Label A | Label B | Intersection | Union | Jaccard |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in jaccard_rows[:30]:
        lines.append(
            f"| {row['label_a']} | {row['label_b']} | {row['intersection']} | "
            f"{row['union']} | {row['jaccard']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- More significant latents for a label indicate stronger fragmentation.",
            "- More multi-label latents indicate stronger representational overlap.",
            "- High top-k Jaccard means two behavior labels rely on similar latent sets.",
            "- These results should be read as structure in representation space, not as a new counseling taxonomy.",
        ]
    )

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_misc_label_mapping(
    *,
    records: list[dict[str, Any]],
    features: torch.Tensor | np.ndarray,
    output_dir: str | Path,
    labels: list[str] | None = None,
    fdr_alpha: float = 0.05,
    precision_k_values: list[int] | None = None,
    min_positive: int = 10,
    min_negative: int = 10,
    chunk_size: int = 512,
    top_k_per_label: int = 50,
    top_example_latents: int = 5,
    top_examples_per_latent: int = 10,
) -> dict[str, Any]:
    """Run the full matrix analysis and persist all report files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    features_np = _as_numpy_features(features)

    if features_np.shape[0] != len(records):
        raise ValueError(
            f"features rows ({features_np.shape[0]}) must match records ({len(records)})."
        )

    selected_labels, label_indicators = select_labels(records, labels=labels)
    matrix, skipped = compute_latent_label_associations(
        features_np,
        label_indicators,
        selected_labels,
        fdr_alpha=fdr_alpha,
        precision_k_values=precision_k_values,
        min_positive=min_positive,
        min_negative=min_negative,
        chunk_size=chunk_size,
    )

    matrix_path = output_path / "latent_label_matrix.csv"
    matrix.to_csv(matrix_path, index=False)

    label_summary = build_label_summary(
        selected_labels,
        label_indicators,
        skipped_labels=skipped,
    )
    fragmentation = build_label_fragmentation(
        matrix,
        top_k_per_label=top_k_per_label,
    )
    overlap = build_latent_overlap(matrix)
    jaccard_rows = build_topk_jaccard(matrix, top_k=top_k_per_label)

    write_json(output_path / "label_summary.json", label_summary)
    write_json(output_path / "label_fragmentation.json", fragmentation)
    write_json(output_path / "latent_overlap.json", overlap)
    write_json(output_path / "label_topk_jaccard.json", jaccard_rows)
    write_jsonl(output_path / "annotation_records.jsonl", records)
    write_label_indicator_csv(
        output_path / "label_indicator_matrix.csv",
        records,
        selected_labels,
        label_indicators,
    )
    write_top_latents_by_label(
        matrix,
        output_path / "top_latents_by_label",
        top_k_per_label=top_k_per_label,
    )
    write_top_example_cards(
        matrix,
        features_np,
        records,
        label_indicators,
        selected_labels,
        output_path / "top_examples_by_label",
        top_latents_per_label=top_example_latents,
        top_examples_per_latent=top_examples_per_latent,
    )
    write_behavior_asymmetry_report(
        output_path / "behavior_asymmetry.md",
        label_summary,
        fragmentation,
        overlap,
        jaccard_rows,
    )

    run_summary = {
        "output_dir": str(output_path),
        "n_records": len(records),
        "feature_shape": list(features_np.shape),
        "labels": selected_labels,
        "fdr_alpha": fdr_alpha,
        "min_positive": min_positive,
        "min_negative": min_negative,
        "precision_k_values": precision_k_values or [10, 50],
        "files": {
            "latent_label_matrix": str(matrix_path),
            "label_summary": str(output_path / "label_summary.json"),
            "label_fragmentation": str(output_path / "label_fragmentation.json"),
            "latent_overlap": str(output_path / "latent_overlap.json"),
            "behavior_asymmetry": str(output_path / "behavior_asymmetry.md"),
        },
    }
    write_json(output_path / "run_summary.json", run_summary)
    return run_summary
