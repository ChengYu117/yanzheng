"""Shared utilities for grouped cross-validation of MISC SAE associations."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .misc_label_mapping import (
    _chunked_auc_by_rank,
    compute_latent_label_associations,
    load_feature_matrix,
)


DEFAULT_LABELS = ("RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF")


@dataclass(frozen=True)
class CrossValInputs:
    """Aligned rows, labels, source groups, and selected SAE features."""

    label_matrix: pd.DataFrame
    features: np.ndarray
    latent_indices: np.ndarray
    labels: tuple[str, ...]


def normalize_text(text: Any) -> str:
    """Normalize utterance text for exact duplicate grouping."""

    value = "" if text is None else str(text)
    value = value.strip().lower()
    return re.sub(r"\s+", " ", value)


def build_dedup_group_mapping(label_matrix: pd.DataFrame) -> pd.DataFrame:
    """Return row-aligned source and duplicate-group metadata."""

    required = {"unit_text", "source_file", "source_split"}
    missing = required.difference(label_matrix.columns)
    if missing:
        raise ValueError(f"label matrix is missing required columns: {sorted(missing)}")

    out = pd.DataFrame()
    out["row_idx"] = (
        label_matrix["row_idx"].astype(int)
        if "row_idx" in label_matrix.columns
        else np.arange(len(label_matrix), dtype=np.int32)
    )
    out["record_id"] = label_matrix.get("record_id", pd.Series([""] * len(label_matrix))).astype(str)
    out["source_split"] = label_matrix["source_split"].astype(str)
    out["source_file"] = label_matrix["source_file"].astype(str)
    out["unit_text"] = label_matrix["unit_text"].astype(str)
    out["text_normalized"] = out["unit_text"].map(normalize_text)
    out["dedup_group_id"] = out.groupby("text_normalized", sort=True).ngroup().astype(int)
    out["source_dedup_group_id"] = (
        out["source_file"].astype(str) + "::" + out["dedup_group_id"].astype(str)
    )
    out["duplicate_text_count"] = out.groupby("dedup_group_id")["dedup_group_id"].transform("size")
    return out


def save_dedup_group_mapping(label_matrix_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Build and save the E0 duplicate-group mapping."""

    label_matrix = pd.read_csv(label_matrix_path)
    mapping = build_dedup_group_mapping(label_matrix)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(output_path, index=False, encoding="utf-8-sig")
    return mapping


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_filtered_inputs(
    *,
    feature_store_path: str | Path,
    label_matrix_path: str | Path,
    feature_filter_audit_path: str | Path,
    labels: Iterable[str] = DEFAULT_LABELS,
) -> CrossValInputs:
    """Load full features and retain only keep=True latents from the filtered pool."""

    label_matrix = pd.read_csv(label_matrix_path)
    features = load_feature_matrix(feature_store_path)
    if len(label_matrix) != features.shape[0]:
        raise ValueError(
            f"label matrix rows ({len(label_matrix)}) must match feature rows ({features.shape[0]})"
        )

    audit = pd.read_csv(feature_filter_audit_path)
    if not {"latent_idx", "keep"}.issubset(audit.columns):
        raise ValueError("feature_filter_audit must contain latent_idx and keep columns")
    keep_mask = audit["keep"].astype(bool)
    latent_indices = audit.loc[keep_mask, "latent_idx"].astype(int).to_numpy()
    if latent_indices.size == 0:
        raise ValueError("filtered latent pool is empty")

    features = np.asarray(features[:, latent_indices], dtype=np.float32)
    selected_labels = tuple(label.upper() for label in labels)
    missing = [label for label in selected_labels if label not in label_matrix.columns]
    if missing:
        raise ValueError(f"label matrix is missing requested labels: {missing}")
    return CrossValInputs(
        label_matrix=label_matrix,
        features=features,
        latent_indices=latent_indices.astype(np.int32),
        labels=selected_labels,
    )


def load_full_inputs(
    *,
    feature_store_path: str | Path,
    label_matrix_path: str | Path,
    labels: Iterable[str] = DEFAULT_LABELS,
) -> CrossValInputs:
    """Load row-aligned labels and all SAE latents without feature prefiltering."""

    label_matrix = pd.read_csv(label_matrix_path)
    features = load_feature_matrix(feature_store_path)
    if len(label_matrix) != features.shape[0]:
        raise ValueError(
            f"label matrix rows ({len(label_matrix)}) must match feature rows ({features.shape[0]})"
        )

    selected_labels = tuple(label.upper() for label in labels)
    missing = [label for label in selected_labels if label not in label_matrix.columns]
    if missing:
        raise ValueError(f"label matrix is missing requested labels: {missing}")

    return CrossValInputs(
        label_matrix=label_matrix,
        features=np.asarray(features, dtype=np.float32),
        latent_indices=np.arange(features.shape[1], dtype=np.int32),
        labels=selected_labels,
    )


def label_indicator_matrix(label_matrix: pd.DataFrame, labels: Iterable[str]) -> np.ndarray:
    """Return boolean [N, L] matrix for requested labels."""

    labels = tuple(labels)
    return label_matrix.loc[:, labels].astype(int).to_numpy(dtype=bool)


def compute_subset_associations(
    inputs: CrossValInputs,
    row_indices: np.ndarray | list[int],
    *,
    min_positive: int = 10,
    min_negative: int = 10,
    chunk_size: int = 512,
    precision_k_values: list[int] | None = None,
    compute_p_values: bool = True,
    compute_auc: bool = True,
) -> pd.DataFrame:
    """Compute label-latent association metrics for a row subset."""

    row_indices = np.asarray(row_indices, dtype=np.int64)
    y = label_indicator_matrix(inputs.label_matrix.iloc[row_indices], inputs.labels)
    matrix, skipped = compute_latent_label_associations(
        inputs.features[row_indices, :],
        y,
        list(inputs.labels),
        min_positive=min_positive,
        min_negative=min_negative,
        chunk_size=chunk_size,
        precision_k_values=precision_k_values,
        compute_p_values=compute_p_values,
        compute_auc=compute_auc,
        candidate_latent_indices=inputs.latent_indices,
    )
    if skipped:
        matrix = matrix.copy()
        matrix.attrs["skipped_labels"] = skipped
    return matrix


def split_rows_by_group(
    groups: Iterable[Any],
    *,
    n_splits: int = 2,
    random_state: int = 42,
) -> list[np.ndarray]:
    """Deterministically split row indices by shuffled unique group IDs."""

    groups_arr = np.asarray(list(groups), dtype=object)
    unique_groups = np.array(sorted(pd.Series(groups_arr).astype(str).unique()), dtype=object)
    rng = np.random.default_rng(random_state)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    group_splits = np.array_split(shuffled, n_splits)
    row_splits: list[np.ndarray] = []
    for group_split in group_splits:
        mask = np.isin(groups_arr.astype(str), group_split.astype(str))
        row_splits.append(np.flatnonzero(mask))
    return row_splits


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    left = set(int(x) for x in a)
    right = set(int(x) for x in b)
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right) / len(union))


def null_jaccard_distribution(
    *,
    pool_size: int,
    top_k: int,
    n_iter: int,
    random_state: int,
) -> np.ndarray:
    """Random TopK Jaccard null distribution for a fixed candidate-pool size."""

    rng = np.random.default_rng(random_state)
    top_k = min(int(top_k), int(pool_size))
    out = np.empty(int(n_iter), dtype=np.float32)
    for i in range(int(n_iter)):
        a = rng.choice(pool_size, size=top_k, replace=False)
        b = rng.choice(pool_size, size=top_k, replace=False)
        out[i] = jaccard(a, b)
    return out


def fisher_ci_for_spearman(rho: float, n: int, *, alpha: float = 0.05) -> tuple[float, float]:
    """Approximate Spearman confidence interval via Fisher z transform."""

    if n <= 3 or not np.isfinite(rho) or abs(rho) >= 1:
        return (float("nan"), float("nan"))
    z = np.arctanh(float(rho))
    se = 1.0 / math.sqrt(n - 3)
    zcrit = 1.959963984540054
    return (float(np.tanh(z - zcrit * se)), float(np.tanh(z + zcrit * se)))


def top_latents_for_label(
    matrix: pd.DataFrame,
    label: str,
    *,
    metric: str,
    top_k: int,
    positive_only: bool = False,
) -> pd.DataFrame:
    """Return top latent rows for one label and ranking metric."""

    sub = matrix[matrix["label"].astype(str) == str(label)].copy()
    if positive_only and "cohens_d" in sub.columns:
        sub = sub[pd.to_numeric(sub["cohens_d"], errors="coerce").fillna(0.0) > 0]
    if metric not in sub.columns:
        raise ValueError(f"metric {metric!r} not found in association matrix")
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    return sub.sort_values([metric, "latent_idx"], ascending=[False, True]).head(top_k)


def compute_auc_and_cohens_d(
    features: np.ndarray,
    positive_mask: np.ndarray,
    *,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute raw AUC and signed Cohen's d for features against one label."""

    features = np.asarray(features, dtype=np.float32)
    positive_mask = np.asarray(positive_mask, dtype=bool)
    n_positive = int(positive_mask.sum())
    n_negative = int(len(positive_mask) - n_positive)
    if n_positive == 0 or n_negative == 0:
        d = np.zeros(features.shape[1], dtype=np.float32)
        auc = np.full(features.shape[1], 0.5, dtype=np.float32)
        return auc, d

    pos = features[positive_mask]
    neg = features[~positive_mask]
    mean_diff = pos.mean(axis=0) - neg.mean(axis=0)
    pos_var = pos.var(axis=0, ddof=1) if n_positive > 1 else np.zeros(features.shape[1])
    neg_var = neg.var(axis=0, ddof=1) if n_negative > 1 else np.zeros(features.shape[1])
    pooled_var = ((n_positive - 1) * pos_var + (n_negative - 1) * neg_var) / max(
        n_positive + n_negative - 2, 1
    )
    d = mean_diff / np.sqrt(np.maximum(pooled_var, 1e-24))
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    auc = _chunked_auc_by_rank(features, positive_mask, chunk_size=chunk_size)
    return auc.astype(np.float32), d


def grouped_bootstrap_row_indices(
    group_to_indices: dict[str, np.ndarray],
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample groups with replacement and concatenate their row indices."""

    group_names = np.array(sorted(group_to_indices), dtype=object)
    sampled = rng.choice(group_names, size=len(group_names), replace=True)
    return np.concatenate([group_to_indices[str(group)] for group in sampled]).astype(np.int64)
