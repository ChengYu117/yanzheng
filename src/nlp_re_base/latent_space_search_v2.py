"""V2 MISC SAE latent-space search analysis.

Top-20 latents are kept as a human-auditable candidate budget.  Formal
structural conclusions are based on thresholded sets, weighted metrics, and
K-sensitivity checks.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_LABELS: tuple[str, ...] = (
    "RE",
    "RES",
    "REC",
    "QU",
    "QUO",
    "QUC",
    "GI",
    "SU",
    "AF",
)
PARENT_CHILDREN: dict[str, list[str]] = {"RE": ["RES", "REC"], "QU": ["QUO", "QUC"]}
PARENT_LABELS: tuple[str, ...] = ("RE", "QU")
LEAF_LABELS: tuple[str, ...] = ("RES", "REC", "QUO", "QUC", "GI", "SU", "AF")
FAMILY_MEMBERS: dict[str, list[str]] = {
    "RE_family": ["RES", "REC"],
    "QU_family": ["QUO", "QUC"],
    "GI": ["GI"],
    "SU": ["SU"],
    "AF": ["AF"],
}
SUPPLEMENTAL_BLOCKS: dict[str, list[str]] = {
    "advice_support_info_block": ["GI", "SU", "AF"],
}
KEY_SENSITIVITY_PAIRS: tuple[tuple[str, str], ...] = (
    ("RE", "REC"),
    ("QU", "QUO"),
    ("RES", "REC"),
    ("QUO", "QUC"),
    ("AF", "SU"),
)


@dataclass(frozen=True)
class LatentSpaceSearchV2Config:
    labels: tuple[str, ...] = DEFAULT_LABELS
    top_candidate_k: int = 20
    k_values: tuple[int, ...] = (5, 10, 20, 50, 100)
    min_directional_auc: float = 0.70
    min_abs_cohens_d: float = 0.50
    precision_delta: float = 0.10
    precision_ks: tuple[int, ...] = (50, 100)
    chunk_size: int = 512
    semantic_top_examples: int = 12
    random_baseline_repeats: int = 100
    random_state: int = 13


def load_feature_store(path: str | Path | None) -> np.ndarray | None:
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    if path.suffix == ".npz":
        payload = np.load(path)
        for key in ("utterance_features", "features", "X", "arr_0"):
            if key in payload:
                return np.asarray(payload[key], dtype=np.float32)
        raise KeyError(f"No feature matrix key found in {path}")
    if path.suffix == ".pt":
        import torch

        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            for key in ("utterance_features", "features", "feature_matrix", "X"):
                if key in payload:
                    payload = payload[key]
                    break
        if hasattr(payload, "detach"):
            payload = payload.detach().cpu().float().numpy()
        return np.asarray(payload, dtype=np.float32)
    raise ValueError(f"Unsupported feature-store extension: {path.suffix}")


def load_records(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _family_for_label(label: str) -> str:
    label = label.upper()
    if label in {"RE", "RES", "REC"}:
        return "RE_family"
    if label in {"QU", "QUO", "QUC"}:
        return "QU_family"
    if label in {"AF", "SU"}:
        return "support_affirm"
    return label


def _leaf_family(label: str) -> str:
    label = label.upper()
    for family, members in FAMILY_MEMBERS.items():
        if label in members:
            return family
    return label


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 0.0


def _generalized_latents_from_matrix(matrix: pd.DataFrame) -> set[int]:
    stable = matrix[matrix["stable_edge"] & matrix["label"].isin(LEAF_LABELS)]
    generalized: set[int] = set()
    for latent_idx, group in stable.groupby("latent_idx"):
        labels = set(group["label"].astype(str).tolist())
        families = {_leaf_family(label) for label in labels}
        if len(labels) >= 4 or len(families) >= 3:
            generalized.add(int(latent_idx))
    return generalized


def _overlap_row_values(
    left_set: set[int],
    right_set: set[int],
    generalized_latents: set[int],
) -> dict[str, Any]:
    shared = left_set & right_set
    shared_generalized = shared & generalized_latents
    left_specific = left_set - generalized_latents
    right_specific = right_set - generalized_latents
    shared_specific = left_specific & right_specific
    specific_union = left_specific | right_specific
    return {
        "intersection": len(shared),
        "union": len(left_set | right_set),
        "jaccard": _jaccard(left_set, right_set),
        "shared_latents": ",".join(str(x) for x in sorted(shared)),
        "generalized_shared_latents": ",".join(str(x) for x in sorted(shared_generalized)),
        "specific_intersection": len(shared_specific),
        "specific_union": len(specific_union),
        "specific_jaccard": _jaccard(left_specific, right_specific),
        "shared_specific_latents": ",".join(str(x) for x in sorted(shared_specific)),
    }


def _weighted_jaccard(left: dict[int, float], right: dict[int, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    numer = sum(min(float(left.get(k, 0.0)), float(right.get(k, 0.0))) for k in keys)
    denom = sum(max(float(left.get(k, 0.0)), float(right.get(k, 0.0))) for k in keys)
    return float(numer / denom) if denom > 0 else 0.0


def _effective_fragmentation(weights: Iterable[float]) -> float:
    values = np.asarray([float(v) for v in weights if float(v) > 0], dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float((values.sum() ** 2) / np.sum(values * values))


def _weighted_entropy(weights: Iterable[float]) -> float:
    values = np.asarray([float(v) for v in weights if float(v) > 0], dtype=np.float64)
    if values.size == 0:
        return 0.0
    probs = values / values.sum()
    return float(-np.sum(probs * np.log(probs)))


def _chunked_precision_at_k(features: np.ndarray, y: np.ndarray, k: int, chunk_size: int) -> np.ndarray:
    n_samples, n_features = features.shape
    if n_samples == 0:
        return np.zeros(n_features, dtype=np.float32)
    k = min(max(int(k), 1), n_samples)
    if k == n_samples:
        return np.full(n_features, float(y.mean()), dtype=np.float32)
    precision = np.empty(n_features, dtype=np.float32)
    kth = n_samples - k
    y_bool = np.asarray(y, dtype=bool)
    for start in range(0, n_features, max(1, int(chunk_size))):
        end = min(start + chunk_size, n_features)
        chunk = features[:, start:end]
        top_idx = np.argpartition(chunk, kth=kth, axis=0)[kth:, :]
        precision[start:end] = y_bool[top_idx].mean(axis=0).astype(np.float32)
    return precision


def _normalize_matrix(
    matrix: pd.DataFrame,
    *,
    label_df: pd.DataFrame | None,
    features: np.ndarray | None,
    config: LatentSpaceSearchV2Config,
) -> pd.DataFrame:
    labels = [label.upper() for label in config.labels]
    out = matrix.copy()
    out["label"] = out["label"].astype(str).str.upper()
    out = out[out["label"].isin(labels)].copy()
    required = {"label", "latent_idx", "cohens_d", "abs_cohens_d", "directional_auc", "significant_fdr"}
    missing = sorted(required.difference(out.columns))
    if missing:
        raise ValueError(f"latent-label matrix missing required columns: {missing}")

    numeric_cols = [
        "latent_idx",
        "cohens_d",
        "abs_cohens_d",
        "auc",
        "directional_auc",
        "p_value",
        "prevalence",
        "precision_at_10",
        "precision_at_50",
        "precision_at_100",
        "precision_lift_at_10",
        "precision_lift_at_50",
        "precision_lift_at_100",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["latent_idx"] = out["latent_idx"].astype(int)
    out["significant_fdr"] = _bool_series(out["significant_fdr"])

    if label_df is not None:
        for label in labels:
            if label in label_df.columns:
                prevalence = float(pd.to_numeric(label_df[label], errors="coerce").fillna(0).mean())
                mask = out["label"] == label
                if "prevalence" not in out.columns:
                    out.loc[mask, "prevalence"] = prevalence
                else:
                    out.loc[mask & (out["prevalence"] <= 0), "prevalence"] = prevalence

    if features is not None and label_df is not None:
        if len(label_df) != features.shape[0]:
            raise ValueError(
                f"label rows ({len(label_df)}) do not match feature rows ({features.shape[0]})"
            )
        for k in config.precision_ks:
            col = f"precision_at_{k}"
            needs_col = col not in out.columns or out[col].isna().any()
            if not needs_col:
                continue
            if any(label not in label_df.columns for label in labels):
                continue
            precision_by_label: dict[str, np.ndarray] = {}
            for label in labels:
                y = pd.to_numeric(label_df[label], errors="coerce").fillna(0).astype(int).to_numpy()
                precision_by_label[label] = _chunked_precision_at_k(
                    features,
                    y,
                    k=k,
                    chunk_size=config.chunk_size,
                )
            out[col] = 0.0
            for label, precision in precision_by_label.items():
                idx = out["label"] == label
                latent_ids = out.loc[idx, "latent_idx"].astype(int).to_numpy()
                valid = latent_ids < len(precision)
                values = np.zeros(len(latent_ids), dtype=np.float32)
                values[valid] = precision[latent_ids[valid]]
                out.loc[idx, col] = values

    for k in config.precision_ks:
        pcol = f"precision_at_{k}"
        if pcol not in out.columns:
            out[pcol] = np.nan
        out[f"precision_lift_absolute_at_{k}"] = out[pcol] - out["prevalence"]
        ratio = out[pcol] / out["prevalence"].replace(0, np.nan)
        out[f"precision_lift_ratio_at_{k}"] = ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        legacy = f"precision_lift_at_{k}"
        if legacy not in out.columns:
            out[legacy] = out[f"precision_lift_absolute_at_{k}"]

    out["association_score"] = np.maximum(out["directional_auc"].astype(float) - 0.5, 0.0)
    out["latent_label_weight"] = out["association_score"] * out["precision_lift_ratio_at_50"]
    out["positive_support"] = (
        out["significant_fdr"]
        & (out["cohens_d"] >= 0)
        & (out["abs_cohens_d"] >= config.min_abs_cohens_d)
        & (out["directional_auc"] >= config.min_directional_auc)
        & (out["precision_at_50"] >= out["prevalence"] + config.precision_delta)
    )
    out["negative_boundary"] = (
        out["significant_fdr"]
        & (out["cohens_d"] < 0)
        & (out["abs_cohens_d"] >= config.min_abs_cohens_d)
        & (out["directional_auc"] >= config.min_directional_auc)
    )
    out["stable_edge"] = out["positive_support"] | out["negative_boundary"]
    out["edge_type"] = np.select(
        [out["positive_support"], out["negative_boundary"]],
        ["positive_support", "negative_boundary"],
        default="weak_or_noise",
    )
    out["formal_edge_weight"] = np.where(
        out["negative_boundary"],
        out["association_score"],
        out["latent_label_weight"],
    )

    ranked_parts: list[pd.DataFrame] = []
    for label in labels:
        group = out[out["label"] == label].copy()
        if group.empty:
            continue
        group = group.sort_values(
            ["abs_cohens_d", "directional_auc", "latent_idx"],
            ascending=[False, False, True],
        ).copy()
        group["association_rank"] = np.arange(1, len(group) + 1)
        ranked_parts.append(group)
    return pd.concat(ranked_parts, ignore_index=True) if ranked_parts else out


def build_top20_candidates(matrix: pd.DataFrame, config: LatentSpaceSearchV2Config) -> pd.DataFrame:
    top = matrix[matrix["association_rank"] <= config.top_candidate_k].copy()
    top["candidate_scope"] = "human_auditable_candidate_budget"
    top["candidate_budget_k"] = config.top_candidate_k
    return top


def build_thresholded_sets(matrix: pd.DataFrame, config: LatentSpaceSearchV2Config) -> pd.DataFrame:
    stable = matrix[matrix["stable_edge"]].copy()
    stable["selection_status"] = "stable_latent"
    rows = [stable]
    for label in config.labels:
        label = label.upper()
        if not bool((stable["label"] == label).any()):
            source = matrix[matrix["label"] == label].head(1)
            placeholder: dict[str, Any] = {
                "label": label,
                "latent_idx": -1,
                "selection_status": "no_stable_latents",
                "edge_type": "no_stable_latents",
                "stable_edge": False,
                "positive_support": False,
                "negative_boundary": False,
            }
            if not source.empty:
                placeholder["prevalence"] = float(source["prevalence"].iloc[0])
            rows.append(pd.DataFrame([placeholder]))
    return pd.concat(rows, ignore_index=True, sort=False)


def build_weighted_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "label",
        "latent_idx",
        "cohens_d",
        "abs_cohens_d",
        "directional_auc",
        "prevalence",
        "precision_at_50",
        "precision_at_100",
        "precision_lift_absolute_at_50",
        "precision_lift_ratio_at_50",
        "association_score",
        "latent_label_weight",
        "formal_edge_weight",
        "edge_type",
        "stable_edge",
        "positive_support",
        "negative_boundary",
        "association_rank",
    ]
    existing = [col for col in cols if col in matrix.columns]
    return matrix[existing].copy()


def build_fragmentation(matrix: pd.DataFrame, config: LatentSpaceSearchV2Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_features = int(matrix["latent_idx"].max()) + 1 if not matrix.empty else 0
    for label in config.labels:
        label = label.upper()
        group = matrix[matrix["label"] == label]
        stable = group[group["stable_edge"]]
        weights = stable["formal_edge_weight"].astype(float).tolist()
        effective = _effective_fragmentation(weights)
        count = int(stable.shape[0])
        if count == 0:
            cls = "no_stable_latents"
        elif count >= 13 or effective > 8:
            cls = "distributed"
        elif count <= 5 or effective <= 4:
            cls = "compact"
        else:
            cls = "moderate"
        top20 = group[group["association_rank"] <= config.top_candidate_k]
        rows.append(
            {
                "label": label,
                "label_scope": "parent_consistency_only" if label in PARENT_LABELS else "formal_or_leaf",
                "n_features": n_features,
                "top20_candidate_count": int(top20.shape[0]),
                "thresholded_latent_count": count,
                "positive_support_count": int(stable["positive_support"].sum()) if not stable.empty else 0,
                "negative_boundary_count": int(stable["negative_boundary"].sum()) if not stable.empty else 0,
                "normalized_fragmentation": float(count / n_features) if n_features else 0.0,
                "effective_fragmentation": effective,
                "effective_fragmentation_fraction": float(effective / n_features) if n_features else 0.0,
                "top20_weak_count": int((~top20["stable_edge"]).sum()) if not top20.empty else 0,
                "total_formal_weight": float(stable["formal_edge_weight"].sum()) if not stable.empty else 0.0,
                "top1_latent_idx": int(group.iloc[0]["latent_idx"]) if not group.empty else -1,
                "top1_directional_auc": _safe_float(group.iloc[0]["directional_auc"], 0.5) if not group.empty else 0.5,
                "selection_status": "stable_latents_found" if count else "no_stable_latents",
                "fragmentation_class": cls,
            }
        )
    return pd.DataFrame(rows)


def _sets_by_label(matrix: pd.DataFrame, labels: Iterable[str], *, stable_only: bool = True) -> dict[str, set[int]]:
    rows = matrix[matrix["stable_edge"]] if stable_only else matrix
    out: dict[str, set[int]] = {}
    for label in labels:
        group = rows[rows["label"] == label.upper()]
        out[label.upper()] = set(group["latent_idx"].astype(int).tolist())
    return out


def _weights_by_label(matrix: pd.DataFrame, labels: Iterable[str]) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for label in labels:
        group = matrix[matrix["label"] == label.upper()]
        weights: dict[int, float] = {}
        for _, row in group.iterrows():
            weights[int(row["latent_idx"])] = max(float(row.get("latent_label_weight", 0.0)), 0.0)
        out[label.upper()] = weights
    return out


def _union_set(label_sets: dict[str, set[int]], members: Iterable[str]) -> set[int]:
    out: set[int] = set()
    for member in members:
        out |= label_sets.get(member.upper(), set())
    return out


def _union_weights(weight_sets: dict[str, dict[int, float]], members: Iterable[str]) -> dict[int, float]:
    out: dict[int, float] = {}
    for member in members:
        for idx, value in weight_sets.get(member.upper(), {}).items():
            out[idx] = max(out.get(idx, 0.0), float(value))
    return out


def build_overlap_thresholded(matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    leaf_sets = _sets_by_label(matrix, LEAF_LABELS, stable_only=True)
    generalized_latents = _generalized_latents_from_matrix(matrix)
    raw_top20_sets = {
        label: set(matrix[(matrix["label"] == label) & (matrix["association_rank"] <= 20)]["latent_idx"].astype(int))
        for label in DEFAULT_LABELS
    }

    for left, right in combinations(LEAF_LABELS, 2):
        left_set = leaf_sets[left]
        right_set = leaf_sets[right]
        raw_left = raw_top20_sets[left]
        raw_right = raw_top20_sets[right]
        rows.append(
            {
                "comparison_scope": "leaf_pair",
                "label_a": left,
                "label_b": right,
                "family_a": _leaf_family(left),
                "family_b": _leaf_family(right),
                "relation_type": "same_family" if _leaf_family(left) == _leaf_family(right) else "cross_family",
                "top20_jaccard": _jaccard(raw_left, raw_right),
                **_overlap_row_values(left_set, right_set, generalized_latents),
            }
        )

    family_defs = {**FAMILY_MEMBERS, **SUPPLEMENTAL_BLOCKS}
    family_names = list(FAMILY_MEMBERS)
    for left, right in combinations(family_names, 2):
        left_set = _union_set(leaf_sets, family_defs[left])
        right_set = _union_set(leaf_sets, family_defs[right])
        rows.append(
            {
                "comparison_scope": "family_union",
                "label_a": left,
                "label_b": right,
                "family_a": left,
                "family_b": right,
                "relation_type": "family_union",
                "top20_jaccard": np.nan,
                **_overlap_row_values(left_set, right_set, generalized_latents),
            }
        )
    for block, members in SUPPLEMENTAL_BLOCKS.items():
        for target in ("RE_family", "QU_family"):
            left_set = _union_set(leaf_sets, members)
            right_set = _union_set(leaf_sets, FAMILY_MEMBERS[target])
            rows.append(
                {
                    "comparison_scope": "supplemental_block",
                    "label_a": block,
                    "label_b": target,
                    "family_a": block,
                    "family_b": target,
                    "relation_type": "supplemental_block",
                    "top20_jaccard": np.nan,
                    **_overlap_row_values(left_set, right_set, generalized_latents),
                }
            )
    return pd.DataFrame(rows)


def build_overlap_weighted(matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    leaf_weights = _weights_by_label(matrix, LEAF_LABELS)
    for left, right in combinations(LEAF_LABELS, 2):
        rows.append(
            {
                "comparison_scope": "leaf_pair",
                "label_a": left,
                "label_b": right,
                "family_a": _leaf_family(left),
                "family_b": _leaf_family(right),
                "relation_type": "same_family" if _leaf_family(left) == _leaf_family(right) else "cross_family",
                "weighted_jaccard": _weighted_jaccard(leaf_weights[left], leaf_weights[right]),
            }
        )
    family_defs = {**FAMILY_MEMBERS, **SUPPLEMENTAL_BLOCKS}
    family_names = list(FAMILY_MEMBERS)
    for left, right in combinations(family_names, 2):
        rows.append(
            {
                "comparison_scope": "family_union",
                "label_a": left,
                "label_b": right,
                "family_a": left,
                "family_b": right,
                "relation_type": "family_union",
                "weighted_jaccard": _weighted_jaccard(
                    _union_weights(leaf_weights, family_defs[left]),
                    _union_weights(leaf_weights, family_defs[right]),
                ),
            }
        )
    for block, members in SUPPLEMENTAL_BLOCKS.items():
        for target in ("RE_family", "QU_family"):
            rows.append(
                {
                    "comparison_scope": "supplemental_block",
                    "label_a": block,
                    "label_b": target,
                    "family_a": block,
                    "family_b": target,
                    "relation_type": "supplemental_block",
                    "weighted_jaccard": _weighted_jaccard(
                        _union_weights(leaf_weights, members),
                        _union_weights(leaf_weights, FAMILY_MEMBERS[target]),
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_polysemanticity(matrix: pd.DataFrame) -> pd.DataFrame:
    stable = matrix[matrix["stable_edge"] & matrix["label"].isin(LEAF_LABELS)].copy()
    rows: list[dict[str, Any]] = []
    for latent_idx, group in stable.groupby("latent_idx"):
        labels = sorted(group["label"].astype(str).unique().tolist())
        families = sorted({_leaf_family(label) for label in labels})
        weights = group["formal_edge_weight"].astype(float).tolist()
        if len(labels) == 1:
            role = "label_specific"
        elif len(families) == 1:
            role = "sibling_shared"
        elif len(labels) >= 4 or len(families) >= 3:
            role = "generalized"
        else:
            role = "cross_family"
        rows.append(
            {
                "latent_idx": int(latent_idx),
                "n_thresholded_labels": len(labels),
                "labels": ",".join(labels),
                "n_families": len(families),
                "families": ",".join(families),
                "weighted_label_entropy": _weighted_entropy(weights),
                "max_formal_weight": float(np.max(weights)) if weights else 0.0,
                "edge_types": ",".join(sorted(group["edge_type"].astype(str).unique().tolist())),
                "role": role,
            }
        )
    return pd.DataFrame(rows)


def build_hierarchy_recovery(matrix: pd.DataFrame, overlap_weighted: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stable_sets = _sets_by_label(matrix, DEFAULT_LABELS, stable_only=True)
    for parent, children in PARENT_CHILDREN.items():
        parent_set = stable_sets[parent]
        child_union = _union_set(stable_sets, children)
        rows.append(
            {
                "metric_scope": "parent_child_consistency",
                "label_a": parent,
                "label_b": ",".join(children),
                "relation": "parent_vs_child_union",
                "jaccard": _jaccard(parent_set, child_union),
                "parent_covered_by_children": float(len(parent_set & child_union) / len(parent_set)) if parent_set else 0.0,
                "children_covered_by_parent": float(len(parent_set & child_union) / len(child_union)) if child_union else 0.0,
                "value": np.nan,
            }
        )
        for child in children:
            child_set = stable_sets[child]
            rows.append(
                {
                    "metric_scope": "parent_child_consistency",
                    "label_a": parent,
                    "label_b": child,
                    "relation": "parent_vs_child",
                    "jaccard": _jaccard(parent_set, child_set),
                    "parent_covered_by_children": float(len(parent_set & child_set) / len(parent_set)) if parent_set else 0.0,
                    "children_covered_by_parent": float(len(parent_set & child_set) / len(child_set)) if child_set else 0.0,
                    "value": np.nan,
                }
            )

    leaf = overlap_weighted[overlap_weighted["comparison_scope"] == "leaf_pair"]
    same = leaf[leaf["relation_type"] == "same_family"]["weighted_jaccard"]
    diff = leaf[leaf["relation_type"] == "cross_family"]["weighted_jaccard"]
    contrast = float(same.mean() - diff.mean()) if len(same) and len(diff) else 0.0
    rows.append(
        {
            "metric_scope": "leaf_hierarchy_recovery",
            "label_a": "same_family",
            "label_b": "cross_family",
            "relation": "weighted_family_contrast",
            "jaccard": np.nan,
            "parent_covered_by_children": np.nan,
            "children_covered_by_parent": np.nan,
            "value": contrast,
        }
    )
    return pd.DataFrame(rows)


def build_k_sensitivity(matrix: pd.DataFrame, config: LatentSpaceSearchV2Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k in config.k_values:
        top_sets = {
            label: set(
                matrix[(matrix["label"] == label) & (matrix["association_rank"] <= k)][
                    "latent_idx"
                ].astype(int)
            )
            for label in config.labels
        }
        leaf_pairs = []
        for left, right in combinations(LEAF_LABELS, 2):
            j = _jaccard(top_sets[left], top_sets[right])
            same = _leaf_family(left) == _leaf_family(right)
            leaf_pairs.append((left, right, j, same))
        same_values = [j for _, _, j, same in leaf_pairs if same]
        diff_values = [j for _, _, j, same in leaf_pairs if not same]
        family_contrast = (
            float(np.mean(same_values) - np.mean(diff_values))
            if same_values and diff_values
            else 0.0
        )
        n_edges = len(config.labels) * min(k, int(matrix["latent_idx"].max()) + 1)
        unique_latents = len(set().union(*top_sets.values())) if top_sets else 0
        latent_counts = Counter(idx for values in top_sets.values() for idx in values)
        multi_share = (
            sum(1 for count in latent_counts.values() if count > 1) / max(len(latent_counts), 1)
        )
        for left, right in KEY_SENSITIVITY_PAIRS:
            if left in top_sets and right in top_sets:
                rows.append(
                    {
                        "k": int(k),
                        "comparison": f"{left}-{right}",
                        "comparison_type": "key_pair",
                        "jaccard": _jaccard(top_sets[left], top_sets[right]),
                        "family_contrast": family_contrast,
                        "topk_edges": int(n_edges),
                        "unique_latents": int(unique_latents),
                        "multi_label_latent_share": float(multi_share),
                    }
                )
        re_family = _union_set(top_sets, FAMILY_MEMBERS["RE_family"])
        qu_family = _union_set(top_sets, FAMILY_MEMBERS["QU_family"])
        rows.append(
            {
                "k": int(k),
                "comparison": "RE_family-QU_family",
                "comparison_type": "family_union",
                "jaccard": _jaccard(re_family, qu_family),
                "family_contrast": family_contrast,
                "topk_edges": int(n_edges),
                "unique_latents": int(unique_latents),
                "multi_label_latent_share": float(multi_share),
            }
        )
        info_block = _union_set(top_sets, SUPPLEMENTAL_BLOCKS["advice_support_info_block"])
        rows.append(
            {
                "k": int(k),
                "comparison": "GI_SU_AF-RE_family",
                "comparison_type": "supplemental_block",
                "jaccard": _jaccard(info_block, re_family),
                "family_contrast": family_contrast,
                "topk_edges": int(n_edges),
                "unique_latents": int(unique_latents),
                "multi_label_latent_share": float(multi_share),
            }
        )
    return pd.DataFrame(rows)


def _record_text(record: dict[str, Any], fallback_row: pd.Series | None) -> str:
    for key in ("unit_text", "text", "utterance", "sentence"):
        value = record.get(key)
        if value:
            return str(value)
    if fallback_row is not None:
        for key in ("unit_text", "text", "utterance", "sentence"):
            if key in fallback_row and pd.notna(fallback_row[key]):
                return str(fallback_row[key])
    return ""


def _record_labels(row: pd.Series, labels: Iterable[str]) -> list[str]:
    out = []
    for label in labels:
        if label in row and _safe_float(row[label]) > 0:
            out.append(label)
    return out


def _common_tokens(texts: list[str], limit: int = 12) -> str:
    tokens: list[str] = []
    for text in texts:
        tokens.extend(re.findall(r"[A-Za-z']{3,}", text.lower()))
    stop = {
        "the",
        "and",
        "you",
        "that",
        "this",
        "with",
        "for",
        "are",
        "but",
        "have",
        "your",
        "what",
    }
    counts = Counter(tok for tok in tokens if tok not in stop)
    return ",".join(tok for tok, _ in counts.most_common(limit))


def build_semantic_review(
    matrix: pd.DataFrame,
    *,
    features: np.ndarray | None,
    label_df: pd.DataFrame | None,
    records: list[dict[str, Any]],
    config: LatentSpaceSearchV2Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    top = build_top20_candidates(matrix, config)
    if features is None or label_df is None:
        review = top.copy()
        review["interpretation_status"] = "needs_feature_store_for_examples"
        return review, pd.DataFrame()

    review_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    labels = list(config.labels)
    for _, cand in top.iterrows():
        label = str(cand["label"])
        latent_idx = int(cand["latent_idx"])
        if latent_idx < 0 or latent_idx >= features.shape[1] or label not in label_df.columns:
            continue
        acts = features[:, latent_idx]
        k = min(config.semantic_top_examples, len(acts))
        order = np.argsort(acts)[::-1][:k]
        target_hits: list[int] = []
        texts: list[str] = []
        label_counter: Counter[str] = Counter()
        quality_counter: Counter[str] = Counter()
        q_mark = 0
        for rank, idx in enumerate(order, start=1):
            row = label_df.iloc[int(idx)]
            rec = records[int(idx)] if int(idx) < len(records) else {}
            active_labels = _record_labels(row, labels)
            text = _record_text(rec, row)
            texts.append(text)
            if "?" in text:
                q_mark += 1
            for active in active_labels:
                label_counter[active] += 1
            for key in ("source_split", "quality", "quality_label"):
                if key in row and pd.notna(row[key]):
                    quality_counter[str(row[key])] += 1
                    break
            target = int(_safe_float(row.get(label, 0.0)) > 0)
            target_hits.append(target)
            example_rows.append(
                {
                    "label": label,
                    "latent_idx": latent_idx,
                    "example_rank": rank,
                    "row_idx": int(idx),
                    "activation": float(acts[int(idx)]),
                    "target_match": target,
                    "active_labels": ",".join(active_labels),
                    "text": text,
                }
            )
        purity = float(np.mean(target_hits)) if target_hits else 0.0
        if bool(cand.get("positive_support", False)) and purity >= 0.70:
            status = "high_purity"
        elif bool(cand.get("stable_edge", False)) or purity >= 0.40:
            status = "mixed_or_shared"
        else:
            status = "low_purity_review_required"
        review_rows.append(
            {
                "label": label,
                "topk_rank": int(cand.get("association_rank", 0)),
                "latent_idx": latent_idx,
                "edge_type": cand.get("edge_type", "weak_or_noise"),
                "directional_auc": float(cand.get("directional_auc", 0.5)),
                "abs_cohens_d": float(cand.get("abs_cohens_d", 0.0)),
                "precision_at_50": float(cand.get("precision_at_50", 0.0)),
                "precision_at_100": float(cand.get("precision_at_100", 0.0)),
                "target_purity_top_examples": purity,
                "dominant_labels_top_examples": ",".join(
                    f"{key}:{value}" for key, value in label_counter.most_common(8)
                ),
                "quality_distribution_top_examples": ",".join(
                    f"{key}:{value}" for key, value in quality_counter.most_common()
                ),
                "question_mark_rate": float(q_mark / max(k, 1)),
                "common_tokens": _common_tokens(texts),
                "recommended_role": status,
                "human_review_priority": "high" if status != "low_purity_review_required" else "normal",
            }
        )
    return pd.DataFrame(review_rows), pd.DataFrame(example_rows)


def build_lightweight_baselines(
    matrix: pd.DataFrame,
    *,
    step6_table_path: str | Path | None,
    config: LatentSpaceSearchV2Config,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_features = int(matrix["latent_idx"].max()) + 1 if not matrix.empty else 0
    rng = np.random.default_rng(config.random_state)
    labels = list(config.labels)
    for label in labels:
        group = matrix[matrix["label"] == label]
        prevalence = float(group["prevalence"].iloc[0]) if not group.empty else 0.0
        rows.append(
            {
                "baseline": "frequency_base_rate",
                "label": label,
                "metric": "expected_precision_at_50",
                "value": prevalence,
                "notes": "Expected top-k precision under no latent-label association.",
            }
        )

    if n_features > 0:
        jaccards = []
        for _ in range(config.random_baseline_repeats):
            sets = {
                label: set(
                    rng.choice(n_features, size=min(config.top_candidate_k, n_features), replace=False).astype(int)
                )
                for label in labels
            }
            for left, right in combinations(LEAF_LABELS, 2):
                jaccards.append(_jaccard(sets[left], sets[right]))
        rows.append(
            {
                "baseline": "random_top20_sets",
                "label": "ALL_LEAF_PAIRS",
                "metric": "mean_random_jaccard",
                "value": float(np.mean(jaccards)) if jaccards else 0.0,
                "notes": f"{config.random_baseline_repeats} random draws with Top{config.top_candidate_k} sets.",
            }
        )

    if step6_table_path:
        path = Path(step6_table_path)
        if path.exists():
            step6 = pd.read_csv(path)
            for _, row in step6.iterrows():
                for metric in ("mean_label_auc", "macro_probe_f1", "mean_effective_n_fraction", "family_contrast"):
                    if metric in row:
                        rows.append(
                            {
                                "baseline": f"step6_{row['representation']}",
                                "label": "ALL",
                                "metric": metric,
                                "value": float(row[metric]),
                                "notes": "Reference from Step 6 SAE/PCA/raw-hidden baseline comparison.",
                            }
                        )
    return pd.DataFrame(rows)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if pd.isna(obj):
        return None
    raise TypeError(f"Cannot serialize {type(obj).__name__}")


def _plot_heatmap(values: pd.DataFrame, path: Path, title: str, *, vmin: float | None = None, vmax: float | None = None) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig_height = min(18.0, max(5.0, 0.09 * max(len(values), 1)))
    fig_width = max(7.0, 0.75 * max(values.shape[1], 1))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    arr = values.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels(values.columns.tolist(), rotation=45, ha="right")
    if len(values) <= 220:
        ax.set_yticks(np.arange(values.shape[0]))
        ax.set_yticklabels(values.index.astype(str).tolist(), fontsize=6)
    else:
        ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.024, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_pair_heatmap(rows: pd.DataFrame, labels: list[str], value_col: str, path: Path, title: str) -> None:
    heat = pd.DataFrame(0.0, index=labels, columns=labels)
    for _, row in rows.iterrows():
        a = str(row["label_a"])
        b = str(row["label_b"])
        if a in heat.index and b in heat.columns:
            heat.loc[a, b] = float(row[value_col])
            heat.loc[b, a] = float(row[value_col])
    for label in labels:
        heat.loc[label, label] = 1.0
    _plot_heatmap(heat, path, title, vmin=0.0, vmax=1.0)


def write_figures(
    output_dir: Path,
    *,
    matrix: pd.DataFrame,
    top20: pd.DataFrame,
    thresholded: pd.DataFrame,
    overlap_weighted: pd.DataFrame,
    fragmentation: pd.DataFrame,
    polysemanticity: pd.DataFrame,
    k_sensitivity: pd.DataFrame,
    config: LatentSpaceSearchV2Config,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    top_ids = sorted(top20["latent_idx"].astype(int).unique().tolist())
    top_values = (
        matrix[matrix["latent_idx"].isin(top_ids)]
        .pivot_table(index="latent_idx", columns="label", values="directional_auc", aggfunc="max")
        .reindex(index=top_ids, columns=list(config.labels))
        .fillna(0.5)
    )
    _plot_heatmap(
        top_values,
        fig_dir / "latent_label_heatmap_top20.png",
        "Top20 candidate budget: directional AUC",
        vmin=0.5,
        vmax=1.0,
    )

    stable_ids = sorted(thresholded[thresholded["stable_edge"].fillna(False)]["latent_idx"].astype(int).unique().tolist())
    if stable_ids:
        stable_values = (
            matrix[matrix["latent_idx"].isin(stable_ids)]
            .pivot_table(index="latent_idx", columns="label", values="formal_edge_weight", aggfunc="max")
            .reindex(index=stable_ids, columns=list(config.labels))
            .fillna(0.0)
        )
    else:
        stable_values = pd.DataFrame(0.0, index=["no_stable_latents"], columns=list(config.labels))
    _plot_heatmap(
        stable_values,
        fig_dir / "latent_label_heatmap_thresholded.png",
        "Thresholded formal latent-label weights",
        vmin=0.0,
    )

    _plot_pair_heatmap(
        overlap_weighted[overlap_weighted["comparison_scope"] == "leaf_pair"],
        list(LEAF_LABELS),
        "weighted_jaccard",
        fig_dir / "weighted_overlap_heatmap.png",
        "Leaf-label weighted overlap",
    )

    frag = fragmentation.copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(frag))
    ax.bar(x - 0.2, frag["thresholded_latent_count"], width=0.4, label="thresholded count")
    ax.bar(x + 0.2, frag["effective_fragmentation"], width=0.4, label="effective fragmentation")
    ax.set_xticks(x)
    ax.set_xticklabels(frag["label"].tolist())
    ax.set_ylabel("Latent count")
    ax.set_title("Fragmentation v2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fragmentation_effective_n_bar.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if not polysemanticity.empty:
        counts = polysemanticity["n_thresholded_labels"].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel("Thresholded labels per latent")
    ax.set_ylabel("N latents")
    ax.set_title("Polysemanticity distribution")
    fig.tight_layout()
    fig.savefig(fig_dir / "polysemanticity_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for comparison, group in k_sensitivity.groupby("comparison"):
        if comparison in {"RES-REC", "QUO-QUC", "RE_family-QU_family", "GI_SU_AF-RE_family"}:
            group = group.sort_values("k")
            ax.plot(group["k"], group["jaccard"], marker="o", label=comparison)
    contrast = k_sensitivity.drop_duplicates("k").sort_values("k")
    if not contrast.empty:
        ax.plot(contrast["k"], contrast["family_contrast"], marker="s", linestyle="--", label="family_contrast")
    ax.set_xlabel("TopK")
    ax.set_ylabel("Jaccard / contrast")
    ax.set_title("K-sensitivity of latent-space structure")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "k_sensitivity_curves.png", dpi=180)
    plt.close(fig)


def write_report(
    output_dir: Path,
    *,
    fragmentation: pd.DataFrame,
    overlap_thresholded: pd.DataFrame,
    overlap_weighted: pd.DataFrame,
    polysemanticity: pd.DataFrame,
    k_sensitivity: pd.DataFrame,
    baselines: pd.DataFrame,
    config: LatentSpaceSearchV2Config,
) -> None:
    lines = [
        "# MISC SAE latent-space search v2",
        "",
        "Top-20 is retained only as a human-auditable candidate budget. Formal structural claims use thresholded sets, weighted metrics, and K-sensitivity analysis.",
        "",
        "## Operational thresholds",
        "",
        f"- `directional_auc >= {config.min_directional_auc:.2f}`",
        f"- `abs_cohens_d >= {config.min_abs_cohens_d:.2f}`",
        "- `significant_fdr == True`",
        f"- positive support requires `precision_at_50 >= prevalence + {config.precision_delta:.2f}`",
        "- negative latents are retained only as `negative_boundary` evidence.",
        "",
        "## Fragmentation",
        "",
        "| Label | Scope | Class | Stable latents | Positive | Boundary | Effective fragmentation | Status |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for _, row in fragmentation.iterrows():
        lines.append(
            f"| {row['label']} | {row['label_scope']} | {row['fragmentation_class']} | "
            f"{int(row['thresholded_latent_count'])} | {int(row['positive_support_count'])} | "
            f"{int(row['negative_boundary_count'])} | {float(row['effective_fragmentation']):.3f} | "
            f"{row['selection_status']} |"
        )

    leaf = overlap_thresholded[overlap_thresholded["comparison_scope"] == "leaf_pair"].sort_values(
        "jaccard", ascending=False
    )
    lines.extend(
        [
            "",
            "## Strongest thresholded leaf overlaps",
            "",
            "Raw thresholded Jaccard is reported together with a specific-only Jaccard that removes generalized latents from both sides before computing overlap.",
            "",
            "| Label A | Label B | Relation | Jaccard | Specific Jaccard | Top20 Jaccard | Generalized shared | Shared latents |",
            "|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    for _, row in leaf.head(12).iterrows():
        lines.append(
            f"| {row['label_a']} | {row['label_b']} | {row['relation_type']} | "
            f"{float(row['jaccard']):.3f} | {float(row['specific_jaccard']):.3f} | "
            f"{float(row['top20_jaccard']):.3f} | {row['generalized_shared_latents'] or '-'} | "
            f"{row['shared_latents'] or '-'} |"
        )

    specific_leaf = overlap_thresholded[
        overlap_thresholded["comparison_scope"] == "leaf_pair"
    ].sort_values("specific_jaccard", ascending=False)
    lines.extend(
        [
            "",
            "## Specific-only thresholded leaf overlaps",
            "",
            "| Label A | Label B | Relation | Specific Jaccard | Shared specific latents |",
            "|---|---|---|---:|---|",
        ]
    )
    for _, row in specific_leaf.head(12).iterrows():
        lines.append(
            f"| {row['label_a']} | {row['label_b']} | {row['relation_type']} | "
            f"{float(row['specific_jaccard']):.3f} | {row['shared_specific_latents'] or '-'} |"
        )

    weighted_leaf = overlap_weighted[overlap_weighted["comparison_scope"] == "leaf_pair"].sort_values(
        "weighted_jaccard", ascending=False
    )
    lines.extend(
        [
            "",
            "## Strongest weighted leaf overlaps",
            "",
            "| Label A | Label B | Relation | Weighted Jaccard |",
            "|---|---|---|---:|",
        ]
    )
    for _, row in weighted_leaf.head(12).iterrows():
        lines.append(
            f"| {row['label_a']} | {row['label_b']} | {row['relation_type']} | "
            f"{float(row['weighted_jaccard']):.3f} |"
        )

    role_counts = polysemanticity["role"].value_counts() if not polysemanticity.empty else pd.Series(dtype=int)
    lines.extend(
        [
            "",
            "## Polysemanticity",
            "",
            "| Role | N latents |",
            "|---|---:|",
        ]
    )
    for role, count in role_counts.items():
        lines.append(f"| {role} | {int(count)} |")

    lines.extend(
        [
            "",
            "## K-sensitivity highlights",
            "",
            "| K | Comparison | Jaccard | Family contrast | Multi-label share |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for _, row in k_sensitivity[k_sensitivity["comparison"].isin(["RES-REC", "QUO-QUC", "RE_family-QU_family"])].iterrows():
        lines.append(
            f"| {int(row['k'])} | {row['comparison']} | {float(row['jaccard']):.3f} | "
            f"{float(row['family_contrast']):.3f} | {float(row['multi_label_latent_share']):.3f} |"
        )

    if not baselines.empty:
        lines.extend(
            [
                "",
                "## Baseline references",
                "",
                "| Baseline | Label | Metric | Value |",
                "|---|---|---|---:|",
            ]
        )
        for _, row in baselines.head(30).iterrows():
            lines.append(
                f"| {row['baseline']} | {row['label']} | {row['metric']} | {float(row['value']):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Reading guide",
            "",
            "- Top20 tables and heatmaps are for inspection, not final inference.",
            "- `thresholded_latent_count` is the formal fragmentation count.",
            "- `effective_fragmentation` downweights weak diffuse tails.",
            "- Raw thresholded overlap can be inflated by generalized latents; use `specific_jaccard` when judging label-pair-specific sharing.",
            "- Parent labels `RE` and `QU` are consistency checks only.",
            "- The main structural conclusions should use leaf labels and family unions.",
        ]
    )
    (output_dir / "latent_space_search_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs(
    *,
    top20: pd.DataFrame,
    fragmentation: pd.DataFrame,
    overlap_thresholded: pd.DataFrame,
    overlap_weighted: pd.DataFrame,
    k_sensitivity: pd.DataFrame,
    output_dir: Path,
    config: LatentSpaceSearchV2Config,
) -> None:
    for label in config.labels:
        n = int((top20["label"] == label.upper()).sum())
        if n != config.top_candidate_k:
            raise AssertionError(f"{label} expected Top{config.top_candidate_k}, found {n}")
    if any(parent in set(overlap_thresholded[overlap_thresholded["comparison_scope"] == "leaf_pair"]["label_a"]) for parent in PARENT_LABELS):
        raise AssertionError("Parent labels leaked into leaf overlap label_a")
    if any(parent in set(overlap_thresholded[overlap_thresholded["comparison_scope"] == "leaf_pair"]["label_b"]) for parent in PARENT_LABELS):
        raise AssertionError("Parent labels leaked into leaf overlap label_b")
    for frame, col in (
        (overlap_thresholded, "jaccard"),
        (overlap_thresholded, "specific_jaccard"),
        (overlap_weighted, "weighted_jaccard"),
    ):
        values = pd.to_numeric(frame[col], errors="coerce").dropna()
        if not values.between(0.0, 1.0).all():
            raise AssertionError(f"{col} contains values outside [0, 1]")
    if not (
        fragmentation["effective_fragmentation"]
        <= fragmentation["thresholded_latent_count"] + 1e-9
    ).all():
        raise AssertionError("effective_fragmentation must be <= thresholded_latent_count")
    expected_ks = set(config.k_values)
    actual_ks = set(k_sensitivity["k"].astype(int).unique().tolist())
    if expected_ks != actual_ks:
        raise AssertionError(f"K-sensitivity missing K values: expected {expected_ks}, got {actual_ks}")
    report = (output_dir / "latent_space_search_report.md").read_text(encoding="utf-8")
    if "Top-20 is retained only as a human-auditable candidate budget" not in report:
        raise AssertionError("Report does not state the Top20 candidate-budget interpretation")


def run_latent_space_search_v2(
    *,
    matrix_path: str | Path,
    feature_store: str | Path | None,
    label_matrix: str | Path | None,
    records_path: str | Path | None,
    output_dir: str | Path,
    step6_table: str | Path | None = None,
    config: LatentSpaceSearchV2Config | None = None,
    make_figures: bool = True,
) -> dict[str, Any]:
    config = config or LatentSpaceSearchV2Config()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(matrix_path)
    label_df = pd.read_csv(label_matrix) if label_matrix and Path(label_matrix).exists() else None
    features = load_feature_store(feature_store)
    records = load_records(records_path)

    matrix = _normalize_matrix(
        source,
        label_df=label_df,
        features=features,
        config=config,
    )
    top20 = build_top20_candidates(matrix, config)
    thresholded = build_thresholded_sets(matrix, config)
    weighted = build_weighted_matrix(matrix)
    fragmentation = build_fragmentation(matrix, config)
    overlap_thresholded = build_overlap_thresholded(matrix)
    overlap_weighted = build_overlap_weighted(matrix)
    polysemanticity = build_polysemanticity(matrix)
    hierarchy = build_hierarchy_recovery(matrix, overlap_weighted)
    k_sensitivity = build_k_sensitivity(matrix, config)
    semantic_review, semantic_examples = build_semantic_review(
        matrix,
        features=features,
        label_df=label_df,
        records=records,
        config=config,
    )
    baselines = build_lightweight_baselines(
        matrix,
        step6_table_path=step6_table,
        config=config,
    )

    files = {
        "latent_label_association_v2": output_path / "latent_label_association_v2.csv",
        "top20_candidate_set_v2": output_path / "top20_candidate_set_v2.csv",
        "thresholded_latent_sets_v2": output_path / "thresholded_latent_sets_v2.csv",
        "weighted_latent_label_matrix_v2": output_path / "weighted_latent_label_matrix_v2.csv",
        "fragmentation_v2": output_path / "fragmentation_v2.csv",
        "overlap_thresholded_v2": output_path / "overlap_thresholded_v2.csv",
        "overlap_weighted_v2": output_path / "overlap_weighted_v2.csv",
        "polysemanticity_v2": output_path / "polysemanticity_v2.csv",
        "latent_role_assignments_v2": output_path / "latent_role_assignments_v2.csv",
        "hierarchy_recovery_v2": output_path / "hierarchy_recovery_v2.csv",
        "k_sensitivity_summary_v2": output_path / "k_sensitivity_summary_v2.csv",
        "semantic_review_candidates_v2": output_path / "semantic_review_candidates_v2.csv",
        "semantic_top_utterances_v2": output_path / "semantic_top_utterances_v2.csv",
        "baseline_lightweight_v2": output_path / "baseline_lightweight_v2.csv",
    }
    _write_csv(matrix, files["latent_label_association_v2"])
    _write_csv(top20, files["top20_candidate_set_v2"])
    _write_csv(thresholded, files["thresholded_latent_sets_v2"])
    _write_csv(weighted, files["weighted_latent_label_matrix_v2"])
    _write_csv(fragmentation, files["fragmentation_v2"])
    _write_csv(overlap_thresholded, files["overlap_thresholded_v2"])
    _write_csv(overlap_weighted, files["overlap_weighted_v2"])
    _write_csv(polysemanticity, files["polysemanticity_v2"])
    _write_csv(polysemanticity, files["latent_role_assignments_v2"])
    _write_csv(hierarchy, files["hierarchy_recovery_v2"])
    _write_csv(k_sensitivity, files["k_sensitivity_summary_v2"])
    _write_csv(semantic_review, files["semantic_review_candidates_v2"])
    _write_csv(semantic_examples, files["semantic_top_utterances_v2"])
    _write_csv(baselines, files["baseline_lightweight_v2"])

    if make_figures:
        write_figures(
            output_path,
            matrix=matrix,
            top20=top20,
            thresholded=thresholded,
            overlap_weighted=overlap_weighted,
            fragmentation=fragmentation,
            polysemanticity=polysemanticity,
            k_sensitivity=k_sensitivity,
            config=config,
        )

    write_report(
        output_path,
        fragmentation=fragmentation,
        overlap_thresholded=overlap_thresholded,
        overlap_weighted=overlap_weighted,
        polysemanticity=polysemanticity,
        k_sensitivity=k_sensitivity,
        baselines=baselines,
        config=config,
    )

    summary = {
        "analysis_version": "latent_space_search_v2",
        "config": asdict(config),
        "inputs": {
            "matrix_path": str(matrix_path),
            "feature_store": str(feature_store) if feature_store else None,
            "label_matrix": str(label_matrix) if label_matrix else None,
            "records_path": str(records_path) if records_path else None,
            "step6_table": str(step6_table) if step6_table else None,
        },
        "n_rows": int(matrix.shape[0]),
        "n_labels": int(len(config.labels)),
        "top20_edges": int(top20.shape[0]),
        "thresholded_edges": int(matrix["stable_edge"].sum()),
        "labels_without_stable_latents": fragmentation.loc[
            fragmentation["selection_status"] == "no_stable_latents", "label"
        ].tolist(),
        "fragmentation_class_counts": fragmentation["fragmentation_class"].value_counts().to_dict(),
        "polysemanticity_role_counts": polysemanticity["role"].value_counts().to_dict()
        if not polysemanticity.empty
        else {},
        "files": {key: str(value) for key, value in files.items()},
        "report": str(output_path / "latent_space_search_report.md"),
    }
    with (output_path / "latent_space_search_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=_json_default)

    validate_outputs(
        top20=top20,
        fragmentation=fragmentation,
        overlap_thresholded=overlap_thresholded,
        overlap_weighted=overlap_weighted,
        k_sensitivity=k_sensitivity,
        output_dir=output_path,
        config=config,
    )
    return {
        "summary": summary,
        "matrix": matrix,
        "top20": top20,
        "thresholded": thresholded,
        "fragmentation": fragmentation,
        "overlap_thresholded": overlap_thresholded,
        "overlap_weighted": overlap_weighted,
        "polysemanticity": polysemanticity,
        "hierarchy": hierarchy,
        "k_sensitivity": k_sensitivity,
        "semantic_review": semantic_review,
        "baselines": baselines,
        "output_dir": output_path,
    }
