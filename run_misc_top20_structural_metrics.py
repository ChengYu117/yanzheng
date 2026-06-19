"""Recompute MISC structural metrics from Top20 latent candidates.

This script is intentionally Top20-only. It does not apply the stable-edge
thresholds used by latent_space_search_v2. Parent labels RE and QU are excluded
from the main leaf-label structural metrics and kept only as a consistency
audit.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_TOP20_PATH = Path(
    "outputs/misc_full_sae_eval/interpretability/latent_space_search_v2/top20_candidate_set_v2.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/misc_full_sae_eval/interpretability/top20_structural_metrics"
)

PARENT_CHILDREN: dict[str, list[str]] = {"RE": ["RES", "REC"], "QU": ["QUO", "QUC"]}
PARENT_LABELS: tuple[str, ...] = ("RE", "QU")
LEAF_LABELS: tuple[str, ...] = ("RES", "REC", "QUO", "QUC", "GI", "SU", "AF")
ALL_LABELS: tuple[str, ...] = PARENT_LABELS + LEAF_LABELS
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
WEIGHT_PRIORITY: tuple[str, ...] = (
    "formal_edge_weight",
    "latent_label_weight",
    "association_score",
    "abs_cohens_d",
)


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _family_for_leaf(label: str) -> str:
    label = label.upper()
    for family, members in FAMILY_MEMBERS.items():
        if label in members:
            return family
    return label


def _block_for_leaf(label: str) -> str:
    label = label.upper()
    if label in {"RES", "REC"}:
        return "RE_family"
    if label in {"QUO", "QUC"}:
        return "QU_family"
    if label in {"GI", "SU", "AF"}:
        return "advice_support_info_block"
    return label


def _pair_relation_type(left: str, right: str) -> str:
    left_family = _family_for_leaf(left)
    right_family = _family_for_leaf(right)
    if left_family == right_family and left_family in {"RE_family", "QU_family"}:
        return "same_family"
    if _block_for_leaf(left) == _block_for_leaf(right):
        return "same_supplemental_block"
    return "cross_family"


def _classify_latent_role(labels: list[str]) -> str:
    families = {_family_for_leaf(label) for label in labels}
    blocks = {_block_for_leaf(label) for label in labels}
    if len(labels) == 1:
        return "label_specific"
    if len(families) == 1:
        return "same_family_shared"
    if len(blocks) == 1 and next(iter(blocks)) == "advice_support_info_block":
        return "same_supplemental_block"
    if len(labels) >= 4 or len(families) >= 3:
        return "generalized"
    return "cross_family_shared"


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 0.0


def _weighted_jaccard(left: dict[int, float], right: dict[int, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    numerator = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    denominator = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    return float(numerator / denominator) if denominator > 0 else 0.0


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


def _fragmentation_band(effective_n: float, unique_n: int) -> str:
    if unique_n <= 0:
        return "no_top20_latents"
    ratio = effective_n / unique_n
    if ratio >= 0.75:
        return "distributed_top20"
    if ratio >= 0.50:
        return "moderate_top20"
    return "concentrated_top20"


def _resolve_weight_col(matrix: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        if requested not in matrix.columns:
            raise ValueError(f"Requested weight column not found: {requested}")
        return requested
    for col in WEIGHT_PRIORITY:
        if col in matrix.columns:
            return col
    raise ValueError(
        "No usable weight column found. Expected one of: "
        + ", ".join(WEIGHT_PRIORITY)
    )


def discover_top20_candidates() -> Path:
    if DEFAULT_TOP20_PATH.exists():
        return DEFAULT_TOP20_PATH
    roots = [
        Path("outputs/misc_full_sae_eval/interpretability"),
        Path("outputs"),
    ]
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        candidates.extend(root.glob("**/top20_candidate_set_v2.csv"))
        candidates.extend(root.glob("**/topk_candidate_matrix.csv"))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {DEFAULT_TOP20_PATH} or another TopK candidate CSV under outputs/."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_top20_candidates(path: Path, *, top_k: int, weight_col: str) -> tuple[pd.DataFrame, str]:
    matrix = pd.read_csv(path)
    required = {"label", "latent_idx"}
    missing = sorted(required.difference(matrix.columns))
    if missing:
        raise ValueError(f"Top20 candidate CSV missing required columns: {missing}")

    matrix = matrix.copy()
    matrix["label"] = matrix["label"].astype(str).str.upper()
    matrix = matrix[matrix["label"].isin(ALL_LABELS)].copy()

    numeric_cols = [
        "latent_idx",
        "association_rank",
        "topk_rank",
        "candidate_budget_k",
        "cohens_d",
        "abs_cohens_d",
        "auc",
        "directional_auc",
        "association_score",
        "latent_label_weight",
        "formal_edge_weight",
        "precision_at_50",
        "precision_lift_at_50",
        "prevalence",
    ]
    for col in numeric_cols:
        if col in matrix.columns:
            matrix[col] = pd.to_numeric(matrix[col], errors="coerce").fillna(0.0)
    for col in ["significant_fdr", "stable_edge", "positive_support", "negative_boundary"]:
        if col in matrix.columns:
            matrix[col] = _bool_series(matrix[col])

    resolved_weight_col = _resolve_weight_col(matrix, weight_col)
    matrix["latent_idx"] = matrix["latent_idx"].astype(int)
    raw_weight = pd.to_numeric(matrix[resolved_weight_col], errors="coerce").fillna(0.0)
    if resolved_weight_col == "cohens_d":
        raw_weight = raw_weight.abs()
    matrix["top20_weight"] = raw_weight.clip(lower=0.0)

    rank_col = "association_rank" if "association_rank" in matrix.columns else None
    if rank_col is None and "topk_rank" in matrix.columns:
        rank_col = "topk_rank"

    if rank_col is not None:
        matrix = matrix.sort_values(["label", rank_col, "latent_idx"]).copy()
        matrix = matrix[matrix[rank_col] <= top_k].copy()
    else:
        sort_cols = [
            col
            for col in ["abs_cohens_d", "directional_auc", "association_score", "latent_idx"]
            if col in matrix.columns
        ]
        ascending = [False] * (len(sort_cols) - 1) + [True] if sort_cols else [True]
        if sort_cols:
            matrix = matrix.sort_values(["label"] + sort_cols, ascending=[True] + ascending)
        else:
            matrix = matrix.sort_values(["label", "latent_idx"])
        matrix = matrix.groupby("label", sort=False).head(top_k).copy()

    matrix = matrix.drop_duplicates(["label", "latent_idx"], keep="first").copy()
    matrix["top20_rank"] = (
        matrix.groupby("label", sort=False).cumcount().astype(int) + 1
    )
    matrix = matrix[matrix["top20_rank"] <= top_k].copy()
    return matrix.reset_index(drop=True), resolved_weight_col


def _weights_for_group(group: pd.DataFrame) -> np.ndarray:
    weights = group["top20_weight"].astype(float).clip(lower=0.0).to_numpy()
    if weights.sum() <= 0 and "abs_cohens_d" in group.columns:
        weights = group["abs_cohens_d"].astype(float).abs().to_numpy()
    return weights


def _sets_by_label(matrix: pd.DataFrame, labels: Iterable[str]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for label in labels:
        group = matrix[matrix["label"] == label.upper()]
        out[label.upper()] = set(group["latent_idx"].astype(int).tolist())
    return out


def _weights_by_label(matrix: pd.DataFrame, labels: Iterable[str]) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for label in labels:
        group = matrix[matrix["label"] == label.upper()].copy()
        weights = _weights_for_group(group)
        latent_weights: dict[int, float] = {}
        for latent_idx, weight in zip(group["latent_idx"].astype(int).tolist(), weights):
            latent_weights[latent_idx] = max(latent_weights.get(latent_idx, 0.0), float(weight))
        out[label.upper()] = latent_weights
    return out


def _union_set(label_sets: dict[str, set[int]], members: Iterable[str]) -> set[int]:
    out: set[int] = set()
    for member in members:
        out |= label_sets.get(member.upper(), set())
    return out


def _union_weights(weight_sets: dict[str, dict[int, float]], members: Iterable[str]) -> dict[int, float]:
    out: dict[int, float] = {}
    for member in members:
        for latent_idx, weight in weight_sets.get(member.upper(), {}).items():
            out[latent_idx] = max(out.get(latent_idx, 0.0), float(weight))
    return out


def build_fragmentation(matrix: pd.DataFrame, *, top_k: int, weight_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label in LEAF_LABELS:
        group = matrix[matrix["label"] == label].copy()
        group = group.sort_values("top20_rank")
        weights = _weights_for_group(group)
        unique_n = int(group["latent_idx"].nunique())
        weight_sum = float(weights.sum())
        effective = _effective_fragmentation(weights)
        entropy = _weighted_entropy(weights)
        top_weight_order = np.argsort(-weights) if len(weights) else np.asarray([], dtype=int)
        top1_share = float(weights[top_weight_order[0]] / weight_sum) if weight_sum > 0 and len(weights) else 0.0
        top5_share = (
            float(weights[top_weight_order[:5]].sum() / weight_sum)
            if weight_sum > 0 and len(weights)
            else 0.0
        )
        top_latents = group.iloc[top_weight_order[:5]]["latent_idx"].astype(int).tolist() if len(weights) else []
        stable_count = int(group["stable_edge"].sum()) if "stable_edge" in group.columns else 0
        rows.append(
            {
                "label": label,
                "family": _family_for_leaf(label),
                "top_k": top_k,
                "top20_count": int(group.shape[0]),
                "unique_top20_count": unique_n,
                "duplicate_count": int(group.shape[0] - unique_n),
                "weight_column": weight_col,
                "weight_sum": weight_sum,
                "effective_fragmentation_by_weight": effective,
                "effective_fragmentation_fraction": float(effective / unique_n) if unique_n else 0.0,
                "weight_entropy": entropy,
                "weight_entropy_fraction": float(entropy / math.log(unique_n)) if unique_n > 1 else 0.0,
                "top1_weight_share": top1_share,
                "top5_weight_share": top5_share,
                "top1_rank_latent_idx": int(group.iloc[0]["latent_idx"]) if not group.empty else -1,
                "top_weight_latent_idx": int(top_latents[0]) if top_latents else -1,
                "top5_weight_latents": ",".join(str(latent) for latent in top_latents),
                "stable_edge_count_aux": stable_count,
                "positive_support_count_aux": int(group["positive_support"].sum())
                if "positive_support" in group.columns
                else 0,
                "negative_boundary_count_aux": int(group["negative_boundary"].sum())
                if "negative_boundary" in group.columns
                else 0,
                "fragmentation_band_descriptive": _fragmentation_band(effective, unique_n),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["effective_fragmentation_by_weight", "unique_top20_count"],
        ascending=[False, False],
    )


def _overlap_values(
    left_set: set[int],
    right_set: set[int],
    left_weights: dict[int, float],
    right_weights: dict[int, float],
) -> dict[str, object]:
    shared = left_set & right_set
    union = left_set | right_set
    return {
        "intersection": int(len(shared)),
        "union": int(len(union)),
        "jaccard": _jaccard(left_set, right_set),
        "weighted_jaccard": _weighted_jaccard(left_weights, right_weights),
        "overlap_fraction_a": float(len(shared) / len(left_set)) if left_set else 0.0,
        "overlap_fraction_b": float(len(shared) / len(right_set)) if right_set else 0.0,
        "shared_latents": ",".join(str(latent) for latent in sorted(shared)),
    }


def build_leaf_pair_overlap(matrix: pd.DataFrame) -> pd.DataFrame:
    leaf_sets = _sets_by_label(matrix, LEAF_LABELS)
    leaf_weights = _weights_by_label(matrix, LEAF_LABELS)
    rows: list[dict[str, object]] = []
    for left, right in combinations(LEAF_LABELS, 2):
        rows.append(
            {
                "comparison_scope": "leaf_pair_top20",
                "label_a": left,
                "label_b": right,
                "family_a": _family_for_leaf(left),
                "family_b": _family_for_leaf(right),
                "block_a": _block_for_leaf(left),
                "block_b": _block_for_leaf(right),
                "relation_type": _pair_relation_type(left, right),
                **_overlap_values(
                    leaf_sets[left],
                    leaf_sets[right],
                    leaf_weights[left],
                    leaf_weights[right],
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["relation_type", "jaccard", "intersection"],
        ascending=[True, False, False],
    )


def build_family_overlap(matrix: pd.DataFrame) -> pd.DataFrame:
    leaf_sets = _sets_by_label(matrix, LEAF_LABELS)
    leaf_weights = _weights_by_label(matrix, LEAF_LABELS)
    rows: list[dict[str, object]] = []
    family_names = list(FAMILY_MEMBERS)
    for left, right in combinations(family_names, 2):
        left_set = _union_set(leaf_sets, FAMILY_MEMBERS[left])
        right_set = _union_set(leaf_sets, FAMILY_MEMBERS[right])
        left_weights = _union_weights(leaf_weights, FAMILY_MEMBERS[left])
        right_weights = _union_weights(leaf_weights, FAMILY_MEMBERS[right])
        rows.append(
            {
                "comparison_scope": "family_union_top20",
                "unit_a": left,
                "unit_b": right,
                "members_a": ",".join(FAMILY_MEMBERS[left]),
                "members_b": ",".join(FAMILY_MEMBERS[right]),
                **_overlap_values(left_set, right_set, left_weights, right_weights),
            }
        )
    family_defs = {**FAMILY_MEMBERS, **SUPPLEMENTAL_BLOCKS}
    for block, members in SUPPLEMENTAL_BLOCKS.items():
        for target in ["RE_family", "QU_family"]:
            left_set = _union_set(leaf_sets, members)
            right_set = _union_set(leaf_sets, family_defs[target])
            left_weights = _union_weights(leaf_weights, members)
            right_weights = _union_weights(leaf_weights, family_defs[target])
            rows.append(
                {
                    "comparison_scope": "supplemental_block_top20",
                    "unit_a": block,
                    "unit_b": target,
                    "members_a": ",".join(members),
                    "members_b": ",".join(family_defs[target]),
                    **_overlap_values(left_set, right_set, left_weights, right_weights),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["comparison_scope", "jaccard", "intersection"],
        ascending=[True, False, False],
    )


def build_parent_child_consistency(matrix: pd.DataFrame) -> pd.DataFrame:
    label_sets = _sets_by_label(matrix, ALL_LABELS)
    label_weights = _weights_by_label(matrix, ALL_LABELS)
    rows: list[dict[str, object]] = []
    for parent, children in PARENT_CHILDREN.items():
        comparisons = [("parent_child_union_top20", children)]
        comparisons.extend(("parent_child_top20", [child]) for child in children)
        for comparison_type, child_members in comparisons:
            parent_set = label_sets.get(parent, set())
            child_set = _union_set(label_sets, child_members)
            parent_weights = label_weights.get(parent, {})
            child_weights = _union_weights(label_weights, child_members)
            shared = parent_set & child_set
            rows.append(
                {
                    "parent_label": parent,
                    "child_labels": ",".join(child_members),
                    "comparison_scope": comparison_type,
                    "interpretation_scope": "consistency_check_only",
                    "parent_top20_n": int(len(parent_set)),
                    "child_top20_union_n": int(len(child_set)),
                    "intersection": int(len(shared)),
                    "union": int(len(parent_set | child_set)),
                    "jaccard": _jaccard(parent_set, child_set),
                    "weighted_jaccard": _weighted_jaccard(parent_weights, child_weights),
                    "parent_covered_by_children": float(len(shared) / len(parent_set))
                    if parent_set
                    else 0.0,
                    "children_covered_by_parent": float(len(shared) / len(child_set))
                    if child_set
                    else 0.0,
                    "shared_latents": ",".join(str(latent) for latent in sorted(shared)),
                }
            )
    return pd.DataFrame(rows)


def build_polysemanticity(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    leaf = matrix[matrix["label"].isin(LEAF_LABELS)].copy()
    rows: list[dict[str, object]] = []
    for latent_idx, group in leaf.groupby("latent_idx"):
        label_set = set(group["label"].astype(str).tolist())
        labels = [label for label in LEAF_LABELS if label in label_set]
        families = sorted({_family_for_leaf(label) for label in labels})
        blocks = sorted({_block_for_leaf(label) for label in labels})
        weights = _weights_for_group(group)
        entropy = _weighted_entropy(weights)
        best_idx = int(np.argmax(weights)) if len(weights) else 0
        best_label = str(group.iloc[best_idx]["label"]) if len(group) else ""
        label_weights = []
        for label, weight in zip(group["label"].astype(str).tolist(), weights):
            label_weights.append(f"{label}:{float(weight):.6g}")
        rows.append(
            {
                "latent_idx": int(latent_idx),
                "n_leaf_labels_supported": int(len(labels)),
                "leaf_labels_supported": ",".join(labels),
                "n_families_supported": int(len(families)),
                "families_supported": ",".join(families),
                "n_blocks_supported": int(len(blocks)),
                "blocks_supported": ",".join(blocks),
                "role": _classify_latent_role(labels),
                "min_top20_rank": int(group["top20_rank"].min()),
                "max_weight": float(weights.max()) if len(weights) else 0.0,
                "mean_weight": float(weights.mean()) if len(weights) else 0.0,
                "total_weight": float(weights.sum()) if len(weights) else 0.0,
                "weighted_label_entropy": entropy,
                "weighted_label_entropy_fraction": float(entropy / math.log(len(labels)))
                if len(labels) > 1
                else 0.0,
                "best_label_by_weight": best_label,
                "label_weight_pairs": ";".join(label_weights),
                "stable_edge_count_aux": int(group["stable_edge"].sum())
                if "stable_edge" in group.columns
                else 0,
            }
        )
    assignments = pd.DataFrame(rows)
    if assignments.empty:
        assignments = pd.DataFrame(
            columns=[
                "latent_idx",
                "n_leaf_labels_supported",
                "leaf_labels_supported",
                "n_families_supported",
                "families_supported",
                "n_blocks_supported",
                "blocks_supported",
                "role",
                "min_top20_rank",
                "max_weight",
                "mean_weight",
                "total_weight",
                "weighted_label_entropy",
                "weighted_label_entropy_fraction",
                "best_label_by_weight",
                "label_weight_pairs",
                "stable_edge_count_aux",
            ]
        )
    else:
        assignments = assignments.sort_values(
            ["n_leaf_labels_supported", "n_families_supported", "total_weight"],
            ascending=[False, False, False],
        )
    distribution = (
        assignments.groupby(["role", "n_leaf_labels_supported"], dropna=False)
        .size()
        .reset_index(name="n_latents")
        .sort_values(["role", "n_leaf_labels_supported"])
    )
    return assignments, distribution


def build_summary(
    *,
    input_path: Path,
    top_k: int,
    weight_col: str,
    matrix: pd.DataFrame,
    fragmentation: pd.DataFrame,
    leaf_overlap: pd.DataFrame,
    family_overlap: pd.DataFrame,
    parent_child: pd.DataFrame,
    polysemanticity: pd.DataFrame,
) -> dict[str, object]:
    same_family = leaf_overlap[leaf_overlap["relation_type"] == "same_family"]
    cross_family = leaf_overlap[leaf_overlap["relation_type"] == "cross_family"]
    role_counts = (
        polysemanticity["role"].value_counts().to_dict()
        if not polysemanticity.empty
        else {}
    )
    return {
        "analysis_version": "top20_structural_metrics_v1",
        "input_path": str(input_path),
        "top_k": top_k,
        "weight_column": weight_col,
        "main_label_scope": "leaf_labels_only",
        "parent_label_scope": "consistency_check_only",
        "leaf_labels": list(LEAF_LABELS),
        "parent_labels": list(PARENT_LABELS),
        "rows_by_label": matrix.groupby("label").size().sort_index().to_dict(),
        "unique_leaf_latents": int(matrix[matrix["label"].isin(LEAF_LABELS)]["latent_idx"].nunique()),
        "fragmentation_top3_by_effective_n": fragmentation.head(3).to_dict("records"),
        "same_family_overlap": same_family.to_dict("records"),
        "top_cross_family_overlap": cross_family.head(5).to_dict("records"),
        "top_family_overlap": family_overlap.head(5).to_dict("records"),
        "parent_child_consistency": parent_child.to_dict("records"),
        "polysemanticity_role_counts": role_counts,
    }


def write_report(
    path: Path,
    *,
    summary: dict[str, object],
    fragmentation: pd.DataFrame,
    leaf_overlap: pd.DataFrame,
    family_overlap: pd.DataFrame,
    parent_child: pd.DataFrame,
    poly_distribution: pd.DataFrame,
) -> None:
    same_family = leaf_overlap[leaf_overlap["relation_type"] == "same_family"]
    top_cross = leaf_overlap[leaf_overlap["relation_type"] == "cross_family"].head(8)
    lines: list[str] = [
        "# Top20 structural metrics",
        "",
        "This report recomputes Fragmentation, Overlap/Jaccard, and Polysemanticity directly from Top20 latent candidate sets.",
        "",
        "## Scope",
        "",
        f"- Input: `{summary['input_path']}`",
        f"- TopK: `{summary['top_k']}`",
        f"- Weight column: `{summary['weight_column']}`",
        "- Main metrics use leaf labels only: RES, REC, QUO, QUC, GI, SU, AF.",
        "- Parent labels RE and QU are excluded from main overlap/polysemanticity conclusions and appear only in the consistency audit.",
        "",
        "## Method",
        "",
        "- Fragmentation uses each label's Top20 set. The raw count is fixed by design, so the informative value is the effective weighted number of latents: `(sum(w)^2 / sum(w^2))`.",
        "- Overlap uses Jaccard on Top20 latent sets: `|A intersect B| / |A union B|`, with weighted Jaccard as a secondary concentration-aware measure.",
        "- Polysemanticity counts how many leaf labels and families each latent appears in across Top20 sets.",
        "",
        "## Fragmentation",
        "",
        "| Label | Family | Unique Top20 | Effective N | Eff. fraction | Top1 share | Top5 share | Band |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in fragmentation.iterrows():
        lines.append(
            f"| {row['label']} | {row['family']} | {int(row['unique_top20_count'])} | "
            f"{_fmt(row['effective_fragmentation_by_weight'])} | "
            f"{_fmt(row['effective_fragmentation_fraction'])} | "
            f"{_fmt(row['top1_weight_share'])} | {_fmt(row['top5_weight_share'])} | "
            f"{row['fragmentation_band_descriptive']} |"
        )

    lines.extend(
        [
            "",
            "## Same-family leaf overlap",
            "",
            "| Label A | Label B | Relation | Intersection | Jaccard | Weighted Jaccard | Shared latents |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for _, row in same_family.iterrows():
        lines.append(
            f"| {row['label_a']} | {row['label_b']} | {row['relation_type']} | "
            f"{int(row['intersection'])} | {_fmt(row['jaccard'])} | "
            f"{_fmt(row['weighted_jaccard'])} | {row['shared_latents'] or '-'} |"
        )

    lines.extend(
        [
            "",
            "## Top cross-family leaf overlap",
            "",
            "| Label A | Label B | Relation | Intersection | Jaccard | Weighted Jaccard | Shared latents |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for _, row in top_cross.iterrows():
        lines.append(
            f"| {row['label_a']} | {row['label_b']} | {row['relation_type']} | "
            f"{int(row['intersection'])} | {_fmt(row['jaccard'])} | "
            f"{_fmt(row['weighted_jaccard'])} | {row['shared_latents'] or '-'} |"
        )

    lines.extend(
        [
            "",
            "## Family-union overlap",
            "",
            "| Unit A | Unit B | Scope | Intersection | Jaccard | Weighted Jaccard | Shared latents |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for _, row in family_overlap.iterrows():
        lines.append(
            f"| {row['unit_a']} | {row['unit_b']} | {row['comparison_scope']} | "
            f"{int(row['intersection'])} | {_fmt(row['jaccard'])} | "
            f"{_fmt(row['weighted_jaccard'])} | {row['shared_latents'] or '-'} |"
        )

    lines.extend(
        [
            "",
            "## Polysemanticity",
            "",
            "| Role | N leaf labels | N latents |",
            "|---|---:|---:|",
        ]
    )
    for _, row in poly_distribution.iterrows():
        lines.append(
            f"| {row['role']} | {int(row['n_leaf_labels_supported'])} | {int(row['n_latents'])} |"
        )

    lines.extend(
        [
            "",
            "## Parent-child consistency audit",
            "",
            "| Parent | Child labels | Scope | Intersection | Jaccard | Parent covered | Children covered |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in parent_child.iterrows():
        lines.append(
            f"| {row['parent_label']} | {row['child_labels']} | {row['comparison_scope']} | "
            f"{int(row['intersection'])} | {_fmt(row['jaccard'])} | "
            f"{_fmt(row['parent_covered_by_children'])} | "
            f"{_fmt(row['children_covered_by_parent'])} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(
    output_dir: Path,
    fragmentation: pd.DataFrame,
    leaf_overlap: pd.DataFrame,
    poly_distribution: pd.DataFrame,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    frag = fragmentation.sort_values("effective_fragmentation_by_weight", ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(frag["label"], frag["effective_fragmentation_by_weight"])
    ax.set_ylabel("effective Top20 latents")
    ax.set_title("Top20 effective fragmentation")
    fig.tight_layout()
    fig.savefig(figure_dir / "top20_effective_fragmentation.png", dpi=220)
    plt.close(fig)

    heat = pd.DataFrame(0.0, index=list(LEAF_LABELS), columns=list(LEAF_LABELS))
    for _, row in leaf_overlap.iterrows():
        a = row["label_a"]
        b = row["label_b"]
        heat.loc[a, b] = float(row["jaccard"])
        heat.loc[b, a] = float(row["jaccard"])
    for label in LEAF_LABELS:
        heat.loc[label, label] = 1.0
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    im = ax.imshow(heat.values, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(LEAF_LABELS)))
    ax.set_yticks(np.arange(len(LEAF_LABELS)))
    ax.set_xticklabels(LEAF_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LEAF_LABELS)
    ax.set_title("Top20 leaf-label Jaccard")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figure_dir / "top20_leaf_jaccard_heatmap.png", dpi=220)
    plt.close(fig)

    role_counts = poly_distribution.groupby("role")["n_latents"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if not role_counts.empty:
        ax.bar(role_counts.index, role_counts.values)
    ax.set_ylabel("n latents")
    ax.set_title("Top20 latent role distribution")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figure_dir / "top20_polysemanticity_roles.png", dpi=220)
    plt.close(fig)


def validate_outputs(
    *,
    matrix: pd.DataFrame,
    fragmentation: pd.DataFrame,
    leaf_overlap: pd.DataFrame,
    family_overlap: pd.DataFrame,
    parent_child: pd.DataFrame,
    polysemanticity: pd.DataFrame,
    top_k: int,
) -> None:
    for label in LEAF_LABELS:
        n_rows = int(matrix[matrix["label"] == label].shape[0])
        if n_rows != top_k:
            raise AssertionError(f"{label} expected {top_k} TopK rows, found {n_rows}")
    pair_labels = set(leaf_overlap["label_a"]) | set(leaf_overlap["label_b"])
    if set(PARENT_LABELS) & pair_labels:
        raise AssertionError("Parent labels leaked into main leaf-pair overlap table.")
    relation_lookup = {
        tuple(sorted((row["label_a"], row["label_b"]))): row["relation_type"]
        for _, row in leaf_overlap.iterrows()
    }
    if relation_lookup.get(("QUC", "QUO")) != "same_family":
        raise AssertionError("QUO-QUC must be treated as a same-family leaf comparison.")
    if relation_lookup.get(("REC", "RES")) != "same_family":
        raise AssertionError("RES-REC must be treated as a same-family leaf comparison.")
    for frame, cols in [
        (leaf_overlap, ["jaccard", "weighted_jaccard", "overlap_fraction_a", "overlap_fraction_b"]),
        (family_overlap, ["jaccard", "weighted_jaccard", "overlap_fraction_a", "overlap_fraction_b"]),
        (parent_child, ["jaccard", "weighted_jaccard", "parent_covered_by_children", "children_covered_by_parent"]),
    ]:
        for col in cols:
            if not frame[col].between(0.0, 1.0).all():
                raise AssertionError(f"{col} has values outside [0, 1].")
    if not (
        fragmentation["effective_fragmentation_by_weight"]
        <= fragmentation["unique_top20_count"] + 1e-9
    ).all():
        raise AssertionError("Effective fragmentation must be <= unique Top20 count.")
    expected_poly_n = int(matrix[matrix["label"].isin(LEAF_LABELS)]["latent_idx"].nunique())
    if int(polysemanticity.shape[0]) != expected_poly_n:
        raise AssertionError(
            f"Polysemanticity rows ({polysemanticity.shape[0]}) do not match unique leaf latents ({expected_poly_n})."
        )


def write_outputs(
    *,
    output_dir: Path,
    input_path: Path,
    top_k: int,
    weight_col: str,
    matrix: pd.DataFrame,
    fragmentation: pd.DataFrame,
    leaf_overlap: pd.DataFrame,
    family_overlap: pd.DataFrame,
    parent_child: pd.DataFrame,
    polysemanticity: pd.DataFrame,
    poly_distribution: pd.DataFrame,
    make_figures: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_dir / "top20_candidate_set_used.csv", index=False)
    fragmentation.to_csv(output_dir / "top20_fragmentation.csv", index=False)
    leaf_overlap.to_csv(output_dir / "top20_leaf_pair_overlap.csv", index=False)
    family_overlap.to_csv(output_dir / "top20_family_overlap.csv", index=False)
    parent_child.to_csv(output_dir / "top20_parent_child_consistency.csv", index=False)
    polysemanticity.to_csv(output_dir / "top20_polysemanticity.csv", index=False)
    poly_distribution.to_csv(output_dir / "top20_role_distribution.csv", index=False)

    summary = build_summary(
        input_path=input_path,
        top_k=top_k,
        weight_col=weight_col,
        matrix=matrix,
        fragmentation=fragmentation,
        leaf_overlap=leaf_overlap,
        family_overlap=family_overlap,
        parent_child=parent_child,
        polysemanticity=polysemanticity,
    )
    (output_dir / "top20_structural_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(
        output_dir / "top20_structural_report.md",
        summary=summary,
        fragmentation=fragmentation,
        leaf_overlap=leaf_overlap,
        family_overlap=family_overlap,
        parent_child=parent_child,
        poly_distribution=poly_distribution,
    )
    if make_figures:
        plot_outputs(output_dir, fragmentation, leaf_overlap, poly_distribution)
    return summary


def run_top20_structural_metrics(
    *,
    top20_candidates: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    top_k: int = 20,
    weight_col: str = "auto",
    make_figures: bool = True,
) -> dict[str, object]:
    input_path = top20_candidates if top20_candidates is not None else discover_top20_candidates()
    matrix, resolved_weight_col = read_top20_candidates(
        input_path,
        top_k=top_k,
        weight_col=weight_col,
    )
    fragmentation = build_fragmentation(matrix, top_k=top_k, weight_col=resolved_weight_col)
    leaf_overlap = build_leaf_pair_overlap(matrix)
    family_overlap = build_family_overlap(matrix)
    parent_child = build_parent_child_consistency(matrix)
    polysemanticity, poly_distribution = build_polysemanticity(matrix)
    validate_outputs(
        matrix=matrix,
        fragmentation=fragmentation,
        leaf_overlap=leaf_overlap,
        family_overlap=family_overlap,
        parent_child=parent_child,
        polysemanticity=polysemanticity,
        top_k=top_k,
    )
    return write_outputs(
        output_dir=output_dir,
        input_path=input_path,
        top_k=top_k,
        weight_col=resolved_weight_col,
        matrix=matrix,
        fragmentation=fragmentation,
        leaf_overlap=leaf_overlap,
        family_overlap=family_overlap,
        parent_child=parent_child,
        polysemanticity=polysemanticity,
        poly_distribution=poly_distribution,
        make_figures=make_figures,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute MISC structural metrics directly from Top20 latent candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--top20-candidates",
        default=None,
        help="Top20 candidate CSV. If omitted, the script searches existing outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for Top20 structural metric outputs.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--weight-col",
        default="auto",
        help="Weight column for effective fragmentation and weighted Jaccard; use 'auto' for priority lookup.",
    )
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    top20_path = Path(args.top20_candidates) if args.top20_candidates else None
    summary = run_top20_structural_metrics(
        top20_candidates=top20_path,
        output_dir=Path(args.output_dir),
        top_k=args.top_k,
        weight_col=args.weight_col,
        make_figures=not args.no_figures,
    )
    print("Completed Top20 structural metrics.")
    print(f"Input: {summary['input_path']}")
    print(f"Output dir: {args.output_dir}")
    print(f"Weight column: {summary['weight_column']}")
    print(f"Unique leaf latents: {summary['unique_leaf_latents']}")
    print(f"Report: {Path(args.output_dir) / 'top20_structural_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
