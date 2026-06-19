"""Top-K convergence analysis for MISC SAE structural metrics.

This script checks whether Fragmentation, Overlap/Jaccard, and
Polysemanticity stabilize as the per-label latent candidate budget increases.
It is intended to justify Top20 as an auditable structural window and Top100 as
a downstream-effect fragmentation candidate-recall budget.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from run_misc_downstream_effect_fragmentation import (
    DownstreamEffectFragmentationConfig,
    _prepare_association_matrix,
    analyze_label,
    load_feature_store,
)


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


DEFAULT_LABELS: tuple[str, ...] = ("RES", "REC", "QUO", "QUC", "GI", "SU", "AF")
DEFAULT_K_GRID: tuple[int, ...] = (5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/misc_full_sae_eval/interpretability/structural_k_convergence"
)

FAMILY_MEMBERS: dict[str, tuple[str, ...]] = {
    "RE_family": ("RES", "REC"),
    "QU_family": ("QUO", "QUC"),
    "GI": ("GI",),
    "SU": ("SU",),
    "AF": ("AF",),
}
SUPPLEMENTAL_BLOCKS: dict[str, tuple[str, ...]] = {
    "advice_support_info_block": ("GI", "SU", "AF"),
}


@dataclass(frozen=True)
class StructuralKConvergenceConfig:
    labels: tuple[str, ...] = DEFAULT_LABELS
    k_grid: tuple[int, ...] = DEFAULT_K_GRID
    ranking_metric: str = "directional_auc"
    cv_folds: int = 5
    min_auc: float = 0.70
    effect_epsilon: float = 0.001
    precision_k: int = 50
    random_state: int = 13
    max_iter: int = 1000


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


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
    if len(labels) <= 1:
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


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _candidate_weight(row: pd.Series) -> float:
    for col in ("formal_edge_weight", "latent_label_weight", "association_score"):
        if col in row and pd.notna(row[col]):
            value = float(row[col])
            if value > 0:
                return value
    if "directional_auc" in row and pd.notna(row["directional_auc"]):
        return max(float(row["directional_auc"]) - 0.5, 0.0)
    if "abs_cohens_d" in row and pd.notna(row["abs_cohens_d"]):
        return max(float(row["abs_cohens_d"]), 0.0)
    return 0.0


def _normalize_ranking_metric(ranking_metric: str) -> str:
    metric = str(ranking_metric).strip().lower()
    aliases = {
        "auc": "directional_auc",
        "directional_auc": "directional_auc",
        "d": "abs_cohens_d",
        "cohens_d": "abs_cohens_d",
        "abs_cohens_d": "abs_cohens_d",
        "formal": "formal_edge_weight",
        "formal_edge": "formal_edge_weight",
        "formal_edge_weight": "formal_edge_weight",
    }
    if metric not in aliases:
        raise ValueError(
            "Unsupported ranking metric. Use one of: directional_auc, abs_cohens_d, formal_edge_weight"
        )
    return aliases[metric]


def _ranking_sort_spec(ranking_metric: str, columns: Iterable[str]) -> tuple[list[str], list[bool]]:
    metric = _normalize_ranking_metric(ranking_metric)
    available = set(columns)
    if metric == "directional_auc":
        required = ["directional_auc", "abs_cohens_d", "latent_idx"]
        missing = [col for col in required if col not in available]
        if missing:
            raise ValueError(f"Association matrix missing ranking columns: {missing}")
        return required, [False, False, True]
    if metric == "abs_cohens_d":
        required = ["abs_cohens_d", "directional_auc", "latent_idx"]
        missing = [col for col in required if col not in available]
        if missing:
            raise ValueError(f"Association matrix missing ranking columns: {missing}")
        return required, [False, False, True]
    required = ["stable_edge", "formal_edge_weight", "abs_cohens_d", "directional_auc", "latent_idx"]
    missing = [col for col in required if col not in available]
    if missing:
        raise ValueError(f"Association matrix missing ranking columns: {missing}")
    return required, [False, False, False, False, True]


def _sort_candidates(group: pd.DataFrame, ranking_metric: str) -> pd.DataFrame:
    sort_cols, ascending = _ranking_sort_spec(ranking_metric, group.columns)
    return group.sort_values(sort_cols, ascending=ascending).copy()


def _topk_rows(
    association: pd.DataFrame,
    labels: Iterable[str],
    k: int,
    ranking_metric: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for label in labels:
        group = association[association["label"] == label].copy()
        group = _sort_candidates(group, ranking_metric)
        group = group.drop_duplicates("latent_idx", keep="first").head(k).copy()
        group["top_k"] = int(k)
        group["ranking_metric"] = _normalize_ranking_metric(ranking_metric)
        group["topk_rank"] = np.arange(1, len(group) + 1)
        group["topk_rank_directional_auc"] = group["topk_rank"]
        group["topk_weight"] = group.apply(_candidate_weight, axis=1)
        rows.append(group)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _build_candidate_pool_for_k(
    association: pd.DataFrame,
    *,
    feature_dim: int,
    k: int,
    ranking_metric: str,
    labels: Iterable[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    metric = _normalize_ranking_metric(ranking_metric)
    for label in labels:
        group = association[association["label"] == label].copy()
        group = group[(group["latent_idx"] >= 0) & (group["latent_idx"] < feature_dim)].copy()
        group = _sort_candidates(group, metric)
        group = group.drop_duplicates("latent_idx", keep="first").head(k).copy()
        group["candidate_rank"] = np.arange(1, len(group) + 1)
        group["candidate_rank_directional_auc"] = group["candidate_rank"]
        group["candidate_ranking_metric"] = metric
        group["candidate_source"] = f"{metric}_top{k}"
        rows.append(group)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _weight_dict(rows: pd.DataFrame) -> dict[int, float]:
    out: dict[int, float] = {}
    for _, row in rows.iterrows():
        latent_idx = int(row["latent_idx"])
        out[latent_idx] = max(out.get(latent_idx, 0.0), float(row.get("topk_weight", 0.0)))
    return out


def _family_union_specs(labels: Iterable[str]) -> dict[str, tuple[str, ...]]:
    label_set = {label.upper() for label in labels}
    specs: dict[str, tuple[str, ...]] = {}
    for family, members in FAMILY_MEMBERS.items():
        selected = tuple(label for label in members if label in label_set)
        if selected:
            specs[family] = selected
    for block, members in SUPPLEMENTAL_BLOCKS.items():
        selected = tuple(label for label in members if label in label_set)
        if len(selected) >= 2:
            specs[block] = selected
    return specs


def compute_overlap_by_k(association: pd.DataFrame, config: StructuralKConvergenceConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = tuple(label.upper() for label in config.labels)
    family_specs = _family_union_specs(labels)
    for k in config.k_grid:
        topk = _topk_rows(association, labels, k, config.ranking_metric)
        label_sets = {
            label: set(topk[topk["label"] == label]["latent_idx"].astype(int).tolist())
            for label in labels
        }
        label_weights = {
            label: _weight_dict(topk[topk["label"] == label])
            for label in labels
        }
        for left, right in combinations(labels, 2):
            rows.append(
                {
                    "top_k": int(k),
                    "comparison_scope": "leaf_pair",
                    "label_a": left,
                    "label_b": right,
                    "relation_type": _pair_relation_type(left, right),
                    "intersection": len(label_sets[left] & label_sets[right]),
                    "union": len(label_sets[left] | label_sets[right]),
                    "jaccard": _jaccard(label_sets[left], label_sets[right]),
                    "weighted_jaccard": _weighted_jaccard(label_weights[left], label_weights[right]),
                    "shared_latents": ",".join(str(v) for v in sorted(label_sets[left] & label_sets[right])),
                    "core_pair": (left, right) in {("RES", "REC"), ("QUO", "QUC")},
                }
            )

        family_sets: dict[str, set[int]] = {}
        family_weights: dict[str, dict[int, float]] = {}
        for family, members in family_specs.items():
            family_rows = topk[topk["label"].isin(members)]
            family_sets[family] = set(family_rows["latent_idx"].astype(int).tolist())
            family_weights[family] = _weight_dict(family_rows)
        for left, right in combinations(family_sets.keys(), 2):
            relation = "same_block" if left == right else "family_union"
            rows.append(
                {
                    "top_k": int(k),
                    "comparison_scope": "family_union",
                    "label_a": left,
                    "label_b": right,
                    "relation_type": relation,
                    "intersection": len(family_sets[left] & family_sets[right]),
                    "union": len(family_sets[left] | family_sets[right]),
                    "jaccard": _jaccard(family_sets[left], family_sets[right]),
                    "weighted_jaccard": _weighted_jaccard(family_weights[left], family_weights[right]),
                    "shared_latents": ",".join(str(v) for v in sorted(family_sets[left] & family_sets[right])),
                    "core_pair": (left, right) == ("RE_family", "QU_family"),
                }
            )
    return pd.DataFrame(rows)


def compute_polysemanticity_by_k(
    association: pd.DataFrame,
    config: StructuralKConvergenceConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    labels = tuple(label.upper() for label in config.labels)
    role_order = [
        "label_specific",
        "same_family_shared",
        "same_supplemental_block",
        "cross_family_shared",
        "generalized",
    ]
    for k in config.k_grid:
        topk = _topk_rows(association, labels, k, config.ranking_metric)
        role_counts = {role: 0 for role in role_order}
        label_counts: list[int] = []
        for latent_idx, group in topk.groupby("latent_idx"):
            latent_labels = sorted(group["label"].astype(str).unique().tolist())
            role = _classify_latent_role(latent_labels)
            role_counts[role] = role_counts.get(role, 0) + 1
            label_counts.append(len(latent_labels))
            strongest = group.sort_values(
                ["topk_weight", "directional_auc", "abs_cohens_d"],
                ascending=[False, False, False],
            ).iloc[0]
            latent_rows.append(
                {
                    "top_k": int(k),
                    "latent_idx": int(latent_idx),
                    "labels": ",".join(latent_labels),
                    "n_labels": len(latent_labels),
                    "families": ",".join(sorted({_family_for_leaf(label) for label in latent_labels})),
                    "latent_role": role,
                    "strongest_label": str(strongest["label"]),
                    "strongest_weight": float(strongest.get("topk_weight", 0.0)),
                }
            )
        unique_n = len(label_counts)
        row: dict[str, Any] = {
            "top_k": int(k),
            "edges": int(topk.shape[0]),
            "unique_latents": int(unique_n),
            "mean_labels_per_latent": float(np.mean(label_counts)) if label_counts else 0.0,
            "max_labels_per_latent": int(max(label_counts)) if label_counts else 0,
        }
        for role in role_order:
            count = role_counts.get(role, 0)
            row[f"{role}_count"] = int(count)
            row[f"{role}_share"] = _safe_div(count, unique_n)
        row["multi_label_latent_count"] = int(unique_n - role_counts.get("label_specific", 0))
        row["multi_label_latent_share"] = _safe_div(row["multi_label_latent_count"], unique_n)
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(latent_rows)


def compute_fragmentation_by_k(
    association: pd.DataFrame,
    features: np.ndarray,
    labels_df: pd.DataFrame,
    config: StructuralKConvergenceConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []
    effect_frames: list[pd.DataFrame] = []
    for k in config.k_grid:
        frag_config = DownstreamEffectFragmentationConfig(
            labels=config.labels,
            candidate_top_k=int(k),
            cv_folds=config.cv_folds,
            min_auc=config.min_auc,
            effect_epsilon=config.effect_epsilon,
            precision_k=config.precision_k,
            random_state=config.random_state,
            max_iter=config.max_iter,
        )
        candidate_pool = _build_candidate_pool_for_k(
            association,
            feature_dim=features.shape[1],
            k=int(k),
            ranking_metric=config.ranking_metric,
            labels=config.labels,
        )
        for label in config.labels:
            fold_metrics, latent_effects, summary = analyze_label(
                features=features,
                labels_df=labels_df,
                candidate_pool=candidate_pool,
                label=label,
                config=frag_config,
            )
            fold_metrics = fold_metrics.copy()
            latent_effects = latent_effects.copy()
            fold_metrics["top_k"] = int(k)
            latent_effects["top_k"] = int(k)
            summary = dict(summary)
            summary["top_k"] = int(k)
            summary["ranking_metric"] = _normalize_ranking_metric(config.ranking_metric)
            summary["candidate_rule"] = f"{_normalize_ranking_metric(config.ranking_metric)}_top{k}"
            summary["effect_equivalent_latents"] = summary["effective_effect_latent_count"]
            summary_rows.append(summary)
            fold_frames.append(fold_metrics)
            effect_frames.append(latent_effects)
    return (
        pd.DataFrame(summary_rows),
        pd.concat(fold_frames, ignore_index=True, sort=False) if fold_frames else pd.DataFrame(),
        pd.concat(effect_frames, ignore_index=True, sort=False) if effect_frames else pd.DataFrame(),
    )


def _first_two_consecutive_stable_k(
    k_values: list[int],
    stable_by_transition: dict[tuple[int, int], bool],
) -> int | None:
    for idx in range(len(k_values) - 2):
        first = (k_values[idx], k_values[idx + 1])
        second = (k_values[idx + 1], k_values[idx + 2])
        if stable_by_transition.get(first, False) and stable_by_transition.get(second, False):
            return int(k_values[idx])
    return None


def _summarize_fragmentation_convergence(fragmentation: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    k_values = sorted(int(k) for k in fragmentation["top_k"].unique())
    detail_rows: list[dict[str, Any]] = []
    label_recommended: dict[str, int | None] = {}
    for label, group in fragmentation.groupby("label"):
        group = group.sort_values("top_k")
        stable: dict[tuple[int, int], bool] = {}
        prev_row: pd.Series | None = None
        for _, row in group.iterrows():
            if prev_row is not None:
                prev_k = int(prev_row["top_k"])
                curr_k = int(row["top_k"])
                prev_value = float(prev_row["effect_equivalent_latents"])
                curr_value = float(row["effect_equivalent_latents"])
                rel_delta = abs(curr_value - prev_value) / max(abs(prev_value), 1e-12)
                class_unchanged = str(curr_value) != "" and row["fragmentation_class"] == prev_row["fragmentation_class"]
                is_stable = rel_delta < 0.10 and bool(class_unchanged)
                stable[(prev_k, curr_k)] = is_stable
                detail_rows.append(
                    {
                        "metric": "fragmentation",
                        "label": label,
                        "from_k": prev_k,
                        "to_k": curr_k,
                        "relative_delta": rel_delta,
                        "class_unchanged": bool(class_unchanged),
                        "transition_stable": bool(is_stable),
                    }
                )
            prev_row = row
        label_recommended[str(label)] = _first_two_consecutive_stable_k(k_values, stable)
    non_converged = sorted(label for label, value in label_recommended.items() if value is None)
    recommended_values = [value for value in label_recommended.values() if value is not None]
    recommended_k = max(recommended_values) if recommended_values and not non_converged else None
    summary = {
        "metric": "fragmentation",
        "converged": not non_converged,
        "recommended_upper_k": recommended_k,
        "criterion": "two consecutive K transitions with <10% effect-equivalent change and unchanged class",
        "non_converged_labels": ",".join(non_converged),
        "label_recommended_k": json.dumps(label_recommended, ensure_ascii=False),
    }
    return summary, pd.DataFrame(detail_rows)


def _overlap_aggregate(overlap: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    leaf = overlap[overlap["comparison_scope"] == "leaf_pair"].copy()
    for k, group in leaf.groupby("top_k"):
        same = group[group["relation_type"] == "same_family"]
        cross = group[group["relation_type"] != "same_family"]
        core = group[group["core_pair"]]
        rows.append(
            {
                "top_k": int(k),
                "mean_leaf_jaccard": float(group["jaccard"].mean()) if not group.empty else 0.0,
                "mean_same_family_jaccard": float(same["jaccard"].mean()) if not same.empty else 0.0,
                "mean_cross_family_jaccard": float(cross["jaccard"].mean()) if not cross.empty else 0.0,
                "family_contrast_jaccard": (
                    float(same["jaccard"].mean()) - float(cross["jaccard"].mean())
                    if not same.empty and not cross.empty
                    else 0.0
                ),
                "mean_weighted_jaccard": float(group["weighted_jaccard"].mean()) if not group.empty else 0.0,
                "max_core_jaccard": float(core["jaccard"].max()) if not core.empty else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("top_k")


def _summarize_overlap_convergence(overlap: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    aggregate = _overlap_aggregate(overlap)
    k_values = aggregate["top_k"].astype(int).tolist()
    transition_rows: list[dict[str, Any]] = []
    stable: dict[tuple[int, int], bool] = {}
    core = overlap[(overlap["comparison_scope"] == "leaf_pair") & (overlap["core_pair"])].copy()
    for idx in range(1, len(k_values)):
        prev_k = k_values[idx - 1]
        curr_k = k_values[idx]
        prev_core = core[core["top_k"] == prev_k].set_index(["label_a", "label_b"])["jaccard"]
        curr_core = core[core["top_k"] == curr_k].set_index(["label_a", "label_b"])["jaccard"]
        common_pairs = sorted(set(prev_core.index) & set(curr_core.index))
        max_delta = max((abs(float(curr_core[pair]) - float(prev_core[pair])) for pair in common_pairs), default=0.0)
        prev_contrast = float(aggregate[aggregate["top_k"] == prev_k]["family_contrast_jaccard"].iloc[0])
        curr_contrast = float(aggregate[aggregate["top_k"] == curr_k]["family_contrast_jaccard"].iloc[0])
        ranking_not_reversed = (prev_contrast == 0 and curr_contrast == 0) or (prev_contrast * curr_contrast >= 0)
        is_stable = max_delta < 0.03 and ranking_not_reversed
        stable[(prev_k, curr_k)] = bool(is_stable)
        transition_rows.append(
            {
                "metric": "overlap",
                "from_k": prev_k,
                "to_k": curr_k,
                "max_core_jaccard_delta": max_delta,
                "ranking_not_reversed": bool(ranking_not_reversed),
                "transition_stable": bool(is_stable),
            }
        )
    recommended_k = _first_two_consecutive_stable_k(k_values, stable)
    summary = {
        "metric": "overlap_jaccard",
        "converged": recommended_k is not None,
        "recommended_upper_k": recommended_k,
        "criterion": "two consecutive K transitions with core-pair Jaccard delta <0.03 and no same/cross ranking reversal",
        "non_converged_labels": "",
        "label_recommended_k": "",
    }
    return summary, pd.DataFrame(transition_rows), aggregate


def _summarize_polysemanticity_convergence(poly: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    poly = poly.sort_values("top_k")
    k_values = poly["top_k"].astype(int).tolist()
    transition_rows: list[dict[str, Any]] = []
    stable: dict[tuple[int, int], bool] = {}
    prev_row: pd.Series | None = None
    for _, row in poly.iterrows():
        if prev_row is not None:
            prev_k = int(prev_row["top_k"])
            curr_k = int(row["top_k"])
            label_specific_delta = abs(float(row["label_specific_share"]) - float(prev_row["label_specific_share"]))
            cross_delta = abs(float(row["cross_family_shared_share"]) - float(prev_row["cross_family_shared_share"]))
            generalized_delta = abs(float(row["generalized_share"]) - float(prev_row["generalized_share"]))
            is_stable = label_specific_delta < 0.05 and cross_delta < 0.03 and generalized_delta < 0.03
            stable[(prev_k, curr_k)] = bool(is_stable)
            transition_rows.append(
                {
                    "metric": "polysemanticity",
                    "from_k": prev_k,
                    "to_k": curr_k,
                    "label_specific_share_delta": label_specific_delta,
                    "cross_family_share_delta": cross_delta,
                    "generalized_share_delta": generalized_delta,
                    "transition_stable": bool(is_stable),
                }
            )
        prev_row = row
    recommended_k = _first_two_consecutive_stable_k(k_values, stable)
    summary = {
        "metric": "polysemanticity",
        "converged": recommended_k is not None,
        "recommended_upper_k": recommended_k,
        "criterion": "two consecutive K transitions with label-specific share delta <5pp and cross/generalized share deltas <3pp",
        "non_converged_labels": "",
        "label_recommended_k": "",
    }
    return summary, pd.DataFrame(transition_rows)


def summarize_convergence(
    fragmentation: pd.DataFrame,
    overlap: pd.DataFrame,
    polysemanticity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frag_summary, frag_transitions = _summarize_fragmentation_convergence(fragmentation)
    overlap_summary, overlap_transitions, overlap_aggregate = _summarize_overlap_convergence(overlap)
    poly_summary, poly_transitions = _summarize_polysemanticity_convergence(polysemanticity)
    summary = pd.DataFrame([frag_summary, overlap_summary, poly_summary])
    transitions = pd.concat([frag_transitions, overlap_transitions, poly_transitions], ignore_index=True, sort=False)
    return summary, transitions, overlap_aggregate


def _fragmentation_aggregate(fragmentation: pd.DataFrame) -> pd.DataFrame:
    return (
        fragmentation.groupby("top_k", as_index=False)
        .agg(
            mean_full_auc=("full_auc_mean", "mean"),
            mean_k80_effect=("k80_effect", "mean"),
            mean_k90_effect=("k90_effect", "mean"),
            mean_effect_equivalent_latents=("effect_equivalent_latents", "mean"),
            mean_top1_effect_share=("top1_effect_share", "mean"),
            mean_top5_effect_share=("top5_effect_share", "mean"),
        )
        .sort_values("top_k")
    )


def write_report(
    path: Path,
    *,
    config: StructuralKConvergenceConfig,
    summary: pd.DataFrame,
    fragmentation: pd.DataFrame,
    overlap_aggregate: pd.DataFrame,
    polysemanticity: pd.DataFrame,
) -> None:
    frag_agg = _fragmentation_aggregate(fragmentation)
    lines: list[str] = [
        "# Top-K 上限收敛实验报告",
        "",
        "## 1. 为什么需要这个实验",
        "",
        "固定 Top20 和 Top100 容易被质疑为人为硬阈值。本实验把 K 改成一组递增上限，观察 Fragmentation、Overlap/Jaccard 和 Polysemanticity 是否在某个 K 后稳定。",
        "",
        "如果指标曲线进入稳定区间，就可以把当前 K 解释为落在收敛后的工作窗口，而不是任意选择。",
        "",
        "## 2. 实验设置",
        "",
        f"- K-grid: `{', '.join(str(k) for k in config.k_grid)}`",
        f"- 叶子标签: `{', '.join(config.labels)}`",
        f"- TopK 排序口径: `{_normalize_ranking_metric(config.ranking_metric)}`。",
        f"- Fragmentation: TopK `{_normalize_ranking_metric(config.ranking_metric)}` 候选池 + logistic probe + validation ablation。",
        "- Overlap/Jaccard: 每个标签 TopK latent 集合之间的 raw/weighted Jaccard。",
        "- Polysemanticity: TopK union 中每个 latent 关联的标签数量和角色分类。",
        "",
        "## 3. 收敛判定",
        "",
        "| 指标 | 是否收敛 | 推荐上限K | 判据 | 备注 |",
        "|---|---:|---:|---|---|",
    ]
    for _, row in summary.iterrows():
        note = row.get("non_converged_labels", "")
        if not note:
            note = "-"
        rec = "NA" if pd.isna(row.get("recommended_upper_k")) else str(int(row["recommended_upper_k"]))
        lines.append(
            f"| {row['metric']} | {bool(row['converged'])} | {rec} | {row['criterion']} | {note} |"
        )

    lines.extend(
        [
            "",
            "## 4. Fragmentation 随 K 的变化",
            "",
            "| K | Mean full AUC | Mean K80 | Mean K90 | Mean effect-equivalent latents | Mean Top1 share | Mean Top5 share |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in frag_agg.iterrows():
        lines.append(
            f"| {int(row['top_k'])} | {_fmt(row['mean_full_auc'])} | {_fmt(row['mean_k80_effect'])} | "
            f"{_fmt(row['mean_k90_effect'])} | {_fmt(row['mean_effect_equivalent_latents'])} | "
            f"{_fmt(row['mean_top1_effect_share'])} | {_fmt(row['mean_top5_effect_share'])} |"
        )

    lines.extend(
        [
            "",
            "## 5. Overlap/Jaccard 随 K 的变化",
            "",
            "| K | Mean leaf Jaccard | Same-family Jaccard | Cross-family Jaccard | Family contrast | Mean weighted Jaccard |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in overlap_aggregate.iterrows():
        lines.append(
            f"| {int(row['top_k'])} | {_fmt(row['mean_leaf_jaccard'])} | "
            f"{_fmt(row['mean_same_family_jaccard'])} | {_fmt(row['mean_cross_family_jaccard'])} | "
            f"{_fmt(row['family_contrast_jaccard'])} | {_fmt(row['mean_weighted_jaccard'])} |"
        )

    lines.extend(
        [
            "",
            "## 6. Polysemanticity 随 K 的变化",
            "",
            "| K | Unique latents | Label-specific share | Same-family share | Cross-family share | Generalized share | Mean labels/latent |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in polysemanticity.sort_values("top_k").iterrows():
        lines.append(
            f"| {int(row['top_k'])} | {int(row['unique_latents'])} | "
            f"{_fmt(row['label_specific_share'])} | {_fmt(row['same_family_shared_share'])} | "
            f"{_fmt(row['cross_family_shared_share'])} | {_fmt(row['generalized_share'])} | "
            f"{_fmt(row['mean_labels_per_latent'])} |"
        )

    lines.extend(
        [
            "",
            "## 7. 对 Top20 / Top100 的判断",
            "",
            "| 问题 | 回答 |",
            "|---|---|",
            "| 为什么不能只用 Top20/Top100 硬阈值？ | 因为不同指标对 K 的敏感性不同。固定 K 只能作为工作窗口，必须用 K-grid 证明结论是否稳定。 |",
            "| 每个 K 下三个指标如何变化？ | 上面三张表分别给出 Fragmentation、Overlap/Jaccard、Polysemanticity 的 K 曲线。 |",
            "| 哪个 K 之后结果基本收敛？ | 以收敛判定表为准；若推荐上限K为 NA，则说明该指标在当前 K-grid 下未满足严格收敛。 |",
            "| 当前 Top20/Top100 是否落在稳定区间内？ | Top20 主要用于人工可审查窗口；只有当 Overlap/Polysemanticity 收敛判据支持时，才可写成稳定结构窗口。Top100 主要用于 Fragmentation 候选召回；只有当 Fragmentation 在 Top75/Top100 附近收敛时，才可写成收敛上限。 |",
            "",
            "实际判读：",
            "",
            "- Polysemanticity 若已收敛，说明 latent 角色分布可以较早稳定，Top20 可作为人工审查窗口的辅助依据。",
            "- Overlap/Jaccard 若未收敛，不能把 Top20 写成正式收敛点，只能写成固定可审查预算，并在正文或附录报告 K-sensitivity。",
            "- Fragmentation 若存在未收敛标签，不能把 Top100 写成全局充分上限；应写成 downstream-effect 候选召回预算，并标记未收敛标签。",
            "",
            "## 8. 输出文件",
            "",
            "- `k_convergence_summary.csv`",
            "- `fragmentation_by_k.csv`",
            "- `overlap_by_k.csv`",
            "- `polysemanticity_by_k.csv`",
            "- `k_convergence_transitions.csv`",
            "- `figures/fragmentation_convergence.png`",
            "- `figures/overlap_convergence.png`",
            "- `figures/polysemanticity_convergence.png`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(
    output_dir: Path,
    *,
    fragmentation: pd.DataFrame,
    overlap_aggregate: pd.DataFrame,
    polysemanticity: pd.DataFrame,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label, group in fragmentation.groupby("label"):
        group = group.sort_values("top_k")
        ax.plot(group["top_k"], group["effect_equivalent_latents"], marker="o", label=label)
    ax.axvline(100, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("TopK candidate upper bound")
    ax.set_ylabel("effect-equivalent latents")
    ax.set_title("Fragmentation convergence")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "fragmentation_convergence.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    overlap_aggregate = overlap_aggregate.sort_values("top_k")
    ax.plot(overlap_aggregate["top_k"], overlap_aggregate["mean_same_family_jaccard"], marker="o", label="same-family")
    ax.plot(overlap_aggregate["top_k"], overlap_aggregate["mean_cross_family_jaccard"], marker="o", label="cross-family")
    ax.plot(overlap_aggregate["top_k"], overlap_aggregate["family_contrast_jaccard"], marker="o", label="family contrast")
    ax.axvline(20, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("TopK candidate upper bound")
    ax.set_ylabel("Jaccard")
    ax.set_title("Overlap convergence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "overlap_convergence.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    polysemanticity = polysemanticity.sort_values("top_k")
    ax.plot(polysemanticity["top_k"], polysemanticity["label_specific_share"], marker="o", label="label-specific")
    ax.plot(polysemanticity["top_k"], polysemanticity["cross_family_shared_share"], marker="o", label="cross-family")
    ax.plot(polysemanticity["top_k"], polysemanticity["generalized_share"], marker="o", label="generalized")
    ax.axvline(20, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("TopK candidate upper bound")
    ax.set_ylabel("share")
    ax.set_title("Polysemanticity convergence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "polysemanticity_convergence.png", dpi=220)
    plt.close(fig)


def validate_outputs(
    *,
    config: StructuralKConvergenceConfig,
    fragmentation: pd.DataFrame,
    overlap: pd.DataFrame,
    polysemanticity: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    expected_k = set(config.k_grid)
    if set(fragmentation["top_k"].astype(int).unique()) != expected_k:
        raise AssertionError("Fragmentation output is missing K values")
    if set(overlap["top_k"].astype(int).unique()) != expected_k:
        raise AssertionError("Overlap output is missing K values")
    if set(polysemanticity["top_k"].astype(int).unique()) != expected_k:
        raise AssertionError("Polysemanticity output is missing K values")
    if not (fragmentation["k90_effect"] >= fragmentation["k80_effect"]).all():
        raise AssertionError("Fragmentation must satisfy K90 >= K80")
    if not overlap["jaccard"].between(0.0, 1.0).all():
        raise AssertionError("Overlap Jaccard outside [0, 1]")
    if not overlap["weighted_jaccard"].between(0.0, 1.0).all():
        raise AssertionError("Weighted Jaccard outside [0, 1]")
    role_count_cols = [
        "label_specific_count",
        "same_family_shared_count",
        "same_supplemental_block_count",
        "cross_family_shared_count",
        "generalized_count",
    ]
    role_counts = polysemanticity[role_count_cols].sum(axis=1)
    if not (role_counts == polysemanticity["unique_latents"]).all():
        raise AssertionError("Polysemanticity role counts must sum to unique_latents")
    required_summary = {"fragmentation", "overlap_jaccard", "polysemanticity"}
    if set(summary["metric"]) != required_summary:
        raise AssertionError("Convergence summary is missing metrics")
    high_k = max(config.k_grid)
    for label in config.labels:
        count = int(fragmentation[(fragmentation["label"] == label) & (fragmentation["top_k"] == high_k)]["candidate_pool_size"].iloc[0])
        if count != high_k:
            raise AssertionError(f"{label} does not have {high_k} candidates at max K")


def run_structural_k_convergence(
    *,
    association_matrix: str | Path,
    feature_store: str | Path,
    label_matrix: str | Path,
    output_dir: str | Path,
    config: StructuralKConvergenceConfig = StructuralKConvergenceConfig(),
    make_figures: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    features = load_feature_store(feature_store)
    labels_df = pd.read_csv(label_matrix)
    association = _prepare_association_matrix(association_matrix, config.labels)

    fragmentation, fold_metrics, latent_effects = compute_fragmentation_by_k(
        association,
        features,
        labels_df,
        config,
    )
    overlap = compute_overlap_by_k(association, config)
    polysemanticity, poly_latents = compute_polysemanticity_by_k(association, config)
    summary, transitions, overlap_aggregate = summarize_convergence(
        fragmentation,
        overlap,
        polysemanticity,
    )
    validate_outputs(
        config=config,
        fragmentation=fragmentation,
        overlap=overlap,
        polysemanticity=polysemanticity,
        summary=summary,
    )

    fragmentation.to_csv(output_path / "fragmentation_by_k.csv", index=False)
    fold_metrics.to_csv(output_path / "fragmentation_fold_metrics_by_k.csv", index=False)
    latent_effects.to_csv(output_path / "fragmentation_latent_effects_by_k.csv", index=False)
    overlap.to_csv(output_path / "overlap_by_k.csv", index=False)
    overlap_aggregate.to_csv(output_path / "overlap_aggregate_by_k.csv", index=False)
    polysemanticity.to_csv(output_path / "polysemanticity_by_k.csv", index=False)
    poly_latents.to_csv(output_path / "polysemanticity_latents_by_k.csv", index=False)
    summary.to_csv(output_path / "k_convergence_summary.csv", index=False)
    transitions.to_csv(output_path / "k_convergence_transitions.csv", index=False)
    payload = {
        "analysis_version": "structural_k_convergence_v1",
        "config": asdict(config),
        "input_paths": {
            "association_matrix": str(association_matrix),
            "feature_store": str(feature_store),
            "label_matrix": str(label_matrix),
        },
        "output_dir": str(output_path),
        "summary": summary.to_dict("records"),
    }
    (output_path / "k_convergence_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    write_report(
        output_path / "k_convergence_report_zh.md",
        config=config,
        summary=summary,
        fragmentation=fragmentation,
        overlap_aggregate=overlap_aggregate,
        polysemanticity=polysemanticity,
    )
    if make_figures:
        plot_outputs(
            output_path,
            fragmentation=fragmentation,
            overlap_aggregate=overlap_aggregate,
            polysemanticity=polysemanticity,
        )
    return {
        "output_dir": output_path,
        "fragmentation": fragmentation,
        "overlap": overlap,
        "overlap_aggregate": overlap_aggregate,
        "polysemanticity": polysemanticity,
        "summary": summary,
        "transitions": transitions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Top-K convergence analysis for MISC SAE structural metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--association-matrix",
        default="outputs/misc_full_sae_eval/interpretability/latent_space_search_v2/latent_label_association_v2.csv",
    )
    parser.add_argument(
        "--feature-store",
        default="outputs/misc_full_sae_eval/feature_store/utterance_features.pt",
    )
    parser.add_argument(
        "--label-matrix",
        default="outputs/misc_full_sae_eval/label_matrix.csv",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--k-grid", nargs="+", type=int, default=list(DEFAULT_K_GRID))
    parser.add_argument(
        "--ranking-metric",
        default="directional_auc",
        choices=["directional_auc", "auc", "abs_cohens_d", "cohens_d", "d", "formal_edge_weight"],
        help="Metric used to rank per-label TopK candidate latents.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--min-auc", type=float, default=0.70)
    parser.add_argument("--effect-epsilon", type=float, default=0.001)
    parser.add_argument("--precision-k", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    k_grid = tuple(sorted({int(k) for k in args.k_grid if int(k) > 0}))
    config = StructuralKConvergenceConfig(
        labels=tuple(label.upper() for label in args.labels),
        k_grid=k_grid,
        ranking_metric=_normalize_ranking_metric(args.ranking_metric),
        cv_folds=args.cv_folds,
        min_auc=args.min_auc,
        effect_epsilon=args.effect_epsilon,
        precision_k=args.precision_k,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )
    result = run_structural_k_convergence(
        association_matrix=args.association_matrix,
        feature_store=args.feature_store,
        label_matrix=args.label_matrix,
        output_dir=args.output_dir,
        config=config,
        make_figures=not args.no_figures,
    )
    print("Completed structural K convergence analysis.")
    print(f"Output dir: {result['output_dir']}")
    print(result["summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
