"""Minimal sufficient SAE latent subspace search for MISC labels.

This module estimates the smallest predictive SAE latent subspace for each
MISC behavior label.  It treats TopK/Top100 as a candidate retrieval budget, not
as the conclusion, and selects the smallest fold-level subset that approaches
the full-candidate probe.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


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
LEAF_LABELS: tuple[str, ...] = ("RES", "REC", "QUO", "QUC", "GI", "SU", "AF")
PARENT_LABELS: tuple[str, ...] = ("RE", "QU")


@dataclass(frozen=True)
class MinimalSufficientSubspaceConfig:
    labels: tuple[str, ...] = DEFAULT_LABELS
    leaf_labels: tuple[str, ...] = LEAF_LABELS
    candidate_top_k: int = 100
    cv_folds: int = 5
    min_auc: float = 0.70
    auc_tolerance: float = 0.02
    auprc_tolerance: float = 0.03
    precision_lift_tolerance: float = 0.05
    precision_k: int = 50
    precision_k_values: tuple[int, ...] = (50, 100)
    max_search_k: int = 100
    stability_jaccard_threshold: float = 0.40
    redundant_auc_epsilon: float = 0.005
    redundant_auprc_epsilon: float = 0.005
    activation_threshold: float = 0.0
    random_state: int = 13
    max_iter: int = 1000


def load_feature_store(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    if path.suffix == ".npz":
        payload = np.load(path)
        for key in ("utterance_features", "features", "feature_matrix", "X", "arr_0"):
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


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return str(value)


def _prepare_association_matrix(matrix: pd.DataFrame, labels: Iterable[str]) -> pd.DataFrame:
    out = matrix.copy()
    label_set = {label.upper() for label in labels}
    out["label"] = out["label"].astype(str).str.upper()
    out = out[out["label"].isin(label_set)].copy()
    for col in (
        "latent_idx",
        "association_rank",
        "directional_auc",
        "abs_cohens_d",
        "cohens_d",
        "prevalence",
        "precision_at_50",
        "precision_lift_absolute_at_50",
        "formal_edge_weight",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["latent_idx"] = out["latent_idx"].astype(int)
    if "association_rank" not in out.columns:
        parts: list[pd.DataFrame] = []
        for label, group in out.groupby("label"):
            ranked = group.sort_values(
                ["abs_cohens_d", "directional_auc", "latent_idx"],
                ascending=[False, False, True],
            ).copy()
            ranked["association_rank"] = np.arange(1, len(ranked) + 1)
            parts.append(ranked)
        out = pd.concat(parts, ignore_index=True) if parts else out
    for col in ("stable_edge", "positive_support", "negative_boundary", "significant_fdr"):
        out[col] = _bool_series(out[col]) if col in out.columns else False
    if "edge_type" not in out.columns:
        out["edge_type"] = np.select(
            [out["positive_support"], out["negative_boundary"], out["stable_edge"]],
            ["positive_support", "negative_boundary", "stable_edge"],
            default="weak_backup",
        )
    return out[out["latent_idx"] >= 0].copy()


def _build_candidate_pool(
    association: pd.DataFrame,
    label: str,
    feature_dim: int,
    config: MinimalSufficientSubspaceConfig,
) -> pd.DataFrame:
    label = label.upper()
    group = association[association["label"] == label].copy()
    if group.empty:
        return group
    group = group[(group["latent_idx"] >= 0) & (group["latent_idx"] < feature_dim)].copy()
    stable = group[group["stable_edge"]].copy()
    backup = group[group["association_rank"] <= config.candidate_top_k].copy()
    pool = pd.concat([stable, backup], ignore_index=True, sort=False)
    pool = pool.sort_values(
        ["stable_edge", "association_rank", "abs_cohens_d", "directional_auc"],
        ascending=[False, True, False, False],
    ).drop_duplicates("latent_idx", keep="first")
    pool["candidate_source"] = np.where(pool["stable_edge"], "legacy_seed", "top100_backup")
    pool["candidate_order"] = np.arange(1, len(pool) + 1)
    return pool.reset_index(drop=True)


def _auc_columns(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    y_bool = np.asarray(y, dtype=bool)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    n = scores.shape[0]
    n_pos = int(y_bool.sum())
    n_neg = int(n - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.full(scores.shape[1], 0.5, dtype=np.float64)
    order = np.argsort(scores, axis=0, kind="mergesort")
    ranks = np.empty(order.shape, dtype=np.float64)
    ranks[order, np.arange(scores.shape[1])] = np.arange(1, n + 1, dtype=np.float64)[:, None]
    pos_rank_sum = ranks[y_bool, :].sum(axis=0)
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc.astype(np.float64)


def _safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    return float(_auc_columns(y, np.asarray(scores, dtype=np.float64))[0])


def _precision_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    if len(scores) == 0:
        return 0.0
    k = min(max(int(k), 1), len(scores))
    idx = np.argsort(scores)[::-1][:k]
    return float(np.asarray(y)[idx].mean()) if k else 0.0


def _score_binary(
    y: np.ndarray,
    logits: np.ndarray,
    *,
    prevalence: float,
    precision_k_values: Iterable[int],
) -> dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    preds = (probs >= 0.5).astype(int)
    out = {
        "auc": _safe_auc(y, probs),
        "average_precision": float(average_precision_score(y, probs)) if len(np.unique(y)) > 1 else prevalence,
        "accuracy": float(accuracy_score(y, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
    }
    for k in precision_k_values:
        p_at = _precision_at_k(y, probs, k)
        out[f"precision_at_{k}"] = p_at
        out[f"precision_lift_at_{k}"] = p_at - prevalence
    return out


def _fit_full_probe(
    features: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    candidate_ids: list[int],
    config: MinimalSufficientSubspaceConfig,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, dict[str, float]]:
    x_train = np.ascontiguousarray(features[train_idx][:, candidate_ids], dtype=np.float32)
    x_val = np.ascontiguousarray(features[val_idx][:, candidate_ids], dtype=np.float32)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        random_state=config.random_state,
        max_iter=config.max_iter,
    )
    clf.fit(x_train, y[train_idx])
    intercept = float(clf.intercept_[0])
    coef = clf.coef_.reshape(-1).astype(np.float64)
    prevalence = float(y[val_idx].mean())
    full_logits = intercept + x_val @ coef
    full_metrics = _score_binary(
        y[val_idx],
        full_logits,
        prevalence=prevalence,
        precision_k_values=config.precision_k_values,
    )
    return x_train, x_val, intercept, coef, full_metrics


def _greedy_additive_selection(
    *,
    label: str,
    fold: int,
    y_val: np.ndarray,
    x_val: np.ndarray,
    intercept: float,
    coef: np.ndarray,
    candidate_ids: list[int],
    full_metrics: dict[str, float],
    config: MinimalSufficientSubspaceConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prevalence = float(y_val.mean())
    selected_positions: list[int] = []
    remaining = list(range(len(candidate_ids)))
    current_logits = np.full(len(y_val), intercept, dtype=np.float64)
    max_steps = min(config.max_search_k, len(candidate_ids))
    rows: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None

    for step in range(1, max_steps + 1):
        if not remaining:
            break
        rem = np.asarray(remaining, dtype=int)
        scores = current_logits[:, None] + x_val[:, rem] * coef[rem]
        aucs = _auc_columns(y_val, scores)
        best_local = int(np.nanargmax(aucs))
        best_pos = int(rem[best_local])
        selected_positions.append(best_pos)
        remaining.remove(best_pos)
        current_logits = current_logits + x_val[:, best_pos] * coef[best_pos]
        metrics = _score_binary(
            y_val,
            current_logits,
            prevalence=prevalence,
            precision_k_values=config.precision_k_values,
        )
        selected_ids = [candidate_ids[pos] for pos in selected_positions]
        sufficient = (
            full_metrics["auc"] >= config.min_auc
            and metrics["auc"] >= config.min_auc
            and metrics["auc"] >= full_metrics["auc"] - config.auc_tolerance
            and metrics["average_precision"]
            >= full_metrics["average_precision"] - config.auprc_tolerance
            and metrics[f"precision_lift_at_{config.precision_k}"]
            >= full_metrics[f"precision_lift_at_{config.precision_k}"] - config.precision_lift_tolerance
        )
        row = {
            "label": label,
            "fold": fold,
            "step_k": step,
            "latent_added": candidate_ids[best_pos],
            "selected_latents": ",".join(str(x) for x in selected_ids),
            "n_selected": len(selected_ids),
            "full_auc": full_metrics["auc"],
            "full_average_precision": full_metrics["average_precision"],
            f"full_precision_lift_at_{config.precision_k}": full_metrics[
                f"precision_lift_at_{config.precision_k}"
            ],
            "auc": metrics["auc"],
            "average_precision": metrics["average_precision"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "f1": metrics["f1"],
            f"precision_at_{config.precision_k}": metrics[f"precision_at_{config.precision_k}"],
            f"precision_lift_at_{config.precision_k}": metrics[f"precision_lift_at_{config.precision_k}"],
            "is_sufficient": sufficient,
        }
        rows.append(row)
        if sufficient and chosen is None:
            chosen = row.copy()

    steps = pd.DataFrame(rows)
    if full_metrics["auc"] < config.min_auc:
        status = "not_recoverable_under_candidate_pool"
        if chosen is None and not steps.empty:
            best_idx = steps["auc"].idxmax()
            chosen = steps.loc[best_idx].to_dict()
    elif chosen is None:
        status = "not_sufficient_within_search_budget"
        best_idx = steps["auc"].idxmax() if not steps.empty else None
        chosen = steps.loc[best_idx].to_dict() if best_idx is not None else {}
    else:
        status = "minimal_sufficient_found"
    selected = {
        "label": label,
        "fold": fold,
        "fold_status": status,
        "selected_k": int(chosen.get("step_k", 0) or 0),
        "selected_latents": str(chosen.get("selected_latents", "")),
        "selected_auc": float(chosen.get("auc", float("nan"))),
        "selected_average_precision": float(chosen.get("average_precision", float("nan"))),
        f"selected_precision_lift_at_{config.precision_k}": float(
            chosen.get(f"precision_lift_at_{config.precision_k}", float("nan"))
        ),
    }
    return steps, selected


def _pairwise_jaccard(sets: list[set[int]]) -> float:
    if len(sets) <= 1:
        return 1.0 if sets else 0.0
    values = []
    for left, right in combinations(sets, 2):
        union = left | right
        values.append(len(left & right) / len(union) if union else 1.0)
    return float(np.mean(values)) if values else 0.0


def _activation_overlap_redundancy(
    features: np.ndarray,
    y: np.ndarray,
    latent_ids: list[int],
    activation_threshold: float,
) -> dict[str, float]:
    if not latent_ids or int(y.sum()) == 0:
        return {
            "activation_overlap_redundancy": 0.0,
            "positive_coverage": 0.0,
            "mean_pairwise_activation_jaccard": 0.0,
        }
    pos = y.astype(bool)
    hit_sets = [(features[:, lid] > activation_threshold) & pos for lid in latent_ids]
    union = np.logical_or.reduce(hit_sets)
    hit_sum = sum(int(h.sum()) for h in hit_sets)
    redundancy = 1.0 - int(union.sum()) / hit_sum if hit_sum else 0.0
    pairwise = []
    for i, j in combinations(range(len(hit_sets)), 2):
        pair_union = hit_sets[i] | hit_sets[j]
        denom = int(pair_union.sum())
        if denom:
            pairwise.append(float((hit_sets[i] & hit_sets[j]).sum()) / denom)
    return {
        "activation_overlap_redundancy": float(max(0.0, redundancy)),
        "positive_coverage": float(union.sum() / pos.sum()) if pos.sum() else 0.0,
        "mean_pairwise_activation_jaccard": float(np.mean(pairwise)) if pairwise else 0.0,
    }


def _redundancy_audit(
    features: np.ndarray,
    y: np.ndarray,
    label: str,
    latent_ids: list[int],
    config: MinimalSufficientSubspaceConfig,
) -> pd.DataFrame:
    if not latent_ids or len(np.unique(y)) < 2:
        return pd.DataFrame()
    scaler = StandardScaler()
    x = scaler.fit_transform(features[:, latent_ids]).astype(np.float32)
    clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        random_state=config.random_state,
        max_iter=config.max_iter,
    )
    clf.fit(x, y)
    intercept = float(clf.intercept_[0])
    coef = clf.coef_.reshape(-1).astype(np.float64)
    prevalence = float(y.mean())
    full_logits = intercept + x @ coef
    full = _score_binary(y, full_logits, prevalence=prevalence, precision_k_values=config.precision_k_values)
    rows: list[dict[str, Any]] = []
    for pos, lid in enumerate(latent_ids):
        logits = full_logits - x[:, pos] * coef[pos]
        metrics = _score_binary(y, logits, prevalence=prevalence, precision_k_values=config.precision_k_values)
        delta_auc = full["auc"] - metrics["auc"]
        delta_ap = full["average_precision"] - metrics["average_precision"]
        if delta_auc <= -config.redundant_auc_epsilon or delta_ap <= -config.redundant_auprc_epsilon:
            role = "conflicting_or_harmful"
        elif delta_auc < config.redundant_auc_epsilon and delta_ap < config.redundant_auprc_epsilon:
            role = "redundant"
        else:
            role = "essential"
        rows.append(
            {
                "label": label,
                "latent_idx": int(lid),
                "full_selected_auc": full["auc"],
                "loo_auc": metrics["auc"],
                "delta_auc": delta_auc,
                "full_selected_average_precision": full["average_precision"],
                "loo_average_precision": metrics["average_precision"],
                "delta_average_precision": delta_ap,
                "redundancy_role": role,
            }
        )
    return pd.DataFrame(rows)


def _classify_label(
    label: str,
    status: str,
    minimal_k_median: float,
    stability_jaccard: float,
    config: MinimalSufficientSubspaceConfig,
) -> tuple[str, str]:
    if status == "not_recoverable_under_candidate_pool":
        return "not_recoverable", "not_recoverable"
    if not math.isfinite(minimal_k_median) or minimal_k_median <= 0:
        return "distributed", "unstable"
    stability = "stable" if stability_jaccard >= config.stability_jaccard_threshold else "unstable"
    if label in PARENT_LABELS:
        return "parent_consistency_only", stability
    if minimal_k_median <= 3 and stability == "stable":
        return "compact", stability
    if minimal_k_median <= 8:
        return "moderate", stability
    return "distributed", stability


def analyze_label(
    *,
    features: np.ndarray,
    label_df: pd.DataFrame,
    association: pd.DataFrame,
    label: str,
    config: MinimalSufficientSubspaceConfig,
) -> dict[str, Any]:
    label = label.upper()
    y = pd.to_numeric(label_df[label], errors="coerce").fillna(0).astype(int).to_numpy()
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    candidate_pool = _build_candidate_pool(association, label, features.shape[1], config)
    if n_pos < 2 or n_neg < 2 or candidate_pool.empty:
        status = "insufficient_data_or_candidates"
        summary = {
            "label": label,
            "label_role": "parent_consistency_only" if label in PARENT_LABELS else "leaf_or_atomic",
            "n_samples": len(y),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "prevalence": float(y.mean()) if len(y) else 0.0,
            "candidate_pool_size": int(candidate_pool.shape[0]),
            "n_stable_candidates": int(candidate_pool["stable_edge"].sum()) if not candidate_pool.empty else 0,
            "formal_status": status,
            "fragmentation_class": "not_recoverable",
            "stability_status": "not_recoverable",
        }
        return {
            "summary": summary,
            "candidates": candidate_pool.assign(label=label),
            "folds": pd.DataFrame(),
            "steps": pd.DataFrame(),
            "selected": pd.DataFrame(),
            "redundancy": pd.DataFrame(),
        }

    candidate_ids = candidate_pool["latent_idx"].astype(int).tolist()
    n_splits = min(config.cv_folds, n_pos, n_neg)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
    fold_rows: list[dict[str, Any]] = []
    step_frames: list[pd.DataFrame] = []
    selected_sets: list[set[int]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
        _x_train, x_val, intercept, coef, full_metrics = _fit_full_probe(
            features,
            y,
            train_idx,
            val_idx,
            candidate_ids,
            config,
        )
        steps, selected = _greedy_additive_selection(
            label=label,
            fold=fold_idx,
            y_val=y[val_idx],
            x_val=x_val,
            intercept=intercept,
            coef=coef,
            candidate_ids=candidate_ids,
            full_metrics=full_metrics,
            config=config,
        )
        step_frames.append(steps)
        selected_ids = [
            int(x)
            for x in str(selected.get("selected_latents", "")).split(",")
            if str(x).strip()
        ]
        if selected["fold_status"] == "minimal_sufficient_found":
            selected_sets.append(set(selected_ids))
        fold_rows.append(
            {
                "label": label,
                "fold": fold_idx,
                "fold_status": selected["fold_status"],
                "candidate_pool_size": len(candidate_ids),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "val_prevalence": float(y[val_idx].mean()),
                "full_auc": full_metrics["auc"],
                "full_average_precision": full_metrics["average_precision"],
                "full_balanced_accuracy": full_metrics["balanced_accuracy"],
                "full_f1": full_metrics["f1"],
                f"full_precision_lift_at_{config.precision_k}": full_metrics[
                    f"precision_lift_at_{config.precision_k}"
                ],
                **selected,
            }
        )

    folds = pd.DataFrame(fold_rows)
    steps_all = pd.concat(step_frames, ignore_index=True) if step_frames else pd.DataFrame()
    sufficient = folds[folds["fold_status"] == "minimal_sufficient_found"].copy()
    full_recoverable = float(folds["full_auc"].mean()) >= config.min_auc
    if not full_recoverable:
        formal_status = "not_recoverable_under_candidate_pool"
    elif len(sufficient) >= math.ceil(n_splits / 2):
        formal_status = "minimal_sufficient_found"
    else:
        formal_status = "not_sufficient_within_search_budget"
        selected_sets = [
            {
                int(x)
                for x in str(row).split(",")
                if str(x).strip()
            }
            for row in folds["selected_latents"].tolist()
        ]

    selection_counts: dict[int, int] = {}
    selection_steps: dict[int, list[int]] = {}
    for _, row in folds.iterrows():
        if formal_status == "minimal_sufficient_found" and row["fold_status"] != "minimal_sufficient_found":
            continue
        ids = [int(x) for x in str(row["selected_latents"]).split(",") if str(x).strip()]
        for pos, lid in enumerate(ids, start=1):
            selection_counts[lid] = selection_counts.get(lid, 0) + 1
            selection_steps.setdefault(lid, []).append(pos)

    denominator = max(1, len(sufficient) if formal_status == "minimal_sufficient_found" else len(folds))
    minimal_k_values = sufficient["selected_k"].astype(float).tolist() if not sufficient.empty else []
    median_k = float(np.median(minimal_k_values)) if minimal_k_values else float("nan")
    final_k = int(math.ceil(median_k)) if math.isfinite(median_k) else 0
    ranked_selected = sorted(
        selection_counts,
        key=lambda lid: (
            -selection_counts[lid] / denominator,
            np.mean(selection_steps.get(lid, [999])),
            int(candidate_pool[candidate_pool["latent_idx"] == lid]["candidate_order"].iloc[0])
            if bool((candidate_pool["latent_idx"] == lid).any())
            else 999,
        ),
    )
    final_selected_ids = ranked_selected[:final_k] if final_k > 0 else []
    stability_jaccard = _pairwise_jaccard(selected_sets)
    redundancy = _redundancy_audit(features, y, label, final_selected_ids, config)
    n_redundant = int((redundancy["redundancy_role"] == "redundant").sum()) if not redundancy.empty else 0
    predictive_redundancy = float(n_redundant / len(final_selected_ids)) if final_selected_ids else 0.0
    activation_overlap = _activation_overlap_redundancy(
        features,
        y,
        final_selected_ids,
        config.activation_threshold,
    )
    fragmentation_class, stability_status = _classify_label(
        label,
        formal_status,
        median_k,
        stability_jaccard,
        config,
    )

    selected_rows: list[dict[str, Any]] = []
    for lid in candidate_ids:
        cand = candidate_pool[candidate_pool["latent_idx"] == lid].iloc[0].to_dict()
        freq = selection_counts.get(lid, 0) / denominator
        selected_rows.append(
            {
                "label": label,
                "latent_idx": int(lid),
                "final_selected": lid in final_selected_ids,
                "selection_count": selection_counts.get(lid, 0),
                "selection_frequency": freq,
                "mean_selected_step": float(np.mean(selection_steps[lid])) if lid in selection_steps else np.nan,
                "candidate_order": int(cand.get("candidate_order", 0)),
                "candidate_source": cand.get("candidate_source", ""),
                "edge_type": cand.get("edge_type", ""),
                "stable_edge": bool(cand.get("stable_edge", False)),
                "positive_support": bool(cand.get("positive_support", False)),
                "negative_boundary": bool(cand.get("negative_boundary", False)),
                "association_rank": int(cand.get("association_rank", 0)),
                "directional_auc": float(cand.get("directional_auc", 0.0)),
                "abs_cohens_d": float(cand.get("abs_cohens_d", 0.0)),
                "formal_edge_weight": float(cand.get("formal_edge_weight", 0.0)),
            }
        )
    selected_df = pd.DataFrame(selected_rows)

    summary = {
        "label": label,
        "label_role": "parent_consistency_only" if label in PARENT_LABELS else "leaf_or_atomic",
        "n_samples": int(len(y)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "prevalence": float(y.mean()),
        "candidate_pool_size": int(len(candidate_ids)),
        "n_stable_candidates": int(candidate_pool["stable_edge"].sum()),
        "n_positive_support_candidates": int(candidate_pool["positive_support"].sum()),
        "n_negative_boundary_candidates": int(candidate_pool["negative_boundary"].sum()),
        "formal_status": formal_status,
        "full_auc_mean": float(folds["full_auc"].mean()),
        "full_auc_std": float(folds["full_auc"].std(ddof=0)),
        "full_average_precision_mean": float(folds["full_average_precision"].mean()),
        "full_f1_mean": float(folds["full_f1"].mean()),
        f"full_precision_lift_at_{config.precision_k}_mean": float(
            folds[f"full_precision_lift_at_{config.precision_k}"].mean()
        ),
        "minimal_k_mean": float(np.mean(minimal_k_values)) if minimal_k_values else np.nan,
        "minimal_k_median": median_k,
        "minimal_k_std": float(np.std(minimal_k_values, ddof=0)) if minimal_k_values else np.nan,
        "n_sufficient_folds": int(len(sufficient)),
        "mean_pairwise_jaccard_between_folds": stability_jaccard,
        "stability_status": stability_status,
        "final_selected_k": int(len(final_selected_ids)),
        "final_selected_latents": ",".join(str(x) for x in final_selected_ids),
        "predictive_redundancy_ratio": predictive_redundancy,
        **activation_overlap,
        "fragmentation_class": fragmentation_class,
    }
    return {
        "summary": summary,
        "candidates": candidate_pool.assign(label=label),
        "folds": folds,
        "steps": steps_all,
        "selected": selected_df,
        "redundancy": redundancy,
    }


def _aggregate_curves(steps: pd.DataFrame) -> pd.DataFrame:
    if steps.empty:
        return pd.DataFrame()
    rows = []
    for (label, k), group in steps.groupby(["label", "step_k"]):
        rows.append(
            {
                "label": label,
                "step_k": int(k),
                "auc_mean": float(group["auc"].mean()),
                "auc_std": float(group["auc"].std(ddof=0)),
                "average_precision_mean": float(group["average_precision"].mean()),
                "f1_mean": float(group["f1"].mean()),
                "precision_lift_at_50_mean": float(group.get("precision_lift_at_50", pd.Series(dtype=float)).mean()),
                "n_folds": int(group["fold"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.write_text("", encoding="utf-8")
    else:
        df.to_csv(path, index=False)


def _write_figures(
    output_dir: Path,
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    redundancy: pd.DataFrame,
    curves: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    leaf = summary[summary["label_role"] != "parent_consistency_only"].copy()
    if not leaf.empty:
        ordered = leaf.sort_values(["fragmentation_class", "minimal_k_median"], na_position="last")
        fig, ax = plt.subplots(figsize=(8, 4))
        values = ordered["minimal_k_median"].fillna(0.0)
        ax.bar(ordered["label"], values, color="#4477aa")
        ax.set_ylabel("Median minimal K")
        ax.set_title("Minimal sufficient latent count by label")
        fig.tight_layout()
        fig.savefig(fig_dir / "minimal_k_by_label.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(leaf["full_auc_mean"], leaf["minimal_k_median"].fillna(0.0), c="#228833")
        for _, row in leaf.iterrows():
            ax.text(row["full_auc_mean"], 0 if pd.isna(row["minimal_k_median"]) else row["minimal_k_median"], row["label"])
        ax.set_xlabel("Full-candidate CV AUC")
        ax.set_ylabel("Median minimal K")
        ax.set_title("Full probe strength vs minimal subspace size")
        fig.tight_layout()
        fig.savefig(fig_dir / "full_vs_minimal_auc.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(leaf["label"], leaf["predictive_redundancy_ratio"], color="#cc6677", label="predictive")
        ax.plot(leaf["label"], leaf["activation_overlap_redundancy"], color="#332288", marker="o", label="activation")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Redundancy ratio")
        ax.set_title("Redundancy audit by label")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "redundancy_by_label.png", dpi=180)
        plt.close(fig)

    if not selected.empty:
        pivot = selected.pivot_table(
            index="latent_idx",
            columns="label",
            values="selection_frequency",
            aggfunc="max",
            fill_value=0.0,
        )
        top_idx = pivot.max(axis=1).sort_values(ascending=False).head(40).index
        pivot = pivot.loc[top_idx]
        fig, ax = plt.subplots(figsize=(8, max(4, 0.16 * len(pivot))))
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.astype(str))
        ax.set_title("Selection stability heatmap")
        fig.colorbar(im, ax=ax, label="Selection frequency")
        fig.tight_layout()
        fig.savefig(fig_dir / "selection_stability_heatmap.png", dpi=180)
        plt.close(fig)

    if not curves.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, group in curves.groupby("label"):
            if label in PARENT_LABELS:
                continue
            group = group.sort_values("step_k")
            ax.plot(group["step_k"], group["auc_mean"], label=label)
        ax.axhline(0.70, color="#888888", linestyle="--", linewidth=1)
        ax.set_xlabel("Selected latent count")
        ax.set_ylabel("CV AUC")
        ax.set_title("Sufficiency curves by label")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "sufficiency_curves_by_label.png", dpi=180)
        plt.close(fig)


def _write_report(output_dir: Path, summary: pd.DataFrame, config: MinimalSufficientSubspaceConfig) -> None:
    leaf = summary[summary["label_role"] != "parent_consistency_only"].copy()
    lines = [
        "# MISC minimal sufficient SAE latent subspace v2",
        "",
        "This report estimates the smallest predictive SAE latent subset for each MISC label.",
        "It is a probe-space sufficiency analysis and should not be written as causal sufficiency.",
        "",
        "## Criteria",
        "",
        f"- Candidate pool: legacy seed candidates plus association-rank Top{config.candidate_top_k} backup.",
        f"- Full-candidate recoverable if mean CV AUC >= {config.min_auc:.2f}.",
        f"- Minimal sufficient K must be within {config.auc_tolerance:.2f} AUC, {config.auprc_tolerance:.2f} AUPRC, and {config.precision_lift_tolerance:.2f} P@{config.precision_k} lift of the full-candidate probe.",
        "- Parent labels `RE` and `QU` are consistency-only rows.",
        "",
        "## Label summary",
        "",
        "| Label | Role | Status | Class | Candidate pool | Legacy seed candidates | Full AUC | Minimal K median | Stability Jaccard | Predictive redundancy | Selected latents |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in summary.iterrows():
        min_k = "-" if pd.isna(row.get("minimal_k_median")) else f"{float(row['minimal_k_median']):.1f}"
        lines.append(
            f"| {row['label']} | {row['label_role']} | {row['formal_status']} | {row['fragmentation_class']} | "
            f"{int(row['candidate_pool_size'])} | {int(row['n_stable_candidates'])} | "
            f"{float(row.get('full_auc_mean', 0.0)):.3f} | {min_k} | "
            f"{float(row.get('mean_pairwise_jaccard_between_folds', 0.0)):.3f} | "
            f"{float(row.get('predictive_redundancy_ratio', 0.0)):.3f} | "
            f"{row.get('final_selected_latents', '') or '-'} |"
        )

    compact = leaf[leaf["fragmentation_class"] == "compact"]["label"].tolist()
    moderate = leaf[leaf["fragmentation_class"] == "moderate"]["label"].tolist()
    distributed = leaf[leaf["fragmentation_class"] == "distributed"]["label"].tolist()
    not_rec = leaf[leaf["fragmentation_class"] == "not_recoverable"]["label"].tolist()
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Compact labels: {', '.join(compact) if compact else 'none'}.",
            f"- Moderate labels: {', '.join(moderate) if moderate else 'none'}.",
            f"- Distributed labels: {', '.join(distributed) if distributed else 'none'}.",
            f"- Not recoverable under the candidate pool: {', '.join(not_rec) if not_rec else 'none'}.",
            "- Selected subspaces can be used as prioritized groups for later ablation or steering, but this report itself is predictive rather than causal.",
        ]
    )
    (output_dir / "minimal_sufficient_subspace_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def validate_outputs(
    summary: pd.DataFrame,
    folds: pd.DataFrame,
    selected: pd.DataFrame,
    output_dir: Path,
    config: MinimalSufficientSubspaceConfig,
) -> None:
    labels = {label.upper() for label in config.labels}
    if set(summary["label"]) != labels:
        raise AssertionError(f"Expected labels {labels}, got {set(summary['label'])}")
    expected_leaf = {label for label in config.leaf_labels if label in labels}
    leaf = summary[summary["label"].isin(expected_leaf)]
    if leaf.shape[0] != len(expected_leaf):
        raise AssertionError("Leaf labels missing from summary")
    if not (
        summary["minimal_k_median"].dropna()
        <= summary.loc[summary["minimal_k_median"].notna(), "candidate_pool_size"]
    ).all():
        raise AssertionError("minimal_k_median exceeds candidate_pool_size")
    for col in ["full_auc", "selected_auc"]:
        if col in folds.columns:
            values = pd.to_numeric(folds[col], errors="coerce").dropna()
            if not values.between(0.0, 1.0).all():
                raise AssertionError(f"{col} out of range")
    recoverable = summary[summary["formal_status"] == "minimal_sufficient_found"]
    for _, row in recoverable.iterrows():
        if not str(row.get("final_selected_latents", "")).strip():
            raise AssertionError(f"Recoverable label {row['label']} has no selected latents")
    leaf_selected = selected[selected["label"].isin(config.leaf_labels)] if not selected.empty else selected
    if not leaf_selected.empty and any(parent in set(leaf_selected["label"]) for parent in PARENT_LABELS):
        raise AssertionError("Parent labels leaked into leaf-selected check")
    required = [
        "minimal_sufficient_summary_v2.csv",
        "minimal_sufficient_selected_latents_v2.csv",
        "minimal_sufficient_fold_results_v2.csv",
        "minimal_sufficient_selection_steps_v2.csv",
        "minimal_sufficient_redundancy_audit_v2.csv",
        "minimal_sufficient_curves_v2.csv",
        "minimal_sufficient_summary.json",
        "minimal_sufficient_subspace_report.md",
    ]
    for name in required:
        if not (output_dir / name).exists():
            raise AssertionError(f"Missing output: {name}")


def run_minimal_sufficient_subspace_v2(
    *,
    association_matrix: str | Path,
    thresholded_sets: str | Path | None,
    feature_store: str | Path,
    label_matrix: str | Path,
    output_dir: str | Path,
    config: MinimalSufficientSubspaceConfig | None = None,
    make_figures: bool = True,
) -> dict[str, Any]:
    config = config or MinimalSufficientSubspaceConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    features = load_feature_store(feature_store)
    labels_df = pd.read_csv(label_matrix)
    if len(labels_df) != features.shape[0]:
        raise ValueError(f"Label rows ({len(labels_df)}) do not match feature rows ({features.shape[0]})")
    association = _prepare_association_matrix(pd.read_csv(association_matrix), config.labels)
    if thresholded_sets and Path(thresholded_sets).exists():
        thresholded = pd.read_csv(thresholded_sets)
        if not thresholded.empty and {"label", "latent_idx"}.issubset(thresholded.columns):
            stable_pairs = {
                (str(row["label"]).upper(), int(row["latent_idx"]))
                for _, row in thresholded.iterrows()
                if int(pd.to_numeric(row["latent_idx"], errors="coerce")) >= 0
                and bool(str(row.get("stable_edge", "True")).lower() in {"true", "1", "yes"})
            }
            if stable_pairs:
                association["stable_edge"] = [
                    bool(stable) or (label, int(latent)) in stable_pairs
                    for stable, label, latent in zip(
                        association["stable_edge"], association["label"], association["latent_idx"]
                    )
                ]

    results = [
        analyze_label(
            features=features,
            label_df=labels_df,
            association=association,
            label=label,
            config=config,
        )
        for label in config.labels
    ]
    summary = pd.DataFrame([result["summary"] for result in results])
    candidates = pd.concat([result["candidates"] for result in results], ignore_index=True, sort=False)
    folds = pd.concat([result["folds"] for result in results if not result["folds"].empty], ignore_index=True, sort=False)
    steps = pd.concat([result["steps"] for result in results if not result["steps"].empty], ignore_index=True, sort=False)
    selected = pd.concat([result["selected"] for result in results if not result["selected"].empty], ignore_index=True, sort=False)
    redundancy = pd.concat([result["redundancy"] for result in results if not result["redundancy"].empty], ignore_index=True, sort=False)
    curves = _aggregate_curves(steps)

    files = {
        "minimal_sufficient_summary_v2": output_path / "minimal_sufficient_summary_v2.csv",
        "minimal_sufficient_selected_latents_v2": output_path / "minimal_sufficient_selected_latents_v2.csv",
        "minimal_sufficient_fold_results_v2": output_path / "minimal_sufficient_fold_results_v2.csv",
        "minimal_sufficient_selection_steps_v2": output_path / "minimal_sufficient_selection_steps_v2.csv",
        "minimal_sufficient_redundancy_audit_v2": output_path / "minimal_sufficient_redundancy_audit_v2.csv",
        "minimal_sufficient_curves_v2": output_path / "minimal_sufficient_curves_v2.csv",
        "candidate_pool_v2": output_path / "candidate_pool_v2.csv",
    }
    _write_csv(summary, files["minimal_sufficient_summary_v2"])
    _write_csv(selected, files["minimal_sufficient_selected_latents_v2"])
    _write_csv(folds, files["minimal_sufficient_fold_results_v2"])
    _write_csv(steps, files["minimal_sufficient_selection_steps_v2"])
    _write_csv(redundancy, files["minimal_sufficient_redundancy_audit_v2"])
    _write_csv(curves, files["minimal_sufficient_curves_v2"])
    _write_csv(candidates, files["candidate_pool_v2"])

    if make_figures:
        _write_figures(output_path, summary, selected, redundancy, curves)
    _write_report(output_path, summary, config)

    payload = {
        "analysis_version": "minimal_sufficient_subspace_v2",
        "config": asdict(config),
        "inputs": {
            "association_matrix": str(association_matrix),
            "thresholded_sets": str(thresholded_sets) if thresholded_sets else None,
            "feature_store": str(feature_store),
            "label_matrix": str(label_matrix),
        },
        "n_labels": int(summary.shape[0]),
        "fragmentation_class_counts": summary["fragmentation_class"].value_counts().to_dict(),
        "formal_status_counts": summary["formal_status"].value_counts().to_dict(),
        "files": {key: str(value) for key, value in files.items()},
        "report": str(output_path / "minimal_sufficient_subspace_report.md"),
    }
    with (output_path / "minimal_sufficient_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)

    validate_outputs(summary, folds, selected, output_path, config)
    return {
        "summary": summary,
        "candidates": candidates,
        "folds": folds,
        "steps": steps,
        "selected": selected,
        "redundancy": redundancy,
        "curves": curves,
        "output_dir": output_path,
        "json_summary": payload,
    }
