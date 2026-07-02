"""Minimal SAE latent subset reaching a ratio of full-SAE probe AUC.

This analysis is a predictive candidate-discovery pass.  For each MISC label it
uses the filtered TopK latent pool, trains a full-SAE probe on each CV fold, and
then searches for the smallest latent subset whose refit logistic probe reaches
a configured fraction of the full-SAE fold AUC.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .full_representation_probe import (
    FullRepresentationProbeConfig,
    _make_splits,
)
from .minimal_sufficient_subspace_v2 import (
    DEFAULT_LABELS,
    LEAF_LABELS,
    PARENT_LABELS,
    MinimalSufficientSubspaceConfig,
    _activation_overlap_redundancy,
    _aggregate_curves,
    _auc_columns,
    _build_candidate_pool,
    _json_default,
    _load_feature_filter_audit,
    _pairwise_jaccard,
    _prepare_association_matrix,
    _redundancy_audit,
    _score_binary,
    _validate_association_in_keep_pool,
    _write_csv,
    load_feature_store,
)


@dataclass(frozen=True)
class MinimalFullSaeRatioSubspaceConfig:
    labels: tuple[str, ...] = DEFAULT_LABELS
    leaf_labels: tuple[str, ...] = LEAF_LABELS
    candidate_top_k: int = 100
    target_ratio: float = 0.95
    cv_folds: int = 5
    split_policy: str = "stratified-group-kfold"
    group_column: str = "file_id"
    min_full_auc: float = 0.70
    max_search_k: int = 100
    precision_k: int = 50
    precision_k_values: tuple[int, ...] = (50, 100)
    stability_jaccard_threshold: float = 0.40
    redundant_auc_epsilon: float = 0.005
    redundant_auprc_epsilon: float = 0.005
    activation_threshold: float = 0.0
    C: float = 1.0
    solver: str = "liblinear"
    random_state: int = 42
    max_iter: int = 1000
    standardize: bool = True


def _candidate_config(config: MinimalFullSaeRatioSubspaceConfig) -> MinimalSufficientSubspaceConfig:
    return MinimalSufficientSubspaceConfig(
        labels=config.labels,
        leaf_labels=config.leaf_labels,
        candidate_policy="filtered_topk_only",
        candidate_top_k=config.candidate_top_k,
        cv_folds=config.cv_folds,
        min_auc=config.min_full_auc,
        precision_k=config.precision_k,
        precision_k_values=config.precision_k_values,
        max_search_k=config.max_search_k,
        stability_jaccard_threshold=config.stability_jaccard_threshold,
        redundant_auc_epsilon=config.redundant_auc_epsilon,
        redundant_auprc_epsilon=config.redundant_auprc_epsilon,
        activation_threshold=config.activation_threshold,
        random_state=config.random_state,
        max_iter=config.max_iter,
    )


def _probe_metrics(
    *,
    features: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    latent_ids: list[int] | None,
    config: MinimalFullSaeRatioSubspaceConfig,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, float, np.ndarray]:
    if latent_ids is None:
        x_train = np.ascontiguousarray(features[train_idx], dtype=np.float32)
        x_test = np.ascontiguousarray(features[test_idx], dtype=np.float32)
    else:
        x_train = np.ascontiguousarray(features[train_idx][:, latent_ids], dtype=np.float32)
        x_test = np.ascontiguousarray(features[test_idx][:, latent_ids], dtype=np.float32)
    if config.standardize:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train).astype(np.float32)
        x_test = scaler.transform(x_test).astype(np.float32)
    clf = LogisticRegression(
        C=config.C,
        solver=config.solver,
        class_weight="balanced",
        random_state=config.random_state,
        max_iter=config.max_iter,
    )
    clf.fit(x_train, y[train_idx])
    intercept = float(clf.intercept_[0])
    coef = clf.coef_.reshape(-1).astype(np.float64)
    logits = intercept + x_test @ coef
    metrics = _score_binary(
        y[test_idx],
        logits,
        prevalence=float(y[test_idx].mean()),
        precision_k_values=config.precision_k_values,
    )
    return metrics, x_train, x_test, intercept, coef


def _empty_label_result(
    label: str,
    y: np.ndarray,
    candidate_pool: pd.DataFrame,
) -> dict[str, Any]:
    role = "parent_consistency_only" if label in PARENT_LABELS else "leaf_or_atomic"
    summary = {
        "label": label,
        "label_role": role,
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "prevalence": float(y.mean()) if len(y) else 0.0,
        "candidate_pool_size": int(candidate_pool.shape[0]),
        "fold_formal_status": "insufficient_data_or_candidates",
        "global_status": "global_ratio_not_found",
        "formal_status": "insufficient_data_or_candidates",
        "full_sae_auc_mean": np.nan,
        "target_auc_mean": np.nan,
        "global_selected_auc_mean": np.nan,
        "global_target_auc_mean": np.nan,
        "global_auc_margin_mean": np.nan,
        "global_auc_margin_min": np.nan,
        "global_n_folds_meet_target": 0,
        "minimal_k_mean": np.nan,
        "minimal_k_median": np.nan,
        "n_sufficient_folds": 0,
        "mean_pairwise_jaccard_between_folds": 0.0,
        "stability_status": "not_recoverable",
        "final_selected_k": 0,
        "final_selected_latents": "",
        "fragmentation_class": "not_recoverable",
    }
    return {
        "summary": summary,
        "candidates": candidate_pool.assign(label=label),
        "folds": pd.DataFrame(),
        "steps": pd.DataFrame(),
        "global_curves": pd.DataFrame(),
        "selected": pd.DataFrame(),
        "redundancy": pd.DataFrame(),
    }


def _classify(
    label: str,
    status: str,
    median_k: float,
    stability_jaccard: float,
    config: MinimalFullSaeRatioSubspaceConfig,
) -> tuple[str, str]:
    if status in {"full_sae_not_recoverable", "insufficient_data_or_candidates"}:
        return "not_recoverable", "not_recoverable"
    stability = "stable" if stability_jaccard >= config.stability_jaccard_threshold else "unstable"
    if label in PARENT_LABELS:
        return "parent_consistency_only", stability
    if not math.isfinite(median_k) or median_k <= 0:
        return "not_recoverable", stability
    if median_k <= 3 and stability == "stable":
        return "compact", stability
    if median_k <= 8:
        return "moderate", stability
    return "distributed", stability


def _search_fold(
    *,
    label: str,
    fold: int,
    features: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    candidate_ids: list[int],
    config: MinimalFullSaeRatioSubspaceConfig,
    full_sae_auc: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if full_sae_auc is None:
        full_metrics, _full_train, _full_test, _full_intercept, _full_coef = _probe_metrics(
            features=features,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            latent_ids=None,
            config=config,
        )
    else:
        full_metrics = {"auc": float(full_sae_auc)}
    target_auc = config.target_ratio * full_metrics["auc"]

    cand_metrics, _cand_train, cand_test, cand_intercept, cand_coef = _probe_metrics(
        features=features,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
        latent_ids=candidate_ids,
        config=config,
    )

    selected_positions: list[int] = []
    remaining = list(range(len(candidate_ids)))
    current_logits = np.full(len(test_idx), cand_intercept, dtype=np.float64)
    max_steps = min(config.max_search_k, len(candidate_ids))
    rows: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None

    for step in range(1, max_steps + 1):
        if not remaining:
            break
        rem = np.asarray(remaining, dtype=int)
        candidate_logits = current_logits[:, None] + cand_test[:, rem] * cand_coef[rem]
        aucs = _auc_columns(y[test_idx], candidate_logits)
        best_local = int(np.nanargmax(aucs))
        best_pos = int(rem[best_local])
        selected_positions.append(best_pos)
        remaining.remove(best_pos)
        current_logits = current_logits + cand_test[:, best_pos] * cand_coef[best_pos]
        selected_ids = [candidate_ids[pos] for pos in selected_positions]
        refit_metrics, _x_train, _x_test, _intercept, _coef = _probe_metrics(
            features=features,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            latent_ids=selected_ids,
            config=config,
        )
        partial_metrics = _score_binary(
            y[test_idx],
            current_logits,
            prevalence=float(y[test_idx].mean()),
            precision_k_values=config.precision_k_values,
        )
        meets_target = (
            full_metrics["auc"] >= config.min_full_auc
            and refit_metrics["auc"] >= target_auc
        )
        row = {
            "label": label,
            "fold": int(fold),
            "step_k": int(step),
            "latent_added": int(candidate_ids[best_pos]),
            "selected_latents": ",".join(str(x) for x in selected_ids),
            "n_selected": int(len(selected_ids)),
            "full_sae_auc": full_metrics["auc"],
            "target_ratio": config.target_ratio,
            "target_auc": target_auc,
            "full_candidate_auc": cand_metrics["auc"],
            "partial_auc": partial_metrics["auc"],
            "selected_auc": refit_metrics["auc"],
            "selected_average_precision": refit_metrics["average_precision"],
            "selected_balanced_accuracy": refit_metrics["balanced_accuracy"],
            "selected_f1": refit_metrics["f1"],
            f"selected_precision_at_{config.precision_k}": refit_metrics[
                f"precision_at_{config.precision_k}"
            ],
            f"selected_precision_lift_at_{config.precision_k}": refit_metrics[
                f"precision_lift_at_{config.precision_k}"
            ],
            "meets_target": bool(meets_target),
        }
        rows.append(row)
        if meets_target and chosen is None:
            chosen = row.copy()
            break

    steps = pd.DataFrame(rows)
    if full_metrics["auc"] < config.min_full_auc:
        status = "full_sae_not_recoverable"
        if chosen is None and not steps.empty:
            chosen = steps.loc[steps["selected_auc"].idxmax()].to_dict()
    elif chosen is None:
        status = "not_sufficient_within_search_budget"
        if not steps.empty:
            chosen = steps.loc[steps["selected_auc"].idxmax()].to_dict()
        else:
            chosen = {}
    else:
        status = "minimal_ratio_found"

    selected = {
        "label": label,
        "fold": int(fold),
        "fold_status": status,
        "selected_k": int(chosen.get("step_k", 0) or 0),
        "selected_latents": str(chosen.get("selected_latents", "")),
        "full_sae_auc": float(full_metrics["auc"]),
        "target_auc": float(target_auc),
        "full_candidate_auc": float(cand_metrics["auc"]),
        "selected_auc": float(chosen.get("selected_auc", np.nan)),
        "selected_average_precision": float(chosen.get("selected_average_precision", np.nan)),
        "selected_balanced_accuracy": float(chosen.get("selected_balanced_accuracy", np.nan)),
        "selected_f1": float(chosen.get("selected_f1", np.nan)),
        "selected_meets_target": bool(status == "minimal_ratio_found"),
    }
    return steps, selected


def _evaluate_global_prefixes(
    *,
    label: str,
    features: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    fold_full_auc: dict[int, float],
    candidate_order: list[int],
    config: MinimalFullSaeRatioSubspaceConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    max_steps = min(config.max_search_k, len(candidate_order))
    for k in range(1, max_steps + 1):
        latent_ids = candidate_order[:k]
        fold_aucs: list[float] = []
        fold_targets: list[float] = []
        fold_margins: list[float] = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            metrics, _x_train, _x_test, _intercept, _coef = _probe_metrics(
                features=features,
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                latent_ids=latent_ids,
                config=config,
            )
            target_auc = config.target_ratio * float(fold_full_auc[fold_idx])
            fold_aucs.append(metrics["auc"])
            fold_targets.append(target_auc)
            fold_margins.append(metrics["auc"] - target_auc)
        selected_auc_mean = float(np.mean(fold_aucs))
        target_auc_mean = float(np.mean(fold_targets))
        rows.append(
            {
                "label": label,
                "step_k": int(k),
                "selected_latents": ",".join(str(x) for x in latent_ids),
                "selected_auc_mean": selected_auc_mean,
                "target_auc_mean": target_auc_mean,
                "auc_margin_mean": selected_auc_mean - target_auc_mean,
                "auc_margin_min": float(np.min(fold_margins)),
                "n_folds_meet_target": int(np.sum(np.asarray(fold_margins) >= 0.0)),
                "n_folds": int(len(splits)),
                "meets_mean_target": bool(selected_auc_mean >= target_auc_mean),
            }
        )
        if selected_auc_mean >= target_auc_mean:
            break
    return pd.DataFrame(rows)


def analyze_label(
    *,
    features: np.ndarray,
    label_df: pd.DataFrame,
    association: pd.DataFrame,
    label: str,
    config: MinimalFullSaeRatioSubspaceConfig,
    full_sae_fold_auc: dict[tuple[str, int], float] | None = None,
) -> dict[str, Any]:
    label = label.upper()
    y = pd.to_numeric(label_df[label], errors="coerce").fillna(0).astype(int).to_numpy()
    candidate_pool = _build_candidate_pool(association, label, features.shape[1], _candidate_config(config))
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos < 2 or n_neg < 2 or candidate_pool.empty:
        return _empty_label_result(label, y, candidate_pool)

    split_config = FullRepresentationProbeConfig(
        labels=config.labels,
        folds=config.cv_folds,
        split_policy=config.split_policy,
        group_column=config.group_column,
        C=config.C,
        solver=config.solver,
        random_state=config.random_state,
        max_iter=config.max_iter,
        standardize=config.standardize,
        verbose=False,
    )
    splits, split_policy_used, split_warnings = _make_splits(y, label_df, split_config)
    if not splits:
        return _empty_label_result(label, y, candidate_pool)

    candidate_ids = candidate_pool["latent_idx"].astype(int).tolist()
    fold_rows: list[dict[str, Any]] = []
    step_frames: list[pd.DataFrame] = []
    selected_sets: list[set[int]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        steps, selected = _search_fold(
            label=label,
            fold=fold_idx,
            features=features,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            candidate_ids=candidate_ids,
            config=config,
            full_sae_auc=(
                full_sae_fold_auc.get((label, fold_idx))
                if full_sae_fold_auc is not None and (label, fold_idx) in full_sae_fold_auc
                else None
            ),
        )
        step_frames.append(steps)
        selected_ids = [int(x) for x in str(selected.get("selected_latents", "")).split(",") if x]
        if selected["fold_status"] == "minimal_ratio_found":
            selected_sets.append(set(selected_ids))
        fold_rows.append(
            {
                "label": label,
                "fold": int(fold_idx),
                "fold_status": selected["fold_status"],
                "split_policy": split_policy_used,
                "split_warnings": "; ".join(split_warnings),
                "candidate_pool_size": int(len(candidate_ids)),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "test_prevalence": float(y[test_idx].mean()),
                **selected,
            }
        )

    folds = pd.DataFrame(fold_rows)
    steps_all = pd.concat(step_frames, ignore_index=True) if step_frames else pd.DataFrame()
    sufficient = folds[folds["fold_status"] == "minimal_ratio_found"].copy()
    if float(folds["full_sae_auc"].mean()) < config.min_full_auc:
        fold_formal_status = "full_sae_not_recoverable"
    elif len(sufficient) >= math.ceil(len(splits) / 2):
        fold_formal_status = "minimal_ratio_found"
    else:
        fold_formal_status = "not_sufficient_within_search_budget"
        selected_sets = [
            {int(x) for x in str(row).split(",") if str(x).strip()}
            for row in folds["selected_latents"].tolist()
        ]

    denominator = max(1, len(sufficient) if fold_formal_status == "minimal_ratio_found" else len(folds))
    selection_counts: dict[int, int] = {}
    selection_steps: dict[int, list[int]] = {}
    for _, row in folds.iterrows():
        if fold_formal_status == "minimal_ratio_found" and row["fold_status"] != "minimal_ratio_found":
            continue
        ids = [int(x) for x in str(row["selected_latents"]).split(",") if str(x).strip()]
        for pos, lid in enumerate(ids, start=1):
            selection_counts[lid] = selection_counts.get(lid, 0) + 1
            selection_steps.setdefault(lid, []).append(pos)

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
    global_candidate_order = ranked_selected + [lid for lid in candidate_ids if lid not in set(ranked_selected)]
    fold_full_auc = {
        int(row["fold"]): float(row["full_sae_auc"])
        for _, row in folds.iterrows()
    }
    global_curves = _evaluate_global_prefixes(
        label=label,
        features=features,
        y=y,
        splits=splits,
        fold_full_auc=fold_full_auc,
        candidate_order=global_candidate_order,
        config=config,
    )
    global_hits = global_curves[global_curves["meets_mean_target"].astype(bool)].copy()
    if not global_hits.empty:
        global_selected = global_hits.iloc[0].to_dict()
        global_status = "global_ratio_found"
    elif not global_curves.empty:
        global_selected = global_curves.loc[global_curves["auc_margin_mean"].idxmax()].to_dict()
        global_status = "global_ratio_not_found"
    else:
        global_selected = {}
        global_status = "global_ratio_not_found"
    final_selected_ids = [
        int(x)
        for x in str(global_selected.get("selected_latents", "")).split(",")
        if str(x).strip()
    ]
    formal_status = (
        "minimal_ratio_found"
        if fold_formal_status == "minimal_ratio_found" and global_status == "global_ratio_found"
        else fold_formal_status
        if fold_formal_status != "minimal_ratio_found"
        else "not_sufficient_within_search_budget"
    )
    stability_jaccard = _pairwise_jaccard(selected_sets)
    fragmentation_class, stability_status = _classify(
        label,
        formal_status,
        median_k,
        stability_jaccard,
        config,
    )
    audit_config = _candidate_config(config)
    redundancy = _redundancy_audit(features, y, label, final_selected_ids, audit_config)
    n_redundant = int((redundancy["redundancy_role"] == "redundant").sum()) if not redundancy.empty else 0
    activation_overlap = _activation_overlap_redundancy(
        features,
        y,
        final_selected_ids,
        config.activation_threshold,
    )

    selected_rows: list[dict[str, Any]] = []
    for lid in candidate_ids:
        cand = candidate_pool[candidate_pool["latent_idx"] == lid].iloc[0].to_dict()
        selected_rows.append(
            {
                "label": label,
                "latent_idx": int(lid),
                "final_selected": bool(lid in final_selected_ids),
                "selection_count": int(selection_counts.get(lid, 0)),
                "selection_frequency": float(selection_counts.get(lid, 0) / denominator),
                "mean_selected_step": float(np.mean(selection_steps[lid])) if lid in selection_steps else np.nan,
                "candidate_order": int(cand.get("candidate_order", 0)),
                "candidate_source": cand.get("candidate_source", ""),
                "edge_type": cand.get("edge_type", ""),
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
        "fold_formal_status": fold_formal_status,
        "global_status": global_status,
        "formal_status": formal_status,
        "full_sae_auc_mean": float(folds["full_sae_auc"].mean()),
        "full_sae_auc_std": float(folds["full_sae_auc"].std(ddof=0)),
        "target_ratio": float(config.target_ratio),
        "target_auc_mean": float(folds["target_auc"].mean()),
        "full_candidate_auc_mean": float(folds["full_candidate_auc"].mean()),
        "selected_auc_mean": float(sufficient["selected_auc"].mean()) if not sufficient.empty else np.nan,
        "global_selected_auc_mean": float(global_selected.get("selected_auc_mean", np.nan)),
        "global_target_auc_mean": float(global_selected.get("target_auc_mean", np.nan)),
        "global_auc_margin_mean": float(global_selected.get("auc_margin_mean", np.nan)),
        "global_auc_margin_min": float(global_selected.get("auc_margin_min", np.nan)),
        "global_n_folds_meet_target": int(global_selected.get("n_folds_meet_target", 0) or 0),
        "minimal_k_mean": float(np.mean(minimal_k_values)) if minimal_k_values else np.nan,
        "minimal_k_median": median_k,
        "minimal_k_std": float(np.std(minimal_k_values, ddof=0)) if minimal_k_values else np.nan,
        "n_sufficient_folds": int(len(sufficient)),
        "mean_pairwise_jaccard_between_folds": stability_jaccard,
        "stability_status": stability_status,
        "final_selected_k": int(len(final_selected_ids)),
        "final_selected_latents": ",".join(str(x) for x in final_selected_ids),
        "predictive_redundancy_ratio": float(n_redundant / len(final_selected_ids)) if final_selected_ids else 0.0,
        **activation_overlap,
        "fragmentation_class": fragmentation_class,
    }
    return {
        "summary": summary,
        "candidates": candidate_pool.assign(label=label),
        "folds": folds,
        "steps": steps_all,
        "global_curves": global_curves,
        "selected": selected_df,
        "redundancy": redundancy,
    }


def _write_report(output_dir: Path, summary: pd.DataFrame, config: MinimalFullSaeRatioSubspaceConfig) -> None:
    leaf = summary[summary["label_role"] != "parent_consistency_only"].copy()
    lines = [
        "# MISC 95% Full SAE minimal latent set report",
        "",
        "This report finds the smallest filtered-Top100 SAE latent subset whose refit",
        f"logistic probe reaches `{config.target_ratio:.0%}` of the same-fold full-SAE AUC.",
        "It is predictive probe sufficiency, not causal sufficiency.",
        "",
        "## Configuration",
        "",
        f"- Target ratio: `{config.target_ratio:.2f}`.",
        f"- Candidate pool: filtered Top{config.candidate_top_k}.",
        f"- CV: `{config.split_policy}`, folds={config.cv_folds}, group column=`{config.group_column}`.",
        f"- Full-SAE recoverable threshold: AUC >= `{config.min_full_auc:.2f}`.",
        "",
        "## Label summary",
        "",
        "| Label | Role | Status | Class | Full SAE AUC | Target AUC | Fold median K | Final K | Final CV AUC | Final margin | Final selected latents |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in summary.iterrows():
        min_k = "-" if pd.isna(row.get("minimal_k_median")) else f"{float(row['minimal_k_median']):.1f}"
        lines.append(
            f"| {row['label']} | {row['label_role']} | {row['formal_status']} | {row['fragmentation_class']} | "
            f"{float(row.get('full_sae_auc_mean', 0.0)):.3f} | "
            f"{float(row.get('target_auc_mean', 0.0)):.3f} | {min_k} | "
            f"{int(row.get('final_selected_k', 0))} | "
            f"{float(row.get('global_selected_auc_mean', 0.0)):.3f} | "
            f"{float(row.get('global_auc_margin_mean', 0.0)):.3f} | "
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
            f"- Not recoverable under this target: {', '.join(not_rec) if not_rec else 'none'}.",
            "- `Fold median K` summarizes fold-specific discovery sets; `Final K` is the globally refit CV-validated set size.",
            "- Parent labels `RE` and `QU` are consistency rows, not the main leaf-label claim.",
        ]
    )
    (output_dir / "minimal_full_sae_ratio_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def validate_outputs(
    summary: pd.DataFrame,
    folds: pd.DataFrame,
    selected: pd.DataFrame,
    output_dir: Path,
    config: MinimalFullSaeRatioSubspaceConfig,
) -> None:
    labels = {label.upper() for label in config.labels}
    if set(summary["label"]) != labels:
        raise AssertionError(f"Expected labels {labels}, got {set(summary['label'])}")
    for col in ("full_sae_auc", "target_auc", "selected_auc"):
        if col in folds.columns:
            values = pd.to_numeric(folds[col], errors="coerce").dropna()
            if not values.between(0.0, 1.0).all():
                raise AssertionError(f"{col} out of range")
    found = folds[folds["fold_status"] == "minimal_ratio_found"].copy()
    if not found.empty:
        if not (found["selected_auc"] + 1e-12 >= found["target_auc"]).all():
            raise AssertionError("A minimal_ratio_found fold is below target_auc")
    for _, row in summary.iterrows():
        latents = [x for x in str(row.get("final_selected_latents", "")).split(",") if x.strip()]
        if int(row.get("final_selected_k", 0)) != len(latents):
            raise AssertionError(f"final_selected_k mismatch for {row['label']}")
        if row.get("formal_status") == "minimal_ratio_found":
            if float(row.get("global_selected_auc_mean", 0.0)) + 1e-12 < float(row.get("global_target_auc_mean", 1.0)):
                raise AssertionError(f"Global final set is below target for {row['label']}")
    if not selected.empty:
        by_label = selected.groupby("label")["candidate_order"].max()
        if (by_label > config.candidate_top_k).any():
            raise AssertionError("Candidate order exceeds filtered TopK limit")
    required = [
        "minimal_full_sae_ratio_summary.csv",
        "minimal_full_sae_ratio_fold_results.csv",
        "minimal_full_sae_ratio_selection_steps.csv",
        "minimal_full_sae_ratio_global_curves.csv",
        "minimal_full_sae_ratio_selected_latents.csv",
        "minimal_full_sae_ratio_redundancy_audit.csv",
        "minimal_full_sae_ratio_curves.csv",
        "candidate_pool.csv",
        "minimal_full_sae_ratio_report.md",
        "manifest.json",
    ]
    for name in required:
        if not (output_dir / name).exists():
            raise AssertionError(f"Missing output: {name}")


def _load_full_sae_fold_auc(path: str | Path, labels: Iterable[str]) -> dict[tuple[str, int], float]:
    rows = pd.read_csv(path)
    required = {"representation", "label", "fold", "probe_auc"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Full probe fold rows are missing required columns: {missing}")
    label_set = {label.upper() for label in labels}
    subset = rows[
        rows["representation"].astype(str).eq("full_sae_latents")
        & rows["label"].astype(str).str.upper().isin(label_set)
    ].copy()
    if subset.empty:
        raise ValueError(f"No full_sae_latents fold rows found in {path}")
    subset["label"] = subset["label"].astype(str).str.upper()
    subset["fold"] = pd.to_numeric(subset["fold"], errors="coerce").astype(int)
    subset["probe_auc"] = pd.to_numeric(subset["probe_auc"], errors="coerce")
    if subset["probe_auc"].isna().any():
        raise ValueError(f"Full probe fold rows contain non-numeric probe_auc values: {path}")
    if subset.duplicated(["label", "fold"]).any():
        dupes = subset[subset.duplicated(["label", "fold"], keep=False)][["label", "fold"]]
        raise ValueError(f"Duplicate full_sae_latents label/fold rows: {dupes.to_dict('records')}")
    return {
        (str(row["label"]), int(row["fold"])): float(row["probe_auc"])
        for _, row in subset.iterrows()
    }


def run_minimal_full_sae_ratio_subspace(
    *,
    association_matrix: str | Path,
    feature_store: str | Path,
    label_matrix: str | Path,
    output_dir: str | Path,
    feature_filter_audit: str | Path | None = None,
    full_probe_fold_rows: str | Path | None = None,
    config: MinimalFullSaeRatioSubspaceConfig | None = None,
) -> dict[str, Any]:
    config = config or MinimalFullSaeRatioSubspaceConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    features = load_feature_store(feature_store)
    label_df = pd.read_csv(label_matrix)
    if len(label_df) != features.shape[0]:
        raise ValueError(f"Label rows ({len(label_df)}) do not match feature rows ({features.shape[0]})")
    association = _prepare_association_matrix(pd.read_csv(association_matrix), config.labels)
    keep_pool_audit: dict[str, Any] | None = None
    if feature_filter_audit is not None:
        keep_pool_audit = _validate_association_in_keep_pool(
            association,
            _load_feature_filter_audit(feature_filter_audit),
        )
    full_sae_fold_auc = (
        _load_full_sae_fold_auc(full_probe_fold_rows, config.labels)
        if full_probe_fold_rows is not None
        else None
    )

    results = [
        analyze_label(
            features=features,
            label_df=label_df,
            association=association,
            label=label,
            config=config,
            full_sae_fold_auc=full_sae_fold_auc,
        )
        for label in config.labels
    ]
    summary = pd.DataFrame([result["summary"] for result in results])
    candidates = pd.concat([result["candidates"] for result in results], ignore_index=True, sort=False)
    folds = pd.concat([result["folds"] for result in results if not result["folds"].empty], ignore_index=True, sort=False)
    steps = pd.concat([result["steps"] for result in results if not result["steps"].empty], ignore_index=True, sort=False)
    global_curves = pd.concat(
        [result["global_curves"] for result in results if not result["global_curves"].empty],
        ignore_index=True,
        sort=False,
    )
    selected = pd.concat([result["selected"] for result in results if not result["selected"].empty], ignore_index=True, sort=False)
    redundancy = pd.concat([result["redundancy"] for result in results if not result["redundancy"].empty], ignore_index=True, sort=False)
    if steps.empty:
        curves = pd.DataFrame()
    else:
        curve_steps = steps.rename(
            columns={
                "selected_auc": "auc",
                "selected_average_precision": "average_precision",
                "selected_f1": "f1",
                f"selected_precision_lift_at_{config.precision_k}": f"precision_lift_at_{config.precision_k}",
            }
        )
        curves = _aggregate_curves(curve_steps)

    files = {
        "minimal_full_sae_ratio_summary": output_path / "minimal_full_sae_ratio_summary.csv",
        "minimal_full_sae_ratio_fold_results": output_path / "minimal_full_sae_ratio_fold_results.csv",
        "minimal_full_sae_ratio_selection_steps": output_path / "minimal_full_sae_ratio_selection_steps.csv",
        "minimal_full_sae_ratio_global_curves": output_path / "minimal_full_sae_ratio_global_curves.csv",
        "minimal_full_sae_ratio_selected_latents": output_path / "minimal_full_sae_ratio_selected_latents.csv",
        "minimal_full_sae_ratio_redundancy_audit": output_path / "minimal_full_sae_ratio_redundancy_audit.csv",
        "minimal_full_sae_ratio_curves": output_path / "minimal_full_sae_ratio_curves.csv",
        "candidate_pool": output_path / "candidate_pool.csv",
    }
    _write_csv(summary, files["minimal_full_sae_ratio_summary"])
    _write_csv(folds, files["minimal_full_sae_ratio_fold_results"])
    _write_csv(steps, files["minimal_full_sae_ratio_selection_steps"])
    _write_csv(global_curves, files["minimal_full_sae_ratio_global_curves"])
    _write_csv(selected, files["minimal_full_sae_ratio_selected_latents"])
    _write_csv(redundancy, files["minimal_full_sae_ratio_redundancy_audit"])
    _write_csv(curves, files["minimal_full_sae_ratio_curves"])
    _write_csv(candidates, files["candidate_pool"])
    _write_report(output_path, summary, config)

    payload = {
        "analysis_version": "minimal_full_sae_ratio_subspace_v1",
        "config": asdict(config),
        "inputs": {
            "association_matrix": str(association_matrix),
            "feature_store": str(feature_store),
            "label_matrix": str(label_matrix),
            "feature_filter_audit": str(feature_filter_audit) if feature_filter_audit else None,
            "full_probe_fold_rows": str(full_probe_fold_rows) if full_probe_fold_rows else None,
        },
        "full_sae_baseline_source": "fold_rows_csv" if full_probe_fold_rows else "recomputed",
        "candidate_policy": "filtered_topk_only",
        "keep_pool_audit": keep_pool_audit,
        "n_labels": int(summary.shape[0]),
        "formal_status_counts": summary["formal_status"].value_counts().to_dict(),
        "fragmentation_class_counts": summary["fragmentation_class"].value_counts().to_dict(),
        "files": {key: str(value) for key, value in files.items()},
        "report": str(output_path / "minimal_full_sae_ratio_report.md"),
    }
    (output_path / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    validate_outputs(summary, folds, selected, output_path, config)
    return {
        "summary": summary,
        "candidates": candidates,
        "folds": folds,
        "steps": steps,
        "global_curves": global_curves,
        "selected": selected,
        "redundancy": redundancy,
        "curves": curves,
        "output_dir": output_path,
        "manifest": payload,
    }


__all__ = [
    "MinimalFullSaeRatioSubspaceConfig",
    "run_minimal_full_sae_ratio_subspace",
]
