"""Minimal non-redundant SAE latent set analysis for MISC labels.

This module generalizes the earlier RE-only group-structure analysis to all
MISC behavior labels using the saved utterance-level SAE feature store.  It is
not a generation-time intervention runner; instead it gives a fast
probe-space, causal-style audit of whether a label can be represented by a
small, low-redundancy latent set.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
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
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


DEFAULT_CORE_LABELS: tuple[str, ...] = (
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
DEFAULT_LEAF_LABELS: tuple[str, ...] = ("RES", "REC", "QUO", "QUC", "GI", "SU", "AF")
PARENT_LABELS: set[str] = {"RE", "QU"}


@dataclass(frozen=True)
class MinimalLatentConfig:
    top_k: int = 20
    loo_k: int = 10
    candidate_mode: str = "topk"
    target_effect_fraction: float = 0.90
    min_auc: float = 0.70
    max_redundancy: float = 0.30
    marginal_auc_epsilon: float = 0.01
    marginal_gap_epsilon: float = 0.05
    activation_threshold: float = 0.0
    precision_k_values: tuple[int, ...] = (50, 100)
    n_bootstrap: int = 20
    random_state: int = 13
    max_iter: int = 1000


def load_feature_store(path: str | Path) -> np.ndarray:
    """Load saved utterance-level SAE features from a torch or numpy artifact."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32, copy=False)
    if path.suffix == ".npz":
        obj = np.load(path)
        for key in ("utterance_features", "features", "arr_0"):
            if key in obj:
                return np.asarray(obj[key], dtype=np.float32)
        raise KeyError(f"No feature array found in {path}")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            f"Loading {path} requires torch. Run in the project ML environment."
        ) from exc

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("utterance_features", "features", "feature_matrix"):
            if key in payload:
                payload = payload[key]
                break
    if hasattr(payload, "detach"):
        payload = payload.detach().cpu().numpy()
    return np.asarray(payload, dtype=np.float32)


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def _effective_n(abs_values: Iterable[float]) -> float:
    values = np.asarray([float(v) for v in abs_values if float(v) > 0], dtype=float)
    if values.size == 0:
        return 0.0
    weights = values / values.sum()
    entropy = -float(np.sum(weights * np.log(weights)))
    return float(math.exp(entropy))


def _prepare_topk(topk_matrix: pd.DataFrame, labels: Iterable[str], top_k: int) -> pd.DataFrame:
    df = topk_matrix.copy()
    df["label"] = df["label"].astype(str).str.upper()
    labels_upper = {label.upper() for label in labels}
    df = df[df["label"].isin(labels_upper)].copy()
    numeric_cols = [
        "topk_rank",
        "latent_idx",
        "cohens_d",
        "abs_cohens_d",
        "directional_auc",
        "precision_at_50",
        "precision_lift_at_50",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "significant_fdr" in df.columns:
        df["significant_fdr"] = _bool_series(df["significant_fdr"])
    else:
        df["significant_fdr"] = False
    df = df[df["topk_rank"] <= top_k].copy()
    df["latent_idx"] = df["latent_idx"].astype(int)
    df["support_edge"] = (
        df["significant_fdr"]
        & (df["abs_cohens_d"] >= 0.5)
        & (df["directional_auc"] >= 0.60)
        & ((df["cohens_d"] < 0) | (df["precision_lift_at_50"] >= 0.10))
    )
    df["edge_type"] = np.where(
        df["support_edge"] & (df["cohens_d"] < 0),
        "negative_boundary",
        np.where(df["support_edge"], "positive_support", "weak_topk"),
    )
    return df


def _fit_probe(
    features: np.ndarray,
    y: np.ndarray,
    latent_ids: list[int],
    *,
    random_state: int,
    max_iter: int,
) -> tuple[StandardScaler, LogisticRegression, np.ndarray, np.ndarray]:
    x = np.ascontiguousarray(features[:, latent_ids], dtype=np.float32)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x).astype(np.float32)
    clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        random_state=random_state,
        max_iter=max_iter,
    )
    clf.fit(x_scaled, y)
    coef = clf.coef_.reshape(-1).astype(np.float64)
    intercept = np.asarray(clf.intercept_, dtype=np.float64)
    return scaler, clf, x_scaled, np.r_[intercept, coef]


def _score_from_logits(y: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.5
    ap = average_precision_score(y, probs) if len(np.unique(y)) > 1 else float(y.mean())
    pos = y == 1
    neg = y == 0
    return {
        "auc": float(auc),
        "average_precision": float(ap),
        "accuracy": float(accuracy_score(y, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "logit_gap": float(logits[pos].mean() - logits[neg].mean()) if pos.any() and neg.any() else 0.0,
        "mean_logit_positive": float(logits[pos].mean()) if pos.any() else 0.0,
        "mean_logit_negative": float(logits[neg].mean()) if neg.any() else 0.0,
    }


def _prefix_logits(
    x_scaled: np.ndarray,
    intercept: float,
    coef: np.ndarray,
    positions: list[int],
    *,
    include_intercept: bool = True,
) -> np.ndarray:
    base = float(intercept) if include_intercept else 0.0
    if not positions:
        return np.full(x_scaled.shape[0], base, dtype=np.float64)
    return base + x_scaled[:, positions] @ coef[positions]


def _precision_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    if len(scores) == 0:
        return 0.0
    k = min(k, len(scores))
    order = np.argsort(scores)[::-1][:k]
    return float(y[order].mean()) if k else 0.0


def _hit_mask(values: np.ndarray, activation_threshold: float) -> np.ndarray:
    return np.asarray(values > activation_threshold, dtype=bool)


def _coverage_redundancy(
    features: np.ndarray,
    y: np.ndarray,
    latent_ids: list[int],
    *,
    activation_threshold: float,
) -> dict[str, float]:
    pos_mask = y == 1
    neg_mask = y == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if not latent_ids or n_pos == 0:
        return {
            "positive_coverage": 0.0,
            "negative_coverage": 0.0,
            "support_union_count": 0,
            "support_hit_sum": 0,
            "redundancy_ratio": 0.0,
            "mean_pairwise_jaccard": 0.0,
        }

    hit_sets = [_hit_mask(features[:, lid], activation_threshold) for lid in latent_ids]
    pos_hit_sets = [hits & pos_mask for hits in hit_sets]
    neg_hit_sets = [hits & neg_mask for hits in hit_sets]
    pos_union = np.logical_or.reduce(pos_hit_sets) if pos_hit_sets else np.zeros_like(pos_mask)
    neg_union = np.logical_or.reduce(neg_hit_sets) if neg_hit_sets else np.zeros_like(neg_mask)
    support_union = int(pos_union.sum())
    support_hit_sum = int(sum(int(h.sum()) for h in pos_hit_sets))
    redundancy = 0.0
    if support_hit_sum > 0:
        redundancy = 1.0 - support_union / support_hit_sum

    pairwise = []
    for i in range(len(pos_hit_sets)):
        for j in range(i + 1, len(pos_hit_sets)):
            union = int((pos_hit_sets[i] | pos_hit_sets[j]).sum())
            if union:
                pairwise.append(float((pos_hit_sets[i] & pos_hit_sets[j]).sum()) / union)
    return {
        "positive_coverage": float(support_union / n_pos) if n_pos else 0.0,
        "negative_coverage": float(neg_union.sum() / n_neg) if n_neg else 0.0,
        "support_union_count": support_union,
        "support_hit_sum": support_hit_sum,
        "redundancy_ratio": float(redundancy),
        "mean_pairwise_jaccard": float(np.mean(pairwise)) if pairwise else 0.0,
    }


def _rank_candidates(
    label_candidates: pd.DataFrame,
    coef: np.ndarray,
    candidate_ids: list[int],
) -> pd.DataFrame:
    ranked = label_candidates.copy().reset_index(drop=True)
    coef_by_latent = {lid: float(coef[pos]) for pos, lid in enumerate(candidate_ids)}
    ranked["probe_coef"] = ranked["latent_idx"].map(coef_by_latent).fillna(0.0)
    ranked["abs_probe_coef"] = ranked["probe_coef"].abs()
    ranked["rank_probe_coef"] = ranked["abs_probe_coef"].rank(ascending=False, method="min")
    ranked["rank_abs_d"] = ranked["abs_cohens_d"].rank(ascending=False, method="min")
    ranked["rank_auc"] = ranked["directional_auc"].rank(ascending=False, method="min")
    ranked["combined_rank_score"] = (
        ranked["rank_probe_coef"] + ranked["rank_abs_d"] + ranked["rank_auc"] + ranked["topk_rank"] / 10.0
    )
    ranked = ranked.sort_values(
        ["combined_rank_score", "abs_probe_coef", "abs_cohens_d", "directional_auc"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    ranked["analysis_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def _bootstrap_stability(
    features: np.ndarray,
    y: np.ndarray,
    candidate_ids: list[int],
    *,
    n_bootstrap: int,
    random_state: int,
    max_iter: int,
) -> pd.DataFrame:
    if n_bootstrap <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(random_state)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    counts = {lid: {"top1": 0, "top5": 0, "top10": 0, "top20": 0} for lid in candidate_ids}
    for _ in range(n_bootstrap):
        sample_idx = np.r_[
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ]
        sample_y = y[sample_idx]
        _scaler, _clf, _x_scaled, params = _fit_probe(
            features[sample_idx],
            sample_y,
            candidate_ids,
            random_state=int(rng.integers(0, 1_000_000)),
            max_iter=max_iter,
        )
        coef = params[1:]
        order = [candidate_ids[i] for i in np.argsort(np.abs(coef))[::-1]]
        for k_name, k in (("top1", 1), ("top5", 5), ("top10", 10), ("top20", 20)):
            for lid in order[: min(k, len(order))]:
                counts[lid][k_name] += 1
    rows = []
    for lid in candidate_ids:
        row = {"latent_idx": lid, "n_bootstrap": n_bootstrap}
        row.update({f"{name}_freq": counts[lid][name] / n_bootstrap for name in counts[lid]})
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_label_minimal_set(
    features: np.ndarray,
    label_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    label: str,
    config: MinimalLatentConfig,
) -> dict[str, Any]:
    label = label.upper()
    if label not in label_df.columns:
        raise KeyError(f"Label {label} not found in label matrix")
    y = label_df[label].astype(int).to_numpy()
    n_positive = int(y.sum())
    n_negative = int(len(y) - n_positive)
    if n_positive < 2 or n_negative < 2:
        raise ValueError(f"Label {label} has insufficient positives/negatives")

    label_candidates = topk_df[topk_df["label"] == label].copy()
    if config.candidate_mode == "support":
        label_candidates = label_candidates[label_candidates["support_edge"]].copy()
    label_candidates = label_candidates.sort_values("topk_rank").head(config.top_k).copy()
    if label_candidates.empty:
        raise ValueError(f"No candidates for label {label}")
    candidate_ids = label_candidates["latent_idx"].astype(int).tolist()

    _scaler, _clf, x_scaled, params = _fit_probe(
        features,
        y,
        candidate_ids,
        random_state=config.random_state,
        max_iter=config.max_iter,
    )
    intercept = float(params[0])
    coef = np.asarray(params[1:], dtype=np.float64)
    full_logits = _prefix_logits(x_scaled, intercept, coef, list(range(len(candidate_ids))))
    full_metrics = _score_from_logits(y, full_logits)
    prevalence = float(y.mean())
    full_metrics.update(
        {
            f"precision_at_{k}": _precision_at_k(y, full_logits, k)
            for k in config.precision_k_values
        }
    )
    for k in config.precision_k_values:
        full_metrics[f"precision_lift_at_{k}"] = full_metrics[f"precision_at_{k}"] - prevalence

    ranked = _rank_candidates(label_candidates, coef, candidate_ids)
    ranked_ids = ranked["latent_idx"].astype(int).tolist()
    candidate_pos = {lid: pos for pos, lid in enumerate(candidate_ids)}

    cumulative_rows: list[dict[str, Any]] = []
    prev_auc: float | None = None
    prev_gap: float | None = None
    full_auc_effect = max(full_metrics["auc"] - 0.5, 1e-12)
    for k in range(1, len(ranked_ids) + 1):
        lids = ranked_ids[:k]
        positions = [candidate_pos[lid] for lid in lids]
        logits = _prefix_logits(x_scaled, intercept, coef, positions)
        metrics = _score_from_logits(y, logits)
        coverage = _coverage_redundancy(
            features,
            y,
            lids,
            activation_threshold=config.activation_threshold,
        )
        row = {
            "label": label,
            "k": k,
            "latent_added": lids[-1],
            "latents": ",".join(str(x) for x in lids),
            "auc": metrics["auc"],
            "auc_effect_fraction": (metrics["auc"] - 0.5) / full_auc_effect,
            "average_precision": metrics["average_precision"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "f1": metrics["f1"],
            "logit_gap": metrics["logit_gap"],
            "marginal_auc_gain": metrics["auc"] - prev_auc if prev_auc is not None else metrics["auc"] - 0.5,
            "marginal_logit_gap": metrics["logit_gap"] - prev_gap if prev_gap is not None else metrics["logit_gap"],
            **coverage,
        }
        for p_at in config.precision_k_values:
            p_val = _precision_at_k(y, logits, p_at)
            row[f"precision_at_{p_at}"] = p_val
            row[f"precision_lift_at_{p_at}"] = p_val - prevalence
        cumulative_rows.append(row)
        prev_auc = metrics["auc"]
        prev_gap = metrics["logit_gap"]

    cumulative_df = pd.DataFrame(cumulative_rows)
    minimal_auc_k = None
    minimal_nonredundant_k = None
    for _, row in cumulative_df.iterrows():
        if row["auc"] >= config.min_auc and row["auc_effect_fraction"] >= config.target_effect_fraction:
            minimal_auc_k = int(row["k"])
            break
    for _, row in cumulative_df.iterrows():
        if (
            row["auc"] >= config.min_auc
            and row["auc_effect_fraction"] >= config.target_effect_fraction
            and row["redundancy_ratio"] <= config.max_redundancy
        ):
            minimal_nonredundant_k = int(row["k"])
            break

    loo_k = min(config.loo_k, len(ranked_ids))
    loo_base_ids = ranked_ids[:loo_k]
    loo_base_pos = [candidate_pos[lid] for lid in loo_base_ids]
    loo_full_logits = _prefix_logits(x_scaled, intercept, coef, loo_base_pos)
    loo_full_metrics = _score_from_logits(y, loo_full_logits)
    loo_rows: list[dict[str, Any]] = []
    for lid in loo_base_ids:
        remaining = [x for x in loo_base_ids if x != lid]
        positions = [candidate_pos[x] for x in remaining]
        logits = _prefix_logits(x_scaled, intercept, coef, positions)
        metrics = _score_from_logits(y, logits)
        delta_auc = loo_full_metrics["auc"] - metrics["auc"]
        delta_gap = loo_full_metrics["logit_gap"] - metrics["logit_gap"]
        if delta_auc >= config.marginal_auc_epsilon or delta_gap >= config.marginal_gap_epsilon:
            role = "essential"
        elif delta_auc <= -config.marginal_auc_epsilon or delta_gap <= -config.marginal_gap_epsilon:
            role = "harmful_or_conflicting"
        else:
            role = "redundant_or_weak"
        loo_rows.append(
            {
                "label": label,
                "latent_idx": lid,
                "full_k": loo_k,
                "full_auc": loo_full_metrics["auc"],
                "loo_auc": metrics["auc"],
                "delta_loo_auc": delta_auc,
                "full_logit_gap": loo_full_metrics["logit_gap"],
                "loo_logit_gap": metrics["logit_gap"],
                "delta_loo_logit_gap": delta_gap,
                "loo_role": role,
            }
        )

    individual_rows: list[dict[str, Any]] = []
    for k, lid in enumerate(loo_base_ids, start=1):
        pos = [candidate_pos[lid]]
        logits = _prefix_logits(x_scaled, intercept, coef, pos)
        metrics = _score_from_logits(y, logits)
        cumulative = cumulative_rows[k - 1]
        individual_rows.append(
            {
                "label": label,
                "k": k,
                "latent_idx": lid,
                "individual_auc": metrics["auc"],
                "individual_logit_gap": metrics["logit_gap"],
                "add_one_auc": cumulative["auc"],
                "delta_add_auc": cumulative["marginal_auc_gain"],
                "add_one_logit_gap": cumulative["logit_gap"],
                "delta_add_logit_gap": cumulative["marginal_logit_gap"],
            }
        )

    support_abs_d = ranked.loc[ranked["support_edge"], "abs_cohens_d"].tolist()
    support_effective_n = _effective_n(support_abs_d)
    stability = _bootstrap_stability(
        features,
        y,
        candidate_ids,
        n_bootstrap=config.n_bootstrap,
        random_state=config.random_state,
        max_iter=config.max_iter,
    )
    if not stability.empty:
        stability["label"] = label
        ranked = ranked.merge(stability[["latent_idx", "top1_freq", "top5_freq", "top10_freq", "top20_freq"]], on="latent_idx", how="left")
    else:
        ranked["top1_freq"] = np.nan
        ranked["top5_freq"] = np.nan
        ranked["top10_freq"] = np.nan
        ranked["top20_freq"] = np.nan

    min_row = cumulative_df[cumulative_df["k"] == minimal_nonredundant_k]
    auc_row = cumulative_df[cumulative_df["k"] == minimal_auc_k]
    if not min_row.empty:
        decision_k = int(minimal_nonredundant_k)
        decision_row = min_row.iloc[0].to_dict()
        decision_status = "minimal_nonredundant"
    elif not auc_row.empty:
        decision_k = int(minimal_auc_k)
        decision_row = auc_row.iloc[0].to_dict()
        decision_status = "minimal_auc_but_redundant"
    else:
        decision_k = None
        decision_row = cumulative_df.iloc[-1].to_dict()
        decision_status = "not_reaching_auc_threshold"

    n_essential = int(sum(row["loo_role"] == "essential" for row in loo_rows))
    n_redundant = int(sum(row["loo_role"] == "redundant_or_weak" for row in loo_rows))
    n_conflict = int(sum(row["loo_role"] == "harmful_or_conflicting" for row in loo_rows))

    is_parent_label = label in PARENT_LABELS
    compact_low_redundancy = bool(
        not is_parent_label
        and decision_k is not None
        and decision_k <= 5
        and decision_status == "minimal_nonredundant"
        and n_conflict <= max(1, math.floor(0.2 * loo_k))
    )

    summary = {
        "label": label,
        "label_role": "parent_consistency_only" if is_parent_label else "leaf_or_atomic",
        "n_samples": int(len(y)),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "prevalence": prevalence,
        "candidate_mode": config.candidate_mode,
        "n_candidates": int(len(candidate_ids)),
        "n_support_edges": int(ranked["support_edge"].sum()),
        "support_effective_n": support_effective_n,
        "full_auc": full_metrics["auc"],
        "full_average_precision": full_metrics["average_precision"],
        "full_balanced_accuracy": full_metrics["balanced_accuracy"],
        "full_f1": full_metrics["f1"],
        "full_logit_gap": full_metrics["logit_gap"],
        "minimal_auc_k": minimal_auc_k,
        "minimal_nonredundant_k": minimal_nonredundant_k,
        "selected_k": decision_k,
        "selected_status": decision_status,
        "selected_latents": decision_row.get("latents", ""),
        "selected_auc": decision_row.get("auc", float("nan")),
        "selected_auc_effect_fraction": decision_row.get("auc_effect_fraction", float("nan")),
        "selected_positive_coverage": decision_row.get("positive_coverage", float("nan")),
        "selected_redundancy_ratio": decision_row.get("redundancy_ratio", float("nan")),
        "selected_mean_pairwise_jaccard": decision_row.get("mean_pairwise_jaccard", float("nan")),
        "loo_k": loo_k,
        "n_essential_in_loo": n_essential,
        "n_redundant_in_loo": n_redundant,
        "n_conflicting_in_loo": n_conflict,
        "compact_low_redundancy": compact_low_redundancy,
        "independent_claim_allowed": compact_low_redundancy,
    }
    for k in config.precision_k_values:
        summary[f"full_precision_at_{k}"] = full_metrics[f"precision_at_{k}"]
        summary[f"full_precision_lift_at_{k}"] = full_metrics[f"precision_lift_at_{k}"]

    ranked["label"] = label
    ranked["candidate_position"] = ranked["latent_idx"].map({lid: i for i, lid in enumerate(candidate_ids)})
    return {
        "summary": summary,
        "ranked": ranked,
        "cumulative": cumulative_df,
        "loo": pd.DataFrame(loo_rows),
        "add_one": pd.DataFrame(individual_rows),
        "stability": stability,
    }


def _write_report(
    output_dir: Path,
    summary_df: pd.DataFrame,
    config: MinimalLatentConfig,
) -> None:
    lines = [
        "# MISC Minimal Non-Redundant Latent Set Analysis",
        "",
        "This report generalizes the earlier RE-only group-structure analysis to the full MISC label matrix.",
        "It uses saved utterance-level SAE features and should be read as a fast probe-space audit, not as generation-time causal intervention.",
        "",
        "## Operational Criteria",
        "",
        f"- Candidate window: per-label Top{config.top_k} (`candidate_mode={config.candidate_mode}`).",
        f"- Minimal AUC K: first K with `AUC >= {config.min_auc:.2f}` and `{config.target_effect_fraction:.0%}` of full Top{config.top_k} AUC effect.",
        f"- Minimal non-redundant K: Minimal AUC K plus `redundancy_ratio <= {config.max_redundancy:.2f}`.",
        "- Redundancy is computed over positive samples hit by latent activations.",
        "- Parent labels `RE` and `QU` are included for consistency checks, but leaf labels should drive paper claims.",
        "",
        "## Label Summary",
        "",
        "| label | role | full AUC | selected K | status | support edges | compact low-redundancy | selected latents |",
        "| --- | --- | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for _, row in summary_df.sort_values(["compact_low_redundancy", "selected_k"], ascending=[False, True], na_position="last").iterrows():
        selected_k = "" if pd.isna(row["selected_k"]) else str(int(row["selected_k"]))
        lines.append(
            f"| {row['label']} | {row['label_role']} | {row['full_auc']:.3f} | {selected_k} | "
            f"{row['selected_status']} | {int(row['n_support_edges'])} | {row['compact_low_redundancy']} | "
            f"{row['selected_latents']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `minimal_auc_but_redundant` means a small K recovers the full probe signal, but the high-activation hit sets overlap too much under the current redundancy threshold.",
            "- `not_reaching_auc_threshold` means even TopK does not reach the operational recognition threshold.",
            "- Negative or boundary latents can help one-vs-rest discrimination, but should not be over-interpreted as positive semantic exemplars.",
        ]
    )
    (output_dir / "minimal_latent_set_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figures(output_dir: Path, summary_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ordered = summary_df.sort_values("selected_k", na_position="last")
    labels = ordered["label"].tolist()
    selected_k = ordered["selected_k"].fillna(0).astype(float).to_numpy()
    support = ordered["n_support_edges"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, selected_k, width=0.4, label="selected_k")
    ax.bar(x + 0.2, support, width=0.4, label="support_edges")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latent count")
    ax.set_title("Minimal selected K vs support-edge count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "minimal_k_vs_support_edges.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#2b8cbe" if bool(v) else "#f03b20" for v in ordered["compact_low_redundancy"]]
    ax.bar(labels, ordered["full_auc"].astype(float), color=colors)
    ax.axhline(0.70, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("Full TopK probe AUC")
    ax.set_title("Full candidate-set AUC by label")
    fig.tight_layout()
    fig.savefig(fig_dir / "full_auc_by_label.png", dpi=180)
    plt.close(fig)


def run_misc_minimal_latent_analysis(
    *,
    features: np.ndarray,
    label_df: pd.DataFrame,
    topk_matrix: pd.DataFrame,
    output_dir: str | Path,
    labels: Iterable[str] = DEFAULT_CORE_LABELS,
    config: MinimalLatentConfig | None = None,
) -> dict[str, Any]:
    config = config or MinimalLatentConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [label.upper() for label in labels]
    topk_df = _prepare_topk(topk_matrix, labels, config.top_k)

    all_summary: list[dict[str, Any]] = []
    all_ranked: list[pd.DataFrame] = []
    all_cumulative: list[pd.DataFrame] = []
    all_loo: list[pd.DataFrame] = []
    all_add: list[pd.DataFrame] = []
    all_stability: list[pd.DataFrame] = []
    payload: dict[str, Any] = {"config": config.__dict__, "labels": {}}

    for label in labels:
        result = analyze_label_minimal_set(features, label_df, topk_df, label, config)
        all_summary.append(result["summary"])
        all_ranked.append(result["ranked"])
        all_cumulative.append(result["cumulative"])
        all_loo.append(result["loo"])
        all_add.append(result["add_one"])
        if not result["stability"].empty:
            all_stability.append(result["stability"])
        payload["labels"][label] = result["summary"]

    summary_df = pd.DataFrame(all_summary)
    ranked_df = pd.concat(all_ranked, ignore_index=True) if all_ranked else pd.DataFrame()
    cumulative_df = pd.concat(all_cumulative, ignore_index=True) if all_cumulative else pd.DataFrame()
    loo_df = pd.concat(all_loo, ignore_index=True) if all_loo else pd.DataFrame()
    add_df = pd.concat(all_add, ignore_index=True) if all_add else pd.DataFrame()
    stability_df = pd.concat(all_stability, ignore_index=True) if all_stability else pd.DataFrame()

    summary_df.to_csv(output_dir / "label_minimal_set_summary.csv", index=False)
    ranked_df.to_csv(output_dir / "latent_group_assignments.csv", index=False)
    cumulative_df.to_csv(output_dir / "cumulative_topk_by_label.csv", index=False)
    loo_df.to_csv(output_dir / "leave_one_out_by_label.csv", index=False)
    add_df.to_csv(output_dir / "add_one_in_by_label.csv", index=False)
    if not stability_df.empty:
        stability_df.to_csv(output_dir / "bootstrap_stability_by_label.csv", index=False)
    with (output_dir / "minimal_latent_sets.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
    _write_report(output_dir, summary_df, config)
    _write_figures(output_dir, summary_df)

    return {
        "summary": summary_df,
        "ranked": ranked_df,
        "cumulative": cumulative_df,
        "leave_one_out": loo_df,
        "add_one_in": add_df,
        "stability": stability_df,
        "output_dir": output_dir,
    }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj):
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
