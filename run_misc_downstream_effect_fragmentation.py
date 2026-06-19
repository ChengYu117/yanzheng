"""Probe-space downstream effect fragmentation for MISC SAE latents.

For each leaf MISC label, this script retrieves the TopK candidate latents by
directional AUC, trains a full candidate label probe, then ablates each candidate
latent in probe-input space and measures the drop in downstream label-prediction
performance.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


DEFAULT_LABELS: tuple[str, ...] = ("RES", "REC", "QUO", "QUC", "GI", "SU", "AF")


@dataclass(frozen=True)
class DownstreamEffectFragmentationConfig:
    labels: tuple[str, ...] = DEFAULT_LABELS
    candidate_top_k: int = 100
    cv_folds: int = 5
    min_auc: float = 0.70
    effect_epsilon: float = 0.001
    precision_k: int = 50
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
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _prepare_association_matrix(path: str | Path, labels: Iterable[str]) -> pd.DataFrame:
    out = pd.read_csv(path)
    out["label"] = out["label"].astype(str).str.upper()
    label_set = {label.upper() for label in labels}
    out = out[out["label"].isin(label_set)].copy()
    required = {"label", "latent_idx", "directional_auc"}
    missing = sorted(required.difference(out.columns))
    if missing:
        raise ValueError(f"Association matrix missing required columns: {missing}")
    for col in (
        "latent_idx",
        "directional_auc",
        "auc",
        "abs_cohens_d",
        "cohens_d",
        "association_rank",
        "formal_edge_weight",
        "precision_at_50",
        "precision_lift_at_50",
        "prevalence",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["latent_idx"] = out["latent_idx"].astype(int)
    if "stable_edge" in out.columns:
        out["stable_edge"] = _bool_series(out["stable_edge"])
    else:
        out["stable_edge"] = False
    return out[out["latent_idx"] >= 0].copy()


def _build_candidate_pool(
    association: pd.DataFrame,
    *,
    feature_dim: int,
    config: DownstreamEffectFragmentationConfig,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for label in config.labels:
        group = association[association["label"] == label].copy()
        group = group[(group["latent_idx"] >= 0) & (group["latent_idx"] < feature_dim)].copy()
        group = group.sort_values(
            ["directional_auc", "abs_cohens_d", "latent_idx"],
            ascending=[False, False, True],
        )
        group = group.drop_duplicates("latent_idx", keep="first").head(config.candidate_top_k).copy()
        group["candidate_rank_directional_auc"] = np.arange(1, len(group) + 1)
        group["candidate_source"] = f"directional_auc_top{config.candidate_top_k}"
        rows.append(group)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


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
    precision_k: int,
) -> dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    preds = (probs >= 0.5).astype(int)
    p_at = _precision_at_k(y, probs, precision_k)
    return {
        "auc": _safe_auc(y, probs),
        "average_precision": float(average_precision_score(y, probs)) if len(np.unique(y)) > 1 else prevalence,
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)) if len(np.unique(y)) > 1 else 0.5,
        "f1": float(f1_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        f"precision_at_{precision_k}": p_at,
        f"precision_lift_at_{precision_k}": p_at - prevalence,
    }


def _fit_full_probe(
    features: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    candidate_ids: list[int],
    config: DownstreamEffectFragmentationConfig,
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
        precision_k=config.precision_k,
    )
    return x_train, x_val, intercept, coef, full_metrics


def _ablation_effects_for_fold(
    *,
    label: str,
    fold: int,
    y_val: np.ndarray,
    x_val: np.ndarray,
    intercept: float,
    coef: np.ndarray,
    candidate_ids: list[int],
    full_metrics: dict[str, float],
    config: DownstreamEffectFragmentationConfig,
) -> pd.DataFrame:
    full_logits = intercept + x_val @ coef
    prevalence = float(y_val.mean())
    rows: list[dict[str, Any]] = []
    for pos, latent_idx in enumerate(candidate_ids):
        ablated_logits = full_logits - x_val[:, pos] * coef[pos]
        metrics = _score_binary(
            y_val,
            ablated_logits,
            prevalence=prevalence,
            precision_k=config.precision_k,
        )
        delta_auc = full_metrics["auc"] - metrics["auc"]
        delta_ap = full_metrics["average_precision"] - metrics["average_precision"]
        p_lift_key = f"precision_lift_at_{config.precision_k}"
        delta_precision_lift = full_metrics[p_lift_key] - metrics[p_lift_key]
        rows.append(
            {
                "label": label,
                "fold": fold,
                "latent_idx": int(latent_idx),
                "candidate_position": pos + 1,
                "full_auc": full_metrics["auc"],
                "ablated_auc": metrics["auc"],
                "delta_auc": delta_auc,
                "positive_delta_auc": max(delta_auc, 0.0),
                "full_average_precision": full_metrics["average_precision"],
                "ablated_average_precision": metrics["average_precision"],
                "delta_average_precision": delta_ap,
                "positive_delta_average_precision": max(delta_ap, 0.0),
                f"full_precision_lift_at_{config.precision_k}": full_metrics[p_lift_key],
                f"ablated_precision_lift_at_{config.precision_k}": metrics[p_lift_key],
                f"delta_precision_lift_at_{config.precision_k}": delta_precision_lift,
                f"positive_delta_precision_lift_at_{config.precision_k}": max(delta_precision_lift, 0.0),
                "coef": float(coef[pos]),
                "mean_abs_activation_val": float(np.mean(np.abs(x_val[:, pos]))),
            }
        )
    return pd.DataFrame(rows)


def _effective_count(values: Iterable[float]) -> float:
    arr = np.asarray([float(v) for v in values if float(v) > 0], dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float((arr.sum() ** 2) / np.sum(arr * arr))


def _k_for_mass(sorted_values: np.ndarray, target: float) -> int:
    if sorted_values.size == 0 or float(sorted_values.sum()) <= 0:
        return 0
    cumulative = np.cumsum(sorted_values) / float(sorted_values.sum())
    return int(np.searchsorted(cumulative, target, side="left") + 1)


def _classify_fragmentation(full_auc: float, effect_sum: float, k90: int, min_auc: float) -> str:
    if full_auc < min_auc:
        return "not_recoverable_under_top100_auc_pool"
    if effect_sum <= 0:
        return "no_positive_downstream_effect"
    if k90 <= 5:
        return "compact"
    if k90 <= 15:
        return "moderate"
    return "distributed"


def _summarize_effects(
    *,
    label: str,
    candidates: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    fold_effects: pd.DataFrame,
    config: DownstreamEffectFragmentationConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    grouped = (
        fold_effects.groupby(["label", "latent_idx"], as_index=False)
        .agg(
            candidate_position=("candidate_position", "first"),
            delta_auc_mean=("delta_auc", "mean"),
            delta_auc_std=("delta_auc", "std"),
            positive_delta_auc_mean=("positive_delta_auc", "mean"),
            delta_average_precision_mean=("delta_average_precision", "mean"),
            delta_average_precision_std=("delta_average_precision", "std"),
            positive_delta_average_precision_mean=("positive_delta_average_precision", "mean"),
            coef_mean=("coef", "mean"),
            mean_abs_activation_val=("mean_abs_activation_val", "mean"),
        )
    )
    p_lift_col = f"delta_precision_lift_at_{config.precision_k}"
    positive_p_lift_col = f"positive_delta_precision_lift_at_{config.precision_k}"
    p_lift = (
        fold_effects.groupby(["label", "latent_idx"], as_index=False)
        .agg(
            delta_precision_lift_mean=(p_lift_col, "mean"),
            delta_precision_lift_std=(p_lift_col, "std"),
            positive_delta_precision_lift_mean=(positive_p_lift_col, "mean"),
        )
    )
    grouped = grouped.merge(p_lift, on=["label", "latent_idx"], how="left")
    meta_cols = [
        "label",
        "latent_idx",
        "directional_auc",
        "auc",
        "abs_cohens_d",
        "candidate_rank_directional_auc",
        "stable_edge",
    ]
    existing = [col for col in meta_cols if col in candidates.columns]
    grouped = grouped.merge(candidates[existing], on=["label", "latent_idx"], how="left")
    grouped["raw_effect_weight_auc"] = grouped["positive_delta_auc_mean"].clip(lower=0.0)
    grouped["passes_effect_epsilon"] = grouped["raw_effect_weight_auc"] >= config.effect_epsilon
    grouped["effect_weight_auc"] = np.where(
        grouped["passes_effect_epsilon"],
        grouped["raw_effect_weight_auc"],
        0.0,
    )
    raw_effect_sum = float(grouped["raw_effect_weight_auc"].sum())
    effect_sum = float(grouped["effect_weight_auc"].sum())
    if effect_sum > 0:
        grouped["effect_share_auc"] = grouped["effect_weight_auc"] / effect_sum
    else:
        grouped["effect_share_auc"] = 0.0
    grouped = grouped.sort_values(
        ["effect_weight_auc", "raw_effect_weight_auc", "directional_auc", "latent_idx"],
        ascending=[False, False, False, True],
    ).copy()
    grouped["effect_rank"] = np.arange(1, len(grouped) + 1)
    sorted_effects = grouped["effect_weight_auc"].to_numpy(dtype=np.float64)
    top1 = float(sorted_effects[0] / effect_sum) if effect_sum > 0 and sorted_effects.size else 0.0
    top5 = float(sorted_effects[:5].sum() / effect_sum) if effect_sum > 0 and sorted_effects.size else 0.0
    k80 = _k_for_mass(sorted_effects, 0.80)
    k90 = _k_for_mass(sorted_effects, 0.90)
    full_auc_mean = float(fold_metrics["full_auc"].mean())
    summary = {
        "label": label,
        "candidate_pool_size": int(candidates.shape[0]),
        "candidate_rule": f"directional_auc_top{config.candidate_top_k}",
        "cv_folds": int(fold_metrics.shape[0]),
        "full_auc_mean": full_auc_mean,
        "full_auc_std": float(fold_metrics["full_auc"].std(ddof=0)),
        "full_average_precision_mean": float(fold_metrics["full_average_precision"].mean()),
        f"full_precision_lift_at_{config.precision_k}_mean": float(
            fold_metrics[f"full_precision_lift_at_{config.precision_k}"].mean()
        ),
        "effect_epsilon_auc": float(config.effect_epsilon),
        "raw_effect_sum_auc": raw_effect_sum,
        "effect_sum_auc": effect_sum,
        "top1_effect_share": top1,
        "top5_effect_share": top5,
        "k80_effect": k80,
        "k90_effect": k90,
        "effective_effect_latent_count": _effective_count(sorted_effects),
        "n_raw_positive_effect_latents": int((grouped["raw_effect_weight_auc"] > 0).sum()),
        "n_positive_effect_latents": int((grouped["effect_weight_auc"] > 0).sum()),
        "top_effect_latent_idx": int(grouped.iloc[0]["latent_idx"]) if not grouped.empty else -1,
        "top5_effect_latents": ",".join(str(int(v)) for v in grouped.head(5)["latent_idx"].tolist()),
        "fragmentation_class": _classify_fragmentation(
            full_auc_mean,
            effect_sum,
            k90,
            config.min_auc,
        ),
    }
    return grouped, summary


def analyze_label(
    *,
    features: np.ndarray,
    labels_df: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    label: str,
    config: DownstreamEffectFragmentationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    label = label.upper()
    if label not in labels_df.columns:
        raise ValueError(f"Label matrix missing label column: {label}")
    candidates = candidate_pool[candidate_pool["label"] == label].copy()
    candidate_ids = candidates["latent_idx"].astype(int).tolist()
    if len(candidate_ids) != config.candidate_top_k:
        raise ValueError(
            f"{label} expected {config.candidate_top_k} directional-AUC candidates, found {len(candidate_ids)}"
        )
    y = pd.to_numeric(labels_df[label], errors="coerce").fillna(0).astype(int).to_numpy()
    if len(y) != features.shape[0]:
        raise ValueError(f"Label rows ({len(y)}) do not match feature rows ({features.shape[0]})")
    n_splits = min(config.cv_folds, int(y.sum()), int((1 - y).sum()))
    if n_splits < 2:
        raise ValueError(f"{label} has too few positive or negative examples for CV")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
    fold_rows: list[dict[str, Any]] = []
    effect_frames: list[pd.DataFrame] = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(features, y), start=1):
        x_train, x_val, intercept, coef, full_metrics = _fit_full_probe(
            features,
            y,
            train_idx,
            val_idx,
            candidate_ids,
            config,
        )
        fold_rows.append(
            {
                "label": label,
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_positive_val": int(y[val_idx].sum()),
                "full_auc": full_metrics["auc"],
                "full_average_precision": full_metrics["average_precision"],
                "full_balanced_accuracy": full_metrics["balanced_accuracy"],
                "full_f1": full_metrics["f1"],
                "full_precision": full_metrics["precision"],
                f"full_precision_lift_at_{config.precision_k}": full_metrics[
                    f"precision_lift_at_{config.precision_k}"
                ],
            }
        )
        effect_frames.append(
            _ablation_effects_for_fold(
                label=label,
                fold=fold,
                y_val=y[val_idx],
                x_val=x_val,
                intercept=intercept,
                coef=coef,
                candidate_ids=candidate_ids,
                full_metrics=full_metrics,
                config=config,
            )
        )
        del x_train
    fold_metrics = pd.DataFrame(fold_rows)
    fold_effects = pd.concat(effect_frames, ignore_index=True, sort=False)
    latent_effects, summary = _summarize_effects(
        label=label,
        candidates=candidates,
        fold_metrics=fold_metrics,
        fold_effects=fold_effects,
        config=config,
    )
    return fold_metrics, latent_effects, summary


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_report(
    path: Path,
    *,
    summary: pd.DataFrame,
    config: DownstreamEffectFragmentationConfig,
) -> None:
    ordered = summary.sort_values(
        ["k90_effect", "effective_effect_latent_count"],
        ascending=[False, False],
    )
    lines = [
        "# Downstream effect fragmentation",
        "",
        "Fragmentation is measured as probe-space downstream effect sparsity.",
        "",
        "## Method",
        "",
        f"- Candidate pool: Top{config.candidate_top_k} latents by `directional_auc` for each leaf label.",
        "- Full probe: balanced logistic regression on all candidate latents.",
        "- Ablation: set one standardized latent column to 0 on validation data, without retraining the probe.",
        "- Effect: `delta_auc = full_auc - ablated_auc`; negative deltas are clipped to 0 for sparsity summaries.",
        f"- Effect threshold: mean positive `delta_auc` below {config.effect_epsilon:.4f} is kept in the raw table but treated as 0 in fragmentation summaries.",
        "- Interpretation: larger `k90_effect` means the label's downstream effect is more fragmented across latents.",
        "",
        "## Summary",
        "",
        "| Label | Class | Full AUC | K80 | K90 | Effective effect latents | Top1 share | Top5 share | Effect latents | Raw positive latents | Top effect latents |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['label']} | {row['fragmentation_class']} | {_fmt(row['full_auc_mean'])} | "
            f"{int(row['k80_effect'])} | {int(row['k90_effect'])} | "
            f"{_fmt(row['effective_effect_latent_count'])} | {_fmt(row['top1_effect_share'])} | "
            f"{_fmt(row['top5_effect_share'])} | {int(row['n_positive_effect_latents'])} | "
            f"{int(row['n_raw_positive_effect_latents'])} | "
            f"{row['top5_effect_latents'] or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is predictive downstream effect sparsity, not model-level causal sufficiency.",
            "- Parent labels RE and QU are excluded from the main analysis.",
            "- Top100 is a candidate-recall budget; the fragmentation conclusion is based on ablation drops within that pool.",
            "- The threshold prevents very small positive validation drops from inflating K80/K90.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(output_dir: Path, summary: pd.DataFrame, latent_effects: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    ordered = summary.sort_values("k90_effect", ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(ordered["label"], ordered["k90_effect"])
    ax.set_ylabel("K90 effect")
    ax.set_title("Downstream effect fragmentation")
    fig.tight_layout()
    fig.savefig(figure_dir / "k90_effect_bar.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(ordered["label"], ordered["effective_effect_latent_count"])
    ax.set_ylabel("effective effect latents")
    ax.set_title("Effective downstream effect latent count")
    fig.tight_layout()
    fig.savefig(figure_dir / "effective_effect_latents_bar.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for label, group in latent_effects.groupby("label"):
        values = group.sort_values("effect_rank")["effect_weight_auc"].to_numpy(dtype=np.float64)
        if values.sum() <= 0:
            continue
        cumulative = np.cumsum(values) / values.sum()
        ax.plot(np.arange(1, len(cumulative) + 1), cumulative, label=label)
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("latents ranked by downstream effect")
    ax.set_ylabel("cumulative positive delta AUC")
    ax.set_title("Cumulative downstream effect curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "cumulative_effect_curves.png", dpi=220)
    plt.close(fig)


def validate_outputs(
    *,
    summary: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    latent_effects: pd.DataFrame,
    config: DownstreamEffectFragmentationConfig,
) -> None:
    for label in config.labels:
        label_candidates = candidate_pool[candidate_pool["label"] == label].sort_values(
            "candidate_rank_directional_auc"
        )
        n_candidates = int(label_candidates.shape[0])
        if n_candidates != config.candidate_top_k:
            raise AssertionError(f"{label} expected {config.candidate_top_k} candidates, found {n_candidates}")
        ranks = label_candidates["candidate_rank_directional_auc"].tolist()
        if ranks != list(range(1, config.candidate_top_k + 1)):
            raise AssertionError(f"{label} candidate ranks are not contiguous")
        directional_auc = label_candidates["directional_auc"].to_numpy(dtype=np.float64)
        if np.any(np.diff(directional_auc) > 1e-12):
            raise AssertionError(f"{label} candidates are not sorted by descending directional_auc")
    required = {
        "label",
        "full_auc_mean",
        "k80_effect",
        "k90_effect",
        "effective_effect_latent_count",
        "top1_effect_share",
        "top5_effect_share",
        "fragmentation_class",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise AssertionError(f"Summary missing columns: {sorted(missing)}")
    if not (summary["k90_effect"] >= summary["k80_effect"]).all():
        raise AssertionError("k90_effect must be >= k80_effect")
    share_cols = ["top1_effect_share", "top5_effect_share"]
    for col in share_cols:
        if not summary[col].between(0.0, 1.0).all():
            raise AssertionError(f"{col} has values outside [0, 1]")
    if not (latent_effects["effect_rank"] >= 1).all():
        raise AssertionError("effect_rank must be positive")
    below_floor = latent_effects["raw_effect_weight_auc"] < config.effect_epsilon
    if not (latent_effects.loc[below_floor, "effect_weight_auc"] == 0.0).all():
        raise AssertionError("Latents below effect_epsilon must have zero summary effect weight")
    for label, group in latent_effects.groupby("label"):
        effect_sum = float(group["effect_weight_auc"].sum())
        share_sum = float(group["effect_share_auc"].sum())
        if effect_sum > 0 and not np.isclose(share_sum, 1.0, atol=1e-8):
            raise AssertionError(f"{label} effect shares should sum to 1.0")


def run_downstream_effect_fragmentation(
    *,
    association_matrix: str | Path,
    feature_store: str | Path,
    label_matrix: str | Path,
    output_dir: str | Path,
    config: DownstreamEffectFragmentationConfig = DownstreamEffectFragmentationConfig(),
    make_figures: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    features = load_feature_store(feature_store)
    labels_df = pd.read_csv(label_matrix)
    association = _prepare_association_matrix(association_matrix, config.labels)
    candidate_pool = _build_candidate_pool(association, feature_dim=features.shape[1], config=config)
    fold_frames: list[pd.DataFrame] = []
    effect_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for label in config.labels:
        fold_metrics, latent_effects, summary = analyze_label(
            features=features,
            labels_df=labels_df,
            candidate_pool=candidate_pool,
            label=label,
            config=config,
        )
        fold_frames.append(fold_metrics)
        effect_frames.append(latent_effects)
        summary_rows.append(summary)
    fold_metrics_all = pd.concat(fold_frames, ignore_index=True, sort=False)
    latent_effects_all = pd.concat(effect_frames, ignore_index=True, sort=False)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["k90_effect", "effective_effect_latent_count"],
        ascending=[False, False],
    )
    validate_outputs(
        summary=summary_df,
        candidate_pool=candidate_pool,
        latent_effects=latent_effects_all,
        config=config,
    )
    candidate_pool.to_csv(output_path / "downstream_effect_candidate_pool.csv", index=False)
    fold_metrics_all.to_csv(output_path / "downstream_effect_fold_metrics.csv", index=False)
    latent_effects_all.to_csv(output_path / "downstream_effect_latent_effects.csv", index=False)
    summary_df.to_csv(output_path / "downstream_effect_fragmentation_summary.csv", index=False)
    summary_payload = {
        "analysis_version": "downstream_effect_fragmentation_v2",
        "config": asdict(config),
        "input_paths": {
            "association_matrix": str(association_matrix),
            "feature_store": str(feature_store),
            "label_matrix": str(label_matrix),
        },
        "output_dir": str(output_path),
        "fragmentation_ranking": summary_df[["label", "k90_effect", "effective_effect_latent_count"]].to_dict("records"),
    }
    (output_path / "downstream_effect_fragmentation_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    write_report(output_path / "downstream_effect_fragmentation_report.md", summary=summary_df, config=config)
    if make_figures:
        plot_outputs(output_path, summary_df, latent_effects_all)
    return {
        "output_dir": output_path,
        "candidate_pool": candidate_pool,
        "fold_metrics": fold_metrics_all,
        "latent_effects": latent_effects_all,
        "summary": summary_df,
        "summary_payload": summary_payload,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure MISC label fragmentation with probe-space downstream effect sparsity.",
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
    parser.add_argument(
        "--output-dir",
        default="outputs/misc_full_sae_eval/interpretability/downstream_effect_fragmentation",
    )
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--candidate-top-k", type=int, default=100)
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
    config = DownstreamEffectFragmentationConfig(
        labels=tuple(label.upper() for label in args.labels),
        candidate_top_k=args.candidate_top_k,
        cv_folds=args.cv_folds,
        min_auc=args.min_auc,
        effect_epsilon=args.effect_epsilon,
        precision_k=args.precision_k,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )
    result = run_downstream_effect_fragmentation(
        association_matrix=args.association_matrix,
        feature_store=args.feature_store,
        label_matrix=args.label_matrix,
        output_dir=args.output_dir,
        config=config,
        make_figures=not args.no_figures,
    )
    print("Completed downstream effect fragmentation.")
    print(f"Output dir: {result['output_dir']}")
    print(
        result["summary"][
            [
                "label",
                "fragmentation_class",
                "full_auc_mean",
                "k90_effect",
                "effective_effect_latent_count",
                "top1_effect_share",
                "top5_effect_share",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
