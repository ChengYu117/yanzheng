"""Probe-space zero-ablation for stable-core latent sets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None  # type: ignore[assignment]

from .contrastive_evidence_pack import (
    DEFAULT_LABELS,
    load_feature_matrix,
    normalise_latents,
    stable_seed,
    write_json,
)


@dataclass(frozen=True)
class ProbeSpaceAblationConfig:
    labels: tuple[str, ...] = DEFAULT_LABELS
    stable_role: str = "stable_core"
    expected_stable_core_count: int | None = 303
    folds: int = 5
    split_policy: str = "stratified-group-kfold"
    group_column: str = "file_id"
    C: float = 1.0
    solver: str = "liblinear"
    max_iter: int = 1000
    random_state: int = 42


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return 0.5


def _safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return float(np.mean(y_true)) if len(y_true) else 0.0


def _make_splits(
    y: np.ndarray,
    label_matrix: pd.DataFrame,
    config: ProbeSpaceAblationConfig,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    y = np.asarray(y, dtype=int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    folds = min(int(config.folds), n_pos, n_neg)
    if folds < 2:
        return [], "insufficient-class-count"

    if config.split_policy == "stratified-group-kfold" and config.group_column in label_matrix.columns and StratifiedGroupKFold is not None:
        groups = label_matrix[config.group_column].fillna("UNKNOWN").astype(str).to_numpy()
        group_df = pd.DataFrame({"group": groups, "y": y})
        grouped = group_df.groupby("group", sort=False)["y"].agg(["sum", "count"])
        n_pos_groups = int((grouped["sum"] > 0).sum())
        n_neg_groups = int(((grouped["count"] - grouped["sum"]) > 0).sum())
        group_folds = min(folds, n_pos_groups, n_neg_groups)
        if group_folds >= 2:
            splitter = StratifiedGroupKFold(n_splits=group_folds, shuffle=True, random_state=config.random_state)
            return list(splitter.split(np.zeros(len(y)), y, groups)), "stratified-group-kfold"

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.random_state)
    return list(splitter.split(np.zeros(len(y)), y)), "stratified-kfold"


def _positions_for_latents(union_latents: list[int], latent_ids: list[int]) -> list[int]:
    position = {latent: idx for idx, latent in enumerate(union_latents)}
    return [position[int(latent)] for latent in latent_ids if int(latent) in position]


def _random_positions(
    *,
    union_latents: list[int],
    target_positions: list[int],
    label: str,
    fold: int,
    random_state: int,
) -> list[int]:
    requested = len(target_positions)
    if requested <= 0:
        return []
    available = [idx for idx in range(len(union_latents)) if idx not in set(target_positions)]
    if len(available) < requested:
        available = list(range(len(union_latents)))
    rng = np.random.default_rng(stable_seed(random_state, label, fold, "probe_space_ablation_random"))
    return [int(idx) for idx in rng.choice(np.asarray(available, dtype=np.int64), size=min(requested, len(available)), replace=False)]


def run_probe_space_ablation(
    *,
    latents_path: str | Path,
    feature_store_path: str | Path,
    label_matrix_path: str | Path,
    output_dir: str | Path,
    config: ProbeSpaceAblationConfig = ProbeSpaceAblationConfig(),
) -> dict[str, Any]:
    output_path = Path(output_dir)
    ablation_dir = output_path / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    latents = normalise_latents(
        pd.read_csv(latents_path),
        tuple(label.upper() for label in config.labels),
        stable_role=config.stable_role,
        expected_count=config.expected_stable_core_count,
    )
    features = load_feature_matrix(feature_store_path)
    label_matrix = pd.read_csv(label_matrix_path)
    label_tuple = tuple(label.upper() for label in config.labels if label.upper() in label_matrix.columns)
    label_to_latents = {
        label: sorted(latents.loc[latents["target_label"] == label, "latent_idx"].astype(int).unique().tolist())
        for label in label_tuple
    }
    union_latents = sorted({latent for values in label_to_latents.values() for latent in values})
    if not union_latents:
        raise ValueError("No stable-core latents available for ablation.")
    X = np.asarray(features[:, union_latents], dtype=np.float32)

    fold_rows: list[dict[str, Any]] = []
    for probe_label in label_tuple:
        y = pd.to_numeric(label_matrix[probe_label], errors="coerce").fillna(0).to_numpy(dtype=int)
        splits, split_policy = _make_splits(y, label_matrix, config)
        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            x_train = X[train_idx]
            x_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            scaler = StandardScaler()
            x_train_std = scaler.fit_transform(x_train).astype(np.float32)
            x_test_std = scaler.transform(x_test).astype(np.float32)
            clf = LogisticRegression(
                C=config.C,
                solver=config.solver,
                class_weight="balanced",
                max_iter=config.max_iter,
                random_state=config.random_state,
            )
            clf.fit(x_train_std, y_train)
            baseline_prob = clf.predict_proba(x_test_std)[:, 1]
            baseline_auc = _safe_auc(y_test, baseline_prob)
            baseline_auprc = _safe_auprc(y_test, baseline_prob)

            for ablated_label in label_tuple:
                positions = _positions_for_latents(union_latents, label_to_latents.get(ablated_label, []))
                random_positions = _random_positions(
                    union_latents=union_latents,
                    target_positions=positions,
                    label=ablated_label,
                    fold=fold_idx,
                    random_state=config.random_state,
                )
                x_ablated = x_test_std.copy()
                if positions:
                    x_ablated[:, positions] = 0.0
                ablated_prob = clf.predict_proba(x_ablated)[:, 1]

                x_random = x_test_std.copy()
                if random_positions:
                    x_random[:, random_positions] = 0.0
                random_prob = clf.predict_proba(x_random)[:, 1]
                fold_rows.append(
                    {
                        "probe_label": probe_label,
                        "ablated_label": ablated_label,
                        "fold": int(fold_idx),
                        "split_policy": split_policy,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "test_positive": int(y_test.sum()),
                        "union_latent_count": int(len(union_latents)),
                        "ablated_latent_count": int(len(positions)),
                        "random_latent_count": int(len(random_positions)),
                        "baseline_auc": baseline_auc,
                        "ablated_auc": _safe_auc(y_test, ablated_prob),
                        "random_auc": _safe_auc(y_test, random_prob),
                        "baseline_auprc": baseline_auprc,
                        "ablated_auprc": _safe_auprc(y_test, ablated_prob),
                        "random_auprc": _safe_auprc(y_test, random_prob),
                        "is_target_probe": bool(probe_label == ablated_label),
                    }
                )

    fold_df = pd.DataFrame(fold_rows)
    if not fold_df.empty:
        fold_df["target_auc_drop"] = fold_df["baseline_auc"] - fold_df["ablated_auc"]
        fold_df["random_auc_drop"] = fold_df["baseline_auc"] - fold_df["random_auc"]

    summary_rows: list[dict[str, Any]] = []
    if not fold_df.empty:
        for ablated_label, group in fold_df.groupby("ablated_label", sort=True):
            target_rows = group[group["is_target_probe"]].copy()
            non_target_rows = group[~group["is_target_probe"]].copy()
            target_drop = float(target_rows["target_auc_drop"].mean()) if not target_rows.empty else np.nan
            random_drop = float(target_rows["random_auc_drop"].mean()) if not target_rows.empty else np.nan
            non_target_drop = float(non_target_rows["target_auc_drop"].mean()) if not non_target_rows.empty else np.nan
            summary_rows.append(
                {
                    "ablated_label": ablated_label,
                    "stable_core_count": int(len(label_to_latents.get(ablated_label, []))),
                    "union_latent_count": int(len(union_latents)),
                    "target_probe_baseline_auc": float(target_rows["baseline_auc"].mean()) if not target_rows.empty else np.nan,
                    "target_probe_ablated_auc": float(target_rows["ablated_auc"].mean()) if not target_rows.empty else np.nan,
                    "target_drop": target_drop,
                    "random_baseline_drop": random_drop,
                    "drop_vs_random": target_drop - random_drop if pd.notna(target_drop) and pd.notna(random_drop) else np.nan,
                    "non_target_drop_mean": non_target_drop,
                    "non_target_preservation_auc": float(non_target_rows["ablated_auc"].mean()) if not non_target_rows.empty else np.nan,
                    "status": (
                        "probe_space_dependency_supported"
                        if pd.notna(target_drop)
                        and pd.notna(random_drop)
                        and target_drop > random_drop
                        and (pd.isna(non_target_drop) or target_drop > non_target_drop)
                        else "weak_or_nonselective"
                    ),
                }
            )
    summary = pd.DataFrame(summary_rows)

    fold_path = ablation_dir / "set_ablation_fold_metrics.csv"
    summary_path = ablation_dir / "set_ablation.csv"
    fold_df.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "step": "ablation",
        "method": "probe_space_mean_zero_ablation_on_union_stable_core_latents",
        "claim_boundary": "probe-space dependency only; not causal mechanism proof",
        "inputs": {
            "latents": str(latents_path),
            "feature_store": str(feature_store_path),
            "label_matrix": str(label_matrix_path),
        },
        "outputs": {
            "set_ablation_fold_metrics": str(fold_path),
            "set_ablation": str(summary_path),
        },
        "parameters": asdict(config),
        "n_union_latents": int(len(union_latents)),
        "label_stable_core_counts": {label: int(len(values)) for label, values in label_to_latents.items()},
        "n_fold_rows": int(len(fold_df)),
    }
    write_json(ablation_dir / "ablation_manifest.json", manifest)
    return manifest


__all__ = ["ProbeSpaceAblationConfig", "run_probe_space_ablation"]
