"""Step 6 baseline comparison for MISC representation structure.

The comparison asks whether SAE latents expose a clearer, sparser, and more
hierarchy-consistent label-representation structure than PCA components or raw
hidden-state coordinates.  Classification is included as a secondary metric;
the main outputs are structural association metrics.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_LABELS: tuple[str, ...] = ("RE", "REC", "QU", "QUO", "GI", "AF", "SU")
DEFAULT_FAMILY_MAP: dict[str, str] = {
    "RE": "reflection",
    "RES": "reflection",
    "REC": "reflection",
    "QU": "question",
    "QUO": "question",
    "QUC": "question",
    "GI": "information",
    "ADP": "information",
    "ADW": "information",
    "AF": "support_affirm",
    "SU": "support_affirm",
    "CO": "confrontation",
}
DEFAULT_EXPECTED_PAIRS: tuple[tuple[str, str], ...] = (
    ("RE", "REC"),
    ("QU", "QUO"),
    ("AF", "SU"),
    ("RE", "RES"),
    ("QU", "QUC"),
)


@dataclass(frozen=True)
class BaselineComparisonConfig:
    pca_components: int | str = 1024
    train_size: float = 0.70
    dev_size: float = 0.15
    random_state: int = 42
    min_directional_auc: float = 0.65
    top_k: int = 20
    concentration_ks: tuple[int, ...] = (10, 20, 50)
    precision_k: int = 50
    classifier_top_features: int = 200
    classifier_max_iter: int = 1000
    association_chunk_size: int = 512


def load_matrix(path: str | Path) -> np.ndarray:
    """Load a matrix from .pt, .npy, or .npz artifacts."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    if path.suffix == ".npz":
        payload = np.load(path)
        for key in (
            "utterance_features",
            "utterance_activations",
            "features",
            "activations",
            "X",
            "arr_0",
        ):
            if key in payload:
                return np.asarray(payload[key], dtype=np.float32)
        raise KeyError(f"No supported matrix key found in {path}")

    if path.suffix == ".pt":
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(f"Loading {path} requires torch.") from exc
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            for key in (
                "utterance_features",
                "utterance_activations",
                "features",
                "activations",
                "X",
            ):
                if key in payload:
                    payload = payload[key]
                    break
        if hasattr(payload, "detach"):
            payload = payload.detach().cpu().float().numpy()
        return np.asarray(payload, dtype=np.float32)

    raise ValueError(f"Unsupported matrix file extension: {path.suffix}")


def _select_available_labels(label_df: pd.DataFrame, labels: Iterable[str]) -> list[str]:
    selected = []
    for label in labels:
        upper = label.upper()
        if upper in label_df.columns:
            selected.append(upper)
    if not selected:
        raise ValueError("None of the requested labels are present in the label matrix.")
    return selected


def _split_indices(
    label_df: pd.DataFrame,
    *,
    train_size: float,
    dev_size: float,
    random_state: int,
) -> dict[str, np.ndarray]:
    n = len(label_df)
    indices = np.arange(n)
    test_size = max(0.0, 1.0 - train_size)
    stratify = None
    if "predicted_code" in label_df.columns:
        series = label_df["predicted_code"].fillna("UNKNOWN").astype(str)
        if series.value_counts().min() >= 2:
            stratify = series.to_numpy()

    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )
    if test_size <= 0 or len(temp_idx) == 0:
        return {"train": np.sort(train_idx), "dev": np.array([], dtype=int), "test": np.array([], dtype=int)}

    dev_fraction_of_temp = dev_size / test_size if test_size > 0 else 0.5
    dev_fraction_of_temp = min(max(dev_fraction_of_temp, 0.0), 1.0)
    temp_stratify = None
    if stratify is not None:
        temp_labels = stratify[temp_idx]
        vc = pd.Series(temp_labels).value_counts()
        if not vc.empty and vc.min() >= 2:
            temp_stratify = temp_labels

    if dev_fraction_of_temp <= 0:
        dev_idx = np.array([], dtype=int)
        test_idx = temp_idx
    elif dev_fraction_of_temp >= 1:
        dev_idx = temp_idx
        test_idx = np.array([], dtype=int)
    else:
        dev_idx, test_idx = train_test_split(
            temp_idx,
            train_size=dev_fraction_of_temp,
            random_state=random_state + 1,
            shuffle=True,
            stratify=temp_stratify,
        )
    return {"train": np.sort(train_idx), "dev": np.sort(dev_idx), "test": np.sort(test_idx)}


def _fit_pca(raw_hidden: np.ndarray, train_idx: np.ndarray, config: BaselineComparisonConfig) -> tuple[np.ndarray, dict[str, Any]]:
    max_components = min(raw_hidden.shape[1], len(train_idx))
    if isinstance(config.pca_components, str):
        requested = config.pca_components.lower()
        if requested in {"full", "same-as-raw", "d_model"}:
            n_components = max_components
        elif requested == "auto":
            n_components = min(1024, max_components)
        else:
            raise ValueError(f"Unknown pca_components value: {config.pca_components}")
    else:
        n_components = min(int(config.pca_components), max_components)
    n_components = max(1, n_components)
    solver = "randomized" if n_components < max_components else "auto"
    pca = PCA(n_components=n_components, svd_solver=solver, random_state=config.random_state)
    pca.fit(raw_hidden[train_idx])
    transformed = pca.transform(raw_hidden).astype(np.float32)
    return transformed, {
        "n_components": int(n_components),
        "max_components": int(max_components),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "svd_solver": solver,
    }


def _chunked_auc(features: np.ndarray, y: np.ndarray, *, chunk_size: int) -> np.ndarray:
    n_samples, n_features = features.shape
    pos = y.astype(bool)
    n_pos = int(pos.sum())
    n_neg = int(n_samples - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.full(n_features, 0.5, dtype=np.float32)

    auc = np.empty(n_features, dtype=np.float32)
    baseline = n_pos * (n_pos + 1) / 2.0
    denom = float(n_pos * n_neg)
    for start in range(0, n_features, chunk_size):
        end = min(start + chunk_size, n_features)
        ranks = stats.rankdata(features[:, start:end], axis=0, method="average")
        pos_rank_sum = ranks[pos, :].sum(axis=0)
        auc[start:end] = ((pos_rank_sum - baseline) / denom).astype(np.float32)
    return np.nan_to_num(auc, nan=0.5, posinf=0.5, neginf=0.5)


def _chunked_precision_at_k(
    features: np.ndarray,
    y: np.ndarray,
    auc: np.ndarray,
    *,
    k: int,
    chunk_size: int,
) -> np.ndarray:
    n_samples, n_features = features.shape
    if n_samples == 0:
        return np.zeros(n_features, dtype=np.float32)
    k = min(max(int(k), 1), n_samples)
    positive = y.astype(bool)
    precision = np.empty(n_features, dtype=np.float32)
    kth = n_samples - k
    for start in range(0, n_features, chunk_size):
        end = min(start + chunk_size, n_features)
        chunk = features[:, start:end]
        direction = np.where(auc[start:end] >= 0.5, 1.0, -1.0).astype(np.float32)
        oriented = chunk * direction.reshape(1, -1)
        top_idx = np.argpartition(oriented, kth=kth, axis=0)[kth:, :]
        precision[start:end] = positive[top_idx].mean(axis=0).astype(np.float32)
    return precision


def _cohens_d(features: np.ndarray, y: np.ndarray) -> np.ndarray:
    pos = y.astype(bool)
    neg = ~pos
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos < 2 or n_neg < 2:
        return np.zeros(features.shape[1], dtype=np.float32)
    x_pos = features[pos]
    x_neg = features[neg]
    diff = x_pos.mean(axis=0) - x_neg.mean(axis=0)
    pos_var = x_pos.var(axis=0, ddof=1)
    neg_var = x_neg.var(axis=0, ddof=1)
    pooled = ((n_pos - 1) * pos_var + (n_neg - 1) * neg_var) / max(n_pos + n_neg - 2, 1)
    d = diff / np.sqrt(np.maximum(pooled, 1e-24))
    return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _association_for_representation(
    name: str,
    features: np.ndarray,
    label_df: pd.DataFrame,
    labels: list[str],
    config: BaselineComparisonConfig,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    score_profiles: dict[str, np.ndarray] = {}
    for label in labels:
        y = label_df[label].astype(int).to_numpy()
        prevalence = float(y.mean())
        auc = _chunked_auc(features, y, chunk_size=config.association_chunk_size)
        directional_auc = np.maximum(auc, 1.0 - auc)
        association = np.maximum(directional_auc - 0.5, 0.0)
        d = _cohens_d(features, y)
        precision = _chunked_precision_at_k(
            features,
            y,
            auc,
            k=config.precision_k,
            chunk_size=config.association_chunk_size,
        )
        lift = precision / max(prevalence, 1e-12)
        order = np.argsort(association)[::-1]
        rank = np.empty_like(order)
        rank[order] = np.arange(1, len(order) + 1)
        score_profiles[label] = association.astype(np.float32)
        rows.append(
            pd.DataFrame(
                {
                    "representation": name,
                    "label": label,
                    "feature_idx": np.arange(features.shape[1], dtype=int),
                    "auc": auc,
                    "directional_auc": directional_auc,
                    "association_score": association,
                    "cohens_d": d,
                    "abs_cohens_d": np.abs(d),
                    f"precision_at_{config.precision_k}": precision,
                    f"lift_at_{config.precision_k}": lift,
                    "association_rank": rank,
                    "selected_threshold": association >= (config.min_directional_auc - 0.5),
                }
            )
        )
    return pd.concat(rows, ignore_index=True), score_profiles


def _effective_number(scores: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    p = values / total
    return float(1.0 / np.sum(p * p))


def _concentration(scores: np.ndarray, k: int) -> float:
    values = np.asarray(scores, dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    k = min(k, len(values))
    return float(np.sort(values)[::-1][:k].sum() / total)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _label_metrics(
    representation: str,
    profiles: dict[str, np.ndarray],
    association_df: pd.DataFrame,
    labels: list[str],
    config: BaselineComparisonConfig,
) -> pd.DataFrame:
    rows = []
    for label in labels:
        scores = profiles[label]
        selected = scores >= (config.min_directional_auc - 0.5)
        label_rows = association_df[association_df["label"] == label]
        top_row = label_rows.sort_values("association_score", ascending=False).iloc[0]
        row = {
            "representation": representation,
            "label": label,
            "n_features": int(len(scores)),
            "fragmentation": int(selected.sum()),
            "normalized_fragmentation": float(selected.mean()),
            "effective_n": _effective_number(scores),
            "effective_n_fraction": _effective_number(scores) / max(float(len(scores)), 1.0),
            "top1_directional_auc": float(top_row["directional_auc"]),
            "top1_feature_idx": int(top_row["feature_idx"]),
            "top1_association_score": float(top_row["association_score"]),
        }
        for k in config.concentration_ks:
            row[f"concentration_at_{k}"] = _concentration(scores, k)
        row["concentration_at_1pct"] = _concentration(
            scores,
            max(1, int(math.ceil(0.01 * len(scores)))),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _overlap_and_similarity(
    representation: str,
    profiles: dict[str, np.ndarray],
    labels: list[str],
    config: BaselineComparisonConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold = config.min_directional_auc - 0.5
    rows = []
    sim = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for i, label_a in enumerate(labels):
        set_a = set(np.flatnonzero(profiles[label_a] >= threshold).astype(int).tolist())
        for j, label_b in enumerate(labels):
            sim.loc[label_a, label_b] = _cosine(profiles[label_a], profiles[label_b])
            if j <= i:
                continue
            set_b = set(np.flatnonzero(profiles[label_b] >= threshold).astype(int).tolist())
            inter = set_a & set_b
            union = set_a | set_b
            rows.append(
                {
                    "representation": representation,
                    "label_a": label_a,
                    "label_b": label_b,
                    "intersection": len(inter),
                    "union": len(union),
                    "jaccard": float(len(inter) / len(union)) if union else 0.0,
                    "weighted_overlap_cosine": _cosine(profiles[label_a], profiles[label_b]),
                    "family_a": DEFAULT_FAMILY_MAP.get(label_a, label_a),
                    "family_b": DEFAULT_FAMILY_MAP.get(label_b, label_b),
                    "same_family": DEFAULT_FAMILY_MAP.get(label_a, label_a)
                    == DEFAULT_FAMILY_MAP.get(label_b, label_b),
                }
            )
    sim.index.name = "label"
    sim.insert(0, "representation", representation)
    return pd.DataFrame(rows), sim.reset_index()


def _polysemanticity(
    representation: str,
    profiles: dict[str, np.ndarray],
    labels: list[str],
    config: BaselineComparisonConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    threshold = config.min_directional_auc - 0.5
    n_features = len(next(iter(profiles.values())))
    selected_by_feature: list[list[str]] = [[] for _ in range(n_features)]
    for label in labels:
        for idx in np.flatnonzero(profiles[label] >= threshold).astype(int):
            selected_by_feature[idx].append(label)
    rows = []
    for idx, supported in enumerate(selected_by_feature):
        if not supported:
            continue
        families = {DEFAULT_FAMILY_MAP.get(label, label) for label in supported}
        if len(supported) == 1:
            role = "label_specific"
        elif len(families) == 1:
            role = "family_shared"
        else:
            role = "cross_family"
        rows.append(
            {
                "representation": representation,
                "feature_idx": idx,
                "n_labels_supported": len(supported),
                "n_families_supported": len(families),
                "labels": ",".join(supported),
                "families": ",".join(sorted(families)),
                "role": role,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        summary = {
            "mean_polysemanticity": 0.0,
            "max_polysemanticity": 0.0,
            "pct_label_specific": 0.0,
            "pct_family_shared": 0.0,
            "pct_cross_family": 0.0,
        }
    else:
        summary = {
            "mean_polysemanticity": float(df["n_labels_supported"].mean()),
            "max_polysemanticity": float(df["n_labels_supported"].max()),
            "pct_label_specific": float((df["role"] == "label_specific").mean()),
            "pct_family_shared": float((df["role"] == "family_shared").mean()),
            "pct_cross_family": float((df["role"] == "cross_family").mean()),
        }
    return df, summary


def _hierarchy_metrics(overlap_df: pd.DataFrame, labels: list[str]) -> dict[str, float]:
    if overlap_df.empty:
        return {"family_contrast": 0.0, "pair_recovery_at_k": 0.0}
    same = overlap_df[overlap_df["same_family"]]
    diff = overlap_df[~overlap_df["same_family"]]
    family_contrast = float(same["weighted_overlap_cosine"].mean() - diff["weighted_overlap_cosine"].mean())
    expected = {
        tuple(sorted(pair))
        for pair in DEFAULT_EXPECTED_PAIRS
        if pair[0] in labels and pair[1] in labels
    }
    k = len(expected)
    if k == 0:
        recovery = 0.0
    else:
        top = overlap_df.sort_values("weighted_overlap_cosine", ascending=False).head(k)
        recovered = {
            tuple(sorted((row["label_a"], row["label_b"])))
            for _, row in top.iterrows()
        }
        recovery = len(expected & recovered) / k
    return {
        "family_contrast": family_contrast,
        "pair_recovery_at_k": float(recovery),
        "expected_pair_count": float(k),
    }


def _classification_metrics(
    representation: str,
    features: np.ndarray,
    label_df: pd.DataFrame,
    labels: list[str],
    splits: dict[str, np.ndarray],
    config: BaselineComparisonConfig,
) -> pd.DataFrame:
    train_idx = splits["train"]
    test_idx = splits["test"] if len(splits["test"]) else splits["dev"]
    if len(test_idx) == 0:
        test_idx = train_idx
    rows = []
    for label in labels:
        y = label_df[label].astype(int).to_numpy()
        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            rows.append(
                {
                    "representation": representation,
                    "label": label,
                    "probe_auc": 0.5,
                    "probe_average_precision": float(y_test.mean()),
                    "probe_f1": 0.0,
                    "probe_balanced_accuracy": 0.5,
                    "n_probe_features": 0,
                }
            )
            continue
        train_auc = _chunked_auc(
            features[train_idx],
            y_train,
            chunk_size=config.association_chunk_size,
        )
        train_assoc = np.maximum(np.maximum(train_auc, 1.0 - train_auc) - 0.5, 0.0)
        n_probe = min(config.classifier_top_features, features.shape[1])
        top_features = np.argsort(train_assoc)[::-1][:n_probe]
        scaler = StandardScaler()
        x_train = scaler.fit_transform(features[train_idx][:, top_features])
        x_test = scaler.transform(features[test_idx][:, top_features])
        clf = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            random_state=config.random_state,
            max_iter=config.classifier_max_iter,
        )
        clf.fit(x_train, y_train)
        probs = clf.predict_proba(x_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        rows.append(
            {
                "representation": representation,
                "label": label,
                "probe_auc": float(roc_auc_score(y_test, probs)),
                "probe_average_precision": float(average_precision_score(y_test, probs)),
                "probe_f1": float(f1_score(y_test, preds, zero_division=0)),
                "probe_balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
                "n_probe_features": int(n_probe),
            }
        )
    return pd.DataFrame(rows)


def _interpretation_for_representation(name: str, row: pd.Series) -> str:
    if name == "sae_latents":
        return "Sparse SAE feature basis; primary interpretable representation."
    if name == "pca_components":
        return "Dense orthogonal baseline; components capture variance, not native semantic units."
    if name == "raw_hidden_dims":
        return "Native hidden coordinates; predictive baseline but coordinate semantics are unclear."
    return "Baseline representation."


def _build_table4(
    label_metrics: pd.DataFrame,
    overlap_rows: pd.DataFrame,
    poly_summary: dict[str, dict[str, float]],
    hierarchy_rows: dict[str, dict[str, float]],
    classification_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for rep in label_metrics["representation"].drop_duplicates().tolist():
        lm = label_metrics[label_metrics["representation"] == rep]
        ov = overlap_rows[overlap_rows["representation"] == rep]
        clf = classification_rows[classification_rows["representation"] == rep]
        row = {
            "representation": rep,
            "mean_label_auc": float(lm["top1_directional_auc"].mean()),
            "macro_probe_f1": float(clf["probe_f1"].mean()) if not clf.empty else 0.0,
            "macro_probe_auc": float(clf["probe_auc"].mean()) if not clf.empty else 0.5,
            "mean_effective_n": float(lm["effective_n"].mean()),
            "mean_effective_n_fraction": float(lm["effective_n_fraction"].mean()),
            "mean_concentration_at_20": float(lm["concentration_at_20"].mean()),
            "mean_concentration_at_1pct": float(lm["concentration_at_1pct"].mean()),
            "mean_label_overlap_jaccard": float(ov["jaccard"].mean()) if not ov.empty else 0.0,
            "mean_weighted_overlap": float(ov["weighted_overlap_cosine"].mean()) if not ov.empty else 0.0,
            "family_contrast": hierarchy_rows[rep]["family_contrast"],
            "pair_recovery_at_k": hierarchy_rows[rep]["pair_recovery_at_k"],
            "mean_polysemanticity": poly_summary[rep]["mean_polysemanticity"],
            "max_polysemanticity": poly_summary[rep]["max_polysemanticity"],
        }
        row["interpretability_summary"] = _interpretation_for_representation(rep, pd.Series(row))
        rows.append(row)
    return pd.DataFrame(rows)


def _write_table_markdown(table: pd.DataFrame, path: Path) -> None:
    headers = [
        "Representation",
        "Mean Label AUC (higher)",
        "Macro F1 (higher)",
        "Mean N_eff (lower)",
        "N_eff / D (lower)",
        "Concentration@20 (higher)",
        "Concentration@1% (higher)",
        "Mean Label Overlap",
        "Family Contrast (higher)",
        "Pair Recovery (higher)",
        "Mean Polysemanticity (lower)",
        "Interpretability Summary",
    ]
    lines = [
        "# Table 4: SAE vs PCA vs Raw Hidden States",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * 10 + ["---"]) + " |",
    ]
    for _, row in table.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["representation"]),
                    f"{row['mean_label_auc']:.3f}",
                    f"{row['macro_probe_f1']:.3f}",
                    f"{row['mean_effective_n']:.1f}",
                    f"{row['mean_effective_n_fraction']:.3f}",
                    f"{row['mean_concentration_at_20']:.3f}",
                    f"{row['mean_concentration_at_1pct']:.3f}",
                    f"{row['mean_label_overlap_jaccard']:.3f}",
                    f"{row['family_contrast']:.3f}",
                    f"{row['pair_recovery_at_k']:.3f}",
                    f"{row['mean_polysemanticity']:.2f}",
                    str(row["interpretability_summary"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(output_dir: Path, table: pd.DataFrame, config: BaselineComparisonConfig, pca_info: dict[str, Any]) -> None:
    best_compact = table.sort_values("mean_effective_n_fraction", ascending=True).iloc[0]
    best_abs_concentration = table.sort_values("mean_concentration_at_20", ascending=False).iloc[0]
    best_hierarchy = table.sort_values("family_contrast", ascending=False).iloc[0]
    lines = [
        "# Step 6 Baseline Comparison Report",
        "",
        "Purpose: compare whether SAE latents expose a clearer and more interpretable MISC label-representation structure than PCA components and raw hidden-state dimensions.",
        "",
        "## Configuration",
        "",
        f"- PCA components: {pca_info['n_components']} / max {pca_info['max_components']}, explained variance={pca_info['explained_variance_ratio_sum']:.3f}",
        f"- Thresholded selected set: directional AUC >= {config.min_directional_auc:.2f}",
        f"- Association score: `max(directional_auc - 0.5, 0)`",
        f"- Classifier top features per label: {config.classifier_top_features}",
        "",
        "## Main Result",
        "",
        f"- Lowest normalized N_eff: `{best_compact['representation']}` ({best_compact['mean_effective_n_fraction']:.3f}).",
        f"- Highest absolute Concentration@20: `{best_abs_concentration['representation']}` ({best_abs_concentration['mean_concentration_at_20']:.3f}); interpret this with dimensionality caveats.",
        f"- Highest Family Contrast: `{best_hierarchy['representation']}` ({best_hierarchy['family_contrast']:.3f}).",
        "",
        "Classification performance is reported as a secondary metric; the primary comparison is structural compactness and hierarchy recovery.",
        "",
        "See `table4_baseline_comparison.md` for the paper-facing table.",
    ]
    (output_dir / "baseline_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figures(output_dir: Path, table: pd.DataFrame, similarity_by_rep: dict[str, pd.DataFrame], label_metrics: pd.DataFrame, labels: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for rep, sim_df in similarity_by_rep.items():
        matrix = sim_df.set_index("label").drop(columns=["representation"]).loc[labels, labels].astype(float)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(matrix.values, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title(f"Label similarity: {rep}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{rep}_label_similarity_heatmap.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for rep in label_metrics["representation"].unique():
        sub = label_metrics[label_metrics["representation"] == rep]
        means = []
        ks = []
        for col in [c for c in sub.columns if c.startswith("concentration_at_")]:
            suffix = col.rsplit("_", 1)[1]
            if not suffix.isdigit():
                continue
            ks.append(int(suffix))
            means.append(float(sub[col].mean()))
        order = np.argsort(ks)
        ax.plot(np.asarray(ks)[order], np.asarray(means)[order], marker="o", label=rep)
    ax.set_xlabel("Top-k features")
    ax.set_ylabel("Mean cumulative association mass")
    ax.set_title("Association concentration curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "association_concentration_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    reps = table["representation"].tolist()
    x = np.arange(len(reps))
    values = table["mean_effective_n_fraction"].astype(float)
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(reps, rotation=20, ha="right")
    ax.set_ylabel("Mean N_eff / D")
    ax.set_title("Mean normalized effective number of associated features")
    fig.tight_layout()
    fig.savefig(fig_dir / "mean_effective_n_by_representation.png", dpi=180)
    plt.close(fig)


def run_baseline_comparison(
    *,
    sae_features: np.ndarray,
    raw_hidden: np.ndarray,
    label_df: pd.DataFrame,
    output_dir: str | Path,
    labels: Iterable[str] = DEFAULT_LABELS,
    config: BaselineComparisonConfig | None = None,
) -> dict[str, Any]:
    config = config or BaselineComparisonConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = _select_available_labels(label_df, labels)
    if sae_features.shape[0] != len(label_df) or raw_hidden.shape[0] != len(label_df):
        raise ValueError("SAE/raw feature row counts must match the label matrix.")

    splits = _split_indices(
        label_df,
        train_size=config.train_size,
        dev_size=config.dev_size,
        random_state=config.random_state,
    )
    pca_features, pca_info = _fit_pca(raw_hidden, splits["train"], config)
    representations = {
        "sae_latents": np.ascontiguousarray(sae_features, dtype=np.float32),
        "pca_components": np.ascontiguousarray(pca_features, dtype=np.float32),
        "raw_hidden_dims": np.ascontiguousarray(raw_hidden, dtype=np.float32),
    }

    all_associations = []
    all_label_metrics = []
    all_overlap = []
    all_poly = []
    all_classification = []
    similarity_by_rep: dict[str, pd.DataFrame] = {}
    poly_summary: dict[str, dict[str, float]] = {}
    hierarchy_rows: dict[str, dict[str, float]] = {}

    for rep, matrix in representations.items():
        assoc_df, profiles = _association_for_representation(rep, matrix, label_df, labels, config)
        label_metrics = _label_metrics(rep, profiles, assoc_df, labels, config)
        overlap_df, sim_df = _overlap_and_similarity(rep, profiles, labels, config)
        poly_df, poly_metrics = _polysemanticity(rep, profiles, labels, config)
        hierarchy = _hierarchy_metrics(overlap_df, labels)
        classification = _classification_metrics(rep, matrix, label_df, labels, splits, config)

        assoc_df.to_csv(output_dir / f"{rep}_feature_label_association.csv", index=False)
        sim_df.to_csv(output_dir / f"{rep}_label_similarity.csv", index=False)
        overlap_df.to_csv(output_dir / f"{rep}_label_overlap.csv", index=False)
        poly_df.to_csv(output_dir / f"{rep}_polysemanticity.csv", index=False)

        all_associations.append(assoc_df)
        all_label_metrics.append(label_metrics)
        all_overlap.append(overlap_df)
        all_poly.append(poly_df)
        all_classification.append(classification)
        similarity_by_rep[rep] = sim_df
        poly_summary[rep] = poly_metrics
        hierarchy_rows[rep] = hierarchy

    label_metrics_all = pd.concat(all_label_metrics, ignore_index=True)
    overlap_all = pd.concat(all_overlap, ignore_index=True)
    poly_all = pd.concat(all_poly, ignore_index=True) if any(not p.empty for p in all_poly) else pd.DataFrame()
    classification_all = pd.concat(all_classification, ignore_index=True)
    table4 = _build_table4(label_metrics_all, overlap_all, poly_summary, hierarchy_rows, classification_all)

    label_metrics_all.to_csv(output_dir / "label_structural_metrics_by_representation.csv", index=False)
    overlap_all.to_csv(output_dir / "label_overlap_all_representations.csv", index=False)
    poly_all.to_csv(output_dir / "polysemanticity_all_representations.csv", index=False)
    classification_all.to_csv(output_dir / "classification_by_label.csv", index=False)
    table4.to_csv(output_dir / "table4_baseline_comparison.csv", index=False)
    _write_table_markdown(table4, output_dir / "table4_baseline_comparison.md")
    _write_report(output_dir, table4, config, pca_info)
    _write_figures(output_dir, table4, similarity_by_rep, label_metrics_all, labels)

    summary = {
        "labels": labels,
        "config": config.__dict__,
        "pca": pca_info,
        "splits": {key: [int(x) for x in value] for key, value in splits.items()},
        "table4": table4.to_dict("records"),
        "hierarchy": hierarchy_rows,
        "polysemanticity": poly_summary,
    }
    with (output_dir / "baseline_comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {
        "table4": table4,
        "label_metrics": label_metrics_all,
        "overlap": overlap_all,
        "polysemanticity": poly_all,
        "classification": classification_all,
        "output_dir": output_dir,
        "pca": pca_info,
    }
