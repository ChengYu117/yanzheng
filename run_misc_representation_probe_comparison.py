"""Compare stable-core SAE probes against Top-n, PCA, random SAE, and full baselines."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal local envs
    plt = None  # type: ignore[assignment]
    HAS_MATPLOTLIB = False
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:  # scikit-learn >= 1.1
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.baseline_comparison import load_matrix  # noqa: E402
from nlp_re_base.cross_val_framework import compute_auc_and_cohens_d  # noqa: E402


DEFAULT_LABELS = ("RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF")
DEFAULT_TOP_NS = (10, 20, 50, 100, 200)
METRICS = ("auc", "average_precision", "f1", "balanced_accuracy")
BASELINE_TOP_N = -1


@dataclass(frozen=True)
class RepresentationProbeComparisonConfig:
    labels: tuple[str, ...] = DEFAULT_LABELS
    top_ns: tuple[int, ...] = DEFAULT_TOP_NS
    random_repeats: int = 20
    random_state: int = 42
    folds: int = 5
    split_policy: str = "stratified-group-kfold"
    group_column: str = "file_id"
    C: float = 1.0
    solver: str = "liblinear"
    max_iter: int = 1000
    standardize: bool = True
    association_chunk_size: int = 512
    quiet: bool = False


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _as_float32(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float32)


def _available_labels(label_df: pd.DataFrame, labels: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(str(label).upper() for label in labels if str(label).upper() in label_df.columns)
    if not selected:
        raise ValueError("None of the requested labels are present in the label matrix.")
    return selected


def _effective_folds(y: np.ndarray, groups: np.ndarray | None, requested: int) -> int:
    y = np.asarray(y, dtype=int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    folds = min(int(requested), n_pos, n_neg)
    if groups is not None:
        group_series = pd.Series(groups)
        y_series = pd.Series(y)
        group_sum = y_series.groupby(group_series, sort=False).sum()
        group_count = y_series.groupby(group_series, sort=False).count()
        folds = min(
            folds,
            int((group_sum > 0).sum()),
            int((group_count - group_sum > 0).sum()),
        )
    return max(0, folds)


def _make_splits(
    y: np.ndarray,
    label_df: pd.DataFrame,
    config: RepresentationProbeComparisonConfig,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str, list[str]]:
    warnings: list[str] = []
    policy = str(config.split_policy).lower()
    groups: np.ndarray | None = None
    if policy == "stratified-group-kfold":
        if config.group_column in label_df.columns:
            groups = label_df[config.group_column].fillna("UNKNOWN").astype(str).to_numpy()
        else:
            warnings.append(
                f"group column {config.group_column!r} not found; falling back to stratified-kfold"
            )
            policy = "stratified-kfold"

    folds = _effective_folds(y, groups if policy == "stratified-group-kfold" else None, config.folds)
    if folds < 2 and policy == "stratified-group-kfold":
        warnings.append("not enough positive/negative groups; falling back to stratified-kfold")
        policy = "stratified-kfold"
        groups = None
        folds = _effective_folds(y, None, config.folds)
    if folds < 2:
        return [], policy, warnings

    if policy == "stratified-group-kfold" and StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=config.random_state,
        )
        return list(splitter.split(np.zeros(len(y)), y, groups)), policy, warnings
    if policy == "stratified-group-kfold":
        warnings.append("StratifiedGroupKFold unavailable; falling back to stratified-kfold")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.random_state)
    return list(splitter.split(np.zeros(len(y)), y)), "stratified-kfold", warnings


def _standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(np.asarray(x_train, dtype=np.float32)).astype(np.float32)
    x_test = scaler.transform(np.asarray(x_test, dtype=np.float32)).astype(np.float32)
    return np.ascontiguousarray(x_train), np.ascontiguousarray(x_test)


def _prepare_features(
    matrix: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    columns: np.ndarray | list[int] | None = None,
    standardize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if columns is None:
        x_train = np.asarray(matrix[train_idx], dtype=np.float32)
        x_test = np.asarray(matrix[test_idx], dtype=np.float32)
    else:
        cols = np.asarray(columns, dtype=np.int64)
        x_train = np.asarray(matrix[np.ix_(train_idx, cols)], dtype=np.float32)
        x_test = np.asarray(matrix[np.ix_(test_idx, cols)], dtype=np.float32)
    if x_train.shape[1] == 0 or not standardize:
        return np.ascontiguousarray(x_train), np.ascontiguousarray(x_test)
    return _standardize_train_test(x_train, x_test)


def _safe_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, probs))
    except ValueError:
        return 0.5


def _score_probe(
    *,
    representation: str,
    label: str,
    fold: int,
    top_n: int,
    seed: int | None,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    split_policy_used: str,
    config: RepresentationProbeComparisonConfig,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "representation": representation,
        "label": label,
        "fold": int(fold),
        "top_n": int(top_n),
        "seed": "" if seed is None else int(seed),
        "split_policy": split_policy_used,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_positive": int(np.asarray(y_train).sum()),
        "test_positive": int(np.asarray(y_test).sum()),
        "n_features": int(x_train.shape[1]),
    }
    if extra:
        row.update(extra)
    prevalence = float(np.mean(y_test)) if len(y_test) else 0.0
    if x_train.shape[1] == 0 or len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        row.update(
            {
                "auc": 0.5,
                "average_precision": prevalence,
                "f1": 0.0,
                "balanced_accuracy": 0.5,
                "status": "empty_or_single_class",
            }
        )
        return row

    clf = LogisticRegression(
        C=config.C,
        solver=config.solver,
        class_weight="balanced",
        random_state=config.random_state,
        max_iter=config.max_iter,
    )
    clf.fit(x_train, y_train)
    probs = clf.predict_proba(x_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    row.update(
        {
            "auc": _safe_auc(y_test, probs),
            "average_precision": float(average_precision_score(y_test, probs)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
            "status": "ok",
        }
    )
    return row


def _fit_pca_max(
    raw_hidden: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    max_n: int,
    random_state: int,
    standardize: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x_train = np.asarray(raw_hidden[train_idx], dtype=np.float32)
    x_test = np.asarray(raw_hidden[test_idx], dtype=np.float32)
    if standardize:
        x_train, x_test = _standardize_train_test(x_train, x_test)
    else:
        x_train = np.ascontiguousarray(x_train)
        x_test = np.ascontiguousarray(x_test)
    n_components = max(1, min(int(max_n), x_train.shape[0], x_train.shape[1]))
    max_components = min(x_train.shape[0], x_train.shape[1])
    solver = "full" if n_components >= max_components else "randomized"
    pca = PCA(n_components=n_components, svd_solver=solver, random_state=random_state)
    pca_train = pca.fit_transform(x_train).astype(np.float32)
    pca_test = pca.transform(x_test).astype(np.float32)
    if standardize:
        # Keep fixed-C L2 probe regularization comparable with the other representations.
        pca_train, pca_test = _standardize_train_test(pca_train, pca_test)
    return (
        np.ascontiguousarray(pca_train),
        np.ascontiguousarray(pca_test),
        {
            "pca_fit_n_components": int(n_components),
            "pca_requested_max_n": int(max_n),
            "pca_svd_solver": solver,
            "pca_input_standardized": bool(standardize),
            "pca_post_standardized": bool(standardize),
            "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        },
    )


def _load_keep_latents(feature_filter_audit_path: str | Path, feature_dim: int) -> np.ndarray:
    audit = pd.read_csv(feature_filter_audit_path)
    if not {"latent_idx", "keep"}.issubset(audit.columns):
        raise ValueError("feature_filter_audit must contain latent_idx and keep columns")
    keep = audit.loc[audit["keep"].astype(bool), "latent_idx"].astype(int).to_numpy()
    keep = keep[(keep >= 0) & (keep < int(feature_dim))]
    if keep.size == 0:
        raise ValueError("filtered SAE keep pool is empty")
    return np.unique(keep).astype(np.int32)


def _stable_core_by_label(path: str | Path, labels: Iterable[str], feature_dim: int) -> dict[str, np.ndarray]:
    stable = pd.read_csv(path)
    required = {"label", "latent_idx", "stable_set_role"}
    missing = required.difference(stable.columns)
    if missing:
        raise ValueError(f"stable core table missing columns: {sorted(missing)}")
    rank_col = "full_data_rank" if "full_data_rank" in stable.columns else None
    out: dict[str, np.ndarray] = {}
    for label in labels:
        sub = stable[
            (stable["label"].astype(str) == str(label))
            & (stable["stable_set_role"].astype(str) == "stable_core")
        ].copy()
        if rank_col:
            sub[rank_col] = pd.to_numeric(sub[rank_col], errors="coerce")
            sub = sub.sort_values([rank_col, "latent_idx"], ascending=[True, True])
        else:
            sub = sub.sort_values("latent_idx")
        latent_ids = sub["latent_idx"].astype(int).to_numpy()
        latent_ids = latent_ids[(latent_ids >= 0) & (latent_ids < int(feature_dim))]
        out[str(label)] = np.unique(latent_ids).astype(np.int32)
    return out


def _full_data_rank_lookup(path: str | Path, labels: Iterable[str]) -> dict[str, dict[int, dict[str, float]]]:
    assoc = pd.read_csv(path)
    required = {"label", "latent_idx", "cohens_d"}
    missing = required.difference(assoc.columns)
    if missing:
        raise ValueError(f"association matrix missing columns: {sorted(missing)}")
    out: dict[str, dict[int, dict[str, float]]] = {}
    for label in labels:
        sub = assoc[
            (assoc["label"].astype(str) == str(label))
            & (pd.to_numeric(assoc["cohens_d"], errors="coerce") > 0)
        ].copy()
        sub["cohens_d"] = pd.to_numeric(sub["cohens_d"], errors="coerce")
        sub = sub.sort_values(["cohens_d", "latent_idx"], ascending=[False, True]).reset_index(drop=True)
        label_lookup: dict[int, dict[str, float]] = {}
        for rank, row in enumerate(sub.itertuples(index=False), start=1):
            label_lookup[int(row.latent_idx)] = {
                "full_data_cohens_d_rank": float(rank),
                "full_data_cohens_d": float(row.cohens_d),
            }
        out[str(label)] = label_lookup
    return out


def _topn_fold_order(
    sae_features: np.ndarray,
    train_idx: np.ndarray,
    y_train: np.ndarray,
    keep_latents: np.ndarray,
    *,
    max_n: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_features = np.asarray(sae_features[np.ix_(train_idx, keep_latents)], dtype=np.float32)
    _, d = compute_auc_and_cohens_d(train_features, y_train.astype(bool), chunk_size=chunk_size)
    positive = d > 0
    candidate_latents = keep_latents[positive]
    candidate_d = d[positive]
    order = np.lexsort((candidate_latents, -candidate_d))
    selected_latents = candidate_latents[order][:max_n].astype(np.int32)
    selected_d = candidate_d[order][:max_n].astype(np.float32)
    return selected_latents, selected_d


def _random_latent_sets(
    *,
    labels: Iterable[str],
    top_ns: Iterable[int],
    keep_latents: np.ndarray,
    repeats: int,
    random_state: int,
) -> dict[tuple[str, int, int], tuple[np.ndarray, int]]:
    out: dict[tuple[str, int, int], tuple[np.ndarray, int]] = {}
    keep_latents = np.asarray(keep_latents, dtype=np.int32)
    for label_i, label in enumerate(labels):
        for n in top_ns:
            size = min(int(n), int(keep_latents.size))
            for repeat_i in range(int(repeats)):
                seed = int(random_state) + repeat_i
                effective_seed = int(random_state) + repeat_i + (label_i * 100_000) + (int(n) * 1_000)
                rng = np.random.default_rng(effective_seed)
                selected = np.sort(rng.choice(keep_latents, size=size, replace=False)).astype(np.int32)
                out[(str(label), int(n), seed)] = (selected, effective_seed)
    return out


def _selection_lookup(lookup: dict[str, dict[int, dict[str, float]]], label: str, latent_idx: int) -> dict[str, Any]:
    return lookup.get(str(label), {}).get(
        int(latent_idx),
        {"full_data_cohens_d_rank": math.nan, "full_data_cohens_d": math.nan},
    )


def _summarize_by_label(fold_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["representation", "top_n", "label"]
    rows: list[dict[str, Any]] = []
    for key, group in fold_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row["n_observations"] = int(group.shape[0])
        row["n_features_mean"] = float(pd.to_numeric(group["n_features"], errors="coerce").mean())
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_se"] = float(row[f"{metric}_std"] / math.sqrt(len(values))) if len(values) else math.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["representation", "top_n", "label"]).reset_index(drop=True)


def _summarize_macro(by_label: pd.DataFrame, fold_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (representation, top_n), group in by_label.groupby(["representation", "top_n"], dropna=False):
        row: dict[str, Any] = {
            "representation": representation,
            "top_n": int(top_n),
            "n_labels": int(group["label"].nunique()),
            "n_features_mean": float(group["n_features_mean"].mean()),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[f"{metric}_mean"], errors="coerce")
            row[f"macro_{metric}"] = float(values.mean())
            row[f"macro_{metric}_label_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)

    macro = pd.DataFrame(rows)
    random_rows = fold_df[fold_df["representation"] == "Random SAE-n"].copy()
    if not random_rows.empty:
        seed_label_rows: list[dict[str, Any]] = []
        for (top_n, seed, label), group in random_rows.groupby(["top_n", "seed", "label"], dropna=False):
            row = {"top_n": int(top_n), "seed": int(seed), "label": label}
            for metric in METRICS:
                row[metric] = float(pd.to_numeric(group[metric], errors="coerce").mean())
            seed_label_rows.append(row)
        seed_label = pd.DataFrame(seed_label_rows)
        seed_macro_rows: list[dict[str, Any]] = []
        for (top_n, seed), group in seed_label.groupby(["top_n", "seed"]):
            row = {"top_n": int(top_n), "seed": int(seed)}
            for metric in METRICS:
                row[f"macro_{metric}"] = float(group[metric].mean())
            seed_macro_rows.append(row)
        seed_macro = pd.DataFrame(seed_macro_rows)
        if not seed_macro.empty:
            for top_n, group in seed_macro.groupby("top_n"):
                mask = (macro["representation"] == "Random SAE-n") & (macro["top_n"] == int(top_n))
                for metric in METRICS:
                    values = pd.to_numeric(group[f"macro_{metric}"], errors="coerce")
                    macro.loc[mask, f"macro_{metric}_seed_std"] = (
                        float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    )
                    macro.loc[mask, f"macro_{metric}_seed_se"] = (
                        float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
                    )
    for metric in METRICS:
        macro[f"macro_{metric}_seed_std"] = macro.get(f"macro_{metric}_seed_std", pd.Series(np.nan, index=macro.index))
        macro[f"macro_{metric}_seed_se"] = macro.get(f"macro_{metric}_seed_se", pd.Series(np.nan, index=macro.index))
    return macro.sort_values(["representation", "top_n"]).reset_index(drop=True)


def _write_macro_plot(macro: pd.DataFrame, output_dir: Path) -> Path:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    out = figures / "performance_curves_macro.png"
    if not HAS_MATPLOTLIB:
        # Valid 1x1 transparent PNG. Full curve plotting is available whenever
        # matplotlib is installed; this keeps smoke tests and headless minimal
        # environments from failing before numeric outputs are produced.
        import base64

        out.write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
            )
        )
        (figures / "performance_curves_macro.warning.txt").write_text(
            "matplotlib is unavailable; wrote a placeholder PNG instead of the performance curve.\n",
            encoding="utf-8",
        )
        return out
    metric_titles = {
        "auc": "AUC",
        "average_precision": "PR-AUC / Average Precision",
        "f1": "F1",
        "balanced_accuracy": "Balanced Accuracy",
    }
    curve_specs = [
        ("Top-n SAE", "o"),
        ("PCA-n", "s"),
        ("Random SAE-n", "^"),
    ]
    baseline_specs = [
        ("Hidden State", "--"),
        ("Full SAE", "-."),
        ("Stable Core SAE", ":"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, metric in zip(axes.ravel(), METRICS):
        y_col = f"macro_{metric}"
        for representation, marker in curve_specs:
            group = macro[(macro["representation"] == representation) & (macro["top_n"] > 0)].sort_values("top_n")
            if group.empty:
                continue
            ax.plot(group["top_n"], group[y_col], marker=marker, linewidth=1.8, label=representation)
            if representation == "Random SAE-n":
                std_col = f"{y_col}_seed_std"
                if std_col in group.columns and group[std_col].notna().any():
                    y = group[y_col].to_numpy(dtype=float)
                    err = group[std_col].fillna(0.0).to_numpy(dtype=float)
                    x = group["top_n"].to_numpy(dtype=float)
                    ax.fill_between(x, y - err, y + err, alpha=0.15)
        for representation, linestyle in baseline_specs:
            base = macro[(macro["representation"] == representation) & (macro["top_n"] == BASELINE_TOP_N)]
            if base.empty:
                continue
            ax.axhline(float(base[y_col].iloc[0]), linestyle=linestyle, linewidth=1.5, label=representation)
        ax.set_title(metric_titles[metric])
        ax.set_xlabel("n")
        ax.set_ylabel(metric_titles[metric])
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    dedup: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        dedup[label] = handle
    fig.legend(dedup.values(), dedup.keys(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def _write_report(
    *,
    output_dir: Path,
    macro: pd.DataFrame,
    by_label: pd.DataFrame,
    config: RepresentationProbeComparisonConfig,
    warnings: list[str],
) -> Path:
    lines: list[str] = [
        "# Stable Core vs Top-n/PCA/Random SAE 线性探针对比报告",
        "",
        "## 实验设置",
        "",
        f"- Labels: `{', '.join(config.labels)}`",
        f"- n grid: `{', '.join(str(n) for n in config.top_ns)}`",
        f"- Random SAE repeats: `{config.random_repeats}`",
        f"- Split: `{config.split_policy}` with group column `{config.group_column}`",
        "- Classifier: `LogisticRegression(class_weight=\"balanced\", C=1.0, solver=\"liblinear\")`",
        (
            "- Preprocessing: train-fold standardization before PCA and again on PCA scores before the probe."
            if config.standardize
            else "- Preprocessing: standardization disabled for all representations, including PCA input and PCA scores."
        ),
        "- Metrics: AUC, PR-AUC / Average Precision, F1, Balanced Accuracy",
        "",
        "## Macro 结果",
        "",
        "| representation | n | AUC | PR-AUC | F1 | Balanced Acc. | features |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    display = macro.copy()
    order = {
        "Hidden State": 0,
        "Full SAE": 1,
        "Stable Core SAE": 2,
        "Top-n SAE": 3,
        "PCA-n": 4,
        "Random SAE-n": 5,
    }
    display["_order"] = display["representation"].map(order).fillna(99)
    display = display.sort_values(["_order", "top_n"])
    for _, row in display.iterrows():
        n = "baseline" if int(row["top_n"]) == BASELINE_TOP_N else str(int(row["top_n"]))
        lines.append(
            f"| {row['representation']} | {n} | {_fmt(row['macro_auc'])} | "
            f"{_fmt(row['macro_average_precision'])} | {_fmt(row['macro_f1'])} | "
            f"{_fmt(row['macro_balanced_accuracy'])} | {_fmt(row['n_features_mean'])} |"
        )

    stable = macro[(macro["representation"] == "Stable Core SAE") & (macro["top_n"] == BASELINE_TOP_N)]
    if not stable.empty:
        stable_auc = float(stable["macro_auc"].iloc[0])
        lines.extend(
            [
                "",
                "## Stable Core 位置",
                "",
                f"- Stable Core SAE macro AUC = `{stable_auc:.3f}`。",
            ]
        )
        for rep in ("Hidden State", "Full SAE"):
            base = macro[(macro["representation"] == rep) & (macro["top_n"] == BASELINE_TOP_N)]
            if not base.empty:
                delta = stable_auc - float(base["macro_auc"].iloc[0])
                lines.append(f"- 相比 `{rep}`，Stable Core SAE macro AUC 差值为 `{delta:+.3f}`。")
        for rep in ("Top-n SAE", "PCA-n", "Random SAE-n"):
            group = macro[(macro["representation"] == rep) & (macro["top_n"] > 0)]
            if not group.empty:
                best = group.sort_values("macro_auc", ascending=False).iloc[0]
                delta = stable_auc - float(best["macro_auc"])
                lines.append(
                    f"- 相比 `{rep}` 最佳点 n={int(best['top_n'])} "
                    f"(macro AUC={float(best['macro_auc']):.3f})，Stable Core 差值为 `{delta:+.3f}`。"
                )

    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- `Stable Core SAE` 是独立 baseline，不参与 Top-n 曲线。",
            "- `Top-n SAE` 的 top 是训练折内正向 Cohen's d 排名，避免测试折信息泄漏。",
            "- `PCA-n` 在每个训练折单独拟合 PCA，不在全量数据上预拟合；默认对 PCA score 再按训练折标准化后送入 probe。",
            "- `Random SAE-n` 从 filtered keep pool 抽样，误差带反映随机 latent 选择方差。",
            "- 本实验是线性可解码性比较，不证明 latent 具有因果机制或临床概念语义。",
        ]
    )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in sorted(set(warnings)):
            lines.append(f"- {warning}")
    path = output_dir / "representation_probe_comparison_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_representation_probe_comparison(
    *,
    sae_features: np.ndarray,
    raw_hidden: np.ndarray,
    label_df: pd.DataFrame,
    filtered_association_path: str | Path,
    feature_filter_audit_path: str | Path,
    stable_core_path: str | Path,
    output_dir: str | Path,
    config: RepresentationProbeComparisonConfig | None = None,
) -> dict[str, Any]:
    config = config or RepresentationProbeComparisonConfig()
    labels = _available_labels(label_df, config.labels)
    config = RepresentationProbeComparisonConfig(**{**asdict(config), "labels": labels})
    sae_features = _as_float32(sae_features)
    raw_hidden = _as_float32(raw_hidden)
    if sae_features.shape[0] != len(label_df) or raw_hidden.shape[0] != len(label_df):
        raise ValueError(
            f"Row mismatch: labels={len(label_df)}, sae={sae_features.shape[0]}, raw={raw_hidden.shape[0]}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    top_ns = tuple(sorted({int(n) for n in config.top_ns if int(n) > 0}))
    max_top_n = max(top_ns)
    keep_latents = _load_keep_latents(feature_filter_audit_path, sae_features.shape[1])
    stable_by_label = _stable_core_by_label(stable_core_path, labels, sae_features.shape[1])
    full_rank_lookup = _full_data_rank_lookup(filtered_association_path, labels)
    random_sets = _random_latent_sets(
        labels=labels,
        top_ns=top_ns,
        keep_latents=keep_latents,
        repeats=config.random_repeats,
        random_state=config.random_state,
    )

    fold_records: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []
    warnings: list[str] = []
    start_time = time.time()

    for label in labels:
        y = label_df[label].astype(int).to_numpy()
        splits, split_policy_used, split_warnings = _make_splits(y, label_df, config)
        warnings.extend(f"{label}: {warning}" for warning in split_warnings)
        if not splits:
            warnings.append(f"{label}: skipped because fewer than two valid folds are available")
            continue
        if not config.quiet:
            print(
                f"[probe] label={label} positives={int(y.sum())}/{len(y)} "
                f"folds={len(splits)} split={split_policy_used}",
                flush=True,
            )

        stable_latents = stable_by_label.get(label, np.array([], dtype=np.int32))
        for rank, latent_idx in enumerate(stable_latents, start=1):
            ref = _selection_lookup(full_rank_lookup, label, int(latent_idx))
            selection_records.append(
                {
                    "representation": "Stable Core SAE",
                    "selection_type": "stable_core",
                    "label": label,
                    "fold": "",
                    "top_n": BASELINE_TOP_N,
                    "seed": "",
                    "rank": int(rank),
                    "latent_idx": int(latent_idx),
                    "train_fold_cohens_d": math.nan,
                    **ref,
                }
            )

        for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
            y_train = y[train_idx]
            y_test = y[test_idx]
            if not config.quiet:
                print(f"[probe] {label} fold={fold_id}/{len(splits)}", flush=True)

            raw_train, raw_test = _prepare_features(
                raw_hidden,
                train_idx,
                test_idx,
                standardize=config.standardize,
            )
            fold_records.append(
                _score_probe(
                    representation="Hidden State",
                    label=label,
                    fold=fold_id,
                    top_n=BASELINE_TOP_N,
                    seed=None,
                    x_train=raw_train,
                    x_test=raw_test,
                    y_train=y_train,
                    y_test=y_test,
                    split_policy_used=split_policy_used,
                    config=config,
                )
            )

            sae_train, sae_test = _prepare_features(
                sae_features,
                train_idx,
                test_idx,
                standardize=config.standardize,
            )
            fold_records.append(
                _score_probe(
                    representation="Full SAE",
                    label=label,
                    fold=fold_id,
                    top_n=BASELINE_TOP_N,
                    seed=None,
                    x_train=sae_train,
                    x_test=sae_test,
                    y_train=y_train,
                    y_test=y_test,
                    split_policy_used=split_policy_used,
                    config=config,
                )
            )
            del sae_train, sae_test

            stable_train, stable_test = _prepare_features(
                sae_features,
                train_idx,
                test_idx,
                columns=stable_latents,
                standardize=config.standardize,
            )
            fold_records.append(
                _score_probe(
                    representation="Stable Core SAE",
                    label=label,
                    fold=fold_id,
                    top_n=BASELINE_TOP_N,
                    seed=None,
                    x_train=stable_train,
                    x_test=stable_test,
                    y_train=y_train,
                    y_test=y_test,
                    split_policy_used=split_policy_used,
                    config=config,
                    extra={"stable_core_latents": int(stable_latents.size)},
                )
            )

            pca_train_max, pca_test_max, pca_info = _fit_pca_max(
                raw_hidden,
                train_idx,
                test_idx,
                max_n=max_top_n,
                random_state=config.random_state,
                standardize=config.standardize,
            )
            for n in top_ns:
                effective_n = min(int(n), pca_train_max.shape[1])
                fold_records.append(
                    _score_probe(
                        representation="PCA-n",
                        label=label,
                        fold=fold_id,
                        top_n=int(n),
                        seed=None,
                        x_train=pca_train_max[:, :effective_n],
                        x_test=pca_test_max[:, :effective_n],
                        y_train=y_train,
                        y_test=y_test,
                        split_policy_used=split_policy_used,
                        config=config,
                        extra={**pca_info, "effective_top_n": int(effective_n)},
                    )
                )

            top_latents, top_d = _topn_fold_order(
                sae_features,
                train_idx,
                y_train,
                keep_latents,
                max_n=max_top_n,
                chunk_size=config.association_chunk_size,
            )
            if top_latents.size < max_top_n:
                warnings.append(
                    f"{label}: fold {fold_id} only has {top_latents.size} positive Cohen's d candidates"
                )
            top_train_max, top_test_max = _prepare_features(
                sae_features,
                train_idx,
                test_idx,
                columns=top_latents,
                standardize=config.standardize,
            )
            for rank, (latent_idx, train_d) in enumerate(zip(top_latents, top_d), start=1):
                ref = _selection_lookup(full_rank_lookup, label, int(latent_idx))
                selection_records.append(
                    {
                        "representation": "Top-n SAE",
                        "selection_type": "fold_positive_cohens_d",
                        "label": label,
                        "fold": int(fold_id),
                        "top_n": max_top_n,
                        "seed": "",
                        "rank": int(rank),
                        "latent_idx": int(latent_idx),
                        "train_fold_cohens_d": float(train_d),
                        **ref,
                    }
                )
            for n in top_ns:
                effective_n = min(int(n), top_train_max.shape[1])
                fold_records.append(
                    _score_probe(
                        representation="Top-n SAE",
                        label=label,
                        fold=fold_id,
                        top_n=int(n),
                        seed=None,
                        x_train=top_train_max[:, :effective_n],
                        x_test=top_test_max[:, :effective_n],
                        y_train=y_train,
                        y_test=y_test,
                        split_policy_used=split_policy_used,
                        config=config,
                        extra={
                            "effective_top_n": int(effective_n),
                            "candidate_pool_size": int(keep_latents.size),
                        },
                    )
                )

            for n in top_ns:
                for seed in range(config.random_state, config.random_state + config.random_repeats):
                    random_latents, effective_seed = random_sets[(label, int(n), int(seed))]
                    if fold_id == 1:
                        for rank, latent_idx in enumerate(random_latents, start=1):
                            ref = _selection_lookup(full_rank_lookup, label, int(latent_idx))
                            selection_records.append(
                                {
                                    "representation": "Random SAE-n",
                                    "selection_type": "random_filtered_keep",
                                    "label": label,
                                    "fold": "",
                                    "top_n": int(n),
                                    "seed": int(seed),
                                    "effective_seed": int(effective_seed),
                                    "rank": int(rank),
                                    "latent_idx": int(latent_idx),
                                    "train_fold_cohens_d": math.nan,
                                    **ref,
                                }
                            )
                    rand_train, rand_test = _prepare_features(
                        sae_features,
                        train_idx,
                        test_idx,
                        columns=random_latents,
                        standardize=config.standardize,
                    )
                    fold_records.append(
                        _score_probe(
                            representation="Random SAE-n",
                            label=label,
                            fold=fold_id,
                            top_n=int(n),
                            seed=int(seed),
                            x_train=rand_train,
                            x_test=rand_test,
                            y_train=y_train,
                            y_test=y_test,
                            split_policy_used=split_policy_used,
                            config=config,
                            extra={
                                "effective_seed": int(effective_seed),
                                "candidate_pool_size": int(keep_latents.size),
                            },
                        )
                    )

    fold_df = pd.DataFrame(fold_records)
    by_label = _summarize_by_label(fold_df)
    macro = _summarize_macro(by_label, fold_df)
    selected = pd.DataFrame(selection_records)

    fold_path = output_path / "probe_fold_metrics.csv"
    by_label_path = output_path / "probe_summary_by_label.csv"
    macro_path = output_path / "probe_macro_summary.csv"
    selected_path = output_path / "selected_latents_by_label_n.csv"
    fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
    by_label.to_csv(by_label_path, index=False, encoding="utf-8-sig")
    macro.to_csv(macro_path, index=False, encoding="utf-8-sig")
    selected.to_csv(selected_path, index=False, encoding="utf-8-sig")
    if not HAS_MATPLOTLIB:
        warnings.append("matplotlib unavailable; performance_curves_macro.png is a placeholder")
    figure_path = _write_macro_plot(macro, output_path)
    report_path = _write_report(output_dir=output_path, macro=macro, by_label=by_label, config=config, warnings=warnings)

    manifest = {
        "analysis": "representation_probe_comparison_stable_core",
        "elapsed_seconds": float(time.time() - start_time),
        "config": asdict(config),
        "inputs": {
            "n_samples": int(len(label_df)),
            "sae_shape": list(map(int, sae_features.shape)),
            "raw_hidden_shape": list(map(int, raw_hidden.shape)),
            "filtered_association": str(filtered_association_path),
            "feature_filter_audit": str(feature_filter_audit_path),
            "stable_core": str(stable_core_path),
        },
        "outputs": {
            "probe_fold_metrics": str(fold_path),
            "probe_summary_by_label": str(by_label_path),
            "probe_macro_summary": str(macro_path),
            "selected_latents_by_label_n": str(selected_path),
            "performance_curves_macro": str(figure_path),
            "report": str(report_path),
        },
        "warnings": sorted(set(warnings)),
    }
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return {
        "fold_metrics": fold_df,
        "summary_by_label": by_label,
        "macro_summary": macro,
        "selected_latents": selected,
        "manifest": manifest,
    }


def _parse_top_ns(values: list[str] | None) -> tuple[int, ...]:
    if not values:
        return DEFAULT_TOP_NS
    out: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return tuple(sorted({n for n in out if n > 0}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae-features", default="outputs/misc_full_sae_eval/feature_store/utterance_features.pt")
    parser.add_argument("--raw-hidden", default="outputs/misc_full_sae_eval/feature_store/utterance_activations.pt")
    parser.add_argument("--label-matrix", default="outputs/misc_full_sae_eval/label_matrix.csv")
    parser.add_argument(
        "--filtered-association",
        default="outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered/latent_label_matrix.csv",
    )
    parser.add_argument(
        "--feature-filter-audit",
        default="outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered/feature_filter_audit.csv",
    )
    parser.add_argument(
        "--stable-core",
        default="outputs/cross_val/stable_topk_selection/stable_topk_latent_set.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/misc_full_sae_eval/interpretability/representation_probe_comparison_stable_core",
    )
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--top-ns", nargs="+", default=[str(n) for n in DEFAULT_TOP_NS])
    parser.add_argument("--random-repeats", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-policy", default="stratified-group-kfold", choices=["stratified-group-kfold", "stratified-kfold"])
    parser.add_argument("--group-column", default="file_id")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--solver", default="liblinear")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--association-chunk-size", type=int, default=512)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = RepresentationProbeComparisonConfig(
        labels=tuple(str(label).upper() for label in args.labels),
        top_ns=_parse_top_ns(args.top_ns),
        random_repeats=args.random_repeats,
        random_state=args.random_state,
        folds=args.folds,
        split_policy=args.split_policy,
        group_column=args.group_column,
        C=args.C,
        solver=args.solver,
        max_iter=args.max_iter,
        standardize=not args.no_standardize,
        association_chunk_size=args.association_chunk_size,
        quiet=args.quiet,
    )
    print(f"[load] SAE features: {args.sae_features}")
    sae_features = load_matrix(args.sae_features)
    print(f"[load] SAE shape: {sae_features.shape}")
    print(f"[load] raw hidden: {args.raw_hidden}")
    raw_hidden = load_matrix(args.raw_hidden)
    print(f"[load] raw hidden shape: {raw_hidden.shape}")
    print(f"[load] labels: {args.label_matrix}")
    label_df = pd.read_csv(args.label_matrix)
    result = run_representation_probe_comparison(
        sae_features=sae_features,
        raw_hidden=raw_hidden,
        label_df=label_df,
        filtered_association_path=args.filtered_association,
        feature_filter_audit_path=args.feature_filter_audit,
        stable_core_path=args.stable_core,
        output_dir=args.output_dir,
        config=config,
    )
    print("[done] Macro summary:")
    print(result["macro_summary"].to_string(index=False))
    print(f"[done] outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
