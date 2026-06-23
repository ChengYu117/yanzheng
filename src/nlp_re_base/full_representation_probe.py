"""Full-representation probes for MISC labels.

This module evaluates whether MISC labels are linearly decodable from complete
SAE feature vectors, raw hidden activations, and PCA-transformed raw activations.
It is intentionally separate from Step 6, whose classifier uses top associated
features as a secondary structural metric.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:  # scikit-learn >= 1.1
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - older sklearn fallback
    StratifiedGroupKFold = None  # type: ignore[assignment]

from .baseline_comparison import load_matrix


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
DEFAULT_SAE_SUBSPACE_TOP_NS: tuple[int, ...] = (1, *range(5, 201, 5))
DEFAULT_SAE_SUBSPACE_RANKINGS: tuple[str, ...] = ("abs_cohens_d", "directional_auc")


@dataclass(frozen=True)
class FullRepresentationProbeConfig:
    labels: tuple[str, ...] = DEFAULT_LABELS
    folds: int = 5
    split_policy: str = "stratified-group-kfold"
    group_column: str = "file_id"
    pca_components: int | str = "full"
    C: float = 1.0
    solver: str = "liblinear"
    max_iter: int = 1000
    random_state: int = 42
    standardize: bool = True
    verbose: bool = True
    include_sae_ranked_subspaces: bool = False
    sae_subspace_top_ns: tuple[int, ...] = DEFAULT_SAE_SUBSPACE_TOP_NS
    sae_subspace_rankings: tuple[str, ...] = DEFAULT_SAE_SUBSPACE_RANKINGS
    association_chunk_size: int = 512


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _select_available_labels(label_df: pd.DataFrame, labels: Iterable[str]) -> list[str]:
    selected: list[str] = []
    for label in labels:
        upper = str(label).upper()
        if upper in label_df.columns:
            selected.append(upper)
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
        n_pos_groups = int((group_sum > 0).sum())
        n_neg_groups = int((group_count - group_sum > 0).sum())
        folds = min(folds, n_pos_groups, n_neg_groups)
    return max(0, folds)


def _make_splits(
    y: np.ndarray,
    label_df: pd.DataFrame,
    config: FullRepresentationProbeConfig,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str, list[str]]:
    warnings: list[str] = []
    policy = config.split_policy.lower()
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

    if policy == "stratified-group-kfold":
        if StratifiedGroupKFold is None:
            warnings.append("StratifiedGroupKFold is unavailable; falling back to stratified-kfold")
            policy = "stratified-kfold"
        else:
            splitter = StratifiedGroupKFold(
                n_splits=folds,
                shuffle=True,
                random_state=config.random_state,
            )
            return list(splitter.split(np.zeros(len(y)), y, groups)), policy, warnings

    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=config.random_state,
    )
    return list(splitter.split(np.zeros(len(y)), y)), "stratified-kfold", warnings


def _resolve_pca_components(
    raw_hidden: np.ndarray,
    train_idx: np.ndarray,
    config: FullRepresentationProbeConfig,
) -> int:
    max_components = min(raw_hidden.shape[1], len(train_idx))
    value = config.pca_components
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"full", "same-as-raw", "d_model"}:
            return max(1, max_components)
        if lowered == "auto":
            return max(1, min(1024, max_components))
        return max(1, min(int(lowered), max_components))
    return max(1, min(int(value), max_components))


def _fit_pca_fold(
    raw_hidden: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: FullRepresentationProbeConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n_components = _resolve_pca_components(raw_hidden, train_idx, config)
    max_components = min(raw_hidden.shape[1], len(train_idx))
    solver = "randomized" if n_components < max_components else "auto"
    pca = PCA(n_components=n_components, svd_solver=solver, random_state=config.random_state)
    x_train = np.asarray(raw_hidden[train_idx], dtype=np.float32)
    x_test = np.asarray(raw_hidden[test_idx], dtype=np.float32)
    pca.fit(x_train)
    return (
        pca.transform(x_train).astype(np.float32),
        pca.transform(x_test).astype(np.float32),
        {
            "n_components": int(n_components),
            "max_components": int(max_components),
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            "svd_solver": solver,
        },
    )


def _standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    return np.ascontiguousarray(x_train), np.ascontiguousarray(x_test)


def _prepare_fold_features(
    features: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    standardize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    x_train = np.asarray(features[train_idx], dtype=np.float32)
    x_test = np.asarray(features[test_idx], dtype=np.float32)
    if not standardize:
        return np.ascontiguousarray(x_train), np.ascontiguousarray(x_test)
    return _standardize_train_test(x_train, x_test)


def _normalize_subspace_top_ns(config: FullRepresentationProbeConfig, feature_dim: int) -> tuple[int, ...]:
    if not config.include_sae_ranked_subspaces:
        return ()
    top_ns: set[int] = set()
    for value in config.sae_subspace_top_ns:
        n = int(value)
        if n > 0:
            top_ns.add(min(n, int(feature_dim)))
    return tuple(sorted(top_ns))


def _normalize_subspace_rankings(config: FullRepresentationProbeConfig) -> tuple[str, ...]:
    allowed = set(DEFAULT_SAE_SUBSPACE_RANKINGS)
    rankings: list[str] = []
    for ranking in config.sae_subspace_rankings:
        normalized = str(ranking).lower()
        if normalized not in allowed:
            raise ValueError(
                f"Unknown SAE subspace ranking {ranking!r}; expected one of {sorted(allowed)}"
            )
        if normalized not in rankings:
            rankings.append(normalized)
    return tuple(rankings)


def _chunked_sae_train_associations(
    features: np.ndarray,
    y: np.ndarray,
    *,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    y = np.asarray(y, dtype=int)
    positive = y.astype(bool)
    n_samples, n_features = features.shape
    n_pos = int(positive.sum())
    n_neg = int(n_samples - n_pos)
    chunk_size = max(1, int(chunk_size))

    auc = np.full(n_features, 0.5, dtype=np.float32)
    cohens_d = np.zeros(n_features, dtype=np.float32)
    if n_pos == 0 or n_neg == 0:
        return {
            "auc": auc,
            "directional_auc": auc.copy(),
            "cohens_d": cohens_d,
            "abs_cohens_d": np.abs(cohens_d),
        }

    auc_baseline = n_pos * (n_pos + 1) / 2.0
    auc_denom = float(n_pos * n_neg)
    negative = ~positive
    for start in range(0, n_features, chunk_size):
        end = min(start + chunk_size, n_features)
        chunk = np.asarray(features[:, start:end], dtype=np.float32)

        ranks = stats.rankdata(chunk, axis=0, method="average")
        pos_rank_sum = ranks[positive, :].sum(axis=0)
        auc[start:end] = ((pos_rank_sum - auc_baseline) / auc_denom).astype(np.float32)

        if n_pos >= 2 and n_neg >= 2:
            pos_chunk = chunk[positive, :]
            neg_chunk = chunk[negative, :]
            diff = pos_chunk.mean(axis=0) - neg_chunk.mean(axis=0)
            pos_var = pos_chunk.var(axis=0, ddof=1)
            neg_var = neg_chunk.var(axis=0, ddof=1)
            pooled = ((n_pos - 1) * pos_var + (n_neg - 1) * neg_var) / max(n_pos + n_neg - 2, 1)
            d = diff / np.sqrt(np.maximum(pooled, 1e-24))
            cohens_d[start:end] = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    auc = np.nan_to_num(auc, nan=0.5, posinf=0.5, neginf=0.5).astype(np.float32)
    cohens_d = np.nan_to_num(cohens_d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    directional_auc = np.maximum(auc, 1.0 - auc).astype(np.float32)
    return {
        "auc": auc,
        "directional_auc": directional_auc,
        "cohens_d": cohens_d,
        "abs_cohens_d": np.abs(cohens_d).astype(np.float32),
    }


def _rank_sae_features(metrics: dict[str, np.ndarray], ranking: str) -> np.ndarray:
    latent_idx = np.arange(len(metrics["auc"]), dtype=np.int64)
    if ranking == "abs_cohens_d":
        primary = metrics["abs_cohens_d"]
        secondary = metrics["directional_auc"]
    elif ranking == "directional_auc":
        primary = metrics["directional_auc"]
        secondary = metrics["abs_cohens_d"]
    else:  # pragma: no cover - guarded by _normalize_subspace_rankings
        raise ValueError(f"Unknown ranking: {ranking}")
    return np.lexsort((latent_idx, -secondary, -primary)).astype(np.int64)


def _subspace_representation_name(ranking: str, top_n: int) -> str:
    return f"sae_top_{ranking}_n{int(top_n):03d}"


def _selected_latent_rows(
    *,
    label: str,
    fold: int,
    ranking: str,
    order: np.ndarray,
    metrics: dict[str, np.ndarray],
    max_n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, latent_idx in enumerate(order[:max_n], start=1):
        idx = int(latent_idx)
        rows.append(
            {
                "label": label,
                "fold": int(fold),
                "subspace_ranking": ranking,
                "rank": int(rank),
                "latent_idx": idx,
                "cohens_d": float(metrics["cohens_d"][idx]),
                "abs_cohens_d": float(metrics["abs_cohens_d"][idx]),
                "auc": float(metrics["auc"][idx]),
                "directional_auc": float(metrics["directional_auc"][idx]),
            }
        )
    return rows

def _safe_auc(y: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, probs))
    except ValueError:
        return 0.5


def _score_probe(
    *,
    representation: str,
    label: str,
    fold: int,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    split_policy_used: str,
    config: FullRepresentationProbeConfig,
    pca_info: dict[str, Any] | None = None,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "representation": representation,
        "label": label,
        "fold": int(fold),
        "split_policy": split_policy_used,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_positive": int(y_train.sum()),
        "test_positive": int(y_test.sum()),
        "n_features": int(x_train.shape[1]),
    }
    if extra_info:
        row.update(extra_info)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        row.update(
            {
                "probe_auc": 0.5,
                "probe_average_precision": float(np.mean(y_test)) if len(y_test) else 0.0,
                "probe_f1": 0.0,
                "probe_balanced_accuracy": 0.5,
                "probe_accuracy": 0.0,
                "status": "single_class_fold",
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
            "probe_auc": _safe_auc(y_test, probs),
            "probe_average_precision": float(average_precision_score(y_test, probs)),
            "probe_f1": float(f1_score(y_test, preds, zero_division=0)),
            "probe_balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
            "probe_accuracy": float(accuracy_score(y_test, preds)),
            "status": "ok",
        }
    )
    if pca_info:
        row.update({f"pca_{key}": value for key, value in pca_info.items()})
    return row


def _summarize_by_label(fold_rows: pd.DataFrame) -> pd.DataFrame:
    if fold_rows.empty:
        return fold_rows
    metrics = [
        "probe_auc",
        "probe_average_precision",
        "probe_f1",
        "probe_balanced_accuracy",
        "probe_accuracy",
    ]
    grouped = fold_rows.groupby(["representation", "label"], as_index=False)
    rows: list[dict[str, Any]] = []
    for (representation, label), group in grouped:
        row: dict[str, Any] = {
            "representation": representation,
            "label": label,
            "n_folds": int(len(group)),
            "n_features_mean": float(group["n_features"].mean()),
            "n_positive": int(group["test_positive"].sum()),
            "n_test": int(group["n_test"].sum()),
        }
        for optional in ("subspace_ranking", "top_n", "source_representation"):
            if optional in group.columns:
                values = group[optional].dropna()
                if not values.empty:
                    value = values.iloc[0]
                    row[optional] = int(value) if optional == "top_n" else str(value)
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["representation", "label"]).reset_index(drop=True)


def _summarize_macro(by_label: pd.DataFrame) -> pd.DataFrame:
    if by_label.empty:
        return by_label
    metrics = [
        "probe_auc_mean",
        "probe_average_precision_mean",
        "probe_f1_mean",
        "probe_balanced_accuracy_mean",
        "probe_accuracy_mean",
    ]
    rows: list[dict[str, Any]] = []
    for representation, group in by_label.groupby("representation", sort=False):
        row: dict[str, Any] = {
            "representation": representation,
            "n_labels": int(len(group)),
            "mean_n_features": float(group["n_features_mean"].mean()),
        }
        for optional in ("subspace_ranking", "top_n", "source_representation"):
            if optional in group.columns:
                values = group[optional].dropna()
                if not values.empty:
                    value = values.iloc[0]
                    row[optional] = int(value) if optional == "top_n" else str(value)
        for metric in metrics:
            macro_name = "macro_" + metric.removeprefix("probe_").removesuffix("_mean")
            row[macro_name] = float(group[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _build_subspace_convergence(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "subspace_ranking" not in summary.columns or "top_n" not in summary.columns:
        return pd.DataFrame()
    subspaces = summary[summary["subspace_ranking"].notna() & summary["top_n"].notna()].copy()
    if subspaces.empty:
        return pd.DataFrame()

    baseline_auc = {
        str(row["representation"]): float(row["macro_auc"])
        for _, row in summary[summary["subspace_ranking"].isna()].iterrows()
        if "macro_auc" in row and pd.notna(row["macro_auc"])
    }
    rows: list[dict[str, Any]] = []
    for ranking, group in subspaces.groupby("subspace_ranking", sort=False):
        group = group.sort_values("top_n")
        max_n = int(group["top_n"].max())
        n200_auc = float(group.loc[group["top_n"] == max_n, "macro_auc"].iloc[-1])
        first_within = group[group["macro_auc"] >= n200_auc - 0.01]
        first_n_within = int(first_within["top_n"].iloc[0]) if not first_within.empty else max_n
        diffs = group["macro_auc"].diff().dropna()
        last_4_gain = float(diffs.tail(4).mean()) if not diffs.empty else 0.0
        for _, row in group.iterrows():
            item = row.to_dict()
            item["auc_to_max_n_gap"] = float(n200_auc - float(row["macro_auc"]))
            item["max_n_macro_auc"] = n200_auc
            item["first_n_within_0.01_of_max_n_macro_auc"] = first_n_within
            item["last_4_step_mean_auc_gain"] = last_4_gain
            item["platformed_by_last_steps"] = bool(abs(last_4_gain) < 0.002)
            for baseline in ("full_sae_latents", "raw_hidden", "pca_raw_hidden"):
                if baseline in baseline_auc:
                    item[f"delta_macro_auc_vs_{baseline}"] = float(row["macro_auc"] - baseline_auc[baseline])
            rows.append(item)
    return pd.DataFrame(rows).sort_values(["subspace_ranking", "top_n"]).reset_index(drop=True)

def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return ""
    view = df.loc[:, columns].copy()
    headers = list(view.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        cells: list[str] = []
        for col in headers:
            value = row[col]
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _fmt_float(value: Any, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def _write_comparison_report_zh(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    by_label: pd.DataFrame,
    convergence: pd.DataFrame,
    config: FullRepresentationProbeConfig,
) -> None:
    lines = [
        "# SAE top-n 子空间探针对比报告",
        "",
        "## 问题",
        "",
        "本实验检验人工 MISC 标签能否从 LLM 内部 SAE features 中被线性识别，并比较三类基线：原始 hidden activations、full SAE latents、full PCA(raw hidden)。新增部分使用每个标签在训练折内按 `abs_cohens_d` 或 `directional_auc` 排名最高的 top-n SAE features 组成子空间训练探针。",
        "",
        "## 关键设置",
        "",
        f"- CV: `{config.split_policy}`, folds={config.folds}, group column=`{config.group_column}`。",
        f"- PCA: `{config.pca_components}`，每个训练折单独拟合。",
        f"- SAE 子空间排序: `{', '.join(config.sae_subspace_rankings)}`。",
        f"- top-n 网格: `{', '.join(str(n) for n in config.sae_subspace_top_ns)}`。",
        "- 重要防泄漏设置: top-n feature 排名只使用当前训练折标签和训练折 SAE features，测试折只用于最终评估。",
        "",
        "## Macro AUC 总览",
        "",
    ]

    if summary.empty:
        lines.append("没有生成 summary。")
    else:
        baseline = summary[summary.get("subspace_ranking", pd.Series(index=summary.index)).isna()].copy()
        if not baseline.empty:
            cols = ["representation", "mean_n_features", "macro_auc", "macro_average_precision", "macro_f1", "macro_balanced_accuracy"]
            lines.append(_markdown_table(baseline, [col for col in cols if col in baseline.columns]))
        if not convergence.empty:
            best_rows = convergence.sort_values("macro_auc", ascending=False).groupby("subspace_ranking", as_index=False).head(1)
            lines.extend(["", "## SAE top-n 子空间最佳点", ""])
            best_cols = [
                "subspace_ranking",
                "top_n",
                "macro_auc",
                "macro_average_precision",
                "macro_f1",
                "delta_macro_auc_vs_full_sae_latents",
                "delta_macro_auc_vs_raw_hidden",
                "delta_macro_auc_vs_pca_raw_hidden",
            ]
            lines.append(_markdown_table(best_rows, [col for col in best_cols if col in best_rows.columns]))

            lines.extend(["", "## 收敛性判断", ""])
            for ranking, group in convergence.groupby("subspace_ranking", sort=False):
                group = group.sort_values("top_n")
                best = group.loc[group["macro_auc"].idxmax()]
                last = group.iloc[-1]
                first_n = int(last["first_n_within_0.01_of_max_n_macro_auc"])
                last_gain = float(last["last_4_step_mean_auc_gain"])
                platformed = "是" if bool(last["platformed_by_last_steps"]) else "否"
                lines.append(
                    f"- `{ranking}`: 最佳 macro AUC={_fmt_float(best['macro_auc'])} at n={int(best['top_n'])}; "
                    f"n={int(last['top_n'])} 时 macro AUC={_fmt_float(last['macro_auc'])}; "
                    f"首次进入最终点 0.01 范围的 n={first_n}; "
                    f"末 4 个步进平均 AUC 增益={last_gain:.4f}; 末端平台化={platformed}。"
                )

            lines.extend(["", "## 解释", ""])
            raw = summary[summary["representation"] == "raw_hidden"]
            full_sae = summary[summary["representation"] == "full_sae_latents"]
            pca = summary[summary["representation"] == "pca_raw_hidden"]
            best_overall = convergence.sort_values("macro_auc", ascending=False).iloc[0]
            lines.append(
                f"- 最强 SAE top-n 子空间为 `{best_overall['subspace_ranking']}` n={int(best_overall['top_n'])}，macro AUC={_fmt_float(best_overall['macro_auc'])}。"
            )
            if not raw.empty:
                lines.append(
                    f"- 相比 raw hidden macro AUC={_fmt_float(raw.iloc[0]['macro_auc'])}，最佳 SAE top-n 子空间差值为 {_fmt_float(best_overall.get('delta_macro_auc_vs_raw_hidden'))}。"
                )
            if not full_sae.empty:
                lines.append(
                    f"- 相比 full SAE macro AUC={_fmt_float(full_sae.iloc[0]['macro_auc'])}，最佳 SAE top-n 子空间差值为 {_fmt_float(best_overall.get('delta_macro_auc_vs_full_sae_latents'))}。"
                )
            if not pca.empty:
                lines.append(
                    f"- 相比 full PCA macro AUC={_fmt_float(pca.iloc[0]['macro_auc'])}，最佳 SAE top-n 子空间差值为 {_fmt_float(best_overall.get('delta_macro_auc_vs_pca_raw_hidden'))}。"
                )
            lines.append(
                "- 审稿口径上，这支持 `MISC 标签信息可从 SAE feature 组合中被线性解码`；但仍不能单独证明这些 SAE features 是因果机制，或等同于人类标注机制。"
            )

        if not by_label.empty and not convergence.empty:
            lines.extend(["", "## 标签层面提示", ""])
            subspace_names = set(convergence["representation"].astype(str))
            sub_by_label = by_label[by_label["representation"].isin(subspace_names)].copy()
            base_by_label = by_label[by_label["representation"].isin(["full_sae_latents", "raw_hidden", "pca_raw_hidden"])].copy()
            if not sub_by_label.empty:
                best_label_rows = sub_by_label.sort_values("probe_auc_mean", ascending=False).groupby("label", as_index=False).head(1)
                display_cols = ["label", "representation", "probe_auc_mean", "probe_f1_mean", "n_features_mean"]
                lines.append(_markdown_table(best_label_rows.sort_values("label"), display_cols))
            if not base_by_label.empty:
                lines.append("")
                lines.append("完整 label 级结果见 `full_probe_by_label_summary.csv`。")

    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `full_probe_summary.csv`: baseline 与所有 top-n 子空间 macro 指标。",
            "- `full_probe_by_label_summary.csv`: label 级均值/方差。",
            "- `full_probe_by_label.csv`: fold 级原始结果。",
            "- `ranked_sae_subspace_convergence.csv`: top-n 收敛与相对 baseline 差值。",
            "- `ranked_sae_subspace_selected_latents.csv`: 每个 label/fold/ranking 的 top feature 排名。",
        ]
    )
    (output_dir / "full_probe_comparison_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def _write_report(
    *,
    output_dir: Path,
    fold_rows: pd.DataFrame,
    by_label: pd.DataFrame,
    summary: pd.DataFrame,
    config: FullRepresentationProbeConfig,
    warnings: list[str],
) -> None:
    lines = [
        "# Full Representation Probe",
        "",
        "This experiment evaluates whether MISC labels are linearly decodable from complete representation vectors.",
        "",
        "It differs from Step 6 `Mean Label AUC`, which measures strongest single-feature association, and from minimal sufficient subspace `Full AUC`, which uses a per-label candidate pool.",
        "",
        "## Configuration",
        "",
        f"- Split policy: `{config.split_policy}`",
        f"- Folds: `{config.folds}`",
        f"- Group column: `{config.group_column}`",
        f"- PCA components: `{config.pca_components}`",
        f"- Logistic C: `{config.C}`",
        f"- Solver: `{config.solver}`",
        f"- Standardize: `{config.standardize}`",
        f"- Include SAE ranked subspaces: `{config.include_sae_ranked_subspaces}`",
        f"- SAE subspace rankings: `{config.sae_subspace_rankings}`",
        f"- SAE subspace top-n grid: `{config.sae_subspace_top_ns}`",
        "",
        "## Macro Summary",
        "",
    ]
    if summary.empty:
        lines.append("No summary rows were produced.")
    else:
        lines.append(_markdown_table(summary, list(summary.columns)))
    lines.extend(["", "## Per-label Summary", ""])
    if by_label.empty:
        lines.append("No per-label rows were produced.")
    else:
        display_cols = [
            "representation",
            "label",
            "probe_auc_mean",
            "probe_average_precision_mean",
            "probe_f1_mean",
            "probe_balanced_accuracy_mean",
            "n_features_mean",
        ]
        lines.append(_markdown_table(by_label, display_cols))
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in sorted(set(warnings)))
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `full_sae_latents` tests the complete SAE feature vector.",
            "- `raw_hidden` tests the original layer activation vector.",
            "- `pca_raw_hidden` tests a dense PCA basis fit on the training fold only.",
            "- High AUC means label information is linearly decodable; it does not prove causal mechanism alignment.",
        ]
    )
    (output_dir / "full_probe_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_full_representation_probe(
    *,
    sae_features: np.ndarray,
    raw_hidden: np.ndarray,
    label_df: pd.DataFrame,
    output_dir: str | Path,
    labels: Iterable[str] = DEFAULT_LABELS,
    config: FullRepresentationProbeConfig | None = None,
) -> dict[str, Any]:
    config = config or FullRepresentationProbeConfig(labels=tuple(str(label).upper() for label in labels))
    labels = _select_available_labels(label_df, labels)
    if sae_features.shape[0] != len(label_df) or raw_hidden.shape[0] != len(label_df):
        raise ValueError(
            f"Row mismatch: labels={len(label_df)}, sae={sae_features.shape[0]}, raw={raw_hidden.shape[0]}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    subspace_top_ns = _normalize_subspace_top_ns(config, sae_features.shape[1])
    subspace_rankings = _normalize_subspace_rankings(config) if subspace_top_ns else ()
    max_subspace_n = max(subspace_top_ns) if subspace_top_ns else 0

    fold_records: list[dict[str, Any]] = []
    selected_latent_records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for label in labels:
        y = label_df[label].astype(int).to_numpy()
        splits, split_policy_used, split_warnings = _make_splits(y, label_df, config)
        warnings.extend(f"{label}: {warning}" for warning in split_warnings)
        if not splits:
            warnings.append(f"{label}: skipped because fewer than two valid folds are available")
            continue
        if config.verbose:
            print(
                f"[probe] label={label} positives={int(y.sum())}/{len(y)} "
                f"folds={len(splits)} split={split_policy_used}",
                flush=True,
            )
        for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
            y_train = y[train_idx]
            y_test = y[test_idx]

            sae_train, sae_test = _prepare_fold_features(
                sae_features,
                train_idx,
                test_idx,
                standardize=config.standardize,
            )
            if config.verbose:
                print(
                    f"[probe] {label} fold={fold_id}/{len(splits)} representation=full_sae_latents "
                    f"features={sae_train.shape[1]}",
                    flush=True,
                )
            fold_records.append(
                _score_probe(
                    representation="full_sae_latents",
                    label=label,
                    fold=fold_id,
                    x_train=sae_train,
                    x_test=sae_test,
                    y_train=y_train,
                    y_test=y_test,
                    split_policy_used=split_policy_used,
                    config=config,
                )
            )

            if subspace_top_ns and len(np.unique(y_train)) >= 2:
                metrics = _chunked_sae_train_associations(
                    sae_train,
                    y_train,
                    chunk_size=config.association_chunk_size,
                )
                for ranking in subspace_rankings:
                    order = _rank_sae_features(metrics, ranking)
                    selected_latent_records.extend(
                        _selected_latent_rows(
                            label=label,
                            fold=fold_id,
                            ranking=ranking,
                            order=order,
                            metrics=metrics,
                            max_n=max_subspace_n,
                        )
                    )
                    selected = order[:max_subspace_n]
                    sub_train = np.ascontiguousarray(sae_train[:, selected], dtype=np.float32)
                    sub_test = np.ascontiguousarray(sae_test[:, selected], dtype=np.float32)
                    if config.verbose:
                        print(
                            f"[probe] {label} fold={fold_id}/{len(splits)} "
                            f"ranking={ranking} top_n_grid={len(subspace_top_ns)} max_n={max_subspace_n}",
                            flush=True,
                        )
                    for top_n in subspace_top_ns:
                        n = min(int(top_n), sub_train.shape[1])
                        fold_records.append(
                            _score_probe(
                                representation=_subspace_representation_name(ranking, n),
                                label=label,
                                fold=fold_id,
                                x_train=sub_train[:, :n],
                                x_test=sub_test[:, :n],
                                y_train=y_train,
                                y_test=y_test,
                                split_policy_used=split_policy_used,
                                config=config,
                                extra_info={
                                    "subspace_ranking": ranking,
                                    "top_n": int(n),
                                    "source_representation": "full_sae_latents",
                                },
                            )
                        )

            raw_train, raw_test = _prepare_fold_features(
                raw_hidden,
                train_idx,
                test_idx,
                standardize=config.standardize,
            )
            if config.verbose:
                print(
                    f"[probe] {label} fold={fold_id}/{len(splits)} representation=raw_hidden "
                    f"features={raw_train.shape[1]}",
                    flush=True,
                )
            fold_records.append(
                _score_probe(
                    representation="raw_hidden",
                    label=label,
                    fold=fold_id,
                    x_train=raw_train,
                    x_test=raw_test,
                    y_train=y_train,
                    y_test=y_test,
                    split_policy_used=split_policy_used,
                    config=config,
                )
            )

            pca_train, pca_test, pca_info = _fit_pca_fold(raw_hidden, train_idx, test_idx, config)
            if config.standardize:
                pca_train, pca_test = _standardize_train_test(pca_train, pca_test)
            else:
                pca_train = np.ascontiguousarray(pca_train)
                pca_test = np.ascontiguousarray(pca_test)
            if config.verbose:
                print(
                    f"[probe] {label} fold={fold_id}/{len(splits)} representation=pca_raw_hidden "
                    f"features={pca_train.shape[1]}",
                    flush=True,
                )
            fold_records.append(
                _score_probe(
                    representation="pca_raw_hidden",
                    label=label,
                    fold=fold_id,
                    x_train=pca_train,
                    x_test=pca_test,
                    y_train=y_train,
                    y_test=y_test,
                    split_policy_used=split_policy_used,
                    config=config,
                    pca_info=pca_info,
                )
            )

    fold_rows = pd.DataFrame(fold_records)
    by_label = _summarize_by_label(fold_rows)
    summary = _summarize_macro(by_label)
    convergence = _build_subspace_convergence(summary)
    selected_latents = pd.DataFrame(
        selected_latent_records,
        columns=[
            "label",
            "fold",
            "subspace_ranking",
            "rank",
            "latent_idx",
            "cohens_d",
            "abs_cohens_d",
            "auc",
            "directional_auc",
        ],
    )

    fold_rows.to_csv(output_path / "full_probe_by_label.csv", index=False)
    by_label.to_csv(output_path / "full_probe_by_label_summary.csv", index=False)
    summary.to_csv(output_path / "full_probe_summary.csv", index=False)
    convergence.to_csv(output_path / "ranked_sae_subspace_convergence.csv", index=False)
    selected_latents.to_csv(output_path / "ranked_sae_subspace_selected_latents.csv", index=False)
    _write_report(
        output_dir=output_path,
        fold_rows=fold_rows,
        by_label=by_label,
        summary=summary,
        config=config,
        warnings=warnings,
    )
    _write_comparison_report_zh(
        output_dir=output_path,
        summary=summary,
        by_label=by_label,
        convergence=convergence,
        config=config,
    )
    metadata = {
        "analysis_version": "full_representation_probe_v2_ranked_sae_subspaces",
        "config": asdict(config),
        "labels": labels,
        "inputs": {
            "n_samples": int(len(label_df)),
            "sae_shape": list(map(int, sae_features.shape)),
            "raw_hidden_shape": list(map(int, raw_hidden.shape)),
        },
        "warnings": sorted(set(warnings)),
        "files": {
            "full_probe_by_label": str(output_path / "full_probe_by_label.csv"),
            "full_probe_by_label_summary": str(output_path / "full_probe_by_label_summary.csv"),
            "full_probe_summary": str(output_path / "full_probe_summary.csv"),
            "ranked_sae_subspace_convergence": str(output_path / "ranked_sae_subspace_convergence.csv"),
            "ranked_sae_subspace_selected_latents": str(output_path / "ranked_sae_subspace_selected_latents.csv"),
            "full_probe_report": str(output_path / "full_probe_report.md"),
            "full_probe_comparison_report_zh": str(output_path / "full_probe_comparison_report_zh.md"),
        },
    }
    with (output_path / "full_probe_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=_json_default)

    return {
        "fold_rows": fold_rows,
        "by_label": by_label,
        "summary": summary,
        "convergence": convergence,
        "selected_latents": selected_latents,
        "metadata": metadata,
    }

__all__ = [
    "DEFAULT_LABELS",
    "FullRepresentationProbeConfig",
    "load_matrix",
    "run_full_representation_probe",
]
