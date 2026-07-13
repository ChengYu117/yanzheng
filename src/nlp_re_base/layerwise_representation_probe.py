"""Layer-wise MISC probes for matched raw, PCA, and SAE representations.

The module deliberately keeps SAE feature identities layer-local.  A latent
index from one OpenMOSS SAE checkpoint is never reused as a candidate at
another transformer layer.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - compatibility fallback
    StratifiedGroupKFold = None  # type: ignore[assignment]

from .cross_val_framework import compute_auc_and_cohens_d


DEFAULT_LABELS = ("RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF")
METRICS = ("auc", "average_precision", "f1", "balanced_accuracy")
BASELINE_TOP_N = -1


@dataclass(frozen=True)
class LayerwiseRepresentationProbeConfig:
    """Fixed protocol shared by every evaluated transformer layer."""

    labels: tuple[str, ...] = DEFAULT_LABELS
    pca_components: int = 100
    sae_top_n: int = 100
    include_full_sae: bool = False
    folds: int = 5
    split_policy: str = "stratified-group-kfold"
    group_column: str = "file_id"
    random_state: int = 42
    C: float = 1.0
    solver: str = "liblinear"
    max_iter: int = 1000
    standardize: bool = True
    association_chunk_size: int = 512
    quiet: bool = False


@dataclass(frozen=True)
class LayerMatrices:
    """Row-aligned representations and the layer-local candidate audit."""

    sae_features: np.ndarray
    raw_hidden: np.ndarray
    feature_filter_audit: pd.DataFrame


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _available_labels(label_df: pd.DataFrame, labels: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(str(label).upper() for label in labels if str(label).upper() in label_df.columns)
    if not selected:
        raise ValueError("None of the requested labels are present in the label matrix.")
    return selected


def _effective_folds(y: np.ndarray, groups: np.ndarray | None, requested: int) -> int:
    y = np.asarray(y, dtype=np.int64)
    folds = min(int(requested), int(y.sum()), int(len(y) - y.sum()))
    if groups is not None:
        group_frame = pd.DataFrame({"group": groups, "label": y})
        per_group = group_frame.groupby("group", sort=False)["label"].agg(["sum", "count"])
        folds = min(
            folds,
            int((per_group["sum"] > 0).sum()),
            int((per_group["count"] - per_group["sum"] > 0).sum()),
        )
    return max(0, folds)


def _make_splits(
    y: np.ndarray,
    label_df: pd.DataFrame,
    config: LayerwiseRepresentationProbeConfig,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str, list[str]]:
    warnings: list[str] = []
    policy = str(config.split_policy).lower()
    groups: np.ndarray | None = None
    if policy == "stratified-group-kfold":
        if config.group_column in label_df.columns:
            groups = label_df[config.group_column].fillna("UNKNOWN").astype(str).to_numpy()
        else:
            warnings.append(f"missing group column {config.group_column!r}; using stratified-kfold")
            policy = "stratified-kfold"

    folds = _effective_folds(y, groups if policy == "stratified-group-kfold" else None, config.folds)
    if folds < 2 and policy == "stratified-group-kfold":
        warnings.append("not enough positive/negative groups; using stratified-kfold")
        policy = "stratified-kfold"
        groups = None
        folds = _effective_folds(y, None, config.folds)
    if folds < 2:
        return [], policy, warnings

    if policy == "stratified-group-kfold" and StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=config.random_state)
        return list(splitter.split(np.zeros(len(y)), y, groups)), policy, warnings
    if policy == "stratified-group-kfold":
        warnings.append("StratifiedGroupKFold unavailable; using stratified-kfold")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.random_state)
    return list(splitter.split(np.zeros(len(y)), y)), "stratified-kfold", warnings


def _standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train = scaler.fit_transform(np.asarray(x_train, dtype=np.float32)).astype(np.float32)
    test = scaler.transform(np.asarray(x_test, dtype=np.float32)).astype(np.float32)
    return np.ascontiguousarray(train), np.ascontiguousarray(test)


def _prepare_features(
    matrix: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    columns: np.ndarray | None = None,
    standardize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if columns is None:
        train = np.asarray(matrix[train_idx], dtype=np.float32)
        test = np.asarray(matrix[test_idx], dtype=np.float32)
    else:
        cols = np.asarray(columns, dtype=np.int64)
        train = np.asarray(matrix[np.ix_(train_idx, cols)], dtype=np.float32)
        test = np.asarray(matrix[np.ix_(test_idx, cols)], dtype=np.float32)
    if train.shape[1] == 0 or not standardize:
        return np.ascontiguousarray(train), np.ascontiguousarray(test)
    return _standardize_train_test(train, test)


def _fit_pca_100(
    raw_hidden: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: LayerwiseRepresentationProbeConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train, test = _prepare_features(raw_hidden, train_idx, test_idx, standardize=config.standardize)
    n_components = min(int(config.pca_components), train.shape[0], train.shape[1])
    if n_components < 1:
        raise ValueError("PCA requires at least one training sample and one hidden dimension.")
    max_components = min(train.shape[0], train.shape[1])
    solver = "full" if n_components >= max_components else "randomized"
    pca = PCA(n_components=n_components, svd_solver=solver, random_state=config.random_state)
    train_pca = pca.fit_transform(train).astype(np.float32)
    test_pca = pca.transform(test).astype(np.float32)
    return (
        np.ascontiguousarray(train_pca),
        np.ascontiguousarray(test_pca),
        {
            "pca_requested_components": int(config.pca_components),
            "pca_effective_components": int(n_components),
            "pca_svd_solver": solver,
            "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        },
    )


def _top_positive_cohens_d(
    sae_features: np.ndarray,
    train_idx: np.ndarray,
    y_train: np.ndarray,
    keep_latents: np.ndarray,
    config: LayerwiseRepresentationProbeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    train_features = np.asarray(sae_features[np.ix_(train_idx, keep_latents)], dtype=np.float32)
    _, cohens_d = compute_auc_and_cohens_d(
        train_features,
        np.asarray(y_train, dtype=bool),
        chunk_size=int(config.association_chunk_size),
    )
    positive = cohens_d > 0
    candidates = keep_latents[positive]
    effects = cohens_d[positive]
    order = np.lexsort((candidates, -effects))
    selected = candidates[order][: int(config.sae_top_n)].astype(np.int32)
    selected_d = effects[order][: int(config.sae_top_n)].astype(np.float32)
    return selected, selected_d


def _safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) >= 2 else 0.5
    except ValueError:
        return 0.5


def _score_probe(
    *,
    layer_idx: int,
    representation: str,
    label: str,
    fold: int,
    top_n: int,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    split_policy: str,
    config: LayerwiseRepresentationProbeConfig,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "layer_idx": int(layer_idx),
        "representation": representation,
        "label": label,
        "fold": int(fold),
        "top_n": int(top_n),
        "split_policy": split_policy,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_positive": int(np.asarray(y_train).sum()),
        "test_positive": int(np.asarray(y_test).sum()),
        "n_features": int(x_train.shape[1]),
    }
    if extra:
        row.update(extra)
    if x_train.shape[1] == 0 or len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        row.update(
            {
                "auc": 0.5,
                "average_precision": float(np.mean(y_test)) if len(y_test) else 0.0,
                "f1": 0.0,
                "balanced_accuracy": 0.5,
                "status": "empty_or_single_class",
            }
        )
        return row

    classifier = LogisticRegression(
        class_weight="balanced",
        C=float(config.C),
        solver=config.solver,
        max_iter=int(config.max_iter),
        random_state=int(config.random_state),
    )
    classifier.fit(x_train, y_train)
    probabilities = classifier.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int64)
    row.update(
        {
            "auc": _safe_auc(y_test, probabilities),
            "average_precision": float(average_precision_score(y_test, probabilities)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "status": "ok",
        }
    )
    return row


def _summarize_by_label(fold_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    keys = ["layer_idx", "representation", "top_n", "label"]
    for key, group in fold_df.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        row["n_observations"] = int(len(group))
        row["n_features_mean"] = float(pd.to_numeric(group["n_features"], errors="coerce").mean())
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_se"] = float(row[f"{metric}_std"] / math.sqrt(len(values))) if len(values) else math.nan
        records.append(row)
    return pd.DataFrame(records).sort_values(keys).reset_index(drop=True)


def _summarize_macro(by_label: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (layer_idx, representation, top_n), group in by_label.groupby(
        ["layer_idx", "representation", "top_n"], dropna=False
    ):
        row: dict[str, Any] = {
            "layer_idx": int(layer_idx),
            "representation": representation,
            "top_n": int(top_n),
            "n_labels": int(group["label"].nunique()),
            "n_features_mean": float(group["n_features_mean"].mean()),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[f"{metric}_mean"], errors="coerce")
            row[f"macro_{metric}"] = float(values.mean())
            row[f"macro_{metric}_label_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        records.append(row)
    return pd.DataFrame(records).sort_values(["layer_idx", "representation", "top_n"]).reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    if frame.empty:
        return ["No rows were generated."]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, (float, np.floating)) and np.isfinite(value):
                values.append(f"{float(value):.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_report(
    *,
    output_dir: Path,
    macro: pd.DataFrame,
    by_label: pd.DataFrame,
    filter_summaries: Mapping[int, Mapping[str, Any]],
    config: LayerwiseRepresentationProbeConfig,
    warnings: list[str],
) -> Path:
    macro_columns = [
        "layer_idx",
        "representation",
        "top_n",
        "macro_auc",
        "macro_average_precision",
        "macro_f1",
        "macro_balanced_accuracy",
        "n_features_mean",
    ]
    label_columns = [
        "layer_idx",
        "label",
        "representation",
        "top_n",
        "auc_mean",
        "average_precision_mean",
        "f1_mean",
        "balanced_accuracy_mean",
    ]
    lines = [
        "# Llama Layer 15/24: Raw Hidden, PCA-100, and SAE Top-100 Probe Report",
        "",
        "## Scope",
        "",
        "- Layers are evaluated with their matching OpenMOSS SAE checkpoints and `max` utterance pooling.",
        "- Every layer uses the same labels, file-grouped folds, linear probe, metrics, and filtered-pool rule.",
        "- Top-100 SAE is selected inside each training fold by positive Cohen's d from that layer's filtered latent pool.",
        "- Stable Core is intentionally omitted: its latent identities and stability assessment are layer-specific and were only established for layer 19.",
        "",
        "## Configuration",
        "",
        f"- Labels: {', '.join(config.labels)}",
        f"- Split: {config.split_policy} grouped by `{config.group_column}`, {config.folds} folds",
        f"- Classifier: LogisticRegression(class_weight=balanced, C={config.C}, solver={config.solver})",
        f"- PCA components: {config.pca_components}; SAE Top-n: {config.sae_top_n}",
        "",
        "## Macro Summary",
        "",
        *_markdown_table(macro, macro_columns),
        "",
        "## Filtered Candidate Pools",
        "",
    ]
    for layer_idx in sorted(filter_summaries):
        summary = filter_summaries[layer_idx]
        lines.append(
            f"- Layer {layer_idx}: kept {summary['n_kept_latents']} / {summary['n_original_latents']} "
            f"latents ({summary['keep_rate']:.1%}) after the common basic feature filter."
        )
    lines.extend(["", "## Per-label Summary", "", *_markdown_table(by_label, label_columns)])
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- This is linear decodability evidence, not a causal test and not evidence that a latent equals a MISC concept.",
            "- PCA-100 is a dense predictive compression baseline; a higher score does not make PCA components more auditable than SAE latents.",
            "- Current data contain counselor utterances only. RE/RES/REC results cannot establish client-context relations.",
        ]
    )
    if warnings:
        lines.extend(["", "## Warnings", "", *[f"- {item}" for item in sorted(set(warnings))]])
    path = output_dir / "layerwise_representation_probe_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_layerwise_representation_probe(
    *,
    layer_matrices: Mapping[int, LayerMatrices],
    label_df: pd.DataFrame,
    output_dir: str | Path,
    config: LayerwiseRepresentationProbeConfig | None = None,
) -> dict[str, Any]:
    """Run the fixed comparison for every supplied layer and write audit artifacts."""

    config = config or LayerwiseRepresentationProbeConfig()
    labels = _available_labels(label_df, config.labels)
    config = LayerwiseRepresentationProbeConfig(**{**asdict(config), "labels": labels})
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fold_records: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    filter_summaries: dict[int, dict[str, Any]] = {}
    warnings: list[str] = []

    for layer_idx, payload in sorted(layer_matrices.items()):
        sae = np.asarray(payload.sae_features, dtype=np.float32)
        raw = np.asarray(payload.raw_hidden, dtype=np.float32)
        if sae.ndim != 2 or raw.ndim != 2:
            raise ValueError(f"layer {layer_idx}: representations must be two-dimensional")
        if sae.shape[0] != len(label_df) or raw.shape[0] != len(label_df):
            raise ValueError(
                f"layer {layer_idx}: row mismatch labels={len(label_df)}, sae={sae.shape[0]}, raw={raw.shape[0]}"
            )
        audit = payload.feature_filter_audit.copy()
        required = {"latent_idx", "keep"}
        missing = required.difference(audit.columns)
        if missing:
            raise ValueError(f"layer {layer_idx}: feature filter audit missing {sorted(missing)}")
        keep = audit.loc[audit["keep"].astype(bool), "latent_idx"].astype(int).to_numpy()
        keep = np.unique(keep[(keep >= 0) & (keep < sae.shape[1])]).astype(np.int32)
        if keep.size == 0:
            raise ValueError(f"layer {layer_idx}: no SAE latents remain after filtering")
        layer_dir = output_path / f"layer_{int(layer_idx):02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        audit.to_csv(layer_dir / "feature_filter_audit.csv", index=False, encoding="utf-8-sig")
        filter_summary = {
            "n_original_latents": int(sae.shape[1]),
            "n_kept_latents": int(keep.size),
            "keep_rate": float(keep.size / sae.shape[1]),
        }
        filter_summaries[int(layer_idx)] = filter_summary
        (layer_dir / "feature_filter_summary.json").write_text(
            json.dumps(filter_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        for label in labels:
            y = label_df[label].astype(int).to_numpy()
            splits, split_policy, split_warnings = _make_splits(y, label_df, config)
            warnings.extend(f"layer {layer_idx}, {label}: {warning}" for warning in split_warnings)
            if not splits:
                warnings.append(f"layer {layer_idx}, {label}: fewer than two valid folds")
                continue
            if not config.quiet:
                print(
                    f"[layer {layer_idx}] label={label} positives={int(y.sum())}/{len(y)} folds={len(splits)}",
                    flush=True,
                )
            for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
                y_train, y_test = y[train_idx], y[test_idx]
                raw_train, raw_test = _prepare_features(
                    raw, train_idx, test_idx, standardize=config.standardize
                )
                fold_records.append(
                    _score_probe(
                        layer_idx=layer_idx,
                        representation="Raw Hidden",
                        label=label,
                        fold=fold_id,
                        top_n=BASELINE_TOP_N,
                        x_train=raw_train,
                        x_test=raw_test,
                        y_train=y_train,
                        y_test=y_test,
                        split_policy=split_policy,
                        config=config,
                    )
                )

                if config.include_full_sae:
                    full_train, full_test = _prepare_features(
                        sae, train_idx, test_idx, standardize=config.standardize
                    )
                    fold_records.append(
                        _score_probe(
                            layer_idx=layer_idx,
                            representation="Full SAE",
                            label=label,
                            fold=fold_id,
                            top_n=BASELINE_TOP_N,
                            x_train=full_train,
                            x_test=full_test,
                            y_train=y_train,
                            y_test=y_test,
                            split_policy=split_policy,
                            config=config,
                        )
                    )
                    del full_train, full_test

                pca_train, pca_test, pca_info = _fit_pca_100(raw, train_idx, test_idx, config)
                fold_records.append(
                    _score_probe(
                        layer_idx=layer_idx,
                        representation=f"PCA-{int(config.pca_components)}",
                        label=label,
                        fold=fold_id,
                        top_n=int(config.pca_components),
                        x_train=pca_train,
                        x_test=pca_test,
                        y_train=y_train,
                        y_test=y_test,
                        split_policy=split_policy,
                        config=config,
                        extra=pca_info,
                    )
                )

                selected, selected_d = _top_positive_cohens_d(sae, train_idx, y_train, keep, config)
                if selected.size < int(config.sae_top_n):
                    warnings.append(
                        f"layer {layer_idx}, {label}, fold {fold_id}: only {selected.size} positive Cohen's d candidates"
                    )
                top_train, top_test = _prepare_features(
                    sae, train_idx, test_idx, columns=selected, standardize=config.standardize
                )
                fold_records.append(
                    _score_probe(
                        layer_idx=layer_idx,
                        representation=f"Top-{int(config.sae_top_n)} SAE",
                        label=label,
                        fold=fold_id,
                        top_n=int(config.sae_top_n),
                        x_train=top_train,
                        x_test=top_test,
                        y_train=y_train,
                        y_test=y_test,
                        split_policy=split_policy,
                        config=config,
                        extra={
                            "effective_top_n": int(selected.size),
                            "candidate_pool_size": int(keep.size),
                        },
                    )
                )
                for rank, (latent_idx, cohens_d) in enumerate(zip(selected, selected_d), start=1):
                    selected_records.append(
                        {
                            "layer_idx": int(layer_idx),
                            "label": label,
                            "fold": int(fold_id),
                            "rank": int(rank),
                            "latent_idx": int(latent_idx),
                            "train_fold_cohens_d": float(cohens_d),
                            "candidate_pool_size": int(keep.size),
                        }
                    )

    fold_df = pd.DataFrame(fold_records)
    by_label = _summarize_by_label(fold_df)
    macro = _summarize_macro(by_label)
    selected_df = pd.DataFrame(selected_records)
    fold_df.to_csv(output_path / "probe_fold_metrics.csv", index=False, encoding="utf-8-sig")
    by_label.to_csv(output_path / "probe_summary_by_layer_label.csv", index=False, encoding="utf-8-sig")
    macro.to_csv(output_path / "probe_macro_summary_by_layer.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(output_path / "selected_top100_latents_by_layer_label.csv", index=False, encoding="utf-8-sig")
    report_path = _write_report(
        output_dir=output_path,
        macro=macro,
        by_label=by_label,
        filter_summaries=filter_summaries,
        config=config,
        warnings=warnings,
    )
    manifest = {
        "analysis": "llama_layerwise_pca100_top100_sae_probe",
        "config": asdict(config),
        "layers": sorted(int(layer) for layer in layer_matrices),
        "filter_summaries": filter_summaries,
        "outputs": {
            "fold_metrics": str(output_path / "probe_fold_metrics.csv"),
            "summary_by_layer_label": str(output_path / "probe_summary_by_layer_label.csv"),
            "macro_summary": str(output_path / "probe_macro_summary_by_layer.csv"),
            "selected_top100": str(output_path / "selected_top100_latents_by_layer_label.csv"),
            "report": str(report_path),
        },
        "warnings": sorted(set(warnings)),
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    return {
        "fold_metrics": fold_df,
        "summary_by_label": by_label,
        "macro_summary": macro,
        "selected_latents": selected_df,
        "filter_summaries": filter_summaries,
        "manifest": manifest,
    }
