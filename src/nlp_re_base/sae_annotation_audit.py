"""Out-of-fold disagreement audit for stable-core SAE probes."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None  # type: ignore[assignment]

from .baseline_comparison import load_matrix


DEFAULT_LABELS = ("RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF")
PARENT_LABELS = frozenset({"RE", "QU"})
REFERENCE_LABEL_PROVENANCE = (
    "label_matrix.csv reference labels derived from LLM-segmented MISC annotations; "
    "not independently verified human gold labels"
)


@dataclass(frozen=True)
class SAEAnnotationAuditConfig:
    labels: tuple[str, ...] = DEFAULT_LABELS
    folds: int = 5
    random_state: int = 42
    threshold: float = 0.5
    high_positive_threshold: float = 0.9
    high_negative_threshold: float = 0.1
    high_per_type: int = 10
    boundary_per_type: int = 5
    C: float = 1.0
    solver: str = "liblinear"
    max_iter: int = 1000
    group_column: str = "file_id"


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value).strip().lower())


def _stable_core_by_label(
    table: pd.DataFrame, labels: Iterable[str], feature_dim: int
) -> dict[str, np.ndarray]:
    required = {"label", "latent_idx", "stable_set_role"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"stable core table missing columns: {sorted(missing)}")
    out: dict[str, np.ndarray] = {}
    for label in labels:
        subset = table[
            table["label"].astype(str).str.upper().eq(label)
            & table["stable_set_role"].astype(str).eq("stable_core")
        ].copy()
        sort_columns = ["latent_idx"]
        if "full_data_rank" in subset.columns:
            subset["full_data_rank"] = pd.to_numeric(subset["full_data_rank"], errors="coerce")
            sort_columns = ["full_data_rank", "latent_idx"]
        latent_ids = subset.sort_values(sort_columns)["latent_idx"].astype(int).to_numpy()
        latent_ids = np.unique(latent_ids[(latent_ids >= 0) & (latent_ids < feature_dim)])
        if latent_ids.size == 0:
            raise ValueError(f"label {label} has no usable stable-core latents")
        out[label] = latent_ids.astype(np.int32)
    return out


def _make_splits(
    y: np.ndarray, groups: np.ndarray, config: SAEAnnotationAuditConfig
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    group_frame = pd.DataFrame({"group": groups, "y": y})
    by_group = group_frame.groupby("group", sort=False)["y"].agg(["sum", "count"])
    folds = min(
        config.folds,
        n_pos,
        n_neg,
        int((by_group["sum"] > 0).sum()),
        int(((by_group["count"] - by_group["sum"]) > 0).sum()),
    )
    if folds < 2:
        raise ValueError("not enough positive and negative groups for grouped cross-validation")
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=config.random_state
        )
        return list(splitter.split(np.zeros(len(y)), y, groups)), "stratified-group-kfold"
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.random_state)
    return list(splitter.split(np.zeros(len(y)), y)), "stratified-kfold-fallback"


def generate_oof_predictions(
    *,
    features: np.ndarray,
    label_matrix: pd.DataFrame,
    stable_core: pd.DataFrame,
    config: SAEAnnotationAuditConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if len(label_matrix) != features.shape[0]:
        raise ValueError("feature and label rows must align")
    missing = [label for label in config.labels if label not in label_matrix.columns]
    if missing:
        raise ValueError(f"label matrix missing labels: {missing}")
    if config.group_column not in label_matrix.columns:
        raise ValueError(f"label matrix missing group column {config.group_column!r}")

    stable_by_label = _stable_core_by_label(stable_core, config.labels, features.shape[1])
    groups = label_matrix[config.group_column].fillna("UNKNOWN").astype(str).to_numpy()
    base_columns = [
        "row_idx", "record_id", "file_id", "source_split", "source_file",
        "predicted_code", "predicted_subcode", "confidence", "unit_text",
    ]
    base = label_matrix.reindex(columns=base_columns).copy()
    if "row_idx" not in label_matrix.columns:
        base["row_idx"] = np.arange(len(label_matrix), dtype=np.int32)

    rows: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    for label in config.labels:
        y = label_matrix[label].astype(int).to_numpy()
        columns = stable_by_label[label]
        splits, split_policy = _make_splits(y, groups, config)
        probabilities = np.full(len(y), np.nan, dtype=np.float64)
        fold_ids = np.full(len(y), -1, dtype=np.int16)
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            overlap = train_groups.intersection(test_groups)
            if split_policy == "stratified-group-kfold" and overlap:
                raise AssertionError(f"group leakage for {label} fold {fold}: {sorted(overlap)[:3]}")
            scaler = StandardScaler()
            x_train = scaler.fit_transform(
                np.asarray(features[np.ix_(train_idx, columns)], dtype=np.float32)
            )
            x_test = scaler.transform(
                np.asarray(features[np.ix_(test_idx, columns)], dtype=np.float32)
            )
            classifier = LogisticRegression(
                C=config.C,
                solver=config.solver,
                class_weight="balanced",
                random_state=config.random_state,
                max_iter=config.max_iter,
            )
            classifier.fit(x_train, y[train_idx])
            probabilities[test_idx] = classifier.predict_proba(x_test)[:, 1]
            fold_ids[test_idx] = fold
            fold_records.append(
                {
                    "label": label,
                    "fold": fold,
                    "split_policy": split_policy,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_train_groups": int(len(train_groups)),
                    "n_test_groups": int(len(test_groups)),
                    "group_overlap_count": int(len(overlap)),
                    "n_features": int(len(columns)),
                }
            )
        if np.isnan(probabilities).any() or (fold_ids < 1).any():
            raise AssertionError(f"incomplete OOF coverage for {label}")
        frame = base.copy()
        frame["label"] = label
        frame["reference_label"] = y
        frame["sae_probability"] = probabilities
        frame["sae_prediction"] = (probabilities >= config.threshold).astype(int)
        frame["fold"] = fold_ids
        frame["threshold"] = config.threshold
        frame["score_margin"] = np.abs(probabilities - config.threshold)
        frame["n_stable_core_latents"] = len(columns)
        frame["split_policy"] = split_policy
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["row_idx", "label"]).any():
        raise AssertionError("duplicate row-label OOF predictions")
    return result, fold_records


def build_disagreement_candidates(
    oof: pd.DataFrame,
    label_matrix: pd.DataFrame,
    config: SAEAnnotationAuditConfig,
) -> pd.DataFrame:
    all_labels = label_matrix.loc[:, list(config.labels)].astype(int).astype(str)
    all_label_strings = all_labels.apply(
        lambda row: "|".join(label for label, value in row.items() if value == "1"), axis=1
    )
    normalized = label_matrix["unit_text"].map(_normalize_text)
    duplicate_counts = normalized.map(normalized.value_counts()).astype(int)
    row_metadata = pd.DataFrame(
        {
            "row_idx": label_matrix.get("row_idx", pd.Series(np.arange(len(label_matrix)))).astype(int),
            "all_reference_labels": all_label_strings,
            "normalized_duplicate_count": duplicate_counts,
        }
    )
    candidates = oof[oof["reference_label"].ne(oof["sae_prediction"])].copy()
    candidates["disagreement_type"] = np.where(
        candidates["reference_label"].eq(1), "false_negative", "false_positive"
    )
    candidates["high_confidence_flag"] = np.where(
        candidates["disagreement_type"].eq("false_negative"),
        candidates["sae_probability"].le(config.high_negative_threshold),
        candidates["sae_probability"].ge(config.high_positive_threshold),
    )
    candidates["hierarchy_audit_flag"] = candidates["label"].isin(PARENT_LABELS)
    candidates["context_limited_flag"] = candidates["label"].isin({"RE", "RES", "REC"})
    candidates = candidates.merge(row_metadata, on="row_idx", how="left", validate="many_to_one")
    return candidates.sort_values(
        ["label", "disagreement_type", "score_margin", "row_idx"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def sample_review_cases(
    candidates: pd.DataFrame, config: SAEAnnotationAuditConfig
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for label in config.labels:
        for disagreement_type in ("false_negative", "false_positive"):
            group = candidates[
                candidates["label"].eq(label)
                & candidates["disagreement_type"].eq(disagreement_type)
            ].copy()
            high = group.sort_values(
                ["score_margin", "row_idx"], ascending=[False, True]
            ).head(config.high_per_type).copy()
            high["sampling_stratum"] = "highest_margin"
            remaining = group[~group["row_idx"].isin(high["row_idx"])].copy()
            boundary = remaining.sort_values(
                ["score_margin", "row_idx"], ascending=[True, True]
            ).head(config.boundary_per_type).copy()
            boundary["sampling_stratum"] = "near_boundary"
            selected.extend([high, boundary])
    review = pd.concat(selected, ignore_index=True) if selected else candidates.head(0).copy()
    review = review.sort_values(
        ["label", "disagreement_type", "sampling_stratum", "score_margin"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    review.insert(0, "case_id", [f"audit_{index + 1:04d}" for index in range(len(review))])
    review["reviewer_target_label"] = review["label"]
    review["adjudicated_label"] = ""
    review["primary_assessment"] = ""
    review["secondary_assessment"] = ""
    review["mixed_behavior_labels"] = ""
    review["surface_trigger"] = ""
    review["missing_context_flag"] = ""
    review["review_rationale"] = ""
    review["recommended_action"] = ""
    review["reviewer_id"] = ""
    review["review_timestamp"] = ""
    review["adjudication_status"] = "pending"
    review_first = [
        "case_id",
        "reviewer_target_label",
        "unit_text",
        "adjudicated_label",
        "primary_assessment",
        "secondary_assessment",
        "mixed_behavior_labels",
        "missing_context_flag",
        "review_rationale",
        "recommended_action",
        "reviewer_id",
        "review_timestamp",
        "adjudication_status",
        "row_idx",
        "record_id",
        "file_id",
        "label",
        "reference_label",
        "all_reference_labels",
        "sae_prediction",
        "sae_probability",
        "score_margin",
        "disagreement_type",
        "high_confidence_flag",
        "sampling_stratum",
        "surface_trigger",
    ]
    return review.loc[:, review_first + [column for column in review.columns if column not in review_first]]


def _write_guidelines(path: Path, config: SAEAnnotationAuditConfig) -> None:
    path.write_text(
        f"""# SAE 标注不一致审查指南

## 标签来源

当前 reference label 来自 `{REFERENCE_LABEL_PROVENANCE}`。在没有独立人工 gold 文件前，不得把它直接称为人工真值。

## 不一致定义

- false negative：reference=1 且 SAE probability < {config.threshold}
- false positive：reference=0 且 SAE probability >= {config.threshold}
- 高模型分数 FN：probability <= {config.high_negative_threshold}
- 高模型分数 FP：probability >= {config.high_positive_threshold}

这些概率来自 balanced logistic regression，未经独立校准；`high_confidence_flag` 仅表示高模型分数。

## 盲审顺序

先阅读 `unit_text` 和目标 `label`，给出初步判断后再查看 SAE probability。依次判断：目标行为是否明确、是否混合行为、参考标签是否可能遗漏或误标、是否缺少上下文、模型是否依赖表面模式。

## primary_assessment 枚举

`reference_label_supported`, `probable_annotation_error`, `label_ambiguity`, `mixed_behavior`, `model_limitation`, `threshold_boundary`, `surface_artifact`, `duplicate_template_artifact`, `insufficient_context`, `other`。

## adjudicated_label 枚举

`positive`, `negative`, `uncertain`, `context_missing`。

## recommended_action 枚举

`keep_reference_label`, `change_reference_label`, `add_secondary_label`, `exclude_as_ambiguous`, `request_context_review`, `retain_as_model_error`, `investigate_surface_artifact`。

## 上下文和层级限制

- `RE`、`QU` 是父标签，只作层级审查。
- 当前 unit_text 没有前一句 client utterance。`RE/RES/REC` 无法仅凭当前文本确认时，使用 `insufficient_context`，不得直接判为标注错误。
- 一条文本可包含多个行为；此时使用 `mixed_behavior` 并填写 `mixed_behavior_labels`。
""",
        encoding="utf-8",
    )


def run_sae_annotation_audit(
    *,
    feature_store_path: str | Path,
    label_matrix_path: str | Path,
    stable_core_path: str | Path,
    output_dir: str | Path,
    config: SAEAnnotationAuditConfig = SAEAnnotationAuditConfig(),
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    label_matrix = pd.read_csv(label_matrix_path)
    features = np.asarray(load_matrix(feature_store_path), dtype=np.float32)
    stable_core = pd.read_csv(stable_core_path)

    oof, fold_records = generate_oof_predictions(
        features=features,
        label_matrix=label_matrix,
        stable_core=stable_core,
        config=config,
    )
    candidates = build_disagreement_candidates(oof, label_matrix, config)
    review = sample_review_cases(candidates, config)
    summary = (
        candidates.groupby(["label", "disagreement_type"], as_index=False)
        .agg(
            n_cases=("row_idx", "count"),
            n_high_model_score=("high_confidence_flag", "sum"),
            mean_probability=("sae_probability", "mean"),
            mean_score_margin=("score_margin", "mean"),
        )
    )

    oof.to_csv(output / "oof_sae_predictions.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(output / "disagreement_candidates.csv", index=False, encoding="utf-8-sig")
    review.to_csv(output / "annotation_review_table.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "disagreement_summary_by_label.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_records).to_csv(output / "fold_integrity.csv", index=False, encoding="utf-8-sig")
    _write_guidelines(output / "annotation_review_guidelines.md", config)

    manifest = {
        "analysis": "stable_core_sae_annotation_disagreement_audit",
        "reference_label_provenance": REFERENCE_LABEL_PROVENANCE,
        "inputs": {
            "feature_store": str(feature_store_path),
            "label_matrix": str(label_matrix_path),
            "stable_core": str(stable_core_path),
        },
        "parameters": asdict(config),
        "counts": {
            "n_samples": int(len(label_matrix)),
            "n_labels": int(len(config.labels)),
            "n_oof_rows": int(len(oof)),
            "n_disagreements": int(len(candidates)),
            "n_review_cases": int(len(review)),
        },
        "probability_caveat": "balanced logistic regression probabilities are not independently calibrated",
        "outputs": {
            "oof_predictions": "oof_sae_predictions.csv",
            "candidates": "disagreement_candidates.csv",
            "review_table": "annotation_review_table.csv",
            "summary": "disagreement_summary_by_label.csv",
            "fold_integrity": "fold_integrity.csv",
            "guidelines": "annotation_review_guidelines.md",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


__all__ = [
    "DEFAULT_LABELS",
    "REFERENCE_LABEL_PROVENANCE",
    "SAEAnnotationAuditConfig",
    "build_disagreement_candidates",
    "generate_oof_predictions",
    "run_sae_annotation_audit",
    "sample_review_cases",
]
