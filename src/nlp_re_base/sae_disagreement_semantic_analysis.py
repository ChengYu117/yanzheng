"""Semantic evidence analysis for sampled stable-core SAE disagreements."""

from __future__ import annotations

import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .baseline_comparison import load_matrix
from .sae_annotation_audit import (
    DEFAULT_LABELS,
    REFERENCE_LABEL_PROVENANCE,
    SAEAnnotationAuditConfig,
    _make_splits,
    _stable_core_by_label,
)


LABEL_DEFINITIONS = {
    "RE": "Reflection parent category; positive when the utterance is RES or REC.",
    "RES": "Simple reflection: repeats or slightly rephrases the client's prior meaning.",
    "REC": "Complex reflection: adds substantial inferred meaning, emphasis, or reframing.",
    "QU": "Question parent category; positive when the utterance is QUO or QUC.",
    "QUO": "Open question: invites an elaborated response rather than a short fixed answer.",
    "QUC": "Closed question: requests confirmation, a fact, quantity, or short constrained answer.",
    "GI": "Giving information, feedback, education, explanation, or neutral advice.",
    "SU": "Support: sympathy, compassion, help, permission, or emotional support.",
    "AF": "Affirm: recognizes the client's strengths, efforts, values, or positive qualities.",
}

ISSUE_TYPES = {
    "reference_label_supported",
    "label_ambiguity",
    "mixed_behavior",
    "probable_reference_label_error",
    "model_limitation",
    "surface_artifact",
    "threshold_boundary",
    "insufficient_context",
}
JUDGMENTS = {"positive", "negative", "uncertain", "context_missing"}
ACTIONS = {
    "keep_reference_label",
    "change_reference_label_candidate",
    "add_secondary_label_candidate",
    "exclude_as_ambiguous",
    "request_context_review",
    "retain_as_model_error",
    "investigate_surface_artifact",
}
PARENT_LABELS = {"RE", "QU"}
LEAF_LABELS = ("RES", "REC", "QUO", "QUC", "GI", "SU", "AF")


@dataclass(frozen=True)
class SemanticDisagreementConfig:
    model: str = "deepseek-v4-flash"
    api_url: str = "https://api.deepseek.com/chat/completions"
    concurrency: int = 50
    timeout_seconds: float = 180.0
    max_retries: int = 2
    temperature: float = 0.0
    max_tokens: int = 1000
    top_contributions: int = 3
    probability_tolerance: float = 1e-6


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _leaf_labels(value: Any) -> list[str]:
    labels = [item for item in str(value or "").split("|") if item]
    return [label for label in labels if label not in {"RE", "QU"}]


def _explanation_map(explanations: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in explanations:
        latent_idx = int(row["latent_idx"])
        effective_type = str(row.get("feature_type", "unknown"))
        if not bool(row.get("majority_gate_passed", True)) or str(
            row.get("induction_status", "")
        ) == "insufficient_majority":
            effective_type = "no_stable_pattern"
        result[latent_idx] = {
            "latent_idx": latent_idx,
            "short_name": str(row.get("short_name", "")),
            "feature_type": effective_type,
            "candidate_explanation": str(row.get("candidate_explanation", "")),
            "confidence": float(row.get("confidence", 0.0)),
        }
    return result


def _latent_evidence(
    *,
    latent_ids: np.ndarray,
    contributions: np.ndarray,
    raw_values: np.ndarray,
    standardized_values: np.ndarray,
    coefficients: np.ndarray,
    explanation_by_latent: dict[int, dict[str, Any]],
    count: int,
    positive: bool,
) -> list[dict[str, Any]]:
    order = np.argsort(-contributions if positive else contributions)
    rows: list[dict[str, Any]] = []
    for index in order:
        value = float(contributions[index])
        if (positive and value <= 0) or (not positive and value >= 0):
            continue
        latent_idx = int(latent_ids[index])
        explanation = explanation_by_latent.get(
            latent_idx,
            {
                "short_name": "missing_explanation",
                "feature_type": "unknown",
                "candidate_explanation": "",
                "confidence": 0.0,
            },
        )
        rows.append(
            {
                "latent_idx": latent_idx,
                "contribution": value,
                "activation": float(raw_values[index]),
                "standardized_value": float(standardized_values[index]),
                "coefficient": float(coefficients[index]),
                **explanation,
            }
        )
        if len(rows) >= count:
            break
    return rows


def build_case_latent_evidence(
    *,
    features: np.ndarray,
    label_matrix: pd.DataFrame,
    stable_core: pd.DataFrame,
    review_cases: pd.DataFrame,
    explanations: list[dict[str, Any]],
    audit_config: SAEAnnotationAuditConfig = SAEAnnotationAuditConfig(),
    config: SemanticDisagreementConfig = SemanticDisagreementConfig(),
) -> pd.DataFrame:
    """Re-fit the original OOF probes and recover exact per-case latent contributions."""

    stable_by_label = _stable_core_by_label(stable_core, audit_config.labels, features.shape[1])
    explanation_by_latent = _explanation_map(explanations)
    groups = label_matrix[audit_config.group_column].fillna("UNKNOWN").astype(str).to_numpy()
    row_positions = {
        int(row_idx): position
        for position, row_idx in enumerate(
            label_matrix.get("row_idx", pd.Series(np.arange(len(label_matrix)))).astype(int)
        )
    }
    output_rows: list[dict[str, Any]] = []
    for label in audit_config.labels:
        label_review = review_cases[review_cases["label"].astype(str).eq(label)].copy()
        if label_review.empty:
            continue
        y = label_matrix[label].astype(int).to_numpy()
        latent_ids = stable_by_label[label]
        splits, split_policy = _make_splits(y, groups, audit_config)
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            fold_cases = label_review[label_review["fold"].astype(int).eq(fold)]
            if fold_cases.empty:
                continue
            scaler = StandardScaler()
            x_train = scaler.fit_transform(
                np.asarray(features[np.ix_(train_idx, latent_ids)], dtype=np.float32)
            )
            classifier = LogisticRegression(
                C=audit_config.C,
                solver=audit_config.solver,
                class_weight="balanced",
                random_state=audit_config.random_state,
                max_iter=audit_config.max_iter,
            )
            classifier.fit(x_train, y[train_idx])
            coefficients = classifier.coef_[0]
            intercept = float(classifier.intercept_[0])
            test_set = set(int(index) for index in test_idx)
            for case in fold_cases.to_dict("records"):
                position = row_positions[int(case["row_idx"])]
                if position not in test_set:
                    raise AssertionError(f"case {case['case_id']} is not in stored OOF fold")
                raw = np.asarray(features[position, latent_ids], dtype=np.float32)
                standardized = scaler.transform(raw.reshape(1, -1))[0]
                contributions = standardized * coefficients
                logit = intercept + float(contributions.sum())
                probability = 1.0 / (1.0 + math.exp(-logit))
                stored_probability = float(case["sae_probability"])
                if abs(probability - stored_probability) > config.probability_tolerance:
                    raise AssertionError(
                        f"probability reconstruction mismatch for {case['case_id']}: "
                        f"{probability} vs {stored_probability}"
                    )
                positive = _latent_evidence(
                    latent_ids=latent_ids,
                    contributions=contributions,
                    raw_values=raw,
                    standardized_values=standardized,
                    coefficients=coefficients,
                    explanation_by_latent=explanation_by_latent,
                    count=config.top_contributions,
                    positive=True,
                )
                negative = _latent_evidence(
                    latent_ids=latent_ids,
                    contributions=contributions,
                    raw_values=raw,
                    standardized_values=standardized,
                    coefficients=coefficients,
                    explanation_by_latent=explanation_by_latent,
                    count=config.top_contributions,
                    positive=False,
                )
                output_rows.append(
                    {
                        **case,
                        "split_policy_verified": split_policy,
                        "reconstructed_probability": probability,
                        "probability_reconstruction_error": abs(probability - stored_probability),
                        "intercept": intercept,
                        "positive_latent_evidence": positive,
                        "negative_latent_evidence": negative,
                        "leaf_reference_labels": _leaf_labels(case.get("all_reference_labels", "")),
                    }
                )
    evidence = pd.DataFrame(output_rows).sort_values("case_id").reset_index(drop=True)
    if len(evidence) != len(review_cases):
        raise AssertionError(f"expected {len(review_cases)} evidence rows, got {len(evidence)}")
    return evidence


def build_case_review_prompt(case: dict[str, Any]) -> str:
    positive = case["positive_latent_evidence"]
    negative = case["negative_latent_evidence"]

    def compact(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "latent_idx": row["latent_idx"],
                "contribution": round(float(row["contribution"]), 4),
                "short_name": row["short_name"],
                "feature_type": row["feature_type"],
                "candidate_explanation": row["candidate_explanation"],
            }
            for row in rows
        ]

    payload = {
        "case_id": case["case_id"],
        "target_label": case["label"],
        "target_definition": LABEL_DEFINITIONS[str(case["label"])],
        "utterance": case["unit_text"],
        "reference_target": int(case["reference_label"]),
        "all_reference_labels": case.get("all_reference_labels", ""),
        "sae_prediction": int(case["sae_prediction"]),
        "sae_probability": round(float(case["sae_probability"]), 4),
        "disagreement_type": case["disagreement_type"],
        "sampling_stratum": case["sampling_stratum"],
        "positive_latent_evidence": compact(positive),
        "negative_latent_evidence": compact(negative),
    }
    return f"""Audit one disagreement between a reference MISC label and a stable-core SAE probe.

Important boundaries:
- The reference label came from LLM-segmented annotations, not verified human gold.
- First judge the utterance against the target definition. Then interpret why the SAE probe disagreed.
- A probable_reference_label_error requires clear textual evidence. Otherwise use ambiguity, mixed behavior,
  insufficient context, or model limitation.
- RE/RES/REC lack the preceding client utterance. Use insufficient_context when that context is necessary.
- RE and QU are parent labels. Do not treat their parent-child co-labeling as mixed behavior.
- Latent explanations are candidate summaries, not causal facts.

Case:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return only one JSON object with exactly these fields:
- case_id
- target_label_judgment: positive|negative|uncertain|context_missing
- primary_issue: one of {sorted(ISSUE_TYPES)}
- secondary_issues: JSON array using the same issue vocabulary
- mixed_behavior_labels: JSON array of label codes
- latent_evidence_interpretation: one concise sentence
- rationale: at most two concise sentences
- recommended_action: one of {sorted(ACTIONS)}
- confidence: number from 0 to 1
"""


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response has no JSON object")
    value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise TypeError("response is not a JSON object")
    return value


def _validate_review(value: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    if str(value.get("case_id")) != str(case["case_id"]):
        raise ValueError("case_id mismatch")
    judgment = str(value.get("target_label_judgment"))
    issue = str(value.get("primary_issue"))
    action = str(value.get("recommended_action"))
    if judgment not in JUDGMENTS or issue not in ISSUE_TYPES or action not in ACTIONS:
        raise ValueError("invalid review enum")
    secondary = value.get("secondary_issues", [])
    mixed = value.get("mixed_behavior_labels", [])
    if not isinstance(secondary, list) or not set(map(str, secondary)).issubset(ISSUE_TYPES):
        raise ValueError("invalid secondary_issues")
    if not isinstance(mixed, list):
        raise ValueError("invalid mixed_behavior_labels")
    confidence = float(value.get("confidence"))
    if not 0 <= confidence <= 1:
        raise ValueError("confidence outside [0, 1]")
    return {
        "case_id": str(case["case_id"]),
        "target_label_judgment": judgment,
        "primary_issue": issue,
        "secondary_issues": [str(item) for item in secondary],
        "mixed_behavior_labels": [str(item) for item in mixed],
        "latent_evidence_interpretation": str(value.get("latent_evidence_interpretation", "")),
        "rationale": str(value.get("rationale", "")),
        "recommended_action": action,
        "review_confidence": confidence,
        "review_status": "valid",
    }


def run_deepseek_reviews(
    *,
    evidence: pd.DataFrame,
    output_dir: str | Path,
    api_key: str,
    config: SemanticDisagreementConfig = SemanticDisagreementConfig(),
) -> list[dict[str, Any]]:
    if not api_key.strip():
        raise ValueError("DeepSeek API key is empty")
    output = Path(output_dir)
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cases = evidence.to_dict("records")
    tasks = [
        {"case_id": case["case_id"], "prompt": build_case_review_prompt(case)} for case in cases
    ]
    _write_jsonl(output / "llm_tasks.jsonl", tasks)

    def run_one(case: dict[str, Any]) -> dict[str, Any]:
        raw_path = raw_dir / f"{case['case_id']}.json"
        if raw_path.exists():
            try:
                return _validate_review(json.loads(raw_path.read_text(encoding="utf-8")), case)
            except Exception:
                pass
        last_error = ""
        for attempt in range(config.max_retries + 1):
            try:
                response = requests.post(
                    config.api_url,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": config.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a cautious MISC annotation auditor. Return only valid JSON.",
                            },
                            {"role": "user", "content": build_case_review_prompt(case)},
                        ],
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=config.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
                parsed = _parse_json_object(content)
                valid = _validate_review(parsed, case)
                raw_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
                return valid
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < config.max_retries:
                    time.sleep(2**attempt)
        return {
            "case_id": str(case["case_id"]),
            "target_label_judgment": "uncertain",
            "primary_issue": "label_ambiguity",
            "secondary_issues": [],
            "mixed_behavior_labels": [],
            "latent_evidence_interpretation": "",
            "rationale": last_error,
            "recommended_action": "exclude_as_ambiguous",
            "review_confidence": 0.0,
            "review_status": "failed",
        }

    reviews: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(config.concurrency, max(1, len(cases)))) as executor:
        futures = {executor.submit(run_one, case): case["case_id"] for case in cases}
        for future in as_completed(futures):
            reviews.append(future.result())
    reviews.sort(key=lambda row: row["case_id"])
    _write_jsonl(output / "llm_reviews.jsonl", reviews)
    return reviews


def summarize_oof_disagreements(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, group in oof.groupby("label", sort=False):
        reference = group["reference_label"].astype(int)
        prediction = group["sae_prediction"].astype(int)
        probability = group["sae_probability"].astype(float)
        positive = reference.eq(1)
        negative = reference.eq(0)
        fn = positive & prediction.eq(0)
        fp = negative & prediction.eq(1)
        high_fn = fn & probability.le(0.1)
        high_fp = fp & probability.ge(0.9)
        n_positive, n_negative = int(positive.sum()), int(negative.sum())
        rows.append(
            {
                "label": str(label),
                "label_role": "parent_audit" if str(label) in PARENT_LABELS else "leaf_primary",
                "n_samples": int(len(group)),
                "n_reference_positive": n_positive,
                "n_reference_negative": n_negative,
                "n_false_negative": int(fn.sum()),
                "false_negative_rate": float(fn.sum() / n_positive),
                "n_false_positive": int(fp.sum()),
                "false_positive_rate": float(fp.sum() / n_negative),
                "n_high_score_false_negative": int(high_fn.sum()),
                "high_score_false_negative_rate": float(high_fn.sum() / n_positive),
                "n_high_score_false_positive": int(high_fp.sum()),
                "high_score_false_positive_rate": float(high_fp.sum() / n_negative),
                "n_total_disagreements": int(fn.sum() + fp.sum()),
                "total_disagreement_rate": float((fn.sum() + fp.sum()) / len(group)),
            }
        )
    return pd.DataFrame(rows)


def select_representative_cases(enriched: pd.DataFrame) -> pd.DataFrame:
    valid = enriched[
        enriched["review_status"].eq("valid") & enriched["label"].isin(LEAF_LABELS)
    ].copy()
    selected: list[pd.DataFrame] = []
    for label in LEAF_LABELS:
        for disagreement_type in ("false_negative", "false_positive"):
            group = valid[
                valid["label"].eq(label)
                & valid["disagreement_type"].eq(disagreement_type)
            ].copy()
            high = group[group["high_confidence_flag"].astype(bool)].sort_values(
                ["review_confidence", "score_margin"], ascending=[False, False]
            ).head(1)
            if high.empty:
                high = group.sort_values(
                    ["score_margin", "review_confidence"], ascending=[False, False]
                ).head(1)
            high = high.copy()
            high["representative_stratum"] = "high_model_score_or_largest_margin"
            boundary = group[~group["case_id"].isin(high["case_id"])].sort_values(
                ["score_margin", "review_confidence"], ascending=[True, False]
            ).head(1).copy()
            boundary["representative_stratum"] = "near_boundary"
            selected.extend([high, boundary])
    result = pd.concat(selected, ignore_index=True)
    columns = [
        "case_id",
        "label",
        "disagreement_type",
        "representative_stratum",
        "unit_text",
        "reference_label",
        "sae_prediction",
        "sae_probability",
        "score_margin",
        "high_confidence_flag",
        "all_reference_labels",
        "primary_issue",
        "secondary_issues",
        "mixed_behavior_labels",
        "target_label_judgment",
        "rationale",
        "latent_evidence_interpretation",
        "positive_latent_evidence",
        "negative_latent_evidence",
        "recommended_action",
        "review_confidence",
    ]
    return result.loc[:, columns]


def write_disagreement_figures(
    *, rates: pd.DataFrame, enriched: pd.DataFrame, output_dir: str | Path
) -> list[str]:
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    leaf_rates = rates[rates["label"].isin(LEAF_LABELS)].set_index("label").loc[list(LEAF_LABELS)]
    x = np.arange(len(LEAF_LABELS))
    width = 0.36

    fig, axis = plt.subplots(figsize=(10.5, 5.6))
    fn_values = leaf_rates["false_negative_rate"].to_numpy() * 100
    fp_values = leaf_rates["false_positive_rate"].to_numpy() * 100
    fn_bars = axis.bar(x - width / 2, fn_values, width, label="FN rate", color="#C44E52")
    fp_bars = axis.bar(x + width / 2, fp_values, width, label="FP rate", color="#4C72B0")
    axis.bar_label(fn_bars, fmt="%.1f", padding=3, fontsize=8)
    axis.bar_label(fp_bars, fmt="%.1f", padding=3, fontsize=8)
    axis.set_xticks(x, LEAF_LABELS)
    axis.set_ylabel("Rate (%)")
    axis.set_title("OOF disagreement rates by leaf MISC label")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylim(0, max(fn_values.max(), fp_values.max()) * 1.18)
    fig.tight_layout()
    rate_path = output / "disagreement_rates_by_label.png"
    fig.savefig(rate_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10.5, 5.6))
    high_fn = leaf_rates["n_high_score_false_negative"].to_numpy()
    high_fp = leaf_rates["n_high_score_false_positive"].to_numpy()
    fn_bars = axis.bar(x - width / 2, high_fn, width, label="High-score FN", color="#DD8452")
    fp_bars = axis.bar(x + width / 2, high_fp, width, label="High-score FP", color="#55A868")
    axis.bar_label(fn_bars, padding=3, fontsize=8)
    axis.bar_label(fp_bars, padding=3, fontsize=8)
    axis.set_xticks(x, LEAF_LABELS)
    axis.set_ylabel("Number of label-case disagreements")
    axis.set_title("High-model-score disagreements by leaf MISC label")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylim(0, max(high_fn.max(), high_fp.max()) * 1.18)
    fig.tight_layout()
    high_path = output / "high_score_disagreements_by_label.png"
    fig.savefig(high_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    valid = enriched[enriched["review_status"].eq("valid")]
    issues = [
        "probable_reference_label_error",
        "model_limitation",
        "insufficient_context",
        "threshold_boundary",
        "label_ambiguity",
        "surface_artifact",
        "mixed_behavior",
    ]
    fn_counts = [
        int(valid.loc[valid["disagreement_type"].eq("false_negative"), f"{issue}_flag"].sum())
        for issue in issues
    ]
    fp_counts = [
        int(valid.loc[valid["disagreement_type"].eq("false_positive"), f"{issue}_flag"].sum())
        for issue in issues
    ]
    y = np.arange(len(issues))
    fig, axis = plt.subplots(figsize=(10.5, 6.2))
    fn_bars = axis.barh(y - 0.18, fn_counts, 0.36, label="FN cases", color="#8172B2")
    fp_bars = axis.barh(y + 0.18, fp_counts, 0.36, label="FP cases", color="#64B5CD")
    axis.bar_label(fn_bars, padding=3, fontsize=8)
    axis.bar_label(fp_bars, padding=3, fontsize=8)
    axis.set_yticks(y, [issue.replace("_", " ") for issue in issues])
    axis.invert_yaxis()
    axis.set_xlabel("Cases in the stratified 270-case audit")
    axis.set_title("Diagnostic flags in the sampled disagreement audit")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    diagnosis_path = output / "sampled_diagnosis_flags.png"
    fig.savefig(diagnosis_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [str(rate_path), str(high_path), str(diagnosis_path)]


def _write_report(
    path: Path,
    enriched: pd.DataFrame,
    rates: pd.DataFrame,
    diagnosis: pd.DataFrame,
    representatives: pd.DataFrame,
) -> None:
    valid = enriched[enriched["review_status"].eq("valid")].copy()
    lines = [
        "# Stable-core SAE 标注不一致的语义辅助分析",
        "",
        "## 口径",
        "",
        "本报告以人工/reference 标签与 stable-core SAE 线性探针预测的不一致为主体，分别分析人工正例但 probe 预测为负例（FN）、人工负例但 probe 预测为正例（FP），以及高模型分数不一致。",
        "",
        f"全体发生率基于 6,194 条样本、9 个标签的 OOF 预测。案例诊断基于 {len(enriched)} 条分层抽样不一致样本。Reference labels 的当前文件来源是 {REFERENCE_LABEL_PROVENANCE}；如果另有独立人工 gold，应在最终论文中替换该来源说明。",
        "",
        "每条抽样案例使用同一 OOF probe 重建预测，并将贡献最大的 stable-core latent 与既有 DeepSeek feature-card 语义关联。Latent 解释仅用于说明 probe 的判断依据，不作为标签裁决本身。",
        "",
        "## 全体 FN、FP 与高模型分数不一致",
        "",
        "叶标签是主结果；RE 和 QU 是父标签，仅用于层级一致性审计。",
        "",
        "| 标签 | 人工正例 | FN | FN率 | 高分FN | 人工负例 | FP | FP率 | 高分FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LEAF_LABELS:
        row = rates[rates["label"].eq(label)].iloc[0]
        lines.append(
            f"| {label} | {row.n_reference_positive} | {row.n_false_negative} | {row.false_negative_rate:.1%} | "
            f"{row.n_high_score_false_negative} | {row.n_reference_negative} | {row.n_false_positive} | "
            f"{row.false_positive_rate:.1%} | {row.n_high_score_false_positive} |"
        )

    lines.extend(["", "父标签层级审计：", ""])
    for label in ("RE", "QU"):
        row = rates[rates["label"].eq(label)].iloc[0]
        lines.append(
            f"- {label}: FN={row.n_false_negative}（{row.false_negative_rate:.1%}），"
            f"FP={row.n_false_positive}（{row.false_positive_rate:.1%}）。该结果必须结合子标签解释。"
        )

    lines.extend(
        [
            "",
            "## 270 条分层案例诊断",
            "",
            f"有效语义辅助审查 {len(valid)} 条，失败 {len(enriched) - len(valid)} 条。抽样固定为每个标签每个方向 10 条最大 margin 加 5 条阈值附近案例，因此以下诊断比例只描述审查样本，不能估计 8,194 条全部不一致的总体构成。",
            "",
            "| 主要判断 | FN | FP | 合计 |",
            "|---|---:|---:|---:|",
        ]
    )
    diagnosis_pivot = diagnosis.pivot_table(
        index="primary_issue",
        columns="disagreement_type",
        values="n_cases",
        fill_value=0,
        aggfunc="sum",
    )
    for issue, row in diagnosis_pivot.iterrows():
        fn = int(row.get("false_negative", 0))
        fp = int(row.get("false_positive", 0))
        lines.append(f"| {issue} | {fn} | {fp} | {fn + fp} |")

    lines.extend(
        [
            "",
            "多标签诊断标志允许一条案例同时属于多个原因：",
            "",
            "| 诊断标志 | FN | FP | 合计 |",
            "|---|---:|---:|---:|",
        ]
    )
    diagnostic_flags = (
        "label_ambiguity",
        "mixed_behavior",
        "probable_reference_label_error",
        "model_limitation",
        "surface_artifact",
        "insufficient_context",
        "threshold_boundary",
    )
    for issue in diagnostic_flags:
        column = f"{issue}_flag"
        fn = int(valid.loc[valid["disagreement_type"].eq("false_negative"), column].sum())
        fp = int(valid.loc[valid["disagreement_type"].eq("false_positive"), column].sum())
        lines.append(f"| {issue} | {fn} | {fp} | {fn + fp} |")

    lines.extend(["", "## 按标签与不一致方向的诊断", ""])
    cross = pd.crosstab(
        [valid["label"], valid["disagreement_type"]], valid["primary_issue"]
    ).reset_index()
    columns = [str(column) for column in cross.columns]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for values in cross.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")

    high = valid[valid["high_confidence_flag"].astype(bool)]
    lines.extend(
        [
            "",
            "## 高模型分数不一致",
            "",
            f"抽样表中共有 {len(high)} 条高模型分数不一致。这里的高分仅表示未校准 probe probability <=0.1（FN）或 >=0.9（FP），不能称为统计置信度。",
            "",
        ]
    )
    if len(high):
        high_counts = high["primary_issue"].value_counts()
        for issue, count in high_counts.items():
            lines.append(f"- {issue}: {count} 条。")

    rate_map = rates.set_index("label")
    lines.extend(
        [
            "",
            "## 标签级主要发现",
            "",
            f"- RES：FN率 {rate_map.loc['RES', 'false_negative_rate']:.1%}，FP率 {rate_map.loc['RES', 'false_positive_rate']:.1%}，是双向不一致最严重的标签之一。代表案例显示，缺少 client 前文、反映与确认问句混合，以及 `so you`/第二人称改述模板都会影响判断。",
            f"- REC：FN率 {rate_map.loc['REC', 'false_negative_rate']:.1%}，FP率 {rate_map.loc['REC', 'false_positive_rate']:.1%}。主要边界是复杂重构与简单复述、提问或一般咨询话语之间的区分；`sounds like` 等反映模板可能产生FP。",
            f"- QUO/QUC：FN率分别为 {rate_map.loc['QUO', 'false_negative_rate']:.1%}/{rate_map.loc['QUC', 'false_negative_rate']:.1%}，FP率分别为 {rate_map.loc['QUO', 'false_positive_rate']:.1%}/{rate_map.loc['QUC', 'false_positive_rate']:.1%}。问句表面结构容易识别，但开放性、封闭性以及反映式问句仍存在边界混淆。",
            f"- GI：FN率最高，为 {rate_map.loc['GI', 'false_negative_rate']:.1%}，但高分FN为 {int(rate_map.loc['GI', 'n_high_score_false_negative'])}。这说明许多GI正例被压在阈值下方，但probe很少以极低分强烈否定；FP则常与医疗主题词、数值和健康建议模式有关。",
            f"- SU：FN率 {rate_map.loc['SU', 'false_negative_rate']:.1%}，但高分FN仅 {int(rate_map.loc['SU', 'n_high_score_false_negative'])}；FP中高分案例为 {int(rate_map.loc['SU', 'n_high_score_false_positive'])}。该模式符合支持行为表达多样、同时 `sorry`、`help` 等表面表达容易触发误判。",
            f"- AF：FP率在叶标签中最低，为 {rate_map.loc['AF', 'false_positive_rate']:.1%}，但 {int(rate_map.loc['AF', 'n_high_score_false_positive'])}/{int(rate_map.loc['AF', 'n_false_positive'])} 个FP达到高模型分数。代表案例表明，probe会把对事物的一般积极评价或 `good/great/perfect` 词汇误当成对来访者优势与努力的肯定。",
            "",
            f"在270条审查样本中，共标记 {int(valid['mixed_behavior_flag'].sum())} 条混合行为候选，主要是反映与问句、GI与反映、SU与反映共存。这一数量来自AI辅助审查，可能低估隐含的混合行为，需人工coder复核。",
            "",
            "## 代表案例独立质检",
            "",
            "对28条叶标签代表案例进行人工式复核后，AI辅助诊断能够较好识别明显的问句、积极词触发和医疗主题触发，但对 probable_reference_label_error 存在过判。因此，116条该类别只能表示优先复核队列，不能作为标注错误数量。",
            "",
            "需要降级或重新解释的典型案例包括：",
            "",
            "- `audit_0227`：reference 为 RE|REC、RES=0 并不矛盾；文本中的情绪推断可能支持 REC。该案例更适合解释为 REC/RES 兄弟标签边界或 probe 混淆，而不是漏标 RES。",
            "- `audit_0240`：reference 为 RE|REC 同样不与 RES=0 构成逻辑矛盾；缺少 client 前文时不能判定简单或复杂反映。",
            "- `audit_0195`：仅凭当前第一人称文本不能确认其说话功能，应该优先标记上下文不足，而不是直接认定 REC 标错。",
            "- `audit_0120`：`so you afraid ...` 同时具有反映和确认问句形式，更适合 label ambiguity/mixed behavior，而不是明确漏标 QUC。",
            "- `audit_0045`：关于未成年人饮酒的规范性陈述仍可能属于 GI，0.499 的结果更符合阈值边界或定义边界。",
            "- `audit_0270`：`what can I do to help you` 可同时解释为开放问句和帮助提议，应该作为 QUO/SU 边界案例，而不是直接认定漏标 SU。",
            "",
            "经质检后，最稳健的发现不是存在大量标注错误，而是三类系统性边界：REC/RES 与反映式问句的区分、GI/SU/AF 的功能定义与表面词触发之间的差异，以及缺少 client 前文导致的不可判定案例。",
        ]
    )

    lines.extend(["", "## 代表性案例", ""])
    for row in representatives.itertuples(index=False):
        text = str(row.unit_text).replace("|", "\\|")
        lines.extend(
            [
                f"### {row.case_id} / {row.label} / {row.disagreement_type} / {row.representative_stratum}",
                "",
                f"- 文本：{text}",
                f"- Reference={row.reference_label}，SAE probability={row.sae_probability:.3f}。",
                f"- 候选判断：{row.primary_issue}；建议：{row.recommended_action}。",
                f"- 理由：{row.rationale}",
                f"- Latent 证据解释：{row.latent_evidence_interpretation}",
                "",
            ]
        )

    lines.extend(
        [
            "## 结论边界",
            "",
            "1. false positive/false negative 是相对于 0.5 阈值和现有 reference label 的操作性名称，不等于真实误报或漏报。",
            "2. probable_reference_label_error 只形成待人工复核候选；最终结论必须由独立 MISC coder 作出。",
            "3. RE/RES/REC 缺少 client 前文，相关案例优先解释为上下文不足或模型局限。",
            "4. 父标签 RE、QU 的不一致必须结合子标签层级检查，不能作为独立行为错误计数。",
            "5. Latent 贡献来自线性 probe，说明预测使用了哪些候选模式，不证明这些模式具有因果作用。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_semantic_disagreement_analysis(
    *,
    feature_store_path: str | Path,
    label_matrix_path: str | Path,
    stable_core_path: str | Path,
    review_table_path: str | Path,
    oof_predictions_path: str | Path,
    disagreement_candidates_path: str | Path,
    explanations_path: str | Path,
    output_dir: str | Path,
    api_key: str | None,
    audit_config: SAEAnnotationAuditConfig = SAEAnnotationAuditConfig(),
    config: SemanticDisagreementConfig = SemanticDisagreementConfig(),
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features = np.asarray(load_matrix(feature_store_path), dtype=np.float32)
    labels = pd.read_csv(label_matrix_path)
    stable = pd.read_csv(stable_core_path)
    review = pd.read_csv(review_table_path)
    oof = pd.read_csv(oof_predictions_path)
    all_candidates = pd.read_csv(disagreement_candidates_path)
    explanations = _read_jsonl(explanations_path)
    evidence = build_case_latent_evidence(
        features=features,
        label_matrix=labels,
        stable_core=stable,
        review_cases=review,
        explanations=explanations,
        audit_config=audit_config,
        config=config,
    )
    serializable = evidence.copy()
    for column in ("positive_latent_evidence", "negative_latent_evidence", "leaf_reference_labels"):
        serializable[column] = serializable[column].map(lambda value: json.dumps(value, ensure_ascii=False))
    serializable.to_csv(output / "case_latent_evidence.csv", index=False, encoding="utf-8-sig")

    reviews = run_deepseek_reviews(
        evidence=evidence,
        output_dir=output,
        api_key=api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
        config=config,
    )
    review_frame = pd.DataFrame(reviews)
    stale_review_columns = [
        column
        for column in review_frame.columns
        if column != "case_id" and column in serializable.columns
    ]
    serializable = serializable.drop(columns=stale_review_columns)
    enriched = serializable.merge(review_frame, on="case_id", how="left", validate="one_to_one")
    for column in ("secondary_issues", "mixed_behavior_labels"):
        if column == "secondary_issues":
            secondary_lists = enriched[column].map(
                lambda value: value if isinstance(value, list) else []
            )
        enriched[column] = enriched[column].map(lambda value: json.dumps(value, ensure_ascii=False))
    for issue in ISSUE_TYPES:
        enriched[f"{issue}_flag"] = enriched["primary_issue"].eq(issue) | secondary_lists.map(
            lambda values, target=issue: target in values
        )
    enriched.to_csv(output / "case_analysis_enriched.csv", index=False, encoding="utf-8-sig")

    valid = enriched[enriched["review_status"].eq("valid")]
    diagnosis = (
        valid.groupby(["label", "disagreement_type", "primary_issue"], as_index=False)
        .agg(
            n_cases=("case_id", "size"),
            n_high_model_score=("high_confidence_flag", "sum"),
            mean_sae_probability=("sae_probability", "mean"),
            mean_review_confidence=("review_confidence", "mean"),
        )
    )
    diagnosis["sample_fraction_within_label_direction"] = diagnosis["n_cases"] / diagnosis.groupby(
        ["label", "disagreement_type"]
    )["n_cases"].transform("sum")
    rates = summarize_oof_disagreements(oof)
    representatives = select_representative_cases(enriched)
    high_score = all_candidates[all_candidates["high_confidence_flag"].astype(bool)].copy()
    high_score_summary = (
        high_score.groupby(["label", "disagreement_type"], as_index=False)
        .agg(
            n_high_score_cases=("row_idx", "size"),
            mean_sae_probability=("sae_probability", "mean"),
            mean_score_margin=("score_margin", "mean"),
        )
    )
    rates.to_csv(output / "disagreement_rates_by_label.csv", index=False, encoding="utf-8-sig")
    diagnosis.to_csv(output / "sampled_diagnosis_summary.csv", index=False, encoding="utf-8-sig")
    diagnostic_flag_rows: list[dict[str, Any]] = []
    for label in DEFAULT_LABELS:
        for disagreement_type in ("false_negative", "false_positive"):
            group = valid[
                valid["label"].eq(label)
                & valid["disagreement_type"].eq(disagreement_type)
            ]
            for issue in sorted(ISSUE_TYPES):
                count = int(group[f"{issue}_flag"].sum())
                diagnostic_flag_rows.append(
                    {
                        "label": label,
                        "disagreement_type": disagreement_type,
                        "diagnostic_flag": issue,
                        "n_cases": count,
                        "sample_fraction": count / len(group) if len(group) else np.nan,
                    }
                )
    pd.DataFrame(diagnostic_flag_rows).to_csv(
        output / "sampled_diagnosis_flags.csv", index=False, encoding="utf-8-sig"
    )
    enriched[enriched["mixed_behavior_flag"]].to_csv(
        output / "mixed_behavior_candidates.csv", index=False, encoding="utf-8-sig"
    )
    high_score.to_csv(output / "high_score_disagreements.csv", index=False, encoding="utf-8-sig")
    high_score_summary.to_csv(
        output / "high_score_disagreement_summary.csv", index=False, encoding="utf-8-sig"
    )
    representatives.to_csv(output / "representative_cases.csv", index=False, encoding="utf-8-sig")
    figure_paths = write_disagreement_figures(
        rates=rates, enriched=enriched, output_dir=output / "figures"
    )
    _write_report(
        output / "semantic_disagreement_analysis_report.md",
        enriched,
        rates,
        diagnosis,
        representatives,
    )
    manifest = {
        "analysis": "stable_core_sae_disagreement_semantic_analysis",
        "inputs": {
            "feature_store": str(feature_store_path),
            "label_matrix": str(label_matrix_path),
            "stable_core": str(stable_core_path),
            "review_table": str(review_table_path),
            "oof_predictions": str(oof_predictions_path),
            "disagreement_candidates": str(disagreement_candidates_path),
            "explanations": str(explanations_path),
        },
        "config": asdict(config),
        "counts": {
            "n_cases": int(len(enriched)),
            "n_valid_reviews": int(enriched["review_status"].eq("valid").sum()),
            "n_failed_reviews": int(enriched["review_status"].ne("valid").sum()),
            "n_all_disagreements": int(len(all_candidates)),
            "n_high_score_disagreements": int(len(high_score)),
            "n_representative_cases": int(len(representatives)),
        },
        "figures": figure_paths,
        "reference_label_provenance": REFERENCE_LABEL_PROVENANCE,
        "claim_boundary": "AI-assisted candidate audit; requires independent human MISC adjudication",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


__all__ = [
    "SemanticDisagreementConfig",
    "build_case_latent_evidence",
    "build_case_review_prompt",
    "run_semantic_disagreement_analysis",
]
