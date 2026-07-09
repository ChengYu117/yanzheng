"""Held-out activation detection scoring for contrastive latent explanations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

from .contrastive_evidence_pack import jsonable, read_jsonl, write_json, write_jsonl
from .contrastive_gemini_io import validate_prediction_outputs


LABEL_DEFINITIONS: dict[str, str] = {
    "RE": "Reflection: counselor statement that reflects, repeats, or infers client meaning or feeling.",
    "RES": "Simple reflection: counselor repeats or slightly rephrases client content with little added meaning.",
    "REC": "Complex reflection: counselor adds inferred meaning, emotion, emphasis, or deeper interpretation.",
    "QU": "Question: counselor utterance that asks the client for information, choice, elaboration, or confirmation.",
    "QUO": "Open question: question inviting elaborated response rather than yes/no or short factual confirmation.",
    "QUC": "Closed question: question answerable with yes/no, a short fact, or a constrained option.",
    "GI": "Giving information: counselor provides facts, advice, explanation, feedback, or professional information.",
    "SU": "Support: counselor offers empathy, encouragement, reassurance, or nonjudgmental support.",
    "AF": "Affirmation: counselor recognizes client strengths, efforts, values, or positive qualities.",
}


@dataclass(frozen=True)
class ScorerTaskConfig:
    include_all_valid_explanations: bool = True


def _explanation_for_prompt(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "short_name": row["short_name"],
        "main_hypothesis": row["main_hypothesis"],
        "positive_triggers": row["positive_triggers"],
        "explicit_exclusions": row["explicit_exclusions"],
        "possible_surface_confounds": row["possible_surface_confounds"],
        "feature_type": row["feature_type"],
        "alternative_hypotheses": row["alternative_hypotheses"],
        "failure_modes": row["failure_modes"],
    }


def _heldout_visible_samples(pack: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for internal_tag, samples in pack["heldout_internal_by_tag"].items():
        visible_tag = "NONACTIVE_NEAR_MISS" if internal_tag == "NONACTIVE_LABEL_MATCH" else str(internal_tag)
        for sample in samples:
            rows.append(
                {
                    "sample_id": str(sample["id"]),
                    "tag": visible_tag,
                    "text": str(sample["text"]),
                }
            )
    return rows


def _heldout_answer_rows(pack: dict[str, Any], task_id: str, *, raw_kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for internal_tag, samples in pack["heldout_internal_by_tag"].items():
        visible_tag = "NONACTIVE_NEAR_MISS" if internal_tag == "NONACTIVE_LABEL_MATCH" else str(internal_tag)
        for sample in samples:
            rows.append(
                {
                    "task_id": task_id,
                    "packet_id": pack["packet_id"],
                    "latent_idx": int(pack["latent_idx"]),
                    "target_label": pack["target_label"],
                    "sample_id": str(sample["id"]),
                    "visible_tag": visible_tag,
                    "internal_tag": internal_tag,
                    "row_idx": int(sample["row_idx"]),
                    "activation": float(sample["activation"]),
                    "activation_threshold": float(sample["activation_threshold"]),
                    "ground_truth_activate": int(sample["ground_truth_activate"]),
                    "raw_kind": raw_kind,
                }
            )
    return rows


def _format_holdout(samples: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- sample_id={sample['sample_id']} tag={sample['tag']}\n  text: {str(sample['text']).replace(chr(10), ' ').strip()}"
        for sample in samples
    )


def build_scorer_prompt(explanation: dict[str, Any], heldout_samples: list[dict[str, Any]]) -> str:
    return f"""Judge whether each sentence should activate the anonymous SAE latent, using only the explanation below.

Explanation JSON:
{json.dumps(explanation, ensure_ascii=False, indent=2)}

Held-out samples:
{_format_holdout(heldout_samples)}

Return only a JSON list. Each item must have:
- sample_id
- pred_activate_prob: continuous probability from 0 to 1
- binary_prediction: 0 or 1
- evidence_span: short copied text span supporting your decision
- reason: one short sentence based on the explanation
"""


def build_label_baseline_prompt(label: str, heldout_samples: list[dict[str, Any]]) -> str:
    definition = LABEL_DEFINITIONS.get(label, f"{label}: no detailed definition available.")
    return f"""Judge whether each sentence matches this counseling behavior definition.

Behavior definition:
{definition}

Important: output a probability that the sentence should activate an internal latent if that latent simply followed this behavior definition. You do not know the real latent explanation.

Held-out samples:
{_format_holdout(heldout_samples)}

Return only a JSON list. Each item must have:
- sample_id
- pred_activate_prob: continuous probability from 0 to 1
- binary_prediction: 0 or 1
- evidence_span: short copied text span supporting your decision
- reason: one short sentence based on the behavior definition
"""


def make_scorer_tasks(
    *,
    packs_path: str | Path,
    explanations_path: str | Path,
    output_dir: str | Path,
    config: ScorerTaskConfig = ScorerTaskConfig(),
) -> dict[str, Any]:
    output_path = Path(output_dir)
    task_dir = output_path / "ide_tasks"
    scorer_raw_dir = output_path / "scorer_outputs" / "raw_explanation_scorer"
    baseline_raw_dir = output_path / "scorer_outputs" / "raw_label_baseline"
    task_dir.mkdir(parents=True, exist_ok=True)
    scorer_raw_dir.mkdir(parents=True, exist_ok=True)
    baseline_raw_dir.mkdir(parents=True, exist_ok=True)

    packs = {pack["packet_id"]: pack for pack in read_jsonl(packs_path)}
    explanations = read_jsonl(explanations_path) if Path(explanations_path).exists() else []
    if not config.include_all_valid_explanations and explanations:
        df = pd.DataFrame(explanations).sort_values(["packet_id", "confidence"], ascending=[True, False])
        explanations = df.groupby("packet_id", sort=False).head(1).to_dict(orient="records")

    scorer_tasks: list[dict[str, Any]] = []
    baseline_tasks: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []

    for explanation in explanations:
        packet_id = str(explanation["packet_id"])
        if packet_id not in packs:
            continue
        pack = packs[packet_id]
        heldout_samples = _heldout_visible_samples(pack)
        task_id = f"{packet_id}_scorer_{explanation['task_id']}"
        scorer_task = {
            "task_id": task_id,
            "task_type": "latent_explanation_activation_scorer",
            "packet_id": packet_id,
            "latent_idx": int(pack["latent_idx"]),
            "explanation_task_id": explanation["task_id"],
            "prompt": build_scorer_prompt(_explanation_for_prompt(explanation), heldout_samples),
            "expected_output_path": str(scorer_raw_dir / f"{task_id}.json"),
            "output_format": "json_list",
            "status": "pending_gemini_in_antigravity",
        }
        scorer_tasks.append(scorer_task)
        answer_rows.extend(_heldout_answer_rows(pack, task_id, raw_kind="explanation_scorer"))

    # Baseline runs once per packet that has at least one valid explanation.
    baseline_packet_ids = sorted({str(explanation["packet_id"]) for explanation in explanations if str(explanation["packet_id"]) in packs})
    for packet_id in baseline_packet_ids:
        pack = packs[packet_id]
        task_id = f"{packet_id}_label_baseline"
        baseline_task = {
            "task_id": task_id,
            "task_type": "label_definition_baseline",
            "packet_id": packet_id,
            "latent_idx": int(pack["latent_idx"]),
            "target_label": pack["target_label"],
            "prompt": build_label_baseline_prompt(pack["target_label"], _heldout_visible_samples(pack)),
            "expected_output_path": str(baseline_raw_dir / f"{task_id}.json"),
            "output_format": "json_list",
            "status": "pending_gemini_in_antigravity",
        }
        baseline_tasks.append(baseline_task)
        answer_rows.extend(_heldout_answer_rows(pack, task_id, raw_kind="label_baseline"))

    scorer_tasks_path = task_dir / "scorer_tasks.jsonl"
    baseline_tasks_path = task_dir / "baseline_tasks.jsonl"
    answer_key_path = output_path / "scorer_outputs" / "heldout_answer_key.csv"
    write_jsonl(scorer_tasks_path, scorer_tasks)
    write_jsonl(baseline_tasks_path, baseline_tasks)
    pd.DataFrame(answer_rows).to_csv(answer_key_path, index=False)

    manifest = {
        "step": "make-scorer-tasks",
        "inputs": {
            "packs": str(packs_path),
            "validated_explanations": str(explanations_path),
        },
        "outputs": {
            "scorer_tasks": str(scorer_tasks_path),
            "baseline_tasks": str(baseline_tasks_path),
            "heldout_answer_key": str(answer_key_path),
            "raw_explanation_scorer_dir": str(scorer_raw_dir),
            "raw_label_baseline_dir": str(baseline_raw_dir),
        },
        "parameters": asdict(config),
        "n_explanations": int(len(explanations)),
        "n_scorer_tasks": int(len(scorer_tasks)),
        "n_baseline_tasks": int(len(baseline_tasks)),
    }
    write_json(output_path / "scorer_task_manifest.json", manifest)
    return manifest


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return 0.5


def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    negatives = y_true == 0
    if int(negatives.sum()) == 0:
        return 0.0
    return float(np.mean(y_pred[negatives] == 0))


def _metrics_for_predictions(predictions: pd.DataFrame, *, kind: str) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for task_id, group in predictions.groupby("task_id", sort=False):
        y_true = group["ground_truth_activate"].to_numpy(dtype=int)
        y_prob = group["pred_activate_prob"].to_numpy(dtype=float)
        y_pred = group["binary_prediction"].to_numpy(dtype=int)
        rows.append(
            {
                "task_id": task_id,
                "packet_id": group["packet_id"].iloc[0],
                "latent_idx": int(group["latent_idx"].iloc[0]),
                "target_label": group["target_label"].iloc[0],
                "raw_kind": kind,
                "n_samples": int(len(group)),
                "n_positive": int(y_true.sum()),
                "n_negative": int(len(y_true) - y_true.sum()),
                "auroc": _safe_auc(y_true, y_prob),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "specificity": _specificity(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows)


def _status(auroc: float, latent_gap: float) -> str:
    if latent_gap <= 0:
        return "no_latent_contribution"
    if auroc >= 0.70:
        return "accepted"
    if auroc >= 0.60:
        return "ambiguous"
    return "rejected"


def validate_scorer_outputs(
    *,
    scorer_tasks_path: str | Path,
    baseline_tasks_path: str | Path,
    answer_key_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    scorer_dir = output_path / "scorer_outputs"
    scorer_dir.mkdir(parents=True, exist_ok=True)
    scorer_tasks = read_jsonl(scorer_tasks_path) if Path(scorer_tasks_path).exists() else []
    baseline_tasks = read_jsonl(baseline_tasks_path) if Path(baseline_tasks_path).exists() else []
    answer_key = pd.read_csv(answer_key_path) if Path(answer_key_path).exists() else pd.DataFrame()

    scorer_key = answer_key[answer_key.get("raw_kind", pd.Series(dtype=str)) == "explanation_scorer"].copy()
    baseline_key = answer_key[answer_key.get("raw_kind", pd.Series(dtype=str)) == "label_baseline"].copy()
    scorer_valid, scorer_errors, scorer_retry = validate_prediction_outputs(
        tasks=scorer_tasks,
        answer_key=scorer_key,
        raw_kind="explanation_scorer",
    )
    baseline_valid, baseline_errors, baseline_retry = validate_prediction_outputs(
        tasks=baseline_tasks,
        answer_key=baseline_key,
        raw_kind="label_baseline",
    )

    scorer_pred = pd.DataFrame(scorer_valid)
    baseline_pred = pd.DataFrame(baseline_valid)
    scorer_metrics = _metrics_for_predictions(scorer_pred, kind="explanation_scorer")
    baseline_metrics = _metrics_for_predictions(baseline_pred, kind="label_baseline")

    if not scorer_metrics.empty:
        baseline_auc = {
            str(row["packet_id"]): float(row["auroc"])
            for _, row in baseline_metrics.iterrows()
        } if not baseline_metrics.empty else {}
        scorer_metrics["baseline_auroc"] = scorer_metrics["packet_id"].map(baseline_auc).fillna(0.5)
        scorer_metrics["latent_gap"] = scorer_metrics["auroc"] - scorer_metrics["baseline_auroc"]
        scorer_metrics["status"] = [
            _status(float(row["auroc"]), float(row["latent_gap"]))
            for _, row in scorer_metrics.iterrows()
        ]

    scorer_pred_path = scorer_dir / "validated_scorer_predictions.jsonl"
    baseline_pred_path = scorer_dir / "validated_baseline_predictions.jsonl"
    scorer_metrics_path = scorer_dir / "scorer_metrics.csv"
    baseline_metrics_path = scorer_dir / "baseline_metrics.csv"
    errors_path = scorer_dir / "validation_errors.csv"
    retry_scorer_path = scorer_dir / "retry_scorer_tasks.jsonl"
    retry_baseline_path = scorer_dir / "retry_baseline_tasks.jsonl"

    write_jsonl(scorer_pred_path, scorer_pred.to_dict(orient="records"))
    write_jsonl(baseline_pred_path, baseline_pred.to_dict(orient="records"))
    scorer_metrics.to_csv(scorer_metrics_path, index=False)
    baseline_metrics.to_csv(baseline_metrics_path, index=False)
    pd.DataFrame(scorer_errors + baseline_errors).to_csv(errors_path, index=False)
    write_jsonl(retry_scorer_path, scorer_retry)
    write_jsonl(retry_baseline_path, baseline_retry)

    manifest = {
        "step": "validate-scorer",
        "inputs": {
            "scorer_tasks": str(scorer_tasks_path),
            "baseline_tasks": str(baseline_tasks_path),
            "heldout_answer_key": str(answer_key_path),
        },
        "outputs": {
            "validated_scorer_predictions": str(scorer_pred_path),
            "validated_baseline_predictions": str(baseline_pred_path),
            "scorer_metrics": str(scorer_metrics_path),
            "baseline_metrics": str(baseline_metrics_path),
            "validation_errors": str(errors_path),
            "retry_scorer_tasks": str(retry_scorer_path),
            "retry_baseline_tasks": str(retry_baseline_path),
        },
        "n_scorer_tasks": int(len(scorer_tasks)),
        "n_baseline_tasks": int(len(baseline_tasks)),
        "n_valid_scorer_predictions": int(len(scorer_pred)),
        "n_valid_baseline_predictions": int(len(baseline_pred)),
        "n_scorer_errors": int(len(scorer_errors)),
        "n_baseline_errors": int(len(baseline_errors)),
        "status_counts": (
            scorer_metrics["status"].value_counts().sort_index().to_dict()
            if "status" in scorer_metrics.columns
            else {}
        ),
    }
    write_json(scorer_dir / "validation_manifest.json", manifest)
    return manifest


__all__ = [
    "LABEL_DEFINITIONS",
    "ScorerTaskConfig",
    "build_label_baseline_prompt",
    "build_scorer_prompt",
    "make_scorer_tasks",
    "validate_scorer_outputs",
]
