"""Import and validate raw Antigravity Gemini JSON outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contrastive_evidence_pack import MISC_LABEL_PATTERN, jsonable, read_jsonl, write_json, write_jsonl
from .contrastive_explainer import EXPLAINER_SCHEMA_FIELDS


def strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    fence = re.fullmatch(r"```(?:json|JSON)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return cleaned


def parse_gemini_json_text(text: str) -> Any:
    cleaned = strip_json_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start_positions = [idx for idx in (cleaned.find("{"), cleaned.find("[")) if idx >= 0]
        if not start_positions:
            raise
        start = min(start_positions)
        end_obj = cleaned.rfind("}")
        end_arr = cleaned.rfind("]")
        end = max(end_obj, end_arr)
        if end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def parse_gemini_json_file(path: str | Path) -> Any:
    return parse_gemini_json_text(Path(path).read_text(encoding="utf-8"))


def _as_string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    out: list[str] = []
    for item in value:
        if not isinstance(item, (str, int, float)):
            raise ValueError(f"{field} contains non-scalar item: {type(item).__name__}")
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _check_no_label_codes(value: Any, *, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _check_no_label_codes(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            _check_no_label_codes(item, path=f"{path}[{idx}]")
    elif isinstance(value, str) and MISC_LABEL_PATTERN.search(value):
        raise ValueError(f"MISC label token leaked at {path}: {value!r}")


def validate_explanation_payload(payload: Any, *, task: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Explanation output must be a JSON object, got {type(payload).__name__}")
    missing = [field for field in EXPLAINER_SCHEMA_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"Explanation output missing fields: {missing}")

    latent_idx = int(payload["latent_idx"])
    expected_latent = int(task["latent_idx"])
    if latent_idx != expected_latent:
        raise ValueError(f"latent_idx mismatch: expected {expected_latent}, got {latent_idx}")
    confidence = float(payload["confidence"])
    if not np.isfinite(confidence) or confidence < 0 or confidence > 1:
        raise ValueError(f"confidence must be in [0, 1], got {payload['confidence']!r}")

    out = {
        "task_id": task["task_id"],
        "packet_id": task["packet_id"],
        "latent_idx": latent_idx,
        "repeat": int(task.get("repeat", 1)),
        "short_name": str(payload["short_name"]).strip(),
        "main_hypothesis": str(payload["main_hypothesis"]).strip(),
        "positive_triggers": _as_string_list(payload["positive_triggers"], "positive_triggers"),
        "explicit_exclusions": _as_string_list(payload["explicit_exclusions"], "explicit_exclusions"),
        "possible_surface_confounds": _as_string_list(
            payload["possible_surface_confounds"],
            "possible_surface_confounds",
        ),
        "feature_type": str(payload["feature_type"]).strip(),
        "confidence": confidence,
        "alternative_hypotheses": _as_string_list(payload["alternative_hypotheses"], "alternative_hypotheses"),
        "key_evidence": _as_string_list(payload["key_evidence"], "key_evidence"),
        "failure_modes": str(payload["failure_modes"]).strip(),
        "raw_output_path": str(task.get("expected_output_path", "")),
        "validation_status": "valid",
    }
    if not out["short_name"]:
        raise ValueError("short_name is empty")
    if not out["main_hypothesis"]:
        raise ValueError("main_hypothesis is empty")
    _check_no_label_codes(out, path="explanation")
    return out


def validate_explainer_outputs(
    *,
    tasks_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    tasks = read_jsonl(tasks_path)
    valid: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    retry_tasks: list[dict[str, Any]] = []

    for task in tasks:
        raw_path = Path(task.get("expected_output_path", ""))
        if not raw_path.exists():
            error = {
                "task_id": task["task_id"],
                "packet_id": task.get("packet_id"),
                "latent_idx": task.get("latent_idx"),
                "error": "raw_output_missing",
                "raw_output_path": str(raw_path),
            }
            errors.append(error)
            retry_tasks.append(task)
            continue
        try:
            payload = parse_gemini_json_file(raw_path)
            valid.append(validate_explanation_payload(payload, task=task))
        except Exception as exc:
            error = {
                "task_id": task["task_id"],
                "packet_id": task.get("packet_id"),
                "latent_idx": task.get("latent_idx"),
                "error": f"{type(exc).__name__}: {exc}",
                "raw_output_path": str(raw_path),
            }
            errors.append(error)
            retry_tasks.append(task)

    explainer_dir = output_path / "explainer_outputs"
    validated_path = explainer_dir / "validated_explanations.jsonl"
    errors_path = explainer_dir / "validation_errors.csv"
    retry_path = explainer_dir / "retry_tasks.jsonl"
    consistency_path = explainer_dir / "consistency_by_latent.csv"
    explainer_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(validated_path, valid)
    pd.DataFrame(errors).to_csv(errors_path, index=False)
    write_jsonl(retry_path, retry_tasks)

    consistency = build_explanation_consistency(valid)
    consistency.to_csv(consistency_path, index=False)
    manifest = {
        "step": "validate-explainer",
        "inputs": {"tasks": str(tasks_path)},
        "outputs": {
            "validated_explanations": str(validated_path),
            "validation_errors": str(errors_path),
            "retry_tasks": str(retry_path),
            "consistency_by_latent": str(consistency_path),
        },
        "n_tasks": int(len(tasks)),
        "n_valid": int(len(valid)),
        "n_errors": int(len(errors)),
        "n_retry": int(len(retry_tasks)),
    }
    write_json(explainer_dir / "validation_manifest.json", manifest)
    return manifest


def build_explanation_consistency(valid_explanations: list[dict[str, Any]]) -> pd.DataFrame:
    if not valid_explanations:
        return pd.DataFrame(
            columns=[
                "packet_id",
                "latent_idx",
                "n_valid_repeats",
                "short_name_match",
                "feature_type_match",
                "confidence_gap",
                "low_consistency",
            ]
        )
    rows: list[dict[str, Any]] = []
    df = pd.DataFrame(valid_explanations)
    for (packet_id, latent_idx), group in df.groupby(["packet_id", "latent_idx"], sort=False):
        names = [str(item).strip() for item in group["short_name"].tolist()]
        types = [str(item).strip() for item in group["feature_type"].tolist()]
        confidences = pd.to_numeric(group["confidence"], errors="coerce").dropna().tolist()
        short_name_match = len(set(names)) <= 1 if len(names) >= 2 else None
        feature_type_match = len(set(types)) <= 1 if len(types) >= 2 else None
        confidence_gap = float(max(confidences) - min(confidences)) if confidences else np.nan
        low_consistency = bool(
            len(names) >= 2
            and len(set(names)) > 1
            and all(float(value) < 0.60 for value in confidences)
        )
        rows.append(
            {
                "packet_id": packet_id,
                "latent_idx": int(latent_idx),
                "n_valid_repeats": int(len(group)),
                "short_name_match": short_name_match,
                "feature_type_match": feature_type_match,
                "confidence_gap": confidence_gap,
                "low_consistency": low_consistency,
            }
        )
    return pd.DataFrame(rows)


def normalize_prediction_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("predictions", "results", "items", "samples"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(f"Scorer output must be a JSON list or wrapped list, got {type(payload).__name__}")
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Prediction {idx} must be an object")
        sample_id = str(item.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"Prediction {idx} missing sample_id")
        prob = float(item.get("pred_activate_prob"))
        if not np.isfinite(prob) or prob < 0 or prob > 1:
            raise ValueError(f"Prediction {sample_id} probability out of range: {prob}")
        binary = int(item.get("binary_prediction", 1 if prob >= 0.5 else 0))
        if binary not in {0, 1}:
            raise ValueError(f"Prediction {sample_id} binary_prediction must be 0/1")
        out.append(
            {
                "sample_id": sample_id,
                "pred_activate_prob": prob,
                "binary_prediction": binary,
                "evidence_span": str(item.get("evidence_span", "")),
                "reason": str(item.get("reason", "")),
            }
        )
    return out


def parse_prediction_file(path: str | Path) -> list[dict[str, Any]]:
    return normalize_prediction_payload(parse_gemini_json_file(path))


def validate_prediction_outputs(
    *,
    tasks: list[dict[str, Any]],
    answer_key: pd.DataFrame,
    raw_kind: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    valid_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    retry_tasks: list[dict[str, Any]] = []
    key_by_task = {
        str(task_id): group.copy()
        for task_id, group in answer_key.groupby("task_id", sort=False)
    }
    for task in tasks:
        task_id = str(task["task_id"])
        raw_path = Path(task.get("expected_output_path", ""))
        expected = key_by_task.get(task_id, pd.DataFrame())
        if not raw_path.exists():
            errors.append(
                {
                    "task_id": task_id,
                    "packet_id": task.get("packet_id"),
                    "raw_kind": raw_kind,
                    "error": "raw_output_missing",
                    "raw_output_path": str(raw_path),
                }
            )
            retry_tasks.append(task)
            continue
        try:
            predictions = parse_prediction_file(raw_path)
            expected_ids = set(expected["sample_id"].astype(str).tolist())
            predicted_ids = {row["sample_id"] for row in predictions}
            missing = sorted(expected_ids.difference(predicted_ids))
            extra = sorted(predicted_ids.difference(expected_ids))
            if missing or extra:
                raise ValueError(f"sample_id mismatch; missing={missing}, extra={extra}")
            for pred in predictions:
                key_row = expected[expected["sample_id"].astype(str) == pred["sample_id"]].iloc[0]
                row = {
                    "task_id": task_id,
                    "packet_id": task.get("packet_id"),
                    "latent_idx": task.get("latent_idx"),
                    "raw_kind": raw_kind,
                    "sample_id": pred["sample_id"],
                    "pred_activate_prob": pred["pred_activate_prob"],
                    "binary_prediction": pred["binary_prediction"],
                    "evidence_span": pred["evidence_span"],
                    "reason": pred["reason"],
                    "ground_truth_activate": int(key_row["ground_truth_activate"]),
                    "visible_tag": key_row.get("visible_tag"),
                    "internal_tag": key_row.get("internal_tag"),
                    "target_label": key_row.get("target_label"),
                    "raw_output_path": str(raw_path),
                }
                valid_rows.append(row)
        except Exception as exc:
            errors.append(
                {
                    "task_id": task_id,
                    "packet_id": task.get("packet_id"),
                    "raw_kind": raw_kind,
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_output_path": str(raw_path),
                }
            )
            retry_tasks.append(task)
    return valid_rows, errors, retry_tasks


__all__ = [
    "build_explanation_consistency",
    "normalize_prediction_payload",
    "parse_gemini_json_file",
    "parse_gemini_json_text",
    "parse_prediction_file",
    "strip_json_fences",
    "validate_explainer_outputs",
    "validate_explanation_payload",
    "validate_prediction_outputs",
]
