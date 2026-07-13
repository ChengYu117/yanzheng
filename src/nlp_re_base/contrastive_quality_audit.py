"""Quality and provenance audit for P3 contrastive latent explanations.

Schema validation answers whether a response can be parsed.  This module adds
the stricter gate needed before an explanation is used by the scorer,
minimal-pair designer, or subconcept clustering stages:

* every cited sample id must exist in the exact prompt;
* the explanation must contrast active evidence with near-miss evidence;
* critical fields must contain enough specific content;
* repeated/template-like outputs are surfaced for review; and
* task, prompt, raw output, and execution manifest provenance must agree.

The audit never deletes raw outputs.  Untrusted rows are written to a separate
quarantine/retry queue and are excluded from downstream inputs.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .contrastive_evidence_pack import read_jsonl, write_json, write_jsonl


_SAMPLE_RE = re.compile(
    r"^\s*-\s+id=(?P<id>\S+)\s+tag=(?P<tag>\S+)\s+activation=(?P<activation>\S+)\s*$"
)
_SAMPLE_ID_RE = re.compile(r"^(?:s|u)\d+$", re.IGNORECASE)
_GENERATOR_FINGERPRINTS = (
    "generate_explanation",
    "build_explanation",
    "MODEL_NAME = \"claude-opus",
    "MODEL = \"claude-opus",
    "llm_execution_manifest.jsonl",
)
_TRUSTED_EXECUTION_MODES = {"claude_code_llm", "zhipu_api"}


def _canonical_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_loose_json(text: str) -> Any:
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json|JSON)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        starts = [index for index in (cleaned.find("{"), cleaned.find("[")) if index >= 0]
        ends = [cleaned.rfind("}"), cleaned.rfind("]")]
        if not starts or max(ends) < min(starts):
            raise
        return json.loads(cleaned[min(starts) : max(ends) + 1])


def _normalise_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _token_set(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", _normalise_text(value)))


def _jaccard(left: Any, right: Any) -> float:
    a, b = _token_set(left), _token_set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def parse_prompt_samples(prompt: str) -> dict[str, dict[str, Any]]:
    """Parse the visible sample tags from a generated explainer prompt."""
    lines = str(prompt).splitlines()
    samples: dict[str, dict[str, Any]] = {}
    for index, line in enumerate(lines):
        match = _SAMPLE_RE.match(line)
        if not match:
            continue
        sample_id = match.group("id")
        text = ""
        if index + 1 < len(lines) and lines[index + 1].strip().lower().startswith("text:"):
            text = lines[index + 1].split(":", 1)[1].strip()
        try:
            activation = float(match.group("activation"))
        except ValueError:
            activation = None
        samples[sample_id] = {
            "id": sample_id,
            "tag": match.group("tag"),
            "activation": activation,
            "text": text,
        }
    return samples


def _find_suspicious_generators(repo_root: Path) -> list[str]:
    """Return local scripts that can synthesize/falsify LLM provenance."""
    findings: list[str] = []
    for path in sorted(repo_root.glob("*.py")) + sorted((repo_root / "scripts").glob("*.py")):
        lower_name = path.name.lower()
        if lower_name.startswith("test_"):
            continue
        # Baseline/scorer batch scripts must not invalidate the explainer gate;
        # only scripts capable of writing explainer outputs are relevant here.
        if "explainer" not in lower_name and "contrastive_tasks" not in lower_name:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        fingerprint_count = sum(token in text for token in _GENERATOR_FINGERPRINTS)
        if fingerprint_count >= 2 and "contrastive_latent_interp" in text:
            findings.append(str(path))
    return findings


def _manifest_audit(
    *,
    task: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    task_id = str(task["task_id"])
    rows = [row for row in manifest_rows if str(row.get("task_id")) == task_id]
    reasons: list[str] = []
    if len(rows) == 0:
        return "missing", ["manifest_missing"]
    if len(rows) != 1:
        reasons.append(f"manifest_duplicate_count={len(rows)}")
    row = rows[-1]
    expected_prompt_hash = hashlib.sha256(str(task.get("prompt", "")).encode("utf-8")).hexdigest()
    if row.get("prompt_sha256") != expected_prompt_hash:
        reasons.append("prompt_hash_mismatch")
    raw_path = Path(str(task.get("expected_output_path", "")))
    if not raw_path.exists():
        reasons.append("raw_output_missing")
    else:
        try:
            payload = _parse_loose_json(raw_path.read_text(encoding="utf-8"))
            accepted_hashes = {
                hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                _canonical_hash(payload),
                hashlib.sha256(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest(),
            }
            if row.get("raw_output_sha256") not in accepted_hashes:
                reasons.append("raw_output_hash_mismatch")
        except Exception as exc:  # pragma: no cover - schema validator owns parse errors
            reasons.append(f"raw_output_unreadable={type(exc).__name__}")
    if not str(row.get("model", "")).strip():
        reasons.append("model_missing")
    if row.get("execution_mode") not in _TRUSTED_EXECUTION_MODES:
        reasons.append("execution_mode_missing_or_invalid")
    if reasons:
        return "mismatch", reasons
    return "declared_complete", []


def _quality_row(
    *,
    task: dict[str, Any],
    explanation: dict[str, Any],
    provenance_status: str,
    provenance_reasons: list[str],
    contamination: bool,
) -> dict[str, Any]:
    samples = parse_prompt_samples(str(task.get("prompt", "")))
    evidence = explanation.get("key_evidence", [])
    evidence = evidence if isinstance(evidence, list) else []
    evidence_ids = [str(item).strip() for item in evidence]
    unknown_ids = sorted({item for item in evidence_ids if item not in samples})
    tags = [samples[item]["tag"] for item in evidence_ids if item in samples]
    active_count = sum(tag.startswith("ACTIVE_") for tag in tags)
    near_count = sum(tag == "NONACTIVE_NEAR_MISS" for tag in tags)
    random_count = sum(tag == "NONACTIVE_RANDOM" for tag in tags)
    critical_lengths = {
        "hypothesis_chars": len(str(explanation.get("main_hypothesis", "")).strip()),
        "positive_trigger_count": len(explanation.get("positive_triggers", []))
        if isinstance(explanation.get("positive_triggers"), list)
        else 0,
        "explicit_exclusion_count": len(explanation.get("explicit_exclusions", []))
        if isinstance(explanation.get("explicit_exclusions"), list)
        else 0,
        "alternative_count": len(explanation.get("alternative_hypotheses", []))
        if isinstance(explanation.get("alternative_hypotheses"), list)
        else 0,
    }
    failure_modes_value = explanation.get("failure_modes", "")
    failure_modes = str(failure_modes_value).strip()
    failure_modes_list_artifact = bool(
        not isinstance(failure_modes_value, list)
        and failure_modes.startswith("[")
        and failure_modes.endswith("]")
    )
    text = " ".join(
        [
            str(explanation.get("main_hypothesis", "")),
            " ".join(map(str, explanation.get("positive_triggers", []))),
            " ".join(map(str, explanation.get("explicit_exclusions", []))),
            " ".join(map(str, explanation.get("possible_surface_confounds", []))),
        ]
    ).lower()
    contrast_language = any(
        phrase in text
        for phrase in ("near-miss", "near miss", "nonactive", "does not", "doesn't", "not trigger")
    )
    reasons: list[str] = list(provenance_reasons)
    if unknown_ids:
        reasons.append("key_evidence_unknown_sample_id")
    if active_count == 0:
        reasons.append("key_evidence_has_no_active_sample")
    if near_count == 0:
        reasons.append("key_evidence_has_no_near_miss_sample")
    if critical_lengths["hypothesis_chars"] < 40:
        reasons.append("hypothesis_too_short")
    if critical_lengths["positive_trigger_count"] < 2:
        reasons.append("too_few_positive_triggers")
    if critical_lengths["explicit_exclusion_count"] < 1:
        reasons.append("no_explicit_exclusion")
    if not contrast_language:
        reasons.append("no_contrastive_language")
    if failure_modes_list_artifact:
        reasons.append("failure_modes_serialized_list_artifact")
    if contamination:
        reasons.append("local_generator_fingerprint_detected")

    hard_fail_reasons = {
        "manifest_missing",
        "manifest_duplicate_count",
        "prompt_hash_mismatch",
        "raw_output_hash_mismatch",
        "raw_output_missing",
        "key_evidence_unknown_sample_id",
        "key_evidence_has_no_active_sample",
        "local_generator_fingerprint_detected",
        "execution_mode_missing_or_invalid",
    }
    is_hard_fail = contamination or any(
        any(reason.startswith(prefix) for prefix in hard_fail_reasons) for reason in reasons
    )
    if is_hard_fail:
        quality_status = "contaminated" if contamination else "fail"
    elif reasons:
        quality_status = "review"
    else:
        quality_status = "pass"
    return {
        "task_id": task.get("task_id"),
        "packet_id": task.get("packet_id"),
        "latent_idx": task.get("latent_idx"),
        "repeat": task.get("repeat"),
        "provenance_status": provenance_status,
        "provenance_reasons": ";".join(provenance_reasons),
        "quality_status": quality_status,
        "quality_reasons": ";".join(dict.fromkeys(reasons)),
        "trusted_for_downstream": bool(quality_status == "pass" and provenance_status == "declared_complete"),
        "n_prompt_samples": len(samples),
        "n_key_evidence": len(evidence_ids),
        "n_unknown_evidence": len(unknown_ids),
        "active_evidence_count": active_count,
        "near_miss_evidence_count": near_count,
        "random_evidence_count": random_count,
        "failure_modes_list_artifact": failure_modes_list_artifact,
        "contrast_language_present": contrast_language,
        "confidence": explanation.get("confidence"),
        **critical_lengths,
    }


def audit_explainer_quality(
    *,
    tasks_path: str | Path,
    validated_explanations_path: str | Path,
    execution_manifest_path: str | Path,
    output_dir: str | Path,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Audit current explainer outputs and create a safe downstream queue."""
    tasks = read_jsonl(tasks_path)
    explanations = read_jsonl(validated_explanations_path)
    manifest_path = Path(execution_manifest_path)
    manifest_rows = read_jsonl(manifest_path) if manifest_path.exists() else []
    task_by_id = {str(task["task_id"]): task for task in tasks}
    explanation_by_id = {str(row["task_id"]): row for row in explanations}

    suspicious_generators = _find_suspicious_generators(Path(repo_root).resolve())
    clean_manifest_declared = bool(
        manifest_rows
        and all(row.get("execution_mode") in _TRUSTED_EXECUTION_MODES for row in manifest_rows)
    )
    contamination = bool(suspicious_generators) and not clean_manifest_declared
    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        explanation = explanation_by_id.get(task_id)
        if explanation is None:
            rows.append(
                {
                    "task_id": task_id,
                    "packet_id": task.get("packet_id"),
                    "latent_idx": task.get("latent_idx"),
                    "repeat": task.get("repeat"),
                    "provenance_status": "missing",
                    "provenance_reasons": "validated_explanation_missing",
                    "quality_status": "fail",
                    "quality_reasons": "validated_explanation_missing",
                    "trusted_for_downstream": False,
                }
            )
            continue
        provenance_status, provenance_reasons = _manifest_audit(task=task, manifest_rows=manifest_rows)
        rows.append(
            _quality_row(
                task=task,
                explanation=explanation,
                provenance_status=provenance_status,
                provenance_reasons=provenance_reasons,
                contamination=contamination,
            )
        )

    audit_df = pd.DataFrame(rows)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    audit_path = output_path / "explainer_quality_audit.csv"
    audit_df.to_csv(audit_path, index=False, encoding="utf-8-sig")

    retry_tasks = [
        task
        for task in tasks
        if not bool(
            audit_df.loc[audit_df["task_id"].astype(str) == str(task["task_id"]), "trusted_for_downstream"].any()
        )
    ]
    retry_path = output_path / "explainer_quality_retry_tasks.jsonl"
    write_jsonl(retry_path, retry_tasks)

    trusted_ids = set(
        audit_df.loc[audit_df["trusted_for_downstream"].astype(bool), "task_id"].astype(str).tolist()
    )
    trusted = [row for row in explanations if str(row.get("task_id")) in trusted_ids]
    trusted_path = output_path / "trusted_explanations.jsonl"
    write_jsonl(trusted_path, trusted)

    status_counts = Counter(audit_df.get("quality_status", pd.Series(dtype=str)).astype(str))
    provenance_counts = Counter(audit_df.get("provenance_status", pd.Series(dtype=str)).astype(str))
    summary = {
        "step": "audit-explainer-quality",
        "inputs": {
            "tasks": str(tasks_path),
            "validated_explanations": str(validated_explanations_path),
            "execution_manifest": str(execution_manifest_path),
        },
        "outputs": {
            "quality_audit": str(audit_path),
            "retry_tasks": str(retry_path),
            "trusted_explanations": str(trusted_path),
        },
        "n_tasks": len(tasks),
        "n_validated_explanations": len(explanations),
        "n_trusted_for_downstream": len(trusted),
        "quality_status_counts": dict(status_counts),
        "provenance_status_counts": dict(provenance_counts),
        "suspicious_generator_scripts": suspicious_generators,
        "contamination_detected": contamination,
        "clean_manifest_declared": clean_manifest_declared,
        "stage_gate_complete": bool(len(tasks) > 0 and len(trusted) == len(tasks) and not contamination),
    }
    summary_path = output_path / "explainer_quality_audit_manifest.json"
    write_json(summary_path, summary)

    lines = [
        "# Explainer Quality Audit",
        "",
        "本审计区分 schema 可解析性与解释质量。只有 provenance 完整、证据引用正确、同时覆盖 active 与 near-miss、且未发现本地规则生成污染的任务，才进入 `trusted_explanations.jsonl`。",
        "",
        f"- tasks: {len(tasks)}",
        f"- schema-validated explanations: {len(explanations)}",
        f"- trusted for downstream: {len(trusted)}",
        f"- contamination detected: `{contamination}`",
        "",
        "## Quality Status",
        "",
        "| status | count |",
        "| --- | ---: |",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Provenance Status",
            "",
            "| status | count |",
            "| --- | ---: |",
        ]
    )
    for status, count in sorted(provenance_counts.items()):
        lines.append(f"| {status} | {count} |")
    if suspicious_generators:
        lines.extend(["", "## Suspicious Local Generators", ""])
        lines.extend(f"- `{path}`" for path in suspicious_generators)
        lines.append("\n这些脚本能够基于规则或硬编码定义生成解释，并自行写入模型名与 hash；当前解释不能据此恢复为可信 LLM 执行结果，必须重新生成。")
    report_path = output_path / "explainer_quality_audit_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {**summary, "outputs": {**summary["outputs"], "report": str(report_path)}}


def prepare_explainer_rerun(
    *,
    tasks_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create a clean Claude Code queue without modifying old raw outputs."""
    tasks = read_jsonl(tasks_path)
    output_path = Path(output_dir)
    task_dir = output_path / "llm_tasks"
    raw_dir = output_path / "explainer_outputs" / "raw_rerun"
    clean_tasks_path = task_dir / "explainer_tasks_clean_rerun.jsonl"
    clean_manifest_path = output_path / "llm_execution_manifest_explainer_rerun.jsonl"
    clean_tasks: list[dict[str, Any]] = []
    for task in tasks:
        row = dict(task)
        row["expected_output_path"] = str(raw_dir / f"{task['task_id']}.json")
        row["status"] = "pending_claude_code_llm_clean_rerun"
        row["rerun_reason"] = "previous_output_failed_quality_or_provenance_gate"
        clean_tasks.append(row)
    task_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(clean_tasks_path, clean_tasks)
    clean_manifest_path.write_text("", encoding="utf-8")
    manifest = {
        "step": "prepare-explainer-rerun",
        "inputs": {"tasks": str(tasks_path)},
        "outputs": {
            "clean_tasks": str(clean_tasks_path),
            "raw_output_dir": str(raw_dir),
            "execution_manifest": str(clean_manifest_path),
        },
        "n_tasks": len(clean_tasks),
        "old_outputs_preserved": True,
        "execution_requirement": "Claude Code must generate each JSON response; local rule-based generators are prohibited.",
    }
    manifest_path = output_path / "explainer_rerun_manifest.json"
    write_json(manifest_path, manifest)
    return manifest


__all__ = ["audit_explainer_quality", "parse_prompt_samples", "prepare_explainer_rerun"]
