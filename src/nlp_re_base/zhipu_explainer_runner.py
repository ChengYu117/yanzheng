"""Direct Zhipu API execution for the stable-core Explainer pilot.

The runner deliberately keeps model execution outside any coding-agent or
sub-agent context. Each task is an independent chat completion, and only the
task prompt is sent as user content. Raw model text and a hashable execution
manifest are written immediately so the run can resume without duplicate
requests.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .contrastive_evidence_pack import read_jsonl, write_json, write_jsonl


SHORT_SYSTEM_PROMPT = (
    "You are an evaluator of anonymous SAE latent activation patterns. "
    "Follow the user task exactly, return only the requested JSON, and use cautious non-causal wording."
)
TRUSTED_EXECUTION_MODE = "zhipu_api"


@dataclass(frozen=True)
class ZhipuExplainerConfig:
    model: str = "glm-4.7"
    per_label: int = 2
    concurrency: int = 2
    max_retries: int = 5
    temperature: float = 0.2
    max_tokens: int = 2048
    random_state: int = 42


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _response_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("Zhipu response has no choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        content = "".join(parts)
    content = str(content or "").strip()
    if not content:
        raise ValueError("Zhipu response content is empty")
    return content


def _response_meta(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    usage_payload: Any = None
    if usage is not None:
        if hasattr(usage, "model_dump"):
            usage_payload = usage.model_dump()
        elif hasattr(usage, "__dict__"):
            usage_payload = dict(usage.__dict__)
        elif isinstance(usage, dict):
            usage_payload = usage
    return {
        "request_id": str(getattr(response, "id", "") or getattr(response, "request_id", "")),
        "usage": usage_payload,
        "model": str(getattr(response, "model", "") or ""),
    }


def _is_retryable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if "1302" in text or "达到速率限制" in text or "rate limit" in text:
        return False
    return any(token in text for token in ("429", "500", "502", "503", "504", "timeout", "timed out", "temporarily"))


def select_pilot_tasks(
    *,
    tasks_path: str | Path,
    summary_path: str | Path,
    output_dir: str | Path,
    config: ZhipuExplainerConfig = ZhipuExplainerConfig(),
) -> dict[str, Any]:
    """Select two packets per label and both explainer repeats."""
    tasks = read_jsonl(tasks_path)
    summary = pd.read_csv(summary_path)
    required = {"packet_id", "target_label", "rank_within_label", "inclusion_frequency"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Evidence summary missing columns: {missing}")
    labels = sorted(summary["target_label"].astype(str).unique().tolist())
    if len(labels) != 9:
        raise ValueError(f"Expected 9 labels for the pilot, found {labels}")
    selected_packets: list[dict[str, Any]] = []
    for label in labels:
        group = summary[summary["target_label"].astype(str) == label].copy()
        group["inclusion_frequency"] = pd.to_numeric(group["inclusion_frequency"], errors="coerce").fillna(-1.0)
        group["rank_within_label"] = pd.to_numeric(group["rank_within_label"], errors="coerce").fillna(10**9)
        group["latent_idx"] = pd.to_numeric(group.get("latent_idx", 10**9), errors="coerce").fillna(10**9)
        group = group.sort_values(
            ["inclusion_frequency", "rank_within_label", "latent_idx"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        if len(group) < int(config.per_label):
            raise ValueError(f"Label {label} has fewer than {config.per_label} candidate packets")
        selected_packets.extend(group.head(int(config.per_label)).to_dict(orient="records"))

    selected_ids = {str(row["packet_id"]) for row in selected_packets}
    selected_tasks = [task for task in tasks if str(task.get("packet_id")) in selected_ids]
    selected_tasks.sort(key=lambda row: str(row.get("task_id", "")))
    expected_count = len(labels) * int(config.per_label) * 2
    if len(selected_tasks) != expected_count:
        raise ValueError(
            f"Pilot task selection expected {expected_count} tasks, found {len(selected_tasks)}; "
            "both r01/r02 tasks must exist for every selected packet."
        )

    output_path = Path(output_dir)
    task_dir = output_path / "llm_tasks"
    raw_dir = output_path / "explainer_outputs" / "raw"
    task_rows: list[dict[str, Any]] = []
    for task in selected_tasks:
        row = dict(task)
        row["expected_output_path"] = str(raw_dir / f"{task['task_id']}.json")
        row["status"] = "pending_zhipu_api"
        row["execution_mode"] = TRUSTED_EXECUTION_MODE
        task_rows.append(row)
    task_path = task_dir / "explainer_tasks.jsonl"
    write_jsonl(task_path, task_rows)
    selection = {
        "step": "select-zhipu-explainer-pilot",
        "inputs": {"tasks": str(tasks_path), "evidence_summary": str(summary_path)},
        "outputs": {"tasks": str(task_path), "raw_output_dir": str(raw_dir)},
        "parameters": asdict(config),
        "labels": labels,
        "selected_packets": selected_packets,
        "n_tasks": len(task_rows),
        "n_packets": len(selected_packets),
    }
    write_json(output_path / "pilot_selection_manifest.json", selection)
    return selection


def _default_client_factory(api_key: str) -> Any:
    try:
        from zai import ZhipuAiClient
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("zai-sdk is not installed; run `conda run -n qwen-env-py311 python -m pip install zai-sdk`") from exc
    return ZhipuAiClient(api_key=api_key)


def run_zhipu_tasks(
    *,
    tasks_path: str | Path,
    output_dir: str | Path,
    api_key: str,
    config: ZhipuExplainerConfig = ZhipuExplainerConfig(),
    client_factory: Callable[[str], Any] = _default_client_factory,
    resume: bool = True,
    max_tasks: int | None = None,
    system_prompt: str = SHORT_SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Run selected tasks through the Zhipu chat API with resumable writes."""
    if not api_key.strip():
        raise ValueError("API key is empty")
    if int(config.concurrency) < 1:
        raise ValueError("concurrency must be >= 1")
    if not str(system_prompt).strip():
        raise ValueError("system_prompt is empty")
    tasks = read_jsonl(tasks_path)
    output_path = Path(output_dir)
    manifest_path = output_path / "llm_execution_manifest.jsonl"
    error_path = output_path / "llm_execution_errors.jsonl"
    response_dir = output_path / "api_responses"
    raw_dir = output_path / "explainer_outputs" / "raw"
    response_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict[str, Any]] = {}
    if resume and manifest_path.exists():
        for row in read_jsonl(manifest_path):
            if row.get("status") == "success" and row.get("task_id"):
                existing[str(row["task_id"])] = row
    manifest_lock = threading.Lock()
    error_lock = threading.Lock()

    def save_manifest() -> None:
        with manifest_lock:
            write_jsonl(manifest_path, [existing[key] for key in sorted(existing)])

    def save_error(row: dict[str, Any]) -> None:
        with error_lock:
            with error_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def run_one(task: dict[str, Any]) -> tuple[str, str]:
        task_id = str(task["task_id"])
        raw_path = Path(str(task["expected_output_path"]))
        old = existing.get(task_id)
        if resume and old and raw_path.exists():
            if old.get("prompt_sha256") == _sha256_text(str(task.get("prompt", ""))) and old.get("raw_output_sha256") == _sha256_bytes(raw_path.read_bytes()):
                return task_id, "skipped"
        client = client_factory(api_key)
        last_error: Exception | None = None
        for attempt in range(1, int(config.max_retries) + 2):
            try:
                response = client.chat.completions.create(
                    model=config.model,
                    messages=[
                        {"role": "system", "content": str(system_prompt)},
                        {"role": "user", "content": str(task["prompt"])},
                    ],
                    temperature=float(config.temperature),
                    max_tokens=int(config.max_tokens),
                    stream=False,
                )
                content = _response_content(response)
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write_text(raw_path, content + "\n")
                meta = _response_meta(response)
                response_payload = {
                    "task_id": task_id,
                    "model": config.model,
                    "attempt": attempt,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **meta,
                }
                write_json(response_dir / f"{task_id}.json", response_payload)
                existing[task_id] = {
                    "task_id": task_id,
                    "model": config.model,
                    "provider": "zhipu",
                    "execution_mode": TRUSTED_EXECUTION_MODE,
                    "status": "success",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompt_sha256": _sha256_text(str(task.get("prompt", ""))),
                    "system_prompt_sha256": _sha256_text(str(system_prompt)),
                    "raw_output_sha256": _sha256_bytes(raw_path.read_bytes()),
                    "request_id": meta.get("request_id", ""),
                    "attempt": attempt,
                }
                save_manifest()
                return task_id, "success"
            except Exception as exc:  # noqa: PERF203 - task-level retry boundary
                last_error = exc
                if attempt > int(config.max_retries) or not _is_retryable_error(exc):
                    break
                delay = min(60.0, 2.0 ** (attempt - 1)) + random.random() * 0.25
                time.sleep(delay)
        save_error(
            {
                "task_id": task_id,
                "model": config.model,
                "provider": "zhipu",
                "execution_mode": TRUSTED_EXECUTION_MODE,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "attempts": int(config.max_retries) + 1,
                "error_type": type(last_error).__name__ if last_error else "UnknownError",
                "error": str(last_error) if last_error else "unknown",
            }
        )
        return task_id, "failed"

    counts = {"success": 0, "skipped": 0, "failed": 0}
    pending = [task for task in tasks if not (resume and str(task["task_id"]) in existing and Path(str(task["expected_output_path"])).exists())]
    if max_tasks is not None and int(max_tasks) > 0:
        pending = pending[: int(max_tasks)]
    with ThreadPoolExecutor(max_workers=int(config.concurrency)) as executor:
        futures = [executor.submit(run_one, task) for task in pending]
        for future in as_completed(futures):
            _, status = future.result()
            counts[status] = counts.get(status, 0) + 1
    summary = {
        "step": "run-zhipu-explainer-pilot",
        "inputs": {"tasks": str(tasks_path)},
        "outputs": {
            "manifest": str(manifest_path),
            "errors": str(error_path),
            "raw_output_dir": str(raw_dir),
            "api_response_dir": str(response_dir),
        },
        "parameters": asdict(config),
        "n_tasks": len(tasks),
        "n_pending": len(pending),
        "max_tasks": max_tasks,
        "counts": counts,
        "cumulative_counts": {
            "success": len(existing),
            "failed": sum(1 for _ in read_jsonl(error_path)) if error_path.exists() else 0,
        },
        "api_key_source": "environment_only",
    }
    write_json(output_path / "zhipu_pilot_run_manifest.json", summary)
    return summary


def check_zhipu_environment(*, api_key_env: str = "ZHIPUAI_API_KEY") -> dict[str, Any]:
    result = {"api_key_env": api_key_env, "api_key_present": bool(os.getenv(api_key_env)), "sdk_importable": False}
    try:
        import zai  # noqa: F401

        result["sdk_importable"] = True
    except ImportError:
        result["sdk_importable"] = False
    result["ready"] = bool(result["api_key_present"] and result["sdk_importable"])
    return result


def write_pilot_report(output_dir: str | Path) -> Path:
    """Write a compact pilot decision report after validation/audit."""
    output_path = Path(output_dir)
    selection = json.loads((output_path / "pilot_selection_manifest.json").read_text(encoding="utf-8"))
    run_manifest_path = output_path / "zhipu_pilot_run_manifest.json"
    audit_path = output_path / "explainer_outputs" / "explainer_quality_audit_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8")) if run_manifest_path.exists() else {}
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    execution_manifest_path = output_path / "llm_execution_manifest.jsonl"
    error_path = output_path / "llm_execution_errors.jsonl"
    cumulative_success = len(read_jsonl(execution_manifest_path)) if execution_manifest_path.exists() else 0
    cumulative_failed = len(read_jsonl(error_path)) if error_path.exists() else 0
    lines = [
        "# GLM-4.7 Explainer Pilot Report",
        "",
        "本报告只覆盖 stable-core latent 的 Explainer pilot，不代表完整 P3 结果。",
        "",
        f"- model: `{selection.get('parameters', {}).get('model', 'glm-4.7')}`",
        f"- execution mode: `{TRUSTED_EXECUTION_MODE}`",
        f"- tasks: {selection.get('n_tasks', 0)}",
        f"- packets: {selection.get('n_packets', 0)}",
        f"- concurrency: {selection.get('parameters', {}).get('concurrency', 2)}",
        "",
        "## Execution",
        "",
        f"- cumulative run counts: `{json.dumps({'success': cumulative_success, 'failed': cumulative_failed}, ensure_ascii=False)}`",
        f"- errors: `{output_path / 'llm_execution_errors.jsonl'}`",
        f"- manifest: `{output_path / 'llm_execution_manifest.jsonl'}`",
        "",
        "## Quality Gate",
        "",
    ]
    if audit:
        lines.extend(
            [
                f"- schema-validated explanations: {audit.get('n_validated_explanations', 0)} / {audit.get('n_tasks', 0)}",
                f"- trusted for downstream: {audit.get('n_trusted_for_downstream', 0)}",
                f"- contamination detected: `{audit.get('contamination_detected', True)}`",
                f"- stage gate complete: `{audit.get('stage_gate_complete', False)}`",
                f"- quality details: `{output_path / 'explainer_outputs' / 'explainer_quality_audit_report.md'}`",
            ]
        )
    else:
        lines.append("尚未执行 validator 和 quality audit；当前不能判断解释是否可信。")
    report_path = output_path / "zhipu_glm47_explainer_pilot_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


__all__ = [
    "SHORT_SYSTEM_PROMPT",
    "TRUSTED_EXECUTION_MODE",
    "ZhipuExplainerConfig",
    "check_zhipu_environment",
    "run_zhipu_tasks",
    "select_pilot_tasks",
    "write_pilot_report",
]
