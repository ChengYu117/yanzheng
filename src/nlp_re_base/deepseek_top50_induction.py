"""Batch Top-50 latent induction through DeepSeek's chat-completions API.

This module deliberately stops after aggregate Top-50 induction and its
structural validation. It does not create held-out scorer or minimal-pair
tasks, because those are separate P3 stages.
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

import numpy as np
import pandas as pd
import requests

from .contrastive_evidence_pack import normalise_text, read_jsonl, write_json, write_jsonl
from .zhipu_single_latent_p3 import TOP50_EXPLAINER_SYSTEM_PROMPT
from .contrastive_llm_io import parse_llm_json_file


DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_EXECUTION_MODE = "deepseek_api"
DEEPSEEK_DUAL_TRACK_FIELDS: tuple[str, ...] = (
    "latent_idx",
    "short_name",
    "candidate_explanation",
    "surface_pattern",
    "surface_supporting_sample_ids",
    "surface_outlier_sample_ids",
    "surface_representative_evidence_ids",
    "surface_confidence",
    "semantic_pattern",
    "semantic_supporting_sample_ids",
    "semantic_outlier_sample_ids",
    "semantic_representative_evidence_ids",
    "semantic_confidence",
    "alternative_hypotheses",
    "failure_modes",
)


@dataclass(frozen=True)
class DeepSeekTop50Config:
    model: str = "deepseek-v4-flash"
    concurrency: int = 225
    max_retries: int = 2
    timeout_seconds: float = 180.0
    temperature: float = 0.1
    max_tokens: int = 4096
    top_n: int = 50


def _format_dual_track_samples(samples: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- id={sample['id']} rank={sample['rank']} activation={float(sample['activation']):.6g}\n"
        f"  text: {str(sample['text']).replace(chr(10), ' ').strip()}"
        for sample in samples
    )


def build_deepseek_dual_track_prompt(pack: dict[str, Any]) -> str:
    samples = list(pack["samples_for_explainer"])
    sample_ids = [str(sample["id"]) for sample in samples]
    if len(samples) != 50 or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Dual-track Top-50 explainer requires exactly 50 uniquely identified samples")
    n_unique = len({normalise_text(sample["text"]) for sample in samples})
    return f"""Analyze one anonymous SAE latent from its 50 highest-activation utterances as one set.

Latent metadata: latent_idx={int(pack['latent_idx'])}
Evidence size: 50 items, {n_unique} unique normalized texts. Each item is the highest-activation
representative of its normalized text group.

Core instruction:
- Produce BOTH required tracks.
- Surface track: identify a recurring lexical, phrase, syntactic, discourse-marker, formatting, or
  domain-terminology pattern that explains a majority of the samples, if one exists.
- Semantic track: identify a recurring semantic content or dialogue function that explains a majority of
  the samples, if one exists.
- The two tracks may describe different patterns and may have different supporting samples. Do not force
  them to agree. If a track has no majority-supported pattern, state that explicitly in its pattern field.
- Reason across all 50 utterances. Do not produce 50 sentence-by-sentence mini-analyses.
- For EACH track, partition every sample id exactly once into that track's supporting ids or outlier ids.
- candidate_explanation is a short synthesis for a human reader. It must state whether the most useful
  candidate is surface-form, semantic/dialogue-function, or unresolved between the two tracks.

Top-50 samples:
{_format_dual_track_samples(samples)}

Return only one JSON object with exactly these fields:
{json.dumps(list(DEEPSEEK_DUAL_TRACK_FIELDS), ensure_ascii=False)}

Requirements:
- When a track has supporting ids, its representative evidence ids must contain 2 to 3 ids from that same
  track's supporting ids. When a track has no supporting ids, its representative evidence ids must be an
  empty array.
- alternative_hypotheses and failure_modes must be JSON arrays of strings.
- Keep each prose field to at most 2 sentences and each prose array to at most 4 items.
- A majority claim requires at least 50% of the 50 unique normalized texts in that track's supporting ids.
- surface_confidence and semantic_confidence must each be numbers from 0 to 1 and use this calibration:
  0.00-0.20: no consistent pattern.
  0.21-0.40: only a weak pattern, or multiple equally plausible explanations.
  0.41-0.60: barely meets the majority condition, with clear confounds.
  0.61-0.80: the majority condition is clearly met and the track's main pattern is fairly consistent.
  0.81-1.00: coverage is very high and alternative explanations or confounds are weak.
"""


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
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _load_feature_tensor(path: str | Path) -> Any:
    feature_path = Path(path)
    if feature_path.suffix != ".pt":
        raise ValueError("Batch DeepSeek Top-50 induction currently requires a .pt feature store")
    import torch

    payload = torch.load(feature_path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        tensor = payload
    elif isinstance(payload, dict):
        for key in ("utterance_features", "features", "X"):
            if key in payload:
                tensor = payload[key]
                break
        else:
            raise KeyError(f"{feature_path} has no utterance feature tensor")
    else:
        raise TypeError(f"Unsupported feature payload: {type(payload).__name__}")
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise ValueError(f"Expected 2D tensor feature store, got {type(tensor).__name__} shape={getattr(tensor, 'shape', None)}")
    return tensor.detach().cpu()


def _load_texts(records_path: str | Path) -> list[str]:
    records = read_jsonl(records_path)
    texts = [str(record.get("unit_text") or record.get("text") or record.get("utterance") or "") for record in records]
    if any(not text.strip() for text in texts):
        raise ValueError("records contain an empty usable utterance text")
    return texts


def _stable_unique_latents(stable_latents_path: str | Path) -> pd.DataFrame:
    source = pd.read_csv(stable_latents_path)
    required = {"label", "latent_idx", "stable_set_role"}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"stable latent table missing columns: {missing}")
    stable = source[source["stable_set_role"].astype(str) == "stable_core"].copy()
    stable["label"] = stable["label"].astype(str).str.upper()
    stable["latent_idx"] = pd.to_numeric(stable["latent_idx"], errors="coerce").astype("Int64")
    stable = stable.dropna(subset=["latent_idx"]).copy()
    stable["latent_idx"] = stable["latent_idx"].astype(int)
    rows: list[dict[str, Any]] = []
    for latent_idx, group in stable.groupby("latent_idx", sort=True):
        labels = sorted(group["label"].astype(str).unique().tolist())
        rows.append(
            {
                "latent_idx": int(latent_idx),
                "associated_labels_internal": labels,
                "n_stable_associations": int(len(group)),
                "max_inclusion_frequency": float(pd.to_numeric(group.get("inclusion_frequency"), errors="coerce").max()),
                "max_abs_cohens_d": float(pd.to_numeric(group.get("abs_cohens_d"), errors="coerce").max()),
            }
        )
    return pd.DataFrame(rows).sort_values("latent_idx", kind="mergesort").reset_index(drop=True)


def _top50_samples(
    *,
    values: np.ndarray,
    texts: list[str],
    normalised_texts: list[str],
    latent_idx: int,
    top_n: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(values) != len(texts) or len(texts) != len(normalised_texts):
        raise ValueError("activation values, texts, and normalized texts must have matching lengths")

    # Keep one representative per normalized text before ranking. Equal activations
    # are resolved by the original row order, making task construction reproducible.
    best_row_by_text: dict[str, int] = {}
    for row_id, normalized_text in enumerate(normalised_texts):
        previous = best_row_by_text.get(normalized_text)
        if previous is None or float(values[row_id]) > float(values[previous]):
            best_row_by_text[normalized_text] = row_id

    candidate_row_ids = np.asarray(list(best_row_by_text.values()), dtype=np.int64)
    order = candidate_row_ids[np.lexsort((candidate_row_ids, -values[candidate_row_ids]))]
    selected = [int(row_id) for row_id in order[:top_n].tolist()]
    if len(selected) != top_n:
        raise ValueError(
            f"latent {latent_idx} has only {len(selected)} unique normalized texts for Top-{top_n}"
        )
    visible: list[dict[str, Any]] = []
    internal: list[dict[str, Any]] = []
    for rank, row_id in enumerate(selected, start=1):
        text = texts[row_id]
        sample_id = f"s{rank:03d}"
        item = {
            "id": sample_id,
            "rank": int(rank),
            "activation": float(values[row_id]),
            "text": text,
        }
        visible.append(item)
        internal.append({**item, "row_idx": int(row_id)})
    return visible, internal


def build_deepseek_top50_tasks(
    *,
    stable_latents_path: str | Path,
    feature_store_path: str | Path,
    records_path: str | Path,
    output_dir: str | Path,
    config: DeepSeekTop50Config = DeepSeekTop50Config(),
) -> dict[str, Any]:
    """Generate one label-blind Top-50 induction task for every unique stable latent."""
    if int(config.top_n) != 50:
        raise ValueError("The aggregate induction protocol is fixed to Top-50")
    output_path = Path(output_dir)
    unique_latents = _stable_unique_latents(stable_latents_path)
    tensor = _load_feature_tensor(feature_store_path)
    texts = _load_texts(records_path)
    normalised_texts = [normalise_text(text) for text in texts]
    if tensor.shape[0] != len(texts):
        raise ValueError(f"feature rows {tensor.shape[0]} do not match record rows {len(texts)}")
    if unique_latents.empty:
        raise ValueError("No stable_core latents found")
    indices = unique_latents["latent_idx"].to_numpy(dtype=np.int64)
    if int(indices.max()) >= int(tensor.shape[1]):
        raise ValueError(f"stable latent index {int(indices.max())} exceeds feature width {tensor.shape[1]}")

    task_dir = output_path / "llm_tasks"
    raw_dir = output_path / "explainer_outputs" / "raw"
    evidence_dir = output_path / "evidence_packs"
    task_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Select only the 225 relevant columns before conversion to NumPy.
    selected_values = tensor[:, indices.tolist()].float().numpy()
    tasks: list[dict[str, Any]] = []
    packs: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    for column, metadata in enumerate(unique_latents.to_dict(orient="records")):
        latent_idx = int(metadata["latent_idx"])
        visible, internal = _top50_samples(
            values=np.asarray(selected_values[:, column], dtype=np.float32),
            texts=texts,
            normalised_texts=normalised_texts,
            latent_idx=latent_idx,
            top_n=int(config.top_n),
        )
        packet_id = f"dst50_{latent_idx:05d}"
        pack = {
            "packet_id": packet_id,
            "latent_idx": latent_idx,
            "associated_labels_internal": metadata["associated_labels_internal"],
            "n_stable_associations": int(metadata["n_stable_associations"]),
            "samples_internal": internal,
            "samples_for_explainer": visible,
            "summary": {
                "top_n": int(config.top_n),
                "selection_unit": "unique_normalized_text",
                "selection_order": "activation_desc_then_row_idx_asc",
                "unique_normalized_texts": int(len({normalise_text(sample["text"]) for sample in visible})),
                "duplicate_rows": 0,
                "source_unique_normalized_texts": int(len(set(normalised_texts))),
                "source_duplicate_rows_suppressed": int(len(texts) - len(set(normalised_texts))),
                "label_blind_explainer": True,
            },
        }
        task_id = f"{packet_id}_top50_explainer"
        task = {
            "task_id": task_id,
            "task_type": "top50_aggregate_latent_explainer",
            "packet_id": packet_id,
            "latent_idx": latent_idx,
            "repeat": 1,
            "selection_unit": pack["summary"]["selection_unit"],
            "selection_order": pack["summary"]["selection_order"],
            "prompt": build_deepseek_dual_track_prompt(pack),
            "visible_samples": visible,
            "expected_output_path": str(raw_dir / f"{task_id}.json"),
            "output_format": "single_json_object",
            "status": "pending_deepseek_api",
        }
        tasks.append(task)
        packs.append(pack)
        index_rows.append(
            {
                "task_id": task_id,
                "packet_id": packet_id,
                "latent_idx": latent_idx,
                "associated_labels_internal": ",".join(metadata["associated_labels_internal"]),
                "n_stable_associations": int(metadata["n_stable_associations"]),
                "selection_unit": pack["summary"]["selection_unit"],
                "selection_order": pack["summary"]["selection_order"],
                "top50_unique_normalized_texts": pack["summary"]["unique_normalized_texts"],
                "top50_duplicate_rows": pack["summary"]["duplicate_rows"],
                "source_duplicate_rows_suppressed": pack["summary"]["source_duplicate_rows_suppressed"],
            }
        )

    tasks_path = task_dir / "deepseek_top50_induction_tasks.jsonl"
    packs_path = evidence_dir / "unique_latent_top50_packs.jsonl"
    index_path = evidence_dir / "unique_latent_top50_task_index.csv"
    write_jsonl(tasks_path, tasks)
    write_jsonl(packs_path, packs)
    pd.DataFrame(index_rows).to_csv(index_path, index=False, encoding="utf-8-sig")
    manifest = {
        "analysis": "deepseek_v4_flash_top50_dual_track_induction",
        "step": "build-top50-dual-track-induction-tasks",
        "inputs": {
            "stable_latents": str(stable_latents_path),
            "feature_store": str(feature_store_path),
            "records": str(records_path),
        },
        "outputs": {"tasks": str(tasks_path), "packs": str(packs_path), "task_index": str(index_path)},
        "parameters": asdict(config),
        "n_stable_label_latent_rows": int(sum(unique_latents["n_stable_associations"])),
        "n_unique_latents": int(len(unique_latents)),
        "n_tasks": int(len(tasks)),
        "selection_unit": "unique_normalized_text",
        "selection_order": "activation_desc_then_row_idx_asc",
        "label_blind": True,
        "stages_not_run": ["heldout_scorer", "label_baseline", "minimal_pairs", "sae_activation_test"],
    }
    write_json(output_path / "task_build_manifest.json", manifest)
    return manifest


def _is_retryable(status_code: int | None, error_text: str) -> bool:
    if status_code is not None and status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    text = error_text.lower()
    return any(token in text for token in ("timeout", "temporarily", "connection reset", "rate limit"))


def _default_request(
    *, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
) -> requests.Response:
    return requests.post(url, headers=headers, json=payload, timeout=(15.0, timeout))


def run_deepseek_top50_tasks(
    *,
    tasks_path: str | Path,
    output_dir: str | Path,
    api_key: str,
    config: DeepSeekTop50Config = DeepSeekTop50Config(),
    system_prompt: str = TOP50_EXPLAINER_SYSTEM_PROMPT,
    endpoint: str = DEEPSEEK_CHAT_COMPLETIONS_URL,
    request_fn: Callable[..., requests.Response] = _default_request,
    resume: bool = True,
    force: bool = False,
    max_tasks: int | None = None,
    analysis_name: str = "deepseek_v4_flash_top50_dual_track_induction",
    step_name: str = "run-deepseek-top50-dual-track-induction",
) -> dict[str, Any]:
    """Run label-blind Top-50 induction tasks with resumable raw-output provenance."""
    if not api_key.strip():
        raise ValueError("API key is empty")
    if int(config.concurrency) < 1:
        raise ValueError("concurrency must be at least one")
    if not str(system_prompt).strip():
        raise ValueError("system prompt is empty")
    tasks = read_jsonl(tasks_path)
    output_path = Path(output_dir)
    manifest_path = output_path / "llm_execution_manifest.jsonl"
    error_path = output_path / "llm_execution_errors.jsonl"
    response_dir = output_path / "api_responses"
    response_dir.mkdir(parents=True, exist_ok=True)

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

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def run_one(task: dict[str, Any]) -> tuple[str, str]:
        task_id = str(task["task_id"])
        raw_path = Path(str(task["expected_output_path"]))
        old = existing.get(task_id)
        if resume and not force and old and raw_path.exists():
            if (
                old.get("prompt_sha256") == _sha256_text(str(task.get("prompt", "")))
                and old.get("raw_output_sha256") == _sha256_bytes(raw_path.read_bytes())
            ):
                return task_id, "skipped"
        last_status: int | None = None
        last_error = "unknown"
        for attempt in range(1, int(config.max_retries) + 2):
            request_payload = {
                "model": config.model,
                "messages": [
                    {"role": "system", "content": str(system_prompt)},
                    {"role": "user", "content": str(task["prompt"])},
                ],
                "temperature": float(config.temperature),
                "max_tokens": int(config.max_tokens),
                "stream": False,
                "thinking": {"type": "disabled"},
                "response_format": {"type": "json_object"},
            }
            try:
                response = request_fn(
                    url=endpoint,
                    headers=headers,
                    payload=request_payload,
                    timeout=float(config.timeout_seconds),
                )
                last_status = int(getattr(response, "status_code", 200))
                if last_status >= 400:
                    last_error = str(getattr(response, "text", "HTTP error"))
                    raise RuntimeError(f"HTTP {last_status}: {last_error}")
                payload = response.json()
                choices = payload.get("choices", []) if isinstance(payload, dict) else []
                if not choices:
                    raise ValueError("DeepSeek response has no choices")
                message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                content = str(message.get("content") or "").strip()
                if not content:
                    raise ValueError("DeepSeek response content is empty")
                _atomic_write_text(raw_path, content + "\n")
                response_meta = {
                    "task_id": task_id,
                    "model": str(payload.get("model") or config.model),
                    "request_id": str(payload.get("id") or ""),
                    "usage": payload.get("usage"),
                    "finish_reason": choices[0].get("finish_reason") if isinstance(choices[0], dict) else "",
                    "attempt": attempt,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                write_json(response_dir / f"{task_id}.json", response_meta)
                existing[task_id] = {
                    "task_id": task_id,
                    "model": config.model,
                    "provider": "deepseek",
                    "execution_mode": DEEPSEEK_EXECUTION_MODE,
                    "status": "success",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompt_sha256": _sha256_text(str(task.get("prompt", ""))),
                    "system_prompt_sha256": _sha256_text(str(system_prompt)),
                    "raw_output_sha256": _sha256_bytes(raw_path.read_bytes()),
                    "request_id": response_meta["request_id"],
                    "attempt": attempt,
                }
                save_manifest()
                return task_id, "success"
            except Exception as exc:  # noqa: PERF203 - task-level API retry boundary
                last_error = str(exc)
                if attempt > int(config.max_retries) or not _is_retryable(last_status, last_error):
                    break
                time.sleep(min(20.0, 1.5 * 2 ** (attempt - 1)) + random.random() * 0.25)
        save_error(
            {
                "task_id": task_id,
                "model": config.model,
                "provider": "deepseek",
                "execution_mode": DEEPSEEK_EXECUTION_MODE,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status_code": last_status,
                "error": last_error,
            }
        )
        return task_id, "failed"

    pending: list[dict[str, Any]] = []
    pre_skipped = 0
    for task in tasks:
        task_id = str(task["task_id"])
        raw_path = Path(str(task["expected_output_path"]))
        old = existing.get(task_id)
        unchanged_success = bool(
            resume
            and not force
            and old
            and raw_path.exists()
            and old.get("prompt_sha256") == _sha256_text(str(task.get("prompt", "")))
            and old.get("raw_output_sha256") == _sha256_bytes(raw_path.read_bytes())
        )
        if unchanged_success:
            pre_skipped += 1
        else:
            pending.append(task)
    if max_tasks is not None and int(max_tasks) > 0:
        pending = pending[: int(max_tasks)]
    counts = {"success": 0, "skipped": pre_skipped, "failed": 0}
    with ThreadPoolExecutor(max_workers=min(int(config.concurrency), max(len(pending), 1))) as executor:
        futures = [executor.submit(run_one, task) for task in pending]
        for future in as_completed(futures):
            _, status = future.result()
            counts[status] += 1
    result = {
        "analysis": str(analysis_name),
        "step": str(step_name),
        "inputs": {"tasks": str(tasks_path)},
        "outputs": {
            "manifest": str(manifest_path),
            "errors": str(error_path),
            "raw_output_dir": str(Path(tasks[0]["expected_output_path"]).parent) if tasks else "",
            "api_response_dir": str(response_dir),
        },
        "parameters": asdict(config),
        "n_tasks": len(tasks),
        "n_pending": len(pending),
        "selection_unit": str(tasks[0].get("selection_unit") or "raw_utterance_row") if tasks else "",
        "counts": counts,
        "cumulative_success": len(existing),
        "api_key_source": "process_only",
    }
    write_json(output_path / "deepseek_run_manifest.json", result)
    return result


def make_deepseek_refined_retry_tasks(
    *,
    all_tasks_path: str | Path,
    retry_tasks_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Refresh failed tasks with the current concise Top-50 prompt without touching valid tasks."""
    all_task_rows = read_jsonl(all_tasks_path)
    all_tasks = {str(task["task_id"]): task for task in all_task_rows}
    retry_tasks = read_jsonl(retry_tasks_path)
    refined: list[dict[str, Any]] = []
    for retry in retry_tasks:
        task_id = str(retry["task_id"])
        task = all_tasks.get(task_id)
        if task is None:
            raise KeyError(f"Retry task {task_id} is missing from all tasks")
        updated = dict(task)
        updated["prompt"] = build_deepseek_dual_track_prompt(
            {
                "latent_idx": int(task["latent_idx"]),
                "samples_for_explainer": task["visible_samples"],
                "summary": {"selection_unit": task.get("selection_unit", "raw_utterance_row")},
            }
        ) + """

Retry-specific compactness rule:
- Write both tracks' supporting ids, outlier ids, and representative evidence ids before all prose fields.
- For EACH track, output exactly 2 representative evidence ids. Each must occur in that same track's
  supporting ids. Before returning JSON, verify both track partitions cover all 50 ids exactly once.
- Every prose string must be no more than 220 characters. Each prose array may contain at most 3 short items.
- Never enumerate example phrases in prose. The sample-id arrays are the only complete enumeration.
- Keep both the surface and semantic tracks. If a track has no majority-supported pattern, say so in that
  track's pattern field and retain its honest partition. Do not broaden a weak pattern just to reach 50%.
"""
        updated["status"] = "retry_deepseek_api"
        refined.append(updated)
    output_path = Path(output_dir)
    refined_path = output_path / "llm_tasks" / "deepseek_top50_refined_retry_tasks.jsonl"
    write_jsonl(refined_path, refined)
    refined_by_id = {str(task["task_id"]): task for task in refined}
    updated_all_tasks = [refined_by_id.get(str(task["task_id"]), task) for task in all_task_rows]
    write_jsonl(all_tasks_path, updated_all_tasks)
    manifest = {
        "analysis": "deepseek_v4_flash_top50_dual_track_induction",
        "step": "make-refined-retry-tasks",
        "inputs": {"all_tasks": str(all_tasks_path), "retry_tasks": str(retry_tasks_path)},
        "outputs": {
            "refined_retry_tasks": str(refined_path),
            "updated_all_tasks": str(all_tasks_path),
        },
        "n_tasks": len(refined),
        "prompt_change": "concise prose and bounded prose arrays; complete id partition unchanged",
    }
    write_json(output_path / "refined_retry_task_manifest.json", manifest)
    return manifest


def _id_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    rows = [str(item).strip() for item in value]
    if any(not row for row in rows):
        raise ValueError(f"{field} contains an empty item")
    return rows


def _text_list(value: Any, field: str) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return _id_list(value, field)


def validate_deepseek_top50_outputs(
    *,
    tasks_path: str | Path,
    execution_manifest_path: str | Path,
    output_dir: str | Path,
    minimum_raw_support_fraction: float = 0.50,
    minimum_unique_support_fraction: float = 0.50,
) -> dict[str, Any]:
    """Validate every task independently so one malformed response cannot abort a batch."""
    tasks = read_jsonl(tasks_path)
    manifest_rows = read_jsonl(execution_manifest_path) if Path(execution_manifest_path).exists() else []
    valid: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    retry_tasks: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        reasons: list[str] = []
        row: dict[str, Any] | None = None
        raw_path = Path(str(task.get("expected_output_path", "")))
        try:
            matching_manifest = [
                item
                for item in manifest_rows
                if str(item.get("task_id")) == task_id and item.get("status") == "success"
            ]
            if len(matching_manifest) != 1:
                reasons.append(f"execution_manifest_success_count={len(matching_manifest)}")
            elif matching_manifest[0].get("execution_mode") != DEEPSEEK_EXECUTION_MODE:
                reasons.append("execution_mode_mismatch")
            if not raw_path.exists():
                raise FileNotFoundError("raw_output_missing")
            if matching_manifest:
                manifest_row = matching_manifest[0]
                if manifest_row.get("prompt_sha256") != _sha256_text(str(task.get("prompt", ""))):
                    reasons.append("prompt_hash_mismatch")
                if manifest_row.get("raw_output_sha256") != _sha256_bytes(raw_path.read_bytes()):
                    reasons.append("raw_output_hash_mismatch")
            payload = parse_llm_json_file(raw_path)
            if not isinstance(payload, dict):
                raise ValueError("output must be a JSON object")
            missing = [field for field in DEEPSEEK_DUAL_TRACK_FIELDS if field not in payload]
            if missing:
                raise ValueError(f"missing_fields={','.join(missing)}")
            if int(payload["latent_idx"]) != int(task["latent_idx"]):
                reasons.append("latent_idx_mismatch")
            visible = {str(sample["id"]): sample for sample in task["visible_samples"]}
            all_ids = set(visible)
            def read_track(prefix: str) -> tuple[list[str], list[str], list[str], float, float, bool]:
                supporting = _id_list(payload[f"{prefix}_supporting_sample_ids"], f"{prefix}_supporting_sample_ids")
                outliers = _id_list(payload[f"{prefix}_outlier_sample_ids"], f"{prefix}_outlier_sample_ids")
                representatives = _id_list(
                    payload[f"{prefix}_representative_evidence_ids"],
                    f"{prefix}_representative_evidence_ids",
                )
                support_set, outlier_set = set(supporting), set(outliers)
                if len(supporting) != len(support_set) or len(outliers) != len(outlier_set):
                    reasons.append(f"{prefix}_duplicate_ids_in_partition")
                if support_set & outlier_set:
                    reasons.append(f"{prefix}_support_outlier_overlap")
                if support_set | outlier_set != all_ids:
                    reasons.append(f"{prefix}_support_outlier_not_full_partition")
                representatives_valid = (
                    (not support_set and not representatives)
                    or (2 <= len(representatives) <= 3 and set(representatives).issubset(support_set))
                )
                if not representatives_valid:
                    reasons.append(f"invalid_{prefix}_representative_evidence_ids")
                confidence = float(payload[f"{prefix}_confidence"])
                if not np.isfinite(confidence) or not 0 <= confidence <= 1:
                    reasons.append(f"invalid_{prefix}_confidence")
                support_norms = {
                    normalise_text(visible[sample_id]["text"])
                    for sample_id in support_set
                    if sample_id in visible
                }
                all_norms = {normalise_text(sample["text"]) for sample in visible.values()}
                raw_fraction = len(support_set) / max(len(all_ids), 1)
                unique_fraction = len(support_norms) / max(len(all_norms), 1)
                majority = bool(
                    raw_fraction >= float(minimum_raw_support_fraction)
                    and unique_fraction >= float(minimum_unique_support_fraction)
                )
                return supporting, outliers, representatives, raw_fraction, unique_fraction, majority

            surface = read_track("surface")
            semantic = read_track("semantic")
            surface_supporting, surface_outliers, surface_representatives, surface_raw, surface_unique, surface_majority = surface
            semantic_supporting, semantic_outliers, semantic_representatives, semantic_raw, semantic_unique, semantic_majority = semantic
            if surface_majority and semantic_majority:
                induction_status = "both_majority"
            elif surface_majority:
                induction_status = "surface_only"
            elif semantic_majority:
                induction_status = "semantic_only"
            else:
                induction_status = "neither_majority"
            candidate_explanation = str(payload["candidate_explanation"]).strip()
            if not candidate_explanation:
                reasons.append("candidate_explanation_missing")
            surface_pattern = str(payload["surface_pattern"]).strip()
            semantic_pattern = str(payload["semantic_pattern"]).strip()
            if not surface_pattern or not semantic_pattern:
                reasons.append("dual_track_pattern_missing")
            row = {
                "task_id": task_id,
                "packet_id": task["packet_id"],
                "latent_idx": int(payload["latent_idx"]),
                "repeat": 1,
                "selection_unit": str(task.get("selection_unit") or "raw_utterance_row"),
                "short_name": str(payload["short_name"]).strip(),
                "candidate_explanation": candidate_explanation,
                "surface_pattern": surface_pattern,
                "surface_supporting_sample_ids": surface_supporting,
                "surface_outlier_sample_ids": surface_outliers,
                "surface_representative_evidence_ids": surface_representatives,
                "surface_confidence": float(payload["surface_confidence"]),
                "surface_support_count": len(surface_supporting),
                "surface_raw_support_fraction": surface_raw,
                "surface_unique_support_fraction": surface_unique,
                "surface_majority_gate_passed": surface_majority,
                "semantic_pattern": semantic_pattern,
                "semantic_supporting_sample_ids": semantic_supporting,
                "semantic_outlier_sample_ids": semantic_outliers,
                "semantic_representative_evidence_ids": semantic_representatives,
                "semantic_confidence": float(payload["semantic_confidence"]),
                "semantic_support_count": len(semantic_supporting),
                "semantic_raw_support_fraction": semantic_raw,
                "semantic_unique_support_fraction": semantic_unique,
                "semantic_majority_gate_passed": semantic_majority,
                "alternative_hypotheses": _text_list(
                    payload["alternative_hypotheses"], "alternative_hypotheses"
                ),
                "failure_modes": _text_list(payload["failure_modes"], "failure_modes"),
                "induction_status": induction_status,
                "raw_output_path": str(raw_path),
                "validation_status": "valid",
            }
        except Exception as exc:  # one malformed task must not stop 224 valid tasks
            reasons.append(f"{type(exc).__name__}: {exc}")

        quality_pass = bool(row is not None and not reasons)
        if quality_pass and row is not None:
            valid.append(row)
        else:
            retry_tasks.append(task)
        audit_rows.append(
            {
                "task_id": task_id,
                "packet_id": task.get("packet_id"),
                "latent_idx": task.get("latent_idx"),
                "selection_unit": task.get("selection_unit", "raw_utterance_row"),
                "quality_pass": quality_pass,
                "quality_reasons": ";".join(reasons),
                "surface_raw_support_fraction": row.get("surface_raw_support_fraction") if row else None,
                "semantic_raw_support_fraction": row.get("semantic_raw_support_fraction") if row else None,
                "surface_confidence": row.get("surface_confidence") if row else None,
                "semantic_confidence": row.get("semantic_confidence") if row else None,
                "induction_status": row.get("induction_status") if row else None,
            }
        )

    explainer_dir = Path(output_dir) / "explainer_outputs"
    explainer_dir.mkdir(parents=True, exist_ok=True)
    validated_path = explainer_dir / "validated_explanations.jsonl"
    audit_path = explainer_dir / "deepseek_top50_quality_audit.csv"
    retry_path = explainer_dir / "retry_tasks.jsonl"
    write_jsonl(validated_path, valid)
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False, encoding="utf-8-sig")
    write_jsonl(retry_path, retry_tasks)
    manifest = {
        "analysis": "deepseek_v4_flash_top50_dual_track_induction",
        "step": "validate-top50-dual-track-induction",
        "inputs": {"tasks": str(tasks_path), "execution_manifest": str(execution_manifest_path)},
        "outputs": {
            "validated_explanations": str(validated_path),
            "quality_audit": str(audit_path),
            "retry_tasks": str(retry_path),
        },
        "n_tasks": len(tasks),
        "n_valid": len(valid),
        "n_failed": len(tasks) - len(valid),
        "selection_unit": str(tasks[0].get("selection_unit") or "raw_utterance_row") if tasks else "",
        "induction_status_counts": (
            pd.DataFrame(valid)["induction_status"].value_counts().sort_index().to_dict()
            if valid
            else {}
        ),
        "majority_gate": {
            "minimum_raw_support_fraction": minimum_raw_support_fraction,
            "minimum_unique_support_fraction": minimum_unique_support_fraction,
        },
        "stages_not_run": ["heldout_scorer", "label_baseline", "minimal_pairs", "sae_activation_test"],
    }
    write_json(explainer_dir / "validation_manifest.json", manifest)
    return manifest


__all__ = [
    "DEEPSEEK_CHAT_COMPLETIONS_URL",
    "DEEPSEEK_DUAL_TRACK_FIELDS",
    "DEEPSEEK_EXECUTION_MODE",
    "DeepSeekTop50Config",
    "build_deepseek_dual_track_prompt",
    "build_deepseek_top50_tasks",
    "make_deepseek_refined_retry_tasks",
    "run_deepseek_top50_tasks",
    "validate_deepseek_top50_outputs",
]
