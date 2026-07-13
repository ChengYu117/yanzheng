"""Generate label-blind latent interpretation cards with DeepSeek.

The model sees only an opaque feature id and a deduplicated sentence set. MISC
labels, activation magnitudes, ranks, and contrast groups remain outside the
prompt.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contrastive_evidence_pack import normalise_text, read_jsonl, write_json, write_jsonl
from .contrastive_llm_io import parse_llm_json_file
from .deepseek_top50_induction import (
    DEEPSEEK_EXECUTION_MODE,
    DeepSeekTop50Config,
    _load_feature_tensor,
    _load_texts,
    _sha256_bytes,
    _sha256_text,
    _stable_unique_latents,
    _top50_samples,
)


LATENT_CARD_SYSTEM_PROMPT = (
    "You are a text-pattern induction and linguistic analysis evaluator. Analyze a cluster of "
    "sentences, infer its most stable shared pattern, and distinguish behavioral function, "
    "linguistic structure, affective content, topic, and surface artifacts. Return only the "
    "requested JSON."
)

LATENT_CARD_FIELDS: tuple[str, ...] = (
    "latent_idx",
    "short_name",
    "primary_explanation",
    "candidate_behavioral_explanation",
    "explanation_type",
    "supporting_sample_ids",
    "outlier_sample_ids",
    "representative_evidence_ids",
    "linguistic_evidence",
    "alternative_explanations",
    "possible_confounds",
    "limitations",
    "confidence",
    "confidence_rationale",
)

EXPLANATION_TYPES = {
    "behavioral_function",
    "linguistic_structure",
    "affective_content",
    "topic",
    "surface_artifact",
    "unclear_or_mixed",
}
EVIDENCE_LEVELS = {
    "lexical",
    "syntactic",
    "discourse_marker",
    "semantic",
    "pragmatic_function",
    "affective",
    "topical",
    "surface_form",
}
EVIDENCE_ROLES = {"support", "limit", "contradict"}


def _format_samples(samples: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- sample_id={sample['id']}\n  text: {str(sample['text']).replace(chr(10), ' ').strip()}"
        for sample in samples
    )


def build_latent_card_prompt(latent_idx: int, samples: list[dict[str, Any]]) -> str:
    ids = [str(sample["id"]) for sample in samples]
    if len(samples) != 50 or len(ids) != len(set(ids)):
        raise ValueError("Latent-card induction requires exactly 50 uniquely identified sentences")
    if len({normalise_text(sample["text"]) for sample in samples}) != 50:
        raise ValueError("Latent-card sentences must be unique after text normalization")
    schema = {
        "latent_idx": int(latent_idx),
        "short_name": "",
        "primary_explanation": "",
        "candidate_behavioral_explanation": "",
        "explanation_type": "unclear_or_mixed",
        "supporting_sample_ids": [],
        "outlier_sample_ids": [],
        "representative_evidence_ids": [],
        "linguistic_evidence": [
            {
                "sample_id": "",
                "evidence": "",
                "evidence_level": "semantic",
                "evidence_role": "support",
            }
        ],
        "alternative_explanations": [],
        "possible_confounds": [],
        "limitations": [],
        "confidence": 1,
        "confidence_rationale": "",
    }
    return f"""Analyze the following sentences as one set and produce a candidate interpretation of this anonymous text feature.

Feature ID:
{int(latent_idx)}

Sentences:

{_format_samples(samples)}

Complete the following tasks:

1. Infer one primary candidate interpretation describing a stable pattern shared by a majority of the sentences.
2. Separately describe any candidate behavioral or discourse function.
3. If the evidence for a behavioral function is insufficient, state this explicitly instead of forcing one.
4. Cite specific sample IDs as linguistic evidence.
5. Identify sentences not adequately covered by the primary interpretation.
6. Provide 1 to 3 reasonable alternative interpretations.
7. Describe possible confounding factors.
8. Assign a confidence score from 1 to 5.

Analyze the sentence set as a whole. Do not produce a separate interpretation for every sentence. If no pattern covers a majority of the sentences, use `unclear_or_mixed`.

`explanation_type` must be exactly one of:

- `behavioral_function`: a request, question, reflection, affirmation, advice, evaluation, or another communicative or discourse function.
- `linguistic_structure`: recurring lexical combinations, syntax, sentence forms, discourse markers, or expression templates.
- `affective_content`: emotion, attitude, evaluative direction, or affective intensity.
- `topic`: a recurring event, object, activity, experience, or domain.
- `surface_artifact`: incidental cues, data formats, or annotation biases unrelated to a genuine behavioral mechanism, including punctuation, text length, transcription conventions, repeated templates, fixed wording, or processing artifacts.
- `unclear_or_mixed`: multiple patterns cannot be separated reliably, or no stable pattern covers a majority of the sentences.

Each `linguistic_evidence` item must contain `sample_id`, `evidence`, `evidence_level`, and `evidence_role`.

`evidence_level` must be exactly one of: `lexical`, `syntactic`, `discourse_marker`, `semantic`, `pragmatic_function`, `affective`, `topical`, `surface_form`.

`evidence_role` must be exactly one of: `support`, `limit`, `contradict`.

`representative_evidence_ids` should contain 2 to 3 sample IDs that best represent the primary interpretation.

Confidence calibration:

- `1`: No consistent pattern can be identified.
- `2`: Only a weak pattern exists, or several interpretations are equally plausible.
- `3`: The primary interpretation covers a majority of the sentences, but clear exceptions or confounds remain.
- `4`: The primary pattern is clear and consistent, while alternative interpretations are weaker.
- `5`: Coverage is very high, the pattern remains stable after sentence deduplication, and alternative interpretations and confounds are weak.

Return only one JSON object matching this structure:

{json.dumps(schema, ensure_ascii=False, indent=2)}

Field requirements:

- Use a short, specific, searchable `short_name`.
- `primary_explanation` must describe the most stable shared pattern and its boundary.
- If no stable behavioral function is supported, set `candidate_behavioral_explanation` to: "The available sentences do not support a stable behavioral-function interpretation."
- Every sample ID must appear exactly once in either `supporting_sample_ids` or `outlier_sample_ids`.
- `representative_evidence_ids` must be selected from `supporting_sample_ids`.
- Keep each prose field within two sentences.
"""


def build_latent_card_tasks(
    *,
    stable_latents_path: str | Path,
    feature_store_path: str | Path,
    records_path: str | Path,
    output_dir: str | Path,
    config: DeepSeekTop50Config = DeepSeekTop50Config(),
) -> dict[str, Any]:
    if int(config.top_n) != 50:
        raise ValueError("The latent-card protocol is fixed to 50 deduplicated sentences")
    output = Path(output_dir)
    unique_latents = _stable_unique_latents(stable_latents_path)
    tensor = _load_feature_tensor(feature_store_path)
    texts = _load_texts(records_path)
    normalized = [normalise_text(text) for text in texts]
    if tensor.shape[0] != len(texts):
        raise ValueError(f"feature rows {tensor.shape[0]} do not match record rows {len(texts)}")
    if unique_latents.empty:
        raise ValueError("No stable_core latents found")
    indices = unique_latents["latent_idx"].to_numpy(dtype=np.int64)
    if int(indices.max()) >= int(tensor.shape[1]):
        raise ValueError("A stable latent index exceeds the feature-store width")

    task_dir = output / "llm_tasks"
    raw_dir = output / "card_outputs" / "raw"
    evidence_dir = output / "evidence_packs"
    task_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    selected_values = tensor[:, indices.tolist()].float().numpy()
    tasks: list[dict[str, Any]] = []
    packs: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    for column, metadata in enumerate(unique_latents.to_dict(orient="records")):
        latent_idx = int(metadata["latent_idx"])
        _, internal = _top50_samples(
            values=np.asarray(selected_values[:, column], dtype=np.float32),
            texts=texts,
            normalised_texts=normalized,
            latent_idx=latent_idx,
            top_n=50,
        )
        visible = [{"id": row["id"], "text": row["text"]} for row in internal]
        packet_id = f"card_{latent_idx:05d}"
        task_id = f"{packet_id}_induction"
        tasks.append(
            {
                "task_id": task_id,
                "task_type": "anonymous_sentence_cluster_latent_card",
                "packet_id": packet_id,
                "latent_idx": latent_idx,
                "prompt": build_latent_card_prompt(latent_idx, visible),
                "visible_samples": visible,
                "expected_output_path": str(raw_dir / f"{task_id}.json"),
                "output_format": "single_json_object",
                "status": "pending_deepseek_api",
            }
        )
        packs.append(
            {
                "packet_id": packet_id,
                "latent_idx": latent_idx,
                "samples_for_model": visible,
                "selection_provenance_internal": internal,
            }
        )
        index_rows.append({"task_id": task_id, "packet_id": packet_id, "latent_idx": latent_idx})

    tasks_path = task_dir / "latent_card_tasks.jsonl"
    packs_path = evidence_dir / "latent_card_sentence_packs.jsonl"
    index_path = evidence_dir / "latent_card_task_index.csv"
    write_jsonl(tasks_path, tasks)
    write_jsonl(packs_path, packs)
    pd.DataFrame(index_rows).to_csv(index_path, index=False, encoding="utf-8-sig")
    manifest = {
        "analysis": "deepseek_v4_flash_latent_cards",
        "step": "build-latent-card-tasks",
        "inputs": {
            "stable_latents": str(stable_latents_path),
            "feature_store": str(feature_store_path),
            "records": str(records_path),
        },
        "outputs": {"tasks": str(tasks_path), "packs": str(packs_path), "task_index": str(index_path)},
        "parameters": asdict(config),
        "n_unique_latents": len(tasks),
        "n_tasks": len(tasks),
        "model_visible_fields": ["latent_idx", "sample_id", "text"],
        "model_hidden_fields": ["MISC labels", "activation values", "activation ranks", "comparison groups"],
    }
    write_json(output / "task_build_manifest.json", manifest)
    return manifest


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    result = [str(item).strip() for item in value]
    if any(not item for item in result):
        raise ValueError(f"{field} contains an empty item")
    return result


def validate_latent_card_outputs(
    *, tasks_path: str | Path, execution_manifest_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    tasks = read_jsonl(tasks_path)
    manifest_rows = read_jsonl(execution_manifest_path) if Path(execution_manifest_path).exists() else []
    success_by_id = {
        str(row["task_id"]): row
        for row in manifest_rows
        if row.get("status") == "success" and row.get("task_id")
    }
    valid: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    retry: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        reasons: list[str] = []
        raw_path = Path(task["expected_output_path"])
        row: dict[str, Any] | None = None
        try:
            execution = success_by_id.get(task_id)
            if execution is None:
                reasons.append("missing_successful_execution")
            if not raw_path.exists():
                raise FileNotFoundError("raw_output_missing")
            if execution is not None:
                if execution.get("prompt_sha256") != _sha256_text(str(task["prompt"])):
                    reasons.append("prompt_hash_mismatch")
                if execution.get("raw_output_sha256") != _sha256_bytes(raw_path.read_bytes()):
                    reasons.append("raw_output_hash_mismatch")
                if execution.get("execution_mode") != DEEPSEEK_EXECUTION_MODE:
                    reasons.append("execution_mode_mismatch")
            payload = parse_llm_json_file(raw_path)
            missing = [field for field in LATENT_CARD_FIELDS if field not in payload]
            if missing:
                raise ValueError(f"missing_fields={','.join(missing)}")
            if int(payload["latent_idx"]) != int(task["latent_idx"]):
                reasons.append("latent_idx_mismatch")
            explanation_type = str(payload["explanation_type"]).strip()
            if explanation_type not in EXPLANATION_TYPES:
                reasons.append("invalid_explanation_type")
            all_ids = {str(sample["id"]) for sample in task["visible_samples"]}
            supporting = _string_list(payload["supporting_sample_ids"], "supporting_sample_ids")
            outliers = _string_list(payload["outlier_sample_ids"], "outlier_sample_ids")
            support_set, outlier_set = set(supporting), set(outliers)
            if len(supporting) != len(support_set) or len(outliers) != len(outlier_set):
                reasons.append("duplicate_ids_in_partition")
            if support_set & outlier_set or support_set | outlier_set != all_ids:
                reasons.append("invalid_full_partition")
            if explanation_type != "unclear_or_mixed" and len(support_set) < (len(all_ids) + 1) // 2:
                reasons.append("nonmajority_primary_explanation")
            representatives = _string_list(payload["representative_evidence_ids"], "representative_evidence_ids")
            if not (2 <= len(representatives) <= 3 and set(representatives).issubset(support_set)):
                reasons.append("invalid_representative_evidence_ids")
            evidence = payload["linguistic_evidence"]
            if not isinstance(evidence, list) or not evidence:
                reasons.append("linguistic_evidence_missing")
                evidence = []
            for item in evidence:
                if not isinstance(item, dict):
                    reasons.append("invalid_linguistic_evidence_item")
                    continue
                if str(item.get("sample_id", "")) not in all_ids:
                    reasons.append("linguistic_evidence_unknown_sample_id")
                if str(item.get("evidence_level", "")) not in EVIDENCE_LEVELS:
                    reasons.append("invalid_evidence_level")
                if str(item.get("evidence_role", "")) not in EVIDENCE_ROLES:
                    reasons.append("invalid_evidence_role")
                if not str(item.get("evidence", "")).strip():
                    reasons.append("empty_linguistic_evidence")
            alternatives = _string_list(payload["alternative_explanations"], "alternative_explanations")
            confounds = _string_list(payload["possible_confounds"], "possible_confounds")
            limitations = _string_list(payload["limitations"], "limitations")
            if not 1 <= len(alternatives) <= 3:
                reasons.append("invalid_alternative_count")
            if len(confounds) > 4:
                reasons.append("too_many_confounds")
            confidence_raw = payload["confidence"]
            confidence = int(confidence_raw)
            if isinstance(confidence_raw, bool) or float(confidence_raw) != confidence or not 1 <= confidence <= 5:
                reasons.append("invalid_confidence")
            for field in ("short_name", "primary_explanation", "candidate_behavioral_explanation", "confidence_rationale"):
                if not str(payload[field]).strip():
                    reasons.append(f"empty_{field}")
            row = {
                "task_id": task_id,
                "packet_id": task["packet_id"],
                "latent_idx": int(payload["latent_idx"]),
                "short_name": str(payload["short_name"]).strip(),
                "primary_explanation": str(payload["primary_explanation"]).strip(),
                "candidate_behavioral_explanation": str(payload["candidate_behavioral_explanation"]).strip(),
                "explanation_type": explanation_type,
                "supporting_sample_ids": supporting,
                "outlier_sample_ids": outliers,
                "representative_evidence_ids": representatives,
                "linguistic_evidence": evidence,
                "alternative_explanations": alternatives,
                "possible_confounds": confounds,
                "limitations": limitations,
                "confidence": confidence,
                "confidence_rationale": str(payload["confidence_rationale"]).strip(),
                "support_count": len(supporting),
                "support_fraction": len(supporting) / max(len(all_ids), 1),
                "raw_output_path": str(raw_path),
            }
        except Exception as exc:
            reasons.append(f"{type(exc).__name__}: {exc}")
        passed = row is not None and not reasons
        if passed:
            valid.append(row)
        else:
            retry.append(task)
        audit.append(
            {
                "task_id": task_id,
                "latent_idx": task.get("latent_idx"),
                "quality_pass": passed,
                "quality_reasons": ";".join(dict.fromkeys(reasons)),
                "explanation_type": row.get("explanation_type") if row else None,
                "confidence": row.get("confidence") if row else None,
                "support_fraction": row.get("support_fraction") if row else None,
            }
        )

    card_dir = Path(output_dir) / "card_outputs"
    card_dir.mkdir(parents=True, exist_ok=True)
    cards_path = card_dir / "validated_cards.jsonl"
    audit_path = card_dir / "card_quality_audit.csv"
    retry_path = card_dir / "retry_tasks.jsonl"
    write_jsonl(cards_path, valid)
    pd.DataFrame(audit).to_csv(audit_path, index=False, encoding="utf-8-sig")
    write_jsonl(retry_path, retry)
    manifest = {
        "analysis": "deepseek_v4_flash_latent_cards",
        "step": "validate-latent-cards",
        "outputs": {"validated_cards": str(cards_path), "quality_audit": str(audit_path), "retry_tasks": str(retry_path)},
        "n_tasks": len(tasks),
        "n_valid": len(valid),
        "n_failed": len(tasks) - len(valid),
        "explanation_type_counts": (
            pd.DataFrame(valid)["explanation_type"].value_counts().sort_index().to_dict() if valid else {}
        ),
    }
    write_json(card_dir / "validation_manifest.json", manifest)
    return manifest


def build_refined_latent_card_retry_tasks(
    *, all_tasks_path: str | Path, retry_tasks_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    all_tasks = read_jsonl(all_tasks_path)
    retry_ids = {str(task["task_id"]) for task in read_jsonl(retry_tasks_path)}
    refined: list[dict[str, Any]] = []
    updated_all: list[dict[str, Any]] = []
    suffix = """

Retry consistency check:
- Count the IDs before returning JSON. `supporting_sample_ids` and `outlier_sample_ids` must be disjoint and together contain all 50 IDs.
- `representative_evidence_ids` must contain 2 to 3 IDs, and every one must also occur in `supporting_sample_ids`.
- A non-`unclear_or_mixed` primary explanation must have at least 25 supporting IDs. If fewer than 25 sentences support one pattern, use `unclear_or_mixed`.
- Ensure the explanation, partition, representative evidence, and confidence rationale describe the same pattern.
"""
    for task in all_tasks:
        updated = dict(task)
        if str(task["task_id"]) in retry_ids:
            prompt = str(task["prompt"]).rstrip()
            updated["prompt"] = prompt if "Retry consistency check:" in prompt else prompt + suffix
            updated["status"] = "refined_retry_deepseek_api"
            refined.append(updated)
        updated_all.append(updated)
    output = Path(output_dir)
    refined_path = output / "llm_tasks" / "refined_latent_card_retry_tasks.jsonl"
    write_jsonl(refined_path, refined)
    write_jsonl(all_tasks_path, updated_all)
    manifest = {
        "analysis": "deepseek_v4_flash_latent_cards",
        "step": "build-refined-latent-card-retry-tasks",
        "outputs": {"refined_retry_tasks": str(refined_path), "updated_all_tasks": str(all_tasks_path)},
        "n_tasks": len(refined),
    }
    write_json(output / "refined_retry_task_manifest.json", manifest)
    return manifest


__all__ = [
    "LATENT_CARD_FIELDS",
    "LATENT_CARD_SYSTEM_PROMPT",
    "build_latent_card_prompt",
    "build_latent_card_tasks",
    "build_refined_latent_card_retry_tasks",
    "validate_latent_card_outputs",
]
