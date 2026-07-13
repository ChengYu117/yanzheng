"""Claude Code LLM task generation for label-blind latent explanations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contrastive_evidence_pack import (
    MISC_LABEL_PATTERN,
    assert_label_blind_samples,
    jsonable,
    read_jsonl,
    stable_seed,
    write_json,
    write_jsonl,
)


EXPLAINER_SCHEMA_FIELDS: tuple[str, ...] = (
    "latent_idx",
    "short_name",
    "main_hypothesis",
    "positive_triggers",
    "explicit_exclusions",
    "possible_surface_confounds",
    "feature_type",
    "confidence",
    "alternative_hypotheses",
    "key_evidence",
    "failure_modes",
)


@dataclass(frozen=True)
class ExplainerTaskConfig:
    repeats_per_latent: int = 2
    random_state: int = 42
    max_samples_per_task: int = 20


def _format_samples(samples: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for sample in samples:
        lines.append(
            "- id={id} tag={tag} activation={activation:.6g}\n  text: {text}".format(
                id=sample["id"],
                tag=sample["tag"],
                activation=float(sample["activation"]),
                text=str(sample["text"]).replace("\n", " ").strip(),
            )
        )
    return "\n".join(lines)


def build_explainer_prompt(samples: list[dict[str, Any]], *, latent_idx: int | None = None) -> str:
    assert_label_blind_samples(samples)
    formatted = _format_samples(samples)
    latent_line = "" if latent_idx is None else f"Latent metadata: latent_idx={int(latent_idx)}\n\n"
    prompt = f"""You are explaining an anonymous internal SAE latent from a language model.

{latent_line}\
Task:
- Do not summarize a broad topic. Propose the narrowest falsifiable trigger hypothesis for this latent.
- Focus on why NONACTIVE_NEAR_MISS samples are not triggered even when they look similar.
- Explicitly state what should not trigger the latent.
- Treat the examples as evidence for a candidate explanation only, not as proof of a mechanism.

Visible sample tags:
- ACTIVE_HIGH: high activation; the latent clearly triggered.
- ACTIVE_MID: medium activation; the latent triggered but less strongly.
- ACTIVE_LOW: low nonzero boundary case.
- NONACTIVE_NEAR_MISS: similar-looking text that should not trigger.
- NONACTIVE_RANDOM: random non-triggering text.

Samples:
{formatted}

Return only one JSON object. Do not wrap it in Markdown. Use exactly these fields:
{json.dumps(list(EXPLAINER_SCHEMA_FIELDS), ensure_ascii=False)}

Field requirements:
- latent_idx must be the integer latent id shown in the task metadata.
- short_name must be concise and must not contain dataset label codes.
- positive_triggers, explicit_exclusions, possible_surface_confounds, alternative_hypotheses, key_evidence, and failure_modes must be JSON arrays.
- confidence must be a continuous number between 0 and 1.
- key_evidence should list sample ids from the prompt.
"""
    if MISC_LABEL_PATTERN.search(prompt):
        raise AssertionError("Explainer prompt leaked a MISC label token.")
    return prompt


def make_explainer_tasks(
    *,
    packs_path: str | Path,
    output_dir: str | Path,
    config: ExplainerTaskConfig = ExplainerTaskConfig(),
) -> dict[str, Any]:
    output_path = Path(output_dir)
    task_dir = output_path / "llm_tasks"
    raw_dir = output_path / "explainer_outputs" / "raw"
    task_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_packs = read_jsonl(packs_path)
    packs = [
        pack
        for pack in all_packs
        if pack.get("summary", {}).get("interpretability_eligible", True)
    ]
    tasks: list[dict[str, Any]] = []
    internal_rows: list[dict[str, Any]] = []
    for pack in packs:
        samples = list(pack["samples_for_explainer"])
        assert_label_blind_samples(samples)
        for repeat in range(1, int(config.repeats_per_latent) + 1):
            rng = np.random.default_rng(
                stable_seed(config.random_state, pack["packet_id"], pack["latent_idx"], repeat, "explainer")
            )
            order = np.arange(len(samples))
            rng.shuffle(order)
            task_samples = [samples[int(idx)] for idx in order[: int(config.max_samples_per_task)]]
            task_id = f"{pack['packet_id']}_explainer_r{repeat:02d}"
            prompt = build_explainer_prompt(task_samples, latent_idx=int(pack["latent_idx"]))
            expected_path = raw_dir / f"{task_id}.json"
            task = {
                "task_id": task_id,
                "task_type": "contrastive_latent_explainer",
                "packet_id": pack["packet_id"],
                "latent_idx": int(pack["latent_idx"]),
                "repeat": int(repeat),
                "prompt": prompt,
                "expected_output_path": str(expected_path),
                "output_format": "single_json_object",
                "status": "pending_claude_code_llm",
            }
            tasks.append(task)
            internal_rows.append(
                {
                    "task_id": task_id,
                    "packet_id": pack["packet_id"],
                    "target_label": pack["target_label"],
                    "latent_idx": int(pack["latent_idx"]),
                    "rank_within_label": int(pack["rank_within_label"]),
                    "repeat": int(repeat),
                    "expected_output_path": str(expected_path),
                }
            )

    tasks_path = task_dir / "explainer_tasks.jsonl"
    internal_path = task_dir / "explainer_task_manifest_internal.jsonl"
    write_jsonl(tasks_path, tasks)
    write_jsonl(internal_path, internal_rows)
    readme = raw_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Claude Code LLM explainer raw outputs",
                "",
                "Save Claude Code LLM responses here as `{task_id}.json`.",
                "Keep malformed or fenced JSON unchanged; the validator preserves raw files and writes retry tasks.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "step": "make-explainer-tasks",
        "inputs": {"packs": str(packs_path)},
        "outputs": {
            "explainer_tasks": str(tasks_path),
            "explainer_task_manifest_internal": str(internal_path),
            "raw_output_dir": str(raw_dir),
        },
        "parameters": asdict(config),
        "n_packs_total": int(len(all_packs)),
        "n_packs": int(len(packs)),
        "n_packs_excluded": int(len(all_packs) - len(packs)),
        "n_tasks": int(len(tasks)),
        "label_blind": True,
        "llm_execution": "claude_code_model_file_queue",
    }
    write_json(output_path / "explainer_task_manifest.json", manifest)
    return manifest


__all__ = [
    "EXPLAINER_SCHEMA_FIELDS",
    "ExplainerTaskConfig",
    "build_explainer_prompt",
    "make_explainer_tasks",
]
