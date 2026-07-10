"""Minimal-pair task generation and optional SAE activation testing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contrastive_evidence_pack import PRIMARY_LABELS, read_jsonl, write_json, write_jsonl
from .contrastive_llm_io import parse_llm_json_file


@dataclass(frozen=True)
class MinimalPairTaskConfig:
    pairs_per_latent: int = 5
    representatives_per_primary_label: int = 3
    primary_labels: tuple[str, ...] = PRIMARY_LABELS


def _prompt_for_minimal_pairs(explanation: dict[str, Any], pairs_per_latent: int) -> str:
    payload = {
        "short_name": explanation.get("short_name"),
        "main_hypothesis": explanation.get("main_hypothesis"),
        "positive_triggers": explanation.get("positive_triggers"),
        "explicit_exclusions": explanation.get("explicit_exclusions"),
        "possible_surface_confounds": explanation.get("possible_surface_confounds"),
        "feature_type": explanation.get("feature_type"),
        "failure_modes": explanation.get("failure_modes"),
    }
    return f"""Design minimal pairs to test this candidate SAE latent explanation.

Explanation:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Create {int(pairs_per_latent)} minimal pairs. Each pair must contain:
- positive_text: should trigger the latent.
- negative_text: should not trigger the latent.
- changed_factor: the single functional factor changed.
- held_constant: what topic, style, and context were held fixed.
- expected_direction: use exactly "positive_greater_than_negative".

Rules:
- Change only one functional factor per pair.
- Keep topic, approximate length, speaker style, and counseling setting as similar as possible.
- Avoid copying examples verbatim.
- Return only a JSON list of pair objects.
"""


def _select_representative_explanations(
    *,
    packs: list[dict[str, Any]],
    explanations: list[dict[str, Any]],
    scorer_metrics: pd.DataFrame | None,
    config: MinimalPairTaskConfig,
) -> list[dict[str, Any]]:
    pack_by_id = {pack["packet_id"]: pack for pack in packs}
    metric_by_explanation: dict[str, dict[str, Any]] = {}
    if scorer_metrics is not None and not scorer_metrics.empty and "explanation_task_id" in scorer_metrics.columns:
        metric_by_explanation = {
            str(row["explanation_task_id"]): row.to_dict()
            for _, row in scorer_metrics.iterrows()
            if pd.notna(row.get("explanation_task_id"))
        }
    status_order = {
        "accepted": 0,
        "ambiguous": 1,
        "no_latent_contribution": 2,
        "rejected": 3,
        "invalid": 4,
        "unscored": 5,
    }
    explanations_by_packet: dict[str, list[dict[str, Any]]] = {}
    for explanation in explanations:
        explanations_by_packet.setdefault(str(explanation["packet_id"]), []).append(explanation)

    pack_rows = [
        {
            "packet_id": str(pack["packet_id"]),
            "target_label": str(pack["target_label"]),
            "latent_idx": int(pack["latent_idx"]),
            "rank_within_label": int(pack["rank_within_label"]),
            "inclusion_frequency": float(pack.get("inclusion_frequency") or 0),
        }
        for pack in packs
        if pack.get("target_label") in set(config.primary_labels)
    ]
    pack_df = pd.DataFrame(pack_rows)
    if pack_df.empty:
        raise ValueError("No primary-label packs available for minimal-pair planning")
    selected_packs: list[dict[str, Any]] = []
    for label in config.primary_labels:
        group = pack_df[pack_df["target_label"] == label].sort_values(
            ["inclusion_frequency", "rank_within_label", "latent_idx"],
            ascending=[False, True, True],
        )
        if len(group) < int(config.representatives_per_primary_label):
            raise ValueError(
                f"Minimal-pair plan requires {config.representatives_per_primary_label} packs for {label}, got {len(group)}"
            )
        selected_packs.extend(group.head(int(config.representatives_per_primary_label)).to_dict(orient="records"))

    selected: list[dict[str, Any]] = []
    for pack in selected_packs:
        packet_id = str(pack["packet_id"])
        candidates = explanations_by_packet.get(packet_id, [])
        if not candidates:
            raise ValueError(f"Minimal-pair representative {packet_id} has no validated explanation")
        ranked = sorted(
            candidates,
            key=lambda explanation: (
                status_order.get(
                    str(metric_by_explanation.get(str(explanation.get("task_id")), {}).get("status", "unscored")),
                    99,
                ),
                -float(metric_by_explanation.get(str(explanation.get("task_id")), {}).get("auroc") or -1),
                -float(explanation.get("confidence", 0)),
                str(explanation.get("task_id", "")),
            ),
        )
        explanation = ranked[0]
        metric = metric_by_explanation.get(str(explanation.get("task_id")), {})
        selected.append(
            {
                **pack,
                "explanation": explanation,
                "scorer_status": str(metric.get("status", "unscored")),
                "scorer_auroc": metric.get("auroc"),
            }
        )

    expected = len(config.primary_labels) * int(config.representatives_per_primary_label)
    if len(selected) != expected:
        raise AssertionError(f"Expected {expected} minimal-pair representatives, got {len(selected)}")
    return selected


def make_minimal_pair_tasks(
    *,
    packs_path: str | Path,
    explanations_path: str | Path,
    scorer_metrics_path: str | Path | None,
    output_dir: str | Path,
    config: MinimalPairTaskConfig = MinimalPairTaskConfig(),
) -> dict[str, Any]:
    output_path = Path(output_dir)
    task_dir = output_path / "llm_tasks"
    raw_dir = output_path / "minimal_pairs" / "raw_designer_outputs"
    task_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    packs = read_jsonl(packs_path)
    explanations = read_jsonl(explanations_path) if Path(explanations_path).exists() else []
    scorer_metrics = (
        pd.read_csv(scorer_metrics_path)
        if scorer_metrics_path and Path(scorer_metrics_path).exists()
        else None
    )
    selected = _select_representative_explanations(
        packs=packs,
        explanations=explanations,
        scorer_metrics=scorer_metrics,
        config=config,
    )
    tasks: list[dict[str, Any]] = []
    for row in selected:
        task_id = f"{row['packet_id']}_minimal_pairs"
        tasks.append(
            {
                "task_id": task_id,
                "task_type": "minimal_pair_designer",
                "packet_id": row["packet_id"],
                "target_label": row["target_label"],
                "latent_idx": int(row["latent_idx"]),
                "rank_within_label": int(row["rank_within_label"]),
                "scorer_status": row["scorer_status"],
                "scorer_auroc": row["scorer_auroc"],
                "prompt": _prompt_for_minimal_pairs(row["explanation"], config.pairs_per_latent),
                "expected_output_path": str(raw_dir / f"{task_id}.json"),
                "output_format": "json_list",
                "status": "pending_claude_code_llm",
            }
        )
    tasks_path = task_dir / "minimal_pair_designer_tasks.jsonl"
    write_jsonl(tasks_path, tasks)
    manifest = {
        "step": "make-minimal-pair-tasks",
        "inputs": {
            "packs": str(packs_path),
            "validated_explanations": str(explanations_path),
            "scorer_metrics": str(scorer_metrics_path or ""),
        },
        "outputs": {
            "minimal_pair_designer_tasks": str(tasks_path),
            "raw_designer_output_dir": str(raw_dir),
        },
        "parameters": asdict(config),
        "expected_tasks": int(len(config.primary_labels) * config.representatives_per_primary_label),
        "n_tasks": int(len(tasks)),
        "plan_complete": bool(len(tasks) == len(config.primary_labels) * config.representatives_per_primary_label),
    }
    write_json(output_path / "minimal_pair_task_manifest.json", manifest)
    return manifest


def _normalize_pairs(payload: Any) -> list[dict[str, str]]:
    if isinstance(payload, dict):
        for key in ("pairs", "minimal_pairs", "items"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError("Minimal-pair output must be a JSON list or wrapped list")
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each minimal pair must be an object")
        positive = str(item.get("positive_text", "")).strip()
        negative = str(item.get("negative_text", "")).strip()
        changed = str(item.get("changed_factor", "")).strip()
        held_constant = str(item.get("held_constant", "")).strip()
        direction = str(item.get("expected_direction", "positive_greater_than_negative")).strip()
        if not positive or not negative:
            raise ValueError("Minimal pair missing positive_text or negative_text")
        if direction != "positive_greater_than_negative":
            raise ValueError(f"Unsupported expected_direction: {direction}")
        key = (positive.lower(), negative.lower())
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "positive_text": positive,
                "negative_text": negative,
                "changed_factor": changed,
                "held_constant": held_constant,
                "expected_direction": direction,
            }
        )
    return rows


def validate_minimal_pair_outputs(
    *,
    tasks_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    pair_dir = output_path / "minimal_pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    tasks = read_jsonl(tasks_path) if Path(tasks_path).exists() else []
    pairs: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    retry: list[dict[str, Any]] = []
    for task in tasks:
        raw_path = Path(task.get("expected_output_path", ""))
        if not raw_path.exists():
            errors.append({"task_id": task["task_id"], "error": "raw_output_missing", "raw_output_path": str(raw_path)})
            retry.append(task)
            continue
        try:
            normalized = _normalize_pairs(parse_llm_json_file(raw_path))
            for idx, pair in enumerate(normalized, start=1):
                pairs.append(
                    {
                        "task_id": task["task_id"],
                        "packet_id": task["packet_id"],
                        "target_label": task["target_label"],
                        "latent_idx": int(task["latent_idx"]),
                        "pair_id": f"{task['task_id']}_p{idx:02d}",
                        **pair,
                    }
                )
        except Exception as exc:
            errors.append({"task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}", "raw_output_path": str(raw_path)})
            retry.append(task)
    pairs_path = pair_dir / "validated_minimal_pairs.jsonl"
    errors_path = pair_dir / "minimal_pair_validation_errors.csv"
    retry_path = pair_dir / "retry_minimal_pair_tasks.jsonl"
    write_jsonl(pairs_path, pairs)
    pd.DataFrame(errors).to_csv(errors_path, index=False)
    write_jsonl(retry_path, retry)
    manifest = {
        "step": "validate-minimal-pairs",
        "inputs": {"tasks": str(tasks_path)},
        "outputs": {
            "validated_minimal_pairs": str(pairs_path),
            "validation_errors": str(errors_path),
            "retry_tasks": str(retry_path),
        },
        "n_tasks": int(len(tasks)),
        "n_pairs": int(len(pairs)),
        "n_errors": int(len(errors)),
    }
    write_json(pair_dir / "minimal_pair_validation_manifest.json", manifest)
    return manifest


def run_minimal_pair_activation_test(
    *,
    pairs_path: str | Path,
    output_dir: str | Path,
    sae_config_path: str | Path,
    model_config_path: str | Path | None = None,
    model_dir: str | None = None,
    device: str | None = None,
    batch_size: int = 4,
    max_seq_len: int = 128,
    aggregation: str = "max",
    checkpoint_topk_semantics: str = "hard",
) -> dict[str, Any]:
    """Run model+SAE forward passes for generated minimal pairs.

    This is the only Step 4 function that requires GPU/local model weights.
    """
    import torch

    from .activations import extract_and_process_streaming
    from .model import load_local_model_and_tokenizer
    from .sae import load_sae_from_hub

    output_path = Path(output_dir)
    pair_dir = output_path / "minimal_pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    pairs = read_jsonl(pairs_path) if Path(pairs_path).exists() else []
    if not pairs:
        raise ValueError(f"No validated minimal pairs found at {pairs_path}")

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sae_cfg = json.loads(Path(sae_config_path).read_text(encoding="utf-8"))
    model, tokenizer, _ = load_local_model_and_tokenizer(
        str(model_config_path) if model_config_path else None,
        model_dir=model_dir,
        device=resolved_device,
    )
    sae = load_sae_from_hub(
        repo_id=sae_cfg["sae_repo_id"],
        subfolder=sae_cfg["sae_subfolder"],
        device=resolved_device,
        dtype=torch.bfloat16,
        checkpoint_topk_semantics=checkpoint_topk_semantics,
    )
    hook_point = sae_cfg.get("hook_point", "blocks.19.hook_resid_post")

    texts: list[str] = []
    for pair in pairs:
        texts.append(str(pair["positive_text"]))
        texts.append(str(pair["negative_text"]))
    result = extract_and_process_streaming(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        texts=texts,
        hook_point=hook_point,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        aggregation=aggregation,
        device=resolved_device,
        collect_structural_samples=0,
    )
    features = result["utterance_features"].detach().cpu().float().numpy()

    rows: list[dict[str, Any]] = []
    for idx, pair in enumerate(pairs):
        latent_idx = int(pair["latent_idx"])
        pos_act = float(features[2 * idx, latent_idx])
        neg_act = float(features[2 * idx + 1, latent_idx])
        rows.append(
            {
                **pair,
                "positive_activation": pos_act,
                "negative_activation": neg_act,
                "activation_gap": pos_act - neg_act,
                "pair_passed": bool(pos_act > neg_act),
            }
        )
    results = pd.DataFrame(rows)
    results_path = pair_dir / "minimal_pair_results.csv"
    summary_path = pair_dir / "minimal_pair_summary.csv"
    results.to_csv(results_path, index=False)
    summary = (
        results.groupby(["target_label", "latent_idx"], as_index=False)
        .agg(
            n_pairs=("pair_id", "count"),
            mean_gap=("activation_gap", "mean"),
            pass_rate=("pair_passed", "mean"),
        )
        .sort_values(["target_label", "latent_idx"])
    )
    summary.to_csv(summary_path, index=False)
    manifest = {
        "step": "run-minimal-pairs",
        "inputs": {"validated_minimal_pairs": str(pairs_path), "sae_config": str(sae_config_path)},
        "outputs": {"minimal_pair_results": str(results_path), "minimal_pair_summary": str(summary_path)},
        "n_pairs": int(len(results)),
        "device": str(resolved_device),
        "hook_point": hook_point,
        "aggregation": aggregation,
    }
    write_json(pair_dir / "minimal_pair_activation_manifest.json", manifest)
    return manifest


__all__ = [
    "MinimalPairTaskConfig",
    "make_minimal_pair_tasks",
    "run_minimal_pair_activation_test",
    "validate_minimal_pair_outputs",
]
