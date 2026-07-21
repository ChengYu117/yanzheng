"""Strong/weak contrastive explanation and held-out sentence prediction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from .codex_latent_cards import CodexLatentCardConfig, default_isolation_paths, run_codex_latent_card_tasks
from .contrastive_evidence_pack import normalise_text, read_jsonl, write_json, write_jsonl


ANALYSIS_NAME = "contrastive_latent_faithfulness_v2_gpt55_low"
SCORER_FROZEN_FIELDS = (
    "short_name",
    "surface_or_linguistic_hypothesis",
    "behavioral_or_discourse_hypothesis",
    "primary_explanation",
    "explanation_type",
)
PILOT_LATENTS = (
    20808, 11435, 31133, 20436, 9959, 664, 21935, 13430, 16345, 8294,
    11948, 30223, 23464, 7143, 10916, 19435, 2995, 32596, 736, 15160,
)


@dataclass(frozen=True)
class SamplingConfig:
    heldout_fraction: float = 0.30
    split_seed: str = "v2-heldout"
    presentation_seed: str = "v3-heldout-order"
    packet_version: str = "v3-randomized-heldout"
    n_group: int = 10
    n_heldout_stratum: int = 5


def _load_features(path: str | Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("utterance_features", "features", "X"):
            if key in payload:
                return payload[key].float()
    if isinstance(payload, torch.Tensor):
        return payload.float()
    raise KeyError(f"Cannot find utterance features in {path}")


def _is_heldout(source_file: str, config: SamplingConfig) -> bool:
    digest = hashlib.sha256(f"{config.split_seed}|{source_file}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF < config.heldout_fraction


def _stable_order(rows: list[int], key: str) -> list[int]:
    return sorted(rows, key=lambda row: hashlib.sha256(f"{key}|{row}".encode()).hexdigest())


def _unique_pick(rows: list[int], texts: list[str], n: int, *, key: str, used: set[str]) -> list[int]:
    selected = []
    for row in _stable_order(rows, key):
        norm = normalise_text(texts[row])
        if not norm or norm in used:
            continue
        selected.append(row)
        used.add(norm)
        if len(selected) == n:
            break
    return selected


def _rank_bands(values: np.ndarray, rows: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    ordered = rows[np.argsort(values[rows], kind="stable")]
    # Keep decile sampling when sufficiently populated, but guarantee a usable
    # candidate pool for five held-out or ten discovery examples. Never allow
    # the low/high candidate bands to overlap.
    width = min(max(int(np.ceil(len(ordered) * 0.10)), 20), len(ordered) // 2)
    if width < 5:
        raise ValueError(f"Only {len(ordered)} positive rows; cannot form disjoint rank bands")
    weak = ordered[:width].tolist()
    strong = ordered[-width:].tolist()
    mid_width = min(max(int(np.ceil(len(ordered) * 0.20)), 20), len(ordered))
    lo = max((len(ordered) - mid_width) // 2, 0)
    mid = ordered[lo : lo + mid_width].tolist()
    return weak, mid, strong


def _public_samples(rows: list[int], texts: list[str], prefix: str) -> list[dict[str, str]]:
    return [{"sample_id": f"{prefix}{i:03d}", "text": texts[row]} for i, row in enumerate(rows, 1)]


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def freeze_randomized_heldout_packet(
    selected_by_stratum: dict[str, list[int]],
    texts: list[str],
    feature_id: str,
    presentation_seed: str,
) -> dict[str, Any]:
    """Freeze a held-out packet and assign public IDs only after permutation."""
    source_order = [
        {"row_idx": int(row_idx), "stratum": str(stratum)}
        for stratum, row_indices in selected_by_stratum.items()
        for row_idx in row_indices
    ]
    row_ids = [item["row_idx"] for item in source_order]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError(f"{feature_id}: duplicate held-out row indices")
    normalized = [normalise_text(texts[row_idx]) for row_idx in row_ids]
    if any(not value for value in normalized) or len(normalized) != len(set(normalized)):
        raise ValueError(f"{feature_id}: empty or duplicate normalized held-out text")

    presentation_order = sorted(
        source_order,
        key=lambda item: hashlib.sha256(
            f"{presentation_seed}|{feature_id}|{item['row_idx']}|{item['stratum']}".encode("utf-8")
        ).hexdigest(),
    )
    if len(presentation_order) > 1 and presentation_order == source_order:
        presentation_order = presentation_order[1:] + presentation_order[:1]

    public_samples: list[dict[str, str]] = []
    private_truth: list[dict[str, Any]] = []
    for position, item in enumerate(presentation_order, 1):
        sample_id = f"H{position:03d}"
        text = str(texts[item["row_idx"]])
        text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        public_samples.append({"sample_id": sample_id, "text": text})
        private_truth.append({
            "public_sample_id": sample_id,
            "row_idx": item["row_idx"],
            "stratum": item["stratum"],
            "text_sha256": text_sha256,
        })
    return {
        "public_samples": public_samples,
        "private_truth": private_truth,
        "packet_sha256": _canonical_sha256(public_samples),
        "private_truth_sha256": _canonical_sha256(private_truth),
        "selected_rows_sha256": _canonical_sha256(sorted(source_order, key=lambda item: item["row_idx"])),
        "presentation_seed": presentation_seed,
    }


def _public_scorer_packets_path(output: Path) -> Path:
    return output / "public_packets" / "scorer_packets.jsonl"


def _private_truth_path(output: Path) -> Path:
    return output / "private" / "heldout_truth.jsonl"


def _validate_public_samples(samples: list[dict[str, Any]], feature_id: str) -> None:
    expected_fields = {"sample_id", "text"}
    if any(set(sample) != expected_fields for sample in samples):
        raise ValueError(f"{feature_id}: public scorer packet contains non-public fields")
    sample_ids = [str(sample["sample_id"]) for sample in samples]
    if len(sample_ids) != 20 or len(set(sample_ids)) != 20:
        raise ValueError(f"{feature_id}: public scorer packet must contain 20 unique IDs")


def align_scorer_predictions_by_id(
    *, public_samples: list[dict[str, Any]], private_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]], feature_id: str,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Join public samples, predictions, and private truth by opaque sample ID."""
    _validate_public_samples(public_samples, feature_id)
    packet_order = [str(sample["sample_id"]) for sample in public_samples]
    prediction_ids = [str(item.get("sample_id", "")) for item in predictions]
    truth_ids = [str(item.get("public_sample_id", "")) for item in private_truth]
    expected = set(packet_order)
    if len(prediction_ids) != len(set(prediction_ids)) or set(prediction_ids) != expected:
        raise ValueError("prediction_id_set_mismatch")
    if len(truth_ids) != len(set(truth_ids)) or set(truth_ids) != expected:
        raise ValueError("truth_id_set_mismatch")
    prediction_by_id = {str(item["sample_id"]): item for item in predictions}
    truth_by_id = {str(item["public_sample_id"]): item for item in private_truth}
    public_by_id = {str(item["sample_id"]): item for item in public_samples}
    aligned = [(prediction_by_id[sid], truth_by_id[sid], public_by_id[sid]) for sid in packet_order]
    for _, truth, public in aligned:
        actual_hash = hashlib.sha256(str(public["text"]).encode("utf-8")).hexdigest()
        if actual_hash != truth.get("text_sha256"):
            raise ValueError("public_private_text_hash_mismatch")
    return aligned


def audit_frozen_packet_integrity(
    *, output_dir: str | Path, expected_strata: tuple[str, ...] = ("high", "mid", "weak", "zero"),
) -> dict[str, Any]:
    """Audit frozen packet privacy, balance, permutation, hashes, and split disjointness."""
    output = Path(output_dir)
    masters = {row["feature_id"]: row for row in read_jsonl(output / "private" / "master_packets.jsonl")}
    truths = {row["feature_id"]: row["samples"] for row in read_jsonl(_private_truth_path(output))}
    scorers = {row["feature_id"]: row["samples"] for row in read_jsonl(_public_scorer_packets_path(output))}
    explainers = {row["feature_id"]: row for row in read_jsonl(output / "public_packets" / "explainer_packets.jsonl")}
    manifest = json.loads((output / "packet_manifest.json").read_text(encoding="utf-8"))
    manifest_entries = {row["feature_id"]: row for row in manifest["entries"]}
    failures: list[dict[str, Any]] = []
    eligible = 0
    randomized = 0
    block_order = [stratum for stratum in expected_strata for _ in range(5)]
    for feature_id, master in masters.items():
        public = scorers.get(feature_id, [])
        truth = truths.get(feature_id, [])
        explainer = explainers.get(feature_id, {})
        if set(explainer) != {"feature_id", "strong_samples", "weak_samples"}:
            failures.append({"feature_id": feature_id, "reason": "explainer_public_fields_invalid"})
        if any(set(item) != {"sample_id", "text"} for group in (explainer.get("strong_samples", []), explainer.get("weak_samples", [])) for item in group):
            failures.append({"feature_id": feature_id, "reason": "explainer_sample_fields_invalid"})
        if not master.get("scorer_eligible", True):
            if public or truth:
                failures.append({"feature_id": feature_id, "reason": "ineligible_packet_not_empty"})
            continue
        eligible += 1
        try:
            _validate_public_samples(public, feature_id)
        except ValueError as exc:
            failures.append({"feature_id": feature_id, "reason": str(exc)})
            continue
        ids = [item["sample_id"] for item in public]
        truth_ids = [item["public_sample_id"] for item in truth]
        if ids != truth_ids:
            failures.append({"feature_id": feature_id, "reason": "public_private_order_or_id_mismatch"})
        counts = {stratum: sum(item["stratum"] == stratum for item in truth) for stratum in expected_strata}
        if counts != {stratum: 5 for stratum in expected_strata}:
            failures.append({"feature_id": feature_id, "reason": "stratum_balance_invalid", "counts": counts})
        strata = [item["stratum"] for item in truth]
        if strata == block_order:
            failures.append({"feature_id": feature_id, "reason": "stratum_block_order_not_randomized"})
        else:
            randomized += 1
        discovery = master.get("discovery_samples_private", [])
        discovery_rows = {int(item["row_idx"]) for item in discovery}
        heldout_rows = {int(item["row_idx"]) for item in truth}
        discovery_texts = {item["normalized_text_sha256"] for item in discovery}
        heldout_texts = {item["normalized_text_sha256"] for item in truth}
        discovery_sources = {item["source_file"] for item in discovery}
        heldout_sources = {item["source_file"] for item in truth}
        if not discovery_rows.isdisjoint(heldout_rows):
            failures.append({"feature_id": feature_id, "reason": "row_overlap"})
        if not discovery_texts.isdisjoint(heldout_texts):
            failures.append({"feature_id": feature_id, "reason": "normalized_text_overlap"})
        if not discovery_sources.isdisjoint(heldout_sources):
            failures.append({"feature_id": feature_id, "reason": "source_file_overlap"})
        entry = manifest_entries.get(feature_id, {})
        if entry.get("scorer_packet_sha256") != _canonical_sha256(public):
            failures.append({"feature_id": feature_id, "reason": "scorer_packet_hash_mismatch"})
        if entry.get("private_truth_sha256") != _canonical_sha256(truth):
            failures.append({"feature_id": feature_id, "reason": "private_truth_hash_mismatch"})
        selected = sorted(
            ({"row_idx": int(item["row_idx"]), "stratum": item["stratum"]} for item in truth),
            key=lambda item: item["row_idx"],
        )
        if entry.get("selected_rows_sha256") != _canonical_sha256(selected):
            failures.append({"feature_id": feature_id, "reason": "selected_rows_hash_mismatch"})
    result = {
        "packet_version": manifest.get("packet_version"),
        "presentation_seed": manifest.get("presentation_seed"),
        "n_master_packets": len(masters),
        "n_scorer_eligible": eligible,
        "n_randomized_nonblock_packets": randomized,
        "public_scorer_fields": ["sample_id", "text"],
        "id_assignment": manifest.get("id_assignment"),
        "scorer_alignment": manifest.get("scorer_alignment"),
        "n_failures": len(failures),
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }
    write_json(output / "packet_integrity_audit.json", result)
    if failures:
        raise ValueError(f"Frozen packet integrity audit failed with {len(failures)} issue(s)")
    return result


def audit_packet_migration(*, source_output_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Verify a v3 rerun changed presentation only, not held-out row selection or Explainer prompts."""
    source, output = Path(source_output_dir), Path(output_dir)
    source_packs = {row["feature_id"]: row for row in read_jsonl(_packets_path(source))}
    target_packs = {row["feature_id"]: row for row in read_jsonl(output / "private" / "master_packets.jsonl")}
    source_tasks = {row["feature_id"]: row for row in read_jsonl(source / "explainer" / "tasks.jsonl")}
    target_tasks = {row["feature_id"]: row for row in read_jsonl(output / "explainer" / "tasks.jsonl")}
    failures: list[dict[str, Any]] = []
    eligible = 0
    changed_id_mappings = 0
    prompt_matches = 0
    for feature_id, target_pack in target_packs.items():
        source_pack = source_packs.get(feature_id)
        if source_pack is None:
            failures.append({"feature_id": feature_id, "reason": "missing_source_packet"})
            continue
        source_hash = hashlib.sha256(source_tasks[feature_id]["prompt"].encode("utf-8")).hexdigest()
        target_hash = hashlib.sha256(target_tasks[feature_id]["prompt"].encode("utf-8")).hexdigest()
        if source_hash != target_hash:
            failures.append({"feature_id": feature_id, "reason": "explainer_prompt_hash_mismatch"})
        else:
            prompt_matches += 1
        if not target_pack.get("scorer_eligible", True):
            continue
        eligible += 1
        source_truth = source_pack["heldout_samples_private"]
        target_truth = target_pack["heldout_samples_private"]
        source_selected = sorted((str(item["stratum"]), int(item["row_idx"])) for item in source_truth)
        target_selected = sorted((str(item["stratum"]), int(item["row_idx"])) for item in target_truth)
        if source_selected != target_selected:
            failures.append({"feature_id": feature_id, "reason": "selected_rows_or_strata_changed"})
        source_mapping = {str(item["sample_id"]): int(item["row_idx"]) for item in source_truth}
        target_mapping = {str(item["public_sample_id"]): int(item["row_idx"]) for item in target_truth}
        if source_mapping != target_mapping:
            changed_id_mappings += 1
        else:
            failures.append({"feature_id": feature_id, "reason": "public_id_mapping_not_changed"})
    result = {
        "source_output_dir": str(source),
        "target_output_dir": str(output),
        "n_target_packets": len(target_packs),
        "n_explainer_prompt_hash_matches": prompt_matches,
        "n_scorer_eligible": eligible,
        "n_same_selected_row_sets": eligible - sum(row["reason"] == "selected_rows_or_strata_changed" for row in failures),
        "n_changed_public_id_row_mappings": changed_id_mappings,
        "n_failures": len(failures),
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }
    write_json(output / "source_packet_migration_audit.json", result)
    if failures:
        raise ValueError(f"Packet migration audit failed with {len(failures)} issue(s)")
    return result


def compare_scorer_runs(*, source_output_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Compare corrected randomized-packet metrics with the superseded scorer run."""
    source, output = Path(source_output_dir), Path(output_dir)
    old = pd.read_csv(source / "scorer" / "faithfulness_metrics.csv")
    new = pd.read_csv(output / "scorer" / "faithfulness_metrics.csv")
    metric_names = (
        "spearman_rho", "pearson_log_activation",
        "positive_vs_zero_auroc", "high_vs_weak_pair_accuracy",
    )
    paired = old[["feature_id", "latent_idx", *metric_names]].merge(
        new[["feature_id", "latent_idx", *metric_names]],
        on=["feature_id", "latent_idx"], suffixes=("_order_leaked", "_randomized"), validate="one_to_one",
    )
    if len(paired) != len(old) or len(paired) != len(new):
        raise ValueError(f"Metric run IDs do not align: old={len(old)}, new={len(new)}, paired={len(paired)}")
    summary_metrics: dict[str, Any] = {}
    for metric in metric_names:
        old_column = f"{metric}_order_leaked"
        new_column = f"{metric}_randomized"
        delta_column = f"{metric}_delta"
        paired[delta_column] = paired[new_column] - paired[old_column]
        summary_metrics[metric] = {
            "order_leaked_mean": float(paired[old_column].mean()),
            "randomized_mean": float(paired[new_column].mean()),
            "mean_delta": float(paired[delta_column].mean()),
            "order_leaked_median": float(paired[old_column].median()),
            "randomized_median": float(paired[new_column].median()),
            "median_delta": float(paired[delta_column].median()),
            "fraction_increased": float((paired[delta_column] > 0).mean()),
            "fraction_decreased": float((paired[delta_column] < 0).mean()),
        }
    analysis = output / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    paired.to_csv(analysis / "order_leakage_correction_per_feature.csv", index=False, encoding="utf-8-sig")
    result = {
        "comparison": "randomized_packet_minus_superseded_order_leaked_deanchored",
        "source_output_dir": str(source),
        "target_output_dir": str(output),
        "n_paired_features": len(paired),
        "metrics": summary_metrics,
    }
    write_json(analysis / "order_leakage_correction_summary.json", result)
    return result


def finalize_scorer_run(*, output_dir: str | Path) -> dict[str, Any]:
    """Apply the final artifact, validation, privacy, and zero-tool completion gate."""
    output = Path(output_dir)
    tasks = read_jsonl(output / "scorer" / "tasks.jsonl")
    main = read_jsonl(output / "scorer" / "llm_execution_manifest.jsonl")
    retry_path = output / "scorer_retry" / "llm_execution_manifest.jsonl"
    retry = read_jsonl(retry_path) if retry_path.exists() else []
    validation = json.loads((output / "scorer" / "validation_manifest.json").read_text(encoding="utf-8"))
    packet_audit = json.loads((output / "packet_integrity_audit.json").read_text(encoding="utf-8"))
    migration_audit = json.loads((output / "source_packet_migration_audit.json").read_text(encoding="utf-8"))
    forbidden = ("true_activation", "true_response", "row_idx", "source_file", "stratum", "latent_idx")
    forbidden_hits = [
        {"task_id": task["task_id"], "token": token}
        for task in tasks for token in forbidden if token in str(task["prompt"])
    ]
    main_ids = [str(row.get("task_id", "")) for row in main]
    failures = []
    if len(main) != len(tasks) or len(set(main_ids)) != len(tasks): failures.append("main_execution_not_one_row_per_task")
    if any(row.get("status") != "success" for row in main): failures.append("main_execution_failure")
    if any(int(row.get("tool_event_count", -1)) != 0 for row in [*main, *retry]): failures.append("tool_event_detected")
    if any(row.get("status") != "success" for row in retry): failures.append("retry_execution_failure")
    if validation != {"n_tasks": len(tasks), "n_valid": len(tasks), "n_failed": 0}: failures.append("validation_gate_failed")
    if packet_audit.get("status") != "pass": failures.append("packet_integrity_gate_failed")
    if migration_audit.get("status") != "pass": failures.append("migration_gate_failed")
    if forbidden_hits: failures.append("forbidden_private_prompt_field")
    if len(list((output / "scorer" / "raw").glob("*.json"))) != len(tasks): failures.append("raw_output_count_mismatch")
    result = {
        "status": "pass" if not failures else "fail",
        "n_tasks": len(tasks), "n_valid": validation.get("n_valid"),
        "n_main_success": sum(row.get("status") == "success" for row in main),
        "n_retry_success": sum(row.get("status") == "success" for row in retry),
        "tool_event_count": sum(int(row.get("tool_event_count", 0)) for row in [*main, *retry]),
        "forbidden_private_prompt_hits": forbidden_hits,
        "packet_integrity_status": packet_audit.get("status"),
        "migration_audit_status": migration_audit.get("status"),
        "failures": failures,
    }
    write_json(output / "final_completion_audit.json", result)
    write_json(output / "pipeline_status.json", {"stage": "complete" if not failures else "failed", "final_completion_audit": result})
    if failures:
        raise ValueError(f"Final scorer completion gate failed: {failures}")
    return result


def _format_samples(samples: list[dict[str, str]]) -> str:
    return "\n".join(f"- sample_id={x['sample_id']}\n  text: {x['text'].replace(chr(10), ' ').strip()}" for x in samples)


def build_explainer_prompt(feature_id: str, strong: list[dict[str, str]], weak: list[dict[str, str]]) -> str:
    return f"""Analyze the following two groups of spoken or transcribed dialogue sentences. They correspond to one anonymous text feature.

Anonymous feature ID:
{feature_id}

GROUP A — STRONG GROUP SENTENCES

{_format_samples(strong)}

GROUP B — WEAK GROUP SENTENCES

{_format_samples(weak)}

Find the narrowest stable natural-language condition that is common in Group A and absent, weaker, or less consistent in Group B.

Requirements:
1. State what must be present and what superficially similar property is insufficient.
2. Separately propose a surface/linguistic hypothesis and a behavioral/discourse hypothesis.
3. Prefer the simpler surface or linguistic explanation when behavioral evidence is insufficient.
4. Partition every Group A ID into strong_supporting_sample_ids or strong_outlier_sample_ids.
5. Partition every Group B ID into weak_boundary_supporting_sample_ids or weak_counterexample_sample_ids.
6. Cite 4–8 contrastive evidence items, including at least one from each group.
7. Give 1–3 alternatives, confounds, limitations, confidence 1–5, and a short rationale.

The sentences may contain disfluencies, repetition, omissions, incomplete grammar, or transcription errors. Do not infer how the feature was produced. Do not infer a predefined label. Return only JSON matching the supplied schema. Use English prose and preserve IDs exactly."""


def load_stable_core_latents(path: str | Path) -> tuple[int, ...]:
    rows = pd.read_csv(path)
    required = {"latent_idx", "stable_set_role"}
    if not required.issubset(rows.columns):
        raise ValueError(f"Stable-latent CSV is missing columns: {sorted(required - set(rows.columns))}")
    selected = rows.loc[rows["stable_set_role"].eq("stable_core"), "latent_idx"].astype(int)
    # Preserve first appearance so anonymous feature IDs remain deterministic.
    return tuple(dict.fromkeys(selected.tolist()))


def build_packets(
    *, feature_store_path: str | Path, records_path: str | Path, output_dir: str | Path,
    latent_indices: tuple[int, ...], config: SamplingConfig = SamplingConfig(),
    scope: str = "custom", skip_ineligible: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(records_path); texts = [str(row["unit_text"]) for row in records]
    features = _load_features(feature_store_path)
    if features.shape[0] != len(records):
        raise ValueError("Feature and record row counts differ")
    held_mask = np.asarray([_is_heldout(str(row["source_file"]), config) for row in records], dtype=bool)
    discovery_rows = np.flatnonzero(~held_mask); heldout_rows = np.flatnonzero(held_mask)
    packs, tasks, audit = [], [], []
    raw_dir = output / "explainer" / "raw"; raw_dir.mkdir(parents=True, exist_ok=True)
    ineligible = []
    scorer_ineligible = []
    for order, latent_idx in enumerate(latent_indices, 1):
        values = features[:, latent_idx].numpy(); used: set[str] = set()
        dpos = discovery_rows[values[discovery_rows] > 0]
        hpos = heldout_rows[values[heldout_rows] > 0]
        hzero = heldout_rows[values[heldout_rows] == 0]
        feature_id = f"F{order:03d}"
        try:
            dweak, _, dstrong = _rank_bands(values, dpos)
            strong_rows = _unique_pick(dstrong, texts, config.n_group, key=f"{latent_idx}|strong", used=used)
            weak_rows = _unique_pick(dweak, texts, config.n_group, key=f"{latent_idx}|weak", used=used)
            if len(strong_rows) != config.n_group or len(weak_rows) != config.n_group:
                raise ValueError("lacks unique discovery strong/weak samples")
        except ValueError as exc:
            if not skip_ineligible:
                raise ValueError(f"Latent {latent_idx}: {exc}") from exc
            failure = {
                "feature_id": feature_id, "latent_idx": latent_idx,
                "sampling_status": "explainer_ineligible", "reason": str(exc),
                "n_discovery_positive": len(dpos), "n_heldout_positive": len(hpos),
            }
            ineligible.append(failure); audit.append(failure)
            continue
        held_rows: list[tuple[str, int]] = []
        selected_by_stratum: dict[str, list[int]] = {}
        heldout_reason = ""
        try:
            hweak, hmid, hstrong = _rank_bands(values, hpos)
            for tag, candidates in (("high", hstrong), ("mid", hmid), ("weak", hweak), ("zero", hzero.tolist())):
                picked = _unique_pick(candidates, texts, config.n_heldout_stratum, key=f"{latent_idx}|held|{tag}", used=used)
                if len(picked) != config.n_heldout_stratum:
                    raise ValueError(f"lacks unique heldout {tag} samples")
                selected_by_stratum[tag] = picked
                held_rows.extend((tag, row) for row in picked)
        except ValueError as exc:
            if not skip_ineligible:
                raise ValueError(f"Latent {latent_idx}: {exc}") from exc
            held_rows = []
            selected_by_stratum = {}
            heldout_reason = str(exc)
            scorer_ineligible.append({
                "feature_id": feature_id, "latent_idx": latent_idx,
                "sampling_status": "scorer_ineligible", "reason": heldout_reason,
                "n_discovery_positive": len(dpos), "n_heldout_positive": len(hpos),
            })
        strong = _public_samples(strong_rows, texts, "A")
        weak = _public_samples(weak_rows, texts, "B")
        frozen = freeze_randomized_heldout_packet(
            selected_by_stratum, texts, feature_id, config.presentation_seed,
        ) if selected_by_stratum else {
            "public_samples": [], "private_truth": [], "packet_sha256": "",
            "private_truth_sha256": "", "selected_rows_sha256": "",
            "presentation_seed": config.presentation_seed,
        }
        held_private = [
            {
                **truth,
                "true_activation": float(values[int(truth["row_idx"])]),
                "source_file": str(records[int(truth["row_idx"])]["source_file"]),
                "normalized_text_sha256": hashlib.sha256(
                    normalise_text(texts[int(truth["row_idx"])]).encode("utf-8")
                ).hexdigest(),
            }
            for truth in frozen["private_truth"]
        ]
        discovery_private = [
            {
                "public_sample_id": sample["sample_id"], "row_idx": int(row), "group": group,
                "source_file": str(records[int(row)]["source_file"]),
                "normalized_text_sha256": hashlib.sha256(normalise_text(texts[int(row)]).encode("utf-8")).hexdigest(),
            }
            for group, samples, rows in (("strong", strong, strong_rows), ("weak", weak, weak_rows))
            for sample, row in zip(samples, rows)
        ]
        pack = {
            "feature_id": feature_id, "latent_idx": latent_idx,
            "strong_samples": strong, "weak_samples": weak,
            "discovery_samples_private": discovery_private,
            "heldout_samples_private": held_private,
            "scorer_eligible": not heldout_reason,
            "scorer_ineligible_reason": heldout_reason,
            "packet_version": config.packet_version,
        }
        packs.append(pack)
        task_id = f"{feature_id}_explainer"
        explainer_prompt = build_explainer_prompt(feature_id, strong, weak)
        tasks.append({"task_id": task_id, "latent_idx": latent_idx, "feature_id": feature_id, "prompt": explainer_prompt, "prompt_sha256": hashlib.sha256(explainer_prompt.encode("utf-8")).hexdigest(), "expected_output_path": str(raw_dir / f"{task_id}.json")})
        audit.append({"feature_id": feature_id, "latent_idx": latent_idx, "sampling_status": "eligible" if not heldout_reason else "explainer_only", "reason": heldout_reason, "n_discovery_positive": len(dpos), "n_heldout_positive": len(hpos), "n_strong": len(strong), "n_weak": len(weak), "n_heldout": len(held_private), "evidence_heldout_text_overlap": False})
    public_explainer_packets = [
        {"feature_id": pack["feature_id"], "strong_samples": pack["strong_samples"], "weak_samples": pack["weak_samples"]}
        for pack in packs
    ]
    public_scorer_packets = []
    private_truth_packets = []
    packet_entries = []
    frozen_by_feature = {
        pack["feature_id"]: (
            freeze_randomized_heldout_packet(
                {
                    stratum: [int(item["row_idx"]) for item in pack["heldout_samples_private"] if item["stratum"] == stratum]
                    for stratum in ("high", "mid", "weak", "zero")
                },
                texts, pack["feature_id"], config.presentation_seed,
            ) if pack["heldout_samples_private"] else None
        )
        for pack in packs
    }
    for pack in packs:
        feature_id = pack["feature_id"]
        frozen = frozen_by_feature[feature_id]
        public_samples = frozen["public_samples"] if frozen else []
        public_scorer_packets.append({"feature_id": feature_id, "samples": public_samples})
        private_truth_packets.append({
            "feature_id": feature_id, "latent_idx": pack["latent_idx"],
            "samples": pack["heldout_samples_private"],
        })
        packet_entries.append({
            "feature_id": feature_id,
            "explainer_packet_sha256": _canonical_sha256(public_explainer_packets[len(packet_entries)]),
            "scorer_packet_sha256": frozen["packet_sha256"] if frozen else "",
            "private_truth_sha256": _canonical_sha256(pack["heldout_samples_private"]),
            "selected_rows_sha256": frozen["selected_rows_sha256"] if frozen else "",
        })
    write_jsonl(output / "private" / "master_packets.jsonl", packs)
    write_jsonl(output / "private" / "heldout_truth.jsonl", private_truth_packets)
    write_jsonl(output / "public_packets" / "explainer_packets.jsonl", public_explainer_packets)
    write_jsonl(output / "public_packets" / "scorer_packets.jsonl", public_scorer_packets)
    write_jsonl(output / "private_packets.jsonl", packs)
    write_jsonl(output / "explainer" / "tasks.jsonl", tasks)
    pd.DataFrame(audit).to_csv(output / "sampling_audit.csv", index=False)
    manifest = {"analysis": ANALYSIS_NAME, "scope": scope, "n_requested_latents": len(latent_indices), "n_latents": len(packs), "n_explainer_eligible": len(packs), "n_scorer_eligible": sum(bool(pack["scorer_eligible"]) for pack in packs), "n_sampling_ineligible": len(ineligible), "n_scorer_ineligible": len(scorer_ineligible), "requested_latents": list(latent_indices), "sampling_ineligible": ineligible, "scorer_ineligible": scorer_ineligible, "split_unit": "source_file", "heldout_fraction": config.heldout_fraction, "discovery_rows": int(len(discovery_rows)), "heldout_rows": int(len(heldout_rows)), "ai_visible_background": "spoken_or_transcribed_dialogue_only", "model_background_exposed": False, "shuffled_explanation_baseline": False, "empty_explanation_baseline": False, "bootstrap_confidence_intervals": False}
    manifest.update({"packet_version": config.packet_version, "presentation_seed": config.presentation_seed})
    write_json(output / "sampling_manifest.json", manifest)
    write_json(output / "packet_manifest.json", {
        "packet_version": config.packet_version,
        "presentation_seed": config.presentation_seed,
        "id_assignment": "after_deterministic_permutation",
        "scorer_alignment": "sample_id_join",
        "entries": packet_entries,
    })
    return manifest


def build_pilot_packets(
    *, feature_store_path: str | Path, records_path: str | Path, output_dir: str | Path,
    config: SamplingConfig = SamplingConfig(), pilot_latents: tuple[int, ...] = PILOT_LATENTS,
) -> dict[str, Any]:
    return build_packets(
        feature_store_path=feature_store_path, records_path=records_path,
        output_dir=output_dir, latent_indices=pilot_latents, config=config, scope="pilot20",
    )


def build_full_packets(
    *, feature_store_path: str | Path, records_path: str | Path,
    stable_latents_path: str | Path, output_dir: str | Path,
    config: SamplingConfig = SamplingConfig(),
) -> dict[str, Any]:
    latents = load_stable_core_latents(stable_latents_path)
    return build_packets(
        feature_store_path=feature_store_path, records_path=records_path,
        output_dir=output_dir, latent_indices=latents, config=config,
        scope="relaxed_leaf7_stable_core_full", skip_ineligible=True,
    )


def run_stage(*, stage_dir: str | Path, tasks_path: str | Path, schema_path: str | Path, instructions_path: str | Path, model: str = "gpt-5.5", reasoning_effort: str = "low", concurrency: int = 4, timeout_seconds: float = 600) -> dict[str, Any]:
    stage = Path(stage_dir); workdir, home = default_isolation_paths(stage)
    return run_codex_latent_card_tasks(tasks_path=tasks_path, output_dir=stage, schema_path=schema_path, instructions_path=instructions_path, auth_source=Path.home()/".codex"/"auth.json", workdir=workdir, codex_home=home, config=CodexLatentCardConfig(model=model, reasoning_effort=reasoning_effort, concurrency=concurrency, timeout_seconds=timeout_seconds))


def validate_explanations(*, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir); tasks = read_jsonl(output/"explainer"/"tasks.jsonl")
    valid, failures = [], []
    for task in tasks:
        retry_path = output / "explainer" / "raw_retry" / f'{task["task_id"]}.json'
        path = retry_path if retry_path.exists() else Path(task["expected_output_path"]); reasons=[]
        try: row=json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc: failures.append({"task_id":task["task_id"],"reasons":[f"read:{exc}"]}); continue
        if row.get("feature_id") != task["feature_id"]: reasons.append("feature_id_mismatch")
        a={f"A{i:03d}" for i in range(1,11)}; b={f"B{i:03d}" for i in range(1,11)}
        sa=set(row.get("strong_supporting_sample_ids",[])); so=set(row.get("strong_outlier_sample_ids",[]))
        wb=set(row.get("weak_boundary_supporting_sample_ids",[])); wc=set(row.get("weak_counterexample_sample_ids",[]))
        if sa & so or sa | so != a: reasons.append("strong_partition_invalid")
        if wb & wc or wb | wc != b: reasons.append("weak_partition_invalid")
        representatives = row.get("representative_evidence_ids", [])
        if not (2 <= len(representatives) <= 3) or not set(representatives).issubset(a | b):
            reasons.append("representative_evidence_ids_invalid")
        evidence = row.get("contrastive_evidence", [])
        evidence_ids = [item.get("sample_id") for item in evidence]
        if not (4 <= len(evidence) <= 8) or any(sample_id not in a | b for sample_id in evidence_ids):
            reasons.append("contrastive_evidence_ids_invalid")
        if any(
            (item.get("sample_id") in a and item.get("group") != "strong")
            or (item.get("sample_id") in b and item.get("group") != "weak")
            for item in evidence
        ):
            reasons.append("contrastive_evidence_group_mismatch")
        if not ({item.get("group") for item in evidence} >= {"strong", "weak"}):
            reasons.append("contrastive_evidence_missing_group")
        if reasons: failures.append({"task_id":task["task_id"],"reasons":reasons})
        else: valid.append({**row,"latent_idx":task["latent_idx"],"task_id":task["task_id"]})
    write_jsonl(output/"explainer"/"validated_explanations.jsonl",valid); write_jsonl(output/"explainer"/"validation_failures.jsonl",failures)
    result={"n_tasks":len(tasks),"n_valid":len(valid),"n_failed":len(failures)}; write_json(output/"explainer"/"validation_manifest.json",result); return result


def make_explainer_retry_tasks(*, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    failures = {row["task_id"] for row in read_jsonl(output / "explainer" / "validation_failures.jsonl")}
    raw_retry = output / "explainer" / "raw_retry"
    raw_retry.mkdir(parents=True, exist_ok=True)
    retry_tasks = []
    correction = """

STRICT OUTPUT CORRECTION:
- representative_evidence_ids must contain exactly 2 or 3 bare sample IDs only.
- Each item must exactly match one supplied ID such as A001 or B003.
- Never place explanations, punctuation, multiple IDs, or descriptive text inside an ID string.
- All other ID fields must likewise contain bare supplied IDs only.
- Check these constraints before returning the JSON object.
"""
    for task in read_jsonl(output / "explainer" / "tasks.jsonl"):
        if task["task_id"] not in failures:
            continue
        retry_tasks.append({**task, "prompt": task["prompt"] + correction, "expected_output_path": str(raw_retry / f'{task["task_id"]}.json')})
    path = output / "explainer" / "retry_tasks.jsonl"
    write_jsonl(path, retry_tasks)
    result = {"n_retry_tasks": len(retry_tasks), "reason": "strict_bare_id_membership"}
    write_json(output / "explainer" / "retry_task_manifest.json", result)
    return result


def build_scorer_prompt(feature_id: str, explanation: dict[str, Any], heldout: list[dict[str, Any]]) -> str:
    frozen={k:explanation[k] for k in SCORER_FROZEN_FIELDS}
    _validate_public_samples(heldout, feature_id)
    public=[{"sample_id":x["sample_id"],"text":x["text"]} for x in heldout]
    return f"""Use the frozen explanation below to predict how strongly each held-out spoken/transcribed-dialogue sentence matches the anonymous text feature.

Feature ID: {feature_id}

Frozen explanation:
{json.dumps(frozen,ensure_ascii=False,indent=2)}

Held-out sentences:
{_format_samples(public)}

For each complete sentence output an integer predicted_feature_score from 0 to 100: 0–10 absent/contradicted; 11–30 weak/incidental; 31–60 partial/ambiguous; 61–80 clear; 81–100 strong and specific. Do not revise the explanation, infer hidden groups, or rely on broad domain membership. matching_evidence_span must be one exact contiguous substring copied from the sentence or empty. Preserve the sentence's original case, punctuation, and whitespace exactly; never normalize spacing, paraphrase, join disjoint spans, or insert ellipses. Return all 20 IDs exactly once. Return only JSON matching the supplied schema."""


def make_scorer_tasks(*, output_dir: str | Path) -> dict[str, Any]:
    output=Path(output_dir); explanations={x["feature_id"]:x for x in read_jsonl(output/"explainer"/"validated_explanations.jsonl")}; packs={x["feature_id"]:x for x in read_jsonl(_packets_path(output))}
    public_packets_path = _public_scorer_packets_path(output)
    if not public_packets_path.exists():
        raise FileNotFoundError(f"Frozen public scorer packets are required: {public_packets_path}")
    public_packets={x["feature_id"]:x["samples"] for x in read_jsonl(public_packets_path)}
    raw=output/"scorer"/"raw"; raw.mkdir(parents=True,exist_ok=True); tasks=[]
    for feature_id in sorted(explanations):
        pack=packs[feature_id]
        if not pack.get("scorer_eligible", True):
            continue
        heldout = public_packets[feature_id]
        _validate_public_samples(heldout, feature_id)
        task_id=f"{feature_id}_scorer"
        prompt=build_scorer_prompt(feature_id,explanations[feature_id],heldout)
        tasks.append({"task_id":task_id,"latent_idx":pack["latent_idx"],"feature_id":feature_id,"prompt":prompt,"prompt_sha256":hashlib.sha256(prompt.encode("utf-8")).hexdigest(),"scorer_packet_sha256":_canonical_sha256(heldout),"expected_output_path":str(raw/f"{task_id}.json")})
    write_jsonl(output/"scorer"/"tasks.jsonl",tasks); result={"n_tasks":len(tasks),"packet_source":"public_packets/scorer_packets.jsonl","alignment":"sample_id_join","frozen_fields_exposed":list(SCORER_FROZEN_FIELDS),"excluded_fields":["contrastive_explanation","necessary_or_characteristic_condition","insufficient_conditions","possible_confounds","limitations","alternative_explanations","confidence","confidence_rationale","discovery_sample_partitions","contrastive_evidence"]}; write_json(output/"scorer"/"task_manifest.json",result); return result


def make_scorer_retry_tasks(*, output_dir: str | Path) -> dict[str, Any]:
    """Create exact-prompt retries only for failed scorer validations."""
    output = Path(output_dir)
    failed = {row["task_id"] for row in read_jsonl(output / "scorer" / "validation_failures.jsonl")}
    raw_retry = output / "scorer" / "raw_retry"
    raw_retry.mkdir(parents=True, exist_ok=True)
    retries = [
        {**task, "expected_output_path": str(raw_retry / f'{task["task_id"]}.json')}
        for task in read_jsonl(output / "scorer" / "tasks.jsonl")
        if task["task_id"] in failed
    ]
    write_jsonl(output / "scorer" / "retry_tasks.jsonl", retries)
    result = {"n_retry_tasks": len(retries), "policy": "same_prompt_no_manual_edit"}
    write_json(output / "scorer" / "retry_task_manifest.json", result)
    return result


def make_scorer_subset_tasks(
    *, source_output_dir: str | Path, output_dir: str | Path,
    feature_ids: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Build a non-overwriting scorer-only reassessment from frozen v2 cards."""
    source = Path(source_output_dir); output = Path(output_dir)
    requested = tuple(dict.fromkeys(str(feature_id) for feature_id in feature_ids))
    if not requested:
        raise ValueError("At least one feature ID is required")
    required_v3 = [
        source / "private" / "master_packets.jsonl",
        _private_truth_path(source),
        _public_scorer_packets_path(source),
        source / "public_packets" / "explainer_packets.jsonl",
        source / "packet_manifest.json",
    ]
    missing_v3 = [str(path) for path in required_v3 if not path.exists()]
    if missing_v3:
        raise ValueError("Source predates randomized frozen packets; rebuild packets before any scorer reassessment")
    explanations = {row["feature_id"]: row for row in read_jsonl(source/"explainer"/"validated_explanations.jsonl")}
    packs = {row["feature_id"]: row for row in read_jsonl(source / "private" / "master_packets.jsonl")}
    missing = [feature_id for feature_id in requested if feature_id not in explanations or feature_id not in packs]
    if missing:
        raise ValueError(f"Requested feature IDs are missing from source artifacts: {missing}")
    ineligible = [feature_id for feature_id in requested if not packs[feature_id].get("scorer_eligible", True)]
    if ineligible:
        raise ValueError(f"Requested feature IDs are scorer-ineligible: {ineligible}")
    (output/"explainer").mkdir(parents=True, exist_ok=True)
    write_jsonl(output/"explainer"/"validated_explanations.jsonl", [explanations[feature_id] for feature_id in requested])
    write_jsonl(output/"private"/"master_packets.jsonl", [packs[feature_id] for feature_id in requested])
    write_jsonl(output/"private_packets.jsonl", [packs[feature_id] for feature_id in requested])
    source_truth = {row["feature_id"]: row for row in read_jsonl(_private_truth_path(source))}
    source_scorer = {row["feature_id"]: row for row in read_jsonl(_public_scorer_packets_path(source))}
    source_explainer = {row["feature_id"]: row for row in read_jsonl(source / "public_packets" / "explainer_packets.jsonl")}
    write_jsonl(_private_truth_path(output), [source_truth[feature_id] for feature_id in requested])
    write_jsonl(_public_scorer_packets_path(output), [source_scorer[feature_id] for feature_id in requested])
    write_jsonl(output / "public_packets" / "explainer_packets.jsonl", [source_explainer[feature_id] for feature_id in requested])
    source_manifest = json.loads((source / "packet_manifest.json").read_text(encoding="utf-8"))
    selected_entries = [row for row in source_manifest["entries"] if row["feature_id"] in set(requested)]
    write_json(output / "packet_manifest.json", {**source_manifest, "entries": selected_entries})
    result = make_scorer_tasks(output_dir=output)
    result.update({
        "analysis": "contrastive_latent_faithfulness_v2_reduced_context_scorer_subset",
        "source_output_dir": str(source),
        "requested_feature_ids": list(requested),
        "source_explanations_reused_verbatim": True,
        "source_heldout_packets_reused_verbatim": True,
    })
    write_json(output/"scorer"/"task_manifest.json",result)
    return result


def prepare_randomized_scorer_rerun(
    *, source_output_dir: str | Path, output_dir: str | Path,
    feature_store_path: str | Path, records_path: str | Path,
    stable_latents_path: str | Path, config: SamplingConfig = SamplingConfig(),
) -> dict[str, Any]:
    """Rebuild frozen packets and safely reuse explanations after prompt-hash verification."""
    source, output = Path(source_output_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite corrected rerun directory: {output}")
    build_manifest = build_full_packets(
        feature_store_path=feature_store_path, records_path=records_path,
        stable_latents_path=stable_latents_path, output_dir=output, config=config,
    )
    source_tasks = {row["feature_id"]: row for row in read_jsonl(source / "explainer" / "tasks.jsonl")}
    target_tasks = {row["feature_id"]: row for row in read_jsonl(output / "explainer" / "tasks.jsonl")}
    explanations = read_jsonl(source / "explainer" / "validated_explanations.jsonl")
    reusable = []
    mismatches = []
    for explanation in explanations:
        feature_id = str(explanation["feature_id"])
        if feature_id not in source_tasks or feature_id not in target_tasks:
            mismatches.append({"feature_id": feature_id, "reason": "missing_explainer_task"})
            continue
        source_hash = hashlib.sha256(source_tasks[feature_id]["prompt"].encode("utf-8")).hexdigest()
        target_hash = hashlib.sha256(target_tasks[feature_id]["prompt"].encode("utf-8")).hexdigest()
        if source_hash != target_hash:
            mismatches.append({"feature_id": feature_id, "source_prompt_sha256": source_hash, "target_prompt_sha256": target_hash})
            continue
        reusable.append(explanation)
    if mismatches or len(reusable) != len(explanations):
        write_json(output / "explainer" / "reuse_verification_failure.json", {
            "n_source_explanations": len(explanations), "n_reusable": len(reusable), "mismatches": mismatches,
        })
        raise ValueError(f"Explainer prompt hash verification failed for {len(mismatches)} feature(s)")
    write_jsonl(output / "explainer" / "validated_explanations.jsonl", reusable)
    write_json(output / "explainer" / "validation_manifest.json", {
        "n_tasks": len(target_tasks), "n_valid": len(reusable),
        "n_failed": len(target_tasks) - len(reusable),
        "source_output_dir": str(source),
        "reuse_policy": "exact_explainer_prompt_sha256_match",
        "all_reused_prompt_hashes_match": True,
    })
    scorer_manifest = make_scorer_tasks(output_dir=output)
    result = {
        "analysis": "contrastive_latent_faithfulness_v3_randomized_packets_scorer_rerun",
        "source_output_dir": str(source),
        "packet_version": config.packet_version,
        "presentation_seed": config.presentation_seed,
        "n_source_explanations": len(explanations),
        "n_reused_explanations": len(reusable),
        "all_reused_prompt_hashes_match": True,
        "n_scorer_tasks": scorer_manifest["n_tasks"],
        "build_manifest": build_manifest,
    }
    write_json(output / "rerun_preparation_manifest.json", result)
    return result


def validate_scorer_and_score(*, output_dir: str | Path) -> dict[str, Any]:
    output=Path(output_dir); tasks=read_jsonl(output/"scorer"/"tasks.jsonl"); predictions=[]; metrics=[]; failures=[]
    public_packets={x["feature_id"]:x["samples"] for x in read_jsonl(_public_scorer_packets_path(output))}
    truth_packets={x["feature_id"]:x for x in read_jsonl(_private_truth_path(output))}
    for task in tasks:
        retry_path=output/"scorer"/"raw_retry"/f'{task["task_id"]}.json'
        response_path=retry_path if retry_path.exists() else Path(task["expected_output_path"])
        try: row=json.loads(response_path.read_text(encoding="utf-8"))
        except Exception as exc: failures.append({"task_id":task["task_id"],"reason":f"read:{exc}"}); continue
        if row.get("feature_id")!=task["feature_id"]: failures.append({"task_id":task["task_id"],"reason":"feature_id_mismatch"}); continue
        feature_id=task["feature_id"]
        try:
            aligned=align_scorer_predictions_by_id(
                public_samples=public_packets[feature_id],
                private_truth=truth_packets[feature_id]["samples"],
                predictions=row.get("predictions",[]), feature_id=feature_id,
            )
        except (KeyError, ValueError) as exc:
            failures.append({"task_id":task["task_id"],"reason":str(exc)}); continue
        bad_spans=[pred.get("sample_id") for pred,_,sample in aligned if pred.get("matching_evidence_span","") and pred["matching_evidence_span"] not in sample["text"]]
        if bad_spans: failures.append({"task_id":task["task_id"],"reason":"non_verbatim_evidence_span","sample_ids":bad_spans}); continue
        y=np.asarray([float(truth["true_activation"]) for _,truth,_ in aligned]); p=np.asarray([float(pred["predicted_feature_score"]) for pred,_,_ in aligned]); strata=[truth["stratum"] for _,truth,_ in aligned]
        rho=float(spearmanr(p,y).statistic); pear=float(pearsonr(p,np.log1p(y)).statistic); binary=np.asarray([s!="zero" for s in strata],dtype=int); auc=float(roc_auc_score(binary,p)); high=p[np.asarray(strata)=="high"]; weak=p[np.asarray(strata)=="weak"]; pair=float(np.mean(high[:,None]>weak[None,:])+0.5*np.mean(high[:,None]==weak[None,:]))
        metrics.append({"feature_id":task["feature_id"],"latent_idx":task["latent_idx"],"spearman_rho":rho,"pearson_log_activation":pear,"positive_vs_zero_auroc":auc,"high_vs_weak_pair_accuracy":pair})
        predictions.extend({"feature_id":task["feature_id"],"latent_idx":task["latent_idx"],**pred,"true_activation":truth["true_activation"],"stratum":truth["stratum"]} for pred,truth,_ in aligned)
    write_jsonl(output/"scorer"/"validated_predictions_private.jsonl",predictions); pd.DataFrame(metrics).to_csv(output/"scorer"/"faithfulness_metrics.csv",index=False); write_jsonl(output/"scorer"/"validation_failures.jsonl",failures)
    result={"n_tasks":len(tasks),"n_valid":len(metrics),"n_failed":len(failures)}; write_json(output/"scorer"/"validation_manifest.json",result); return result


def _packets_path(output: Path) -> Path:
    current = output / "private_packets.jsonl"
    return current if current.exists() else output / "private_pilot_packets.jsonl"


__all__=["PILOT_LATENTS","SCORER_FROZEN_FIELDS","SamplingConfig","align_scorer_predictions_by_id","audit_frozen_packet_integrity","audit_packet_migration","build_packets","build_full_packets","build_pilot_packets","compare_scorer_runs","finalize_scorer_run","freeze_randomized_heldout_packet","load_stable_core_latents","make_explainer_retry_tasks","make_scorer_retry_tasks","make_scorer_tasks","make_scorer_subset_tasks","prepare_randomized_scorer_rerun","run_stage","validate_explanations","validate_scorer_and_score"]
