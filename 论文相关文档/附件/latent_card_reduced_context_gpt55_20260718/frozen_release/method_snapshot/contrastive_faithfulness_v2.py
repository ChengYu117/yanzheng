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
        held_rows = []
        heldout_reason = ""
        try:
            hweak, hmid, hstrong = _rank_bands(values, hpos)
            for tag, candidates in (("high", hstrong), ("mid", hmid), ("weak", hweak), ("zero", hzero.tolist())):
                picked = _unique_pick(candidates, texts, config.n_heldout_stratum, key=f"{latent_idx}|held|{tag}", used=used)
                if len(picked) != config.n_heldout_stratum:
                    raise ValueError(f"lacks unique heldout {tag} samples")
                held_rows.extend((tag, row) for row in picked)
        except ValueError as exc:
            if not skip_ineligible:
                raise ValueError(f"Latent {latent_idx}: {exc}") from exc
            held_rows = []
            heldout_reason = str(exc)
            scorer_ineligible.append({
                "feature_id": feature_id, "latent_idx": latent_idx,
                "sampling_status": "scorer_ineligible", "reason": heldout_reason,
                "n_discovery_positive": len(dpos), "n_heldout_positive": len(hpos),
            })
        strong = _public_samples(strong_rows, texts, "A")
        weak = _public_samples(weak_rows, texts, "B")
        held_public = _public_samples([row for _, row in held_rows], texts, "H")
        held_private = [
            {**sample, "row_idx": int(row), "stratum": tag, "true_activation": float(values[row])}
            for sample, (tag, row) in zip(held_public, held_rows)
        ]
        pack = {"feature_id": feature_id, "latent_idx": latent_idx, "strong_samples": strong, "weak_samples": weak, "heldout_samples_private": held_private, "scorer_eligible": not heldout_reason, "scorer_ineligible_reason": heldout_reason}
        packs.append(pack)
        task_id = f"{feature_id}_explainer"
        tasks.append({"task_id": task_id, "latent_idx": latent_idx, "feature_id": feature_id, "prompt": build_explainer_prompt(feature_id, strong, weak), "expected_output_path": str(raw_dir / f"{task_id}.json")})
        audit.append({"feature_id": feature_id, "latent_idx": latent_idx, "sampling_status": "eligible" if not heldout_reason else "explainer_only", "reason": heldout_reason, "n_discovery_positive": len(dpos), "n_heldout_positive": len(hpos), "n_strong": len(strong), "n_weak": len(weak), "n_heldout": len(held_private), "evidence_heldout_text_overlap": False})
    write_jsonl(output / "private_packets.jsonl", packs)
    write_jsonl(output / "explainer" / "tasks.jsonl", tasks)
    pd.DataFrame(audit).to_csv(output / "sampling_audit.csv", index=False)
    manifest = {"analysis": ANALYSIS_NAME, "scope": scope, "n_requested_latents": len(latent_indices), "n_latents": len(packs), "n_explainer_eligible": len(packs), "n_scorer_eligible": sum(bool(pack["scorer_eligible"]) for pack in packs), "n_sampling_ineligible": len(ineligible), "n_scorer_ineligible": len(scorer_ineligible), "requested_latents": list(latent_indices), "sampling_ineligible": ineligible, "scorer_ineligible": scorer_ineligible, "split_unit": "source_file", "heldout_fraction": config.heldout_fraction, "discovery_rows": int(len(discovery_rows)), "heldout_rows": int(len(heldout_rows)), "ai_visible_background": "spoken_or_transcribed_dialogue_only", "model_background_exposed": False, "shuffled_explanation_baseline": False, "empty_explanation_baseline": False, "bootstrap_confidence_intervals": False}
    write_json(output / "sampling_manifest.json", manifest)
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
    public=[{"sample_id":x["sample_id"],"text":x["text"]} for x in heldout]
    return f"""Use the frozen explanation below to predict how strongly each held-out spoken/transcribed-dialogue sentence matches the anonymous text feature.

Feature ID: {feature_id}

Frozen explanation:
{json.dumps(frozen,ensure_ascii=False,indent=2)}

Held-out sentences:
{_format_samples(public)}

For each complete sentence output an integer predicted_feature_score from 0 to 100: 0–10 absent/contradicted; 11–30 weak/incidental; 31–60 partial/ambiguous; 61–80 clear; 81–100 strong and specific. Do not revise the explanation, infer hidden groups, or rely on broad domain membership. matching_evidence_span must be one exact contiguous substring copied from the sentence or empty. Preserve the sentence's original case, punctuation, and whitespace exactly; never normalize spacing, paraphrase, join disjoint spans, or insert ellipses. Return all 20 IDs exactly once and in input order. Return only JSON matching the supplied schema."""


def make_scorer_tasks(*, output_dir: str | Path) -> dict[str, Any]:
    output=Path(output_dir); explanations={x["feature_id"]:x for x in read_jsonl(output/"explainer"/"validated_explanations.jsonl")}; packs={x["feature_id"]:x for x in read_jsonl(_packets_path(output))}
    raw=output/"scorer"/"raw"; raw.mkdir(parents=True,exist_ok=True); tasks=[]
    for feature_id in sorted(explanations):
        pack=packs[feature_id]
        if not pack.get("scorer_eligible", True):
            continue
        task_id=f"{feature_id}_scorer"
        tasks.append({"task_id":task_id,"latent_idx":pack["latent_idx"],"feature_id":feature_id,"prompt":build_scorer_prompt(feature_id,explanations[feature_id],pack["heldout_samples_private"]),"expected_output_path":str(raw/f"{task_id}.json")})
    write_jsonl(output/"scorer"/"tasks.jsonl",tasks); result={"n_tasks":len(tasks),"frozen_fields_exposed":list(SCORER_FROZEN_FIELDS),"excluded_fields":["contrastive_explanation","necessary_or_characteristic_condition","insufficient_conditions","possible_confounds","limitations","alternative_explanations","confidence","confidence_rationale","discovery_sample_partitions","contrastive_evidence"]}; write_json(output/"scorer"/"task_manifest.json",result); return result


def make_scorer_subset_tasks(
    *, source_output_dir: str | Path, output_dir: str | Path,
    feature_ids: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Build a non-overwriting scorer-only reassessment from frozen v2 cards."""
    source = Path(source_output_dir); output = Path(output_dir)
    requested = tuple(dict.fromkeys(str(feature_id) for feature_id in feature_ids))
    if not requested:
        raise ValueError("At least one feature ID is required")
    explanations = {row["feature_id"]: row for row in read_jsonl(source/"explainer"/"validated_explanations.jsonl")}
    packs = {row["feature_id"]: row for row in read_jsonl(_packets_path(source))}
    missing = [feature_id for feature_id in requested if feature_id not in explanations or feature_id not in packs]
    if missing:
        raise ValueError(f"Requested feature IDs are missing from source artifacts: {missing}")
    ineligible = [feature_id for feature_id in requested if not packs[feature_id].get("scorer_eligible", True)]
    if ineligible:
        raise ValueError(f"Requested feature IDs are scorer-ineligible: {ineligible}")
    (output/"explainer").mkdir(parents=True, exist_ok=True)
    write_jsonl(output/"explainer"/"validated_explanations.jsonl", [explanations[feature_id] for feature_id in requested])
    write_jsonl(output/"private_packets.jsonl", [packs[feature_id] for feature_id in requested])
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


def validate_scorer_and_score(*, output_dir: str | Path) -> dict[str, Any]:
    output=Path(output_dir); tasks=read_jsonl(output/"scorer"/"tasks.jsonl"); packs={x["feature_id"]:x for x in read_jsonl(_packets_path(output))}; predictions=[]; metrics=[]; failures=[]
    for task in tasks:
        try: row=json.loads(Path(task["expected_output_path"]).read_text(encoding="utf-8"))
        except Exception as exc: failures.append({"task_id":task["task_id"],"reason":f"read:{exc}"}); continue
        expected=[f"H{i:03d}" for i in range(1,21)]; got=[x.get("sample_id") for x in row.get("predictions",[])]
        if row.get("feature_id")!=task["feature_id"] or got!=expected: failures.append({"task_id":task["task_id"],"reason":"id_or_order_mismatch"}); continue
        truth=packs[task["feature_id"]]["heldout_samples_private"]
        bad_spans=[pred.get("sample_id") for pred,sample in zip(row["predictions"],truth) if pred.get("matching_evidence_span","") and pred["matching_evidence_span"] not in sample["text"]]
        if bad_spans: failures.append({"task_id":task["task_id"],"reason":"non_verbatim_evidence_span","sample_ids":bad_spans}); continue
        y=np.asarray([float(x["true_activation"]) for x in truth]); p=np.asarray([float(x["predicted_feature_score"]) for x in row["predictions"]]); strata=[x["stratum"] for x in truth]
        rho=float(spearmanr(p,y).statistic); pear=float(pearsonr(p,np.log1p(y)).statistic); binary=np.asarray([s!="zero" for s in strata],dtype=int); auc=float(roc_auc_score(binary,p)); high=p[np.asarray(strata)=="high"]; weak=p[np.asarray(strata)=="weak"]; pair=float(np.mean(high[:,None]>weak[None,:])+0.5*np.mean(high[:,None]==weak[None,:]))
        metrics.append({"feature_id":task["feature_id"],"latent_idx":task["latent_idx"],"spearman_rho":rho,"pearson_log_activation":pear,"positive_vs_zero_auroc":auc,"high_vs_weak_pair_accuracy":pair})
        predictions.extend({"feature_id":task["feature_id"],"latent_idx":task["latent_idx"],**pred,"true_activation":truth[i]["true_activation"],"stratum":truth[i]["stratum"]} for i,pred in enumerate(row["predictions"]))
    write_jsonl(output/"scorer"/"validated_predictions_private.jsonl",predictions); pd.DataFrame(metrics).to_csv(output/"scorer"/"faithfulness_metrics.csv",index=False); write_jsonl(output/"scorer"/"validation_failures.jsonl",failures)
    result={"n_tasks":len(tasks),"n_valid":len(metrics),"n_failed":len(failures)}; write_json(output/"scorer"/"validation_manifest.json",result); return result


def _packets_path(output: Path) -> Path:
    current = output / "private_packets.jsonl"
    return current if current.exists() else output / "private_pilot_packets.jsonl"


__all__=["PILOT_LATENTS","SCORER_FROZEN_FIELDS","SamplingConfig","build_packets","build_full_packets","build_pilot_packets","load_stable_core_latents","make_explainer_retry_tasks","make_scorer_tasks","make_scorer_subset_tasks","run_stage","validate_explanations","validate_scorer_and_score"]
