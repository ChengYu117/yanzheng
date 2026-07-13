"""DeepSeek-generated dual-track 2x2 text blocks and SAE activation tests."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contrastive_evidence_pack import normalise_text, read_jsonl, write_json, write_jsonl
from .contrastive_llm_io import parse_llm_json_file


CELLS = ("s_plus_f_plus", "s_minus_f_plus", "s_plus_f_minus", "s_minus_f_minus")


@dataclass(frozen=True)
class DualTrackPairConfig:
    blocks_per_latent: int = 5
    pilot_per_label: int = 2
    mode: str = "pilot"
    max_length_ratio: float = 1.25
    min_token_jaccard: float = 0.15
    max_word_delta: int = 5


def _prompt_context(explanation: dict[str, Any], top5_templates: list[str]) -> str:
    payload = {
        "surface_pattern": explanation["surface_pattern"],
        "surface_confidence": explanation["surface_confidence"],
        "surface_support_fraction": explanation["surface_raw_support_fraction"],
        "semantic_pattern": explanation["semantic_pattern"],
        "semantic_confidence": explanation["semantic_confidence"],
        "semantic_support_fraction": explanation["semantic_raw_support_fraction"],
        "candidate_explanation": explanation["candidate_explanation"],
        "alternative_hypotheses": explanation["alternative_hypotheses"],
        "failure_modes": explanation["failure_modes"],
        "reference_utterances": top5_templates,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _single_rewrite_output_contract() -> str:
    return '''Return only this JSON object:
{"generation_status":"feasible","generated_text":"one rewritten utterance"}'''


def _prompt_b(explanation: dict[str, Any], top5_templates: list[str], template_text: str) -> str:
    return f"""Rewrite one counselor utterance.

Requirements:
1. Keep the same topic, meaning, and the specified communicative function.
2. Change the identified surface pattern substantially.
3. Keep the rewrite natural and within 5 words of the original word count.
4. Do not merely change punctuation, contractions, or one synonym.
5. Return one rewrite only.

Surface pattern to change:
{explanation["surface_pattern"]}

Communicative function to preserve:
{explanation["semantic_pattern"]}

Original utterance:
{template_text}

Example:
Original: What has helped you stay on track this week?
Rewrite: Could you describe the things that helped you stay on track this week?

Final instruction: Change the surface feature "{explanation["surface_pattern"]}" while preserving the functional feature "{explanation["semantic_pattern"]}".

{_single_rewrite_output_contract()}
"""


def _prompt_c(explanation: dict[str, Any], top5_templates: list[str], template_text: str) -> str:
    return f"""Rewrite one counselor utterance.

Requirements:
1. Preserve the identified surface pattern and main topic.
2. Do not perform the specified original communicative function. Change to a clearly different communicative-act category.
3. Keep the rewrite natural and within 5 words of the original word count.
4. Changing only details, polarity, sentiment, certainty, intensity, importance, or confidence is invalid. A question rewritten as another question is invalid.
5. Return one rewrite only.

Surface pattern to preserve:
{explanation["surface_pattern"]}

Original communicative function to avoid:
{explanation["semantic_pattern"]}

Original utterance:
{template_text}

Example:
Original: What changes have you noticed this week?
Rewrite: What meaningful changes you have made this week.

Final instruction: Preserve the surface feature "{explanation["surface_pattern"]}" while changing away from the functional feature "{explanation["semantic_pattern"]}" to a clearly different communicative act.

{_single_rewrite_output_contract()}
"""


def _prompt(explanation: dict[str, Any], top5_templates: list[str], template_text: str, target_cell: str) -> str:
    if target_cell == "B_surface_minus_function_plus":
        return _prompt_b(explanation, top5_templates, template_text)
    if target_cell == "C_surface_plus_function_minus":
        return _prompt_c(explanation, top5_templates, template_text)
    raise ValueError(f"Unsupported target_cell: {target_cell}")


def _representative_texts(explanation: dict[str, Any], task: dict[str, Any]) -> list[str]:
    by_id = {str(row["id"]): str(row["text"]) for row in task["visible_samples"]}
    ids = list(explanation.get("surface_representative_evidence_ids", []))
    ids += list(explanation.get("semantic_representative_evidence_ids", []))
    seen: set[str] = set()
    texts: list[str] = []
    for sample_id in ids:
        text = by_id.get(str(sample_id), "")
        norm = normalise_text(text)
        if text and norm not in seen:
            texts.append(text)
            seen.add(norm)
    return texts[:6]


def _select_latents(explanations: pd.DataFrame, stable: pd.DataFrame, cfg: DualTrackPairConfig) -> pd.DataFrame:
    associations = stable[stable["stable_set_role"].astype(str).eq("stable_core")].copy()
    associations["label"] = associations["label"].astype(str).str.upper()
    associations["latent_idx"] = pd.to_numeric(associations["latent_idx"]).astype(int)
    eligible = explanations[explanations["induction_status"].eq("both_majority")].copy()
    eligible["track_score"] = eligible[
        ["surface_confidence", "semantic_confidence", "surface_raw_support_fraction", "semantic_raw_support_fraction"]
    ].min(axis=1)
    joined = associations.merge(eligible, on="latent_idx", how="inner")
    if cfg.mode == "full":
        return joined.sort_values(["latent_idx", "label"]).drop_duplicates("latent_idx")
    selected: list[pd.Series] = []
    used: set[int] = set()
    for label in sorted(joined["label"].unique()):
        group = joined[joined["label"].eq(label)].sort_values(
            ["track_score", "latent_idx"], ascending=[False, True]
        )
        rows = [row for _, row in group.iterrows() if int(row["latent_idx"]) not in used]
        if len(rows) < cfg.pilot_per_label:
            raise ValueError(f"Not enough unused both-majority latents for pilot label {label}")
        for row in rows[: cfg.pilot_per_label]:
            selected.append(row)
            used.add(int(row["latent_idx"]))
    return pd.DataFrame(selected).reset_index(drop=True)


def build_dual_track_pair_tasks(
    *,
    explanations_path: str | Path,
    top50_tasks_path: str | Path,
    stable_latents_path: str | Path,
    output_dir: str | Path,
    config: DualTrackPairConfig = DualTrackPairConfig(),
) -> dict[str, Any]:
    output = Path(output_dir)
    explanations = pd.DataFrame(read_jsonl(explanations_path))
    stable = pd.read_csv(stable_latents_path)
    selected = _select_latents(explanations, stable, config)
    top_tasks = {int(row["latent_idx"]): row for row in read_jsonl(top50_tasks_path)}
    raw_dir = output / "raw_designer_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    for row in selected.to_dict(orient="records"):
        latent = int(row["latent_idx"])
        task_id = f"dtm_{latent:05d}"
        explanation = dict(row)
        top5 = [str(sample["text"]) for sample in top_tasks[latent]["visible_samples"][:5]]
        for template_index, template_text in enumerate(top5, 1):
            for target_cell in ("B_surface_minus_function_plus", "C_surface_plus_function_minus"):
                suffix = "B" if target_cell.startswith("B_") else "C"
                mutation_id = f"{task_id}_t{template_index:02d}_{suffix}"
                task = {
                    "task_id": mutation_id,
                    "task_type": "single_cell_minimal_mutation",
                    "latent_idx": latent,
                    "associated_label": str(row["label"]),
                    "template_index": template_index,
                    "template_text": template_text,
                    "target_cell": target_cell,
                    "prompt": _prompt(explanation, top5, template_text, target_cell),
                    "visible_samples": [],
                    "expected_output_path": str(raw_dir / f"{mutation_id}.json"),
                    "selection_unit": "top5_template_single_mutation",
                }
                tasks.append(task)
                index_rows.append({"task_id": mutation_id, "latent_idx": latent, "associated_label": row["label"], "template_index": template_index, "target_cell": target_cell, "track_score": row["track_score"]})
    tasks_path = output / "llm_tasks" / "dual_track_minimal_pair_tasks.jsonl"
    write_jsonl(tasks_path, tasks)
    pd.DataFrame(index_rows).to_csv(output / "pilot_latent_index.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "analysis": "dual_track_minimal_pair_2x2",
        "step": "build",
        "inputs": {"explanations": str(explanations_path), "top50_tasks": str(top50_tasks_path), "stable_latents": str(stable_latents_path)},
        "outputs": {"tasks": str(tasks_path), "raw": str(raw_dir)},
        "parameters": asdict(config),
        "n_tasks": len(tasks),
        "n_templates": int(len(selected) * 5),
        "n_phase1_mutation_tasks": len(tasks),
        "generation_design": "A=observed_top5_template; B and C generated in separate requests; D generated after phase1 validation",
    }
    write_json(output / "build_manifest.json", manifest)
    return manifest


def _jaccard(a: str, b: str) -> float:
    left, right = set(normalise_text(a).split()), set(normalise_text(b).split())
    return len(left & right) / max(len(left | right), 1)


def validate_phase1_mutations(*, tasks_path: str | Path, output_dir: str | Path, config: DualTrackPairConfig) -> dict[str, Any]:
    output = Path(output_dir)
    rows, errors, retry = [], [], []
    tasks = read_jsonl(tasks_path)
    for task in tasks:
        try:
            payload = parse_llm_json_file(task["expected_output_path"])
            if not isinstance(payload, dict) or payload.get("generation_status") != "feasible":
                raise ValueError("mutation is not feasible")
            generated = str(payload.get("generated_text", "")).strip()
            template = str(task["template_text"]).strip()
            if not generated or normalise_text(generated) == normalise_text(template):
                raise ValueError("generated text is empty or unchanged")
            lengths = [max(len(normalise_text(x).split()), 1) for x in (template, generated)]
            if abs(lengths[0] - lengths[1]) > config.max_word_delta:
                raise ValueError("template/mutation word-count difference too high")
            if _jaccard(template, generated) < config.min_token_jaccard:
                raise ValueError("template/mutation lexical overlap too low")
            rows.append({**{k: task[k] for k in ("task_id","latent_idx","associated_label","template_index","template_text","target_cell")},
                "generated_text": generated, "preserved_factor": str(payload.get("preserved_factor", "")),
                "changed_factor": str(payload.get("changed_factor", "")), "surface_realization": str(payload.get("surface_realization", "")),
                "semantic_function": str(payload.get("semantic_function", "")), "naturalness_note": str(payload.get("naturalness_note", ""))})
        except Exception as exc:
            errors.append({"task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"}); retry.append(task)
    path = output / "validated_phase1_mutations.jsonl"; write_jsonl(path, rows)
    pd.DataFrame(errors).to_csv(output / "phase1_validation_errors.csv", index=False, encoding="utf-8-sig")
    write_jsonl(output / "phase1_retry_tasks.jsonl", retry)
    manifest={"step":"validate-phase1","n_tasks":len(tasks),"n_valid":len(rows),"n_errors":len(errors),"outputs":{"mutations":str(path)}}
    write_json(output / "phase1_validation_manifest.json",manifest); return manifest


def build_phase2_d_tasks(*, mutations_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output=Path(output_dir); rows=read_jsonl(mutations_path); grouped={}
    for row in rows: grouped.setdefault((int(row["latent_idx"]),int(row["template_index"])),{})[str(row["target_cell"])[0]]=row
    tasks=[]; raw=output/"raw_d_outputs"; raw.mkdir(parents=True,exist_ok=True)
    for (latent,idx), cells in grouped.items():
        if set(cells)!={"B","C"}: continue
        b,c=cells["B"],cells["C"]; task_id=f"dtm_{latent:05d}_t{idx:02d}_D"
        prompt=f'''Write one counselor utterance.

Requirements:
1. Use the surface form shown by the surface reference.
2. Use the communicative act shown by the function reference.
3. Keep the original topic and counselor role.
4. Keep the result natural and within 5 words of the original word count.
5. The result must differ from all three input utterances.
6. Return one utterance only.

Original utterance:
{b["template_text"]}

Surface reference:
{b["generated_text"]}

Function reference:
{c["generated_text"]}

{_single_rewrite_output_contract()}'''
        tasks.append({"task_id":task_id,"task_type":"single_cell_D_mutation","latent_idx":latent,"associated_label":b["associated_label"],"template_index":idx,"template_text":b["template_text"],"B_text":b["generated_text"],"C_text":c["generated_text"],"prompt":prompt,"visible_samples":[],"expected_output_path":str(raw/f"{task_id}.json"),"selection_unit":"sequential_D"})
    path=output/"llm_tasks"/"phase2_d_tasks.jsonl"; write_jsonl(path,tasks)
    manifest={"step":"build-phase2-d","n_tasks":len(tasks),"outputs":{"tasks":str(path)}}; write_json(output/"phase2_build_manifest.json",manifest); return manifest


def validate_phase2_and_assemble(*, d_tasks_path: str | Path, output_dir: str | Path, config: DualTrackPairConfig) -> dict[str, Any]:
    output=Path(output_dir); blocks=[]; errors=[]
    for task in read_jsonl(d_tasks_path):
        try:
            p=parse_llm_json_file(task["expected_output_path"]); d=str(p.get("generated_text","")).strip()
            if p.get("generation_status")!="feasible" or not d: raise ValueError("D not feasible")
            texts=[task["template_text"],task["B_text"],task["C_text"],d]
            if len({normalise_text(x) for x in texts})!=4: raise ValueError("A/B/C/D not distinct")
            lengths=[max(len(normalise_text(x).split()),1) for x in texts]
            if any(abs(length-lengths[0])>config.max_word_delta for length in lengths[1:]): raise ValueError("four-cell word-count difference too high")
            blocks.append({"task_id":task["task_id"],"latent_idx":task["latent_idx"],"associated_label":task["associated_label"],"block_id":f"dtm_{int(task['latent_idx']):05d}_b{int(task['template_index']):02d}","s_plus_f_plus":texts[0],"s_minus_f_plus":texts[1],"s_plus_f_minus":texts[2],"s_minus_f_minus":texts[3]})
        except Exception as exc: errors.append({"task_id":task["task_id"],"error":f"{type(exc).__name__}: {exc}"})
    path=output/"validated_2x2_blocks.jsonl"; write_jsonl(path,blocks); pd.DataFrame(errors).to_csv(output/"phase2_validation_errors.csv",index=False,encoding="utf-8-sig")
    manifest={"step":"validate-phase2","n_blocks":len(blocks),"n_errors":len(errors),"outputs":{"blocks":str(path)}}; write_json(output/"phase2_validation_manifest.json",manifest); return manifest


def validate_dual_track_pair_outputs(*, tasks_path: str | Path, output_dir: str | Path, config: DualTrackPairConfig) -> dict[str, Any]:
    output = Path(output_dir)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    retry: list[dict[str, Any]] = []
    for task in read_jsonl(tasks_path):
        try:
            payload = parse_llm_json_file(task["expected_output_path"])
            blocks = payload.get("blocks") if isinstance(payload, dict) else None
            if not isinstance(blocks, list) or len(blocks) != int(task["expected_blocks"]):
                raise ValueError("wrong block count")
            task_rows: list[dict[str, Any]] = []
            for idx, block in enumerate(blocks, 1):
                if block.get("generation_status") != "feasible":
                    raise ValueError(f"block {idx} is not feasible")
                texts = {cell: str(block.get(cell, "")).strip() for cell in CELLS}
                if any(not text for text in texts.values()) or len({normalise_text(v) for v in texts.values()}) != 4:
                    raise ValueError(f"block {idx} has empty or duplicate cells")
                anchor = str(block.get("surface_anchor", "")).strip().lower()
                if not anchor or anchor not in texts["s_plus_f_plus"].lower() or anchor not in texts["s_plus_f_minus"].lower():
                    raise ValueError(f"block {idx} surface anchor missing from A/C")
                if anchor in texts["s_minus_f_plus"].lower() or anchor in texts["s_minus_f_minus"].lower():
                    raise ValueError(f"block {idx} surface anchor leaks into B/D")
                lengths = [max(len(normalise_text(v).split()), 1) for v in texts.values()]
                if max(lengths) / min(lengths) > config.max_length_ratio:
                    raise ValueError(f"block {idx} length ratio too high")
                if _jaccard(texts["s_plus_f_plus"], texts["s_minus_f_plus"]) < config.min_token_jaccard:
                    raise ValueError(f"block {idx} A/B lexical overlap too low")
                if _jaccard(texts["s_plus_f_plus"], texts["s_plus_f_minus"]) < config.min_token_jaccard:
                    raise ValueError(f"block {idx} A/C lexical overlap too low")
                task_rows.append({
                    "task_id": task["task_id"], "latent_idx": int(task["latent_idx"]),
                    "associated_label": task["associated_label"], "block_id": f"{task['task_id']}_b{idx:02d}",
                    "topic": str(block.get("topic", "")), "surface_anchor": anchor,
                    "semantic_function": str(block.get("semantic_function", "")),
                    "surface_changed": str(block.get("surface_changed", "")),
                    "semantic_changed": str(block.get("semantic_changed", "")),
                    "held_constant": str(block.get("held_constant", "")), **texts,
                })
            rows.extend(task_rows)
        except Exception as exc:
            errors.append({"task_id": task["task_id"], "latent_idx": task["latent_idx"], "error": f"{type(exc).__name__}: {exc}"})
            retry.append(task)
    blocks_path = output / "validated_2x2_blocks.jsonl"
    write_jsonl(blocks_path, rows)
    pd.DataFrame(errors).to_csv(output / "validation_errors.csv", index=False, encoding="utf-8-sig")
    write_jsonl(output / "retry_tasks.jsonl", retry)
    manifest = {"step": "validate", "n_tasks": len(read_jsonl(tasks_path)), "n_valid_tasks": len(set(r["task_id"] for r in rows)), "n_blocks": len(rows), "n_errors": len(errors), "outputs": {"blocks": str(blocks_path)}}
    write_json(output / "validation_manifest.json", manifest)
    return manifest


def run_dual_track_activation_test(
    *, blocks_path: str | Path, output_dir: str | Path, sae_config_path: str | Path,
    model_config_path: str | Path, model_dir: str | None, feature_store_path: str | Path,
    stable_latents_path: str | Path, device: str | None = None, batch_size: int = 1,
) -> dict[str, Any]:
    import torch
    from .activations import extract_and_process_streaming
    from .model import load_local_model_and_tokenizer
    from .sae import load_sae_from_hub

    blocks = read_jsonl(blocks_path)
    texts = [str(row[cell]) for row in blocks for cell in CELLS]
    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = json.loads(Path(sae_config_path).read_text(encoding="utf-8"))
    model, tokenizer, _ = load_local_model_and_tokenizer(str(model_config_path), model_dir=model_dir, device=resolved)
    sae = load_sae_from_hub(repo_id=cfg["sae_repo_id"], subfolder=cfg["sae_subfolder"], device=resolved, dtype=torch.bfloat16, checkpoint_topk_semantics="hard")
    result = extract_and_process_streaming(model=model, tokenizer=tokenizer, sae=sae, texts=texts,
        hook_point=cfg["hook_point"], max_seq_len=int(cfg.get("max_seq_len", 128)), batch_size=batch_size,
        aggregation=str(cfg.get("aggregation", "max")), device=resolved, collect_structural_samples=0)
    features = result["utterance_features"].detach().cpu().float().numpy()
    natural = torch.load(feature_store_path, map_location="cpu").float().numpy()
    stable = pd.read_csv(stable_latents_path)
    stable_ids = sorted(stable[stable["stable_set_role"].astype(str).eq("stable_core")]["latent_idx"].astype(int).unique())
    rows: list[dict[str, Any]] = []
    for idx, block in enumerate(blocks):
        acts = {cell: float(features[4 * idx + pos, int(block["latent_idx"])]) for pos, cell in enumerate(CELLS)}
        surface = ((acts[CELLS[0]] - acts[CELLS[1]]) + (acts[CELLS[2]] - acts[CELLS[3]])) / 2
        semantic = ((acts[CELLS[0]] - acts[CELLS[2]]) + (acts[CELLS[1]] - acts[CELLS[3]])) / 2
        interaction = acts[CELLS[0]] - acts[CELLS[1]] - acts[CELLS[2]] + acts[CELLS[3]]
        latent = int(block["latent_idx"])
        q25, q75 = np.quantile(natural[:, latent], [0.25, 0.75])
        scale = max(float(q75 - q25), 1e-6)
        full_surface = ((features[4*idx] - features[4*idx+1]) + (features[4*idx+2] - features[4*idx+3])) / 2
        rank = float(np.mean(np.abs(full_surface[stable_ids]) <= abs(surface)))
        rows.append({**block, **{f"activation_{k}": v for k, v in acts.items()},
            "surface_effect": surface, "semantic_effect": semantic, "interaction_effect": interaction,
            "surface_effect_iqr": surface / scale, "semantic_effect_iqr": semantic / scale,
            "target_surface_specificity_percentile": rank})
    frame = pd.DataFrame(rows)
    output = Path(output_dir)
    frame.to_csv(output / "activation_results.csv", index=False, encoding="utf-8-sig")
    summary = frame.groupby(["associated_label", "latent_idx"], as_index=False).agg(
        n_blocks=("block_id", "count"), median_surface_effect_iqr=("surface_effect_iqr", "median"),
        median_semantic_effect_iqr=("semantic_effect_iqr", "median"),
        surface_direction_rate=("surface_effect", lambda x: float(np.mean(np.asarray(x) > 0))),
        semantic_direction_rate=("semantic_effect", lambda x: float(np.mean(np.asarray(x) > 0))),
        median_specificity_percentile=("target_surface_specificity_percentile", "median"))
    summary.to_csv(output / "activation_summary_by_latent.csv", index=False, encoding="utf-8-sig")
    manifest = {"step": "activate", "n_blocks": len(frame), "n_texts": len(texts), "device": str(resolved), "hook_point": cfg["hook_point"], "aggregation": cfg.get("aggregation", "max"), "outputs": {"results": str(output / "activation_results.csv"), "summary": str(output / "activation_summary_by_latent.csv")}}
    write_json(output / "activation_manifest.json", manifest)
    return manifest


__all__ = ["DualTrackPairConfig", "build_dual_track_pair_tasks", "validate_phase1_mutations", "build_phase2_d_tasks", "validate_phase2_and_assemble", "validate_dual_track_pair_outputs", "run_dual_track_activation_test"]
