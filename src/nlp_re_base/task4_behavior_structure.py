"""Build an evidence-bounded, bottom-up Task 4 behavior structure draft."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .contrastive_evidence_pack import read_jsonl, write_json, write_jsonl
from .contrastive_llm_io import parse_llm_json_file
from .deepseek_top50_induction import DEEPSEEK_EXECUTION_MODE, _sha256_bytes, _sha256_text


LABEL_ORDER = ("RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF")
LEAF_LABELS = {"RES", "REC", "QUO", "QUC", "GI", "SU", "AF"}
EVIDENCE_CLASSES = {
    "behavioral_function",
    "linguistic_structure",
    "affective_content",
    "topic",
    "surface_artifact",
    "unclear_or_mixed",
}
CARD_CODING_FIELDS = (
    "latent_idx",
    "primary_evidence_class",
    "secondary_evidence_classes",
    "open_behavior_code",
    "behavior_component_name_zh",
    "behavior_component_name_en",
    "behavior_definition",
    "allow_behavior_grouping",
    "supporting_evidence_ids",
    "contradicting_evidence_ids",
    "coding_rationale",
    "confidence",
    "boundary_note",
)

SYSTEM_PROMPT = (
    "You are an evidence-focused linguistic analyst. Independently code supplied feature cards, "
    "separate communicative function from form, affect, topic, and artifacts, and return only JSON."
)
GROUPING_SYSTEM_PROMPT = (
    "You are an evidence-focused qualitative analyst. Form bottom-up behavioral components only "
    "when multiple supplied features support the same communicative function. Return only JSON."
)


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    rows = [str(item).strip() for item in value]
    if any(not item for item in rows) or len(rows) != len(set(rows)):
        raise ValueError(f"{field} contains empty or duplicate values")
    return rows


def _execution_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        str(row["task_id"]): row
        for row in read_jsonl(path)
        if row.get("status") == "success" and row.get("task_id")
    }


def _check_provenance(task: dict[str, Any], execution: dict[str, Any] | None) -> list[str]:
    reasons: list[str] = []
    raw_path = Path(task["expected_output_path"])
    if execution is None:
        reasons.append("missing_successful_execution")
    elif execution.get("execution_mode") != DEEPSEEK_EXECUTION_MODE:
        reasons.append("execution_mode_mismatch")
    else:
        if execution.get("prompt_sha256") != _sha256_text(str(task["prompt"])):
            reasons.append("prompt_hash_mismatch")
        if raw_path.exists() and execution.get("raw_output_sha256") != _sha256_bytes(raw_path.read_bytes()):
            reasons.append("raw_output_hash_mismatch")
    return reasons


def _sample_lookup(pack: dict[str, Any]) -> dict[str, str]:
    return {str(row["id"]): str(row["text"]).strip() for row in pack["samples_for_model"]}


def _cited_evidence(card: dict[str, Any], pack: dict[str, Any]) -> list[dict[str, str]]:
    lookup = _sample_lookup(pack)
    ids: list[str] = []
    for sample_id in card.get("representative_evidence_ids", []):
        if str(sample_id) not in ids:
            ids.append(str(sample_id))
    for item in card.get("linguistic_evidence", []):
        sample_id = str(item.get("sample_id", ""))
        if sample_id and sample_id not in ids:
            ids.append(sample_id)
    for sample_id in card.get("outlier_sample_ids", [])[:3]:
        if str(sample_id) not in ids:
            ids.append(str(sample_id))
    return [{"sample_id": sample_id, "text": lookup.get(sample_id, "")} for sample_id in ids]


def _card_prompt(card: dict[str, Any], evidence: list[dict[str, str]]) -> str:
    visible_ids = [row["sample_id"] for row in evidence]
    source = {key: value for key, value in card.items() if key != "raw_output_path"}
    schema = {
        "latent_idx": int(card["latent_idx"]),
        "primary_evidence_class": "unclear_or_mixed",
        "secondary_evidence_classes": [],
        "open_behavior_code": "",
        "behavior_component_name_zh": "",
        "behavior_component_name_en": "",
        "behavior_definition": "",
        "allow_behavior_grouping": False,
        "supporting_evidence_ids": [],
        "contradicting_evidence_ids": [],
        "coding_rationale": "",
        "confidence": 1,
        "boundary_note": "",
    }
    return f"""Independently recode one previously generated feature card. The source card is evidence, not ground truth.

Do not infer a dataset label. Do not upgrade a question form, repeated phrase, sentiment, or health topic into a behavioral function unless the cited utterances directly support a consistent communicative action. A behavioral grouping candidate must be supported by pragmatic evidence, not merely by the source card's proposed name. If evidence is insufficient, set allow_behavior_grouping=false.

Primary class must be one of: {sorted(EVIDENCE_CLASSES)}.
Use a short bottom-up open_behavior_code only when allow_behavior_grouping=true; do not choose from a predefined taxonomy. Confidence is 1-5. supporting_evidence_ids and contradicting_evidence_ids must come from the visible IDs below. Use 2-3 supporting IDs when grouping is allowed.

Source feature card:
{json.dumps(source, ensure_ascii=False, indent=2)}

Cited utterance texts:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Visible evidence IDs: {json.dumps(visible_ids, ensure_ascii=False)}

Return only this JSON structure:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def build_card_coding_tasks(*, package_dir: Path, output_dir: Path) -> dict[str, Any]:
    frozen = package_dir / "frozen_all_cards"
    cards = read_jsonl(frozen / "validated_cards.jsonl")
    packs = {int(row["latent_idx"]): row for row in read_jsonl(frozen / "sentence_packs.jsonl")}
    task_dir = output_dir / "llm_tasks"
    raw_dir = output_dir / "llm_outputs" / "card_codings" / "raw"
    task_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    for card in sorted(cards, key=lambda row: int(row["latent_idx"])):
        latent_idx = int(card["latent_idx"])
        evidence = _cited_evidence(card, packs[latent_idx])
        task_id = f"task4_card_{latent_idx:05d}"
        tasks.append(
            {
                "task_id": task_id,
                "task_type": "task4_independent_card_open_coding",
                "latent_idx": latent_idx,
                "prompt": _card_prompt(card, evidence),
                "visible_evidence_ids": [row["sample_id"] for row in evidence],
                "expected_output_path": str(raw_dir / f"{task_id}.json"),
            }
        )
    path = task_dir / "card_coding_tasks.jsonl"
    write_jsonl(path, tasks)
    result = {"step": "build_card_coding_tasks", "n_tasks": len(tasks), "tasks": str(path)}
    write_json(output_dir / "card_coding_task_manifest.json", result)
    return result


def validate_card_codings(*, tasks_path: Path, execution_manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    tasks = read_jsonl(tasks_path)
    executions = _execution_rows(execution_manifest_path)
    valid: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for task in tasks:
        reasons = _check_provenance(task, executions.get(str(task["task_id"])))
        payload: dict[str, Any] | None = None
        try:
            payload = parse_llm_json_file(task["expected_output_path"])
            missing = [field for field in CARD_CODING_FIELDS if field not in payload]
            if missing:
                raise ValueError(f"missing_fields={','.join(missing)}")
            if int(payload["latent_idx"]) != int(task["latent_idx"]):
                reasons.append("latent_idx_mismatch")
            primary = str(payload["primary_evidence_class"]).strip()
            if primary not in EVIDENCE_CLASSES:
                reasons.append("invalid_primary_evidence_class")
            secondary = _string_list(payload["secondary_evidence_classes"], "secondary_evidence_classes")
            if any(item not in EVIDENCE_CLASSES or item == primary for item in secondary):
                reasons.append("invalid_secondary_evidence_classes")
            allowed = payload["allow_behavior_grouping"]
            if not isinstance(allowed, bool):
                reasons.append("allow_behavior_grouping_not_boolean")
                allowed = False
            supporting = _string_list(payload["supporting_evidence_ids"], "supporting_evidence_ids")
            contradicting = _string_list(payload["contradicting_evidence_ids"], "contradicting_evidence_ids")
            visible = set(task["visible_evidence_ids"])
            if not set(supporting + contradicting).issubset(visible):
                reasons.append("unknown_evidence_id")
            if set(supporting) & set(contradicting):
                reasons.append("support_contradiction_overlap")
            open_fields = (
                "open_behavior_code",
                "behavior_component_name_zh",
                "behavior_component_name_en",
                "behavior_definition",
            )
            if allowed:
                if primary != "behavioral_function":
                    reasons.append("grouping_allowed_for_nonbehavior_primary")
                if not 2 <= len(supporting) <= 3:
                    reasons.append("invalid_behavior_support_count")
                if any(not str(payload[field]).strip() for field in open_fields):
                    reasons.append("missing_open_behavior_fields")
            confidence = int(payload["confidence"])
            if isinstance(payload["confidence"], bool) or float(payload["confidence"]) != confidence or not 1 <= confidence <= 5:
                reasons.append("invalid_confidence")
            if not str(payload["coding_rationale"]).strip() or not str(payload["boundary_note"]).strip():
                reasons.append("missing_rationale_or_boundary")
            if not reasons:
                valid.append(
                    {
                        "task_id": str(task["task_id"]),
                        "latent_idx": int(payload["latent_idx"]),
                        "primary_evidence_class": primary,
                        "secondary_evidence_classes": secondary,
                        "open_behavior_code": str(payload["open_behavior_code"]).strip(),
                        "behavior_component_name_zh": str(payload["behavior_component_name_zh"]).strip(),
                        "behavior_component_name_en": str(payload["behavior_component_name_en"]).strip(),
                        "behavior_definition": str(payload["behavior_definition"]).strip(),
                        "allow_behavior_grouping": bool(allowed),
                        "supporting_evidence_ids": supporting,
                        "contradicting_evidence_ids": contradicting,
                        "coding_rationale": str(payload["coding_rationale"]).strip(),
                        "confidence": confidence,
                        "boundary_note": str(payload["boundary_note"]).strip(),
                    }
                )
        except Exception as exc:
            reasons.append(f"parse_or_schema_error:{exc}")
        audit.append({"task_id": task["task_id"], "latent_idx": task["latent_idx"], "valid": not reasons, "reasons": "|".join(reasons)})
    validated_path = output_dir / "llm_outputs" / "card_codings" / "validated_card_codings.jsonl"
    audit_path = output_dir / "llm_outputs" / "card_codings" / "validation_audit.csv"
    write_jsonl(validated_path, valid)
    pd.DataFrame(audit).to_csv(audit_path, index=False, encoding="utf-8-sig")
    result = {"step": "validate_card_codings", "n_tasks": len(tasks), "n_valid": len(valid), "n_invalid": len(tasks) - len(valid), "validated": str(validated_path), "audit": str(audit_path)}
    write_json(output_dir / "card_coding_validation_manifest.json", result)
    return result


def _grouping_prompt(label: str, candidates: list[dict[str, Any]]) -> str:
    schema = {
        "label": label,
        "components": [
            {
                "component_key": "",
                "name_zh": "",
                "name_en": "",
                "definition": "",
                "member_latent_ids": [],
                "representative_evidence": [{"latent_idx": 0, "sample_id": ""}],
                "confidence": 1,
                "boundary_note": "",
            }
        ],
        "unassigned_behavior_latent_ids": [],
    }
    return f"""Create a bottom-up candidate behavior structure for label {label} from independently coded feature cards.

Do not use a predefined component taxonomy. Group features only when their utterance evidence supports the same communicative action. Shared words, sentence form, topic, sentiment, or the common label are insufficient. Every component must contain at least two distinct latents. Keep singletons and incompatible candidates unassigned. Every supplied latent must appear exactly once, either in one component or in unassigned_behavior_latent_ids.

Candidate feature codings:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Return only this JSON structure:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def build_label_grouping_tasks(*, package_dir: Path, output_dir: Path) -> dict[str, Any]:
    frozen = package_dir / "frozen_all_cards"
    stable = pd.read_csv(frozen / "stable_topk_latent_set.csv")
    stable = stable[stable["stable_set_role"].astype(str).eq("stable_core")].copy()
    stable["label"] = stable["label"].astype(str).str.upper()
    codings = {int(row["latent_idx"]): row for row in read_jsonl(output_dir / "llm_outputs" / "card_codings" / "validated_card_codings.jsonl")}
    packs = {int(row["latent_idx"]): row for row in read_jsonl(frozen / "sentence_packs.jsonl")}
    raw_dir = output_dir / "llm_outputs" / "label_groupings" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    for label in LABEL_ORDER:
        latent_ids = stable.loc[stable["label"].eq(label), "latent_idx"].astype(int).tolist()
        candidates: list[dict[str, Any]] = []
        evidence_ids: dict[str, list[str]] = {}
        for latent_idx in latent_ids:
            coding = codings[latent_idx]
            if not coding["allow_behavior_grouping"]:
                continue
            lookup = _sample_lookup(packs[latent_idx])
            ids = list(coding["supporting_evidence_ids"])
            evidence_ids[str(latent_idx)] = ids
            candidates.append(
                {
                    "latent_idx": latent_idx,
                    "open_behavior_code": coding["open_behavior_code"],
                    "candidate_name_zh": coding["behavior_component_name_zh"],
                    "candidate_name_en": coding["behavior_component_name_en"],
                    "definition": coding["behavior_definition"],
                    "coding_rationale": coding["coding_rationale"],
                    "boundary_note": coding["boundary_note"],
                    "confidence": coding["confidence"],
                    "supporting_evidence": [{"sample_id": sid, "text": lookup.get(sid, "")} for sid in ids],
                }
            )
        task_id = f"task4_group_{label.lower()}"
        tasks.append(
            {
                "task_id": task_id,
                "task_type": "task4_bottom_up_label_grouping",
                "label": label,
                "eligible_latent_ids": [row["latent_idx"] for row in candidates],
                "evidence_ids_by_latent": evidence_ids,
                "prompt": _grouping_prompt(label, candidates),
                "expected_output_path": str(raw_dir / f"{task_id}.json"),
            }
        )
    path = output_dir / "llm_tasks" / "label_grouping_tasks.jsonl"
    write_jsonl(path, tasks)
    result = {"step": "build_label_grouping_tasks", "n_tasks": len(tasks), "tasks": str(path), "n_eligible_associations": sum(len(row["eligible_latent_ids"]) for row in tasks)}
    write_json(output_dir / "label_grouping_task_manifest.json", result)
    return result


def validate_label_groupings(*, tasks_path: Path, execution_manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    tasks = read_jsonl(tasks_path)
    executions = _execution_rows(execution_manifest_path)
    valid: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for task in tasks:
        reasons = _check_provenance(task, executions.get(str(task["task_id"])))
        try:
            payload = parse_llm_json_file(task["expected_output_path"])
            if str(payload.get("label", "")).upper() != str(task["label"]):
                reasons.append("label_mismatch")
            components = payload.get("components")
            if not isinstance(components, list):
                raise ValueError("components must be a list")
            unassigned = [int(value) for value in payload.get("unassigned_behavior_latent_ids", [])]
            seen: list[int] = []
            normalized_components: list[dict[str, Any]] = []
            component_keys: set[str] = set()
            evidence_ids = task["evidence_ids_by_latent"]
            for component in components:
                key = str(component.get("component_key", "")).strip()
                members = [int(value) for value in component.get("member_latent_ids", [])]
                if not key or key in component_keys:
                    reasons.append("empty_or_duplicate_component_key")
                component_keys.add(key)
                if len(members) < 2 or len(members) != len(set(members)):
                    reasons.append(f"invalid_component_members:{key}")
                representatives = component.get("representative_evidence", [])
                if not isinstance(representatives, list) or not 2 <= len(representatives) <= 4:
                    reasons.append(f"invalid_representative_count:{key}")
                    representatives = []
                for item in representatives:
                    latent_idx = int(item.get("latent_idx", -1))
                    sample_id = str(item.get("sample_id", ""))
                    if latent_idx not in members or sample_id not in evidence_ids.get(str(latent_idx), []):
                        reasons.append(f"invalid_representative_reference:{key}")
                confidence = int(component.get("confidence", 0))
                if not 1 <= confidence <= 5:
                    reasons.append(f"invalid_component_confidence:{key}")
                for field in ("name_zh", "name_en", "definition", "boundary_note"):
                    if not str(component.get(field, "")).strip():
                        reasons.append(f"missing_{field}:{key}")
                seen.extend(members)
                normalized_components.append(
                    {
                        "component_key": key,
                        "name_zh": str(component.get("name_zh", "")).strip(),
                        "name_en": str(component.get("name_en", "")).strip(),
                        "definition": str(component.get("definition", "")).strip(),
                        "member_latent_ids": members,
                        "representative_evidence": representatives,
                        "confidence": confidence,
                        "boundary_note": str(component.get("boundary_note", "")).strip(),
                    }
                )
            seen.extend(unassigned)
            eligible = [int(value) for value in task["eligible_latent_ids"]]
            if len(seen) != len(set(seen)) or set(seen) != set(eligible):
                reasons.append("eligible_latents_not_exactly_partitioned")
            if not reasons:
                valid.append({"task_id": task["task_id"], "label": task["label"], "components": normalized_components, "unassigned_behavior_latent_ids": unassigned})
        except Exception as exc:
            reasons.append(f"parse_or_schema_error:{exc}")
        audit.append({"task_id": task["task_id"], "label": task["label"], "valid": not reasons, "reasons": "|".join(reasons)})
    validated_path = output_dir / "llm_outputs" / "label_groupings" / "validated_label_groupings.jsonl"
    audit_path = output_dir / "llm_outputs" / "label_groupings" / "validation_audit.csv"
    write_jsonl(validated_path, valid)
    pd.DataFrame(audit).to_csv(audit_path, index=False, encoding="utf-8-sig")
    result = {"step": "validate_label_groupings", "n_tasks": len(tasks), "n_valid": len(valid), "n_invalid": len(tasks) - len(valid), "validated": str(validated_path), "audit": str(audit_path)}
    write_json(output_dir / "label_grouping_validation_manifest.json", result)
    return result


def _scope(labels: set[str]) -> str:
    if labels.issubset({"RE", "RES", "REC"}) or labels.issubset({"QU", "QUO", "QUC"}):
        return "sibling_leaf" if labels.issubset(LEAF_LABELS) else "parent_child"
    return "cross_family"


def build_cross_label_task(*, output_dir: Path) -> dict[str, Any]:
    groupings = read_jsonl(output_dir / "llm_outputs" / "label_groupings" / "validated_label_groupings.jsonl")
    components = []
    for row in groupings:
        for component in row["components"]:
            components.append({"component_ref": f"{row['label']}:{component['component_key']}", "label": row["label"], **component})
    schema = {
        "shared_component_families": [{"family_key": "", "name_zh": "", "definition": "", "component_refs": [], "shared_basis": "", "label_specific_differences": "", "confidence": 1}],
        "label_specific_component_refs": [],
    }
    prompt = f"""Compare the bottom-up behavior components across labels. Group components only when their definitions and evidence support the same communicative function. Similar wording, topic, sentiment, or question form is insufficient. Every component_ref must appear exactly once, either in one shared family containing at least two labels or in label_specific_component_refs.

Components:
{json.dumps(components, ensure_ascii=False, indent=2)}

Return only this JSON structure:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""
    task_id = "task4_cross_label_components"
    raw_dir = output_dir / "llm_outputs" / "cross_label" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    task = {"task_id": task_id, "task_type": "task4_cross_label_component_comparison", "component_refs": [row["component_ref"] for row in components], "component_labels": {row["component_ref"]: row["label"] for row in components}, "prompt": prompt, "expected_output_path": str(raw_dir / f"{task_id}.json")}
    path = output_dir / "llm_tasks" / "cross_label_task.jsonl"
    write_jsonl(path, [task])
    result = {"step": "build_cross_label_task", "n_components": len(components), "tasks": str(path)}
    write_json(output_dir / "cross_label_task_manifest.json", result)
    return result


def validate_cross_label(*, tasks_path: Path, execution_manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    task = read_jsonl(tasks_path)[0]
    execution = _execution_rows(execution_manifest_path).get(str(task["task_id"]))
    reasons = _check_provenance(task, execution)
    normalized: dict[str, Any] = {}
    try:
        payload = parse_llm_json_file(task["expected_output_path"])
        families = payload.get("shared_component_families")
        specifics = _string_list(payload.get("label_specific_component_refs"), "label_specific_component_refs")
        if not isinstance(families, list):
            raise ValueError("shared_component_families must be a list")
        seen: list[str] = list(specifics)
        normalized_families = []
        keys: set[str] = set()
        for family in families:
            key = str(family.get("family_key", "")).strip()
            refs = _string_list(family.get("component_refs"), "component_refs")
            labels = {task["component_labels"].get(ref, "") for ref in refs}
            if not key or key in keys:
                reasons.append("empty_or_duplicate_family_key")
            keys.add(key)
            if len(refs) < 2 or len(labels) < 2:
                reasons.append(f"family_not_cross_label:{key}")
            confidence = int(family.get("confidence", 0))
            if not 1 <= confidence <= 5:
                reasons.append(f"invalid_family_confidence:{key}")
            for field in ("name_zh", "definition", "shared_basis", "label_specific_differences"):
                if not str(family.get(field, "")).strip():
                    reasons.append(f"missing_{field}:{key}")
            seen.extend(refs)
            normalized_families.append({**family, "family_key": key, "component_refs": refs, "confidence": confidence})
        expected = set(task["component_refs"])
        if len(seen) != len(set(seen)) or set(seen) != expected:
            reasons.append("components_not_exactly_partitioned")
        normalized = {"shared_component_families": normalized_families, "label_specific_component_refs": specifics}
    except Exception as exc:
        reasons.append(f"parse_or_schema_error:{exc}")
    validated_path = output_dir / "llm_outputs" / "cross_label" / "validated_cross_label.json"
    if not reasons:
        write_json(validated_path, normalized)
    result = {"step": "validate_cross_label", "valid": not reasons, "reasons": reasons, "validated": str(validated_path)}
    write_json(output_dir / "cross_label_validation_manifest.json", result)
    return result


def build_report(*, package_dir: Path, output_dir: Path) -> dict[str, Any]:
    frozen = package_dir / "frozen_all_cards"
    cards = {int(row["latent_idx"]): row for row in read_jsonl(frozen / "validated_cards.jsonl")}
    packs = {int(row["latent_idx"]): row for row in read_jsonl(frozen / "sentence_packs.jsonl")}
    stable = pd.read_csv(frozen / "stable_topk_latent_set.csv")
    stable = stable[stable["stable_set_role"].astype(str).eq("stable_core")].copy()
    stable["label"] = stable["label"].astype(str).str.upper()
    stable["latent_idx"] = stable["latent_idx"].astype(int)
    stable["label_order"] = stable["label"].map({label: index for index, label in enumerate(LABEL_ORDER)})
    stable = stable.sort_values(["label_order", "rank_within_label", "latent_idx"])
    codings = {int(row["latent_idx"]): row for row in read_jsonl(output_dir / "llm_outputs" / "card_codings" / "validated_card_codings.jsonl")}
    groupings = {str(row["label"]): row for row in read_jsonl(output_dir / "llm_outputs" / "label_groupings" / "validated_label_groupings.jsonl")}
    cross = json.loads((output_dir / "llm_outputs" / "cross_label" / "validated_cross_label.json").read_text(encoding="utf-8"))

    feature_rows = []
    for row in stable.itertuples(index=False):
        latent_idx = int(row.latent_idx)
        coding = codings[latent_idx]
        feature_rows.append({"item_id": f"{row.label}_{latent_idx}", "label": row.label, "is_leaf_label": row.label in LEAF_LABELS, "latent_idx": latent_idx, "rank_within_label": int(row.rank_within_label), "inclusion_frequency": float(row.inclusion_frequency), "abs_cohens_d": float(row.abs_cohens_d), "source_card_type": cards[latent_idx]["explanation_type"], **{key: ("|".join(value) if isinstance(value, list) else value) for key, value in coding.items() if key != "task_id"}, "analysis_status": "automated_open_coding_not_human_validated"})
    feature_df = pd.DataFrame(feature_rows)

    component_rows = []
    singleton_rows = []
    for label in LABEL_ORDER:
        grouping = groupings[label]
        for component in grouping["components"]:
            evidence_rows = []
            for item in component["representative_evidence"]:
                latent_idx = int(item["latent_idx"])
                sample_id = str(item["sample_id"])
                evidence_rows.append(f"F{latent_idx}/{sample_id}: {_sample_lookup(packs[latent_idx]).get(sample_id, '')}")
            component_rows.append({"label": label, "is_leaf_label": label in LEAF_LABELS, "component_ref": f"{label}:{component['component_key']}", "behavior_component": component["name_zh"], "behavior_component_en": component["name_en"], "definition": component["definition"], "supporting_feature_count": len(component["member_latent_ids"]), "supporting_sae_features": "|".join(f"F{idx}" for idx in component["member_latent_ids"]), "representative_evidence": " || ".join(evidence_rows), "confidence": component["confidence"], "boundary_note": component["boundary_note"], "status": "automated_bottom_up_candidate"})
        for latent_idx in grouping["unassigned_behavior_latent_ids"]:
            coding = codings[int(latent_idx)]
            singleton_rows.append({"label": label, "latent_idx": int(latent_idx), "sae_feature": f"F{latent_idx}", "open_behavior_code": coding["open_behavior_code"], "candidate_name_zh": coding["behavior_component_name_zh"], "definition": coding["behavior_definition"], "confidence": coding["confidence"], "boundary_note": coding["boundary_note"], "status": "unassigned_singleton_or_incompatible_behavior_candidate"})
    components_df = pd.DataFrame(component_rows)
    singleton_df = pd.DataFrame(singleton_rows)
    nonbehavior_df = feature_df[~feature_df["allow_behavior_grouping"].astype(bool)].copy()

    exact_rows = []
    for latent_idx, group in feature_df.groupby("latent_idx", sort=True):
        labels = set(group["label"].astype(str))
        if len(labels) < 2:
            continue
        leaf = labels & LEAF_LABELS
        exact_rows.append({"latent_idx": int(latent_idx), "all_labels": "|".join(label for label in LABEL_ORDER if label in labels), "leaf_labels": "|".join(label for label in LABEL_ORDER if label in leaf), "sharing_scope_all_labels": _scope(labels), "sharing_scope_leaf_labels": _scope(leaf) if len(leaf) >= 2 else "no_multi_leaf_overlap", "primary_evidence_class": group.iloc[0]["primary_evidence_class"], "open_behavior_code": group.iloc[0]["open_behavior_code"], "allow_behavior_grouping": bool(group.iloc[0]["allow_behavior_grouping"]), "confidence": int(group.iloc[0]["confidence"])})
    exact_df = pd.DataFrame(exact_rows)

    cross_rows = []
    component_lookup = components_df.set_index("component_ref").to_dict(orient="index") if not components_df.empty else {}
    for family in cross["shared_component_families"]:
        labels = {ref.split(":", 1)[0] for ref in family["component_refs"]}
        cross_rows.append({"shared_family": family["name_zh"], "family_key": family["family_key"], "component_refs": "|".join(family["component_refs"]), "labels": "|".join(label for label in LABEL_ORDER if label in labels), "sharing_scope": _scope(labels), "shared_basis": family["shared_basis"], "label_specific_differences": family["label_specific_differences"], "confidence": family["confidence"], "supporting_sae_features_by_component": " || ".join(f"{ref}:{component_lookup.get(ref, {}).get('supporting_sae_features', '')}" for ref in family["component_refs"]), "status": "automated_cross_label_candidate"})
    cross_df = pd.DataFrame(cross_rows)

    summary_rows = []
    for label in LABEL_ORDER:
        subset = feature_df[feature_df["label"].eq(label)]
        summary_rows.append({"label": label, "n_feature_cards": len(subset), "n_behavior_grouping_eligible": int(subset["allow_behavior_grouping"].astype(bool).sum()), **{f"n_{kind}": int(subset["primary_evidence_class"].eq(kind).sum()) for kind in sorted(EVIDENCE_CLASSES)}, "n_behavior_components": int(components_df["label"].eq(label).sum()), "n_unassigned_behavior_candidates": int(singleton_df["label"].eq(label).sum())})
    summary_df = pd.DataFrame(summary_rows)

    paths = {
        "feature_codings": output_dir / "task4_feature_codings.csv",
        "behavior_components": output_dir / "task4_behavior_components.csv",
        "behavior_singletons": output_dir / "task4_behavior_singletons_unresolved.csv",
        "nonbehavior_features": output_dir / "task4_nonbehavior_features.csv",
        "cross_label_components": output_dir / "task4_cross_label_components.csv",
        "exact_shared_latents": output_dir / "task4_exact_shared_latents.csv",
        "label_summary": output_dir / "task4_label_summary.csv",
    }
    for frame, key in ((feature_df, "feature_codings"), (components_df, "behavior_components"), (singleton_df, "behavior_singletons"), (nonbehavior_df, "nonbehavior_features"), (cross_df, "cross_label_components"), (exact_df, "exact_shared_latents"), (summary_df, "label_summary")):
        frame.to_csv(paths[key], index=False, encoding="utf-8-sig")

    lines = ["# Task 4：行为表征结构（自动候选稿）", "", "## 范围与边界", "", f"本次覆盖冻结包中的 {len(cards)} 张唯一 feature card 和 {len(feature_df)} 条 stable-core 标签–latent 关联。逐卡编码为标签盲的开放编码；标签内成分随后自下而上归组，没有使用固定成分词典。", "", "当前结果尚未经过人工双审查或逐成分人工质量审核，因此只能作为 `automated_bottom_up_candidate`。LLM 自报支持比例不作为独立证据。`RE/RES/REC` 缺少 client 前文，涉及反映功能的名称均为候选解释。", "", "## 标签概览", "", "| 标签 | 卡片 | 可归组行为候选 | 行为成分 | 未归组行为候选 |", "|---|---:|---:|---:|---:|"]
    for row in summary_df.itertuples(index=False):
        lines.append(f"| {row.label} | {row.n_feature_cards} | {row.n_behavior_grouping_eligible} | {row.n_behavior_components} | {row.n_unassigned_behavior_candidates} |")
    lines.extend(["", "## 标签内行为成分", ""])
    for label in LABEL_ORDER:
        lines.extend([f"### {label}", "", "| 行为成分 | 支持特征 | 代表性证据 | 置信度 |", "|---|---|---|---:|"])
        subset = components_df[components_df["label"].eq(label)]
        if subset.empty:
            lines.append("| 无满足双特征门槛的自动候选成分 | - | - | - |")
        for row in subset.itertuples(index=False):
            lines.append(f"| {row.behavior_component} | `{row.supporting_sae_features}` | {str(row.representative_evidence).replace('|', '/')} | {row.confidence} |")
        lines.append("")
    lines.extend(["## 非行为证据构成", ""])
    for label in LABEL_ORDER:
        subset = feature_df[(feature_df["label"].eq(label)) & (~feature_df["allow_behavior_grouping"].astype(bool))]
        counts = subset["primary_evidence_class"].value_counts().to_dict()
        lines.append(f"- `{label}`：" + "；".join(f"{key}={value}" for key, value in sorted(counts.items())) + "。")
    lines.extend(["", "## 跨标签候选", "", "| 共享成分 | 标签 | 范围 | 证据基础 | 差异 |", "|---|---|---|---|---|"])
    for row in cross_df.itertuples(index=False):
        lines.append(f"| {row.shared_family} | `{row.labels}` | `{row.sharing_scope}` | {row.shared_basis} | {row.label_specific_differences} |")
    lines.extend(["", "精确共享 latent 另见 `task4_exact_shared_latents.csv`；未进入共享家族的标签特异候选保留在自动跨标签 JSON 中。人工审核与一致率计算本轮未执行。", ""])
    report_path = output_dir / "task4_behavior_representation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {"analysis": "task4_behavior_representation_structure", "status": "automated_draft_human_review_not_run", "inputs": {"package": str(package_dir)}, "outputs": {**{key: str(value) for key, value in paths.items()}, "report": str(report_path)}, "counts": {"unique_cards": len(cards), "label_latent_associations": len(feature_df), "labels": len(LABEL_ORDER), "behavior_components": len(components_df), "unassigned_behavior_candidates": len(singleton_df), "cross_label_families": len(cross_df), "exact_shared_latents": len(exact_df)}, "human_review": {"performed": False, "agreement": None, "quality_audit": "deferred_by_user"}}
    write_json(output_dir / "manifest.json", manifest)
    return manifest


__all__ = [
    "CARD_CODING_FIELDS",
    "EVIDENCE_CLASSES",
    "GROUPING_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "build_card_coding_tasks",
    "build_cross_label_task",
    "build_label_grouping_tasks",
    "build_report",
    "validate_card_codings",
    "validate_cross_label",
    "validate_label_groupings",
]
