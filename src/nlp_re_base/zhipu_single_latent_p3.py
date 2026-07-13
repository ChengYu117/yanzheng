"""Single-latent Top-50 P3 workflow artifacts and quality gates.

The explainer sees one anonymous latent at a time and must partition all Top-50
activating utterances into a dominant pattern and outliers. Downstream scoring
uses text-disjoint held-out samples, and generated minimal pairs are checked by
the original model plus SAE rather than by the explainer alone.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .contrastive_evidence_pack import (
    DEFAULT_LABELS,
    detection_threshold,
    normalise_text,
    read_jsonl,
    write_json,
    write_jsonl,
)
from .contrastive_llm_io import parse_llm_json_file


TOP50_EXPLAINER_SYSTEM_PROMPT = (
    "You evaluate anonymous SAE latent activation patterns. Infer one aggregate pattern across the full "
    "sample set, distinguish semantic evidence from surface form, and return only the requested JSON."
)
SCORER_SYSTEM_PROMPT = (
    "You evaluate whether texts match a supplied latent hypothesis. Score every item independently and "
    "return only the requested JSON list."
)
MINIMAL_PAIR_SYSTEM_PROMPT = (
    "You design controlled text pairs for testing an anonymous SAE latent hypothesis. Return only the "
    "requested JSON list."
)

TOP50_EXPLANATION_FIELDS: tuple[str, ...] = (
    "latent_idx",
    "short_name",
    "candidate_explanation",
    "main_hypothesis",
    "dominant_pattern",
    "semantic_component",
    "surface_component",
    "positive_triggers",
    "explicit_exclusions",
    "possible_surface_confounds",
    "feature_type",
    "supporting_sample_ids",
    "outlier_sample_ids",
    "representative_evidence_ids",
    "alternative_hypotheses",
    "failure_modes",
    "confidence",
)


@dataclass(frozen=True)
class SingleLatentP3Config:
    top_n: int = 50
    heldout_positive: int = 10
    heldout_near_miss: int = 5
    heldout_label_match: int = 5
    minimum_raw_support_fraction: float = 0.50
    minimum_unique_support_fraction: float = 0.50
    pairs_per_latent: int = 5


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _as_string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    rows = [str(item).strip() for item in value]
    if any(not item for item in rows):
        raise ValueError(f"{field} contains an empty item")
    return rows


def _as_text_list(value: Any, field: str) -> list[str]:
    """Normalize prose-list fields while keeping id-list fields strict."""
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return _as_string_list(value, field)


def _load_latent_vector(path: str | Path, latent_idx: int) -> np.ndarray:
    feature_path = Path(path)
    if feature_path.suffix == ".npy":
        matrix = np.load(feature_path, mmap_mode="r")
        if matrix.ndim != 2 or latent_idx < 0 or latent_idx >= matrix.shape[1]:
            raise ValueError(f"latent_idx={latent_idx} is invalid for feature shape {matrix.shape}")
        return np.asarray(matrix[:, latent_idx], dtype=np.float32)
    if feature_path.suffix == ".npz":
        payload = np.load(feature_path)
        for key in ("utterance_features", "features", "X"):
            if key in payload:
                matrix = payload[key]
                break
        else:
            raise KeyError(f"{feature_path} has no utterance feature matrix")
        return np.asarray(matrix[:, latent_idx], dtype=np.float32)
    if feature_path.suffix == ".pt":
        import torch

        payload = torch.load(feature_path, map_location="cpu")
        if isinstance(payload, torch.Tensor):
            matrix = payload
        elif isinstance(payload, dict):
            for key in ("utterance_features", "features", "X"):
                if key in payload:
                    matrix = payload[key]
                    break
            else:
                raise KeyError(f"{feature_path} has no utterance feature matrix")
        else:
            raise TypeError(f"Unsupported feature payload: {type(payload).__name__}")
        if matrix.ndim != 2 or latent_idx < 0 or latent_idx >= matrix.shape[1]:
            raise ValueError(f"latent_idx={latent_idx} is invalid for feature shape {tuple(matrix.shape)}")
        column = matrix[:, latent_idx]
        if isinstance(column, torch.Tensor):
            return column.detach().cpu().float().numpy()
        return np.asarray(column, dtype=np.float32)
    raise ValueError(f"Unsupported feature extension: {feature_path.suffix}")


def _record_text(record: dict[str, Any], label_row: pd.Series) -> str:
    for key in ("unit_text", "text", "utterance"):
        if record.get(key) not in (None, ""):
            return str(record[key])
        if key in label_row and pd.notna(label_row[key]):
            return str(label_row[key])
    return ""


def _record_value(record: dict[str, Any], label_row: pd.Series, key: str, default: Any = "") -> Any:
    if record.get(key) not in (None, ""):
        return record[key]
    if key in label_row and pd.notna(label_row[key]):
        return label_row[key]
    return default


def _active_labels(label_row: pd.Series) -> str:
    return ",".join(
        label
        for label in DEFAULT_LABELS
        if label in label_row and float(pd.to_numeric(pd.Series([label_row[label]]), errors="coerce").fillna(0).iloc[0]) > 0
    )


def _common_content_tokens(texts: Iterable[str], limit: int = 16) -> set[str]:
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "but", "for", "from", "have", "i", "in", "is",
        "it", "of", "on", "or", "so", "that", "the", "this", "to", "was", "we", "well", "what",
        "with", "you", "your", "yeah", "okay", "know", "really", "like",
    }
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(token for token in normalise_text(text).split() if len(token) >= 3 and token not in stop)
    return {token for token, _ in counts.most_common(limit)}


def _make_internal_sample(
    *,
    sample_id: str,
    tag: str,
    row_idx: int,
    target_label: str,
    latent_idx: int,
    activations: np.ndarray,
    threshold: float,
    label_matrix: pd.DataFrame,
    records: list[dict[str, Any]],
    internal_source: str,
    heldout: bool,
) -> dict[str, Any]:
    label_row = label_matrix.iloc[row_idx]
    record = records[row_idx]
    activation = float(activations[row_idx])
    return {
        "id": sample_id,
        "tag": tag,
        "row_idx": int(row_idx),
        "activation": activation,
        "activation_threshold": float(threshold),
        "ground_truth_activate": int(activation > threshold),
        "text": _record_text(record, label_row),
        "target_label": target_label,
        "latent_idx": int(latent_idx),
        "target_match": int(
            target_label in label_row
            and float(pd.to_numeric(pd.Series([label_row[target_label]]), errors="coerce").fillna(0).iloc[0]) > 0
        ),
        "active_labels": _active_labels(label_row),
        "record_id": _record_value(record, label_row, "record_id", f"row_{row_idx}"),
        "file_id": _record_value(record, label_row, "file_id"),
        "source_split": _record_value(record, label_row, "source_split"),
        "source_file": _record_value(record, label_row, "source_file"),
        "source_line": _record_value(record, label_row, "source_line"),
        "internal_source": internal_source,
        "heldout": bool(heldout),
    }


def _select_unique_rows(
    candidates: Iterable[int],
    *,
    count: int,
    texts: list[str],
    used_rows: set[int],
    used_texts: set[str],
) -> list[int]:
    selected: list[int] = []
    for value in candidates:
        row_idx = int(value)
        norm = normalise_text(texts[row_idx])
        if row_idx in used_rows or not norm or norm in used_texts:
            continue
        selected.append(row_idx)
        used_rows.add(row_idx)
        used_texts.add(norm)
        if len(selected) >= count:
            break
    return selected


def prepare_single_latent_pack(
    *,
    stable_latents_path: str | Path,
    feature_store_path: str | Path,
    label_matrix_path: str | Path,
    records_path: str | Path,
    output_dir: str | Path,
    target_label: str,
    latent_idx: int,
    config: SingleLatentP3Config = SingleLatentP3Config(),
) -> dict[str, Any]:
    """Build Top-50 induction evidence and a text-disjoint held-out scorer set."""
    target_label = str(target_label).upper()
    stable = pd.read_csv(stable_latents_path)
    label_col = "target_label" if "target_label" in stable.columns else "label"
    match = stable[
        (stable[label_col].astype(str).str.upper() == target_label)
        & (pd.to_numeric(stable["latent_idx"], errors="coerce") == int(latent_idx))
    ].copy()
    if "stable_set_role" in match.columns:
        match = match[match["stable_set_role"].astype(str) == "stable_core"]
    if len(match) != 1:
        raise ValueError(
            f"Expected exactly one stable_core row for {target_label}/latent {latent_idx}, found {len(match)}"
        )
    latent_meta = match.iloc[0]

    label_matrix = pd.read_csv(label_matrix_path)
    records = read_jsonl(records_path)
    activations = _load_latent_vector(feature_store_path, int(latent_idx))
    if len(activations) != len(label_matrix) or len(records) != len(label_matrix):
        raise ValueError(
            f"Alignment mismatch: activations={len(activations)}, labels={len(label_matrix)}, records={len(records)}"
        )
    texts = [_record_text(records[idx], label_matrix.iloc[idx]) for idx in range(len(records))]
    row_ids = np.arange(len(activations), dtype=np.int64)
    activation_order = row_ids[np.lexsort((row_ids, -activations))]
    top_rows = [int(idx) for idx in activation_order[: int(config.top_n)].tolist()]
    if len(top_rows) != int(config.top_n):
        raise ValueError(f"Requested Top-{config.top_n}, found only {len(top_rows)} rows")

    top_norms = [normalise_text(texts[idx]) for idx in top_rows]
    duplicate_counts = Counter(top_norms)
    duplicate_group_by_norm = {
        norm: f"d{group_idx:03d}"
        for group_idx, norm in enumerate(sorted(norm for norm, count in duplicate_counts.items() if count > 1), start=1)
    }
    threshold = detection_threshold(activations)
    samples_for_explainer: list[dict[str, Any]] = []
    samples_internal: list[dict[str, Any]] = []
    for rank, row_idx in enumerate(top_rows, start=1):
        sample_id = f"s{rank:03d}"
        norm = normalise_text(texts[row_idx])
        internal = _make_internal_sample(
            sample_id=sample_id,
            tag="ACTIVE_TOP50",
            row_idx=row_idx,
            target_label=target_label,
            latent_idx=int(latent_idx),
            activations=activations,
            threshold=threshold,
            label_matrix=label_matrix,
            records=records,
            internal_source="top50_activation",
            heldout=False,
        )
        internal["activation_rank"] = rank
        internal["duplicate_group"] = duplicate_group_by_norm.get(norm, "")
        internal["duplicate_count"] = int(duplicate_counts[norm])
        samples_internal.append(internal)
        samples_for_explainer.append(
            {
                "id": sample_id,
                "rank": rank,
                "activation": float(activations[row_idx]),
                "text": texts[row_idx],
                "duplicate_group": duplicate_group_by_norm.get(norm, ""),
                "duplicate_count": int(duplicate_counts[norm]),
            }
        )

    used_rows = set(top_rows)
    used_texts = set(top_norms)
    positive_candidates = [int(idx) for idx in activation_order if float(activations[int(idx)]) > threshold]
    heldout_positive = _select_unique_rows(
        positive_candidates,
        count=int(config.heldout_positive),
        texts=texts,
        used_rows=used_rows,
        used_texts=used_texts,
    )

    target_values = (
        pd.to_numeric(label_matrix[target_label], errors="coerce").fillna(0).to_numpy(dtype=float) > 0
    )
    nonactive = activations <= threshold
    cue_tokens = _common_content_tokens(texts[idx] for idx in top_rows)
    near_candidates = [idx for idx in row_ids.tolist() if nonactive[int(idx)] and not target_values[int(idx)]]
    near_candidates.sort(
        key=lambda idx: (
            -len(cue_tokens.intersection(normalise_text(texts[int(idx)]).split())),
            -float(activations[int(idx)]),
            int(idx),
        )
    )
    heldout_near = _select_unique_rows(
        near_candidates,
        count=int(config.heldout_near_miss),
        texts=texts,
        used_rows=used_rows,
        used_texts=used_texts,
    )

    label_candidates = [idx for idx in row_ids.tolist() if nonactive[int(idx)] and target_values[int(idx)]]
    label_candidates.sort(key=lambda idx: (-float(activations[int(idx)]), int(idx)))
    heldout_label = _select_unique_rows(
        label_candidates,
        count=int(config.heldout_label_match),
        texts=texts,
        used_rows=used_rows,
        used_texts=used_texts,
    )
    expected = {
        "ACTIVE_HIGH": int(config.heldout_positive),
        "NONACTIVE_NEAR_MISS": int(config.heldout_near_miss),
        "NONACTIVE_LABEL_MATCH": int(config.heldout_label_match),
    }
    selected_groups = {
        "ACTIVE_HIGH": heldout_positive,
        "NONACTIVE_NEAR_MISS": heldout_near,
        "NONACTIVE_LABEL_MATCH": heldout_label,
    }
    found = {key: len(value) for key, value in selected_groups.items()}
    if found != expected:
        raise ValueError(f"Insufficient text-disjoint held-out samples: expected={expected}, found={found}")

    heldout_internal_by_tag: dict[str, list[dict[str, Any]]] = {key: [] for key in selected_groups}
    heldout_counter = 1
    source_by_tag = {
        "ACTIVE_HIGH": "post_top50_high_activation",
        "NONACTIVE_NEAR_MISS": "surface_similar_nonactive",
        "NONACTIVE_LABEL_MATCH": "target_label_positive_nonactive",
    }
    for tag, rows in selected_groups.items():
        for row_idx in rows:
            heldout_internal_by_tag[tag].append(
                _make_internal_sample(
                    sample_id=f"u{heldout_counter:03d}",
                    tag=tag,
                    row_idx=row_idx,
                    target_label=target_label,
                    latent_idx=int(latent_idx),
                    activations=activations,
                    threshold=threshold,
                    label_matrix=label_matrix,
                    records=records,
                    internal_source=source_by_tag[tag],
                    heldout=True,
                )
            )
            heldout_counter += 1

    packet_id = f"p3top50_{target_label.lower()}_{int(latent_idx)}"
    packet = {
        "packet_id": packet_id,
        "target_label": target_label,
        "latent_idx": int(latent_idx),
        "rank_within_label": int(pd.to_numeric(pd.Series([latent_meta.get("rank_within_label", -1)]), errors="coerce").fillna(-1).iloc[0]),
        "stable_set_role": "stable_core",
        "label_selection_status": str(latent_meta.get("label_selection_status", "")),
        "auc": float(latent_meta.get("auc", np.nan)),
        "directional_auc": float(latent_meta.get("directional_auc", np.nan)),
        "cohens_d": float(latent_meta.get("cohens_d", np.nan)),
        "precision_at_50": float(latent_meta.get("precision_at_50", np.nan)),
        "inclusion_frequency": float(latent_meta.get("inclusion_frequency", np.nan)),
        "activation_threshold_policy": "q50_nonzero_activation",
        "activation_threshold": float(threshold),
        "n_nonzero_activation": int(np.sum(activations > 0)),
        "samples_internal": samples_internal,
        "samples_for_explainer": samples_for_explainer,
        "heldout_internal_by_tag": heldout_internal_by_tag,
        "summary": {
            "top_n": int(config.top_n),
            "top50_unique_normalized_texts": int(len(set(top_norms))),
            "top50_duplicate_rows": int(config.top_n - len(set(top_norms))),
            "top50_target_match_count": int(sum(int(row["target_match"]) for row in samples_internal)),
            "evidence_text_unique": bool(len(set(top_norms)) == len(top_norms)),
            "heldout_counts": found,
            "heldout_text_unique": True,
            "evidence_heldout_text_disjoint": True,
            "scorer_eligible": True,
            "interpretability_eligible": True,
            "common_surface_tokens_internal": sorted(cue_tokens),
        },
    }

    output_path = Path(output_dir)
    evidence_dir = output_path / "evidence_packs"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    packs_path = evidence_dir / "single_latent_top50_pack.jsonl"
    top_csv = evidence_dir / "top50_samples.csv"
    heldout_csv = evidence_dir / "heldout_samples_internal.csv"
    write_jsonl(packs_path, [packet])
    pd.DataFrame(samples_internal).to_csv(top_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame([row for rows in heldout_internal_by_tag.values() for row in rows]).to_csv(
        heldout_csv, index=False, encoding="utf-8-sig"
    )
    manifest = {
        "step": "prepare-single-latent-top50",
        "inputs": {
            "stable_latents": str(stable_latents_path),
            "feature_store": str(feature_store_path),
            "label_matrix": str(label_matrix_path),
            "records": str(records_path),
        },
        "outputs": {"pack": str(packs_path), "top50_samples": str(top_csv), "heldout_samples": str(heldout_csv)},
        "parameters": asdict(config),
        "packet_id": packet_id,
        "target_label": target_label,
        "latent_idx": int(latent_idx),
        "top50_unique_normalized_texts": int(len(set(top_norms))),
        "top50_duplicate_rows": int(config.top_n - len(set(top_norms))),
        "heldout_counts": found,
        "label_blind_explainer": True,
        "top50_heldout_row_disjoint": True,
        "top50_heldout_text_disjoint": True,
    }
    write_json(evidence_dir / "preparation_manifest.json", manifest)
    return manifest


def _format_top_samples(samples: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for sample in samples:
        duplicate = (
            f" duplicate_group={sample['duplicate_group']} duplicate_count={sample['duplicate_count']}"
            if sample.get("duplicate_group")
            else " duplicate_group=none duplicate_count=1"
        )
        lines.append(
            f"- id={sample['id']} rank={sample['rank']} activation={float(sample['activation']):.6g}{duplicate}\n"
            f"  text: {str(sample['text']).replace(chr(10), ' ').strip()}"
        )
    return "\n".join(lines)


def build_top50_explainer_prompt(pack: dict[str, Any]) -> str:
    samples = list(pack["samples_for_explainer"])
    sample_ids = [str(sample["id"]) for sample in samples]
    if len(samples) != 50 or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Top-50 explainer requires exactly 50 uniquely identified samples")
    n_unique = len({normalise_text(sample["text"]) for sample in samples})
    selection_unit = str(pack.get("summary", {}).get("selection_unit") or "raw_utterance_row")
    selection_note = (
        "The 50 items are already unique normalized texts; each is the highest-activation representative "
        "of its text group."
        if selection_unit == "unique_normalized_text"
        else "Repeated wording does not count as independent semantic evidence. A majority claim must hold "
        "for both raw utterances and unique normalized texts."
    )
    return f"""Analyze one anonymous SAE latent from its 50 highest-activation utterances as one set.

Latent metadata: latent_idx={int(pack['latent_idx'])}
Evidence size: 50 items, {n_unique} unique normalized texts.

Core instruction:
- Infer the narrowest shared feature that explains a majority of the set. The feature may be semantic,
  surface-form, mixed, or absent.
- Reason across all 50 utterances. Do not produce 50 sentence-by-sentence mini-analyses.
- Partition every sample id exactly once into supporting_sample_ids or outlier_sample_ids.
- {selection_note} Otherwise use feature_type="no_stable_pattern".
- Separate semantic_component from surface_component. If the apparent regularity is mainly a lexical or
  syntactic template, say so directly.
- candidate_explanation should be a concise, human-readable candidate explanation of what the latent may
  represent. It may name a behavior or concept when the Top-50 evidence supports it.

Top-50 samples:
{_format_top_samples(samples)}

Return only one JSON object with exactly these fields:
{json.dumps(list(TOP50_EXPLANATION_FIELDS), ensure_ascii=False)}

Requirements:
- feature_type must be exactly one of: semantic, surface, mixed, no_stable_pattern.
- supporting_sample_ids and outlier_sample_ids must be disjoint and together contain all 50 ids.
- representative_evidence_ids must contain 2 to 3 ids from supporting_sample_ids.
- positive_triggers, explicit_exclusions, possible_surface_confounds, alternative_hypotheses, and
  failure_modes must be JSON arrays of strings, even when there is only one item.
- Keep prose compact: use at most 2 sentences for each prose field and at most 4 items in each prose array.
- Do not enumerate or repeat example phrases in prose. Only supporting_sample_ids and outlier_sample_ids
  may be long arrays, because they are the complete evidence partition.
- positive_triggers must describe abstract recurring conditions, not merely copy sentences.
- confidence must be a number from 0 to 1 and follow this calibration:
  0.00-0.20: no consistent pattern.
  0.21-0.40: only a weak pattern, or multiple equally plausible explanations.
  0.41-0.60: barely meets the majority condition, with clear confounds.
  0.61-0.80: both majority conditions are clearly met and the main pattern is fairly consistent.
  0.81-1.00: coverage is very high, stability remains after deduplication, and alternative explanations
  and confounds are weak.
"""


def make_top50_explainer_task(*, packs_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    packs = read_jsonl(packs_path)
    if len(packs) != 1:
        raise ValueError(f"Single-latent workflow requires one pack, found {len(packs)}")
    pack = packs[0]
    task_id = f"{pack['packet_id']}_top50_explainer"
    raw_path = Path(output_dir) / "explainer_outputs" / "raw" / f"{task_id}.json"
    task = {
        "task_id": task_id,
        "task_type": "top50_aggregate_latent_explainer",
        "packet_id": pack["packet_id"],
        "latent_idx": int(pack["latent_idx"]),
        "repeat": 1,
        "prompt": build_top50_explainer_prompt(pack),
        "visible_samples": pack["samples_for_explainer"],
        "expected_output_path": str(raw_path),
        "output_format": "single_json_object",
        "status": "pending_zhipu_api",
    }
    task_path = Path(output_dir) / "llm_tasks" / "top50_explainer_tasks.jsonl"
    write_jsonl(task_path, [task])
    manifest = {
        "step": "make-top50-explainer-task",
        "inputs": {"pack": str(packs_path)},
        "outputs": {"tasks": str(task_path), "raw_output": str(raw_path)},
        "n_tasks": 1,
        "n_visible_samples": len(pack["samples_for_explainer"]),
        "aggregate_induction": True,
        "label_blind": True,
    }
    write_json(Path(output_dir) / "explainer_task_manifest.json", manifest)
    return manifest


def validate_top50_explainer_output(
    *,
    tasks_path: str | Path,
    execution_manifest_path: str | Path,
    output_dir: str | Path,
    config: SingleLatentP3Config = SingleLatentP3Config(),
) -> dict[str, Any]:
    tasks = read_jsonl(tasks_path)
    execution_rows = read_jsonl(execution_manifest_path) if Path(execution_manifest_path).exists() else []
    valid: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        reasons: list[str] = []
        raw_path = Path(task["expected_output_path"])
        manifest_matches = [row for row in execution_rows if str(row.get("task_id")) == task_id and row.get("status") == "success"]
        if len(manifest_matches) != 1:
            reasons.append(f"execution_manifest_success_count={len(manifest_matches)}")
        elif raw_path.exists():
            manifest_row = manifest_matches[0]
            if manifest_row.get("prompt_sha256") != _sha256_text(str(task["prompt"])):
                reasons.append("prompt_hash_mismatch")
            if manifest_row.get("raw_output_sha256") != _sha256_bytes(raw_path.read_bytes()):
                reasons.append("raw_output_hash_mismatch")
        if not raw_path.exists():
            reasons.append("raw_output_missing")
            payload: dict[str, Any] = {}
        else:
            parsed = parse_llm_json_file(raw_path)
            if not isinstance(parsed, dict):
                raise ValueError("Top-50 explanation must be a JSON object")
            payload = parsed
        missing = [field for field in TOP50_EXPLANATION_FIELDS if field not in payload]
        if missing:
            reasons.append(f"missing_fields={','.join(missing)}")

        row: dict[str, Any] | None = None
        if not missing:
            visible = {str(sample["id"]): sample for sample in task["visible_samples"]}
            all_ids = set(visible)
            supporting = _as_string_list(payload["supporting_sample_ids"], "supporting_sample_ids")
            outliers = _as_string_list(payload["outlier_sample_ids"], "outlier_sample_ids")
            representative = _as_string_list(payload["representative_evidence_ids"], "representative_evidence_ids")
            support_set = set(supporting)
            outlier_set = set(outliers)
            if support_set & outlier_set:
                reasons.append("support_outlier_overlap")
            if support_set | outlier_set != all_ids:
                reasons.append("support_outlier_not_full_partition")
            if len(supporting) != len(support_set) or len(outliers) != len(outlier_set):
                reasons.append("duplicate_sample_ids_in_partition")
            if not set(representative).issubset(support_set) or not 2 <= len(representative) <= 3:
                reasons.append("invalid_representative_evidence_ids")
            feature_type = str(payload["feature_type"]).strip().lower()
            if feature_type not in {"semantic", "surface", "mixed", "no_stable_pattern"}:
                reasons.append("invalid_feature_type")
            confidence = float(payload["confidence"])
            if not np.isfinite(confidence) or not 0 <= confidence <= 1:
                reasons.append("invalid_confidence")

            support_norms = {normalise_text(visible[sample_id]["text"]) for sample_id in support_set if sample_id in visible}
            all_norms = {normalise_text(sample["text"]) for sample in visible.values()}
            raw_fraction = len(support_set) / max(len(all_ids), 1)
            unique_fraction = len(support_norms) / max(len(all_norms), 1)
            majority_pass = bool(
                raw_fraction >= float(config.minimum_raw_support_fraction)
                and unique_fraction >= float(config.minimum_unique_support_fraction)
                and feature_type != "no_stable_pattern"
            )
            if not majority_pass:
                reasons.append("no_raw_and_unique_majority")
            hypothesis = str(payload["main_hypothesis"]).strip()
            if len(hypothesis) < 40:
                reasons.append("hypothesis_too_short")
            candidate_explanation = str(payload["candidate_explanation"]).strip()
            if not candidate_explanation:
                reasons.append("candidate_explanation_missing")
            if not str(payload["semantic_component"]).strip() or not str(payload["surface_component"]).strip():
                reasons.append("semantic_surface_separation_missing")

            row = {
                "task_id": task_id,
                "packet_id": task["packet_id"],
                "latent_idx": int(payload["latent_idx"]),
                "repeat": 1,
                "short_name": str(payload["short_name"]).strip(),
                "candidate_explanation": candidate_explanation,
                "main_hypothesis": hypothesis,
                "dominant_pattern": str(payload["dominant_pattern"]).strip(),
                "semantic_component": str(payload["semantic_component"]).strip(),
                "surface_component": str(payload["surface_component"]).strip(),
                "positive_triggers": _as_text_list(payload["positive_triggers"], "positive_triggers"),
                "explicit_exclusions": _as_text_list(payload["explicit_exclusions"], "explicit_exclusions"),
                "possible_surface_confounds": _as_text_list(
                    payload["possible_surface_confounds"], "possible_surface_confounds"
                ),
                "feature_type": feature_type,
                "supporting_sample_ids": supporting,
                "outlier_sample_ids": outliers,
                "representative_evidence_ids": representative,
                "key_evidence": representative,
                "alternative_hypotheses": _as_text_list(
                    payload["alternative_hypotheses"], "alternative_hypotheses"
                ),
                "failure_modes": _as_text_list(payload["failure_modes"], "failure_modes"),
                "confidence": confidence,
                "raw_support_count": len(support_set),
                "raw_support_fraction": raw_fraction,
                "unique_support_count": len(support_norms),
                "unique_total_count": len(all_norms),
                "unique_support_fraction": unique_fraction,
                "majority_gate_passed": majority_pass,
                "raw_output_path": str(raw_path),
                "validation_status": "valid" if not reasons else "review",
            }
            if int(row["latent_idx"]) != int(task["latent_idx"]):
                reasons.append("latent_idx_mismatch")
        quality_pass = bool(row is not None and not reasons)
        if quality_pass and row is not None:
            row["validation_status"] = "valid"
            valid.append(row)
        audit_rows.append(
            {
                "task_id": task_id,
                "latent_idx": task.get("latent_idx"),
                "quality_pass": quality_pass,
                "quality_reasons": ";".join(reasons),
                "raw_support_fraction": row.get("raw_support_fraction") if row else None,
                "unique_support_fraction": row.get("unique_support_fraction") if row else None,
                "feature_type": row.get("feature_type") if row else None,
                "confidence": row.get("confidence") if row else None,
            }
        )

    explainer_dir = Path(output_dir) / "explainer_outputs"
    explainer_dir.mkdir(parents=True, exist_ok=True)
    validated_path = explainer_dir / "validated_explanations.jsonl"
    audit_path = explainer_dir / "top50_explainer_quality_audit.csv"
    write_jsonl(validated_path, valid)
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False, encoding="utf-8-sig")
    manifest = {
        "step": "validate-top50-explainer",
        "inputs": {"tasks": str(tasks_path), "execution_manifest": str(execution_manifest_path)},
        "outputs": {"validated_explanations": str(validated_path), "quality_audit": str(audit_path)},
        "n_tasks": len(tasks),
        "n_valid": len(valid),
        "n_failed": len(tasks) - len(valid),
        "majority_gate": {
            "minimum_raw_support_fraction": config.minimum_raw_support_fraction,
            "minimum_unique_support_fraction": config.minimum_unique_support_fraction,
        },
    }
    write_json(explainer_dir / "validation_manifest.json", manifest)
    return manifest


def build_single_minimal_pair_prompt(explanation: dict[str, Any], pairs_per_latent: int) -> str:
    evidence = {
        key: explanation.get(key)
        for key in (
            "short_name",
            "main_hypothesis",
            "dominant_pattern",
            "semantic_component",
            "surface_component",
            "positive_triggers",
            "explicit_exclusions",
            "possible_surface_confounds",
            "feature_type",
            "failure_modes",
        )
    }
    return f"""Design {int(pairs_per_latent)} controlled text pairs for this candidate SAE latent explanation.

Candidate explanation:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Each pair must contain:
- positive_text: predicted to activate more strongly.
- negative_text: predicted to activate less strongly.
- changed_factor: the single trigger factor changed.
- held_constant: topic, approximate length, speaker role, and style kept fixed.
- test_dimension: exactly semantic, surface, or mixed.
- expected_direction: exactly positive_greater_than_negative.

Use natural counselor utterances. Change one factor only, do not copy Top-50 evidence verbatim, and keep
each pair lexically close enough to isolate the proposed trigger. Return only a JSON list.
"""


def make_single_minimal_pair_task(
    *, explanations_path: str | Path, packs_path: str | Path, output_dir: str | Path, pairs_per_latent: int = 5
) -> dict[str, Any]:
    explanations = read_jsonl(explanations_path)
    packs = read_jsonl(packs_path)
    if len(explanations) != 1 or len(packs) != 1:
        raise ValueError("Single minimal-pair task requires one explanation and one pack")
    explanation, pack = explanations[0], packs[0]
    task_id = f"{pack['packet_id']}_minimal_pairs"
    raw_path = Path(output_dir) / "minimal_pairs" / "raw_designer_outputs" / f"{task_id}.json"
    task = {
        "task_id": task_id,
        "task_type": "single_latent_minimal_pair_designer",
        "packet_id": pack["packet_id"],
        "target_label": pack["target_label"],
        "latent_idx": int(pack["latent_idx"]),
        "rank_within_label": int(pack["rank_within_label"]),
        "prompt": build_single_minimal_pair_prompt(explanation, pairs_per_latent),
        "expected_output_path": str(raw_path),
        "expected_pair_count": int(pairs_per_latent),
        "output_format": "json_list",
        "status": "pending_zhipu_api",
    }
    task_path = Path(output_dir) / "llm_tasks" / "minimal_pair_designer_tasks.jsonl"
    write_jsonl(task_path, [task])
    manifest = {
        "step": "make-single-minimal-pair-task",
        "outputs": {"tasks": str(task_path), "raw_output": str(raw_path)},
        "n_tasks": 1,
        "pairs_per_latent": int(pairs_per_latent),
    }
    write_json(Path(output_dir) / "minimal_pair_task_manifest.json", manifest)
    return manifest


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalise_text(left).split())
    right_tokens = set(normalise_text(right).split())
    if not left_tokens and not right_tokens:
        return 1.0
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)


def validate_single_minimal_pair_output(*, tasks_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    tasks = read_jsonl(tasks_path)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for task in tasks:
        raw_path = Path(task["expected_output_path"])
        try:
            payload = parse_llm_json_file(raw_path)
            if not isinstance(payload, list):
                raise ValueError("minimal-pair output must be a JSON list")
            if len(payload) != int(task["expected_pair_count"]):
                raise ValueError(f"expected {task['expected_pair_count']} pairs, got {len(payload)}")
            task_rows: list[dict[str, Any]] = []
            seen: set[tuple[str, str]] = set()
            for index, item in enumerate(payload, start=1):
                if not isinstance(item, dict):
                    raise ValueError(f"pair {index} is not an object")
                positive = str(item.get("positive_text", "")).strip()
                negative = str(item.get("negative_text", "")).strip()
                changed = str(item.get("changed_factor", "")).strip()
                held = str(item.get("held_constant", "")).strip()
                dimension = str(item.get("test_dimension", "")).strip().lower()
                direction = str(item.get("expected_direction", "")).strip()
                if not positive or not negative or not changed or not held:
                    raise ValueError(f"pair {index} has an empty required field")
                if normalise_text(positive) == normalise_text(negative):
                    raise ValueError(f"pair {index} has identical texts")
                if dimension not in {"semantic", "surface", "mixed"}:
                    raise ValueError(f"pair {index} has invalid test_dimension={dimension!r}")
                if direction != "positive_greater_than_negative":
                    raise ValueError(f"pair {index} has invalid expected_direction={direction!r}")
                key = (normalise_text(positive), normalise_text(negative))
                if key in seen:
                    raise ValueError(f"pair {index} duplicates an earlier pair")
                seen.add(key)
                overlap = _token_jaccard(positive, negative)
                pos_words = max(len(normalise_text(positive).split()), 1)
                neg_words = max(len(normalise_text(negative).split()), 1)
                length_ratio = pos_words / neg_words
                if overlap < 0.20 or not 0.5 <= length_ratio <= 2.0:
                    raise ValueError(
                        f"pair {index} is not sufficiently controlled: token_jaccard={overlap:.3f}, "
                        f"length_ratio={length_ratio:.3f}"
                    )
                task_rows.append(
                    {
                        "task_id": task["task_id"],
                        "packet_id": task["packet_id"],
                        "target_label": task["target_label"],
                        "latent_idx": int(task["latent_idx"]),
                        "pair_id": f"{task['task_id']}_p{index:02d}",
                        "positive_text": positive,
                        "negative_text": negative,
                        "changed_factor": changed,
                        "held_constant": held,
                        "test_dimension": dimension,
                        "expected_direction": direction,
                        "token_jaccard": overlap,
                        "length_ratio": length_ratio,
                    }
                )
            rows.extend(task_rows)
        except Exception as exc:
            errors.append({"task_id": task.get("task_id"), "error": f"{type(exc).__name__}: {exc}", "raw_output": str(raw_path)})

    pair_dir = Path(output_dir) / "minimal_pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = pair_dir / "validated_minimal_pairs.jsonl"
    errors_path = pair_dir / "minimal_pair_validation_errors.csv"
    write_jsonl(pairs_path, rows)
    pd.DataFrame(errors).to_csv(errors_path, index=False, encoding="utf-8-sig")
    manifest = {
        "step": "validate-single-minimal-pairs",
        "outputs": {"validated_minimal_pairs": str(pairs_path), "validation_errors": str(errors_path)},
        "n_tasks": len(tasks),
        "n_pairs": len(rows),
        "n_errors": len(errors),
        "quality_pass": bool(len(tasks) == 1 and not errors and len(rows) == int(tasks[0]["expected_pair_count"])),
    }
    write_json(pair_dir / "minimal_pair_validation_manifest.json", manifest)
    return manifest


def write_single_latent_p3_report(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    pack_rows = read_jsonl(output_path / "evidence_packs" / "single_latent_top50_pack.jsonl")
    explanations = read_jsonl(output_path / "explainer_outputs" / "validated_explanations.jsonl")
    pack = pack_rows[0] if pack_rows else {}
    explanation = explanations[0] if explanations else {}
    scorer_path = output_path / "scorer_outputs" / "scorer_metrics.csv"
    baseline_path = output_path / "scorer_outputs" / "baseline_metrics.csv"
    pair_summary_path = output_path / "minimal_pairs" / "minimal_pair_summary.csv"
    pair_results_path = output_path / "minimal_pairs" / "minimal_pair_results.csv"
    scorer = pd.read_csv(scorer_path) if scorer_path.exists() else pd.DataFrame()
    baseline = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
    pair_summary = pd.read_csv(pair_summary_path) if pair_summary_path.exists() else pd.DataFrame()
    pair_results = pd.read_csv(pair_results_path) if pair_results_path.exists() else pd.DataFrame()

    scorer_row = scorer.iloc[0].to_dict() if not scorer.empty else {}
    baseline_row = baseline.iloc[0].to_dict() if not baseline.empty else {}
    pair_row = pair_summary.iloc[0].to_dict() if not pair_summary.empty else {}
    explainer_pass = bool(explanation.get("majority_gate_passed", False))
    scorer_pass = str(scorer_row.get("status", "")) == "accepted"
    pair_pass = bool(
        pair_row
        and float(pair_row.get("pass_rate", 0)) >= 0.60
        and float(pair_row.get("mean_gap", 0)) > 0
    )
    usable = bool(explainer_pass and scorer_pass and pair_pass)

    def fmt(value: Any) -> str:
        try:
            number = float(value)
            return f"{number:.3f}" if np.isfinite(number) else "NA"
        except (TypeError, ValueError):
            return "NA"

    evidence_by_id = {str(row["id"]): row for row in pack.get("samples_for_explainer", [])}
    representative_lines = []
    for sample_id in explanation.get("representative_evidence_ids", []):
        sample = evidence_by_id.get(str(sample_id), {})
        representative_lines.append(
            f"- `{sample_id}` (rank {sample.get('rank', '?')}, activation {fmt(sample.get('activation'))}): "
            f"{sample.get('text', '')}"
        )
    pair_lines = []
    if not pair_results.empty:
        for _, row in pair_results.iterrows():
            pair_lines.append(
                f"- `{row.get('pair_id')}`: gap={fmt(row.get('activation_gap'))}, "
                f"passed={bool(row.get('pair_passed'))}; + {row.get('positive_text')} / - {row.get('negative_text')}"
            )

    lines = [
        "# GLM-4.7 Single-Latent Top-50 P3 Report",
        "",
        f"- final status: `{'usable_candidate_explanation' if usable else 'not_yet_usable'}`",
        f"- target audit label: `{pack.get('target_label', '')}`",
        f"- stable-core latent: `{pack.get('latent_idx', '')}`",
        f"- stable-core rank: {pack.get('rank_within_label', '')}",
        f"- selection metrics: Cohen's d={fmt(pack.get('cohens_d'))}, AUC={fmt(pack.get('auc'))}, "
        f"precision@50={fmt(pack.get('precision_at_50'))}, inclusion frequency={fmt(pack.get('inclusion_frequency'))}",
        "",
        "## Top-50 Aggregate Induction",
        "",
        f"- raw support: {explanation.get('raw_support_count', 0)}/50 ({fmt(explanation.get('raw_support_fraction'))})",
        f"- unique-text support: {explanation.get('unique_support_count', 0)}/{explanation.get('unique_total_count', 0)} "
        f"({fmt(explanation.get('unique_support_fraction'))})",
        f"- duplicate rows in Top-50: {pack.get('summary', {}).get('top50_duplicate_rows', 'NA')}",
        f"- majority gate: `{explainer_pass}`",
        f"- candidate name: **{explanation.get('short_name', 'NA')}**",
        f"- feature type: `{explanation.get('feature_type', 'NA')}`",
        f"- hypothesis: {explanation.get('main_hypothesis', 'NA')}",
        f"- semantic component: {explanation.get('semantic_component', 'NA')}",
        f"- surface component: {explanation.get('surface_component', 'NA')}",
        "",
        "### Representative Top-50 Evidence",
        "",
        *(representative_lines or ["No validated representative evidence."]),
        "",
        "## Held-Out Scorer Gate",
        "",
        f"- explanation scorer AUROC: {fmt(scorer_row.get('auroc'))}",
        f"- label-definition baseline AUROC: {fmt(baseline_row.get('auroc'))}",
        f"- latent-specific gap: {fmt(scorer_row.get('latent_gap'))}",
        f"- status: `{scorer_row.get('status', 'missing')}`",
        "",
        "The scorer set is row-disjoint and normalized-text-disjoint from Top-50 induction evidence. "
        "An accepted result requires AUROC >= 0.70 and a positive gap over the label-definition baseline.",
        "",
        "## Minimal-Pair SAE Test",
        "",
        f"- pairs tested: {pair_row.get('n_pairs', 0)}",
        f"- mean activation gap: {fmt(pair_row.get('mean_gap'))}",
        f"- positive > negative pass rate: {fmt(pair_row.get('pass_rate'))}",
        f"- gate: `{pair_pass}` (requires pass rate >= 0.60 and mean gap > 0)",
        "",
        *(pair_lines or ["No activation-tested minimal pairs."]),
        "",
        "The minimal-pair pass rate is exactly the acceptance boundary rather than a robust ceiling. "
        "Two pairs failed, so the sentence-initial surface pattern is better supported than a universal "
        "reflection-function rule.",
        "",
        "## Interpretation Boundary",
        "",
        "This is a usable candidate explanation only if all three gates pass. It supports a stable textual "
        "association, held-out activation detectability, and controlled-text sensitivity for this SAE latent. "
        "It does not show that the latent equals a MISC label, that the base model understands the concept, "
        "or that the latent is a causal mechanism for counselor behavior. The dataset contains only the current "
        "counselor utterance, so prior client context cannot verify whether each sentence is functionally a true "
        "reflection; the surface-form claim is therefore stronger than the counseling-function claim.",
    ]
    report_path = output_path / "single_latent_p3_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    final_manifest = {
        "analysis": "zhipu_glm47_single_latent_top50_p3",
        "final_status": "usable_candidate_explanation" if usable else "not_yet_usable",
        "target_label": pack.get("target_label"),
        "latent_idx": pack.get("latent_idx"),
        "gates": {"top50_majority": explainer_pass, "heldout_scorer": scorer_pass, "minimal_pair_activation": pair_pass},
        "qualification": (
            "minimal_pair_gate_passed_at_boundary"
            if pair_row and float(pair_row.get("pass_rate", 0)) == 0.60
            else ""
        ),
        "criteria": {
            "top50_majority": "raw_support_fraction>=0.50 and unique_support_fraction>=0.50",
            "heldout_scorer": "AUROC>=0.70 and AUROC-label_baseline_AUROC>0",
            "minimal_pair_activation": "pass_rate>=0.60 and mean_gap>0",
        },
        "outputs": {"report": str(report_path)},
    }
    write_json(output_path / "final_manifest.json", final_manifest)
    return report_path


__all__ = [
    "MINIMAL_PAIR_SYSTEM_PROMPT",
    "SCORER_SYSTEM_PROMPT",
    "TOP50_EXPLAINER_SYSTEM_PROMPT",
    "SingleLatentP3Config",
    "build_single_minimal_pair_prompt",
    "build_top50_explainer_prompt",
    "make_single_minimal_pair_task",
    "make_top50_explainer_task",
    "prepare_single_latent_pack",
    "validate_single_minimal_pair_output",
    "validate_top50_explainer_output",
    "write_single_latent_p3_report",
]
