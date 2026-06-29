"""Generate P3 dry-run SAE feature cards for MISC latent interpretation.

This is a descriptive, read-only analysis export. It consumes the existing
Top20 positive Cohen's d latent packets and adds contrast examples, dry-run
LLM prompts, and scoring task sets. It does not call an LLM and does not claim
causal mechanism evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_LABELS = ("RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF")
DEFAULT_INPUT_DIR = (
    "outputs/misc_full_sae_eval/interpretability/top20_cohensd_latent_utterances"
)
DEFAULT_EVIDENCE_DIR = (
    "outputs/misc_full_sae_eval/interpretability/top20_cohensd_latent_utterances/"
    "latent_evidence_packets"
)
DEFAULT_OUTPUT_DIR = "outputs/misc_full_sae_eval/interpretability/p3_feature_cards"
P3_VERSION = "misc_p3_feature_cards_v1"
CONTEXT_LIMITATION = (
    "Only counselor current utterance is available; prior client context is unavailable. "
    "RES/REC/RE context-relation claims must be treated as limited."
)

SIBLING_LABELS = {
    "QU": ("QUO", "QUC"),
    "QUO": ("QUC", "QU"),
    "QUC": ("QUO", "QU"),
    "RE": ("RES", "REC", "GI"),
    "RES": ("REC", "RE"),
    "REC": ("RES", "RE"),
    "GI": ("RE", "QU", "SU"),
    "SU": ("AF", "GI", "QU"),
    "AF": ("SU", "RE", "RES", "REC"),
}

SURFACE_WORDS = {
    "question": re.compile(
        r"(\?|^|\b)(what|how|why|when|where|who|do|does|did|are|is|can|could|would|will|have|has)\b",
        flags=re.IGNORECASE,
    ),
    "reflection": re.compile(
        r"\b(you|you're|you are|your|sounds like|it sounds|seems like|it seems|what i hear|so you|so it)\b",
        flags=re.IGNORECASE,
    ),
    "positive": re.compile(
        r"\b(good|great|nice|wow|proud|appreciate|appreciated|strength|strong|effort|really|excellent|amazing)\b",
        flags=re.IGNORECASE,
    ),
    "information": re.compile(
        r"\b(because|means|mean|need to|should|could|can|if you|when you|information|risk|helps|works|recommend)\b",
        flags=re.IGNORECASE,
    ),
}

LABEL_SURFACE_KEYS = {
    "QU": ("question",),
    "QUO": ("question",),
    "QUC": ("question",),
    "RE": ("reflection",),
    "RES": ("reflection",),
    "REC": ("reflection", "positive"),
    "GI": ("information",),
    "SU": ("information",),
    "AF": ("positive",),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _normalise_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text.strip().lower())


def _safe_slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def _fmt_float(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _load_feature_matrix(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    if path.suffix == ".npz":
        payload = np.load(path)
        for key in ("utterance_features", "features", "X"):
            if key in payload:
                return np.asarray(payload[key], dtype=np.float32)
        raise KeyError(f"{path} does not contain one of: utterance_features, features, X")
    if path.suffix == ".pt":
        import torch

        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, torch.Tensor):
            tensor = payload
        elif isinstance(payload, dict):
            for key in ("utterance_features", "features", "X"):
                if key in payload:
                    tensor = payload[key]
                    break
            else:
                raise KeyError(f"{path} does not contain one of: utterance_features, features, X")
        else:
            raise TypeError(f"Unsupported torch payload type: {type(payload).__name__}")
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().float().numpy()
        return np.asarray(tensor, dtype=np.float32)
    raise ValueError(f"Unsupported feature matrix extension: {path.suffix}")


def _numeric_positive(value: Any) -> bool:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0]
    return float(number) > 0


def _active_labels(row: pd.Series, labels: tuple[str, ...]) -> str:
    return ",".join(label for label in labels if label in row and _numeric_positive(row[label]))


def _record_value(record: dict[str, Any], label_row: pd.Series, *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
        if key in label_row and pd.notna(label_row[key]):
            return label_row[key]
    return default


def _ordered_by_activation(activations: np.ndarray, candidates: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or len(candidates) == 0:
        return np.asarray([], dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)
    order = np.lexsort((candidates, -activations[candidates]))
    return candidates[order[: min(limit, len(order))]]


def _ordered_low_activation(activations: np.ndarray, candidates: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or len(candidates) == 0:
        return np.asarray([], dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)
    order = np.lexsort((candidates, activations[candidates]))
    return candidates[order[: min(limit, len(order))]]


def _as_row_idx_set(examples: list[dict[str, Any]]) -> set[int]:
    rows: set[int] = set()
    for example in examples:
        try:
            rows.add(int(example["row_idx"]))
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def _label_mask(label_matrix: pd.DataFrame, label: str) -> np.ndarray:
    if label not in label_matrix.columns:
        return np.zeros(len(label_matrix), dtype=bool)
    return pd.to_numeric(label_matrix[label], errors="coerce").fillna(0).to_numpy(dtype=float) > 0


def _surface_score(text: str, target_label: str, median_words: float | None) -> int:
    normalised = _normalise_text(text)
    score = 0
    for key in LABEL_SURFACE_KEYS.get(target_label, ()):
        if SURFACE_WORDS[key].search(normalised):
            score += 3
    if "?" in normalised and target_label in {"QU", "QUO", "QUC"}:
        score += 2
    if median_words is not None and median_words > 0:
        words = len(normalised.split())
        if 0.6 * median_words <= words <= 1.6 * median_words:
            score += 1
    return score


def _example_from_row(
    *,
    packet_id: str,
    target_label: str,
    latent_idx: int,
    row_idx: int,
    activation: float,
    example_group: str,
    rank: int,
    label_matrix: pd.DataFrame,
    records: list[dict[str, Any]],
    labels: tuple[str, ...],
    surface_match_score: int | None = None,
) -> dict[str, Any]:
    label_row = label_matrix.iloc[int(row_idx)]
    record = records[int(row_idx)]
    target_match = int(target_label in label_matrix.columns and _numeric_positive(label_row[target_label]))
    text = _record_value(record, label_row, "unit_text", "text", "utterance", default="")
    example = {
        "packet_id": packet_id,
        "target_label": target_label,
        "latent_idx": int(latent_idx),
        "example_group": example_group,
        "rank_within_group": int(rank),
        "row_idx": int(row_idx),
        "activation": float(activation),
        "target_match": target_match,
        "active_labels": _active_labels(label_row, labels),
        "record_id": _record_value(record, label_row, "record_id", "sample_id", default=f"row_{row_idx}"),
        "file_id": _record_value(record, label_row, "file_id", default=""),
        "source_line": _record_value(record, label_row, "source_line", default=""),
        "source_split": _record_value(record, label_row, "source_split", default=""),
        "quality_label": _record_value(record, label_row, "quality_label", default=""),
        "predicted_code": _record_value(record, label_row, "predicted_code", default=""),
        "predicted_subcode": _record_value(record, label_row, "predicted_subcode", default=""),
        "confidence": _record_value(record, label_row, "confidence", default=""),
        "unit_text": str(text),
        "normalized_text": _normalise_text(text),
    }
    if surface_match_score is not None:
        example["surface_match_score"] = int(surface_match_score)
    return example


def _dedupe_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int]] = set()
    out: list[dict[str, Any]] = []
    for example in examples:
        key = (str(example.get("example_group")), int(example.get("row_idx", -1)))
        if key in seen:
            continue
        seen.add(key)
        out.append(example)
    return out


def _load_packets(path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    packets: dict[tuple[str, int, int], dict[str, Any]] = {}
    for packet in _read_jsonl(path):
        key = (
            str(packet.get("target_label", "")).upper(),
            int(packet.get("latent_idx")),
            int(packet.get("rank_within_label")),
        )
        packets[key] = packet
    return packets


def _normalise_latents(latents: pd.DataFrame, labels: tuple[str, ...], top_features: int | None) -> pd.DataFrame:
    label_col = "target_label" if "target_label" in latents.columns else "label"
    required = {label_col, "latent_idx", "rank_within_label"}
    missing = sorted(required.difference(latents.columns))
    if missing:
        raise ValueError(f"Latents table missing columns: {missing}")
    out = latents.copy()
    out["target_label"] = out[label_col].astype(str).str.upper()
    out["latent_idx"] = pd.to_numeric(out["latent_idx"], errors="coerce").fillna(-1).astype(int)
    out["rank_within_label"] = pd.to_numeric(out["rank_within_label"], errors="coerce").fillna(-1).astype(int)
    for col in ("cohens_d", "directional_auc", "precision_at_50"):
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    label_order = {label: idx for idx, label in enumerate(labels)}
    out = out[out["target_label"].isin(label_order)].copy()
    if top_features is not None:
        out = out[out["rank_within_label"] <= int(top_features)].copy()
    out["_label_order"] = out["target_label"].map(label_order)
    out = out.sort_values(["_label_order", "rank_within_label", "latent_idx"], kind="mergesort")
    return out.drop(columns=["_label_order"]).reset_index(drop=True)


def _layer_metadata(layer_selection: pd.DataFrame, target_label: str) -> dict[str, Any]:
    if layer_selection.empty or "target_label" not in layer_selection.columns:
        return {
            "sae_canonical_layer": 19,
            "sae_hook_point": "blocks.19.hook_resid_post",
            "llama_label_specific_best_layer": None,
            "llama_early_stable_layer": None,
            "note": "layer_selection_unavailable",
        }
    row_df = layer_selection[layer_selection["target_label"].astype(str).str.upper() == target_label]
    if row_df.empty:
        return {
            "sae_canonical_layer": 19,
            "sae_hook_point": "blocks.19.hook_resid_post",
            "llama_label_specific_best_layer": None,
            "llama_early_stable_layer": None,
            "note": "label_not_found_in_layer_selection",
        }
    row = row_df.iloc[0].to_dict()
    return {
        "sae_canonical_layer": row.get("canonical_layer"),
        "sae_hook_point": row.get("hook_point"),
        "llama_label_specific_best_layer": row.get("label_specific_best_layer"),
        "llama_label_specific_best_auc": row.get("label_specific_best_auc"),
        "llama_early_stable_layer": row.get("early_stable_layer"),
        "llama_early_stable_auc": row.get("early_stable_auc"),
        "near_delta": row.get("near_delta"),
        "note": row.get("availability_note"),
    }


def _top_examples_from_packet(packet: dict[str, Any], group: str, limit: int) -> list[dict[str, Any]]:
    examples = [dict(ex) for ex in packet.get("examples", []) if ex.get("example_group") == group]
    examples = sorted(examples, key=lambda ex: int(ex.get("rank_within_group", 10**9)))
    return examples[:limit]


def _stable_seed(*parts: Any) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)


def _sample_random_target(
    candidates: np.ndarray,
    *,
    exclude_rows: set[int],
    requested: int,
    seed: int,
) -> np.ndarray:
    if requested <= 0:
        return np.asarray([], dtype=np.int64)
    pool = np.asarray([int(idx) for idx in candidates.tolist() if int(idx) not in exclude_rows], dtype=np.int64)
    if len(pool) == 0:
        return np.asarray([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(pool, size=min(int(requested), len(pool)), replace=False)
    return np.asarray(chosen, dtype=np.int64)


def _source_packet_summary(examples: list[dict[str, Any]]) -> dict[str, Any]:
    top_examples = [ex for ex in examples if ex.get("example_group") == "top_activating"]
    top_match_rate = (
        float(np.mean([int(ex.get("target_match", 0)) for ex in top_examples])) if top_examples else np.nan
    )
    normalized_counts = Counter(str(ex.get("normalized_text", "")) for ex in examples if ex.get("normalized_text"))
    duplicate_rows = sum(
        1
        for ex in examples
        if ex.get("normalized_text") and normalized_counts[str(ex.get("normalized_text"))] > 1
    )
    return {
        "top_activating_target_match_rate": top_match_rate,
        "duplicate_text_row_count": int(duplicate_rows),
    }


def _synthesise_source_packet(
    *,
    latent_row: pd.Series,
    features: np.ndarray,
    label_matrix: pd.DataFrame,
    records: list[dict[str, Any]],
    labels: tuple[str, ...],
    top_activating: int,
    high_non_target: int,
    random_target: int,
) -> dict[str, Any]:
    target_label = str(latent_row["target_label"])
    latent_idx = int(latent_row["latent_idx"])
    rank = int(latent_row["rank_within_label"])
    packet_id = f"p3_fallback_{target_label}_{rank:02d}_latent_{latent_idx}"
    activations = np.asarray(features[:, latent_idx], dtype=np.float32)
    all_indices = np.arange(len(activations), dtype=np.int64)
    target_mask = _label_mask(label_matrix, target_label)
    top_indices = _ordered_by_activation(activations, all_indices, top_activating)
    non_target_indices = all_indices[~target_mask]
    high_non_target_indices = _ordered_by_activation(activations, non_target_indices, high_non_target)
    random_target_indices = _sample_random_target(
        all_indices[target_mask],
        exclude_rows={int(idx) for idx in top_indices.tolist()},
        requested=random_target,
        seed=_stable_seed(P3_VERSION, target_label, latent_idx, rank, "random_target"),
    )

    examples: list[dict[str, Any]] = []
    for group, indices in (
        ("top_activating", top_indices),
        ("high_non_target", high_non_target_indices),
        ("random_target", random_target_indices),
    ):
        for example_rank, row_idx in enumerate(indices.tolist(), start=1):
            examples.append(
                _example_from_row(
                    packet_id=packet_id,
                    target_label=target_label,
                    latent_idx=latent_idx,
                    row_idx=int(row_idx),
                    activation=float(activations[int(row_idx)]),
                    example_group=group,
                    rank=example_rank,
                    label_matrix=label_matrix,
                    records=records,
                    labels=labels,
                )
            )

    return {
        "packet_id": packet_id,
        "target_label": target_label,
        "latent_idx": latent_idx,
        "rank_within_label": rank,
        "cohens_d": latent_row.get("cohens_d"),
        "directional_auc": latent_row.get("directional_auc"),
        "precision_at_50": latent_row.get("precision_at_50"),
        "evidence_source": "p3_generated_from_feature_store",
        "summary": _source_packet_summary(examples),
        "examples": examples,
    }


def _build_sibling_contrast(
    *,
    packet_id: str,
    target_label: str,
    latent_idx: int,
    activations: np.ndarray,
    label_matrix: pd.DataFrame,
    records: list[dict[str, Any]],
    labels: tuple[str, ...],
    exclude_rows: set[int],
    limit: int,
) -> list[dict[str, Any]]:
    target_mask = _label_mask(label_matrix, target_label)
    sibling_mask = np.zeros(len(label_matrix), dtype=bool)
    for sibling in SIBLING_LABELS.get(target_label, ()):
        sibling_mask |= _label_mask(label_matrix, sibling)
    candidates = np.where((~target_mask) & sibling_mask)[0]
    candidates = np.asarray([idx for idx in candidates.tolist() if idx not in exclude_rows], dtype=np.int64)
    selected = _ordered_by_activation(activations, candidates, limit)
    return [
        _example_from_row(
            packet_id=packet_id,
            target_label=target_label,
            latent_idx=latent_idx,
            row_idx=int(row_idx),
            activation=float(activations[int(row_idx)]),
            example_group="sibling_code_contrast",
            rank=rank,
            label_matrix=label_matrix,
            records=records,
            labels=labels,
        )
        for rank, row_idx in enumerate(selected, start=1)
    ]


def _build_surface_contrast(
    *,
    packet_id: str,
    target_label: str,
    latent_idx: int,
    activations: np.ndarray,
    label_matrix: pd.DataFrame,
    records: list[dict[str, Any]],
    labels: tuple[str, ...],
    top_examples: list[dict[str, Any]],
    exclude_rows: set[int],
    limit: int,
) -> list[dict[str, Any]]:
    target_mask = _label_mask(label_matrix, target_label)
    top_lengths = [
        len(str(ex.get("unit_text") or "").split())
        for ex in top_examples
        if str(ex.get("unit_text") or "").strip()
    ]
    median_words = float(np.median(top_lengths)) if top_lengths else None
    scored: list[tuple[int, float, int]] = []
    for idx, record in enumerate(records):
        if target_mask[idx] or idx in exclude_rows:
            continue
        label_row = label_matrix.iloc[idx]
        text = _record_value(record, label_row, "unit_text", "text", "utterance", default="")
        score = _surface_score(str(text), target_label, median_words)
        if score > 0:
            scored.append((score, float(activations[idx]), idx))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    selected = scored[:limit]
    return [
        _example_from_row(
            packet_id=packet_id,
            target_label=target_label,
            latent_idx=latent_idx,
            row_idx=int(row_idx),
            activation=float(activation),
            example_group="surface_matched_contrast",
            rank=rank,
            label_matrix=label_matrix,
            records=records,
            labels=labels,
            surface_match_score=int(score),
        )
        for rank, (score, activation, row_idx) in enumerate(selected, start=1)
    ]


def _build_low_activation_examples(
    *,
    packet_id: str,
    target_label: str,
    latent_idx: int,
    activations: np.ndarray,
    label_matrix: pd.DataFrame,
    records: list[dict[str, Any]],
    labels: tuple[str, ...],
    exclude_rows: set[int],
    limit: int,
) -> list[dict[str, Any]]:
    candidates = np.asarray([idx for idx in range(len(activations)) if idx not in exclude_rows], dtype=np.int64)
    selected = _ordered_low_activation(activations, candidates, limit)
    return [
        _example_from_row(
            packet_id=packet_id,
            target_label=target_label,
            latent_idx=latent_idx,
            row_idx=int(row_idx),
            activation=float(activations[int(row_idx)]),
            example_group="low_activation_scoring",
            rank=rank,
            label_matrix=label_matrix,
            records=records,
            labels=labels,
        )
        for rank, row_idx in enumerate(selected, start=1)
    ]


def _prompt_examples(examples: list[dict[str, Any]], groups: tuple[str, ...], per_group: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group in groups:
        group_examples = [ex for ex in examples if ex.get("example_group") == group]
        for ex in group_examples[:per_group]:
            out.append(
                {
                    "group": group,
                    "rank": ex.get("rank_within_group"),
                    "activation": ex.get("activation"),
                    "target_match": ex.get("target_match"),
                    "active_labels": ex.get("active_labels"),
                    "text": ex.get("unit_text"),
                }
            )
    return out


def _build_input_prompt(card: dict[str, Any], *, prompt_examples_per_group: int) -> list[dict[str, str]]:
    target_label = card["target_label"]
    system = (
        "You are a cautious research assistant reviewing anonymous SAE latents from "
        "Motivational Interviewing counselor utterances. Do not say 'this feature is "
        "the target label'. Use hedged language such as appears associated with, may "
        "capture, candidate explanation. Separate surface form, dialogue function, "
        "MI-principle, artifact, and mixed/unclear evidence. No prior client utterance "
        "is available."
    )
    user_payload = {
        "task": "Generate an input-centric candidate explanation for one SAE latent.",
        "p3_version": P3_VERSION,
        "target_misc_label_for_alignment": target_label,
        "latent_idx": card["latent_idx"],
        "rank_within_label": card["rank_within_label"],
        "association_metrics": card["association_metrics"],
        "layer_metadata": card["layer_metadata"],
        "context_limitation": CONTEXT_LIMITATION,
        "critical_instruction": (
            "Do not directly conclude that this latent is a MISC label. Explain what "
            "shared input pattern may drive activation and whether that pattern is "
            "surface-level, counseling-functional, artifact-like, or unclear."
        ),
        "examples": _prompt_examples(
            card["examples"],
            (
                "top_activating",
                "high_non_target",
                "sibling_code_contrast",
                "surface_matched_contrast",
                "random_target",
            ),
            prompt_examples_per_group,
        ),
        "required_json": {
            "one_sentence_tentative_interpretation": "string",
            "main_patterns": [
                {
                    "pattern_name": "string",
                    "pattern_type": "surface_form|dialogue_function|context_relation|mi_principle|artifact|mixed_unclear",
                    "evidence": "string",
                }
            ],
            "relationship_to_target_label": "string",
            "artifact_risk": "low|medium|high|unclear",
            "evidence_quality": "high|medium|low|uninterpretable",
            "candidate_feature_name": "string",
            "alternative_explanations": ["string", "string"],
            "recommended_followup_checks": ["string"],
            "final_concise_conclusion": "string",
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def _task_examples_from_group(
    examples: list[dict[str, Any]],
    group: str,
    *,
    start_after: int = 0,
    limit: int = 8,
    expected_high_activation: int | None = None,
    expected_target_match: int | None = None,
) -> list[dict[str, Any]]:
    group_examples = [ex for ex in examples if ex.get("example_group") == group]
    group_examples = sorted(group_examples, key=lambda ex: int(ex.get("rank_within_group", 10**9)))
    selected = group_examples[start_after : start_after + limit]
    out: list[dict[str, Any]] = []
    for ex in selected:
        item = {
            "row_idx": ex.get("row_idx"),
            "group": group,
            "activation": ex.get("activation"),
            "active_labels": ex.get("active_labels"),
            "text": ex.get("unit_text"),
        }
        if expected_high_activation is not None:
            item["expected_high_activation"] = int(expected_high_activation)
        if expected_target_match is not None:
            item["expected_target_match"] = int(expected_target_match)
        out.append(item)
    return out


def _build_scoring_tasks(card: dict[str, Any], *, heldout_start: int, per_class: int) -> list[dict[str, Any]]:
    examples = card["examples"]
    activation_examples = []
    activation_examples.extend(
        _task_examples_from_group(
            examples,
            "top_activating",
            start_after=heldout_start,
            limit=per_class,
            expected_high_activation=1,
        )
    )
    activation_examples.extend(
        _task_examples_from_group(
            examples,
            "low_activation_scoring",
            limit=per_class,
            expected_high_activation=0,
        )
    )

    code_examples = []
    code_examples.extend(
        _task_examples_from_group(
            examples,
            "random_target",
            limit=per_class,
            expected_target_match=1,
        )
    )
    code_examples.extend(
        _task_examples_from_group(
            examples,
            "sibling_code_contrast",
            limit=per_class,
            expected_target_match=0,
        )
    )
    code_examples.extend(
        _task_examples_from_group(
            examples,
            "surface_matched_contrast",
            limit=per_class,
            expected_target_match=0,
        )
    )

    return [
        {
            "packet_id": card["packet_id"],
            "target_label": card["target_label"],
            "latent_idx": card["latent_idx"],
            "task_type": "activation_prediction_task",
            "status": "pending_score",
            "instructions": (
                "After reading a candidate explanation, predict whether each held-out "
                "utterance should be high-activating for this latent."
            ),
            "examples": activation_examples,
        },
        {
            "packet_id": card["packet_id"],
            "target_label": card["target_label"],
            "latent_idx": card["latent_idx"],
            "task_type": "code_discrimination_task",
            "status": "pending_score",
            "instructions": (
                "After reading a candidate explanation, decide whether each utterance "
                "matches the target MISC label rather than sibling/surface negatives."
            ),
            "examples": code_examples,
        },
    ]


def _count_groups(examples: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(ex.get("example_group")) for ex in examples))


def _build_card(
    *,
    latent_row: pd.Series,
    source_packet: dict[str, Any],
    layer_selection: pd.DataFrame,
    features: np.ndarray,
    label_matrix: pd.DataFrame,
    records: list[dict[str, Any]],
    labels: tuple[str, ...],
    top_activating: int,
    high_non_target: int,
    random_target: int,
    sibling_contrast: int,
    surface_contrast: int,
    low_activation: int,
) -> dict[str, Any]:
    target_label = str(latent_row["target_label"])
    latent_idx = int(latent_row["latent_idx"])
    rank = int(latent_row["rank_within_label"])
    packet_id = str(source_packet.get("packet_id") or f"{target_label}_latent_{latent_idx}")
    activations = np.asarray(features[:, latent_idx], dtype=np.float32)

    examples: list[dict[str, Any]] = []
    examples.extend(_top_examples_from_packet(source_packet, "top_activating", top_activating))
    examples.extend(_top_examples_from_packet(source_packet, "high_non_target", high_non_target))
    examples.extend(_top_examples_from_packet(source_packet, "random_target", random_target))
    existing_rows = _as_row_idx_set(examples)
    top_examples = [ex for ex in examples if ex.get("example_group") == "top_activating"]

    sibling_examples = _build_sibling_contrast(
        packet_id=packet_id,
        target_label=target_label,
        latent_idx=latent_idx,
        activations=activations,
        label_matrix=label_matrix,
        records=records,
        labels=labels,
        exclude_rows=existing_rows,
        limit=sibling_contrast,
    )
    existing_rows.update(_as_row_idx_set(sibling_examples))
    surface_examples = _build_surface_contrast(
        packet_id=packet_id,
        target_label=target_label,
        latent_idx=latent_idx,
        activations=activations,
        label_matrix=label_matrix,
        records=records,
        labels=labels,
        top_examples=top_examples,
        exclude_rows=existing_rows,
        limit=surface_contrast,
    )
    existing_rows.update(_as_row_idx_set(surface_examples))
    low_examples = _build_low_activation_examples(
        packet_id=packet_id,
        target_label=target_label,
        latent_idx=latent_idx,
        activations=activations,
        label_matrix=label_matrix,
        records=records,
        labels=labels,
        exclude_rows=existing_rows,
        limit=low_activation,
    )
    examples.extend(sibling_examples)
    examples.extend(surface_examples)
    examples.extend(low_examples)
    examples = _dedupe_examples(examples)

    group_counts = _count_groups(examples)
    scarcity = {
        "top_activating": group_counts.get("top_activating", 0) < top_activating,
        "high_non_target": group_counts.get("high_non_target", 0) < high_non_target,
        "random_target": group_counts.get("random_target", 0) < random_target,
        "sibling_code_contrast": group_counts.get("sibling_code_contrast", 0) < sibling_contrast,
        "surface_matched_contrast": group_counts.get("surface_matched_contrast", 0) < surface_contrast,
        "low_activation_scoring": group_counts.get("low_activation_scoring", 0) < low_activation,
    }
    target_rates = {
        group: float(np.mean([ex["target_match"] for ex in examples if ex.get("example_group") == group]))
        for group in group_counts
        if group_counts[group] > 0 and group != "low_activation_scoring"
    }

    return {
        "p3_version": P3_VERSION,
        "packet_id": packet_id,
        "target_label": target_label,
        "latent_idx": latent_idx,
        "rank_within_label": rank,
        "dry_run": True,
        "source_packet_status": source_packet.get("p3_source_packet_status", "phase1_packet"),
        "context_limitation": CONTEXT_LIMITATION,
        "association_metrics": {
            "cohens_d": latent_row.get("cohens_d"),
            "directional_auc": latent_row.get("directional_auc"),
            "precision_at_50": latent_row.get("precision_at_50"),
        },
        "layer_metadata": _layer_metadata(layer_selection, target_label),
        "dry_run_explanation_slots": {
            "input_centric_explanation": "pending",
            "artifact_risk": "pending",
            "mi_coder_judgment": "pending",
            "final_status": "pending",
        },
        "scoring_slots": {
            "activation_prediction_score": "pending_score",
            "code_discrimination_score": "pending_score",
        },
        "examples": examples,
        "summary": {
            "group_counts": group_counts,
            "target_match_rates": target_rates,
            "scarcity": scarcity,
            "top_activating_target_match_rate": source_packet.get("summary", {}).get("top_activating_target_match_rate"),
            "duplicate_text_row_count": source_packet.get("summary", {}).get("duplicate_text_row_count"),
        },
    }


def build_p3_feature_cards(
    *,
    latents: pd.DataFrame,
    packets: dict[tuple[str, int, int], dict[str, Any]],
    features: np.ndarray,
    label_matrix: pd.DataFrame,
    records: list[dict[str, Any]],
    layer_selection: pd.DataFrame,
    labels: tuple[str, ...],
    top_activating: int,
    high_non_target: int,
    random_target: int,
    sibling_contrast: int,
    surface_contrast: int,
    low_activation: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if features.ndim != 2:
        raise ValueError(f"Feature matrix must be 2D, got shape {features.shape}")
    if len(label_matrix) != features.shape[0] or len(records) != features.shape[0]:
        raise ValueError(
            f"Row mismatch: features={features.shape[0]}, labels={len(label_matrix)}, records={len(records)}"
        )

    cards: list[dict[str, Any]] = []
    fallback_packets: list[dict[str, Any]] = []
    unresolved_missing_packets: list[dict[str, Any]] = []
    for _, row in latents.iterrows():
        target_label = str(row["target_label"])
        latent_idx = int(row["latent_idx"])
        rank = int(row["rank_within_label"])
        if latent_idx < 0 or latent_idx >= features.shape[1]:
            raise ValueError(f"latent_idx {latent_idx} outside feature dimension {features.shape[1]}")
        packet = packets.get((target_label, latent_idx, rank))
        if packet is None:
            fallback_record = {
                "target_label": target_label,
                "latent_idx": latent_idx,
                "rank_within_label": rank,
                "reason": "phase1_packet_key_not_found",
            }
            try:
                packet = _synthesise_source_packet(
                    latent_row=row,
                    features=features,
                    label_matrix=label_matrix,
                    records=records,
                    labels=labels,
                    top_activating=top_activating,
                    high_non_target=high_non_target,
                    random_target=random_target,
                )
                fallback_packets.append(fallback_record)
            except Exception as exc:  # pragma: no cover - defensive audit path
                fallback_record["error"] = repr(exc)
                unresolved_missing_packets.append(fallback_record)
                continue
        packet["p3_source_packet_status"] = (
            "fallback_generated_from_feature_store"
            if str(packet.get("evidence_source")) == "p3_generated_from_feature_store"
            else "phase1_packet"
        )
        cards.append(
            _build_card(
                latent_row=row,
                source_packet=packet,
                layer_selection=layer_selection,
                features=features,
                label_matrix=label_matrix,
                records=records,
                labels=labels,
                top_activating=top_activating,
                high_non_target=high_non_target,
                random_target=random_target,
                sibling_contrast=sibling_contrast,
                surface_contrast=surface_contrast,
                low_activation=low_activation,
            )
        )
    return cards, fallback_packets, unresolved_missing_packets


def _examples_dataframe(cards: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for card in cards:
        for example in card["examples"]:
            row = dict(example)
            row["rank_within_label"] = card["rank_within_label"]
            row["cohens_d"] = card["association_metrics"].get("cohens_d")
            row["directional_auc"] = card["association_metrics"].get("directional_auc")
            row["precision_at_50"] = card["association_metrics"].get("precision_at_50")
            rows.append(row)
    return pd.DataFrame(rows)


def _summary_dataframe(cards: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for card in cards:
        summary = card["summary"]
        row = {
            "packet_id": card["packet_id"],
            "target_label": card["target_label"],
            "latent_idx": card["latent_idx"],
            "rank_within_label": card["rank_within_label"],
            "cohens_d": card["association_metrics"].get("cohens_d"),
            "directional_auc": card["association_metrics"].get("directional_auc"),
            "precision_at_50": card["association_metrics"].get("precision_at_50"),
            "source_packet_status": card.get("source_packet_status"),
            "sae_canonical_layer": card["layer_metadata"].get("sae_canonical_layer"),
            "llama_label_specific_best_layer": card["layer_metadata"].get("llama_label_specific_best_layer"),
            "llama_early_stable_layer": card["layer_metadata"].get("llama_early_stable_layer"),
            "activation_prediction_score": "pending_score",
            "code_discrimination_score": "pending_score",
            "top_activating_target_match_rate": summary.get("top_activating_target_match_rate"),
            "duplicate_text_row_count": summary.get("duplicate_text_row_count"),
        }
        for group, count in summary["group_counts"].items():
            row[f"n_{group}"] = count
        for group, scarce in summary["scarcity"].items():
            row[f"scarce_{group}"] = bool(scarce)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_prompts(
    *,
    cards: list[dict[str, Any]],
    prompts_dir: Path,
    prompt_examples_per_group: int,
) -> list[dict[str, str]]:
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_manifest: list[dict[str, str]] = []
    for card in cards:
        slug = f"{card['target_label']}_rank{int(card['rank_within_label']):02d}_latent{card['latent_idx']}"
        path = prompts_dir / f"{slug}_input_prompt.json"
        messages = _build_input_prompt(card, prompt_examples_per_group=prompt_examples_per_group)
        _write_json(path, {"messages": messages})
        prompt_manifest.append(
            {
                "packet_id": card["packet_id"],
                "target_label": card["target_label"],
                "latent_idx": str(card["latent_idx"]),
                "prompt_path": str(path),
            }
        )
    return prompt_manifest


def _write_markdown_report(path: Path, cards: list[dict[str, Any]], summary_df: pd.DataFrame) -> None:
    lines: list[str] = [
        "# P3 SAE Feature Cards Dry-Run Report",
        "",
        "This report exports dry-run P3 feature card materials. No LLM explanation, score, or causal intervention result has been computed.",
        "",
        "## Method Boundary",
        "",
        "- Candidate features are Top20 positive Cohen's d SAE latents per core MISC label.",
        "- SAE canonical layer remains Llama layer 19 (`blocks.19.hook_resid_post`).",
        "- Llama cross-layer probe results are included only as label-level localization metadata.",
        "- Only counselor current utterance is available; prior client context is unavailable.",
        "- Scoring fields are `pending_score`; this run only builds task sets.",
        "",
        "## Coverage",
        "",
        f"- feature cards: {len(cards)}",
        f"- labels: {', '.join(sorted(summary_df['target_label'].unique())) if not summary_df.empty else 'NA'}",
        f"- fallback-generated source packets: {int((summary_df.get('source_packet_status', pd.Series(dtype=str)) == 'fallback_generated_from_feature_store').sum()) if not summary_df.empty else 0}",
        "",
        "| label | cards | mean top target-match | scarce sibling | scarce surface |",
        "|---|---:|---:|---:|---:|",
    ]
    if not summary_df.empty:
        for label, group in summary_df.groupby("target_label", sort=False):
            scarce_sibling = int(group.get("scarce_sibling_code_contrast", pd.Series(dtype=bool)).fillna(False).sum())
            scarce_surface = int(group.get("scarce_surface_matched_contrast", pd.Series(dtype=bool)).fillna(False).sum())
            match_mean = pd.to_numeric(group["top_activating_target_match_rate"], errors="coerce").mean()
            lines.append(
                f"| {label} | {len(group)} | {_fmt_float(match_mean, 3)} | {scarce_sibling} | {scarce_surface} |"
            )

    lines.extend(["", "## Feature Card Index", ""])
    for label, group in summary_df.groupby("target_label", sort=False):
        lines.extend(
            [
                f"### {label}",
                "",
                "| rank | latent | Cohen's d | dir. AUC | P@50 | evidence source | Llama best | early stable | prompts/scoring |",
                "|---:|---:|---:|---:|---:|---|---:|---:|---|",
            ]
        )
        for _, row in group.sort_values("rank_within_label").iterrows():
            lines.append(
                "| {rank} | {latent} | {d} | {auc} | {p50} | {source} | {best} | {early} | pending |".format(
                    rank=int(row["rank_within_label"]),
                    latent=int(row["latent_idx"]),
                    d=_fmt_float(row["cohens_d"]),
                    auc=_fmt_float(row["directional_auc"]),
                    p50=_fmt_float(row["precision_at_50"]),
                    source=row.get("source_packet_status", "NA"),
                    best=row.get("llama_label_specific_best_layer", "NA"),
                    early=row.get("llama_early_stable_layer", "NA"),
                )
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_p3_feature_card_export(
    *,
    latents_path: str | Path,
    packets_path: str | Path,
    feature_store_path: str | Path,
    label_matrix_path: str | Path,
    records_path: str | Path,
    layer_selection_path: str | Path,
    output_dir: str | Path,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    top_features: int | None = 20,
    top_activating: int = 20,
    high_non_target: int = 10,
    random_target: int = 10,
    sibling_contrast: int = 10,
    surface_contrast: int = 10,
    low_activation: int = 10,
    prompt_examples_per_group: int = 8,
    scoring_heldout_start: int = 8,
    scoring_examples_per_class: int = 6,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    label_tuple = tuple(label.upper() for label in labels)
    latents = _normalise_latents(pd.read_csv(latents_path), label_tuple, top_features)
    packets = _load_packets(Path(packets_path))
    features = _load_feature_matrix(Path(feature_store_path))
    label_matrix = pd.read_csv(label_matrix_path)
    records = _read_jsonl(Path(records_path))
    layer_selection = pd.read_csv(layer_selection_path) if Path(layer_selection_path).exists() else pd.DataFrame()

    cards, fallback_packets, unresolved_missing_packets = build_p3_feature_cards(
        latents=latents,
        packets=packets,
        features=features,
        label_matrix=label_matrix,
        records=records,
        layer_selection=layer_selection,
        labels=label_tuple,
        top_activating=top_activating,
        high_non_target=high_non_target,
        random_target=random_target,
        sibling_contrast=sibling_contrast,
        surface_contrast=surface_contrast,
        low_activation=low_activation,
    )
    summary_df = _summary_dataframe(cards)
    examples_df = _examples_dataframe(cards)
    scoring_tasks: list[dict[str, Any]] = []
    for card in cards:
        scoring_tasks.extend(
            _build_scoring_tasks(
                card,
                heldout_start=scoring_heldout_start,
                per_class=scoring_examples_per_class,
            )
        )

    paths = {
        "packets_jsonl": output_path / "p3_feature_card_packets.jsonl",
        "examples_csv": output_path / "p3_feature_card_examples.csv",
        "prompts_dir": output_path / "p3_input_explanation_prompts",
        "scoring_tasks_jsonl": output_path / "p3_scoring_tasks.jsonl",
        "report_md": output_path / "p3_feature_cards_dryrun.md",
        "summary_csv": output_path / "p3_feature_card_summary.csv",
        "manifest_json": output_path / "manifest.json",
    }

    _write_jsonl(paths["packets_jsonl"], cards)
    examples_df.to_csv(paths["examples_csv"], index=False)
    summary_df.to_csv(paths["summary_csv"], index=False)
    _write_jsonl(paths["scoring_tasks_jsonl"], scoring_tasks)
    prompt_manifest = _write_prompts(
        cards=cards,
        prompts_dir=paths["prompts_dir"],
        prompt_examples_per_group=prompt_examples_per_group,
    )
    _write_markdown_report(paths["report_md"], cards, summary_df)

    scarcity_counts: dict[str, int] = defaultdict(int)
    for card in cards:
        for group, scarce in card["summary"]["scarcity"].items():
            if scarce:
                scarcity_counts[group] += 1

    manifest = {
        "analysis": P3_VERSION,
        "dry_run": True,
        "labels": list(label_tuple),
        "candidate_source": "top20_positive_cohens_d_latents",
        "context_limitation": CONTEXT_LIMITATION,
        "inputs": {
            "latents": str(latents_path),
            "packets": str(packets_path),
            "feature_store": str(feature_store_path),
            "label_matrix": str(label_matrix_path),
            "records": str(records_path),
            "layer_selection": str(layer_selection_path),
        },
        "outputs": {key: str(value) for key, value in paths.items()},
        "n_cards": int(len(cards)),
        "n_expected_cards": int(len(latents)),
        "n_fallback_packets": int(len(fallback_packets)),
        "fallback_packets": fallback_packets[:50],
        "n_missing_packets": int(len(unresolved_missing_packets)),
        "missing_packets": unresolved_missing_packets[:20],
        "n_example_rows": int(len(examples_df)),
        "n_scoring_tasks": int(len(scoring_tasks)),
        "n_prompts": int(len(prompt_manifest)),
        "scarcity_counts": dict(scarcity_counts),
        "parameters": {
            "top_features": top_features,
            "top_activating": top_activating,
            "high_non_target": high_non_target,
            "random_target": random_target,
            "sibling_contrast": sibling_contrast,
            "surface_contrast": surface_contrast,
            "low_activation": low_activation,
            "prompt_examples_per_group": prompt_examples_per_group,
            "scoring_heldout_start": scoring_heldout_start,
            "scoring_examples_per_class": scoring_examples_per_class,
        },
    }
    _write_json(paths["manifest_json"], manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate P3 dry-run SAE feature cards for MISC latents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--latents",
        default=f"{DEFAULT_INPUT_DIR}/top20_cohensd_latents_by_label.csv",
    )
    parser.add_argument(
        "--packets",
        default=f"{DEFAULT_EVIDENCE_DIR}/latent_evidence_packets_labeled.jsonl",
    )
    parser.add_argument(
        "--feature-store",
        default="outputs/misc_full_sae_eval/feature_store/utterance_features.pt",
    )
    parser.add_argument("--label-matrix", default="outputs/misc_full_sae_eval/label_matrix.csv")
    parser.add_argument("--records", default="outputs/misc_full_sae_eval/records.jsonl")
    parser.add_argument(
        "--layer-selection",
        default="outputs/layer_selection_strategy/llama_layer_selection.csv",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--top-features", type=int, default=20)
    parser.add_argument("--top-activating", type=int, default=20)
    parser.add_argument("--high-non-target", type=int, default=10)
    parser.add_argument("--random-target", type=int, default=10)
    parser.add_argument("--sibling-contrast", type=int, default=10)
    parser.add_argument("--surface-contrast", type=int, default=10)
    parser.add_argument("--low-activation", type=int, default=10)
    parser.add_argument("--prompt-examples-per-group", type=int, default=8)
    parser.add_argument("--scoring-heldout-start", type=int, default=8)
    parser.add_argument("--scoring-examples-per-class", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_p3_feature_card_export(
        latents_path=args.latents,
        packets_path=args.packets,
        feature_store_path=args.feature_store,
        label_matrix_path=args.label_matrix,
        records_path=args.records,
        layer_selection_path=args.layer_selection,
        output_dir=args.output_dir,
        labels=tuple(args.labels),
        top_features=args.top_features,
        top_activating=args.top_activating,
        high_non_target=args.high_non_target,
        random_target=args.random_target,
        sibling_contrast=args.sibling_contrast,
        surface_contrast=args.surface_contrast,
        low_activation=args.low_activation,
        prompt_examples_per_group=args.prompt_examples_per_group,
        scoring_heldout_start=args.scoring_heldout_start,
        scoring_examples_per_class=args.scoring_examples_per_class,
    )
    print("Completed P3 dry-run feature card export.")
    print(f"Output dir: {args.output_dir}")
    print(f"Feature cards: {manifest['n_cards']}")
    print(f"Example rows: {manifest['n_example_rows']}")
    print(f"Scoring tasks: {manifest['n_scoring_tasks']}")
    print(f"Prompts: {manifest['n_prompts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
