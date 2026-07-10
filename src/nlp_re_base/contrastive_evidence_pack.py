"""Contrastive evidence packs for stable-core SAE latent interpretation.

This module builds the local, auditable inputs for the revised P3 workflow.
It keeps label-bearing audit metadata internally, while exposing only
label-blind samples to the LLM explainer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_LABELS: tuple[str, ...] = ("RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF")
PRIMARY_LABELS: tuple[str, ...] = ("QU", "QUO", "QUC", "RE", "REC", "AF")
SECONDARY_LABELS: tuple[str, ...] = ("SU", "GI", "RES")
CONTRASTIVE_P3_VERSION = "contrastive_latent_interp_v1_stable_core"

VISIBLE_SAMPLE_KEYS: tuple[str, ...] = ("id", "tag", "activation", "text")
FORBIDDEN_EXPLAINER_KEYS: set[str] = {
    "target_label",
    "label",
    "target_match",
    "active_labels",
    "predicted_code",
    "predicted_subcode",
    "rationale",
    "record_id",
    "file_id",
    "source_file",
    "source_split",
    "source_line",
    "quality_label",
    "confidence",
    "ground_truth_activate",
    "internal_source",
}
MISC_LABEL_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(?:RE|RES|REC|QU|QUO|QUC|GI|SU|AF)(?![A-Za-z0-9_])")

SIBLING_LABELS: dict[str, tuple[str, ...]] = {
    "QU": ("QUO", "QUC"),
    "QUO": ("QU", "QUC"),
    "QUC": ("QU", "QUO"),
    "RE": ("RES", "REC", "GI", "AF"),
    "RES": ("RE", "REC", "AF"),
    "REC": ("RE", "RES", "AF"),
    "GI": ("QU", "SU", "RE"),
    "SU": ("GI", "AF", "QU"),
    "AF": ("RE", "RES", "REC", "SU"),
}

SURFACE_PATTERNS: dict[str, re.Pattern[str]] = {
    "question": re.compile(
        r"(\?|^|\b)(what|how|why|when|where|who|do|does|did|are|is|can|could|would|will|have|has)\b",
        flags=re.IGNORECASE,
    ),
    "reflection": re.compile(
        r"\b(you|you're|you are|your|sounds like|it sounds|seems like|it seems|what i hear|so you|so it)\b",
        flags=re.IGNORECASE,
    ),
    "positive": re.compile(
        r"\b(good|great|nice|proud|appreciate|strength|strong|effort|excellent|amazing)\b",
        flags=re.IGNORECASE,
    ),
    "information": re.compile(
        r"\b(because|means|need to|should|could|can|if you|when you|information|risk|helps|works|recommend)\b",
        flags=re.IGNORECASE,
    ),
}

LABEL_SURFACE_KEYS: dict[str, tuple[str, ...]] = {
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


@dataclass(frozen=True)
class ContrastiveEvidenceConfig:
    labels: tuple[str, ...] = DEFAULT_LABELS
    stable_role: str = "stable_core"
    expected_stable_core_count: int | None = 303
    random_state: int = 42
    active_high: int = 5
    active_mid: int = 4
    active_low: int = 2
    near_miss_surface: int = 4
    nonactive_label_match: int = 3
    nonactive_random: int = 2
    heldout_active_high: int = 5
    heldout_active_mid: int = 5
    heldout_near_miss: int = 5
    heldout_label_match: int = 5


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
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


def load_feature_matrix(path: str | Path) -> np.ndarray:
    feature_path = Path(path)
    if feature_path.suffix == ".npy":
        return np.asarray(np.load(feature_path), dtype=np.float32)
    if feature_path.suffix == ".npz":
        payload = np.load(feature_path)
        for key in ("utterance_features", "features", "X"):
            if key in payload:
                return np.asarray(payload[key], dtype=np.float32)
        raise KeyError(f"{feature_path} does not contain utterance_features/features/X")
    if feature_path.suffix == ".pt":
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
                raise KeyError(f"{feature_path} does not contain utterance_features/features/X")
        else:
            raise TypeError(f"Unsupported torch payload in {feature_path}: {type(payload).__name__}")
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().float().numpy()
        return np.asarray(tensor, dtype=np.float32)
    raise ValueError(f"Unsupported feature matrix extension: {feature_path.suffix}")


def stable_seed(*parts: Any) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)


def normalise_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def numeric_positive(value: Any) -> bool:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0]
    return float(number) > 0


def record_value(record: dict[str, Any], label_row: pd.Series, *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
        if key in label_row and pd.notna(label_row[key]):
            return label_row[key]
    return default


def active_labels(label_row: pd.Series, labels: tuple[str, ...]) -> str:
    return ",".join(label for label in labels if label in label_row and numeric_positive(label_row[label]))


def label_mask(label_matrix: pd.DataFrame, label: str) -> np.ndarray:
    if label not in label_matrix.columns:
        return np.zeros(len(label_matrix), dtype=bool)
    return pd.to_numeric(label_matrix[label], errors="coerce").fillna(0).to_numpy(dtype=float) > 0


def prepare_pack_context(
    label_matrix: pd.DataFrame,
    records: list[dict[str, Any]],
    labels: tuple[str, ...],
) -> dict[str, Any]:
    texts: list[str] = []
    active_by_row: list[str] = []
    for idx in range(len(label_matrix)):
        label_row = label_matrix.iloc[idx]
        record = records[idx]
        texts.append(str(record_value(record, label_row, "unit_text", "text", "utterance", default="")))
        active_by_row.append(active_labels(label_row, labels))
    word_counts = np.asarray([len(normalise_text(text).split()) for text in texts if text], dtype=float)
    median_words = float(np.median(word_counts)) if len(word_counts) else None
    label_masks = {label: label_mask(label_matrix, label) for label in labels}
    surface_scores = {
        label: np.asarray([surface_score(text, label, median_words) for text in texts], dtype=np.int32)
        for label in labels
    }
    return {
        "texts": texts,
        "normalised_texts": [normalise_text(text) for text in texts],
        "active_labels": active_by_row,
        "label_masks": label_masks,
        "surface_scores": surface_scores,
        "median_words": median_words,
    }


def normalise_latents(
    latents: pd.DataFrame,
    labels: tuple[str, ...],
    *,
    stable_role: str = "stable_core",
    expected_count: int | None = None,
) -> pd.DataFrame:
    label_col = "target_label" if "target_label" in latents.columns else "label"
    missing = sorted({label_col, "latent_idx"}.difference(latents.columns))
    if missing:
        raise ValueError(f"Latents table missing columns: {missing}")

    out = latents.copy()
    out["target_label"] = out[label_col].astype(str).str.upper()
    out["latent_idx"] = pd.to_numeric(out["latent_idx"], errors="coerce").fillna(-1).astype(int)
    if stable_role and stable_role.lower() not in {"all", "*"} and "stable_set_role" in out.columns:
        out = out[out["stable_set_role"].astype(str).str.lower() == stable_role.lower()].copy()

    if "rank_within_label" in out.columns:
        out["rank_within_label"] = pd.to_numeric(out["rank_within_label"], errors="coerce").fillna(-1).astype(int)
    elif "full_data_rank" in out.columns:
        out["rank_within_label"] = pd.to_numeric(out["full_data_rank"], errors="coerce").fillna(-1).astype(int)
    else:
        out = out.sort_values(["target_label", "latent_idx"], kind="mergesort")
        out["rank_within_label"] = out.groupby("target_label").cumcount() + 1

    numeric_cols = (
        "cohens_d",
        "abs_cohens_d",
        "auc",
        "directional_auc",
        "precision_at_10",
        "precision_at_50",
        "full_data_rank",
        "inclusion_frequency",
        "inclusion_count",
        "n_half_runs",
        "cohens_d_ci_lo",
        "cohens_d_ci_hi",
        "top_k_star",
        "cross_quality_max_auc_drop",
    )
    for col in numeric_cols:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in (
        "label_selection_status",
        "stable_claim_allowed",
        "ci_supported",
        "cross_quality_stable",
        "cross_quality_risk_or_missing",
        "stable_set_role",
    ):
        if col not in out.columns:
            out[col] = ""

    label_order = {label.upper(): idx for idx, label in enumerate(labels)}
    out = out[out["target_label"].isin(label_order)].copy()
    out["_label_order"] = out["target_label"].map(label_order)
    out = out.sort_values(["_label_order", "rank_within_label", "latent_idx"], kind="mergesort")
    out = out.drop(columns=["_label_order"]).reset_index(drop=True)
    if expected_count is not None and len(out) != int(expected_count):
        raise ValueError(
            f"Expected {expected_count} rows after stable_core filter, got {len(out)}. "
            "Check stable_topk_latent_set.csv version before running P3."
        )
    return out


def validate_matrix_alignment(features: np.ndarray, label_matrix: pd.DataFrame, records: list[dict[str, Any]]) -> None:
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {features.shape}")
    if len(label_matrix) != features.shape[0]:
        raise ValueError(f"Label rows {len(label_matrix)} do not match feature rows {features.shape[0]}")
    if len(records) != features.shape[0]:
        raise ValueError(f"Record rows {len(records)} do not match feature rows {features.shape[0]}")


def surface_score(text: str, target_label: str, median_words: float | None = None) -> int:
    score = 0
    normalised = normalise_text(text)
    for key in LABEL_SURFACE_KEYS.get(target_label, ()):
        if SURFACE_PATTERNS[key].search(normalised):
            score += 3
    if "?" in normalised and target_label in {"QU", "QUO", "QUC"}:
        score += 2
    if median_words and median_words > 0:
        n_words = len(normalised.split())
        if 0.6 * median_words <= n_words <= 1.6 * median_words:
            score += 1
    return int(score)


def detection_threshold(activations: np.ndarray) -> float:
    nonzero = np.asarray(activations[activations > 0], dtype=np.float32)
    if len(nonzero) == 0:
        return 0.0
    return float(np.median(nonzero))


def _order_desc(activations: np.ndarray, candidates: np.ndarray, limit: int, exclude: set[int]) -> list[int]:
    if limit <= 0 or len(candidates) == 0:
        return []
    pool = np.asarray([int(idx) for idx in candidates.tolist() if int(idx) not in exclude], dtype=np.int64)
    if len(pool) == 0:
        return []
    order = np.lexsort((pool, -activations[pool]))
    return [int(idx) for idx in pool[order[: min(limit, len(order))]].tolist()]


def _order_asc(activations: np.ndarray, candidates: np.ndarray, limit: int, exclude: set[int]) -> list[int]:
    if limit <= 0 or len(candidates) == 0:
        return []
    pool = np.asarray([int(idx) for idx in candidates.tolist() if int(idx) not in exclude], dtype=np.int64)
    if len(pool) == 0:
        return []
    order = np.lexsort((pool, activations[pool]))
    return [int(idx) for idx in pool[order[: min(limit, len(order))]].tolist()]


def _range_candidates(activations: np.ndarray, low_q: float, high_q: float) -> np.ndarray:
    active = np.flatnonzero(activations > 0).astype(np.int64)
    if len(active) == 0:
        return active
    values = activations[active]
    low = float(np.quantile(values, low_q))
    high = float(np.quantile(values, high_q))
    if low > high:
        low, high = high, low
    return active[(values >= low) & (values <= high)]


def _near_quantile_candidates(activations: np.ndarray, quantile: float) -> np.ndarray:
    active = np.flatnonzero(activations > 0).astype(np.int64)
    if len(active) == 0:
        return active
    target = float(np.quantile(activations[active], quantile))
    distances = np.abs(activations[active] - target)
    order = np.lexsort((active, distances))
    return active[order]


def _near_miss_order(
    *,
    activations: np.ndarray,
    label_matrix: pd.DataFrame,
    records: list[dict[str, Any]],
    target_label: str,
    labels: tuple[str, ...],
    threshold: float,
    exclude: set[int],
    context: dict[str, Any] | None = None,
) -> list[int]:
    if context is not None:
        target = np.asarray(context["label_masks"].get(target_label, np.zeros(len(label_matrix), dtype=bool)), dtype=bool)
    else:
        target = label_mask(label_matrix, target_label)
    sibling = np.zeros(len(label_matrix), dtype=bool)
    for label in SIBLING_LABELS.get(target_label, ()):
        if label in labels:
            if context is not None:
                sibling |= np.asarray(context["label_masks"].get(label, np.zeros(len(label_matrix), dtype=bool)), dtype=bool)
            else:
                sibling |= label_mask(label_matrix, label)
    if context is not None:
        scores = np.asarray(context["surface_scores"].get(target_label, np.zeros(len(label_matrix), dtype=np.int32)))
    else:
        texts = [
            str(record_value(records[idx], label_matrix.iloc[idx], "unit_text", "text", "utterance", default=""))
            for idx in range(len(label_matrix))
        ]
        target_words = np.asarray([len(normalise_text(text).split()) for text in texts if text], dtype=float)
        median_words = float(np.median(target_words)) if len(target_words) else None
        scores = np.asarray([surface_score(text, target_label, median_words) for text in texts], dtype=np.int32)
    candidates = np.flatnonzero((activations <= threshold) & (~target) & ((scores > 0) | sibling)).astype(np.int64)
    candidates = np.asarray([int(idx) for idx in candidates.tolist() if int(idx) not in exclude], dtype=np.int64)
    if len(candidates) == 0:
        return []
    sibling_bonus = sibling[candidates].astype(np.int32) * 2
    total_scores = scores[candidates] + sibling_bonus
    # Prefer surface/sibling matches, then nonactive examples closest to the threshold.
    order = np.lexsort((candidates, -activations[candidates], -total_scores))
    return [int(idx) for idx in candidates[order].tolist()]


def _sample_row(
    *,
    sample_id: str,
    tag: str,
    internal_source: str,
    target_label: str,
    latent_idx: int,
    row_idx: int,
    activation: float,
    threshold: float,
    label_matrix: pd.DataFrame,
    records: list[dict[str, Any]],
    labels: tuple[str, ...],
    heldout: bool,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    label_row = label_matrix.iloc[int(row_idx)]
    record = records[int(row_idx)]
    text = (
        context["texts"][int(row_idx)]
        if context is not None
        else record_value(record, label_row, "unit_text", "text", "utterance", default="")
    )
    return {
        "id": sample_id,
        "tag": tag,
        "row_idx": int(row_idx),
        "activation": float(activation),
        "activation_threshold": float(threshold),
        "ground_truth_activate": int(float(activation) > float(threshold)),
        "text": str(text),
        "target_label": target_label,
        "latent_idx": int(latent_idx),
        "target_match": int(target_label in label_matrix.columns and numeric_positive(label_row[target_label])),
        "active_labels": (
            context["active_labels"][int(row_idx)]
            if context is not None
            else active_labels(label_row, labels)
        ),
        "record_id": record_value(record, label_row, "record_id", "sample_id", default=f"row_{row_idx}"),
        "file_id": record_value(record, label_row, "file_id", default=""),
        "source_split": record_value(record, label_row, "source_split", default=""),
        "source_file": record_value(record, label_row, "source_file", default=""),
        "source_line": record_value(record, label_row, "source_line", default=""),
        "quality_label": record_value(record, label_row, "quality_label", default=""),
        "predicted_code": record_value(record, label_row, "predicted_code", default=""),
        "predicted_subcode": record_value(record, label_row, "predicted_subcode", default=""),
        "internal_source": internal_source,
        "heldout": bool(heldout),
    }


def project_sample_for_explainer(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(sample["id"]),
        "tag": str(sample["tag"]),
        "activation": float(sample["activation"]),
        "text": str(sample["text"]),
    }


def assert_label_blind_samples(samples: list[dict[str, Any]]) -> None:
    for idx, sample in enumerate(samples):
        keys = set(sample)
        missing = set(VISIBLE_SAMPLE_KEYS).difference(keys)
        if missing:
            raise AssertionError(f"samples_for_explainer[{idx}] missing keys: {sorted(missing)}")
        extra = keys.difference(VISIBLE_SAMPLE_KEYS)
        if extra:
            raise AssertionError(f"samples_for_explainer[{idx}] has non-blind extra keys: {sorted(extra)}")
        overlap = keys.intersection(FORBIDDEN_EXPLAINER_KEYS)
        if overlap:
            raise AssertionError(f"samples_for_explainer[{idx}] leaked keys: {sorted(overlap)}")
        if sample.get("tag") == "NONACTIVE_LABEL_MATCH":
            raise AssertionError("NONACTIVE_LABEL_MATCH must not be exposed to explainer")


def _append_samples(
    *,
    selected: list[dict[str, Any]],
    used_rows: set[int],
    used_texts: set[str],
    id_state: dict[str, int],
    row_indices: list[int],
    tag: str,
    internal_source: str,
    target_label: str,
    latent_idx: int,
    activations: np.ndarray,
    threshold: float,
    label_matrix: pd.DataFrame,
    records: list[dict[str, Any]],
    labels: tuple[str, ...],
    heldout: bool,
    prefix: str,
    max_count: int | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    for row_idx in row_indices:
        if max_count is not None and len(selected) >= int(max_count):
            break
        if int(row_idx) in used_rows:
            continue
        text = (
            context["texts"][int(row_idx)]
            if context is not None
            else record_value(records[int(row_idx)], label_matrix.iloc[int(row_idx)], "unit_text", "text", "utterance", default="")
        )
        normalised = normalise_text(text)
        if not normalised or normalised in used_texts:
            continue
        next_id = int(id_state.get("next", 1))
        sample_id = f"{prefix}{next_id:03d}"
        id_state["next"] = next_id + 1
        selected.append(
            _sample_row(
                sample_id=sample_id,
                tag=tag,
                internal_source=internal_source,
                target_label=target_label,
                latent_idx=latent_idx,
                row_idx=int(row_idx),
                activation=float(activations[int(row_idx)]),
                threshold=threshold,
                label_matrix=label_matrix,
                records=records,
                labels=labels,
                heldout=heldout,
                context=context,
            )
        )
        used_rows.add(int(row_idx))
        used_texts.add(normalised)


def _merge_ordered_rows(*orders: Iterable[int]) -> list[int]:
    merged: list[int] = []
    seen: set[int] = set()
    for order in orders:
        for value in order:
            row_idx = int(value)
            if row_idx not in seen:
                seen.add(row_idx)
                merged.append(row_idx)
    return merged


def _build_single_pack(
    *,
    packet_number: int,
    latent_row: pd.Series,
    features: np.ndarray,
    label_matrix: pd.DataFrame,
    records: list[dict[str, Any]],
    labels: tuple[str, ...],
    config: ContrastiveEvidenceConfig,
    context: dict[str, Any],
) -> dict[str, Any]:
    target_label = str(latent_row["target_label"]).upper()
    latent_idx = int(latent_row["latent_idx"])
    rank = int(latent_row["rank_within_label"])
    activations = np.asarray(features[:, latent_idx], dtype=np.float32)
    threshold = detection_threshold(activations)
    target = label_mask(label_matrix, target_label)
    nonactive = activations <= threshold
    positive = activations > threshold

    packet_id = f"ctli_{packet_number:04d}"
    used_rows: set[int] = set()
    used_texts: set[str] = set()
    evidence_id_state = {"next": 1}
    samples: list[dict[str, Any]] = []

    positive_rows = np.flatnonzero(positive).astype(np.int64)
    active_high = _order_desc(activations, positive_rows, len(positive_rows), used_rows)
    _append_samples(
        selected=samples,
        used_rows=used_rows,
        used_texts=used_texts,
        id_state=evidence_id_state,
        row_indices=active_high,
        tag="ACTIVE_HIGH",
        internal_source="top_activation",
        target_label=target_label,
        latent_idx=latent_idx,
        activations=activations,
        threshold=threshold,
        label_matrix=label_matrix,
        records=records,
        labels=labels,
        heldout=False,
        prefix="s",
        max_count=config.active_high,
        context=context,
    )

    mid_candidates = _range_candidates(activations, 0.50, 0.75)
    active_mid = _merge_ordered_rows(
        _order_desc(activations, mid_candidates, len(mid_candidates), used_rows),
        [int(idx) for idx in _near_quantile_candidates(activations, 0.625).tolist() if bool(positive[int(idx)])],
    )
    _append_samples(
        selected=samples,
        used_rows=used_rows,
        used_texts=used_texts,
        id_state=evidence_id_state,
        row_indices=active_mid,
        tag="ACTIVE_MID",
        internal_source="mid_nonzero_activation",
        target_label=target_label,
        latent_idx=latent_idx,
        activations=activations,
        threshold=threshold,
        label_matrix=label_matrix,
        records=records,
        labels=labels,
        heldout=False,
        prefix="s",
        max_count=config.active_high + config.active_mid,
        context=context,
    )

    low_candidates = _range_candidates(activations, 0.05, 0.20)
    active_low = _merge_ordered_rows(
        _order_desc(activations, low_candidates, len(low_candidates), used_rows),
        _near_quantile_candidates(activations, 0.125).tolist(),
    )
    _append_samples(
        selected=samples,
        used_rows=used_rows,
        used_texts=used_texts,
        id_state=evidence_id_state,
        row_indices=active_low,
        tag="ACTIVE_LOW",
        internal_source="low_nonzero_boundary",
        target_label=target_label,
        latent_idx=latent_idx,
        activations=activations,
        threshold=threshold,
        label_matrix=label_matrix,
        records=records,
        labels=labels,
        heldout=False,
        prefix="s",
        max_count=config.active_high + config.active_mid + config.active_low,
        context=context,
    )

    near_order = _near_miss_order(
        activations=activations,
        label_matrix=label_matrix,
        records=records,
        target_label=target_label,
        labels=labels,
        threshold=threshold,
        exclude=used_rows,
        context=context,
    )
    _append_samples(
        selected=samples,
        used_rows=used_rows,
        used_texts=used_texts,
        id_state=evidence_id_state,
        row_indices=near_order,
        tag="NONACTIVE_NEAR_MISS",
        internal_source="surface_or_adjacent_nonactive",
        target_label=target_label,
        latent_idx=latent_idx,
        activations=activations,
        threshold=threshold,
        label_matrix=label_matrix,
        records=records,
        labels=labels,
        heldout=False,
        prefix="s",
        max_count=config.active_high + config.active_mid + config.active_low + config.near_miss_surface,
        context=context,
    )

    label_nonactive = np.flatnonzero(target & nonactive).astype(np.int64)
    label_match_rows = _order_asc(activations, label_nonactive, len(label_nonactive), used_rows)
    _append_samples(
        selected=samples,
        used_rows=used_rows,
        used_texts=used_texts,
        id_state=evidence_id_state,
        row_indices=label_match_rows,
        tag="NONACTIVE_NEAR_MISS",
        internal_source="target_label_positive_nonactive_hidden_from_explainer",
        target_label=target_label,
        latent_idx=latent_idx,
        activations=activations,
        threshold=threshold,
        label_matrix=label_matrix,
        records=records,
        labels=labels,
        heldout=False,
        prefix="s",
        max_count=(
            config.active_high
            + config.active_mid
            + config.active_low
            + config.near_miss_surface
            + config.nonactive_label_match
        ),
        context=context,
    )

    random_pool = np.flatnonzero(nonactive).astype(np.int64)
    random_rows = _order_asc(activations, random_pool, len(random_pool), used_rows)
    _append_samples(
        selected=samples,
        used_rows=used_rows,
        used_texts=used_texts,
        id_state=evidence_id_state,
        row_indices=random_rows,
        tag="NONACTIVE_RANDOM",
        internal_source="lowest_activation_random_nonactive",
        target_label=target_label,
        latent_idx=latent_idx,
        activations=activations,
        threshold=threshold,
        label_matrix=label_matrix,
        records=records,
        labels=labels,
        heldout=False,
        prefix="s",
        max_count=(
            config.active_high
            + config.active_mid
            + config.active_low
            + config.near_miss_surface
            + config.nonactive_label_match
            + config.nonactive_random
        ),
        context=context,
    )

    evidence_rows = {int(sample["row_idx"]) for sample in samples}
    heldout_used: set[int] = set(evidence_rows)
    heldout_id_state = {"next": 1}
    heldout: dict[str, list[dict[str, Any]]] = {
        "ACTIVE_HIGH": [],
        "ACTIVE_MID": [],
        "NONACTIVE_NEAR_MISS": [],
        "NONACTIVE_LABEL_MATCH": [],
    }

    heldout_high = _order_desc(activations, positive_rows, len(positive_rows), heldout_used)
    _append_samples(
        selected=heldout["ACTIVE_HIGH"],
        used_rows=heldout_used,
        used_texts=used_texts,
        id_state=heldout_id_state,
        row_indices=heldout_high,
        tag="ACTIVE_HIGH",
        internal_source="heldout_top_activation",
        target_label=target_label,
        latent_idx=latent_idx,
        activations=activations,
        threshold=threshold,
        label_matrix=label_matrix,
        records=records,
        labels=labels,
        heldout=True,
        prefix="u",
        max_count=config.heldout_active_high,
        context=context,
    )

    heldout_mid_candidates = _range_candidates(activations, 0.50, 0.75)
    heldout_mid = _merge_ordered_rows(
        _order_desc(activations, heldout_mid_candidates, len(heldout_mid_candidates), heldout_used),
        [int(idx) for idx in _near_quantile_candidates(activations, 0.625).tolist() if bool(positive[int(idx)])],
    )
    _append_samples(
        selected=heldout["ACTIVE_MID"],
        used_rows=heldout_used,
        used_texts=used_texts,
        id_state=heldout_id_state,
        row_indices=heldout_mid,
        tag="ACTIVE_MID",
        internal_source="heldout_mid_nonzero_activation",
        target_label=target_label,
        latent_idx=latent_idx,
        activations=activations,
        threshold=threshold,
        label_matrix=label_matrix,
        records=records,
        labels=labels,
        heldout=True,
        prefix="u",
        max_count=config.heldout_active_mid,
        context=context,
    )

    heldout_near_order = _near_miss_order(
        activations=activations,
        label_matrix=label_matrix,
        records=records,
        target_label=target_label,
        labels=labels,
        threshold=threshold,
        exclude=heldout_used,
        context=context,
    )
    _append_samples(
        selected=heldout["NONACTIVE_NEAR_MISS"],
        used_rows=heldout_used,
        used_texts=used_texts,
        id_state=heldout_id_state,
        row_indices=heldout_near_order,
        tag="NONACTIVE_NEAR_MISS",
        internal_source="heldout_surface_or_adjacent_nonactive",
        target_label=target_label,
        latent_idx=latent_idx,
        activations=activations,
        threshold=threshold,
        label_matrix=label_matrix,
        records=records,
        labels=labels,
        heldout=True,
        prefix="u",
        max_count=config.heldout_near_miss,
        context=context,
    )

    heldout_label_rows = _order_asc(activations, label_nonactive, len(label_nonactive), heldout_used)
    _append_samples(
        selected=heldout["NONACTIVE_LABEL_MATCH"],
        used_rows=heldout_used,
        used_texts=used_texts,
        id_state=heldout_id_state,
        row_indices=heldout_label_rows,
        tag="NONACTIVE_LABEL_MATCH",
        internal_source="heldout_target_label_positive_nonactive_hidden_from_scorer",
        target_label=target_label,
        latent_idx=latent_idx,
        activations=activations,
        threshold=threshold,
        label_matrix=label_matrix,
        records=records,
        labels=labels,
        heldout=True,
        prefix="u",
        max_count=config.heldout_label_match,
        context=context,
    )

    heldout_rows = {
        int(sample["row_idx"])
        for group in heldout.values()
        for sample in group
    }
    overlap = evidence_rows.intersection(heldout_rows)
    if overlap:
        raise AssertionError(f"Evidence/heldout row overlap for latent {latent_idx}: {sorted(overlap)[:10]}")
    evidence_texts = {normalise_text(sample["text"]) for sample in samples}
    heldout_samples = [sample for group in heldout.values() for sample in group]
    heldout_texts = {normalise_text(sample["text"]) for sample in heldout_samples}
    heldout_ids = [str(sample["id"]) for sample in heldout_samples]
    if len(evidence_texts) != len(samples):
        raise AssertionError(f"Duplicate evidence text for latent {latent_idx}")
    if len(heldout_texts) != len(heldout_samples):
        raise AssertionError(f"Duplicate heldout text for latent {latent_idx}")
    text_overlap = evidence_texts.intersection(heldout_texts)
    if text_overlap:
        raise AssertionError(f"Evidence/heldout text overlap for latent {latent_idx}: {sorted(text_overlap)[:3]}")
    if len(heldout_ids) != len(set(heldout_ids)):
        raise AssertionError(f"Duplicate heldout sample id for latent {latent_idx}")

    samples_for_explainer = [project_sample_for_explainer(sample) for sample in samples]
    assert_label_blind_samples(samples_for_explainer)

    requested_counts = {
        "ACTIVE_HIGH": config.active_high,
        "ACTIVE_MID": config.active_mid,
        "ACTIVE_LOW": config.active_low,
        "NONACTIVE_NEAR_MISS": config.near_miss_surface + config.nonactive_label_match,
        "NONACTIVE_RANDOM": config.nonactive_random,
    }
    actual_counts = {
        tag: sum(1 for sample in samples if sample["tag"] == tag)
        for tag in ("ACTIVE_HIGH", "ACTIVE_MID", "ACTIVE_LOW", "NONACTIVE_NEAR_MISS", "NONACTIVE_RANDOM")
    }
    heldout_counts = {tag: len(rows) for tag, rows in heldout.items()}
    heldout_truth = {
        tag: int(sum(int(sample["ground_truth_activate"]) for sample in rows))
        for tag, rows in heldout.items()
    }
    scarcity = {
        tag: int(actual_counts.get(tag, 0)) < int(requested_counts.get(tag, 0))
        for tag in requested_counts
    } | {
        f"heldout_{tag}": int(heldout_counts.get(tag, 0))
        < int(
            {
                "ACTIVE_HIGH": config.heldout_active_high,
                "ACTIVE_MID": config.heldout_active_mid,
                "NONACTIVE_NEAR_MISS": config.heldout_near_miss,
                "NONACTIVE_LABEL_MATCH": config.heldout_label_match,
            }[tag]
        )
        for tag in heldout
    }
    n_heldout_positive = int(sum(int(sample["ground_truth_activate"]) for sample in heldout_samples))
    n_heldout_negative = int(len(heldout_samples) - n_heldout_positive)
    interpretability_eligible = not any(scarcity[tag] for tag in requested_counts)
    scorer_eligible = bool(
        not any(scarcity[f"heldout_{tag}"] for tag in heldout)
        and n_heldout_positive == config.heldout_active_high + config.heldout_active_mid
        and n_heldout_negative == config.heldout_near_miss + config.heldout_label_match
    )

    return {
        "packet_id": packet_id,
        "latent_idx": latent_idx,
        "target_label": target_label,
        "rank_within_label": rank,
        "stable_set_role": str(latent_row.get("stable_set_role", "")),
        "label_selection_status": str(latent_row.get("label_selection_status", "")),
        "tier": "primary" if target_label in PRIMARY_LABELS else "secondary",
        "auc": latent_row.get("auc"),
        "directional_auc": latent_row.get("directional_auc"),
        "cohens_d": latent_row.get("cohens_d"),
        "abs_cohens_d": latent_row.get("abs_cohens_d"),
        "precision_at_50": latent_row.get("precision_at_50"),
        "inclusion_frequency": latent_row.get("inclusion_frequency"),
        "inclusion_count": latent_row.get("inclusion_count"),
        "n_half_runs": latent_row.get("n_half_runs"),
        "cohens_d_ci_lo": latent_row.get("cohens_d_ci_lo"),
        "cohens_d_ci_hi": latent_row.get("cohens_d_ci_hi"),
        "top_k_star": latent_row.get("top_k_star"),
        "activation_threshold_policy": "q50_nonzero_activation",
        "q50_nonzero_activation": threshold,
        "n_nonzero_activation": int(np.count_nonzero(activations > 0)),
        "client_context_available": False,
        "context_limitation": (
            "Only counselor current utterance is available; prior client context is unavailable. "
            "RES/REC/RE functional claims need extra caution."
        ),
        "samples_internal": samples,
        "samples_for_explainer": samples_for_explainer,
        "heldout_internal_by_tag": heldout,
        "summary": {
            "requested_counts": requested_counts,
            "actual_counts": actual_counts,
            "heldout_counts": heldout_counts,
            "heldout_ground_truth_positive_counts": heldout_truth,
            "n_evidence_rows": int(len(evidence_rows)),
            "n_heldout_rows": int(len(heldout_rows)),
            "evidence_heldout_disjoint": True,
            "n_heldout_positive": n_heldout_positive,
            "n_heldout_negative": n_heldout_negative,
            "evidence_text_unique": True,
            "heldout_text_unique": True,
            "evidence_heldout_text_disjoint": True,
            "heldout_ids_unique": True,
            "interpretability_eligible": interpretability_eligible,
            "scorer_eligible": scorer_eligible,
            "scarcity_reason": sorted(key for key, value in scarcity.items() if value),
            "scarcity": scarcity,
        },
    }


def build_contrastive_evidence_packs(
    *,
    latents: pd.DataFrame,
    features: np.ndarray,
    label_matrix: pd.DataFrame,
    records: list[dict[str, Any]],
    config: ContrastiveEvidenceConfig = ContrastiveEvidenceConfig(),
) -> list[dict[str, Any]]:
    label_tuple = tuple(label.upper() for label in config.labels)
    selected = normalise_latents(
        latents,
        label_tuple,
        stable_role=config.stable_role,
        expected_count=config.expected_stable_core_count,
    )
    validate_matrix_alignment(features, label_matrix, records)
    bad = selected[(selected["latent_idx"] < 0) | (selected["latent_idx"] >= features.shape[1])]
    if not bad.empty:
        preview = bad[["target_label", "latent_idx"]].head(5).to_dict(orient="records")
        raise ValueError(f"Latent indices outside feature dimension {features.shape[1]}: {preview}")
    missing_labels = [label for label in label_tuple if label not in label_matrix.columns]
    if missing_labels:
        raise ValueError(f"Label matrix missing labels: {missing_labels}")

    context = prepare_pack_context(label_matrix, records, label_tuple)
    packs: list[dict[str, Any]] = []
    for packet_number, (_, row) in enumerate(selected.iterrows(), start=1):
        packs.append(
            _build_single_pack(
                packet_number=packet_number,
                latent_row=row,
                features=features,
                label_matrix=label_matrix,
                records=records,
                labels=label_tuple,
                config=config,
                context=context,
            )
        )
    return packs


def pack_summary_rows(packs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pack in packs:
        summary = pack.get("summary", {})
        row: dict[str, Any] = {
            "packet_id": pack["packet_id"],
            "target_label": pack["target_label"],
            "latent_idx": pack["latent_idx"],
            "rank_within_label": pack["rank_within_label"],
            "tier": pack["tier"],
            "stable_set_role": pack.get("stable_set_role"),
            "label_selection_status": pack.get("label_selection_status"),
            "auc": pack.get("auc"),
            "directional_auc": pack.get("directional_auc"),
            "cohens_d": pack.get("cohens_d"),
            "precision_at_50": pack.get("precision_at_50"),
            "inclusion_frequency": pack.get("inclusion_frequency"),
            "q50_nonzero_activation": pack.get("q50_nonzero_activation"),
            "n_nonzero_activation": pack.get("n_nonzero_activation"),
            "n_evidence_rows": summary.get("n_evidence_rows"),
            "n_heldout_rows": summary.get("n_heldout_rows"),
            "n_heldout_positive": summary.get("n_heldout_positive"),
            "n_heldout_negative": summary.get("n_heldout_negative"),
            "evidence_text_unique": summary.get("evidence_text_unique"),
            "heldout_text_unique": summary.get("heldout_text_unique"),
            "evidence_heldout_text_disjoint": summary.get("evidence_heldout_text_disjoint"),
            "heldout_ids_unique": summary.get("heldout_ids_unique"),
            "interpretability_eligible": summary.get("interpretability_eligible"),
            "scorer_eligible": summary.get("scorer_eligible"),
            "scarcity_reason": ",".join(summary.get("scarcity_reason", [])),
        }
        for key, value in summary.get("actual_counts", {}).items():
            row[f"n_{key}"] = value
        for key, value in summary.get("heldout_counts", {}).items():
            row[f"heldout_n_{key}"] = value
        for key, value in summary.get("scarcity", {}).items():
            row[f"scarce_{key}"] = bool(value)
        rows.append(row)
    return rows


def run_build_contrastive_evidence_packs(
    *,
    latents_path: str | Path,
    feature_store_path: str | Path,
    label_matrix_path: str | Path,
    records_path: str | Path,
    output_dir: str | Path,
    config: ContrastiveEvidenceConfig = ContrastiveEvidenceConfig(),
) -> dict[str, Any]:
    output_path = Path(output_dir)
    evidence_dir = output_path / "evidence_packs"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    latents = pd.read_csv(latents_path)
    features = load_feature_matrix(feature_store_path)
    label_matrix = pd.read_csv(label_matrix_path)
    records = read_jsonl(records_path)
    packs = build_contrastive_evidence_packs(
        latents=latents,
        features=features,
        label_matrix=label_matrix,
        records=records,
        config=config,
    )

    packs_path = evidence_dir / "contrastive_evidence_packs.jsonl"
    summary_path = evidence_dir / "contrastive_evidence_pack_summary.csv"
    write_jsonl(packs_path, packs)
    pd.DataFrame(pack_summary_rows(packs)).to_csv(summary_path, index=False)

    label_counts = pd.Series([pack["target_label"] for pack in packs]).value_counts().sort_index().to_dict()
    role_counts = pd.Series([pack.get("stable_set_role", "") for pack in packs]).value_counts().sort_index().to_dict()
    scarcity_counts: dict[str, int] = {}
    for pack in packs:
        for key, value in pack.get("summary", {}).get("scarcity", {}).items():
            if value:
                scarcity_counts[key] = scarcity_counts.get(key, 0) + 1

    manifest = {
        "analysis": CONTRASTIVE_P3_VERSION,
        "step": "build-packs",
        "inputs": {
            "latents": str(latents_path),
            "feature_store": str(feature_store_path),
            "label_matrix": str(label_matrix_path),
            "records": str(records_path),
        },
        "outputs": {
            "evidence_packs": str(packs_path),
            "evidence_pack_summary": str(summary_path),
        },
        "parameters": asdict(config),
        "n_packs": int(len(packs)),
        "label_counts": {str(key): int(value) for key, value in label_counts.items()},
        "stable_role_counts": {str(key): int(value) for key, value in role_counts.items()},
        "feature_shape": [int(features.shape[0]), int(features.shape[1])],
        "label_matrix_rows": int(len(label_matrix)),
        "records_rows": int(len(records)),
        "scarcity_counts": scarcity_counts,
        "integrity": {
            "evidence_duplicate_text_packs": int(
                sum(not bool(pack.get("summary", {}).get("evidence_text_unique")) for pack in packs)
            ),
            "heldout_duplicate_text_packs": int(
                sum(not bool(pack.get("summary", {}).get("heldout_text_unique")) for pack in packs)
            ),
            "evidence_heldout_text_overlap_packs": int(
                sum(not bool(pack.get("summary", {}).get("evidence_heldout_text_disjoint")) for pack in packs)
            ),
            "heldout_duplicate_id_packs": int(
                sum(not bool(pack.get("summary", {}).get("heldout_ids_unique")) for pack in packs)
            ),
            "interpretability_eligible_packs": int(
                sum(bool(pack.get("summary", {}).get("interpretability_eligible")) for pack in packs)
            ),
            "scorer_eligible_packs": int(
                sum(bool(pack.get("summary", {}).get("scorer_eligible")) for pack in packs)
            ),
        },
        "label_blind_projection": {
            "visible_sample_keys": list(VISIBLE_SAMPLE_KEYS),
            "forbidden_keys": sorted(FORBIDDEN_EXPLAINER_KEYS),
            "nonactive_label_match_hidden_from_explainer": True,
        },
    }
    write_json(output_path / "manifest.json", manifest)
    return manifest


__all__ = [
    "CONTRASTIVE_P3_VERSION",
    "ContrastiveEvidenceConfig",
    "DEFAULT_LABELS",
    "FORBIDDEN_EXPLAINER_KEYS",
    "MISC_LABEL_PATTERN",
    "PRIMARY_LABELS",
    "assert_label_blind_samples",
    "build_contrastive_evidence_packs",
    "jsonable",
    "load_feature_matrix",
    "normalise_latents",
    "normalise_text",
    "pack_summary_rows",
    "read_jsonl",
    "run_build_contrastive_evidence_packs",
    "stable_seed",
    "write_json",
    "write_jsonl",
]
