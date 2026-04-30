"""Core data loading utilities.

The current main experiment uses the full MISC annotation dataset, where each
``unit_text`` behavior unit is the analysis sample. Legacy MI-RE split JSONL
and CACTUS unified JSONL are still supported for compatibility.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEGACY_DATA_DIR = PROJECT_ROOT / "data" / "mi_re"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "mi_quality_counseling_misc"
CACTUS_JSONL = "cactus_re_small_1500.jsonl"
MISC_ANNOTATION_DIR = "misc_annotations"
CORE_MISC_LABELS = {"RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF"}
PREFERRED_LABEL_ORDER = [
    "RE",
    "RES",
    "REC",
    "QU",
    "QUO",
    "QUC",
    "GI",
    "SU",
    "AF",
    "OTHER",
]


@dataclass
class ExperimentDataset:
    """Unified dataset view used by the SAE and causal pipelines."""

    data_dir: Path
    data_format: str
    records: list[dict[str, Any]]
    texts: list[str]
    binary_labels: list[int]
    re_records: list[dict[str, Any]]
    nonre_records: list[dict[str, Any]]
    label_names: list[str]
    label_matrix: list[list[int]]
    summary: dict[str, Any]


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root


def infer_data_format(data_dir: str | Path | None = None) -> str:
    """Infer the dataset format from files on disk."""
    root = resolve_data_dir(data_dir)
    if root.name == MISC_ANNOTATION_DIR or (root / MISC_ANNOTATION_DIR).exists():
        return "misc_full"
    if (root / CACTUS_JSONL).exists():
        return "cactus"
    if (root / "re_dataset.jsonl").exists() or (root / "nonre_dataset.jsonl").exists():
        return "legacy_re_nonre"
    raise FileNotFoundError(
        f"Cannot infer dataset format under {root}. Expected MISC annotations, "
        f"{CACTUS_JSONL}, or re_dataset.jsonl/nonre_dataset.jsonl."
    )


def _normalize_label(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"", "NONE", "NULL", "N/A", "NA", "NAN"}:
        return ""
    return text


def misc_label_set(record: dict[str, Any]) -> set[str]:
    """Return the canonical hierarchical MISC labels for a record.

    Unknown or out-of-scope MISC codes are collapsed to OTHER while the raw
    predicted_code/subcode fields are preserved on the record.
    """
    code = _normalize_label(record.get("predicted_code"))
    subcode = _normalize_label(record.get("predicted_subcode"))
    labels: set[str] = set()

    if code in CORE_MISC_LABELS:
        labels.add(code)
    if subcode in CORE_MISC_LABELS:
        labels.add(subcode)

    return labels or {"OTHER"}


def is_reflection_record(record: dict[str, Any]) -> bool:
    code = _normalize_label(record.get("predicted_code"))
    subcode = _normalize_label(record.get("predicted_subcode"))
    return code == "RE" or subcode in {"RES", "REC"}


def _ordered_labels(label_sets: list[set[str]]) -> list[str]:
    all_labels = set().union(*label_sets) if label_sets else set()
    ordered = [label for label in PREFERRED_LABEL_ORDER if label in all_labels]
    ordered.extend(sorted(all_labels.difference(ordered)))
    return ordered


def _build_label_matrix(
    records: list[dict[str, Any]],
    labels: list[str],
) -> list[list[int]]:
    matrix: list[list[int]] = []
    for record in records:
        label_set = set(record.get("labels") or misc_label_set(record))
        matrix.append([1 if label in label_set else 0 for label in labels])
    return matrix


def _load_quality_labels(root: Path) -> dict[str, str]:
    labels_path = root / "metadata" / "labels.csv"
    if not labels_path.exists():
        return {}
    out: dict[str, str] = {}
    with labels_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_id = str(row.get("id") or "").strip()
            label = str(row.get("label") or "").strip()
            if file_id:
                out[file_id] = label
    return out


def _misc_annotation_root(data_dir: str | Path) -> Path:
    root = resolve_data_dir(data_dir)
    if root.name == MISC_ANNOTATION_DIR:
        return root
    return root / MISC_ANNOTATION_DIR


def iter_misc_annotation_files(data_dir: str | Path) -> list[Path]:
    root = _misc_annotation_root(data_dir)
    files = list(root.glob("*.jsonl")) + list(root.glob("*/*.jsonl"))
    unique = sorted({path.resolve(): path for path in files}.values())
    if not unique:
        raise FileNotFoundError(
            f"No MISC annotation JSONL files found under {root}. "
            "Expected misc_annotations/high/*.jsonl and misc_annotations/low/*.jsonl."
        )
    return unique


def _safe_confidence(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_misc_full_records(
    data_dir: str | Path | None = None,
    *,
    confidence_threshold: float | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load full MISC annotation records as normalized behavior-unit samples."""
    root = resolve_data_dir(data_dir)
    annotation_root = _misc_annotation_root(root)
    quality_labels = _load_quality_labels(root if root.name != MISC_ANNOTATION_DIR else root.parent)
    records: list[dict[str, Any]] = []

    for path in iter_misc_annotation_files(annotation_root):
        source_split = path.parent.name if path.parent != annotation_root else ""
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                text = str(
                    raw.get("unit_text")
                    or raw.get("utterance")
                    or raw.get("formatted_text")
                    or ""
                ).strip()
                if not text:
                    continue
                confidence = _safe_confidence(raw.get("confidence"))
                if (
                    confidence_threshold is not None
                    and (confidence is None or confidence < confidence_threshold)
                ):
                    continue

                file_id = str(raw.get("file_id") or path.stem).strip()
                labels = sorted(
                    misc_label_set(raw),
                    key=lambda label: (
                        PREFERRED_LABEL_ORDER.index(label)
                        if label in PREFERRED_LABEL_ORDER
                        else len(PREFERRED_LABEL_ORDER)
                    ),
                )
                label_re = int(is_reflection_record(raw))
                code = _normalize_label(raw.get("predicted_code")) or None
                label_family = code if code in CORE_MISC_LABELS else "OTHER"
                sample_id = f"{file_id}:{line_no:04d}"

                record = dict(raw)
                record.update(
                    {
                        "sample_id": sample_id,
                        "record_id": sample_id,
                        "file_id": file_id,
                        "quality_label": quality_labels.get(file_id, source_split or None),
                        "text": text,
                        "unit_text": text,
                        "predicted_code": code,
                        "predicted_subcode": _normalize_label(raw.get("predicted_subcode")) or None,
                        "label_re": label_re,
                        "label_family": label_family,
                        "labels": labels,
                        "confidence": confidence,
                        "rationale": raw.get("rationale"),
                        "source_split": source_split,
                        "source_file": path.name,
                        "source_path": str(path),
                        "source_line": line_no,
                    }
                )
                records.append(record)

                if limit is not None and len(records) >= limit:
                    return records

    return records


def _normalize_legacy_record(
    record: dict[str, Any],
    *,
    label_re: int,
    fallback_id: str,
) -> dict[str, Any]:
    text = str(
        record.get("unit_text")
        or record.get("formatted_text")
        or record.get("therapist_curr")
        or record.get("text")
        or ""
    ).strip()
    sample_id = str(record.get("sample_id") or record.get("record_id") or fallback_id)
    labels = ["RE"] if label_re else ["OTHER"]
    normalized = dict(record)
    normalized.update(
        {
            "sample_id": sample_id,
            "record_id": sample_id,
            "text": text,
            "unit_text": text,
            "label_re": int(label_re),
            "labels": labels,
            "label_family": "RE" if label_re else "OTHER",
            "quality_label": record.get("quality_label"),
        }
    )
    return normalized


def load_experiment_dataset(
    data_dir: str | Path | None = None,
    *,
    data_format: str = "auto",
    confidence_threshold: float | None = None,
    limit: int | None = None,
) -> ExperimentDataset:
    """Load any supported dataset into the unified experiment representation."""
    root = resolve_data_dir(data_dir)
    resolved_format = infer_data_format(root) if data_format == "auto" else data_format
    if resolved_format not in {"misc_full", "legacy_re_nonre", "cactus"}:
        raise ValueError(
            "data_format must be one of: auto, misc_full, legacy_re_nonre, cactus"
        )

    if resolved_format == "misc_full":
        records = load_misc_full_records(
            root,
            confidence_threshold=confidence_threshold,
            limit=limit,
        )
    else:
        re_records, nonre_records, _ = load_cactus_dataset(root)
        records = []
        for idx, record in enumerate(re_records):
            records.append(
                _normalize_legacy_record(record, label_re=1, fallback_id=f"re:{idx:05d}")
            )
            if limit is not None and len(records) >= limit:
                break
        if limit is None or len(records) < limit:
            for idx, record in enumerate(nonre_records):
                records.append(
                    _normalize_legacy_record(
                        record,
                        label_re=0,
                        fallback_id=f"nonre:{idx:05d}",
                    )
                )
                if limit is not None and len(records) >= limit:
                    break

    if not records:
        raise RuntimeError(f"No records loaded from {root} using format={resolved_format}.")

    texts = [str(record.get("text") or record.get("unit_text") or "") for record in records]
    binary_labels = [int(record.get("label_re", 0)) for record in records]
    re_records = [record for record in records if int(record.get("label_re", 0)) == 1]
    nonre_records = [record for record in records if int(record.get("label_re", 0)) == 0]
    label_sets = [set(record.get("labels") or misc_label_set(record)) for record in records]
    label_names = _ordered_labels(label_sets)
    label_matrix = _build_label_matrix(records, label_names)

    label_counts = {
        label: int(sum(row[idx] for row in label_matrix))
        for idx, label in enumerate(label_names)
    }
    quality_counts: dict[str, int] = {}
    for record in records:
        quality = str(record.get("quality_label") or "unknown")
        quality_counts[quality] = quality_counts.get(quality, 0) + 1

    summary = {
        "data_dir": str(root),
        "data_format": resolved_format,
        "n_records": len(records),
        "re_count": len(re_records),
        "nonre_count": len(nonre_records),
        "label_names": label_names,
        "label_counts": label_counts,
        "quality_counts": quality_counts,
        "confidence_threshold": confidence_threshold,
        "limit": limit,
        "example": records[0] if records else None,
    }

    return ExperimentDataset(
        data_dir=root,
        data_format=resolved_format,
        records=records,
        texts=texts,
        binary_labels=binary_labels,
        re_records=re_records,
        nonre_records=nonre_records,
        label_names=label_names,
        label_matrix=label_matrix,
        summary=summary,
    )


def load_cactus_dataset(
    data_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load the default legacy MI-RE split or a compatible CACTUS JSONL.

    Returns:
        (re_records, nonre_records, all_records)
        - re_records: RE samples
        - nonre_records: non-RE samples
        - all_records: all samples
    If a CACTUS unified JSONL is present, it is loaded and adapted for backward
    compatibility by mapping ``formatted_text`` to ``unit_text``. Otherwise the
    loader falls back to the legacy ``re_dataset.jsonl`` / ``nonre_dataset.jsonl``
    split used by the original RE experiments.
    """
    root = Path(data_dir) if data_dir else DEFAULT_LEGACY_DATA_DIR
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    jsonl_path = root / CACTUS_JSONL

    if not jsonl_path.exists():
        # Default path for the current branch: legacy MI-RE split format.
        return _load_legacy_split(root)

    raw = load_jsonl(jsonl_path)
    re_records: list[dict] = []
    nonre_records: list[dict] = []

    for r in raw:
        # Backward compatibility: add 'unit_text' mapped from 'formatted_text'
        r["unit_text"] = r.get("formatted_text", r.get("therapist_curr", ""))
        r["label_re"] = 1 if r.get("label") == "RE" else 0

        if r.get("label") == "RE":
            re_records.append(r)
        else:
            nonre_records.append(r)

    return re_records, nonre_records, raw


def _load_legacy_split(root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Fallback: load legacy MI-RE split JSONL format."""
    re_path = root / "re_dataset.jsonl"
    nonre_path = root / "nonre_dataset.jsonl"
    re_records = load_jsonl(re_path) if re_path.exists() else []
    nonre_records = load_jsonl(nonre_path) if nonre_path.exists() else []
    for r in re_records:
        r["label_re"] = 1
    for r in nonre_records:
        r["label_re"] = 0
    return re_records, nonre_records, re_records + nonre_records


def dataset_summary(data_dir: str | Path | None = None) -> dict:
    dataset = load_experiment_dataset(data_dir, data_format="auto")
    return dataset.summary
