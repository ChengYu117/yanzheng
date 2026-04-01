"""Core data loading utilities.

Supports both CACTUS (unified JSONL) and legacy MI-RE (split JSONL) formats.
Default data source: data/cactus/cactus_re_small_1500.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "cactus"
CACTUS_JSONL = "cactus_re_small_1500.jsonl"


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_cactus_dataset(
    data_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load CACTUS unified JSONL and split by label.

    Returns:
        (re_records, nonre_records, all_records)
        - re_records: samples with label='RE'
        - nonre_records: samples with label!='RE' (NonRE_CBT + NonTech_Process)
        - all_records: all samples
    Each record has a 'unit_text' key mapped from 'formatted_text' for
    backward compatibility with the rest of the pipeline.
    """
    root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    jsonl_path = root / CACTUS_JSONL

    if not jsonl_path.exists():
        # Fallback to legacy format
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
    re_records, nonre_records, _ = load_cactus_dataset(data_dir)
    return {
        "data_dir": str(Path(data_dir) if data_dir else DEFAULT_DATA_DIR),
        "re_count": len(re_records),
        "nonre_count": len(nonre_records),
        "total_count": len(re_records) + len(nonre_records),
        "re_example": re_records[0] if re_records else None,
        "nonre_example": nonre_records[0] if nonre_records else None,
    }
