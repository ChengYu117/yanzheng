from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "mi_re"


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def dataset_summary(data_dir: str | Path | None = None) -> dict:
    root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    re_path = root / "re_dataset.jsonl"
    nonre_path = root / "nonre_dataset.jsonl"
    re_records = load_jsonl(re_path)
    nonre_records = load_jsonl(nonre_path)

    return {
        "data_dir": str(root),
        "re_count": len(re_records),
        "nonre_count": len(nonre_records),
        "total_count": len(re_records) + len(nonre_records),
        "re_example": re_records[0] if re_records else None,
        "nonre_example": nonre_records[0] if nonre_records else None,
    }
