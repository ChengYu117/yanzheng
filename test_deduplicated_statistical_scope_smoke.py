from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import torch

from src.nlp_re_base.deduplicated_statistical_scope import build_deduplicated_statistical_scope


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_deduplicated_scope() -> None:
    root = (Path.cwd() / "outputs" / "_smoke_deduplicated_scope").resolve()
    if root.exists():
        shutil.rmtree(root)
    source = root / "source"
    output = root / "output"
    (source / "feature_store").mkdir(parents=True)
    rows = [
        {"row_idx": 0, "original_row_idx": 10, "record_id": "r0", "source_split": "high", "source_file": "a.jsonl", "predicted_code": "AF", "predicted_subcode": "", "confidence": 0.8, "unit_text": "That is great", "RE": 0, "RES": 0, "REC": 0, "QU": 0, "QUO": 0, "QUC": 0, "GI": 0, "SU": 0, "AF": 1, "OTHER": 0},
        {"row_idx": 1, "original_row_idx": 11, "record_id": "r1", "source_split": "low", "source_file": "b.jsonl", "predicted_code": "AF", "predicted_subcode": "", "confidence": 0.9, "unit_text": "  that   is GREAT ", "RE": 0, "RES": 0, "REC": 0, "QU": 0, "QUO": 0, "QUC": 0, "GI": 0, "SU": 0, "AF": 1, "OTHER": 0},
        {"row_idx": 2, "original_row_idx": 12, "record_id": "r2", "source_split": "high", "source_file": "c.jsonl", "predicted_code": "QU", "predicted_subcode": "QUC", "confidence": 0.95, "unit_text": "Okay?", "RE": 0, "RES": 0, "REC": 0, "QU": 1, "QUO": 0, "QUC": 1, "GI": 0, "SU": 0, "AF": 0, "OTHER": 0},
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(source / "label_matrix.csv", index=False)
    _write_jsonl(source / "records.jsonl", rows)
    torch.save({"utterance_features": torch.arange(12).reshape(3, 4)}, source / "feature_store" / "utterance_features.pt")
    torch.save({"utterance_activations": torch.arange(9).reshape(3, 3)}, source / "feature_store" / "utterance_activations.pt")
    result = build_deduplicated_statistical_scope(source_root=source, output_root=output)
    assert result["source_rows"] == 3
    assert result["retained_rows"] == 2
    assert result["removed_duplicate_rows"] == 1
    kept = pd.read_csv(output / "label_matrix.csv")
    assert kept["record_id"].tolist() == ["r1", "r2"]
    assert kept["row_idx"].tolist() == [0, 1]
    payload = torch.load(output / "feature_store" / "utterance_features.pt", weights_only=True)
    assert payload["utterance_features"][:, 0].tolist() == [4, 8]
    assert len(pd.read_csv(output / "removed_duplicates.csv")) == 1
    shutil.rmtree(root)


if __name__ == "__main__":
    test_deduplicated_scope()
    print("deduplicated statistical scope smoke test passed")
