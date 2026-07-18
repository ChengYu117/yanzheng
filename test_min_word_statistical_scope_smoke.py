from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import torch

from src.nlp_re_base.statistical_scope_filter import build_min_word_statistical_scope


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source"
        output = root / "filtered"
        (source / "feature_store").mkdir(parents=True)
        texts = ["okay", "one two three four", "one two three four five", "six words are retained right here"]
        labels = pd.DataFrame(
            {
                "row_idx": range(4),
                "record_id": [f"r{i}" for i in range(4)],
                "unit_text": texts,
                "AF": [0, 1, 0, 1],
                "OTHER": [1, 0, 1, 0],
            }
        )
        labels.to_csv(source / "label_matrix.csv", index=False)
        with (source / "records.jsonl").open("w", encoding="utf-8") as handle:
            for i, text in enumerate(texts):
                handle.write(json.dumps({"record_id": f"r{i}", "unit_text": text}) + "\n")
        torch.save(torch.arange(12).reshape(4, 3), source / "feature_store" / "utterance_features.pt")
        torch.save(torch.arange(8).reshape(4, 2), source / "feature_store" / "utterance_activations.pt")
        result = build_min_word_statistical_scope(source_root=source, output_root=output)
        assert result["archived_rows"] == 2
        assert result["retained_rows"] == 2
        filtered = pd.read_csv(output / "label_matrix.csv")
        assert filtered["row_idx"].tolist() == [0, 1]
        assert filtered["original_row_idx"].tolist() == [2, 3]
        assert torch.load(output / "feature_store" / "utterance_features.pt", weights_only=True).shape == (2, 3)
        assert len((output / "archive_lt5_words" / "records_lt5_words.jsonl").read_text().splitlines()) == 2
    print("test_min_word_statistical_scope_smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
