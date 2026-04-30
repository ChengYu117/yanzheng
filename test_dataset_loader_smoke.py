"""Smoke tests for the unified experiment dataset loader and feature store."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_misc_full_dataset_loader():
    from nlp_re_base.data import load_experiment_dataset

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "metadata").mkdir(parents=True, exist_ok=True)
        (root / "metadata" / "labels.csv").write_text(
            "id,label\nhigh_001,high\nlow_001,low\n",
            encoding="utf-8",
        )
        _write_jsonl(
            root / "misc_annotations" / "high" / "high_001.jsonl",
            [
                {
                    "file_id": "high_001",
                    "unit_text": "you feel stuck and worried",
                    "predicted_code": "RE",
                    "predicted_subcode": "RES",
                    "confidence": 0.95,
                },
                {
                    "file_id": "high_001",
                    "unit_text": "what would help right now",
                    "predicted_code": "QU",
                    "predicted_subcode": "QUO",
                    "confidence": 0.88,
                },
                {
                    "file_id": "high_001",
                    "unit_text": "uncategorized behavior",
                    "predicted_code": "XX",
                    "predicted_subcode": "YY",
                    "confidence": 0.7,
                },
            ],
        )
        _write_jsonl(
            root / "misc_annotations" / "low" / "low_001.jsonl",
            [
                {
                    "file_id": "low_001",
                    "unit_text": "here is some information",
                    "predicted_code": "GI",
                    "predicted_subcode": None,
                    "confidence": 0.91,
                }
            ],
        )

        dataset = load_experiment_dataset(root)

        assert dataset.data_format == "misc_full"
        assert len(dataset.records) == 4
        assert dataset.binary_labels == [1, 0, 0, 0]
        assert dataset.records[0]["quality_label"] == "high"
        assert dataset.records[-1]["quality_label"] == "low"
        assert "OTHER" in dataset.label_names
        assert dataset.summary["label_counts"]["RE"] == 1
        assert dataset.summary["label_counts"]["RES"] == 1
        assert dataset.summary["label_counts"]["OTHER"] == 1


def test_legacy_dataset_loader():
    from nlp_re_base.data import load_experiment_dataset

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_jsonl(root / "re_dataset.jsonl", [{"unit_text": "reflection"}])
        _write_jsonl(root / "nonre_dataset.jsonl", [{"unit_text": "question"}])

        dataset = load_experiment_dataset(root)

        assert dataset.data_format == "legacy_re_nonre"
        assert dataset.texts == ["reflection", "question"]
        assert dataset.binary_labels == [1, 0]
        assert dataset.label_names == ["RE", "OTHER"]


def test_feature_store_writer():
    from nlp_re_base.data import ExperimentDataset
    from run_sae_evaluation import _save_feature_store

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        records = [
            {
                "sample_id": "a:0001",
                "record_id": "a:0001",
                "file_id": "a",
                "unit_text": "reflection",
                "text": "reflection",
                "label_re": 1,
                "labels": ["RE"],
            },
            {
                "sample_id": "a:0002",
                "record_id": "a:0002",
                "file_id": "a",
                "unit_text": "other",
                "text": "other",
                "label_re": 0,
                "labels": ["OTHER"],
            },
        ]
        dataset = ExperimentDataset(
            data_dir=root,
            data_format="misc_full",
            records=records,
            texts=["reflection", "other"],
            binary_labels=[1, 0],
            re_records=[records[0]],
            nonre_records=[records[1]],
            label_names=["RE", "OTHER"],
            label_matrix=[[1, 0], [0, 1]],
            summary={"n_records": 2},
        )
        out = root / "out"

        paths = _save_feature_store(
            output_dir=out,
            dataset=dataset,
            utterance_features=torch.ones(2, 4),
            utterance_activations=torch.ones(2, 3),
            aggregation="max",
            hook_point="blocks.19.hook_resid_post",
            max_seq_len=128,
            batch_size=2,
            save_token_topk=False,
        )

        assert Path(paths["records"]).exists()
        assert Path(paths["label_matrix"]).exists()
        payload = torch.load(paths["utterance_features"], map_location="cpu")
        assert tuple(payload["utterance_features"].shape) == (2, 4)


def main() -> int:
    tests = [
        test_misc_full_dataset_loader,
        test_legacy_dataset_loader,
        test_feature_store_writer,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"Results: {len(tests)} passed, 0 failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
