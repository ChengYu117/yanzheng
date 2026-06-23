from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import torch

from run_misc_latent_evidence_packets import FORBIDDEN_BLIND_KEYS, run_latent_evidence_packet_export


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _safe_smoke_root() -> Path:
    root = (Path.cwd() / "outputs" / "_smoke_latent_evidence_packets").resolve()
    cwd = Path.cwd().resolve()
    if cwd not in root.parents:
        raise RuntimeError(f"Refusing to use smoke output outside repo: {root}")
    return root


def _assert_no_forbidden_keys(obj: object) -> None:
    if isinstance(obj, dict):
        overlap = FORBIDDEN_BLIND_KEYS.intersection(obj)
        assert not overlap, f"Blind packet leaked keys: {sorted(overlap)}"
        for value in obj.values():
            _assert_no_forbidden_keys(value)
    elif isinstance(obj, list):
        for item in obj:
            _assert_no_forbidden_keys(item)


def test_latent_evidence_packets() -> None:
    root = _safe_smoke_root()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    try:
        latents = pd.DataFrame(
            [
                {
                    "label": "A",
                    "rank_within_label": 1,
                    "latent_idx": 0,
                    "cohens_d": 1.2,
                    "directional_auc": 0.9,
                    "precision_at_50": 0.8,
                }
            ]
        )
        features = torch.tensor(
            [
                [9.0, 0.0],
                [8.0, 0.0],
                [7.0, 0.0],
                [6.0, 0.0],
                [5.0, 0.0],
                [4.0, 0.0],
                [3.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=torch.float32,
        )
        labels = pd.DataFrame(
            [
                {"row_idx": 0, "record_id": "r0", "file_id": "f1", "source_line": 1, "source_split": "high", "predicted_code": "A", "predicted_subcode": "", "confidence": 0.9, "unit_text": "target top one", "A": 1, "B": 0},
                {"row_idx": 1, "record_id": "r1", "file_id": "f1", "source_line": 2, "source_split": "high", "predicted_code": "A", "predicted_subcode": "", "confidence": 0.8, "unit_text": "target top two", "A": 1, "B": 0},
                {"row_idx": 2, "record_id": "r2", "file_id": "f2", "source_line": 3, "source_split": "low", "predicted_code": "B", "predicted_subcode": "", "confidence": 0.7, "unit_text": "non target top", "A": 0, "B": 1},
                {"row_idx": 3, "record_id": "r3", "file_id": "f2", "source_line": 4, "source_split": "low", "predicted_code": "B", "predicted_subcode": "", "confidence": 0.6, "unit_text": "Duplicate Text", "A": 0, "B": 1},
                {"row_idx": 4, "record_id": "r4", "file_id": "f2", "source_line": 5, "source_split": "low", "predicted_code": "B", "predicted_subcode": "", "confidence": 0.5, "unit_text": " duplicate   text ", "A": 0, "B": 1},
                {"row_idx": 5, "record_id": "r5", "file_id": "f3", "source_line": 6, "source_split": "high", "predicted_code": "A", "predicted_subcode": "", "confidence": 0.4, "unit_text": "random target one", "A": 1, "B": 0},
                {"row_idx": 6, "record_id": "r6", "file_id": "f3", "source_line": 7, "source_split": "high", "predicted_code": "A", "predicted_subcode": "", "confidence": 0.3, "unit_text": "random target two", "A": 1, "B": 0},
                {"row_idx": 7, "record_id": "r7", "file_id": "f4", "source_line": 8, "source_split": "low", "predicted_code": "B", "predicted_subcode": "", "confidence": 0.2, "unit_text": "extra non target", "A": 0, "B": 1},
            ]
        )
        records = labels.to_dict(orient="records")
        for record in records:
            record["rationale"] = f"rationale for {record['record_id']}"
            record["quality_label"] = record["source_split"]

        latents_path = root / "latents.csv"
        feature_path = root / "features.pt"
        label_path = root / "labels.csv"
        records_path = root / "records.jsonl"
        output_dir = root / "out"

        latents.to_csv(latents_path, index=False)
        torch.save({"utterance_features": features}, feature_path)
        labels.to_csv(label_path, index=False)
        _write_jsonl(records_path, records)

        manifest = run_latent_evidence_packet_export(
            latents_path=latents_path,
            feature_store_path=feature_path,
            label_matrix_path=label_path,
            records_path=records_path,
            output_dir=output_dir,
            labels=("A", "B"),
            top_activating=3,
            high_non_target=3,
            random_target=3,
            random_state=42,
            markdown_examples_per_group=2,
        )

        expected_files = [
            "latent_evidence_packets_labeled.jsonl",
            "latent_evidence_packets_blind.jsonl",
            "latent_evidence_examples_labeled.csv",
            "latent_evidence_examples_blind.csv",
            "latent_evidence_packets_labeled.md",
            "latent_evidence_packets_blind.md",
            "latent_evidence_rationale_appendix.csv",
            "latent_evidence_packet_summary.csv",
            "manifest.json",
        ]
        for name in expected_files:
            assert (output_dir / name).exists(), name

        labeled = pd.read_csv(output_dir / "latent_evidence_examples_labeled.csv")
        blind = pd.read_csv(output_dir / "latent_evidence_examples_blind.csv")
        summary = pd.read_csv(output_dir / "latent_evidence_packet_summary.csv")
        rationale = pd.read_csv(output_dir / "latent_evidence_rationale_appendix.csv")

        top_rows = labeled[labeled["example_group"] == "top_activating"]
        assert top_rows["row_idx"].tolist() == [0, 1, 2]
        assert top_rows["activation"].tolist() == [9.0, 8.0, 7.0]

        high_non_target = labeled[labeled["example_group"] == "high_non_target"]
        assert high_non_target["row_idx"].tolist() == [2, 3, 4]
        assert high_non_target["target_match"].tolist() == [0, 0, 0]

        random_target = labeled[labeled["example_group"] == "random_target"]
        assert set(random_target["row_idx"].tolist()) == {5, 6}
        assert random_target["target_match"].tolist() == [1, 1]
        assert not set(random_target["row_idx"].tolist()).intersection({0, 1, 2})

        duplicate_mask = labeled["duplicate_text_within_packet"].map(lambda value: str(value).lower() == "true")
        duplicate_rows = labeled[duplicate_mask]
        assert set(duplicate_rows["row_idx"].tolist()) == {2, 3, 4}
        assert bool(summary.iloc[0]["scarce_random_target"])
        assert int(summary.iloc[0]["n_random_target"]) == 2
        assert manifest["scarcity"]["random_target"] == 1

        assert "rationale" not in labeled.columns
        assert len(rationale) == len(labeled)
        assert "rationale for r0" in rationale["rationale"].tolist()

        assert set(blind.columns).isdisjoint(FORBIDDEN_BLIND_KEYS)
        assert len(blind) == len(labeled)
        with (output_dir / "latent_evidence_packets_blind.jsonl").open(encoding="utf-8") as handle:
            blind_packet = json.loads(next(handle))
        _assert_no_forbidden_keys(blind_packet)
        assert blind_packet["client_context_available"] is False
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_latent_evidence_packets()
    print("test_latent_evidence_packets_smoke passed")
