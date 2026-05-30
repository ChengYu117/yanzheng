from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import torch

from run_misc_top3_latent_utterances import run_top3_latent_utterance_export


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_top3_selection_and_utterance_export() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        association = pd.DataFrame(
            [
                # Label A: three stable candidates, ordered by weight.
                {"label": "A", "latent_idx": 0, "latent_label_weight": 0.70, "precision_lift_absolute_at_50": 0.2, "directional_auc": 0.71, "abs_cohens_d": 0.7, "association_rank": 3, "cohens_d": 0.7, "mean_diff": 1.0, "negative_boundary": False, "edge_type": "positive_support", "stable_edge": True, "positive_support": True, "precision_at_50": 0.8, "precision_at_100": 0.7},
                {"label": "A", "latent_idx": 1, "latent_label_weight": 0.95, "precision_lift_absolute_at_50": 0.1, "directional_auc": 0.72, "abs_cohens_d": 0.6, "association_rank": 2, "cohens_d": 0.6, "mean_diff": 1.0, "negative_boundary": False, "edge_type": "positive_support", "stable_edge": True, "positive_support": True, "precision_at_50": 0.7, "precision_at_100": 0.6},
                {"label": "A", "latent_idx": 2, "latent_label_weight": 0.80, "precision_lift_absolute_at_50": 0.3, "directional_auc": 0.73, "abs_cohens_d": 0.8, "association_rank": 1, "cohens_d": 0.8, "mean_diff": 1.0, "negative_boundary": False, "edge_type": "positive_support", "stable_edge": True, "positive_support": True, "precision_at_50": 0.9, "precision_at_100": 0.8},
                # Label B: one stable candidate; fallback should use latent 2 then 1, excluding negative boundary latent 4.
                {"label": "B", "latent_idx": 3, "latent_label_weight": 0.60, "precision_lift_absolute_at_50": 0.2, "directional_auc": 0.71, "abs_cohens_d": 0.9, "association_rank": 1, "cohens_d": 0.9, "mean_diff": 1.0, "negative_boundary": False, "edge_type": "positive_support", "stable_edge": True, "positive_support": True, "precision_at_50": 0.6, "precision_at_100": 0.5},
                {"label": "B", "latent_idx": 4, "latent_label_weight": 9.00, "precision_lift_absolute_at_50": 9.0, "directional_auc": 0.99, "abs_cohens_d": 9.0, "association_rank": 2, "cohens_d": -9.0, "mean_diff": -1.0, "negative_boundary": True, "edge_type": "negative_boundary", "stable_edge": False, "positive_support": False, "precision_at_50": 0.1, "precision_at_100": 0.1},
                {"label": "B", "latent_idx": 2, "latent_label_weight": 0.50, "precision_lift_absolute_at_50": 0.4, "directional_auc": 0.70, "abs_cohens_d": 0.7, "association_rank": 3, "cohens_d": 0.7, "mean_diff": 1.0, "negative_boundary": False, "edge_type": "weak_or_noise", "stable_edge": False, "positive_support": False, "precision_at_50": 0.5, "precision_at_100": 0.4},
                {"label": "B", "latent_idx": 1, "latent_label_weight": 0.40, "precision_lift_absolute_at_50": 0.3, "directional_auc": 0.69, "abs_cohens_d": 0.6, "association_rank": 4, "cohens_d": 0.6, "mean_diff": 1.0, "negative_boundary": False, "edge_type": "weak_or_noise", "stable_edge": False, "positive_support": False, "precision_at_50": 0.4, "precision_at_100": 0.3},
            ]
        )
        thresholded = association[association["stable_edge"]].copy()
        features = torch.tensor(
            [
                [0.1, 5.0, 1.0, 0.2, 9.0],
                [0.2, 4.0, 2.0, 0.3, 8.0],
                [0.3, 3.0, 3.0, 0.4, 7.0],
                [0.4, 2.0, 4.0, 0.5, 6.0],
                [0.5, 1.0, 5.0, 0.6, 5.0],
            ],
            dtype=torch.float32,
        )
        labels = pd.DataFrame(
            [
                {"row_idx": i, "record_id": f"r{i}", "file_id": "f", "source_line": i + 1, "quality_label": "high", "A": int(i < 2), "B": int(i >= 2)}
                for i in range(5)
            ]
        )
        records = [
            {"record_id": f"r{i}", "file_id": "f", "source_line": i + 1, "quality_label": "high", "text": f"utterance {i}"}
            for i in range(5)
        ]

        association_path = root / "association.csv"
        thresholded_path = root / "thresholded.csv"
        feature_path = root / "features.pt"
        label_path = root / "labels.csv"
        records_path = root / "records.jsonl"
        output_dir = root / "out"
        association.to_csv(association_path, index=False)
        thresholded.to_csv(thresholded_path, index=False)
        torch.save({"utterance_features": features}, feature_path)
        labels.to_csv(label_path, index=False)
        _write_jsonl(records_path, records)

        audit = run_top3_latent_utterance_export(
            association_path=association_path,
            thresholded_path=thresholded_path,
            feature_store_path=feature_path,
            label_matrix_path=label_path,
            records_path=records_path,
            output_dir=output_dir,
            labels=("A", "B"),
            top_latents=3,
            top_utterances=3,
        )

        selected = pd.read_csv(output_dir / "top3_latents_by_label.csv")
        utterances = pd.read_csv(output_dir / "top20_utterances_by_top3_latents.csv")
        assert len(selected) == 6
        assert len(utterances) == 18
        assert selected[selected["label"] == "A"]["latent_idx"].tolist() == [1, 2, 0]
        assert selected[selected["label"] == "B"]["latent_idx"].tolist() == [3, 2, 1]
        assert 4 not in selected["latent_idx"].tolist()
        assert selected[selected["label"] == "B"]["selection_source"].tolist() == [
            "stable_positive",
            "fallback_top_association",
            "fallback_top_association",
        ]
        first_b2 = utterances[(utterances["label"] == "B") & (utterances["latent_idx"] == 2)].iloc[0]
        assert int(first_b2["row_idx"]) == 4
        assert first_b2["record_id"] == "r4"
        assert first_b2["text"] == "utterance 4"
        assert audit["labels"]["B"]["fallback_count"] == 2


if __name__ == "__main__":
    test_top3_selection_and_utterance_export()
    print("test_top3_latent_utterances_smoke passed")
