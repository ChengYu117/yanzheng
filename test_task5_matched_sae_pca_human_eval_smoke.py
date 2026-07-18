"""Synthetic smoke test for Task 5 material preparation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.nlp_re_base.task5_matched_human_eval import (
    Task5PreparationConfig,
    prepare_task5_materials,
)


def main() -> int:
    rng = np.random.default_rng(7)
    n_groups = 30
    rows_per_group = 4
    n = n_groups * rows_per_group
    groups = np.repeat([f"g{i:02d}" for i in range(n_groups)], rows_per_group)
    labels = np.zeros((n, 2), dtype=int)
    labels[:, 0] = (np.arange(n) % 3 == 0).astype(int)
    labels[:, 1] = (np.arange(n) % 4 == 0).astype(int)
    raw = rng.normal(size=(n, 12)).astype(np.float32)
    raw[:, 0] += labels[:, 0] * 2.0
    raw[:, 1] -= labels[:, 0] * 1.5
    raw[:, 2] += labels[:, 1] * 2.2
    raw[:, 3] -= labels[:, 1] * 1.4
    sae = np.maximum(rng.normal(scale=0.2, size=(n, 16)), 0).astype(np.float32)
    for latent in range(6):
        sae[:, latent] += labels[:, 0] * (2.5 - latent * 0.1)
    for latent in range(6, 12):
        sae[:, latent] += labels[:, 1] * (2.5 - (latent - 6) * 0.1)
    label_df = pd.DataFrame(
        {
            "row_idx": np.arange(n),
            "record_id": [f"r{i:03d}" for i in range(n)],
            "file_id": groups,
            "source_split": np.where(np.arange(n) % 2, "high", "low"),
            "unit_text": [f"synthetic utterance number {i} with token {i % 11}" for i in range(n)],
            "L1": labels[:, 0],
            "L2": labels[:, 1],
        }
    )
    stable_rows = []
    for label, latent_ids in (("L1", range(6)), ("L2", range(6, 12))):
        for rank, latent_idx in enumerate(latent_ids, start=1):
            stable_rows.append(
                {
                    "label": label,
                    "latent_idx": latent_idx,
                    "stable_set_role": "stable_core",
                    "rank_within_label": rank,
                }
            )
    stable = pd.DataFrame(stable_rows)
    config = Task5PreparationConfig(
        labels=("L1", "L2"),
        pca_dimensions=(3, 5),
        units_per_label=2,
        folds=3,
        fold_index=0,
        random_state=11,
        discovery_count=3,
        heldout_positive_count=3,
        heldout_control_count=3,
        near_duplicate_threshold=0.99,
        max_auc_difference=0.5,
    )
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "task5"
        result = prepare_task5_materials(
            sae_features=sae,
            raw_hidden=raw,
            label_df=label_df,
            stable_core=stable,
            output_dir=output,
            config=config,
        )
        assert result["validation"]["overall_status"] == "PASS"
        assert len(result["matches"]) == 8
        assert result["selected_sae"]["latent_idx"].nunique() == 4
        assert set(result["matches"]["pca_direction"]) <= {"+", "-"}
        assert (result["matches"]["pca_train_auc"] > 0.5).all()
        for arm in ("pca_3", "pca_5"):
            discovery = pd.read_csv(output / "blind_materials" / arm / "stage1_discovery_examples.csv")
            heldout = pd.read_csv(output / "release_pending" / arm / "stage2_heldout_examples.csv")
            assert len(discovery) == 8 * 3
            assert len(heldout) == 8 * 6
            forbidden = {"representation", "target_label", "unit_score", "truth_role", "row_idx"}
            assert not forbidden.intersection(discovery.columns)
            assert not forbidden.intersection(heldout.columns)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["human_stage"]["interpretations_generated"] is False
        assert manifest["human_stage"]["statistical_comparison_run"] is False
        assert (output / "private" / "blind_mapping_key.csv").exists()
        assert (output / "selected_sae_for_future_b_review.csv").exists()
        assert (output / "SHA256SUMS.txt").exists()
    print("task5 matched SAE/PCA preparation smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
