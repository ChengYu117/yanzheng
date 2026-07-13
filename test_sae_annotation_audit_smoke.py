from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.nlp_re_base.sae_annotation_audit import (
    SAEAnnotationAuditConfig,
    build_disagreement_candidates,
    generate_oof_predictions,
    sample_review_cases,
)


def main() -> None:
    root = Path("outputs/_smoke_sae_annotation_audit")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    rng = np.random.default_rng(7)
    n_groups = 20
    rows_per_group = 4
    n = n_groups * rows_per_group
    groups = np.repeat([f"file_{i:02d}" for i in range(n_groups)], rows_per_group)
    y_re = np.tile([0, 1, 0, 1], n_groups)
    y_qu = np.tile([1, 0, 1, 0], n_groups)
    features = rng.normal(size=(n, 8)).astype(np.float32)
    features[:, 0] += y_re * 0.8
    features[:, 2] += y_qu * 0.8
    labels = pd.DataFrame(
        {
            "row_idx": np.arange(n),
            "record_id": [f"r{i}" for i in range(n)],
            "file_id": groups,
            "source_split": "high",
            "source_file": [f"{g}.jsonl" for g in groups],
            "predicted_code": "OTHER",
            "predicted_subcode": "",
            "confidence": 0.9,
            "unit_text": [f"synthetic utterance {i % 13}" for i in range(n)],
            "RE": y_re,
            "QU": y_qu,
        }
    )
    stable = pd.DataFrame(
        {
            "label": ["RE", "RE", "QU", "QU"],
            "latent_idx": [0, 1, 2, 3],
            "stable_set_role": "stable_core",
            "full_data_rank": [1, 2, 1, 2],
        }
    )
    config = SAEAnnotationAuditConfig(
        labels=("RE", "QU"), folds=4, high_per_type=3, boundary_per_type=2
    )
    oof, folds = generate_oof_predictions(
        features=features, label_matrix=labels, stable_core=stable, config=config
    )
    assert len(oof) == n * 2
    assert not oof.duplicated(["row_idx", "label"]).any()
    assert oof["sae_probability"].between(0, 1).all()
    assert all(row["group_overlap_count"] == 0 for row in folds)
    candidates = build_disagreement_candidates(oof, labels, config)
    assert set(candidates["disagreement_type"]).issubset({"false_negative", "false_positive"})
    review_a = sample_review_cases(candidates, config)
    review_b = sample_review_cases(candidates, config)
    assert review_a[["row_idx", "label"]].equals(review_b[["row_idx", "label"]])
    assert review_a["case_id"].is_unique
    assert review_a["reviewer_target_label"].equals(review_a["label"])
    assert list(review_a.columns[:3]) == ["case_id", "reviewer_target_label", "unit_text"]
    assert set(review_a["sampling_stratum"]).issubset({"highest_margin", "near_boundary"})
    shutil.rmtree(root, ignore_errors=True)
    print("test_sae_annotation_audit_smoke passed")


if __name__ == "__main__":
    main()
