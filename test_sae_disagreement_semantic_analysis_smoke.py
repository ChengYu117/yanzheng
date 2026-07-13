from __future__ import annotations

import numpy as np
import pandas as pd

from src.nlp_re_base.sae_annotation_audit import (
    SAEAnnotationAuditConfig,
    build_disagreement_candidates,
    generate_oof_predictions,
    sample_review_cases,
)
from src.nlp_re_base.sae_disagreement_semantic_analysis import (
    SemanticDisagreementConfig,
    build_case_latent_evidence,
    build_case_review_prompt,
    summarize_oof_disagreements,
)


def main() -> None:
    rng = np.random.default_rng(11)
    n_groups, rows_per_group = 12, 4
    n = n_groups * rows_per_group
    groups = np.repeat([f"f{i:02d}" for i in range(n_groups)], rows_per_group)
    y = np.tile([0, 1, 0, 1], n_groups)
    features = rng.normal(size=(n, 4)).astype(np.float32)
    features[:, 0] += y * 0.7
    labels = pd.DataFrame(
        {
            "row_idx": np.arange(n),
            "record_id": [f"r{i}" for i in range(n)],
            "file_id": groups,
            "source_split": "high",
            "source_file": "synthetic.jsonl",
            "predicted_code": "AF",
            "predicted_subcode": "",
            "confidence": 0.9,
            "unit_text": [f"synthetic positive statement {i}" for i in range(n)],
            "AF": y,
        }
    )
    stable = pd.DataFrame(
        {
            "label": ["AF", "AF"],
            "latent_idx": [0, 1],
            "stable_set_role": ["stable_core", "stable_core"],
            "full_data_rank": [1, 2],
        }
    )
    audit_config = SAEAnnotationAuditConfig(
        labels=("AF",), folds=3, high_per_type=2, boundary_per_type=1
    )
    oof, _ = generate_oof_predictions(
        features=features, label_matrix=labels, stable_core=stable, config=audit_config
    )
    candidates = build_disagreement_candidates(oof, labels, audit_config)
    review = sample_review_cases(candidates, audit_config)
    assert len(review) > 0
    explanations = [
        {
            "latent_idx": 0,
            "short_name": "positive evaluation",
            "feature_type": "mixed",
            "candidate_explanation": "positive evaluation with good/great wording",
            "confidence": 0.8,
            "majority_gate_passed": True,
            "induction_status": "candidate_pattern",
        },
        {
            "latent_idx": 1,
            "short_name": "generic wording",
            "feature_type": "surface",
            "candidate_explanation": "generic phrase pattern",
            "confidence": 0.6,
            "majority_gate_passed": True,
            "induction_status": "candidate_pattern",
        },
    ]
    evidence = build_case_latent_evidence(
        features=features,
        label_matrix=labels,
        stable_core=stable,
        review_cases=review,
        explanations=explanations,
        audit_config=audit_config,
        config=SemanticDisagreementConfig(probability_tolerance=1e-6),
    )
    assert len(evidence) == len(review)
    assert evidence["probability_reconstruction_error"].max() < 1e-6
    assert evidence["positive_latent_evidence"].map(lambda x: isinstance(x, list)).all()
    rates = summarize_oof_disagreements(oof)
    assert rates.loc[0, "n_false_negative"] == int(
        ((oof["reference_label"] == 1) & (oof["sae_prediction"] == 0)).sum()
    )
    assert rates.loc[0, "n_false_positive"] == int(
        ((oof["reference_label"] == 0) & (oof["sae_prediction"] == 1)).sum()
    )
    prompt = build_case_review_prompt(evidence.iloc[0].to_dict())
    assert "positive_latent_evidence" in prompt
    assert "not verified human gold" in prompt
    print("test_sae_disagreement_semantic_analysis_smoke passed")


if __name__ == "__main__":
    main()
