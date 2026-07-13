from __future__ import annotations

import pandas as pd

from src.nlp_re_base.feature_card_representation_analysis import (
    build_feature_card_analysis,
    build_single_choice_feature_card_analysis,
)


def main() -> None:
    stable = pd.DataFrame(
        {
            "label": ["QU", "QUO", "GI"],
            "latent_idx": [1, 1, 2],
            "stable_set_role": ["stable_core"] * 3,
            "full_data_rank": [1, 1, 1],
        }
    )
    explanations = pd.DataFrame(
        {
            "latent_idx": [1, 2],
            "short_name": ["what_questions", "medication_advice"],
            "candidate_explanation": ["open question", "medication information"],
            "surface_pattern": ["what question", "medication terms"],
            "semantic_pattern": ["asks a question", "provides medication advice"],
            "induction_status": ["both_majority", "both_majority"],
            "surface_majority_gate_passed": [True, True],
            "semantic_majority_gate_passed": [True, True],
            "surface_confidence": [0.9, 0.8],
            "semantic_confidence": [0.8, 0.9],
            "surface_raw_support_fraction": [0.9, 0.8],
            "semantic_raw_support_fraction": [0.8, 0.9],
        }
    )
    joined, summary, themes, matrix = build_feature_card_analysis(stable, explanations)
    assert len(joined) == 3
    assert joined.loc[joined["label"].eq("QU"), "shared_latent_flag"].all()
    assert joined.loc[joined["label"].eq("GI"), "candidate_theme"].item() == "medication_information"
    assert summary.set_index("label").loc["QU", "shared_fraction"] == 1.0
    assert matrix.set_index("label").loc["QU", "QUO"] == 1
    assert themes["n_associations"].sum() == 3

    single_stable = pd.DataFrame(
        {
            "label": ["RE", "REC", "REC", "RES", "QUO", "QUC"],
            "latent_idx": [10, 10, 11, 11, 12, 12],
            "stable_set_role": ["stable_core"] * 6,
            "full_data_rank": [1, 1, 2, 1, 1, 1],
        }
    )
    single_explanations = pd.DataFrame(
        {
            "latent_idx": [10, 11, 12],
            "short_name": ["reflection", "shared reflection", "question form"],
            "candidate_explanation": ["reflects", "reflects", "question"],
            "main_hypothesis": ["reflection", "reflection", "question"],
            "dominant_pattern": ["semantic", "mixed", "surface"],
            "surface_component": ["so you", "you", "what"],
            "semantic_component": ["reflection", "reflection", "asking"],
            "feature_type": ["semantic", "mixed", "surface"],
            "confidence": [0.8, 0.7, 0.9],
            "majority_gate_passed": [True, True, True],
            "induction_status": ["candidate_pattern"] * 3,
        }
    )
    single_joined, single_summary, _, pair_audit = build_single_choice_feature_card_analysis(
        single_stable, single_explanations
    )
    re_row = single_joined[single_joined["label"].eq("RE")].iloc[0]
    assert not re_row["non_hierarchical_shared_flag"]
    rec_11 = single_joined[
        single_joined["label"].eq("REC") & single_joined["latent_idx"].eq(11)
    ].iloc[0]
    assert rec_11["non_hierarchical_shared_labels"] == "RES"
    assert single_summary.set_index("label").loc["REC", "mixed"] == 1
    re_rec = pair_audit[
        pair_audit["label_left"].eq("RE") & pair_audit["label_right"].eq("REC")
    ].iloc[0]
    assert re_rec["relation_type"] == "hierarchical_definition"
    assert not re_rec["eligible_as_representation_evidence"]
    print("test_feature_card_representation_analysis_smoke passed")


if __name__ == "__main__":
    main()
