"""Run latent-semantic analysis for sampled stable-core SAE disagreements."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nlp_re_base.sae_disagreement_semantic_analysis import (
    SemanticDisagreementConfig,
    run_semantic_disagreement_analysis,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-store",
        default="outputs/misc_full_sae_eval/feature_store/utterance_features.pt",
    )
    parser.add_argument(
        "--label-matrix", default="outputs/misc_full_sae_eval/label_matrix.csv"
    )
    parser.add_argument(
        "--stable-core",
        default="outputs/cross_val/stable_topk_selection/stable_topk_latent_set.csv",
    )
    parser.add_argument(
        "--review-table",
        default="outputs/misc_full_sae_eval/interpretability/sae_annotation_audit_stable_core/annotation_review_table.csv",
    )
    parser.add_argument(
        "--oof-predictions",
        default="outputs/misc_full_sae_eval/interpretability/sae_annotation_audit_stable_core/oof_sae_predictions.csv",
    )
    parser.add_argument(
        "--disagreement-candidates",
        default="outputs/misc_full_sae_eval/interpretability/sae_annotation_audit_stable_core/disagreement_candidates.csv",
    )
    parser.add_argument(
        "--explanations",
        default="outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/deepseek_v4_flash_top50_unique_text_induction/explainer_outputs/validated_explanations.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/misc_full_sae_eval/interpretability/sae_annotation_audit_stable_core/semantic_case_analysis",
    )
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--concurrency", type=int, default=50)
    args = parser.parse_args()
    config = SemanticDisagreementConfig(model=args.model, concurrency=args.concurrency)
    result = run_semantic_disagreement_analysis(
        feature_store_path=args.feature_store,
        label_matrix_path=args.label_matrix,
        stable_core_path=args.stable_core,
        review_table_path=args.review_table,
        oof_predictions_path=args.oof_predictions,
        disagreement_candidates_path=args.disagreement_candidates,
        explanations_path=args.explanations,
        output_dir=args.output_dir,
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        config=config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
