"""Aggregate current stable-core feature-card explanations by MISC label."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nlp_re_base.feature_card_representation_analysis import run_single_choice_feature_card_analysis


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stable-core", default="outputs/cross_val/stable_topk_selection/stable_topk_latent_set.csv")
    parser.add_argument(
        "--explanations",
        default="outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/deepseek_v4_flash_top50_unique_text_induction/explainer_outputs/validated_explanations.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/misc_full_sae_eval/interpretability/feature_card_representation_analysis_single_type",
    )
    args = parser.parse_args()
    result = run_single_choice_feature_card_analysis(
        stable_core_path=args.stable_core,
        explanations_path=args.explanations,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
