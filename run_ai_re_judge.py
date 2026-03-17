"""Standalone CLI for the SAE-RE AI expert-proxy review pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.ai_re_judge import DEFAULT_GROUP_NAMES, run_ai_judge_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the standalone AI judge pipeline on an exported judge bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Evaluation output directory or judge_bundle directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for AI judge outputs.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Judge model name. Falls back to OPENAI_MODEL.",
    )
    parser.add_argument(
        "--top-latents",
        type=int,
        default=20,
        help="Number of top candidate latents to review.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top high-activation utterances per latent/group.",
    )
    parser.add_argument(
        "--control-n",
        type=int,
        default=5,
        help="Number of control utterances per latent/group.",
    )
    parser.add_argument(
        "--groups",
        default=",".join(DEFAULT_GROUP_NAMES),
        help="Comma-separated group names to review.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge sampling temperature.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for malformed or failed judge responses.",
    )
    parser.add_argument(
        "--dry-run-prompts",
        action="store_true",
        help="Only render prompts and placeholder outputs without calling the API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = [item.strip() for item in args.groups.split(",") if item.strip()]
    results = run_ai_judge_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model=args.model,
        top_latents=args.top_latents,
        top_n=args.top_n,
        control_n=args.control_n,
        groups=groups,
        temperature=args.temperature,
        max_retries=args.max_retries,
        dry_run_prompts=args.dry_run_prompts,
    )

    print("\n=== AI Judge Pipeline Complete ===")
    print(f"status: {results['status']}")
    print(f"judge_model: {results['judge_model']}")
    print(f"output_dir: {results['output_dir']}")
    print(f"utterance_reviews: {results['n_utterance_reviews']}")
    print(f"latent_reviews: {results['n_latent_reviews']}")
    print(f"group_reviews: {results['n_group_reviews']}")


if __name__ == "__main__":
    main()
