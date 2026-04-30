"""Run follow-up MISC SAE interpretability analyses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.behavior_interpretability import (  # noqa: E402
    DEFAULT_STAGE_OUTPUT,
    run_followup_interpretability_analysis,
)
from nlp_re_base.mapping_structure import DEFAULT_CORE_LABELS, DEFAULT_HIERARCHY_SPECS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run R2 behavior asymmetry and latent case analyses for MISC SAE outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eval-dir", default="outputs/misc_full_sae_eval")
    parser.add_argument("--output-dir", default=DEFAULT_STAGE_OUTPUT)
    parser.add_argument("--labels", nargs="+", default=DEFAULT_CORE_LABELS)
    parser.add_argument("--label-hierarchy", nargs="+", default=DEFAULT_HIERARCHY_SPECS)
    parser.add_argument("--top-latents-per-label", type=int, default=5)
    parser.add_argument("--top-examples-per-latent", type=int, default=12)
    parser.add_argument(
        "--doc-report",
        default="doc/MISC后续可解释性阶段分析报告.md",
        help="Optional doc report path. Pass an empty string to skip.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_followup_interpretability_analysis(
        eval_dir=args.eval_dir,
        output_dir=args.output_dir,
        labels=args.labels,
        hierarchy_specs=args.label_hierarchy,
        top_latents_per_label=args.top_latents_per_label,
        top_examples_per_latent=args.top_examples_per_latent,
        doc_report=args.doc_report or None,
    )
    print("Completed follow-up MISC interpretability analysis.")
    print(f"Output dir: {metrics['output_dir']}")
    print(f"Stage report: {metrics['files']['stage_report']}")
    if metrics["files"].get("doc_report"):
        print(f"Doc report: {metrics['files']['doc_report']}")
    print(f"Case cards: {metrics['case_metrics']['n_case_cards']}")


if __name__ == "__main__":
    main()
