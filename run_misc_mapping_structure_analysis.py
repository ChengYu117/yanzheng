"""Analyze the structure of the MISC label-to-SAE latent mapping."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.mapping_structure import (  # noqa: E402
    DEFAULT_CORE_LABELS,
    DEFAULT_HIERARCHY_SPECS,
    DEFAULT_TOP_K_VALUES,
    run_mapping_structure_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MISC Mapping Structure analysis from an existing Latent x Label matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mapping-dir",
        default="outputs/misc_full_sae_eval/functional/misc_label_mapping",
        help="Directory containing latent_label_matrix.csv and label_summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/misc_full_sae_eval/interpretability/mapping_structure",
        help="Directory for mapping-structure outputs.",
    )
    parser.add_argument("--top-k", nargs="+", type=int, default=DEFAULT_TOP_K_VALUES)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument(
        "--label-hierarchy",
        nargs="+",
        default=DEFAULT_HIERARCHY_SPECS,
        help="Hierarchy specs such as RE:RES,REC QU:QUO,QUC.",
    )
    parser.add_argument(
        "--core-labels",
        nargs="+",
        default=DEFAULT_CORE_LABELS,
        help="Core counseling behavior labels for latent role taxonomy.",
    )
    parser.add_argument(
        "--doc-report",
        default="doc/MISC Mapping Structure分析报告.md",
        help="Optional doc report path. Pass an empty string to skip.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip optional matplotlib figure generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    doc_report = args.doc_report if args.doc_report else None
    metrics = run_mapping_structure_analysis(
        mapping_dir=args.mapping_dir,
        output_dir=args.output_dir,
        top_k_values=args.top_k,
        hierarchy_specs=args.label_hierarchy,
        core_labels=args.core_labels,
        fdr_alpha=args.fdr_alpha,
        doc_report=doc_report,
        make_figures=not args.no_figures,
    )
    print("Completed MISC Mapping Structure analysis.")
    print(f"Output dir: {metrics['output_dir']}")
    print(f"Report: {metrics['files']['mapping_structure_report']}")
    if metrics["files"].get("doc_report"):
        print(f"Doc report: {metrics['files']['doc_report']}")
    print(f"Significant latent-label edges: {metrics['significant_latent_label_edges']}")
    print(f"Latents with any significant label: {metrics['latents_with_any_significant_label']}")
    print(f"Single-label latents: {metrics['single_label_latents']}")
    print(f"Multi-label latents: {metrics['multi_label_latents']}")


if __name__ == "__main__":
    main()
