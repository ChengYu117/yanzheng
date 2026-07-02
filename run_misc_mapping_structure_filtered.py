"""Run mapping-structure analysis on the filtered MISC latent pool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.mapping_structure import (  # noqa: E402
    DEFAULT_CORE_LABELS,
    DEFAULT_HIERARCHY_SPECS,
    DEFAULT_INTERPRETABILITY_TOP_K,
    DEFAULT_TOP_K_VALUES,
    run_mapping_structure_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MISC Mapping Structure analysis from the filtered Latent x Label matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mapping-dir",
        default="outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered",
        help="Directory containing filtered latent_label_matrix.csv and label_summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/misc_full_sae_eval/interpretability/mapping_structure_filtered",
        help="Directory for filtered mapping-structure outputs.",
    )
    parser.add_argument("--top-k", nargs="+", type=int, default=DEFAULT_TOP_K_VALUES)
    parser.add_argument("--analysis-top-k", type=int, default=DEFAULT_INTERPRETABILITY_TOP_K)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--label-hierarchy", nargs="+", default=DEFAULT_HIERARCHY_SPECS)
    parser.add_argument("--core-labels", nargs="+", default=DEFAULT_CORE_LABELS)
    parser.add_argument(
        "--doc-report",
        default="",
        help="Optional doc report path. Empty by default to avoid overwriting legacy docs.",
    )
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_mapping_structure_analysis(
        mapping_dir=args.mapping_dir,
        output_dir=args.output_dir,
        top_k_values=args.top_k,
        analysis_top_k=args.analysis_top_k,
        hierarchy_specs=args.label_hierarchy,
        core_labels=args.core_labels,
        fdr_alpha=args.fdr_alpha,
        doc_report=args.doc_report or None,
        make_figures=not args.no_figures,
    )
    print("Completed filtered MISC Mapping Structure analysis.")
    print(f"Output dir: {metrics['output_dir']}")
    print(f"Report: {metrics['files']['mapping_structure_report']}")
    print(f"Interpretability scope: Top-{metrics['analysis_top_k']} per label")
    print(f"TopK latent-label edges: {metrics['topk_latent_label_edges']}")
    print(f"TopK unique latents: {metrics['topk_unique_latents']}")
    print(f"TopK multi-label latents: {metrics['topk_multi_label_latents']}")


if __name__ == "__main__":
    main()

