"""Select stable per-label Top-K latent sets from AUC@K and CV evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.cross_val_framework import DEFAULT_LABELS  # noqa: E402
from nlp_re_base.stable_topk_selection import (  # noqa: E402
    StableTopKSelectionConfig,
    run_stable_topk_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--auc-curve",
        default="outputs/misc_full_sae_eval/interpretability/ranked_sae_subspace_probe_k001_100/auc_by_k_curve_0_100.csv",
    )
    parser.add_argument(
        "--topk-grid-summary",
        default="outputs/cross_val/topk_reproducibility/repeated_split_topk_grid_summary.csv",
    )
    parser.add_argument(
        "--inclusion-frequency",
        default="outputs/cross_val/topk_reproducibility/topk_inclusion_frequency.csv",
    )
    parser.add_argument(
        "--association-matrix",
        default="outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered/latent_label_matrix.csv",
    )
    parser.add_argument(
        "--bootstrap-ci",
        default="outputs/cross_val/bootstrap_ci/bootstrap_ci_by_label_latent.csv",
    )
    parser.add_argument(
        "--cross-quality",
        default="outputs/cross_val/cross_quality_validation/cross_quality_auc_comparison.csv",
    )
    parser.add_argument("--output-dir", default="outputs/cross_val/stable_topk_selection")
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--ranking-metric", default="cohens_d")
    parser.add_argument("--max-k", type=int, default=100)
    parser.add_argument("--auc-gap-floor", type=float, default=0.01)
    parser.add_argument("--pass-rate-threshold", type=float, default=0.95)
    parser.add_argument("--jaccard-p05-threshold", type=float, default=0.40)
    parser.add_argument("--inclusion-stable-threshold", type=float, default=0.70)
    parser.add_argument("--inclusion-boundary-threshold", type=float, default=0.40)
    parser.add_argument("--cross-quality-drop-threshold", type=float, default=0.05)
    parser.add_argument("--require-cross-quality-gate", action="store_true")
    parser.add_argument("--require-label-stability-gate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = StableTopKSelectionConfig(
        labels=tuple(label.upper() for label in args.labels),
        ranking_metric=args.ranking_metric,
        max_k=args.max_k,
        auc_gap_floor=args.auc_gap_floor,
        pass_rate_threshold=args.pass_rate_threshold,
        jaccard_p05_threshold=args.jaccard_p05_threshold,
        inclusion_stable_threshold=args.inclusion_stable_threshold,
        inclusion_boundary_threshold=args.inclusion_boundary_threshold,
        cross_quality_drop_threshold=args.cross_quality_drop_threshold,
        require_cross_quality_gate=args.require_cross_quality_gate,
        require_label_stability_gate=args.require_label_stability_gate,
    )
    result = run_stable_topk_selection(
        auc_curve_path=args.auc_curve,
        topk_grid_summary_path=args.topk_grid_summary,
        inclusion_frequency_path=args.inclusion_frequency,
        association_matrix_path=args.association_matrix,
        bootstrap_ci_path=args.bootstrap_ci,
        cross_quality_path=args.cross_quality,
        output_dir=args.output_dir,
        config=config,
    )
    print("[done] Stable Top-K selection")
    print(result["summary"].to_string(index=False))
    print(f"[done] outputs: {result['manifest']['outputs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
