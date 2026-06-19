"""Run MISC minimal sufficient SAE latent subspace v2 analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.minimal_sufficient_subspace_v2 import (  # noqa: E402
    DEFAULT_LABELS,
    MinimalSufficientSubspaceConfig,
    run_minimal_sufficient_subspace_v2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search minimal sufficient SAE latent subspaces for MISC labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--association-matrix",
        default="outputs/misc_full_sae_eval/interpretability/latent_space_search_v2/latent_label_association_v2.csv",
        help="Legacy latent-label association matrix used as the minimal-subspace candidate source.",
    )
    parser.add_argument(
        "--thresholded-sets",
        default="outputs/misc_full_sae_eval/interpretability/latent_space_search_v2/thresholded_latent_sets_v2.csv",
        help="Optional legacy candidate seed table; kept for compatibility, not current formal evidence.",
    )
    parser.add_argument(
        "--feature-store",
        default="outputs/misc_full_sae_eval/feature_store/utterance_features.pt",
        help="Utterance-level SAE feature store.",
    )
    parser.add_argument(
        "--label-matrix",
        default="outputs/misc_full_sae_eval/label_matrix.csv",
        help="MISC label matrix aligned to the feature store.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/misc_full_sae_eval/interpretability/minimal_sufficient_subspace_v2",
        help="Output directory.",
    )
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--candidate-top-k", type=int, default=100)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--min-auc", type=float, default=0.70)
    parser.add_argument("--auc-tolerance", type=float, default=0.02)
    parser.add_argument("--auprc-tolerance", type=float, default=0.03)
    parser.add_argument("--precision-lift-tolerance", type=float, default=0.05)
    parser.add_argument("--precision-k", type=int, default=50)
    parser.add_argument("--max-search-k", type=int, default=100)
    parser.add_argument("--stability-jaccard-threshold", type=float, default=0.40)
    parser.add_argument("--activation-threshold", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = MinimalSufficientSubspaceConfig(
        labels=tuple(label.upper() for label in args.labels),
        candidate_top_k=args.candidate_top_k,
        cv_folds=args.cv_folds,
        min_auc=args.min_auc,
        auc_tolerance=args.auc_tolerance,
        auprc_tolerance=args.auprc_tolerance,
        precision_lift_tolerance=args.precision_lift_tolerance,
        precision_k=args.precision_k,
        max_search_k=args.max_search_k,
        stability_jaccard_threshold=args.stability_jaccard_threshold,
        activation_threshold=args.activation_threshold,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )
    result = run_minimal_sufficient_subspace_v2(
        association_matrix=args.association_matrix,
        thresholded_sets=args.thresholded_sets,
        feature_store=args.feature_store,
        label_matrix=args.label_matrix,
        output_dir=args.output_dir,
        config=config,
        make_figures=not args.no_figures,
    )
    summary = result["summary"]
    print("Completed minimal sufficient subspace v2.")
    print(f"Output dir: {result['output_dir']}")
    print(summary[["label", "formal_status", "fragmentation_class", "full_auc_mean", "minimal_k_median", "final_selected_latents"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
