"""Run MISC minimal filtered-TopK latent set search against full-SAE AUC."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.minimal_full_sae_ratio_subspace import (  # noqa: E402
    MinimalFullSaeRatioSubspaceConfig,
    run_minimal_full_sae_ratio_subspace,
)
from nlp_re_base.minimal_sufficient_subspace_v2 import DEFAULT_LABELS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find minimal filtered-TopK SAE latent sets reaching a ratio of full-SAE probe AUC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--association-matrix",
        default="outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered/latent_label_matrix.csv",
        help="Filtered latent-label association matrix used as the candidate source.",
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
        "--feature-filter-audit",
        default="outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered/feature_filter_audit.csv",
        help="Feature filter audit used to verify candidates are in the keep=True pool.",
    )
    parser.add_argument(
        "--full-probe-fold-rows",
        default="outputs/misc_full_sae_eval/interpretability/ranked_sae_subspace_probe_pca_fixed/full_probe_by_label.csv",
        help="Existing fold-level probe rows used for full_sae_latents AUC targets.",
    )
    parser.add_argument(
        "--recompute-full-sae-baseline",
        action="store_true",
        help="Ignore --full-probe-fold-rows and refit full-SAE probes inside this run.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/misc_full_sae_eval/interpretability/minimal_full_sae_ratio_subspace_95",
        help="Output directory.",
    )
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--target-ratio", type=float, default=0.95)
    parser.add_argument("--candidate-top-k", type=int, default=100)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--split-policy",
        default="stratified-group-kfold",
        choices=["stratified-group-kfold", "stratified-kfold"],
    )
    parser.add_argument("--group-column", default="file_id")
    parser.add_argument("--min-full-auc", type=float, default=0.70)
    parser.add_argument("--max-search-k", type=int, default=100)
    parser.add_argument("--precision-k", type=int, default=50)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--solver", default="liblinear")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--no-standardize", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = MinimalFullSaeRatioSubspaceConfig(
        labels=tuple(label.upper() for label in args.labels),
        target_ratio=args.target_ratio,
        candidate_top_k=args.candidate_top_k,
        cv_folds=args.cv_folds,
        split_policy=args.split_policy,
        group_column=args.group_column,
        min_full_auc=args.min_full_auc,
        max_search_k=args.max_search_k,
        precision_k=args.precision_k,
        C=args.C,
        solver=args.solver,
        random_state=args.random_state,
        max_iter=args.max_iter,
        standardize=not args.no_standardize,
    )
    result = run_minimal_full_sae_ratio_subspace(
        association_matrix=args.association_matrix,
        feature_store=args.feature_store,
        label_matrix=args.label_matrix,
        output_dir=args.output_dir,
        feature_filter_audit=args.feature_filter_audit,
        full_probe_fold_rows=None if args.recompute_full_sae_baseline else args.full_probe_fold_rows,
        config=config,
    )
    summary = result["summary"]
    print("Completed minimal full-SAE ratio subspace search.")
    print(f"Output dir: {result['output_dir']}")
    print(
        summary[
            [
                "label",
                "formal_status",
                "fragmentation_class",
                "full_sae_auc_mean",
                "target_auc_mean",
                "minimal_k_median",
                "final_selected_latents",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
