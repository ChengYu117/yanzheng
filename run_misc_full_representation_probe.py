"""Run full-representation MISC label probes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.full_representation_probe import (  # noqa: E402
    DEFAULT_LABELS,
    DEFAULT_SAE_SUBSPACE_RANKINGS,
    FullRepresentationProbeConfig,
    load_matrix,
    run_full_representation_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate full SAE/raw/PCA representation decodability for MISC labels."
    )
    parser.add_argument(
        "--sae-features",
        default="outputs/misc_full_sae_eval/feature_store/utterance_features.pt",
        help="Utterance-level SAE latent feature matrix.",
    )
    parser.add_argument(
        "--raw-hidden",
        default="outputs/misc_full_sae_eval/feature_store/utterance_activations.pt",
        help="Utterance-level raw hidden-state matrix aligned to SAE features.",
    )
    parser.add_argument(
        "--label-matrix",
        default="outputs/misc_full_sae_eval/label_matrix.csv",
        help="MISC label matrix aligned to feature rows.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/misc_full_sae_eval/interpretability/full_representation_probe",
        help="Output directory for full-representation probe results.",
    )
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--split-policy",
        default="stratified-group-kfold",
        choices=["stratified-group-kfold", "stratified-kfold"],
        help="Cross-validation split policy.",
    )
    parser.add_argument("--group-column", default="file_id")
    parser.add_argument(
        "--pca-components",
        default="full",
        help="PCA components: full, auto, same-as-raw, d_model, or an integer.",
    )
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--solver", default="liblinear")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable train-fold standardization before fitting probes.",
    )
    parser.add_argument(
        "--include-sae-ranked-subspaces",
        action="store_true",
        help="Add SAE top-n subspace probes ranked within each training fold.",
    )
    parser.add_argument(
        "--subspace-max-n",
        type=int,
        default=200,
        help="Maximum top-n SAE features per label/fold/ranking.",
    )
    parser.add_argument(
        "--subspace-step",
        type=int,
        default=5,
        help="Step for top-n SAE feature grid after n=1.",
    )
    parser.add_argument(
        "--subspace-rankings",
        nargs="+",
        default=list(DEFAULT_SAE_SUBSPACE_RANKINGS),
        choices=list(DEFAULT_SAE_SUBSPACE_RANKINGS),
        help="Feature-ranking metrics for SAE top-n subspace probes.",
    )
    parser.add_argument(
        "--no-filter-sae-subspace-candidates",
        action="store_true",
        help="Disable the train-fold SAE feature quality filter before top-n ranking.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-label/fold progress logs.")
    return parser.parse_args()


def _parse_pca_components(value: str) -> int | str:
    lowered = str(value).lower()
    if lowered in {"auto", "full", "same-as-raw", "d_model"}:
        return lowered
    return int(value)


def _build_subspace_top_ns(max_n: int, step: int) -> tuple[int, ...]:
    max_n = max(1, int(max_n))
    step = max(1, int(step))
    values = {1}
    values.update(range(step, max_n + 1, step))
    return tuple(sorted(n for n in values if n <= max_n))


def main() -> int:
    args = parse_args()
    labels = tuple(label.upper() for label in args.labels)
    config = FullRepresentationProbeConfig(
        labels=labels,
        folds=args.folds,
        split_policy=args.split_policy,
        group_column=args.group_column,
        pca_components=_parse_pca_components(args.pca_components),
        C=args.C,
        solver=args.solver,
        max_iter=args.max_iter,
        random_state=args.random_state,
        standardize=not args.no_standardize,
        verbose=not args.quiet,
        include_sae_ranked_subspaces=args.include_sae_ranked_subspaces,
        sae_subspace_top_ns=_build_subspace_top_ns(args.subspace_max_n, args.subspace_step),
        sae_subspace_rankings=tuple(args.subspace_rankings),
        filter_sae_subspace_candidates=not args.no_filter_sae_subspace_candidates,
    )

    print(f"[load] SAE features: {args.sae_features}")
    sae_features = load_matrix(args.sae_features)
    print(f"[load] SAE shape: {sae_features.shape}")
    print(f"[load] raw hidden: {args.raw_hidden}")
    raw_hidden = load_matrix(args.raw_hidden)
    print(f"[load] raw hidden shape: {raw_hidden.shape}")
    print(f"[load] labels: {args.label_matrix}")
    label_df = pd.read_csv(args.label_matrix)

    result = run_full_representation_probe(
        sae_features=sae_features,
        raw_hidden=raw_hidden,
        label_df=label_df,
        output_dir=args.output_dir,
        labels=labels,
        config=config,
    )
    print("[done] Full-representation macro summary:")
    print(result["summary"].to_string(index=False))
    print(f"[done] outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
