"""Prepare Task 5 SAE/PCA matched human-evaluation materials."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.baseline_comparison import load_matrix  # noqa: E402
from nlp_re_base.task5_matched_human_eval import (  # noqa: E402
    DEFAULT_LABELS,
    DEFAULT_PCA_DIMENSIONS,
    Task5PreparationConfig,
    prepare_task5_materials,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare blinded matched SAE/PCA materials; no human judgments are generated."
    )
    parser.add_argument(
        "--sae-features",
        default="outputs/misc_full_sae_eval/feature_store/utterance_features.pt",
    )
    parser.add_argument(
        "--raw-hidden",
        default="outputs/misc_full_sae_eval/feature_store/utterance_activations.pt",
    )
    parser.add_argument(
        "--label-matrix",
        default="outputs/misc_full_sae_eval/label_matrix.csv",
    )
    parser.add_argument(
        "--stable-core",
        default="outputs/cross_val/stable_topk_selection/stable_topk_latent_set.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/misc_full_sae_eval/interpretability/task5_matched_sae_pca_human_eval",
    )
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument(
        "--pca-dimensions",
        nargs="+",
        type=int,
        default=list(DEFAULT_PCA_DIMENSIONS),
    )
    parser.add_argument("--units-per-label", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--group-column", default="file_id")
    parser.add_argument("--discovery-count", type=int, default=10)
    parser.add_argument("--heldout-positive-count", type=int, default=10)
    parser.add_argument("--heldout-control-count", type=int, default=10)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.90)
    parser.add_argument("--matching-rank-penalty", type=float, default=0.001)
    parser.add_argument("--max-auc-difference", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Task5PreparationConfig(
        labels=tuple(str(label).upper() for label in args.labels),
        pca_dimensions=tuple(sorted(set(int(value) for value in args.pca_dimensions))),
        units_per_label=args.units_per_label,
        folds=args.folds,
        fold_index=args.fold_index,
        random_state=args.random_state,
        group_column=args.group_column,
        discovery_count=args.discovery_count,
        heldout_positive_count=args.heldout_positive_count,
        heldout_control_count=args.heldout_control_count,
        near_duplicate_threshold=args.near_duplicate_threshold,
        matching_rank_penalty=args.matching_rank_penalty,
        max_auc_difference=args.max_auc_difference,
    )
    print(f"[load] SAE features: {args.sae_features}")
    sae_features = load_matrix(args.sae_features)
    print(f"[load] SAE shape: {sae_features.shape}")
    print(f"[load] raw hidden: {args.raw_hidden}")
    raw_hidden = load_matrix(args.raw_hidden)
    print(f"[load] raw hidden shape: {raw_hidden.shape}")
    label_df = pd.read_csv(args.label_matrix)
    stable_core = pd.read_csv(args.stable_core)
    result = prepare_task5_materials(
        sae_features=sae_features,
        raw_hidden=raw_hidden,
        label_df=label_df,
        stable_core=stable_core,
        output_dir=args.output_dir,
        config=config,
    )
    print(f"[done] matched pairs: {len(result['matches'])}")
    print(f"[done] validation: {result['validation']['overall_status']}")
    print(f"[done] outputs: {args.output_dir}")
    print("[pending] human interpretation and review have not been run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
