"""Run split-half TopK reproducibility on the filtered MISC SAE latent pool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.cross_val_framework import (  # noqa: E402
    DEFAULT_LABELS,
    load_filtered_inputs,
    save_dedup_group_mapping,
)
from nlp_re_base.topk_reproducibility import run_topk_reproducibility  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-store", default="outputs/misc_full_sae_eval/feature_store/utterance_features.pt")
    parser.add_argument("--label-matrix", default="outputs/misc_full_sae_eval/label_matrix.csv")
    parser.add_argument(
        "--feature-filter-audit",
        default="outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered/feature_filter_audit.csv",
    )
    parser.add_argument(
        "--reference-matrix",
        default="outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered/latent_label_matrix.csv",
        help="Full-data filtered association matrix used to compare repeated split TopK coverage.",
    )
    parser.add_argument("--output-dir", default="outputs/cross_val/topk_reproducibility")
    parser.add_argument("--dedup-output", default="outputs/cross_val/dedup_group_mapping.csv")
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--group-column", default="source_file")
    parser.add_argument("--ranking-metrics", nargs="+", default=["cohens_d", "directional_auc"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--top-k-grid-max",
        type=int,
        default=None,
        help="If set, compute TopK stability for every K from 1..top-k-grid-max in addition to --top-k.",
    )
    parser.add_argument(
        "--top-k-grid",
        nargs="+",
        type=int,
        default=None,
        help="Explicit TopK grid. Overrides --top-k-grid-max when provided.",
    )
    parser.add_argument("--n-repeats", type=int, default=50)
    parser.add_argument("--null-iter", type=int, default=10000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-positive", type=int, default=10)
    parser.add_argument("--min-negative", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[E0] writing dedup mapping: {args.dedup_output}")
    save_dedup_group_mapping(args.label_matrix, args.dedup_output)
    print("[load] filtered inputs")
    inputs = load_filtered_inputs(
        feature_store_path=args.feature_store,
        label_matrix_path=args.label_matrix,
        feature_filter_audit_path=args.feature_filter_audit,
        labels=args.labels,
    )
    print(f"[E1] rows={len(inputs.label_matrix)} filtered_latents={len(inputs.latent_indices)}")
    top_k_grid = args.top_k_grid
    if top_k_grid is None and args.top_k_grid_max is not None:
        top_k_grid = list(range(1, int(args.top_k_grid_max) + 1))
    manifest = run_topk_reproducibility(
        inputs,
        output_dir=args.output_dir,
        reference_matrix_path=args.reference_matrix,
        group_column=args.group_column,
        ranking_metrics=args.ranking_metrics,
        top_k=args.top_k,
        top_k_grid=top_k_grid,
        n_repeats=args.n_repeats,
        null_iter=args.null_iter,
        random_state=args.random_state,
        min_positive=args.min_positive,
        min_negative=args.min_negative,
        chunk_size=args.chunk_size,
    )
    print(f"[done] outputs: {manifest['outputs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
