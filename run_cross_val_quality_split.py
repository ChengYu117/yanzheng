"""Run high/low cross-quality validation for filtered MISC SAE latent candidates."""

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
from nlp_re_base.cross_quality_val import run_cross_quality_validation  # noqa: E402


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
        help="Full-data filtered association matrix used to ensure full TopK candidates get high/low checks.",
    )
    parser.add_argument("--output-dir", default="outputs/cross_val/cross_quality_validation")
    parser.add_argument("--dedup-output", default="outputs/cross_val/dedup_group_mapping.csv")
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--ranking-metric", default="cohens_d")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--stable-drop-threshold", type=float, default=0.05)
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
    print(f"[E3] rows={len(inputs.label_matrix)} filtered_latents={len(inputs.latent_indices)}")
    manifest = run_cross_quality_validation(
        inputs,
        output_dir=args.output_dir,
        reference_matrix_path=args.reference_matrix,
        ranking_metric=args.ranking_metric,
        top_k=args.top_k,
        stable_drop_threshold=args.stable_drop_threshold,
        min_positive=args.min_positive,
        min_negative=args.min_negative,
        chunk_size=args.chunk_size,
    )
    print(f"[done] outputs: {manifest['outputs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
