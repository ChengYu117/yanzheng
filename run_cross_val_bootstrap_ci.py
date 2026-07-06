"""Run grouped bootstrap CIs for filtered MISC SAE latent associations."""

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
from nlp_re_base.effect_size_ci import run_bootstrap_ci  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-store", default="outputs/misc_full_sae_eval/feature_store/utterance_features.pt")
    parser.add_argument("--label-matrix", default="outputs/misc_full_sae_eval/label_matrix.csv")
    parser.add_argument(
        "--feature-filter-audit",
        default="outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered/feature_filter_audit.csv",
    )
    parser.add_argument(
        "--association",
        default="outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered/latent_label_matrix.csv",
    )
    parser.add_argument("--output-dir", default="outputs/cross_val/bootstrap_ci")
    parser.add_argument("--dedup-output", default="outputs/cross_val/dedup_group_mapping.csv")
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--group-column", default="source_file")
    parser.add_argument("--ranking-metric", default="cohens_d")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
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
    print(
        f"[E2] rows={len(inputs.label_matrix)} filtered_latents={len(inputs.latent_indices)} "
        f"top_k={args.top_k} n_bootstrap={args.n_bootstrap}"
    )
    manifest = run_bootstrap_ci(
        inputs,
        association_matrix_path=args.association,
        output_dir=args.output_dir,
        group_column=args.group_column,
        ranking_metric=args.ranking_metric,
        top_k=args.top_k,
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
        chunk_size=args.chunk_size,
    )
    print(f"[done] outputs: {manifest['outputs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

