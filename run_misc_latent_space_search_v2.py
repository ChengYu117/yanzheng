"""Run the v2 MISC SAE latent-space search pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.latent_space_search_v2 import (  # noqa: E402
    DEFAULT_LABELS,
    LatentSpaceSearchV2Config,
    run_latent_space_search_v2,
)


def _parse_k_values(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected comma-separated K values.")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("K values must be integers.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refactored MISC SAE latent-space search v2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--matrix",
        default="outputs/misc_full_sae_eval/functional/misc_label_mapping/latent_label_matrix.csv",
        help="Latent x MISC label association matrix.",
    )
    parser.add_argument(
        "--feature-store",
        default="outputs/misc_full_sae_eval/feature_store/utterance_features.pt",
        help="Utterance-level SAE feature store used for Precision@100 and top examples.",
    )
    parser.add_argument(
        "--label-matrix",
        default="outputs/misc_full_sae_eval/label_matrix.csv",
        help="MISC label indicator matrix aligned to the feature store.",
    )
    parser.add_argument(
        "--records",
        default="outputs/misc_full_sae_eval/records.jsonl",
        help="Records JSONL for top-activating utterance text.",
    )
    parser.add_argument(
        "--step6-table",
        default="outputs/misc_full_sae_eval/interpretability/baseline_comparison_step6/table4_baseline_comparison.csv",
        help="Optional Step 6 SAE/PCA/raw-hidden baseline table.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/misc_full_sae_eval/interpretability/latent_space_search_v2",
        help="Output directory for v2 artifacts.",
    )
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--top-candidate-k", type=int, default=20)
    parser.add_argument("--k-values", type=_parse_k_values, default=(5, 10, 20, 50, 100))
    parser.add_argument("--min-directional-auc", type=float, default=0.70)
    parser.add_argument("--min-abs-cohens-d", type=float, default=0.50)
    parser.add_argument("--precision-delta", type=float, default=0.10)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--semantic-top-examples", type=int, default=12)
    parser.add_argument("--random-baseline-repeats", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = LatentSpaceSearchV2Config(
        labels=tuple(label.upper() for label in args.labels),
        top_candidate_k=args.top_candidate_k,
        k_values=tuple(args.k_values),
        min_directional_auc=args.min_directional_auc,
        min_abs_cohens_d=args.min_abs_cohens_d,
        precision_delta=args.precision_delta,
        chunk_size=args.chunk_size,
        semantic_top_examples=args.semantic_top_examples,
        random_baseline_repeats=args.random_baseline_repeats,
        random_state=args.random_state,
    )
    result = run_latent_space_search_v2(
        matrix_path=args.matrix,
        feature_store=args.feature_store,
        label_matrix=args.label_matrix,
        records_path=args.records,
        output_dir=args.output_dir,
        step6_table=args.step6_table,
        config=config,
        make_figures=not args.no_figures,
    )
    summary = result["summary"]
    print("Completed latent-space search v2.")
    print(f"Output dir: {result['output_dir']}")
    print(f"Top20 edges: {summary['top20_edges']}")
    print(f"Thresholded edges: {summary['thresholded_edges']}")
    print(f"Labels without stable latents: {summary['labels_without_stable_latents']}")
    print(f"Report: {summary['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
