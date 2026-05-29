"""Run Step 6 SAE-vs-PCA-vs-raw baseline comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.baseline_comparison import (  # noqa: E402
    DEFAULT_LABELS,
    BaselineComparisonConfig,
    load_matrix,
    run_baseline_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 6 structural baseline comparison.")
    parser.add_argument(
        "--sae-features",
        default="outputs/misc_full_sae_eval/feature_store/utterance_features.pt",
        help="Utterance-level SAE latent feature matrix.",
    )
    parser.add_argument(
        "--raw-hidden",
        default="outputs/misc_full_sae_eval/feature_store/utterance_activations.pt",
        help="Utterance-level raw hidden-state matrix from the same layer/pooling run.",
    )
    parser.add_argument(
        "--label-matrix",
        default="outputs/misc_full_sae_eval/label_matrix.csv",
        help="MISC label matrix aligned to the feature rows.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/misc_full_sae_eval/interpretability/baseline_comparison_step6",
        help="Output directory for Table 4, metrics, and figures.",
    )
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument(
        "--pca-components",
        default="1024",
        help="PCA component count, or one of: auto, full, same-as-raw, d_model.",
    )
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--dev-size", type=float, default=0.15)
    parser.add_argument("--min-directional-auc", type=float, default=0.65)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--precision-k", type=int, default=50)
    parser.add_argument("--classifier-top-features", type=int, default=200)
    parser.add_argument("--association-chunk-size", type=int, default=512)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _parse_pca_components(value: str) -> int | str:
    lowered = str(value).lower()
    if lowered in {"auto", "full", "same-as-raw", "d_model"}:
        return lowered
    return int(value)


def main() -> int:
    args = parse_args()
    config = BaselineComparisonConfig(
        pca_components=_parse_pca_components(args.pca_components),
        train_size=args.train_size,
        dev_size=args.dev_size,
        min_directional_auc=args.min_directional_auc,
        top_k=args.top_k,
        precision_k=args.precision_k,
        classifier_top_features=args.classifier_top_features,
        association_chunk_size=args.association_chunk_size,
        random_state=args.random_state,
    )

    print(f"[load] SAE features: {args.sae_features}")
    sae_features = load_matrix(args.sae_features)
    print(f"[load] SAE shape: {sae_features.shape}")
    print(f"[load] raw hidden: {args.raw_hidden}")
    raw_hidden = load_matrix(args.raw_hidden)
    print(f"[load] raw hidden shape: {raw_hidden.shape}")
    label_df = pd.read_csv(args.label_matrix)
    if len(label_df) != sae_features.shape[0] or len(label_df) != raw_hidden.shape[0]:
        raise ValueError(
            f"Row mismatch: labels={len(label_df)}, sae={sae_features.shape[0]}, raw={raw_hidden.shape[0]}"
        )

    output_dir = Path(args.output_dir)
    result = run_baseline_comparison(
        sae_features=sae_features,
        raw_hidden=raw_hidden,
        label_df=label_df,
        output_dir=output_dir,
        labels=[label.upper() for label in args.labels],
        config=config,
    )
    table = result["table4"]
    print("[done] Table 4:")
    print(table.to_string(index=False))
    print(f"[done] outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
