"""Run minimal non-redundant SAE latent set analysis for MISC labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.misc_minimal_latents import (  # noqa: E402
    DEFAULT_CORE_LABELS,
    DEFAULT_LEAF_LABELS,
    MinimalLatentConfig,
    load_feature_store,
    run_misc_minimal_latent_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generalize the RE-only latent group-structure analysis to all MISC "
            "labels using saved utterance-level SAE features."
        )
    )
    parser.add_argument(
        "--feature-store",
        default="outputs/misc_full_sae_eval/feature_store/utterance_features.pt",
        help="Saved utterance-level SAE features (.pt/.npy/.npz).",
    )
    parser.add_argument(
        "--label-matrix",
        default="outputs/misc_full_sae_eval/label_matrix.csv",
        help="MISC label indicator matrix aligned to the feature store.",
    )
    parser.add_argument(
        "--topk-matrix",
        default="outputs/misc_full_sae_eval/interpretability/mapping_structure/topk_candidate_matrix.csv",
        help="Per-label TopK latent candidate matrix.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/misc_full_sae_eval/interpretability/minimal_latent_sets",
        help="Directory for CSV/JSON/Markdown/figure outputs.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(DEFAULT_CORE_LABELS),
        help="Labels to analyze. Default: all core labels except OTHER.",
    )
    parser.add_argument(
        "--leaf-only",
        action="store_true",
        help="Analyze only leaf/atomic labels: RES REC QUO QUC GI SU AF.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--loo-k", type=int, default=10)
    parser.add_argument(
        "--candidate-mode",
        choices=("topk", "support"),
        default="topk",
        help="Use all TopK candidates or only support-edge candidates.",
    )
    parser.add_argument("--target-effect-fraction", type=float, default=0.90)
    parser.add_argument("--min-auc", type=float, default=0.70)
    parser.add_argument("--max-redundancy", type=float, default=0.30)
    parser.add_argument("--activation-threshold", type=float, default=0.0)
    parser.add_argument("--n-bootstrap", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = list(DEFAULT_LEAF_LABELS) if args.leaf_only else [label.upper() for label in args.labels]
    config = MinimalLatentConfig(
        top_k=args.top_k,
        loo_k=args.loo_k,
        candidate_mode=args.candidate_mode,
        target_effect_fraction=args.target_effect_fraction,
        min_auc=args.min_auc,
        max_redundancy=args.max_redundancy,
        activation_threshold=args.activation_threshold,
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )

    print(f"[load] features: {args.feature_store}")
    features = load_feature_store(args.feature_store)
    print(f"[load] feature shape: {features.shape}")
    label_df = pd.read_csv(args.label_matrix)
    topk_df = pd.read_csv(args.topk_matrix)
    if len(label_df) != features.shape[0]:
        raise ValueError(
            f"Label rows ({len(label_df)}) do not match feature rows ({features.shape[0]})"
        )

    output_dir = Path(args.output_dir)
    print(f"[run] labels={labels}")
    result = run_misc_minimal_latent_analysis(
        features=features,
        label_df=label_df,
        topk_matrix=topk_df,
        output_dir=output_dir,
        labels=labels,
        config=config,
    )
    summary = result["summary"]
    print("[done] minimal latent set summary:")
    cols = [
        "label",
        "full_auc",
        "selected_k",
        "selected_status",
        "n_support_edges",
        "compact_low_redundancy",
    ]
    print(summary[cols].to_string(index=False))
    print(f"[done] outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
