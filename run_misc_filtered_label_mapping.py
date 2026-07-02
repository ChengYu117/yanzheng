"""Recompute MISC latent-label mapping with an SAE feature quality filter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.misc_label_mapping import (  # noqa: E402
    FeatureFilterConfig,
    load_feature_matrix,
    run_misc_label_mapping,
)

DEFAULT_LABELS = ("RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF")


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _attach_labels_from_matrix_if_needed(
    records: list[dict[str, Any]],
    label_matrix_path: str | Path | None,
    labels: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not label_matrix_path:
        return records
    path = Path(label_matrix_path)
    if not path.exists() or all(record.get("labels") for record in records):
        return records
    label_df = pd.read_csv(path)
    if len(label_df) != len(records):
        raise ValueError(
            f"label matrix rows ({len(label_df)}) must match records ({len(records)})"
        )
    missing = [label for label in labels if label not in label_df.columns]
    if missing:
        raise ValueError(f"label matrix is missing requested labels: {missing}")
    out: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        updated = dict(record)
        updated["labels"] = [label for label in labels if int(label_df.at[idx, label]) == 1]
        out.append(updated)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute MISC latent-label associations after filtering low-quality SAE features."
    )
    parser.add_argument(
        "--feature-store",
        default="outputs/misc_full_sae_eval/feature_store/utterance_features.pt",
        help="Utterance-level SAE feature matrix (.pt/.npy/.npz).",
    )
    parser.add_argument(
        "--records",
        default="outputs/misc_full_sae_eval/records.jsonl",
        help="JSONL records aligned to feature rows. Existing labels are used when present.",
    )
    parser.add_argument(
        "--label-matrix",
        default="outputs/misc_full_sae_eval/label_matrix.csv",
        help="Optional label matrix used only when records do not already contain labels.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered",
        help="Output directory for the filtered mapping artifacts.",
    )
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--precision-k-values", nargs="+", type=int, default=[10, 50])
    parser.add_argument("--min-positive", type=int, default=10)
    parser.add_argument("--min-negative", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--top-k-per-label", type=int, default=50)
    parser.add_argument("--top-example-latents", type=int, default=5)
    parser.add_argument("--top-examples-per-latent", type=int, default=10)

    parser.add_argument("--disable-feature-filter", action="store_true")
    parser.add_argument("--activation-threshold", type=float, default=1e-8)
    parser.add_argument("--min-activation-rate", type=float, default=0.001)
    parser.add_argument("--min-active-count", type=int, default=10)
    parser.add_argument("--max-activation-rate", type=float, default=0.995)
    parser.add_argument("--min-std", type=float, default=1e-8)
    parser.add_argument("--min-outlier-active-count", type=int, default=20)
    parser.add_argument("--outlier-iqr-multiplier", type=float, default=10.0)
    parser.add_argument("--outlier-z-threshold", type=float, default=8.0)
    parser.add_argument("--max-outlier-fraction", type=float, default=0.50)
    parser.add_argument("--max-top1-activation-share", type=float, default=0.80)
    parser.add_argument("--max-top5-activation-share", type=float, default=0.95)
    parser.add_argument("--filter-chunk-size", type=int, default=512)
    return parser.parse_args()


def _build_filter_config(args: argparse.Namespace) -> FeatureFilterConfig:
    return FeatureFilterConfig(
        enabled=not args.disable_feature_filter,
        activation_threshold=args.activation_threshold,
        min_activation_rate=args.min_activation_rate,
        min_active_count=args.min_active_count,
        max_activation_rate=args.max_activation_rate,
        min_std=args.min_std,
        min_outlier_active_count=args.min_outlier_active_count,
        outlier_iqr_multiplier=args.outlier_iqr_multiplier,
        outlier_z_threshold=args.outlier_z_threshold,
        max_outlier_fraction=args.max_outlier_fraction,
        max_top1_activation_share=args.max_top1_activation_share,
        max_top5_activation_share=args.max_top5_activation_share,
        chunk_size=args.filter_chunk_size,
    )


def main() -> int:
    args = parse_args()
    labels = tuple(label.upper() for label in args.labels)

    print(f"[load] records: {args.records}")
    records = _load_jsonl(args.records)
    records = _attach_labels_from_matrix_if_needed(records, args.label_matrix, labels)
    print(f"[load] records shape: {len(records)}")

    print(f"[load] SAE features: {args.feature_store}")
    features = load_feature_matrix(args.feature_store)
    print(f"[load] feature shape: {features.shape}")

    summary = run_misc_label_mapping(
        records=records,
        features=features,
        output_dir=args.output_dir,
        labels=list(labels),
        fdr_alpha=args.fdr_alpha,
        precision_k_values=args.precision_k_values,
        min_positive=args.min_positive,
        min_negative=args.min_negative,
        chunk_size=args.chunk_size,
        top_k_per_label=args.top_k_per_label,
        top_example_latents=args.top_example_latents,
        top_examples_per_latent=args.top_examples_per_latent,
        feature_filter_config=_build_filter_config(args),
    )

    feature_filter = summary.get("feature_filter", {})
    row_checks = summary.get("row_count_checks", {})
    print(
        "[done] kept "
        f"{feature_filter.get('n_kept_latents')} / {feature_filter.get('n_original_latents')} "
        "SAE latents"
    )
    print(
        "[done] metric rows "
        f"{row_checks.get('metric_rows_actual')} / {row_checks.get('metric_rows_expected')} "
        f"(match={row_checks.get('metric_rows_match')})"
    )
    print(f"[done] outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
