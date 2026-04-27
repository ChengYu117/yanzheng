"""Build a MISC Latent x Label association matrix.

The script can either reuse precomputed utterance-level SAE features or extract
them with the current Llama + OpenMOSS SAE pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.activations import extract_and_process_streaming
from nlp_re_base.config import resolve_output_dir, resolve_repo_path
from nlp_re_base.misc_label_mapping import (
    load_feature_matrix,
    load_misc_annotation_records,
    run_misc_label_mapping,
    write_json,
)
from nlp_re_base.model import load_local_model_and_tokenizer
from nlp_re_base.sae import load_sae_from_hub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MISC multi-label Latent x Label mapping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default="data/mi_quality_counseling_misc",
        help="MISC dataset root or misc_annotations directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Falls back to OUTPUT_ROOT/misc_label_mapping.",
    )
    parser.add_argument(
        "--features-path",
        default=None,
        help="Optional precomputed utterance feature matrix (.pt/.npy/.npz).",
    )
    parser.add_argument(
        "--save-features",
        default=None,
        help="Where to save extracted features. Defaults to output_dir/utterance_features.pt.",
    )
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--sae-config", default="config/sae_config.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument(
        "--aggregation",
        choices=["max", "mean", "sum", "binarized_sum"],
        default=None,
        help="Utterance-level pooling. Defaults to sae_config aggregation.",
    )
    parser.add_argument(
        "--binarized-threshold",
        type=float,
        default=0.0,
        help="Threshold used when aggregation=binarized_sum.",
    )
    parser.add_argument(
        "--checkpoint-topk-semantics",
        choices=["disabled", "hard"],
        default="hard",
        help="Whether to enforce hard top-k after JumpReLU for the public checkpoint.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Optional MISC annotation confidence filter.",
    )
    parser.add_argument(
        "--limit-records",
        type=int,
        default=None,
        help="Optional debug limit before feature extraction/analysis.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional label subset. Defaults to all observed hierarchical labels.",
    )
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--min-positive", type=int, default=10)
    parser.add_argument("--min-negative", type=int, default=10)
    parser.add_argument("--precision-k", nargs="+", type=int, default=[10, 50])
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--top-k-per-label", type=int, default=50)
    parser.add_argument("--top-example-latents", type=int, default=5)
    parser.add_argument("--top-examples-per-latent", type=int, default=10)
    return parser.parse_args()


def _load_sae_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_repo_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _extract_features(
    *,
    texts: list[str],
    args: argparse.Namespace,
    sae_config: dict[str, Any],
    output_dir: Path,
) -> torch.Tensor:
    device = _resolve_device(args.device)
    print(f"Device: {device}")
    print("Loading base model...")
    model, tokenizer, _ = load_local_model_and_tokenizer(
        args.model_config,
        model_dir=args.model_dir,
        device=device,
    )

    print("Loading SAE...")
    sae = load_sae_from_hub(
        repo_id=sae_config["sae_repo_id"],
        subfolder=sae_config["sae_subfolder"],
        device=device,
        dtype=torch.bfloat16,
        checkpoint_topk_semantics=args.checkpoint_topk_semantics,
    )

    aggregation = args.aggregation or sae_config.get("aggregation", "max")
    print(
        "Extracting MISC utterance features "
        f"(n={len(texts)}, batch_size={args.batch_size}, max_seq_len={args.max_seq_len}, "
        f"aggregation={aggregation})..."
    )
    result = extract_and_process_streaming(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        texts=texts,
        hook_point=sae_config["hook_point"],
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        aggregation=aggregation,
        binarized_threshold=args.binarized_threshold,
        device=device,
        collect_structural_samples=0,
    )
    features = result["utterance_features"].detach().cpu().float()

    feature_path = Path(args.save_features) if args.save_features else output_dir / "utterance_features.pt"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "utterance_features": features,
            "aggregation": aggregation,
            "hook_point": sae_config["hook_point"],
            "max_seq_len": args.max_seq_len,
            "batch_size": args.batch_size,
        },
        feature_path,
    )
    print(f"Saved extracted features to {feature_path}")
    return features


def main() -> None:
    args = parse_args()
    start = time.time()
    output_dir = resolve_output_dir(args.output_dir, default_subdir="misc_label_mapping")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = resolve_repo_path(args.data_dir)
    print("=" * 60)
    print("MISC Latent x Label Mapping")
    print("=" * 60)
    print(f"Data dir:   {data_dir}")
    print(f"Output dir: {output_dir}")

    records = load_misc_annotation_records(
        data_dir,
        confidence_threshold=args.confidence_threshold,
        limit=args.limit_records,
    )
    if not records:
        raise RuntimeError("No MISC annotation records loaded.")

    texts = [str(record["unit_text"]) for record in records]
    print(f"Loaded records: {len(records)}")

    sae_config = _load_sae_config(args.sae_config)
    if args.features_path:
        feature_path = resolve_repo_path(args.features_path)
        print(f"Loading precomputed features from {feature_path}")
        features = load_feature_matrix(feature_path)
    else:
        features = _extract_features(
            texts=texts,
            args=args,
            sae_config=sae_config,
            output_dir=output_dir,
        )

    run_config = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "features_path": args.features_path,
        "confidence_threshold": args.confidence_threshold,
        "limit_records": args.limit_records,
        "labels": args.labels,
        "fdr_alpha": args.fdr_alpha,
        "min_positive": args.min_positive,
        "min_negative": args.min_negative,
        "precision_k": args.precision_k,
        "chunk_size": args.chunk_size,
        "top_k_per_label": args.top_k_per_label,
        "top_example_latents": args.top_example_latents,
        "top_examples_per_latent": args.top_examples_per_latent,
        "sae_config": args.sae_config,
        "aggregation": args.aggregation or sae_config.get("aggregation", "max"),
        "checkpoint_topk_semantics": args.checkpoint_topk_semantics,
    }
    write_json(output_dir / "run_config.json", run_config)

    summary = run_misc_label_mapping(
        records=records,
        features=features,
        output_dir=output_dir,
        labels=args.labels,
        fdr_alpha=args.fdr_alpha,
        precision_k_values=args.precision_k,
        min_positive=args.min_positive,
        min_negative=args.min_negative,
        chunk_size=args.chunk_size,
        top_k_per_label=args.top_k_per_label,
        top_example_latents=args.top_example_latents,
        top_examples_per_latent=args.top_examples_per_latent,
    )

    elapsed = time.time() - start
    print("\nCompleted MISC label mapping.")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Latent x Label matrix: {summary['files']['latent_label_matrix']}")
    print(f"Report: {summary['files']['behavior_asymmetry']}")


if __name__ == "__main__":
    main()
