"""SAE-RE evaluation pipeline end-to-end runner.

Uses a streaming architecture: activations are extracted, processed through
the SAE, and aggregated batch-by-batch. Only utterance-level features
([N, d_sae] / [N, d_model]) are kept in memory instead of the full
[N, T, d_sae] tensor.

Usage:
    python run_sae_evaluation.py [--output-dir DIR] [--batch-size N]
                                 [--max-seq-len N] [--device DEVICE]
                                 [--skip-ce-kl]
                                 [--ce-kl-batch-size N]
                                 [--ce-kl-max-texts N]
                                 [--aggregation max|mean]
                                 [--compare-mean]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.activations import extract_and_process_streaming
from nlp_re_base.config import resolve_output_dir
from nlp_re_base.data import DEFAULT_DATA_DIR, ExperimentDataset, load_experiment_dataset
from nlp_re_base.diagnostics import apply_full_structural_metrics
from nlp_re_base.eval_functional import run_functional_evaluation
from nlp_re_base.eval_structural import (
    OfficialMetricsAccumulator,
    OnlineStructuralAccumulator,
    compute_ce_kl_with_intervention,
    run_structural_evaluation,
)
from nlp_re_base.misc_label_mapping import (
    run_misc_label_mapping,
    write_json,
    write_jsonl,
    write_label_indicator_csv,
)
from nlp_re_base.model import load_local_model_and_tokenizer
from nlp_re_base.sae import load_sae_from_hub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAE-MISC Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for evaluation outputs. Falls back to OUTPUT_ROOT/sae_eval if set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for model inference (default lowered for memory safety).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (e.g. 'cuda', 'cpu'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--skip-ce-kl",
        action="store_true",
        help="Skip CE/KL computation even though it is part of the default fidelity evaluation.",
    )
    parser.add_argument(
        "--ce-kl-batch-size",
        type=int,
        default=None,
        help="Override batch size used by CE/KL intervention evaluation.",
    )
    parser.add_argument(
        "--ce-kl-max-texts",
        type=int,
        default=None,
        help="Optionally limit the number of texts used for CE/KL evaluation.",
    )
    parser.add_argument(
        "--sae-config",
        default="config/sae_config.json",
        help="Path to SAE configuration JSON.",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Path to model_config.json.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Override the local model directory. Takes precedence over MODEL_DIR and model_config.json.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR.relative_to(PROJECT_ROOT)),
        help=(
            "Dataset root. Defaults to the full MISC dataset; legacy MI-RE split "
            "and CACTUS remain supported through --data-format auto."
        ),
    )
    parser.add_argument(
        "--data-format",
        choices=["auto", "misc_full", "legacy_re_nonre", "cactus"],
        default="auto",
        help="Dataset format override.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["misc_multilabel", "binary_re"],
        default="misc_multilabel",
        help="Functional label analysis mode. binary_re skips the MISC Latent x Label matrix.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Optional confidence filter for MISC annotation records.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional debug limit applied after dataset loading.",
    )
    parser.add_argument(
        "--full-structural",
        action="store_true",
        help="Compute structural metrics (EV, MSE, Cosine, L0) on the full "
             "dataset via online Welford algorithm instead of just 5 sample batches.",
    )
    parser.add_argument(
        "--aggregation",
        choices=["max", "mean"],
        default=None,
        help="Override the utterance aggregation method from the SAE config.",
    )
    parser.add_argument(
        "--compare-mean",
        action="store_true",
        help="Run both max and mean aggregation and save a comparison summary.",
    )
    parser.add_argument(
        "--judge-bundle-only",
        action="store_true",
        help="Run univariate analysis and export judge_bundle only, skipping the slower late functional evaluation stages.",
    )
    parser.add_argument(
        "--save-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save records, label matrix, utterance SAE features, and utterance activations.",
    )
    parser.add_argument(
        "--save-token-topk",
        action="store_true",
        help="Reserved for token-level top-k latent activation export. Not enabled by default.",
    )
    parser.add_argument(
        "--checkpoint-topk-semantics",
        choices=["disabled", "hard"],
        default="hard",
        help=(
            "Whether to enforce an extra hard top-k after JumpReLU when running "
            "the public checkpoint. 'hard' matches the currently selected main experimental path."
        ),
    )
    return parser.parse_args()


def _unique_ordered(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _summarize_functional_metrics(
    functional_metrics: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if functional_metrics is None:
        return None

    summary = functional_metrics.get("univariate_summary", {})
    top10 = summary.get("top10_latents", [])
    return {
        "total_latents": summary.get("total_latents"),
        "significant_fdr": summary.get("significant_fdr"),
        "fdr_alpha": summary.get("fdr_alpha"),
        "top10_latent_indices": [entry["latent_idx"] for entry in top10],
        "probe_results": functional_metrics.get("probe_results"),
        "maxact_summary": functional_metrics.get("maxact_summary"),
    }


def _print_banner() -> None:
    print("\n" + "=" * 38)
    print("  SAE-MISC Evaluation Pipeline")
    print("=" * 38 + "\n")


def _label_sets_for_records(records: list[dict[str, Any]]) -> list[set[str]]:
    return [set(record.get("labels") or []) for record in records]


def _save_feature_store(
    *,
    output_dir: Path,
    dataset: ExperimentDataset,
    utterance_features: torch.Tensor,
    utterance_activations: torch.Tensor,
    aggregation: str,
    hook_point: str,
    max_seq_len: int,
    batch_size: int,
    save_token_topk: bool,
) -> dict[str, str]:
    feature_dir = output_dir / "feature_store"
    feature_dir.mkdir(parents=True, exist_ok=True)

    records_path = output_dir / "records.jsonl"
    label_matrix_path = output_dir / "label_matrix.csv"
    features_path = feature_dir / "utterance_features.pt"
    activations_path = feature_dir / "utterance_activations.pt"
    metadata_path = feature_dir / "feature_metadata.json"

    write_jsonl(records_path, dataset.records)
    write_label_indicator_csv(
        label_matrix_path,
        dataset.records,
        dataset.label_names,
        torch.tensor(dataset.label_matrix, dtype=torch.int64).numpy(),
    )
    torch.save(
        {
            "utterance_features": utterance_features.detach().cpu().float(),
            "aggregation": aggregation,
            "hook_point": hook_point,
            "data_format": dataset.data_format,
        },
        features_path,
    )
    torch.save(
        {
            "utterance_activations": utterance_activations.detach().cpu().float(),
            "aggregation": aggregation,
            "hook_point": hook_point,
            "data_format": dataset.data_format,
        },
        activations_path,
    )
    metadata = {
        "data_dir": str(dataset.data_dir),
        "data_format": dataset.data_format,
        "n_records": len(dataset.records),
        "feature_shape": list(utterance_features.shape),
        "activation_shape": list(utterance_activations.shape),
        "label_names": dataset.label_names,
        "aggregation": aggregation,
        "hook_point": hook_point,
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "save_token_topk": bool(save_token_topk),
        "token_topk_note": (
            "Full token-level latent export is intentionally disabled by default. "
            "This run saved utterance-level SAE features and model activations."
        ),
    }
    write_json(metadata_path, metadata)
    return {
        "records": str(records_path),
        "label_matrix": str(label_matrix_path),
        "utterance_features": str(features_path),
        "utterance_activations": str(activations_path),
        "feature_metadata": str(metadata_path),
    }


def _mirror_binary_functional_outputs(binary_dir: Path, output_dir: Path) -> None:
    """Keep legacy root-level outputs available for causal and judge scripts."""
    for filename in ("candidate_latents.csv", "metrics_functional.json"):
        src = binary_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)
    for dirname in ("judge_bundle", "latent_cards"):
        src_dir = binary_dir / dirname
        dst_dir = output_dir / dirname
        if src_dir.exists():
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)


def _binary_reordered_metadata(
    texts: list[str],
    labels: list[int],
    records: list[dict[str, Any]],
) -> tuple[list[str], list[int], list[dict[str, Any]]]:
    """Match legacy functional feature order: all RE rows, then all NonRE rows."""
    re_indices = [idx for idx, label in enumerate(labels) if int(label) == 1]
    nonre_indices = [idx for idx, label in enumerate(labels) if int(label) == 0]
    order = re_indices + nonre_indices
    return (
        [texts[idx] for idx in order],
        [labels[idx] for idx in order],
        [records[idx] for idx in order],
    )


def _run_single_aggregation(
    *,
    aggregation: str,
    output_dir: Path,
    args: argparse.Namespace,
    sae_config: dict[str, Any],
    dataset: ExperimentDataset,
    model: Any,
    tokenizer: Any,
    model_cfg: dict[str, Any] | None,
    sae: Any,
    hook_point: str,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
    allow_partial_functional: bool = False,
    cached_ce_kl_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print(f"  Aggregation Mode: {aggregation}")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[3-5/7] Extracting activations, running SAE, aggregating (streaming)...")
    print(
        f"  batch_size={batch_size}, max_seq_len={max_seq_len}, aggregation={aggregation}"
    )
    all_texts = dataset.texts
    all_labels = dataset.binary_labels
    all_records = dataset.records
    # Create optional online accumulator for full-dataset structural metrics
    accumulator = None
    if args.full_structural:
        group_accumulators = {
            label: {
                "raw": OnlineStructuralAccumulator(),
                "normalized": OnlineStructuralAccumulator(),
            }
            for label in dataset.label_names
        }
        accumulator = {
            "raw": OnlineStructuralAccumulator(),
            "normalized": OnlineStructuralAccumulator(),
            "official": OfficialMetricsAccumulator(),
            "groups": group_accumulators,
        }
        print(
            "  [full-structural] Online accumulator enabled - metrics computed over ALL tokens, "
            "including official OpenMOSS legacy EV."
        )

    result = extract_and_process_streaming(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        texts=all_texts,
        hook_point=hook_point,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        aggregation=aggregation,
        device=device,
        collect_structural_samples=5,
        structural_accumulator=accumulator,
        structural_group_labels=_label_sets_for_records(dataset.records),
    )

    utterance_features = result["utterance_features"]
    utterance_activations = result["utterance_activations"]

    binary_texts, binary_labels, binary_records = _binary_reordered_metadata(
        all_texts,
        all_labels,
        all_records,
    )

    label_mask = torch.tensor(all_labels, dtype=torch.bool)
    re_features = utterance_features[label_mask]
    nonre_features = utterance_features[~label_mask]
    re_activations = utterance_activations[label_mask]
    nonre_activations = utterance_activations[~label_mask]

    print(f"  Utterance features shape: {utterance_features.shape}")
    print(f"  RE features: {re_features.shape}, NonRE features: {nonre_features.shape}")
    if args.save_features:
        feature_paths = _save_feature_store(
            output_dir=output_dir,
            dataset=dataset,
            utterance_features=utterance_features,
            utterance_activations=utterance_activations,
            aggregation=aggregation,
            hook_point=hook_point,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            save_token_topk=args.save_token_topk,
        )
        print(f"  Saved feature store to {output_dir / 'feature_store'}")
    else:
        feature_paths = {}

    print("\n[6/7] Running structural evaluation...")
    ce_kl_results = cached_ce_kl_results
    ce_kl_batch_size = args.ce_kl_batch_size or sae_config.get(
        "ce_kl_batch_size",
        max(1, batch_size // 2),
    )
    ce_kl_max_texts = (
        args.ce_kl_max_texts
        if args.ce_kl_max_texts is not None
        else sae_config.get("ce_kl_max_texts")
    )
    if ce_kl_max_texts is not None and ce_kl_max_texts <= 0:
        ce_kl_max_texts = None

    if not args.skip_ce_kl and ce_kl_results is None:
        scope = f"{ce_kl_max_texts} texts" if ce_kl_max_texts is not None else "all texts"
        print(
            "  Computing CE/KL "
            f"(batch_size={ce_kl_batch_size}, scope={scope}; this may take a while)..."
        )
        ce_kl_results = compute_ce_kl_with_intervention(
            model=model,
            tokenizer=tokenizer,
            texts=all_texts,
            sae=sae,
            hook_point=hook_point,
            max_seq_len=max_seq_len,
            batch_size=ce_kl_batch_size,
            max_texts=ce_kl_max_texts,
        )
    elif ce_kl_results is not None:
        print("  Reusing cached CE/KL results for this aggregation.")

    structural_metrics = run_structural_evaluation(
        activations=result["sample_activations"],
        reconstructed=result["sample_reconstructed"],
        normalized_activations=result["sample_activations_normalized"],
        normalized_reconstructed=result["sample_reconstructed_normalized"],
        latents=result["sample_latents"],
        attention_mask=result["sample_mask"],
        ce_kl_results=ce_kl_results,
        output_dir=output_dir,
    )

    # If the online accumulator was used, overwrite with full-data metrics
    if accumulator is not None:
        raw_full_metrics = accumulator["raw"].result()
        norm_full_metrics = accumulator["normalized"].result()
        official_runtime_metrics = accumulator["official"].result()
        print(
            "  [full-structural][raw] "
            f"n_tokens={raw_full_metrics.get('n_tokens')}, "
            f"EV={raw_full_metrics.get('explained_variance', float('nan')):.4f}, "
            f"FVU={raw_full_metrics.get('fvu', float('nan')):.4f}, "
            f"MSE={raw_full_metrics.get('mse', float('nan')):.4f}"
        )
        print(
            "  [full-structural][normalized] "
            f"n_tokens={norm_full_metrics.get('n_tokens')}, "
            f"EV={norm_full_metrics.get('explained_variance', float('nan')):.4f}, "
            f"FVU={norm_full_metrics.get('fvu', float('nan')):.4f}, "
            f"MSE={norm_full_metrics.get('mse', float('nan')):.4f}"
        )
        if official_runtime_metrics:
            print(
                "  [full-structural][official] "
                f"EV={official_runtime_metrics.get('metrics/explained_variance', float('nan')):.4f}, "
                f"legacy_EV={official_runtime_metrics.get('metrics/explained_variance_legacy', float('nan')):.4f}, "
                f"L0={official_runtime_metrics.get('metrics/l0', float('nan')):.4f}"
            )
        structural_metrics = apply_full_structural_metrics(
            structural_metrics,
            raw_full_metrics=raw_full_metrics,
            norm_full_metrics=norm_full_metrics,
            official_runtime_metrics=official_runtime_metrics,
            output_dir=output_dir,
        )
        structural_metrics["by_label"] = {
            label: {
                "raw": accs["raw"].result(),
                "normalized": accs["normalized"].result(),
            }
            for label, accs in accumulator.get("groups", {}).items()
        }
        with (output_dir / "metrics_structural.json").open("w", encoding="utf-8") as f:
            json.dump(structural_metrics, f, indent=2, ensure_ascii=False, default=str)
        print(f"  [full-structural] Updated {output_dir / 'metrics_structural.json'}")

    print("\n[7/7] Running functional evaluation...")
    sae_decoder_weight = sae.W_dec.detach().cpu()

    functional_metrics = None
    functional_error = None
    binary_output_dir = output_dir / "functional" / "re_binary"
    try:
        functional_metrics = run_functional_evaluation(
            re_features=re_features,
            nonre_features=nonre_features,
            all_texts=binary_texts,
            all_labels=binary_labels,
            all_records=binary_records,
            re_activations=re_activations,
            nonre_activations=nonre_activations,
            sae_decoder_weight=sae_decoder_weight,
            fdr_alpha=sae_config.get("fdr_alpha", 0.05),
            k_values=sae_config.get("probe_k_values", [1, 5, 20]),
            top_k_candidates=sae_config.get("top_k_candidates", 50),
            aggregation=aggregation,
            hook_point=hook_point,
            model_name=(model_cfg or {}).get("base_model_path"),
            sae_repo_id=sae_config.get("sae_repo_id"),
            sae_subfolder=sae_config.get("sae_subfolder"),
            judge_bundle_only=args.judge_bundle_only,
            output_dir=binary_output_dir,
        )
        _mirror_binary_functional_outputs(binary_output_dir, output_dir)
    except Exception as exc:
        functional_error = f"{type(exc).__name__}: {exc}"
        if not allow_partial_functional:
            raise
        print(
            f"  Functional evaluation did not complete for aggregation={aggregation}: "
            f"{functional_error}"
        )

    misc_mapping_summary = None
    if (
        args.label_mode == "misc_multilabel"
        and dataset.data_format == "misc_full"
        and not args.judge_bundle_only
    ):
        print("\n[7/7] Running MISC multi-label latent mapping...")
        misc_mapping_summary = run_misc_label_mapping(
            records=all_records,
            features=utterance_features,
            output_dir=output_dir / "functional" / "misc_label_mapping",
            labels=dataset.label_names,
            fdr_alpha=sae_config.get("fdr_alpha", 0.05),
            precision_k_values=sae_config.get("misc_precision_k_values", [10, 50]),
            min_positive=sae_config.get("misc_min_positive", 10),
            min_negative=sae_config.get("misc_min_negative", 10),
            chunk_size=sae_config.get("misc_chunk_size", 512),
            top_k_per_label=sae_config.get("misc_top_k_per_label", 50),
            top_example_latents=sae_config.get("misc_top_example_latents", 5),
            top_examples_per_latent=sae_config.get("misc_top_examples_per_latent", 10),
        )
        print(
            "  Saved MISC label mapping to "
            f"{output_dir / 'functional' / 'misc_label_mapping'}"
        )
    elif args.judge_bundle_only and args.label_mode == "misc_multilabel":
        print("  Skipping MISC multi-label mapping because --judge-bundle-only is set.")

    return {
        "aggregation": aggregation,
        "output_dir": str(output_dir),
        "utterance_features_shape": list(utterance_features.shape),
        "utterance_activations_shape": list(utterance_activations.shape),
        "feature_store": feature_paths,
        "structural_metrics": structural_metrics,
        "ce_kl_results": ce_kl_results,
        "functional_metrics": functional_metrics,
        "misc_mapping_summary": misc_mapping_summary,
        "functional_error": functional_error,
    }


def main() -> None:
    args = parse_args()
    start_time = time.time()

    base_output_dir = resolve_output_dir(args.output_dir, default_subdir="sae_eval")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    sae_config_path = Path(args.sae_config)
    if not sae_config_path.is_absolute():
        sae_config_path = PROJECT_ROOT / sae_config_path
    with open(sae_config_path, "r", encoding="utf-8") as f:
        sae_config = json.load(f)

    hook_point = sae_config["hook_point"]
    max_seq_len = args.max_seq_len
    batch_size = args.batch_size
    base_aggregation = args.aggregation or sae_config.get("aggregation", "max")
    aggregations = (
        _unique_ordered([base_aggregation, "max", "mean"])
        if args.compare_mean
        else [base_aggregation]
    )

    _print_banner()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    if args.save_token_topk:
        print(
            "  --save-token-topk requested, but token-level export is not part of "
            "the default memory-safe path yet. This run will save utterance-level "
            "feature and activation stores."
        )

    dataset = load_experiment_dataset(
        data_dir,
        data_format=args.data_format,
        confidence_threshold=args.confidence_threshold,
        limit=args.max_samples,
    )
    write_json(base_output_dir / "dataset_summary.json", dataset.summary)

    print(f"  Data format:   {dataset.data_format}")
    print(f"  Data dir:      {dataset.data_dir}")
    print(f"  RE samples:    {len(dataset.re_records)}")
    print(f"  NonRE samples: {len(dataset.nonre_records)}")
    print(f"  Total:         {len(dataset.texts)}")
    print(f"  Labels:        {', '.join(dataset.label_names)}")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  Device:        {device}")
    print(f"  Aggregations:  {', '.join(aggregations)}")
    if args.model_dir:
        print(f"  Model dir:     {args.model_dir}")

    print("\n[1/7] Loading base model...")
    model, tokenizer, _model_cfg = load_local_model_and_tokenizer(
        args.model_config,
        model_dir=args.model_dir,
        device=device,
    )

    print("\n[2/7] Loading SAE from HuggingFace Hub...")
    sae = load_sae_from_hub(
        repo_id=sae_config["sae_repo_id"],
        subfolder=sae_config["sae_subfolder"],
        device=device,
        dtype=torch.bfloat16,
        checkpoint_topk_semantics=args.checkpoint_topk_semantics,
    )

    comparison_records = []
    cached_ce_kl_results: dict[str, Any] | None = None
    for aggregation in aggregations:
        run_output_dir = (
            base_output_dir / aggregation
            if len(aggregations) > 1
            else base_output_dir
        )
        record = _run_single_aggregation(
            aggregation=aggregation,
            output_dir=run_output_dir,
            args=args,
            sae_config=sae_config,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            model_cfg=_model_cfg,
            sae=sae,
            hook_point=hook_point,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            device=device,
            allow_partial_functional=len(aggregations) > 1,
            cached_ce_kl_results=cached_ce_kl_results,
        )
        comparison_records.append(record)
        if cached_ce_kl_results is None and record["ce_kl_results"] is not None:
            cached_ce_kl_results = record["ce_kl_results"]

    if len(aggregations) > 1:
        comparison_payload = {
            "aggregations": aggregations,
            "base_aggregation": base_aggregation,
            "note": (
                "Structural metrics and CE/KL should match across aggregation modes "
                "because aggregation only changes utterance-level features used in "
                "functional evaluation."
            ),
            "results": [
                {
                    "aggregation": record["aggregation"],
                    "output_dir": record["output_dir"],
                    "utterance_features_shape": record["utterance_features_shape"],
                    "utterance_activations_shape": record["utterance_activations_shape"],
                    "feature_store": record["feature_store"],
                    "structural_metrics": record["structural_metrics"],
                    "functional_summary": _summarize_functional_metrics(
                        record["functional_metrics"]
                    ),
                    "misc_mapping_summary": record["misc_mapping_summary"],
                    "functional_error": record["functional_error"],
                }
                for record in comparison_records
            ],
        }
        comparison_path = base_output_dir / "aggregation_comparison.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison_payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved aggregation comparison to {comparison_path}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print("  EVALUATION COMPLETE")
    print("=" * 50)
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"  Output dir:   {base_output_dir}")
    print("  Files generated:")
    for path in sorted(base_output_dir.rglob("*")):
        if path.is_file():
            size_kb = path.stat().st_size / 1024
            print(f"    {path.relative_to(base_output_dir)}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
