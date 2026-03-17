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
from nlp_re_base.data import load_jsonl
from nlp_re_base.eval_functional import run_functional_evaluation
from nlp_re_base.eval_structural import (
    OnlineStructuralAccumulator,
    compute_ce_kl_with_intervention,
    run_structural_evaluation,
)
from nlp_re_base.model import load_local_model_and_tokenizer
from nlp_re_base.sae import load_sae_from_hub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAE-RE Evaluation Pipeline",
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
        default="data/mi_re",
        help="Directory containing re_dataset.jsonl and nonre_dataset.jsonl.",
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
    print("  SAE-RE Evaluation Pipeline")
    print("=" * 38 + "\n")


def _run_single_aggregation(
    *,
    aggregation: str,
    output_dir: Path,
    args: argparse.Namespace,
    sae_config: dict[str, Any],
    all_texts: list[str],
    all_labels: list[int],
    all_records: list[dict[str, Any]],
    n_re: int,
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
    # Create optional online accumulator for full-dataset structural metrics
    accumulator = OnlineStructuralAccumulator() if args.full_structural else None
    if accumulator:
        print("  [full-structural] Online accumulator enabled - metrics computed over ALL tokens.")

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
    )

    utterance_features = result["utterance_features"]
    utterance_activations = result["utterance_activations"]

    re_features = utterance_features[:n_re]
    nonre_features = utterance_features[n_re:]
    re_activations = utterance_activations[:n_re]
    nonre_activations = utterance_activations[n_re:]

    print(f"  Utterance features shape: {utterance_features.shape}")
    print(f"  RE features: {re_features.shape}, NonRE features: {nonre_features.shape}")

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
        latents=result["sample_latents"],
        attention_mask=result["sample_mask"],
        ce_kl_results=ce_kl_results,
        output_dir=output_dir,
    )

    # If the online accumulator was used, overwrite with full-data metrics
    if accumulator is not None:
        full_metrics = accumulator.result()
        print(f"  [full-structural] n_tokens={full_metrics.get('n_tokens')}, "
              f"EV={full_metrics.get('explained_variance', float('nan')):.4f}, "
              f"FVU={full_metrics.get('fvu', float('nan')):.4f}, "
              f"MSE={full_metrics.get('mse', float('nan')):.4f}")
        structural_metrics.update(full_metrics)
        # Re-save the merged structural metrics JSON
        import json
        struct_path = output_dir / "metrics_structural.json"
        with open(struct_path, "w", encoding="utf-8") as _f:
            json.dump(structural_metrics, _f, indent=2, ensure_ascii=False)
        print(f"  [full-structural] Updated {struct_path}")

    print("\n[7/7] Running functional evaluation...")
    sae_decoder_weight = sae.W_dec.detach().cpu()

    functional_metrics = None
    functional_error = None
    try:
        functional_metrics = run_functional_evaluation(
            re_features=re_features,
            nonre_features=nonre_features,
            all_texts=all_texts,
            all_labels=all_labels,
            all_records=all_records,
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
            output_dir=output_dir,
        )
    except Exception as exc:
        functional_error = f"{type(exc).__name__}: {exc}"
        if not allow_partial_functional:
            raise
        print(
            f"  Functional evaluation did not complete for aggregation={aggregation}: "
            f"{functional_error}"
        )

    return {
        "aggregation": aggregation,
        "output_dir": str(output_dir),
        "utterance_features_shape": list(utterance_features.shape),
        "utterance_activations_shape": list(utterance_activations.shape),
        "structural_metrics": structural_metrics,
        "ce_kl_results": ce_kl_results,
        "functional_metrics": functional_metrics,
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

    re_records = load_jsonl(data_dir / "re_dataset.jsonl")
    nonre_records = load_jsonl(data_dir / "nonre_dataset.jsonl")

    re_texts = [record["unit_text"] for record in re_records]
    nonre_texts = [record["unit_text"] for record in nonre_records]
    all_records = re_records + nonre_records
    all_texts = re_texts + nonre_texts
    all_labels = [1] * len(re_texts) + [0] * len(nonre_texts)

    n_re = len(re_texts)
    print(f"  RE samples:    {n_re}")
    print(f"  NonRE samples: {len(nonre_texts)}")
    print(f"  Total:         {len(all_texts)}")

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
    )

    print("\n[2/7] Loading SAE from HuggingFace Hub...")
    sae = load_sae_from_hub(
        repo_id=sae_config["sae_repo_id"],
        subfolder=sae_config["sae_subfolder"],
        device=device,
        dtype=torch.bfloat16,
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
            all_texts=all_texts,
            all_labels=all_labels,
            all_records=all_records,
            n_re=n_re,
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
                    "structural_metrics": record["structural_metrics"],
                    "functional_summary": _summarize_functional_metrics(
                        record["functional_metrics"]
                    ),
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
