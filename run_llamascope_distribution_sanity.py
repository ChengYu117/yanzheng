"""Run a structural sanity check on a paper-like text distribution and MI-RE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.config import resolve_output_dir
from nlp_re_base.diagnostics import (
    build_llamascope_evidence_table,
    build_comparison_payload,
    load_hf_texts_with_metadata,
    load_mi_re_texts,
    run_structural_diagnostic,
    save_json,
)
from nlp_re_base.model import load_local_model_and_tokenizer
from nlp_re_base.sae import load_sae_from_hub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a paper-distribution SAE structural sanity check.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-id",
        default="cerebras/SlimPajama-627B",
        help="HuggingFace dataset id used as the paper-distribution proxy.",
    )
    parser.add_argument(
        "--dataset-config-name",
        default=None,
        help="Optional HuggingFace dataset config name.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split for the paper-distribution proxy.",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Text field name in the HuggingFace dataset.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=400,
        help="Maximum number of paper-distribution documents to evaluate.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Load the HuggingFace dataset in streaming mode.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Disable streaming mode for the HuggingFace dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the sanity-check outputs.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1024,
        help="Maximum token sequence length.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for model inference.",
    )
    parser.add_argument(
        "--ce-kl-batch-size",
        type=int,
        default=None,
        help="Optional CE/KL batch size override.",
    )
    parser.add_argument(
        "--ce-kl-max-texts",
        type=int,
        default=None,
        help="Optional CE/KL scope limit for the paper-distribution proxy.",
    )
    parser.add_argument(
        "--mi-re-data-dir",
        default="data/mi_re",
        help="Path to the legacy MI-RE split used for the comparison baseline.",
    )
    parser.add_argument(
        "--mi-re-ce-kl-max-texts",
        type=int,
        default=None,
        help="Optional CE/KL scope limit for the MI-RE baseline.",
    )
    parser.add_argument(
        "--sae-config",
        default="config/sae_config.json",
        help="Path to sae_config.json.",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Path to model_config.json.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Override the local model directory.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (e.g. 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--checkpoint-topk-semantics",
        choices=("disabled", "hard"),
        default="disabled",
        help=(
            "Whether to enforce an extra hard top-k after JumpReLU when running "
            "the public checkpoint. 'disabled' follows the official runtime path."
        ),
    )
    return parser.parse_args()


def _load_sae_config(path: str | Path) -> dict:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(
        args.output_dir,
        default_subdir="llamascope_sanity",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    sae_config = _load_sae_config(args.sae_config)
    hook_point = sae_config["hook_point"]

    print("\n==============================================")
    print("  Llama Scope Distribution Sanity Check")
    print("==============================================")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  Device:       {device}")
    print(f"  Dataset:      {args.dataset_id} [{args.split}]")
    print(f"  Max docs:     {args.max_docs}")
    print(f"  MI-RE dir:    {args.mi_re_data_dir}")

    print("\n[1/5] Loading paper-distribution texts...")
    paper_bundle = load_hf_texts_with_metadata(
        dataset_id=args.dataset_id,
        config_name=args.dataset_config_name,
        split=args.split,
        text_field=args.text_field,
        max_docs=args.max_docs,
        streaming=args.streaming,
    )
    paper_texts = paper_bundle["texts"]
    dataset_access = paper_bundle["dataset_access"]
    if not paper_texts:
        raise SystemExit(
            "No paper-distribution texts were loaded. "
            "Check --dataset-id/--split/--text-field."
        )
    print(f"  Loaded {len(paper_texts)} documents from the paper-distribution proxy.")
    if dataset_access["resolved_dataset_id"] != dataset_access["requested_dataset_id"]:
        print(
            "  Dataset fallback: "
            f"{dataset_access['requested_dataset_id']} -> {dataset_access['resolved_dataset_id']}"
        )
    save_json(
        build_llamascope_evidence_table(dataset_access=dataset_access),
        output_dir / "evidence_table.json",
    )

    print("\n[2/5] Loading MI-RE baseline texts...")
    mire_texts, mire_metadata = load_mi_re_texts(args.mi_re_data_dir)
    print(
        "  Loaded MI-RE baseline: "
        f"RE={mire_metadata['re_count']}, NonRE={mire_metadata['nonre_count']}"
    )

    print("\n[3/5] Loading model and SAE...")
    model, tokenizer, _ = load_local_model_and_tokenizer(
        args.model_config,
        model_dir=args.model_dir,
        device=device,
    )
    sae = load_sae_from_hub(
        repo_id=sae_config["sae_repo_id"],
        subfolder=sae_config["sae_subfolder"],
        device=device,
        dtype=torch.bfloat16,
        checkpoint_topk_semantics=args.checkpoint_topk_semantics,
    )

    print("\n[4/5] Running paper-distribution structural diagnostic...")
    paper_result = run_structural_diagnostic(
        texts=paper_texts,
        output_dir=output_dir,
        dataset_label="paper_distribution",
        tokenizer=tokenizer,
        model=model,
        sae=sae,
        hook_point=hook_point,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=device,
        ce_kl_batch_size=args.ce_kl_batch_size,
        ce_kl_max_texts=args.ce_kl_max_texts,
    )
    paper_result["dataset_proxy"] = {
        **dataset_access,
    }
    save_json(paper_result, output_dir / "paper_distribution_result.json")

    print("\n[5/5] Running MI-RE structural baseline and saving comparison...")
    mire_output_dir = output_dir / "mi_re_baseline"
    mire_ce_kl_max_texts = (
        args.mi_re_ce_kl_max_texts
        if args.mi_re_ce_kl_max_texts is not None
        else args.max_docs
    )
    mire_result = run_structural_diagnostic(
        texts=mire_texts,
        output_dir=mire_output_dir,
        dataset_label="mi_re_baseline",
        tokenizer=tokenizer,
        model=model,
        sae=sae,
        hook_point=hook_point,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=device,
        ce_kl_batch_size=args.ce_kl_batch_size,
        ce_kl_max_texts=mire_ce_kl_max_texts,
    )
    mire_result["dataset_proxy"] = mire_metadata
    save_json(mire_result, mire_output_dir / "mi_re_result.json")

    comparison_payload = build_comparison_payload(
        paper_result=paper_result,
        mire_result=mire_result,
    )
    save_json(comparison_payload, output_dir / "comparison_vs_mire.json")

    print("\nCompleted structural sanity check.")
    print(f"  Paper metrics: {output_dir / 'metrics_structural.json'}")
    print(f"  MI-RE metrics: {mire_output_dir / 'metrics_structural.json'}")
    print(f"  Comparison:    {output_dir / 'comparison_vs_mire.json'}")
    print(f"  Diagnosis:     {comparison_payload['diagnosis_hint']}")


if __name__ == "__main__":
    main()
