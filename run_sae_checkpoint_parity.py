"""Compare the local SAE forward pass against a reference backend."""

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
    build_parity_report,
    collect_residual_batches,
    compute_text_statistics,
    load_hf_texts_with_metadata,
    load_reference_sae_backend,
    save_json,
)
from nlp_re_base.model import load_local_model_and_tokenizer
from nlp_re_base.sae import load_sae_from_hub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a parity check between the local SAE and a reference backend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-id",
        default="cerebras/SlimPajama-627B",
        help="HuggingFace dataset id used to source residual activations.",
    )
    parser.add_argument(
        "--dataset-config-name",
        default=None,
        help="Optional HuggingFace dataset config name.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split for parity inputs.",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Text field name in the HuggingFace dataset.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=2,
        help="Maximum number of documents used for parity inputs.",
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
        "--max-seq-len",
        type=int,
        default=256,
        help="Maximum token sequence length.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for parity input capture.",
    )
    parser.add_argument(
        "--topk-compare",
        type=int,
        default=10,
        help="How many top latents to compare in the parity report.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for parity outputs.",
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
        "--official-loader",
        default=None,
        help=(
            "Optional import path for a reference SAE factory, in the form "
            "'module.submodule:callable'. When omitted, the script falls back "
            "to an independently coded checkpoint-formula adapter."
        ),
    )
    parser.add_argument(
        "--official-repo-dir",
        default=None,
        help="Optional local checkout of the official SAE repo for custom loaders.",
    )
    parser.add_argument(
        "--checkpoint-topk-semantics",
        choices=("disabled", "hard"),
        default="disabled",
        help=(
            "Whether to enforce an extra hard top-k after JumpReLU when evaluating "
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
    output_dir = resolve_output_dir(args.output_dir, default_subdir="sae_parity")
    output_dir.mkdir(parents=True, exist_ok=True)

    sae_config = _load_sae_config(args.sae_config)
    hook_point = sae_config["hook_point"]

    print("\n======================================")
    print("  SAE Checkpoint Parity")
    print("======================================")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  Device:       {device}")
    print(f"  Dataset:      {args.dataset_id} [{args.split}]")
    print(f"  Max docs:     {args.max_docs}")

    print("\n[1/4] Loading parity texts...")
    text_bundle = load_hf_texts_with_metadata(
        dataset_id=args.dataset_id,
        config_name=args.dataset_config_name,
        split=args.split,
        text_field=args.text_field,
        max_docs=args.max_docs,
        streaming=args.streaming,
    )
    texts = text_bundle["texts"]
    dataset_access = text_bundle["dataset_access"]
    if not texts:
        raise SystemExit(
            "No parity texts were loaded. Check --dataset-id/--split/--text-field."
        )
    save_json(
        compute_text_statistics(texts),
        output_dir / "sample_stats.json",
    )
    save_json(
        build_llamascope_evidence_table(dataset_access=dataset_access),
        output_dir / "evidence_table.json",
    )
    print(f"  Loaded {len(texts)} text samples for parity.")

    print("\n[2/4] Loading model and local SAE...")
    model, tokenizer, _ = load_local_model_and_tokenizer(
        args.model_config,
        model_dir=args.model_dir,
        device=device,
    )
    local_sae = load_sae_from_hub(
        repo_id=sae_config["sae_repo_id"],
        subfolder=sae_config["sae_subfolder"],
        device=device,
        dtype=torch.bfloat16,
        checkpoint_topk_semantics=args.checkpoint_topk_semantics,
    )

    print("\n[3/4] Capturing residual activations...")
    residual_batches = collect_residual_batches(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        hook_point=hook_point,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=device,
    )

    print("\n[4/4] Running local vs reference SAE forward passes...")
    reference_backend, backend_name = load_reference_sae_backend(
        repo_id=sae_config["sae_repo_id"],
        subfolder=sae_config["sae_subfolder"],
        device=device,
        dtype=torch.bfloat16,
        official_loader=args.official_loader,
        official_repo_dir=args.official_repo_dir,
        checkpoint_topk_semantics=args.checkpoint_topk_semantics,
    )

    local_details_list = []
    reference_details_list = []
    attention_masks = []
    sae_dtype = getattr(local_sae, "sae_dtype", next(local_sae.parameters()).dtype)

    for batch in residual_batches:
        residual = batch["residual"].to(device=device, dtype=sae_dtype)
        with torch.inference_mode():
            local_details = local_sae.forward_with_details(residual)
            reference_details = reference_backend.forward_with_details(residual)
        local_details_list.append(local_details)
        reference_details_list.append(reference_details)
        attention_masks.append(batch["attention_mask"])

    report = build_parity_report(
        local_details_list=local_details_list,
        reference_details_list=reference_details_list,
        attention_masks=attention_masks,
        topk_compare=args.topk_compare,
    )
    report["reference_backend"] = backend_name
    report["official_backend_status"] = (
        "loaded"
        if backend_name == "official_loader"
        else "fallback_checkpoint_formula"
    )
    report["input_dataset"] = {
        **dataset_access,
        "hook_point": hook_point,
        "max_seq_len": args.max_seq_len,
        "checkpoint_topk_semantics": args.checkpoint_topk_semantics,
    }
    save_json(report, output_dir / "parity_report.json")

    print("\nCompleted parity check.")
    print(f"  Backend:       {backend_name}")
    print(f"  Parity passed: {report['parity_passed']}")
    print(f"  Report:        {output_dir / 'parity_report.json'}")


if __name__ == "__main__":
    main()
