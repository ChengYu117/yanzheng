"""Audit whether the current SAE forward matches the published Llama Scope variant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.config import resolve_output_dir
from nlp_re_base.diagnostics import (
    build_llamascope_evidence_table,
    build_parity_report,
    build_variant_evidence_matrix,
    collect_residual_batches,
    compute_text_statistics,
    load_hf_texts_with_metadata,
    load_mi_re_texts,
    load_reference_sae_backend,
    run_structural_diagnostic,
    save_json,
    summarize_variant_compare,
)
from nlp_re_base.model import load_local_model_and_tokenizer
from nlp_re_base.sae import load_sae_from_hub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the Llama Scope TopK/JumpReLU runtime semantics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-id", default="cerebras/SlimPajama-627B")
    parser.add_argument("--dataset-config-name", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--smoke-docs", type=int, default=32)
    parser.add_argument("--full-docs", type=int, default=400)
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ce-kl-batch-size", type=int, default=None)
    parser.add_argument("--mi-re-data-dir", default="data/mi_re")
    parser.add_argument("--sae-config", default="config/sae_config.json")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _load_json(path: str | Path) -> dict:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_checkpoint_hyperparams(repo_id: str, subfolder: str) -> dict:
    hp_path = hf_hub_download(repo_id=repo_id, filename=f"{subfolder}/hyperparams.json")
    with open(hp_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_parity_for_semantics(
    *,
    semantics: str,
    residual_batches: list[dict],
    sae_config: dict,
    device: torch.device,
) -> dict:
    local_sae = load_sae_from_hub(
        repo_id=sae_config["sae_repo_id"],
        subfolder=sae_config["sae_subfolder"],
        device=device,
        dtype=torch.bfloat16,
        checkpoint_topk_semantics=semantics,
    )
    reference_backend, backend_name = load_reference_sae_backend(
        repo_id=sae_config["sae_repo_id"],
        subfolder=sae_config["sae_subfolder"],
        device=device,
        dtype=torch.bfloat16,
        checkpoint_topk_semantics=semantics,
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
        topk_compare=10,
    )
    report["checkpoint_topk_semantics"] = semantics
    report["reference_backend"] = backend_name
    return report


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir, default_subdir="llamascope_variant_audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    sae_config = _load_json(args.sae_config)
    hook_point = sae_config["hook_point"]
    checkpoint_hyperparams = _load_checkpoint_hyperparams(
        sae_config["sae_repo_id"],
        sae_config["sae_subfolder"],
    )

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("\n==============================================")
    print("  Llama Scope TopK Variant Audit")
    print("==============================================")
    print(f"  Device:       {device}")
    print(f"  Smoke docs:   {args.smoke_docs}")
    print(f"  Full docs:    {args.full_docs}")

    smoke_bundle = load_hf_texts_with_metadata(
        dataset_id=args.dataset_id,
        config_name=args.dataset_config_name,
        split=args.split,
        text_field=args.text_field,
        max_docs=args.smoke_docs,
        streaming=args.streaming,
    )
    smoke_texts = smoke_bundle["texts"]
    dataset_access = smoke_bundle["dataset_access"]
    if not smoke_texts:
        raise SystemExit("No smoke texts were loaded for variant audit.")

    mire_texts, mire_metadata = load_mi_re_texts(args.mi_re_data_dir)
    evidence = build_llamascope_evidence_table(dataset_access=dataset_access)
    evidence["variant_evidence_matrix"] = build_variant_evidence_matrix(
        checkpoint_hyperparams=checkpoint_hyperparams
    )
    save_json(evidence, output_dir / "variant_evidence_matrix.json")
    save_json(compute_text_statistics(smoke_texts), output_dir / "smoke_sample_stats.json")

    model, tokenizer, _ = load_local_model_and_tokenizer(
        args.model_config,
        model_dir=args.model_dir,
        device=device,
    )
    residual_batches = collect_residual_batches(
        model=model,
        tokenizer=tokenizer,
        texts=smoke_texts,
        hook_point=hook_point,
        max_seq_len=args.max_seq_len,
        batch_size=max(1, min(args.batch_size, 2)),
        device=device,
    )

    semantics_reports: dict[str, dict] = {}
    semantics_compare: dict[str, dict] = {}
    for semantics in ("disabled", "hard"):
        print(f"\n[Audit] Running semantics={semantics} ...")
        semantics_dir = output_dir / semantics
        semantics_dir.mkdir(parents=True, exist_ok=True)

        parity = _run_parity_for_semantics(
            semantics=semantics,
            residual_batches=residual_batches,
            sae_config=sae_config,
            device=device,
        )
        save_json(parity, semantics_dir / "parity_report.json")

        sae = load_sae_from_hub(
            repo_id=sae_config["sae_repo_id"],
            subfolder=sae_config["sae_subfolder"],
            device=device,
            dtype=torch.bfloat16,
            checkpoint_topk_semantics=semantics,
        )
        paper_result = run_structural_diagnostic(
            texts=smoke_texts,
            output_dir=semantics_dir / "paper_distribution",
            dataset_label="paper_distribution",
            tokenizer=tokenizer,
            model=model,
            sae=sae,
            hook_point=hook_point,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            device=device,
            ce_kl_batch_size=args.ce_kl_batch_size,
            ce_kl_max_texts=args.smoke_docs,
        )
        mire_result = run_structural_diagnostic(
            texts=mire_texts,
            output_dir=semantics_dir / "mi_re_baseline",
            dataset_label="mi_re_baseline",
            tokenizer=tokenizer,
            model=model,
            sae=sae,
            hook_point=hook_point,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            device=device,
            ce_kl_batch_size=args.ce_kl_batch_size,
            ce_kl_max_texts=args.smoke_docs,
        )
        compare = summarize_variant_compare(
            semantics=semantics,
            paper_result=paper_result,
            mire_result=mire_result,
            parity_report=parity,
        )
        semantics_compare[semantics] = compare
        save_json(compare, semantics_dir / "variant_smoke_summary.json")
        semantics_reports[semantics] = {
            "paper_result": paper_result,
            "mire_result": mire_result,
            "parity": parity,
        }

    disabled_summary = semantics_compare["disabled"]["paper_distribution"]
    hard_summary = semantics_compare["hard"]["paper_distribution"]
    disabled_bad = (
        (disabled_summary["mse"] is not None and hard_summary["mse"] is not None and disabled_summary["mse"] > hard_summary["mse"] * 10)
        or (
            disabled_summary["l0_mean"] is not None
            and checkpoint_hyperparams.get("top_k") is not None
            and disabled_summary["l0_mean"] > float(checkpoint_hyperparams["top_k"]) * 2
        )
        or (
            disabled_summary["ev_openmoss_aligned"] is not None
            and hard_summary["ev_openmoss_aligned"] is not None
            and disabled_summary["ev_openmoss_aligned"] < hard_summary["ev_openmoss_aligned"] - 0.1
        )
    )
    selected_semantics = "hard" if disabled_bad else "disabled"
    decision = {
        "selected_semantics": selected_semantics,
        "decision_basis": (
            "paper_does_not_fully_specify_vanilla_runtime_topk; "
            "selection_uses_checkpoint_metadata_plus_smoke_results"
        ),
        "paper_writes_vanilla_runtime_topk_clearly": False,
        "notes": [
            "The paper and local evidence support post-processing TopK SAEs into JumpReLU variants.",
            "The public checkpoint still exposes top_k=50 in hyperparams.",
            "Official vanilla SAE JumpReLU runtime does not automatically enforce hard top-k.",
            "When jumprelu_only causes severe L0 / MSE / EV regression on the public checkpoint, prefer explicit checkpoint semantics layer with hard top-k.",
        ],
        "selection_rationale": {
            "disabled_semantics_catastrophic_regression": disabled_bad,
            "checkpoint_top_k": checkpoint_hyperparams.get("top_k"),
        },
        "semantics_compare": semantics_compare,
    }
    save_json(decision, output_dir / "variant_decision.json")

    full_bundle = load_hf_texts_with_metadata(
        dataset_id=args.dataset_id,
        config_name=args.dataset_config_name,
        split=args.split,
        text_field=args.text_field,
        max_docs=args.full_docs,
        streaming=args.streaming,
    )
    full_texts = full_bundle["texts"]
    selected_semantics = decision["selected_semantics"]
    selected_sae = load_sae_from_hub(
        repo_id=sae_config["sae_repo_id"],
        subfolder=sae_config["sae_subfolder"],
        device=device,
        dtype=torch.bfloat16,
        checkpoint_topk_semantics=selected_semantics,
    )
    selected_dir = output_dir / f"selected_{selected_semantics}_{args.full_docs}"
    paper_full = run_structural_diagnostic(
        texts=full_texts,
        output_dir=selected_dir,
        dataset_label="paper_distribution",
        tokenizer=tokenizer,
        model=model,
        sae=selected_sae,
        hook_point=hook_point,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=device,
        ce_kl_batch_size=args.ce_kl_batch_size,
        ce_kl_max_texts=args.full_docs,
    )
    mire_full = run_structural_diagnostic(
        texts=mire_texts,
        output_dir=selected_dir / "mi_re_baseline",
        dataset_label="mi_re_baseline",
        tokenizer=tokenizer,
        model=model,
        sae=selected_sae,
        hook_point=hook_point,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=device,
        ce_kl_batch_size=args.ce_kl_batch_size,
        ce_kl_max_texts=args.full_docs,
    )
    final_summary = {
        "selected_semantics": selected_semantics,
        "paper_full": summarize_variant_compare(
            semantics=selected_semantics,
            paper_result=paper_full,
            mire_result=mire_full,
            parity_report=semantics_reports[selected_semantics]["parity"],
        ),
        "dataset_access": full_bundle["dataset_access"],
    }
    save_json(final_summary, selected_dir / "selected_variant_summary.json")

    print("\nCompleted Llama Scope variant audit.")
    print(f"  Evidence matrix: {output_dir / 'variant_evidence_matrix.json'}")
    print(f"  Decision:        {output_dir / 'variant_decision.json'}")
    print(f"  Selected run:    {selected_dir / 'metrics_structural.json'}")


if __name__ == "__main__":
    main()
