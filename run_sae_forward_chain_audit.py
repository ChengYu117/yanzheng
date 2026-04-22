"""Audit intermediate SAE forward states for the public Llama Scope checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.config import resolve_output_dir
from nlp_re_base.diagnostics import (
    collect_residual_batches,
    load_hf_texts_with_metadata,
    save_json,
)
from nlp_re_base.eval_structural import OfficialMetricsAccumulator, OnlineStructuralAccumulator
from nlp_re_base.model import load_local_model_and_tokenizer
from nlp_re_base.sae import OpenMossLmSaesAdapter, load_sae_from_hub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit intermediate SAE forward states and top-k placement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-id", default="cerebras/SlimPajama-627B")
    parser.add_argument("--dataset-config-name", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-docs", type=int, default=8)
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sae-config", default="config/sae_config.json")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)


def _apply_hard_topk(values: torch.Tensor, k: int | None) -> torch.Tensor:
    if k is None or k <= 0 or k >= values.shape[-1]:
        return values
    topk_indices = values.topk(k, dim=-1).indices
    keep_mask = torch.zeros_like(values, dtype=torch.bool)
    keep_mask.scatter_(-1, topk_indices, True)
    return values * keep_mask.to(values.dtype)


def _summarize_topk_overlap(
    post_division_acts: torch.Tensor,
    pre_division_acts: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    flat_mask = mask.bool().reshape(-1)
    post_active = (post_division_acts.reshape(-1, post_division_acts.shape[-1])[flat_mask] > 0)
    pre_active = (pre_division_acts.reshape(-1, pre_division_acts.shape[-1])[flat_mask] > 0)
    if post_active.numel() == 0:
        return {
            "exact_match_rate": 1.0,
            "mean_jaccard": 1.0,
            "mean_overlap_ratio": 1.0,
        }

    exact = (post_active == pre_active).all(dim=-1).float().mean().item()
    intersection = (post_active & pre_active).float().sum(dim=-1)
    union = (post_active | pre_active).float().sum(dim=-1).clamp(min=1.0)
    post_count = post_active.float().sum(dim=-1).clamp(min=1.0)
    overlap_ratio = intersection / post_count
    return {
        "exact_match_rate": float(exact),
        "mean_jaccard": float((intersection / union).mean().item()),
        "mean_overlap_ratio": float(overlap_ratio.mean().item()),
    }


def _accumulators() -> dict[str, dict[str, Any]]:
    return {
        "jumprelu_only": {
            "raw": OnlineStructuralAccumulator(),
            "official": OfficialMetricsAccumulator(),
        },
        "hard_post_division": {
            "raw": OnlineStructuralAccumulator(),
            "official": OfficialMetricsAccumulator(),
        },
        "hard_pre_division": {
            "raw": OnlineStructuralAccumulator(),
            "official": OfficialMetricsAccumulator(),
        },
    }


def _result_payload(name: str, accs: dict[str, Any]) -> dict[str, Any]:
    raw = accs["raw"].result()
    official = accs["official"].result()
    return {
        "variant": name,
        "raw_metrics": raw,
        "official_metrics": official,
        "summary": {
            "mse": raw.get("mse"),
            "cosine_similarity": raw.get("cosine_similarity"),
            "ev_centered": raw.get("explained_variance"),
            "ev_paper": raw.get("explained_variance_paper"),
            "ev_official_batch": official.get("metrics/explained_variance"),
            "l0_mean": raw.get("l0_mean"),
            "dead_ratio": raw.get("dead_ratio"),
            "l2_norm_error_official": official.get("metrics/l2_norm_error"),
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir, default_subdir="sae_forward_chain_audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("\n============================================")
    print("  SAE Forward Chain Audit")
    print("============================================")
    print(f"  Device:   {device}")
    print(f"  Dataset:  {args.dataset_id} [{args.split}]")
    print(f"  Max docs: {args.max_docs}")

    sae_config = _load_json(args.sae_config)
    hook_point = sae_config["hook_point"]

    text_bundle = load_hf_texts_with_metadata(
        dataset_id=args.dataset_id,
        config_name=args.dataset_config_name,
        split=args.split,
        text_field=args.text_field,
        max_docs=args.max_docs,
        streaming=args.streaming,
    )
    texts = text_bundle["texts"]
    save_json({"dataset_access": text_bundle["dataset_access"], "n_texts": len(texts)}, output_dir / "input_bundle.json")

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
        checkpoint_topk_semantics="hard",
    )
    if not isinstance(sae, OpenMossLmSaesAdapter):
        raise RuntimeError("Forward-chain audit requires the official OpenMOSS adapter backend.")

    backend = sae.backend_sae
    if not hasattr(backend, "W_E") or not hasattr(backend, "W_D"):
        raise RuntimeError("Official backend does not expose the expected SAE weights.")

    residual_batches = collect_residual_batches(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        hook_point=hook_point,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=device,
    )

    accumulators = _accumulators()
    overlap_summaries: list[dict[str, float]] = []
    batch_debug: list[dict[str, Any]] = []

    sae_dtype = getattr(sae, "sae_dtype", next(sae.parameters()).dtype)
    decoder_norm = backend.decoder_norm().to(device=device, dtype=sae_dtype)
    top_k = int(getattr(backend.cfg, "top_k", 0))
    use_decoder_norm = bool(getattr(backend.cfg, "sparsity_include_decoder_norm", False))

    for batch_index, batch in enumerate(residual_batches):
        x = batch["residual"].to(device=device, dtype=sae_dtype)
        mask = batch["attention_mask"].to(device=device)
        x_normalized, scale_factor = sae.normalize_with_stats(x)

        hidden_pre_raw = x_normalized @ backend.W_E + backend.b_E
        hidden_pre_scaled = hidden_pre_raw * decoder_norm if use_decoder_norm else hidden_pre_raw

        activated_scaled = backend.activation_function(hidden_pre_scaled)
        if use_decoder_norm:
            feature_acts_official = activated_scaled / decoder_norm
        else:
            feature_acts_official = activated_scaled

        feature_acts_post = _apply_hard_topk(feature_acts_official, top_k)
        feature_acts_pre = _apply_hard_topk(activated_scaled, top_k)
        if use_decoder_norm:
            feature_acts_pre = feature_acts_pre / decoder_norm

        recon_official_norm = backend.decode(feature_acts_official)
        recon_post_norm = backend.decode(feature_acts_post)
        recon_pre_norm = backend.decode(feature_acts_pre)

        recon_official_raw = sae.denormalize_reconstruction(recon_official_norm, scale_factor)
        recon_post_raw = sae.denormalize_reconstruction(recon_post_norm, scale_factor)
        recon_pre_raw = sae.denormalize_reconstruction(recon_pre_norm, scale_factor)

        x_cpu = x.detach().cpu().float()
        mask_cpu = mask.detach().cpu()
        variants = {
            "jumprelu_only": (feature_acts_official.detach().cpu().float(), recon_official_raw.detach().cpu().float()),
            "hard_post_division": (feature_acts_post.detach().cpu().float(), recon_post_raw.detach().cpu().float()),
            "hard_pre_division": (feature_acts_pre.detach().cpu().float(), recon_pre_raw.detach().cpu().float()),
        }
        for name, (lat_cpu, recon_cpu) in variants.items():
            accumulators[name]["raw"].update(z=x_cpu, z_hat=recon_cpu, latents=lat_cpu, mask=mask_cpu)
            accumulators[name]["official"].update(z=x_cpu, z_hat=recon_cpu, latents=lat_cpu, mask=mask_cpu)

        overlap = _summarize_topk_overlap(feature_acts_post.detach().cpu(), feature_acts_pre.detach().cpu(), mask_cpu)
        overlap_summaries.append(overlap)
        batch_debug.append(
            {
                "batch_index": batch_index,
                "n_tokens": int(mask.sum().item()),
                "topk_overlap": overlap,
                "feature_mean_abs_diff_post_vs_pre": float(
                    (feature_acts_post - feature_acts_pre).abs().mean().item()
                ),
                "reconstruction_mean_abs_diff_post_vs_pre": float(
                    (recon_post_raw - recon_pre_raw).abs().mean().item()
                ),
                "hidden_pre_raw_mean_abs": float(hidden_pre_raw.abs().mean().item()),
                "hidden_pre_scaled_mean_abs": float(hidden_pre_scaled.abs().mean().item()),
                "decoder_norm_mean": float(decoder_norm.mean().item()),
                "decoder_norm_std": float(decoder_norm.std(unbiased=False).item()),
            }
        )

    results = {name: _result_payload(name, accs) for name, accs in accumulators.items()}
    overlap_summary = {
        "mean_exact_match_rate": sum(item["exact_match_rate"] for item in overlap_summaries) / max(len(overlap_summaries), 1),
        "mean_jaccard": sum(item["mean_jaccard"] for item in overlap_summaries) / max(len(overlap_summaries), 1),
        "mean_overlap_ratio": sum(item["mean_overlap_ratio"] for item in overlap_summaries) / max(len(overlap_summaries), 1),
    }
    decision_hint = "topk_placement_likely_not_primary_issue"
    post_ev = results["hard_post_division"]["summary"]["ev_official_batch"]
    pre_ev = results["hard_pre_division"]["summary"]["ev_official_batch"]
    if pre_ev is not None and post_ev is not None and abs(float(pre_ev) - float(post_ev)) > 0.05:
        decision_hint = (
            "hard_topk_placement_changes_reconstruction_meaningfully"
            if float(pre_ev) > float(post_ev)
            else "current_post_division_topk_is_no_worse_than_pre_division_topk"
        )

    payload = {
        "dataset_access": text_bundle["dataset_access"],
        "n_batches": len(residual_batches),
        "n_texts": len(texts),
        "hook_point": hook_point,
        "checkpoint_topk": top_k,
        "sparsity_include_decoder_norm": use_decoder_norm,
        "runtime_norm_activation": getattr(backend.cfg, "norm_activation", None),
        "overlap_summary": overlap_summary,
        "decision_hint": decision_hint,
        "variants": results,
    }
    save_json(payload, output_dir / "forward_chain_audit_summary.json")
    save_json(batch_debug, output_dir / "forward_chain_audit_batches.json")

    print("\nForward-chain audit complete.")
    print(f"  Output: {output_dir / 'forward_chain_audit_summary.json'}")
    print(f"  Decision: {decision_hint}")


if __name__ == "__main__":
    main()
