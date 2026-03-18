"""Activation extraction from the Llama model and SAE forward pass utilities.

Handles:
  1. Extracting residual stream activations from the specified hook point
  2. Running SAE forward pass on extracted activations
  3. Aggregating token-level latents to utterance-level features

MEMORY DESIGN:
  Instead of concatenating all [N, T, 32768] latents into one giant tensor,
  this module processes data in streaming chunks and only keeps the
  aggregated utterance-level features [N, d_sae] in memory.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .sae import SparseAutoencoder


# ────────────────────────────────────────────────────────────────
# Hook-point mapping:  "blocks.19.hook_resid_post"
#   → model.model.layers[19] output (residual stream after block 19)
# ────────────────────────────────────────────────────────────────

def _parse_hook_point(hook_point: str) -> int:
    """Parse 'blocks.N.hook_resid_post' and return the layer index N."""
    parts = hook_point.split(".")
    if len(parts) != 3 or parts[0] != "blocks" or parts[2] != "hook_resid_post":
        raise ValueError(
            f"Unsupported hook_point format: '{hook_point}'. "
            f"Expected 'blocks.<N>.hook_resid_post'."
        )
    return int(parts[1])


def _tokenize_batch(
    batch_texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
) -> dict[str, torch.Tensor]:
    """Tokenize a batch of texts."""
    return tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )


# ────────────────────────────────────────────────────────────────
# STREAMING PIPELINE
#   Processes data batch by batch. Each batch:
#     1) extract activations from base model
#     2) run SAE forward
#     3) aggregate to utterance level
#     4) append only the aggregated [B, d_sae] / [B, d_model]
#   Total memory: O(batch_size * T * max(d_model, d_sae))
# ────────────────────────────────────────────────────────────────

def extract_and_process_streaming(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sae: SparseAutoencoder,
    texts: list[str],
    hook_point: str = "blocks.19.hook_resid_post",
    max_seq_len: int = 128,
    batch_size: int = 8,
    aggregation: str = "max",
    binarized_threshold: float = 0.0,
    device: str | torch.device | None = None,
    collect_structural_samples: int = 5,
    structural_accumulator: Optional[Any] = None,
) -> dict[str, Any]:
    """Extract activations, run SAE, and aggregate — all in streaming fashion.

    Only keeps utterance-level features in memory. Token-level data is
    accumulated online via `structural_accumulator` (if provided) or
    stored for the first `collect_structural_samples` batches.

    Args:
        model: HuggingFace causal LM (Llama-3.1-8B).
        tokenizer: Corresponding tokenizer.
        sae: Loaded SparseAutoencoder.
        texts: List of utterance strings.
        hook_point: e.g. 'blocks.19.hook_resid_post'.
        max_seq_len: Maximum token sequence length.
        batch_size: Batch size for inference.
        aggregation: 'max', 'mean', 'sum', or 'binarized_sum' pooling.
        binarized_threshold: Threshold used when aggregation='binarized_sum'.
        device: Device for model inference.
        collect_structural_samples: Number of batches to keep full
            token-level data for structural metric computation.
        structural_accumulator: Optional OnlineStructuralAccumulator from
            eval_structural. When provided, receives every batch's token-level
            data and computes metrics over the full dataset at O(1) memory.
            When None, falls back to legacy sample-based approach.

    Returns:
        Dictionary with:
            'utterance_features': [N, d_sae]  (aggregated SAE latents)
            'utterance_activations': [N, d_model]  (aggregated raw activations)
            'sample_activations': [S, T, d_model]  (small sample for structural)
            'sample_reconstructed': [S, T, d_model]
            'sample_activations_normalized': [S, T, d_model]
            'sample_reconstructed_normalized': [S, T, d_model]
            'sample_latents': [S, T, d_sae]
            'sample_mask': [S, T]
    """
    layer_idx = _parse_hook_point(hook_point)

    if device is None:
        device = next(model.parameters()).device

    target_layer = model.model.layers[layer_idx]

    # Determine SAE dtype for alignment
    sae_dtype = getattr(sae, "sae_dtype", next(sae.parameters()).dtype)
    sae_device = next(sae.parameters()).device

    # Accumulators for aggregated (utterance-level) results
    all_utterance_features = []     # [B, d_sae] each
    all_utterance_activations = []  # [B, d_model] each

    # Small sample of full token-level data for structural metrics
    sample_acts = []
    sample_recon = []
    sample_acts_normed = []
    sample_recon_normed = []
    sample_latents = []
    sample_masks = []
    samples_collected = 0

    captured: dict[str, Any] = {}

    def _hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured["hidden_states"] = output[0].detach()
        else:
            captured["hidden_states"] = output.detach()

    handle = target_layer.register_forward_hook(_hook_fn)

    try:
        for start_idx in tqdm(
            range(0, len(texts), batch_size),
            desc="Extract+SAE+Aggregate",
            unit="batch",
        ):
            batch_texts = texts[start_idx : start_idx + batch_size]
            encoded = _tokenize_batch(batch_texts, tokenizer, max_seq_len)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            batch_mask = encoded["attention_mask"]

            # ── Step 1: Extract activations ──
            with torch.inference_mode():
                _ = model(**encoded)

            # hidden_states is [B, T, d_model] on model device
            batch_activations = captured["hidden_states"]  # keep on device

            # ── Step 2: SAE forward (with dtype alignment) ──
            sae_input = batch_activations.to(device=sae_device, dtype=sae_dtype)
            with torch.inference_mode():
                sae_outputs = sae.forward_with_details(sae_input)

            # Move results to CPU immediately to free GPU memory
            batch_latents_cpu = sae_outputs["latents"].cpu().float()
            batch_recon_cpu = sae_outputs["reconstructed_raw"].cpu().float()
            batch_recon_normed_cpu = sae_outputs["reconstructed_normalized"].cpu().float()
            batch_acts_cpu = batch_activations.cpu().float()
            batch_acts_normed_cpu = sae_outputs["input_normalized"].cpu().float()
            batch_mask_cpu = batch_mask.cpu()

            # ── Step 3: Aggregate to utterance level ──
            utt_features = _aggregate_batch(
                batch_latents_cpu,
                batch_mask_cpu,
                aggregation,
                binarized_threshold=binarized_threshold,
            )
            utt_activations = _aggregate_batch(
                batch_acts_cpu,
                batch_mask_cpu,
                aggregation,
                binarized_threshold=binarized_threshold,
            )

            all_utterance_features.append(utt_features)
            all_utterance_activations.append(utt_activations)

            # ── Step 4a: Feed the online accumulator (full-data structural) ──
            if structural_accumulator is not None:
                if isinstance(structural_accumulator, dict):
                    raw_acc = structural_accumulator.get("raw")
                    norm_acc = structural_accumulator.get("normalized")
                else:
                    raw_acc = structural_accumulator
                    norm_acc = None
                raw_acc.update(
                    z=batch_acts_cpu,
                    z_hat=batch_recon_cpu,
                    latents=batch_latents_cpu,
                    mask=batch_mask_cpu,
                )
                if norm_acc is not None:
                    norm_acc.update(
                        z=batch_acts_normed_cpu,
                        z_hat=batch_recon_normed_cpu,
                        latents=batch_latents_cpu,
                        mask=batch_mask_cpu,
                    )

            # ── Step 4b: Legacy sample collection (first N batches only) ──
            if samples_collected < collect_structural_samples:
                sample_acts.append(batch_acts_cpu.clone())
                sample_recon.append(batch_recon_cpu.clone())
                sample_acts_normed.append(batch_acts_normed_cpu.clone())
                sample_recon_normed.append(batch_recon_normed_cpu.clone())
                sample_latents.append(batch_latents_cpu.clone())
                sample_masks.append(batch_mask_cpu.clone())
                samples_collected += 1

            # Explicitly free batch tensors
            del batch_activations, sae_input, sae_outputs
            del batch_latents_cpu, batch_recon_cpu, batch_recon_normed_cpu
            del batch_acts_cpu, batch_acts_normed_cpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        handle.remove()

    # ── Pad structural samples to uniform seq len and cat ──
    sample_activations, sample_reconstructed, sample_lat, sample_msk = \
        _pad_and_cat_samples(sample_acts, sample_recon, sample_latents, sample_masks)
    sample_activations_normed = _pad_and_cat_tensor_list(sample_acts_normed)
    sample_reconstructed_normed = _pad_and_cat_tensor_list(sample_recon_normed)

    return {
        "utterance_features": torch.cat(all_utterance_features, dim=0),       # [N, d_sae]
        "utterance_activations": torch.cat(all_utterance_activations, dim=0), # [N, d_model]
        "sample_activations": sample_activations,      # [S, T, d_model]
        "sample_reconstructed": sample_reconstructed,   # [S, T, d_model]
        "sample_activations_normalized": sample_activations_normed,   # [S, T, d_model]
        "sample_reconstructed_normalized": sample_reconstructed_normed,   # [S, T, d_model]
        "sample_latents": sample_lat,                   # [S, T, d_sae]
        "sample_mask": sample_msk,                      # [S, T]
    }


def _aggregate_batch(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    method: str = "max",
    binarized_threshold: float = 0.0,
) -> torch.Tensor:
    """Aggregate a [B, T, D] tensor to [B, D] using mask.

    This is the inner per-batch aggregation used by the streaming pipeline.
    """
    mask_3d = mask.unsqueeze(-1).float()

    if method == "max":
        masked = tensor.clone()
        masked[mask_3d.squeeze(-1) == 0] = -float("inf")
        result, _ = masked.max(dim=1)
        result[result == -float("inf")] = 0.0
    elif method == "mean":
        masked = tensor * mask_3d
        token_counts = mask_3d.sum(dim=1).clamp(min=1)
        result = masked.sum(dim=1) / token_counts
    elif method == "sum":
        result = (tensor * mask_3d).sum(dim=1)
    elif method == "binarized_sum":
        summed = (tensor * mask_3d).sum(dim=1)
        result = (summed > binarized_threshold).to(tensor.dtype)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    return result


def _pad_and_cat_samples(
    acts_list: list[torch.Tensor],
    recon_list: list[torch.Tensor],
    latents_list: list[torch.Tensor],
    masks_list: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad structural sample batches to uniform seq len and concatenate."""
    if not acts_list:
        return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

    max_t = max(a.shape[1] for a in acts_list)

    def _pad(tensor_list, pad_val=0.0):
        padded = []
        for t in tensor_list:
            b, seq, *rest = t.shape
            if seq < max_t:
                pad_shape = [b, max_t - seq] + rest
                t = torch.cat([t, torch.full(pad_shape, pad_val, dtype=t.dtype)], dim=1)
            padded.append(t)
        return torch.cat(padded, dim=0)

    return (
        _pad(acts_list),
        _pad(recon_list),
        _pad(latents_list),
        _pad(masks_list, pad_val=0),
    )


def _pad_and_cat_tensor_list(
    tensor_list: list[torch.Tensor],
    pad_val: float = 0.0,
) -> torch.Tensor:
    """Pad a list of [B, T, ...] tensors to a common sequence length and cat."""
    if not tensor_list:
        return torch.empty(0)

    max_t = max(t.shape[1] for t in tensor_list)
    padded = []
    for tensor in tensor_list:
        bsz, seq_len, *rest = tensor.shape
        if seq_len < max_t:
            pad_shape = [bsz, max_t - seq_len] + rest
            tensor = torch.cat(
                [tensor, torch.full(pad_shape, pad_val, dtype=tensor.dtype)],
                dim=1,
            )
        padded.append(tensor)
    return torch.cat(padded, dim=0)


# ────────────────────────────────────────────────────────────────
# Legacy API (kept for backward compatibility with smoke tests)
# ────────────────────────────────────────────────────────────────

def aggregate_to_utterance(
    latents: torch.Tensor,
    attention_mask: torch.Tensor,
    method: str = "max",
    binarized_threshold: float = 0.0,
) -> torch.Tensor:
    """Aggregate token-level latent activations to utterance-level features.

    Args:
        latents: [N, T, d_sae] token-level latent activations.
        attention_mask: [N, T] binary mask (1 = real token, 0 = padding).
        method: Aggregation method ('max', 'mean', 'sum', or 'binarized_sum').
        binarized_threshold: Threshold used when method='binarized_sum'.

    Returns:
        Utterance-level features: [N, d_sae]
    """
    return _aggregate_batch(
        latents,
        attention_mask,
        method,
        binarized_threshold=binarized_threshold,
    )
