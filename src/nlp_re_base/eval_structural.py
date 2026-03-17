"""Structural evaluation metrics for the SAE.

Metrics implemented:
  - MSE (Mean Squared Error)
  - Cosine Similarity
  - Explained Variance / FVU
  - L₀ Sparsity
  - Firing Frequency & Dead Feature Ratio
  - CE Loss Delta & KL Divergence (requires model forward pass)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .sae import SparseAutoencoder


# ─────────────────── Online structural accumulator ───────────────────────────
# Computes MSE, Cosine Similarity, and EV/FVU over the full dataset in
# O(1) extra memory using Welford's online algorithm.  Feed it one batch
# at a time via .update(); call .result() at the end.


class OnlineStructuralAccumulator:
    """Online (streaming) accumulator for token-level structural metrics.

    Processes one [B, T, d] batch at a time, masked by an attention mask.
    Uses Welford's algorithm for numerically stable variance estimation.
    Memory cost: O(d_model) — constant regardless of dataset size.
    """

    def __init__(self) -> None:
        # MSE accumulator
        self._n_tokens: int = 0
        self._mse_sum: float = 0.0

        # Cosine similarity accumulator
        self._cos_sum: float = 0.0

        # Welford's online variance for z and residual (z - z_hat)
        # shape: (d_model,) — kept in float64 for numerical stability
        self._count_w: int = 0
        self._mean_z: torch.Tensor | None = None     # (d_model,)
        self._M2_z: torch.Tensor | None = None       # (d_model,)
        self._mean_r: torch.Tensor | None = None     # (d_model,)
        self._M2_r: torch.Tensor | None = None       # (d_model,)

        # L0 accumulator
        self._l0_sum: float = 0.0
        self._l0_sq_sum: float = 0.0

        # Firing frequency accumulator: shape (d_sae,)
        self._freq_sum: torch.Tensor | None = None
        self._freq_token_count: int = 0

    def update(
        self,
        z: torch.Tensor,          # [B, T, d_model] — float32, CPU
        z_hat: torch.Tensor,      # [B, T, d_model]
        latents: torch.Tensor,    # [B, T, d_sae]
        mask: torch.Tensor,       # [B, T] — bool or 0/1 int
    ) -> None:
        """Ingest one batch of token-level data."""
        # Work in float64 for stability
        mask_bool = mask.bool()                               # [B, T]
        flat_mask = mask_bool.reshape(-1)                     # [B*T]

        z_flat = z.reshape(-1, z.shape[-1]).double()[flat_mask]       # [M, d]
        z_hat_flat = z_hat.reshape(-1, z_hat.shape[-1]).double()[flat_mask]
        r_flat = z_flat - z_hat_flat                          # residual [M, d]
        lat_flat = latents.reshape(-1, latents.shape[-1]).float()[flat_mask]

        m = z_flat.shape[0]
        if m == 0:
            return

        d = z_flat.shape[1]

        # ── MSE ──
        self._mse_sum += float((r_flat ** 2).sum())
        self._n_tokens += m

        # ── Cosine similarity ──
        cos = F.cosine_similarity(
            z_flat.float(), z_hat_flat.float(), dim=-1
        )
        self._cos_sum += float(cos.sum())

        # ── Welford online variance for z and residual ──
        if self._mean_z is None:
            self._mean_z = torch.zeros(d, dtype=torch.float64)
            self._M2_z   = torch.zeros(d, dtype=torch.float64)
            self._mean_r = torch.zeros(d, dtype=torch.float64)
            self._M2_r   = torch.zeros(d, dtype=torch.float64)

        for i in range(m):
            self._count_w += 1
            # Welford update for z
            delta_z = z_flat[i] - self._mean_z
            self._mean_z += delta_z / self._count_w
            delta2_z = z_flat[i] - self._mean_z
            self._M2_z  += delta_z * delta2_z
            # Welford update for residual
            delta_r = r_flat[i] - self._mean_r
            self._mean_r += delta_r / self._count_w
            delta2_r = r_flat[i] - self._mean_r
            self._M2_r   += delta_r * delta2_r

        # ── L0 sparsity ──
        active = (lat_flat.abs() > 1e-8)
        l0_per_tok = active.float().sum(dim=-1)     # [M]
        self._l0_sum    += float(l0_per_tok.sum())
        self._l0_sq_sum += float((l0_per_tok ** 2).sum())

        # ── Firing frequency ──
        if self._freq_sum is None:
            self._freq_sum = torch.zeros(lat_flat.shape[1], dtype=torch.float64)
        self._freq_sum += active.float().sum(dim=0).double()
        self._freq_token_count += m

    def result(self) -> dict:
        """Return all accumulated structural metrics as a plain dict."""
        n = self._n_tokens
        if n == 0:
            return {}

        mse    = self._mse_sum / n
        cos    = self._cos_sum / n
        l0_mean = self._l0_sum / n
        l0_std  = float(
            (self._l0_sq_sum / n - (self._l0_sum / n) ** 2) ** 0.5
        ) if n > 1 else 0.0

        # Variance from Welford accumulators
        if self._count_w > 1 and self._M2_z is not None:
            var_z = self._M2_z / (self._count_w - 1)
            var_r = self._M2_r / (self._count_w - 1)
        else:
            var_z = torch.ones(1)
            var_r = torch.zeros(1)

        total_var_z  = float(var_z.mean())
        total_var_r  = float(var_r.mean())

        if total_var_z < 1e-12:
            ev, fvu = 1.0, 0.0
        else:
            fvu = total_var_r / total_var_z
            ev  = 1.0 - fvu

        # Firing frequency
        freq_result: dict = {}
        if self._freq_sum is not None and self._freq_token_count > 0:
            freq = self._freq_sum / self._freq_token_count  # per-latent [d_sae]
            d_sae = freq.shape[0]
            dead_mask = freq < 1e-8
            dead_count = int(dead_mask.sum().item())
            top10 = freq.topk(min(10, d_sae)).indices.tolist()
            freq_result = {
                "dead_count": dead_count,
                "dead_ratio": dead_count / d_sae,
                "alive_count": int(d_sae - dead_count),
                "top10_freq_indices": top10,
            }

        return {
            "mse":                mse,
            "cosine_similarity":  cos,
            "explained_variance": ev,
            "fvu":                fvu,
            "l0_mean":            l0_mean,
            "l0_std":             l0_std,
            "n_tokens":           n,
            **freq_result,
        }


# ──────────────────────────── Token-level metrics ────────────────────────────

def compute_mse(
    z: torch.Tensor,
    z_hat: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    """Mean Squared Error between original and reconstructed activations.

    Args:
        z: Original activations [N, T, d_model] or [N, d_model].
        z_hat: Reconstructed activations, same shape as z.
        mask: Optional attention mask [N, T].

    Returns:
        Scalar MSE value.
    """
    diff = (z - z_hat).pow(2)
    if mask is not None and diff.dim() == 3:
        mask_3d = mask.unsqueeze(-1).float()
        diff = diff * mask_3d
        total = mask_3d.sum() * diff.shape[-1]
        return (diff.sum() / total.clamp(min=1)).item()
    return diff.mean().item()


def compute_cosine_similarity(
    z: torch.Tensor,
    z_hat: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    """Mean cosine similarity between original and reconstructed activations.

    Args:
        z: Original activations [N, T, d_model] or [N, d_model].
        z_hat: Reconstructed activations, same shape as z.
        mask: Optional attention mask [N, T].

    Returns:
        Mean cosine similarity (scalar).
    """
    if z.dim() == 3:
        # Flatten to [N*T, d_model] respecting mask
        if mask is not None:
            flat_mask = mask.bool().reshape(-1)
            z_flat = z.reshape(-1, z.shape[-1])[flat_mask]
            z_hat_flat = z_hat.reshape(-1, z_hat.shape[-1])[flat_mask]
        else:
            z_flat = z.reshape(-1, z.shape[-1])
            z_hat_flat = z_hat.reshape(-1, z_hat.shape[-1])
    else:
        z_flat = z
        z_hat_flat = z_hat

    cos_sim = F.cosine_similarity(z_flat, z_hat_flat, dim=-1)
    return cos_sim.mean().item()


def compute_explained_variance(
    z: torch.Tensor,
    z_hat: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Explained Variance and Fraction of Variance Unexplained (FVU).

    Returns:
        {'explained_variance': float, 'fvu': float}
    """
    if z.dim() == 3 and mask is not None:
        flat_mask = mask.bool().reshape(-1)
        z_flat = z.reshape(-1, z.shape[-1])[flat_mask]
        z_hat_flat = z_hat.reshape(-1, z_hat.shape[-1])[flat_mask]
    else:
        z_flat = z.reshape(-1, z.shape[-1])
        z_hat_flat = z_hat.reshape(-1, z_hat.shape[-1])

    residual_var = (z_flat - z_hat_flat).var(dim=0).mean().item()
    total_var = z_flat.var(dim=0).mean().item()

    if total_var < 1e-12:
        return {"explained_variance": 1.0, "fvu": 0.0}

    fvu = residual_var / total_var
    ev = 1.0 - fvu
    return {"explained_variance": ev, "fvu": fvu}


# ──────────────────────────── Sparsity metrics ───────────────────────────────


def compute_l0_sparsity(
    latents: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """L₀ sparsity: average number of non-zero latents per token.

    Args:
        latents: [N, T, d_sae] or [N, d_sae].
        mask: Optional attention mask [N, T].

    Returns:
        {'l0_mean': float, 'l0_std': float}
    """
    active = (latents.abs() > 1e-8).float()

    if latents.dim() == 3:
        # Per-token L0
        l0_per_token = active.sum(dim=-1)  # [N, T]
        if mask is not None:
            flat_mask = mask.bool().reshape(-1)
            l0_values = l0_per_token.reshape(-1)[flat_mask]
        else:
            l0_values = l0_per_token.reshape(-1)
    else:
        l0_values = active.sum(dim=-1)

    return {
        "l0_mean": l0_values.mean().item(),
        "l0_std": l0_values.std().item(),
    }


def compute_firing_frequency(
    latents: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Per-latent firing frequency and dead feature statistics.

    Args:
        latents: [N, T, d_sae] or [N, d_sae].
        mask: Optional attention mask [N, T].

    Returns:
        {
            'firing_freq': Tensor[d_sae],  (fraction of tokens that activate each latent)
            'dead_count': int,
            'dead_ratio': float,
            'alive_count': int,
            'top10_freq_indices': list[int],
        }
    """
    active = (latents.abs() > 1e-8).float()

    if latents.dim() == 3:
        if mask is not None:
            mask_3d = mask.unsqueeze(-1).float()
            active = active * mask_3d
            total_tokens = mask.sum().item()
        else:
            total_tokens = latents.shape[0] * latents.shape[1]
        freq = active.sum(dim=(0, 1)) / max(total_tokens, 1)
    else:
        freq = active.mean(dim=0)

    d_sae = freq.shape[0]
    dead_mask = freq < 1e-8
    dead_count = dead_mask.sum().item()

    top10_indices = freq.topk(min(10, d_sae)).indices.tolist()

    return {
        "firing_freq": freq,
        "dead_count": int(dead_count),
        "dead_ratio": dead_count / d_sae,
        "alive_count": int(d_sae - dead_count),
        "top10_freq_indices": top10_indices,
    }


# ──────────────────────── CE / KL (requires model) ──────────────────────────


def compute_ce_kl_with_intervention(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    sae: SparseAutoencoder,
    hook_point: str = "blocks.19.hook_resid_post",
    max_seq_len: int = 128,
    batch_size: int = 4,
    max_texts: int | None = None,
) -> dict[str, float | int]:
    """Compute CE loss delta and KL divergence by replacing activations with SAE reconstructions.

    Runs the model twice per batch:
      1. Original forward pass → get logits_orig and layer activations
      2. Intervention forward pass → replace hook-point activations with SAE(activations)

    Returns:
        {'ce_loss_orig': float, 'ce_loss_sae': float,
         'ce_loss_delta': float, 'kl_divergence': float,
         'n_eval_texts': int, 'n_eval_pred_tokens': int,
         'ce_kl_batch_size': int}
    """
    from .activations import _parse_hook_point

    layer_idx = _parse_hook_point(hook_point)
    target_layer = model.model.layers[layer_idx]
    device = next(model.parameters()).device
    sae_device = next(sae.parameters()).device
    eval_texts = texts[:max_texts] if max_texts is not None else texts

    total_ce_orig = 0.0
    total_ce_sae = 0.0
    total_kl = 0.0
    total_tokens = 0

    for start in tqdm(
        range(0, len(eval_texts), batch_size),
        desc="CE/KL evaluation",
        unit="batch",
    ):
        batch_texts = eval_texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100

        # ── Pass 1: Original ──
        captured = {}

        def _capture_hook(module, input, output):
            if isinstance(output, tuple):
                captured["hidden"] = output[0]
            else:
                captured["hidden"] = output

        handle = target_layer.register_forward_hook(_capture_hook)
        with torch.inference_mode():
            out_orig = model(**encoded, labels=labels)
        handle.remove()

        logits_orig = out_orig.logits
        ce_orig = out_orig.loss.item()
        original_hidden = captured["hidden"]

        # ── Pass 2: SAE intervention ──
        sae_dtype = getattr(sae, "sae_dtype", next(sae.parameters()).dtype)
        sae_hidden = original_hidden.to(device=sae_device, dtype=sae_dtype)
        with torch.inference_mode():
            recon, _ = sae(sae_hidden)
        recon = recon.to(device=device, dtype=original_hidden.dtype)

        def _replace_hook(module, input, output):
            if isinstance(output, tuple):
                return (recon,) + output[1:]
            return recon

        handle = target_layer.register_forward_hook(_replace_hook)
        with torch.inference_mode():
            out_sae = model(**encoded, labels=labels)
        handle.remove()

        logits_sae = out_sae.logits
        ce_sae = out_sae.loss.item()

        # Match CE/KL to the next-token positions used by causal LM loss.
        pred_mask = labels[:, 1:] != -100
        n_tokens = int(pred_mask.sum().item())
        if n_tokens == 0:
            continue

        logits_orig_shifted = logits_orig[:, :-1, :]
        logits_sae_shifted = logits_sae[:, :-1, :]
        log_probs_orig = F.log_softmax(
            logits_orig_shifted.reshape(-1, logits_orig_shifted.size(-1)),
            dim=-1,
        )
        log_probs_sae = F.log_softmax(
            logits_sae_shifted.reshape(-1, logits_sae_shifted.size(-1)),
            dim=-1,
        )
        probs_orig = log_probs_orig.exp()

        kl_per_token = F.kl_div(log_probs_sae, probs_orig, reduction="none", log_target=False)
        kl_per_token = kl_per_token.sum(dim=-1)  # sum over vocab
        kl_masked = kl_per_token[pred_mask.reshape(-1)]

        total_ce_orig += ce_orig * n_tokens
        total_ce_sae += ce_sae * n_tokens
        total_kl += kl_masked.sum().item()
        total_tokens += n_tokens

    avg_ce_orig = total_ce_orig / max(total_tokens, 1)
    avg_ce_sae = total_ce_sae / max(total_tokens, 1)

    return {
        "ce_loss_orig": avg_ce_orig,
        "ce_loss_sae": avg_ce_sae,
        "ce_loss_delta": avg_ce_sae - avg_ce_orig,
        "kl_divergence": total_kl / max(total_tokens, 1),
        "n_eval_texts": len(eval_texts),
        "n_eval_pred_tokens": total_tokens,
        "ce_kl_batch_size": batch_size,
    }


# ──────────────────────────── Orchestration ──────────────────────────────────


def run_structural_evaluation(
    activations: torch.Tensor,
    reconstructed: torch.Tensor,
    latents: torch.Tensor,
    attention_mask: torch.Tensor,
    ce_kl_results: dict[str, float | int] | None = None,
    output_dir: str | Path = "outputs/sae_eval",
) -> dict[str, Any]:
    """Run all structural metrics and save results.

    Args:
        activations: Original activations [N, T, d_model].
        reconstructed: SAE-reconstructed activations [N, T, d_model].
        latents: SAE latent activations [N, T, d_sae].
        attention_mask: [N, T] binary mask.
        ce_kl_results: Pre-computed CE/KL results (optional).
        output_dir: Directory for saving results.

    Returns:
        Dictionary of all structural metrics.
    """
    print("\n=== Structural Evaluation ===")

    mse = compute_mse(activations, reconstructed, attention_mask)
    print(f"  MSE:               {mse:.6f}")

    cosine = compute_cosine_similarity(activations, reconstructed, attention_mask)
    print(f"  Cosine Similarity: {cosine:.6f}")

    ev_result = compute_explained_variance(activations, reconstructed, attention_mask)
    print(f"  Explained Variance:{ev_result['explained_variance']:.6f}")
    print(f"  FVU:               {ev_result['fvu']:.6f}")

    l0_result = compute_l0_sparsity(latents, attention_mask)
    print(f"  L0 Mean:           {l0_result['l0_mean']:.1f}")
    print(f"  L0 Std:            {l0_result['l0_std']:.1f}")

    ff_result = compute_firing_frequency(latents, attention_mask)
    print(f"  Dead Features:     {ff_result['dead_count']} / {ff_result['dead_count'] + ff_result['alive_count']}"
          f" ({ff_result['dead_ratio']:.2%})")
    print(f"  Alive Features:    {ff_result['alive_count']}")

    metrics = {
        "mse": mse,
        "cosine_similarity": cosine,
        **ev_result,
        **l0_result,
        "dead_count": ff_result["dead_count"],
        "dead_ratio": ff_result["dead_ratio"],
        "alive_count": ff_result["alive_count"],
        "top10_freq_indices": ff_result["top10_freq_indices"],
    }

    if ce_kl_results is not None:
        metrics.update(ce_kl_results)
        print(f"  CE Loss (orig):    {ce_kl_results['ce_loss_orig']:.4f}")
        print(f"  CE Loss (SAE):     {ce_kl_results['ce_loss_sae']:.4f}")
        print(f"  CE Loss Delta:     {ce_kl_results['ce_loss_delta']:.4f}")
        print(f"  KL Divergence:     {ce_kl_results['kl_divergence']:.6f}")

    # Save to JSON
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_path = out_path / "metrics_structural.json"

    serializable = {
        k: v for k, v in metrics.items()
        if not isinstance(v, torch.Tensor)
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    if ce_kl_results is not None:
        ce_kl_path = out_path / "metrics_ce_kl.json"
        ce_kl_serializable = {
            k: v for k, v in ce_kl_results.items()
            if not isinstance(v, torch.Tensor)
        }
        with open(ce_kl_path, "w", encoding="utf-8") as f:
            json.dump(ce_kl_serializable, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved to {save_path}")
    return metrics
