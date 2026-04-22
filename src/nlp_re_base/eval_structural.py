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

from .model import is_transformer_lens_model
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
        # SSE accumulator: Σ(z - z_hat)² — used for both MSE and FVU
        self._n_tokens: int = 0
        self._mse_sum: float = 0.0
        self._d_model: int | None = None

        # Cosine similarity accumulator
        self._cos_sum: float = 0.0

        # Welford's online variance for z and residual (z - z_hat)
        # shape: (d_model,) — kept in float64 for numerical stability
        self._count_w: int = 0
        self._mean_z: torch.Tensor | None = None     # (d_model,)
        self._M2_z: torch.Tensor | None = None       # (d_model,)
        self._energy_sum: float = 0.0

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
        if self._d_model is None:
            self._d_model = d

        # ── MSE ──
        self._mse_sum += float((r_flat ** 2).sum())
        self._energy_sum += float((z_flat ** 2).sum())
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

        for i in range(m):
            self._count_w += 1
            # Welford update for z (needed for SST = M2_z)
            delta_z = z_flat[i] - self._mean_z
            self._mean_z += delta_z / self._count_w
            delta2_z = z_flat[i] - self._mean_z
            self._M2_z  += delta_z * delta2_z

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

        mse    = self._mse_sum / (n * max(self._d_model or 1, 1))
        cos    = self._cos_sum / n
        l0_mean = self._l0_sum / n
        l0_std  = float(
            (self._l0_sq_sum / n - (self._l0_sum / n) ** 2) ** 0.5
        ) if n > 1 else 0.0

        # FVU = SSE / SST_centered (standard R²-based definition)
        # SSE = _mse_sum = Σ(z - z_hat)² across all tokens and dims
        # SST = M2_z.sum() = Σⱼ Σᵢ (zᵢⱼ - z̄ⱼ)² (Welford final result)
        sse = self._mse_sum
        sst = float(self._M2_z.sum()) if (self._count_w > 1 and self._M2_z is not None) else 0.0

        if sst < 1e-12:
            ev, fvu = (1.0, 0.0) if sse < 1e-12 else (0.0, 1.0)
        else:
            fvu = sse / sst
            ev  = 1.0 - fvu

        total_energy = self._energy_sum
        if total_energy < 1e-12:
            paper_ev = 1.0 if sse < 1e-12 else 0.0
        else:
            paper_ev = 1.0 - (sse / total_energy)

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
            "explained_variance_openmoss": ev,
            "explained_variance_openmoss_legacy": None,
            "fvu":                fvu,
            "explained_variance_paper": paper_ev,
            "paper_ev_denominator_energy": total_energy,
            "l0_mean":            l0_mean,
            "l0_std":             l0_std,
            "n_tokens":           n,
            **freq_result,
        }


class BatchEnergyDebugCollector:
    """Collect per-batch reconstruction-energy diagnostics for space alignment."""

    def __init__(self, *, space_id: str) -> None:
        self.space_id = space_id
        self.entries: list[dict[str, Any]] = []

    def update(
        self,
        *,
        batch_index: int,
        z: torch.Tensor,
        z_hat: torch.Tensor,
        mask: torch.Tensor,
        input_scale_factor: torch.Tensor | None = None,
        normalized_reference: torch.Tensor | None = None,
    ) -> None:
        z_flat, z_hat_flat = _flatten_valid_tokens(z, z_hat, mask)
        if z_flat.numel() == 0:
            return

        residual = z_flat - z_hat_flat
        sse = float((residual ** 2).sum().item())
        z_mean = z_flat.mean(dim=0, keepdim=True)
        sst_centered = float(((z_flat - z_mean) ** 2).sum().item())
        total_energy = float((z_flat ** 2).sum().item())
        token_norm = z_flat.norm(p=2, dim=-1)

        entry: dict[str, Any] = {
            "batch_index": batch_index,
            "space_id": self.space_id,
            "n_tokens": int(z_flat.shape[0]),
            "sse": sse,
            "sst_centered": sst_centered,
            "sum_x2": total_energy,
            "openmoss_explained_variance": (
                None if sst_centered < 1e-12 else 1.0 - (sse / sst_centered)
            ),
            "paper_explained_variance": (
                None if total_energy < 1e-12 else 1.0 - (sse / total_energy)
            ),
            "mean_token_norm": float(token_norm.mean().item()),
            "std_token_norm": float(token_norm.std(unbiased=False).item()),
            "min_token_norm": float(token_norm.min().item()),
            "max_token_norm": float(token_norm.max().item()),
        }

        if input_scale_factor is not None:
            flat_mask = mask.bool().reshape(-1)
            if input_scale_factor.ndim == 0:
                scale_valid = input_scale_factor.expand(int(flat_mask.sum().item())).reshape(-1, 1)
            elif input_scale_factor.ndim == 1:
                scale_valid = input_scale_factor.reshape(-1, 1)
            else:
                flat_scale = input_scale_factor.reshape(-1, input_scale_factor.shape[-1])
                scale_valid = flat_scale[flat_mask]
            entry.update(
                {
                    "input_scale_factor_mean": float(scale_valid.mean().item()),
                    "input_scale_factor_std": float(scale_valid.std(unbiased=False).item()),
                    "input_scale_factor_min": float(scale_valid.min().item()),
                    "input_scale_factor_max": float(scale_valid.max().item()),
                }
            )

        if normalized_reference is not None:
            z_norm_flat, _ = _flatten_valid_tokens(normalized_reference, normalized_reference, mask)
            entry["mean_abs_x_minus_x_norm"] = float((z_flat - z_norm_flat).abs().mean().item())

        self.entries.append(entry)

    def result(self) -> list[dict[str, Any]]:
        return self.entries


class OfficialMetricsAccumulator:
    """Approximate the official lm_saes evaluator aggregation protocol."""

    def __init__(self) -> None:
        self._explained_variance_sum = 0.0
        self._explained_variance_batches = 0
        self._explained_variance_legacy_sum = 0.0
        self._explained_variance_legacy_batches = 0
        self._l2_norm_error_sum = 0.0
        self._l2_norm_error_batches = 0
        self._l2_norm_error_ratio_sum = 0.0
        self._l2_norm_error_ratio_batches = 0
        self._mean_feature_act_sum = 0.0
        self._mean_feature_act_batches = 0
        self._l0_sum = 0.0
        self._l0_weight = 0
        self._act_freq_sum: torch.Tensor | None = None
        self._act_freq_weight = 0

    def update(
        self,
        *,
        z: torch.Tensor,
        z_hat: torch.Tensor,
        latents: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        z_flat, z_hat_flat = _flatten_valid_tokens(z, z_hat, mask)
        lat_flat = latents.reshape(-1, latents.shape[-1]).float()[mask.bool().reshape(-1)]
        if z_flat.numel() == 0:
            return

        label_mean = z_flat.mean(dim=0, keepdim=True)
        per_token_l2_loss = (z_hat_flat - z_flat).pow(2).sum(dim=-1)
        total_variance = (z_flat - label_mean).pow(2).sum(dim=-1)

        l2_loss_mean = float(per_token_l2_loss.mean().item())
        total_variance_mean = float(total_variance.mean().item())
        if total_variance_mean < 1e-12:
            ev = 1.0 if l2_loss_mean < 1e-12 else 0.0
        else:
            ev = 1.0 - (l2_loss_mean / total_variance_mean)
        legacy_ratio = torch.where(
            total_variance > 1e-12,
            1.0 - (per_token_l2_loss / total_variance),
            torch.where(per_token_l2_loss < 1e-12, torch.ones_like(per_token_l2_loss), torch.zeros_like(per_token_l2_loss)),
        )
        ev_legacy = float(legacy_ratio.mean().item())
        self._explained_variance_sum += ev
        self._explained_variance_batches += 1
        self._explained_variance_legacy_sum += ev_legacy
        self._explained_variance_legacy_batches += 1

        l2_norm_error = float((z_hat_flat - z_flat).pow(2).sum(dim=-1).sqrt().mean().item())
        label_norm = float(z_flat.norm(p=2, dim=-1).mean().item())
        self._l2_norm_error_sum += l2_norm_error
        self._l2_norm_error_batches += 1
        self._l2_norm_error_ratio_sum += (l2_norm_error / label_norm) if label_norm > 1e-12 else 0.0
        self._l2_norm_error_ratio_batches += 1

        positive = lat_flat[lat_flat > 0]
        mean_feature_act = float(positive.mean().item()) if positive.numel() > 0 else 0.0
        self._mean_feature_act_sum += mean_feature_act
        self._mean_feature_act_batches += 1

        l0 = float((lat_flat > 0).float().sum(dim=-1).mean().item())
        n_tokens = int(z_flat.shape[0])
        self._l0_sum += l0 * n_tokens
        self._l0_weight += n_tokens

        act_freq_scores = (lat_flat > 0).float().sum(dim=0)
        if self._act_freq_sum is None:
            self._act_freq_sum = torch.zeros_like(act_freq_scores, dtype=torch.float64)
        self._act_freq_sum += act_freq_scores.to(torch.float64)
        self._act_freq_weight += n_tokens

    def result(self) -> dict[str, Any]:
        if self._explained_variance_batches == 0:
            return {}

        freq = (
            self._act_freq_sum / max(self._act_freq_weight, 1)
            if self._act_freq_sum is not None
            else torch.empty(0, dtype=torch.float64)
        )
        return {
            "metrics/explained_variance": self._explained_variance_sum / max(self._explained_variance_batches, 1),
            "metrics/explained_variance_legacy": self._explained_variance_legacy_sum / max(self._explained_variance_legacy_batches, 1),
            "metrics/l2_norm_error": self._l2_norm_error_sum / max(self._l2_norm_error_batches, 1),
            "metrics/l2_norm_error_ratio": self._l2_norm_error_ratio_sum / max(self._l2_norm_error_ratio_batches, 1),
            "metrics/mean_feature_act": self._mean_feature_act_sum / max(self._mean_feature_act_batches, 1),
            "metrics/l0": self._l0_sum / max(self._l0_weight, 1),
            "sparsity/above_1e-1": int((freq > 1e-1).sum().item()),
            "sparsity/above_1e-2": int((freq > 1e-2).sum().item()),
            "sparsity/below_1e-5": int((freq < 1e-5).sum().item()),
            "sparsity/below_1e-6": int((freq < 1e-6).sum().item()),
            "sparsity/below_1e-7": int((freq < 1e-7).sum().item()),
        }


# ──────────────────────────── Token-level metrics ────────────────────────────

def _flatten_valid_tokens(
    z: torch.Tensor,
    z_hat: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten activations to [M, d_model], respecting an optional token mask."""
    if z.dim() == 3:
        if mask is not None:
            flat_mask = mask.bool().reshape(-1)
            z_flat = z.reshape(-1, z.shape[-1])[flat_mask]
            z_hat_flat = z_hat.reshape(-1, z_hat.shape[-1])[flat_mask]
        else:
            z_flat = z.reshape(-1, z.shape[-1])
            z_hat_flat = z_hat.reshape(-1, z_hat.shape[-1])
    else:
        z_flat = z.reshape(-1, z.shape[-1])
        z_hat_flat = z_hat.reshape(-1, z_hat.shape[-1])
    return z_flat, z_hat_flat


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
    """Explained Variance (1 - FVU) and Fraction of Variance Unexplained.

    Uses the SSE/SST_centered definition:
        FVU = Σ(z - z_hat)² / Σ(z - mean(z))²
        EV  = 1 - FVU

    This correctly penalizes systematic bias (constant offset) in
    reconstruction, unlike Var(residual)/Var(z) which would miss it.

    Returns:
        {'explained_variance': float, 'fvu': float}
    """
    z_flat, z_hat_flat = _flatten_valid_tokens(z, z_hat, mask)

    # SSE per dimension: Σᵢ (zᵢⱼ - ẑᵢⱼ)²
    sse_per_dim = ((z_flat - z_hat_flat) ** 2).sum(dim=0)  # [d]
    # SST per dimension: Σᵢ (zᵢⱼ - z̄ⱼ)²
    z_mean = z_flat.mean(dim=0, keepdim=True)
    sst_per_dim = ((z_flat - z_mean) ** 2).sum(dim=0)  # [d]

    total_sst = sst_per_dim.sum().item()
    if total_sst < 1e-12:
        if sse_per_dim.sum().item() < 1e-12:
            return {"explained_variance": 1.0, "fvu": 0.0}
        return {"explained_variance": 0.0, "fvu": 1.0}

    fvu = sse_per_dim.sum().item() / total_sst
    ev = 1.0 - fvu
    return {"explained_variance": ev, "fvu": fvu}


def compute_explained_variance_openmoss(
    z: torch.Tensor,
    z_hat: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """OpenMOSS lm-saes EV implementation.

    Official metric:
      explained_variance = 1 - mean(per_token_l2_loss) / mean(total_variance)

    Official legacy metric:
      explained_variance_legacy = mean(1 - per_token_l2_loss / total_variance)
    """
    z_flat, z_hat_flat = _flatten_valid_tokens(z, z_hat, mask)
    if z_flat.shape[0] == 0:
        return {
            "explained_variance_openmoss": 0.0,
            "explained_variance_openmoss_legacy": 0.0,
        }

    label_mean = z_flat.mean(dim=0, keepdim=True)
    per_token_l2_loss = (z_hat_flat - z_flat).pow(2).sum(dim=-1)
    total_variance = (z_flat - label_mean).pow(2).sum(dim=-1)

    l2_loss_mean = per_token_l2_loss.mean()
    total_variance_mean = total_variance.mean()
    if float(total_variance_mean.item()) < 1e-12:
        explained_variance = 1.0 if float(l2_loss_mean.item()) < 1e-12 else 0.0
    else:
        explained_variance = 1.0 - float((l2_loss_mean / total_variance_mean).item())

    safe_ratio = torch.where(
        total_variance > 1e-12,
        1.0 - (per_token_l2_loss / total_variance),
        torch.where(per_token_l2_loss < 1e-12, torch.ones_like(per_token_l2_loss), torch.zeros_like(per_token_l2_loss)),
    )
    explained_variance_legacy = float(safe_ratio.mean().item())
    return {
        "explained_variance_openmoss": explained_variance,
        "explained_variance_openmoss_legacy": explained_variance_legacy,
    }


def compute_explained_variance_paper(
    z: torch.Tensor,
    z_hat: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Llama Scope paper EV: 1 - sum((z - z_hat)^2) / sum(z^2)."""
    z_flat, z_hat_flat = _flatten_valid_tokens(z, z_hat, mask)
    sse = ((z_flat - z_hat_flat) ** 2).sum().item()
    total_energy = (z_flat ** 2).sum().item()
    if total_energy < 1e-12:
        ev = 1.0 if sse < 1e-12 else 0.0
    else:
        ev = 1.0 - (sse / total_energy)
    return {
        "explained_variance_paper": ev,
        "paper_ev_denominator_energy": total_energy,
    }


def compute_reconstruction_metrics(
    z: torch.Tensor,
    z_hat: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Bundle the reconstruction-space metrics used in structural evaluation."""
    return {
        "mse": compute_mse(z, z_hat, mask),
        "cosine_similarity": compute_cosine_similarity(z, z_hat, mask),
        **compute_explained_variance(z, z_hat, mask),
        **compute_explained_variance_openmoss(z, z_hat, mask),
        **compute_explained_variance_paper(z, z_hat, mask),
    }


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

    device = next(model.parameters()).device
    sae_device = next(sae.parameters()).device
    eval_texts = texts[:max_texts] if max_texts is not None else texts

    total_ce_orig = 0.0
    total_ce_sae = 0.0
    total_ce_ablated = 0.0
    total_kl = 0.0
    total_downstream_ratio = 0.0
    downstream_ratio_batches = 0
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
        if is_transformer_lens_model(model):
            with torch.inference_mode():
                logits_orig, cache = model.run_with_cache(
                    encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    return_type="logits",
                    return_cache_object=False,
                    names_filter=lambda name: name == hook_point,
                )
            original_hidden = cache[hook_point]
        else:
            layer_idx = _parse_hook_point(hook_point)
            target_layer = model.model.layers[layer_idx]
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
            original_hidden = captured["hidden"]

        # ── Pass 2: SAE intervention ──
        sae_dtype = getattr(sae, "sae_dtype", next(sae.parameters()).dtype)
        sae_hidden = original_hidden.to(device=sae_device, dtype=sae_dtype)
        with torch.inference_mode():
            if hasattr(sae, "forward_with_details"):
                sae_outputs = sae.forward_with_details(sae_hidden)
                recon = sae_outputs["reconstructed_raw"]
            else:
                recon, _ = sae(sae_hidden)
        recon = recon.to(device=device, dtype=original_hidden.dtype)

        if is_transformer_lens_model(model):
            def _replace_hook(activation, hook):
                return recon

            with torch.inference_mode():
                logits_sae = model.run_with_hooks(
                    encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    return_type="logits",
                    fwd_hooks=[(hook_point, _replace_hook)],
                )
            def _zero_hook(activation, hook):
                return torch.zeros_like(activation)
            with torch.inference_mode():
                logits_ablated = model.run_with_hooks(
                    encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    return_type="logits",
                    fwd_hooks=[(hook_point, _zero_hook)],
                )
        else:
            def _replace_hook(module, input, output):
                if isinstance(output, tuple):
                    return (recon,) + output[1:]
                return recon

            handle = target_layer.register_forward_hook(_replace_hook)
            with torch.inference_mode():
                out_sae = model(**encoded, labels=labels)
            handle.remove()

            logits_sae = out_sae.logits

            def _zero_hook(module, input, output):
                zero = torch.zeros_like(output[0] if isinstance(output, tuple) else output)
                if isinstance(output, tuple):
                    return (zero,) + output[1:]
                return zero

            handle = target_layer.register_forward_hook(_zero_hook)
            with torch.inference_mode():
                out_ablated = model(**encoded, labels=labels)
            handle.remove()
            logits_ablated = out_ablated.logits

        # Match CE/KL to the next-token positions used by causal LM loss.
        pred_mask = labels[:, 1:] != -100
        n_tokens = int(pred_mask.sum().item())
        if n_tokens == 0:
            continue

        ce_orig = F.cross_entropy(
            logits_orig[:, :-1, :].reshape(-1, logits_orig.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ).item() / n_tokens
        ce_sae = F.cross_entropy(
            logits_sae[:, :-1, :].reshape(-1, logits_sae.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ).item() / n_tokens
        ce_ablated = F.cross_entropy(
            logits_ablated[:, :-1, :].reshape(-1, logits_ablated.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ).item() / n_tokens

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
        total_ce_ablated += ce_ablated * n_tokens
        total_kl += kl_masked.sum().item()
        total_tokens += n_tokens
        denom = ce_ablated - ce_sae
        if abs(denom) > 1e-12:
            total_downstream_ratio += (ce_ablated - ce_orig) / denom
            downstream_ratio_batches += 1

    avg_ce_orig = total_ce_orig / max(total_tokens, 1)
    avg_ce_sae = total_ce_sae / max(total_tokens, 1)
    avg_ce_ablated = total_ce_ablated / max(total_tokens, 1)

    return {
        "ce_loss_orig": avg_ce_orig,
        "ce_loss_sae": avg_ce_sae,
        "ce_loss_delta": avg_ce_sae - avg_ce_orig,
        "delta_lm_loss": avg_ce_sae - avg_ce_orig,
        "downstream_loss_original": avg_ce_orig,
        "downstream_loss_reconstructed": avg_ce_sae,
        "downstream_loss_ablated": avg_ce_ablated,
        "downstream_loss_ratio": (
            total_downstream_ratio / downstream_ratio_batches
            if downstream_ratio_batches > 0
            else None
        ),
        "kl_divergence": total_kl / max(total_tokens, 1),
        "n_eval_texts": len(eval_texts),
        "n_eval_pred_tokens": total_tokens,
        "ce_kl_batch_size": batch_size,
    }


# ──────────────────────────── Orchestration ──────────────────────────────────


def run_structural_evaluation(
    activations: torch.Tensor,
    reconstructed: torch.Tensor,
    normalized_activations: torch.Tensor | None,
    normalized_reconstructed: torch.Tensor | None,
    latents: torch.Tensor,
    attention_mask: torch.Tensor,
    ce_kl_results: dict[str, float | int] | None = None,
    output_dir: str | Path = "outputs/sae_eval",
) -> dict[str, Any]:
    """Run all structural metrics and save results.

    Args:
        activations: Original activations [N, T, d_model].
        reconstructed: SAE raw-space reconstructed activations [N, T, d_model].
        normalized_activations: Normalized SAE inputs [N, T, d_model].
        normalized_reconstructed: Normalized-space reconstructions [N, T, d_model].
        latents: SAE latent activations [N, T, d_sae].
        attention_mask: [N, T] binary mask.
        ce_kl_results: Pre-computed CE/KL results (optional).
        output_dir: Directory for saving results.

    Returns:
        Dictionary of all structural metrics.
    """
    print("\n=== Structural Evaluation ===")

    raw_metrics = compute_reconstruction_metrics(activations, reconstructed, attention_mask)
    print("  Raw-space metrics:")
    print(f"    MSE:               {raw_metrics['mse']:.6f}")
    print(f"    Cosine Similarity: {raw_metrics['cosine_similarity']:.6f}")
    print(f"    EV (centered):     {raw_metrics['explained_variance']:.6f}")
    print(f"    FVU (centered):    {raw_metrics['fvu']:.6f}")
    print(f"    EV (paper):        {raw_metrics['explained_variance_paper']:.6f}")

    normalized_metrics: dict[str, float] | None = None
    if normalized_activations is not None and normalized_reconstructed is not None:
        normalized_metrics = compute_reconstruction_metrics(
            normalized_activations,
            normalized_reconstructed,
            attention_mask,
        )
        print("  Normalized-space metrics:")
        print(f"    MSE:               {normalized_metrics['mse']:.6f}")
        print(f"    Cosine Similarity: {normalized_metrics['cosine_similarity']:.6f}")
        print(f"    EV (centered):     {normalized_metrics['explained_variance']:.6f}")
        print(f"    FVU (centered):    {normalized_metrics['fvu']:.6f}")
        print(f"    EV (paper):        {normalized_metrics['explained_variance_paper']:.6f}")

    l0_result = compute_l0_sparsity(latents, attention_mask)
    print(f"  L0 Mean:           {l0_result['l0_mean']:.1f}")
    print(f"  L0 Std:            {l0_result['l0_std']:.1f}")

    ff_result = compute_firing_frequency(latents, attention_mask)
    print(f"  Dead Features:     {ff_result['dead_count']} / {ff_result['dead_count'] + ff_result['alive_count']}"
          f" ({ff_result['dead_ratio']:.2%})")
    print(f"  Alive Features:    {ff_result['alive_count']}")

    sample_n_tokens = int(attention_mask.sum().item())

    metrics = {
        "metric_definition_version": 3,
        "structural_scope": "sample_batches",
        "n_tokens": sample_n_tokens,
        **raw_metrics,
        "explained_variance_centered_raw": raw_metrics["explained_variance"],
        "fvu_centered_raw": raw_metrics["fvu"],
        "explained_variance_paper_raw": raw_metrics["explained_variance_paper"],
        "paper_ev_denominator_energy_raw": raw_metrics["paper_ev_denominator_energy"],
        **l0_result,
        "dead_count": ff_result["dead_count"],
        "dead_ratio": ff_result["dead_ratio"],
        "alive_count": ff_result["alive_count"],
        "top10_freq_indices": ff_result["top10_freq_indices"],
        "metric_primary": "ev_openmoss_legacy",
        "metric_primary_status": "preferred",
        "metric_primary_note": (
            "Use official legacy EV as the primary literature-facing metric for now. "
            "Negative EV variants are retained only as auxiliary diagnostics."
        ),
        "paper_metric_primary": "ev_openmoss_legacy",
        "paper_metric_formula": "mean(1 - per_token_l2_loss / total_variance)",
        "metric_source_ref": "lm_saes.metrics.ExplainedVarianceMetric.legacy",
        "space_metrics": {
            "raw": {
                **raw_metrics,
                "explained_variance_centered": raw_metrics["explained_variance"],
                "fvu_centered": raw_metrics["fvu"],
                "n_tokens": sample_n_tokens,
            },
            "normalized": (
                {
                    **normalized_metrics,
                    "explained_variance_centered": normalized_metrics["explained_variance"],
                    "fvu_centered": normalized_metrics["fvu"],
                    "n_tokens": sample_n_tokens,
                }
                if normalized_metrics is not None
                else None
            ),
        },
    }

    if normalized_metrics is not None:
        metrics.update(
            {
                "explained_variance_centered_normalized": normalized_metrics["explained_variance"],
                "fvu_centered_normalized": normalized_metrics["fvu"],
                "explained_variance_paper_normalized": normalized_metrics[
                    "explained_variance_paper"
                ],
                "paper_ev_denominator_energy_normalized": normalized_metrics[
                    "paper_ev_denominator_energy"
                ],
            }
        )

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
