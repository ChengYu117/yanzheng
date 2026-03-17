"""causal/intervention.py — Latent-space interventions on the residual stream.

All interventions operate on SAE latent tensors (z) and project the delta
back to residual space via the SAE decoder weight W_dec.

Shapes:
  resid   : [B, T, d_model]
  z       : [B, T, d_sae]
  span_mask: [B, T]  (bool, 1 = counselor token)
  W_dec   : [d_sae, d_model]  (SAE.W_dec — note: some SAEs use [d_model, d_sae])
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _decoder_vectors(W_dec: torch.Tensor, latent_ids: list[int]) -> torch.Tensor:
    """Return decoder column vectors [K, d_model] for the given latent ids.

    Handles both [d_sae, d_model] and [d_model, d_sae] W_dec layouts.
    - [d_sae, d_model]:  rows are latent vectors → W_dec[latent_ids]
    - [d_model, d_sae]:  columns are latent vectors → W_dec[:, latent_ids].T
    For the LXR SAE: W_dec has shape [4096, 32768] = [d_model, d_sae].
    """
    if W_dec.shape[0] < W_dec.shape[1]:
        # [d_model, d_sae] layout — columns are latents
        return W_dec[:, latent_ids].T   # [K, d_model]
    else:
        # [d_sae, d_model] layout — rows are latents
        return W_dec[latent_ids]        # [K, d_model]


def decode_delta(delta_z: torch.Tensor, W_dec: torch.Tensor) -> torch.Tensor:
    """Project latent delta to residual space.

    Args:
        delta_z : [B, T, d_sae]
        W_dec   : [d_model, d_sae] or [d_sae, d_model]
    Returns:
        [B, T, d_model]
    For LXR SAE: W_dec is [4096, 32768] = [d_model, d_sae].
    """
    if W_dec.shape[0] < W_dec.shape[1]:
        # [d_model, d_sae] — latents are columns, need to transpose
        return delta_z @ W_dec.T          # [B,T,d_sae] @ [d_sae,d_model]
    else:
        # [d_sae, d_model] — direct multiply
        return delta_z @ W_dec            # [B,T,d_sae] @ [d_sae,d_model]


# ─────────────────────────────────────────────────────────────────────────────
# Ablations (Necessity)  §4
# ─────────────────────────────────────────────────────────────────────────────

def zero_ablate(
    z: torch.Tensor,            # [B, T, d_sae]
    span_mask: torch.Tensor,    # [B, T] bool
    latent_ids: list[int],
) -> torch.Tensor:
    """Zero out target latents within the counselor span. (§4.1)
    Returns modified z (in-place copy).
    """
    z_new = z.clone()
    mask3d = span_mask.unsqueeze(-1)                    # [B, T, 1]
    latent_mask = torch.zeros(z.shape[-1], dtype=torch.bool, device=z.device)
    latent_mask[latent_ids] = True
    # Zero where span AND target latent
    z_new = z_new.masked_fill(mask3d & latent_mask.unsqueeze(0).unsqueeze(0), 0.0)
    return z_new


def mean_ablate(
    z: torch.Tensor,            # [B, T, d_sae]
    span_mask: torch.Tensor,    # [B, T] bool
    latent_ids: list[int],
    ref_mean: torch.Tensor,     # [d_sae]  — population mean activations
) -> torch.Tensor:
    """Replace target latents with their population mean. (§4.2)"""
    z_new = z.clone()
    mask3d = span_mask.unsqueeze(-1)                    # [B, T, 1]
    for lid in latent_ids:
        # Replace with ref_mean[lid] wherever span_mask is True
        fill_val = ref_mean[lid].item()
        cond = mask3d.squeeze(-1)                       # [B, T] bool
        z_new[:, :, lid] = torch.where(cond, torch.full_like(z_new[:, :, lid], fill_val), z_new[:, :, lid])
    return z_new


def cond_token_ablate(
    z: torch.Tensor,            # [B, T, d_sae]
    span_mask: torch.Tensor,    # [B, T] bool
    latent_ids: list[int],
    tau: float = 0.0,
) -> torch.Tensor:
    """Zero target latents only when they exceed threshold tau. (§4.3)"""
    z_new = z.clone()
    for lid in latent_ids:
        active = (z_new[:, :, lid] > tau) & span_mask  # [B, T]
        z_new[:, :, lid] = z_new[:, :, lid].masked_fill(active, 0.0)
    return z_new


# ─────────────────────────────────────────────────────────────────────────────
# Steerings (Sufficiency)  §5
# ─────────────────────────────────────────────────────────────────────────────

def constant_steer(
    resid: torch.Tensor,        # [B, T, d_model]
    span_mask: torch.Tensor,    # [B, T] bool
    latent_ids: list[int],
    weights: list[float],       # per-latent weight α_i
    W_dec: torch.Tensor,        # SAE decoder weight
    strength: float = 1.0,
) -> torch.Tensor:
    """Add weighted decoder vectors to all counselor span tokens. (§5.1/5.2A)"""
    vecs = _decoder_vectors(W_dec, latent_ids)    # [K, d_model]
    alphas = torch.tensor(weights, dtype=resid.dtype, device=resid.device)
    delta = (alphas.unsqueeze(-1) * vecs).sum(dim=0)  # [d_model]

    resid_new = resid.clone()
    resid_new[span_mask] += strength * delta.unsqueeze(0)
    return resid_new


def cond_token_steer(
    z: torch.Tensor,            # [B, T, d_sae]
    resid: torch.Tensor,        # [B, T, d_model]
    span_mask: torch.Tensor,    # [B, T] bool
    latent_ids: list[int],
    weights: list[float],
    W_dec: torch.Tensor,
    strength: float = 1.0,
    tau: float = 0.0,
) -> torch.Tensor:
    """Steer only tokens where target latent already exceeds threshold. (§5.3C)

    This is the recommended per-token conditional version.
    """
    resid_new = resid.clone()
    vecs = _decoder_vectors(W_dec, latent_ids)    # [K, d_model]

    for j, (lid, w) in enumerate(zip(latent_ids, weights)):
        active = (z[:, :, lid] > tau) & span_mask  # [B, T]
        delta_tok = strength * w * vecs[j]          # [d_model]
        resid_new[active] += delta_tok.unsqueeze(0)

    return resid_new


def cond_input_steer(
    z: torch.Tensor,            # [B, T, d_sae]
    resid: torch.Tensor,        # [B, T, d_model]
    span_mask: torch.Tensor,    # [B, T] bool
    latent_ids: list[int],
    weights: list[float],
    W_dec: torch.Tensor,
    strength: float = 1.0,
    tau: float = 0.0,
) -> torch.Tensor:
    """Steer the full counselor span if any target latent fires in the input."""
    resid_new = resid.clone()
    vecs = _decoder_vectors(W_dec, latent_ids)    # [K, d_model]
    alphas = torch.tensor(weights, dtype=resid.dtype, device=resid.device)
    delta = (alphas.unsqueeze(-1) * vecs.to(resid.device, resid.dtype)).sum(dim=0)  # [d_model]

    any_active = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)
    for lid in latent_ids:
        any_active = any_active | ((z[:, :, lid] > tau) & span_mask).any(dim=1)

    for batch_idx in range(z.shape[0]):
        if any_active[batch_idx]:
            resid_new[batch_idx, span_mask[batch_idx]] += strength * delta.unsqueeze(0)

    return resid_new


# ─────────────────────────────────────────────────────────────────────────────
# Control directions  (§6.1)
# ─────────────────────────────────────────────────────────────────────────────

def make_steering_direction(
    W_dec: torch.Tensor,
    latent_ids: list[int],
    weights: list[float],
) -> torch.Tensor:
    """Compute the weighted steering direction u = Σ α_i v_i. [d_model]"""
    vecs   = _decoder_vectors(W_dec, latent_ids)      # [K, d_model]
    alphas = torch.tensor(weights, dtype=W_dec.dtype, device=W_dec.device)
    return (alphas.unsqueeze(-1) * vecs).sum(dim=0)   # [d_model]


def make_orthogonal_direction(u: torch.Tensor, seed: int = 123) -> torch.Tensor:
    """Random direction orthogonal to u (Gram-Schmidt). [d_model]"""
    torch.manual_seed(seed)
    r = torch.randn_like(u)
    # Project out component along u
    u_hat = F.normalize(u, dim=0)
    r = r - (r @ u_hat) * u_hat
    return F.normalize(r, dim=0) * u.norm()   # same magnitude as u


def make_random_direction(d_model: int, dtype: torch.dtype, device: torch.device, seed: int = 99) -> torch.Tensor:
    """Unit-norm random direction scaled to unit magnitude."""
    torch.manual_seed(seed)
    r = torch.randn(d_model, dtype=dtype, device=device)
    return F.normalize(r, dim=0)


def steer_with_direction(
    resid: torch.Tensor,        # [B, T, d_model]
    span_mask: torch.Tensor,    # [B, T] bool
    direction: torch.Tensor,    # [d_model]
    strength: float = 1.0,
) -> torch.Tensor:
    """Apply a pre-computed direction to the counselor span (for controls)."""
    resid_new = resid.clone()
    resid_new[span_mask] += strength * direction.unsqueeze(0)
    return resid_new
