"""Latent-space interventions on the residual stream."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def decode_delta(delta_z: torch.Tensor, decoder_matrix: torch.Tensor) -> torch.Tensor:
    """Project latent deltas to residual space with an explicit decoder matrix.

    Args:
        delta_z: [B, T, d_sae]
        decoder_matrix: [d_model, d_sae] or [d_sae, d_model]
    """
    if decoder_matrix.shape[0] < decoder_matrix.shape[1]:
        return delta_z @ decoder_matrix.T
    return delta_z @ decoder_matrix


def zero_ablate(
    z: torch.Tensor,
    span_mask: torch.Tensor,
    latent_ids: list[int],
) -> torch.Tensor:
    """Zero out target latents within the intervention span."""
    z_new = z.clone()
    mask3d = span_mask.unsqueeze(-1)
    latent_mask = torch.zeros(z.shape[-1], dtype=torch.bool, device=z.device)
    latent_mask[latent_ids] = True
    z_new = z_new.masked_fill(mask3d & latent_mask.unsqueeze(0).unsqueeze(0), 0.0)
    return z_new


def mean_ablate(
    z: torch.Tensor,
    span_mask: torch.Tensor,
    latent_ids: list[int],
    ref_mean: torch.Tensor,
) -> torch.Tensor:
    """Replace target latents with their reference mean inside the span."""
    z_new = z.clone()
    cond = span_mask
    for lid in latent_ids:
        fill_val = ref_mean[lid].item()
        z_new[:, :, lid] = torch.where(cond, torch.full_like(z_new[:, :, lid], fill_val), z_new[:, :, lid])
    return z_new


def cond_token_ablate(
    z: torch.Tensor,
    span_mask: torch.Tensor,
    latent_ids: list[int],
    tau: float = 0.0,
) -> torch.Tensor:
    """Zero target latents only when they exceed threshold tau inside the span."""
    z_new = z.clone()
    for lid in latent_ids:
        active = (z_new[:, :, lid] > tau) & span_mask
        z_new[:, :, lid] = z_new[:, :, lid].masked_fill(active, 0.0)
    return z_new


def constant_steer(
    resid: torch.Tensor,
    span_mask: torch.Tensor,
    decoder_vectors: torch.Tensor,
    weights: list[float],
    strength: float = 1.0,
) -> torch.Tensor:
    """Add a weighted decoder direction to all span tokens."""
    vecs = decoder_vectors.to(device=resid.device, dtype=resid.dtype)
    alphas = torch.tensor(weights, dtype=resid.dtype, device=resid.device)
    delta = (alphas.unsqueeze(-1) * vecs).sum(dim=0)

    resid_new = resid.clone()
    resid_new[span_mask] += strength * delta.unsqueeze(0)
    return resid_new


def cond_token_steer(
    z: torch.Tensor,
    resid: torch.Tensor,
    span_mask: torch.Tensor,
    latent_ids: list[int],
    decoder_vectors: torch.Tensor,
    weights: list[float],
    strength: float = 1.0,
    tau: float = 0.0,
) -> torch.Tensor:
    """Steer only tokens where the target latent already exceeds tau."""
    resid_new = resid.clone()
    vecs = decoder_vectors.to(device=resid.device, dtype=resid.dtype)

    for j, (lid, weight) in enumerate(zip(latent_ids, weights)):
        active = (z[:, :, lid] > tau) & span_mask
        delta_tok = strength * weight * vecs[j]
        resid_new[active] += delta_tok.unsqueeze(0)

    return resid_new


def cond_input_steer(
    z: torch.Tensor,
    resid: torch.Tensor,
    span_mask: torch.Tensor,
    latent_ids: list[int],
    decoder_vectors: torch.Tensor,
    weights: list[float],
    strength: float = 1.0,
    tau: float = 0.0,
) -> torch.Tensor:
    """Steer the full span if any target latent fires in the input."""
    resid_new = resid.clone()
    vecs = decoder_vectors.to(device=resid.device, dtype=resid.dtype)
    alphas = torch.tensor(weights, dtype=resid.dtype, device=resid.device)
    delta = (alphas.unsqueeze(-1) * vecs).sum(dim=0)

    any_active = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)
    for lid in latent_ids:
        any_active = any_active | ((z[:, :, lid] > tau) & span_mask).any(dim=1)

    for batch_idx in range(z.shape[0]):
        if any_active[batch_idx]:
            resid_new[batch_idx, span_mask[batch_idx]] += strength * delta.unsqueeze(0)

    return resid_new


def make_steering_direction(
    decoder_vectors: torch.Tensor,
    weights: list[float],
) -> torch.Tensor:
    """Compute the weighted steering direction u = sum_i w_i v_i."""
    vecs = decoder_vectors
    alphas = torch.tensor(weights, dtype=vecs.dtype, device=vecs.device)
    return (alphas.unsqueeze(-1) * vecs).sum(dim=0)


def make_orthogonal_direction(u: torch.Tensor, seed: int = 123) -> torch.Tensor:
    """Random direction orthogonal to u (Gram-Schmidt)."""
    torch.manual_seed(seed)
    r = torch.randn_like(u)
    u_hat = F.normalize(u, dim=0)
    r = r - (r @ u_hat) * u_hat
    return F.normalize(r, dim=0) * u.norm()


def make_random_direction(
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 99,
) -> torch.Tensor:
    """Unit-norm random direction scaled to unit magnitude."""
    torch.manual_seed(seed)
    r = torch.randn(d_model, dtype=dtype, device=device)
    return F.normalize(r, dim=0)


def steer_with_direction(
    resid: torch.Tensor,
    span_mask: torch.Tensor,
    direction: torch.Tensor,
    strength: float = 1.0,
) -> torch.Tensor:
    """Apply a pre-computed residual-space direction to the intervention span."""
    resid_new = resid.clone()
    resid_new[span_mask] += strength * direction.unsqueeze(0)
    return resid_new
