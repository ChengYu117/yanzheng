"""causal/selection.py — Stabilised latent group selection.

Combines Cohen's d rank from the prior run (candidate_latents.csv)
with a GradSAE-style influence score to produce G1/G5/G10/G20.
Bootstrap stability filtering is also provided.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

GROUP_SIZE_MAP = {
    "G1": 1,
    "G5": 5,
    "G10": 10,
    "G20": 20,
}


def _fit_probe_weight_scores(
    candidate_latent_ids: list[int],
    re_features: np.ndarray,
    nonre_features: np.ndarray,
) -> np.ndarray:
    """Fit a lightweight probe on candidate latents and return |w_i| scores."""
    if not candidate_latent_ids:
        return np.zeros(0, dtype=np.float32)

    from nlp_re_base.eval_functional import _fit_torch_probe

    labels = np.concatenate([
        np.ones(len(re_features), dtype=np.float32),
        np.zeros(len(nonre_features), dtype=np.float32),
    ])
    X = np.concatenate([re_features, nonre_features], axis=0)
    X_sub = np.ascontiguousarray(X[:, candidate_latent_ids], dtype=np.float32)
    probe_state = _fit_torch_probe(X_sub, labels, max_steps=300, learning_rate=0.05)
    with torch.no_grad():
        weights = probe_state["model"].weight.detach().cpu().numpy().reshape(-1)
    return np.abs(np.asarray(weights, dtype=np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Influence score  (GradSAE §3.2 proxy)
# ─────────────────────────────────────────────────────────────────────────────

def compute_influence_scores(
    re_features: np.ndarray,    # [N_re, d_sae]
    nonre_features: np.ndarray, # [N_nonre, d_sae]
) -> np.ndarray:
    """Fast proxy influence: mean activation in RE minus mean in NonRE.

    Positive = fires more for RE; negative = fires more for NonRE.
    """
    mu_re    = re_features.mean(axis=0)    # [d_sae]
    mu_nonre = nonre_features.mean(axis=0)
    return mu_re - mu_nonre                # [d_sae]


# ─────────────────────────────────────────────────────────────────────────────
# Combined ranking  (§3.3)
# ─────────────────────────────────────────────────────────────────────────────

def rank_latents(
    candidate_df: pd.DataFrame,
    re_features: np.ndarray,
    nonre_features: np.ndarray,
    top_k: int = 20,
) -> dict[str, list[int]]:
    """Combine |probe weight| rank + influence rank to select G1/G5/G10/G20.

    Returns {'G1': [...], 'G5': [...], 'G10': [...], 'G20': [...]}.
    Only considers latents already in candidate_df (BH-FDR significant).
    """
    df = candidate_df.copy()
    if "significant_fdr" in df.columns:
        df = df[df["significant_fdr"]].copy()
    if df.empty:
        raise ValueError("No significant candidate latents available for ranking.")

    # Influence for all d_sae latents
    infl_all = compute_influence_scores(re_features, nonre_features)

    # Map influence to candidate latents
    candidate_ids = df["latent_idx"].astype(int).tolist()
    infl_cand = infl_all[np.asarray(candidate_ids, dtype=np.int64)]
    probe_weight_abs = _fit_probe_weight_scores(candidate_ids, re_features, nonre_features)

    df["influence"] = infl_cand
    df["influence_abs"] = np.abs(infl_cand)
    df["probe_weight_abs"] = probe_weight_abs

    # Rank both metrics (higher = better)
    df["rank_probe_weight"] = df["probe_weight_abs"].rank(ascending=False, method="min")
    df["rank_influence"] = df["influence_abs"].rank(ascending=False, method="min")
    df["combined_score"] = df["rank_probe_weight"] + df["rank_influence"]

    sort_cols = ["combined_score", "probe_weight_abs", "influence_abs"]
    if "abs_cohens_d" in df.columns:
        sort_cols.append("abs_cohens_d")
    ascending = [True, False, False] + ([False] if "abs_cohens_d" in df.columns else [])
    df_sorted = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    ranked_indices = df_sorted["latent_idx"].tolist()

    return {
        "G1": ranked_indices[:GROUP_SIZE_MAP["G1"]],
        "G5": ranked_indices[:GROUP_SIZE_MAP["G5"]],
        "G10": ranked_indices[:min(GROUP_SIZE_MAP["G10"], top_k)],
        "G20": ranked_indices[:top_k],
        "ranked_df": df_sorted,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap stability  (§3.4)
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_stability(
    re_features: np.ndarray,
    nonre_features: np.ndarray,
    candidate_df: pd.DataFrame,
    n_seeds: int = 20,
    g5_k: int = 5,
    g10_k: int = 10,
    g20_k: int = 20,
    g5_threshold: float = 0.60,
    g10_threshold: float = 0.65,
    g20_threshold: float = 0.70,
) -> dict[str, Any]:
    """Estimate how often each latent appears in G5/G10/G20 across bootstrap resamples.

    Returns stabilised G5 / G10 / G20 index lists.
    """
    n_re    = len(re_features)
    n_nonre = len(nonre_features)

    g5_counts:  dict[int, int] = {}
    g10_counts: dict[int, int] = {}
    g20_counts: dict[int, int] = {}

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        re_idx    = rng.choice(n_re,    size=n_re,    replace=True)
        nonre_idx = rng.choice(n_nonre, size=n_nonre, replace=True)
        re_boot    = re_features[re_idx]
        nonre_boot = nonre_features[nonre_idx]

        result = rank_latents(candidate_df, re_boot, nonre_boot, top_k=g20_k)
        for idx in result["G5"]:
            g5_counts[idx] = g5_counts.get(idx, 0) + 1
        for idx in result["G10"][:g10_k]:
            g10_counts[idx] = g10_counts.get(idx, 0) + 1
        for idx in result["G20"]:
            g20_counts[idx] = g20_counts.get(idx, 0) + 1

    stable_g5  = [i for i, c in g5_counts.items()  if c / n_seeds >= g5_threshold]
    stable_g10 = [i for i, c in g10_counts.items() if c / n_seeds >= g10_threshold]
    stable_g20 = [i for i, c in g20_counts.items() if c / n_seeds >= g20_threshold]

    # Sort by count descending
    stable_g5  = sorted(stable_g5,  key=lambda i: -g5_counts[i])
    stable_g10 = sorted(stable_g10, key=lambda i: -g10_counts[i])
    stable_g20 = sorted(stable_g20, key=lambda i: -g20_counts[i])

    return {
        "stable_G5":   stable_g5,
        "stable_G10":  stable_g10,
        "stable_G20":  stable_g20,
        "g5_counts":   g5_counts,
        "g10_counts":  g10_counts,
        "g20_counts":  g20_counts,
        "n_seeds":     n_seeds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Control groups  (§4.4)
# ─────────────────────────────────────────────────────────────────────────────

def make_bottom_k(
    ranked_df: pd.DataFrame,
    k: int,
) -> list[int]:
    """Return the k worst-ranked (lowest combined_score → highest combined_score)."""
    return ranked_df["latent_idx"].tolist()[-k:][::-1]


def make_random_control(
    candidate_df: pd.DataFrame,
    k: int,
    reference_latents: list[int] | None = None,
    all_features: np.ndarray | None = None,
    seed: int = 0,
) -> list[int]:
    """Random sample of k candidates, optionally matched by activation frequency."""
    rng = random.Random(seed)
    df = candidate_df.copy()
    if "significant_fdr" in df.columns:
        df = df[df["significant_fdr"]].copy()
    pool = [int(x) for x in df["latent_idx"].tolist()]

    if not pool:
        return []

    if all_features is None or not reference_latents:
        return rng.sample(pool, min(k, len(pool)))

    firing_freq = (np.asarray(all_features) > 0).mean(axis=0)
    target_freq = float(np.mean([firing_freq[idx] for idx in reference_latents]))
    candidates = [idx for idx in pool if idx not in set(reference_latents)]
    candidates.sort(key=lambda idx: (abs(float(firing_freq[idx]) - target_freq), rng.random()))
    return candidates[:min(k, len(candidates))]
