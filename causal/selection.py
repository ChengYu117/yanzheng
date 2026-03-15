"""causal/selection.py — Stabilised latent group selection.

Combines Cohen's d rank from the prior run (candidate_latents.csv)
with a GradSAE-style influence score to produce G1/G5/G20.
Bootstrap stability filtering is also provided.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


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
    """Combine |Cohen's d| rank + |influence| rank to select G1/G5/G20.

    Returns {'G1': [...], 'G5': [...], 'G20': [...]}.
    Only considers latents already in candidate_df (BH-FDR significant).
    """
    df = candidate_df.copy()
    d_sae = re_features.shape[1]

    # Influence for all d_sae latents
    infl_all = compute_influence_scores(re_features, nonre_features)

    # Map influence to candidate latents
    infl_cand = infl_all[df["latent_idx"].values]
    df["influence"] = np.abs(infl_cand)

    # Rank both metrics (higher = better)
    n = len(df)
    df["rank_cohens_d"]  = df["abs_cohens_d"].rank(ascending=False)
    df["rank_influence"] = df["influence"].rank(ascending=False)
    df["combined_score"] = df["rank_cohens_d"] + df["rank_influence"]

    df_sorted = df.sort_values("combined_score").reset_index(drop=True)
    ranked_indices = df_sorted["latent_idx"].tolist()

    return {
        "G1":  ranked_indices[:1],
        "G5":  ranked_indices[:5],
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
    g20_k: int = 20,
    g5_threshold: float = 0.60,
    g20_threshold: float = 0.70,
) -> dict[str, Any]:
    """Estimate how often each latent appears in G5/G20 across bootstrap resamples.

    Returns stabilised G5 and G20 index lists.
    """
    n_re    = len(re_features)
    n_nonre = len(nonre_features)

    g5_counts:  dict[int, int] = {}
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
        for idx in result["G20"]:
            g20_counts[idx] = g20_counts.get(idx, 0) + 1

    stable_g5  = [i for i, c in g5_counts.items()  if c / n_seeds >= g5_threshold]
    stable_g20 = [i for i, c in g20_counts.items() if c / n_seeds >= g20_threshold]

    # Sort by count descending
    stable_g5  = sorted(stable_g5,  key=lambda i: -g5_counts[i])
    stable_g20 = sorted(stable_g20, key=lambda i: -g20_counts[i])

    return {
        "stable_G5":   stable_g5,
        "stable_G20":  stable_g20,
        "g5_counts":   g5_counts,
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
    seed: int = 0,
) -> list[int]:
    """Random sample of k latents from significant candidates."""
    rng = random.Random(seed)
    pool = candidate_df["latent_idx"].tolist()
    return rng.sample(pool, min(k, len(pool)))
