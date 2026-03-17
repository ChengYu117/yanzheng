"""Functional evaluation metrics for SAE-RE analysis.

Metrics implemented:
  - Univariate analysis: Cohen's d, ROC-AUC with BH-FDR correction
  - Sparse probing: Logistic regression with top-k latents vs baselines
  - MaxAct analysis: Top-activating utterances for candidate latents
  - Feature Absorption: detects redundant encoding
  - Feature Geometry: decoder column cosine similarity
  - TPP (Targeted Probe Perturbation): causal verification
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from .ai_re_judge import export_judge_bundle


# ──────────────────────── Univariate Analysis ────────────────────────────────


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    mean_diff = group1.mean() - group2.mean()
    pooled_var = ((n1 - 1) * group1.var(ddof=1) + (n2 - 1) * group2.var(ddof=1)) / (
        n1 + n2 - 2
    )
    pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 1e-12
    return mean_diff / pooled_std


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Returns:
        Boolean array indicating which tests are significant after correction.
    """
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # BH thresholds: (rank / m) * alpha
    thresholds = (np.arange(1, m + 1) / m) * alpha

    # Find the largest rank where p <= threshold
    rejections = sorted_p <= thresholds
    if not rejections.any():
        return np.zeros(m, dtype=bool)

    max_reject_idx = np.max(np.where(rejections))
    significant = np.zeros(m, dtype=bool)
    significant[sorted_indices[: max_reject_idx + 1]] = True
    return significant


def univariate_analysis(
    re_features: np.ndarray,
    nonre_features: np.ndarray,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Univariate analysis for each SAE latent: Cohen's d, AUC, p-value, BH-FDR.

    Args:
        re_features: [N_re, d_sae] utterance-level features for RE samples.
        nonre_features: [N_nonre, d_sae] utterance-level features for NonRE samples.
        fdr_alpha: FDR significance level.

    Returns:
        DataFrame with columns: latent_idx, cohens_d, auc, p_value, significant_fdr
        Sorted by |cohens_d| descending.
    """
    d_sae = re_features.shape[1]
    results = []

    labels = np.concatenate(
        [np.ones(len(re_features)), np.zeros(len(nonre_features))]
    )
    all_features = np.concatenate([re_features, nonre_features], axis=0)

    print(f"Running univariate analysis on {d_sae} latents...")

    for j in tqdm(range(d_sae), desc="Univariate", unit="latent"):
        re_vals = re_features[:, j]
        nonre_vals = nonre_features[:, j]

        d = cohens_d(re_vals, nonre_vals)

        # t-test
        if re_vals.std() == 0 and nonre_vals.std() == 0:
            p_val = 1.0
        else:
            _, p_val = stats.ttest_ind(re_vals, nonre_vals, equal_var=False)

        # AUC
        feat_col = all_features[:, j]
        if feat_col.std() == 0:
            auc = 0.5
        else:
            try:
                auc = roc_auc_score(labels, feat_col)
            except ValueError:
                auc = 0.5

        results.append({
            "latent_idx": j,
            "cohens_d": d,
            "abs_cohens_d": abs(d),
            "auc": auc,
            "p_value": p_val,
        })

    df = pd.DataFrame(results)

    # BH-FDR correction
    p_values = df["p_value"].values
    df["significant_fdr"] = benjamini_hochberg(p_values, alpha=fdr_alpha)

    # Sort by |Cohen's d|
    df = df.sort_values("abs_cohens_d", ascending=False).reset_index(drop=True)

    n_sig = df["significant_fdr"].sum()
    print(f"  Significant latents (BH-FDR α={fdr_alpha}): {n_sig} / {d_sae}")

    return df


# ──────────────────────── Sparse Probing ─────────────────────────────────────


def _cross_val_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
) -> dict[str, float]:
    """Train and evaluate a logistic regression probe with cross-validation."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.int64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    accs, f1s, aucs = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train = np.ascontiguousarray(X_train, dtype=np.float32)
        X_test = np.ascontiguousarray(X_test, dtype=np.float32)
        y_train = np.ascontiguousarray(y_train, dtype=np.int64)
        y_test = np.ascontiguousarray(y_test, dtype=np.int64)

        probe_state = _fit_torch_probe(X_train, y_train)
        y_pred, y_prob = _predict_torch_probe(probe_state, X_test)

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        try:
            aucs.append(roc_auc_score(y_test, y_prob))
        except ValueError:
            aucs.append(0.5)

    return {
        "accuracy": float(np.mean(accs)),
        "f1": float(np.mean(f1s)),
        "auc": float(np.mean(aucs)),
    }


def _standardize_probe_features(
    X: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize probe features using train-set statistics."""
    X = np.ascontiguousarray(X, dtype=np.float32)

    if mean is None:
        mean = X.mean(axis=0, keepdims=True)
    else:
        mean = np.ascontiguousarray(mean, dtype=np.float32)

    if std is None:
        std = X.std(axis=0, keepdims=True)
    else:
        std = np.ascontiguousarray(std, dtype=np.float32)

    std = np.where(std < 1e-6, 1.0, std).astype(np.float32, copy=False)
    X_std = np.ascontiguousarray((X - mean) / std, dtype=np.float32)
    return X_std, mean, std


def _fit_torch_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_steps: int = 200,
    learning_rate: float = 0.05,
) -> dict[str, Any]:
    """Fit a lightweight logistic probe with torch."""
    X_std, mean, std = _standardize_probe_features(X_train)
    y_train = np.ascontiguousarray(y_train, dtype=np.float32)

    X_tensor = torch.from_numpy(X_std)
    y_tensor = torch.from_numpy(y_train).unsqueeze(1)

    torch.manual_seed(42)
    model = torch.nn.Linear(X_tensor.shape[1], 1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    pos_count = float(y_train.sum())
    neg_count = float(len(y_train) - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_loss = float("inf")
    stale_steps = 0

    model.train()
    for _ in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_tensor)
        loss = loss_fn(logits, y_tensor)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        if best_loss - loss_value > 1e-6:
            best_loss = loss_value
            stale_steps = 0
        else:
            stale_steps += 1

        if stale_steps >= 20:
            break

    model.eval()
    return {
        "model": model,
        "mean": mean,
        "std": std,
    }


def _predict_torch_probe(
    probe_state: dict[str, Any],
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict labels and probabilities with the torch probe."""
    X_std, _, _ = _standardize_probe_features(
        X,
        mean=probe_state["mean"],
        std=probe_state["std"],
    )
    X_tensor = torch.from_numpy(X_std)

    with torch.no_grad():
        logits = probe_state["model"](X_tensor).squeeze(1).detach().cpu().numpy()

    logits = np.clip(logits, -30.0, 30.0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int64)
    return preds, probs


def _extract_probe_weights(probe_state: dict[str, Any]) -> np.ndarray:
    """Extract the fitted linear probe weights as a 1D numpy array."""
    with torch.no_grad():
        weight = probe_state["model"].weight.detach().cpu().numpy().reshape(-1)
    return np.asarray(weight, dtype=np.float32)


def _build_judge_group_weights(
    all_features: np.ndarray,
    labels: np.ndarray,
    candidate_indices: list[int],
) -> dict[str, list[float]]:
    """Fit full-data probes for G1/G5/G20 and export normalized absolute weights."""
    group_weights: dict[str, list[float]] = {}
    for group_name, k in (("G1", 1), ("G5", 5), ("G20", 20)):
        sel_indices = candidate_indices[:k]
        if not sel_indices:
            continue
        X_group = np.ascontiguousarray(all_features[:, sel_indices], dtype=np.float32)
        probe_state = _fit_torch_probe(X_group, labels)
        weights = np.abs(_extract_probe_weights(probe_state))
        total = float(weights.sum())
        if total <= 1e-12:
            group_weights[group_name] = [1.0 / len(sel_indices)] * len(sel_indices)
        else:
            group_weights[group_name] = (weights / total).astype(np.float32).tolist()
    return group_weights


def _safe_l2_norm(
    x: np.ndarray,
    axis: int | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    """Compute an L2 norm without calling numpy.linalg.norm."""
    x64 = np.asarray(x, dtype=np.float64)
    squared = np.sum(x64 * x64, axis=axis, keepdims=keepdims, dtype=np.float64)
    return np.sqrt(squared)


def sparse_probing(
    re_features: np.ndarray,
    nonre_features: np.ndarray,
    candidate_df: pd.DataFrame,
    re_activations: np.ndarray | None = None,
    nonre_activations: np.ndarray | None = None,
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """Sparse probing evaluation with baselines.

    Args:
        re_features: [N_re, d_sae] RE utterance features.
        nonre_features: [N_nonre, d_sae] NonRE utterance features.
        candidate_df: DataFrame from univariate_analysis (with latent_idx, cohens_d).
        re_activations: [N_re, d_model] original activations (for dense probe baseline).
        nonre_activations: [N_nonre, d_model] original activations.
        k_values: List of k values for top-k probing.

    Returns:
        Dictionary of probe results including baselines.
    """
    if k_values is None:
        k_values = [1, 5, 20]

    labels = np.concatenate(
        [np.ones(len(re_features)), np.zeros(len(nonre_features))]
    )
    all_features = np.concatenate([re_features, nonre_features], axis=0)

    results = {}

    # ── Sparse probes (top-k latents by |Cohen's d|) ──
    print("Running sparse probes...")
    top_indices = candidate_df["latent_idx"].values

    for k in k_values:
        sel_indices = top_indices[:k]
        X_sparse = all_features[:, sel_indices]
        probe_result = _cross_val_probe(X_sparse, labels)
        results[f"sparse_probe_k{k}"] = {
            "k": k,
            "latent_indices": sel_indices.tolist(),
            **probe_result,
        }
        print(f"  Sparse Probe k={k}: acc={probe_result['accuracy']:.3f}, "
              f"f1={probe_result['f1']:.3f}, auc={probe_result['auc']:.3f}")

    # ── Dense Probe baseline (original activations) ──
    if re_activations is not None and nonre_activations is not None:
        print("Running dense probe baseline...")
        all_raw = np.concatenate([re_activations, nonre_activations], axis=0)
        dense_result = _cross_val_probe(all_raw, labels)
        results["dense_probe"] = dense_result
        print(f"  Dense Probe: acc={dense_result['accuracy']:.3f}, "
              f"f1={dense_result['f1']:.3f}, auc={dense_result['auc']:.3f}")

    # ── DiffMean baseline ──
    print("Running DiffMean baseline...")
    re_mean = re_features.mean(axis=0)
    nonre_mean = nonre_features.mean(axis=0)
    diff_direction = re_mean - nonre_mean
    diff_norm = float(_safe_l2_norm(diff_direction))
    if diff_norm > 1e-12:
        diff_direction = diff_direction / diff_norm

    # Project onto difference direction → 1D score
    scores = all_features @ diff_direction
    X_diffmean = scores.reshape(-1, 1)
    diffmean_result = _cross_val_probe(X_diffmean, labels)
    results["diffmean"] = diffmean_result
    print(f"  DiffMean: acc={diffmean_result['accuracy']:.3f}, "
          f"f1={diffmean_result['f1']:.3f}, auc={diffmean_result['auc']:.3f}")

    return results


# ──────────────────────── MaxAct Analysis ────────────────────────────────────


def maxact_analysis(
    utterance_features: np.ndarray,
    texts: list[str],
    labels: list[int],
    candidate_indices: list[int],
    top_n: int = 10,
    output_dir: str | Path = "outputs/sae_eval/latent_cards",
) -> list[dict]:
    """Generate MaxAct cards for top candidate latents.

    For each candidate latent, find the top-N highest-activating utterances
    and generate a Markdown report card.

    Args:
        utterance_features: [N_total, d_sae] all utterance features.
        texts: List of all utterance texts.
        labels: List of RE labels (1=RE, 0=NonRE).
        candidate_indices: Latent indices to analyze.
        top_n: Number of top-activating utterances per latent.
        output_dir: Where to save .md card files.

    Returns:
        List of card dicts with summary info.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cards = []

    for rank, lat_idx in enumerate(candidate_indices):
        activations = utterance_features[:, lat_idx]
        top_indices = np.argsort(activations)[::-1][:top_n]

        card_entries = []
        re_count = 0
        for i, idx in enumerate(top_indices):
            is_re = labels[idx] == 1
            if is_re:
                re_count += 1
            card_entries.append({
                "rank": i + 1,
                "utterance_idx": int(idx),
                "text": texts[idx],
                "activation": float(activations[idx]),
                "label": "RE" if is_re else "NonRE",
            })

        re_purity = re_count / top_n if top_n > 0 else 0.0

        card = {
            "latent_idx": int(lat_idx),
            "candidate_rank": rank + 1,
            "re_purity_top_n": re_purity,
            "top_entries": card_entries,
        }
        cards.append(card)

        # Write Markdown card
        md_lines = [
            f"# Latent {lat_idx} (Candidate Rank #{rank + 1})",
            "",
            f"**RE Purity (top-{top_n}):** {re_purity:.0%}",
            "",
            "## Top Activating Utterances",
            "",
            "| Rank | Act. | Label | Text |",
            "|------|------|-------|------|",
        ]
        for entry in card_entries:
            text_short = entry["text"][:100] + ("..." if len(entry["text"]) > 100 else "")
            md_lines.append(
                f"| {entry['rank']} | {entry['activation']:.4f} "
                f"| {entry['label']} | {text_short} |"
            )
        md_lines.append("")

        card_path = out_path / f"latent_{lat_idx:05d}.md"
        with open(card_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

    print(f"  Generated {len(cards)} MaxAct cards in {out_path}")
    return cards


# ──────────────────────── Feature Absorption ─────────────────────────────────


def feature_absorption(
    utterance_features: np.ndarray,
    candidate_indices: list[int],
    top_k: int = 10,
) -> dict[str, Any]:
    """Detect feature absorption: when a target latent is inactive but
    semantically-similar latents fire instead.

    For each candidate latent j, find the top-k most correlated other latents.
    Then measure how often j is inactive while those neighbours are active
    (= absorption events).

    Returns:
        {'per_latent': [{latent_idx, mean_absorption, full_absorption, neighbours}],
         'overall_mean_absorption': float}
    """
    n_samples, d_sae = utterance_features.shape
    results_per_latent = []

    for lat_idx in candidate_indices:
        target_col = utterance_features[:, lat_idx]
        target_active = target_col > 1e-8

        # Find top-k most correlated latents (excluding self)
        correlations = np.zeros(d_sae)
        for j in range(d_sae):
            if j == lat_idx:
                correlations[j] = -np.inf
                continue
            other_col = utterance_features[:, j]
            if target_col.std() < 1e-12 or other_col.std() < 1e-12:
                correlations[j] = 0.0
            else:
                correlations[j] = np.corrcoef(target_col, other_col)[0, 1]

        neighbour_indices = np.argsort(correlations)[::-1][:top_k]

        # Mean absorption: fraction of samples where target is OFF but
        # at least one neighbour is ON
        target_inactive = ~target_active
        if target_inactive.sum() == 0:
            mean_abs = 0.0
            full_abs = 0.0
        else:
            neighbour_active_any = np.zeros(n_samples, dtype=bool)
            for ni in neighbour_indices:
                neighbour_active_any |= (utterance_features[:, ni] > 1e-8)
            absorbed = target_inactive & neighbour_active_any
            mean_abs = float(absorbed.sum() / target_inactive.sum())

            # Full absorption: target NEVER fires, but neighbours do
            full_abs = 1.0 if (target_active.sum() == 0 and neighbour_active_any.any()) else 0.0

        results_per_latent.append({
            "latent_idx": int(lat_idx),
            "mean_absorption": mean_abs,
            "full_absorption": full_abs,
            "top_neighbours": neighbour_indices.tolist(),
        })

    overall = float(np.mean([r["mean_absorption"] for r in results_per_latent]))
    print(f"  Feature Absorption: overall_mean={overall:.3f}")

    return {
        "per_latent": results_per_latent,
        "overall_mean_absorption": overall,
    }


# ──────────────────────── Feature Geometry ───────────────────────────────────


def feature_geometry(
    sae_decoder_weight: np.ndarray,
    candidate_indices: list[int],
    top_n_pairs: int = 20,
) -> dict[str, Any]:
    """Analyse decoder column cosine similarity among candidate latents.

    High cosine similarity between decoder columns suggests
    redundant (non-independent) feature encoding.

    Args:
        sae_decoder_weight: [d_model, d_sae] the W_dec matrix.
        candidate_indices: Latent indices to compare.
        top_n_pairs: Number of most-similar pairs to report.

    Returns:
        {'mean_cosine': float, 'max_cosine': float,
         'top_pairs': [(idx_i, idx_j, cosine)]}
    """
    # Extract decoder columns for candidates: [n_candidates, d_model]
    cols = sae_decoder_weight[:, candidate_indices].T  # [n_cand, d_model]

    # Normalise
    norms = _safe_l2_norm(cols, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    cols_normed = cols / norms

    # Pairwise cosine similarity
    cos_matrix = cols_normed @ cols_normed.T
    n = len(candidate_indices)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((
                int(candidate_indices[i]),
                int(candidate_indices[j]),
                float(cos_matrix[i, j]),
            ))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    cosines = [abs(p[2]) for p in pairs] if pairs else [0.0]
    mean_cos = float(np.mean(cosines))
    max_cos = float(np.max(cosines)) if cosines else 0.0

    print(f"  Feature Geometry: mean_cos={mean_cos:.3f}, max_cos={max_cos:.3f}")

    return {
        "mean_cosine": mean_cos,
        "max_cosine": max_cos,
        "top_pairs": pairs[:top_n_pairs],
    }


# ──────────────────────── TPP (Targeted Probe Perturbation) ──────────────────


def targeted_probe_perturbation(
    re_features: np.ndarray,
    nonre_features: np.ndarray,
    candidate_indices: list[int],
    k: int = 20,
) -> dict[str, Any]:
    """TPP: train a probe, then zero out each candidate latent individually
    and measure the accuracy drop. Large drop = causal evidence.

    Args:
        re_features: [N_re, d_sae]
        nonre_features: [N_nonre, d_sae]
        candidate_indices: Latent indices to perturb.
        k: Number of top latents used in the probe.

    Returns:
        {'baseline_accuracy': float, 'perturbation_results': [...]}
    """
    labels = np.ascontiguousarray(
        np.concatenate([np.ones(len(re_features)), np.zeros(len(nonre_features))]),
        dtype=np.int64,
    )
    all_features = np.ascontiguousarray(
        np.concatenate([re_features, nonre_features], axis=0),
        dtype=np.float32,
    )

    # Use top-k latents for the probe
    probe_indices = candidate_indices[:k]
    X_probe = np.ascontiguousarray(all_features[:, probe_indices], dtype=np.float32)

    # Train a single probe on all data (for perturbation, not CV)
    probe_state = _fit_torch_probe(X_probe, labels)
    baseline_pred, _ = _predict_torch_probe(probe_state, X_probe)
    baseline_acc = float(accuracy_score(labels, baseline_pred))

    perturbation_results = []
    for target_idx in candidate_indices[:min(k, len(candidate_indices))]:
        if target_idx not in probe_indices:
            continue
        local_pos = probe_indices.index(target_idx)

        X_perturbed = X_probe.copy()
        X_perturbed[:, local_pos] = 0.0

        X_perturbed = np.ascontiguousarray(X_perturbed, dtype=np.float32)
        perturbed_pred, _ = _predict_torch_probe(probe_state, X_perturbed)
        perturbed_acc = float(accuracy_score(labels, perturbed_pred))
        acc_drop = baseline_acc - perturbed_acc

        perturbation_results.append({
            "latent_idx": int(target_idx),
            "accuracy_drop": acc_drop,
            "perturbed_accuracy": perturbed_acc,
        })

    perturbation_results.sort(key=lambda x: x["accuracy_drop"], reverse=True)
    print(f"  TPP: baseline_acc={baseline_acc:.3f}, "
          f"max_drop={perturbation_results[0]['accuracy_drop']:.3f}" if perturbation_results else "")

    return {
        "baseline_accuracy": baseline_acc,
        "probe_k": k,
        "perturbation_results": perturbation_results,
    }


def run_functional_evaluation(
    re_features: torch.Tensor | np.ndarray,
    nonre_features: torch.Tensor | np.ndarray,
    all_texts: list[str],
    all_labels: list[int],
    all_records: list[dict[str, Any]] | None = None,
    re_activations: torch.Tensor | np.ndarray | None = None,
    nonre_activations: torch.Tensor | np.ndarray | None = None,
    sae_decoder_weight: torch.Tensor | np.ndarray | None = None,
    fdr_alpha: float = 0.05,
    k_values: list[int] | None = None,
    top_k_candidates: int = 50,
    aggregation: str = "max",
    hook_point: str | None = None,
    model_name: str | None = None,
    sae_repo_id: str | None = None,
    sae_subfolder: str | None = None,
    judge_top_latents: int = 20,
    judge_top_n: int = 10,
    judge_control_n: int = 5,
    output_dir: str | Path = "outputs/sae_eval",
) -> dict[str, Any]:
    """Run complete functional evaluation pipeline.

    Returns:
        Dictionary with univariate results, probe results, MaxAct summaries,
        feature absorption, feature geometry, and TPP results.
    """
    print("\n=== Functional Evaluation ===")

    # Convert to numpy
    if isinstance(re_features, torch.Tensor):
        re_features = re_features.float().numpy()
    if isinstance(nonre_features, torch.Tensor):
        nonre_features = nonre_features.float().numpy()
    if isinstance(re_activations, torch.Tensor):
        re_activations = re_activations.float().numpy()
    if isinstance(nonre_activations, torch.Tensor):
        nonre_activations = nonre_activations.float().numpy()
    if isinstance(sae_decoder_weight, torch.Tensor):
        sae_decoder_weight = sae_decoder_weight.float().numpy()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Univariate analysis ──
    candidate_df = univariate_analysis(re_features, nonre_features, fdr_alpha)

    # Save candidate latents CSV
    csv_path = out_path / "candidate_latents.csv"
    candidate_df.to_csv(csv_path, index=False)
    print(f"  Saved candidate latents to {csv_path}")

    labels = np.ascontiguousarray(
        np.concatenate([np.ones(len(re_features)), np.zeros(len(nonre_features))]),
        dtype=np.int64,
    )
    all_features = np.ascontiguousarray(
        np.concatenate([re_features, nonre_features], axis=0),
        dtype=np.float32,
    )

    # ── Step 2: Sparse probing ──
    probe_results = sparse_probing(
        re_features,
        nonre_features,
        candidate_df,
        re_activations=re_activations,
        nonre_activations=nonre_activations,
        k_values=k_values,
    )

    # ── Step 3: MaxAct analysis ──
    top_candidates = candidate_df["latent_idx"].values[:top_k_candidates].tolist()

    maxact_cards = maxact_analysis(
        utterance_features=all_features,
        texts=all_texts,
        labels=all_labels,
        candidate_indices=top_candidates,
        top_n=10,
        output_dir=out_path / "latent_cards",
    )

    # ── Step 4: Feature Absorption ──
    absorption_results = feature_absorption(
        utterance_features=all_features,
        candidate_indices=top_candidates[:20],  # top-20 for efficiency
    )

    # ── Step 5: Feature Geometry ──
    geometry_results = None
    if sae_decoder_weight is not None:
        geometry_results = feature_geometry(
            sae_decoder_weight=sae_decoder_weight,
            candidate_indices=top_candidates[:20],
        )

    # ── Step 6: TPP ──
    tpp_results = targeted_probe_perturbation(
        re_features=re_features,
        nonre_features=nonre_features,
        candidate_indices=top_candidates,
        k=min(20, len(top_candidates)),
    )

    # ── Assemble results ──
    functional_metrics = {
        "univariate_summary": {
            "total_latents": int(candidate_df.shape[0]),
            "significant_fdr": int(candidate_df["significant_fdr"].sum()),
            "fdr_alpha": fdr_alpha,
            "top10_latents": candidate_df.head(10)[
                ["latent_idx", "cohens_d", "auc", "p_value", "significant_fdr"]
            ].to_dict("records"),
        },
        "probe_results": probe_results,
        "maxact_summary": {
            "n_candidates": len(top_candidates),
            "avg_re_purity": float(
                np.mean([c["re_purity_top_n"] for c in maxact_cards])
            ),
        },
        "feature_absorption": absorption_results,
        "tpp": tpp_results,
    }
    if geometry_results is not None:
        functional_metrics["feature_geometry"] = geometry_results

    group_weights = _build_judge_group_weights(
        all_features=all_features,
        labels=labels,
        candidate_indices=top_candidates,
    )

    # Save functional metrics JSON
    json_path = out_path / "metrics_functional.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(functional_metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved functional metrics to {json_path}")

    bundle_path = export_judge_bundle(
        output_dir=out_path,
        candidate_df=candidate_df,
        utterance_features=all_features,
        texts=all_texts,
        labels=all_labels,
        records=all_records,
        aggregation=aggregation,
        hook_point=hook_point,
        model_name=model_name,
        sae_repo_id=sae_repo_id,
        sae_subfolder=sae_subfolder,
        group_weights=group_weights,
        top_latents=judge_top_latents,
        top_n=judge_top_n,
        control_n=judge_control_n,
    )
    functional_metrics["judge_bundle"] = {
        "path": str(bundle_path),
        "top_latents": judge_top_latents,
        "top_n": judge_top_n,
        "control_n": judge_control_n,
        "group_names": ["G1", "G5", "G20"],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(functional_metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Exported judge bundle to {bundle_path}")

    return functional_metrics
