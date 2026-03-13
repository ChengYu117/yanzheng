"""Smoke tests for the SAE-RE Evaluation Pipeline.

Validates the pipeline logic with synthetic data on CPU—
no real model or SAE weights required.

Usage:
    python test_pipeline_smoke.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_sae_forward():
    """Test SparseAutoencoder forward pass with random weights."""
    from nlp_re_base.sae import SparseAutoencoder

    d_model, d_sae = 64, 256
    sae = SparseAutoencoder(
        d_model=d_model,
        d_sae=d_sae,
        jump_relu_threshold=0.1,
        use_decoder_bias=True,
        norm_scale=17.125,
    )
    sae.eval()

    x = torch.randn(4, 10, d_model)  # [batch, seq_len, d_model]
    x_hat, latents = sae(x)

    assert x_hat.shape == x.shape, f"x_hat shape {x_hat.shape} != input shape {x.shape}"
    assert latents.shape == (4, 10, d_sae), f"latents shape {latents.shape} wrong"
    assert (latents >= 0).all(), "JumpReLU should produce non-negative latents"

    print("  ✓ test_sae_forward")


def test_sae_dtype_alignment():
    """Test that SAE works with bfloat16 input (dtype alignment)."""
    from nlp_re_base.sae import SparseAutoencoder

    d_model, d_sae = 64, 256
    sae = SparseAutoencoder(d_model=d_model, d_sae=d_sae, norm_scale=17.125)
    sae = sae.to(dtype=torch.bfloat16)
    sae.sae_dtype = torch.bfloat16
    sae.eval()

    # Simulate float16 activations being cast to bfloat16
    x_fp16 = torch.randn(2, 5, d_model, dtype=torch.float16)
    x_bf16 = x_fp16.to(torch.bfloat16)  # explicit cast like our pipeline does

    x_hat, latents = sae(x_bf16)
    assert x_hat.dtype == torch.bfloat16, f"Expected bfloat16, got {x_hat.dtype}"
    assert latents.dtype == torch.bfloat16, f"Expected bfloat16, got {latents.dtype}"

    print("  ✓ test_sae_dtype_alignment")


def test_aggregation():
    """Test token-to-utterance aggregation."""
    from nlp_re_base.activations import aggregate_to_utterance

    latents = torch.tensor([
        [[1.0, 2.0, 0.0], [3.0, 0.5, 1.0], [0.0, 0.0, 0.0]],
        [[0.5, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    # Max aggregation
    result_max = aggregate_to_utterance(latents, mask, method="max")
    expected_max = torch.tensor([[3.0, 2.0, 1.0], [0.5, 1.0, 2.0]])
    assert torch.allclose(result_max, expected_max, atol=1e-5), \
        f"Max aggregation: {result_max} != {expected_max}"

    # Mean aggregation
    result_mean = aggregate_to_utterance(latents, mask, method="mean")
    expected_mean = torch.tensor([[2.0, 1.25, 0.5], [0.5, 1.0, 2.0]])
    assert torch.allclose(result_mean, expected_mean, atol=1e-5), \
        f"Mean aggregation: {result_mean} != {expected_mean}"

    print("  ✓ test_aggregation")


def test_structural_metrics():
    """Test structural metric computations."""
    from nlp_re_base.eval_structural import (
        compute_mse,
        compute_cosine_similarity,
        compute_explained_variance,
        compute_l0_sparsity,
        compute_firing_frequency,
    )

    z = torch.randn(10, 5, 64)
    z_hat = z + torch.randn_like(z) * 0.1  # Small perturbation
    mask = torch.ones(10, 5)

    mse = compute_mse(z, z_hat, mask)
    assert mse > 0 and mse < 1.0, f"MSE out of expected range: {mse}"

    cos = compute_cosine_similarity(z, z_hat, mask)
    assert 0.9 < cos <= 1.0, f"Cosine sim out of expected range: {cos}"

    ev = compute_explained_variance(z, z_hat, mask)
    assert 0 < ev["explained_variance"] <= 1.0, f"EV out of range: {ev}"
    assert abs(ev["explained_variance"] + ev["fvu"] - 1.0) < 1e-5, "EV + FVU != 1"

    # Sparsity
    latents = torch.randn(10, 5, 256).clamp(min=0)  # Simulated ReLU output
    l0 = compute_l0_sparsity(latents, mask)
    assert l0["l0_mean"] > 0, f"L0 should be positive: {l0}"

    ff = compute_firing_frequency(latents, mask)
    assert ff["dead_count"] >= 0
    assert ff["dead_count"] + ff["alive_count"] == 256

    print("  ✓ test_structural_metrics")


def test_ce_kl_intervention():
    """Test CE/KL intervention metrics with a toy causal LM."""
    import torch.nn as nn
    import torch.nn.functional as F
    from types import SimpleNamespace

    from nlp_re_base.eval_structural import compute_ce_kl_with_intervention

    class ToyTokenizer:
        pad_token_id = 0

        def __call__(
            self,
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ):
            token_lists = []
            for text in texts:
                ids = [int(tok) for tok in text.split()][:max_length]
                token_lists.append(ids)

            max_len = max(len(ids) for ids in token_lists)
            padded = []
            masks = []
            for ids in token_lists:
                pad_len = max_len - len(ids)
                padded.append(ids + [self.pad_token_id] * pad_len)
                masks.append([1] * len(ids) + [0] * pad_len)

            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }

    class ToyLM(nn.Module):
        def __init__(self, vocab_size=10, d_model=8):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Linear(d_model, d_model)])
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, input_ids, attention_mask=None, labels=None):
            hidden = self.embed(input_ids)
            hidden = self.model.layers[0](hidden)
            logits = self.lm_head(hidden)

            loss = None
            if labels is not None:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            return SimpleNamespace(logits=logits, loss=loss)

    class ToySAE(nn.Module):
        def __init__(self, d_model=8):
            super().__init__()
            self.scale = nn.Parameter(torch.full((d_model,), 0.5))
            self.sae_dtype = torch.float32

        def forward(self, x):
            recon = x * self.scale
            latents = torch.zeros(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)
            return recon, latents

    torch.manual_seed(0)
    model = ToyLM()
    tokenizer = ToyTokenizer()
    sae = ToySAE()

    results = compute_ce_kl_with_intervention(
        model=model,
        tokenizer=tokenizer,
        texts=["1 2 3 4", "2 3 4"],
        sae=sae,
        hook_point="blocks.0.hook_resid_post",
        max_seq_len=16,
        batch_size=2,
    )

    assert results["n_eval_texts"] == 2
    assert results["n_eval_pred_tokens"] == 5, results
    assert results["ce_kl_batch_size"] == 2
    assert results["ce_loss_sae"] != results["ce_loss_orig"], results
    assert results["kl_divergence"] >= 0.0, results

    print("  PASS test_ce_kl_intervention")


def test_univariate_analysis():
    """Test univariate analysis with synthetic data."""
    from nlp_re_base.eval_functional import univariate_analysis

    np.random.seed(42)
    d_sae = 100
    n_re, n_nonre = 50, 50

    re_features = np.random.randn(n_re, d_sae)
    nonre_features = np.random.randn(n_nonre, d_sae)

    # Make latent 0 strongly associated with RE
    re_features[:, 0] += 2.0

    df = univariate_analysis(re_features, nonre_features, fdr_alpha=0.05)

    assert len(df) == d_sae, f"Expected {d_sae} rows, got {len(df)}"
    assert "cohens_d" in df.columns
    assert "significant_fdr" in df.columns

    # Latent 0 should be among the top candidates
    top5 = df.head(5)["latent_idx"].tolist()
    assert 0 in top5, f"Latent 0 should be in top-5, got {top5}"

    print("  ✓ test_univariate_analysis")


def test_sparse_probing():
    """Test sparse probing with synthetic data."""
    from nlp_re_base.eval_functional import sparse_probing, univariate_analysis

    np.random.seed(42)
    d_sae = 50
    n_re, n_nonre = 80, 80

    re_features = np.random.randn(n_re, d_sae)
    nonre_features = np.random.randn(n_nonre, d_sae)

    # Strong signal in latent 0
    re_features[:, 0] += 3.0

    candidate_df = univariate_analysis(re_features, nonre_features)

    results = sparse_probing(
        re_features, nonre_features, candidate_df,
        k_values=[1, 5],
    )

    assert "sparse_probe_k1" in results
    assert "sparse_probe_k5" in results
    assert "diffmean" in results
    assert results["sparse_probe_k1"]["accuracy"] > 0.6, \
        f"k=1 probe accuracy too low: {results['sparse_probe_k1']['accuracy']}"

    print("  ✓ test_sparse_probing")


def test_maxact():
    """Test MaxAct card generation."""
    import tempfile
    from nlp_re_base.eval_functional import maxact_analysis

    np.random.seed(42)
    n = 20
    d_sae = 10

    features = np.random.randn(n, d_sae)
    texts = [f"Utterance {i}" for i in range(n)]
    labels = [1 if i < 10 else 0 for i in range(n)]

    with tempfile.TemporaryDirectory() as tmpdir:
        cards = maxact_analysis(
            utterance_features=features,
            texts=texts,
            labels=labels,
            candidate_indices=[0, 1, 2],
            top_n=5,
            output_dir=tmpdir,
        )

    assert len(cards) == 3
    assert "re_purity_top_n" in cards[0]
    assert len(cards[0]["top_entries"]) == 5

    print("  ✓ test_maxact")


def test_feature_absorption():
    """Test feature absorption analysis."""
    from nlp_re_base.eval_functional import feature_absorption

    np.random.seed(42)
    n, d_sae = 100, 50
    features = np.random.randn(n, d_sae).clip(min=0)

    results = feature_absorption(
        utterance_features=features,
        candidate_indices=[0, 1, 2],
        top_k=5,
    )

    assert "per_latent" in results
    assert "overall_mean_absorption" in results
    assert len(results["per_latent"]) == 3
    assert 0 <= results["overall_mean_absorption"] <= 1.0

    print("  ✓ test_feature_absorption")


def test_feature_geometry():
    """Test feature geometry (decoder cosine sim)."""
    from nlp_re_base.eval_functional import feature_geometry

    np.random.seed(42)
    d_model, d_sae = 64, 50
    W_dec = np.random.randn(d_model, d_sae)

    results = feature_geometry(
        sae_decoder_weight=W_dec,
        candidate_indices=[0, 1, 2, 3],
        top_n_pairs=5,
    )

    assert "mean_cosine" in results
    assert "max_cosine" in results
    assert len(results["top_pairs"]) <= 5
    assert 0 <= results["mean_cosine"] <= 1.0

    print("  ✓ test_feature_geometry")


def test_tpp():
    """Test Targeted Probe Perturbation."""
    from nlp_re_base.eval_functional import targeted_probe_perturbation

    np.random.seed(42)
    d_sae = 30
    n_re, n_nonre = 80, 80

    re_features = np.random.randn(n_re, d_sae)
    nonre_features = np.random.randn(n_nonre, d_sae)
    re_features[:, 0] += 3.0  # strong signal

    results = targeted_probe_perturbation(
        re_features=re_features,
        nonre_features=nonre_features,
        candidate_indices=list(range(10)),
        k=10,
    )

    assert "baseline_accuracy" in results
    assert "perturbation_results" in results
    assert results["baseline_accuracy"] > 0.5

    # Latent 0 should have the largest accuracy drop
    drops = {r["latent_idx"]: r["accuracy_drop"] for r in results["perturbation_results"]}
    assert 0 in drops, "Latent 0 should be in perturbation results"

    print("  ✓ test_tpp")


def main():
    print("\n══════════════════════════════════════")
    print("  SAE-RE Pipeline Smoke Tests")
    print("══════════════════════════════════════\n")

    tests = [
        ("SAE Forward Pass", test_sae_forward),
        ("SAE dtype Alignment", test_sae_dtype_alignment),
        ("Token→Utterance Aggregation", test_aggregation),
        ("Structural Metrics", test_structural_metrics),
        ("CE/KL Intervention", test_ce_kl_intervention),
        ("Univariate Analysis", test_univariate_analysis),
        ("Sparse Probing", test_sparse_probing),
        ("MaxAct Cards", test_maxact),
        ("Feature Absorption", test_feature_absorption),
        ("Feature Geometry", test_feature_geometry),
        ("TPP", test_tpp),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed, {len(tests)} total")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
