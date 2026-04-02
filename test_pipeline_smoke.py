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
    details = sae.forward_with_details(x)

    assert x_hat.shape == x.shape, f"x_hat shape {x_hat.shape} != input shape {x.shape}"
    assert latents.shape == (4, 10, d_sae), f"latents shape {latents.shape} wrong"
    assert (latents >= 0).all(), "JumpReLU should produce non-negative latents"
    assert details["reconstructed_raw"].shape == x.shape
    assert details["reconstructed_normalized"].shape == x.shape
    assert details["input_normalized"].shape == x.shape
    assert not torch.allclose(
        details["reconstructed_raw"],
        details["reconstructed_normalized"],
    ), "raw-space reconstruction should differ from normalized-space reconstruction"

    print("  PASS test_sae_forward")


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

    print("  PASS test_sae_dtype_alignment")


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

    # Sum aggregation
    result_sum = aggregate_to_utterance(latents, mask, method="sum")
    expected_sum = torch.tensor([[4.0, 2.5, 1.0], [0.5, 1.0, 2.0]])
    assert torch.allclose(result_sum, expected_sum, atol=1e-5), \
        f"Sum aggregation: {result_sum} != {expected_sum}"

    # Binarized sum
    result_bin = aggregate_to_utterance(latents, mask, method="binarized_sum", binarized_threshold=0.0)
    expected_bin = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert torch.allclose(result_bin, expected_bin, atol=1e-5), \
        f"Binarized sum aggregation: {result_bin} != {expected_bin}"

    # Last-token pooling
    result_last = aggregate_to_utterance(latents, mask, method="last_token")
    expected_last = torch.tensor([[3.0, 0.5, 1.0], [0.5, 1.0, 2.0]])
    assert torch.allclose(result_last, expected_last, atol=1e-5), \
        f"Last-token aggregation: {result_last} != {expected_last}"

    # Weighted mean pooling (later tokens get larger weights)
    result_weighted = aggregate_to_utterance(latents, mask, method="weighted_mean")
    expected_weighted = torch.tensor([
        [(1.0 * 1 + 3.0 * 2) / 3, (2.0 * 1 + 0.5 * 2) / 3, (0.0 * 1 + 1.0 * 2) / 3],
        [0.5, 1.0, 2.0],
    ])
    assert torch.allclose(result_weighted, expected_weighted, atol=1e-5), \
        f"Weighted-mean aggregation: {result_weighted} != {expected_weighted}"

    print("  PASS test_aggregation")


def test_pooling_scope_mask():
    """Therapist-span pooling mask should only include therapist tokens."""
    from nlp_re_base.activations import _build_pooling_mask

    class DummyTokenizer:
        def __call__(
            self,
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_offsets_mapping=False,
            return_tensors=None,
        ):
            offsets = []
            for text in texts:
                rows = []
                for idx, chunk in enumerate(text.split(" ")):
                    start = text.find(chunk, rows[-1][1] if rows else 0)
                    end = start + len(chunk)
                    rows.append((start, end))
                offsets.append(rows)
            max_len = max(len(row) for row in offsets)
            padded = []
            for row in offsets:
                padded.append(row + [(0, 0)] * (max_len - len(row)))
            return {"offset_mapping": torch.tensor(padded, dtype=torch.long)}

    text = "client therapist response"
    record = {
        "therapist_char_start": text.find("therapist"),
        "therapist_char_end": len(text),
    }
    batch_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
    pooling_mask = _build_pooling_mask(
        batch_texts=[text],
        batch_records=[record],
        batch_mask=batch_mask,
        tokenizer=DummyTokenizer(),
        max_seq_len=16,
        pooling_scope="therapist_span",
    )
    expected = torch.tensor([[False, True, True]])
    assert torch.equal(pooling_mask, expected), f"Unexpected therapist-span pooling mask: {pooling_mask}"

    print("  PASS test_pooling_scope_mask")


def test_structural_metrics():
    """Test structural metric computations."""
    from nlp_re_base.eval_structural import (
        compute_mse,
        compute_cosine_similarity,
        compute_explained_variance,
        compute_l0_sparsity,
        compute_firing_frequency,
        OnlineStructuralAccumulator,
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

    acc = OnlineStructuralAccumulator()
    acc.update(z, z_hat, latents, mask)
    online = acc.result()
    assert abs(online["mse"] - mse) < 1e-6, (online["mse"], mse)

    print("  PASS test_structural_metrics")


def test_fvu_constant_offset_regression():
    """Regression: FVU must detect systematic bias (constant offset).

    Old (buggy) Var(residual)/Var(z) implementation would give artificially
    good EV when z_hat = z + constant, because Var(z-z_hat) = Var(constant) = 0.
    The correct SSE/SST definition must show FVU > 0 for any offset.
    """
    from nlp_re_base.eval_structural import (
        compute_explained_variance,
        OnlineStructuralAccumulator,
    )

    torch.manual_seed(42)
    z = torch.randn(20, 8, 64)
    mask = torch.ones(20, 8)
    offset = torch.full_like(z, 5.0)  # Large constant offset
    z_hat_biased = z + offset

    ev_biased = compute_explained_variance(z, z_hat_biased, mask)
    # With a large constant offset, FVU should be very high (bad reconstruction)
    assert ev_biased["fvu"] > 1.0, (
        f"FVU should be > 1 for large constant offset, got {ev_biased['fvu']:.4f}. "
        f"This suggests the old Var(residual)/Var(z) bug is still present."
    )
    assert ev_biased["explained_variance"] < 0.0, (
        f"EV should be negative for large offset, got {ev_biased['explained_variance']:.4f}"
    )

    # Perfect reconstruction should give FVU ~ 0
    ev_perfect = compute_explained_variance(z, z.clone(), mask)
    assert ev_perfect["fvu"] < 1e-6, f"Perfect recon should give FVU~0, got {ev_perfect['fvu']}"

    # Online accumulator should agree with batch version
    latents = torch.randn(20, 8, 256).clamp(min=0)
    acc = OnlineStructuralAccumulator()
    acc.update(z, z_hat_biased, latents, mask)
    online = acc.result()
    assert online["fvu"] > 1.0, (
        f"Online FVU should also detect constant offset, got {online['fvu']:.4f}"
    )

    # Check that online and batch FVU are reasonably close
    batch_fvu = ev_biased["fvu"]
    online_fvu = online["fvu"]
    rel_diff = abs(batch_fvu - online_fvu) / max(abs(batch_fvu), 1e-8)
    assert rel_diff < 0.01, (
        f"Online FVU ({online_fvu:.4f}) and batch FVU ({batch_fvu:.4f}) "
        f"differ by {rel_diff:.2%} — should be < 1%"
    )

    print("  PASS test_fvu_constant_offset_regression")


def test_sae_top_k():
    """Test that top_k truncation limits non-zero latents per token."""
    from nlp_re_base.sae import SparseAutoencoder

    d_model, d_sae = 64, 256
    top_k = 10
    sae = SparseAutoencoder(
        d_model=d_model,
        d_sae=d_sae,
        jump_relu_threshold=0.01,  # Low threshold so many latents pass JumpReLU
        use_decoder_bias=True,
        norm_scale=17.125,
        top_k=top_k,
    )
    sae.eval()

    x = torch.randn(4, 10, d_model)

    # encode() and forward_with_details() should produce identical latents
    latents_enc = sae.encode(x)
    details = sae.forward_with_details(x)
    latents_fwd = details["latents"]

    assert torch.allclose(latents_enc, latents_fwd, atol=1e-6), \
        "encode() and forward_with_details() latents must match"

    # Check that every token has at most top_k non-zero latents
    nonzero_per_token = (latents_enc.abs() > 1e-8).sum(dim=-1)  # [4, 10]
    assert (nonzero_per_token <= top_k).all(), (
        f"Some tokens have > {top_k} non-zero latents: "
        f"max={nonzero_per_token.max().item()}"
    )

    # Without top_k, more latents should be active
    sae_no_topk = SparseAutoencoder(
        d_model=d_model, d_sae=d_sae,
        jump_relu_threshold=0.01,
        norm_scale=17.125,
        top_k=None,
    )
    sae_no_topk.load_state_dict(sae.state_dict())
    sae_no_topk.eval()
    latents_no_topk = sae_no_topk.encode(x)
    nonzero_no_topk = (latents_no_topk.abs() > 1e-8).sum(dim=-1)
    assert (nonzero_no_topk >= nonzero_per_token).all(), \
        "Without top_k, latent count should be >= with top_k"

    print("  PASS test_sae_top_k")


def test_structural_metadata():
    """Test that structural evaluation output includes metadata fields."""
    import tempfile
    import json
    from nlp_re_base.eval_structural import run_structural_evaluation

    z = torch.randn(4, 5, 64)
    z_hat = z + torch.randn_like(z) * 0.1
    latents = torch.randn(4, 5, 256).clamp(min=0)
    mask = torch.ones(4, 5)

    with tempfile.TemporaryDirectory() as tmpdir:
        metrics = run_structural_evaluation(
            activations=z,
            reconstructed=z_hat,
            normalized_activations=z,
            normalized_reconstructed=z_hat,
            latents=latents,
            attention_mask=mask,
            output_dir=tmpdir,
        )

        # Check metadata fields exist
        assert metrics["metric_definition_version"] == 2, \
            f"Expected version 2, got {metrics.get('metric_definition_version')}"
        assert metrics["structural_scope"] == "sample_batches", \
            f"Expected 'sample_batches', got {metrics.get('structural_scope')}"

        # Check space_metrics present
        assert "space_metrics" in metrics
        assert "raw" in metrics["space_metrics"]
        assert "normalized" in metrics["space_metrics"]

        # Verify JSON was saved with metadata
        with open(Path(tmpdir) / "metrics_structural.json") as f:
            saved = json.load(f)
        assert saved["metric_definition_version"] == 2

    print("  PASS test_structural_metadata")


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

    print("  PASS test_univariate_analysis")


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

    print("  PASS test_sparse_probing")


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

    print("  PASS test_maxact")


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

    print("  PASS test_feature_absorption")


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

    print("  PASS test_feature_geometry")


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

    print("  PASS test_tpp")


def test_judge_bundle_export():
    """Test judge bundle export with synthetic utterance features."""
    import tempfile

    import pandas as pd

    from nlp_re_base.ai_re_judge import export_judge_bundle

    np.random.seed(42)
    n, d_sae = 16, 24
    features = np.random.randn(n, d_sae).clip(min=0)
    texts = [f"example {i}" for i in range(n)]
    labels = [1] * 8 + [0] * 8
    records = [
        {
            "predicted_code": "RE" if label == 1 else "NonRE",
            "predicted_subcode": "RES" if label == 1 else "QUC",
            "rationale": f"why {i}",
        }
        for i, label in enumerate(labels)
    ]
    candidate_df = pd.DataFrame(
        {
            "latent_idx": list(range(d_sae)),
            "cohens_d": np.linspace(2.0, 0.1, d_sae),
            "abs_cohens_d": np.linspace(2.0, 0.1, d_sae),
            "auc": np.linspace(0.9, 0.5, d_sae),
            "p_value": np.linspace(0.001, 0.5, d_sae),
            "significant_fdr": [True] * d_sae,
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = export_judge_bundle(
            output_dir=tmpdir,
            candidate_df=candidate_df,
            utterance_features=features,
            texts=texts,
            labels=labels,
            records=records,
            aggregation="max",
            hook_point="blocks.19.hook_resid_post",
            model_name="dummy-model",
            sae_repo_id="dummy/repo",
            sae_subfolder="layer19",
            group_weights={"G1": [1.0], "G5": [0.4, 0.2, 0.2, 0.1, 0.1]},
            top_latents=5,
            top_n=3,
            control_n=2,
        )

        assert (bundle_path / "manifest.json").exists()
        assert (bundle_path / "latent_examples.jsonl").exists()
        assert (bundle_path / "group_examples.json").exists()
        assert (bundle_path / "rubric_snapshot.json").exists()

    print("  PASS test_judge_bundle_export")


def main():
    print("\n══════════════════════════════════════")
    print("  SAE-RE Pipeline Smoke Tests")
    print("══════════════════════════════════════\n")

    tests = [
        ("SAE Forward Pass", test_sae_forward),
        ("SAE dtype Alignment", test_sae_dtype_alignment),
        ("Token→Utterance Aggregation", test_aggregation),
        ("Pooling Scope Mask", test_pooling_scope_mask),
        ("Structural Metrics", test_structural_metrics),
        ("FVU Constant-Offset Regression", test_fvu_constant_offset_regression),
        ("SAE top_k Enforcement", test_sae_top_k),
        ("Structural Metadata", test_structural_metadata),
        ("CE/KL Intervention", test_ce_kl_intervention),
        ("Univariate Analysis", test_univariate_analysis),
        ("Sparse Probing", test_sparse_probing),
        ("MaxAct Cards", test_maxact),
        ("Feature Absorption", test_feature_absorption),
        ("Feature Geometry", test_feature_geometry),
        ("TPP", test_tpp),
        ("Judge Bundle Export", test_judge_bundle_export),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed, {len(tests)} total")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
