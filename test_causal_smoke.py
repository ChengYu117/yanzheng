"""Smoke tests for the causal validation modules (CPU-only, synthetic data)."""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import unittest


class TestIntervention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B, self.T, self.d_sae, self.d_model = 2, 6, 32, 16
        self.z     = torch.randn(self.B, self.T, self.d_sae)
        self.resid = torch.randn(self.B, self.T, self.d_model)
        self.span_mask = torch.ones(self.B, self.T, dtype=torch.bool)
        self.span_mask[:, -1] = False  # last token = padding
        self.W_dec = torch.randn(self.d_sae, self.d_model)
        self.lids = [0, 1, 2]

    def test_zero_ablate_shape(self):
        from causal.intervention import zero_ablate
        z_new = zero_ablate(self.z, self.span_mask, self.lids)
        self.assertEqual(z_new.shape, self.z.shape)

    def test_zero_ablate_zeros_targets(self):
        from causal.intervention import zero_ablate
        z_new = zero_ablate(self.z, self.span_mask, self.lids)
        for lid in self.lids:
            self.assertTrue((z_new[:, :-1, lid] == 0).all(),
                            f"latent {lid} not zeroed in span")
        # Padding position should remain unchanged
        torch.testing.assert_close(z_new[:, -1, :], self.z[:, -1, :])

    def test_mean_ablate_shape(self):
        from causal.intervention import mean_ablate
        ref = torch.zeros(self.d_sae)
        z_new = mean_ablate(self.z, self.span_mask, self.lids, ref)
        self.assertEqual(z_new.shape, self.z.shape)

    def test_cond_token_ablate_shape(self):
        from causal.intervention import cond_token_ablate
        z_new = cond_token_ablate(self.z, self.span_mask, self.lids, tau=0.0)
        self.assertEqual(z_new.shape, self.z.shape)

    def test_decode_delta_shape(self):
        from causal.intervention import decode_delta
        delta_z = torch.randn(self.B, self.T, self.d_sae)
        out = decode_delta(delta_z, self.W_dec)
        self.assertEqual(out.shape, (self.B, self.T, self.d_model))

    def test_constant_steer_shape(self):
        from causal.intervention import constant_steer
        weights = [1.0 / len(self.lids)] * len(self.lids)
        out = constant_steer(self.resid, self.span_mask, self.lids, weights, self.W_dec, strength=1.0)
        self.assertEqual(out.shape, self.resid.shape)

    def test_cond_token_steer_shape(self):
        from causal.intervention import cond_token_steer
        weights = [1.0 / len(self.lids)] * len(self.lids)
        out = cond_token_steer(self.z, self.resid, self.span_mask, self.lids, weights, self.W_dec, 1.0)
        self.assertEqual(out.shape, self.resid.shape)

    def test_orthogonal_direction(self):
        from causal.intervention import make_steering_direction, make_orthogonal_direction
        w = [1.0, 1.0, 1.0]
        u = make_steering_direction(self.W_dec, self.lids, w)
        orth = make_orthogonal_direction(u)
        dot = float((u / u.norm()) @ (orth / orth.norm()))
        self.assertAlmostEqual(dot, 0.0, places=4)

    def test_random_direction_shape(self):
        from causal.intervention import make_random_direction
        r = make_random_direction(self.d_model, torch.float32, torch.device("cpu"))
        self.assertEqual(r.shape, (self.d_model,))


class TestSelection(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.d_sae = 100
        self.re_feats    = np.random.randn(50, self.d_sae).astype(np.float32)
        self.nonre_feats = np.random.randn(50, self.d_sae).astype(np.float32)
        # Synthetic candidate_df
        import pandas as pd
        indices = list(range(self.d_sae))
        self.df = pd.DataFrame({
            "latent_idx":   indices,
            "abs_cohens_d": np.abs(np.random.randn(self.d_sae)),
        })

    def test_influence_scores_shape(self):
        from causal.selection import compute_influence_scores
        scores = compute_influence_scores(self.re_feats, self.nonre_feats)
        self.assertEqual(scores.shape, (self.d_sae,))

    def test_rank_latents_returns_groups(self):
        from causal.selection import rank_latents
        result = rank_latents(self.df, self.re_feats, self.nonre_feats, top_k=20)
        self.assertEqual(len(result["G1"]), 1)
        self.assertEqual(len(result["G5"]), 5)
        self.assertEqual(len(result["G20"]), 20)

    def test_make_bottom_k(self):
        from causal.selection import rank_latents, make_bottom_k
        result = rank_latents(self.df, self.re_feats, self.nonre_feats, top_k=20)
        bottom = make_bottom_k(result["ranked_df"], k=5)
        self.assertEqual(len(bottom), 5)

    def test_make_random_control(self):
        from causal.selection import make_random_control
        ctrl = make_random_control(self.df, k=5)
        self.assertEqual(len(ctrl), 5)


class TestData(unittest.TestCase):
    def test_counselor_span_mask(self):
        from causal.data import make_counselor_span_mask
        attn = torch.tensor([[1,1,1,0,0], [1,1,0,0,0]])
        mask = make_counselor_span_mask(attn)
        self.assertEqual(mask.shape, attn.shape)
        self.assertTrue(mask[0, 2])
        self.assertFalse(mask[0, 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
