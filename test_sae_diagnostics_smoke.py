"""Smoke tests for the new SAE diagnostics helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestDiagnosticsHelpers(unittest.TestCase):
    def test_paper_ev_matches_reference_cases(self):
        from nlp_re_base.eval_structural import compute_explained_variance_paper

        z = torch.tensor([[[1.0, -2.0], [3.0, 4.0]]])
        mask = torch.tensor([[1, 1]])

        perfect = compute_explained_variance_paper(z, z.clone(), mask)
        zero = compute_explained_variance_paper(z, torch.zeros_like(z), mask)
        shifted = compute_explained_variance_paper(
            z,
            torch.tensor([[[1.0, -1.0], [2.0, 4.0]]]),
            mask,
        )

        self.assertAlmostEqual(perfect["explained_variance_paper"], 1.0, places=9)
        self.assertAlmostEqual(zero["explained_variance_paper"], 0.0, places=9)
        self.assertAlmostEqual(
            shifted["explained_variance_paper"],
            1.0 - (2.0 / 30.0),
            places=9,
        )
        self.assertAlmostEqual(
            zero["paper_ev_denominator_energy"],
            float((z ** 2).sum().item()),
            places=9,
        )

    def test_paper_ev_differs_from_centered_ev_on_constant_offset(self):
        from nlp_re_base.eval_structural import (
            compute_explained_variance,
            compute_explained_variance_openmoss,
            compute_explained_variance_paper,
        )

        z = torch.tensor([[[2.0, 2.0], [4.0, 4.0]]])
        z_hat = z + 1.0
        mask = torch.tensor([[1, 1]])

        centered = compute_explained_variance(z, z_hat, mask)
        openmoss = compute_explained_variance_openmoss(z, z_hat, mask)
        paper = compute_explained_variance_paper(z, z_hat, mask)

        self.assertAlmostEqual(
            centered["explained_variance"],
            openmoss["explained_variance_openmoss"],
            places=9,
        )
        self.assertLess(centered["explained_variance"], paper["explained_variance_paper"])

    def test_official_metrics_accumulator_matches_batch_aggregated_semantics(self):
        from nlp_re_base.eval_structural import (
            OfficialMetricsAccumulator,
            compute_explained_variance_openmoss,
        )

        batch1 = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
        batch2 = torch.tensor([[[10.0, 10.0], [11.0, 11.0]]])
        recon1 = batch1.clone()
        recon1[:, 1, :] += 0.5
        recon2 = batch2.clone()
        recon2[:, 1, :] += 0.5
        mask = torch.tensor([[1, 1]])
        latents = torch.tensor([[[1.0, 0.0], [0.5, 0.0]]])

        acc = OfficialMetricsAccumulator()
        acc.update(z=batch1, z_hat=recon1, latents=latents, mask=mask)
        acc.update(z=batch2, z_hat=recon2, latents=latents, mask=mask)
        official = acc.result()

        ev1 = compute_explained_variance_openmoss(batch1, recon1, mask)["explained_variance_openmoss"]
        ev2 = compute_explained_variance_openmoss(batch2, recon2, mask)["explained_variance_openmoss"]
        expected_batch_mean = (ev1 + ev2) / 2.0

        combined = compute_explained_variance_openmoss(
            torch.cat([batch1, batch2], dim=0),
            torch.cat([recon1, recon2], dim=0),
            torch.cat([mask, mask], dim=0),
        )["explained_variance_openmoss"]

        self.assertAlmostEqual(
            official["metrics/explained_variance"],
            expected_batch_mean,
            places=6,
        )
        self.assertNotAlmostEqual(
            official["metrics/explained_variance"],
            combined,
            places=3,
        )

    def test_apply_full_structural_metrics_prefers_official_ev_field(self):
        from nlp_re_base.diagnostics import apply_full_structural_metrics

        out_dir = PROJECT_ROOT / "outputs" / "tmp_test_apply_full_structural_metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        merged = apply_full_structural_metrics(
            {"metric_definition_version": 1},
            raw_full_metrics={
                "explained_variance": -0.7,
                "explained_variance_openmoss": -0.7,
                "explained_variance_openmoss_legacy": -0.6,
                "fvu": 1.7,
                "explained_variance_paper": -0.65,
                "paper_ev_denominator_energy": 123.0,
                "mse": 1.0,
                "cosine_similarity": 0.8,
                "n_tokens": 4,
            },
            norm_full_metrics={},
            official_runtime_metrics={
                "metrics/explained_variance": 0.61,
                "metrics/explained_variance_legacy": 0.59,
                "metrics/l2_norm_error": 0.2,
                "metrics/l2_norm_error_ratio": 0.1,
                "metrics/mean_feature_act": 0.3,
                "metrics/l0": 42.0,
            },
            output_dir=out_dir,
        )
        self.assertAlmostEqual(merged["ev_openmoss_aligned"], 0.61, places=9)
        self.assertAlmostEqual(merged["ev_openmoss_legacy"], 0.59, places=9)
        self.assertAlmostEqual(merged["ev_centered_legacy"], -0.7, places=9)

    def test_reference_adapter_matches_local_sae(self):
        from nlp_re_base.diagnostics import ReferenceCheckpointAdapter
        from nlp_re_base.sae import SparseAutoencoder

        torch.manual_seed(7)
        sae = SparseAutoencoder(
            d_model=8,
            d_sae=16,
            jump_relu_threshold=0.15,
            use_decoder_bias=True,
            norm_scale=3.5,
            top_k=4,
        )
        sae.eval()

        hyperparams = {
            "d_model": 8,
            "d_sae": 16,
            "act_fn": "jumprelu",
            "jump_relu_threshold": 0.15,
            "use_decoder_bias": True,
            "dataset_average_activation_norm": {"in": 3.5},
            "top_k": 4,
        }
        state_dict = {
            "encoder.weight": sae.W_enc.detach().clone(),
            "encoder.bias": sae.b_enc.detach().clone(),
            "decoder.weight": sae.W_dec.detach().clone(),
            "decoder.bias": sae.b_dec.detach().clone(),
            "pre_bias": sae.b_pre.detach().clone(),
        }
        adapter = ReferenceCheckpointAdapter.from_state_dict(
            hyperparams=hyperparams,
            state_dict=state_dict,
            device="cpu",
            dtype=torch.float32,
            checkpoint_topk_semantics="hard",
        )

        x = torch.randn(3, 5, 8)
        local = sae.forward_with_details(x)
        reference = adapter.forward_with_details(x)

        torch.testing.assert_close(
            local["reconstructed_raw"].float(),
            reference["reconstructed_raw"].float(),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            local["reconstructed_normalized"].float(),
            reference["reconstructed_normalized"].float(),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            local["latents"].float(),
            reference["latents"].float(),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_build_comparison_payload_prefers_healthier_paper_distribution(self):
        from nlp_re_base.diagnostics import build_comparison_payload

        payload = build_comparison_payload(
            paper_result={
                "dataset_label": "paper_distribution",
                "sample_stats": {"count": 8},
                "structural_metrics": {
                    "cosine_similarity": 0.95,
                    "explained_variance": -0.40,
                    "ev_openmoss_aligned": -0.40,
                    "explained_variance_paper_raw": 0.66,
                    "explained_variance_paper_normalized": 0.72,
                    "fvu": 0.60,
                    "ce_loss_delta": 0.30,
                    "kl_divergence": 0.25,
                    "dead_ratio": 0.05,
                },
            },
            mire_result={
                "dataset_label": "mi_re_baseline",
                "sample_stats": {"count": 10},
                "structural_metrics": {
                    "cosine_similarity": 0.82,
                    "explained_variance": 0.08,
                    "ev_openmoss_aligned": 0.08,
                    "explained_variance_paper_raw": 0.21,
                    "explained_variance_paper_normalized": 0.31,
                    "fvu": 0.92,
                    "ce_loss_delta": 2.10,
                    "kl_divergence": 2.80,
                    "dead_ratio": 0.53,
                },
            },
        )
        self.assertEqual(
            payload["diagnosis_hint"],
            "paper_distribution_structurally_healthier_than_mi_re",
        )
        self.assertGreater(payload["delta_summary"]["cosine_similarity"], 0.0)
        self.assertLess(payload["delta_summary"]["fvu"], 0.0)
        self.assertEqual(
            payload["literature_alignment"]["alignment_status"],
            "metric_mismatch_supported",
        )
        self.assertLess(payload["delta_summary"]["ev_openmoss_aligned"], 0.0)

    def test_space_resolution_marks_inference_collapse(self):
        from nlp_re_base.diagnostics import (
            build_ev_alignment_report,
            build_metric_provenance,
            infer_space_resolution,
        )

        space_resolution = infer_space_resolution(
            sae=type(
                "DummyAdapter",
                (),
                {
                    "backend_sae": type(
                        "Backend",
                        (),
                        {"cfg": type("Cfg", (), {"norm_activation": "inference"})()},
                    )(),
                },
            )(),
            raw_debug_entries=[
                {
                    "input_scale_factor_mean": 1.0,
                    "input_scale_factor_std": 0.0,
                    "mean_abs_x_minus_x_norm": 0.0,
                }
            ],
            normalized_debug_entries=[{}],
        )
        self.assertTrue(space_resolution["normalized_space_matches_raw"])
        provenance = build_metric_provenance(
            sae=type(
                "DummyAdapter",
                (),
                {
                    "backend_sae": type(
                        "Backend",
                        (),
                        {"cfg": type("Cfg", (), {"norm_activation": "inference"})()},
                    )(),
                },
            )(),
            hook_point="blocks.19.hook_resid_post",
            space_resolution=space_resolution,
        )
        self.assertEqual(provenance["alignment_basis"], "openmoss_official_implementation")
        report = build_ev_alignment_report(
            structural_metrics={
                "ev_openmoss_aligned": -0.7,
                "ev_openmoss_legacy": -0.2,
                "ev_llamascope_paper": -0.6,
                "ev_centered_legacy": -0.7,
            },
            space_resolution=space_resolution,
        )
        self.assertIn("normalized_fields_collapse_to_inference_space", report["diagnosis"])

    def test_build_parity_report_on_identical_outputs(self):
        from nlp_re_base.diagnostics import build_parity_report

        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        latents = torch.tensor(
            [
                [[1.0, 0.0, 0.5], [0.4, 0.2, 0.0], [0.0, 0.0, 0.0]],
                [[0.1, 0.0, 0.7], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        )
        recon = torch.randn(2, 3, 4)
        report = build_parity_report(
            local_details_list=[
                {
                    "reconstructed_raw": recon,
                    "latents": latents,
                }
            ],
            reference_details_list=[
                {
                    "reconstructed_raw": recon.clone(),
                    "latents": latents.clone(),
                }
            ],
            attention_masks=[mask],
            topk_compare=2,
        )
        self.assertTrue(report["parity_passed"])
        self.assertEqual(report["reconstruction_max_abs_diff"], 0.0)
        self.assertTrue(report["topk_exact_match"])
        self.assertEqual(report["topk_compare"], 2)


if __name__ == "__main__":
    unittest.main()
