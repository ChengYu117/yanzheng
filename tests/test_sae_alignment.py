from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nlp_re_base.sae import SparseAutoencoder


class SAEAlignmentTests(unittest.TestCase):
    def _build_aligned_sae(self) -> SparseAutoencoder:
        sae = SparseAutoencoder(
            d_model=4,
            d_sae=2,
            jump_relu_threshold=3.0,
            use_decoder_bias=True,
            norm_scale=1.0,
            output_norm_scale=1.0,
            sparsity_include_decoder_norm=True,
            runtime_inference_mode="aligned_datasetwise",
        )
        with torch.no_grad():
            sae.W_enc.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                    ]
                )
            )
            sae.b_enc.zero_()
            sae.b_pre.zero_()
            sae.W_dec.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 2.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ]
                )
            )
            sae.b_dec.zero_()
        return sae

    def _build_legacy_sae(self) -> SparseAutoencoder:
        sae = SparseAutoencoder(
            d_model=4,
            d_sae=2,
            jump_relu_threshold=3.0,
            use_decoder_bias=True,
            norm_scale=1.0,
            output_norm_scale=1.0,
            sparsity_include_decoder_norm=True,
            runtime_inference_mode="legacy",
        )
        with torch.no_grad():
            sae.W_enc.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                    ]
                )
            )
            sae.b_enc.zero_()
            sae.b_pre.zero_()
            sae.W_dec.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 2.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ]
                )
            )
            sae.b_dec.zero_()
        return sae

    def test_aligned_datasetwise_forward_matches_manual_raw_reconstruction(self) -> None:
        sae = self._build_aligned_sae()
        x_raw = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

        x_hat_raw, latents = sae.forward_raw(x_raw)

        expected_latents = torch.tensor([[0.0, 2.0]])
        expected_x_hat_raw = torch.tensor([[0.0, 2.0, 0.0, 0.0]])
        self.assertTrue(torch.allclose(latents, expected_latents, atol=1e-6))
        self.assertTrue(torch.allclose(x_hat_raw, expected_x_hat_raw, atol=1e-6))

        details = sae.forward_with_details(x_raw)
        expected_scale = math.sqrt(4) / 1.0
        self.assertTrue(
            torch.allclose(
                details["input_normalized"],
                x_raw * expected_scale,
                atol=1e-6,
            )
        )

    def test_decoder_norm_changes_activation_in_aligned_mode(self) -> None:
        x_raw = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        aligned = self._build_aligned_sae()
        legacy = self._build_legacy_sae()

        _, aligned_latents = aligned(x_raw)
        _, legacy_latents = legacy(x_raw)

        self.assertGreater(float(aligned_latents[0, 1]), 0.0)
        self.assertTrue(torch.allclose(legacy_latents, torch.zeros_like(legacy_latents), atol=1e-6))

    def test_decode_delta_raw_and_decoder_vectors_raw_are_consistent(self) -> None:
        sae = self._build_aligned_sae()
        delta_z = torch.tensor([[[0.0, 2.0]]])

        delta_h = sae.decode_delta_raw(delta_z)
        expected_delta_h = torch.tensor([[[0.0, 2.0, 0.0, 0.0]]])
        self.assertTrue(torch.allclose(delta_h, expected_delta_h, atol=1e-6))

        decoder_vectors_raw = sae.decoder_vectors_raw([1])
        expected_vector = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        self.assertTrue(torch.allclose(decoder_vectors_raw, expected_vector, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
