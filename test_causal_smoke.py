"""Smoke tests for the causal validation modules (CPU-only, synthetic data)."""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import json
import torch
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace


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

    def test_cond_input_steer_shape(self):
        from causal.intervention import cond_input_steer
        weights = [1.0 / len(self.lids)] * len(self.lids)
        out = cond_input_steer(self.z, self.resid, self.span_mask, self.lids, weights, self.W_dec, 1.0)
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
        self.assertIn("probe_weight_abs", result["ranked_df"].columns)
        self.assertIn("influence_abs", result["ranked_df"].columns)

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


class TestEvaluation(unittest.TestCase):
    def test_eval_text_quality_intervention_fields(self):
        from causal.evaluation import eval_text_quality

        baseline = [
            "I hear how difficult this feels for you",
            "You are trying hard to keep going",
        ]
        intervened = [
            "I hear how difficult this feels",
            "You are trying hard to keep moving forward",
        ]
        result = eval_text_quality(baseline, intervened)
        self.assertIn("mean_content_retention", result)
        self.assertIn("delta_ttr", result)
        self.assertIn("delta_bigram_repetition", result)

    def test_generate_summary_tables_with_side_effects(self):
        from causal.run_experiment import generate_summary_tables

        necessity = {
            "G1": {
                "zero": {
                    "mean_delta_re": -0.2,
                    "mean_delta_nonre": -0.1,
                    "fraction_improved": 0.3,
                }
            }
        }
        sufficiency = {
            "G1": {
                "cond_token": {
                    "lam_1.0": {
                        "mean_delta_re": 0.4,
                        "mean_delta_nonre": 0.1,
                        "fraction_improved": 0.7,
                    }
                }
            }
        }
        group_structure = {
            "cumulative_topk": [{"k": 1, "latent_idx": 1, "mean_delta_re": 0.4}],
            "leave_one_out": [{"latent_idx": 1, "full_effect": 0.4, "loo_effect": 0.1, "delta_loo": 0.3}],
            "add_one_in": [{"k": 1, "latent_idx": 1, "effect": 0.4, "delta_add": 0.4}],
            "synergy": {"synergy_score": 0.1, "interpretation": "near-zero (additive)"},
        }
        side_effects = {
            "groups": {
                "G1": {
                    "mode": "cond_token",
                    "mean_generated_re_logit_delta": 0.2,
                    "quality": {
                        "mean_content_retention": 0.8,
                        "delta_ttr": 0.01,
                        "delta_bigram_repetition": -0.02,
                    },
                }
            },
            "controls": {},
        }

        with TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "summary.md"
            generate_summary_tables(
                necessity,
                sufficiency,
                group_structure,
                side_effects,
                out_path,
            )
            text = out_path.read_text(encoding="utf-8")
            self.assertIn("Table 3: Selectivity / Side Effects", text)
            self.assertIn("G1", text)

    def test_pooling_comparison_report(self):
        from causal.run_experiment import (
            build_pooling_comparison_summary,
            generate_pooling_comparison_report,
        )

        run_payloads = {
            "max": {
                "binarized_threshold": 0.0,
                "selected_groups": {"G1": [1], "G5": [1, 2, 3, 4, 5], "G20": list(range(20))},
                "probe_baseline": {"accuracy": 0.8, "auc": 0.9},
                "necessity": {
                    "G1": {"cond_token": {"mean_delta_re": -0.2}, "zero": {"mean_delta_re": -0.1}},
                    "G5": {"cond_token": {"mean_delta_re": -0.3}, "zero": {"mean_delta_re": -0.2}},
                    "G20": {"cond_token": {"mean_delta_re": -0.4}, "zero": {"mean_delta_re": -0.3}},
                },
                "sufficiency": {
                    "G1": {"cond_token": {"lam_1.0": {"mean_delta_re": 0.2}}},
                    "G5": {"cond_token": {"lam_1.0": {"mean_delta_re": 0.3}}},
                    "G20": {"cond_token": {"lam_1.0": {"mean_delta_re": 0.4}}},
                },
                "side_effects": {
                    "groups": {
                        "G1": {"quality": {"mean_content_retention": 0.8, "delta_bigram_repetition": 0.02}},
                        "G5": {"quality": {"mean_content_retention": 0.79, "delta_bigram_repetition": 0.03}},
                        "G20": {"quality": {"mean_content_retention": 0.78, "delta_bigram_repetition": 0.04}},
                    }
                },
                "group": {"synergy": {"synergy_score": 0.1}},
            },
            "sum": {
                "binarized_threshold": 0.0,
                "selected_groups": {"G1": [7], "G5": [7, 8, 9, 10, 11], "G20": list(range(20, 40))},
                "probe_baseline": {"accuracy": 0.82, "auc": 0.91},
                "necessity": {
                    "G1": {"cond_token": {"mean_delta_re": -0.25}, "zero": {"mean_delta_re": -0.2}},
                    "G5": {"cond_token": {"mean_delta_re": -0.35}, "zero": {"mean_delta_re": -0.3}},
                    "G20": {"cond_token": {"mean_delta_re": -0.45}, "zero": {"mean_delta_re": -0.4}},
                },
                "sufficiency": {
                    "G1": {"cond_token": {"lam_1.0": {"mean_delta_re": 0.25}}},
                    "G5": {"cond_token": {"lam_1.0": {"mean_delta_re": 0.35}}},
                    "G20": {"cond_token": {"lam_1.0": {"mean_delta_re": 0.45}}},
                },
                "side_effects": {
                    "groups": {
                        "G1": {"quality": {"mean_content_retention": 0.83, "delta_bigram_repetition": 0.01}},
                        "G5": {"quality": {"mean_content_retention": 0.82, "delta_bigram_repetition": 0.02}},
                        "G20": {"quality": {"mean_content_retention": 0.81, "delta_bigram_repetition": 0.02}},
                    }
                },
                "group": {"synergy": {"synergy_score": 0.12}},
            },
            "binarized_sum": {
                "binarized_threshold": 0.0,
                "selected_groups": {"G1": [13], "G5": [13, 14, 15, 16, 17], "G20": list(range(40, 60))},
                "probe_baseline": {"accuracy": 0.75, "auc": 0.84},
                "necessity": {
                    "G1": {"cond_token": {"mean_delta_re": -0.1}, "zero": {"mean_delta_re": -0.08}},
                    "G5": {"cond_token": {"mean_delta_re": -0.15}, "zero": {"mean_delta_re": -0.1}},
                    "G20": {"cond_token": {"mean_delta_re": -0.2}, "zero": {"mean_delta_re": -0.12}},
                },
                "sufficiency": {
                    "G1": {"cond_token": {"lam_1.0": {"mean_delta_re": 0.1}}},
                    "G5": {"cond_token": {"lam_1.0": {"mean_delta_re": 0.15}}},
                    "G20": {"cond_token": {"lam_1.0": {"mean_delta_re": 0.2}}},
                },
                "side_effects": {"groups": {}},
                "group": {"synergy": {"synergy_score": 0.05}},
            },
        }

        comparison = build_pooling_comparison_summary(run_payloads)
        self.assertEqual(comparison["best_probe_pooling"], "sum")
        self.assertEqual(comparison["best_sufficiency_pooling"], "sum")

        with TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "pooling.md"
            generate_pooling_comparison_report(comparison, out_path)
            text = out_path.read_text(encoding="utf-8")
            self.assertIn("Pooling Comparison", text)
            self.assertIn("sum", text)


class TestPooling(unittest.TestCase):
    def test_pool_features_variants(self):
        from causal.run_experiment import _pool_features

        z = torch.tensor([
            [[1.0, -1.0], [2.0, 3.0], [10.0, 10.0]],
        ])
        mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)

        max_result = _pool_features(z, mask, method="max")
        sum_result = _pool_features(z, mask, method="sum")
        bin_result = _pool_features(z, mask, method="binarized_sum", threshold=0.0)

        np.testing.assert_allclose(max_result, np.array([[2.0, 3.0]], dtype=np.float32))
        np.testing.assert_allclose(sum_result, np.array([[3.0, 2.0]], dtype=np.float32))
        np.testing.assert_allclose(bin_result, np.array([[1.0, 1.0]], dtype=np.float32))

    def test_extract_utterance_features_passes_pooling(self):
        from causal import run_experiment

        with patch.object(run_experiment, "extract_and_process_streaming") as mock_extract:
            mock_extract.return_value = {"utterance_features": torch.zeros(2, 3)}
            out = run_experiment.extract_utterance_features(
                model=object(),
                tokenizer=object(),
                sae=object(),
                texts=["a", "b"],
                hook_point="blocks.19.hook_resid_post",
                device=torch.device("cpu"),
                aggregation="binarized_sum",
                binarized_threshold=0.0,
            )

        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(mock_extract.call_args.kwargs["aggregation"], "binarized_sum")
        self.assertEqual(mock_extract.call_args.kwargs["binarized_threshold"], 0.0)

    def test_main_passes_records_to_single_pooling_experiment(self):
        from causal import run_experiment
        import pandas as pd

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sae_config_path = tmp / "sae_config.json"
            candidate_csv_path = tmp / "candidate_latents.csv"
            sae_config_path.write_text(
                json.dumps(
                    {
                        "hook_point": "blocks.19.hook_resid_post",
                        "sae_repo_id": "dummy/repo",
                        "sae_subfolder": "dummy-subfolder",
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame({"latent_idx": [1], "abs_cohens_d": [1.0]}).to_csv(
                candidate_csv_path,
                index=False,
            )

            args = SimpleNamespace(
                output_dir=str(tmp / "out"),
                sae_config=str(sae_config_path),
                model_config=None,
                model_dir=None,
                batch_size=2,
                max_seq_len=16,
                device="cpu",
                lambdas=[0.5, 1.0],
                sentence_pooling="max",
                compare_pooling=False,
                binarized_threshold=0.0,
                skip_group_structure=False,
                skip_side_effects=False,
                side_effect_max_samples=4,
                side_effect_max_new_tokens=8,
                side_effect_lambda=1.0,
                n_bootstrap=0,
                data_dir="data/mi_re",
                candidate_csv=str(candidate_csv_path),
            )
            expected_records = [{"unit_text": "a"}, {"unit_text": "b"}]

            with patch.object(run_experiment, "parse_args", return_value=args), \
                 patch.object(run_experiment, "resolve_output_dir", return_value=tmp / "out"), \
                 patch.object(run_experiment, "resolve_repo_path", side_effect=lambda value: Path(value)), \
                 patch.object(run_experiment, "load_local_model_and_tokenizer", return_value=(object(), object(), {})), \
                 patch.object(run_experiment, "load_sae_from_hub", return_value=object()), \
                 patch.object(run_experiment, "build_dataset", return_value=(["a", "b"], [1, 0], expected_records)), \
                 patch.object(run_experiment, "_run_single_pooling_experiment", return_value={}) as mock_run:
                run_experiment.main()

            self.assertEqual(mock_run.call_count, 1)
            self.assertIs(mock_run.call_args.args[6], expected_records)


if __name__ == "__main__":
    unittest.main(verbosity=2)
