"""Smoke tests for full-MISC minimal latent set analysis."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _synthetic_inputs() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(123)
    n = 240
    d = 32
    features = rng.gamma(shape=0.5, scale=0.05, size=(n, d)).astype(np.float32)

    y_res = np.zeros(n, dtype=int)
    y_res[:120] = 1
    # Two complementary positive latents for RES.
    features[:60, 0] += 3.0
    features[60:120, 1] += 3.0
    features[120:, 0] += rng.gamma(shape=0.5, scale=0.02, size=120)
    features[120:, 1] += rng.gamma(shape=0.5, scale=0.02, size=120)

    y_gi = np.zeros(n, dtype=int)
    y_gi[120:] = 1
    features[120:, 10] += 4.0

    label_df = pd.DataFrame(
        {
            "row_idx": np.arange(n),
            "RES": y_res,
            "GI": y_gi,
        }
    )

    rows = []
    for label, latent_ids in {"RES": [0, 1, 2, 3, 4], "GI": [10, 11, 12, 13, 14]}.items():
        for rank, lid in enumerate(latent_ids, start=1):
            rows.append(
                {
                    "topk_rank": rank,
                    "label": label,
                    "latent_idx": lid,
                    "cohens_d": 2.0 / rank,
                    "abs_cohens_d": 2.0 / rank,
                    "directional_auc": 0.95 - rank * 0.02,
                    "significant_fdr": True,
                    "precision_lift_at_50": 0.40,
                }
            )
    topk_df = pd.DataFrame(rows)
    return features, label_df, topk_df


def test_misc_minimal_latent_analysis_module() -> None:
    from nlp_re_base.misc_minimal_latents import MinimalLatentConfig, run_misc_minimal_latent_analysis

    features, label_df, topk_df = _synthetic_inputs()
    with TemporaryDirectory() as tmp:
        out = Path(tmp) / "minimal"
        result = run_misc_minimal_latent_analysis(
            features=features,
            label_df=label_df,
            topk_matrix=topk_df,
            output_dir=out,
            labels=["RES", "GI"],
            config=MinimalLatentConfig(top_k=5, loo_k=5, n_bootstrap=3, max_redundancy=0.50),
        )
        summary = result["summary"]
        assert (out / "label_minimal_set_summary.csv").exists()
        assert (out / "cumulative_topk_by_label.csv").exists()
        assert (out / "leave_one_out_by_label.csv").exists()
        assert (out / "add_one_in_by_label.csv").exists()
        assert (out / "minimal_latent_sets.json").exists()
        assert set(summary["label"]) == {"RES", "GI"}

        gi = summary[summary["label"] == "GI"].iloc[0]
        assert gi["full_auc"] > 0.95
        assert int(gi["selected_k"]) <= 2

        res_cumulative = result["cumulative"][result["cumulative"]["label"] == "RES"]
        assert len(res_cumulative) == 5
        assert res_cumulative["auc"].max() > 0.90


def main() -> int:
    tests = [test_misc_minimal_latent_analysis_module]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # pragma: no cover
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
            raise
        else:
            print(f"PASS {test.__name__}")
    print(f"Results: {len(tests) - failed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
