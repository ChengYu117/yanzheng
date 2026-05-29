"""Smoke tests for Step 6 baseline comparison."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _synthetic_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(7)
    n = 180
    sae_d = 48
    raw_d = 24
    label_df = pd.DataFrame({"row_idx": np.arange(n)})
    label_df["predicted_code"] = "OTHER"

    re = np.zeros(n, dtype=int)
    rec = np.zeros(n, dtype=int)
    qu = np.zeros(n, dtype=int)
    quo = np.zeros(n, dtype=int)
    af = np.zeros(n, dtype=int)
    su = np.zeros(n, dtype=int)

    re[:60] = 1
    rec[20:60] = 1
    qu[60:120] = 1
    quo[60:100] = 1
    af[120:150] = 1
    su[150:180] = 1
    label_df["RE"] = re
    label_df["REC"] = rec
    label_df["QU"] = qu
    label_df["QUO"] = quo
    label_df["AF"] = af
    label_df["SU"] = su
    label_df.loc[re == 1, "predicted_code"] = "RE"
    label_df.loc[qu == 1, "predicted_code"] = "QU"
    label_df.loc[af == 1, "predicted_code"] = "AF"
    label_df.loc[su == 1, "predicted_code"] = "SU"

    sae = rng.gamma(shape=0.4, scale=0.04, size=(n, sae_d)).astype(np.float32)
    sae[re == 1, 0] += 3.0
    sae[rec == 1, 1] += 2.5
    sae[qu == 1, 2] += 3.0
    sae[quo == 1, 3] += 2.5
    sae[af == 1, 4] += 2.5
    sae[su == 1, 5] += 2.5

    raw = rng.normal(0.0, 0.5, size=(n, raw_d)).astype(np.float32)
    raw[:, 0] += re * 1.0 + qu * 0.5 + af * 0.3
    raw[:, 1] += qu * 1.0 + re * 0.4 + su * 0.3
    raw[:, 2] += rec * 0.8 + quo * 0.6
    raw[:, 3] += af * 0.7 + su * 0.7
    return sae, raw, label_df


def test_step6_baseline_comparison_module() -> None:
    from nlp_re_base.baseline_comparison import BaselineComparisonConfig, run_baseline_comparison

    sae, raw, label_df = _synthetic_data()
    with TemporaryDirectory() as tmp:
        out = Path(tmp) / "step6"
        result = run_baseline_comparison(
            sae_features=sae,
            raw_hidden=raw,
            label_df=label_df,
            output_dir=out,
            labels=["RE", "REC", "QU", "QUO", "AF", "SU"],
            config=BaselineComparisonConfig(
                pca_components=12,
                classifier_top_features=8,
                association_chunk_size=16,
                random_state=3,
            ),
        )
        table = result["table4"]
        assert set(table["representation"]) == {"sae_latents", "pca_components", "raw_hidden_dims"}
        assert (out / "table4_baseline_comparison.csv").exists()
        assert (out / "table4_baseline_comparison.md").exists()
        assert (out / "baseline_comparison_report.md").exists()
        assert (out / "sae_latents_label_similarity.csv").exists()
        assert (out / "pca_components_label_similarity.csv").exists()
        assert (out / "raw_hidden_dims_label_similarity.csv").exists()
        assert (out / "figures" / "sae_latents_label_similarity_heatmap.png").exists()
        sae_row = table[table["representation"] == "sae_latents"].iloc[0]
        assert sae_row["mean_label_auc"] > 0.8
        assert result["label_metrics"]["effective_n"].notna().all()


def main() -> int:
    tests = [test_step6_baseline_comparison_module]
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
