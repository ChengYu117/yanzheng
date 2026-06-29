"""Smoke tests for full-representation MISC probes."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _synthetic_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(11)
    n_groups = 30
    samples_per_group = 6
    n = n_groups * samples_per_group
    sae_d = 32
    raw_d = 12
    file_ids = np.repeat([f"session_{idx:02d}" for idx in range(n_groups)], samples_per_group)
    label_df = pd.DataFrame(
        {
            "row_idx": np.arange(n),
            "file_id": file_ids,
            "predicted_code": "OTHER",
        }
    )

    re = np.zeros(n, dtype=int)
    qu = np.zeros(n, dtype=int)
    af = np.zeros(n, dtype=int)
    for group_idx in range(n_groups):
        start = group_idx * samples_per_group
        end = start + samples_per_group
        if group_idx % 3 == 0:
            re[start:end] = 1
            label_df.loc[start:end - 1, "predicted_code"] = "RE"
        elif group_idx % 3 == 1:
            qu[start:end] = 1
            label_df.loc[start:end - 1, "predicted_code"] = "QU"
        else:
            af[start:end] = 1
            label_df.loc[start:end - 1, "predicted_code"] = "AF"
    label_df["RE"] = re
    label_df["QU"] = qu
    label_df["AF"] = af

    sae = rng.normal(0.0, 0.25, size=(n, sae_d)).astype(np.float32)
    sae[:, 0] += re * 2.5
    sae[:, 1] += qu * 2.5
    sae[:, 2] += af * 2.5
    sae[:, 30] = 0.0
    sae[:, 31] = 1.0

    raw = rng.normal(0.0, 0.4, size=(n, raw_d)).astype(np.float32)
    raw[:, 0] += re * 1.0 + qu * 0.2
    raw[:, 1] += qu * 1.0 + af * 0.2
    raw[:, 2] += af * 1.0 + re * 0.2
    return sae, raw, label_df


def test_full_representation_probe_module() -> None:
    from nlp_re_base.full_representation_probe import (
        FullRepresentationProbeConfig,
        run_full_representation_probe,
    )

    sae, raw, label_df = _synthetic_data()
    out = PROJECT_ROOT / "outputs" / "_full_probe_smoke"
    if out.exists():
        shutil.rmtree(out)
    try:
        result = run_full_representation_probe(
            sae_features=sae,
            raw_hidden=raw,
            label_df=label_df,
            output_dir=out,
            labels=["RE", "QU", "AF"],
            config=FullRepresentationProbeConfig(
                labels=("RE", "QU", "AF"),
                folds=3,
                pca_components="full",
                random_state=5,
                max_iter=500,
                verbose=False,
                include_sae_ranked_subspaces=True,
                sae_subspace_top_ns=(1, 5, 10),
                sae_subspace_rankings=("abs_cohens_d", "directional_auc"),
                association_chunk_size=8,
            ),
        )
        summary = result["summary"]
        by_label = result["by_label"]
        representations = set(summary["representation"])
        assert {
            "full_sae_latents",
            "raw_hidden",
            "pca_raw_hidden",
        }.issubset(representations)
        assert {
            "sae_top_abs_cohens_d_n001",
            "sae_top_abs_cohens_d_n005",
            "sae_top_abs_cohens_d_n010",
            "sae_top_directional_auc_n001",
            "sae_top_directional_auc_n005",
            "sae_top_directional_auc_n010",
        }.issubset(representations)
        assert set(by_label["label"]) == {"RE", "QU", "AF"}
        assert (out / "full_probe_by_label.csv").exists()
        assert (out / "full_probe_by_label_summary.csv").exists()
        assert (out / "full_probe_summary.csv").exists()
        assert (out / "ranked_sae_subspace_convergence.csv").exists()
        assert (out / "ranked_sae_subspace_selected_latents.csv").exists()
        assert (out / "ranked_sae_subspace_feature_filter_summary.csv").exists()
        assert (out / "full_probe_summary.json").exists()
        assert (out / "full_probe_report.md").exists()
        assert (out / "full_probe_comparison_report_zh.md").exists()
        sae_row = summary[summary["representation"] == "full_sae_latents"].iloc[0]
        raw_row = summary[summary["representation"] == "raw_hidden"].iloc[0]
        pca_row = summary[summary["representation"] == "pca_raw_hidden"].iloc[0]
        top1_row = summary[summary["representation"] == "sae_top_abs_cohens_d_n001"].iloc[0]
        assert sae_row["macro_auc"] > 0.8
        assert abs(float(raw_row["macro_auc"]) - float(pca_row["macro_auc"])) < 0.02
        assert top1_row["mean_n_features"] == 1.0
        assert top1_row["macro_auc"] > 0.8
        assert not result["convergence"].empty
        assert not result["selected_latents"].empty
        assert not result["feature_filter_summary"].empty
        assert {30, 31}.isdisjoint(set(result["selected_latents"]["latent_idx"].astype(int)))
        assert set(result["summary"].loc[result["summary"]["subspace_ranking"].notna(), "candidate_filter_enabled"].dropna().astype(bool)) == {True}
        fold_rows = result["fold_rows"]
        pca_rows = fold_rows[fold_rows["representation"] == "pca_raw_hidden"]
        assert set(pca_rows["pca_pre_standardized"].dropna().astype(bool)) == {True}
        assert set(pca_rows["pca_post_standardized"].dropna().astype(bool)) == {False}
    finally:
        shutil.rmtree(out, ignore_errors=True)


def main() -> int:
    tests = [test_full_representation_probe_module]
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
