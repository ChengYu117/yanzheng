from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.cross_val_framework import (
    build_dedup_group_mapping,
    load_filtered_inputs,
)
from nlp_re_base.cross_quality_val import run_cross_quality_validation
from nlp_re_base.effect_size_ci import run_bootstrap_ci
from nlp_re_base.misc_label_mapping import compute_latent_label_associations
from nlp_re_base.stable_topk_selection import StableTopKSelectionConfig, run_stable_topk_selection
from nlp_re_base.topk_reproducibility import run_topk_reproducibility


def _safe_root() -> Path:
    root = (Path.cwd() / "outputs" / "_smoke_cross_val_experiments").resolve()
    if Path.cwd().resolve() not in root.parents:
        raise RuntimeError(f"Refusing to use smoke path outside repo: {root}")
    return root


def test_cross_val_experiment_stack() -> None:
    root = _safe_root()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    try:
        rng = np.random.default_rng(42)
        n = 120
        d = 32
        labels = ("A", "B", "C")
        label_a = np.array([(i % 4) in (0, 1) for i in range(n)], dtype=bool)
        label_b = np.array([(i % 4) in (1, 2) for i in range(n)], dtype=bool)
        label_c = np.array([(i % 5) == 0 for i in range(n)], dtype=bool)
        features = rng.normal(0, 0.2, size=(n, d)).astype(np.float32)
        features[label_a, 0] += 2.0
        features[label_b, 1] += 1.8
        features[label_c, 2] += 1.5

        label_matrix = pd.DataFrame(
            {
                "row_idx": np.arange(n),
                "record_id": [f"r{i:03d}" for i in range(n)],
                "source_split": ["high" if i < n // 2 else "low" for i in range(n)],
                "source_file": [f"file_{i // 10:02d}.jsonl" for i in range(n)],
                "unit_text": [f"Utterance {i // 2}" for i in range(n)],
                "A": label_a.astype(int),
                "B": label_b.astype(int),
                "C": label_c.astype(int),
            }
        )
        feature_audit = pd.DataFrame(
            {
                "latent_idx": np.arange(d),
                "keep": [True] * 16 + [False] * 16,
            }
        )

        feature_path = root / "features.pt"
        label_path = root / "label_matrix.csv"
        audit_path = root / "feature_filter_audit.csv"
        association_path = root / "latent_label_matrix.csv"
        torch.save({"utterance_features": torch.tensor(features)}, feature_path)
        label_matrix.to_csv(label_path, index=False)
        feature_audit.to_csv(audit_path, index=False)

        dedup = build_dedup_group_mapping(label_matrix)
        assert dedup["dedup_group_id"].nunique() == n // 2
        assert int(dedup["duplicate_text_count"].max()) == 2

        inputs = load_filtered_inputs(
            feature_store_path=feature_path,
            label_matrix_path=label_path,
            feature_filter_audit_path=audit_path,
            labels=labels,
        )
        association, skipped = compute_latent_label_associations(
            inputs.features,
            label_matrix.loc[:, labels].to_numpy(dtype=bool),
            list(labels),
            min_positive=2,
            min_negative=2,
            candidate_latent_indices=inputs.latent_indices,
        )
        assert not skipped
        association.to_csv(association_path, index=False)

        e1 = run_topk_reproducibility(
            inputs,
            output_dir=root / "topk_reproducibility",
            reference_matrix_path=association_path,
            top_k=5,
            top_k_grid=range(1, 6),
            n_repeats=3,
            null_iter=100,
            min_positive=2,
            min_negative=2,
        )
        rank_df = pd.read_csv(root / "topk_reproducibility" / "split_half_rank_correlation.csv")
        jaccard_df = pd.read_csv(root / "topk_reproducibility" / "split_half_top20_jaccard.csv")
        repeated_rank = pd.read_csv(root / "topk_reproducibility" / "repeated_split_rank_correlation.csv")
        repeated_jaccard = pd.read_csv(root / "topk_reproducibility" / "repeated_split_top20_jaccard.csv")
        repeated_summary = pd.read_csv(root / "topk_reproducibility" / "repeated_split_summary.csv")
        grid_summary = pd.read_csv(root / "topk_reproducibility" / "repeated_split_topk_grid_summary.csv")
        inclusion = pd.read_csv(root / "topk_reproducibility" / "topk_inclusion_frequency.csv")
        assert Path(e1["outputs"]["rank_correlation"]).exists()
        assert Path(e1["outputs"]["repeated_summary"]).exists()
        assert Path(e1["outputs"]["repeated_topk_grid_summary"]).exists()
        assert Path(e1["outputs"]["topk_inclusion_frequency"]).exists()
        assert e1["n_repeats"] == 3
        assert set(e1["top_k_grid"]) == {1, 2, 3, 4, 5}
        assert {"A", "B", "C"}.issubset(set(rank_df["label"]))
        assert not jaccard_df.empty
        assert repeated_rank["repeat_index"].nunique() == 3
        assert repeated_jaccard["repeat_index"].nunique() == 3
        assert set(grid_summary["top_k"].astype(int)).issuperset({1, 2, 3, 4, 5})
        assert "observed_jaccard_mean" in repeated_summary.columns
        assert "reference_vs_cv_union_overlap_fraction_mean" in repeated_summary.columns
        assert not inclusion.empty
        injected = inclusion[
            (inclusion["label"] == "A")
            & (inclusion["ranking_metric"] == "cohens_d")
            & (inclusion["top_k"].astype(int) == 1)
            & (inclusion["latent_idx"].astype(int) == 0)
        ]
        assert not injected.empty
        assert float(injected["inclusion_frequency"].iloc[0]) >= 0.5

        e2 = run_bootstrap_ci(
            inputs,
            association_matrix_path=association_path,
            output_dir=root / "bootstrap_ci",
            top_k=3,
            n_bootstrap=20,
            group_column="source_file",
            random_state=7,
        )
        ci_df = pd.read_csv(root / "bootstrap_ci" / "bootstrap_ci_by_label_latent.csv")
        assert Path(e2["outputs"]["bootstrap_ci"]).exists()
        assert {"auc_ci_lo", "auc_ci_hi", "cohens_d_ci_lo", "cohens_d_ci_hi"}.issubset(ci_df.columns)
        assert len(ci_df) == 9

        e3 = run_cross_quality_validation(
            inputs,
            output_dir=root / "cross_quality_validation",
            reference_matrix_path=association_path,
            top_k=3,
            min_positive=2,
            min_negative=2,
        )
        comp = pd.read_csv(root / "cross_quality_validation" / "cross_quality_auc_comparison.csv")
        summary = pd.read_csv(root / "cross_quality_validation" / "cross_quality_summary.csv")
        assert Path(e3["outputs"]["comparison"]).exists()
        assert {"high_to_low", "low_to_high", "full_to_high", "full_to_low"}.issubset(set(comp["direction"]))
        assert len(summary) == 3

        auc_curve_path = root / "auc_by_k_curve_0_100.csv"
        mock_grid_path = root / "mock_topk_grid_summary.csv"
        mock_inclusion_path = root / "mock_inclusion_frequency.csv"
        mock_ci_path = root / "mock_bootstrap_ci.csv"
        mock_cross_quality_path = root / "mock_cross_quality.csv"
        curve_rows = []
        grid_rows = []
        inclusion_rows = []
        ci_rows = []
        cq_rows = []
        for label, latent_idx in [("A", 0), ("B", 1), ("C", 2)]:
            for k, auc in [(0, 0.5), (1, 0.80), (2, 0.86), (3, 0.90), (4, 0.901), (5, 0.902)]:
                curve_rows.append(
                    {
                        "label": label,
                        "subspace_ranking": "cohens_d",
                        "top_k": k,
                        "auc_mean": auc,
                        "auc_std": 0.01,
                        "n_folds": 3,
                    }
                )
            for k in range(1, 6):
                grid_rows.append(
                    {
                        "label": label,
                        "ranking_metric": "cohens_d",
                        "top_k": k,
                        "passes_null_2sd_rate": 1.0,
                        "observed_jaccard_p05": 0.20 if k < 3 else 0.50,
                    }
                )
                inclusion_rows.append(
                    {
                        "label": label,
                        "ranking_metric": "cohens_d",
                        "top_k": k,
                        "latent_idx": latent_idx,
                        "inclusion_frequency": 1.0,
                        "inclusion_count": 6,
                        "n_half_runs": 6,
                    }
                )
            ci_rows.append(
                {
                    "label": label,
                    "latent_idx": latent_idx,
                    "cohens_d_ci_lo": 0.5,
                    "cohens_d_ci_hi": 2.0,
                    "ci_excludes_zero": True,
                    "rank_within_label": 1,
                }
            )
            cq_rows.append(
                {
                    "label": label,
                    "latent_idx": latent_idx,
                    "auc_cross_drop": 0.01,
                }
            )
        pd.DataFrame(curve_rows).to_csv(auc_curve_path, index=False)
        pd.DataFrame(grid_rows).to_csv(mock_grid_path, index=False)
        pd.DataFrame(inclusion_rows).to_csv(mock_inclusion_path, index=False)
        pd.DataFrame(ci_rows).to_csv(mock_ci_path, index=False)
        pd.DataFrame(cq_rows).to_csv(mock_cross_quality_path, index=False)
        stable = run_stable_topk_selection(
            auc_curve_path=auc_curve_path,
            topk_grid_summary_path=mock_grid_path,
            inclusion_frequency_path=mock_inclusion_path,
            association_matrix_path=association_path,
            bootstrap_ci_path=mock_ci_path,
            cross_quality_path=mock_cross_quality_path,
            output_dir=root / "stable_topk_selection",
            config=StableTopKSelectionConfig(labels=labels, max_k=5),
        )
        stable_summary = stable["summary"]
        stable_latents = stable["latent_set"]
        assert set(stable_summary["label"]) == {"A", "B", "C"}
        assert set(stable_summary["k_auc"].astype(int)) == {3}
        assert set(stable_summary["k_star"].astype(int)) == {3}
        assert set(stable_latents["stable_set_role"]).issuperset({"stable_core"})
        assert (root / "stable_topk_selection" / "stable_k_by_label.csv").exists()
        assert (root / "stable_topk_selection" / "stable_topk_latent_set.csv").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_cross_val_experiment_stack()
    print("ok")
