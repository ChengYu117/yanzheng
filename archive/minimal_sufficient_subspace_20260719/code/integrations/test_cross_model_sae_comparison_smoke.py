"""Smoke test for cross-model SAE comparison report generation."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def _write_model_root(root: Path, auc_offset: float) -> None:
    labels = ["RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF"]
    mapping_dir = root / "functional" / "misc_label_mapping"
    latent_dir = root / "interpretability" / "latent_space_search_v2"
    minimal_dir = root / "interpretability" / "minimal_sufficient_subspace_v2"
    mapping_dir.mkdir(parents=True)
    latent_dir.mkdir(parents=True)
    minimal_dir.mkdir(parents=True)
    rows = []
    frag = []
    minimal = []
    for label_idx, label in enumerate(labels):
        for latent_idx in range(4):
            rows.append(
                {
                    "label": label,
                    "latent_idx": latent_idx,
                    "directional_auc": 0.70 + auc_offset + label_idx * 0.005 - latent_idx * 0.01,
                    "abs_cohens_d": 0.5 + label_idx * 0.01,
                }
            )
        frag.append(
            {
                "label": label,
                "thresholded_latent_count": 2,
                "effective_fragmentation": 1.7,
                "fragmentation_class": "compact",
                "selection_status": "stable",
            }
        )
        minimal.append(
            {
                "label": label,
                "formal_status": "recoverable",
                "full_auc_mean": 0.80 + auc_offset,
                "minimal_k_median": 5 + label_idx,
                "mean_pairwise_jaccard_between_folds": 0.4,
                "predictive_redundancy_ratio": 0.1,
            }
        )
    pd.DataFrame(rows).to_csv(mapping_dir / "latent_label_matrix.csv", index=False)
    pd.DataFrame(frag).to_csv(latent_dir / "fragmentation_v2.csv", index=False)
    pd.DataFrame(minimal).to_csv(minimal_dir / "minimal_sufficient_summary_v2.csv", index=False)


def test_cross_model_comparison_smoke() -> None:
    from run_cross_model_sae_comparison import main

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        llama = root / "llama"
        gemma = root / "gemma"
        out = root / "out"
        _write_model_root(llama, 0.00)
        _write_model_root(gemma, 0.03)

        old_argv = sys.argv
        try:
            sys.argv = [
                "run_cross_model_sae_comparison.py",
                "--llama-root",
                str(llama),
                "--gemma-root",
                str(gemma),
                "--output-dir",
                str(out),
            ]
            assert main() == 0
        finally:
            sys.argv = old_argv

        assert (out / "label_level_model_comparison.csv").exists()
        assert (out / "label_level_model_comparison_wide.csv").exists()
        report = out / "model_specificity_report.md"
        assert report.exists()
        text = report.read_text(encoding="utf-8")
        assert "Cross-Model SAE Specificity Comparison" in text
        assert "Gemma layer 18 is RE-selected" in text


if __name__ == "__main__":
    test_cross_model_comparison_smoke()
    print("cross_model_sae_comparison smoke passed")

