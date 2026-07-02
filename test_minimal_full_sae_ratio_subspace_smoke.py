"""Smoke tests for full-SAE ratio minimal latent subset search."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _make_synthetic() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(23)
    n = 360
    d = 24
    features = rng.normal(0, 0.20, size=(n, d)).astype(np.float32)

    af = np.zeros(n, dtype=int)
    af[:90] = 1
    features[af == 1, 0] += 3.0

    gi = np.zeros(n, dtype=int)
    gi[120:240] = 1
    features[120:180, 5] += 3.2
    features[180:240, 6] += 3.2
    features[gi == 0, 5] -= 0.3
    features[gi == 0, 6] -= 0.3

    labels = pd.DataFrame(
        {
            "file_id": np.repeat([f"session_{idx:02d}" for idx in range(36)], 10),
            "AF": af,
            "GI": gi,
        }
    )
    rows = []
    for label, signal_ids in {"AF": [0], "GI": [5, 6]}.items():
        backup = [idx for idx in range(d) if idx not in signal_ids][:10]
        for rank, lid in enumerate(signal_ids + backup, start=1):
            strong = lid in signal_ids
            rows.append(
                {
                    "label": label,
                    "latent_idx": lid,
                    "association_rank": rank,
                    "directional_auc": 0.95 if strong else 0.55,
                    "abs_cohens_d": 1.5 if strong else 0.10,
                    "cohens_d": 1.5 if strong else 0.10,
                    "prevalence": float(labels[label].mean()),
                    "precision_at_50": float(labels[label].mean()) + (0.4 if strong else 0.0),
                    "precision_lift_absolute_at_50": 0.4 if strong else 0.0,
                    "formal_edge_weight": 1.0 if strong else 0.0,
                }
            )
    audit = pd.DataFrame({"latent_idx": list(range(d)), "keep": [True] * d})
    return features, labels, pd.DataFrame(rows), audit


def test_minimal_full_sae_ratio_subspace_smoke() -> None:
    from nlp_re_base.minimal_full_sae_ratio_subspace import (
        MinimalFullSaeRatioSubspaceConfig,
        run_minimal_full_sae_ratio_subspace,
    )

    features, labels, association, audit = _make_synthetic()
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        feature_path = root / "features.npy"
        label_path = root / "labels.csv"
        association_path = root / "association.csv"
        audit_path = root / "feature_filter_audit.csv"
        out = root / "out"
        np.save(feature_path, features)
        labels.to_csv(label_path, index=False)
        association.to_csv(association_path, index=False)
        audit.to_csv(audit_path, index=False)

        result = run_minimal_full_sae_ratio_subspace(
            association_matrix=association_path,
            feature_store=feature_path,
            label_matrix=label_path,
            output_dir=out,
            feature_filter_audit=audit_path,
            config=MinimalFullSaeRatioSubspaceConfig(
                labels=("AF", "GI"),
                leaf_labels=("AF", "GI"),
                candidate_top_k=8,
                target_ratio=0.95,
                cv_folds=3,
                split_policy="stratified-kfold",
                max_search_k=5,
                random_state=3,
                max_iter=500,
            ),
        )

        summary = result["summary"].set_index("label")
        assert set(summary.index) == {"AF", "GI"}
        assert summary.loc["AF", "formal_status"] == "minimal_ratio_found"
        assert summary.loc["GI", "formal_status"] == "minimal_ratio_found"
        assert int(summary.loc["AF", "final_selected_k"]) >= 1
        assert int(summary.loc["GI", "final_selected_k"]) >= 1

        found = result["folds"][result["folds"]["fold_status"] == "minimal_ratio_found"]
        assert not found.empty
        assert (found["selected_auc"] >= found["target_auc"]).all()
        assert (out / "minimal_full_sae_ratio_summary.csv").exists()
        assert (out / "minimal_full_sae_ratio_fold_results.csv").exists()
        assert (out / "minimal_full_sae_ratio_selection_steps.csv").exists()
        assert (out / "minimal_full_sae_ratio_selected_latents.csv").exists()
        assert (out / "minimal_full_sae_ratio_report.md").exists()
        assert (out / "manifest.json").exists()


def main() -> int:
    test_minimal_full_sae_ratio_subspace_smoke()
    print("minimal_full_sae_ratio_subspace smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
