"""Smoke test for MISC structural Top-K convergence analysis."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from run_misc_structural_k_convergence import (
    StructuralKConvergenceConfig,
    run_structural_k_convergence,
)


def _make_fixture(root: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(17)
    n = 360
    d = 24
    labels = ("RES", "REC", "QUO", "QUC", "GI", "SU", "AF")
    features = rng.normal(size=(n, d)).astype(np.float32)

    def sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-values))

    label_defs = {
        "RES": [0, 1],
        "REC": [1, 2],
        "QUO": [4, 5],
        "QUC": [5, 6],
        "GI": [8],
        "SU": [10, 11],
        "AF": [12, 13],
    }
    label_payload: dict[str, np.ndarray] = {}
    for label, signal_ids in label_defs.items():
        score = rng.normal(0, 0.9, size=n)
        for idx in signal_ids:
            score += 1.2 * features[:, idx]
        label_payload[label] = (rng.random(n) < sigmoid(score)).astype(int)

    feature_path = root / "features.npy"
    label_path = root / "labels.csv"
    association_path = root / "association.csv"
    np.save(feature_path, features)
    pd.DataFrame(label_payload).to_csv(label_path, index=False)

    rows: list[dict[str, object]] = []
    for label, signal_ids in label_defs.items():
        shared_bonus = {1: 0.02, 5: 0.02}.get
        for latent_idx in range(d):
            is_signal = latent_idx in signal_ids
            directional_auc = (
                0.92 - 0.02 * signal_ids.index(latent_idx)
                if is_signal
                else 0.68 - 0.003 * latent_idx + shared_bonus(latent_idx, 0.0)
            )
            rows.append(
                {
                    "label": label,
                    "latent_idx": latent_idx,
                    "directional_auc": directional_auc,
                    "auc": directional_auc,
                    "abs_cohens_d": 1.2 if is_signal else max(0.05, 0.35 - 0.005 * latent_idx),
                    "cohens_d": 1.2 if is_signal else 0.1,
                    "formal_edge_weight": max(directional_auc - 0.5, 0.0),
                    "stable_edge": is_signal,
                }
            )
    pd.DataFrame(rows).to_csv(association_path, index=False)
    return association_path, feature_path, label_path


def test_structural_k_convergence_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        association_path, feature_path, label_path = _make_fixture(root)
        output_dir = root / "out"
        result = run_structural_k_convergence(
            association_matrix=association_path,
            feature_store=feature_path,
            label_matrix=label_path,
            output_dir=output_dir,
            config=StructuralKConvergenceConfig(
                k_grid=(3, 5, 8, 10),
                ranking_metric="abs_cohens_d",
                cv_folds=2,
                precision_k=20,
                random_state=5,
                max_iter=300,
            ),
            make_figures=False,
        )

        fragmentation = result["fragmentation"]
        overlap = result["overlap"]
        poly = result["polysemanticity"]
        summary = result["summary"]

        assert set(fragmentation["top_k"].unique()) == {3, 5, 8, 10}
        assert (fragmentation["candidate_pool_size"] <= fragmentation["top_k"]).all()
        assert (fragmentation["k90_effect"] >= fragmentation["k80_effect"]).all()
        assert overlap["jaccard"].between(0.0, 1.0).all()
        assert overlap["weighted_jaccard"].between(0.0, 1.0).all()

        role_cols = [
            "label_specific_count",
            "same_family_shared_count",
            "same_supplemental_block_count",
            "cross_family_shared_count",
            "generalized_count",
        ]
        assert (poly[role_cols].sum(axis=1) == poly["unique_latents"]).all()
        assert set(summary["metric"]) == {"fragmentation", "overlap_jaccard", "polysemanticity"}

        required = [
            "k_convergence_summary.csv",
            "fragmentation_by_k.csv",
            "overlap_by_k.csv",
            "polysemanticity_by_k.csv",
            "k_convergence_report_zh.md",
        ]
        for name in required:
            assert (output_dir / name).exists(), name

        report = (output_dir / "k_convergence_report_zh.md").read_text(encoding="utf-8")
        assert "推荐上限K" in report
        assert "abs_cohens_d" in report


if __name__ == "__main__":
    test_structural_k_convergence_smoke()
    print("structural K convergence smoke passed")
