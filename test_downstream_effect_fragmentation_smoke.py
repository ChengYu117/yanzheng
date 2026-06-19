"""Smoke test for downstream-effect fragmentation analysis."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from run_misc_downstream_effect_fragmentation import (
    DownstreamEffectFragmentationConfig,
    run_downstream_effect_fragmentation,
)


def _make_fixture(root: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(7)
    n = 600
    d = 30
    features = rng.normal(size=(n, d)).astype(np.float32)

    def sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-values))

    compact_score = 1.5 * features[:, 0] + 1.5 * features[:, 1] + rng.normal(0, 1.2, size=n)
    compact_y = (rng.random(n) < sigmoid(compact_score)).astype(int)

    spread_score = rng.normal(0, 1.2, size=n)
    for idx in range(10, 18):
        spread_score += 0.7 * features[:, idx]
    spread_y = (rng.random(n) < sigmoid(spread_score)).astype(int)

    feature_path = root / "features.npy"
    label_path = root / "labels.csv"
    association_path = root / "association.csv"
    np.save(feature_path, features)
    pd.DataFrame({"COMPACT": compact_y, "SPREAD": spread_y}).to_csv(label_path, index=False)

    rows: list[dict[str, object]] = []
    for label, signal_ids in {"COMPACT": [0, 1], "SPREAD": list(range(10, 18))}.items():
        for latent_idx in range(d):
            is_signal = latent_idx in signal_ids
            rows.append(
                {
                    "label": label,
                    "latent_idx": latent_idx,
                    "directional_auc": 0.95 - 0.01 * signal_ids.index(latent_idx)
                    if is_signal
                    else 0.70 - latent_idx * 0.001,
                    "auc": 0.95 if is_signal else 0.55,
                    "abs_cohens_d": 1.5 if is_signal else 0.2,
                    "stable_edge": is_signal,
                }
            )
    pd.DataFrame(rows).to_csv(association_path, index=False)
    return association_path, feature_path, label_path


def test_downstream_effect_fragmentation_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        association_path, feature_path, label_path = _make_fixture(root)
        output_dir = root / "out"
        result = run_downstream_effect_fragmentation(
            association_matrix=association_path,
            feature_store=feature_path,
            label_matrix=label_path,
            output_dir=output_dir,
            config=DownstreamEffectFragmentationConfig(
                labels=("COMPACT", "SPREAD"),
                candidate_top_k=20,
                cv_folds=3,
                min_auc=0.70,
                precision_k=20,
                random_state=3,
            ),
            make_figures=False,
        )

        summary = result["summary"].set_index("label")
        assert "effect_epsilon_auc" in summary.columns
        assert summary.loc["SPREAD", "k90_effect"] > summary.loc["COMPACT", "k90_effect"]
        assert summary.loc["SPREAD", "effective_effect_latent_count"] > summary.loc[
            "COMPACT", "effective_effect_latent_count"
        ]
        effects = result["latent_effects"]
        below_floor = effects["raw_effect_weight_auc"] < 0.001
        assert (effects.loc[below_floor, "effect_weight_auc"] == 0.0).all()

        candidates = result["candidate_pool"]
        compact_top = candidates[candidates["label"] == "COMPACT"].sort_values(
            "candidate_rank_directional_auc"
        )
        assert compact_top.iloc[0]["latent_idx"] == 0
        assert compact_top.iloc[1]["latent_idx"] == 1
        assert compact_top.shape[0] == 20

        required = [
            "downstream_effect_candidate_pool.csv",
            "downstream_effect_fold_metrics.csv",
            "downstream_effect_latent_effects.csv",
            "downstream_effect_fragmentation_summary.csv",
            "downstream_effect_fragmentation_report.md",
            "downstream_effect_fragmentation_summary.json",
        ]
        for name in required:
            assert (output_dir / name).exists(), name


if __name__ == "__main__":
    test_downstream_effect_fragmentation_smoke()
    print("downstream effect fragmentation smoke passed")
