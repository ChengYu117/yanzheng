"""Smoke tests for the layer-wise PCA / SAE Top-n probe module."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.layerwise_representation_probe import (
    LayerMatrices,
    LayerwiseRepresentationProbeConfig,
    run_layerwise_representation_probe,
)


def _synthetic_layer(seed: int, y_a: np.ndarray, y_b: np.ndarray) -> LayerMatrices:
    rng = np.random.default_rng(seed)
    n = len(y_a)
    sae = rng.normal(0.0, 0.2, size=(n, 12)).astype(np.float32)
    sae[:, 0] += y_a * 2.5
    sae[:, 1] += y_a * 1.5
    sae[:, 2] += y_b * 2.5
    sae[:, 3] += y_b * 1.5
    sae[:, 4] -= y_a * 2.0  # Must not enter the positive Cohen's d Top-n set for A.
    raw = rng.normal(0.0, 0.3, size=(n, 10)).astype(np.float32)
    raw[:, 0] += y_a * 1.5
    raw[:, 1] += y_b * 1.5
    audit = pd.DataFrame({"latent_idx": np.arange(sae.shape[1]), "keep": True})
    audit.loc[audit["latent_idx"] == 11, "keep"] = False
    return LayerMatrices(sae_features=sae, raw_hidden=raw, feature_filter_audit=audit)


def test_layerwise_representation_probe_smoke() -> None:
    n = 72
    groups = np.repeat([f"file_{idx:02d}" for idx in range(12)], 6)
    y_a = np.zeros(n, dtype=int)
    y_b = np.zeros(n, dtype=int)
    for idx, group in enumerate(np.unique(groups)):
        rows = np.flatnonzero(groups == group)
        if idx % 2 == 0:
            y_a[rows[:3]] = 1
        else:
            y_b[rows[:3]] = 1
    labels = pd.DataFrame({"file_id": groups, "A": y_a, "B": y_b})

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        result = run_layerwise_representation_probe(
            layer_matrices={2: _synthetic_layer(11, y_a, y_b), 5: _synthetic_layer(22, y_a, y_b)},
            label_df=labels,
            output_dir=output_dir,
            config=LayerwiseRepresentationProbeConfig(
                labels=("A", "B"),
                pca_components=4,
                sae_top_n=4,
                include_full_sae=False,
                folds=3,
                quiet=True,
            ),
        )

        for name in (
            "probe_fold_metrics.csv",
            "probe_summary_by_layer_label.csv",
            "probe_macro_summary_by_layer.csv",
            "selected_top100_latents_by_layer_label.csv",
            "layerwise_representation_probe_report.md",
            "manifest.json",
            "layer_02/feature_filter_audit.csv",
            "layer_05/feature_filter_summary.json",
        ):
            assert (output_dir / name).exists(), name

        fold = result["fold_metrics"]
        assert set(fold["representation"]) == {"Raw Hidden", "PCA-4", "Top-4 SAE"}
        for metric in ("auc", "average_precision", "f1", "balanced_accuracy"):
            assert metric in fold.columns
        fold_shapes = fold.groupby(["layer_idx", "label", "fold"])[["n_train", "n_test"]].nunique()
        assert (fold_shapes["n_train"] == 1).all()
        assert (fold_shapes["n_test"] == 1).all()

        selected = result["selected_latents"]
        assert 4 not in set(selected[selected["label"] == "A"]["latent_idx"].astype(int))
        assert 11 not in set(selected["latent_idx"].astype(int))
        macro = result["macro_summary"]
        assert set(macro["layer_idx"].astype(int)) == {2, 5}
        assert set(macro["representation"]) == {"Raw Hidden", "PCA-4", "Top-4 SAE"}


if __name__ == "__main__":
    test_layerwise_representation_probe_smoke()
    print("test_layerwise_representation_probe_smoke passed")
