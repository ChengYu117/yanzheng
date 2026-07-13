"""Smoke test for the frozen representation-comparison exporter."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.nlp_re_base.representation_comparison_freeze import FreezeConfig, REPRESENTATIONS, freeze_representation_comparison


def test_freeze_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source"
        output = root / "output"
        source.mkdir()
        labels = ("RES", "REC")
        rows = []
        for label_i, label in enumerate(labels):
            for fold in (1, 2):
                for representation in REPRESENTATIONS:
                    top_ns = (10, 20) if representation in {"Top-n SAE", "PCA-n", "Random SAE-n"} else (-1,)
                    seeds = (42, 43) if representation == "Random SAE-n" else (None,)
                    for top_n in top_ns:
                        for seed in seeds:
                            rows.append({
                                "representation": representation, "label": label, "fold": fold, "top_n": top_n,
                                "seed": seed, "split_policy": "stratified-group-kfold", "n_train": 80, "n_test": 20,
                                "train_positive": 20 + label_i, "test_positive": 5 + label_i, "n_features": max(top_n, 3),
                                "auc": 0.7, "average_precision": 0.5, "f1": 0.4, "balanced_accuracy": 0.6,
                                "status": "ok", "pca_input_standardized": representation == "PCA-n",
                                "pca_post_standardized": representation == "PCA-n",
                            })
        pd.DataFrame(rows).to_csv(source / "probe_fold_metrics.csv", index=False)
        selected = pd.DataFrame({"label": list(labels), "representation": ["Top-n SAE", "Top-n SAE"], "latent_idx": [1, 2]})
        selected.to_csv(source / "selected_latents_by_label_n.csv", index=False)
        manifest = {
            "config": {"top_ns": [10, 20, 50, 100, 200], "C": 1.0, "solver": "liblinear", "max_iter": 1000, "standardize": True, "random_state": 42},
            "protocol": {"sae_rows_imported": True, "sae_rows_rerun": False},
        }
        (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        plot_script = root / "plot.py"
        plot_script.write_text("print('plot')\n", encoding="utf-8")
        result = freeze_representation_comparison(
            source_dir=source, output_dir=output, plot_script=plot_script,
            config=FreezeConfig(labels=labels, package_name="smoke"),
        )
        assert set(result["by_label"]["label"]) == set(labels)
        assert result["macro"]["n_labels"].eq(2).all()
        audit = result["audit"].set_index("check")
        assert audit.loc["same_outer_folds", "status"] == "PASS"
        assert audit.loc["stable_core_selection_train_only", "status"] == "FAIL"
        assert (output / "SHA256SUMS.txt").exists()


if __name__ == "__main__":
    test_freeze_smoke()
    print("test_representation_comparison_freeze_smoke passed")
