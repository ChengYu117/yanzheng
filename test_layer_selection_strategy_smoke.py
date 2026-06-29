from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from run_layer_selection_strategy import (
    parse_hook_layer,
    run_layer_selection_strategy,
)


def _safe_smoke_root() -> Path:
    root = (Path.cwd() / "outputs" / "_smoke_layer_selection_strategy").resolve()
    cwd = Path.cwd().resolve()
    if cwd not in root.parents:
        raise RuntimeError(f"Refusing to use smoke output outside repo: {root}")
    return root


def test_parse_hook_layer() -> None:
    assert parse_hook_layer("blocks.19.hook_resid_post") == 19
    assert parse_hook_layer("blocks.0.hook_resid_post") == 0


def test_layer_selection_strategy() -> None:
    root = _safe_smoke_root()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    try:
        sae_config_path = root / "sae_config.json"
        metadata_path = root / "feature_metadata.json"
        metrics_path = root / "layer_probe_metrics.csv"
        output_dir = root / "out"

        sae_config_path.write_text(
            json.dumps(
                {
                    "sae_repo_id": "toy/llama",
                    "sae_subfolder": "toy-L19",
                    "hook_point": "blocks.19.hook_resid_post",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        metadata_path.write_text(
            json.dumps(
                {
                    "hook_point": "blocks.19.hook_resid_post",
                    "n_records": 8,
                    "feature_shape": [8, 16],
                    "activation_shape": [8, 4],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        pd.DataFrame(
            [
                {
                    "label": "A",
                    "layer_idx": 0,
                    "pooling": "mean",
                    "auc_mean": 0.896,
                    "auc_std": 0.01,
                    "f1_mean": 0.50,
                    "balanced_accuracy_mean": 0.70,
                    "recognized": True,
                },
                {
                    "label": "A",
                    "layer_idx": 1,
                    "pooling": "mean",
                    "auc_mean": 0.900,
                    "auc_std": 0.02,
                    "f1_mean": 0.55,
                    "balanced_accuracy_mean": 0.75,
                    "recognized": True,
                },
                {
                    "label": "A",
                    "layer_idx": 1,
                    "pooling": "last",
                    "auc_mean": 0.850,
                    "auc_std": 0.03,
                    "f1_mean": 0.40,
                    "balanced_accuracy_mean": 0.65,
                    "recognized": True,
                },
                {
                    "label": "A",
                    "layer_idx": 2,
                    "pooling": "last",
                    "auc_mean": 0.899,
                    "auc_std": 0.01,
                    "f1_mean": 0.54,
                    "balanced_accuracy_mean": 0.74,
                    "recognized": True,
                },
                {
                    "label": "B",
                    "layer_idx": 2,
                    "pooling": "mean",
                    "auc_mean": 0.700,
                    "auc_std": 0.01,
                    "f1_mean": 0.35,
                    "balanced_accuracy_mean": 0.60,
                    "recognized": False,
                },
                {
                    "label": "B",
                    "layer_idx": 2,
                    "pooling": "last",
                    "auc_mean": 0.910,
                    "auc_std": 0.02,
                    "f1_mean": 0.57,
                    "balanced_accuracy_mean": 0.77,
                    "recognized": True,
                },
                {
                    "label": "B",
                    "layer_idx": 3,
                    "pooling": "mean",
                    "auc_mean": 0.905,
                    "auc_std": 0.01,
                    "f1_mean": 0.56,
                    "balanced_accuracy_mean": 0.76,
                    "recognized": True,
                },
            ]
        ).to_csv(metrics_path, index=False)

        strategy = run_layer_selection_strategy(
            llama_sae_config_path=sae_config_path,
            llama_feature_metadata_path=metadata_path,
            gemma_layer_metrics_path=metrics_path,
            output_dir=output_dir,
            labels=("A", "B"),
            near_delta=0.005,
        )

        expected_files = [
            "layer_selection_strategy.json",
            "llama_layer_selection.csv",
            "gemma_control_layer_selection.csv",
            "layer_selection_strategy_report.md",
        ]
        for name in expected_files:
            assert (output_dir / name).exists(), name

        assert strategy["llama_main"]["global_canonical_layer"] == 19
        assert strategy["gemma_control"]["interpretation_boundary"] == "control_only_not_used_to_choose_llama_layer"

        llama = pd.read_csv(output_dir / "llama_layer_selection.csv")
        assert set(llama["target_label"]) == {"A", "B"}
        assert llama["canonical_layer"].tolist() == [19, 19]
        assert set(llama["label_specific_best_layer"]) == {"not_available"}
        assert set(llama["early_stable_layer"]) == {"not_available"}
        assert set(llama["availability_note"]) == {"requires_llama_cross_layer_probe_metrics"}

        gemma = pd.read_csv(output_dir / "gemma_control_layer_selection.csv")
        row_a = gemma[gemma["target_label"] == "A"].iloc[0]
        assert int(row_a["best_layer"]) == 1
        assert abs(float(row_a["best_auc"]) - 0.900) < 1e-12
        assert int(row_a["earliest_near_optimal_layer"]) == 0
        assert abs(float(row_a["earliest_delta_from_best"]) - 0.004) < 1e-12
        assert str(row_a["near_optimal_layers"]) == "0,1,2"

        row_b = gemma[gemma["target_label"] == "B"].iloc[0]
        assert int(row_b["best_layer"]) == 2
        assert row_b["best_pooling"] == "last"
        assert abs(float(row_b["best_auc"]) - 0.910) < 1e-12
        assert int(row_b["earliest_near_optimal_layer"]) == 2

        report = (output_dir / "layer_selection_strategy_report.md").read_text(encoding="utf-8")
        assert "Gemma 仅为对照，不能决定 Llama 主层" in report
        assert "这不是 Llama 全层搜索后的最优层结论" in report

        output_dir_with_llama = root / "out_with_llama"
        strategy_with_llama = run_layer_selection_strategy(
            llama_sae_config_path=sae_config_path,
            llama_feature_metadata_path=metadata_path,
            llama_layer_metrics_path=metrics_path,
            gemma_layer_metrics_path=metrics_path,
            output_dir=output_dir_with_llama,
            labels=("A", "B"),
            near_delta=0.005,
        )
        assert strategy_with_llama["llama_main"]["layer_metrics_available"] is True
        assert (
            strategy_with_llama["llama_main"]["label_specific_best_status"]
            == "computed_from_llama_cross_layer_probe_metrics"
        )

        llama_with_metrics = pd.read_csv(output_dir_with_llama / "llama_layer_selection.csv")
        row_a_llama = llama_with_metrics[llama_with_metrics["target_label"] == "A"].iloc[0]
        assert int(row_a_llama["canonical_layer"]) == 19
        assert int(row_a_llama["label_specific_best_layer"]) == 1
        assert int(row_a_llama["early_stable_layer"]) == 0
        assert abs(float(row_a_llama["early_stable_delta_from_best"]) - 0.004) < 1e-12

        report_with_llama = (output_dir_with_llama / "layer_selection_strategy_report.md").read_text(
            encoding="utf-8"
        )
        assert "已接入 Llama cross-layer probe metrics" in report_with_llama
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_parse_hook_layer()
    test_layer_selection_strategy()
    print("test_layer_selection_strategy_smoke passed")
