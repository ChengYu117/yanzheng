from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from run_misc_phase3_integrated_research_report import run_phase3_report


def _safe_root() -> Path:
    root = (Path.cwd() / "outputs" / "_smoke_phase3_integrated_report").resolve()
    cwd = Path.cwd().resolve()
    if cwd not in root.parents:
        raise RuntimeError(f"Refusing to use smoke output outside repo: {root}")
    return root


def _write_inputs(root: Path) -> dict[str, Path]:
    probe = root / "probe"
    evidence = root / "evidence"
    function = root / "function"
    for path in (probe, evidence, function):
        path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "representation": "full_sae_latents",
                "n_labels": 2,
                "mean_n_features": 8,
                "macro_auc": 0.82,
                "macro_average_precision": 0.50,
                "macro_f1": 0.45,
                "macro_balanced_accuracy": 0.75,
                "macro_accuracy": 0.80,
                "subspace_ranking": None,
                "top_n": None,
            },
            {
                "representation": "raw_hidden",
                "n_labels": 2,
                "mean_n_features": 4,
                "macro_auc": 0.84,
                "macro_average_precision": 0.52,
                "macro_f1": 0.48,
                "macro_balanced_accuracy": 0.76,
                "macro_accuracy": 0.81,
                "subspace_ranking": None,
                "top_n": None,
            },
            {
                "representation": "pca_raw_hidden",
                "n_labels": 2,
                "mean_n_features": 4,
                "macro_auc": 0.66,
                "macro_average_precision": 0.31,
                "macro_f1": 0.30,
                "macro_balanced_accuracy": 0.62,
                "macro_accuracy": 0.70,
                "subspace_ranking": None,
                "top_n": None,
            },
            {
                "representation": "sae_top_directional_auc_n005",
                "n_labels": 2,
                "mean_n_features": 5,
                "macro_auc": 0.80,
                "macro_average_precision": 0.47,
                "macro_f1": 0.42,
                "macro_balanced_accuracy": 0.73,
                "macro_accuracy": 0.79,
                "subspace_ranking": "directional_auc",
                "top_n": 5,
            },
            {
                "representation": "sae_top_directional_auc_n010",
                "n_labels": 2,
                "mean_n_features": 10,
                "macro_auc": 0.86,
                "macro_average_precision": 0.55,
                "macro_f1": 0.50,
                "macro_balanced_accuracy": 0.78,
                "macro_accuracy": 0.83,
                "subspace_ranking": "directional_auc",
                "top_n": 10,
            },
        ]
    ).to_csv(probe / "full_probe_summary.csv", index=False)

    by_rows = []
    for label in ("QUO", "RES"):
        by_rows.extend(
            [
                {
                    "representation": "full_sae_latents",
                    "label": label,
                    "probe_auc_mean": 0.82 if label == "QUO" else 0.70,
                    "subspace_ranking": None,
                    "top_n": None,
                },
                {
                    "representation": "raw_hidden",
                    "label": label,
                    "probe_auc_mean": 0.84 if label == "QUO" else 0.72,
                    "subspace_ranking": None,
                    "top_n": None,
                },
                {
                    "representation": "pca_raw_hidden",
                    "label": label,
                    "probe_auc_mean": 0.66 if label == "QUO" else 0.60,
                    "subspace_ranking": None,
                    "top_n": None,
                },
                {
                    "representation": "sae_top_directional_auc_n005",
                    "label": label,
                    "probe_auc_mean": 0.80 if label == "QUO" else 0.68,
                    "subspace_ranking": "directional_auc",
                    "top_n": 5,
                },
                {
                    "representation": "sae_top_directional_auc_n010",
                    "label": label,
                    "probe_auc_mean": 0.86 if label == "QUO" else 0.73,
                    "subspace_ranking": "directional_auc",
                    "top_n": 10,
                },
            ]
        )
    pd.DataFrame(by_rows).to_csv(probe / "full_probe_by_label_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "representation": "sae_top_directional_auc_n005",
                "subspace_ranking": "directional_auc",
                "top_n": 5,
                "macro_auc": 0.80,
                "last_4_step_mean_auc_gain": 0.006,
                "platformed_by_last_steps": False,
            },
            {
                "representation": "sae_top_directional_auc_n010",
                "subspace_ranking": "directional_auc",
                "top_n": 10,
                "macro_auc": 0.86,
                "last_4_step_mean_auc_gain": 0.001,
                "platformed_by_last_steps": True,
            },
        ]
    ).to_csv(probe / "ranked_sae_subspace_convergence.csv", index=False)

    pd.DataFrame(
        [
            {
                "target_label": "QUO",
                "latent_idx": 1,
                "top_activating_target_match_rate": 0.80,
                "precision_at_50": 0.80,
                "directional_auc": 0.90,
                "cohens_d": 1.2,
                "n_top_activating": 50,
                "n_high_non_target": 20,
                "n_random_target": 20,
                "duplicate_text_row_count": 9,
                "unique_file_count": 25,
                "active_label_counts_top_activating": "QUO:40,QU:42,QUC:5",
            },
            {
                "target_label": "RES",
                "latent_idx": 2,
                "top_activating_target_match_rate": 0.20,
                "precision_at_50": 0.20,
                "directional_auc": 0.65,
                "cohens_d": 0.7,
                "n_top_activating": 50,
                "n_high_non_target": 20,
                "n_random_target": 20,
                "duplicate_text_row_count": 18,
                "unique_file_count": 18,
                "active_label_counts_top_activating": "RES:10,REC:25,RE:30",
            },
        ]
    ).to_csv(evidence / "latent_evidence_packet_summary.csv", index=False)

    (function / "manifest.json").write_text(
        json.dumps(
            {
                "dry_run_prompts": True,
                "n_reviews": 2,
                "n_success": 0,
                "n_failed": 0,
                "n_pending_dry_run": 2,
                "status_counts": {"pending_dry_run": 2},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"probe": probe, "evidence": evidence, "function": function}


def test_phase3_integrated_report_smoke() -> None:
    root = _safe_root()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        paths = _write_inputs(root)
        out = root / "out"
        manifest = run_phase3_report(
            full_probe_summary_path=paths["probe"] / "full_probe_summary.csv",
            full_probe_by_label_summary_path=paths["probe"] / "full_probe_by_label_summary.csv",
            ranked_convergence_path=paths["probe"] / "ranked_sae_subspace_convergence.csv",
            evidence_summary_path=paths["evidence"] / "latent_evidence_packet_summary.csv",
            function_manifest_path=paths["function"] / "manifest.json",
            output_dir=out,
        )
        assert manifest["n_labels"] == 2
        assert manifest["phase2_review_status"]["n_pending_dry_run"] == 2
        assert (out / "phase3_integrated_research_report_zh.md").exists()
        assert (out / "phase3_label_integrated_summary.csv").exists()
        assert (out / "phase3_label_topn_convergence.csv").exists()
        text = (out / "phase3_integrated_research_report_zh.md").read_text(encoding="utf-8")
        assert "不能证明" in text
        assert "dry-run" in text
        table = pd.read_csv(out / "phase3_label_integrated_summary.csv")
        assert set(table["label"]) == {"QUO", "RES"}
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_phase3_integrated_report_smoke()
    print("test_phase3_integrated_research_report_smoke passed")
