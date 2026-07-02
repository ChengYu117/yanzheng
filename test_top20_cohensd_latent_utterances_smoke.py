from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import torch

from run_misc_top20_cohensd_latent_utterances import run_top20_cohensd_latent_utterance_export


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _safe_smoke_root() -> Path:
    root = (Path.cwd() / "outputs" / "_smoke_top20_cohensd_latent_utterances").resolve()
    cwd = Path.cwd().resolve()
    if cwd not in root.parents:
        raise RuntimeError(f"Refusing to use smoke output outside repo: {root}")
    return root


def test_top20_cohensd_selection_and_utterance_export() -> None:
    root = _safe_smoke_root()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    try:
        association = pd.DataFrame(
            [
                # Label A: latent 2 has huge absolute d but negative d, so it must be excluded.
                {"label": "A", "latent_idx": 0, "cohens_d": 0.70, "abs_cohens_d": 0.70, "auc": 0.70, "directional_auc": 0.80, "precision_at_10": 0.8, "precision_at_50": 0.6, "p_value": 0.01, "significant_fdr": True},
                {"label": "A", "latent_idx": 1, "cohens_d": 1.00, "abs_cohens_d": 1.00, "auc": 0.75, "directional_auc": 0.75, "precision_at_10": 0.7, "precision_at_50": 0.5, "p_value": 0.02, "significant_fdr": True},
                {"label": "A", "latent_idx": 2, "cohens_d": -9.00, "abs_cohens_d": 9.00, "auc": 0.05, "directional_auc": 0.95, "precision_at_10": 0.1, "precision_at_50": 0.1, "p_value": 0.0, "significant_fdr": True},
                {"label": "A", "latent_idx": 3, "cohens_d": 0.70, "abs_cohens_d": 0.70, "auc": 0.90, "directional_auc": 0.90, "precision_at_10": 0.9, "precision_at_50": 0.7, "p_value": 0.03, "significant_fdr": False},
                # Label B: selected by positive Cohen's d only.
                {"label": "B", "latent_idx": 4, "cohens_d": 1.10, "abs_cohens_d": 1.10, "auc": 0.88, "directional_auc": 0.88, "precision_at_10": 0.9, "precision_at_50": 0.8, "p_value": 0.01, "significant_fdr": True},
                {"label": "B", "latent_idx": 0, "cohens_d": 0.90, "abs_cohens_d": 0.90, "auc": 0.81, "directional_auc": 0.81, "precision_at_10": 0.8, "precision_at_50": 0.7, "p_value": 0.02, "significant_fdr": True},
                {"label": "B", "latent_idx": 2, "cohens_d": 0.80, "abs_cohens_d": 0.80, "auc": 0.79, "directional_auc": 0.79, "precision_at_10": 0.7, "precision_at_50": 0.6, "p_value": 0.04, "significant_fdr": False},
            ]
        )
        features = torch.tensor(
            [
                [1.0, 5.0, 0.0, 1.0, 0.1],
                [2.0, 4.0, 1.0, 5.0, 0.2],
                [3.0, 3.0, 2.0, 4.0, 0.3],
                [4.0, 2.0, 3.0, 3.0, 0.4],
                [5.0, 1.0, 4.0, 2.0, 9.0],
            ],
            dtype=torch.float32,
        )
        labels = pd.DataFrame(
            [
                {"row_idx": 0, "record_id": "r0", "file_id": "f", "source_line": 1, "predicted_code": "A", "predicted_subcode": "", "confidence": 0.9, "unit_text": "utterance 0", "A": 1, "B": 0},
                {"row_idx": 1, "record_id": "r1", "file_id": "f", "source_line": 2, "predicted_code": "A", "predicted_subcode": "", "confidence": 0.8, "unit_text": "utterance 1", "A": 1, "B": 0},
                {"row_idx": 2, "record_id": "r2", "file_id": "f", "source_line": 3, "predicted_code": "B", "predicted_subcode": "", "confidence": 0.7, "unit_text": "utterance 2", "A": 0, "B": 1},
                {"row_idx": 3, "record_id": "r3", "file_id": "f", "source_line": 4, "predicted_code": "B", "predicted_subcode": "", "confidence": 0.6, "unit_text": "utterance 3", "A": 0, "B": 1},
                {"row_idx": 4, "record_id": "r4", "file_id": "f", "source_line": 5, "predicted_code": "B", "predicted_subcode": "", "confidence": 0.5, "unit_text": "utterance 4", "A": 0, "B": 1},
            ]
        )
        records = labels.to_dict(orient="records")

        association_path = root / "association.csv"
        feature_path = root / "features.pt"
        label_path = root / "labels.csv"
        records_path = root / "records.jsonl"
        feature_filter_audit_path = root / "feature_filter_audit.csv"
        output_dir = root / "out"

        association.to_csv(association_path, index=False)
        torch.save({"utterance_features": features}, feature_path)
        labels.to_csv(label_path, index=False)
        _write_jsonl(records_path, records)
        pd.DataFrame(
            [
                {"latent_idx": 0, "keep": True},
                {"latent_idx": 1, "keep": True},
                {"latent_idx": 2, "keep": False},
                {"latent_idx": 3, "keep": True},
                {"latent_idx": 4, "keep": True},
            ]
        ).to_csv(feature_filter_audit_path, index=False)

        audit = run_top20_cohensd_latent_utterance_export(
            association_path=association_path,
            feature_store_path=feature_path,
            label_matrix_path=label_path,
            records_path=records_path,
            output_dir=output_dir,
            feature_filter_audit_path=feature_filter_audit_path,
            labels=("A", "B"),
            top_features=2,
            top_utterances=3,
        )

        selected = pd.read_csv(output_dir / "top2_cohensd_latents_by_label.csv")
        utterances = pd.read_csv(output_dir / "top3_utterances_by_top2_cohensd_latents.csv")
        metrics = pd.read_csv(output_dir / "top2_latent_label_metrics_with_top3_precision.csv")
        report = (output_dir / "top2_cohensd_feature_activation_report.md").read_text(encoding="utf-8")
        audit_path = output_dir / "top2_cohensd_feature_activation_audit.json"

        assert len(selected) == 4
        assert len(utterances) == 12
        assert len(metrics) == 4
        assert audit["n_selected_latents"] == 4
        assert audit["n_top_utterance_rows"] == 12
        assert audit["filter_summary"]["selected_latents_all_keep_true"] is True
        assert audit["row_count_checks"]["utterance_rows_match"] is True
        assert audit["row_count_checks"]["every_label_latent_has_top_n"] is True
        assert audit_path.exists()
        assert (output_dir / "top2_cohensd_feature_activation_report.md").exists()
        assert "top3_target_match_rate" in metrics.columns

        assert selected[selected["label"] == "A"]["latent_idx"].tolist() == [1, 3]
        assert 2 not in selected["latent_idx"].tolist()
        assert selected[selected["label"] == "B"]["latent_idx"].tolist() == [4, 0]

        first_a1 = utterances[(utterances["label"] == "A") & (utterances["latent_idx"] == 1)].iloc[0]
        assert int(first_a1["row_idx"]) == 0
        assert first_a1["record_id"] == "r0"
        assert first_a1["unit_text"] == "utterance 0"
        assert int(first_a1["target_match"]) == 1
        assert first_a1["active_labels"] == "A"

        first_b4 = utterances[(utterances["label"] == "B") & (utterances["latent_idx"] == 4)].iloc[0]
        assert int(first_b4["row_idx"]) == 4
        assert int(first_b4["target_match"]) == 1
        assert first_b4["active_labels"] == "B"

        assert "不使用 `abs_cohens_d`" in report
        assert "target_match=1" in report
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_feature_filter_audit_rejects_dropped_selected_latent() -> None:
    root = _safe_smoke_root()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    try:
        association = pd.DataFrame(
            [
                {"label": "A", "latent_idx": 0, "cohens_d": 0.70, "abs_cohens_d": 0.70, "auc": 0.70, "directional_auc": 0.80, "precision_at_10": 0.8, "precision_at_50": 0.6, "p_value": 0.01, "significant_fdr": True},
                {"label": "A", "latent_idx": 2, "cohens_d": 9.00, "abs_cohens_d": 9.00, "auc": 0.99, "directional_auc": 0.99, "precision_at_10": 1.0, "precision_at_50": 1.0, "p_value": 0.0, "significant_fdr": True},
            ]
        )
        features = torch.ones((5, 3), dtype=torch.float32)
        labels = pd.DataFrame(
            [
                {"row_idx": i, "record_id": f"r{i}", "unit_text": f"utterance {i}", "A": int(i < 3)}
                for i in range(5)
            ]
        )
        records = labels.to_dict(orient="records")

        association_path = root / "association.csv"
        feature_path = root / "features.pt"
        label_path = root / "labels.csv"
        records_path = root / "records.jsonl"
        feature_filter_audit_path = root / "feature_filter_audit.csv"
        output_dir = root / "out"

        association.to_csv(association_path, index=False)
        torch.save({"utterance_features": features}, feature_path)
        labels.to_csv(label_path, index=False)
        _write_jsonl(records_path, records)
        pd.DataFrame(
            [
                {"latent_idx": 0, "keep": True},
                {"latent_idx": 1, "keep": True},
                {"latent_idx": 2, "keep": False},
            ]
        ).to_csv(feature_filter_audit_path, index=False)

        try:
            run_top20_cohensd_latent_utterance_export(
                association_path=association_path,
                feature_store_path=feature_path,
                label_matrix_path=label_path,
                records_path=records_path,
                output_dir=output_dir,
                feature_filter_audit_path=feature_filter_audit_path,
                labels=("A",),
                top_features=1,
                top_utterances=3,
            )
        except ValueError as exc:
            assert "keep=True" in str(exc)
        else:
            raise AssertionError("Expected dropped selected latent to be rejected")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_top20_cohensd_selection_and_utterance_export()
    test_feature_filter_audit_rejects_dropped_selected_latent()
    print("test_top20_cohensd_latent_utterances_smoke passed")
