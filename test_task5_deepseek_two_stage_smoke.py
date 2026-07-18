"""Smoke test for Task 5 two-stage DeepSeek task and score processing."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from src.nlp_re_base.task5_deepseek_two_stage import (
    build_generation_tasks,
    build_scoring_tasks,
    render_two_stage_family_review_document,
    render_two_stage_report,
    validate_generation_outputs,
    validate_scoring_outputs,
)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "task5"
        private = root / "private"
        private.mkdir(parents=True)
        mapping = pd.DataFrame(
            [
                {"arm": "pca_50", "blind_unit_id": "U1", "representation": "SAE", "label": "L", "index": 7, "direction_sign": 1},
                {"arm": "pca_100", "blind_unit_id": "U2", "representation": "SAE", "label": "L", "index": 7, "direction_sign": 1},
                {"arm": "pca_50", "blind_unit_id": "U3", "representation": "PCA-50", "label": "L", "index": 2, "direction_sign": -1},
            ]
        )
        mapping.to_csv(private / "blind_mapping_key.csv", index=False)
        rows = []
        for arm, unit in (("pca_50", "U1"), ("pca_100", "U2"), ("pca_50", "U3")):
            for i in range(10):
                rows.append(
                    {
                        "arm": arm,
                        "blind_unit_id": unit,
                        "phase": "discovery",
                        "source_order": i + 1,
                        "row_idx": i,
                        "unit_text": f"unique sample {unit} number {i}",
                    }
                )
        pd.DataFrame(rows).to_csv(private / "example_inventory.csv", index=False)
        output = root / "two_stage"
        built = build_generation_tasks(task5_root=root, output_dir=output)
        assert built["n_tasks"] == 2
        generation_tasks = [
            json.loads(line)
            for line in (output / "llm_tasks" / "generation_tasks.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert "置信度" in generation_tasks[0]["prompt"]
        assert "\"confidence\"" not in generation_tasks[0]["prompt"]
        for task in generation_tasks:
            ids = [sample["sample_id"] for sample in task["visible_samples"]]
            payload = {
                "anonymous_unit_id": task["anonymous_unit_id"],
                "short_name": "synthetic template",
                "primary_explanation": "Most sentences share a numbered synthetic template.",
                "candidate_behavioral_explanation": "现有句子不足以支持稳定的行为功能解释。",
                "explanation_type": "pragmatic_function" if task is generation_tasks[0] else "linguistic_structure",
                "supporting_sample_ids": ids[:8],
                "outlier_sample_ids": ids[8:],
                "representative_evidence_ids": ids[:2],
                "linguistic_evidence": [
                    {
                        "sample_id": ids[0],
                        "evidence": "shared numbered template",
                        "evidence_level": "surface_form",
                        "evidence_role": "support",
                    }
                ],
                "alternative_explanations": ["synthetic data"],
                "possible_confounds": ["numbering"],
                "limitations": ["ten samples"],
            }
            Path(task["expected_output_path"]).write_text(json.dumps(payload), encoding="utf-8")
        generation_validation = validate_generation_outputs(
            tasks_path=output / "llm_tasks" / "generation_tasks.jsonl",
            output_dir=output,
        )
        assert generation_validation["status"] == "PASS"
        generation_audit = pd.read_csv(output / "generation_structure_audit.csv")
        assert generation_audit["normalizations"].fillna("").str.contains(
            "pragmatic_function->behavioral_function"
        ).sum() == 1
        scoring_build = build_scoring_tasks(output_dir=output)
        assert scoring_build["n_tasks"] == 2
        scoring_tasks = [
            json.loads(line)
            for line in (output / "llm_tasks" / "scoring_tasks.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert "supporting_sample_ids" not in scoring_tasks[0]["prompt"]
        assert "完整数据集中的选择特异性" in scoring_tasks[0]["prompt"]
        for task in scoring_tasks:
            judgments = []
            for i, sample in enumerate(task["visible_samples"]):
                judgments.append(
                    {
                        "sample_id": sample["sample_id"],
                        "verdict": "supports" if i < 8 else "partial",
                        "evidence": "numbered synthetic wording",
                    }
                )
            payload = {
                "anonymous_unit_id": task["anonymous_unit_id"],
                "evaluation_scope": "within_cluster_explanation_quality_only",
                "sample_judgments": judgments,
                "within_cluster_fit_summary": "The frozen explanation fits the shared template.",
                "unsupported_within_cluster_claims": [],
                "within_cluster_coherence": 5,
                "claim_specificity": 4,
                "evidence_grounding": 5,
                "score_rationales": {
                    "within_cluster_coherence": "All samples share the form.",
                    "claim_specificity": "The numbered template is specific.",
                    "evidence_grounding": "The wording is directly visible.",
                },
            }
            Path(task["expected_output_path"]).write_text(json.dumps(payload), encoding="utf-8")
        scoring_validation = validate_scoring_outputs(
            tasks_path=output / "llm_tasks" / "scoring_tasks.jsonl",
            output_dir=output,
        )
        assert scoring_validation["status"] == "PASS"
        scores = pd.read_csv(output / "final_scores.csv")
        assert set(scores["overall_score"]) == {4}
        assert set(scores["equivalent_coverage"]) == {9.0}
        report = render_two_stage_report(output_dir=output)
        text = report.read_text(encoding="utf-8")
        assert "两阶段句子簇解释与评分报告" in text
        assert "SAE" in text and "PCA" in text
        assert "程序总分" in text
        pca_review = render_two_stage_family_review_document(
            output_dir=output, representation_family="PCA"
        ).read_text(encoding="utf-8")
        sae_review = render_two_stage_family_review_document(
            output_dir=output, representation_family="SAE"
        ).read_text(encoding="utf-8")
        assert "PCA 两阶段解释 Card 人工审查文档" in pca_review
        assert "SAE 两阶段解释 Card 人工审查文档" in sae_review
        assert "完整句子簇与逐句判断" in pca_review
        assert "unique sample U3" in pca_review
        assert "unique sample U2" in sae_review
        assert "unique sample U3" not in sae_review
    print("task5 two-stage DeepSeek smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
