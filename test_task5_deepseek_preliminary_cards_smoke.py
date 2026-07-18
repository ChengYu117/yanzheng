"""Smoke test for Task 5 DeepSeek task construction and validation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from src.nlp_re_base.task5_deepseek_preliminary import (
    Task5DeepSeekCardConfig,
    build_task5_deepseek_tasks,
    render_task5_pca_human_review_document,
    render_task5_readable_card_document,
    render_task5_sae_human_review_document,
    validate_task5_deepseek_cards,
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
                rows.append({"arm": arm, "blind_unit_id": unit, "phase": "discovery", "source_order": i + 1, "row_idx": i, "unit_text": f"unique sample {unit} number {i}"})
        pd.DataFrame(rows).to_csv(private / "example_inventory.csv", index=False)
        output = root / "deepseek"
        built = build_task5_deepseek_tasks(task5_root=root, output_dir=output)
        assert built["n_source_unit_instances"] == 3
        assert built["n_unique_cluster_tasks"] == 2
        tasks = [json.loads(line) for line in (output / "llm_tasks" / "task5_cluster_card_tasks.jsonl").read_text(encoding="utf-8").splitlines()]
        for task in tasks:
            ids = [sample["sample_id"] for sample in task["visible_samples"]]
            payload = {
                "anonymous_unit_id": task["anonymous_unit_id"],
                "short_name": "synthetic pattern",
                "primary_explanation": "Most samples share a synthetic numbered template.",
                "candidate_behavioral_explanation": "The available sentences do not support a stable behavioral-function interpretation.",
                "explanation_type": "linguistic_structure",
                "supporting_sample_ids": ids[:8],
                "outlier_sample_ids": ids[8:],
                "representative_evidence_ids": ids[:2],
                "linguistic_evidence": [{"sample_id": ids[0], "evidence": "shared template", "evidence_level": "surface_form", "evidence_role": "support"}],
                "alternative_explanations": ["numbered examples"],
                "possible_confounds": ["synthetic data"],
                "limitations": ["ten examples"],
                "confidence": 4,
                "confidence_rationale": "Eight examples share the same form.",
            }
            path = Path(task["expected_output_path"])
            path.write_text(json.dumps(payload), encoding="utf-8")
        validated = validate_task5_deepseek_cards(
            tasks_path=output / "llm_tasks" / "task5_cluster_card_tasks.jsonl",
            output_dir=output,
        )
        assert validated["status"] == "PASS"
        assert validated["n_valid"] == 2
        pd.DataFrame(
            [
                {
                    "anonymous_unit_id": task["anonymous_unit_id"],
                    "representation_family": "SAE" if "SAE" in task["task_id"] else "PCA",
                    "grade": "A",
                    "pattern_supported": "yes",
                    "boundary_quality": "good",
                    "overclaim_risk": "low",
                    "review_note": "Synthetic review note.",
                }
                for task in tasks
            ]
        ).to_csv(output / "codex_card_quality_review.csv", index=False)
        document = render_task5_readable_card_document(output_dir=output)
        rendered = document.read_text(encoding="utf-8")
        assert "Task 5 DeepSeek 初步解释卡片全集" in rendered
        assert "原始 discovery 语句" in rendered
        assert tasks[0]["anonymous_unit_id"] in rendered
        sae_document = render_task5_sae_human_review_document(output_dir=output)
        sae_rendered = sae_document.read_text(encoding="utf-8")
        assert "Task 5 SAE latent 归纳人工审核" in sae_rendered
        assert "## Latent 7" in sae_rendered
        assert "unique sample" in sae_rendered
        assert "PCA" not in sae_rendered
        pca_document = render_task5_pca_human_review_document(output_dir=output)
        pca_rendered = pca_document.read_text(encoding="utf-8")
        assert "Task 5 PCA latent 归纳人工审核" in pca_rendered
        assert "## Latent 2" in pca_rendered
        assert "unique sample U3 number 0" in pca_rendered
        pca_only_output = root / "pca_only_cards"
        pca_only = build_task5_deepseek_tasks(
            task5_root=root,
            output_dir=pca_only_output,
            config=Task5DeepSeekCardConfig(representation_families=("PCA",)),
        )
        assert pca_only["n_unique_cluster_tasks"] == 1
        pca_only_index = pd.read_csv(pca_only_output / "card_index_private.csv")
        assert pca_only_index["representation_internal"].str.startswith("PCA").all()
    print("task5 DeepSeek preliminary-card smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
