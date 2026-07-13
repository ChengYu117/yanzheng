from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.nlp_re_base.contrastive_evidence_pack import read_jsonl, write_jsonl
from src.nlp_re_base.zhipu_single_latent_p3 import (
    make_single_minimal_pair_task,
    make_top50_explainer_task,
    prepare_single_latent_pack,
    validate_single_minimal_pair_output,
    validate_top50_explainer_output,
)


def _write_inputs(root: Path) -> dict[str, Path]:
    n_rows = 160
    records = []
    for idx in range(n_rows):
        text = f"counselor utterance unique {idx}"
        if idx in {0, 1}:
            text = "shared positive evaluation template"
        records.append({"record_id": f"r{idx}", "file_id": f"f{idx // 4}", "unit_text": text})
    records_path = root / "records.jsonl"
    write_jsonl(records_path, records)

    labels = pd.DataFrame(
        {
            "row_idx": np.arange(n_rows),
            "record_id": [f"r{idx}" for idx in range(n_rows)],
            "file_id": [f"f{idx // 4}" for idx in range(n_rows)],
            "unit_text": [row["unit_text"] for row in records],
            "RE": 0,
            "RES": 0,
            "REC": 0,
            "QU": 0,
            "QUO": 0,
            "QUC": 0,
            "GI": 0,
            "SU": 0,
            "AF": [1 if idx < 35 or idx >= 140 else 0 for idx in range(n_rows)],
        }
    )
    labels_path = root / "labels.csv"
    labels.to_csv(labels_path, index=False)

    features = np.zeros((n_rows, 4), dtype=np.float32)
    features[:, 2] = np.linspace(3.0, 0.1, n_rows, dtype=np.float32)
    features_path = root / "features.npy"
    np.save(features_path, features)

    stable = pd.DataFrame(
        [
            {
                "label": "AF",
                "latent_idx": 2,
                "rank_within_label": 1,
                "stable_set_role": "stable_core",
                "label_selection_status": "stable_topk_found",
                "auc": 0.8,
                "directional_auc": 0.8,
                "cohens_d": 2.0,
                "precision_at_50": 0.7,
                "inclusion_frequency": 1.0,
            }
        ]
    )
    stable_path = root / "stable.csv"
    stable.to_csv(stable_path, index=False)
    return {
        "records": records_path,
        "labels": labels_path,
        "features": features_path,
        "stable": stable_path,
    }


def test_single_latent_top50(root: Path) -> None:
    inputs = _write_inputs(root)
    output_dir = root / "run"
    prep = prepare_single_latent_pack(
        stable_latents_path=inputs["stable"],
        feature_store_path=inputs["features"],
        label_matrix_path=inputs["labels"],
        records_path=inputs["records"],
        output_dir=output_dir,
        target_label="AF",
        latent_idx=2,
    )
    assert prep["top50_duplicate_rows"] == 1
    assert prep["top50_heldout_text_disjoint"] is True
    pack_path = output_dir / "evidence_packs" / "single_latent_top50_pack.jsonl"
    pack = read_jsonl(pack_path)[0]
    assert len(pack["samples_for_explainer"]) == 50
    assert sum(len(rows) for rows in pack["heldout_internal_by_tag"].values()) == 20

    task_manifest = make_top50_explainer_task(packs_path=pack_path, output_dir=output_dir)
    task_path = Path(task_manifest["outputs"]["tasks"])
    task = read_jsonl(task_path)[0]
    assert "Reason across all 50 utterances" in task["prompt"]
    support = [f"s{idx:03d}" for idx in range(1, 31)]
    outliers = [f"s{idx:03d}" for idx in range(31, 51)]
    payload = {
        "latent_idx": 2,
        "short_name": "positive evaluative wording",
        "main_hypothesis": "The candidate pattern is recurring positive evaluative wording applied directly to an idea or event.",
        "dominant_pattern": "Direct positive evaluation",
        "semantic_component": "The speaker positively evaluates an idea, outcome, or action.",
        "surface_component": "Short adjective-led evaluative constructions recur but are not required.",
        "positive_triggers": ["direct positive evaluation", "brief approval of an idea"],
        "explicit_exclusions": ["neutral factual statements"],
        "possible_surface_confounds": ["repeated adjective templates"],
        "feature_type": "mixed",
        "supporting_sample_ids": support,
        "outlier_sample_ids": outliers,
        "representative_evidence_ids": support[:5],
        "alternative_hypotheses": ["short conversational acknowledgements"],
        "failure_modes": ["may miss longer paraphrases"],
        "confidence": 0.8,
    }
    raw_path = Path(task["expected_output_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(payload), encoding="utf-8")
    execution_path = output_dir / "llm_execution_manifest.jsonl"
    write_jsonl(
        execution_path,
        [
            {
                "task_id": task["task_id"],
                "status": "success",
                "execution_mode": "zhipu_api",
                "model": "glm-4.7",
                "prompt_sha256": hashlib.sha256(task["prompt"].encode("utf-8")).hexdigest(),
                "raw_output_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            }
        ],
    )
    validation = validate_top50_explainer_output(
        tasks_path=task_path, execution_manifest_path=execution_path, output_dir=output_dir
    )
    assert validation["n_valid"] == 1
    explanation_path = output_dir / "explainer_outputs" / "validated_explanations.jsonl"
    explanation = read_jsonl(explanation_path)[0]
    assert explanation["raw_support_fraction"] == 0.6
    assert explanation["unique_support_fraction"] > 0.5

    minimal_manifest = make_single_minimal_pair_task(
        explanations_path=explanation_path, packs_path=pack_path, output_dir=output_dir
    )
    minimal_task_path = Path(minimal_manifest["outputs"]["tasks"])
    minimal_task = read_jsonl(minimal_task_path)[0]
    pairs = []
    for idx in range(5):
        pairs.append(
            {
                "positive_text": f"That is a great plan for step {idx}",
                "negative_text": f"That is a plan for step {idx}",
                "changed_factor": "positive evaluation",
                "held_constant": "topic, syntax, speaker role, and step number",
                "test_dimension": "semantic",
                "expected_direction": "positive_greater_than_negative",
            }
        )
    minimal_raw = Path(minimal_task["expected_output_path"])
    minimal_raw.parent.mkdir(parents=True, exist_ok=True)
    minimal_raw.write_text(json.dumps(pairs), encoding="utf-8")
    pair_validation = validate_single_minimal_pair_output(tasks_path=minimal_task_path, output_dir=output_dir)
    assert pair_validation["quality_pass"] is True
    assert pair_validation["n_pairs"] == 5


def main() -> None:
    root = (Path.cwd() / "outputs" / "_smoke_zhipu_single_latent_p3").resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        test_single_latent_top50(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("test_zhipu_single_latent_p3_smoke passed")


if __name__ == "__main__":
    main()
