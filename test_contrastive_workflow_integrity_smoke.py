from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from run_misc_contrastive_latent_interp import write_report
from src.nlp_re_base.contrastive_evidence_pack import (
    PRIMARY_LABELS,
    _append_samples,
    normalise_text,
    write_jsonl,
)
from src.nlp_re_base.contrastive_llm_io import validate_prediction_outputs
from src.nlp_re_base.contrastive_minimal_pairs import make_minimal_pair_tasks
from src.nlp_re_base.contrastive_scorer import _metrics_for_predictions
from src.nlp_re_base.contrastive_subconcepts import make_subconcept_tasks


def _safe_smoke_root() -> Path:
    root = (Path.cwd() / "outputs" / "_smoke_contrastive_workflow_integrity").resolve()
    if Path.cwd().resolve() not in root.parents:
        raise RuntimeError(f"Refusing smoke output outside repo: {root}")
    return root


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_normalized_text_deduplication() -> None:
    label_matrix = pd.DataFrame(
        [
            {"unit_text": "Hello, WORLD!", "QU": 1},
            {"unit_text": " hello world ", "QU": 1},
            {"unit_text": "A distinct sample", "QU": 0},
        ]
    )
    records = label_matrix.to_dict(orient="records")
    selected: list[dict] = []
    _append_samples(
        selected=selected,
        used_rows=set(),
        used_texts=set(),
        id_state={"next": 1},
        row_indices=[0, 1, 2],
        tag="ACTIVE_HIGH",
        internal_source="smoke",
        target_label="QU",
        latent_idx=0,
        activations=np.asarray([3.0, 2.0, 1.0]),
        threshold=1.5,
        label_matrix=label_matrix,
        records=records,
        labels=("QU",),
        heldout=False,
        prefix="s",
        max_count=3,
    )
    normalized = [normalise_text(row["text"]) for row in selected]
    assert normalise_text("Hello, WORLD!") == normalise_text(" hello world ")
    assert len(selected) == 2
    assert len(normalized) == len(set(normalized))


def test_duplicate_prediction_ids_are_rejected(root: Path) -> None:
    raw_path = root / "duplicate_predictions.json"
    _write_json(
        raw_path,
        [
            {"sample_id": "u001", "pred_activate_prob": 0.9, "binary_prediction": 1},
            {"sample_id": "u001", "pred_activate_prob": 0.1, "binary_prediction": 0},
        ],
    )
    tasks = [{"task_id": "t1", "packet_id": "p1", "latent_idx": 1, "expected_output_path": str(raw_path)}]
    answer_key = pd.DataFrame(
        [
            {"task_id": "t1", "sample_id": "u001", "ground_truth_activate": 1, "target_label": "QU"},
            {"task_id": "t1", "sample_id": "u002", "ground_truth_activate": 0, "target_label": "QU"},
        ]
    )
    valid, errors, retry = validate_prediction_outputs(
        tasks=tasks,
        answer_key=answer_key,
        raw_kind="explanation_scorer",
    )
    assert not valid
    assert len(errors) == 1 and "duplicate sample_id" in errors[0]["error"]
    assert retry == tasks


def test_single_class_metric_is_invalid() -> None:
    predictions = pd.DataFrame(
        [
            {
                "task_id": "single",
                "packet_id": "p1",
                "latent_idx": 1,
                "target_label": "QU",
                "explanation_task_id": "e1",
                "ground_truth_activate": 1,
                "pred_activate_prob": 0.9,
                "binary_prediction": 1,
            },
            {
                "task_id": "single",
                "packet_id": "p1",
                "latent_idx": 1,
                "target_label": "QU",
                "explanation_task_id": "e1",
                "ground_truth_activate": 1,
                "pred_activate_prob": 0.8,
                "binary_prediction": 1,
            },
        ]
    )
    metrics = _metrics_for_predictions(predictions, kind="explanation_scorer")
    assert not bool(metrics.iloc[0]["metric_valid"])
    assert pd.isna(metrics.iloc[0]["auroc"])
    assert "both ground-truth classes" in metrics.iloc[0]["invalid_reason"]


def _fixed_plan_rows() -> tuple[list[dict], list[dict]]:
    packs: list[dict] = []
    explanations: list[dict] = []
    latent_idx = 100
    for label in PRIMARY_LABELS:
        for rank in range(1, 4):
            packet_id = f"{label.lower()}_{rank}"
            packs.append(
                {
                    "packet_id": packet_id,
                    "target_label": label,
                    "latent_idx": latent_idx,
                    "rank_within_label": rank,
                    "inclusion_frequency": 1.0 - rank / 10,
                }
            )
            explanations.append(
                {
                    "task_id": f"{packet_id}_explainer_r01",
                    "packet_id": packet_id,
                    "target_label": label,
                    "latent_idx": latent_idx,
                    "short_name": "candidate pattern",
                    "main_hypothesis": "candidate functional wording pattern",
                    "positive_triggers": ["functional cue"],
                    "explicit_exclusions": ["contrast cue"],
                    "possible_surface_confounds": ["length"],
                    "feature_type": "discourse",
                    "failure_modes": "context may change the interpretation",
                    "confidence": 0.7,
                }
            )
            latent_idx += 1
    return packs, explanations


def test_fixed_minimal_pair_plan(root: Path) -> tuple[list[dict], list[dict]]:
    packs, explanations = _fixed_plan_rows()
    packs_path = root / "fixed_plan_packs.jsonl"
    explanations_path = root / "fixed_plan_explanations.jsonl"
    write_jsonl(packs_path, packs)
    write_jsonl(explanations_path, explanations)
    manifest = make_minimal_pair_tasks(
        packs_path=packs_path,
        explanations_path=explanations_path,
        scorer_metrics_path=None,
        output_dir=root / "minimal_plan",
    )
    assert manifest["expected_tasks"] == 18
    assert manifest["n_tasks"] == 18
    assert manifest["plan_complete"] is True
    tasks = [json.loads(line) for line in Path(manifest["outputs"]["minimal_pair_designer_tasks"]).read_text(encoding="utf-8").splitlines()]
    assert len(tasks) == 18
    assert {task["target_label"] for task in tasks} == set(PRIMARY_LABELS)
    return packs, explanations


def test_subconcept_gate_uses_distinct_stable_latents(root: Path, packs: list[dict], explanations: list[dict]) -> None:
    selected_packs = [pack for pack in packs if pack["target_label"] == "QU"]
    selected_ids = {pack["packet_id"] for pack in selected_packs}
    selected_explanations = [row for row in explanations if row["packet_id"] in selected_ids]
    duplicate = dict(selected_explanations[0])
    duplicate["task_id"] = duplicate["task_id"].replace("r01", "r02")
    selected_explanations.append(duplicate)

    packs_path = root / "subconcept_packs.jsonl"
    explanations_path = root / "subconcept_explanations.jsonl"
    metrics_path = root / "subconcept_metrics.csv"
    status_path = root / "latent_status.csv"
    write_jsonl(packs_path, selected_packs)
    write_jsonl(explanations_path, selected_explanations)
    pd.DataFrame(
        [
            {
                "explanation_task_id": row["task_id"],
                "target_label": row["target_label"],
                "latent_idx": row["latent_idx"],
                "auroc": 0.8,
                "latent_gap": 0.2,
            }
            for row in selected_explanations
        ]
    ).to_csv(metrics_path, index=False)
    pd.DataFrame(
        [
            {
                "packet_id": pack["packet_id"],
                "target_label": pack["target_label"],
                "latent_idx": pack["latent_idx"],
                "latent_status": "accepted_stable",
            }
            for pack in selected_packs
        ]
    ).to_csv(status_path, index=False)

    output_dir = root / "subconcept_gate"
    manifest = make_subconcept_tasks(
        packs_path=packs_path,
        explanations_path=explanations_path,
        scorer_metrics_path=metrics_path,
        latent_status_path=status_path,
        output_dir=output_dir,
    )
    cluster_input = pd.read_csv(output_dir / "subconcepts" / "subconcept_cluster_input.csv")
    assert manifest["n_tasks"] == 1
    assert manifest["n_input_rows"] == 3
    assert manifest["n_distinct_latents"] == 3
    assert cluster_input[["target_label", "latent_idx"]].drop_duplicates().shape[0] == 3


def test_partial_report_is_truthful(root: Path) -> None:
    output_dir = root / "partial_report"
    report_path = write_report(output_dir)
    report = report_path.read_text(encoding="utf-8")
    status = pd.read_csv(output_dir / "workflow_stage_status.csv")
    assert set(status["status"]).issubset({"not_started", "blocked", "partial", "complete"})
    assert status.loc[status["step"] == 1, "status"].iloc[0] == "not_started"
    assert status.loc[status["step"] == 4, "status"].iloc[0] == "blocked"
    assert "尚无 SAE 前向测试结果" in report
    assert "本报告不主张 minimal-pair 功能敏感性证据" in report


def test_contrastive_workflow_integrity() -> None:
    root = _safe_smoke_root()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        test_normalized_text_deduplication()
        test_duplicate_prediction_ids_are_rejected(root)
        test_single_class_metric_is_invalid()
        packs, explanations = test_fixed_minimal_pair_plan(root)
        test_subconcept_gate_uses_distinct_stable_latents(root, packs, explanations)
        test_partial_report_is_truthful(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_contrastive_workflow_integrity()
    print("test_contrastive_workflow_integrity_smoke passed")
