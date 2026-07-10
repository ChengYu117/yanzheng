from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from src.nlp_re_base.contrastive_evidence_pack import write_jsonl
from src.nlp_re_base.contrastive_explainer import ExplainerTaskConfig, make_explainer_tasks
from src.nlp_re_base.contrastive_llm_io import validate_explainer_outputs
from src.nlp_re_base.contrastive_scorer import ScorerTaskConfig, make_scorer_tasks, validate_scorer_outputs


def _safe_smoke_root() -> Path:
    root = (Path.cwd() / "outputs" / "_smoke_contrastive_llm_io").resolve()
    cwd = Path.cwd().resolve()
    if cwd not in root.parents:
        raise RuntimeError(f"Refusing smoke output outside repo: {root}")
    return root


def _write_raw_json(path: Path, payload: object, *, fenced: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if fenced:
        text = "```json\n" + text + "\n```"
    path.write_text(text, encoding="utf-8")


def _toy_pack() -> dict:
    return {
        "packet_id": "ctli_0001",
        "latent_idx": 123,
        "target_label": "QUO",
        "rank_within_label": 1,
        "stable_set_role": "stable_core",
        "samples_for_explainer": [
            {"id": "s001", "tag": "ACTIVE_HIGH", "activation": 5.0, "text": "What would make this easier for you?"},
            {"id": "s002", "tag": "ACTIVE_MID", "activation": 2.0, "text": "How might that change help?"},
            {"id": "s003", "tag": "ACTIVE_LOW", "activation": 0.8, "text": "What else could matter?"},
            {"id": "s004", "tag": "NONACTIVE_NEAR_MISS", "activation": 0.1, "text": "Do you want to do that today?"},
            {"id": "s005", "tag": "NONACTIVE_RANDOM", "activation": 0.0, "text": "That sounds like a long week."},
        ],
        "heldout_internal_by_tag": {
            "ACTIVE_HIGH": [
                {
                    "id": "u001",
                    "tag": "ACTIVE_HIGH",
                    "row_idx": 1,
                    "activation": 4.0,
                    "activation_threshold": 1.0,
                    "ground_truth_activate": 1,
                    "text": "What would help you take the next step?",
                }
            ],
            "ACTIVE_MID": [
                {
                    "id": "u002",
                    "tag": "ACTIVE_MID",
                    "row_idx": 2,
                    "activation": 2.0,
                    "activation_threshold": 1.0,
                    "ground_truth_activate": 1,
                    "text": "How could you approach that conversation?",
                }
            ],
            "NONACTIVE_NEAR_MISS": [
                {
                    "id": "u003",
                    "tag": "NONACTIVE_NEAR_MISS",
                    "row_idx": 3,
                    "activation": 0.2,
                    "activation_threshold": 1.0,
                    "ground_truth_activate": 0,
                    "text": "Do you want to call today?",
                }
            ],
            "NONACTIVE_LABEL_MATCH": [
                {
                    "id": "u004",
                    "tag": "NONACTIVE_LABEL_MATCH",
                    "row_idx": 4,
                    "activation": 0.1,
                    "activation_threshold": 1.0,
                    "ground_truth_activate": 0,
                    "text": "What happened next?",
                }
            ],
        },
    }


def test_contrastive_llm_io() -> None:
    root = _safe_smoke_root()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        packs_path = root / "packs.jsonl"
        write_jsonl(packs_path, [_toy_pack()])
        task_manifest = make_explainer_tasks(
            packs_path=packs_path,
            output_dir=root,
            config=ExplainerTaskConfig(repeats_per_latent=2),
        )
        assert task_manifest["n_tasks"] == 2
        tasks = [json.loads(line) for line in (root / "llm_tasks" / "explainer_tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        assert "QUO" not in tasks[0]["prompt"]

        valid_explanation = {
            "latent_idx": 123,
            "short_name": "open invite",
            "main_hypothesis": "activates for open invitations that ask the client to elaborate on a possible next step",
            "positive_triggers": ["open what/how phrasing", "invites elaboration", "client choice or next step"],
            "explicit_exclusions": ["yes/no confirmation", "simple supportive statement"],
            "possible_surface_confounds": ["question mark", "second person"],
            "feature_type": "discourse",
            "confidence": 0.82,
            "alternative_hypotheses": ["next step exploration"],
            "key_evidence": ["s001", "s002"],
            "failure_modes": "brief what-questions without elaboration may be overpredicted",
        }
        _write_raw_json(Path(tasks[0]["expected_output_path"]), valid_explanation, fenced=True)
        validation = validate_explainer_outputs(
            tasks_path=root / "llm_tasks" / "explainer_tasks.jsonl",
            output_dir=root,
        )
        assert validation["n_valid"] == 1
        assert validation["n_retry"] == 1
        assert (root / "explainer_outputs" / "validated_explanations.jsonl").exists()
        assert (root / "explainer_outputs" / "retry_tasks.jsonl").exists()

        scorer_manifest = make_scorer_tasks(
            packs_path=packs_path,
            explanations_path=root / "explainer_outputs" / "validated_explanations.jsonl",
            output_dir=root,
            config=ScorerTaskConfig(),
        )
        assert scorer_manifest["n_scorer_tasks"] == 1
        assert scorer_manifest["n_baseline_tasks"] == 1
        scorer_tasks = [json.loads(line) for line in (root / "llm_tasks" / "scorer_tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        baseline_tasks = [json.loads(line) for line in (root / "llm_tasks" / "baseline_tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        forbidden_prompt_fragments = (
            "tag=",
            "ACTIVE_",
            "NONACTIVE",
            "activation=",
            "ground_truth",
            "row_idx",
            "internal_source",
        )
        for task in scorer_tasks + baseline_tasks:
            assert not any(fragment in task["prompt"] for fragment in forbidden_prompt_fragments)

        perfect_predictions = [
            {"sample_id": "u001", "pred_activate_prob": 0.95, "binary_prediction": 1, "evidence_span": "What would help", "reason": "matches open invitation"},
            {"sample_id": "u002", "pred_activate_prob": 0.85, "binary_prediction": 1, "evidence_span": "How could", "reason": "matches open invitation"},
            {"sample_id": "u003", "pred_activate_prob": 0.20, "binary_prediction": 0, "evidence_span": "Do you want", "reason": "closed confirmation"},
            {"sample_id": "u004", "pred_activate_prob": 0.10, "binary_prediction": 0, "evidence_span": "What happened", "reason": "too generic"},
        ]
        weak_baseline = [
            {"sample_id": "u001", "pred_activate_prob": 0.50, "binary_prediction": 0, "evidence_span": "", "reason": "unclear"},
            {"sample_id": "u002", "pred_activate_prob": 0.50, "binary_prediction": 0, "evidence_span": "", "reason": "unclear"},
            {"sample_id": "u003", "pred_activate_prob": 0.50, "binary_prediction": 0, "evidence_span": "", "reason": "unclear"},
            {"sample_id": "u004", "pred_activate_prob": 0.50, "binary_prediction": 0, "evidence_span": "", "reason": "unclear"},
        ]
        _write_raw_json(Path(scorer_tasks[0]["expected_output_path"]), perfect_predictions)
        _write_raw_json(Path(baseline_tasks[0]["expected_output_path"]), {"predictions": weak_baseline})

        scorer_validation = validate_scorer_outputs(
            scorer_tasks_path=root / "llm_tasks" / "scorer_tasks.jsonl",
            baseline_tasks_path=root / "llm_tasks" / "baseline_tasks.jsonl",
            answer_key_path=root / "scorer_outputs" / "heldout_answer_key.csv",
            output_dir=root,
        )
        assert scorer_validation["n_valid_scorer_predictions"] == 4
        metrics = pd.read_csv(root / "scorer_outputs" / "scorer_metrics.csv")
        assert metrics.iloc[0]["status"] == "accepted"
        assert metrics.iloc[0]["auroc"] == 1.0
        assert metrics.iloc[0]["latent_gap"] > 0
        latent_status = pd.read_csv(root / "scorer_outputs" / "latent_level_status.csv")
        assert latent_status.iloc[0]["latent_status"] == "accepted_single"
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_contrastive_llm_io()
    print("test_contrastive_llm_io_smoke passed")
