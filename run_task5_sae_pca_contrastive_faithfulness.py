"""Run sampled SAE/PCA contrastive explanation and held-out scoring with Codex."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.nlp_re_base.contrastive_faithfulness_v2 import (
    audit_frozen_packet_integrity,
    make_explainer_retry_tasks,
    make_scorer_retry_tasks,
    run_stage,
    validate_explanations,
)
from src.nlp_re_base.task5_sae_pca_contrastive_faithfulness import (
    build_packets,
    make_reduced_context_scorer_reassessment,
    make_scorer_tasks,
    render_report,
    validate_scorer_and_score,
)


def _status(output: Path, payload: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "pipeline_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "run-all", "run-explainer", "validate-explainer", "make-scorer", "make-scorer-reassessment", "make-scorer-retry", "run-scorer-retry", "run-scorer", "validate-scorer", "render"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-output-dir", type=Path)
    parser.add_argument("--task5-root", type=Path, default=Path("outputs/rerun_new_dataset_20260716/min5_words/interpretability/task5_matched_sae_pca_human_eval_n20"))
    parser.add_argument("--sae-features", default="outputs/rerun_new_dataset_20260716/min5_words/feature_store/utterance_features.pt")
    parser.add_argument("--raw-hidden", default="outputs/rerun_new_dataset_20260716/min5_words/feature_store/utterance_activations.pt")
    parser.add_argument("--label-matrix", default="outputs/rerun_new_dataset_20260716/min5_words/label_matrix.csv")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    args = parser.parse_args(); output = args.output_dir

    def build():
        return build_packets(task5_root=args.task5_root, sae_features_path=args.sae_features, raw_hidden_path=args.raw_hidden, label_matrix_path=args.label_matrix, output_dir=output)

    def run_explainer():
        return run_stage(stage_dir=output / "explainer", tasks_path=output / "explainer" / "tasks.jsonl", schema_path="config/contrastive_explainer_v2_schema.json", instructions_path="config/contrastive_explainer_v2_base_instructions.txt", model=args.model, reasoning_effort=args.reasoning_effort, concurrency=args.concurrency, timeout_seconds=args.timeout_seconds)

    def run_scorer():
        return run_stage(stage_dir=output / "scorer", tasks_path=output / "scorer" / "tasks.jsonl", schema_path="config/contrastive_scorer_v2_schema.json", instructions_path="config/contrastive_scorer_v2_base_instructions.txt", model=args.model, reasoning_effort=args.reasoning_effort, concurrency=args.concurrency, timeout_seconds=args.timeout_seconds)

    if args.action == "build": result = build()
    elif args.action == "run-explainer": result = run_explainer()
    elif args.action == "validate-explainer": result = validate_explanations(output_dir=output)
    elif args.action == "make-scorer": result = make_scorer_tasks(output_dir=output)
    elif args.action == "make-scorer-reassessment":
        if args.source_output_dir is None:
            parser.error("--source-output-dir is required for make-scorer-reassessment")
        result = make_reduced_context_scorer_reassessment(source_output_dir=args.source_output_dir, output_dir=output)
    elif args.action == "make-scorer-retry": result = make_scorer_retry_tasks(output_dir=output)
    elif args.action == "run-scorer-retry":
        result = run_stage(stage_dir=output / "scorer_retry", tasks_path=output / "scorer" / "retry_tasks.jsonl", schema_path="config/contrastive_scorer_v2_schema.json", instructions_path="config/contrastive_scorer_v2_base_instructions.txt", model=args.model, reasoning_effort=args.reasoning_effort, concurrency=args.concurrency, timeout_seconds=args.timeout_seconds)
    elif args.action == "run-scorer": result = run_scorer()
    elif args.action == "validate-scorer": result = validate_scorer_and_score(output_dir=output)
    elif args.action == "render":
        result = render_report(output_dir=output)
        explainer = json.loads((output / "explainer" / "validation_manifest.json").read_text(encoding="utf-8"))
        scorer = json.loads((output / "scorer" / "validation_manifest.json").read_text(encoding="utf-8"))
        _status(output, {"stage": "complete", "explainer_validation": explainer, "scorer_validation": scorer, "report": result})
    else:
        if not (output / "explainer" / "tasks.jsonl").exists(): build()
        audit_frozen_packet_integrity(output_dir=output, expected_strata=("high", "mid", "weak", "control"))
        _status(output, {"stage": "explainer_running"}); run_explainer()
        validation = validate_explanations(output_dir=output)
        if validation["n_failed"]:
            retry = make_explainer_retry_tasks(output_dir=output)
            if retry["n_retry_tasks"]:
                _status(output, {"stage": "explainer_retry_running", **retry})
                run_stage(stage_dir=output / "explainer_retry", tasks_path=output / "explainer" / "retry_tasks.jsonl", schema_path="config/contrastive_explainer_v2_schema.json", instructions_path="config/contrastive_explainer_v2_base_instructions.txt", model=args.model, reasoning_effort=args.reasoning_effort, concurrency=args.concurrency, timeout_seconds=args.timeout_seconds)
                validation = validate_explanations(output_dir=output)
        _status(output, {"stage": "scorer_running", "explainer_validation": validation})
        make_scorer_tasks(output_dir=output); run_scorer()
        scorer = validate_scorer_and_score(output_dir=output)
        if scorer["n_failed"]:
            retry = make_scorer_retry_tasks(output_dir=output)
            if retry["n_retry_tasks"]:
                _status(output, {"stage": "scorer_retry_running", **retry})
                run_stage(stage_dir=output / "scorer_retry", tasks_path=output / "scorer" / "retry_tasks.jsonl", schema_path="config/contrastive_scorer_v2_schema.json", instructions_path="config/contrastive_scorer_v2_base_instructions.txt", model=args.model, reasoning_effort=args.reasoning_effort, concurrency=args.concurrency, timeout_seconds=args.timeout_seconds)
                scorer = validate_scorer_and_score(output_dir=output)
        report = render_report(output_dir=output) if scorer["n_valid"] else {}
        result = {"explainer_validation": validation, "scorer_validation": scorer, "report": report}
        _status(output, {"stage": "complete", **result})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
