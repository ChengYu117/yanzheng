"""Run two-stage DeepSeek explanation generation and independent scoring."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.nlp_re_base.deepseek_top50_induction import DeepSeekTop50Config, run_deepseek_top50_tasks
from src.nlp_re_base.task5_deepseek_two_stage import (
    GENERATION_SYSTEM_PROMPT,
    SCORING_SYSTEM_PROMPT,
    Task5TwoStageConfig,
    build_generation_tasks,
    build_scoring_tasks,
    render_two_stage_family_review_document,
    render_two_stage_report,
    validate_generation_outputs,
    validate_scoring_outputs,
)


DEFAULT_TASK5_ROOT = Path(
    "outputs/misc_full_sae_eval_min5_words/interpretability/task5_matched_sae_pca_human_eval"
)
DEFAULT_OUTPUT = DEFAULT_TASK5_ROOT / "deepseek_two_stage_cards"


def _api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key or os.name != "nt":
        return key
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as handle:
            value, _ = winreg.QueryValueEx(handle, "DEEPSEEK_API_KEY")
        return str(value).strip()
    except (FileNotFoundError, OSError):
        return ""


def _run_phase(
    *,
    tasks_path: Path,
    execution_dir: Path,
    api_key: str,
    system_prompt: str,
    temperature: float,
    concurrency: int,
    max_retries: int,
    force: bool,
    phase: str,
) -> dict[str, object]:
    return run_deepseek_top50_tasks(
        tasks_path=tasks_path,
        output_dir=execution_dir,
        api_key=api_key,
        config=DeepSeekTop50Config(
            model="deepseek-v4-flash",
            concurrency=concurrency,
            max_retries=max_retries,
            temperature=temperature,
            top_n=10,
        ),
        system_prompt=system_prompt,
        force=force,
        analysis_name="task5_deepseek_two_stage",
        step_name=f"run-{phase}",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "build-generation",
            "run-generation",
            "validate-generation",
            "build-scoring",
            "run-scoring",
            "validate-scoring",
            "render",
            "all",
        ),
        nargs="?",
        default="all",
    )
    parser.add_argument("--task5-root", type=Path, default=DEFAULT_TASK5_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    generation_tasks = args.output_dir / "llm_tasks" / "generation_tasks.jsonl"
    scoring_tasks = args.output_dir / "llm_tasks" / "scoring_tasks.jsonl"
    results: dict[str, object] = {}
    if args.action in {"build-generation", "all"}:
        results["build_generation"] = build_generation_tasks(
            task5_root=args.task5_root,
            output_dir=args.output_dir,
            config=Task5TwoStageConfig(representation_families=("SAE", "PCA")),
        )
    if args.action in {"run-generation", "run-scoring", "all"}:
        key = _api_key()
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not available")
    else:
        key = ""
    if args.action in {"run-generation", "all"}:
        results["run_generation"] = _run_phase(
            tasks_path=generation_tasks,
            execution_dir=args.output_dir / "generation_execution",
            api_key=key,
            system_prompt=GENERATION_SYSTEM_PROMPT,
            temperature=0.2,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
            force=args.force,
            phase="generation",
        )
    if args.action in {"validate-generation", "all"}:
        results["validate_generation"] = validate_generation_outputs(
            tasks_path=generation_tasks, output_dir=args.output_dir
        )
        if results["validate_generation"]["status"] != "PASS":
            raise RuntimeError("Generation validation did not pass; scoring was not started")
    if args.action in {"build-scoring", "all"}:
        results["build_scoring"] = build_scoring_tasks(output_dir=args.output_dir)
    if args.action in {"run-scoring", "all"}:
        results["run_scoring"] = _run_phase(
            tasks_path=scoring_tasks,
            execution_dir=args.output_dir / "scoring_execution",
            api_key=key,
            system_prompt=SCORING_SYSTEM_PROMPT,
            temperature=0.0,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
            force=args.force,
            phase="scoring",
        )
    if args.action in {"validate-scoring", "all"}:
        results["validate_scoring"] = validate_scoring_outputs(
            tasks_path=scoring_tasks, output_dir=args.output_dir
        )
        if results["validate_scoring"]["status"] != "PASS":
            raise RuntimeError("Scoring validation did not pass")
    if args.action in {"render", "all"}:
        results["report"] = str(render_two_stage_report(output_dir=args.output_dir))
        results["pca_review_document"] = str(
            render_two_stage_family_review_document(
                output_dir=args.output_dir, representation_family="PCA"
            )
        )
        results["sae_review_document"] = str(
            render_two_stage_family_review_document(
                output_dir=args.output_dir, representation_family="SAE"
            )
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
