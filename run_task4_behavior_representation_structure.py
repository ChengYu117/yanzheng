"""Run the automated, bottom-up Task 4 behavior-structure workflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.nlp_re_base.deepseek_top50_induction import DeepSeekTop50Config, run_deepseek_top50_tasks
from src.nlp_re_base.task4_behavior_structure import (
    GROUPING_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_card_coding_tasks,
    build_cross_label_task,
    build_label_grouping_tasks,
    build_report,
    validate_card_codings,
    validate_cross_label,
    validate_label_groupings,
)


DEFAULT_PACKAGE = Path("论文相关文档/latent_card_B级人工审查包")


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


def _run(tasks: Path, output: Path, system_prompt: str, args: argparse.Namespace, step: str) -> dict:
    key = _api_key()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not available")
    return run_deepseek_top50_tasks(
        tasks_path=tasks,
        output_dir=output,
        api_key=key,
        config=DeepSeekTop50Config(
            concurrency=args.concurrency,
            max_retries=args.max_retries,
            max_tokens=args.max_tokens,
        ),
        system_prompt=system_prompt,
        force=args.force,
        analysis_name="task4_behavior_representation_structure",
        step_name=step,
    )


def main() -> None:
    actions = (
        "build-codings", "run-codings", "validate-codings",
        "build-groupings", "run-groupings", "validate-groupings",
        "build-cross", "run-cross", "validate-cross", "report", "all",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=actions, nargs="?", default="all")
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output_dir or args.package_dir / "Task4_行为表征结构"
    coding_tasks = output / "llm_tasks" / "card_coding_tasks.jsonl"
    grouping_tasks = output / "llm_tasks" / "label_grouping_tasks.jsonl"
    cross_tasks = output / "llm_tasks" / "cross_label_task.jsonl"
    execution_manifest = output / "llm_execution_manifest.jsonl"
    results: dict[str, object] = {}

    if args.action in {"build-codings", "all"}:
        results["build_codings"] = build_card_coding_tasks(package_dir=args.package_dir, output_dir=output)
    if args.action in {"run-codings", "all"}:
        results["run_codings"] = _run(coding_tasks, output, SYSTEM_PROMPT, args, "run-task4-card-open-coding")
    if args.action in {"validate-codings", "all"}:
        results["validate_codings"] = validate_card_codings(tasks_path=coding_tasks, execution_manifest_path=execution_manifest, output_dir=output)
        if results["validate_codings"]["n_invalid"]:
            raise RuntimeError("Card-coding validation failed; inspect validation_audit.csv")
    if args.action in {"build-groupings", "all"}:
        results["build_groupings"] = build_label_grouping_tasks(package_dir=args.package_dir, output_dir=output)
    if args.action in {"run-groupings", "all"}:
        results["run_groupings"] = _run(grouping_tasks, output, GROUPING_SYSTEM_PROMPT, args, "run-task4-bottom-up-grouping")
    if args.action in {"validate-groupings", "all"}:
        results["validate_groupings"] = validate_label_groupings(tasks_path=grouping_tasks, execution_manifest_path=execution_manifest, output_dir=output)
        if results["validate_groupings"]["n_invalid"]:
            raise RuntimeError("Label-grouping validation failed; inspect validation_audit.csv")
    if args.action in {"build-cross", "all"}:
        results["build_cross"] = build_cross_label_task(output_dir=output)
    if args.action in {"run-cross", "all"}:
        results["run_cross"] = _run(cross_tasks, output, GROUPING_SYSTEM_PROMPT, args, "run-task4-cross-label-comparison")
    if args.action in {"validate-cross", "all"}:
        results["validate_cross"] = validate_cross_label(tasks_path=cross_tasks, execution_manifest_path=execution_manifest, output_dir=output)
        if not results["validate_cross"]["valid"]:
            raise RuntimeError("Cross-label validation failed; inspect cross_label_validation_manifest.json")
    if args.action in {"report", "all"}:
        results["report"] = build_report(package_dir=args.package_dir, output_dir=output)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
