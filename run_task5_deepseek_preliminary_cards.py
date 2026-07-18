"""Build, run, and validate preliminary DeepSeek cards for Task 5 clusters."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from src.nlp_re_base.deepseek_top50_induction import DeepSeekTop50Config, run_deepseek_top50_tasks
from src.nlp_re_base.task5_deepseek_preliminary import (
    TASK5_CARD_SYSTEM_PROMPT,
    Task5DeepSeekCardConfig,
    build_task5_deepseek_tasks,
    render_task5_pca_human_review_document,
    render_task5_readable_card_document,
    render_task5_sae_human_review_document,
    validate_task5_deepseek_cards,
)


DEFAULT_TASK5_ROOT = Path(
    "outputs/misc_full_sae_eval/interpretability/task5_matched_sae_pca_human_eval"
)
DEFAULT_OUTPUT = DEFAULT_TASK5_ROOT / "deepseek_preliminary_cards"


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "run", "validate", "render", "all"), nargs="?", default="all")
    parser.add_argument("--task5-root", type=Path, default=DEFAULT_TASK5_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks-path", type=Path)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--representation-families",
        nargs="+",
        choices=("SAE", "PCA"),
        default=("SAE", "PCA"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    tasks_path = args.tasks_path or args.output_dir / "llm_tasks" / "task5_cluster_card_tasks.jsonl"
    results: dict[str, object] = {}
    if args.action in {"build", "all"}:
        results["build"] = build_task5_deepseek_tasks(
            task5_root=args.task5_root,
            output_dir=args.output_dir,
            config=Task5DeepSeekCardConfig(
                representation_families=tuple(args.representation_families)
            ),
        )
    if args.action in {"run", "all"}:
        key = _api_key()
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not available")
        results["run"] = run_deepseek_top50_tasks(
            tasks_path=tasks_path,
            output_dir=args.output_dir,
            api_key=key,
            config=DeepSeekTop50Config(
                model="deepseek-v4-flash",
                concurrency=args.concurrency,
                max_retries=args.max_retries,
                top_n=10,
            ),
            system_prompt=TASK5_CARD_SYSTEM_PROMPT,
            force=args.force,
            analysis_name="task5_deepseek_preliminary_cluster_cards",
            step_name="run-task5-preliminary-card-generation",
        )
    if args.action in {"validate", "all"}:
        results["validate"] = validate_task5_deepseek_cards(
            tasks_path=tasks_path,
            output_dir=args.output_dir,
        )
    if args.action in {"render", "all"}:
        render_outputs: dict[str, str] = {}
        if (args.output_dir / "codex_card_quality_review.csv").exists():
            render_outputs["document"] = str(
                render_task5_readable_card_document(output_dir=args.output_dir)
            )
        card_index = pd.read_csv(args.output_dir / "card_index_private.csv")
        representations = card_index["representation_internal"].astype(str)
        if representations.str.startswith("SAE").any():
            render_outputs["sae_human_review_document"] = str(
                render_task5_sae_human_review_document(output_dir=args.output_dir)
            )
        if representations.str.startswith("PCA").any():
            render_outputs["pca_human_review_document"] = str(
                render_task5_pca_human_review_document(output_dir=args.output_dir)
            )
        results["render"] = render_outputs
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
