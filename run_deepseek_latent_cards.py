"""Build, run, and validate DeepSeek latent interpretation cards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.nlp_re_base.deepseek_latent_cards import (
    LATENT_CARD_SYSTEM_PROMPT,
    build_latent_card_tasks,
    build_refined_latent_card_retry_tasks,
    render_stable_core_top5_human_review_document,
    validate_latent_card_outputs,
)
from src.nlp_re_base.deepseek_top50_induction import DeepSeekTop50Config, run_deepseek_top50_tasks


DEFAULT_OUTPUT = Path(
    "outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/"
    "deepseek_v4_flash_latent_cards"
)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "run", "validate", "retry", "render-review", "all"), nargs="?", default="all")
    parser.add_argument("--stable-latents", default="outputs/cross_val/stable_topk_selection/stable_topk_latent_set.csv")
    parser.add_argument("--feature-store", default="outputs/misc_full_sae_eval/feature_store/utterance_features.pt")
    parser.add_argument("--records", default="outputs/misc_full_sae_eval/records.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks-path", type=Path)
    parser.add_argument("--concurrency", type=int, default=225)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = DeepSeekTop50Config(concurrency=args.concurrency, max_retries=args.max_retries)
    tasks_path = args.tasks_path or args.output_dir / "llm_tasks" / "latent_card_tasks.jsonl"
    execution_manifest = args.output_dir / "llm_execution_manifest.jsonl"
    results: dict[str, object] = {}

    if args.action in {"build", "all"}:
        results["build"] = build_latent_card_tasks(
            stable_latents_path=args.stable_latents,
            feature_store_path=args.feature_store,
            records_path=args.records,
            output_dir=args.output_dir,
            config=config,
        )
    if args.action in {"run", "all"}:
        key = _api_key()
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not available")
        results["run"] = run_deepseek_top50_tasks(
            tasks_path=tasks_path,
            output_dir=args.output_dir,
            api_key=key,
            config=config,
            system_prompt=LATENT_CARD_SYSTEM_PROMPT,
            force=args.force,
            max_tasks=args.max_tasks,
            analysis_name="deepseek_v4_flash_latent_cards",
            step_name="run-latent-card-generation",
        )
    if args.action == "retry":
        retry_build = build_refined_latent_card_retry_tasks(
            all_tasks_path=args.output_dir / "llm_tasks" / "latent_card_tasks.jsonl",
            retry_tasks_path=args.output_dir / "card_outputs" / "retry_tasks.jsonl",
            output_dir=args.output_dir,
        )
        results["retry_build"] = retry_build
        key = _api_key()
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not available")
        results["retry_run"] = run_deepseek_top50_tasks(
            tasks_path=retry_build["outputs"]["refined_retry_tasks"],
            output_dir=args.output_dir,
            api_key=key,
            config=DeepSeekTop50Config(concurrency=min(args.concurrency, max(retry_build["n_tasks"], 1)), max_retries=args.max_retries),
            system_prompt=LATENT_CARD_SYSTEM_PROMPT,
            force=True,
            analysis_name="deepseek_v4_flash_latent_cards",
            step_name="retry-latent-card-generation",
        )
        results["validate"] = validate_latent_card_outputs(
            tasks_path=args.output_dir / "llm_tasks" / "latent_card_tasks.jsonl",
            execution_manifest_path=execution_manifest,
            output_dir=args.output_dir,
        )
    if args.action in {"validate", "all"}:
        results["validate"] = validate_latent_card_outputs(
            tasks_path=tasks_path,
            execution_manifest_path=execution_manifest,
            output_dir=args.output_dir,
        )
    if args.action in {"render-review", "all"}:
        results["render_review"] = {
            "document": str(
                render_stable_core_top5_human_review_document(
                    output_dir=args.output_dir,
                    stable_latents_path=args.stable_latents,
                )
            )
        }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
