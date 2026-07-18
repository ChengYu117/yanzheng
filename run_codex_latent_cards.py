"""Build, run, and validate isolated GPT-5.5 latent cards via Codex exec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.nlp_re_base.codex_latent_cards import (
    ANALYSIS_NAME,
    CODEX_EXECUTION_MODE,
    CodexLatentCardConfig,
    build_codex_latent_card_tasks,
    default_isolation_paths,
    render_codex_latent_card_catalog,
    run_codex_latent_card_tasks,
)
from src.nlp_re_base.deepseek_latent_cards import (
    render_stable_core_top5_human_review_document,
    validate_latent_card_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "run", "validate", "render-review", "render-catalog", "all"), nargs="?", default="all")
    parser.add_argument("--source-tasks", required=True)
    parser.add_argument("--stable-latents")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--schema", default="config/latent_card_output_schema.json")
    parser.add_argument("--base-instructions", default="config/codex_latent_evaluator_base_instructions.txt")
    parser.add_argument("--auth-source", default=str(Path.home() / ".codex" / "auth.json"))
    parser.add_argument("--isolated-workdir")
    parser.add_argument("--isolated-codex-home")
    parser.add_argument("--codex-bin")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="low", choices=("minimal", "low", "medium", "high", "xhigh"))
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    default_workdir, default_home = default_isolation_paths(args.output_dir)
    workdir = Path(args.isolated_workdir) if args.isolated_workdir else default_workdir
    codex_home = Path(args.isolated_codex_home) if args.isolated_codex_home else default_home
    tasks_path = args.output_dir / "llm_tasks" / "latent_card_tasks.jsonl"
    execution_manifest = args.output_dir / "llm_execution_manifest.jsonl"
    results: dict[str, object] = {}

    if args.action in {"build", "all"}:
        results["build"] = build_codex_latent_card_tasks(
            source_tasks_path=args.source_tasks,
            output_dir=args.output_dir,
        )
    if args.action in {"run", "all"}:
        if not tasks_path.exists():
            build_codex_latent_card_tasks(source_tasks_path=args.source_tasks, output_dir=args.output_dir)
        results["run"] = run_codex_latent_card_tasks(
            tasks_path=tasks_path,
            output_dir=args.output_dir,
            schema_path=args.schema,
            instructions_path=args.base_instructions,
            auth_source=args.auth_source,
            workdir=workdir,
            codex_home=codex_home,
            codex_bin=args.codex_bin,
            config=CodexLatentCardConfig(
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                concurrency=args.concurrency,
                timeout_seconds=args.timeout_seconds,
            ),
            max_tasks=args.max_tasks,
            force=args.force,
        )
    if args.action in {"validate", "all"}:
        results["validate"] = validate_latent_card_outputs(
            tasks_path=tasks_path,
            execution_manifest_path=execution_manifest,
            output_dir=args.output_dir,
            expected_execution_mode=CODEX_EXECUTION_MODE,
            analysis_name=ANALYSIS_NAME,
        )
    if args.action in {"render-review", "all"}:
        if not args.stable_latents:
            raise ValueError("--stable-latents is required for render-review")
        results["render_review"] = {
            "document": str(
                render_stable_core_top5_human_review_document(
                    output_dir=args.output_dir,
                    stable_latents_path=args.stable_latents,
                )
            )
        }
    if args.action in {"render-catalog", "all"}:
        if not args.stable_latents:
            raise ValueError("--stable-latents is required for render-catalog")
        results["render_catalog"] = render_codex_latent_card_catalog(
            output_dir=args.output_dir,
            stable_latents_path=args.stable_latents,
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
