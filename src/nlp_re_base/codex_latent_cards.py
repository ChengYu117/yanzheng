"""Run label-blind latent-card prompts through isolated Codex exec requests."""

from __future__ import annotations

import hashlib
import csv
import json
import os
import shutil
import subprocess
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .contrastive_evidence_pack import read_jsonl, write_json, write_jsonl


CODEX_EXECUTION_MODE = "codex_exec_chatgpt_managed_auth"
ANALYSIS_NAME = "codex_gpt55_light_latent_cards"
TOOL_ITEM_TYPES = {
    "command_execution",
    "file_change",
    "mcp_tool_call",
    "web_search",
    "plan_update",
    "collab_tool_call",
}


@dataclass(frozen=True)
class CodexLatentCardConfig:
    model: str = "gpt-5.5"
    reasoning_effort: str = "low"
    concurrency: int = 2
    timeout_seconds: float = 600.0


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_codex_latent_card_tasks(
    *, source_tasks_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    source = Path(source_tasks_path)
    output = Path(output_dir)
    raw_dir = output / "card_outputs" / "raw"
    task_dir = output / "llm_tasks"
    raw_dir.mkdir(parents=True, exist_ok=True)
    task_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    for source_task in read_jsonl(source):
        task = dict(source_task)
        task_id = str(task["task_id"])
        task["expected_output_path"] = str(raw_dir / f"{task_id}.json")
        task["status"] = "pending_codex_exec"
        tasks.append(task)
    tasks_path = task_dir / "latent_card_tasks.jsonl"
    write_jsonl(tasks_path, tasks)
    manifest = {
        "analysis": ANALYSIS_NAME,
        "step": "build-codex-latent-card-tasks",
        "inputs": {"source_tasks": str(source)},
        "outputs": {"tasks": str(tasks_path)},
        "n_tasks": len(tasks),
        "prompt_reused_verbatim": True,
    }
    write_json(output / "task_build_manifest.json", manifest)
    return manifest


def default_isolation_paths(output_dir: str | Path) -> tuple[Path, Path]:
    suffix = _sha256_text(str(Path(output_dir).resolve()))[:12]
    root = Path(tempfile.gettempdir()) / f"codex-latent-eval-{suffix}"
    return root / "workdir", root / "codex-home"


def prepare_isolation(
    *, workdir: str | Path, codex_home: str | Path, auth_source: str | Path
) -> dict[str, Any]:
    work = Path(workdir)
    home = Path(codex_home)
    auth = Path(auth_source)
    if not auth.exists():
        raise FileNotFoundError(f"Codex auth file not found: {auth}")
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        raise ValueError(f"Isolated Codex workdir must be empty: {work}")
    home.mkdir(parents=True, exist_ok=True)
    forbidden = [
        home / "config.toml",
        home / "AGENTS.md",
        home / "AGENTS.override.md",
    ]
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise ValueError(f"Isolated CODEX_HOME contains forbidden context: {present}")
    skills_root = home / "skills"
    non_system_skills = []
    if skills_root.exists():
        non_system_skills = [
            str(path)
            for path in skills_root.iterdir()
            if path.name != ".system"
        ]
    if non_system_skills:
        raise ValueError(f"Isolated CODEX_HOME contains user skills: {non_system_skills}")
    shutil.copy2(auth, home / "auth.json")
    system_skill_files = list((skills_root / ".system").glob("*/SKILL.md")) if skills_root.exists() else []
    return {
        "workdir": str(work.resolve()),
        "codex_home": str(home.resolve()),
        "auth_source": str(auth.resolve()),
        "workdir_empty": True,
        "forbidden_context_present": False,
        "user_or_project_skills_present": False,
        "runtime_system_skill_files": len(system_skill_files),
        "plugins_feature_disabled_per_request": True,
    }


def audit_event_stream(stdout: str) -> dict[str, Any]:
    event_counts: Counter[str] = Counter()
    item_counts: Counter[str] = Counter()
    malformed = 0
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        event_counts[str(event.get("type", "unknown"))] += 1
        item = event.get("item")
        if isinstance(item, dict):
            item_counts[str(item.get("type", "unknown"))] += 1
    tool_calls = sum(item_counts[item_type] for item_type in TOOL_ITEM_TYPES)
    return {
        "event_type_counts": dict(sorted(event_counts.items())),
        "item_type_counts": dict(sorted(item_counts.items())),
        "tool_event_count": int(tool_calls),
        "malformed_event_lines": int(malformed),
    }


def render_codex_latent_card_catalog(
    *, output_dir: str | Path, stable_latents_path: str | Path
) -> dict[str, Any]:
    """Render all validated Codex cards with their stable-core label memberships."""

    output = Path(output_dir)
    cards = read_jsonl(output / "card_outputs" / "validated_cards.jsonl")
    with Path(stable_latents_path).open("r", encoding="utf-8-sig", newline="") as handle:
        stable_rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("stable_set_role") == "stable_core"
        ]

    memberships: dict[int, list[dict[str, str]]] = {}
    for row in stable_rows:
        memberships.setdefault(int(row["latent_idx"]), []).append(row)

    missing = sorted({int(card["latent_idx"]) for card in cards} - set(memberships))
    if missing:
        raise ValueError(f"Validated cards missing stable-core membership: {missing}")

    catalog_rows: list[dict[str, Any]] = []
    for card in sorted(cards, key=lambda row: int(row["latent_idx"])):
        latent_idx = int(card["latent_idx"])
        member_rows = sorted(
            memberships[latent_idx], key=lambda row: (row["label"], int(row["rank_within_label"]))
        )
        catalog_rows.append(
            {
                "latent_idx": latent_idx,
                "labels": "|".join(row["label"] for row in member_rows),
                "label_ranks": "|".join(
                    f'{row["label"]}:{int(row["rank_within_label"])}' for row in member_rows
                ),
                "short_name": card["short_name"],
                "explanation_type": card["explanation_type"],
                "confidence": int(card["confidence"]),
                "support_count": int(card["support_count"]),
                "support_fraction": float(card["support_fraction"]),
                "primary_explanation": card["primary_explanation"],
                "candidate_behavioral_explanation": card["candidate_behavioral_explanation"],
                "confidence_rationale": card["confidence_rationale"],
                "possible_confounds": " || ".join(card.get("possible_confounds", [])),
                "limitations": " || ".join(card.get("limitations", [])),
                "representative_evidence_ids": "|".join(card.get("representative_evidence_ids", [])),
            }
        )

    catalog_dir = output / "analysis"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    csv_path = catalog_dir / "all_latent_explanations.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(catalog_rows[0]))
        writer.writeheader()
        writer.writerows(catalog_rows)

    type_counts = Counter(row["explanation_type"] for row in catalog_rows)
    confidence_counts = Counter(str(row["confidence"]) for row in catalog_rows)
    support_values = [float(row["support_fraction"]) for row in catalog_rows]
    high_reliability = sum(
        int(row["confidence"]) >= 4 and float(row["support_fraction"]) >= 0.8
        for row in catalog_rows
    )
    summary = {
        "n_cards": len(catalog_rows),
        "n_stable_label_latent_rows": len(stable_rows),
        "n_shared_latents": sum(len(rows) > 1 for rows in memberships.values()),
        "explanation_type_counts": dict(sorted(type_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "mean_confidence": sum(int(row["confidence"]) for row in catalog_rows) / len(catalog_rows),
        "mean_support_fraction": sum(support_values) / len(support_values),
        "min_support_fraction": min(support_values),
        "max_support_fraction": max(support_values),
        "high_reliability_count": high_reliability,
        "high_reliability_definition": "confidence >= 4 and support_fraction >= 0.8",
        "catalog_csv": str(csv_path),
    }
    summary_path = catalog_dir / "latent_card_analysis_summary.json"
    write_json(summary_path, summary)

    md_path = catalog_dir / "all_latent_explanations.md"
    lines = [
        "# GPT-5.5 Stable-core Latent 解释全集",
        "",
        f"> 共 {len(catalog_rows)} 个去重 latent，对应 {len(stable_rows)} 条标签–latent stable-core 关系。解释来自标签盲归纳；标签仅在生成后用于索引。",
        "",
        "## 总览",
        "",
        f"- 解释类型：`{dict(sorted(type_counts.items()))}`",
        f"- 平均模型置信度：`{summary['mean_confidence']:.2f}/5`",
        f"- 平均模型自报支持比例：`{summary['mean_support_fraction']:.1%}`",
        f"- 高支持候选（confidence >= 4 且 support_fraction >= 0.8）：`{high_reliability}/{len(catalog_rows)}`",
        "",
        "注意：模型置信度和自报支持比例不是独立验证结果，只用于安排人工审核优先级。",
        "",
        "## 索引",
        "",
        "| Latent | Stable-core 标签 | 名称 | 类型 | 置信度 | 支持比例 |",
        "|---:|---|---|---|---:|---:|",
    ]
    for row in catalog_rows:
        lines.append(
            f'| [{row["latent_idx"]}](#latent-{row["latent_idx"]}) | {row["labels"]} | '
            f'{row["short_name"]} | {row["explanation_type"]} | {row["confidence"]}/5 | '
            f'{row["support_fraction"]:.0%} |'
        )
    for row in catalog_rows:
        lines.extend(
            [
                "",
                f'<a id="latent-{row["latent_idx"]}"></a>',
                f'## Latent {row["latent_idx"]}: {row["short_name"]}',
                "",
                f'- Stable-core 标签及排名：`{row["label_ranks"]}`',
                f'- 解释类型：`{row["explanation_type"]}`',
                f'- 模型置信度：`{row["confidence"]}/5`',
                f'- 模型自报支持：`{row["support_count"]}/50 ({row["support_fraction"]:.0%})`',
                "",
                "**主要解释**",
                "",
                str(row["primary_explanation"]),
                "",
                "**候选行为或话语功能**",
                "",
                str(row["candidate_behavioral_explanation"]),
                "",
                "**可靠性理由**",
                "",
                str(row["confidence_rationale"]),
                "",
                "**可能混淆因素**",
                "",
                str(row["possible_confounds"]),
                "",
                "**局限**",
                "",
                str(row["limitations"]),
            ]
        )
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    summary["catalog_markdown"] = str(md_path)
    write_json(summary_path, summary)
    return summary


def _codex_args(
    *,
    codex_bin: str,
    workdir: Path,
    output_path: Path,
    schema_path: Path,
    instructions_path: Path,
    codex_home: Path,
    config: CodexLatentCardConfig,
) -> list[str]:
    disabled_skills = sorted((codex_home / "skills").glob("**/SKILL.md"))
    disabled_skills_toml = "[" + ",".join(
        "{path='" + str(path).replace("'", "") + "',enabled=false}"
        for path in disabled_skills
    ) + "]"
    return [
        codex_bin,
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--strict-config",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--cd",
        str(workdir),
        "--model",
        config.model,
        "--config",
        f'model_reasoning_effort="{config.reasoning_effort}"',
        "--config",
        'approval_policy="never"',
        "--config",
        f"model_instructions_file='{instructions_path}'",
        "--config",
        'web_search="disabled"',
        "--config",
        "project_doc_max_bytes=0",
        "--config",
        f"skills.config={disabled_skills_toml}",
        "--disable",
        "shell_tool",
        "--disable",
        "apps",
        "--disable",
        "goals",
        "--disable",
        "hooks",
        "--disable",
        "multi_agent",
        "--disable",
        "memories",
        "--disable",
        "remote_plugin",
        "--disable",
        "personality",
        "--disable",
        "shell_snapshot",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
        "--json",
        "-",
    ]


def _run_one(
    *,
    task: dict[str, Any],
    codex_bin: str,
    workdir: Path,
    codex_home: Path,
    schema_path: Path,
    instructions_path: Path,
    config: CodexLatentCardConfig,
) -> dict[str, Any]:
    task_id = str(task["task_id"])
    prompt = str(task["prompt"])
    output_path = Path(task["expected_output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CODEX_HOME"] = str(codex_home)
    for name in ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN"):
        env.pop(name, None)
    args = _codex_args(
        codex_bin=codex_bin,
        workdir=workdir,
        output_path=output_path,
        schema_path=schema_path,
        instructions_path=instructions_path,
        codex_home=codex_home,
        config=config,
    )
    try:
        completed = subprocess.run(
            args,
            input=prompt,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            env=env,
            timeout=config.timeout_seconds,
            check=False,
        )
        event_audit = audit_event_stream(completed.stdout)
        status = "success"
        error = ""
        if completed.returncode != 0:
            status = "failed"
            error = f"codex_exit_code={completed.returncode}"
        elif not output_path.exists():
            status = "failed"
            error = "missing_output_file"
        elif event_audit["tool_event_count"] != 0:
            status = "failed"
            error = f"tool_use_detected={event_audit['tool_event_count']}"
        else:
            json.loads(output_path.read_text(encoding="utf-8"))
        return {
            "task_id": task_id,
            "latent_idx": int(task["latent_idx"]),
            "model": config.model,
            "reasoning_effort": config.reasoning_effort,
            "provider": "openai_codex",
            "auth_mode": "chatgpt_managed_saved_auth",
            "execution_mode": CODEX_EXECUTION_MODE,
            "status": status,
            "error": error,
            "prompt_sha256": _sha256_text(prompt),
            "base_instructions_sha256": _sha256_file(instructions_path),
            "output_schema_sha256": _sha256_file(schema_path),
            "raw_output_sha256": _sha256_file(output_path) if output_path.exists() else "",
            "codex_returncode": int(completed.returncode),
            **event_audit,
        }
    except Exception as exc:
        return {
            "task_id": task_id,
            "latent_idx": int(task.get("latent_idx", -1)),
            "model": config.model,
            "reasoning_effort": config.reasoning_effort,
            "provider": "openai_codex",
            "auth_mode": "chatgpt_managed_saved_auth",
            "execution_mode": CODEX_EXECUTION_MODE,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "prompt_sha256": _sha256_text(prompt),
            "base_instructions_sha256": _sha256_file(instructions_path),
            "output_schema_sha256": _sha256_file(schema_path),
            "raw_output_sha256": _sha256_file(output_path) if output_path.exists() else "",
            "tool_event_count": -1,
        }


def run_codex_latent_card_tasks(
    *,
    tasks_path: str | Path,
    output_dir: str | Path,
    schema_path: str | Path,
    instructions_path: str | Path,
    auth_source: str | Path,
    workdir: str | Path,
    codex_home: str | Path,
    codex_bin: str | None = None,
    config: CodexLatentCardConfig = CodexLatentCardConfig(),
    max_tasks: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    schema = Path(schema_path).resolve()
    instructions = Path(instructions_path).resolve()
    if not schema.exists() or not instructions.exists():
        raise FileNotFoundError("Schema or base-instructions file is missing")
    isolation = prepare_isolation(workdir=workdir, codex_home=codex_home, auth_source=auth_source)
    executable = codex_bin or shutil.which("codex")
    if not executable:
        raise FileNotFoundError("codex executable was not found")
    tasks = read_jsonl(tasks_path)
    manifest_path = output / "llm_execution_manifest.jsonl"
    prior = read_jsonl(manifest_path) if manifest_path.exists() else []
    successful = {
        str(row["task_id"])
        for row in prior
        if row.get("status") == "success" and row.get("task_id")
    }
    pending = [
        task
        for task in tasks
        if force
        or str(task["task_id"]) not in successful
        or not Path(task["expected_output_path"]).exists()
    ]
    if max_tasks is not None:
        pending = pending[: max(int(max_tasks), 0)]
    rows = list(prior)
    write_lock = threading.Lock()
    counts: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=max(int(config.concurrency), 1)) as pool:
        futures = {
            pool.submit(
                _run_one,
                task=task,
                codex_bin=str(executable),
                workdir=Path(workdir).resolve(),
                codex_home=Path(codex_home).resolve(),
                schema_path=schema,
                instructions_path=instructions,
                config=config,
            ): task
            for task in pending
        }
        for future in as_completed(futures):
            row = future.result()
            counts[str(row["status"])] += 1
            with write_lock:
                rows.append(row)
                write_jsonl(manifest_path, rows)
    manifest = {
        "analysis": ANALYSIS_NAME,
        "step": "run-codex-latent-card-generation",
        "inputs": {
            "tasks": str(tasks_path),
            "schema": str(schema),
            "base_instructions": str(instructions),
        },
        "outputs": {"execution_manifest": str(manifest_path)},
        "parameters": asdict(config),
        "isolation": isolation,
        "n_tasks_total": len(tasks),
        "n_pending_this_run": len(pending),
        "counts_this_run": dict(sorted(counts.items())),
        "tool_use_policy": "zero tool events required",
    }
    write_json(output / "codex_run_manifest.json", manifest)
    return manifest


__all__ = [
    "ANALYSIS_NAME",
    "CODEX_EXECUTION_MODE",
    "CodexLatentCardConfig",
    "audit_event_stream",
    "render_codex_latent_card_catalog",
    "build_codex_latent_card_tasks",
    "default_isolation_paths",
    "prepare_isolation",
    "run_codex_latent_card_tasks",
]
