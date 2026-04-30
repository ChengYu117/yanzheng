from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nlp_re_base.config import PROJECT_ROOT


OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", "/mnt/pai/outputs"))
DEFAULT_MODEL_DIR = os.getenv("MODEL_DIR", "/mnt/pai/models/Llama-3.1-8B")
DEFAULT_DEVICE = os.getenv("PAI_DEVICE", "cuda")

_SAFE_SUBDIR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")

JOB_SPECS: dict[str, dict[str, Any]] = {
    "sae_eval": {
        "script": "run_sae_evaluation.py",
        "allowed_args": {
            "batch_size",
            "max_seq_len",
            "device",
            "skip_ce_kl",
            "ce_kl_batch_size",
            "ce_kl_max_texts",
            "data_dir",
            "full_structural",
            "aggregation",
            "compare_mean",
            "judge_bundle_only",
        },
        "default_args": {
            "device": DEFAULT_DEVICE,
            "model_dir": DEFAULT_MODEL_DIR,
            "full_structural": True,
        },
        "key_artifacts": [
            "metrics_structural.json",
            "metrics_functional.json",
            "candidate_latents.csv",
        ],
    },
    "llamascope_sanity": {
        "script": "run_llamascope_distribution_sanity.py",
        "allowed_args": {
            "dataset_id",
            "dataset_config_name",
            "split",
            "text_field",
            "max_docs",
            "streaming",
            "max_seq_len",
            "batch_size",
            "ce_kl_batch_size",
            "ce_kl_max_texts",
            "mi_re_data_dir",
            "mi_re_ce_kl_max_texts",
            "device",
            "checkpoint_topk_semantics",
        },
        "default_args": {
            "device": DEFAULT_DEVICE,
            "model_dir": DEFAULT_MODEL_DIR,
            "streaming": True,
            "checkpoint_topk_semantics": "hard",
        },
        "key_artifacts": [
            "metrics_structural.json",
            "comparison_vs_mire.json",
            "metric_provenance.json",
        ],
    },
    "sae_parity": {
        "script": "run_sae_checkpoint_parity.py",
        "allowed_args": {
            "dataset_id",
            "dataset_config_name",
            "split",
            "text_field",
            "max_docs",
            "streaming",
            "max_seq_len",
            "batch_size",
            "topk_compare",
            "device",
            "official_loader",
            "official_repo_dir",
            "checkpoint_topk_semantics",
        },
        "default_args": {
            "device": DEFAULT_DEVICE,
            "model_dir": DEFAULT_MODEL_DIR,
            "streaming": True,
            "checkpoint_topk_semantics": "hard",
        },
        "key_artifacts": [
            "parity_report.json",
            "sample_stats.json",
            "evidence_table.json",
        ],
    },
    "variant_audit": {
        "script": "run_llamascope_variant_audit.py",
        "allowed_args": {
            "dataset_id",
            "dataset_config_name",
            "split",
            "text_field",
            "smoke_docs",
            "full_docs",
            "streaming",
            "max_seq_len",
            "batch_size",
            "ce_kl_batch_size",
            "mi_re_data_dir",
            "device",
        },
        "default_args": {
            "device": DEFAULT_DEVICE,
            "model_dir": DEFAULT_MODEL_DIR,
            "streaming": True,
        },
        "key_artifacts": [
            "variant_evidence_matrix.json",
            "variant_decision.json",
        ],
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_output_subdir(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("output_subdir must not be empty")
    if value.startswith("/") or ".." in value.split("/"):
        raise ValueError("output_subdir must stay within OUTPUT_ROOT")
    if not _SAFE_SUBDIR_RE.match(value):
        raise ValueError(
            "output_subdir may only contain letters, digits, dot, underscore, dash, and slash"
        )
    return value


def validate_job_type(job_type: str) -> dict[str, Any]:
    if job_type not in JOB_SPECS:
        raise ValueError(
            f"Unsupported job_type={job_type!r}. Expected one of {sorted(JOB_SPECS)}"
        )
    return JOB_SPECS[job_type]


def filter_args(job_type: str, args: dict[str, Any] | None) -> dict[str, Any]:
    spec = validate_job_type(job_type)
    if not args:
        return {}
    filtered: dict[str, Any] = {}
    for key, value in args.items():
        if key not in spec["allowed_args"]:
            raise ValueError(
                f"Argument {key!r} is not allowed for job_type={job_type!r}. "
                f"Allowed keys: {sorted(spec['allowed_args'])}"
            )
        filtered[key] = value
    return filtered


def _flag_name(name: str) -> str:
    return "--" + name.replace("_", "-")


def args_to_cli(args: dict[str, Any]) -> list[str]:
    cli: list[str] = []
    for key, value in args.items():
        if value is None:
            continue
        flag = _flag_name(key)
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            continue
        cli.extend([flag, str(value)])
    return cli


def build_job_command(
    *,
    job_type: str,
    output_subdir: str,
    args: dict[str, Any] | None = None,
) -> tuple[list[str], Path]:
    spec = validate_job_type(job_type)
    user_args = filter_args(job_type, args)
    merged_args = {**spec["default_args"], **user_args}
    output_dir = OUTPUT_ROOT / sanitize_output_subdir(output_subdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = PROJECT_ROOT / spec["script"]
    command = [
        sys.executable,
        str(script_path),
        "--output-dir",
        str(output_dir),
    ]
    if "model_dir" not in user_args and merged_args.get("model_dir"):
        command.extend(["--model-dir", str(merged_args.pop("model_dir"))])
    command.extend(args_to_cli(merged_args))
    return command, output_dir


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def list_artifacts(job_type: str, output_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir)
    spec = validate_job_type(job_type)
    existing = []
    for artifact in spec["key_artifacts"]:
        path = output_path / artifact
        if path.exists():
            existing.append({"name": artifact, "path": str(path)})

    children = []
    if output_path.exists():
        for item in sorted(output_path.iterdir()):
            children.append(
                {
                    "name": item.name,
                    "path": str(item),
                    "is_dir": item.is_dir(),
                }
            )
    return {
        "output_dir": str(output_path),
        "key_artifacts": existing,
        "children": children,
    }


@dataclass
class JobHandle:
    job_id: str
    job_type: str
    output_dir: Path
    status_path: Path
    thread: threading.Thread | None = None


def run_job_subprocess(
    *,
    job_id: str,
    job_type: str,
    output_subdir: str,
    args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    command, output_dir = build_job_command(
        job_type=job_type,
        output_subdir=output_subdir,
        args=args,
    )
    status_path = output_dir / "job_status.json"
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"

    base_status = {
        "job_id": job_id,
        "job_type": job_type,
        "status": "running",
        "created_at": utc_now_iso(),
        "started_at": utc_now_iso(),
        "finished_at": None,
        "output_dir": str(output_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "command": command,
        "return_code": None,
        "error": None,
    }
    _write_status(status_path, base_status)

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=stdout,
            stderr=stderr,
            text=True,
            env=os.environ.copy(),
            check=False,
        )

    final_status = {
        **base_status,
        "status": "succeeded" if completed.returncode == 0 else "failed",
        "finished_at": utc_now_iso(),
        "return_code": completed.returncode,
        "artifacts": list_artifacts(job_type, output_dir),
    }
    if completed.returncode != 0:
        final_status["error"] = (
            f"Subprocess exited with code {completed.returncode}. "
            f"See stderr log: {stderr_path}"
        )
    _write_status(status_path, final_status)
    return final_status
