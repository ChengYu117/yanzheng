"""Automated Gemma3-4B MISC layer-probe execution loop.

This wrapper runs the engineering checks, small Gemma smoke, full MISC run, and
result validation. It retries the things that are safe to adjust automatically
such as batch size and endpoint fallback, but stops on external blockers such as
invalid Hugging Face credentials.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = "google/gemma-3-4b-pt"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
EXPECTED_LABELS = ["RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF"]


@dataclass
class CommandResult:
    name: str
    returncode: int
    log_path: Path
    tail: str
    seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-run Gemma3 MISC layer probe until results or external blocker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--data-dir", default="data/mi_quality_counseling_misc")
    parser.add_argument("--output-dir", default="outputs/gemma3_misc_layer_probe")
    parser.add_argument(
        "--smoke-output-dir",
        default="outputs/gemma3_misc_layer_probe_smoke",
    )
    parser.add_argument("--smoke-limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--hf-endpoint", default=DEFAULT_HF_ENDPOINT)
    parser.add_argument(
        "--ignore-hf-token-env",
        action="store_true",
        help="Ignore HF_TOKEN/HUGGING_FACE_HUB_TOKEN and use cached hf auth login token.",
    )
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_command(
    name: str,
    command: list[str],
    *,
    log_dir: Path,
    timeout: int | None = None,
) -> CommandResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    tail: deque[str] = deque(maxlen=120)
    started = time.time()
    print(f"\n==> {name}")
    print(" ".join(command))
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                safe_line = line.encode(
                    sys.stdout.encoding or "utf-8",
                    errors="replace",
                ).decode(sys.stdout.encoding or "utf-8", errors="replace")
                print(safe_line, end="")
                log_file.write(line)
                tail.append(line)
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            returncode = proc.wait()
            message = f"\nTIMEOUT after {timeout} seconds\n"
            print(message, end="")
            log_file.write(message)
            tail.append(message)
    seconds = time.time() - started
    return CommandResult(
        name=name,
        returncode=returncode,
        log_path=log_path,
        tail="".join(tail),
        seconds=round(seconds, 3),
    )


def classify_failure(result: CommandResult) -> str:
    text = result.tail.lower()
    if "invalid user token" in text or "invalid token" in text:
        return "external_auth_invalid_token"
    if "gatedrepoerror" in text or "cannot access gated repo" in text or "401 client error" in text:
        return "external_auth_gated_repo"
    if "out of memory" in text or "cuda error: out of memory" in text:
        return "retry_oom"
    if "couldn't connect to 'https://hf-mirror.com'" in text or "we couldn't connect to 'https://hf-mirror.com'" in text:
        return "retry_endpoint_fallback"
    if "no space left on device" in text:
        return "external_disk_space"
    return "failure"


def validate_probe_output(output_dir: Path, *, expected_limit: int | None = None) -> dict:
    best_path = output_dir / "best_layers_by_label.csv"
    metrics_path = output_dir / "layer_probe_metrics.csv"
    summary_path = output_dir / "dataset_summary.json"
    manifest_path = output_dir / "probe_artifacts" / "manifest.json"

    missing = [
        str(path)
        for path in (best_path, metrics_path, summary_path, manifest_path)
        if not path.exists()
    ]
    if missing:
        return {"ok": False, "reason": "missing_outputs", "missing": missing}

    with best_path.open("r", encoding="utf-8", newline="") as f:
        best_rows = list(csv.DictReader(f))
    labels = [row["label"] for row in best_rows]
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if labels != EXPECTED_LABELS:
        return {
            "ok": False,
            "reason": "unexpected_labels",
            "labels": labels,
            "expected": EXPECTED_LABELS,
        }
    if "OTHER" in labels:
        return {"ok": False, "reason": "other_label_present", "labels": labels}
    if len(manifest.get("artifacts", [])) != len(EXPECTED_LABELS):
        return {
            "ok": False,
            "reason": "unexpected_artifact_count",
            "n_artifacts": len(manifest.get("artifacts", [])),
        }
    if expected_limit is not None and int(summary.get("n_records", -1)) != expected_limit:
        return {
            "ok": False,
            "reason": "unexpected_record_count",
            "n_records": summary.get("n_records"),
            "expected": expected_limit,
        }

    recognized = sum(1 for row in best_rows if str(row.get("recognized")).lower() == "true")
    return {
        "ok": True,
        "labels": labels,
        "n_records": summary.get("n_records"),
        "n_layers": summary.get("n_layers"),
        "hidden_size": summary.get("hidden_size"),
        "recognized_labels": recognized,
        "best_layers_path": str(best_path),
        "metrics_path": str(metrics_path),
    }


def probe_command(
    args: argparse.Namespace,
    *,
    output_dir: str,
    batch_size: int,
    hf_endpoint: str,
    limit: int | None,
    overwrite_cache: bool,
) -> list[str]:
    command = [
        args.python_exe,
        "run_gemma_layer_probe.py",
        "--model-id",
        args.model_id,
        "--data-dir",
        args.data_dir,
        "--output-dir",
        output_dir,
        "--batch-size",
        str(batch_size),
        "--max-seq-len",
        str(args.max_seq_len),
        "--hf-endpoint",
        hf_endpoint,
    ]
    if args.ignore_hf_token_env:
        command.append("--ignore-hf-token-env")
    if limit is not None:
        command.extend(["--limit-records", str(limit)])
    if overwrite_cache:
        command.append("--overwrite-cache")
    return command


def run_probe_with_retries(
    args: argparse.Namespace,
    *,
    name: str,
    output_dir: str,
    limit: int | None,
    initial_batch_size: int,
    log_dir: Path,
    overwrite_cache: bool,
) -> tuple[CommandResult, int, str, str | None]:
    batch_size = max(1, initial_batch_size)
    endpoints = [args.hf_endpoint]
    if args.hf_endpoint:
        endpoints.append("")

    last_result: CommandResult | None = None
    for endpoint in endpoints:
        while batch_size >= 1:
            result = run_command(
                f"{name}_batch{batch_size}_{'mirror' if endpoint else 'official'}",
                probe_command(
                    args,
                    output_dir=output_dir,
                    batch_size=batch_size,
                    hf_endpoint=endpoint,
                    limit=limit,
                    overwrite_cache=overwrite_cache,
                ),
                log_dir=log_dir,
            )
            last_result = result
            if result.ok:
                return result, batch_size, endpoint, None

            failure = classify_failure(result)
            if failure == "retry_oom" and batch_size > 1:
                batch_size = max(1, batch_size // 2)
                print(f"OOM detected; retrying {name} with batch_size={batch_size}.")
                continue
            if failure == "retry_endpoint_fallback" and endpoint:
                print(f"Mirror endpoint failed; retrying {name} with official Hugging Face endpoint.")
                break
            return result, batch_size, endpoint, failure

    assert last_result is not None
    return last_result, batch_size, endpoints[-1], classify_failure(last_result)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    status_path = PROJECT_ROOT / "outputs" / "gemma3_misc_layer_probe_auto_status.json"
    log_dir = PROJECT_ROOT / "outputs" / "gemma3_misc_layer_probe_auto_logs"
    status: dict = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": args.model_id,
        "data_dir": args.data_dir,
        "expected_labels": EXPECTED_LABELS,
        "stages": [],
    }
    write_status(status_path, status)

    if not args.skip_tests:
        checks = [
            (
                "py_compile",
                [
                    args.python_exe,
                    "-m",
                    "py_compile",
                    "run_gemma_layer_probe.py",
                    "src/nlp_re_base/layer_probe.py",
                    "test_layer_probe_smoke.py",
                ],
            ),
            ("test_layer_probe_smoke", [args.python_exe, "test_layer_probe_smoke.py"]),
            ("test_dataset_loader_smoke", [args.python_exe, "test_dataset_loader_smoke.py"]),
        ]
        for name, command in checks:
            result = run_command(name, command, log_dir=log_dir)
            status["stages"].append(
                {
                    "stage": name,
                    "ok": result.ok,
                    "log_path": str(result.log_path),
                    "seconds": result.seconds,
                }
            )
            write_status(status_path, status)
            if not result.ok:
                status["final_status"] = "failed_engineering_check"
                write_status(status_path, status)
                return result.returncode

    smoke_result, batch_size, endpoint, failure = run_probe_with_retries(
        args,
        name="gemma_smoke",
        output_dir=args.smoke_output_dir,
        limit=args.smoke_limit,
        initial_batch_size=args.batch_size,
        log_dir=log_dir,
        overwrite_cache=True,
    )
    smoke_stage = {
        "stage": "gemma_smoke",
        "ok": smoke_result.ok,
        "failure": failure,
        "batch_size": batch_size,
        "hf_endpoint": endpoint,
        "log_path": str(smoke_result.log_path),
        "seconds": smoke_result.seconds,
    }
    if smoke_result.ok:
        smoke_stage["validation"] = validate_probe_output(
            PROJECT_ROOT / args.smoke_output_dir,
            expected_limit=args.smoke_limit,
        )
    status["stages"].append(smoke_stage)
    write_status(status_path, status)
    if not smoke_result.ok or not smoke_stage.get("validation", {}).get("ok", False):
        status["final_status"] = failure or "failed_smoke_validation"
        status["requires_human_intervention"] = str(status["final_status"]).startswith("external_")
        write_status(status_path, status)
        return smoke_result.returncode or 1

    full_result, batch_size, endpoint, failure = run_probe_with_retries(
        args,
        name="gemma_full",
        output_dir=args.output_dir,
        limit=None,
        initial_batch_size=batch_size,
        log_dir=log_dir,
        overwrite_cache=False,
    )
    full_stage = {
        "stage": "gemma_full",
        "ok": full_result.ok,
        "failure": failure,
        "batch_size": batch_size,
        "hf_endpoint": endpoint,
        "log_path": str(full_result.log_path),
        "seconds": full_result.seconds,
    }
    if full_result.ok:
        full_stage["validation"] = validate_probe_output(PROJECT_ROOT / args.output_dir)
    status["stages"].append(full_stage)
    if full_result.ok and full_stage.get("validation", {}).get("ok", False):
        status["final_status"] = "completed"
        status["requires_human_intervention"] = False
        status["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_status(status_path, status)
        print(f"\nCompleted. Status: {status_path}")
        return 0

    status["final_status"] = failure or "failed_full_validation"
    status["requires_human_intervention"] = str(status["final_status"]).startswith("external_")
    write_status(status_path, status)
    return full_result.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
