from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .job_runner import (
    OUTPUT_ROOT,
    filter_args,
    list_artifacts,
    run_job_subprocess,
    sanitize_output_subdir,
    utc_now_iso,
    validate_job_type,
)


app = FastAPI(
    title="SAE-RE PAI Job Service",
    version="0.1.0",
    description="Submit long-running SAE evaluation jobs on Aliyun PAI-EAS.",
)

_JOB_LOCK = threading.Lock()
_JOB_REGISTRY: dict[str, dict[str, Any]] = {}


class JobRunRequest(BaseModel):
    job_type: str = Field(
        ...,
        description="One of sae_eval, llamascope_sanity, sae_parity, variant_audit",
    )
    output_subdir: str = Field(..., description="Output subdirectory under OUTPUT_ROOT")
    args: dict[str, Any] = Field(default_factory=dict)


def _load_persisted_status(job_id: str) -> dict[str, Any] | None:
    for candidate in OUTPUT_ROOT.glob("**/job_status.json"):
        try:
            payload = candidate.read_text(encoding="utf-8")
            import json

            data = json.loads(payload)
        except Exception:
            continue
        if data.get("job_id") == job_id:
            return data
    return None


def _store_job(job_id: str, payload: dict[str, Any]) -> None:
    with _JOB_LOCK:
        _JOB_REGISTRY[job_id] = payload


def _get_job(job_id: str) -> dict[str, Any] | None:
    with _JOB_LOCK:
        current = _JOB_REGISTRY.get(job_id)
    if current is not None:
        status_path = current.get("status_path")
        if status_path:
            path = Path(status_path)
            if path.exists():
                import json

                data = json.loads(path.read_text(encoding="utf-8"))
                _store_job(job_id, {**current, **data})
                return {**current, **data}
        return current
    persisted = _load_persisted_status(job_id)
    if persisted is not None:
        _store_job(job_id, persisted)
    return persisted


def _run_and_refresh(job_id: str, job_type: str, output_subdir: str, args: dict[str, Any]) -> None:
    final_status = run_job_subprocess(
        job_id=job_id,
        job_type=job_type,
        output_subdir=output_subdir,
        args=args,
    )
    _store_job(job_id, final_status)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return {
        "status": "ok",
        "service": "sae-re-pai-job-service",
        "output_root": str(OUTPUT_ROOT),
        "task_mode": os.getenv("EAS_TASK_MODE", "0"),
        "time": utc_now_iso(),
    }


@app.post("/jobs/run")
def run_job(request: JobRunRequest) -> dict[str, Any]:
    try:
        spec = validate_job_type(request.job_type)
        output_subdir = sanitize_output_subdir(request.output_subdir)
        args = filter_args(request.job_type, request.args)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    output_dir = OUTPUT_ROOT / output_subdir
    status_path = output_dir / "job_status.json"
    if status_path.exists():
        raise HTTPException(
            status_code=409,
            detail=(
                f"output_subdir={output_subdir!r} already exists. "
                "Choose a new output_subdir for a new job."
            ),
        )

    job_id = uuid.uuid4().hex
    initial_status = {
        "job_id": job_id,
        "job_type": request.job_type,
        "status": "queued",
        "created_at": utc_now_iso(),
        "started_at": None,
        "finished_at": None,
        "output_dir": str(output_dir),
        "status_path": str(status_path),
        "allowed_args": sorted(spec["allowed_args"]),
        "args": args,
    }
    _store_job(job_id, initial_status)

    thread = threading.Thread(
        target=_run_and_refresh,
        kwargs={
            "job_id": job_id,
            "job_type": request.job_type,
            "output_subdir": output_subdir,
            "args": args,
        },
        daemon=True,
        name=f"sae-re-job-{job_id[:8]}",
    )
    thread.start()

    return {
        "job_id": job_id,
        "status": "queued",
        "job_type": request.job_type,
        "output_dir": str(output_dir),
        "status_url": f"/jobs/{job_id}",
        "artifacts_url": f"/jobs/{job_id}/artifacts",
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job_id={job_id!r} not found")
    return job


@app.get("/jobs/{job_id}/artifacts")
def get_job_artifacts(job_id: str) -> dict[str, Any]:
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job_id={job_id!r} not found")
    output_dir = job.get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=409, detail="job has no output directory yet")
    artifacts = list_artifacts(job["job_type"], output_dir)
    return {
        "job_id": job_id,
        "job_type": job["job_type"],
        "status": job.get("status"),
        **artifacts,
    }
