"""Subconcept aggregation task generation and table building."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .contrastive_evidence_pack import read_jsonl, write_json, write_jsonl
from .contrastive_gemini_io import parse_gemini_json_file


def make_subconcept_tasks(
    *,
    packs_path: str | Path,
    explanations_path: str | Path,
    scorer_metrics_path: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    task_dir = output_path / "ide_tasks"
    raw_dir = output_path / "subconcepts" / "raw_cluster_outputs"
    task_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    packs = {pack["packet_id"]: pack for pack in read_jsonl(packs_path)}
    explanations = read_jsonl(explanations_path) if Path(explanations_path).exists() else []
    metrics = pd.read_csv(scorer_metrics_path) if scorer_metrics_path and Path(scorer_metrics_path).exists() else pd.DataFrame()
    metric_by_packet = {}
    if not metrics.empty:
        metric_by_packet = {
            str(row["packet_id"]): row.to_dict()
            for _, row in metrics.sort_values(["packet_id", "auroc"], ascending=[True, False]).groupby("packet_id", sort=False).head(1).iterrows()
        }

    rows: list[dict[str, Any]] = []
    for explanation in explanations:
        pack = packs.get(str(explanation["packet_id"]))
        if not pack:
            continue
        metric = metric_by_packet.get(str(explanation["packet_id"]), {})
        rows.append(
            {
                "target_label": pack["target_label"],
                "packet_id": pack["packet_id"],
                "latent_idx": int(pack["latent_idx"]),
                "rank_within_label": int(pack["rank_within_label"]),
                "short_name": explanation.get("short_name"),
                "main_hypothesis": explanation.get("main_hypothesis"),
                "feature_type": explanation.get("feature_type"),
                "confidence": explanation.get("confidence"),
                "auroc": metric.get("auroc"),
                "latent_gap": metric.get("latent_gap"),
                "status": metric.get("status", "unscored"),
            }
        )

    tasks: list[dict[str, Any]] = []
    df = pd.DataFrame(rows)
    if not df.empty:
        for label, group in df.groupby("target_label", sort=True):
            items = group.sort_values(["status", "auroc", "rank_within_label"], ascending=[True, False, True]).to_dict(orient="records")
            task_id = f"subconcept_cluster_{label}"
            prompt = f"""Cluster these candidate latent explanations for one counseling behavior label into 3-6 subconcepts.

Input rows:
{json.dumps(items, ensure_ascii=False, indent=2)}

Return only a JSON list. Each item must contain:
- subconcept
- representative_latents: list of latent_idx integers
- explanation: one sentence
- confidence: number from 0 to 1
- caveats: one sentence

Use cautious wording. Do not claim a causal mechanism.
"""
            tasks.append(
                {
                    "task_id": task_id,
                    "task_type": "subconcept_cluster",
                    "target_label": label,
                    "prompt": prompt,
                    "expected_output_path": str(raw_dir / f"{task_id}.json"),
                    "output_format": "json_list",
                    "status": "pending_gemini_in_antigravity",
                }
            )
    tasks_path = task_dir / "subconcept_cluster_tasks.jsonl"
    cluster_input_path = output_path / "subconcepts" / "subconcept_cluster_input.csv"
    write_jsonl(tasks_path, tasks)
    pd.DataFrame(rows).to_csv(cluster_input_path, index=False)
    manifest = {
        "step": "make-subconcept-tasks",
        "inputs": {
            "packs": str(packs_path),
            "validated_explanations": str(explanations_path),
            "scorer_metrics": str(scorer_metrics_path or ""),
        },
        "outputs": {
            "subconcept_cluster_tasks": str(tasks_path),
            "subconcept_cluster_input": str(cluster_input_path),
            "raw_cluster_output_dir": str(raw_dir),
        },
        "n_tasks": int(len(tasks)),
        "n_input_rows": int(len(rows)),
    }
    write_json(output_path / "subconcept_task_manifest.json", manifest)
    return manifest


def _normalize_cluster_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("subconcepts", "clusters", "items"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError("Subconcept cluster output must be a JSON list or wrapped list")
    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each subconcept item must be an object")
        latents = item.get("representative_latents", [])
        if not isinstance(latents, list):
            raise ValueError("representative_latents must be a list")
        rows.append(
            {
                "subconcept": str(item.get("subconcept", "")).strip(),
                "representative_latents": ",".join(str(int(value)) for value in latents if str(value).strip()),
                "explanation": str(item.get("explanation", "")).strip(),
                "confidence": float(item.get("confidence", 0.0)),
                "caveats": str(item.get("caveats", "")).strip(),
            }
        )
    return rows


def build_subconcept_table(
    *,
    cluster_tasks_path: str | Path,
    cluster_input_path: str | Path,
    scorer_metrics_path: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    subconcept_dir = output_path / "subconcepts"
    subconcept_dir.mkdir(parents=True, exist_ok=True)
    tasks = read_jsonl(cluster_tasks_path) if Path(cluster_tasks_path).exists() else []
    input_rows = pd.read_csv(cluster_input_path) if Path(cluster_input_path).exists() else pd.DataFrame()
    metrics = pd.read_csv(scorer_metrics_path) if scorer_metrics_path and Path(scorer_metrics_path).exists() else pd.DataFrame()
    metric_by_latent = {}
    if not metrics.empty:
        metric_by_latent = {
            int(row["latent_idx"]): row.to_dict()
            for _, row in metrics.sort_values(["latent_idx", "auroc"], ascending=[True, False]).groupby("latent_idx", sort=False).head(1).iterrows()
        }

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for task in tasks:
        raw_path = Path(task.get("expected_output_path", ""))
        if raw_path.exists():
            try:
                for item in _normalize_cluster_payload(parse_gemini_json_file(raw_path)):
                    latent_ids = [int(value) for value in item["representative_latents"].split(",") if value.strip()]
                    aurocs = [float(metric_by_latent[idx]["auroc"]) for idx in latent_ids if idx in metric_by_latent and pd.notna(metric_by_latent[idx].get("auroc"))]
                    gaps = [float(metric_by_latent[idx]["latent_gap"]) for idx in latent_ids if idx in metric_by_latent and pd.notna(metric_by_latent[idx].get("latent_gap"))]
                    rows.append(
                        {
                            "target_label": task["target_label"],
                            **item,
                            "mean_auroc": float(sum(aurocs) / len(aurocs)) if aurocs else None,
                            "mean_latent_gap": float(sum(gaps) / len(gaps)) if gaps else None,
                            "status": "tentative" if task["target_label"] in {"SU", "GI", "RES"} else "gemini_clustered",
                            "source": "gemini_cluster_output",
                        }
                    )
            except Exception as exc:
                errors.append({"task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}", "raw_output_path": str(raw_path)})

    if not rows and not input_rows.empty:
        # Fallback table keeps the pipeline locally usable before Gemini clustering.
        for (label, short_name), group in input_rows.groupby(["target_label", "short_name"], sort=True):
            latents = sorted({int(value) for value in group["latent_idx"].tolist()})
            aurocs = pd.to_numeric(group.get("auroc", pd.Series(dtype=float)), errors="coerce").dropna()
            gaps = pd.to_numeric(group.get("latent_gap", pd.Series(dtype=float)), errors="coerce").dropna()
            rows.append(
                {
                    "target_label": label,
                    "subconcept": short_name,
                    "representative_latents": ",".join(str(value) for value in latents[:3]),
                    "explanation": str(group["main_hypothesis"].iloc[0]),
                    "confidence": float(pd.to_numeric(group["confidence"], errors="coerce").mean()),
                    "caveats": "Local fallback grouping by identical short_name; run Gemini clustering for final wording.",
                    "mean_auroc": float(aurocs.mean()) if not aurocs.empty else None,
                    "mean_latent_gap": float(gaps.mean()) if not gaps.empty else None,
                    "status": "tentative" if label in {"SU", "GI", "RES"} else "local_fallback",
                    "source": "local_short_name_fallback",
                }
            )

    table = pd.DataFrame(rows)
    table_path = subconcept_dir / "subconcept_table.csv"
    errors_path = subconcept_dir / "subconcept_cluster_errors.csv"
    table.to_csv(table_path, index=False)
    pd.DataFrame(errors).to_csv(errors_path, index=False)
    manifest = {
        "step": "build-subconcept-table",
        "inputs": {
            "cluster_tasks": str(cluster_tasks_path),
            "cluster_input": str(cluster_input_path),
            "scorer_metrics": str(scorer_metrics_path or ""),
        },
        "outputs": {
            "subconcept_table": str(table_path),
            "cluster_errors": str(errors_path),
        },
        "n_rows": int(len(table)),
        "n_errors": int(len(errors)),
    }
    write_json(subconcept_dir / "subconcept_manifest.json", manifest)
    return manifest


__all__ = ["build_subconcept_table", "make_subconcept_tasks"]
