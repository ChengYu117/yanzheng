"""Subconcept aggregation task generation and table building."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .contrastive_evidence_pack import read_jsonl, write_json, write_jsonl
from .contrastive_llm_io import parse_llm_json_file


def make_subconcept_tasks(
    *,
    packs_path: str | Path,
    explanations_path: str | Path,
    scorer_metrics_path: str | Path | None,
    latent_status_path: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    task_dir = output_path / "llm_tasks"
    raw_dir = output_path / "subconcepts" / "raw_cluster_outputs"
    task_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    packs = {pack["packet_id"]: pack for pack in read_jsonl(packs_path)}
    explanations = read_jsonl(explanations_path) if Path(explanations_path).exists() else []
    metrics = pd.read_csv(scorer_metrics_path) if scorer_metrics_path and Path(scorer_metrics_path).exists() else pd.DataFrame()
    latent_status = (
        pd.read_csv(latent_status_path)
        if latent_status_path and Path(latent_status_path).exists()
        else pd.DataFrame()
    )
    if latent_status.empty or "latent_status" not in latent_status.columns:
        raise ValueError("latent_level_status.csv is required before subconcept task generation")
    accepted_packets = set(
        latent_status.loc[
            latent_status["latent_status"].astype(str) == "accepted_stable", "packet_id"
        ].astype(str)
    )
    metric_by_explanation = {
        str(row["explanation_task_id"]): row.to_dict()
        for _, row in metrics.iterrows()
        if "explanation_task_id" in metrics.columns and pd.notna(row.get("explanation_task_id"))
    } if not metrics.empty else {}
    explanations_by_packet: dict[str, list[dict[str, Any]]] = {}
    for explanation in explanations:
        packet_id = str(explanation.get("packet_id", ""))
        if packet_id in accepted_packets:
            explanations_by_packet.setdefault(packet_id, []).append(explanation)

    rows: list[dict[str, Any]] = []
    for packet_id in sorted(accepted_packets):
        pack = packs.get(packet_id)
        candidates = explanations_by_packet.get(packet_id, [])
        if not pack or not candidates:
            continue
        candidates = sorted(
            candidates,
            key=lambda explanation: (
                -float(metric_by_explanation.get(str(explanation.get("task_id")), {}).get("auroc") or -1),
                -float(explanation.get("confidence", 0)),
                str(explanation.get("task_id", "")),
            ),
        )
        explanation = candidates[0]
        metric = metric_by_explanation.get(str(explanation.get("task_id")), {})
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
                "status": "accepted_stable",
            }
        )

    tasks: list[dict[str, Any]] = []
    excluded_labels: list[dict[str, Any]] = []
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["target_label", "rank_within_label", "latent_idx"], kind="mergesort"
        ).drop_duplicates(["target_label", "latent_idx"], keep="first")
        for label, group in df.groupby("target_label", sort=True):
            if len(group) < 3:
                excluded_labels.append(
                    {"target_label": label, "n_validated_latents": int(len(group)), "reason": "insufficient_validated_latents"}
                )
                continue
            items = group.sort_values(["auroc", "rank_within_label"], ascending=[False, True]).to_dict(orient="records")
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
                    "allowed_latent_ids": sorted(int(value) for value in group["latent_idx"].tolist()),
                    "prompt": prompt,
                    "expected_output_path": str(raw_dir / f"{task_id}.json"),
                    "output_format": "json_list",
                    "status": "pending_claude_code_llm",
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
            "latent_level_status": str(latent_status_path or ""),
        },
        "outputs": {
            "subconcept_cluster_tasks": str(tasks_path),
            "subconcept_cluster_input": str(cluster_input_path),
            "raw_cluster_output_dir": str(raw_dir),
        },
        "n_tasks": int(len(tasks)),
        "n_input_rows": int(len(df)),
        "n_distinct_latents": int(len(df)),
        "excluded_labels": excluded_labels,
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
    metric_by_latent: dict[tuple[str, int], dict[str, Any]] = {}
    if not metrics.empty:
        metric_by_latent = {
            (str(row["target_label"]), int(row["latent_idx"])): row.to_dict()
            for _, row in metrics.sort_values(
                ["target_label", "latent_idx", "auroc"], ascending=[True, True, False]
            ).groupby(["target_label", "latent_idx"], sort=False).head(1).iterrows()
        }

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for task in tasks:
        raw_path = Path(task.get("expected_output_path", ""))
        if not raw_path.exists():
            errors.append({"task_id": task["task_id"], "error": "raw_output_missing", "raw_output_path": str(raw_path)})
            continue
        try:
            normalized_items = _normalize_cluster_payload(parse_llm_json_file(raw_path))
            if not 3 <= len(normalized_items) <= 6:
                raise ValueError(f"Expected 3-6 subconcepts, got {len(normalized_items)}")
            allowed = {int(value) for value in task.get("allowed_latent_ids", [])}
            assigned: set[int] = set()
            task_rows: list[dict[str, Any]] = []
            for item in normalized_items:
                latent_ids = [int(value) for value in item["representative_latents"].split(",") if value.strip()]
                if not latent_ids:
                    raise ValueError(f"Subconcept {item['subconcept']!r} has no representative_latents")
                unknown = sorted(set(latent_ids).difference(allowed))
                if unknown:
                    raise ValueError(f"Subconcept references latents outside task input: {unknown}")
                duplicate_assignment = sorted(set(latent_ids).intersection(assigned))
                if duplicate_assignment:
                    raise ValueError(f"Latents assigned to multiple primary subconcepts: {duplicate_assignment}")
                assigned.update(latent_ids)
                keys = [(str(task["target_label"]), idx) for idx in latent_ids]
                aurocs = [float(metric_by_latent[key]["auroc"]) for key in keys if key in metric_by_latent and pd.notna(metric_by_latent[key].get("auroc"))]
                gaps = [float(metric_by_latent[key]["latent_gap"]) for key in keys if key in metric_by_latent and pd.notna(metric_by_latent[key].get("latent_gap"))]
                task_rows.append(
                        {
                            "target_label": task["target_label"],
                            **item,
                            "mean_auroc": float(sum(aurocs) / len(aurocs)) if aurocs else None,
                            "mean_latent_gap": float(sum(gaps) / len(gaps)) if gaps else None,
                            "status": (
                                "tentative_singleton"
                                if len(latent_ids) == 1
                                else "tentative"
                                if task["target_label"] in {"SU", "GI", "RES"}
                                else "llm_clustered"
                            ),
                            "source": "claude_code_llm_cluster_output",
                        }
                    )
            missing_assignments = sorted(allowed.difference(assigned))
            if missing_assignments:
                raise ValueError(f"Subconcept output omitted input latents: {missing_assignments}")
            rows.extend(task_rows)
        except Exception as exc:
            errors.append({"task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}", "raw_output_path": str(raw_path)})

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
        "fallback_used": False,
    }
    write_json(subconcept_dir / "subconcept_manifest.json", manifest)
    return manifest


__all__ = ["build_subconcept_table", "make_subconcept_tasks"]
