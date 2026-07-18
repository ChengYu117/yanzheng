"""Sampled SAE/PCA contrastive explanations with held-out faithfulness scoring."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from .contrastive_evidence_pack import normalise_text, read_jsonl, write_json, write_jsonl
from .contrastive_faithfulness_v2 import (
    _format_samples,
    _rank_bands,
    _unique_pick,
    build_explainer_prompt,
)


ANALYSIS_NAME = "task5_sae_pca_contrastive_faithfulness_gpt55_low_sampled"


@dataclass(frozen=True)
class SamplingConfig:
    n_group: int = 10
    n_heldout_stratum: int = 5
    random_seed: str = "task5-sae-pca-contrastive-v1"


def _load_tensor(path: str | Path, keys: tuple[str, ...]) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, torch.Tensor):
        return payload
    for key in keys:
        if key in payload:
            return payload[key]
    raise KeyError(f"No matrix found in {path}; tried {keys}")


def _pca_scores_all(raw_hidden: np.ndarray, model_path: Path) -> np.ndarray:
    model = np.load(model_path)
    scaled = (raw_hidden - model["input_mean"]) / model["input_scale"]
    scores = (scaled - model["pca_mean"]) @ model["pca_components"].T
    return ((scores - model["output_mean"]) / model["output_scale"]).astype(np.float32)


def _public_samples(rows: list[int], texts: list[str], prefix: str) -> list[dict[str, str]]:
    return [{"sample_id": f"{prefix}{i:03d}", "text": texts[row]} for i, row in enumerate(rows, 1)]


def _scorer_prompt(feature_id: str, explanation: dict[str, Any], heldout: list[dict[str, Any]]) -> str:
    fields = (
        "short_name", "contrastive_explanation", "necessary_or_characteristic_condition",
        "insufficient_conditions", "surface_or_linguistic_hypothesis",
        "behavioral_or_discourse_hypothesis", "primary_explanation", "explanation_type",
        "possible_confounds", "limitations",
    )
    frozen = {key: explanation[key] for key in fields}
    public = [{"sample_id": row["sample_id"], "text": row["text"]} for row in heldout]
    return f"""Use the frozen explanation below to predict how strongly each held-out spoken/transcribed-dialogue sentence matches the anonymous text feature.

Feature ID: {feature_id}

Frozen explanation:
{json.dumps(frozen, ensure_ascii=False, indent=2)}

Held-out sentences:
{_format_samples(public)}

For each complete sentence output an integer predicted_feature_score from 0 to 100: 0–10 absent/contradicted; 11–30 weak/incidental; 31–60 partial/ambiguous; 61–80 clear; 81–100 strong and specific. Do not revise the explanation, infer hidden groups, or rely on broad domain membership. matching_evidence_span must be copied from the sentence or empty. Return all 20 IDs exactly once and in input order. Return only JSON matching the supplied schema."""


def _canonical_units(mapping: pd.DataFrame, seed: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: dict[str, dict[str, Any]] = {}
    for row in mapping.itertuples(index=False):
        representation = str(row.representation)
        if representation == "SAE":
            key = f"SAE:{int(row.index)}"
            family = "SAE"
        else:
            key = f"{representation}:{row.label}:{int(row.index)}:{int(row.direction_sign)}"
            family = representation
        record = rows.setdefault(
            key,
            {
                "canonical_key": key,
                "representation_family": family,
                "label": str(row.label),
                "index": int(row.index),
                "direction_sign": int(row.direction_sign),
                "arms": [],
                "source_blind_unit_ids": [],
            },
        )
        record["arms"].append(str(row.arm))
        record["source_blind_unit_ids"].append(str(row.blind_unit_id))
    ordered = sorted(rows.values(), key=lambda row: hashlib.sha256(f"{seed}|{row['canonical_key']}".encode()).hexdigest())
    for order, row in enumerate(ordered, 1):
        row["feature_id"] = f"F{order:03d}"
        row["unit_order"] = order
        row["arms"] = "|".join(sorted(set(row["arms"])))
        row["source_blind_unit_ids"] = "|".join(sorted(set(row["source_blind_unit_ids"])))
    units = pd.DataFrame(ordered)
    feature_by_key = dict(zip(units["canonical_key"], units["feature_id"]))
    return units, pd.DataFrame({"canonical_key": list(feature_by_key), "feature_id": list(feature_by_key.values())})


def build_packets(
    *, task5_root: str | Path, sae_features_path: str | Path, raw_hidden_path: str | Path,
    label_matrix_path: str | Path, output_dir: str | Path,
    config: SamplingConfig = SamplingConfig(),
) -> dict[str, Any]:
    root, output = Path(task5_root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    private = output / "private"; private.mkdir(parents=True, exist_ok=True)
    raw_dir = output / "explainer" / "raw"; raw_dir.mkdir(parents=True, exist_ok=True)

    mapping = pd.read_csv(root / "private" / "blind_mapping_key.csv")
    matches = pd.read_csv(root / "private" / "matched_units.csv")
    split = pd.read_csv(root / "private" / "split_assignments.csv")
    labels = pd.read_csv(label_matrix_path)
    texts = labels["unit_text"].astype(str).tolist()
    sae = _load_tensor(sae_features_path, ("utterance_features", "features", "X"))
    raw_hidden = _load_tensor(raw_hidden_path, ("utterance_activations", "activations", "X")).float().numpy()
    units, _ = _canonical_units(mapping, config.random_seed)

    pca_cache: dict[str, np.ndarray] = {}
    split_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label in sorted(units["label"].unique()):
        rows = split[split["label"].astype(str).eq(label)]
        split_cache[label] = (
            rows.loc[rows["split"].eq("train"), "row_idx"].astype(int).to_numpy(),
            rows.loc[rows["split"].eq("test"), "row_idx"].astype(int).to_numpy(),
        )
        pca_cache[label] = _pca_scores_all(
            raw_hidden,
            root / "private" / "pca_models" / f"{label}_pca_max100.npz",
        )

    packets: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    explainer_ineligible: list[dict[str, Any]] = []
    scorer_ineligible: list[dict[str, Any]] = []
    for unit in units.itertuples(index=False):
        label = str(unit.label); train_idx, test_idx = split_cache[label]
        if str(unit.representation_family) == "SAE":
            values = sae[:, int(unit.index)].float().numpy()
        else:
            values = pca_cache[label][:, int(unit.index)] * int(unit.direction_sign)
        used: set[str] = set()
        train_pos = train_idx[values[train_idx] > 0]
        test_pos = test_idx[values[test_idx] > 0]
        test_control = test_idx[values[test_idx] <= 0]
        try:
            weak_band, _, strong_band = _rank_bands(values, train_pos)
            strong_rows = _unique_pick(strong_band, texts, config.n_group, key=f"{unit.canonical_key}|strong", used=used)
            weak_rows = _unique_pick(weak_band, texts, config.n_group, key=f"{unit.canonical_key}|weak", used=used)
            if len(strong_rows) != config.n_group or len(weak_rows) != config.n_group:
                raise ValueError("lacks 10 unique discovery strong/weak sentences")
        except ValueError as exc:
            failure = {"feature_id": unit.feature_id, "reason": str(exc), "n_train_positive": len(train_pos), "n_test_positive": len(test_pos)}
            explainer_ineligible.append(failure); audit.append({**failure, "sampling_status": "explainer_ineligible"})
            continue

        held_rows: list[tuple[str, int]] = []
        heldout_reason = ""
        try:
            weak_band, mid_band, strong_band = _rank_bands(values, test_pos)
            for stratum, candidates in (("high", strong_band), ("mid", mid_band), ("weak", weak_band), ("control", test_control.tolist())):
                picked = _unique_pick(candidates, texts, config.n_heldout_stratum, key=f"{unit.canonical_key}|held|{stratum}", used=used)
                if len(picked) != config.n_heldout_stratum:
                    raise ValueError(f"lacks 5 unique held-out {stratum} sentences")
                held_rows.extend((stratum, row) for row in picked)
        except ValueError as exc:
            held_rows = []; heldout_reason = str(exc)
            scorer_ineligible.append({"feature_id": unit.feature_id, "reason": heldout_reason, "n_train_positive": len(train_pos), "n_test_positive": len(test_pos)})

        strong = _public_samples(strong_rows, texts, "A")
        weak = _public_samples(weak_rows, texts, "B")
        held_public = _public_samples([row for _, row in held_rows], texts, "H")
        held_private = [
            {**sample, "row_idx": int(row), "stratum": stratum, "true_response": float(values[row])}
            for sample, (stratum, row) in zip(held_public, held_rows)
        ]
        packets.append({
            "feature_id": unit.feature_id, "unit_order": int(unit.unit_order),
            "strong_samples": strong, "weak_samples": weak,
            "heldout_samples_private": held_private,
            "scorer_eligible": not heldout_reason,
            "scorer_ineligible_reason": heldout_reason,
        })
        task_id = f"{unit.feature_id}_explainer"
        tasks.append({
            "task_id": task_id, "latent_idx": int(unit.unit_order), "feature_id": unit.feature_id,
            "prompt": build_explainer_prompt(unit.feature_id, strong, weak),
            "expected_output_path": str(raw_dir / f"{task_id}.json"),
        })
        audit.append({
            "feature_id": unit.feature_id,
            "sampling_status": "eligible" if not heldout_reason else "explainer_only",
            "reason": heldout_reason, "n_train_positive": len(train_pos), "n_test_positive": len(test_pos),
            "n_strong": len(strong), "n_weak": len(weak), "n_heldout": len(held_private),
        })

    units.to_csv(private / "unit_mapping.csv", index=False, encoding="utf-8-sig")
    key_to_feature = dict(zip(units["canonical_key"], units["feature_id"]))
    pair_rows = []
    for row in matches.itertuples(index=False):
        sae_key = f"SAE:{int(row.sae_latent_idx)}"
        pca_key = f"PCA-{int(row.pca_dimension)}:{row.label}:{int(row.pca_component_idx)}:{int(row.pca_direction_sign)}"
        pair_rows.append({
            "pca_dimension": int(row.pca_dimension), "label": str(row.label), "pair_order": int(row.pair_order),
            "sae_feature_id": key_to_feature[sae_key], "pca_feature_id": key_to_feature[pca_key],
            "absolute_auc_difference": float(row.absolute_auc_difference),
        })
    pd.DataFrame(pair_rows).to_csv(private / "pair_index.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(output / "sampling_audit.csv", index=False, encoding="utf-8-sig")
    write_jsonl(output / "private_packets.jsonl", packets)
    write_jsonl(output / "explainer" / "tasks.jsonl", tasks)
    manifest = {
        "analysis": ANALYSIS_NAME, "n_frozen_units": len(units), "family_counts": units["representation_family"].value_counts().sort_index().to_dict(),
        "n_matched_pairs": len(pair_rows), "n_explainer_tasks": len(tasks),
        "n_scorer_eligible": sum(bool(packet["scorer_eligible"]) for packet in packets),
        "n_explainer_ineligible": len(explainer_ineligible), "n_scorer_ineligible": len(scorer_ineligible),
        "explainer_ineligible": explainer_ineligible, "scorer_ineligible": scorer_ineligible,
        "discovery": {"strong": 10, "weak_positive": 10},
        "heldout": {"high": 5, "mid": 5, "weak_positive": 5, "nonpositive_control": 5},
        "label_blind": True, "representation_blind": True, "model_visible_numeric_response": False,
        "excluded": ["shuffled_explanation_baseline", "empty_explanation_baseline", "bootstrap_confidence_intervals"],
    }
    write_json(output / "sampling_manifest.json", manifest)
    return manifest


def make_scorer_tasks(*, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    explanations = {row["feature_id"]: row for row in read_jsonl(output / "explainer" / "validated_explanations.jsonl")}
    packets = {row["feature_id"]: row for row in read_jsonl(output / "private_packets.jsonl")}
    raw = output / "scorer" / "raw"; raw.mkdir(parents=True, exist_ok=True)
    tasks = []
    for feature_id in sorted(explanations):
        packet = packets[feature_id]
        if not packet["scorer_eligible"]:
            continue
        task_id = f"{feature_id}_scorer"
        tasks.append({
            "task_id": task_id, "latent_idx": int(packet["unit_order"]), "feature_id": feature_id,
            "prompt": _scorer_prompt(feature_id, explanations[feature_id], packet["heldout_samples_private"]),
            "expected_output_path": str(raw / f"{task_id}.json"),
        })
    write_jsonl(output / "scorer" / "tasks.jsonl", tasks)
    result = {"n_tasks": len(tasks)}; write_json(output / "scorer" / "task_manifest.json", result)
    return result


def validate_scorer_and_score(*, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    tasks = read_jsonl(output / "scorer" / "tasks.jsonl")
    packets = {row["feature_id"]: row for row in read_jsonl(output / "private_packets.jsonl")}
    mapping = pd.read_csv(output / "private" / "unit_mapping.csv")
    mapping_by_id = mapping.set_index("feature_id").to_dict("index")
    predictions, metrics, failures = [], [], []
    for task in tasks:
        try:
            row = json.loads(Path(task["expected_output_path"]).read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append({"task_id": task["task_id"], "reason": f"read:{exc}"}); continue
        expected = [f"H{i:03d}" for i in range(1, 21)]
        got = [item.get("sample_id") for item in row.get("predictions", [])]
        if row.get("feature_id") != task["feature_id"] or got != expected:
            failures.append({"task_id": task["task_id"], "reason": "id_or_order_mismatch"}); continue
        truth = packets[task["feature_id"]]["heldout_samples_private"]
        y = np.asarray([float(item["true_response"]) for item in truth])
        p = np.asarray([float(item["predicted_feature_score"]) for item in row["predictions"]])
        strata = np.asarray([item["stratum"] for item in truth])
        rho = float(spearmanr(p, y).statistic)
        pear = float(pearsonr(p, y).statistic)
        binary = (strata != "control").astype(int)
        auc = float(roc_auc_score(binary, p))
        high, weak = p[strata == "high"], p[strata == "weak"]
        pair = float(np.mean(high[:, None] > weak[None, :]) + 0.5 * np.mean(high[:, None] == weak[None, :]))
        private_row = mapping_by_id[task["feature_id"]]
        metrics.append({
            "feature_id": task["feature_id"], "representation_family": private_row["representation_family"],
            "target_label_private": private_row["label"], "spearman_rho": rho,
            "pearson_response": pear, "positive_vs_control_auroc": auc,
            "high_vs_weak_pair_accuracy": pair,
        })
        predictions.extend({
            "feature_id": task["feature_id"], **pred, "true_response": truth[i]["true_response"],
            "stratum": truth[i]["stratum"],
        } for i, pred in enumerate(row["predictions"]))
    write_jsonl(output / "scorer" / "validated_predictions_private.jsonl", predictions)
    pd.DataFrame(metrics).to_csv(output / "scorer" / "faithfulness_metrics.csv", index=False, encoding="utf-8-sig")
    write_jsonl(output / "scorer" / "validation_failures.jsonl", failures)
    result = {"n_tasks": len(tasks), "n_valid": len(metrics), "n_failed": len(failures)}
    write_json(output / "scorer" / "validation_manifest.json", result)
    return result


def render_report(*, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    analysis_dir = output / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(output / "scorer" / "faithfulness_metrics.csv")
    pairs = pd.read_csv(output / "private" / "pair_index.csv")
    explanations = {row["feature_id"]: row for row in read_jsonl(output / "explainer" / "validated_explanations.jsonl")}
    by_id = metrics.set_index("feature_id")
    pair_rows = []
    for row in pairs.itertuples(index=False):
        if row.sae_feature_id not in by_id.index or row.pca_feature_id not in by_id.index:
            continue
        s, p = by_id.loc[row.sae_feature_id], by_id.loc[row.pca_feature_id]
        pair_rows.append({
            **row._asdict(),
            "sae_spearman": float(s.spearman_rho), "pca_spearman": float(p.spearman_rho),
            "sae_minus_pca_spearman": float(s.spearman_rho - p.spearman_rho),
            "sae_pair_accuracy": float(s.high_vs_weak_pair_accuracy), "pca_pair_accuracy": float(p.high_vs_weak_pair_accuracy),
        })
    comparison = pd.DataFrame(pair_rows)
    comparison.to_csv(analysis_dir / "paired_comparison.csv", index=False, encoding="utf-8-sig")
    summary = metrics.groupby("representation_family").agg(
        n=("feature_id", "count"), mean_spearman=("spearman_rho", "mean"),
        median_spearman=("spearman_rho", "median"), mean_auroc=("positive_vs_control_auroc", "mean"),
        mean_pair_accuracy=("high_vs_weak_pair_accuracy", "mean"),
    ).reset_index()
    summary.to_csv(analysis_dir / "family_summary.csv", index=False, encoding="utf-8-sig")
    summary_lines = [
        "| Family | N | Mean Spearman | Median Spearman | Mean AUROC | Mean high-weak accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        summary_lines.append(
            f"| {row.representation_family} | {int(row.n)} | {row.mean_spearman:.4f} | "
            f"{row.median_spearman:.4f} | {row.mean_auroc:.4f} | {row.mean_pair_accuracy:.4f} |"
        )
    lines = [
        "# SAE–PCA 抽样对比式解释与 Held-out 忠实度报告 / Sampled Contrastive Faithfulness Report", "",
        "> 本报告比较匿名表示单元解释的 held-out 预测忠实度，不提供因果机制证明，也不证明单元等同于 MISC 标签。", "",
        "## Family summary", "", *summary_lines, "",
        f"- Valid matched comparisons: {len(comparison)}/{len(pairs)}", "",
        "## Explanation index / 解释索引", "",
        "| Feature | Family (private audit) | Short name | Type | Confidence |", "|---|---|---|---|---:|",
    ]
    for row in metrics.sort_values(["representation_family", "feature_id"]).itertuples(index=False):
        explanation = explanations[row.feature_id]
        lines.append(f"| {row.feature_id} | {row.representation_family} | {explanation['short_name']} | {explanation['explanation_type']} | {explanation['confidence']}/5 |")
    lines.extend(["", "注意：解释文本由标签盲、表示类型盲的 Explainer 生成；family仅在生成完成后用于私有审计和配对比较。", ""])
    report = analysis_dir / "sae_pca_contrastive_faithfulness_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    result = {"n_metrics": len(metrics), "n_pair_comparisons": len(comparison), "report": str(report)}
    write_json(output / "analysis" / "report_manifest.json", result)
    return result


__all__ = ["ANALYSIS_NAME", "SamplingConfig", "build_packets", "make_scorer_tasks", "render_report", "validate_scorer_and_score"]
