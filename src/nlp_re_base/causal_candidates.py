"""Prepare label-specific latent groups for downstream causal validation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .mapping_structure import DEFAULT_CORE_LABELS, load_mapping_matrix, write_json


DEFAULT_CAUSAL_CANDIDATE_OUTPUT = (
    "outputs/misc_full_sae_eval/interpretability/causal_candidates"
)
DEFAULT_GROUP_SIZES = (1, 5, 10, 20)
STATUS_SCORE = {
    "high_purity_candidate": 1.0,
    "mixed_but_label_relevant": 0.55,
    "low_purity_review_required": 0.1,
}


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    lo = float(values.min()) if len(values) else 0.0
    hi = float(values.max()) if len(values) else 0.0
    if hi <= lo:
        return pd.Series(np.ones(len(values), dtype=np.float32), index=series.index)
    return (values - lo) / (hi - lo)


def _split_ints(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    out: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def _group_name(size: int) -> str:
    return f"G{size}"


def _ordered_unique(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _build_label_pool(
    *,
    matrix: pd.DataFrame,
    label: str,
    case_summary: pd.DataFrame,
    role_assignments: pd.DataFrame,
    behavior_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    label = label.upper()
    df = matrix[(matrix["label"] == label) & matrix["significant_fdr"]].copy()
    direction = "positive"
    positive = df[df["cohens_d"] > 0].copy()
    if positive.empty:
        positive = df.copy()
        direction = "absolute_fallback"

    if positive.empty:
        return positive, direction

    if not case_summary.empty:
        case_cols = [
            "label",
            "latent_idx",
            "top_example_target_purity",
            "dominant_labels_top_examples",
            "quality_distribution_top_examples",
            "interpretation_status",
            "card_path",
        ]
        available = [col for col in case_cols if col in case_summary.columns]
        if {"label", "latent_idx"}.issubset(available):
            positive = positive.merge(
                case_summary[available],
                on=["label", "latent_idx"],
                how="left",
            )

    if not role_assignments.empty:
        role_cols = [
            "latent_idx",
            "n_labels",
            "labels",
            "positive_labels",
            "negative_labels",
            "direction_type",
            "role",
        ]
        available = [col for col in role_cols if col in role_assignments.columns]
        if "latent_idx" in available:
            positive = positive.merge(
                role_assignments[available],
                on="latent_idx",
                how="left",
            )

    behavior = behavior_summary[behavior_summary["label"] == label]
    if not behavior.empty:
        behavior_row = behavior.iloc[0]
        behavior_top = _split_ints(behavior_row.get("top_positive_latents"))
        behavior_rank = {latent_idx: rank for rank, latent_idx in enumerate(behavior_top, start=1)}
        positive["behavior_top_rank"] = positive["latent_idx"].map(behavior_rank)
        positive["behavior_pattern"] = behavior_row.get("pattern", "")
    else:
        positive["behavior_top_rank"] = np.nan
        positive["behavior_pattern"] = ""

    if "precision_lift_at_50" in positive.columns:
        precision_signal = positive["precision_lift_at_50"]
    elif "precision_at_50" in positive.columns:
        precision_signal = positive["precision_at_50"]
    else:
        precision_signal = pd.Series(np.zeros(len(positive)), index=positive.index)

    auc_signal = (pd.to_numeric(positive["directional_auc"], errors="coerce").fillna(0.5) - 0.5) / 0.5
    auc_signal = auc_signal.clip(lower=0.0, upper=1.0)
    purity = pd.to_numeric(
        positive.get("top_example_target_purity", pd.Series(np.nan, index=positive.index)),
        errors="coerce",
    )
    purity = purity.fillna(pd.to_numeric(positive.get("precision_at_50", 0.0), errors="coerce")).fillna(0.0)
    status = positive.get("interpretation_status", pd.Series("", index=positive.index))
    status_boost = status.map(STATUS_SCORE).fillna(0.0)
    top_rank_boost = positive["behavior_top_rank"].apply(
        lambda rank: 1.0 / float(rank) if pd.notna(rank) and float(rank) > 0 else 0.0
    )

    positive["candidate_score"] = (
        0.40 * _minmax(positive["abs_cohens_d"])
        + 0.20 * auc_signal
        + 0.18 * _minmax(precision_signal)
        + 0.12 * purity.clip(lower=0.0, upper=1.0)
        + 0.06 * status_boost
        + 0.04 * top_rank_boost
    )
    positive["selection_direction"] = direction

    sort_cols = [
        "candidate_score",
        "abs_cohens_d",
        "directional_auc",
        "precision_at_50",
        "latent_idx",
    ]
    ascending = [False, False, False, False, True]
    positive = positive.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    positive.insert(0, "candidate_rank", np.arange(1, len(positive) + 1))
    return positive, direction


def _make_controls(
    pool: pd.DataFrame,
    selected: list[int],
    *,
    k: int,
    seed: int,
) -> dict[str, list[int]]:
    selected_set = set(selected)
    eligible = [int(x) for x in pool["latent_idx"].tolist() if int(x) not in selected_set]
    bottom = _ordered_unique([int(x) for x in pool.sort_values("candidate_score")["latent_idx"].tolist()])
    bottom = [idx for idx in bottom if idx not in selected_set][:k]
    rng = random.Random(seed)
    shuffled = eligible[:]
    rng.shuffle(shuffled)
    return {
        "bottom": bottom[:k],
        "random": shuffled[:k],
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def _write_report(
    *,
    path: Path,
    summary_rows: list[dict[str, Any]],
    output_dir: Path,
    group_sizes: list[int],
) -> None:
    table_rows = []
    for row in summary_rows:
        table_rows.append(
            {
                "Label": row["label"],
                "Pattern": row.get("behavior_pattern", ""),
                "Direction": row.get("selection_direction", ""),
                "Candidates": row.get("n_candidates", 0),
                "G20": row.get("G20", ""),
                "High-purity": row.get("n_high_purity_in_g20", 0),
            }
        )

    text = "\n".join(
        [
            "# MISC Causal Candidate Groups",
            "",
            "本阶段只做候选组整理，不执行模型干预实验。它把已经得到的 MISC latent-label 矩阵、Mapping Structure 角色分类和 latent case card 汇总为后续因果验证可直接读取的 G1/G5/G10/G20 latent 组。",
            "",
            "## 输出文件",
            "",
            f"- 候选目录：`{output_dir}`",
            "- `causal_candidate_groups.json`：每个标签的候选组、bottom control 和 random control。",
            "- `candidate_group_summary.csv`：每个标签的组摘要。",
            "- `label_candidates/<LABEL>_candidate_latents.csv`：每个标签的完整排序候选池。",
            "",
            "## 选择口径",
            "",
            "- 优先选择 FDR 显著且 `cohens_d > 0` 的 latent，代表该标签正向激活更高。",
            "- 若某标签没有正向显著 latent，则回退为绝对效应排序，并在 `selection_direction` 中标记。",
            "- 排序综合 `abs_cohens_d`、`directional_auc`、`precision@50`、case card 纯度和行为差异分析中的 top-positive 排名。",
            "- 这些组仍是候选解释对象，不能直接写成因果结论；真正因果性需要后续 `causal/run_experiment.py` 进行 ablation/steering 验证。",
            "",
            "## 标签候选组摘要",
            "",
            _markdown_table(
                table_rows,
                ["Label", "Pattern", "Direction", "Candidates", "G20", "High-purity"],
            ),
            "",
            "## 后续运行建议",
            "",
            "- 第一轮建议优先跑 `RE`，因为它与既有 RE/NonRE 因果流程兼容。",
            "- 第二轮建议跑 `QU` 或 `QUO`，因为它们在当前 SAE 空间中最紧凑，适合作为正例对照。",
            "- `AF/SU/RES` 更像 negative-boundary 或稀疏边界信号，干预解释要更谨慎，优先观察 necessity 而不是强 sufficiency。",
            f"- 当前导出的组大小：`{', '.join(_group_name(size) for size in group_sizes)}`。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def export_misc_causal_candidates(
    *,
    eval_dir: str | Path = "outputs/misc_full_sae_eval",
    mapping_dir: str | Path | None = None,
    mapping_structure_dir: str | Path | None = None,
    followup_dir: str | Path | None = None,
    output_dir: str | Path = DEFAULT_CAUSAL_CANDIDATE_OUTPUT,
    labels: list[str] | None = None,
    group_sizes: tuple[int, ...] | list[int] = DEFAULT_GROUP_SIZES,
    seed: int = 0,
    doc_report: str | Path | None = "doc/MISC因果验证候选组说明.md",
) -> dict[str, Any]:
    """Export label-specific latent groups for later causal validation."""

    eval_path = Path(eval_dir)
    mapping_path = Path(mapping_dir) if mapping_dir else eval_path / "functional" / "misc_label_mapping"
    structure_path = (
        Path(mapping_structure_dir)
        if mapping_structure_dir
        else eval_path / "interpretability" / "mapping_structure"
    )
    followup_path = (
        Path(followup_dir)
        if followup_dir
        else eval_path / "interpretability" / "followup_analysis"
    )
    output_path = Path(output_dir)
    label_dir = output_path / "label_candidates"
    label_dir.mkdir(parents=True, exist_ok=True)

    labels = [label.upper() for label in (labels or DEFAULT_CORE_LABELS)]
    group_sizes = sorted({int(size) for size in group_sizes if int(size) > 0})
    max_group = max(group_sizes) if group_sizes else 20

    matrix = load_mapping_matrix(mapping_path / "latent_label_matrix.csv")
    case_summary = _read_csv_if_exists(followup_path / "latent_cases" / "latent_case_summary.csv")
    behavior_summary = _read_csv_if_exists(
        followup_path / "behavior_asymmetry" / "behavior_asymmetry_summary.csv"
    )
    role_assignments = _read_csv_if_exists(structure_path / "latent_role_assignments.csv")

    groups: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []

    for label in labels:
        pool, direction = _build_label_pool(
            matrix=matrix,
            label=label,
            case_summary=case_summary,
            role_assignments=role_assignments,
            behavior_summary=behavior_summary,
        )
        if pool.empty:
            groups[label] = {
                "selection_direction": direction,
                "candidate_groups": {},
                "controls": {},
                "n_candidates": 0,
            }
            continue

        candidate_path = label_dir / f"{label}_candidate_latents.csv"
        pool.to_csv(candidate_path, index=False)

        ranked_latents = _ordered_unique([int(x) for x in pool["latent_idx"].tolist()])
        candidate_groups = {
            _group_name(size): ranked_latents[: min(size, len(ranked_latents))]
            for size in group_sizes
        }
        controls = _make_controls(
            pool,
            candidate_groups.get(_group_name(max_group), []),
            k=min(max_group, len(ranked_latents)),
            seed=seed + sum(ord(ch) for ch in label),
        )

        g20 = candidate_groups.get(_group_name(max_group), [])
        g20_df = pool[pool["latent_idx"].isin(g20)]
        high_purity = int(
            (g20_df.get("interpretation_status", pd.Series([], dtype=str)) == "high_purity_candidate").sum()
        )
        behavior_pattern = ""
        behavior = behavior_summary[behavior_summary["label"] == label]
        if not behavior.empty:
            behavior_pattern = str(behavior.iloc[0].get("pattern", ""))

        groups[label] = {
            "selection_direction": direction,
            "behavior_pattern": behavior_pattern,
            "n_candidates": int(len(pool)),
            "candidate_groups": candidate_groups,
            "controls": controls,
            "candidate_csv": str(candidate_path),
        }
        row = {
            "label": label,
            "selection_direction": direction,
            "behavior_pattern": behavior_pattern,
            "n_candidates": int(len(pool)),
            "n_high_purity_in_g20": high_purity,
            "candidate_csv": str(candidate_path),
        }
        for size in group_sizes:
            name = _group_name(size)
            row[name] = ",".join(str(idx) for idx in candidate_groups[name])
        row["bottom_control"] = ",".join(str(idx) for idx in controls["bottom"])
        row["random_control"] = ",".join(str(idx) for idx in controls["random"])
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_path / "candidate_group_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    payload = {
        "eval_dir": str(eval_path),
        "mapping_dir": str(mapping_path),
        "mapping_structure_dir": str(structure_path),
        "followup_dir": str(followup_path),
        "output_dir": str(output_path),
        "labels": labels,
        "group_sizes": group_sizes,
        "seed": seed,
        "groups": groups,
        "files": {
            "candidate_group_summary": str(summary_path),
            "label_candidates_dir": str(label_dir),
        },
    }
    groups_path = output_path / "causal_candidate_groups.json"
    write_json(groups_path, payload)

    report_path = output_path / "causal_candidate_report.md"
    _write_report(
        path=report_path,
        summary_rows=summary_rows,
        output_dir=output_path,
        group_sizes=group_sizes,
    )
    payload["files"]["causal_candidate_groups"] = str(groups_path)
    payload["files"]["causal_candidate_report"] = str(report_path)

    if doc_report is not None:
        doc_path = Path(doc_report)
        _write_report(
            path=doc_path,
            summary_rows=summary_rows,
            output_dir=output_path,
            group_sizes=group_sizes,
        )
        payload["files"]["doc_report"] = str(doc_path)

    metrics = {
        "n_labels": len(labels),
        "group_sizes": group_sizes,
        "summary": summary_rows,
        "files": payload["files"],
    }
    metrics_path = output_path / "causal_candidate_metrics.json"
    write_json(metrics_path, metrics)
    payload["files"]["causal_candidate_metrics"] = str(metrics_path)
    write_json(groups_path, payload)
    return payload
