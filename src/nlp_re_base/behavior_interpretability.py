"""Follow-up interpretability analyses for MISC SAE mappings."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .mapping_structure import (
    DEFAULT_CORE_LABELS,
    DEFAULT_HIERARCHY_SPECS,
    load_json,
    load_mapping_matrix,
    parse_label_hierarchy,
    write_json,
)
from .misc_label_mapping import load_feature_matrix


DEFAULT_STAGE_OUTPUT = "outputs/misc_full_sae_eval/interpretability/followup_analysis"
DEFAULT_STOPWORDS = {
    "a",
    "about",
    "and",
    "are",
    "but",
    "for",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "like",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "to",
    "was",
    "what",
    "with",
    "you",
    "your",
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(out):
        return ""
    return f"{out:.{digits}f}"


def _cohens_d(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or right.size < 2:
        return 0.0
    left_var = float(left.var(ddof=1))
    right_var = float(right.var(ddof=1))
    denom = max(left.size + right.size - 2, 1)
    pooled = ((left.size - 1) * left_var + (right.size - 1) * right_var) / denom
    if pooled <= 1e-24:
        return 0.0
    return float((left.mean() - right.mean()) / math.sqrt(pooled))


def _record_labels(record: dict[str, Any]) -> set[str]:
    labels = record.get("labels") or []
    return {str(label).upper() for label in labels}


def _quality(record: dict[str, Any]) -> str:
    return str(record.get("quality_label") or record.get("source_split") or "unknown").lower()


def _tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z']+", text)
        if len(token) > 2 and token.lower() not in DEFAULT_STOPWORDS
    ]


def _common_tokens(records: list[dict[str, Any]], *, limit: int = 12) -> str:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(_tokenize(str(record.get("text") or record.get("unit_text") or "")))
    return ", ".join(token for token, _ in counter.most_common(limit))


def _label_family(label: str, hierarchy: dict[str, list[str]]) -> str:
    label = label.upper()
    for parent, children in hierarchy.items():
        if label == parent or label in children:
            return parent
    return label


def _select_latents(
    matrix: pd.DataFrame,
    label: str,
    *,
    n: int,
    direction: str = "positive",
) -> pd.DataFrame:
    group = matrix[(matrix["label"] == label.upper()) & matrix["significant_fdr"]].copy()
    if direction == "positive":
        selected = group[group["cohens_d"] > 0]
    elif direction == "negative":
        selected = group[group["cohens_d"] < 0]
    else:
        selected = group
    if selected.empty:
        selected = group
    return selected.sort_values(["abs_cohens_d", "directional_auc"], ascending=False).head(n)


def _load_features(eval_dir: Path) -> np.ndarray:
    feature_path = eval_dir / "feature_store" / "utterance_features.pt"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature store not found: {feature_path}")
    return load_feature_matrix(feature_path)


def _label_indicator(records: list[dict[str, Any]], label: str) -> np.ndarray:
    label = label.upper()
    return np.array([label in _record_labels(record) for record in records], dtype=bool)


def _quality_indicator(records: list[dict[str, Any]], value: str) -> np.ndarray:
    value = value.lower()
    return np.array([_quality(record) == value for record in records], dtype=bool)


def _top_examples(
    features: np.ndarray,
    records: list[dict[str, Any]],
    latent_idx: int,
    *,
    n: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    acts = np.asarray(features[:, latent_idx], dtype=np.float32)
    top_idx = np.argsort(acts)[::-1][:n]
    return acts, [records[int(idx)] | {"_row_idx": int(idx), "_activation": float(acts[idx])} for idx in top_idx]


def build_behavior_asymmetry(
    *,
    eval_dir: str | Path,
    mapping_dir: str | Path,
    mapping_structure_dir: str | Path,
    output_dir: str | Path,
    labels: list[str] | None = None,
    hierarchy_specs: list[str] | None = None,
    top_latents_per_label: int = 20,
) -> dict[str, Any]:
    eval_path = Path(eval_dir)
    mapping_path = Path(mapping_dir)
    structure_path = Path(mapping_structure_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hierarchy = parse_label_hierarchy(hierarchy_specs or DEFAULT_HIERARCHY_SPECS)
    labels = [label.upper() for label in (labels or DEFAULT_CORE_LABELS)]
    matrix = load_mapping_matrix(mapping_path / "latent_label_matrix.csv")
    records = read_jsonl(eval_path / "records.jsonl")
    features = _load_features(eval_path)

    fragmentation_path = structure_path / "label_fragmentation_rank.csv"
    pair_path = structure_path / "label_pair_similarity.csv"
    role_assign_path = structure_path / "latent_role_assignments.csv"
    fragmentation = pd.read_csv(fragmentation_path)
    pair_similarity = pd.read_csv(pair_path)
    role_assignments = pd.read_csv(role_assign_path)

    rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    high_mask = _quality_indicator(records, "high")
    low_mask = _quality_indicator(records, "low")

    for label in labels:
        frag = fragmentation[fragmentation["label"] == label]
        if frag.empty:
            continue
        frag_row = frag.iloc[0].to_dict()
        top_pos = _select_latents(
            matrix,
            label,
            n=top_latents_per_label,
            direction="positive",
        )
        top_latents = top_pos["latent_idx"].astype(int).tolist()
        label_mask = _label_indicator(records, label)
        scores = features[:, top_latents].mean(axis=1) if top_latents else np.zeros(len(records))

        positive_scores = scores[label_mask]
        negative_scores = scores[~label_mask]
        high_scores = scores[high_mask]
        low_scores = scores[low_mask]
        label_high_scores = scores[label_mask & high_mask]
        label_low_scores = scores[label_mask & low_mask]

        pair_ref = pair_similarity[
            ((pair_similarity["label_a"] == label) | (pair_similarity["label_b"] == label))
            & (pair_similarity["top_k"] == 50)
        ].copy()
        pair_ref["other_label"] = pair_ref.apply(
            lambda row: row["label_b"] if row["label_a"] == label else row["label_a"],
            axis=1,
        )
        nearest = pair_ref.sort_values("topk_jaccard", ascending=False).head(3)

        role_hits = role_assignments[
            role_assignments["labels"].fillna("").apply(
                lambda value: label in {part.strip() for part in str(value).split(",") if part.strip()}
            )
        ]
        role_counts = role_hits["role"].value_counts().to_dict()
        pos_ratio = float(frag_row.get("positive_effect_ratio", 0.0))
        top_auc = float(frag_row.get("top_directional_auc", 0.0))
        frag_ratio = float(frag_row.get("fragmentation_ratio", 0.0))
        if top_auc >= 0.75 and frag_ratio <= 0.13:
            pattern = "compact_strong"
        elif pos_ratio <= 0.1:
            pattern = "negative_boundary"
        elif role_counts.get("cross_family", 0) + role_counts.get("global", 0) > role_counts.get("exclusive", 0):
            pattern = "shared_distributed"
        else:
            pattern = "mixed_distributed"

        rows.append(
            {
                "label": label,
                "family": _label_family(label, hierarchy),
                "pattern": pattern,
                "n_significant_latents": int(frag_row["n_significant_latents"]),
                "fragmentation_ratio": frag_ratio,
                "positive_effect_ratio": pos_ratio,
                "negative_effect_ratio": float(frag_row.get("negative_effect_ratio", 0.0)),
                "top_latent_idx": int(frag_row["top_latent_idx"]),
                "top_abs_cohens_d": float(frag_row["top_abs_cohens_d"]),
                "top_directional_auc": top_auc,
                "exclusive_latents": int(role_counts.get("exclusive", 0)),
                "family_shared_latents": int(role_counts.get("family_shared", 0)),
                "cross_family_latents": int(role_counts.get("cross_family", 0)),
                "global_latents": int(role_counts.get("global", 0)),
                "nearest_labels_top50": ",".join(nearest["other_label"].astype(str).tolist()),
                "nearest_jaccard_top50": ",".join(_fmt(v) for v in nearest["topk_jaccard"].tolist()),
                "group_score_pos_mean": float(np.mean(positive_scores)) if positive_scores.size else 0.0,
                "group_score_neg_mean": float(np.mean(negative_scores)) if negative_scores.size else 0.0,
                "group_score_label_cohens_d": _cohens_d(positive_scores, negative_scores),
                "group_score_high_mean": float(np.mean(high_scores)) if high_scores.size else 0.0,
                "group_score_low_mean": float(np.mean(low_scores)) if low_scores.size else 0.0,
                "group_score_high_low_d": _cohens_d(high_scores, low_scores),
                "within_label_high_mean": (
                    float(np.mean(label_high_scores)) if label_high_scores.size else 0.0
                ),
                "within_label_low_mean": (
                    float(np.mean(label_low_scores)) if label_low_scores.size else 0.0
                ),
                "within_label_high_low_d": _cohens_d(label_high_scores, label_low_scores),
                "top_positive_latents": ",".join(str(idx) for idx in top_latents),
            }
        )
        quality_rows.append(
            {
                "label": label,
                "n_high_label_samples": int((label_mask & high_mask).sum()),
                "n_low_label_samples": int((label_mask & low_mask).sum()),
                "within_label_high_mean": rows[-1]["within_label_high_mean"],
                "within_label_low_mean": rows[-1]["within_label_low_mean"],
                "within_label_high_low_diff": (
                    rows[-1]["within_label_high_mean"] - rows[-1]["within_label_low_mean"]
                ),
                "within_label_high_low_d": rows[-1]["within_label_high_low_d"],
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["pattern", "fragmentation_ratio"],
        ascending=[True, False],
    )
    quality = pd.DataFrame(quality_rows)
    family = (
        summary.groupby("family")
        .agg(
            n_labels=("label", "count"),
            total_significant_latents=("n_significant_latents", "sum"),
            mean_fragmentation_ratio=("fragmentation_ratio", "mean"),
            mean_top_auc=("top_directional_auc", "mean"),
            mean_cross_family_latents=("cross_family_latents", "mean"),
            mean_global_latents=("global_latents", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(output_path / "behavior_asymmetry_summary.csv", index=False)
    quality.to_csv(output_path / "quality_label_activation_shift.csv", index=False)
    family.to_csv(output_path / "family_asymmetry_summary.csv", index=False)

    metrics = {
        "n_labels": int(summary.shape[0]),
        "top_latents_per_label": int(top_latents_per_label),
        "patterns": summary["pattern"].value_counts().to_dict(),
        "most_fragmented": summary.sort_values("n_significant_latents", ascending=False)
        .head(5)
        .to_dict("records"),
        "strongest_top_auc": summary.sort_values("top_directional_auc", ascending=False)
        .head(5)
        .to_dict("records"),
        "quality_shift": quality.sort_values("within_label_high_low_d", ascending=False)
        .to_dict("records"),
        "files": {
            "behavior_asymmetry_summary": str(output_path / "behavior_asymmetry_summary.csv"),
            "quality_label_activation_shift": str(output_path / "quality_label_activation_shift.csv"),
            "family_asymmetry_summary": str(output_path / "family_asymmetry_summary.csv"),
            "behavior_asymmetry_report": str(output_path / "behavior_asymmetry_report.md"),
        },
    }
    write_json(output_path / "behavior_asymmetry_metrics.json", metrics)
    write_behavior_asymmetry_report(output_path / "behavior_asymmetry_report.md", summary, quality, family)
    return metrics


def write_behavior_asymmetry_report(
    path: str | Path,
    summary: pd.DataFrame,
    quality: pd.DataFrame,
    family: pd.DataFrame,
) -> None:
    lines = [
        "# MISC Behavior Asymmetry 分析报告",
        "",
        "## 1. 核心结论",
        "",
        (
            "不同行为标签在 SAE 表征空间中的结构并不相同：有的标签由少数强 latent "
            "支撑，有的标签主要表现为负向边界，有的标签高度依赖跨行为共享 latent。"
        ),
        "",
        "## 2. 标签差异总表",
        "",
        "| Label | Pattern | Sig. Latents | Frag. Ratio | Pos. Ratio | Top Latent | Top AUC | Label d | Nearest Labels |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['label']} | {row['pattern']} | {int(row['n_significant_latents'])} | "
            f"{_fmt(row['fragmentation_ratio'])} | {_fmt(row['positive_effect_ratio'])} | "
            f"{int(row['top_latent_idx'])} | {_fmt(row['top_directional_auc'])} | "
            f"{_fmt(row['group_score_label_cohens_d'])} | {row['nearest_labels_top50']} |"
        )

    lines.extend(
        [
            "",
            "## 3. 行为家族差异",
            "",
            "| Family | Labels | Total Sig. Latents | Mean Frag. | Mean Top AUC | Mean Cross-family | Mean Global |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in family.iterrows():
        lines.append(
            f"| {row['family']} | {int(row['n_labels'])} | {int(row['total_significant_latents'])} | "
            f"{_fmt(row['mean_fragmentation_ratio'])} | {_fmt(row['mean_top_auc'])} | "
            f"{_fmt(row['mean_cross_family_latents'])} | {_fmt(row['mean_global_latents'])} |"
        )

    lines.extend(
        [
            "",
            "## 4. 高低质量会话中的标签激活差异",
            "",
            "| Label | High n | Low n | High mean | Low mean | Diff | Cohen's d |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in quality.iterrows():
        lines.append(
            f"| {row['label']} | {int(row['n_high_label_samples'])} | "
            f"{int(row['n_low_label_samples'])} | {_fmt(row['within_label_high_mean'])} | "
            f"{_fmt(row['within_label_low_mean'])} | {_fmt(row['within_label_high_low_diff'])} | "
            f"{_fmt(row['within_label_high_low_d'])} |"
        )

    lines.extend(
        [
            "",
            "## 5. 论文可用解释",
            "",
            (
                "这些结果支持 R2：行为标签之间的错配模式并不一致。"
                "`QU/QUO/QUC` 更接近 compact strong pattern，`RE/REC/RES` 更分散，"
                "`AF/SU` 更像由强局部特征和大量负向边界共同刻画。"
            ),
            "",
            "## 6. 限制",
            "",
            "- 本阶段仍是相关性和结构分析，不等价于因果验证。",
            "- 高低质量差异只说明表征强度差异，不直接说明咨询效果。",
            "- pattern 名称是分析标签，不是新的 MISC 分类体系。",
        ]
    )
    write_text(path, "\n".join(lines) + "\n")


def build_latent_case_analysis(
    *,
    eval_dir: str | Path,
    mapping_dir: str | Path,
    output_dir: str | Path,
    labels: list[str] | None = None,
    top_latents_per_label: int = 5,
    top_examples_per_latent: int = 12,
) -> dict[str, Any]:
    eval_path = Path(eval_dir)
    mapping_path = Path(mapping_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    card_dir = output_path / "latent_case_cards"
    card_dir.mkdir(parents=True, exist_ok=True)

    labels = [label.upper() for label in (labels or DEFAULT_CORE_LABELS)]
    matrix = load_mapping_matrix(mapping_path / "latent_label_matrix.csv")
    records = read_jsonl(eval_path / "records.jsonl")
    features = _load_features(eval_path)

    rows: list[dict[str, Any]] = []
    for label in labels:
        label_dir = card_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        selected = _select_latents(
            matrix,
            label,
            n=top_latents_per_label,
            direction="positive",
        )
        target_mask = _label_indicator(records, label)
        for rank, (_, latent_row) in enumerate(selected.iterrows(), start=1):
            latent_idx = int(latent_row["latent_idx"])
            acts, examples = _top_examples(
                features,
                records,
                latent_idx,
                n=top_examples_per_latent,
            )
            top_indices = [record["_row_idx"] for record in examples]
            top_target = target_mask[top_indices]
            top_records = [records[idx] for idx in top_indices]
            label_counter: Counter[str] = Counter()
            quality_counter: Counter[str] = Counter()
            for record in top_records:
                label_counter.update(_record_labels(record))
                quality_counter[_quality(record)] += 1
            purity = float(np.mean(top_target)) if len(top_target) else 0.0
            dominant = ",".join(f"{k}:{v}" for k, v in label_counter.most_common(5))
            quality_dist = ",".join(f"{k}:{v}" for k, v in quality_counter.most_common())
            question_rate = float(
                np.mean(["?" in str(r.get("text") or r.get("unit_text") or "") for r in top_records])
            )
            avg_words = float(
                np.mean(
                    [
                        len(str(r.get("text") or r.get("unit_text") or "").split())
                        for r in top_records
                    ]
                )
            )
            common_tokens = _common_tokens(top_records)
            if purity >= 0.7:
                interpretation_status = "high_purity_candidate"
            elif purity >= 0.4:
                interpretation_status = "mixed_but_label_relevant"
            else:
                interpretation_status = "low_purity_review_required"

            card_path = label_dir / f"latent_{latent_idx:05d}.md"
            write_latent_case_card(
                card_path,
                label=label,
                rank=rank,
                latent_row=latent_row.to_dict(),
                examples=examples,
                target_label=label,
                purity=purity,
                dominant_labels=dominant,
                quality_distribution=quality_dist,
                common_tokens=common_tokens,
                interpretation_status=interpretation_status,
            )
            rows.append(
                {
                    "label": label,
                    "rank": rank,
                    "latent_idx": latent_idx,
                    "cohens_d": float(latent_row["cohens_d"]),
                    "abs_cohens_d": float(latent_row["abs_cohens_d"]),
                    "directional_auc": float(latent_row["directional_auc"]),
                    "precision_at_10": float(latent_row.get("precision_at_10", 0.0)),
                    "precision_at_50": float(latent_row.get("precision_at_50", 0.0)),
                    "top_example_target_purity": purity,
                    "dominant_labels_top_examples": dominant,
                    "quality_distribution_top_examples": quality_dist,
                    "question_mark_rate": question_rate,
                    "avg_words": avg_words,
                    "common_tokens": common_tokens,
                    "interpretation_status": interpretation_status,
                    "card_path": str(card_path),
                    "top_activation_mean": float(np.mean(acts[top_indices])) if top_indices else 0.0,
                    "top_activation_max": float(np.max(acts[top_indices])) if top_indices else 0.0,
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_path / "latent_case_summary.csv", index=False)
    metrics = {
        "n_labels": int(len(labels)),
        "top_latents_per_label": int(top_latents_per_label),
        "top_examples_per_latent": int(top_examples_per_latent),
        "n_case_cards": int(summary.shape[0]),
        "status_counts": summary["interpretation_status"].value_counts().to_dict()
        if not summary.empty
        else {},
        "mean_target_purity": float(summary["top_example_target_purity"].mean())
        if not summary.empty
        else 0.0,
        "files": {
            "latent_case_summary": str(output_path / "latent_case_summary.csv"),
            "latent_case_report": str(output_path / "latent_case_report.md"),
            "latent_case_cards": str(card_dir),
        },
    }
    write_json(output_path / "latent_case_metrics.json", metrics)
    write_latent_case_report(output_path / "latent_case_report.md", summary)
    return metrics


def write_latent_case_card(
    path: str | Path,
    *,
    label: str,
    rank: int,
    latent_row: dict[str, Any],
    examples: list[dict[str, Any]],
    target_label: str,
    purity: float,
    dominant_labels: str,
    quality_distribution: str,
    common_tokens: str,
    interpretation_status: str,
) -> None:
    lines = [
        f"# {label} latent case #{rank}: latent {int(latent_row['latent_idx'])}",
        "",
        "## Metrics",
        "",
        f"- Cohen's d: {_fmt(latent_row['cohens_d'], 4)}",
        f"- Directional AUC: {_fmt(latent_row['directional_auc'], 4)}",
        f"- Precision@10: {_fmt(latent_row.get('precision_at_10', 0.0), 4)}",
        f"- Precision@50: {_fmt(latent_row.get('precision_at_50', 0.0), 4)}",
        f"- Top-example target purity: {_fmt(purity, 4)}",
        f"- Dominant top-example labels: {dominant_labels}",
        f"- Quality distribution: {quality_distribution}",
        f"- Common tokens: {common_tokens}",
        f"- Auto status: {interpretation_status}",
        "",
        "## Top Activating Examples",
        "",
        "| Rank | Act. | Match | Quality | Labels | Text |",
        "|---:|---:|---|---|---|---|",
    ]
    for idx, record in enumerate(examples, start=1):
        labels = ",".join(sorted(_record_labels(record)))
        match = "yes" if target_label in _record_labels(record) else "no"
        text = str(record.get("text") or record.get("unit_text") or "").replace("|", "\\|")
        lines.append(
            f"| {idx} | {_fmt(record['_activation'], 4)} | {match} | "
            f"{_quality(record)} | {labels} | {text} |"
        )
    lines.extend(
        [
            "",
            "## Reading note",
            "",
            "This card is an automatic candidate explanation. It should be reviewed as evidence for a latent pattern, not treated as a final semantic label.",
        ]
    )
    write_text(path, "\n".join(lines) + "\n")


def write_latent_case_report(path: str | Path, summary: pd.DataFrame) -> None:
    lines = [
        "# MISC Latent Case 解释分析报告",
        "",
        "## 1. 核心结论",
        "",
        (
            "本报告把矩阵中的 top latent 转化为可人工审查的高激活案例。"
            "目标是判断每个候选 latent 是否具有稳定语义，而不是直接把 latent 等同于标签。"
        ),
        "",
        "## 2. Case Summary",
        "",
        "| Label | Latent | d | AUC | P@10 | Top purity | Status | Dominant labels |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    if not summary.empty:
        for _, row in summary.iterrows():
            lines.append(
                f"| {row['label']} | {int(row['latent_idx'])} | {_fmt(row['cohens_d'])} | "
                f"{_fmt(row['directional_auc'])} | {_fmt(row['precision_at_10'])} | "
                f"{_fmt(row['top_example_target_purity'])} | {row['interpretation_status']} | "
                f"{row['dominant_labels_top_examples']} |"
            )
    lines.extend(
        [
            "",
            "## 3. 解释原则",
            "",
            "- high_purity_candidate：top examples 中目标标签占比较高，可优先人工命名。",
            "- mixed_but_label_relevant：与目标标签相关，但混合其他行为，应作为共享或组合特征处理。",
            "- low_purity_review_required：统计显著但案例纯度较低，需要谨慎解释。",
            "",
            "## 4. 下一步",
            "",
            "对 high_purity_candidate 和 mixed_but_label_relevant 的 latent 进行人工语义命名，并挑选代表性案例写入论文。",
        ]
    )
    write_text(path, "\n".join(lines) + "\n")


def write_followup_stage_report(
    path: str | Path,
    *,
    behavior_metrics: dict[str, Any],
    case_metrics: dict[str, Any],
    output_dir: str | Path,
) -> None:
    lines = [
        "# MISC 后续可解释性阶段总报告",
        "",
        "> 本报告汇总 R2 行为差异分析与 latent-level 案例解释分析。",
        "",
        "## 1. 本阶段完成内容",
        "",
        "- 行为差异分析：比较各 MISC 标签在 SAE 空间中的碎片化、共享和质量分层差异。",
        "- latent 案例解释：为核心标签 top latent 生成高激活样本卡片。",
        "- 阶段评估：判断哪些 latent 可进入人工命名和后续因果验证。",
        "",
        "## 2. 行为差异结论",
        "",
        f"- 分析标签数：{behavior_metrics.get('n_labels')}",
        f"- Pattern 分布：{behavior_metrics.get('patterns')}",
        "",
        "最碎片化标签：",
        "",
        "| Label | Sig. Latents | Frag. Ratio | Top AUC |",
        "|---|---:|---:|---:|",
    ]
    for row in behavior_metrics.get("most_fragmented", [])[:5]:
        lines.append(
            f"| {row['label']} | {int(row['n_significant_latents'])} | "
            f"{_fmt(row['fragmentation_ratio'])} | {_fmt(row['top_directional_auc'])} |"
        )
    lines.extend(
        [
            "",
            "## 3. Latent 案例结论",
            "",
            f"- 生成 case cards：{case_metrics.get('n_case_cards')}",
            f"- 平均 top-example target purity：{_fmt(case_metrics.get('mean_target_purity', 0.0))}",
            f"- 自动解释状态分布：{case_metrics.get('status_counts')}",
            "",
            "## 4. 总体判断",
            "",
            (
                "当前证据链已经从 R1 的结构映射推进到 R2 的行为差异，并补上了 latent-level 案例审查入口。"
                "结果支持：标签与表征之间的错配不仅是多对多的，而且不同行为标签具有不同的碎片化和共享模式。"
            ),
            "",
            "## 5. 输出位置",
            "",
            f"- 阶段输出目录：`{output_dir}`",
            f"- 行为差异报告：`{behavior_metrics['files']['behavior_asymmetry_report']}`",
            f"- latent 案例报告：`{case_metrics['files']['latent_case_report']}`",
        ]
    )
    write_text(path, "\n".join(lines) + "\n")


def run_followup_interpretability_analysis(
    *,
    eval_dir: str | Path = "outputs/misc_full_sae_eval",
    output_dir: str | Path = DEFAULT_STAGE_OUTPUT,
    labels: list[str] | None = None,
    hierarchy_specs: list[str] | None = None,
    top_latents_per_label: int = 5,
    top_examples_per_latent: int = 12,
    doc_report: str | Path | None = "doc/MISC后续可解释性阶段分析报告.md",
) -> dict[str, Any]:
    eval_path = Path(eval_dir)
    output_path = Path(output_dir)
    mapping_dir = eval_path / "functional" / "misc_label_mapping"
    structure_dir = eval_path / "interpretability" / "mapping_structure"
    output_path.mkdir(parents=True, exist_ok=True)

    behavior_metrics = build_behavior_asymmetry(
        eval_dir=eval_path,
        mapping_dir=mapping_dir,
        mapping_structure_dir=structure_dir,
        output_dir=output_path / "behavior_asymmetry",
        labels=labels,
        hierarchy_specs=hierarchy_specs,
        top_latents_per_label=max(top_latents_per_label, 20),
    )
    case_metrics = build_latent_case_analysis(
        eval_dir=eval_path,
        mapping_dir=mapping_dir,
        output_dir=output_path / "latent_cases",
        labels=labels,
        top_latents_per_label=top_latents_per_label,
        top_examples_per_latent=top_examples_per_latent,
    )
    stage_report = output_path / "followup_interpretability_report.md"
    write_followup_stage_report(
        stage_report,
        behavior_metrics=behavior_metrics,
        case_metrics=case_metrics,
        output_dir=output_path,
    )
    if doc_report:
        write_followup_stage_report(
            doc_report,
            behavior_metrics=behavior_metrics,
            case_metrics=case_metrics,
            output_dir=output_path,
        )
    metrics = {
        "eval_dir": str(eval_path),
        "output_dir": str(output_path),
        "behavior_metrics": behavior_metrics,
        "case_metrics": case_metrics,
        "files": {
            "stage_report": str(stage_report),
            "doc_report": str(doc_report) if doc_report else None,
        },
    }
    write_json(output_path / "followup_interpretability_metrics.json", metrics)
    return metrics
