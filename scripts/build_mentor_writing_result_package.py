"""Build the frozen mentor-sync and paper-writing result package.

The package summarizes current evidence without becoming an experiment SSOT.
All experiment decisions remain governed by docs/current/experiment_workflow.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "outputs" / "rerun_new_dataset_20260716" / "min5_words"
SSOT = PROJECT / "docs" / "current" / "experiment_workflow.md"
STABLE_ROOT = ROOT / "cross_val" / "stable_topk_selection_n20_relaxed_leaf7"
PROBE_ROOT = ROOT / "interpretability" / "representation_probe_comparison_stable_core_leaf7_n20_relaxed"
CARD_ROOT = ROOT / "interpretability" / "contrastive_latent_faithfulness_v3_gpt55_low_full218_randomized_packets_20260719"
TASK5_ROOT = ROOT / "interpretability" / "task5_sae_pca_contrastive_faithfulness_v3_randomized_packets_build_20260719"
ARCHIVE_ROOT = PROJECT / "archive" / "minimal_sufficient_subspace_20260719"
LABELS = ("RES", "REC", "QUO", "QUC", "GI", "SU", "AF")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or (rows[0].keys() if rows else []))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def f(value: Any) -> float:
    return float(value)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(item).replace("\n", " ") for item in row) + " |")
    return "\n".join(lines)


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT.resolve()).as_posix()


def feature_sort_key(feature_id: str) -> tuple[str, int]:
    prefix = "".join(char for char in feature_id if not char.isdigit())
    digits = "".join(char for char in feature_id if char.isdigit())
    return prefix, int(digits or 0)


def markdown_bullets(values: Any, empty_text: str = "未提供") -> list[str]:
    if values is None:
        return [f"- {empty_text}"]
    if isinstance(values, str):
        values = [values]
    items = list(values)
    if not items:
        return [f"- {empty_text}"]
    return [f"- {item}" for item in items]


def render_sentence_cluster(title: str, samples: list[dict[str, Any]]) -> list[str]:
    lines = [f"#### {title}", ""]
    lines.extend(
        f"- **{sample['sample_id']}** — {str(sample['text']).replace(chr(10), ' ')}"
        for sample in samples
    )
    lines.append("")
    return lines


def render_explainer_card_fields(card: dict[str, Any]) -> list[str]:
    lines = [
        "#### 解释字段（模型原文）",
        "",
        f"- **Short name：** {card['short_name']}",
        f"- **Explanation type：** `{card['explanation_type']}`",
        "",
        "**Primary explanation / 总体概括**",
        "",
        str(card["primary_explanation"]),
        "",
        "**Surface or linguistic hypothesis / 表面或语言形式假设**",
        "",
        str(card["surface_or_linguistic_hypothesis"]),
        "",
        "**Behavioral or discourse hypothesis / 行为或话语功能假设**",
        "",
        str(card["behavioral_or_discourse_hypothesis"]),
        "",
        "**Contrastive explanation / 强弱组总体区别**",
        "",
        str(card["contrastive_explanation"]),
        "",
        "**Necessary or characteristic condition / 必要或典型条件**",
        "",
        str(card["necessary_or_characteristic_condition"]),
        "",
        "**Insufficient conditions / 单独出现仍不足的条件**",
        "",
    ]
    lines.extend(markdown_bullets(card.get("insufficient_conditions")))
    lines.extend(["", "**Alternative explanations / 备选解释**", ""])
    lines.extend(markdown_bullets(card.get("alternative_explanations")))
    lines.extend(
        [
            "",
            "**样本角色ID（仅保留ID映射，不展开逐条语言学证据）**",
            "",
            f"- Strong supporting：`{', '.join(card.get('strong_supporting_sample_ids', [])) or '无'}`",
            f"- Strong outlier：`{', '.join(card.get('strong_outlier_sample_ids', [])) or '无'}`",
            f"- Weak boundary supporting：`{', '.join(card.get('weak_boundary_supporting_sample_ids', [])) or '无'}`",
            f"- Weak counterexample：`{', '.join(card.get('weak_counterexample_sample_ids', [])) or '无'}`",
            f"- Representative evidence：`{', '.join(card.get('representative_evidence_ids', [])) or '无'}`",
            "",
            f"**Explainer confidence：** `{card.get('confidence', 'NA')}/5`",
            "",
            f"**Confidence rationale：** {card.get('confidence_rationale', '未提供')}",
            "",
        ]
    )
    return lines


def build_auditable_card_catalog(
    *,
    main_explanations: dict[str, dict[str, Any]],
    main_packets: dict[str, dict[str, Any]],
    main_metrics: dict[str, dict[str, Any]],
    stable_labels: dict[int, set[str]],
    task5_explanations: dict[str, dict[str, Any]],
    task5_packets: dict[str, dict[str, Any]],
    task5_metrics: dict[str, dict[str, Any]],
    task5_pairs: list[dict[str, Any]],
    source_paths: list[tuple[str, Path, int]],
) -> str:
    main_ids = set(main_explanations)
    if main_ids != set(main_packets):
        raise ValueError("Main-card explanation/packet feature IDs do not match")
    if not set(main_metrics).issubset(main_ids):
        raise ValueError("Main-card metrics contain unknown feature IDs")
    task5_ids = set(task5_explanations)
    if task5_ids != set(task5_packets) or task5_ids != set(task5_metrics):
        raise ValueError("Task5 explanation/packet/metric feature IDs do not match")

    pair_notes: dict[str, list[str]] = defaultdict(list)
    for row in task5_pairs:
        dimension = int(float(row["pca_dimension"]))
        label = row["label"]
        sae_id = row["sae_feature_id"]
        pca_id = row["pca_feature_id"]
        pair_notes[sae_id].append(f"{label} / paired PCA-{dimension}: {pca_id}")
        pair_notes[pca_id].append(f"{label} / paired SAE: {sae_id}")

    source_table = markdown_table(
        ["数据角色", "正式源路径", "记录数", "SHA-256"],
        [[role, f"`{rel(path)}`", count, sha256(path)] for role, path, count in source_paths],
    )
    lines = [
        "# 主实验与SAE–PCA对比实验：Feature Card可审计全集",
        "",
        "> Status: Generated from frozen v3 artifacts  ",
        "> 生成方式：确定性文件连接与Markdown渲染；未重新调用AI，未翻译、润色或改写解释。  ",
        "> 解释边界：Card是候选自然语言解释及held-out忠实度证据，不是latent与MISC标签等价或因果机制证明。",
        "",
        "## 1. 收录范围",
        "",
        f"- 主实验：{len(main_explanations)}张有效解释；其中{len(main_metrics)}张具有正式held-out评分。",
        f"- SAE–PCA对比：{len(task5_explanations)}张有效解释，全部具有正式held-out评分。",
        f"- 合计：{len(main_explanations) + len(task5_explanations)}张Card。",
        "- 主实验与对比实验使用各自独立的`feature_id`命名空间；文档锚点使用`main-`和`task5-`前缀避免混淆。",
        "",
        "## 2. 明确排除的字段",
        "",
        "按当前人工审核要求，本文件不渲染以下三个Explainer字段：",
        "",
        "- `contrastive_evidence`：逐条语言学/语义/话语证据说明；",
        "- `limitations`：模型生成的局限列表；",
        "- `possible_confounds`：模型生成的可能混杂因素列表。",
        "",
        "强响应与弱正响应的原始句子簇仍然保留；其余解释字段、样本角色ID、置信度与held-out指标按正式产物原样输出。",
        "",
        "## 3. 审计与连接规则",
        "",
        "- 主实验以`feature_id`连接Explainer packet、validated explanation与Scorer metric，再以`latent_idx`连接宽松Stable Core标签。",
        "- SAE–PCA实验以`feature_id`连接packet、explanation和metric；表示族与私有目标标签来自正式Scorer metric，配对关系来自`paired_comparison.csv`。",
        "- 主实验按Feature编号升序排列；SAE–PCA部分按`SAE → PCA-50 → PCA-100`、再按Feature编号升序排列。",
        "- 所有句子和解释保持源文件原文；Markdown只改变排版。",
        "",
        source_table,
        "",
        "## 4. 主实验Card索引",
        "",
        "<details>",
        f"<summary>展开{len(main_explanations)}张主实验Card索引</summary>",
        "",
        "| Feature | SAE latent | Stable-core labels | Short name | Type | Confidence | Scorer | Spearman |",
        "|---|---:|---|---|---|---:|---|---:|",
    ]
    for feature_id in sorted(main_explanations, key=feature_sort_key):
        card = main_explanations[feature_id]
        latent_idx = int(card["latent_idx"])
        metric = main_metrics.get(feature_id)
        labels = "+".join(sorted(stable_labels.get(latent_idx, set()))) or "—"
        scorer = "scored" if metric else "ineligible"
        spearman = f"{f(metric['spearman_rho']):.3f}" if metric else "—"
        short_name = str(card["short_name"]).replace("|", "\\|")
        lines.append(
            f"| [{feature_id}](#main-{feature_id.lower()}) | {latent_idx} | {labels} | {short_name} | "
            f"{card['explanation_type']} | {card.get('confidence', '—')} | {scorer} | {spearman} |"
        )
    lines.extend(["", "</details>", "", "## 5. 主实验Feature Cards", ""])

    for feature_id in sorted(main_explanations, key=feature_sort_key):
        card = main_explanations[feature_id]
        packet = main_packets[feature_id]
        metric = main_metrics.get(feature_id)
        latent_idx = int(card["latent_idx"])
        labels = ", ".join(sorted(stable_labels.get(latent_idx, set()))) or "未映射"
        lines.extend(
            [
                f'<a id="main-{feature_id.lower()}"></a>',
                "",
                "<details>",
                f"<summary><strong>{feature_id} / SAE latent {latent_idx} / {labels} / {card['short_name']}</strong></summary>",
                "",
                f"### {feature_id} — SAE latent {latent_idx}",
                "",
                f"- **Stable-core labels：** `{labels}`",
                f"- **Task ID：** `{card.get('task_id', 'NA')}`",
                f"- **Scorer status：** `{'scored' if metric else 'scorer_ineligible'}`",
                "",
                "#### Held-out忠实度",
                "",
            ]
        )
        if metric:
            lines.extend(
                [
                    "| Spearman | Pearson(log activation) | Positive-vs-zero AUROC | High-vs-weak accuracy |",
                    "|---:|---:|---:|---:|",
                    f"| {f(metric['spearman_rho']):.4f} | {f(metric['pearson_log_activation']):.4f} | "
                    f"{f(metric['positive_vs_zero_auroc']):.4f} | {f(metric['high_vs_weak_pair_accuracy']):.4f} |",
                    "",
                ]
            )
        else:
            lines.extend(["该Card没有完整正式held-out分层，因此无Scorer指标。", ""])
        lines.extend(render_explainer_card_fields(card))
        lines.extend(render_sentence_cluster("强响应句（10条）", packet["strong_samples"]))
        lines.extend(render_sentence_cluster("弱正响应句 / hard negatives（10条）", packet["weak_samples"]))
        lines.extend(["</details>", "", "---", ""])

    family_order = {"SAE": 0, "PCA-50": 1, "PCA-100": 2}
    task5_order = sorted(
        task5_explanations,
        key=lambda feature_id: (
            family_order.get(task5_metrics[feature_id]["representation_family"], 99),
            feature_sort_key(feature_id),
        ),
    )
    lines.extend(
        [
            "## 6. SAE–PCA对比实验Card索引",
            "",
            "<details>",
            f"<summary>展开{len(task5_explanations)}张SAE–PCA Card索引</summary>",
            "",
            "| Feature | Family | Private target label | Short name | Type | Confidence | Spearman | AUROC |",
            "|---|---|---|---|---|---:|---:|---:|",
        ]
    )
    for feature_id in task5_order:
        card = task5_explanations[feature_id]
        metric = task5_metrics[feature_id]
        short_name = str(card["short_name"]).replace("|", "\\|")
        lines.append(
            f"| [{feature_id}](#task5-{feature_id.lower()}) | {metric['representation_family']} | "
            f"{metric['target_label_private']} | {short_name} | {card['explanation_type']} | "
            f"{card.get('confidence', '—')} | {f(metric['spearman_rho']):.3f} | "
            f"{f(metric['positive_vs_control_auroc']):.3f} |"
        )
    lines.extend(["", "</details>", "", "## 7. SAE–PCA对比实验Feature Cards", ""])

    for feature_id in task5_order:
        card = task5_explanations[feature_id]
        packet = task5_packets[feature_id]
        metric = task5_metrics[feature_id]
        family = metric["representation_family"]
        label = metric["target_label_private"]
        pair_text = "; ".join(pair_notes.get(feature_id, [])) or "无配对记录"
        lines.extend(
            [
                f'<a id="task5-{feature_id.lower()}"></a>',
                "",
                "<details>",
                f"<summary><strong>{feature_id} / {family} / {label} / {card['short_name']}</strong></summary>",
                "",
                f"### {feature_id} — {family}",
                "",
                f"- **Private target label：** `{label}`",
                f"- **Anonymous unit order：** `{card.get('latent_idx', 'NA')}`",
                f"- **Pair mapping：** {pair_text}",
                f"- **Task ID：** `{card.get('task_id', 'NA')}`",
                "",
                "#### Held-out忠实度",
                "",
                "| Spearman | Pearson(response) | Positive-vs-control AUROC | High-vs-weak accuracy |",
                "|---:|---:|---:|---:|",
                f"| {f(metric['spearman_rho']):.4f} | {f(metric['pearson_response']):.4f} | "
                f"{f(metric['positive_vs_control_auroc']):.4f} | {f(metric['high_vs_weak_pair_accuracy']):.4f} |",
                "",
            ]
        )
        lines.extend(render_explainer_card_fields(card))
        lines.extend(render_sentence_cluster("强响应句（10条）", packet["strong_samples"]))
        lines.extend(render_sentence_cluster("弱正响应句 / hard negatives（10条）", packet["weak_samples"]))
        lines.extend(["</details>", "", "---", ""])

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "论文相关文档" / "导师同步与写作结果包_20260719",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty package: {output}")
    output.mkdir(parents=True, exist_ok=True)
    docs = output / "文档"
    stage_docs = docs / "实验设置"
    tables = output / "结果表"
    attachments = output / "附件"

    required = [
        SSOT,
        STABLE_ROOT / "stable_topk_latent_set.csv",
        STABLE_ROOT / "stable_topk_selection_report.md",
        PROBE_ROOT / "probe_macro_summary.csv",
        PROBE_ROOT / "probe_summary_by_label.csv",
        PROBE_ROOT / "reuse_and_refresh_audit.json",
        CARD_ROOT / "explainer" / "validated_explanations.jsonl",
        CARD_ROOT / "public_packets" / "explainer_packets.jsonl",
        CARD_ROOT / "scorer" / "faithfulness_metrics.csv",
        CARD_ROOT / "final_completion_audit.json",
        TASK5_ROOT / "scorer" / "faithfulness_metrics.csv",
        TASK5_ROOT / "explainer" / "validated_explanations.jsonl",
        TASK5_ROOT / "public_packets" / "explainer_packets.jsonl",
        TASK5_ROOT / "analysis" / "family_summary.csv",
        TASK5_ROOT / "analysis" / "paired_comparison.csv",
        TASK5_ROOT / "pipeline_status.json",
        TASK5_ROOT / "packet_integrity_audit.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen inputs:\n" + "\n".join(missing))

    # Data and stable-core summaries.
    label_rows = read_csv(ROOT / "label_matrix.csv")
    data_summary = [
        {
            "label": label,
            "n_positive": sum(int(float(row[label])) for row in label_rows),
            "n_total": len(label_rows),
            "prevalence": sum(int(float(row[label])) for row in label_rows) / len(label_rows),
        }
        for label in LABELS
    ]
    write_csv(tables / "data_label_summary.csv", data_summary)

    stable_all = read_csv(STABLE_ROOT / "stable_topk_latent_set.csv")
    stable = [row for row in stable_all if row["stable_set_role"] == "stable_core"]
    by_label_edges = {label: [row for row in stable if row["label"] == label] for label in LABELS}
    stable_summary = []
    for label in LABELS:
        rows = by_label_edges[label]
        stable_summary.append(
            {
                "label": label,
                "k_star": int(float(rows[0]["top_k_star"])),
                "stable_core_edges": len(rows),
                "selection_status": rows[0]["label_selection_status"],
                "mean_inclusion_frequency": mean(f(row["inclusion_frequency"]) for row in rows),
                "min_inclusion_frequency": min(f(row["inclusion_frequency"]) for row in rows),
            }
        )
    write_csv(tables / "stable_core_summary_by_label.csv", stable_summary)

    latent_labels: dict[int, set[str]] = defaultdict(set)
    for row in stable:
        latent_labels[int(row["latent_idx"])].add(row["label"])
    sharing_distribution = Counter(len(labels) for labels in latent_labels.values())
    sharing_rows = [
        {"labels_per_latent": count, "n_latents": sharing_distribution[count]}
        for count in sorted(sharing_distribution)
    ]
    write_csv(tables / "stable_core_sharing_distribution.csv", sharing_rows)
    overlap_rows = []
    for i, left in enumerate(LABELS):
        left_set = {int(row["latent_idx"]) for row in by_label_edges[left]}
        for right in LABELS[i + 1 :]:
            right_set = {int(row["latent_idx"]) for row in by_label_edges[right]}
            overlap = left_set & right_set
            if overlap:
                overlap_rows.append(
                    {
                        "label_a": left,
                        "label_b": right,
                        "n_shared_latents": len(overlap),
                        "jaccard": len(overlap) / len(left_set | right_set),
                        "latent_indices": ",".join(str(value) for value in sorted(overlap)),
                    }
                )
    write_csv(tables / "stable_core_pair_overlap.csv", overlap_rows)

    # Representation probe summary.
    probe_macro = read_csv(PROBE_ROOT / "probe_macro_summary.csv")
    key_probe_rows: list[dict[str, Any]] = []
    for representation in ("Hidden State", "Full SAE", "Stable Core SAE"):
        row = next(row for row in probe_macro if row["representation"] == representation)
        key_probe_rows.append({**row, "selection": "baseline"})
    for representation in ("Top-n SAE", "PCA-n", "Random SAE-n"):
        candidates = [row for row in probe_macro if row["representation"] == representation]
        row = max(candidates, key=lambda item: f(item["macro_auc"]))
        key_probe_rows.append({**row, "selection": "best_macro_auc"})
    probe_fields = [
        "representation", "top_n", "selection", "n_features_mean", "macro_auc",
        "macro_average_precision", "macro_f1", "macro_balanced_accuracy",
    ]
    write_csv(tables / "representation_key_results.csv", key_probe_rows, probe_fields)

    # Main card faithfulness, joined to stable label edges.
    card_metrics = read_csv(CARD_ROOT / "scorer" / "faithfulness_metrics.csv")
    card_metric_by_latent = {int(row["latent_idx"]): row for row in card_metrics}
    card_overall = {
        "n_scored_latents": len(card_metrics),
        "mean_spearman": mean(f(row["spearman_rho"]) for row in card_metrics),
        "mean_pearson_log_activation": mean(f(row["pearson_log_activation"]) for row in card_metrics),
        "mean_positive_vs_zero_auroc": mean(f(row["positive_vs_zero_auroc"]) for row in card_metrics),
        "mean_high_vs_weak_pair_accuracy": mean(f(row["high_vs_weak_pair_accuracy"]) for row in card_metrics),
    }
    write_csv(tables / "main_card_faithfulness_overall.csv", [card_overall])
    card_label_summary = []
    for label in LABELS:
        metrics = [card_metric_by_latent[int(row["latent_idx"])] for row in by_label_edges[label] if int(row["latent_idx"]) in card_metric_by_latent]
        card_label_summary.append(
            {
                "label": label,
                "n_stable_edges": len(by_label_edges[label]),
                "n_scored_edges": len(metrics),
                "coverage": len(metrics) / len(by_label_edges[label]),
                "mean_spearman": mean(f(row["spearman_rho"]) for row in metrics),
                "mean_pearson_log_activation": mean(f(row["pearson_log_activation"]) for row in metrics),
                "mean_positive_vs_zero_auroc": mean(f(row["positive_vs_zero_auroc"]) for row in metrics),
                "mean_high_vs_weak_pair_accuracy": mean(f(row["high_vs_weak_pair_accuracy"]) for row in metrics),
            }
        )
    write_csv(tables / "main_card_faithfulness_by_label.csv", card_label_summary)

    # Select one high-faithfulness current card per label, preferring unique latents.
    explanations = {row["feature_id"]: row for row in read_jsonl(CARD_ROOT / "explainer" / "validated_explanations.jsonl")}
    packets = {row["feature_id"]: row for row in read_jsonl(CARD_ROOT / "public_packets" / "explainer_packets.jsonl")}
    metric_by_feature = {row["feature_id"]: row for row in card_metrics}
    feature_by_latent = {int(row["latent_idx"]): row["feature_id"] for row in card_metrics}
    selected_examples: list[dict[str, Any]] = []
    used_latents: set[int] = set()
    for label in LABELS:
        candidates = []
        for edge in by_label_edges[label]:
            latent = int(edge["latent_idx"])
            feature_id = feature_by_latent.get(latent)
            if feature_id and feature_id in explanations and feature_id in packets:
                candidates.append((f(metric_by_feature[feature_id]["spearman_rho"]), latent, feature_id))
        candidates.sort(reverse=True)
        chosen = next((item for item in candidates if item[1] not in used_latents), candidates[0])
        _, latent, feature_id = chosen
        used_latents.add(latent)
        selected_examples.append(
            {
                "label": label,
                "latent_idx": latent,
                "feature_id": feature_id,
                **metric_by_feature[feature_id],
                "explanation": explanations[feature_id],
                "packet": packets[feature_id],
            }
        )
    write_csv(
        tables / "representative_card_index.csv",
        [
            {
                "label": row["label"],
                "feature_id": row["feature_id"],
                "latent_idx": row["latent_idx"],
                "short_name": row["explanation"]["short_name"],
                "explanation_type": row["explanation"]["explanation_type"],
                "spearman_rho": row["spearman_rho"],
                "positive_vs_zero_auroc": row["positive_vs_zero_auroc"],
                "high_vs_weak_pair_accuracy": row["high_vs_weak_pair_accuracy"],
            }
            for row in selected_examples
        ],
    )

    # SAE-PCA summaries.
    task5_family = read_csv(TASK5_ROOT / "analysis" / "family_summary.csv")
    task5_pairs = read_csv(TASK5_ROOT / "analysis" / "paired_comparison.csv")
    task5_explanations = {
        row["feature_id"]: row
        for row in read_jsonl(TASK5_ROOT / "explainer" / "validated_explanations.jsonl")
    }
    task5_packets = {
        row["feature_id"]: row
        for row in read_jsonl(TASK5_ROOT / "public_packets" / "explainer_packets.jsonl")
    }
    task5_metric_rows = read_csv(TASK5_ROOT / "scorer" / "faithfulness_metrics.csv")
    task5_metric_by_feature = {row["feature_id"]: row for row in task5_metric_rows}
    paired_summary = []
    for dimension in (50, 100):
        rows = [row for row in task5_pairs if int(row["pca_dimension"]) == dimension]
        paired_summary.append(
            {
                "pca_dimension": dimension,
                "n_pairs": len(rows),
                "mean_sae_minus_pca_spearman": mean(f(row["sae_minus_pca_spearman"]) for row in rows),
                "sae_spearman_wins": sum(f(row["sae_minus_pca_spearman"]) > 0 for row in rows),
                "mean_sae_pair_accuracy": mean(f(row["sae_pair_accuracy"]) for row in rows),
                "mean_pca_pair_accuracy": mean(f(row["pca_pair_accuracy"]) for row in rows),
            }
        )
    write_csv(tables / "task5_paired_summary.csv", paired_summary)
    paired_by_label = []
    for dimension in (50, 100):
        for label in ("REC", "QUO", "QUC", "AF"):
            rows = [row for row in task5_pairs if int(row["pca_dimension"]) == dimension and row["label"] == label]
            paired_by_label.append(
                {
                    "pca_dimension": dimension,
                    "label": label,
                    "n_pairs": len(rows),
                    "mean_sae_minus_pca_spearman": mean(f(row["sae_minus_pca_spearman"]) for row in rows),
                    "sae_spearman_wins": sum(f(row["sae_minus_pca_spearman"]) > 0 for row in rows),
                }
            )
    write_csv(tables / "task5_paired_summary_by_label.csv", paired_by_label)

    # Human-readable documents.
    stable_table = markdown_table(
        ["标签", "K*", "Stable-core边", "标签选择状态"],
        [[row["label"], row["k_star"], row["stable_core_edges"], row["selection_status"]] for row in stable_summary],
    )
    probe_table = markdown_table(
        ["表示", "n", "Macro AUC", "PR-AUC", "F1", "Balanced Acc."],
        [[row["representation"], row["top_n"], fmt(f(row["macro_auc"])), fmt(f(row["macro_average_precision"])), fmt(f(row["macro_f1"])), fmt(f(row["macro_balanced_accuracy"]))] for row in key_probe_rows],
    )
    card_table = markdown_table(
        ["标签", "已评分/Stable边", "Spearman", "AUROC", "High–weak"],
        [[row["label"], f'{row["n_scored_edges"]}/{row["n_stable_edges"]}', fmt(row["mean_spearman"]), fmt(row["mean_positive_vs_zero_auroc"]), fmt(row["mean_high_vs_weak_pair_accuracy"])] for row in card_label_summary],
    )
    task5_table = markdown_table(
        ["表示族", "N", "Spearman均值", "Spearman中位数", "AUROC", "High–weak"],
        [[row["representation_family"], row["n"], fmt(f(row["mean_spearman"])), fmt(f(row["median_spearman"])), fmt(f(row["mean_auroc"])), fmt(f(row["mean_pair_accuracy"]))] for row in task5_family],
    )

    write_text(
        output / "README.md",
        f"""# MISC–SAE 导师同步与论文写作结果包

> Status: Current writing/evidence package
>
> Generated: {date.today().isoformat()}
>
> Experiment SSOT: `../../docs/current/experiment_workflow.md`

本目录整合当前可用于导师同步和论文写作的冻结结果，但不是新的实验规范。任何方法、输入、运行命令或完成门禁冲突均以 [当前唯一事实来源](../../docs/current/experiment_workflow.md) 为准。

## 建议阅读顺序

1. [研究问题与证据链](文档/01_研究问题与证据链.md)
2. [实验设置总览与分阶段文档](文档/02_实验设置冻结摘要.md)
3. [结果总览与科学说明](文档/03_结果总览与科学说明.md)
4. [Stable-core结果说明](文档/04_Stable_Core结果说明.md)
5. [表示信息验证结果说明](文档/05_表示信息验证结果说明.md)
6. [主latent-card解释与可靠性](文档/06_主latent_card解释与可靠性.md)
7. [SAE–PCA解释忠实度对照](文档/07_SAE_PCA解释忠实度对照.md)
8. [代表性latent card](文档/08_代表性latent_card.md)
9. [人工子概念编码计划](文档/09_人工子概念编码计划.md)
10. [论文主张、限制与禁止引用](文档/10_论文主张限制与禁止引用.md)
11. [决策日志](文档/11_决策日志.md)
12. [导师版结果阅读指南](文档/实验设置/结果阅读指南_导师版.md)
13. [主实验与SAE–PCA Card可审计全集](文档/12_主实验与SAE_PCA_Card可审计全集.md)
14. [产物索引](00_产物索引.md)

## 当前状态

| 模块 | 状态 | 可用于论文 |
|---|---|---|
| 新数据与layer-19表征 | FROZEN | 是 |
| 宽松stable-core | FROZEN | 是 |
| 表示probe局部刷新 | FROZEN | 是 |
| 218 latent解释与207 held-out评分 | FROZEN | 是 |
| 48单元SAE–PCA解释对照 | FROZEN | 是，限冻结抽样 |
| 人工子概念编码 | PENDING | 否 |
| Minimal sufficient | ARCHIVED | 否 |
""",
    )

    write_text(
        docs / "01_研究问题与证据链.md",
        """# 研究问题与证据链

## 总体问题

Llama-3.1-8B第19层中，与MISC心理咨询行为标签相关的稀疏内部证据如何组织，这些证据能否稳定复现、保留行为信息并获得可预测未见响应的自然语言解释？

## RQ1：能否找到可复现的候选内部特征？

- 证据：训练折内Cohen’s d排序、AUC@K、20次按`file_id`分组split-half、2000次grouped bootstrap。
- 结果对象：宽松7叶标签stable-core，228条label–latent边、218个去重latent。
- 结论类型：可重复关联候选，不是一一对应或因果机制。

## RQ2：这些特征是否保留标签行为信息？

- 证据：Raw Hidden、Full SAE、Top-n SAE、Stable Core SAE、Random SAE和PCA的统一grouped probe。
- 决策指标：ROC-AUC、PR-AUC、F1、Balanced Accuracy。
- 关键比较：Stable Core与Raw Hidden/Full SAE的性能差距，以及相对Random SAE的增益。

## RQ3：行为信息如何组织？

- 证据：每标签stable-core规模、latent独占/共享、同家族叶标签重叠。
- 当前可回答：多latent、相对标签特异、少量家族内共享。
- 当前不可回答：RE/QU父标签的直接stable-core层级分解，因为主线只分析7叶标签。

## RQ4：内部单元提供什么证据，解释有多可靠？

- 证据：10条强响应与10条弱正响应生成解释；独立20条held-out由Scorer预测响应强弱。
- 指标：Spearman、Pearson、positive-vs-zero AUROC、high-vs-weak排序准确率。
- 解释边界：反映句法、模板、话语功能、主题或情感线索，不自动等于心理机制。

## RQ5：PCA在解释忠实度上是否弱于SAE？

- 证据：16个SAE、16个PCA-50、16个PCA-100匿名单元，形成32个冻结配对。
- 结论范围：只针对REC、QUO、QUC、AF四标签的冻结抽样，不推广到全部PCA component或全部SAE latent。

## 尚待完成的问题

当前尚未完成基于冻结子概念词表的人工编码，因此“card揭示的MI心理学子概念组织结构”仍是待验证分析，不应提前写成最终科学发现。
""",
    )

    write_text(
        docs / "02_实验设置冻结摘要.md",
        f"""# 实验设置冻结摘要

> 本文只提供写作摘要。完整运行命令和完成门禁见 [实验SSOT](../../../docs/current/experiment_workflow.md)。

## 如何阅读

主线被拆成7个可独立阅读的阶段。每份阶段文档都回答：研究子目标是什么、输入是什么、实验如何实现、用什么指标判断、产物在哪里、结果说明什么以及不能说明什么。

| 阶段 | 状态 | 研究子目标 | 独立文档 |
|---|---|---|---|
| 1 | FROZEN | 建立文本、标签、Raw Hidden与SAE特征严格对齐的分析数据 | [数据范围与Layer-19特征](实验设置/阶段1_数据范围与Layer19特征.md) |
| 2 | FROZEN | 从32768维SAE空间排除低质量latent，得到可比较候选池 | [Filtered latent pool](实验设置/阶段2_Filtered_Latent_Pool.md) |
| 3 | FROZEN | 找到与标签相关且能在重复抽样中复现的stable-core候选 | [Stable Core筛选](实验设置/阶段3_Stable_Core筛选.md) |
| 4 | FROZEN | 验证不同表示中是否存在可解码的MISC行为信息 | [表示信息验证](实验设置/阶段4_表示信息验证.md) |
| 5 | FROZEN | 归纳latent证据并用独立held-out句子检验解释忠实度 | [Latent Card与held-out评分](实验设置/阶段5_Latent_Card与Heldout评分.md) |
| 6 | FROZEN | 在统一协议下比较SAE与PCA单元的解释忠实度 | [SAE–PCA解释对照](实验设置/阶段6_SAE_PCA解释忠实度对照.md) |
| 7 | PENDING | 将card人工编码为MISC子概念，分析心理学组织结构 | [人工子概念编码](实验设置/阶段7_人工子概念编码_待执行.md) |
| 阅读指南 | GUIDE | 用非机制可解释性背景也能读懂的方式串联结果、指标和结论边界 | [导师版结果阅读指南](实验设置/结果阅读指南_导师版.md) |

## 全阶段共同冻结口径

- 数据：`data/mi_quality_counseling_misc`。
- 分析单位：不少于5个空格分词词项的counselor `unit_text`。
- 样本：5018条，252个`file_id`。
- 标签：`RES, REC, QUO, QUC, GI, SU, AF`。
- 模型：Llama-3.1-8B；只使用`blocks.19.hook_resid_post`。
- 聚合：token维度max pooling。
- Raw hidden：4096维；SAE：32768维JumpReLU特征。

## Stable-core

- filtered keep pool：12589个latent。
- 候选排序：训练折内正向Cohen’s d。
- K选择：held-out AUC@K，存在稳定平台时使用K_stab，否则使用K_auc。
- 重采样：20次按`file_id`分组split-half，seed=42。
- CI：2000次按`file_id`分组bootstrap。
- 成员规则：`inclusion_frequency >= 0.70`且`cohens_d_ci_lo > 0`。
- 跨质量结果保留为审计，不作为成员硬门禁。

## 表示probe

- 5-fold `StratifiedGroupKFold`，group=`file_id`，seed=42。
- 标准化、PCA拟合和Top-n排序只使用训练折。
- Random SAE重复20次。
- 本次刷新复用3920条配置完全相同的非Stable fold结果，仅重算35条Stable Core SAE fold结果。

## 解释与Scorer

- 模型：GPT-5.5，`reasoning_effort=low`。
- 每单元独立ephemeral请求，零工具调用。
- Explainer：10强＋10弱正响应。
- Scorer：独立5 high＋5 mid＋5 weak-positive＋5 control。
- held-out先冻结、再随机排列、最后赋公开ID；验证只按`sample_id` join。
- 不运行shuffled/empty基线，不计算解释指标bootstrap CI。

## 样本标签规模

{markdown_table(["标签", "阳性数", "占比"], [[row["label"], row["n_positive"], fmt(row["prevalence"])] for row in data_summary])}

## 给首次接触该方向的读者

如果对SAE、latent、probe或解释忠实度还不熟悉，请在阅读各阶段后查看[导师版结果阅读指南](实验设置/结果阅读指南_导师版.md)。该文档从“每个结果究竟回答了什么问题”出发解释数字，不要求预先掌握机制可解释性背景。
""",
    )

    write_text(
        stage_docs / "阶段1_数据范围与Layer19特征.md",
        f"""# 阶段1：数据范围与Llama Layer-19特征

> 状态：FROZEN  
> 导航：[返回实验设置总览](../02_实验设置冻结摘要.md)  
> 执行规范：以[实验SSOT](../../../../docs/current/experiment_workflow.md)为准；本文用于解释实验设计，不另立运行口径。

## 1. 研究子目标

建立一个行级严格对齐的分析数据集，使每一条咨询师语句同时拥有：原始文本、`file_id`、7个MISC叶标签、Llama第19层Raw Hidden表示和SAE稀疏特征。后续所有统计比较都必须基于同一批语句、同一行顺序。

## 2. 为什么需要这一阶段

如果文本、标签、hidden或SAE特征错位，后续即使得到很高的AUC，也可能只是把某条语句的表征错误地配给了另一条语句。因此，本阶段首先固定“研究对象是谁”，再进入latent筛选。

## 3. 输入与冻结范围

- 数据来源：`data/mi_quality_counseling_misc`。
- 分析单位：counselor的`unit_text`，不是完整会话。
- 初始提取：6194条counselor utterance。
- 统计范围：按空格分词后至少5词，最终保留5018条，归档1176条。
- 分组变量：252个`file_id`；后续划分以此为组，避免同一文件跨训练/测试。
- 标签：`RES, REC, QUO, QUC, GI, SU, AF`。
- 模型：Llama-3.1-8B，只读取`blocks.19.hook_resid_post`。
- SAE：OpenMOSS `Llama3_1-8B-Base-L19R-8x`，JumpReLU，`d_model=4096`，`d_sae=32768`。

## 4. 实验实现方法

1. 将每条咨询师语句输入Llama-3.1-8B，取得第19层每个token的residual-stream向量。
2. 在token维度做max pooling，把变长语句压缩为一个4096维Raw Hidden向量。
3. 用固定SAE对同一层表示编码，再在token维度做max pooling，得到一个32768维utterance-level SAE向量。
4. 根据“至少5词”规则生成保留掩码，并用同一掩码同步过滤文本、标签、Raw Hidden和SAE特征。
5. 保存过滤后的四类对象；任何下游脚本只读取该冻结结果根，不再自行改变样本范围。

直观地说：最终矩阵中的第`i`行都指向同一条咨询师语句；区别只是Raw Hidden用4096个连续维度描述它，SAE用32768个稀疏latent响应描述它。

## 5. 主要产物

| 产物 | 规模 | 含义 |
|---|---:|---|
| `records.jsonl` | 5018行 | 文本、`file_id`、质量来源和标签provenance |
| `label_matrix.csv` | 5018行 | 7叶标签目标与分组变量 |
| `feature_store/utterance_activations.pt` | 5018 × 4096 | 第19层Raw Hidden语句表示 |
| `feature_store/utterance_features.pt` | 5018 × 32768 | 第19层SAE语句特征 |

结果根：`outputs/rerun_new_dataset_20260716/min5_words`。

## 6. 完成门禁

- 四类产物第0维均为5018。
- 行顺序完全一致。
- 模型、层、hook和pooling方式与冻结配置一致。
- `source_split=high/low`只表示数据质量来源，不能被误当作训练/测试划分。

## 7. 本阶段得到什么

本阶段得到的是后续实验的统一测量基座，而不是关于latent语义的结论。它保证后续Stable Core、probe和card实验讨论的是同一批5018条语句。

## 8. 限制

`records.jsonl`不包含前一句client utterance，因此不能据此直接声称模型识别了RES/REC相对来访者上文“新增了什么”。当前证据以咨询师语句自身可见内容为边界。

## 9. 标签规模

{markdown_table(["标签", "阳性数", "占比"], [[row["label"], row["n_positive"], fmt(row["prevalence"])] for row in data_summary])}
""",
    )

    write_text(
        stage_docs / "阶段2_Filtered_Latent_Pool.md",
        """# 阶段2：Filtered Latent Pool

> 状态：FROZEN  
> 导航：[返回实验设置总览](../02_实验设置冻结摘要.md)  
> 执行规范：以[实验SSOT](../../../../docs/current/experiment_workflow.md)为准。

## 1. 研究子目标

从32768个SAE维度中排除几乎从不激活、几乎总在激活、缺少方差或由极少数异常值支配的latent，建立后续候选筛选的统一质量池。

## 2. 为什么不能直接使用全部32768维

SAE字典中包含大量在当前咨询数据上不工作的维度。一个只在极少数句子中出现的latent可能产生很大的偶然效应值；一个几乎总激活的latent也难以区分标签。先做与标签无关的激活质量过滤，可以减少这类不可靠候选，同时避免根据目标标签预先挑选特征。

## 3. 输入

- 5018 × 32768的`utterance_features.pt`。
- 与其逐行对齐的`records.jsonl`和`label_matrix.csv`。
- 本阶段过滤只使用latent激活分布；不会根据某个MISC标签决定latent是否进入候选池。

## 4. 实验实现方法

对每个latent统计激活次数、激活率、均值、标准差、非零激活分位数、异常值比例以及Top-1/Top-5激活占比。当前主要门槛为：

- 激活判定阈值：`activation > 1e-8`；
- 至少激活10条语句，且激活率至少0.001；
- 激活率不得超过0.995；
- 标准差必须大于`1e-8`；
- 对激活数不少于20的latent，检查IQR/Z-score异常值比例以及激活质量是否被极少数样本支配；
- Top-1激活占比不得超过0.80，Top-5不得超过0.95，异常值比例不得超过0.50。

只要命中任一排除原因，`keep=False`。同一latent可能同时命中多个原因，因此各drop reason数量不能简单相加。

过滤后，再针对7个叶标签在保留池内计算关联统计表，为下一阶段的Cohen's d排序提供统一输入。

## 5. 主要产物

- `functional/misc_label_mapping_filtered/feature_filter_audit.csv`：逐latent保留状态和诊断统计。
- `functional/misc_label_mapping_filtered/feature_filter_summary.json`：过滤配置与汇总数量。
- `functional/misc_label_mapping_filtered/latent_label_matrix.csv`：保留latent与各标签的关联统计。

## 6. 冻结结果

| 项目 | 数量 |
|---|---:|
| 原始SAE latent | 32768 |
| 保留latent | 12589 |
| 排除latent | 20179 |
| 保留率 | 38.42% |

排除原因中，20127个latent属于`rarely_active`，52个属于`almost_always_active`，10776个属于`zero_or_near_zero_variance`；原因存在重叠。

## 7. 完成门禁

- 后续候选只能来自`feature_filter_audit.csv`中`keep=True`的12589个latent。
- 过滤配置、样本数和输入文件hash必须落盘。
- 不允许后续阶段静默恢复被排除维度。

## 8. 本阶段结果如何理解

12589并不是“与MI有关的latent数量”，而只是“在当前数据上有足够激活质量、值得继续比较的latent数量”。标签相关性和可复现性要到下一阶段才判断。
""",
    )

    write_text(
        stage_docs / "阶段3_Stable_Core筛选.md",
        f"""# 阶段3：AUC@K与宽松Stable Core筛选

> 状态：FROZEN  
> 导航：[返回实验设置总览](../02_实验设置冻结摘要.md)  
> 执行规范：以[实验SSOT](../../../../docs/current/experiment_workflow.md)为准。

## 1. 研究子目标

在训练数据划分发生变化时，仍能找回与某一MISC标签正相关的候选latent，从而区分“可重复的标签关联”与“一次划分中的偶然高分”。

## 2. 输入

- 阶段2保留的12589个filtered latent。
- 5018条语句的7叶标签矩阵。
- `file_id`分组变量。

## 3. 实验实现方法

### 3.1 训练折内关联排序

对每个标签，把该标签阳性语句与阴性语句的latent激活进行比较，并计算正向Cohen's d。d越大，表示该latent在标签阳性语句中的激活相对更高。排序必须只在训练折内完成，不能看到测试折。

### 3.2 用held-out AUC选择集合规模

按训练折排序依次加入Top-K latent，计算`K=0..200`时的held-out分类AUC。`K_auc`定义为首个达到`best_auc - max(0.01, SE)`的K；它回答“需要多少个高排名latent，预测性能已接近最佳”。

### 3.3 20次grouped split-half重复抽样

按`file_id`把数据重复划成两半，共20次，随机种子42。每次重新计算排名，记录Top-K集合重叠和单latent inclusion frequency。频率0.80表示该latent在20次重采样中约有16次进入目标集合。

### 3.4 2000次grouped bootstrap

按`file_id`为单位执行2000次bootstrap，得到每条候选label–latent关联的Cohen's d置信区间。`cohens_d_ci_lo > 0`要求区间下界仍为正，用于排除方向不确定的候选。

### 3.5 宽松成员口径

每个标签先确定`K*`：若存在集合稳定平台则取`K_stab`，否则取`K_auc`。随后只在full-data Top-K*中保留同时满足以下条件的成员：

```text
inclusion_frequency >= 0.70
cohens_d_ci_lo > 0
```

跨high/low质量来源的结果和标签级稳定平台状态保留为审计字段，但不作为成员硬剔除条件。这就是当前唯一有效的“宽松stable-core”口径。

## 4. 冻结结果

{stable_table}

- 总计228条label–latent边。
- 去重后为218个latent。
- 208个latent只关联一个叶标签，10个关联两个标签。
- 共享关系：RES–REC 4个、QUO–QUC 5个、REC–SU 1个。

## 5. 主要产物

- 正式成员表：`cross_val/stable_topk_selection_n20_relaxed_leaf7/stable_topk_latent_set.csv`。
- 选择报告：`stable_topk_selection_report.md`。
- 支撑产物：AUC@K、20次split-half、bootstrap CI与跨质量审计目录。

## 6. 完成门禁

- 只分析7个叶标签，不混入RE/QU父标签。
- split-half必须为20次、group=`file_id`、seed=42。
- 成员必须同时满足Top-K*、频率阈值和正向CI下界。
- 最终必须复现228条边和218个去重latent。

## 7. 结果如何回答研究问题

该结果支持：与MISC标签相关的信息由多个可重复候选latent共同承载，并以标签相对特异的成员为主、少量同家族共享为辅。它不支持“一latent等于一标签”。

## 8. 限制

RES、GI和SU的标签级状态为`performance_only_unstable`。这表示其内部一部分成员满足宽松可复现条件，但整个Top-K集合没有形成稳定平台；论文中不能把这三个标签描述为“标签整体稳定”。
""",
    )

    write_text(
        stage_docs / "阶段4_表示信息验证.md",
        f"""# 阶段4：表示信息验证

> 状态：FROZEN  
> 导航：[返回实验设置总览](../02_实验设置冻结摘要.md)  
> 执行规范：以[实验SSOT](../../../../docs/current/experiment_workflow.md)为准。

## 1. 研究子目标

验证MISC行为标签信息是否存在于Llama第19层表示中、SAE是否保留这类信息，以及定向筛选的Stable Core是否优于随机选择的SAE latent并接近完整表示性能。

## 2. 比较对象

| 表示 | 它回答的问题 |
|---|---|
| Raw Hidden | 原始第19层表示中最多能解码出多少标签信息？ |
| Full SAE | SAE整体是否保留Raw Hidden中的标签信息？ |
| Top-n SAE | 只用训练折内高Cohen's d特征时性能如何？ |
| Stable Core SAE | 可重复候选集合是否具有预测充分性？ |
| Random SAE-n | 任意抽取相同候选池特征能否得到类似结果？ |
| PCA-n | 稠密低维方差方向能保留多少可解码信息？ |

## 3. 实验实现方法

1. 对每个叶标签单独做二分类probe。
2. 使用5-fold `StratifiedGroupKFold`，以`file_id`分组，seed=42；同一文件不会同时进入训练折和测试折。
3. 分类器为`LogisticRegression(class_weight="balanced", C=1.0, solver="liblinear")`。
4. 标准化、PCA拟合和Top-n排序全部只在训练折完成，再应用到测试折，防止信息泄漏。
5. Top-n、PCA-n和Random SAE-n使用`n=10,20,50,100,200`。
6. Random SAE从阶段2的filtered keep pool抽样，每个n重复20次。
7. 指标包括ROC-AUC、PR-AUC、F1和Balanced Accuracy；macro结果是7个标签的平均。

通俗地说，probe不是让模型重新学习咨询行为，而是测试一个简单线性分类器能否从固定表示中读出标签。高AUC说明信息可线性解码，不等于表示具有可读语义或因果作用。

## 4. 本次局部刷新如何实现

Raw Hidden、Full SAE、Top-n SAE、PCA和Random SAE使用的数据、fold与seed没有改变，因此3920条非Stable fold结果从原正式运行逐行复用。Stable Core成员口径改变后，只重算7标签 × 5 folds = 35条Stable Core SAE结果。

复用审计检查了配置、fold大小、标签阳性数、seed、源hash和非Stable逐行结果，均通过。这一做法避免无信息增益的重复运行，同时保持可追溯性。

## 5. Macro结果

{probe_table}

## 6. 主要产物

- `probe_macro_summary.csv`：各表示的macro指标。
- `probe_summary_by_label.csv`：逐标签、逐fold与汇总指标。
- `reuse_and_refresh_audit.json`：3920条复用与35条重算审计。
- `representation_probe_comparison_report.md`和性能曲线图。

## 7. 结果如何回答研究问题

- Raw Hidden AUC 0.8865，Full SAE 0.8762：SAE总体保留了大部分行为标签信息。
- Stable Core SAE只使用平均32.57个特征，AUC仍为0.8528，较Raw Hidden低0.0337、较Full SAE低0.0234。
- Stable Core比Random SAE-200高0.1119，说明其信息不是任取一批latent即可获得。
- PCA-100达到0.9207，表明PCA具有很强线性可解码性；这并不回答PCA component是否容易形成忠实自然语言解释。

## 8. 限制

本实验验证的是相关性的线性可解码信息。它不能证明Stable Core是模型产生行为判断时的必要或充分因果机制，也不能仅凭probe性能给出“SAE比PCA更可解释”的结论。
""",
    )

    write_text(
        stage_docs / "阶段5_Latent_Card与Heldout评分.md",
        f"""# 阶段5：Latent Card生成与Held-out解释评分

> 状态：FROZEN  
> 导航：[返回实验设置总览](../02_实验设置冻结摘要.md)  
> 执行规范：以[实验SSOT](../../../../docs/current/experiment_workflow.md)为准。

## 1. 研究子目标

回答两个问题：一个stable-core latent在什么样的咨询师语句上响应；根据这些句子归纳出的解释，能否预测没有参与解释生成的新句子上的真实响应强弱。

## 2. 为什么分为Explainer和Scorer两阶段

只展示Top激活句并生成流畅解释容易过拟合。为此，Explainer用强响应与弱正响应做对比，弱正响应是“相似但不满足全部条件”的hard negatives；随后Scorer只在独立held-out句子上检验冻结解释。解释生成与解释评估因此使用不同文本。

## 3. Explainer阶段如何实现

### 3.1 输入给AI的内容

- 10条强响应句；
- 10条弱但非零响应句；
- 文本来源的基本背景：口语/转录对话；
- 任务：找出区分强组与弱组的最窄自然语言条件。

AI只看到句子及强/弱组别，不看到token激活、数值激活、SAE背景、MISC标签、latent编号或项目代码。这样做是为了减少标签定义和机制术语对解释的锚定。

### 3.2 输出内容

冻结解释包括简短名称、语言/表面假设、行为/话语功能假设、一句话主解释和综合解释类型。表面线索与功能线索允许同时存在；主解释负责说明该latent最可能依赖哪种组合，而不是强迫二者互斥。

## 4. Held-out Scorer如何实现

每个可评分latent在解释前就冻结20条独立句子：5 high、5 mid、5 weak-positive和5 control。随后按确定性种子随机排列，排列后才赋`H001...H020`公开ID。

Scorer只能看到：

- 冻结解释中的`short_name`、两类hypothesis、`primary_explanation`和`explanation_type`；
- 每条held-out样本的`sample_id`和句子文本。

Scorer看不到真实响应层级、激活值、标签、latent编号、发现集样本，也看不到必要条件、不足条件、混杂因素、局限或Explainer自评置信度。它根据解释预测每条句子的相对响应，最后通过`sample_id`与私有truth连接计分，而不是按数组位置对齐。

## 5. AI执行隔离

- 模型：GPT-5.5，`reasoning_effort=low`。
- 每个latent单独一次`codex exec --ephemeral`请求。
- 使用空目录，忽略个人配置和rules。
- 只读sandbox，禁用Shell、Web、apps/plugins、memory、multi-agent和plan tool。
- 并发上限20；成功请求工具调用总数为0。

这些设置保证第200个任务不会继承前面任务的latent解释上下文。

## 6. 评分指标怎样理解

| 指标 | 它检验什么 |
|---|---|
| Spearman | 预测排序与真实响应排序是否一致 |
| Pearson(log activation) | 预测强度与对数真实激活幅度是否线性一致 |
| Positive-vs-zero AUROC | 能否区分有响应句与零响应句 |
| High-vs-weak accuracy | 能否在强响应和弱正响应之间选对更强者 |

## 7. 冻结完成状态与结果

- Stable-core候选：218个。
- 可构建Explainer packet并通过验证：214个。
- 可构建完整held-out分层并通过Scorer：207个。
- 4个`explainer_ineligible`和7个`scorer_ineligible`均显式记录，没有复制句子或放宽采样。
- packet完整性、迁移与最终完成审计全部pass。

| 指标 | 207个单元均值 |
|---|---:|
| Spearman | {fmt(card_overall["mean_spearman"])} |
| Pearson(log activation) | {fmt(card_overall["mean_pearson_log_activation"])} |
| Positive-vs-zero AUROC | {fmt(card_overall["mean_positive_vs_zero_auroc"])} |
| High-vs-weak accuracy | {fmt(card_overall["mean_high_vs_weak_pair_accuracy"])} |

## 8. 主要产物

- `private/master_packets.jsonl`与`private/heldout_truth.jsonl`：私有采样和计分truth。
- `public_packets/explainer_packets.jsonl`与`scorer_packets.jsonl`：AI可见输入。
- `explainer/validated_explanations.jsonl`：214张冻结解释。
- `scorer/faithfulness_metrics.csv`：207个单元的held-out指标。
- `packet_manifest.json`、`packet_integrity_audit.json`和`final_completion_audit.json`：协议审计。

## 9. 结果如何回答研究问题

平均Spearman 0.6170和AUROC 0.8485说明，多数解释包含能够迁移到未见句子的真实响应规律，而不只是复述发现集句子。不同latent的分数仍有明显差异，因此论文应把解释可靠性视为连续量，并优先展示高忠实度card。

## 10. 限制

held-out忠实度证明“这段自然语言描述能预测latent响应”，不证明描述抓住了模型内部真实因果计算，也不证明模型已经理解MI。本轮没有shuffled/empty explanation基线，也没有为解释指标计算bootstrap CI。
""",
    )

    write_text(
        stage_docs / "阶段6_SAE_PCA解释忠实度对照.md",
        f"""# 阶段6：SAE–PCA解释忠实度对照

> 状态：FROZEN  
> 导航：[返回实验设置总览](../02_实验设置冻结摘要.md)  
> 执行规范：以[实验SSOT](../../../../docs/current/experiment_workflow.md)为准。

## 1. 研究子目标

在相同句子采样、相同AI提示词和相同held-out评分协议下，比较SAE latent与PCA component哪一种更容易形成能够预测未见响应的自然语言解释。

该问题与阶段4不同：阶段4比较“标签信息能否被线性probe读出”，本阶段比较“单个表示方向能否被人类语言忠实描述”。

## 2. 冻结抽样范围

- 标签：REC、QUO、QUC、AF。
- 每标签选择4个去重SAE单元，共16个SAE。
- 对每个SAE匹配一个PCA-50方向和一个PCA-100方向。
- 总计16个SAE、16个PCA-50、16个PCA-100，即48个匿名单元。
- 形成16个SAE–PCA50和16个SAE–PCA100配对，共32个配对。

## 3. 实验实现方法

1. PCA只在训练数据上拟合，并根据其与目标标签的训练折关联确定方向，使正响应方向可比较。
2. 对SAE、PCA-50和PCA-100都使用阶段5相同的10强＋10弱Explainer与独立20条held-out Scorer。
3. 所有held-out在解释前冻结，公开样本只含随机化后的`sample_id + text`，计分按ID join。
4. Explainer和Scorer不知道当前单元来自SAE还是PCA，也不知道目标标签。
5. 三类表示使用同一GPT-5.5低推理配置、Schema、并发与零工具隔离要求。
6. 先比较表示族均值，再在同一标签和匹配单元内计算`SAE Spearman - PCA Spearman`，减少不同抽样对象造成的混淆。

PCA的control样本不强制真实响应等于零，因为稠密PCA方向通常不会产生SAE式的大量精确零值；其余流程保持一致。

## 4. Family结果

{task5_table}

## 5. 配对结果

{markdown_table(["PCA维数", "配对数", "SAE-PCA Spearman", "SAE胜出", "SAE high–weak", "PCA high–weak"], [[row["pca_dimension"], row["n_pairs"], fmt(row["mean_sae_minus_pca_spearman"]), f'{row["sae_spearman_wins"]}/{row["n_pairs"]}', fmt(row["mean_sae_pair_accuracy"]), fmt(row["mean_pca_pair_accuracy"])] for row in paired_summary])}

## 6. 完成门禁与产物

- 48/48 Explainer与48/48 Scorer通过严格验证。
- 32/32配对完整。
- 所有公开packet字段、5/5/5/5分层、随机顺序、hash和discovery/held-out不重叠审计通过。
- 主要产物：`faithfulness_metrics.csv`、`family_summary.csv`、`paired_comparison.csv`、正式报告和`pipeline_status.json`。

## 7. 结果如何回答研究问题

冻结样本内，SAE平均Spearman为0.5996，高于PCA-50的0.3625和PCA-100的0.3128。SAE相对PCA-50平均高0.2371并在11/16对胜出；相对PCA-100平均高0.2868并在13/16对胜出。

因此，当前结果支持“在这组冻结匹配单元中，SAE比PCA更容易获得能预测响应排序的自然语言解释”。它与PCA在阶段4拥有更高probe AUC并不矛盾：PCA可以保留更多线性预测信息，但单个PCA方向仍可能较难用简洁概念描述。

## 8. 限制

该实验不是全量PCA评估，只覆盖4个叶标签和32个配对。不能推广为“所有SAE latent都比所有PCA component更可解释”，也不能据此声称SAE在预测任务上优于PCA。
""",
    )

    write_text(
        stage_docs / "阶段7_人工子概念编码_待执行.md",
        """# 阶段7：人工子概念编码与组织结构分析

> 状态：PENDING，尚未执行  
> 导航：[返回实验设置总览](../02_实验设置冻结摘要.md)  
> 本文件记录计划，不包含已经获得的科学结果。

## 1. 研究子目标

把AI生成且通过held-out评估的latent card映射到可审计的MISC子概念词表，回答：不同标签主要由哪些行为功能、语言代理、主题或转录伪影支撑；哪些子概念标签特异，哪些跨标签共享。

## 2. 为什么需要人工编码

Stable Core只能告诉我们哪些latent与标签稳定相关，held-out Scorer只能告诉我们自然语言解释能否预测响应。要讨论“心理咨询行为在内部如何组织”，还必须把自由文本解释转换成统一、可复核的子概念代码，否则不同措辞无法可靠统计。

## 3. 编码对象与分母

- 主编码对象：214个去重latent/card；每张card只编码一次。
- 关联分析对象：228条label–latent边；共享latent会连接到多个标签。
- 报告必须同时给出“按214个去重latent”和“按228条边”两种分母，不能混用。
- held-out忠实度可作为card可靠性字段，但不自动替代人工判断。

## 4. 计划实现方法

1. 从推荐的MISC子概念词表开始，字段包括代码、定义、包含例、排除例和上位类别。
2. 先抽20–30张card做试编码，检查词表是否覆盖功能、表面线索和伪影证据。
3. 根据试编码修订一次后冻结`codebook_v1`；正式编码期间不随意增加近义代码。
4. 每个latent指定一个主子概念代码，依据完整card中的功能解释、表面解释和一句话概括共同判断。
5. 至少20%的card由第二名编码者独立复核，记录一致率、分歧原因与裁决结果。
6. 汇总每标签子概念频率、latent共享、子概念共享以及高/低忠实度解释的分布。

## 5. 计划输出

- `codebook_v1.md`
- `latent_subconcept_coding.csv`
- `coding_disagreement_log.csv`
- `subconcept_summary_by_label.csv`
- `shared_subconcept_summary.csv`
- `human_coding_report.md`

## 6. 完成门禁

- 214张当前v3 card全部有明确状态：已编码、不可编码或待裁决。
- 词表版本、编码者和裁决过程可追溯。
- 至少20%完成独立复核并报告一致性。
- 汇总结果与Stable Core边表按latent ID正确连接。
- 论文区分“行为功能证据”“语言代理/模板”“主题内容”和“潜在伪影”。

## 7. 完成后能够回答什么

完成后才能系统回答：解释后的稀疏特征集合揭示了怎样的MI行为组织结构、各标签依赖哪些心理学子概念、哪些证据跨标签共享，以及高忠实度证据主要属于功能还是语言代理。

## 8. 当前不能写什么

在人工编码与复核完成前，不能把旧225-card分组或AI自由文本聚类当作当前主线的心理学组织结构发现，也不能声称214张card已经自动给出了最终子概念体系。
""",
    )

    write_text(
        stage_docs / "结果阅读指南_导师版.md",
        f"""# 导师版结果阅读指南：这些实验结果应该怎么读

> 文档类型：GUIDE，不是新的实验阶段  
> 适用读者：第一次接触SAE、latent或机制可解释性研究的导师与合作者  
> 导航：[返回实验设置总览](../02_实验设置冻结摘要.md)  
> 数据、命令和口径冲突时，以[实验SSOT](../../../../docs/current/experiment_workflow.md)为准。

## 1. 先用一句话理解这项研究

这项研究不是试图证明“某一个神经元就是某一种咨询行为”，而是在问：**Llama第19层中与MISC咨询行为标签相关的信息，能否被压缩为一组可重复出现的稀疏内部证据；这些证据能否被自然语言描述，并在未见句子上验证这种描述是否可靠。**

目前最简洁的结果是：

1. MISC标签信息可以从Llama第19层和SAE表示中被线性读出。
2. 宽松Stable Core把候选压缩为228条label–latent关联、218个去重latent，而且明显优于随机选择的SAE特征。
3. 这些latent并非都能被清楚解释，但207个可评分单元的自然语言解释平均能够较好地预测未见句子的响应。
4. 在冻结的32个SAE–PCA配对中，SAE单元的解释忠实度整体高于PCA方向。
5. 这些card具体形成了哪些心理学子概念结构，仍需阶段7人工编码后才能回答。

## 2. 先理解几个核心名词

| 名词 | 通俗解释 | 在本研究中的作用 |
|---|---|---|
| Utterance | 一条咨询师说的话 | 本研究的基本分析单位，共5018条 |
| Raw Hidden | Llama第19层对整条语句形成的4096维内部向量 | 作为原始内部表示基线 |
| SAE | 把稠密hidden表示拆成大量稀疏响应维度的方法 | 产生32768个候选latent |
| Latent | SAE中的一个内部响应维度 | 可能响应某类词汇、句式、话语功能、主题或多者组合 |
| Label–latent边 | 某个latent与某个MISC标签之间的一条统计关联 | 同一latent可与两个标签关联，因此边数可以大于latent数 |
| Stable Core | 在重复分组抽样中仍反复被选中、且效应方向CI为正的候选关联 | 用来减少一次数据划分造成的偶然发现 |
| Probe | 在固定表示上训练的简单线性分类器 | 检验标签信息能否从表示中被读出 |
| Latent card | 用强响应句和弱正响应句归纳出的自然语言解释 | 说明一个latent可能在检测什么证据 |
| Held-out | 没有参与解释生成的新句子 | 检验解释能否预测未见响应，而不是只复述生成样例 |
| PCA | 把hidden压缩为最大方差方向的稠密降维方法 | 作为预测信息与解释忠实度的强对照 |

## 3. 整个证据链应该怎样连起来

| 研究问题 | 使用的实验 | 当前结果 | 能得出的结论 |
|---|---|---|---|
| 标签信息是否存在？ | Raw Hidden、Full SAE和PCA线性probe | AUC均明显高于随机水平 | 第19层表示中存在可线性解码的MISC标签信息 |
| 少量定向latent是否仍含信息？ | Stable Core、Top-n和Random SAE对照 | Stable Core AUC 0.8528，Random SAE-200为0.7409 | 定向stable候选比任意抽取latent更有信息 |
| 信息由一个还是多个latent组成？ | 每标签Stable Core规模和共享分析 | 228条边、218个latent，大部分标签特异、少量共享 | 更符合多latent、多对多和局部共享结构 |
| Latent究竟响应什么？ | 10强＋10弱正Explainer | 214张有效card | 获得可审计的候选自然语言解释 |
| 解释是否只是在讲故事？ | 独立20条held-out Scorer | 207个单元平均Spearman 0.6170 | 解释对未见响应具有预测忠实度，但强弱因latent而异 |
| SAE是否比PCA更可解释？ | 统一匿名协议下的32个配对 | SAE配对Spearman整体更高 | 只在当前冻结抽样中支持SAE解释更忠实 |
| 心理学子概念如何组织？ | 人工词表编码 | 尚未执行 | 当前还不能给出最终心理学组织结构结论 |

## 4. Stable Core结果怎么读

### 4.1 为什么是228条边，但只有218个latent

“边”指标签与latent之间的关联。例如一个latent同时进入QUO和QUC的Stable Core，它会产生两条边，但仍然只算一个去重latent。因此：

- 228是论文分析标签关联时使用的分母；
- 218是生成和统计去重latent card时使用的候选数；
- 不能把218写成218个独立心理学概念。

### 4.2 独占和共享意味着什么

- 208个latent只关联一个叶标签；
- 10个latent关联两个标签；
- 共享主要出现在RES–REC、QUO–QUC和REC–SU。

这说明内部结构以“相对标签特异的证据”为主，同时保留少量同类行为共享证据。但“只关联一个标签”不等于该latent在神经网络中只服务该标签，它只表示在当前统计口径下没有进入其他标签的Stable Core。

### 4.3 `performance_only_unstable`怎么读

RES、GI和SU的部分latent满足当前宽松成员条件，因此可以进入Stable Core；但这三个标签的整个Top-K集合没有形成清楚的稳定平台。稳妥表述是“标签内部存在可重复候选成员”，不能写成“该标签整体具有稳定内部集合”。

详细结果见[阶段3：Stable Core筛选](阶段3_Stable_Core筛选.md)。

## 5. Probe结果怎么读

### 5.1 AUC表示什么

ROC-AUC可以理解为：随机抽取一条标签阳性语句和一条阴性语句时，分类器把阳性排得更高的概率。0.5接近随机，1.0表示完全区分。它衡量表示中是否含有可线性读取的信息，不衡量概念是否易懂，也不证明因果作用。

| 表示 | Macro AUC | 应该怎样解释 |
|---|---:|---|
| Raw Hidden | 0.8865 | 原始第19层包含较强标签信息 |
| Full SAE | 0.8762 | SAE整体保留了大部分Raw Hidden信息 |
| Top-n SAE最佳 | 0.8655 | 训练折内定向排名的少量latent已能保留较多信息 |
| Stable Core SAE | 0.8528 | 平均约32.57个可重复latent仍保留较多信息 |
| Random SAE-200 | 0.7409 | 随机抽取即使数量更多，表现仍明显较低 |
| PCA-100 | 0.9207 | 稠密低维PCA方向具有很强线性可解码性 |

### 5.2 最关键的比较

Stable Core相对Raw Hidden低0.0337，相对Full SAE低0.0234，但相对Random SAE-200高0.1119。核心信息不是“Stable Core获得最高分”，而是“用很少且可重复的定向latent，保留了接近完整表示的标签信息，并明显优于随机latent”。

### 5.3 为什么PCA AUC更高，却可能更难解释

Probe和解释实验测量的是两个不同问题：

- Probe问：多个维度组合后能否预测标签？
- Card问：一个单独方向的响应规律能否被简洁自然语言描述，并预测新句子？

PCA专门保留数据方差，多个PCA方向组合后很适合分类；但单个PCA方向可能混合许多语言和行为因素。SAE的预测AUC可以略低，却可能把响应拆成更局部、更容易命名的单位。因此“PCA预测更强”和“SAE单元解释更忠实”可以同时成立。

详细结果见[阶段4：表示信息验证](阶段4_表示信息验证.md)。

## 6. Latent card和held-out分数怎么读

### 6.1 Card不是只看Top句子起名字

Explainer同时看10条强响应和10条弱但非零响应。弱正响应通常与强响应较相似，相当于hard negatives。例如强句都出现疑问词时，弱句也可能包含疑问词，这会迫使AI进一步判断latent响应的是某个具体疑问模板、提问功能，还是二者组合。

### 6.2 为什么还需要另一个Scorer

一个解释在生成句子上听起来合理并不够。Scorer拿到冻结解释和20条完全未参与解释生成的句子，预测这些句子的相对响应；再与私有真实激活比较。这样评估的是解释能否外推，而不是措辞是否流畅。

### 6.3 四个指标分别意味着什么

| 指标 | 当前均值 | 直观解释 |
|---|---:|---|
| Spearman | {fmt(card_overall["mean_spearman"])} | 解释预测的响应排序与真实排序具有中等偏强一致性 |
| Pearson(log activation) | {fmt(card_overall["mean_pearson_log_activation"])} | 预测强度与对数激活幅度具有中等偏强线性一致性 |
| Positive-vs-zero AUROC | {fmt(card_overall["mean_positive_vs_zero_auroc"])} | 解释较好地区分“会响应”和“零响应”句子 |
| High-vs-weak accuracy | {fmt(card_overall["mean_high_vs_weak_pair_accuracy"])} | 约78%的强—弱正配对中，解释把强响应句排在更高位置 |

这些是207个可评分单元的平均值，不代表每张card都同样可靠。论文示例应同时展示解释文本、强弱句簇和该card自己的held-out指标。

### 6.4 高分和低分分别说明什么

- 高Spearman、高AUROC且high–weak也高：解释覆盖了较完整的响应边界，适合作为强案例。
- AUROC高但high–weak较低：解释知道“是否会响应”，但没有准确抓住强弱程度。
- 发现句簇看起来整齐但held-out低：解释可能过拟合Top句子或命名得过细。
- 功能解释听起来合理但表面模板更稳定：该latent可能主要是语言代理，不能直接当作核心心理学功能证据。

详细结果见[阶段5：Latent Card与Held-out评分](阶段5_Latent_Card与Heldout评分.md)，具体案例见[代表性latent card](../08_代表性latent_card.md)。

## 7. SAE–PCA解释对照怎么读

本实验不是比较两种表示谁的分类AUC更高，而是把16个SAE单元分别与PCA-50和PCA-100方向匹配，使用完全相同的匿名Explainer与Scorer协议。

- SAE平均Spearman：0.5996；
- PCA-50：0.3625；
- PCA-100：0.3128；
- SAE相对PCA-50平均高0.2371，在11/16对胜出；
- SAE相对PCA-100平均高0.2868，在13/16对胜出。

这支持“当前冻结样本中，SAE单元更容易获得能预测未见响应的自然语言解释”。由于只覆盖REC、QUO、QUC、AF和32个配对，不能推广为所有SAE都优于所有PCA。

详细结果见[阶段6：SAE–PCA解释忠实度对照](阶段6_SAE_PCA解释忠实度对照.md)。

## 8. 阅读一张具体card时看什么

建议按以下顺序，而不是只看一句解释：

1. **关联对象**：该latent进入哪个标签的Stable Core，是否与其他标签共享。
2. **强弱句簇**：强句有什么共同点，弱正句保留了哪些相似表面形式。
3. **表面假设**：是否主要依赖词汇、疑问词、句法或模板。
4. **行为假设**：是否体现提问、反映、肯定、建议等话语功能。
5. **主解释**：是否清楚说明表面形式与行为功能的关系，而不是堆叠两个独立标签。
6. **Held-out指标**：解释能否预测新句子；若分数不高，应把card当作不确定证据。
7. **标签角色**：只有在人工编码后，才能判断它是核心行为证据、语言代理、标签特异证据还是跨标签共享证据。

## 9. 结论强度分三级

### 9.1 当前可以直接支持

- Llama第19层和SAE空间中存在可线性解码的MISC叶标签信息。
- 通过重复分组抽样和bootstrap可以找到一组可重复的label–latent候选关联。
- Stable Core用较少特征保留了较多标签信息，并优于随机SAE子集。
- 部分自然语言解释能够预测未见句子的latent响应。
- 当前冻结抽样中，SAE解释忠实度高于匹配PCA方向。

### 9.2 需要带限定词

- “稳定”：指当前宽松统计规则下的可重复关联，不是跨模型、跨层或跨数据集稳定。
- “解释可靠”：指held-out响应预测较好，不是解释一定对应真实内部算法。
- “标签特异”：指没有进入其他叶标签Stable Core，不是神经网络中只服务一个标签。
- “SAE更可解释”：只限当前四标签、32个配对的冻结抽样。

### 9.3 当前不能支持

- 一个latent等价于一个MISC标签。
- 模型已经理解MI或具有人类咨询师式心理机制。
- Probe AUC高就证明表示更可解释。
- Card高分就证明该latent因果地产生某种行为。
- 214张解释已经自动揭示最终心理学子概念结构。

## 10. 导师可能会问的几个问题

### Q1：为什么只研究第19层？

本轮把模型与层固定，是为了避免跨层选择自由度污染主实验。当前结论因此是第19层的局部结果，不声称代表整个模型；跨层复现可作为未来扩展。

### Q2：为什么不用随机负例，而用弱正响应句？

随机负例通常与强句差异太大，AI容易生成过宽或过细的解释。弱正响应与强句更相似，可以迫使AI找出真正区分高响应与低响应的最窄条件。

### Q3：为什么不给Explainer看激活值？

当前任务是从句子内容归纳概念边界，而不是让AI从数值猜测机制。只提供强/弱相对组别还能减少AI对具体数值分布的过拟合。

### Q4：218个latent就是218个咨询心理学概念吗？

不是。它们是统计稳定候选，其中可能包含心理学功能、语言形式、主题内容或转录伪影。需要阶段7人工词表编码后，才能统计这些类型的组成。

### Q5：现在最缺的结果是什么？

最直接的缺口不是再跑更多AI解释，而是完成214张当前v3 card的人工子概念编码与复核。该阶段决定我们能否从“稳定且可解释的latent”进一步写到“咨询行为证据如何在内部组织”。

## 11. 建议的10分钟阅读顺序

1. [结果总览与科学说明](../03_结果总览与科学说明.md)：先理解论文故事。
2. [阶段3 Stable Core](阶段3_Stable_Core筛选.md)：理解候选如何获得。
3. [阶段4 表示信息验证](阶段4_表示信息验证.md)：理解信息保留证据。
4. [阶段5 Latent Card](阶段5_Latent_Card与Heldout评分.md)：理解解释如何验证。
5. [代表性latent card](../08_代表性latent_card.md)：查看具体句簇。
6. [阶段6 SAE–PCA](阶段6_SAE_PCA解释忠实度对照.md)：理解解释性对照。
7. [阶段7人工编码](阶段7_人工子概念编码_待执行.md)：理解下一步及当前结论缺口。

## 12. 可直接用于汇报的结果表述

> 我们在固定的Llama-3.1-8B第19层上，首先确认MISC咨询行为标签可以从原始hidden和SAE表示中线性解码；随后通过训练折内效应排序、20次按会话文件分组的重复抽样和grouped bootstrap，得到228条可重复的叶标签–latent关联，对应218个去重latent。由这些候选构成的Stable Core仅使用平均约33个特征，仍保留接近完整表示的标签信息，并明显优于随机SAE子集。进一步地，我们用强响应与弱正响应句生成自然语言解释，再用独立随机化held-out句子检验解释能否预测真实响应，207个可评分单元取得平均Spearman 0.617和AUROC 0.849。在统一匿名协议的冻结抽样中，SAE单元的解释忠实度高于匹配PCA方向。当前证据支持可重复关联、表示可解码性和解释忠实度，但不构成因果机制或模型已经理解MI的证明；心理学子概念组织仍需人工编码完成。
""",
    )

    write_text(
        docs / "03_结果总览与科学说明.md",
        f"""# 结果总览与科学说明

## 1. 找到了稳定、包含信息的候选内部证据

宽松成员口径得到228条label–latent边和218个去重latent。208个latent只进入一个叶标签，10个latent进入两个叶标签。当前结构因此不是单个latent对应单个标签，而是以标签相对特异的多latent集合为主，并伴随少量共享证据。

## 2. Stable-core保留了接近完整表示的行为信息

Stable Core SAE仅使用平均32.57个特征，macro AUC为0.8528；Raw Hidden为0.8865，Full SAE为0.8762。Stable Core相对Raw Hidden低0.0337、相对Full SAE低0.0234，但比Random SAE-200高0.1119。

这支持：定向筛选出的少量稀疏特征保留了大部分线性可解码的MISC标签信息。它不证明这些latent是因果必要条件，也不证明PCA因AUC更高而更可解释。

## 3. 自然语言解释能预测未见响应

218个候选latent中214个可生成解释，207个可完成严格held-out评分。207个单元平均Spearman为0.6170、Pearson为0.6101、positive-vs-zero AUROC为0.8485、high-vs-weak准确率为0.7832。

因此，部分解释不仅在发现句簇上“听起来合理”，还能预测未参与解释生成的句子响应。但指标仍存在latent间差异，不能将所有card视为同等可靠。

## 4. 冻结抽样中SAE解释忠实度高于PCA

SAE平均Spearman为0.5996，PCA-50为0.3625，PCA-100为0.3128。SAE相对PCA-50的配对Spearman平均高0.2371，相对PCA-100高0.2868。

这说明在相同匿名解释流程、相同句子数量和相同held-out评分口径下，所抽取SAE单元的自然语言解释更能预测响应排序。结论只适用于当前32个配对。

## 5. 当前最稳妥的论文故事

Llama layer-19中的MISC行为信息可在SAE空间中被压缩为可重复的多latent候选集合；这些集合整体具有较强线性可解码性，其中一部分latent还具有可跨未见句子验证的自然语言解释。行为标签主要由多个相对特异的内部证据共同支撑，并存在少量同家族共享。冻结抽样进一步显示，SAE单元比匹配PCA方向更容易获得忠实的响应解释。

## 尚未完成

对214张当前card进行人工子概念编码后，才能回答这些证据在心理学上具体形成了哪些功能簇、语言代理和跨标签共享子概念。
""",
    )

    write_text(
        docs / "04_Stable_Core结果说明.md",
        f"""# Stable Core结果说明

## 冻结结果

{stable_table}

- 总label–latent边：228。
- 去重latent：218。
- 单标签latent：208。
- 两标签共享latent：10。
- RES–REC共享4个；QUO–QUC共享5个；REC–SU共享1个。

## 科学解释

结果更符合“标签由多个碎片化证据共同表示”，而不是“一标签一latent”。共享主要发生在同一行为家族的叶标签之间，但共享比例较低，说明家族共性与叶标签差异同时存在。

## 必须保留的限定

- RES、GI、SU的标签级状态是`performance_only_unstable`：其内部成员满足宽松可复现条件，但不能声称整个标签集合形成稳定平台。
- 跨质量差异属于稳健性审计，不是当前成员硬剔除条件。
- 主线没有RE/QU父标签stable-core，不能直接写父子层级分解。
""",
    )

    write_text(
        docs / "05_表示信息验证结果说明.md",
        f"""# 表示信息验证结果说明

## Macro结果

{probe_table}

## 结果解释

1. Raw Hidden和Full SAE均具有较强MISC标签可解码性，说明SAE总体保留了大部分行为信息。
2. Stable Core SAE以平均32.57个特征取得0.8528 macro AUC，接近Full SAE的0.8762和Raw Hidden的0.8865。
3. Stable Core明显高于相同候选池中的Random SAE-200，说明性能来自定向选择，而不仅是任取若干激活特征。
4. PCA-100达到最高macro AUC 0.9207，但这是线性可解码性，不是解释性。PCA是否更难解释必须由独立card忠实度实验回答。

## 复用审计

- 原样复用非Stable fold结果：3920条。
- 重算Stable Core SAE：35条label-fold结果。
- 复用配置、fold规模、标签阳性数、seed及源hash检查：全部pass。
- 非Stable结果与旧表逐行完全一致。
""",
    )

    write_text(
        docs / "06_主latent_card解释与可靠性.md",
        f"""# 主latent-card解释与可靠性

## 完成状态

- stable-core候选：218。
- Explainer可采样且有效：214/214。
- Scorer可采样且有效：207/207。
- 工具调用：0。
- 随机化packet、迁移和最终完成审计：全部pass。

## 总体指标

- Spearman：{fmt(card_overall["mean_spearman"])}。
- Pearson(log activation)：{fmt(card_overall["mean_pearson_log_activation"])}。
- Positive-vs-zero AUROC：{fmt(card_overall["mean_positive_vs_zero_auroc"])}。
- High-vs-weak准确率：{fmt(card_overall["mean_high_vs_weak_pair_accuracy"])}。

## 映射到标签后的描述性结果

{card_table}

逐标签均值按stable label–latent边计算，共217条可评分边；共享latent会在其关联的每个标签中出现。该表没有解释指标bootstrap CI，也不用于标签间显著性检验。

## 如何回答“证据有多可靠”

- Spearman/Pearson衡量解释预测连续响应排序或幅度的能力。
- AUROC衡量解释区分正响应与零响应句子的能力。
- High–weak准确率衡量解释能否区分强响应与弱正响应。
- 可靠性是连续量，不能仅凭一张card的措辞流畅度判断。

## 限制

4个latent不能生成完整Explainer packet，另有7个无法形成完整Scorer held-out分层；这些单元均显式排除，没有复制句子或放宽协议。
""",
    )

    write_text(
        docs / "07_SAE_PCA解释忠实度对照.md",
        f"""# SAE–PCA解释忠实度对照

## 范围

- 标签：REC、QUO、QUC、AF。
- 16个SAE、16个PCA-50、16个PCA-100匿名单元。
- 32个SAE–PCA配对。
- Explainer和Scorer均不知道表示类型与标签。

## Family结果

{task5_table}

## 配对结果

{markdown_table(["PCA维数", "配对数", "SAE-PCA Spearman", "SAE胜出", "SAE high–weak", "PCA high–weak"], [[row["pca_dimension"], row["n_pairs"], fmt(row["mean_sae_minus_pca_spearman"]), f'{row["sae_spearman_wins"]}/{row["n_pairs"]}', fmt(row["mean_sae_pair_accuracy"]), fmt(row["mean_pca_pair_accuracy"])] for row in paired_summary])}

## 结论

冻结样本内，SAE解释在连续响应排序、正响应识别和强弱区分上整体优于匹配PCA方向。尤其是Spearman差距在PCA-50和PCA-100两组中方向一致。

## 不能扩大解释的地方

- 这不是全量PCA component评估。
- 只覆盖四个叶标签。
- Probe AUC与解释忠实度是不同性质的证据：PCA probe更强不意味着PCA card更忠实。
- 当前结果支持“冻结抽样下SAE更易获得忠实解释”，不支持“SAE在任何任务上都优于PCA”。
""",
    )

    example_sections = ["# 代表性latent card", "", "> 每标签选择一个当前v3中Spearman较高且尽量不重复的可评分stable-core latent。解释和句子保留原始英文，未重新调用模型或人工润色。", ""]
    for row in selected_examples:
        explanation = row["explanation"]
        packet = row["packet"]
        example_sections.extend(
            [
                f'## {row["label"]} — {row["feature_id"]} / latent {row["latent_idx"]}',
                "",
                f'- Short name: `{explanation["short_name"]}`',
                f'- Type: `{explanation["explanation_type"]}`',
                f'- Spearman: `{fmt(f(row["spearman_rho"]))}`',
                f'- AUROC: `{fmt(f(row["positive_vs_zero_auroc"]))}`',
                f'- High–weak accuracy: `{fmt(f(row["high_vs_weak_pair_accuracy"]))}`',
                "",
                "**Primary explanation**",
                "",
                explanation["primary_explanation"],
                "",
                "**Surface/linguistic hypothesis**",
                "",
                explanation["surface_or_linguistic_hypothesis"],
                "",
                "**Behavioral/discourse hypothesis**",
                "",
                explanation["behavioral_or_discourse_hypothesis"],
                "",
                "**Strong examples**",
                "",
            ]
        )
        example_sections.extend(f'- `{sample["sample_id"]}` {sample["text"]}' for sample in packet["strong_samples"][:3])
        example_sections.extend(["", "**Weak-positive contrasts**", ""])
        example_sections.extend(f'- `{sample["sample_id"]}` {sample["text"]}' for sample in packet["weak_samples"][:2])
        example_sections.append("")
    write_text(docs / "08_代表性latent_card.md", "\n".join(example_sections))

    write_text(
        docs / "09_人工子概念编码计划.md",
        """# 人工子概念编码计划

> Status: PENDING。该阶段尚未完成，当前包不包含人工编码科学发现。

## 编码目标

将214张有效当前v3 card映射到冻结的MISC子概念词表，回答各标签主要由哪些心理学功能、语言代理、主题或伪影证据组成。

## 最小编码单位

- 编码对象：一个去重latent/card。
- 每张card只填写一个主子概念代码。
- 代码表预先带有上位类别，编码者不再填写第二套复杂分类。
- 多标签latent先编码一次，再连接到所有stable label–latent边。

## 建议流程

1. 用20–30张card试编码并修订词表。
2. 冻结词表版本、定义、包含和排除例。
3. 对214张card完成主代码。
4. 抽取至少20%由第二名编码者独立复核。
5. 记录一致率、分歧及裁决结果。
6. 分别统计214个去重latent与228条label–latent边，避免混淆分母。

## 计划输出

- `codebook_v1.md`
- `latent_subconcept_coding.csv`
- `coding_disagreement_log.csv`
- `subconcept_summary_by_label.csv`
- `shared_subconcept_summary.csv`
- `human_coding_report.md`

## 完成门禁

在人工编码与复核完成前，不得写“card已经揭示了最终MI心理学组织结构”；当前只能报告stable-core结构和AI解释忠实度。
""",
    )

    write_text(
        docs / "10_论文主张限制与禁止引用.md",
        """# 论文主张、限制与禁止引用

## 当前允许的主张

- MISC叶标签在Llama layer-19 SAE空间中对应可重复的多latent关联集合。
- 少量定向stable-core特征保留了接近Raw Hidden和Full SAE的线性可解码信息。
- 部分自然语言解释能够预测未见句子的latent响应。
- 在冻结抽样和统一匿名协议下，SAE解释忠实度高于匹配PCA方向。

## 当前禁止的主张

- 单个latent等价于一个MISC标签。
- 模型已经理解MI或心理咨询机制。
- Probe AUC高证明某种表示更可解释。
- 当前关联、probe或card结果构成因果机制证明。
- 当前32个SAE–PCA配对代表全部SAE和全部PCA。
- 未完成人工编码前声称已经获得最终心理学子概念结构。

## 禁止作为主结果引用的历史产物

- `contrastive_latent_faithfulness_v2_gpt55_low_full218`：Scorer字段锚定且公开ID顺序泄漏。
- `contrastive_latent_faithfulness_v2_gpt55_low_full218_scorer_deanchored_20260718`：仍有公开ID顺序泄漏。
- `task5_sae_pca_contrastive_faithfulness_gpt55_low_sampled`：旧Task5 held-out顺序泄漏。
- `论文相关文档/latent_card_B级人工审查包/Task4_行为表征结构`：225张旧card、303条旧关联并包含父标签，不符合当前214解释/228叶标签边口径。
- `archive/minimal_sufficient_subspace_20260719`：已退出研究主线。

## 核心限制

- 结果只来自Llama-3.1-8B第19层。
- 数据是咨询转录句子，词汇模板、主题和转录伪影可能构成标签代理。
- 主线只含7叶标签，不能直接分析RE/QU父标签stable-core。
- 解释评估不含shuffled/empty基线或解释指标bootstrap CI。
- card忠实度说明解释能预测响应，不证明解释对应真实因果计算。
""",
    )

    write_text(
        docs / "11_决策日志.md",
        f"""# 决策日志

## D-20260719-01：Minimal sufficient退出主线

- 状态：accepted。
- 决策：归档专用代码、测试和结果，当前主线不再包含该研究问题。
- 原因：避免与当前“稳定候选—表示验证—解释忠实度”证据链混杂。
- 结果：当前SSOT、README和活动源码入口不再引用该内容。
- 反转条件：只有在形成新的明确研究问题并经导师确认后，才能从归档重新立项。

## D-20260719-02：表示probe只更新Stable Core SAE

- 状态：accepted。
- 决策：复用Raw Hidden、Full SAE、Top-n SAE、PCA、Random SAE的3920条旧fold结果，只重算35条Stable Core SAE结果。
- 证据：数据、标签、fold、seed和全部配置一致；复用审计及逐行比较通过。
- 后果：避免无信息增益的重复计算，同时保留完整可追溯性。

## D-20260719-03：随机化held-out成为唯一正式解释评分

- 状态：accepted。
- 决策：旧顺序泄漏Scorer全部降级为superseded；只使用先随机排列、后赋公开ID的v3 packet。
- 证据：主card 207/207、Task5 48/48验证通过。
- 后果：论文只引用当前v3解释忠实度指标。

## D-20260719-04：人工组织结构结论暂缓

- 状态：accepted。
- 决策：旧225-card AI分组不能作为当前人工编码结果；待214张v3 card按冻结子概念词表重新编码。
- 后果：当前科学结论停在稳定结构、表示可解码性和解释忠实度层面。
""",
    )

    card_catalog_sources = [
        ("主实验Explainer packet", CARD_ROOT / "public_packets" / "explainer_packets.jsonl", len(packets)),
        ("主实验有效解释", CARD_ROOT / "explainer" / "validated_explanations.jsonl", len(explanations)),
        ("主实验held-out指标", CARD_ROOT / "scorer" / "faithfulness_metrics.csv", len(card_metrics)),
        ("Stable-core标签映射", STABLE_ROOT / "stable_topk_latent_set.csv", len(stable_all)),
        ("SAE-PCA Explainer packet", TASK5_ROOT / "public_packets" / "explainer_packets.jsonl", len(task5_packets)),
        ("SAE-PCA有效解释", TASK5_ROOT / "explainer" / "validated_explanations.jsonl", len(task5_explanations)),
        ("SAE-PCA held-out指标", TASK5_ROOT / "scorer" / "faithfulness_metrics.csv", len(task5_metric_rows)),
        ("SAE-PCA配对表", TASK5_ROOT / "analysis" / "paired_comparison.csv", len(task5_pairs)),
    ]
    write_text(
        docs / "12_主实验与SAE_PCA_Card可审计全集.md",
        build_auditable_card_catalog(
            main_explanations=explanations,
            main_packets=packets,
            main_metrics=metric_by_feature,
            stable_labels=latent_labels,
            task5_explanations=task5_explanations,
            task5_packets=task5_packets,
            task5_metrics=task5_metric_by_feature,
            task5_pairs=task5_pairs,
            source_paths=card_catalog_sources,
        ),
    )

    # Copy the compact, audit-relevant frozen artifacts.
    copy_specs = [
        (STABLE_ROOT / "stable_topk_latent_set.csv", "stable_core/stable_topk_latent_set.csv", "宽松stable-core完整成员表"),
        (STABLE_ROOT / "stable_topk_selection_report.md", "stable_core/stable_topk_selection_report.md", "stable-core选择报告"),
        (PROBE_ROOT / "probe_macro_summary.csv", "representation_probe/probe_macro_summary.csv", "表示probe macro结果"),
        (PROBE_ROOT / "probe_summary_by_label.csv", "representation_probe/probe_summary_by_label.csv", "表示probe逐标签结果"),
        (PROBE_ROOT / "reuse_and_refresh_audit.json", "representation_probe/reuse_and_refresh_audit.json", "非Stable复用与Stable刷新审计"),
        (PROBE_ROOT / "manifest.json", "representation_probe/manifest.json", "表示probe manifest"),
        (PROBE_ROOT / "representation_probe_comparison_report.md", "representation_probe/representation_probe_comparison_report.md", "表示probe报告"),
        (PROBE_ROOT / "figures" / "performance_curves_macro.png", "representation_probe/performance_curves_macro.png", "表示probe图"),
        (CARD_ROOT / "scorer" / "faithfulness_metrics.csv", "main_latent_cards/faithfulness_metrics.csv", "主card逐latent忠实度"),
        (CARD_ROOT / "explainer" / "validated_explanations.jsonl", "main_latent_cards/validated_explanations.jsonl", "214张冻结解释"),
        (CARD_ROOT / "final_completion_audit.json", "main_latent_cards/final_completion_audit.json", "主card完成审计"),
        (CARD_ROOT / "packet_integrity_audit.json", "main_latent_cards/packet_integrity_audit.json", "主card packet审计"),
        (TASK5_ROOT / "scorer" / "faithfulness_metrics.csv", "sae_pca/faithfulness_metrics.csv", "Task5逐单元忠实度"),
        (TASK5_ROOT / "analysis" / "family_summary.csv", "sae_pca/family_summary.csv", "Task5表示族汇总"),
        (TASK5_ROOT / "analysis" / "paired_comparison.csv", "sae_pca/paired_comparison.csv", "Task5 32个配对"),
        (TASK5_ROOT / "analysis" / "sae_pca_contrastive_faithfulness_report.md", "sae_pca/sae_pca_contrastive_faithfulness_report.md", "Task5正式报告"),
        (TASK5_ROOT / "pipeline_status.json", "sae_pca/pipeline_status.json", "Task5完成状态"),
        (TASK5_ROOT / "packet_integrity_audit.json", "sae_pca/packet_integrity_audit.json", "Task5 packet审计"),
    ]
    artifact_records: list[dict[str, Any]] = []
    for source, relative, role in copy_specs:
        destination = attachments / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = sha256(source)
        copied_hash = sha256(destination)
        if source_hash != copied_hash:
            raise ValueError(f"Copy hash mismatch: {source}")
        artifact_records.append(
            {
                "status": "FROZEN",
                "role": role,
                "source_path": rel(source),
                "package_path": destination.relative_to(output).as_posix(),
                "sha256": source_hash,
                "bytes": source.stat().st_size,
            }
        )

    write_csv(output / "冻结产物清单.csv", artifact_records)
    index_rows = [[row["status"], row["role"], f'`{row["package_path"]}`', row["sha256"][:12], row["bytes"]] for row in artifact_records]
    write_text(
        output / "00_产物索引.md",
        f"""# 冻结产物索引

本表列出复制进入同步包的正式结果。完整SHA-256见`冻结产物清单.csv`；复制文件与源文件hash逐项一致。

{markdown_table(["状态", "用途", "包内路径", "SHA-256前12位", "字节"], index_rows)}

## 派生结果表

- `结果表/data_label_summary.csv`
- `结果表/stable_core_summary_by_label.csv`
- `结果表/stable_core_sharing_distribution.csv`
- `结果表/stable_core_pair_overlap.csv`
- `结果表/representation_key_results.csv`
- `结果表/main_card_faithfulness_overall.csv`
- `结果表/main_card_faithfulness_by_label.csv`
- `结果表/representative_card_index.csv`
- `结果表/task5_paired_summary.csv`
- `结果表/task5_paired_summary_by_label.csv`

## 分阶段实验设置

- `文档/实验设置/阶段1_数据范围与Layer19特征.md`
- `文档/实验设置/阶段2_Filtered_Latent_Pool.md`
- `文档/实验设置/阶段3_Stable_Core筛选.md`
- `文档/实验设置/阶段4_表示信息验证.md`
- `文档/实验设置/阶段5_Latent_Card与Heldout评分.md`
- `文档/实验设置/阶段6_SAE_PCA解释忠实度对照.md`
- `文档/实验设置/阶段7_人工子概念编码_待执行.md`
- `文档/实验设置/结果阅读指南_导师版.md`

## Card人工审查

- `文档/12_主实验与SAE_PCA_Card可审计全集.md`
""",
    )

    package_files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "package_manifest.json":
            package_files.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    manifest = {
        "status": "complete",
        "package_version": "mentor-writing-sync-20260720-v4",
        "generated_on": date.today().isoformat(),
        "ssot": rel(SSOT),
        "source_roots": {
            "stable_core": rel(STABLE_ROOT),
            "representation_probe": rel(PROBE_ROOT),
            "main_latent_cards": rel(CARD_ROOT),
            "sae_pca": rel(TASK5_ROOT),
            "minimal_sufficient_archive": rel(ARCHIVE_ROOT),
        },
        "frozen_counts": {
            "samples": len(label_rows),
            "stable_edges": len(stable),
            "unique_stable_latents": len(latent_labels),
            "valid_explanations": len(explanations),
            "main_scorer_metrics": len(card_metrics),
            "task5_units": sum(int(row["n"]) for row in task5_family),
            "task5_pairs": len(task5_pairs),
        },
        "package_files": package_files,
    }
    (output / "package_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_dir": str(output), **manifest["frozen_counts"], "n_files": len(package_files) + 1}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
