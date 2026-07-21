"""Freeze the current latent-card release and render selected example packets."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPLAINER_ROOT = ROOT / "outputs/rerun_new_dataset_20260716/min5_words/interpretability/contrastive_latent_faithfulness_v2_gpt55_low_full218"
SCORER_ROOT = ROOT / "outputs/rerun_new_dataset_20260716/min5_words/interpretability/contrastive_latent_faithfulness_v2_reduced_context_scorer_full207"
STABLE_PATH = ROOT / "outputs/rerun_new_dataset_20260716/min5_words/cross_val/stable_topk_selection_n20_relaxed_leaf7/stable_topk_latent_set.csv"
OUTPUT_ROOT = ROOT / "论文相关文档/附件/latent_card_reduced_context_gpt55_20260718"

SCORER_VISIBLE_FIELDS = (
    "short_name",
    "surface_or_linguistic_hypothesis",
    "behavioral_or_discourse_hypothesis",
    "primary_explanation",
    "explanation_type",
)

SELECTIONS = {
    "F060": {
        "tier": "paper_ready_primary",
        "title_zh": "直接 what 问句",
        "why_zh": "最窄的直接 what 主句问法；四项 held-out 指标均接近或达到 1，适合作为语言形式证据的主例。",
        "caution_zh": "它首先是问句形式特征；不能仅凭该卡断言 latent 独立编码开放式提问功能。",
    },
    "F113": {
        "tier": "paper_ready_primary",
        "title_zh": "数字量表评分引导",
        "why_zh": "形式条件非常具体，同时对应量化自评这一临床/咨询评估动作，展示语言模板与交际功能可以共同构成解释。",
        "caution_zh": "其强可预测性可能主要来自数字与量表模板。",
    },
    "F058": {
        "tier": "paper_ready_primary",
        "title_zh": "以客户为中心的信息引导问题",
        "why_zh": "同时保留 what/how 等表面线索与邀请对方提供经历、判断或计划的行为功能，且 held-out 区分稳定。",
        "caution_zh": "QUO 与 QUC 共享，适合说明跨标签共享证据，而非标签特异 latent。",
    },
    "F150": {
        "tier": "paper_ready_primary",
        "title_zh": "患者特异的临床解释或指令",
        "why_zh": "较清楚地区分信息给予/医疗指令与提问、反映和协作规划，属于较强行为功能例。",
        "caution_zh": "具体医疗实体和第二人称结构可能同时充当语言代理。",
    },
    "F165": {
        "tier": "paper_ready_primary",
        "title_zh": "明确认可或许可框架",
        "why_zh": "明确的 acceptable/understandable/okay 类形式与验证、征求许可功能相互对应，四项指标稳定。",
        "caution_zh": "解释仍以显式词汇框架为主，不能扩大为一般性的同理心机制。",
    },
    "F176": {
        "tier": "paper_ready_primary",
        "title_zh": "明确的亲和性承接",
        "why_zh": "感谢、同情、认可和积极回应共同指向关系建立功能，是当前最强的行为/话语功能卡之一。",
        "caution_zh": "表面公式较异质，建议论文中展示具体句子而非只给抽象标签。",
    },
    "F015": {
        "tier": "strong_caveated",
        "title_zh": "sounds/seems like 反映模板",
        "why_zh": "held-out 指标极强，清楚展示反映性话语常由稳定语言模板承载。",
        "caution_zh": "records 缺少前一轮 client 话语，因此只能可靠确认模板，不能确认反映内容是否准确。",
    },
    "F047": {
        "tier": "strong_caveated",
        "title_zh": "对负面自我认知的第二人称反映",
        "why_zh": "相比纯模板卡更接近行为功能，并在未见句子上保持较强排序和区分表现。",
        "caution_zh": "功能解释依赖对负面内部状态的语义归纳，且缺少前文，可靠性低于纯语言结构卡。",
    },
    "F014": {
        "tier": "strong_caveated",
        "title_zh": "you are/you're 陈述式刻画",
        "why_zh": "四项指标较高，适合展示 RES 候选中稳定、可预测的第二人称陈述结构。",
        "caution_zh": "主要支持语言代理而不是反映功能；无 client 前文时不能据此认定为简单反映。",
    },
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def metric_text(value: object) -> str:
    return f"{float(value):.3f}"


def load_metrics(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["feature_id"]: row for row in csv.DictReader(handle)}


def load_labels(path: Path) -> dict[int, list[str]]:
    result: dict[int, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["stable_set_role"] == "stable_core":
                result[int(row["latent_idx"])].add(row["label"])
    return {latent: sorted(labels) for latent, labels in result.items()}


def render_card(packet: dict) -> str:
    card = packet["full_explainer_card"]
    metrics = packet["faithfulness_metrics"]
    lines = [
        f"# {packet['feature_id']} / latent {packet['latent_idx']}：{packet['selection']['title_zh']}",
        "",
        f"> Selection tier: `{packet['selection']['tier']}`  ",
        f"> Stable-core labels: `{', '.join(packet['stable_core_labels'])}`  ",
        f"> Explanation type: `{card['explanation_type']}`",
        "",
        "## 中文审核结论",
        "",
        f"- 入选理由：{packet['selection']['why_zh']}",
        f"- 解释边界：{packet['selection']['caution_zh']}",
        "- 证据等级：held-out 相关性/排序忠实度证据；不是因果或标签等价证明。",
        "",
        "## 冻结解释（英文原文）",
        "",
        f"- **Short name:** {card['short_name']}",
        f"- **Surface/linguistic hypothesis:** {card['surface_or_linguistic_hypothesis']}",
        f"- **Behavioral/discourse hypothesis:** {card['behavioral_or_discourse_hypothesis']}",
        f"- **Primary explanation:** {card['primary_explanation']}",
        f"- **Explanation type:** `{card['explanation_type']}`",
        "",
        "## Held-out 忠实度",
        "",
        "| Spearman | Pearson(log activation) | Positive-vs-zero AUROC | High-vs-weak accuracy |",
        "|---:|---:|---:|---:|",
        f"| {metric_text(metrics['spearman_rho'])} | {metric_text(metrics['pearson_log_activation'])} | {metric_text(metrics['positive_vs_zero_auroc'])} | {metric_text(metrics['high_vs_weak_pair_accuracy'])} |",
        "",
        "## Discovery：强响应句",
        "",
        "| ID | Sentence |",
        "|---|---|",
    ]
    lines.extend(f"| {sample['sample_id']} | {md_cell(sample['text'])} |" for sample in packet["strong_samples"])
    lines.extend(["", "## Discovery：弱正响应句（hard negatives）", "", "| ID | Sentence |", "|---|---|"])
    lines.extend(f"| {sample['sample_id']} | {md_cell(sample['text'])} |" for sample in packet["weak_samples"])
    lines.extend(
        [
            "",
            "## Held-out 预测",
            "",
            "| ID | Stratum | True activation | Predicted score | Matching evidence | Sentence |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for row in packet["heldout_predictions"]:
        lines.append(
            f"| {row['sample_id']} | {row['stratum']} | {float(row['true_activation']):.6g} | "
            f"{int(row['predicted_feature_score'])} | {md_cell(row.get('matching_evidence_span', ''))} | {md_cell(row['text'])} |"
        )
    lines.extend(
        [
            "",
            "## 方法提醒",
            "",
            "该卡只说明冻结解释能够在未见句子上预测相对响应。stable-core 标签来自相关性筛选，不能写成该 latent 等价于该标签。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen attachment: {OUTPUT_ROOT}")

    frozen = OUTPUT_ROOT / "frozen_release"
    examples = OUTPUT_ROOT / "example_packets"
    cards_dir = examples / "cards"
    method_dir = frozen / "method_snapshot"
    for directory in (frozen, examples, cards_dir, method_dir):
        directory.mkdir(parents=True, exist_ok=False)

    copy_map = {
        EXPLAINER_ROOT / "explainer/validated_explanations.jsonl": frozen / "validated_explanations.jsonl",
        EXPLAINER_ROOT / "explainer/validation_manifest.json": frozen / "explainer_validation_manifest.json",
        SCORER_ROOT / "private_packets.jsonl": frozen / "private_packets.jsonl",
        SCORER_ROOT / "scorer/faithfulness_metrics.csv": frozen / "faithfulness_metrics.csv",
        SCORER_ROOT / "scorer/validated_predictions_private.jsonl": frozen / "validated_predictions_private.jsonl",
        SCORER_ROOT / "scorer/task_manifest.json": frozen / "scorer_task_manifest.json",
        SCORER_ROOT / "scorer/validation_failures.jsonl": frozen / "scorer_validation_failures.jsonl",
        SCORER_ROOT / "pipeline_status.json": frozen / "pipeline_status.json",
        SCORER_ROOT / "full207_reduced_context_scorer_report.md": frozen / "full207_reduced_context_scorer_report.md",
        STABLE_PATH: frozen / "stable_topk_latent_set.csv",
        ROOT / "config/contrastive_explainer_v2_base_instructions.txt": method_dir / "contrastive_explainer_v2_base_instructions.txt",
        ROOT / "config/contrastive_explainer_v2_schema.json": method_dir / "contrastive_explainer_v2_schema.json",
        ROOT / "config/contrastive_scorer_v2_base_instructions.txt": method_dir / "contrastive_scorer_v2_base_instructions.txt",
        ROOT / "config/contrastive_scorer_v2_schema.json": method_dir / "contrastive_scorer_v2_schema.json",
        ROOT / "src/nlp_re_base/contrastive_faithfulness_v2.py": method_dir / "contrastive_faithfulness_v2.py",
        ROOT / "run_contrastive_faithfulness_v2.py": method_dir / "run_contrastive_faithfulness_v2.py",
        ROOT / "doc/强弱对比解释与Heldout忠实度评估_当前规范.md": method_dir / "强弱对比解释与Heldout忠实度评估_当前规范.md",
    }
    for source, destination in copy_map.items():
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)

    explanations = {row["feature_id"]: row for row in read_jsonl(EXPLAINER_ROOT / "explainer/validated_explanations.jsonl")}
    discovery = {row["feature_id"]: row for row in read_jsonl(EXPLAINER_ROOT / "private_packets.jsonl")}
    metrics = load_metrics(SCORER_ROOT / "scorer/faithfulness_metrics.csv")
    labels = load_labels(STABLE_PATH)
    predictions_by_feature: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(SCORER_ROOT / "scorer/validated_predictions_private.jsonl"):
        predictions_by_feature[row["feature_id"]].append(row)

    selected_packets = []
    for feature_id, selection in SELECTIONS.items():
        card = explanations[feature_id]
        pack = discovery[feature_id]
        heldout_text = {row["sample_id"]: row["text"] for row in pack["heldout_samples_private"]}
        heldout_predictions = []
        for prediction in predictions_by_feature[feature_id]:
            heldout_predictions.append({**prediction, "text": heldout_text[prediction["sample_id"]]})
        selected = {
            "feature_id": feature_id,
            "latent_idx": int(card["latent_idx"]),
            "stable_core_labels": labels.get(int(card["latent_idx"]), []),
            "selection": selection,
            "scorer_visible_explanation": {field: card[field] for field in SCORER_VISIBLE_FIELDS},
            "full_explainer_card": card,
            "faithfulness_metrics": metrics[feature_id],
            "strong_samples": pack["strong_samples"],
            "weak_samples": pack["weak_samples"],
            "heldout_predictions": heldout_predictions,
        }
        selected_packets.append(selected)
        (cards_dir / f"{feature_id}_latent_{card['latent_idx']}.md").write_text(render_card(selected), encoding="utf-8")

    write_jsonl(examples / "selected_example_packets.jsonl", selected_packets)
    with (examples / "selected_example_index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_id", "latent_idx", "stable_core_labels", "tier", "title_zh", "explanation_type", "spearman", "pearson", "auroc", "high_vs_weak", "why_zh", "caution_zh"])
        for packet in selected_packets:
            card = packet["full_explainer_card"]
            metric = packet["faithfulness_metrics"]
            selection = packet["selection"]
            writer.writerow([
                packet["feature_id"], packet["latent_idx"], "+".join(packet["stable_core_labels"]), selection["tier"],
                selection["title_zh"], card["explanation_type"], metric["spearman_rho"], metric["pearson_log_activation"],
                metric["positive_vs_zero_auroc"], metric["high_vs_weak_pair_accuracy"], selection["why_zh"], selection["caution_zh"],
            ])

    primary = [packet for packet in selected_packets if packet["selection"]["tier"] == "paper_ready_primary"]
    caveated = [packet for packet in selected_packets if packet["selection"]["tier"] == "strong_caveated"]
    overview = [
        "# Latent card 论文示例包",
        "",
        "> Status: Frozen attachment / Pending final author selection",
        ">",
        "> 解释与指标均逐字复制自 2026-07-18 的 GPT-5.5 reduced-context 正式产物；中文入选理由是人工审核注释，不属于模型输出。",
        "",
        "## 推荐优先用于论文的 6 张主例",
        "",
        "| Feature | Latent | Stable-core labels | 类型 | 中文概括 | Spearman | AUROC | High–weak |",
        "|---|---:|---|---|---|---:|---:|---:|",
    ]
    for packet in primary:
        metric = packet["faithfulness_metrics"]
        overview.append(
            f"| [{packet['feature_id']}](cards/{packet['feature_id']}_latent_{packet['latent_idx']}.md) | {packet['latent_idx']} | "
            f"{'+'.join(packet['stable_core_labels'])} | {packet['full_explainer_card']['explanation_type']} | {packet['selection']['title_zh']} | "
            f"{metric_text(metric['spearman_rho'])} | {metric_text(metric['positive_vs_zero_auroc'])} | {metric_text(metric['high_vs_weak_pair_accuracy'])} |"
        )
    overview.extend(
        [
            "",
            "## 3 张强但需谨慎解释的对照例",
            "",
            "| Feature | Latent | Stable-core labels | 类型 | 中文概括 | Spearman | AUROC | 主要风险 |",
            "|---|---:|---|---|---|---:|---:|---|",
        ]
    )
    for packet in caveated:
        metric = packet["faithfulness_metrics"]
        overview.append(
            f"| [{packet['feature_id']}](cards/{packet['feature_id']}_latent_{packet['latent_idx']}.md) | {packet['latent_idx']} | "
            f"{'+'.join(packet['stable_core_labels'])} | {packet['full_explainer_card']['explanation_type']} | {packet['selection']['title_zh']} | "
            f"{metric_text(metric['spearman_rho'])} | {metric_text(metric['positive_vs_zero_auroc'])} | {packet['selection']['caution_zh']} |"
        )
    overview.extend(
        [
            "",
            "## 选择原则",
            "",
            "1. Scorer 结构验证通过，且证据片段逐字有效；",
            "2. held-out Spearman、AUROC 与 high–weak 区分整体较高；",
            "3. 解释条件足够窄，能够指出句子中可审核的证据；",
            "4. 主例覆盖语言结构与行为功能，并覆盖 QUO、QUC、GI、SU、AF；",
            "5. REC/RES 因缺少前一轮 client 话语，单列为带边界的强例，不把形式代理升级为功能或因果结论。",
            "",
            "## 使用建议",
            "",
            "- 图或正文优先使用 F060、F113、F058、F150、F165、F176；",
            "- F015、F047、F014 更适合方法讨论、误差边界或语言代理分析；",
            "- 每张卡的完整 discovery 强/弱句与 held-out 预测见 `cards/`；机器可读版本见 `selected_example_packets.jsonl`。",
            "",
        ]
    )
    (examples / "README_示例卡总览.md").write_text("\n".join(overview), encoding="utf-8")

    manifest = {
        "status": "frozen",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_explainer_root": str(EXPLAINER_ROOT.relative_to(ROOT)),
        "source_scorer_root": str(SCORER_ROOT.relative_to(ROOT)),
        "model": "gpt-5.5",
        "reasoning_effort": "low",
        "scorer_concurrency": 20,
        "n_valid_explanations": len(explanations),
        "n_scorer_cards": len(metrics),
        "n_selected_examples": len(selected_packets),
        "selected_feature_ids": list(SELECTIONS),
        "paper_ready_primary": [packet["feature_id"] for packet in primary],
        "strong_caveated": [packet["feature_id"] for packet in caveated],
        "frozen_files": {str(destination.relative_to(OUTPUT_ROOT)): sha256(destination) for destination in copy_map.values()},
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    readme = """# Latent card 冻结附件

本目录是 2026-07-18 GPT-5.5 reduced-context 两阶段流程的论文附件快照。

- `frozen_release/`：完整冻结解释、207 张卡的最终指标、held-out 预测、稳定集合映射及方法快照；
- `example_packets/`：6 张论文主例与 3 张强但需谨慎解释的对照例；
- `manifest.json`：来源、规模、模型设置及冻结文件哈希；
- `checksums.sha256`：本附件内全部文件的完整性校验。

冻结规则：不要直接修改本目录中的原始卡、指标或预测。如需增加人工注释，请新建 review 文件并保留本快照不变。
"""
    (OUTPUT_ROOT / "README.md").write_text(readme, encoding="utf-8")

    checksum_rows = []
    for path in sorted(p for p in OUTPUT_ROOT.rglob("*") if p.is_file() and p.name != "checksums.sha256"):
        checksum_rows.append(f"{sha256(path)}  {path.relative_to(OUTPUT_ROOT).as_posix()}")
    (OUTPUT_ROOT / "checksums.sha256").write_text("\n".join(checksum_rows) + "\n", encoding="utf-8")

    print(json.dumps({
        "output_root": str(OUTPUT_ROOT),
        "n_frozen_files": len(copy_map),
        "n_selected_examples": len(selected_packets),
        "paper_ready_primary": [packet["feature_id"] for packet in primary],
        "strong_caveated": [packet["feature_id"] for packet in caveated],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
