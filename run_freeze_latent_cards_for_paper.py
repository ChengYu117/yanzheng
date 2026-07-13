"""Freeze DeepSeek latent cards and build the paper B-level human-review package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.nlp_re_base.contrastive_evidence_pack import read_jsonl, write_json, write_jsonl


SOURCE_BASE = Path(
    "outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/"
    "deepseek_v4_flash_latent_cards"
)
DEFAULT_OUTPUT = Path("论文相关文档/latent_card_B级人工审查包")
LEAF_LABELS = ("RES", "REC", "QUO", "QUC", "GI", "SU", "AF")


LEAF_SELECTIONS: dict[str, list[tuple[int, str]]] = {
    "RES": [
        (20808, "标签内稳定排名第1；明确的第二人称表层模式，作为避免行为过度解释的结构基线。"),
        (28269, "标签内稳定排名第2；so/okay 引导的改述模式，代表候选反映功能。"),
        (29759, "标签内稳定排名第4，且与 REC 共享；用于审查反映与提问的功能混合。"),
        (13966, "标签内稳定排名第6；you're + 状态描述结构，区别于纯 so-you 模板。"),
        (19435, "标签内稳定排名第8，且与 REC 共享；so you 表层结构清晰，行为功能需人工收窄。"),
    ],
    "REC": [
        (31133, "标签内稳定排名第1；it/that sounds like 模板与反映候选直接对应。"),
        (20436, "标签内稳定排名第2；on one hand/on the other hand 的矛盾心理改述，功能较具体。"),
        (30224, "标签内稳定排名第4；反映、改述与开放提问共现，适合审查复杂反映边界。"),
        (19435, "标签内稳定排名第5，且与 RES 共享；保留可观察的 so you 结构。"),
        (15068, "标签内稳定排名第8；it + copula/perception + evaluation 结构，提供句法多样性。"),
    ],
    "QUO": [
        (9959, "标签内稳定排名第1；围绕个人经历和原因的开放式提问。"),
        (23183, "标签内稳定排名第4；what-initial 结构明确，作为表层形式代表。"),
        (26485, "标签内稳定排名第5；how 引导的感受、方式和观点询问。"),
        (11660, "标签内稳定排名第7，且与 QUC 共享；wh + would 假设/偏好问句。"),
        (18310, "标签内稳定排名第8；围绕改变、障碍和计划的开放式功能候选。"),
    ],
    "QUC": [
        (21935, "标签内稳定排名第1；健康行为相关的 do/have/can 问句代表。"),
        (14014, "标签内稳定排名第3；did/have 引导的病史与行为事实询问。"),
        (12555, "标签内稳定排名第8；do you 是可直接核查的封闭问句结构。"),
        (20463, "标签内稳定排名第15；are you 结构补充系词/进行体问句类型。"),
        (2504, "标签内稳定排名第21；辅助动词前置是最明确的 yes/no 句法候选，因多样性纳入。"),
    ],
    "GI": [
        (16345, "标签内稳定排名第1；药物和健康信息主题稳定，且未强行指定单一行为功能。"),
        (26879, "标签内稳定排名第3；健康风险、益处和统计证据说明。"),
        (27515, "标签内稳定排名第5；治疗方案、药物调整和生活方式建议。"),
        (26236, "标签内稳定排名第6；胆固醇和血压管理的具体信息主题。"),
        (16515, "标签内稳定排名第8；剂量、指标、数量和量表等数值信息。"),
    ],
    "SU": [
        (24760, "标签内稳定排名第1；I understand 引导的理解与共情表达。"),
        (29825, "标签内稳定排名第3；理解、正常化与征求许可共现，适合审查支持功能边界。"),
        (16736, "标签内稳定排名第4；hard/difficult/tough 等困难承认模式。"),
        (9720, "标签内稳定排名第7；提供帮助和询问需求的 help 模板。"),
        (28795, "标签内稳定排名第15；道歉、遗憾和共情内容，提供情感类型多样性。"),
    ],
    "AF": [
        (23464, "标签内稳定排名第1；that's/that sounds + 正向形容词的肯定评价。"),
        (7143, "标签内稳定排名第2；对行动、动机和进展的肯定性改述。"),
        (17793, "标签内稳定排名第4；thank you for coming/sharing 的参与感谢模板。"),
        (30870, "标签内稳定排名第5；对目标、计划和想法的正向评价。"),
        (2434, "标签内稳定排名第10；内容抽样确认质量高的 thank you for + gerund 模板。"),
    ],
}


SHARED_SELECTIONS: list[tuple[int, str, str]] = [
    (19435, "REC|RES", "同家族共享；so you 结构清晰，但反映、总结和指令功能需要人工区分。"),
    (29759, "REC|RES", "同家族共享；反映与提问混合，适合审查简单/复杂反映边界。"),
    (11660, "QUC|QUO", "同家族共享；wh + would 可同时表现开放性和可回答边界。"),
    (12340, "QUC|QUO", "同家族共享；do you 句式说明问句表层结构不能自动区分开放/封闭功能。"),
    (26485, "QUC|QUO", "同家族共享；how 问句在两个子标签中共享，适合审查功能标注边界。"),
    (29590, "QUC|QUO", "同家族共享；how long/much/often/many 的数量问句具有开放形式和受限答案双重属性。"),
    (23464, "AF|SU", "跨家族共享；正向评价可能同时承担肯定和支持作用。"),
    (24856, "AF|SU", "跨家族共享；that + copula + evaluative adjective 是可核查的共同表层模板。"),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _as_text_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _card_markdown(
    *, label: str, card: dict[str, Any], pack: dict[str, Any], stable_row: pd.Series, rationale: str
) -> list[str]:
    text_by_id = {str(row["id"]): str(row["text"]) for row in pack["samples_for_model"]}
    representative_ids = _as_text_list(card.get("representative_evidence_ids"))
    extra_support = [
        sample_id
        for sample_id in _as_text_list(card.get("supporting_sample_ids"))
        if sample_id not in representative_ids
    ][:5]
    outliers = _as_text_list(card.get("outlier_sample_ids"))[:5]
    lines = [
        f"## {label} / latent {int(card['latent_idx'])}: {card['short_name']}",
        "",
        f"- B级状态：`pending_human_review`",
        f"- 标签内 stable rank：{int(stable_row['rank_within_label'])}",
        f"- inclusion frequency：{float(stable_row['inclusion_frequency']):.3f}",
        f"- |Cohen's d|：{float(stable_row['abs_cohens_d']):.3f}",
        f"- 自动解释类型：`{card['explanation_type']}`",
        f"- 自动置信度：{int(card['confidence'])}/5",
        f"- LLM 自报支持比例：{float(card['support_fraction']):.1%}",
        f"- 入选理由：{rationale}",
        "",
        "**主要候选解释**",
        "",
        str(card["primary_explanation"]),
        "",
        "**候选行为解释**",
        "",
        str(card["candidate_behavioral_explanation"]),
        "",
        "**代表证据**",
        "",
    ]
    lines.extend(f"- `{sample_id}`：{text_by_id.get(sample_id, '')}" for sample_id in representative_ids)
    lines.extend(["", "**其他 supporting 抽样**", ""])
    lines.extend(f"- `{sample_id}`：{text_by_id.get(sample_id, '')}" for sample_id in extra_support)
    lines.extend(["", "**Outlier 抽样**", ""])
    lines.extend(
        [f"- `{sample_id}`：{text_by_id.get(sample_id, '')}" for sample_id in outliers]
        or ["- 模型没有提供 outlier；人工审查必须检查是否存在被错误纳入 supporting 的反例。"]
    )
    lines.extend(["", "**替代解释**", ""])
    lines.extend(f"- {item}" for item in _as_text_list(card.get("alternative_explanations")))
    lines.extend(["", "**混淆与限制**", ""])
    lines.extend(f"- {item}" for item in _as_text_list(card.get("possible_confounds")))
    lines.extend(f"- {item}" for item in _as_text_list(card.get("limitations")))
    lines.extend(
        [
            "",
            "**人工审查填写**",
            "",
            "- [ ] 主要模式得到支持",
            "- [ ] 行为功能与表层结构已分开",
            "- [ ] supporting/outlier 分区合理",
            "- [ ] 候选名称需要修改",
            "- reviewer 1：",
            "- reviewer 2：",
            "- 最终名称：",
            "- 审查意见：",
            "",
        ]
    )
    return lines


def freeze_package(*, source_base: Path, stable_path: Path, output_dir: Path) -> dict[str, Any]:
    source_files = {
        "validated_cards": source_base / "card_outputs" / "validated_cards.jsonl",
        "sentence_packs": source_base / "evidence_packs" / "latent_card_sentence_packs.jsonl",
        "quality_by_label": source_base / "quality_report" / "latent_card_quality_by_label.csv",
        "quality_by_latent": source_base / "quality_report" / "latent_card_quality_by_latent.csv",
        "content_audit_md": source_base / "quality_report" / "card_content_sample_audit.md",
        "content_audit_csv": source_base / "quality_report" / "card_content_sample_audit.csv",
    }
    missing = [str(path) for path in source_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source artifacts: {missing}")

    cards = {int(row["latent_idx"]): row for row in read_jsonl(source_files["validated_cards"])}
    packs = {int(row["latent_idx"]): row for row in read_jsonl(source_files["sentence_packs"])}
    stable = pd.read_csv(stable_path)
    stable = stable[stable["stable_set_role"].astype(str).eq("stable_core")].copy()
    stable["label"] = stable["label"].astype(str).str.upper()
    stable["latent_idx"] = pd.to_numeric(stable["latent_idx"], errors="raise").astype(int)
    stable_index = stable.set_index(["label", "latent_idx"], drop=False)

    frozen_dir = output_dir / "frozen_all_cards"
    destinations: dict[str, Path] = {}
    for name, source in source_files.items():
        suffix = source.suffix
        destination = frozen_dir / f"{name}{suffix}"
        _copy(source, destination)
        destinations[name] = destination
    _copy(stable_path, frozen_dir / "stable_topk_latent_set.csv")
    destinations["stable_latents"] = frozen_dir / "stable_topk_latent_set.csv"

    review_dir = output_dir / "B_level_human_review"
    cards_dir = review_dir / "cards_by_leaf_label"
    cards_dir.mkdir(parents=True, exist_ok=True)
    leaf_rows: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for label in LEAF_LABELS:
        lines = [f"# {label} B级 Latent Card 人工审查", ""]
        for order, (latent_idx, rationale) in enumerate(LEAF_SELECTIONS[label], 1):
            if latent_idx not in cards or latent_idx not in packs:
                raise KeyError(f"Missing selected latent {latent_idx}")
            key = (label, latent_idx)
            if key not in stable_index.index:
                raise KeyError(f"Selected latent {latent_idx} is not stable_core for {label}")
            stable_row = stable_index.loc[key]
            if isinstance(stable_row, pd.DataFrame):
                stable_row = stable_row.iloc[0]
            card = cards[latent_idx]
            selected_ids.add(latent_idx)
            leaf_rows.append(
                {
                    "grade": "B",
                    "review_status": "pending_human_review",
                    "label": label,
                    "selection_order": order,
                    "latent_idx": latent_idx,
                    "rank_within_label": int(stable_row["rank_within_label"]),
                    "inclusion_frequency": float(stable_row["inclusion_frequency"]),
                    "abs_cohens_d": float(stable_row["abs_cohens_d"]),
                    "short_name": card["short_name"],
                    "explanation_type": card["explanation_type"],
                    "confidence": int(card["confidence"]),
                    "support_fraction_model_reported": float(card["support_fraction"]),
                    "selection_rationale": rationale,
                    "human_final_name": "",
                    "reviewer_1": "",
                    "reviewer_2": "",
                    "human_review_notes": "",
                }
            )
            lines.extend(
                _card_markdown(
                    label=label,
                    card=card,
                    pack=packs[latent_idx],
                    stable_row=stable_row,
                    rationale=rationale,
                )
            )
        (cards_dir / f"{label}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    shared_rows: list[dict[str, Any]] = []
    shared_lines = ["# 跨叶级标签共享 Latent Card", ""]
    for order, (latent_idx, labels_text, rationale) in enumerate(SHARED_SELECTIONS, 1):
        labels = labels_text.split("|")
        card = cards[latent_idx]
        selected_ids.add(latent_idx)
        ranks: list[str] = []
        for label in labels:
            stable_row = stable_index.loc[(label, latent_idx)]
            if isinstance(stable_row, pd.DataFrame):
                stable_row = stable_row.iloc[0]
            ranks.append(f"{label}:{int(stable_row['rank_within_label'])}")
        sharing_type = "cross_family" if set(labels) == {"AF", "SU"} else "sibling_leaf_labels"
        shared_rows.append(
            {
                "grade": "B",
                "review_status": "pending_human_review",
                "selection_order": order,
                "latent_idx": latent_idx,
                "leaf_labels": labels_text,
                "sharing_type": sharing_type,
                "rank_within_labels": "|".join(ranks),
                "short_name": card["short_name"],
                "explanation_type": card["explanation_type"],
                "confidence": int(card["confidence"]),
                "support_fraction_model_reported": float(card["support_fraction"]),
                "selection_rationale": rationale,
                "human_shared_interpretation": "",
                "reviewer_1": "",
                "reviewer_2": "",
                "human_review_notes": "",
            }
        )
        shared_lines.extend(
            [
                f"## {labels_text} / latent {latent_idx}: {card['short_name']}",
                "",
                f"- 共享类型：`{sharing_type}`",
                f"- 标签内排名：`{' | '.join(ranks)}`",
                f"- 自动解释类型：`{card['explanation_type']}`",
                f"- 自动置信度：{int(card['confidence'])}/5",
                f"- 入选理由：{rationale}",
                "",
                str(card["primary_explanation"]),
                "",
                "**代表证据**",
                "",
            ]
        )
        text_by_id = {str(row["id"]): str(row["text"]) for row in packs[latent_idx]["samples_for_model"]}
        shared_lines.extend(
            f"- `{sample_id}`：{text_by_id.get(str(sample_id), '')}"
            for sample_id in card["representative_evidence_ids"]
        )
        shared_lines.extend(
            [
                "",
                "- [ ] 共享来自真实功能而非表层结构",
                "- [ ] 同家族共享与跨家族共享已区分",
                "- 人工共享解释：",
                "- 审查意见：",
                "",
            ]
        )

    leaf_path = review_dir / "leaf_label_B_candidates.csv"
    shared_path = review_dir / "shared_B_candidates.csv"
    pd.DataFrame(leaf_rows).to_csv(leaf_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(shared_rows).to_csv(shared_path, index=False, encoding="utf-8-sig")
    (review_dir / "shared_features.md").write_text("\n".join(shared_lines) + "\n", encoding="utf-8")
    write_jsonl(review_dir / "selected_cards.jsonl", [cards[idx] for idx in sorted(selected_ids)])
    write_jsonl(review_dir / "selected_sentence_packs.jsonl", [packs[idx] for idx in sorted(selected_ids)])

    readme = f"""# Latent Card B级人工审查包

冻结日期：{date.today().isoformat()}

## B级定义

B级表示：来自 stable-core、通过自动结构校验，并经过候选内容筛选，适合进入论文人工审查；它不表示已经完成人工确认，也不表示 latent 已被证明具有唯一行为功能。

## 选择范围

- 叶级标签：`RES, REC, QUO, QUC, GI, SU, AF`。
- 每个叶级标签固定 5 个候选，共 35 条 label-latent 关联。
- 共享候选 8 个：RES-REC 2 个、QUO-QUC 4 个、AF-SU 2 个。
- 父标签 `RE` 和 `QU` 不进入叶级候选主表。

## 选择原则

1. 优先标签内 stable rank 靠前的 latent。
2. 排除 `unclear_or_mixed`、明显低覆盖和内容抽样中不可靠的 card。
3. 同一标签内保留行为功能、句法结构、情感内容和主题等不同模式，避免五张卡片表达同一个模板。
4. 对 RES/REC 和 QUO/QUC，优先保留可观察的表层模式，并将治疗/MI 功能写成待审候选。
5. `support_fraction_model_reported` 是 LLM 自报分区，不是独立准确率。

## 人工审查顺序

1. 打开 `B_level_human_review/cards_by_leaf_label/<LABEL>.md`，逐卡核对代表证据、其他 supporting 和 outlier。
2. 在 `leaf_label_B_candidates.csv` 填写最终名称、审查者和意见。
3. 审查 `shared_features.md`，先判断共享是否仅来自问句、第二人称或评价模板。
4. 两名审查者达成一致后，才能把 `review_status` 更新为人工确认状态。

## 冻结说明

`frozen_all_cards` 保存全部 225 张 card、全部句子包、质量统计和内容抽样报告。`package_checksums.sha256` 用于确认论文审查期间输入没有变化。
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    checksum_path = output_dir / "package_checksums.sha256"
    manifest = {
        "analysis": "paper_frozen_latent_card_B_review_package",
        "freeze_date": date.today().isoformat(),
        "source_base": str(source_base),
        "leaf_labels": list(LEAF_LABELS),
        "grade_definition": "automatically generated stable-core candidate selected for human review; not human validated",
        "counts": {
            "all_frozen_cards": len(cards),
            "leaf_label_candidate_rows": len(leaf_rows),
            "leaf_candidates_per_label": pd.DataFrame(leaf_rows).groupby("label").size().to_dict(),
            "shared_candidates": len(shared_rows),
            "unique_selected_latents": len(selected_ids),
        },
        "outputs": {
            "readme": str(output_dir / "README.md"),
            "leaf_candidates": str(leaf_path),
            "shared_candidates": str(shared_path),
            "checksums": str(checksum_path),
        },
        "source_sha256": {name: _sha256(path) for name, path in source_files.items()},
    }
    write_json(output_dir / "manifest.json", manifest)
    output_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    checksum_lines = [
        f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}"
        for path in output_files
        if path != checksum_path
    ]
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="ascii")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-base", type=Path, default=SOURCE_BASE)
    parser.add_argument(
        "--stable-latents",
        type=Path,
        default=Path("outputs/cross_val/stable_topk_selection/stable_topk_latent_set.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = freeze_package(
        source_base=args.source_base,
        stable_path=args.stable_latents,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
