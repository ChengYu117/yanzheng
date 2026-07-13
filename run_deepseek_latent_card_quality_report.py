"""Audit DeepSeek latent-card sentence sets and summarize them by stable-core label."""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.nlp_re_base.contrastive_evidence_pack import normalise_text, read_jsonl, write_json


DEFAULT_BASE = Path(
    "outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/"
    "deepseek_v4_flash_latent_cards"
)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")
TRAILING_FRAGMENT_RE = re.compile(
    r"\b(?:and|or|but|because|if|that|the|a|an|to|of|for|with|from|your|my|our|this|these|those)\s*[?.!,]*$",
    re.IGNORECASE,
)
REPEATED_TOKEN_RE = re.compile(r"\b([A-Za-z]+)(?:\s+\1){2,}\b", re.IGNORECASE)
BROKEN_PUNCT_RE = re.compile(r"(?:\.{2,}|_{2,}|-{2,}|\s-[\s?.!,]*$)")


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def _sample_flags(text: str) -> dict[str, Any]:
    words = _words(text)
    stripped = text.strip()
    very_short = len(words) <= 3
    fragment_like = bool(
        TRAILING_FRAGMENT_RE.search(stripped)
        or stripped.endswith(("-", "…"))
        or BROKEN_PUNCT_RE.search(stripped)
    )
    transcription_noise = bool(
        REPEATED_TOKEN_RE.search(stripped)
        or BROKEN_PUNCT_RE.search(stripped)
        or "�" in stripped
        or re.search(r"\b\d*\s*%\s+to\s+%\b", stripped)
    )
    return {
        "word_count": len(words),
        "very_short": very_short,
        "fragment_like": fragment_like,
        "transcription_noise": transcription_noise,
    }


def _near_duplicate_ids(samples: list[dict[str, Any]], threshold: float = 0.92) -> set[str]:
    normalized = [(str(row["id"]), normalise_text(row["text"])) for row in samples]
    flagged: set[str] = set()
    for left in range(len(normalized)):
        left_id, left_text = normalized[left]
        for right in range(left + 1, len(normalized)):
            right_id, right_text = normalized[right]
            if abs(len(left_text) - len(right_text)) > max(len(left_text), len(right_text)) * 0.20:
                continue
            if SequenceMatcher(None, left_text, right_text, autojunk=False).ratio() >= threshold:
                flagged.update((left_id, right_id))
    return flagged


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _rate(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0).mean())


def build_report(
    *, cards_path: Path, packs_path: Path, stable_path: Path, output_dir: Path
) -> dict[str, Any]:
    cards = {int(row["latent_idx"]): row for row in read_jsonl(cards_path)}
    packs = {int(row["latent_idx"]): row for row in read_jsonl(packs_path)}
    stable = pd.read_csv(stable_path)
    stable = stable[stable["stable_set_role"].astype(str).eq("stable_core")].copy()
    stable["label"] = stable["label"].astype(str).str.upper()
    stable["latent_idx"] = pd.to_numeric(stable["latent_idx"], errors="raise").astype(int)
    if set(cards) != set(packs):
        raise ValueError("Card and sentence-pack latent sets do not match")
    missing_cards = sorted(set(stable["latent_idx"]) - set(cards))
    if missing_cards:
        raise ValueError(f"Stable-core latents missing cards: {missing_cards[:10]}")

    latent_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for latent_idx in sorted(cards):
        card = cards[latent_idx]
        samples = list(packs[latent_idx]["samples_for_model"])
        if len(samples) != 50:
            raise ValueError(f"latent {latent_idx} does not have 50 model-visible samples")
        near_duplicate_ids = _near_duplicate_ids(samples)
        flags_by_id: dict[str, dict[str, Any]] = {}
        for sample in samples:
            sample_id = str(sample["id"])
            flags = _sample_flags(str(sample["text"]))
            flags_by_id[sample_id] = flags
            sample_rows.append(
                {
                    "latent_idx": latent_idx,
                    "sample_id": sample_id,
                    "text": str(sample["text"]),
                    **flags,
                    "near_duplicate": sample_id in near_duplicate_ids,
                    "card_supporting": sample_id in set(card["supporting_sample_ids"]),
                    "card_outlier": sample_id in set(card["outlier_sample_ids"]),
                }
            )
        support_fraction = float(card.get("support_fraction", len(card["supporting_sample_ids"]) / 50))
        clear_explanation = str(card["explanation_type"]) != "unclear_or_mixed"
        latent_rows.append(
            {
                "latent_idx": latent_idx,
                "short_name": str(card["short_name"]),
                "explanation_type": str(card["explanation_type"]),
                "confidence": int(card["confidence"]),
                "support_fraction": support_fraction,
                "clear_explanation": clear_explanation,
                "clear_pattern_support_fraction": support_fraction if clear_explanation else np.nan,
                "n_supporting": len(card["supporting_sample_ids"]),
                "n_outliers": len(card["outlier_sample_ids"]),
                "n_linguistic_evidence": len(card["linguistic_evidence"]),
                "n_alternatives": len(card["alternative_explanations"]),
                "n_confounds": len(card["possible_confounds"]),
                "mean_word_count": float(np.mean([row["word_count"] for row in flags_by_id.values()])),
                "very_short_rate": float(np.mean([row["very_short"] for row in flags_by_id.values()])),
                "fragment_like_rate": float(np.mean([row["fragment_like"] for row in flags_by_id.values()])),
                "transcription_noise_rate": float(
                    np.mean([row["transcription_noise"] for row in flags_by_id.values()])
                ),
                "near_duplicate_rate": len(near_duplicate_ids) / 50,
                "low_support_flag": clear_explanation and support_fraction < 0.75,
                "high_confidence_low_support_flag": (
                    clear_explanation and int(card["confidence"]) >= 4 and support_fraction < 0.75
                ),
                "unclear_full_support_flag": (
                    str(card["explanation_type"]) == "unclear_or_mixed" and support_fraction >= 0.95
                ),
                "primary_explanation": str(card["primary_explanation"]),
                "candidate_behavioral_explanation": str(card["candidate_behavioral_explanation"]),
                "representative_evidence_ids": _json_cell(card["representative_evidence_ids"]),
                "alternative_explanations": _json_cell(card["alternative_explanations"]),
                "possible_confounds": _json_cell(card["possible_confounds"]),
                "limitations": _json_cell(card["limitations"]),
            }
        )

    latent_df = pd.DataFrame(latent_rows)
    sample_df = pd.DataFrame(sample_rows)
    association_df = stable.merge(latent_df, on="latent_idx", how="left", validate="many_to_one")
    labels = sorted(association_df["label"].unique())
    label_rows: list[dict[str, Any]] = []
    for label in labels:
        group = association_df[association_df["label"].eq(label)].copy()
        latent_ids = set(group["latent_idx"].astype(int))
        label_samples = sample_df[sample_df["latent_idx"].isin(latent_ids)]
        type_counts = group["explanation_type"].value_counts()
        clear_group = group[group["clear_explanation"].astype(bool)]
        label_rows.append(
            {
                "label": label,
                "n_associations": len(group),
                "n_unique_latents": group["latent_idx"].nunique(),
                "mean_confidence": float(group["confidence"].mean()),
                "confidence_ge_4_rate": float((group["confidence"] >= 4).mean()),
                "mean_support_fraction": float(group["support_fraction"].mean()),
                "n_clear_explanations": len(clear_group),
                "mean_clear_pattern_support_fraction": float(clear_group["support_fraction"].mean()),
                "low_support_rate": _rate(clear_group["low_support_flag"]),
                "high_confidence_low_support_rate": _rate(clear_group["high_confidence_low_support_flag"]),
                "behavioral_function_rate": float(type_counts.get("behavioral_function", 0) / len(group)),
                "linguistic_structure_rate": float(type_counts.get("linguistic_structure", 0) / len(group)),
                "affective_content_rate": float(type_counts.get("affective_content", 0) / len(group)),
                "topic_rate": float(type_counts.get("topic", 0) / len(group)),
                "surface_artifact_rate": float(type_counts.get("surface_artifact", 0) / len(group)),
                "unclear_or_mixed_rate": float(type_counts.get("unclear_or_mixed", 0) / len(group)),
                "mean_word_count": float(label_samples["word_count"].mean()),
                "very_short_sample_rate": _rate(label_samples["very_short"]),
                "fragment_like_sample_rate": _rate(label_samples["fragment_like"]),
                "transcription_noise_sample_rate": _rate(label_samples["transcription_noise"]),
                "near_duplicate_sample_rate": _rate(label_samples["near_duplicate"]),
            }
        )
    label_df = pd.DataFrame(label_rows)

    review_rows: list[dict[str, Any]] = []
    sample_text_by_latent = {
        latent_idx: {str(row["id"]): str(row["text"]) for row in packs[latent_idx]["samples_for_model"]}
        for latent_idx in packs
    }
    for label in labels:
        group = association_df[association_df["label"].eq(label)].copy()
        group["review_risk"] = (
            np.where(group["clear_explanation"], 1 - group["support_fraction"], 0.50) * 0.45
            + group["fragment_like_rate"] * 0.20
            + group["transcription_noise_rate"] * 0.15
            + group["near_duplicate_rate"] * 0.10
            + (5 - group["confidence"]) / 4 * 0.10
        )
        selected = group.sort_values(["review_risk", "latent_idx"], ascending=[False, True]).head(3)
        for _, row in selected.iterrows():
            latent_idx = int(row["latent_idx"])
            card = cards[latent_idx]
            evidence_texts = [
                sample_text_by_latent[latent_idx].get(str(sample_id), "")
                for sample_id in card["representative_evidence_ids"]
            ]
            noisy = sample_df[
                sample_df["latent_idx"].eq(latent_idx)
                & (sample_df["fragment_like"] | sample_df["transcription_noise"] | sample_df["near_duplicate"])
            ].head(3)
            review_rows.append(
                {
                    "label": label,
                    "latent_idx": latent_idx,
                    "short_name": row["short_name"],
                    "explanation_type": row["explanation_type"],
                    "confidence": int(row["confidence"]),
                    "support_fraction": float(row["support_fraction"]),
                    "fragment_like_rate": float(row["fragment_like_rate"]),
                    "transcription_noise_rate": float(row["transcription_noise_rate"]),
                    "near_duplicate_rate": float(row["near_duplicate_rate"]),
                    "primary_explanation": row["primary_explanation"],
                    "representative_evidence_texts": _json_cell(evidence_texts),
                    "flagged_sample_texts": _json_cell(noisy["text"].astype(str).tolist()),
                    "review_priority": "high" if float(row["review_risk"]) >= 0.25 else "medium",
                }
            )
    review_df = pd.DataFrame(review_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    latent_path = output_dir / "latent_card_quality_by_latent.csv"
    sample_path = output_dir / "sentence_quality_details.csv"
    association_path = output_dir / "label_latent_card_associations.csv"
    label_path = output_dir / "latent_card_quality_by_label.csv"
    review_path = output_dir / "stratified_quality_review_sample.csv"
    latent_df.to_csv(latent_path, index=False, encoding="utf-8-sig")
    sample_df.to_csv(sample_path, index=False, encoding="utf-8-sig")
    association_df.to_csv(association_path, index=False, encoding="utf-8-sig")
    label_df.to_csv(label_path, index=False, encoding="utf-8-sig")
    review_df.to_csv(review_path, index=False, encoding="utf-8-sig")

    type_counts = latent_df["explanation_type"].value_counts()
    report_lines = [
        "# DeepSeek Latent Card 样本质量与标签统计报告",
        "",
        "## 1. 统计范围",
        "",
        f"本报告覆盖 {len(latent_df)} 个 unique stable-core latent、{len(association_df)} 条 label-latent 关联和 {len(sample_df):,} 条 latent 内句子记录。一个 latent 可以关联多个标签，因此标签表按关联口径统计；同一 latent 的 50 条句子会分别计入其关联的每个标签。",
        "",
        "DeepSeek 在生成时只看到 latent ID 与 50 条去重文本，没有看到 MISC 标签、激活值、排名或对照分组。所有 225 张 card 已通过 JSON schema、ID 完整分区、代表证据归属和多数门槛校验。",
        "",
        "## 2. 质量指标定义",
        "",
        "- `support_fraction`：模型归入主要解释支持集合的句子比例。低于 0.75 记为低覆盖，仅表示解释边界较弱，不自动等于解释错误。",
        "- `very_short_sample_rate`：不超过 3 个词的句子比例。",
        "- `fragment_like_sample_rate`：以连接词/限定词结尾、包含破碎标点或明显未完结的句子比例。该指标是启发式审查线索。",
        "- `transcription_noise_sample_rate`：连续重复词、破碎标点、替换字符等转录噪声比例。口语中的自然重复也可能被计入。",
        "- `near_duplicate_sample_rate`：同一 latent 的 50 条唯一文本中，字符相似度至少 0.92 的近重复句子所占比例。",
        "- `high_confidence_low_support_rate`：置信度至少 4，但支持比例低于 0.75 的 card 比例，用于发现可能的置信度校准问题。",
        "",
        "这些规则用于定位需要人工复核的样本，不是人工 gold 质量标签，也不证明 latent 的真实机制。",
        "",
        "## 3. 总体结果",
        "",
        f"- 明确类型 card 的平均支持比例：{latent_df['clear_pattern_support_fraction'].mean():.1%}",
        f"- 低覆盖明确类型 card：{int(latent_df['low_support_flag'].sum())}/{int(latent_df['clear_explanation'].sum())} ({latent_df.loc[latent_df['clear_explanation'], 'low_support_flag'].mean():.1%})",
        f"- 高置信度但低覆盖：{int(latent_df['high_confidence_low_support_flag'].sum())}/{int(latent_df['clear_explanation'].sum())} ({latent_df.loc[latent_df['clear_explanation'], 'high_confidence_low_support_flag'].mean():.1%})",
        f"- 极短句：{sample_df['very_short'].mean():.1%}",
        f"- 疑似残句：{sample_df['fragment_like'].mean():.1%}",
        f"- 疑似转录噪声：{sample_df['transcription_noise'].mean():.1%}",
        f"- 近重复句：{sample_df['near_duplicate'].mean():.1%}",
        "",
        "解释类型（unique latent 口径）：",
        "",
    ]
    for name, count in type_counts.items():
        report_lines.append(f"- `{name}`：{int(count)} ({count / len(latent_df):.1%})")
    report_lines.extend(["", "置信度分布（unique latent 口径）：", ""])
    for confidence, count in latent_df["confidence"].value_counts().sort_index().items():
        report_lines.append(f"- `{int(confidence)}` 分：{int(count)} ({count / len(latent_df):.1%})")
    report_lines.extend(
        [
            "",
            "## 4. 各标签统计",
            "",
            "| 标签 | 关联数 | 平均置信度 | 明确类型平均支持 | 低覆盖 | 行为功能 | 语言结构 | 不清晰/混合 | 极短句 | 疑似残句 | 转录噪声 | 近重复 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in label_df.itertuples(index=False):
        report_lines.append(
            f"| {row.label} | {int(row.n_associations)} | {row.mean_confidence:.2f} | "
            f"{row.mean_clear_pattern_support_fraction:.1%} | {row.low_support_rate:.1%} | "
            f"{row.behavioral_function_rate:.1%} | {row.linguistic_structure_rate:.1%} | "
            f"{row.unclear_or_mixed_rate:.1%} | {row.very_short_sample_rate:.1%} | "
            f"{row.fragment_like_sample_rate:.1%} | {row.transcription_noise_sample_rate:.1%} | "
            f"{row.near_duplicate_sample_rate:.1%} |"
        )
    by_label = label_df.set_index("label")
    report_lines.extend(
        [
            "",
            "## 5. 标签层面观察",
            "",
            f"- `QU/QUC/QUO` 的明确解释支持率均在 {min(by_label.loc['QU', 'mean_clear_pattern_support_fraction'], by_label.loc['QUC', 'mean_clear_pattern_support_fraction'], by_label.loc['QUO', 'mean_clear_pattern_support_fraction']):.1%} 以上，且行为功能与语言结构合计接近或达到 100%。这说明问题类 latent 的句子簇较一致，但也提示问句结构可能与行为功能共同驱动解释。",
            f"- `GI` 的低覆盖率最高（{by_label.loc['GI', 'low_support_rate']:.1%}），并且 {by_label.loc['GI', 'topic_rate']:.1%} 的 card 被归为主题类型。部分解释停留在宽泛的 healthcare/health behavior 主题，需要优先检查是否足够具体。",
            f"- `AF` 的明确解释平均支持率最低（{by_label.loc['AF', 'mean_clear_pattern_support_fraction']:.1%}），极短句率也最高（{by_label.loc['AF', 'very_short_sample_rate']:.1%}）。赞许、感谢和一般正向评价容易被合并，需要检查行为功能与情感内容是否混淆。",
            f"- `RE` 与 `REC` 的构成接近：明确解释支持率分别为 {by_label.loc['RE', 'mean_clear_pattern_support_fraction']:.1%} 和 {by_label.loc['REC', 'mean_clear_pattern_support_fraction']:.1%}，但低覆盖率均超过 13%。低覆盖项多使用 reflection/advice 等宽泛名称，不能仅凭名称视为已识别反映功能。",
            f"- `RES` 的平均置信度最低（{by_label.loc['RES', 'mean_confidence']:.2f}），`unclear_or_mixed` 比例最高（{by_label.loc['RES', 'unclear_or_mixed_rate']:.1%}）。其明确类型 card 虽有较高支持率，但总体可解释性受不清晰子集限制。",
            f"- `SU` 的疑似残句率（{by_label.loc['SU', 'fragment_like_sample_rate']:.1%}）和转录噪声率（{by_label.loc['SU', 'transcription_noise_sample_rate']:.1%}）在标签中最高，建议人工审查时同时查看原句，不只阅读 card 摘要。",
            "",
            "总体判断：句子簇和生成 card 足以支持后续分层人工复核，但不能直接作为最终人工确认解释。主要风险集中在宽泛行为命名、问题句式与功能混淆、近重复模板，以及少量残句和转录噪声。",
            "",
            "## 6. 各标签优先复核对象",
            "",
        ]
    )
    for label in labels:
        report_lines.append(f"### {label}")
        report_lines.append("")
        for row in review_df[review_df["label"].eq(label)].itertuples(index=False):
            report_lines.append(
                f"- latent `{int(row.latent_idx)}`，`{row.short_name}`：类型 `{row.explanation_type}`，"
                f"置信度 {int(row.confidence)}，支持 {row.support_fraction:.1%}，疑似残句 "
                f"{row.fragment_like_rate:.1%}，转录噪声 {row.transcription_noise_rate:.1%}，"
                f"近重复 {row.near_duplicate_rate:.1%}。"
            )
        report_lines.append("")
    report_lines.extend(
        [
            "## 7. 结论与使用边界",
            "",
            "1. 当前 card 在结构层面完整，但结构通过不等于概念解释已经人工确认。优先审查低支持、高置信度低支持以及不清晰/混合 card。",
            "2. `unclear_or_mixed` card 的 supporting/outlier 分区不具有统一语义，部分结果将全部句子列为 supporting；因此主表的平均支持率只在明确解释类型中计算。",
            "3. 句子簇已经消除规范化完全重复，但仍存在近重复、转录残句和口语重复；这些现象可能让模型更容易归纳表层结构。",
            "4. 标签统计描述的是 stable-core latent 的 card 构成，不表示标签本身由某一种解释类型定义。共享 latent 会出现在多个标签中。",
            "5. 分层抽样表每个标签列出 3 个风险最高的关联，适合作为第一轮人工复核入口。",
            "",
            "## 8. 输出文件",
            "",
            f"- 标签汇总：`{label_path}`",
            f"- latent 明细：`{latent_path}`",
            f"- 标签-latent 关联：`{association_path}`",
            f"- 句子级质量明细：`{sample_path}`",
            f"- 分层抽样复核：`{review_path}`",
        ]
    )
    report_path = output_dir / "latent_card_quality_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest = {
        "analysis": "deepseek_latent_card_quality_by_label",
        "inputs": {"cards": str(cards_path), "packs": str(packs_path), "stable_latents": str(stable_path)},
        "outputs": {
            "report": str(report_path),
            "by_label": str(label_path),
            "by_latent": str(latent_path),
            "associations": str(association_path),
            "sentence_details": str(sample_path),
            "review_sample": str(review_path),
        },
        "counts": {"unique_latents": len(latent_df), "label_latent_associations": len(association_df), "sentence_rows": len(sample_df), "labels": len(label_df)},
        "heuristic_thresholds": {"low_support": 0.75, "very_short_words": 3, "near_duplicate_similarity": 0.92},
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--stable-latents", type=Path, default=Path("outputs/cross_val/stable_topk_selection/stable_topk_latent_set.csv"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.base_dir / "quality_report"
    manifest = build_report(
        cards_path=args.base_dir / "card_outputs" / "validated_cards.jsonl",
        packs_path=args.base_dir / "evidence_packs" / "latent_card_sentence_packs.jsonl",
        stable_path=args.stable_latents,
        output_dir=output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
