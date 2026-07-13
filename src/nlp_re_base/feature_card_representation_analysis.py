"""Aggregate stable-core feature-card explanations into label-level patterns."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LABEL_ORDER = ("RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF")
HIERARCHICAL_PAIRS = {
    frozenset(("RE", "RES")),
    frozenset(("RE", "REC")),
    frozenset(("QU", "QUO")),
    frozenset(("QU", "QUC")),
}
SIBLING_PAIRS = {frozenset(("RES", "REC")), frozenset(("QUO", "QUC"))}


def _read_jsonl(path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _has(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def classify_candidate_theme(row: pd.Series) -> str:
    """Assign an auditable exploratory theme from the generated explanation text."""

    label = str(row["label"])
    text = " ".join(
        str(row.get(column, ""))
        for column in (
            "short_name",
            "candidate_explanation",
            "main_hypothesis",
            "dominant_pattern",
            "surface_pattern",
            "semantic_pattern",
            "surface_component",
            "semantic_component",
        )
    )
    if label == "AF":
        if _has(text, r"thank|gratitude|appreciat|welcome"):
            return "gratitude_or_welcome"
        if _has(text, r"praise|positive|affirm|encourag|approval|great|good|glad"):
            return "praise_or_positive_evaluation"
        return "other_affiliative_pattern"
    if label == "GI":
        if _has(text, r"medication|medicine|drug|pharmac"):
            return "medication_information"
        if _has(text, r"risk|cholesterol|blood pressure|weight|health"):
            return "health_risk_or_measurement"
        if _has(text, r"advice|instruction|recommend|educat|explain|information"):
            return "advice_or_information_delivery"
        return "other_information_pattern"
    if label in {"QU", "QUO", "QUC"}:
        if _has(text, r"scale|rating|one to ten|zero to ten|likert"):
            return "numeric_scale_question"
        if _has(text, r"what question|what_questions|question_what|what_question"):
            return "what_question"
        if _has(text, r"how question|how_questions|how_do_you|how_long|how_much"):
            return "how_question"
        if _has(text, r"do_you|did_you|are_you|yes.?no|closed|question formation|question_marker"):
            return "auxiliary_or_closed_question"
        if _has(text, r"hypothetical|would you|could you"):
            return "hypothetical_question"
        return "general_therapeutic_question"
    if label in {"RE", "REC", "RES"}:
        if _has(text, r"sounds.like|seems.like|refram"):
            return "sounds_seems_reframing"
        if _has(text, r"hedg|kind of|sort of|tentative"):
            return "hedged_interpretation"
        if _has(text, r"advice|warning|directed|you-directed|healthcare|medical"):
            return "advice_or_health_spillover"
        if _has(text, r"\bso\b|discourse marker|okay|and so|so_you"):
            return "so_okay_discourse_frame"
        if _has(text, r"reflect|paraphras|summar|validat|therapeutic"):
            return "general_reflection"
        return "other_reflection_related"
    if label == "SU":
        if _has(text, r"help|assist|support"):
            return "help_or_support_offer"
        if _has(text, r"apolog|sorry|permission"):
            return "apology_or_permission"
        if _has(text, r"empath|understand|validat|difficulty"):
            return "empathy_or_validation"
        if _has(text, r"first.person|I-utterance|\bi know\b|\bi think\b"):
            return "first_person_acknowledgment"
        return "other_supportive_pattern"
    return "other"


def build_feature_card_analysis(
    stable_core: pd.DataFrame, explanations: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stable = stable_core[stable_core["stable_set_role"].astype(str).eq("stable_core")].copy()
    stable["label"] = stable["label"].astype(str).str.upper()
    explanations = explanations.drop_duplicates("latent_idx").copy()
    joined = stable.merge(explanations, on="latent_idx", how="left", validate="many_to_one")
    if joined["short_name"].isna().any():
        missing = joined.loc[joined["short_name"].isna(), "latent_idx"].astype(int).tolist()
        raise ValueError(f"missing explanations for stable-core latents: {missing[:10]}")

    shared = stable.groupby("latent_idx")["label"].agg(lambda x: "|".join(sorted(set(x))))
    joined = joined.merge(shared.rename("shared_labels"), on="latent_idx", how="left")
    joined["n_associated_labels"] = joined["shared_labels"].str.count(r"\|") + 1
    joined["shared_latent_flag"] = joined["n_associated_labels"].gt(1)
    joined["candidate_theme"] = joined.apply(classify_candidate_theme, axis=1)

    summary = (
        joined.groupby("label", as_index=False)
        .agg(
            n_label_latent_associations=("latent_idx", "size"),
            n_unique_latents=("latent_idx", "nunique"),
            n_shared_latents=("shared_latent_flag", "sum"),
            shared_fraction=("shared_latent_flag", "mean"),
            surface_majority_fraction=("surface_majority_gate_passed", "mean"),
            semantic_majority_fraction=("semantic_majority_gate_passed", "mean"),
            mean_surface_confidence=("surface_confidence", "mean"),
            mean_semantic_confidence=("semantic_confidence", "mean"),
            mean_surface_support=("surface_raw_support_fraction", "mean"),
            mean_semantic_support=("semantic_raw_support_fraction", "mean"),
        )
    )
    summary["n_exclusive_latents"] = (
        summary["n_unique_latents"] - summary["n_shared_latents"]
    )
    status = pd.crosstab(joined["label"], joined["induction_status"]).reset_index()
    summary = summary.merge(status, on="label", how="left").fillna(0)

    themes = (
        joined.groupby(["label", "candidate_theme"], as_index=False)
        .agg(n_associations=("latent_idx", "size"))
    )
    totals = themes.groupby("label")["n_associations"].transform("sum")
    themes["fraction_within_label"] = themes["n_associations"] / totals

    labels = [label for label in LABEL_ORDER if label in set(stable["label"])]
    matrix = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for _, group in stable.groupby("latent_idx"):
        present = sorted(set(group["label"]).intersection(labels))
        for left in present:
            for right in present:
                matrix.loc[left, right] += 1
    matrix.index.name = "label"
    return joined, summary, themes, matrix.reset_index()


def _fmt(value: float) -> str:
    return f"{value:.1%}"


def _write_report(
    path: Path,
    joined: pd.DataFrame,
    summary: pd.DataFrame,
    themes: pd.DataFrame,
) -> None:
    summary_map = summary.set_index("label")
    narrative = {
        "AF": "以肯定、赞扬、积极评价、感谢和欢迎为主，表现为多个亲和性子功能，而不是单一的 affirmation 模块。共享率低，说明该簇相对标签专属，但仍需区分真正行为功能与 good/great/thank 等情感词触发。",
        "GI": "主要围绕药物、风险、指标和健康建议。没有 stable-core latent 与其他标签共享，标签专属性最强；但这种专属性很可能同时来自医疗主题词，因此 GI 更像内容领域与信息提供功能的混合表征。",
        "QU": "几乎全部由疑问形式支持，并与 QUO/QUC 高度共享。它更像问题家族的父级表示，而不是独立功能簇；量表、what/how、助动词和假设式问题构成多个子通路。",
        "QUO": "以 what/how 开放式问句和探索性提问为主，表面模式覆盖极高。大量 latent 与 QU 共享，部分也与 QUC 共享，说明开放性可能建立在通用疑问骨架上，再由较少特征编码开放程度。",
        "QUC": "包含 do/did/are 等助动词起始、事实核验、数量频率和量表问题。相比 QUO 有更多标签独占 latent，但仍与 QU 共享核心疑问结构，呈现通用问句骨架加封闭式子类型的组织。",
        "RE": "由 so/okay 话语框架、sounds/seems like、hedging、第二人称和反映/总结功能共同构成。多数 latent 与 REC/RES 共享，符合父标签结构；RE 本身不应被解释为独立于两个子类的单一概念。",
        "REC": "最集中于 sounds/seems like、改述、重构、试探性解释和复杂反映。语义多数门覆盖很高，但与 RE 的共享也最高之一；当前无 client 前文，不能仅凭这些模式确认复杂反映相对简单反映的增量语义。",
        "RES": "stable-core 数量最少，语义归纳多于稳定表面模式，且混入第二人称、建议、医疗和一般互动模式。它与 RE 有部分共享，但没有形成像 REC 那样清晰的专属形式簇，可能反映上下文依赖、标签噪声或 probe 的区分困难。",
        "SU": "以理解、共情、验证、提供帮助、道歉和请求许可为主，形成多个支持性人际行为子簇。大多数 latent 为标签独占，说明支持性功能与问题/反映家族相对分离，但功能范围内部仍较宽。",
    }
    lines = [
        "# Stable-core Feature Cards 的标签级表征模式分析",
        "",
        "## 分析范围",
        "",
        f"本分析将 {joined['latent_idx'].nunique()} 个 unique stable-core latent 的 Top-50 去重文本解释映射回 {len(joined)} 个 label-latent 关联。解释由 LLM 自动归纳，以下结论是候选表征模式，不等同于人工确认概念或因果机制。主题分类基于 short_name、surface pattern 和 semantic pattern 的透明关键词规则，仅用于汇总。",
        "",
        "## 总体结论",
        "",
        "1. 标签不是以 one-label-one-latent 方式组织，而是由多个表面和功能子模式共同支持。",
        "2. QU/QUO/QUC 构成高度共享的问题家族；RE/REC/RES 构成高度共享的反映家族。父标签主要复用子标签特征。",
        "3. GI、AF 和 SU 更接近标签专属簇，但 GI 显著受医疗主题词影响，AF 和 SU 也包含多个相邻人际功能。",
        "4. 表面形式和语义功能普遍同时出现，不能把全部 stable-core latent 直接解释为心理咨询功能特征。",
        "",
        "## 定量概览",
        "",
        "| 标签 | 关联数 | 独占数 | 共享率 | 表面多数门 | 语义多数门 | 平均表面置信度 | 平均语义置信度 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LABEL_ORDER:
        if label not in summary_map.index:
            continue
        row = summary_map.loc[label]
        lines.append(
            f"| {label} | {int(row.n_label_latent_associations)} | {int(row.n_exclusive_latents)} | {_fmt(row.shared_fraction)} | "
            f"{_fmt(row.surface_majority_fraction)} | {_fmt(row.semantic_majority_fraction)} | "
            f"{row.mean_surface_confidence:.3f} | {row.mean_semantic_confidence:.3f} |"
        )
    latent_labels = joined.groupby("latent_idx")["label"].agg(lambda x: set(x))
    key_pairs = (("RE", "REC"), ("RE", "RES"), ("REC", "RES"), ("QU", "QUO"), ("QU", "QUC"), ("QUO", "QUC"), ("SU", "AF"))
    lines.extend(
        [
            "",
            "## 关键标签重叠",
            "",
            "| 标签对 | 共享 latent | Jaccard 重叠率 |",
            "|---|---:|---:|",
        ]
    )
    label_sets = {
        label: set(joined.loc[joined["label"].eq(label), "latent_idx"].astype(int))
        for label in LABEL_ORDER
    }
    for left, right in key_pairs:
        intersection = len(label_sets[left] & label_sets[right])
        union = len(label_sets[left] | label_sets[right])
        lines.append(f"| {left}-{right} | {intersection} | {intersection / union:.1%} |")
    triple_reflection = sum({"RE", "REC", "RES"}.issubset(labels) for labels in latent_labels)
    triple_question = sum({"QU", "QUO", "QUC"}.issubset(labels) for labels in latent_labels)
    lines.extend(
        [
            "",
            f"三标签共同 latent：RE/REC/RES = {triple_reflection}；QU/QUO/QUC = {triple_question}。",
            "",
            "## 各标签模式",
            "",
        ]
    )
    for label in LABEL_ORDER:
        if label not in summary_map.index:
            continue
        top = themes[themes["label"].eq(label)].sort_values(
            ["n_associations", "candidate_theme"], ascending=[False, True]
        )
        theme_text = "；".join(
            f"{row.candidate_theme} {int(row.n_associations)}（{row.fraction_within_label:.1%}）"
            for row in top.itertuples(index=False)
        )
        lines.extend(
            [
                f"### {label}",
                "",
                narrative[label],
                "",
                f"启发式主题分布：{theme_text}。",
                "",
            ]
        )
    lines.extend(
        [
            "## 论文层面的解释",
            "",
            "这些 feature cards 更支持 compositional and hierarchical organization（组合式与层级式组织），而不是每个 MISC 标签对应一个单独神经概念。问题类标签主要共享疑问句骨架，再由 what/how、助动词、量表和开放探索等模式细分；反映类标签共享 so/okay、sounds/seems、hedging 和第二人称改述框架，再由复杂重构、简单复述及相邻建议模式分化。GI、AF、SU 的低共享说明部分行为可形成相对专属簇，但这些簇仍混合主题词、情感词和人际功能。",
            "",
            "最稳健的主张是：stable-core SAE 特征揭示了多个可检查的语言与行为成分，它们以共享、复用和碎片化的方式共同支持 MISC 标签判断。当前结果不支持单个 latent 等价于标签，也不能单凭自动 feature card 区分真正功能编码与表面相关性。",
            "",
            "## 后续验证优先级",
            "",
            "1. 人工复核每个标签最高排名、标签独占和跨标签共享的代表 cards。",
            "2. 对 QUO/QUC 和 REC/RES 构造匹配表面形式的对照，检验子类差异是否超出通用问句或反映模板。",
            "3. 对 GI 检查医疗主题词相同但话语功能不同的文本，区分 topic feature 与 information-giving feature。",
            "4. 对 RES/REC 补充 client 前文，否则复杂度和反映关系的结论保持保守。",
            "5. 将人工审查结论与共享矩阵结合，只在审核通过后选择论文 Figure 3 的代表 cards。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_single_choice_feature_card_analysis(
    stable_core: pd.DataFrame, explanations: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate the earlier surface/semantic/mixed single-choice explanations."""

    required = {"latent_idx", "feature_type", "confidence", "majority_gate_passed"}
    missing = required.difference(explanations.columns)
    if missing:
        raise ValueError(f"single-choice explanations missing columns: {sorted(missing)}")
    stable = stable_core[stable_core["stable_set_role"].astype(str).eq("stable_core")].copy()
    stable["label"] = stable["label"].astype(str).str.upper()
    explanations = explanations.drop_duplicates("latent_idx").copy()
    explanations["reported_feature_type"] = explanations["feature_type"].astype(str)
    no_stable_mask = ~explanations["majority_gate_passed"].astype(bool)
    if "induction_status" in explanations:
        no_stable_mask |= explanations["induction_status"].astype(str).eq(
            "insufficient_majority"
        )
    explanations["effective_feature_type"] = explanations["reported_feature_type"]
    explanations.loc[no_stable_mask, "effective_feature_type"] = "no_stable_pattern"
    joined = stable.merge(explanations, on="latent_idx", how="left", validate="many_to_one")
    if joined["feature_type"].isna().any():
        missing_ids = joined.loc[joined["feature_type"].isna(), "latent_idx"].astype(int).tolist()
        raise ValueError(f"missing explanations for stable-core latents: {missing_ids[:10]}")

    latent_labels = stable.groupby("latent_idx")["label"].agg(lambda x: sorted(set(x)))
    joined = joined.merge(latent_labels.rename("shared_label_list"), on="latent_idx", how="left")
    joined["shared_labels"] = joined["shared_label_list"].map(lambda x: "|".join(x))
    joined["raw_shared_latent_flag"] = joined["shared_label_list"].map(len).gt(1)

    def eligible_labels(row: pd.Series) -> list[str]:
        label = str(row["label"])
        return [
            other
            for other in row["shared_label_list"]
            if other != label and frozenset((label, other)) not in HIERARCHICAL_PAIRS
        ]

    joined["non_hierarchical_shared_label_list"] = joined.apply(eligible_labels, axis=1)
    joined["non_hierarchical_shared_labels"] = joined[
        "non_hierarchical_shared_label_list"
    ].map(lambda x: "|".join(x))
    joined["non_hierarchical_shared_flag"] = joined[
        "non_hierarchical_shared_label_list"
    ].map(bool)
    joined["candidate_theme"] = joined.apply(classify_candidate_theme, axis=1)

    summary = (
        joined.groupby("label", as_index=False)
        .agg(
            n_associations=("latent_idx", "size"),
            n_non_hierarchical_shared=("non_hierarchical_shared_flag", "sum"),
            non_hierarchical_shared_fraction=("non_hierarchical_shared_flag", "mean"),
            mean_confidence=("confidence", "mean"),
            majority_gate_fraction=("majority_gate_passed", "mean"),
        )
    )
    type_counts = pd.crosstab(joined["label"], joined["effective_feature_type"])
    for feature_type in ("surface", "semantic", "mixed", "no_stable_pattern"):
        if feature_type not in type_counts:
            type_counts[feature_type] = 0
    type_counts = type_counts.reset_index()
    summary = summary.merge(type_counts, on="label", how="left")
    for feature_type in ("surface", "semantic", "mixed", "no_stable_pattern"):
        summary[f"{feature_type}_fraction"] = summary[feature_type] / summary["n_associations"]

    themes = (
        joined.groupby(["label", "candidate_theme"], as_index=False)
        .agg(n_associations=("latent_idx", "size"))
    )
    themes["fraction_within_label"] = themes["n_associations"] / themes.groupby("label")[
        "n_associations"
    ].transform("sum")

    label_sets = {
        label: set(stable.loc[stable["label"].eq(label), "latent_idx"].astype(int))
        for label in LABEL_ORDER
    }
    pair_rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(LABEL_ORDER):
        for right in LABEL_ORDER[left_index + 1 :]:
            pair = frozenset((left, right))
            intersection = len(label_sets[left] & label_sets[right])
            union = len(label_sets[left] | label_sets[right])
            relation = (
                "hierarchical_definition"
                if pair in HIERARCHICAL_PAIRS
                else "sibling_labels"
                if pair in SIBLING_PAIRS
                else "cross_family"
            )
            pair_rows.append(
                {
                    "label_left": left,
                    "label_right": right,
                    "shared_latents": intersection,
                    "jaccard": intersection / union if union else np.nan,
                    "relation_type": relation,
                    "eligible_as_representation_evidence": relation != "hierarchical_definition",
                }
            )
    return joined, summary, themes, pd.DataFrame(pair_rows)


def _write_single_choice_report(
    path: Path,
    joined: pd.DataFrame,
    summary: pd.DataFrame,
    themes: pd.DataFrame,
    pair_audit: pd.DataFrame,
) -> None:
    summary_map = summary.set_index("label")
    unique_types = (
        joined.drop_duplicates("latent_idx")["effective_feature_type"].value_counts().to_dict()
    )
    lines = [
        "# Stable-core Feature Cards 的标签级表征模式分析（单选类型版）",
        "",
        "## 分析口径",
        "",
        f"本分析使用早期 DeepSeek 单选归纳结果：每个 latent 被判断为 surface、semantic、mixed 或 no_stable_pattern。共覆盖 {joined['latent_idx'].nunique()} 个唯一 latent 和 {len(joined)} 条标签-latent 关联。",
        "",
        "RE 是 RES/REC 的父标签，QU 是 QUO/QUC 的父标签。父子标签在样本定义上天然重合，因此父子 stable-core 集合重合不作为 SAE 表征共享的独立证据。只有同层标签或跨家族标签的共享才进入探索性解释。",
        "",
        "## 总体类型构成",
        "",
        f"按唯一 latent 计数：mixed = {unique_types.get('mixed', 0)}，semantic = {unique_types.get('semantic', 0)}，surface = {unique_types.get('surface', 0)}，no_stable_pattern = {unique_types.get('no_stable_pattern', 0)}。未通过多数门或 induction_status=insufficient_majority 的记录统一归入 no_stable_pattern，同时在明细表保留 DeepSeek 原始 reported_feature_type。",
        "",
        "该版本明显偏向 mixed。这里的 mixed 表示 Top-50 中同时观察到稳定表面线索与功能/语义共性，不等于已经证明 latent 同时编码两个可独立操纵的因素。",
        "",
        "## 本部分工作的贡献",
        "",
        "本部分使用 SAE 将 LLM 中可用于区分 MI/MISC 行为标签的内部表征分解到更细粒度的 latent 层面，并结合 feature cards 归纳各标签由哪些候选概念成分共同构成。与只报告标签级 probe 性能相比，该分析进一步揭示了标签内部的表征异质性：同一标签通常由词汇和句法形式、话语结构、语义主题及咨询功能等多个成分组合支持。",
        "",
        "具体而言，分析观察到问题类行为由通用问句骨架、what/how 结构、助动词、量表和数量表达等成分构成；反映类行为由 so/okay、sounds/seems like、第二人称改述、总结和重构等成分构成；GI、SU 与 AF 则分别呈现医疗信息、支持性互动和积极评价等相对集中的候选成分。由此，本部分提供了一种从标签级可解码性进一步下钻到标签内部概念组成的分析路径。",
        "",
        "这里的“概念组成”指由 Top-50 激活文本支持、经 LLM 归纳得到的候选表征模式。它们是可检查的描述性和结构性证据，不表示单个 latent 等价于完整 MI 概念，也不构成 LLM 采用相应心理咨询机制的因果证明。",
        "",
        "## 各标签类型构成",
        "",
        "| 标签 | n | Surface | Semantic | Mixed | 无稳定模式 | 平均置信度 | 多数门通过率 | 非层级共享数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LABEL_ORDER:
        row = summary_map.loc[label]
        lines.append(
            f"| {label} | {int(row.n_associations)} | {int(row.surface)} ({row.surface_fraction:.1%}) | "
            f"{int(row.semantic)} ({row.semantic_fraction:.1%}) | {int(row.mixed)} ({row.mixed_fraction:.1%}) | "
            f"{int(row.no_stable_pattern)} ({row.no_stable_pattern_fraction:.1%}) | {row.mean_confidence:.3f} | "
            f"{row.majority_gate_fraction:.1%} | {int(row.n_non_hierarchical_shared)} |"
        )

    hierarchy = pair_audit[pair_audit["relation_type"].eq("hierarchical_definition")]
    eligible = pair_audit[
        pair_audit["eligible_as_representation_evidence"] & pair_audit["shared_latents"].gt(0)
    ].sort_values(["shared_latents", "label_left", "label_right"], ascending=[False, True, True])
    lines.extend(["", "## 标签共享审计", "", "以下父子重合仅作数据层级核验，不解释为 SAE 发现：", ""])
    for row in hierarchy.itertuples(index=False):
        lines.append(f"- {row.label_left}-{row.label_right}: {row.shared_latents} 个。")
    lines.extend(["", "可用于探索的非层级共享：", ""])
    for row in eligible.itertuples(index=False):
        lines.append(
            f"- {row.label_left}-{row.label_right}: {row.shared_latents} 个，Jaccard = {row.jaccard:.1%}，关系类型 = {row.relation_type}。"
        )

    lines.extend(["", "## 各标签候选模式", ""])
    for label in LABEL_ORDER:
        top = themes[themes["label"].eq(label)].sort_values(
            ["n_associations", "candidate_theme"], ascending=[False, True]
        )
        theme_text = "；".join(
            f"{row.candidate_theme} {int(row.n_associations)}（{row.fraction_within_label:.1%}）"
            for row in top.itertuples(index=False)
        )
        lines.extend([f"### {label}", "", f"候选主题：{theme_text}。", ""])

    lines.extend(
        [
            "## 结论边界",
            "",
            "1. 最直接的结果是：绝大多数 stable-core latent 被 DeepSeek 判断为 mixed，而不是纯表面或纯功能特征。",
            "2. GI 的纯 semantic 比例相对最高；问题类出现少量纯 surface 特征；AF 和 SU 几乎全部是 mixed。",
            "3. feature_type 是 LLM 对 Top-50 文本的归纳标签，不是独立干预实验，因此只能作为 feature-card 类型审计。",
            "4. RE/REC/RES 缺少前一句 client context，不能据此确认简单反映与复杂反映的功能边界。",
            "5. 当前标签来自现有分段标注产物，结论属于结构性和描述性证据，不是因果机制证明。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_feature_card_representation_analysis(
    *, stable_core_path: str | Path, explanations_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stable = pd.read_csv(stable_core_path)
    explanations = _read_jsonl(explanations_path)
    joined, summary, themes, shared_matrix = build_feature_card_analysis(stable, explanations)
    joined.to_csv(output / "feature_card_associations_enriched.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "representation_summary_by_label.csv", index=False, encoding="utf-8-sig")
    themes.to_csv(output / "candidate_themes_by_label.csv", index=False, encoding="utf-8-sig")
    shared_matrix.to_csv(output / "shared_latent_matrix.csv", index=False, encoding="utf-8-sig")
    top = joined.sort_values(["label", "full_data_rank"]).groupby("label", as_index=False).head(15)
    top.loc[:, [
        "label",
        "latent_idx",
        "full_data_rank",
        "short_name",
        "candidate_theme",
        "surface_confidence",
        "semantic_confidence",
        "surface_majority_gate_passed",
        "semantic_majority_gate_passed",
        "induction_status",
        "shared_labels",
        "shared_latent_flag",
    ]].to_csv(
        output / "top15_feature_patterns_by_label.csv", index=False, encoding="utf-8-sig"
    )
    _write_report(output / "feature_card_representation_analysis_report.md", joined, summary, themes)
    manifest = {
        "analysis": "stable_core_feature_card_representation_patterns",
        "inputs": {"stable_core": str(stable_core_path), "explanations": str(explanations_path)},
        "counts": {
            "n_label_latent_associations": int(len(joined)),
            "n_unique_latents": int(joined["latent_idx"].nunique()),
            "n_labels": int(joined["label"].nunique()),
        },
        "theme_assignment": "deterministic heuristic over LLM-generated short_name, surface_pattern, semantic_pattern, and candidate_explanation",
        "claim_boundary": "candidate descriptive structure; not human-validated concepts or causal mechanisms",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run_single_choice_feature_card_analysis(
    *, stable_core_path: str | Path, explanations_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stable = pd.read_csv(stable_core_path)
    explanations = _read_jsonl(explanations_path)
    joined, summary, themes, pair_audit = build_single_choice_feature_card_analysis(
        stable, explanations
    )
    joined.drop(columns=["shared_label_list", "non_hierarchical_shared_label_list"]).to_csv(
        output / "feature_card_associations_enriched.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        output / "representation_type_by_label.csv", index=False, encoding="utf-8-sig"
    )
    themes.to_csv(output / "candidate_themes_by_label.csv", index=False, encoding="utf-8-sig")
    pair_audit.to_csv(output / "label_pair_overlap_audit.csv", index=False, encoding="utf-8-sig")
    top = joined.sort_values(["label", "full_data_rank"]).groupby("label", as_index=False).head(15)
    top.loc[:, [
        "label",
        "latent_idx",
        "full_data_rank",
        "short_name",
        "reported_feature_type",
        "effective_feature_type",
        "candidate_theme",
        "confidence",
        "majority_gate_passed",
        "shared_labels",
        "non_hierarchical_shared_labels",
    ]].to_csv(output / "top15_feature_patterns_by_label.csv", index=False, encoding="utf-8-sig")
    _write_single_choice_report(
        output / "feature_card_representation_analysis_report.md",
        joined,
        summary,
        themes,
        pair_audit,
    )
    unique_types = joined.drop_duplicates("latent_idx")["effective_feature_type"].value_counts().to_dict()
    manifest = {
        "analysis": "stable_core_single_choice_feature_type_patterns",
        "inputs": {"stable_core": str(stable_core_path), "explanations": str(explanations_path)},
        "counts": {
            "n_label_latent_associations": int(len(joined)),
            "n_unique_latents": int(joined["latent_idx"].nunique()),
            "n_labels": int(joined["label"].nunique()),
            "unique_latent_feature_types": {key: int(value) for key, value in unique_types.items()},
        },
        "hierarchy_policy": "RE-RES, RE-REC, QU-QUO, and QU-QUC overlap is structural and excluded as representation-sharing evidence",
        "claim_boundary": "LLM feature-type audit over Top-50 examples; not causal or human-validated",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


__all__ = [
    "build_feature_card_analysis",
    "build_single_choice_feature_card_analysis",
    "classify_candidate_theme",
    "run_feature_card_representation_analysis",
    "run_single_choice_feature_card_analysis",
]
