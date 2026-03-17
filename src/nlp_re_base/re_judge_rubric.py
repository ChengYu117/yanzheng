"""Fixed RE rubric for AI expert-proxy review.

This module defines the only allowed rubric for AI judging in this repo.
The prompt layer must not redefine RE ad hoc.
"""

from __future__ import annotations

from copy import deepcopy

RUBRIC_VERSION = "1.0"

RE_JUDGE_RUBRIC: dict[str, object] = {
    "version": RUBRIC_VERSION,
    "method_positioning": {
        "label": "automatic_interpretation_evidence",
        "statement_zh": (
            "该评审是自动解释/专家代理证据，不等于因果证明。"
            "如果要做因果结论，仍需和 ablation、steering、控制组结果联合报告。"
        ),
    },
    "sources": [
        "MITI 4",
        "MISC 2.5",
        "Bills et al. 2023",
        "Huang et al. 2023",
        "Paulo et al. 2024",
        "CounselBench 2025",
    ],
    "task_context": {
        "current_default": (
            "当前数据集只有 counselor utterance（unit_text），没有稳定的前一句来访者上下文。"
            "因此默认执行句内 RE 线索评审，而不是严格的对前句回应准确性评审。"
        ),
    },
    "re_definition": {
        "simple_reflection": (
            "复述或近义改写来访者已明确表达的内容、感受或立场，不新增明显新意义。"
        ),
        "complex_reflection": (
            "在忠于来访者表达的前提下，合理深化其情绪、意义、重点、双面性、"
            "隐含动机或总结性理解。"
        ),
        "non_re": (
            "纯问题、建议、解释、教育、命令、说服、泛泛安慰、模板式回应，"
            "或与来访者表达弱相关的回应。"
        ),
    },
    "dimensions": [
        {
            "key": "mirrors_client_meaning",
            "description_zh": "是否镜像了来访者已经表达的内容、感受或意义。",
        },
        {
            "key": "adds_valid_meaning_or_empathy",
            "description_zh": "是否加入了合理的情绪理解、重点提炼或复杂反映。",
        },
        {
            "key": "non_directive_non_question",
            "description_zh": "是否避免提问、命令、建议、教育或说服。",
        },
        {
            "key": "natural_therapeutic_language",
            "description_zh": "语言是否自然、像真实咨询回应，而不是模板化拼接。",
        },
    ],
    "risk_flags": [
        "question_like",
        "advice_like",
        "information_giving",
        "lexical_template_only",
        "context_needed",
    ],
    "output_schema": {
        "utterance_review": {
            "has_clear_re_feature": ["yes", "partial", "no"],
            "re_type": ["simple", "complex", "mixed", "non_re", "unclear"],
            "clarity_score": "1-5 integer",
            "dimension_scores": {
                "mirrors_client_meaning": "1-5 integer",
                "adds_valid_meaning_or_empathy": "1-5 integer",
                "non_directive_non_question": "1-5 integer",
                "natural_therapeutic_language": "1-5 integer",
            },
            "evidence_spans": "list[str]",
            "reason_zh": "str",
            "risk_flags": "list[str]",
        },
        "synthesis_review": {
            "shared_feature_name": "str",
            "shared_feature_description_zh": "str",
            "common_positive_evidence": "list[str]",
            "common_counterevidence": "list[str]",
            "failure_modes": "list[str]",
        },
    },
}


def get_rubric_snapshot() -> dict[str, object]:
    """Return a deep copy of the fixed rubric."""
    return deepcopy(RE_JUDGE_RUBRIC)


def render_rubric_markdown() -> str:
    """Render a compact Markdown snapshot for docs or reports."""
    lines = [
        "# RE 定义与 AI 评审 Rubric",
        "",
        f"- 版本: `{RUBRIC_VERSION}`",
        "",
        "## 方法定位",
        "",
        RE_JUDGE_RUBRIC["method_positioning"]["statement_zh"],  # type: ignore[index]
        "",
        "## RE 定义",
        "",
        "### Simple Reflection",
        str(RE_JUDGE_RUBRIC["re_definition"]["simple_reflection"]),  # type: ignore[index]
        "",
        "### Complex Reflection",
        str(RE_JUDGE_RUBRIC["re_definition"]["complex_reflection"]),  # type: ignore[index]
        "",
        "### 非 RE",
        str(RE_JUDGE_RUBRIC["re_definition"]["non_re"]),  # type: ignore[index]
        "",
        "## 评审维度",
        "",
    ]
    for dim in RE_JUDGE_RUBRIC["dimensions"]:  # type: ignore[assignment]
        lines.append(f"- `{dim['key']}`: {dim['description_zh']}")

    lines.extend([
        "",
        "## 风险标记",
        "",
    ])
    for flag in RE_JUDGE_RUBRIC["risk_flags"]:  # type: ignore[assignment]
        lines.append(f"- `{flag}`")

    lines.extend([
        "",
        "## 参考来源",
        "",
    ])
    for source in RE_JUDGE_RUBRIC["sources"]:  # type: ignore[assignment]
        lines.append(f"- {source}")
    return "\n".join(lines) + "\n"
