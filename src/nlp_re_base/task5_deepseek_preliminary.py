"""Build and validate label-blind DeepSeek cards for Task 5 discovery clusters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .contrastive_evidence_pack import read_jsonl, write_json, write_jsonl
from .contrastive_llm_io import parse_llm_json_file
from .deepseek_latent_cards import (
    EVIDENCE_LEVELS,
    EVIDENCE_ROLES,
    EXPLANATION_TYPES,
    LATENT_CARD_SYSTEM_PROMPT,
)
from .deepseek_top50_induction import _sha256_text


TASK5_CARD_SYSTEM_PROMPT = LATENT_CARD_SYSTEM_PROMPT
TASK5_CARD_FIELDS = (
    "anonymous_unit_id",
    "short_name",
    "primary_explanation",
    "candidate_behavioral_explanation",
    "explanation_type",
    "supporting_sample_ids",
    "outlier_sample_ids",
    "representative_evidence_ids",
    "linguistic_evidence",
    "alternative_explanations",
    "possible_confounds",
    "limitations",
    "confidence",
    "confidence_rationale",
)


@dataclass(frozen=True)
class Task5DeepSeekCardConfig:
    samples_per_unit: int = 10
    min_representative_ids: int = 2
    max_representative_ids: int = 3
    representation_families: tuple[str, ...] = ("SAE", "PCA")


def _format_samples(samples: list[dict[str, str]]) -> str:
    return "\n".join(
        f"- sample_id={sample['sample_id']}\n  text: {sample['text'].replace(chr(10), ' ').strip()}"
        for sample in samples
    )


def build_task5_cluster_prompt(
    anonymous_unit_id: str,
    samples: list[dict[str, str]],
    config: Task5DeepSeekCardConfig = Task5DeepSeekCardConfig(),
) -> str:
    ids = [sample["sample_id"] for sample in samples]
    texts = [" ".join(sample["text"].strip().lower().split()) for sample in samples]
    if len(samples) != config.samples_per_unit or len(set(ids)) != len(ids):
        raise ValueError(f"Expected {config.samples_per_unit} uniquely identified samples")
    if len(set(texts)) != len(texts):
        raise ValueError("Task 5 discovery samples must be unique after normalization")
    schema = {
        "anonymous_unit_id": anonymous_unit_id,
        "short_name": "",
        "primary_explanation": "",
        "candidate_behavioral_explanation": "",
        "explanation_type": "unclear_or_mixed",
        "supporting_sample_ids": [],
        "outlier_sample_ids": [],
        "representative_evidence_ids": [],
        "linguistic_evidence": [
            {
                "sample_id": "",
                "evidence": "",
                "evidence_level": "semantic",
                "evidence_role": "support",
            }
        ],
        "alternative_explanations": [],
        "possible_confounds": [],
        "limitations": [],
        "confidence": 1,
        "confidence_rationale": "",
    }
    return f"""Analyze the following ten sentences as one set and produce a candidate interpretation of this anonymous text feature.

Anonymous feature ID:
{anonymous_unit_id}

Sentences:

{_format_samples(samples)}

Complete the following tasks:

1. Infer one primary candidate interpretation describing a stable pattern shared by a majority of the sentences.
2. Separately describe any candidate behavioral or discourse function.
3. If the evidence for a behavioral function is insufficient, state this explicitly instead of forcing one.
4. Cite specific sample IDs as linguistic evidence.
5. Identify sentences not adequately covered by the primary interpretation.
6. Provide 1 to 3 reasonable alternative interpretations.
7. Describe possible confounding factors.
8. Assign a confidence score from 1 to 5.

Analyze the sentence set as a whole. Do not produce a separate interpretation for every sentence. If no pattern covers at least 6 of the 10 sentences, use `unclear_or_mixed`.

`explanation_type` must be exactly one of:

- `behavioral_function`: a request, question, reflection, affirmation, advice, evaluation, or another communicative or discourse function.
- `linguistic_structure`: recurring lexical combinations, syntax, sentence forms, discourse markers, or expression templates.
- `affective_content`: emotion, attitude, evaluative direction, or affective intensity.
- `topic`: a recurring event, object, activity, experience, or domain.
- `surface_artifact`: incidental cues, data formats, annotation biases, punctuation, text length, transcription conventions, repeated templates, fixed wording, or processing artifacts.
- `unclear_or_mixed`: multiple patterns cannot be separated reliably, or no stable pattern covers a majority of the sentences.

Each `linguistic_evidence` item must contain `sample_id`, `evidence`, `evidence_level`, and `evidence_role`.
`evidence_level` must be exactly one of: `lexical`, `syntactic`, `discourse_marker`, `semantic`, `pragmatic_function`, `affective`, `topical`, `surface_form`.
`evidence_role` must be exactly one of: `support`, `limit`, `contradict`.
`representative_evidence_ids` should contain 2 to 3 sample IDs that best represent the primary interpretation.

Confidence calibration:
- `1`: No consistent pattern can be identified.
- `2`: Only a weak pattern exists, or several interpretations are equally plausible.
- `3`: The primary interpretation covers a majority of the sentences, but clear exceptions or confounds remain.
- `4`: The primary pattern is clear and consistent, while alternative interpretations are weaker.
- `5`: Coverage is very high, the pattern remains stable after sentence deduplication, and alternative interpretations and confounds are weak.

Return only one JSON object matching this structure:

{json.dumps(schema, ensure_ascii=False, indent=2)}

Field requirements:
- Use a short, specific, searchable `short_name`.
- `primary_explanation` must describe the most stable shared pattern and its boundary.
- If no stable behavioral function is supported, set `candidate_behavioral_explanation` to: "The available sentences do not support a stable behavioral-function interpretation."
- Every sample ID must appear exactly once in either `supporting_sample_ids` or `outlier_sample_ids`.
- `representative_evidence_ids` must be selected from `supporting_sample_ids`.
- Keep each prose field within two sentences.
"""


def _canonical_key(row: Any) -> str:
    representation = str(row.representation)
    label = str(row.label)
    index = int(row.index)
    sign = int(row.direction_sign)
    if representation == "SAE":
        return f"SAE|{label}|{index}"
    return f"PCA|{label}|{index}|{sign}"


def build_task5_deepseek_tasks(
    *,
    task5_root: str | Path,
    output_dir: str | Path,
    config: Task5DeepSeekCardConfig = Task5DeepSeekCardConfig(),
) -> dict[str, Any]:
    task5_root = Path(task5_root)
    output = Path(output_dir)
    task_dir = output / "llm_tasks"
    raw_dir = output / "raw_cards"
    task_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(task5_root / "private" / "blind_mapping_key.csv")
    inventory = pd.read_csv(task5_root / "private" / "example_inventory.csv")
    allowed_families = {str(value).upper() for value in config.representation_families}
    unknown_families = allowed_families - {"SAE", "PCA"}
    if unknown_families:
        raise ValueError(f"Unsupported representation families: {sorted(unknown_families)}")
    representation_family = mapping["representation"].astype(str).map(
        lambda value: "SAE" if value.startswith("SAE") else "PCA"
    )
    mapping = mapping.loc[representation_family.isin(allowed_families)].copy()
    if mapping.empty:
        raise ValueError("No Task 5 units remain after representation filtering")
    mapping["canonical_key"] = [_canonical_key(row) for row in mapping.itertuples(index=False)]

    tasks: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    pack_rows: list[dict[str, Any]] = []
    grouped = list(mapping.groupby("canonical_key", sort=True))
    for order, (canonical_key, group) in enumerate(grouped, start=1):
        group = group.sort_values(["arm", "blind_unit_id"])
        source = group.iloc[0]
        rows = inventory[
            inventory["arm"].eq(source["arm"])
            & inventory["blind_unit_id"].eq(source["blind_unit_id"])
            & inventory["phase"].eq("discovery")
        ].sort_values(["source_order", "row_idx"])
        if len(rows) != config.samples_per_unit:
            raise ValueError(f"{canonical_key} has {len(rows)} discovery rows")
        anonymous_id = f"T5C{order:03d}"
        samples = [
            {"sample_id": f"S{i:02d}", "text": str(row.unit_text)}
            for i, row in enumerate(rows.itertuples(index=False), start=1)
        ]
        task_id = f"{anonymous_id}_cluster_card"
        task = {
            "task_id": task_id,
            "task_type": "task5_discovery_cluster_card",
            "anonymous_unit_id": anonymous_id,
            "prompt": build_task5_cluster_prompt(anonymous_id, samples, config),
            "visible_samples": samples,
            "expected_output_path": str(raw_dir / f"{anonymous_id}.json"),
            "output_format": "single_json_object",
            "status": "pending_deepseek_api",
        }
        tasks.append(task)
        index_rows.append(
            {
                "anonymous_unit_id": anonymous_id,
                "canonical_key": canonical_key,
                "representation_internal": str(source["representation"]),
                "target_label_internal": str(source["label"]),
                "unit_index_internal": int(source["index"]),
                "direction_sign_internal": int(source["direction_sign"]),
                "arms": ",".join(sorted(group["arm"].astype(str).unique())),
                "blind_unit_ids": ",".join(sorted(group["blind_unit_id"].astype(str).unique())),
                "n_discovery_samples": len(samples),
            }
        )
        pack_rows.append({"anonymous_unit_id": anonymous_id, "samples": samples})

    tasks_path = task_dir / "task5_cluster_card_tasks.jsonl"
    index_path = output / "card_index_private.csv"
    packs_path = output / "discovery_sentence_packs.jsonl"
    write_jsonl(tasks_path, tasks)
    pd.DataFrame(index_rows).to_csv(index_path, index=False, encoding="utf-8-sig")
    write_jsonl(packs_path, pack_rows)
    manifest = {
        "analysis": "task5_deepseek_preliminary_cluster_cards",
        "status": "tasks_built_pending_api",
        "config": asdict(config),
        "n_source_unit_instances": int(len(mapping)),
        "n_unique_cluster_tasks": int(len(tasks)),
        "label_blind": True,
        "model_visible_fields": ["anonymous_unit_id", "sample_id", "text"],
        "model_hidden_fields": [
            "SAE/PCA identity",
            "MISC target label",
            "activation or PCA scores",
            "component direction",
            "matching scores",
            "held-out examples",
        ],
        "prompt_sha256": _sha256_text(tasks[0]["prompt"]) if tasks else "",
        "outputs": {"tasks": str(tasks_path), "index": str(index_path), "packs": str(packs_path)},
    }
    write_json(output / "task_build_manifest.json", manifest)
    return manifest


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return [str(item).strip() for item in value if str(item).strip()]


def validate_task5_deepseek_cards(
    *, tasks_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    output = Path(output_dir)
    tasks = read_jsonl(tasks_path)
    valid_cards: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for task in tasks:
        path = Path(task["expected_output_path"])
        reasons: list[str] = []
        card: dict[str, Any] | None = None
        if not path.exists():
            reasons.append("missing_output")
        else:
            try:
                payload = parse_llm_json_file(path)
                if not isinstance(payload, dict):
                    raise ValueError("output is not an object")
                missing = [field for field in TASK5_CARD_FIELDS if field not in payload]
                if missing:
                    raise ValueError(f"missing fields: {missing}")
                expected_id = str(task["anonymous_unit_id"])
                if str(payload["anonymous_unit_id"]) != expected_id:
                    reasons.append("anonymous_unit_id_mismatch")
                visible = {str(row["sample_id"]) for row in task["visible_samples"]}
                supporting = _string_list(payload["supporting_sample_ids"], "supporting_sample_ids")
                outliers = _string_list(payload["outlier_sample_ids"], "outlier_sample_ids")
                representatives = _string_list(payload["representative_evidence_ids"], "representative_evidence_ids")
                if set(supporting).intersection(outliers) or set(supporting).union(outliers) != visible:
                    reasons.append("invalid_sample_partition")
                if not 2 <= len(representatives) <= 3 or not set(representatives).issubset(supporting):
                    reasons.append("invalid_representative_evidence_ids")
                if str(payload["explanation_type"]) not in EXPLANATION_TYPES:
                    reasons.append("invalid_explanation_type")
                confidence = int(payload["confidence"])
                if isinstance(payload["confidence"], bool) or not 1 <= confidence <= 5:
                    reasons.append("invalid_confidence")
                evidence = payload["linguistic_evidence"]
                if not isinstance(evidence, list) or not evidence:
                    reasons.append("invalid_linguistic_evidence")
                else:
                    for item in evidence:
                        if not isinstance(item, dict):
                            reasons.append("invalid_linguistic_evidence_item")
                            continue
                        if str(item.get("sample_id")) not in visible:
                            reasons.append("unknown_evidence_sample_id")
                        if str(item.get("evidence_level")) not in EVIDENCE_LEVELS:
                            reasons.append("invalid_evidence_level")
                        if str(item.get("evidence_role")) not in EVIDENCE_ROLES:
                            reasons.append("invalid_evidence_role")
                card = dict(payload)
                card["task_id"] = task["task_id"]
                card["support_fraction"] = len(supporting) / max(len(visible), 1)
            except Exception as exc:
                reasons.append(f"parse_or_schema_error:{exc}")
        status = "valid" if not reasons else "invalid"
        audit_rows.append(
            {
                "task_id": task["task_id"],
                "anonymous_unit_id": task["anonymous_unit_id"],
                "status": status,
                "reasons": "|".join(sorted(set(reasons))),
                "output_path": str(path),
                "explanation_type": card.get("explanation_type") if card else "",
                "confidence": card.get("confidence") if card else "",
                "support_fraction": card.get("support_fraction") if card else "",
            }
        )
        if status == "valid" and card is not None:
            valid_cards.append(card)
    validated_path = output / "validated_cards.jsonl"
    audit_path = output / "card_structure_audit.csv"
    write_jsonl(validated_path, valid_cards)
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False, encoding="utf-8-sig")
    result = {
        "analysis": "task5_deepseek_preliminary_cluster_cards",
        "status": "PASS" if len(valid_cards) == len(tasks) else "INCOMPLETE",
        "n_tasks": len(tasks),
        "n_valid": len(valid_cards),
        "n_invalid": len(tasks) - len(valid_cards),
        "outputs": {"validated_cards": str(validated_path), "audit": str(audit_path)},
    }
    write_json(output / "validation_manifest.json", result)
    return result


def _md_text(value: Any) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|").strip()


def _md_list(values: Any, empty: str = "无") -> list[str]:
    if not isinstance(values, list) or not values:
        return [f"- {empty}"]
    return [f"- {_md_text(value)}" for value in values]


def render_task5_readable_card_document(
    *,
    output_dir: str | Path,
    document_path: str | Path | None = None,
) -> Path:
    """Render validated cards, source sentences, and Codex audit into one Markdown file."""

    output = Path(output_dir)
    destination = Path(document_path) if document_path else output / "task5_deepseek_cards_readable.md"
    index = pd.read_csv(output / "card_index_private.csv")
    review = pd.read_csv(output / "codex_card_quality_review.csv")
    cards = pd.DataFrame(read_jsonl(output / "validated_cards.jsonl"))
    packs = {
        str(row["anonymous_unit_id"]): list(row["samples"])
        for row in read_jsonl(output / "discovery_sentence_packs.jsonl")
    }
    merged = index.merge(cards, on="anonymous_unit_id", validate="one_to_one").merge(
        review, on="anonymous_unit_id", validate="one_to_one"
    )
    if len(merged) != len(index) or len(merged) != len(cards):
        raise ValueError("Readable card document inputs are not one-to-one")

    representation_order = pd.Categorical(
        merged["representation_family"], categories=["SAE", "PCA"], ordered=True
    )
    merged = merged.assign(_representation_order=representation_order).sort_values(
        ["_representation_order", "target_label_internal", "anonymous_unit_id"]
    )
    grade_counts = pd.crosstab(merged["representation_family"], merged["grade"])
    for grade in ("A", "B", "C"):
        if grade not in grade_counts.columns:
            grade_counts[grade] = 0

    lines = [
        "# Task 5 DeepSeek 初步解释卡片全集",
        "",
        "> 状态：`PRELIMINARY_LLM_REVIEW_COMPLETE_PENDING_HUMAN_REVIEW`",
        "",
        "本文档汇总 36 个匿名 discovery 语句簇的 DeepSeek 初步解释，并附上 Codex 对照原始语句后的内容审核。DeepSeek 未看到 SAE/PCA 身份、MISC 标签、方向、分数、匹配信息或 held-out 样例。",
        "",
        "这些结果只用于检查材料与自动解释质量，不能替代正式双审查者盲评，也不能单独证明 SAE 或 PCA 的解释性优势。",
        "",
        "## 阅读说明",
        "",
        "- `A`：主要模式真实覆盖多数语句，边界清楚，没有实质性外推。",
        "- `B`：解释有参考价值，但存在模式合并、边界偏宽或功能外推。",
        "- `C`：多数条件不成立、支持集合明显膨胀，或解释存在实质误导。",
        "- `support/outlier` 是 DeepSeek 自己的划分，不是人工真值。",
        "- 内部标签和表征仅在生成完成后合并到本文档，未进入模型提示词。",
        "",
        "## 总体统计",
        "",
        "| 表征 | A | B | C | 总计 |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in ("SAE", "PCA"):
        row = grade_counts.loc[family] if family in grade_counts.index else pd.Series(dtype=int)
        a, b, c = (int(row.get(grade, 0)) for grade in ("A", "B", "C"))
        lines.append(f"| {family} | {a} | {b} | {c} | {a + b + c} |")
    lines.extend(
        [
            "",
            "初步自动解释没有显示 SAE 明显优于 PCA。PCA 更常形成清楚的词法或句法模板；SAE 更常产生行为功能候选，但功能外推风险也更高。",
            "",
            "## 卡片索引",
            "",
            "| 卡片 | 表征 | 内部标签 | 单元 | DeepSeek 名称 | 类型 | 置信度 | Codex 评级 |",
            "|---|---|---|---:|---|---|---:|---|",
        ]
    )
    for row in merged.itertuples(index=False):
        lines.append(
            f"| [{row.anonymous_unit_id}](#{row.anonymous_unit_id}) | {row.representation_family} | "
            f"{row.target_label_internal} | {int(row.unit_index_internal)} | {_md_text(row.short_name)} | "
            f"{row.explanation_type} | {int(row.confidence)} | **{row.grade}** |"
        )

    current_family = ""
    current_label = ""
    for row in merged.itertuples(index=False):
        if row.representation_family != current_family:
            current_family = str(row.representation_family)
            current_label = ""
            lines.extend(["", f"## {current_family} 卡片", ""])
        if row.target_label_internal != current_label:
            current_label = str(row.target_label_internal)
            lines.extend([f"### 标签 {current_label}", ""])
        card_id = str(row.anonymous_unit_id)
        samples = packs.get(card_id, [])
        supporting = set(row.supporting_sample_ids if isinstance(row.supporting_sample_ids, list) else [])
        lines.extend(
            [
                f'<a id="{card_id}"></a>',
                f"#### {card_id} · {_md_text(row.short_name)}",
                "",
                f"- 表征：`{row.representation_family}`（原始记录：`{row.representation_internal}`）",
                f"- 内部标签：`{row.target_label_internal}`",
                f"- 单元索引：`{int(row.unit_index_internal)}`",
                f"- 所属方案：`{row.arms}`",
                f"- 解释类型：`{row.explanation_type}`",
                f"- DeepSeek 置信度：`{int(row.confidence)}/5`",
                f"- DeepSeek 自报支持比例：`{float(row.support_fraction):.0%}`",
                f"- Codex 评级：**{row.grade}**",
                f"- 模式支持判断：`{row.pattern_supported}`；边界质量：`{row.boundary_quality}`；外推风险：`{row.overclaim_risk}`",
                "",
                "**Codex 审核意见**",
                "",
                _md_text(row.review_note),
                "",
                "**DeepSeek 主要解释**",
                "",
                _md_text(row.primary_explanation),
                "",
                "**候选行为/话语功能解释**",
                "",
                _md_text(row.candidate_behavioral_explanation),
                "",
                "**原始 discovery 语句**",
                "",
                "| ID | DeepSeek 划分 | 文本 |",
                "|---|---|---|",
            ]
        )
        for sample in samples:
            sample_id = str(sample["sample_id"])
            role = "support" if sample_id in supporting else "outlier"
            lines.append(f"| {sample_id} | {role} | {_md_text(sample['text'])} |")
        representatives = row.representative_evidence_ids if isinstance(row.representative_evidence_ids, list) else []
        lines.extend(
            [
                "",
                f"**代表性证据 ID**：{', '.join(str(x) for x in representatives) or '无'}",
                "",
                "**语言学证据**",
                "",
                "| 样例 | 层级 | 角色 | 证据说明 |",
                "|---|---|---|---|",
            ]
        )
        evidence_items = row.linguistic_evidence if isinstance(row.linguistic_evidence, list) else []
        for item in evidence_items:
            lines.append(
                f"| {_md_text(item.get('sample_id', ''))} | {_md_text(item.get('evidence_level', ''))} | "
                f"{_md_text(item.get('evidence_role', ''))} | {_md_text(item.get('evidence', ''))} |"
            )
        if not evidence_items:
            lines.append("| - | - | - | 无 |")
        lines.extend(["", "**替代解释**", ""] + _md_list(row.alternative_explanations))
        lines.extend(["", "**可能混淆因素**", ""] + _md_list(row.possible_confounds))
        lines.extend(["", "**局限**", ""] + _md_list(row.limitations))
        lines.extend(["", "---", ""])

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return destination


def _render_task5_representation_human_review_document(
    *,
    output_dir: str | Path,
    representation_family: str,
    document_path: str | Path | None = None,
) -> Path:
    """Render one representation family with every discovery sentence for human review."""

    output = Path(output_dir)
    family = representation_family.upper()
    if family not in {"SAE", "PCA"}:
        raise ValueError(f"Unsupported representation family: {representation_family}")
    destination = (
        Path(document_path)
        if document_path
        else output / f"task5_{family.lower()}_latent_human_review.md"
    )
    index = pd.read_csv(output / "card_index_private.csv")
    cards = pd.DataFrame(read_jsonl(output / "validated_cards.jsonl"))
    packs = {
        str(row["anonymous_unit_id"]): list(row["samples"])
        for row in read_jsonl(output / "discovery_sentence_packs.jsonl")
    }
    merged = index.merge(cards, on="anonymous_unit_id", validate="one_to_one")
    merged = merged.loc[
        merged["representation_internal"].astype(str).str.startswith(family)
    ].sort_values(
        ["target_label_internal", "unit_index_internal"]
    )
    if merged.empty:
        raise ValueError(f"No {family} cards found for human review")

    lines = [
        f"# Task 5 {family} latent 归纳人工审核",
        "",
        f"> 共 {len(merged)} 个 {family} latent。每个 latent 均完整展示模型看到的 10 条去重句子。",
        "",
        "## 索引",
        "",
        "| Latent | 内部标签 | Card | 模型归纳名称 | 置信度 |",
        "|---:|---|---|---|---:|",
    ]
    for row in merged.itertuples(index=False):
        anchor = f"latent-{int(row.unit_index_internal)}"
        lines.append(
            f"| [{int(row.unit_index_internal)}](#{anchor}) | {row.target_label_internal} | "
            f"{row.anonymous_unit_id} | {_md_text(row.short_name)} | {int(row.confidence)}/5 |"
        )

    for row in merged.itertuples(index=False):
        latent_idx = int(row.unit_index_internal)
        card_id = str(row.anonymous_unit_id)
        samples = packs.get(card_id, [])
        if len(samples) != 10:
            raise ValueError(f"{card_id} has {len(samples)} discovery sentences, expected 10")
        supporting = set(
            row.supporting_sample_ids
            if isinstance(row.supporting_sample_ids, list)
            else []
        )
        lines.extend(
            [
                "",
                f'<a id="latent-{latent_idx}"></a>',
                f"## Latent {latent_idx}",
                "",
                f"- 内部标签：`{row.target_label_internal}`",
                f"- Card ID：`{card_id}`",
                f"- 模型归纳名称：`{_md_text(row.short_name)}`",
                f"- 解释类型：`{row.explanation_type}`",
                f"- 模型置信度：`{int(row.confidence)}/5`",
                f"- 模型自报支持比例：`{float(row.support_fraction):.0%}`",
                "",
                "### 输入模型的完整句子簇",
                "",
                "| ID | 模型划分 | 句子内容 |",
                "|---|---|---|",
            ]
        )
        for sample in samples:
            sample_id = str(sample["sample_id"])
            role = "support" if sample_id in supporting else "outlier"
            lines.append(f"| {sample_id} | {role} | {_md_text(sample['text'])} |")
        lines.extend(
            [
                "",
                "### DeepSeek 归纳结果",
                "",
                "**主要模式**",
                "",
                _md_text(row.primary_explanation),
                "",
                "**候选行为或话语功能**",
                "",
                _md_text(row.candidate_behavioral_explanation),
                "",
                "**替代解释**",
                "",
                *_md_list(row.alternative_explanations),
                "",
                "**可能混淆因素**",
                "",
                *_md_list(row.possible_confounds),
                "",
                "### 人工审核",
                "",
                "- 多数句子是否支持主要模式：`待审核`",
                "- 归纳是否准确：`待审核`",
                "- 是否存在过度外推：`待审核`",
                "- 人工备注：",
                "",
                "---",
            ]
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return destination


def render_task5_sae_human_review_document(
    *,
    output_dir: str | Path,
    document_path: str | Path | None = None,
) -> Path:
    return _render_task5_representation_human_review_document(
        output_dir=output_dir,
        representation_family="SAE",
        document_path=document_path,
    )


def render_task5_pca_human_review_document(
    *,
    output_dir: str | Path,
    document_path: str | Path | None = None,
) -> Path:
    return _render_task5_representation_human_review_document(
        output_dir=output_dir,
        representation_family="PCA",
        document_path=document_path,
    )


__all__ = [
    "TASK5_CARD_SYSTEM_PROMPT",
    "Task5DeepSeekCardConfig",
    "build_task5_cluster_prompt",
    "build_task5_deepseek_tasks",
    "render_task5_pca_human_review_document",
    "render_task5_readable_card_document",
    "render_task5_sae_human_review_document",
    "validate_task5_deepseek_cards",
]
