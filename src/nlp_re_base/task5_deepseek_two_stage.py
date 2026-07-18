"""Two-stage DeepSeek generation and within-cluster scoring for Task 5."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .contrastive_evidence_pack import read_jsonl, write_json, write_jsonl
from .contrastive_llm_io import parse_llm_json_file
from .deepseek_latent_cards import EVIDENCE_LEVELS, EVIDENCE_ROLES, EXPLANATION_TYPES
from .deepseek_top50_induction import _sha256_text


GENERATION_SYSTEM_PROMPT = """你是一名文本模式归纳分析员。

请根据一个句子簇归纳候选解释。所有结论必须仅建立在句子中可以直接观察到的语言证据上。除非句子明确提供相关信息，否则不得推断外部场景、说话者身份、专业领域或对话背景。

你的任务是生成解释，而不是评价解释质量。不要输出置信度、质量分数、等级、概率或通过/不通过判断。

仅返回指定的 JSON 对象。"""


SCORING_SYSTEM_PROMPT = """你是一名独立且保守的句子簇解释质量评价员。

你的唯一核心任务是判断：一个已经冻结的候选解释，能否准确、具体且有证据地概括给定句子簇内部的共同模式。

本次评价不考察该解释在完整数据集中的选择特异性、外部泛化能力或因果功能。不得利用这些无法由当前句子簇检验的性质提高或降低分数。

不得改写、完善、扩展或替换冻结解释。不得因为该解释由另一个模型生成，就预设它是正确的。

在给出维度分数之前，必须逐条独立判断所有句子。所有判断只能建立在给定句子中可以直接观察到的证据上。

仅返回指定的 JSON 对象。"""


GENERATION_FIELDS = (
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
)

SCORING_FIELDS = (
    "anonymous_unit_id",
    "evaluation_scope",
    "sample_judgments",
    "within_cluster_fit_summary",
    "unsupported_within_cluster_claims",
    "within_cluster_coherence",
    "claim_specificity",
    "evidence_grounding",
    "score_rationales",
)

SCORING_VERDICTS = {"supports", "partial", "contradicts", "not_relevant"}


@dataclass(frozen=True)
class Task5TwoStageConfig:
    samples_per_unit: int = 10
    min_representative_ids: int = 2
    max_representative_ids: int = 3
    representation_families: tuple[str, ...] = ("SAE", "PCA")


def _format_samples(samples: list[dict[str, str]]) -> str:
    return "\n".join(
        f"- sample_id={sample['sample_id']}\n  句子：{sample['text'].replace(chr(10), ' ').strip()}"
        for sample in samples
    )


def _canonical_key(row: Any) -> str:
    representation = str(row.representation)
    if representation.startswith("SAE"):
        return f"SAE|{row.label}|{int(row.index)}"
    return f"PCA|{row.label}|{int(row.index)}|{int(row.direction_sign)}"


def build_generation_prompt(anonymous_unit_id: str, samples: list[dict[str, str]]) -> str:
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
    }
    return f"""请将以下 10 条去重后的句子作为一个整体进行分析，并生成该匿名文本特征的候选解释。

匿名特征 ID：
{anonymous_unit_id}

句子：

{_format_samples(samples)}

请完成以下任务：

1. 找出至少被多数句子共同支持的最稳定模式。
2. 为该模式提供一个简短、具体、便于检索的名称。
3. 说明该模式的可观察边界，即哪些特征是共同的，哪些特征并非必要条件。
4. 只有当句子措辞直接支持时，才单独提出可能的行为功能或话语功能解释。
5. 区分支持主要解释的样本，以及主要解释不能充分覆盖的样本。
6. 从支持样本中选出 2 至 3 条最具代表性的样本。
7. 引用具体样本 ID，给出词汇、句法、语义、情感、主题或话语层面的证据。
8. 给出 1 至 3 个合理的替代解释。
9. 记录可能的混淆因素和当前证据的局限。

必须将句子簇作为一个整体进行分析，不要为每条句子分别生成一个独立解释。

一个多数模式必须覆盖至少 10 条句子中的 6 条。如果没有任何模式达到该标准，请将 `explanation_type` 设置为 `unclear_or_mixed`，并描述可能并存的模式，不要强行生成统一解释。

`explanation_type` 必须是以下值之一：
- `behavioral_function`
- `linguistic_structure`
- `affective_content`
- `topic`
- `surface_artifact`
- `unclear_or_mixed`

`evidence_level` 必须是以下值之一：
- `lexical`
- `syntactic`
- `discourse_marker`
- `semantic`
- `pragmatic_function`
- `affective`
- `topical`
- `surface_form`

`evidence_role` 必须是以下值之一：`support`、`limit`、`contradict`。

仅返回一个符合以下结构的 JSON 对象：

{json.dumps(schema, ensure_ascii=False, indent=2)}

字段要求：
- `short_name` 应当简短、具体并便于检索。
- `primary_explanation` 必须描述可观察的语言模式及其边界，不能依赖假设的数据来源、专业场景或说话者身份。
- 如果没有足够证据支持稳定的行为功能，请将 `candidate_behavioral_explanation` 写为：“现有句子不足以支持稳定的行为功能解释。”
- 每个样本 ID 必须且只能出现在 `supporting_sample_ids` 或 `outlier_sample_ids` 中的一处。
- `representative_evidence_ids` 必须包含 2 至 3 个来自 `supporting_sample_ids` 的样本 ID。
- 每个文字字段最多使用两句话。
- 不得输出置信度、质量分数、概率、等级或其他评价字段。"""


def build_scoring_prompt(
    anonymous_unit_id: str,
    samples: list[dict[str, str]],
    explanation: dict[str, Any],
) -> str:
    frozen = {
        "short_name": str(explanation["short_name"]),
        "primary_explanation": str(explanation["primary_explanation"]),
        "candidate_behavioral_explanation": str(explanation["candidate_behavioral_explanation"]),
        "explanation_type": str(explanation["explanation_type"]),
    }
    schema = {
        "anonymous_unit_id": anonymous_unit_id,
        "evaluation_scope": "within_cluster_explanation_quality_only",
        "sample_judgments": [
            {"sample_id": "", "verdict": "supports", "evidence": ""}
        ],
        "within_cluster_fit_summary": "",
        "unsupported_within_cluster_claims": [],
        "within_cluster_coherence": 1,
        "claim_specificity": 1,
        "evidence_grounding": 1,
        "score_rationales": {
            "within_cluster_coherence": "",
            "claim_specificity": "",
            "evidence_grounding": "",
        },
    }
    return f"""请根据以下 10 条句子，评价已经冻结的候选解释。

匿名特征 ID：
{anonymous_unit_id}

冻结解释：

{json.dumps(frozen, ensure_ascii=False, indent=2)}

句子：

{_format_samples(samples)}

本次评价的唯一核心任务是衡量句子簇内部的解释质量，即判断冻结解释是否准确概括了这 10 条句子中实际存在的共同模式。

不要评价该解释在完整数据集中的选择特异性、对其他句子的泛化能力或任何因果功能。这些性质不属于本次评分范围。

请完成以下任务：

1. 独立判断每条句子与冻结解释之间的关系。每条句子必须且只能使用以下一种判断：
   - `supports`：该句子清楚地体现了冻结后的主要解释。
   - `partial`：该句子只体现了主要模式的一部分，不能完整支持该解释。
   - `contradicts`：该句子提供了与主要解释相冲突的证据。
   - `not_relevant`：该句子既不能支持该解释，也不能直接反驳该解释。
2. 对每条判断引用具体的语言线索作为依据。
3. 找出冻结解释中没有得到当前句子簇支持的所有表述。
4. 简要总结该解释在句子簇内部准确捕捉了什么，以及遗漏或错误概括了什么。
5. 按照以下标准分别给出 1 至 5 分。

`within_cluster_coherence`：簇内模式一致性
- 1 分：解释不能识别句子簇中的一致模式，或者只适用于极少数句子。
- 2 分：解释只覆盖较弱或彼此不一致的局部现象。
- 3 分：解释基本覆盖多数句子，但存在明显例外或多个竞争模式。
- 4 分：解释清楚地覆盖大多数句子，主要模式在句子之间较为一致。
- 5 分：解释以同一种明确模式覆盖几乎全部句子，且没有重要的簇内冲突。

`claim_specificity`：解释具体性
- 1 分：解释模糊、循环论证，或者没有指出当前句子簇的具体共同点。
- 2 分：识别出一个宽泛模式，但无法说明当前句子为什么属于同一模式。
- 3 分：解释可以根据当前句子进行检验，但仍然较宽泛或定义不足。
- 4 分：解释具体地描述了当前句子簇的主要共同点，适用边界基本清楚。
- 5 分：解释精确描述了当前句子簇的共同模式，并区分了核心特征与簇内偶然差异。

`evidence_grounding`：证据基础
- 1 分：主要解释与当前句子的实际措辞缺乏对应关系。
- 2 分：只有少量结论能够在当前句子中找到直接依据。
- 3 分：核心模式得到部分文本证据支持，但仍混有未经句子支持的推断。
- 4 分：主要结论基本建立在当前句子中可观察的语言证据上。
- 5 分：主要解释的所有核心内容都能由当前句子中的具体线索直接支持。

不要主观给出覆盖率分数或总体分数。程序将根据逐句判断计算覆盖率和最终分数。

仅返回一个符合以下结构的 JSON 对象：

{json.dumps(schema, ensure_ascii=False, indent=2)}

字段要求：
- 必须为每个给定样本 ID 输出且只输出一条判断。
- 不得将冻结解释本身的表述作为支持证据。
- 不得改写解释，也不得提出更好的解释版本。
- 外部场景、专业领域、说话者身份和因果功能只有在当前句子明确提供证据时，才能被视为簇内解释的一部分。
- 不得因为无法评价完整数据集选择性、外部泛化或因果功能而扣分；这些内容不属于本次评价范围。
- 只有解释准确覆盖几乎全部句子、模式高度一致且核心结论均有直接文本证据时，才能给出 5 分。"""


def build_generation_tasks(
    *,
    task5_root: str | Path,
    output_dir: str | Path,
    config: Task5TwoStageConfig = Task5TwoStageConfig(),
) -> dict[str, Any]:
    task5_root = Path(task5_root)
    output = Path(output_dir)
    task_dir = output / "llm_tasks"
    raw_dir = output / "generation_raw"
    task_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(task5_root / "private" / "blind_mapping_key.csv")
    inventory = pd.read_csv(task5_root / "private" / "example_inventory.csv")
    allowed = {str(value).upper() for value in config.representation_families}
    if allowed - {"SAE", "PCA"}:
        raise ValueError(f"Unsupported representation families: {sorted(allowed)}")
    mapping["representation_family"] = mapping["representation"].astype(str).map(
        lambda value: "SAE" if value.startswith("SAE") else "PCA"
    )
    mapping = mapping[mapping["representation_family"].isin(allowed)].copy()
    mapping["canonical_key"] = [_canonical_key(row) for row in mapping.itertuples(index=False)]
    tasks: list[dict[str, Any]] = []
    indexes: list[dict[str, Any]] = []
    packs: list[dict[str, Any]] = []
    for order, (canonical_key, group) in enumerate(mapping.groupby("canonical_key", sort=True), start=1):
        group = group.sort_values(["arm", "blind_unit_id"])
        source = group.iloc[0]
        rows = inventory[
            inventory["arm"].eq(source["arm"])
            & inventory["blind_unit_id"].eq(source["blind_unit_id"])
            & inventory["phase"].eq("discovery")
        ].sort_values(["source_order", "row_idx"])
        if len(rows) != config.samples_per_unit:
            raise ValueError(f"{canonical_key} has {len(rows)} discovery rows")
        anonymous_id = f"T5TS{order:03d}"
        samples = [
            {"sample_id": f"S{i:02d}", "text": str(row.unit_text)}
            for i, row in enumerate(rows.itertuples(index=False), start=1)
        ]
        ids = [sample["sample_id"] for sample in samples]
        texts = [" ".join(sample["text"].lower().split()) for sample in samples]
        if len(set(ids)) != len(ids) or len(set(texts)) != len(texts):
            raise ValueError(f"{canonical_key} discovery samples are not unique")
        task_id = f"{anonymous_id}_generation"
        tasks.append(
            {
                "task_id": task_id,
                "task_type": "task5_two_stage_explanation_generation",
                "anonymous_unit_id": anonymous_id,
                "prompt": build_generation_prompt(anonymous_id, samples),
                "visible_samples": samples,
                "expected_output_path": str(raw_dir / f"{anonymous_id}.json"),
                "output_format": "single_json_object",
                "status": "pending_deepseek_api",
            }
        )
        indexes.append(
            {
                "anonymous_unit_id": anonymous_id,
                "canonical_key": canonical_key,
                "representation_internal": str(source["representation"]),
                "representation_family": str(source["representation_family"]),
                "target_label_internal": str(source["label"]),
                "unit_index_internal": int(source["index"]),
                "direction_sign_internal": int(source["direction_sign"]),
                "arms": ",".join(sorted(group["arm"].astype(str).unique())),
                "n_discovery_samples": len(samples),
            }
        )
        packs.append({"anonymous_unit_id": anonymous_id, "samples": samples})
    tasks_path = task_dir / "generation_tasks.jsonl"
    index_path = output / "card_index_private.csv"
    packs_path = output / "sentence_packs.jsonl"
    write_jsonl(tasks_path, tasks)
    pd.DataFrame(indexes).to_csv(index_path, index=False, encoding="utf-8-sig")
    write_jsonl(packs_path, packs)
    result = {
        "analysis": "task5_deepseek_two_stage",
        "step": "build-generation-tasks",
        "status": "tasks_built_pending_api",
        "config": asdict(config),
        "n_tasks": len(tasks),
        "generation_has_scores": False,
        "label_blind": True,
        "model_visible_fields": ["anonymous_unit_id", "sample_id", "text"],
        "prompt_sha256": _sha256_text(tasks[0]["prompt"]) if tasks else "",
        "outputs": {"tasks": str(tasks_path), "index": str(index_path), "packs": str(packs_path)},
    }
    write_json(output / "generation_task_manifest.json", result)
    return result


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return [str(item).strip() for item in value if str(item).strip()]


def validate_generation_outputs(*, tasks_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    tasks = read_jsonl(tasks_path)
    valid: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for task in tasks:
        path = Path(task["expected_output_path"])
        reasons: list[str] = []
        normalizations: list[str] = []
        payload: dict[str, Any] | None = None
        try:
            parsed = parse_llm_json_file(path)
            if not isinstance(parsed, dict):
                raise ValueError("output is not an object")
            payload = dict(parsed)
            missing = [field for field in GENERATION_FIELDS if field not in payload]
            if missing:
                reasons.append(f"missing_fields:{','.join(missing)}")
            forbidden = {"confidence", "confidence_rationale", "quality_score", "overall_score"}
            if forbidden.intersection(payload):
                reasons.append("generation_contains_score_fields")
            if str(payload.get("anonymous_unit_id")) != str(task["anonymous_unit_id"]):
                reasons.append("anonymous_unit_id_mismatch")
            visible = {str(row["sample_id"]) for row in task["visible_samples"]}
            supporting = _string_list(payload.get("supporting_sample_ids"), "supporting_sample_ids")
            outliers = _string_list(payload.get("outlier_sample_ids"), "outlier_sample_ids")
            representatives = _string_list(
                payload.get("representative_evidence_ids"), "representative_evidence_ids"
            )
            if set(supporting).intersection(outliers) or set(supporting).union(outliers) != visible:
                reasons.append("invalid_sample_partition")
            if not 2 <= len(representatives) <= 3 or not set(representatives).issubset(supporting):
                reasons.append("invalid_representative_evidence_ids")
            if str(payload.get("explanation_type")) == "pragmatic_function":
                payload["explanation_type"] = "behavioral_function"
                normalizations.append("explanation_type:pragmatic_function->behavioral_function")
            if str(payload.get("explanation_type")) not in EXPLANATION_TYPES:
                reasons.append("invalid_explanation_type")
            evidence = payload.get("linguistic_evidence")
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
            if payload is not None:
                payload["task_id"] = str(task["task_id"])
        except Exception as exc:
            reasons.append(f"parse_or_schema_error:{exc}")
        status = "valid" if not reasons else "invalid"
        audit.append(
            {
                "task_id": task["task_id"],
                "anonymous_unit_id": task["anonymous_unit_id"],
                "status": status,
                "reasons": "|".join(sorted(set(reasons))),
                "normalizations": "|".join(normalizations),
                "output_path": str(path),
            }
        )
        if status == "valid" and payload is not None:
            valid.append(payload)
    write_jsonl(output / "generation_validated.jsonl", valid)
    pd.DataFrame(audit).to_csv(output / "generation_structure_audit.csv", index=False, encoding="utf-8-sig")
    result = {
        "analysis": "task5_deepseek_two_stage",
        "step": "validate-generation",
        "status": "PASS" if len(valid) == len(tasks) else "INCOMPLETE",
        "n_tasks": len(tasks),
        "n_valid": len(valid),
        "n_invalid": len(tasks) - len(valid),
    }
    write_json(output / "generation_validation_manifest.json", result)
    return result


def build_scoring_tasks(*, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    explanations = {str(row["anonymous_unit_id"]): row for row in read_jsonl(output / "generation_validated.jsonl")}
    packs = {str(row["anonymous_unit_id"]): row["samples"] for row in read_jsonl(output / "sentence_packs.jsonl")}
    if set(explanations) != set(packs):
        raise ValueError("Validated explanations and sentence packs do not match one-to-one")
    raw_dir = output / "scoring_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    for anonymous_id in sorted(explanations):
        explanation = explanations[anonymous_id]
        samples = list(packs[anonymous_id])
        task_id = f"{anonymous_id}_scoring"
        tasks.append(
            {
                "task_id": task_id,
                "task_type": "task5_two_stage_within_cluster_scoring",
                "anonymous_unit_id": anonymous_id,
                "prompt": build_scoring_prompt(anonymous_id, samples, explanation),
                "visible_samples": samples,
                "frozen_explanation_fields": {
                    key: explanation[key]
                    for key in (
                        "short_name",
                        "primary_explanation",
                        "candidate_behavioral_explanation",
                        "explanation_type",
                    )
                },
                "expected_output_path": str(raw_dir / f"{anonymous_id}.json"),
                "output_format": "single_json_object",
                "status": "pending_deepseek_api",
            }
        )
    tasks_path = output / "llm_tasks" / "scoring_tasks.jsonl"
    write_jsonl(tasks_path, tasks)
    result = {
        "analysis": "task5_deepseek_two_stage",
        "step": "build-scoring-tasks",
        "status": "tasks_built_pending_api",
        "n_tasks": len(tasks),
        "evaluation_scope": "within_cluster_explanation_quality_only",
        "excluded_generation_fields": [
            "supporting_sample_ids",
            "outlier_sample_ids",
            "representative_evidence_ids",
            "linguistic_evidence",
            "alternative_explanations",
            "possible_confounds",
            "limitations",
        ],
        "prompt_sha256": _sha256_text(tasks[0]["prompt"]) if tasks else "",
        "outputs": {"tasks": str(tasks_path)},
    }
    write_json(output / "scoring_task_manifest.json", result)
    return result


def _score_1_to_5(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} cannot be bool")
    score = int(value)
    if not 1 <= score <= 5:
        raise ValueError(f"{field} must be in [1, 5]")
    return score


def _coverage_score(equivalent_coverage: float) -> int:
    if equivalent_coverage <= 2.0:
        return 1
    if equivalent_coverage <= 4.0:
        return 2
    if equivalent_coverage <= 6.0:
        return 3
    if equivalent_coverage <= 8.0:
        return 4
    return 5


def validate_scoring_outputs(*, tasks_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    tasks = read_jsonl(tasks_path)
    valid: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for task in tasks:
        path = Path(task["expected_output_path"])
        reasons: list[str] = []
        payload: dict[str, Any] | None = None
        try:
            parsed = parse_llm_json_file(path)
            if not isinstance(parsed, dict):
                raise ValueError("output is not an object")
            payload = dict(parsed)
            missing = [field for field in SCORING_FIELDS if field not in payload]
            if missing:
                reasons.append(f"missing_fields:{','.join(missing)}")
            if str(payload.get("anonymous_unit_id")) != str(task["anonymous_unit_id"]):
                reasons.append("anonymous_unit_id_mismatch")
            if payload.get("evaluation_scope") != "within_cluster_explanation_quality_only":
                reasons.append("invalid_evaluation_scope")
            visible = {str(row["sample_id"]) for row in task["visible_samples"]}
            judgments = payload.get("sample_judgments")
            if not isinstance(judgments, list):
                raise ValueError("sample_judgments must be a list")
            judged_ids = [str(item.get("sample_id")) for item in judgments if isinstance(item, dict)]
            if len(judged_ids) != len(visible) or set(judged_ids) != visible or len(set(judged_ids)) != len(judged_ids):
                reasons.append("invalid_sample_judgment_coverage")
            verdict_counts = {verdict: 0 for verdict in sorted(SCORING_VERDICTS)}
            for item in judgments:
                if not isinstance(item, dict):
                    reasons.append("invalid_sample_judgment_item")
                    continue
                verdict = str(item.get("verdict"))
                if verdict not in SCORING_VERDICTS:
                    reasons.append("invalid_sample_verdict")
                else:
                    verdict_counts[verdict] += 1
                if not str(item.get("evidence", "")).strip():
                    reasons.append("missing_sample_evidence")
            coherence = _score_1_to_5(payload.get("within_cluster_coherence"), "within_cluster_coherence")
            specificity = _score_1_to_5(payload.get("claim_specificity"), "claim_specificity")
            grounding = _score_1_to_5(payload.get("evidence_grounding"), "evidence_grounding")
            rationales = payload.get("score_rationales")
            required_rationales = {"within_cluster_coherence", "claim_specificity", "evidence_grounding"}
            if not isinstance(rationales, dict) or any(
                not str(rationales.get(key, "")).strip() for key in required_rationales
            ):
                reasons.append("invalid_score_rationales")
            unsupported = payload.get("unsupported_within_cluster_claims")
            if not isinstance(unsupported, list):
                reasons.append("invalid_unsupported_claims")
            equivalent = verdict_counts["supports"] + 0.5 * verdict_counts["partial"]
            coverage = _coverage_score(equivalent)
            overall = min(coverage, coherence, specificity, grounding)
            payload.update(
                {
                    "task_id": str(task["task_id"]),
                    "supports_count": verdict_counts["supports"],
                    "partial_count": verdict_counts["partial"],
                    "contradicts_count": verdict_counts["contradicts"],
                    "not_relevant_count": verdict_counts["not_relevant"],
                    "equivalent_coverage": equivalent,
                    "coverage_score": coverage,
                    "overall_score": overall,
                }
            )
        except Exception as exc:
            reasons.append(f"parse_or_schema_error:{exc}")
        status = "valid" if not reasons else "invalid"
        audit.append(
            {
                "task_id": task["task_id"],
                "anonymous_unit_id": task["anonymous_unit_id"],
                "status": status,
                "reasons": "|".join(sorted(set(reasons))),
                "output_path": str(path),
                "overall_score": payload.get("overall_score", "") if payload else "",
            }
        )
        if status == "valid" and payload is not None:
            valid.append(payload)
    write_jsonl(output / "scoring_validated.jsonl", valid)
    pd.DataFrame(audit).to_csv(output / "scoring_structure_audit.csv", index=False, encoding="utf-8-sig")
    result = {
        "analysis": "task5_deepseek_two_stage",
        "step": "validate-scoring",
        "status": "PASS" if len(valid) == len(tasks) else "INCOMPLETE",
        "n_tasks": len(tasks),
        "n_valid": len(valid),
        "n_invalid": len(tasks) - len(valid),
    }
    write_json(output / "scoring_validation_manifest.json", result)
    if result["status"] == "PASS":
        _merge_final_results(output)
    return result


def _merge_final_results(output: Path) -> None:
    index = pd.read_csv(output / "card_index_private.csv")
    generation_rows = read_jsonl(output / "generation_validated.jsonl")
    scoring_rows = read_jsonl(output / "scoring_validated.jsonl")
    generation = {str(row["anonymous_unit_id"]): row for row in generation_rows}
    scoring = {str(row["anonymous_unit_id"]): row for row in scoring_rows}
    if set(generation) != set(scoring) or set(generation) != set(index["anonymous_unit_id"].astype(str)):
        raise ValueError("Index, generation, and scoring outputs do not match one-to-one")
    final: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    index_lookup = index.set_index("anonymous_unit_id").to_dict(orient="index")
    for anonymous_id in sorted(generation):
        metadata = index_lookup[anonymous_id]
        row = {
            "anonymous_unit_id": anonymous_id,
            **metadata,
            "generation": generation[anonymous_id],
            "scoring": scoring[anonymous_id],
        }
        final.append(row)
        score_rows.append(
            {
                "anonymous_unit_id": anonymous_id,
                **metadata,
                "short_name": generation[anonymous_id]["short_name"],
                "explanation_type": generation[anonymous_id]["explanation_type"],
                "coverage_score": scoring[anonymous_id]["coverage_score"],
                "within_cluster_coherence": scoring[anonymous_id]["within_cluster_coherence"],
                "claim_specificity": scoring[anonymous_id]["claim_specificity"],
                "evidence_grounding": scoring[anonymous_id]["evidence_grounding"],
                "overall_score": scoring[anonymous_id]["overall_score"],
                "equivalent_coverage": scoring[anonymous_id]["equivalent_coverage"],
            }
        )
    write_jsonl(output / "final_cards.jsonl", final)
    pd.DataFrame(score_rows).to_csv(output / "final_scores.csv", index=False, encoding="utf-8-sig")


def _md(value: Any) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|").strip()


def render_two_stage_report(*, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    scores = pd.read_csv(output / "final_scores.csv")
    final = {str(row["anonymous_unit_id"]): row for row in read_jsonl(output / "final_cards.jsonl")}
    packs = {str(row["anonymous_unit_id"]): row["samples"] for row in read_jsonl(output / "sentence_packs.jsonl")}
    summary = (
        scores.groupby("representation_family", as_index=False)
        .agg(
            n_units=("anonymous_unit_id", "size"),
            overall_mean=("overall_score", "mean"),
            coverage_mean=("coverage_score", "mean"),
            coherence_mean=("within_cluster_coherence", "mean"),
            specificity_mean=("claim_specificity", "mean"),
            grounding_mean=("evidence_grounding", "mean"),
        )
        .sort_values("representation_family")
    )
    lines = [
        "# Task 5 DeepSeek 两阶段句子簇解释与评分报告",
        "",
        "> 评价范围：`within_cluster_explanation_quality_only`。本报告不评价完整数据集选择性、外部泛化或因果功能。",
        "",
        "## 汇总",
        "",
        "| 表征 | 单元数 | 总分均值 | 覆盖 | 簇内一致性 | 具体性 | 证据基础 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.representation_family} | {int(row.n_units)} | {row.overall_mean:.2f} | "
            f"{row.coverage_mean:.2f} | {row.coherence_mean:.2f} | {row.specificity_mean:.2f} | {row.grounding_mean:.2f} |"
        )
    lines.extend(["", "## 单元结果", ""])
    ordered = scores.sort_values(["representation_family", "target_label_internal", "unit_index_internal"])
    for score in ordered.itertuples(index=False):
        anonymous_id = str(score.anonymous_unit_id)
        row = final[anonymous_id]
        generation = row["generation"]
        scoring = row["scoring"]
        lines.extend(
            [
                f"### {anonymous_id} · {score.representation_family} · {score.target_label_internal} · 单元 {int(score.unit_index_internal)}",
                "",
                f"- 名称：{_md(generation['short_name'])}",
                f"- 类型：`{generation['explanation_type']}`",
                f"- 程序总分：**{int(scoring['overall_score'])}/5**",
                f"- 覆盖 / 一致性 / 具体性 / 证据：`{int(scoring['coverage_score'])}` / `{int(scoring['within_cluster_coherence'])}` / `{int(scoring['claim_specificity'])}` / `{int(scoring['evidence_grounding'])}`",
                "",
                "**冻结解释**",
                "",
                _md(generation["primary_explanation"]),
                "",
                "**候选行为或话语功能**",
                "",
                _md(generation["candidate_behavioral_explanation"]),
                "",
                "**独立评分摘要**",
                "",
                _md(scoring["within_cluster_fit_summary"]),
                "",
                "**句子与独立判断**",
                "",
                "| ID | 判断 | 句子 | 评分证据 |",
                "|---|---|---|---|",
            ]
        )
        judgment_map = {str(item["sample_id"]): item for item in scoring["sample_judgments"]}
        for sample in packs[anonymous_id]:
            judgment = judgment_map[str(sample["sample_id"])]
            lines.append(
                f"| {sample['sample_id']} | {judgment['verdict']} | {_md(sample['text'])} | {_md(judgment['evidence'])} |"
            )
        unsupported = scoring.get("unsupported_within_cluster_claims") or []
        lines.extend(["", "**当前句子簇不支持的解释表述**", ""])
        lines.extend([f"- {_md(item)}" for item in unsupported] or ["- 无"])
        lines.extend(["", "---", ""])
    path = output / "task5_two_stage_evaluation_report.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def render_two_stage_family_review_document(
    *, output_dir: str | Path, representation_family: str
) -> Path:
    """Render one complete, human-readable review document for PCA or SAE cards."""
    output = Path(output_dir)
    family = str(representation_family).upper()
    if family not in {"PCA", "SAE"}:
        raise ValueError(f"Unsupported representation family: {representation_family}")
    scores = pd.read_csv(output / "final_scores.csv")
    scores = scores[scores["representation_family"].astype(str).eq(family)].sort_values(
        ["target_label_internal", "unit_index_internal", "anonymous_unit_id"]
    )
    if scores.empty:
        raise ValueError(f"No {family} cards are available")
    final = {
        str(row["anonymous_unit_id"]): row
        for row in read_jsonl(output / "final_cards.jsonl")
        if str(row["representation_family"]).upper() == family
    }
    packs = {
        str(row["anonymous_unit_id"]): row["samples"]
        for row in read_jsonl(output / "sentence_packs.jsonl")
    }
    lines = [
        f"# Task 5 {family} 两阶段解释 Card 人工审查文档",
        "",
        f"> 共 `{len(scores)}` 个 {family} 单元。评价范围仅限句子簇内部解释质量，不涉及完整数据集选择性、外部泛化或因果功能。",
        "",
        "## 审查索引",
        "",
        "| Card | 内部标签 | 单元 | 候选名称 | 类型 | 总分 | 覆盖 | 一致性 | 具体性 | 证据 |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for score in scores.itertuples(index=False):
        anchor = f"card-{str(score.anonymous_unit_id).lower()}"
        lines.append(
            f"| [{score.anonymous_unit_id}](#{anchor}) | {score.target_label_internal} | "
            f"{int(score.unit_index_internal)} | {_md(score.short_name)} | `{score.explanation_type}` | "
            f"{int(score.overall_score)} | {int(score.coverage_score)} | "
            f"{int(score.within_cluster_coherence)} | {int(score.claim_specificity)} | "
            f"{int(score.evidence_grounding)} |"
        )
    for score in scores.itertuples(index=False):
        anonymous_id = str(score.anonymous_unit_id)
        row = final[anonymous_id]
        generation = row["generation"]
        scoring = row["scoring"]
        judgments = {str(item["sample_id"]): item for item in scoring["sample_judgments"]}
        lines.extend(
            [
                "",
                f'<a id="card-{anonymous_id.lower()}"></a>',
                f"## {anonymous_id} · {score.target_label_internal} · 单元 {int(score.unit_index_internal)}",
                "",
                "### Card 信息",
                "",
                f"- 表征：`{family}`",
                f"- 内部标签：`{score.target_label_internal}`",
                f"- 单元索引：`{int(score.unit_index_internal)}`",
                f"- 方向符号：`{int(row['direction_sign_internal'])}`",
                f"- 候选名称：{_md(generation['short_name'])}",
                f"- 解释类型：`{generation['explanation_type']}`",
                "",
                "### 第一阶段冻结解释",
                "",
                "**主要解释**",
                "",
                _md(generation["primary_explanation"]),
                "",
                "**候选行为或话语功能**",
                "",
                _md(generation["candidate_behavioral_explanation"]),
                "",
                "**替代解释**",
                "",
            ]
        )
        alternatives = generation.get("alternative_explanations") or []
        lines.extend([f"- {_md(item)}" for item in alternatives] or ["- 无"])
        lines.extend(["", "**可能混淆因素**", ""])
        confounds = generation.get("possible_confounds") or []
        lines.extend([f"- {_md(item)}" for item in confounds] or ["- 无"])
        lines.extend(["", "**解释局限**", ""])
        limitations = generation.get("limitations") or []
        lines.extend([f"- {_md(item)}" for item in limitations] or ["- 无"])
        lines.extend(
            [
                "",
                "### 第二阶段独立评分",
                "",
                "| 总分 | 覆盖分 | 等效覆盖数 | 簇内一致性 | 具体性 | 证据基础 |",
                "|---:|---:|---:|---:|---:|---:|",
                f"| **{int(scoring['overall_score'])}/5** | {int(scoring['coverage_score'])}/5 | "
                f"{float(scoring['equivalent_coverage']):.1f}/10 | {int(scoring['within_cluster_coherence'])}/5 | "
                f"{int(scoring['claim_specificity'])}/5 | {int(scoring['evidence_grounding'])}/5 |",
                "",
                "**簇内拟合摘要**",
                "",
                _md(scoring["within_cluster_fit_summary"]),
                "",
                "**评分理由**",
                "",
                f"- 簇内一致性：{_md(scoring['score_rationales']['within_cluster_coherence'])}",
                f"- 解释具体性：{_md(scoring['score_rationales']['claim_specificity'])}",
                f"- 证据基础：{_md(scoring['score_rationales']['evidence_grounding'])}",
                "",
                "### 完整句子簇与逐句判断",
                "",
                "| ID | 第一阶段划分 | 第二阶段判断 | 句子 | 判断依据 |",
                "|---|---|---|---|---|",
            ]
        )
        generation_support = set(str(item) for item in generation["supporting_sample_ids"])
        for sample in packs[anonymous_id]:
            sample_id = str(sample["sample_id"])
            judgment = judgments[sample_id]
            generation_role = "support" if sample_id in generation_support else "outlier"
            lines.append(
                f"| {sample_id} | {generation_role} | {judgment['verdict']} | "
                f"{_md(sample['text'])} | {_md(judgment['evidence'])} |"
            )
        lines.extend(["", "**当前句子簇不支持的解释表述**", ""])
        unsupported = scoring.get("unsupported_within_cluster_claims") or []
        lines.extend([f"- {_md(item)}" for item in unsupported] or ["- 无"])
        lines.extend(
            [
                "",
                "### 人工审核",
                "",
                "- 主要解释是否准确：`待审核`",
                "- 逐句判断是否合理：`待审核`",
                "- 程序总分是否合理：`待审核`",
                "- 人工备注：",
                "",
                "---",
            ]
        )
    path = output / f"task5_{family.lower()}_two_stage_cards_human_review.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


__all__ = [
    "GENERATION_SYSTEM_PROMPT",
    "SCORING_SYSTEM_PROMPT",
    "Task5TwoStageConfig",
    "build_generation_prompt",
    "build_generation_tasks",
    "build_scoring_prompt",
    "build_scoring_tasks",
    "render_two_stage_family_review_document",
    "render_two_stage_report",
    "validate_generation_outputs",
    "validate_scoring_outputs",
]
