"""Second-stage cautious function induction for MISC SAE latent packets.

This script consumes the first-stage latent evidence packets and produces
structured candidate interpretations. It is deliberately conservative: the
outputs are review hypotheses, not feature names or causal mechanism claims.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.ai_re_judge import OpenAICompatibleChatClient  # noqa: E402


DEFAULT_INPUT_DIR = Path(
    "outputs/misc_full_sae_eval/interpretability/top20_cohensd_latent_utterances/latent_evidence_packets"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/misc_full_sae_eval/interpretability/top20_cohensd_latent_utterances/latent_function_induction"
)
PROMPT_VERSION = "misc_latent_function_induction_v1"
PATTERN_TYPES = {
    "surface_form",
    "dialogue_function",
    "context_relation",
    "mi_principle",
    "artifact",
    "mixed_unclear",
}
EVIDENCE_QUALITIES = {"high", "medium", "low", "uninterpretable"}
CONTEXT_NOTE = "Only counselor current utterance is available in phase 1; prior client context is unavailable."


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _extract_json_blob(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Response did not contain a JSON object.")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object.")
    return obj


def _trim_examples(packet: dict[str, Any], max_examples_per_group: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for group in ("top_activating", "high_non_target", "random_target"):
        group_examples = [ex for ex in packet.get("examples", []) if ex.get("example_group") == group]
        examples.extend(group_examples[:max_examples_per_group])
    return examples


def _examples_for_prompt(examples: list[dict[str, Any]], *, blind: bool) -> list[dict[str, Any]]:
    blind_group_names = {
        "top_activating": "top",
        "high_non_target": "contrast_high",
        "random_target": "comparison_random",
    }
    prompt_examples: list[dict[str, Any]] = []
    for ex in examples:
        item = {
            "example_group": blind_group_names.get(ex.get("example_group"), ex.get("example_group"))
            if blind
            else ex.get("example_group"),
            "rank_within_group": ex.get("rank_within_group"),
            "activation": ex.get("activation"),
            "text": ex.get("unit_text"),
            "duplicate_text_within_packet": bool(ex.get("duplicate_text_within_packet", False)),
        }
        if not blind:
            item.update(
                {
                    "target_match": ex.get("target_match"),
                    "active_labels": ex.get("active_labels", ""),
                }
            )
        prompt_examples.append(item)
    return prompt_examples


def build_blind_prompt(packet: dict[str, Any], *, max_examples_per_group: int) -> list[dict[str, str]]:
    examples = _examples_for_prompt(_trim_examples(packet, max_examples_per_group), blind=True)
    system = (
        "You are a cautious reviewer of anonymous Sparse Autoencoder latents from Motivational "
        "Interviewing counselor utterances. Treat the latent as unlabeled. Do not infer or mention "
        "any target MISC label. Use hedged language. Separate surface form, dialogue function, "
        "MI-principle, artifact, and mixed/unclear patterns. Current data has only counselor current "
        "utterance; no prior client context is available."
    )
    user = {
        "task": "Blind pattern induction for one anonymous SAE latent.",
        "packet_id": packet.get("packet_id"),
        "latent_alias": packet.get("latent_alias"),
        "context_limitation": CONTEXT_NOTE,
        "allowed_pattern_types": sorted(PATTERN_TYPES),
        "evidence_groups": {
            "top_activating": "Highest activation utterances.",
            "contrast_high": "High activation contrast utterances from the comparison pool.",
            "comparison_random": "Random comparison utterances; use only as lower/ordinary activation comparison.",
        },
        "examples": examples,
        "required_json": {
            "tentative_interpretation": "string",
            "patterns": [
                {
                    "pattern_name": "string",
                    "pattern_type": "surface_form|dialogue_function|context_relation|mi_principle|artifact|mixed_unclear",
                    "evidence": "string",
                }
            ],
            "artifact_risks": ["string"],
            "evidence_quality": "high|medium|low|uninterpretable",
            "candidate_name": "string",
            "alternative_explanations": ["string", "string"],
            "recommended_followup_checks": ["string"],
            "final_conclusion": "string",
            "context_limitation_note": "string",
        },
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def build_labeled_prompt(
    *,
    blind_review: dict[str, Any],
    labeled_packet: dict[str, Any],
    summary_row: dict[str, Any],
    max_examples_per_group: int,
) -> list[dict[str, str]]:
    examples = _examples_for_prompt(_trim_examples(labeled_packet, max_examples_per_group), blind=False)
    system = (
        "You audit whether a blind SAE latent interpretation relates to a target MISC label. "
        "Do not convert the candidate explanation into a definitive feature name. Do not claim causal "
        "mechanism evidence. Explicitly separate surface-form support from counseling-function support. "
        "For RES/REC and context-dependent labels, downgrade context-relation claims because no prior "
        "client utterance is available."
    )
    user = {
        "task": "Labeled alignment audit for one SAE latent candidate interpretation.",
        "packet_id": labeled_packet.get("packet_id"),
        "target_label": labeled_packet.get("target_label"),
        "latent_idx": labeled_packet.get("latent_idx"),
        "rank_within_label": labeled_packet.get("rank_within_label"),
        "cohens_d": labeled_packet.get("cohens_d"),
        "directional_auc": labeled_packet.get("directional_auc"),
        "precision_at_50": labeled_packet.get("precision_at_50"),
        "context_limitation": CONTEXT_NOTE,
        "alignment_metrics": {
            "top_activating_target_match_rate": summary_row.get("top_activating_target_match_rate"),
            "active_label_counts_top_activating": summary_row.get("active_label_counts_top_activating"),
            "duplicate_text_row_count": summary_row.get("duplicate_text_row_count"),
            "unique_file_count": summary_row.get("unique_file_count"),
        },
        "blind_review": blind_review,
        "examples": examples,
        "required_json": {
            "target_label_relationship": "string",
            "adjacent_label_risks": ["string"],
            "artifact_risks": ["string"],
            "evidence_quality": "high|medium|low|uninterpretable",
            "final_conclusion": "string",
            "context_limitation_note": "string",
        },
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def _normalise_pattern(pattern: Any) -> dict[str, str]:
    if not isinstance(pattern, dict):
        return {"pattern_name": "unclear", "pattern_type": "mixed_unclear", "evidence": str(pattern)}
    pattern_type = str(pattern.get("pattern_type", "mixed_unclear")).strip().lower()
    if pattern_type not in PATTERN_TYPES:
        pattern_type = "mixed_unclear"
    return {
        "pattern_name": str(pattern.get("pattern_name") or pattern.get("name") or "unclear"),
        "pattern_type": pattern_type,
        "evidence": str(pattern.get("evidence") or ""),
    }


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]


def _normalise_review(
    *,
    blind_review: dict[str, Any] | None,
    labeled_review: dict[str, Any] | None,
    labeled_packet: dict[str, Any],
    summary_row: dict[str, Any],
    status: str,
    error: str = "",
) -> dict[str, Any]:
    blind_review = blind_review or {}
    labeled_review = labeled_review or {}
    patterns = [_normalise_pattern(p) for p in blind_review.get("patterns", [])]
    if not patterns:
        patterns = [
            {
                "pattern_name": "pending or unclear",
                "pattern_type": "mixed_unclear",
                "evidence": "No model review was available.",
            }
        ]
    patterns = patterns[:5]

    quality = str(labeled_review.get("evidence_quality") or blind_review.get("evidence_quality") or "uninterpretable").lower()
    if quality not in EVIDENCE_QUALITIES:
        quality = "uninterpretable"
    artifact_risks = _as_list(blind_review.get("artifact_risks")) + _as_list(labeled_review.get("artifact_risks"))
    candidate_name = str(blind_review.get("candidate_name") or "unclear / mixed pattern")
    context_note = str(labeled_review.get("context_limitation_note") or blind_review.get("context_limitation_note") or CONTEXT_NOTE)

    if labeled_packet.get("target_label") in {"RE", "RES", "REC"} and "client" not in context_note.lower():
        context_note = f"{context_note} Prior client utterance is unavailable, so reflection/context claims are limited."

    return {
        "packet_id": labeled_packet.get("packet_id"),
        "latent_alias": summary_row.get("latent_alias") or labeled_packet.get("packet_id"),
        "target_label": labeled_packet.get("target_label"),
        "latent_idx": labeled_packet.get("latent_idx"),
        "rank_within_label": labeled_packet.get("rank_within_label"),
        "cohens_d": labeled_packet.get("cohens_d"),
        "directional_auc": labeled_packet.get("directional_auc"),
        "precision_at_50": labeled_packet.get("precision_at_50"),
        "status": status,
        "error": error,
        "tentative_interpretation": str(
            blind_review.get("tentative_interpretation")
            or "No stable interpretation can be assigned based on the current examples."
        ),
        "patterns": patterns,
        "target_label_relationship": str(labeled_review.get("target_label_relationship") or ""),
        "adjacent_label_risks": _as_list(labeled_review.get("adjacent_label_risks")),
        "artifact_risks": artifact_risks,
        "evidence_quality": quality,
        "candidate_name": candidate_name,
        "alternative_explanations": _as_list(blind_review.get("alternative_explanations")),
        "recommended_followup_checks": _as_list(blind_review.get("recommended_followup_checks"))
        + _as_list(labeled_review.get("recommended_followup_checks")),
        "final_conclusion": str(labeled_review.get("final_conclusion") or blind_review.get("final_conclusion") or ""),
        "context_limitation_note": context_note,
        "top_activating_target_match_rate": summary_row.get("top_activating_target_match_rate"),
        "active_label_counts_top_activating": summary_row.get("active_label_counts_top_activating"),
        "duplicate_text_row_count": summary_row.get("duplicate_text_row_count"),
        "unique_file_count": summary_row.get("unique_file_count"),
    }


def _pending_review(labeled_packet: dict[str, Any], summary_row: dict[str, Any]) -> dict[str, Any]:
    blind = {
        "tentative_interpretation": "Dry run only; no model interpretation has been generated.",
        "patterns": [
            {
                "pattern_name": "pending dry-run review",
                "pattern_type": "mixed_unclear",
                "evidence": "Prompt generated but not sent to a model.",
            }
        ],
        "evidence_quality": "uninterpretable",
        "candidate_name": "pending dry-run review",
        "alternative_explanations": ["No model review was run.", CONTEXT_NOTE],
        "recommended_followup_checks": ["Run without --dry-run-prompts using a configured review model."],
        "final_conclusion": "No functional conclusion should be drawn from dry-run output.",
        "context_limitation_note": CONTEXT_NOTE,
    }
    labeled = {
        "target_label_relationship": "Pending dry-run; no label alignment audit has been performed.",
        "evidence_quality": "uninterpretable",
        "context_limitation_note": CONTEXT_NOTE,
    }
    return _normalise_review(
        blind_review=blind,
        labeled_review=labeled,
        labeled_packet=labeled_packet,
        summary_row=summary_row,
        status="pending_dry_run",
    )


def _failed_review(labeled_packet: dict[str, Any], summary_row: dict[str, Any], error: str) -> dict[str, Any]:
    return _normalise_review(
        blind_review=None,
        labeled_review=None,
        labeled_packet=labeled_packet,
        summary_row=summary_row,
        status="failed",
        error=error,
    )


def _dominant_pattern_type(review: dict[str, Any]) -> str:
    patterns = review.get("patterns") or []
    if not patterns:
        return "mixed_unclear"
    return str(patterns[0].get("pattern_type", "mixed_unclear"))


def _target_support(review: dict[str, Any]) -> str:
    text = f"{review.get('target_label_relationship', '')} {review.get('final_conclusion', '')}".lower()
    if review.get("status") != "ok":
        return "not_reviewed"
    if any(term in text for term in ("does not", "weak", "limited", "unclear", "mixed")):
        return "limited_or_mixed"
    if any(term in text for term in ("support", "consistent", "align", "fits")):
        return "supports_candidate"
    return "unclear"


def reviews_to_tables(reviews: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    review_rows: list[dict[str, Any]] = []
    pattern_rows: list[dict[str, Any]] = []
    for review in reviews:
        dominant = _dominant_pattern_type(review)
        artifact_risk = "yes" if review.get("artifact_risks") else "no"
        review_rows.append(
            {
                "packet_id": review["packet_id"],
                "target_label": review["target_label"],
                "latent_idx": review["latent_idx"],
                "rank_within_label": review["rank_within_label"],
                "status": review["status"],
                "candidate_name": review["candidate_name"],
                "dominant_pattern_type": dominant,
                "confidence": review["evidence_quality"],
                "artifact_risk": artifact_risk,
                "target_support": _target_support(review),
                "top_activating_target_match_rate": review.get("top_activating_target_match_rate"),
                "duplicate_text_row_count": review.get("duplicate_text_row_count"),
                "context_limitation_note": review.get("context_limitation_note"),
            }
        )
        for idx, pattern in enumerate(review.get("patterns", []), start=1):
            pattern_rows.append(
                {
                    "packet_id": review["packet_id"],
                    "target_label": review["target_label"],
                    "latent_idx": review["latent_idx"],
                    "pattern_rank": idx,
                    "pattern_name": pattern.get("pattern_name"),
                    "pattern_type": pattern.get("pattern_type"),
                    "evidence": pattern.get("evidence"),
                    "review_status": review["status"],
                }
            )

    reviews_df = pd.DataFrame(review_rows)
    patterns_df = pd.DataFrame(pattern_rows)
    label_rows: list[dict[str, Any]] = []
    for label, group in reviews_df.groupby("target_label", sort=False):
        pattern_counts = Counter(group["dominant_pattern_type"].fillna("mixed_unclear"))
        quality_counts = Counter(group["confidence"].fillna("uninterpretable"))
        artifact_count = int((group["artifact_risk"] == "yes").sum())
        label_rows.append(
            {
                "target_label": label,
                "n_latents": int(len(group)),
                "status_counts": dict(Counter(group["status"])),
                "dominant_pattern_counts": dict(pattern_counts),
                "evidence_quality_counts": dict(quality_counts),
                "artifact_risk_latents": artifact_count,
                "uninterpretable_latents": int((group["confidence"] == "uninterpretable").sum()),
                "mean_top_activating_target_match_rate": float(
                    pd.to_numeric(group["top_activating_target_match_rate"], errors="coerce").mean()
                ),
            }
        )
    label_summary_df = pd.DataFrame(label_rows)
    return reviews_df, patterns_df, label_summary_df


def _format_counts(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{k}:{v}" for k, v in value.items())
    return str(value)


def _write_report_legacy_mojibake(
    *,
    reviews: list[dict[str, Any]],
    reviews_df: pd.DataFrame,
    label_summary_df: pd.DataFrame,
    output_path: Path,
    dry_run: bool,
) -> None:
    status_counts = Counter(review["status"] for review in reviews)
    lines = [
        "# SAE Latent 谨慎功能归纳报告",
        "",
        "本报告消费第一阶段 latent evidence packets，对每个 SAE latent 给出候选功能解释或 dry-run 占位结果。",
        "",
        "## 方法边界",
        "",
        "- 输出是 candidate functional interpretation，不是因果机制证明。",
        "- 第一阶段只有 counselor current utterance，没有前一句 client utterance。",
        "- RES/REC 等上下文依赖标签的 context-relation 解释必须降级处理。",
        "- top activating examples 只说明 activation preference，不等于完整标签机制。",
        f"- dry run: `{dry_run}`。",
        "",
        "## 覆盖情况",
        "",
        f"- latent reviews: {len(reviews)}",
        f"- status counts: {_format_counts(dict(status_counts))}",
        "",
        "## 每标签概览",
        "",
        "| label | n | status | dominant patterns | evidence quality | artifact-risk latents | uninterpretable | mean target-match |",
        "|---|---:|---|---|---|---:|---:|---:|",
    ]
    for _, row in label_summary_df.iterrows():
        lines.append(
            "| {label} | {n} | {status} | {patterns} | {quality} | {artifact} | {uninterp} | {match} |".format(
                label=row["target_label"],
                n=int(row["n_latents"]),
                status=_format_counts(row["status_counts"]),
                patterns=_format_counts(row["dominant_pattern_counts"]),
                quality=_format_counts(row["evidence_quality_counts"]),
                artifact=int(row["artifact_risk_latents"]),
                uninterp=int(row["uninterpretable_latents"]),
                match=f"{float(row['mean_top_activating_target_match_rate']):.3f}"
                if pd.notna(row["mean_top_activating_target_match_rate"])
                else "NA",
            )
        )

    lines.extend(["", "## 代表性条目", ""])
    ok_reviews = [review for review in reviews if review["status"] == "ok"]
    examples = ok_reviews[:12] if ok_reviews else reviews[:12]
    for review in examples:
        lines.extend(
            [
                f"### {review['target_label']} latent {review['latent_idx']} ({review['packet_id']})",
                "",
                f"- status: `{review['status']}`",
                f"- candidate name: {review['candidate_name']}",
                f"- evidence quality: {review['evidence_quality']}",
                f"- dominant pattern: {_dominant_pattern_type(review)}",
                f"- tentative interpretation: {review['tentative_interpretation']}",
                f"- target relationship: {review['target_label_relationship']}",
                f"- conclusion: {review['final_conclusion']}",
                "",
            ]
        )

    lines.extend(
        [
            "## 可发表表述建议",
            "",
            "可以说：SAE top-latents 的高激活语句呈现若干候选 surface/function/artifact 模式，可用于解释标签可解码性的表征证据。",
            "",
            "不要说：这些 latent 已经证明 LLM 以 MISC codebook 的方式理解咨询行为，或这些 latent 是标签判断的因果机制。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_report(
    *,
    reviews: list[dict[str, Any]],
    reviews_df: pd.DataFrame,
    label_summary_df: pd.DataFrame,
    output_path: Path,
    dry_run: bool,
) -> None:
    status_counts = Counter(review["status"] for review in reviews)
    lines = [
        "# SAE Latent 谨慎功能归纳报告",
        "",
        "本报告消费第一阶段 latent evidence packets，对每个 SAE latent 给出候选功能解释或 dry-run 占位结果。",
        "",
        "## 方法边界",
        "",
        "- 输出是 candidate functional interpretation，不是因果机制证明。",
        "- 第一阶段只有 counselor current utterance，没有前一句 client utterance。",
        "- RES/REC 等上下文依赖标签的 context-relation 解释必须降级处理。",
        "- top activating examples 只能说明 activation preference，不等于完整标签机制。",
        f"- dry run: `{dry_run}`。",
        "",
        "## 覆盖情况",
        "",
        f"- latent reviews: {len(reviews)}",
        f"- status counts: {_format_counts(dict(status_counts))}",
        "",
        "## 每标签概览",
        "",
        "| label | n | status | dominant patterns | evidence quality | artifact-risk latents | uninterpretable | mean target-match |",
        "|---|---:|---|---|---|---:|---:|---:|",
    ]
    for _, row in label_summary_df.iterrows():
        match = "NA"
        if pd.notna(row["mean_top_activating_target_match_rate"]):
            match = f"{float(row['mean_top_activating_target_match_rate']):.3f}"
        lines.append(
            "| {label} | {n} | {status} | {patterns} | {quality} | {artifact} | {uninterp} | {match} |".format(
                label=row["target_label"],
                n=int(row["n_latents"]),
                status=_format_counts(row["status_counts"]),
                patterns=_format_counts(row["dominant_pattern_counts"]),
                quality=_format_counts(row["evidence_quality_counts"]),
                artifact=int(row["artifact_risk_latents"]),
                uninterp=int(row["uninterpretable_latents"]),
                match=match,
            )
        )

    lines.extend(["", "## 代表性条目", ""])
    ok_reviews = [review for review in reviews if review["status"] == "ok"]
    examples = ok_reviews[:12] if ok_reviews else reviews[:12]
    for review in examples:
        lines.extend(
            [
                f"### {review['target_label']} latent {review['latent_idx']} ({review['packet_id']})",
                "",
                f"- status: `{review['status']}`",
                f"- candidate name: {review['candidate_name']}",
                f"- evidence quality: {review['evidence_quality']}",
                f"- dominant pattern: {_dominant_pattern_type(review)}",
                f"- tentative interpretation: {review['tentative_interpretation']}",
                f"- target relationship: {review['target_label_relationship']}",
                f"- conclusion: {review['final_conclusion']}",
                "",
            ]
        )

    lines.extend(
        [
            "## 可发表表述建议",
            "",
            "可以说：SAE top-latents 的高激活语句呈现若干候选 surface/function/artifact 模式，可作为解释标签可解码性的表征证据。",
            "",
            "不要说：这些 latent 已经证明 LLM 以 MISC codebook 的方式理解咨询行为，或这些 latent 是标签判断的因果机制。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _packet_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["packet_id"]): row for row in rows}


def run_latent_function_induction(
    *,
    blind_packets_path: str | Path,
    labeled_packets_path: str | Path,
    summary_path: str | Path,
    output_dir: str | Path,
    dry_run_prompts: bool = False,
    model: str | None = None,
    max_examples_per_group: int = 15,
    temperature: float = 0.0,
    limit: int | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    blind_packets_path = Path(blind_packets_path)
    labeled_packets_path = Path(labeled_packets_path)
    summary_path = Path(summary_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prompts_dir = output_path / "prompts"
    raw_dir = output_path / "raw_responses"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    blind_packets = _read_jsonl(blind_packets_path)
    labeled_packets = _read_jsonl(labeled_packets_path)
    summary_df = pd.read_csv(summary_path)
    summary_map = {str(row["packet_id"]): row.to_dict() for _, row in summary_df.iterrows()}
    blind_map = _packet_by_id(blind_packets)

    if limit is not None:
        labeled_packets = labeled_packets[: int(limit)]

    resolved_model = model or os.getenv("OPENAI_MODEL")
    if not dry_run_prompts and client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required unless --dry-run-prompts is used or a custom client is provided.")
        if not resolved_model:
            raise RuntimeError("Review model must be provided via --model or OPENAI_MODEL.")
        client = OpenAICompatibleChatClient(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL"),
            timeout=int(os.getenv("OPENAI_HTTP_TIMEOUT", "600")),
        )
    elif client is not None and not resolved_model:
        resolved_model = "custom-client-model"
    elif dry_run_prompts and not resolved_model:
        resolved_model = "dry-run-model"

    reviews: list[dict[str, Any]] = []
    for labeled_packet in labeled_packets:
        packet_id = str(labeled_packet["packet_id"])
        blind_packet = blind_map.get(packet_id)
        summary_row = summary_map.get(packet_id, {})
        if blind_packet is None:
            reviews.append(_failed_review(labeled_packet, summary_row, "Missing blind packet."))
            continue

        blind_messages = build_blind_prompt(blind_packet, max_examples_per_group=max_examples_per_group)
        labeled_prompt_stub: list[dict[str, str]] | None = None
        slug = _safe_slug(packet_id)
        _write_json(prompts_dir / f"{slug}_blind_prompt.json", {"messages": blind_messages})

        if dry_run_prompts:
            placeholder = _pending_review(labeled_packet, summary_row)
            labeled_prompt_stub = build_labeled_prompt(
                blind_review=placeholder,
                labeled_packet=labeled_packet,
                summary_row=summary_row,
                max_examples_per_group=max_examples_per_group,
            )
            _write_json(prompts_dir / f"{slug}_labeled_prompt.json", {"messages": labeled_prompt_stub})
            reviews.append(placeholder)
            continue

        try:
            blind_content, blind_raw = client.chat(model=resolved_model, messages=blind_messages, temperature=temperature)
            _write_json(raw_dir / f"{slug}_blind_response.json", blind_raw)
            blind_review = _extract_json_blob(blind_content)

            labeled_messages = build_labeled_prompt(
                blind_review=blind_review,
                labeled_packet=labeled_packet,
                summary_row=summary_row,
                max_examples_per_group=max_examples_per_group,
            )
            _write_json(prompts_dir / f"{slug}_labeled_prompt.json", {"messages": labeled_messages})
            labeled_content, labeled_raw = client.chat(model=resolved_model, messages=labeled_messages, temperature=temperature)
            _write_json(raw_dir / f"{slug}_labeled_response.json", labeled_raw)
            labeled_review = _extract_json_blob(labeled_content)
            reviews.append(
                _normalise_review(
                    blind_review=blind_review,
                    labeled_review=labeled_review,
                    labeled_packet=labeled_packet,
                    summary_row=summary_row,
                    status="ok",
                )
            )
        except Exception as exc:  # Keep full-batch review robust.
            _write_json(raw_dir / f"{slug}_error.json", {"error": f"{type(exc).__name__}: {exc}"})
            reviews.append(_failed_review(labeled_packet, summary_row, f"{type(exc).__name__}: {exc}"))

    reviews_df, patterns_df, label_summary_df = reviews_to_tables(reviews)
    output_files = {
        "reviews_jsonl": output_path / "latent_function_reviews.jsonl",
        "reviews_csv": output_path / "latent_function_reviews.csv",
        "patterns_csv": output_path / "latent_function_patterns.csv",
        "label_summary_csv": output_path / "label_function_cluster_summary.csv",
        "report_md": output_path / "latent_function_induction_report.md",
        "manifest_json": output_path / "manifest.json",
        "prompts_dir": prompts_dir,
        "raw_responses_dir": raw_dir,
    }
    _write_jsonl(output_files["reviews_jsonl"], reviews)
    reviews_df.to_csv(output_files["reviews_csv"], index=False)
    patterns_df.to_csv(output_files["patterns_csv"], index=False)
    label_summary_df.to_csv(output_files["label_summary_csv"], index=False)
    write_report(
        reviews=reviews,
        reviews_df=reviews_df,
        label_summary_df=label_summary_df,
        output_path=output_files["report_md"],
        dry_run=dry_run_prompts,
    )

    status_counts = dict(Counter(review["status"] for review in reviews))
    manifest = {
        "analysis": "misc_sae_latent_function_induction_phase2",
        "prompt_version": PROMPT_VERSION,
        "dry_run_prompts": bool(dry_run_prompts),
        "model": resolved_model,
        "max_examples_per_group": int(max_examples_per_group),
        "context_note": CONTEXT_NOTE,
        "inputs": {
            "blind_packets": str(blind_packets_path),
            "labeled_packets": str(labeled_packets_path),
            "summary": str(summary_path),
        },
        "outputs": {key: str(value) for key, value in output_files.items()},
        "n_reviews": int(len(reviews)),
        "status_counts": status_counts,
        "n_success": int(status_counts.get("ok", 0)),
        "n_failed": int(status_counts.get("failed", 0)),
        "n_pending_dry_run": int(status_counts.get("pending_dry_run", 0)),
    }
    _write_json(output_files["manifest_json"], manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cautious two-pass function induction for MISC SAE latent evidence packets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--blind-packets", default=None)
    parser.add_argument("--labeled-packets", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dry-run-prompts", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-examples-per-group", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    manifest = run_latent_function_induction(
        blind_packets_path=args.blind_packets or input_dir / "latent_evidence_packets_blind.jsonl",
        labeled_packets_path=args.labeled_packets or input_dir / "latent_evidence_packets_labeled.jsonl",
        summary_path=args.summary or input_dir / "latent_evidence_packet_summary.csv",
        output_dir=args.output_dir,
        dry_run_prompts=args.dry_run_prompts,
        model=args.model,
        max_examples_per_group=args.max_examples_per_group,
        temperature=args.temperature,
        limit=args.limit,
    )
    print("Completed latent function induction.")
    print(f"Output dir: {args.output_dir}")
    print(f"Reviews: {manifest['n_reviews']}")
    print(f"Status counts: {manifest['status_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
