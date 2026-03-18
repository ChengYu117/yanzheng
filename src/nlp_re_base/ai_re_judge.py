"""AI expert-proxy review pipeline for SAE-RE."""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from .re_judge_rubric import RE_JUDGE_RUBRIC, get_rubric_snapshot

JUDGE_BUNDLE_VERSION = "1.0"
DEFAULT_GROUP_NAMES = ("G1", "G5", "G20")
REVIEW_SCORE_MAP = {"yes": 1.0, "partial": 0.5, "no": 0.0}
UTTERANCE_REVIEW_ENUMS = {
    "has_clear_re_feature": {"yes", "partial", "no"},
    "re_type": {"simple", "complex", "mixed", "non_re", "unclear"},
}
RISK_FLAGS = set(RE_JUDGE_RUBRIC["risk_flags"])  # type: ignore[arg-type]


def _safe_std(x: np.ndarray) -> np.ndarray:
    std = x.std(axis=0, keepdims=True)
    return np.where(std < 1e-6, 1.0, std)


def normalize_weights(weights: list[float] | np.ndarray | None, size: int) -> list[float]:
    """Normalise absolute weights with equal-weight fallback."""
    if size <= 0:
        return []
    if weights is None:
        return [1.0 / size] * size
    arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if arr.size != size:
        return [1.0 / size] * size
    arr = np.abs(arr)
    denom = float(arr.sum())
    if denom <= 1e-12:
        return [1.0 / size] * size
    return (arr / denom).astype(np.float64).tolist()


def compute_group_scores(
    utterance_features: np.ndarray,
    latent_ids: list[int],
    weights: list[float] | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Compute weighted group scores from z-scored utterance features."""
    if not latent_ids:
        return np.zeros(utterance_features.shape[0], dtype=np.float32), []

    X = np.asarray(utterance_features[:, latent_ids], dtype=np.float32)
    mean = X.mean(axis=0, keepdims=True)
    std = _safe_std(X)
    Xz = (X - mean) / std

    norm_weights = normalize_weights(weights, len(latent_ids))
    scores = Xz @ np.asarray(norm_weights, dtype=np.float32)
    return np.asarray(scores, dtype=np.float32), norm_weights


def _record_to_metadata(record: dict[str, Any] | None) -> dict[str, Any]:
    record = record or {}
    return {
        "file_id": record.get("file_id"),
        "source_file": record.get("source_file"),
        "predicted_code": record.get("predicted_code"),
        "predicted_subcode": record.get("predicted_subcode"),
        "rationale": record.get("rationale"),
        "confidence": record.get("confidence"),
    }


def _label_name(label: int, record: dict[str, Any] | None) -> str:
    if record and isinstance(record.get("predicted_code"), str):
        return str(record["predicted_code"])
    return "RE" if label == 1 else "NonRE"


def _select_control_indices(
    scores: np.ndarray,
    excluded: set[int],
    control_n: int,
) -> list[int]:
    """Select stable controls from the 40%-60% score band."""
    if scores.size == 0 or control_n <= 0:
        return []

    q_low, q_high = np.quantile(scores, [0.4, 0.6])
    median = float(np.median(scores))
    candidates = [
        idx for idx, value in enumerate(scores)
        if idx not in excluded and q_low <= value <= q_high
    ]
    candidates.sort(key=lambda idx: (abs(float(scores[idx]) - median), idx))

    if len(candidates) < control_n:
        fallback = [
            idx for idx in np.argsort(np.abs(scores - median)).tolist()
            if idx not in excluded and idx not in candidates
        ]
        candidates.extend(fallback)

    return candidates[:control_n]


def _make_example(
    *,
    utterance_idx: int,
    score_name: str,
    score_value: float,
    texts: list[str],
    labels: list[int],
    records: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    record = records[utterance_idx] if records and utterance_idx < len(records) else None
    metadata = _record_to_metadata(record)
    return {
        "utterance_idx": int(utterance_idx),
        "text": texts[utterance_idx],
        "label": "RE" if labels[utterance_idx] == 1 else "NonRE",
        "label_int": int(labels[utterance_idx]),
        "score_name": score_name,
        "score_value": float(score_value),
        **metadata,
        "predicted_code_or_label": _label_name(labels[utterance_idx], record),
    }


def export_judge_bundle(
    *,
    output_dir: str | Path,
    candidate_df,
    utterance_features: np.ndarray,
    texts: list[str],
    labels: list[int],
    records: list[dict[str, Any]] | None = None,
    aggregation: str = "max",
    hook_point: str | None = None,
    model_name: str | None = None,
    sae_repo_id: str | None = None,
    sae_subfolder: str | None = None,
    group_weights: dict[str, list[float]] | None = None,
    top_latents: int = 20,
    top_n: int = 10,
    control_n: int = 5,
) -> Path:
    """Export a lightweight bundle for downstream AI review."""
    out_path = Path(output_dir) / "judge_bundle"
    out_path.mkdir(parents=True, exist_ok=True)

    top_latent_ids = candidate_df["latent_idx"].head(top_latents).astype(int).tolist()
    latent_records: list[dict[str, Any]] = []
    for rank, lat_idx in enumerate(top_latent_ids, start=1):
        scores = np.asarray(utterance_features[:, lat_idx], dtype=np.float32)
        top_indices = np.argsort(scores)[::-1][:top_n].tolist()
        control_indices = _select_control_indices(scores, set(top_indices), control_n)
        top_examples = [
            _make_example(
                utterance_idx=idx,
                score_name="activation",
                score_value=float(scores[idx]),
                texts=texts,
                labels=labels,
                records=records,
            )
            for idx in top_indices
        ]
        control_examples = [
            _make_example(
                utterance_idx=idx,
                score_name="activation",
                score_value=float(scores[idx]),
                texts=texts,
                labels=labels,
                records=records,
            )
            for idx in control_indices
        ]
        latent_records.append({
            "latent_idx": int(lat_idx),
            "candidate_rank": rank,
            "re_purity_top_n": float(np.mean([x["label_int"] for x in top_examples])) if top_examples else 0.0,
            "top_examples": top_examples,
            "control_examples": control_examples,
        })

    groups = {
        "G1": top_latent_ids[:1],
        "G5": top_latent_ids[:5],
        "G20": top_latent_ids[:20],
    }
    group_weights = group_weights or {}
    group_payload: dict[str, Any] = {}
    for group_name, latent_ids in groups.items():
        scores, norm_weights = compute_group_scores(
            utterance_features=utterance_features,
            latent_ids=latent_ids,
            weights=group_weights.get(group_name),
        )
        top_indices = np.argsort(scores)[::-1][:top_n].tolist()
        control_indices = _select_control_indices(scores, set(top_indices), control_n)
        group_payload[group_name] = {
            "group_name": group_name,
            "latent_ids": [int(idx) for idx in latent_ids],
            "weights": norm_weights,
            "top_examples": [
                _make_example(
                    utterance_idx=idx,
                    score_name="group_score",
                    score_value=float(scores[idx]),
                    texts=texts,
                    labels=labels,
                    records=records,
                )
                for idx in top_indices
            ],
            "control_examples": [
                _make_example(
                    utterance_idx=idx,
                    score_name="group_score",
                    score_value=float(scores[idx]),
                    texts=texts,
                    labels=labels,
                    records=records,
                )
                for idx in control_indices
            ],
        }

    manifest = {
        "bundle_version": JUDGE_BUNDLE_VERSION,
        "dataset_size": len(texts),
        "top_latents": top_latents,
        "top_n": top_n,
        "control_n": control_n,
        "aggregation": aggregation,
        "hook_point": hook_point,
        "model_name": model_name,
        "sae_repo_id": sae_repo_id,
        "sae_subfolder": sae_subfolder,
        "method_positioning": RE_JUDGE_RUBRIC["method_positioning"],
    }

    with open(out_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(out_path / "latent_examples.jsonl", "w", encoding="utf-8") as f:
        for row in latent_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out_path / "group_examples.json", "w", encoding="utf-8") as f:
        json.dump(group_payload, f, indent=2, ensure_ascii=False)
    with open(out_path / "rubric_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(get_rubric_snapshot(), f, indent=2, ensure_ascii=False)

    return out_path


def _rubric_prompt_block() -> str:
    dims = ", ".join(dim["key"] for dim in RE_JUDGE_RUBRIC["dimensions"])  # type: ignore[index]
    risk_flags = ", ".join(RE_JUDGE_RUBRIC["risk_flags"])  # type: ignore[arg-type]
    return (
        "Use the following fixed rubric.\n"
        "Simple Reflection: paraphrases or mirrors content/feeling already expressed.\n"
        "Complex Reflection: adds valid empathic or meaning-level inference while staying faithful.\n"
        "Non-RE: question, advice, information-giving, directive language, persuasion, or empty template.\n"
        f"Score dimensions (1-5): {dims}.\n"
        f"Allowed risk flags: {risk_flags}.\n"
        "Client context is unavailable; judge only sentence-internal evidence."
    )


def build_utterance_review_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    """Build the per-utterance judge prompt."""
    schema = {
        "has_clear_re_feature": "yes|partial|no",
        "re_type": "simple|complex|mixed|non_re|unclear",
        "clarity_score": 1,
        "dimension_scores": {
            "mirrors_client_meaning": 1,
            "adds_valid_meaning_or_empathy": 1,
            "non_directive_non_question": 1,
            "natural_therapeutic_language": 1,
        },
        "evidence_spans": ["..."],
        "reason_zh": "中文理由",
        "risk_flags": ["context_needed"],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are simulating a senior motivational interviewing expert. "
                "Judge whether a single counselor utterance clearly expresses reflective listening. "
                "Return JSON only and never mention hidden labels."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{_rubric_prompt_block()}\n\n"
                "Return valid JSON only with this schema:\n"
                f"{json.dumps(schema, ensure_ascii=False)}\n\n"
                f"Utterance:\n{example['text']}"
            ),
        },
    ]


def build_synthesis_messages(
    *,
    object_kind: str,
    object_name: str,
    reviewed_examples: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build the latent/group synthesis prompt."""
    payload = []
    for item in reviewed_examples:
        payload.append({
            "text": item["text"],
            "score_value": item["score_value"],
            "review": item["review"],
        })
    schema = {
        "shared_feature_name": "简洁特征名",
        "shared_feature_description_zh": "中文描述",
        "common_positive_evidence": ["..."],
        "common_counterevidence": ["..."],
        "failure_modes": ["..."],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are summarizing whether a latent or latent group has a coherent RE-related feature. "
                "Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{_rubric_prompt_block()}\n\n"
                f"Object kind: {object_kind}\n"
                f"Object name: {object_name}\n\n"
                "Reviewed utterances:\n"
                f"{json.dumps(payload, ensure_ascii=False)}\n\n"
                "Return valid JSON only with this schema:\n"
                f"{json.dumps(schema, ensure_ascii=False)}"
            ),
        },
    ]


def _extract_json_blob(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if "```" in text:
        chunks = [chunk.strip() for chunk in text.split("```") if chunk.strip()]
        for chunk in chunks:
            if chunk.startswith("json"):
                chunk = chunk[4:].strip()
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                continue

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("Could not parse JSON from judge response.")


def _validate_dimension_scores(raw: dict[str, Any]) -> dict[str, int]:
    dims = {}
    input_scores = raw.get("dimension_scores") or {}
    for dim in RE_JUDGE_RUBRIC["dimensions"]:  # type: ignore[assignment]
        key = dim["key"]
        value = int(input_scores.get(key, 1))
        dims[key] = max(1, min(5, value))
    return dims


def validate_utterance_review(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and coerce the per-utterance review schema."""
    has_clear = str(raw.get("has_clear_re_feature", "no")).strip().lower()
    if has_clear not in UTTERANCE_REVIEW_ENUMS["has_clear_re_feature"]:
        has_clear = "no"

    re_type = str(raw.get("re_type", "unclear")).strip().lower()
    if re_type not in UTTERANCE_REVIEW_ENUMS["re_type"]:
        re_type = "unclear"

    clarity = max(1, min(5, int(raw.get("clarity_score", 1))))
    evidence_spans = raw.get("evidence_spans") or []
    if not isinstance(evidence_spans, list):
        evidence_spans = []
    risk_flags = raw.get("risk_flags") or []
    if not isinstance(risk_flags, list):
        risk_flags = []

    return {
        "has_clear_re_feature": has_clear,
        "re_type": re_type,
        "clarity_score": clarity,
        "dimension_scores": _validate_dimension_scores(raw),
        "evidence_spans": [str(item).strip() for item in evidence_spans if str(item).strip()],
        "reason_zh": str(raw.get("reason_zh", "")).strip(),
        "risk_flags": [flag for flag in map(str, risk_flags) if flag in RISK_FLAGS],
    }


def validate_synthesis_review(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and coerce the synthesis schema."""
    return {
        "shared_feature_name": str(raw.get("shared_feature_name", "")).strip(),
        "shared_feature_description_zh": str(raw.get("shared_feature_description_zh", "")).strip(),
        "common_positive_evidence": [
            str(item).strip() for item in (raw.get("common_positive_evidence") or [])
            if str(item).strip()
        ],
        "common_counterevidence": [
            str(item).strip() for item in (raw.get("common_counterevidence") or [])
            if str(item).strip()
        ],
        "failure_modes": [
            str(item).strip() for item in (raw.get("failure_modes") or [])
            if str(item).strip()
        ],
    }


class OpenAICompatibleChatClient:
    """Small HTTP client for OpenAI-compatible chat completions."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: int = 600,
    ) -> None:
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.extra_body = dict(extra_body or {})
        self.timeout = timeout

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        if self.extra_body:
            payload.update(self.extra_body)
        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw_response = json.loads(resp.read().decode("utf-8"))
        content = raw_response["choices"][0]["message"]["content"]
        return content, raw_response


def _parse_json_env(name: str) -> dict[str, Any] | None:
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must decode to a JSON object.")
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _call_with_retry(
    *,
    client: OpenAICompatibleChatClient,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_retries: int,
    log_prefix: str,
    logs_dir: Path,
) -> dict[str, Any]:
    prompt_payload = {"model": model, "messages": messages, "temperature": temperature}
    _write_json(logs_dir / f"{log_prefix}_prompt.json", prompt_payload)

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            content, raw_response = client.chat(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            _write_json(
                logs_dir / f"{log_prefix}_response_attempt_{attempt}.json",
                raw_response,
            )
            return _extract_json_blob(content)
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            socket.timeout,
            ConnectionResetError,
            ValueError,
            KeyError,
        ) as exc:
            last_error = exc
            _write_json(
                logs_dir / f"{log_prefix}_error_attempt_{attempt}.json",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            time.sleep(min(2 * attempt, 5))

    raise RuntimeError(f"Judge request failed after {max_retries} attempts: {last_error}")


def _review_strength(review: dict[str, Any]) -> float:
    return float(REVIEW_SCORE_MAP.get(review.get("has_clear_re_feature", "no"), 0.0))


def _dominant_re_type(reviews: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for review in reviews:
        key = str(review.get("re_type", "unclear"))
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return "unclear"
    return max(sorted(counts), key=lambda key: counts[key])


def _flag_rate(reviews: list[dict[str, Any]], flag: str) -> float:
    if not reviews:
        return 0.0
    return float(np.mean([flag in review.get("risk_flags", []) for review in reviews]))


def classify_latent_review(
    *,
    judge_re_rate: float,
    control_re_rate: float,
    avg_clarity_score: float,
    lexical_template_only_rate: float,
) -> str:
    """Apply the fixed latent judgement thresholds."""
    if (
        judge_re_rate >= 0.70
        and avg_clarity_score >= 4.0
        and (judge_re_rate - control_re_rate) >= 0.25
        and lexical_template_only_rate < 0.30
    ):
        return "clear_re_feature"
    if judge_re_rate >= 0.40:
        return "mixed_feature"
    if judge_re_rate >= 0.20:
        return "weak_re_signal"
    return "non_re_feature"


def classify_group_review(
    *,
    judge_re_rate: float,
    control_re_rate: float,
    avg_clarity_score: float,
) -> str:
    """Apply a fixed group-level classification heuristic."""
    if judge_re_rate >= 0.70 and avg_clarity_score >= 4.0 and (judge_re_rate - control_re_rate) >= 0.25:
        return "clear_re_subspace"
    if judge_re_rate >= 0.40:
        return "mixed_subspace"
    if judge_re_rate >= 0.20:
        return "weak_re_subspace"
    return "non_re_subspace"


def _load_bundle(input_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    bundle_dir = Path(input_dir)
    if bundle_dir.name != "judge_bundle":
        bundle_dir = bundle_dir / "judge_bundle"

    manifest_path = bundle_dir / "manifest.json"
    latent_path = bundle_dir / "latent_examples.jsonl"
    group_path = bundle_dir / "group_examples.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing bundle manifest: {manifest_path}")
    if not latent_path.exists():
        raise FileNotFoundError(f"Missing latent examples: {latent_path}")
    if not group_path.exists():
        raise FileNotFoundError(f"Missing group examples: {group_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    latent_records = []
    with open(latent_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                latent_records.append(json.loads(line))
    with open(group_path, "r", encoding="utf-8") as f:
        group_records = json.load(f)
    return manifest, latent_records, group_records


def _make_scope_name(scope_type: str, identifier: str | int) -> str:
    if scope_type == "latent":
        return f"latent_{int(identifier):05d}"
    return str(identifier)


def _iter_scope_examples(
    *,
    scope_type: str,
    scope_name: str,
    examples: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    rows = []
    for position, example in enumerate(examples, start=1):
        rows.append({
            "scope_type": scope_type,
            "scope_name": scope_name,
            "split": split,
            "position": position,
            **example,
        })
    return rows


def _predicted_re_binary(example: dict[str, Any]) -> int:
    predicted = str(example.get("predicted_code_or_label") or "").upper()
    return 1 if predicted == "RE" else 0


def _predicted_re_type(example: dict[str, Any]) -> str:
    subcode = str(example.get("predicted_subcode") or "").upper()
    if subcode == "RES":
        return "simple"
    if subcode == "REC":
        return "complex"
    return "non_re"


def _coerce_macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def build_calibration_summary(
    utterance_reviews: list[dict[str, Any]],
    latent_reviews: list[dict[str, Any]],
    group_reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute label-alignment and top-vs-control summaries."""
    y_true = [_predicted_re_binary(row) for row in utterance_reviews]
    y_pred = [1 if _review_strength(row["review"]) >= 0.5 else 0 for row in utterance_reviews]

    label_alignment = {
        "n_examples": len(utterance_reviews),
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "macro_f1": _coerce_macro_f1(y_true, y_pred) if y_true else 0.0,
        "cohens_kappa": float(cohen_kappa_score(y_true, y_pred)) if y_true else 0.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[1, 0]).tolist() if y_true else [[0, 0], [0, 0]],
        "labels": ["RE", "NonRE"],
    }

    subtype_rows = [
        row for row in utterance_reviews
        if _predicted_re_binary(row) == 1 and row["review"]["re_type"] in {"simple", "complex"}
    ]
    subtype_true = [_predicted_re_type(row) for row in subtype_rows]
    subtype_pred = [row["review"]["re_type"] for row in subtype_rows]
    subtype_alignment = {
        "n_examples": len(subtype_rows),
        "accuracy": float(accuracy_score(subtype_true, subtype_pred)) if subtype_true else 0.0,
        "macro_f1": float(f1_score(subtype_true, subtype_pred, average="macro", zero_division=0)) if subtype_true else 0.0,
        "cohens_kappa": float(cohen_kappa_score(subtype_true, subtype_pred)) if subtype_true else 0.0,
        "confusion_matrix": confusion_matrix(
            subtype_true,
            subtype_pred,
            labels=["simple", "complex"],
        ).tolist() if subtype_true else [[0, 0], [0, 0]],
        "labels": ["simple", "complex"],
    }

    return {
        "label_alignment": label_alignment,
        "subtype_alignment": subtype_alignment,
        "top_vs_control": {
            "latents": [
                {
                    "latent_idx": review["latent_idx"],
                    "judge_re_rate_gap": float(review["judge_re_rate"] - review["control_re_rate"]),
                    "clarity_gap": float(review["avg_clarity_score"] - review["control_avg_clarity_score"]),
                }
                for review in latent_reviews
            ],
            "groups": [
                {
                    "group_name": review["group_name"],
                    "judge_re_rate_gap": float(review["group_judge_re_rate"] - review["group_control_re_rate"]),
                    "clarity_gap": float(review["group_avg_clarity"] - review["group_control_avg_clarity"]),
                }
                for review in group_reviews
            ],
        },
    }


def build_report_markdown(
    *,
    manifest: dict[str, Any],
    latent_reviews: list[dict[str, Any]],
    group_reviews: list[dict[str, Any]],
    calibration: dict[str, Any],
) -> str:
    """Render the final Markdown report."""
    lines = [
        "# AI 专家代理评审报告",
        "",
        "## 方法声明",
        "",
        str(RE_JUDGE_RUBRIC["method_positioning"]["statement_zh"]),  # type: ignore[index]
        "",
        "## Judge 配置",
        "",
        f"- Bundle 版本: `{manifest.get('bundle_version')}`",
        f"- 数据规模: `{manifest.get('dataset_size')}`",
        f"- 聚合方式: `{manifest.get('aggregation')}`",
        f"- Hook 点: `{manifest.get('hook_point')}`",
        f"- SAE: `{manifest.get('sae_repo_id')} / {manifest.get('sae_subfolder')}`",
        "",
        "## RE 定义摘要",
        "",
        f"- Simple Reflection: {RE_JUDGE_RUBRIC['re_definition']['simple_reflection']}",  # type: ignore[index]
        f"- Complex Reflection: {RE_JUDGE_RUBRIC['re_definition']['complex_reflection']}",  # type: ignore[index]
        f"- 非 RE: {RE_JUDGE_RUBRIC['re_definition']['non_re']}",  # type: ignore[index]
        "",
        "## 校准结果",
        "",
        f"- RE/非RE accuracy: `{calibration['label_alignment']['accuracy']:.3f}`",
        f"- RE/非RE macro-F1: `{calibration['label_alignment']['macro_f1']:.3f}`",
        f"- RE/非RE Cohen's kappa: `{calibration['label_alignment']['cohens_kappa']:.3f}`",
        f"- Simple/Complex accuracy: `{calibration['subtype_alignment']['accuracy']:.3f}`",
        "",
        "## 最清晰的 RE latents",
        "",
    ]

    ranked = sorted(
        latent_reviews,
        key=lambda item: (
            item["final_latent_judgement"] != "clear_re_feature",
            -item["judge_re_rate"],
            -item["avg_clarity_score"],
        ),
    )
    for item in ranked[:10]:
        lines.append(
            f"- Latent `{item['latent_idx']}`: `{item['final_latent_judgement']}`, "
            f"judge_re_rate=`{item['judge_re_rate']:.3f}`, "
            f"clarity=`{item['avg_clarity_score']:.2f}`, "
            f"特征=`{item['shared_feature_name']}`"
        )

    lines.extend(["", "## 组级结论", ""])
    for item in group_reviews:
        lines.extend([
            f"### {item['group_name']}",
            f"- 组判断: `{item['final_group_judgement']}`",
            f"- judge_re_rate: `{item['group_judge_re_rate']:.3f}`",
            f"- avg_clarity: `{item['group_avg_clarity']:.2f}`",
            f"- distributed_subspace: `{item['is_distributed_re_subspace']}`",
            f"- 说明: {item['why_group_clearer_or_not_than_single_latent']}",
            "",
        ])

    lines.extend([
        "## 结论说明",
        "",
        "这份报告只能作为自动解释/专家代理评审证据。"
        "若要得出因果结论，仍需结合 ablation、steering 和对照实验。",
        "",
    ])
    return "\n".join(lines) + "\n"


def build_utterance_review_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    """Build the per-utterance judge prompt."""
    schema = {
        "has_clear_re_feature": "yes|partial|no",
        "re_type": "simple|complex|mixed|non_re|unclear",
        "clarity_score": 1,
        "dimension_scores": {
            "mirrors_client_meaning": 1,
            "adds_valid_meaning_or_empathy": 1,
            "non_directive_non_question": 1,
            "natural_therapeutic_language": 1,
        },
        "evidence_spans": ["..."],
        "reason_zh": "中文理由",
        "risk_flags": ["context_needed"],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are simulating a senior motivational interviewing expert. "
                "Judge whether a single counselor utterance clearly expresses reflective listening. "
                "Return JSON only and never mention hidden labels."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{_rubric_prompt_block()}\n\n"
                "Return valid JSON only with this schema:\n"
                f"{json.dumps(schema, ensure_ascii=False)}\n\n"
                f"Utterance:\n{example['text']}"
            ),
        },
    ]


def build_synthesis_messages(
    *,
    object_kind: str,
    object_name: str,
    reviewed_examples: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build the latent/group synthesis prompt."""
    payload = []
    for item in reviewed_examples:
        payload.append({
            "text": item["text"],
            "score_value": item["score_value"],
            "review": item["review"],
        })
    schema = {
        "shared_feature_name": "简洁特征名",
        "shared_feature_description_zh": "中文描述",
        "common_positive_evidence": ["..."],
        "common_counterevidence": ["..."],
        "failure_modes": ["..."],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are summarizing whether a latent or latent group has a coherent RE-related feature. "
                "Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{_rubric_prompt_block()}\n\n"
                f"Object kind: {object_kind}\n"
                f"Object name: {object_name}\n\n"
                "Reviewed utterances:\n"
                f"{json.dumps(payload, ensure_ascii=False)}\n\n"
                "Return valid JSON only with this schema:\n"
                f"{json.dumps(schema, ensure_ascii=False)}"
            ),
        },
    ]


def build_report_markdown(
    *,
    manifest: dict[str, Any],
    latent_reviews: list[dict[str, Any]],
    group_reviews: list[dict[str, Any]],
    calibration: dict[str, Any],
) -> str:
    """Render the final Markdown report."""
    lines = [
        "# AI 专家代理评审报告",
        "",
        "## 方法声明",
        "",
        str(RE_JUDGE_RUBRIC["method_positioning"]["statement_zh"]),  # type: ignore[index]
        "",
        "## Judge 配置",
        "",
        f"- Bundle 版本: `{manifest.get('bundle_version')}`",
        f"- 数据规模: `{manifest.get('dataset_size')}`",
        f"- 聚合方式: `{manifest.get('aggregation')}`",
        f"- Hook 点: `{manifest.get('hook_point')}`",
        f"- SAE: `{manifest.get('sae_repo_id')} / {manifest.get('sae_subfolder')}`",
        "",
        "## RE 定义摘要",
        "",
        f"- Simple Reflection: {RE_JUDGE_RUBRIC['re_definition']['simple_reflection']}",  # type: ignore[index]
        f"- Complex Reflection: {RE_JUDGE_RUBRIC['re_definition']['complex_reflection']}",  # type: ignore[index]
        f"- 非 RE: {RE_JUDGE_RUBRIC['re_definition']['non_re']}",  # type: ignore[index]
        "",
        "## 校准结果",
        "",
        f"- RE/NonRE accuracy: `{calibration['label_alignment']['accuracy']:.3f}`",
        f"- RE/NonRE macro-F1: `{calibration['label_alignment']['macro_f1']:.3f}`",
        f"- RE/NonRE Cohen's kappa: `{calibration['label_alignment']['cohens_kappa']:.3f}`",
        f"- Simple/Complex accuracy: `{calibration['subtype_alignment']['accuracy']:.3f}`",
        "",
        "## 最清晰的 RE latents",
        "",
    ]

    ranked = sorted(
        latent_reviews,
        key=lambda item: (
            item["final_latent_judgement"] != "clear_re_feature",
            -item["judge_re_rate"],
            -item["avg_clarity_score"],
        ),
    )
    for item in ranked[:10]:
        lines.append(
            f"- Latent `{item['latent_idx']}`: `{item['final_latent_judgement']}`, "
            f"judge_re_rate=`{item['judge_re_rate']:.3f}`, "
            f"clarity=`{item['avg_clarity_score']:.2f}`, "
            f"特征=`{item['shared_feature_name']}`"
        )

    lines.extend(["", "## 组级结论", ""])
    for item in group_reviews:
        lines.extend([
            f"### {item['group_name']}",
            f"- 组判断: `{item['final_group_judgement']}`",
            f"- judge_re_rate: `{item['group_judge_re_rate']:.3f}`",
            f"- avg_clarity: `{item['group_avg_clarity']:.2f}`",
            f"- distributed_subspace: `{item['is_distributed_re_subspace']}`",
            f"- 说明: {item['why_group_clearer_or_not_than_single_latent']}",
            "",
        ])

    lines.extend([
        "## 结论说明",
        "",
        "这份报告只能作为自动解释/专家代理评审证据。",
        "若要得出因果结论，仍需结合 ablation、steering 和对照实验。",
        "",
    ])
    return "\n".join(lines) + "\n"


def _summarise_scope_reviews(
    *,
    top_reviews: list[dict[str, Any]],
    control_reviews: list[dict[str, Any]],
) -> dict[str, float | str]:
    judge_re_rate = float(np.mean([_review_strength(item["review"]) for item in top_reviews])) if top_reviews else 0.0
    control_re_rate = float(np.mean([_review_strength(item["review"]) for item in control_reviews])) if control_reviews else 0.0
    avg_clarity = float(np.mean([item["review"]["clarity_score"] for item in top_reviews])) if top_reviews else 0.0
    control_avg_clarity = float(np.mean([item["review"]["clarity_score"] for item in control_reviews])) if control_reviews else 0.0
    return {
        "judge_re_rate": judge_re_rate,
        "control_re_rate": control_re_rate,
        "avg_clarity_score": avg_clarity,
        "control_avg_clarity_score": control_avg_clarity,
        "dominant_re_type": _dominant_re_type([item["review"] for item in top_reviews]),
        "lexical_template_only_rate": _flag_rate([item["review"] for item in top_reviews], "lexical_template_only"),
    }


def run_ai_judge_pipeline(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    model: str | None = None,
    top_latents: int = 20,
    top_n: int = 10,
    control_n: int = 5,
    groups: list[str] | tuple[str, ...] = DEFAULT_GROUP_NAMES,
    temperature: float = 0.0,
    max_retries: int = 3,
    request_timeout: int | None = None,
    dry_run_prompts: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    """Run the standalone AI judge pipeline against an exported bundle."""
    manifest, latent_records, group_records = _load_bundle(input_dir)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = out_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    selected_latent_records = latent_records[:top_latents]
    selected_group_names = [name for name in groups if name in group_records]

    resolved_model = model or os.getenv("OPENAI_MODEL")
    if dry_run_prompts:
        resolved_model = resolved_model or "dry-run-model"
    else:
        if client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required unless a custom client is provided.")
            if not resolved_model:
                raise RuntimeError("Judge model must be provided via --model or OPENAI_MODEL.")
            resolved_timeout = request_timeout
            if resolved_timeout is None:
                raw_timeout = os.getenv("OPENAI_HTTP_TIMEOUT") or os.getenv("JUDGE_REQUEST_TIMEOUT")
                resolved_timeout = int(raw_timeout) if raw_timeout else 600
            client = OpenAICompatibleChatClient(
                api_key=api_key,
                base_url=os.getenv("OPENAI_BASE_URL"),
                extra_body=_parse_json_env("OPENAI_EXTRA_BODY_JSON"),
                timeout=resolved_timeout,
            )
        elif not resolved_model:
            resolved_model = "custom-client-model"

    def _trim_examples(examples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        return list(examples[:limit])

    def _store_prompt(path: Path, messages: list[dict[str, str]]) -> None:
        _write_json(path, {"model": resolved_model, "messages": messages, "temperature": temperature})

    def _review_scope_examples(
        *,
        scope_type: str,
        scope_name: str,
        examples: list[dict[str, Any]],
        split: str,
    ) -> list[dict[str, Any]]:
        reviewed_rows: list[dict[str, Any]] = []
        for row in _iter_scope_examples(
            scope_type=scope_type,
            scope_name=scope_name,
            examples=examples,
            split=split,
        ):
            prompt_name = f"{scope_name}_{split}_{row['position']:02d}"
            messages = build_utterance_review_messages(row)
            _store_prompt(prompts_dir / f"{prompt_name}.json", messages)

            if dry_run_prompts:
                review = validate_utterance_review({})
            else:
                raw = _call_with_retry(
                    client=client,
                    model=resolved_model,
                    messages=messages,
                    temperature=temperature,
                    max_retries=max_retries,
                    log_prefix=prompt_name,
                    logs_dir=logs_dir,
                )
                review = validate_utterance_review(raw)

            reviewed_rows.append({**row, "review": review})
        return reviewed_rows

    utterance_reviews: list[dict[str, Any]] = []
    latent_reviews: list[dict[str, Any]] = []
    group_reviews: list[dict[str, Any]] = []

    for latent_record in selected_latent_records:
        latent_idx = int(latent_record["latent_idx"])
        scope_name = _make_scope_name("latent", latent_idx)
        top_examples = _trim_examples(latent_record.get("top_examples", []), top_n)
        control_examples = _trim_examples(latent_record.get("control_examples", []), control_n)

        top_reviews = _review_scope_examples(
            scope_type="latent",
            scope_name=scope_name,
            examples=top_examples,
            split="top",
        )
        control_reviews = _review_scope_examples(
            scope_type="latent",
            scope_name=scope_name,
            examples=control_examples,
            split="control",
        )
        utterance_reviews.extend(top_reviews)
        utterance_reviews.extend(control_reviews)

        synth_messages = build_synthesis_messages(
            object_kind="latent",
            object_name=str(latent_idx),
            reviewed_examples=top_reviews,
        )
        _store_prompt(prompts_dir / f"{scope_name}_synthesis.json", synth_messages)
        if dry_run_prompts:
            synthesis = validate_synthesis_review({})
        else:
            synth_raw = _call_with_retry(
                client=client,
                model=resolved_model,
                messages=synth_messages,
                temperature=temperature,
                max_retries=max_retries,
                log_prefix=f"{scope_name}_synthesis",
                logs_dir=logs_dir,
            )
            synthesis = validate_synthesis_review(synth_raw)

        summary = _summarise_scope_reviews(
            top_reviews=top_reviews,
            control_reviews=control_reviews,
        )
        latent_reviews.append({
            "latent_idx": latent_idx,
            "candidate_rank": int(latent_record.get("candidate_rank", 0)),
            "judge_re_rate": float(summary["judge_re_rate"]),
            "control_re_rate": float(summary["control_re_rate"]),
            "avg_clarity_score": float(summary["avg_clarity_score"]),
            "control_avg_clarity_score": float(summary["control_avg_clarity_score"]),
            "dominant_re_type": str(summary["dominant_re_type"]),
            "shared_feature_name": synthesis["shared_feature_name"],
            "shared_feature_description_zh": synthesis["shared_feature_description_zh"],
            "common_positive_evidence": synthesis["common_positive_evidence"],
            "common_counterevidence": synthesis["common_counterevidence"],
            "failure_modes": synthesis["failure_modes"],
            "lexical_template_only_rate": float(summary["lexical_template_only_rate"]),
            "final_latent_judgement": classify_latent_review(
                judge_re_rate=float(summary["judge_re_rate"]),
                control_re_rate=float(summary["control_re_rate"]),
                avg_clarity_score=float(summary["avg_clarity_score"]),
                lexical_template_only_rate=float(summary["lexical_template_only_rate"]),
            ),
        })

    for group_name in selected_group_names:
        group_record = group_records[group_name]
        scope_name = _make_scope_name("group", group_name)
        top_examples = _trim_examples(group_record.get("top_examples", []), top_n)
        control_examples = _trim_examples(group_record.get("control_examples", []), control_n)

        top_reviews = _review_scope_examples(
            scope_type="group",
            scope_name=scope_name,
            examples=top_examples,
            split="top",
        )
        control_reviews = _review_scope_examples(
            scope_type="group",
            scope_name=scope_name,
            examples=control_examples,
            split="control",
        )
        utterance_reviews.extend(top_reviews)
        utterance_reviews.extend(control_reviews)

        synth_messages = build_synthesis_messages(
            object_kind="group",
            object_name=group_name,
            reviewed_examples=top_reviews,
        )
        _store_prompt(prompts_dir / f"{scope_name}_synthesis.json", synth_messages)
        if dry_run_prompts:
            synthesis = validate_synthesis_review({})
        else:
            synth_raw = _call_with_retry(
                client=client,
                model=resolved_model,
                messages=synth_messages,
                temperature=temperature,
                max_retries=max_retries,
                log_prefix=f"{scope_name}_synthesis",
                logs_dir=logs_dir,
            )
            synthesis = validate_synthesis_review(synth_raw)

        summary = _summarise_scope_reviews(
            top_reviews=top_reviews,
            control_reviews=control_reviews,
        )
        group_reviews.append({
            "group_name": group_name,
            "latent_ids": [int(idx) for idx in group_record.get("latent_ids", [])],
            "weights": [float(w) for w in group_record.get("weights", [])],
            "group_judge_re_rate": float(summary["judge_re_rate"]),
            "group_control_re_rate": float(summary["control_re_rate"]),
            "group_avg_clarity": float(summary["avg_clarity_score"]),
            "group_control_avg_clarity": float(summary["control_avg_clarity_score"]),
            "group_feature_name": synthesis["shared_feature_name"],
            "group_feature_description_zh": synthesis["shared_feature_description_zh"],
            "common_positive_evidence": synthesis["common_positive_evidence"],
            "common_counterevidence": synthesis["common_counterevidence"],
            "failure_modes": synthesis["failure_modes"],
            "final_group_judgement": classify_group_review(
                judge_re_rate=float(summary["judge_re_rate"]),
                control_re_rate=float(summary["control_re_rate"]),
                avg_clarity_score=float(summary["avg_clarity_score"]),
            ),
            "is_distributed_re_subspace": False,
            "why_group_clearer_or_not_than_single_latent": "",
        })

    baseline_g1 = next((item for item in group_reviews if item["group_name"] == "G1"), None)
    for item in group_reviews:
        if item["group_name"] == "G1":
            item["why_group_clearer_or_not_than_single_latent"] = "G1 is the single-latent baseline group."
            continue
        if baseline_g1 is None:
            item["why_group_clearer_or_not_than_single_latent"] = "G1 baseline is unavailable in this bundle."
            continue

        better_rate = item["group_judge_re_rate"] >= baseline_g1["group_judge_re_rate"] + 0.10
        better_clarity = item["group_avg_clarity"] >= baseline_g1["group_avg_clarity"] + 0.50
        item["is_distributed_re_subspace"] = bool(better_rate and better_clarity)
        if item["is_distributed_re_subspace"]:
            item["why_group_clearer_or_not_than_single_latent"] = (
                f"{item['group_name']} exceeds G1 on both judge_re_rate and clarity, "
                "suggesting a more distributed RE subspace."
            )
        else:
            item["why_group_clearer_or_not_than_single_latent"] = (
                f"{item['group_name']} does not exceed G1 by the required margin on both "
                "judge_re_rate and clarity."
            )

    calibration = build_calibration_summary(
        utterance_reviews=utterance_reviews,
        latent_reviews=latent_reviews,
        group_reviews=group_reviews,
    )

    _write_jsonl(out_dir / "utterance_reviews.jsonl", utterance_reviews)
    _write_json(out_dir / "latent_reviews.json", latent_reviews)
    _write_json(out_dir / "group_reviews.json", group_reviews)
    _write_json(out_dir / "calibration.json", calibration)

    report_text = build_report_markdown(
        manifest={**manifest, "judge_model": resolved_model, "dry_run_prompts": dry_run_prompts},
        latent_reviews=latent_reviews,
        group_reviews=group_reviews,
        calibration=calibration,
    )
    (out_dir / "report.md").write_text(report_text, encoding="utf-8")

    if dry_run_prompts:
        _write_json(
            out_dir / "dry_run_manifest.json",
            {
                "status": "dry_run",
                "judge_model": resolved_model,
                "latent_count": len(selected_latent_records),
                "group_count": len(selected_group_names),
                "prompt_dir": str(prompts_dir),
            },
        )

    return {
        "status": "dry_run" if dry_run_prompts else "completed",
        "judge_model": resolved_model,
        "bundle_version": manifest.get("bundle_version"),
        "n_utterance_reviews": len(utterance_reviews),
        "n_latent_reviews": len(latent_reviews),
        "n_group_reviews": len(group_reviews),
        "output_dir": str(out_dir),
    }
