"""Smoke tests for the standalone SAE-RE AI judge pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _build_synthetic_bundle(tmpdir: str) -> Path:
    from nlp_re_base.ai_re_judge import export_judge_bundle

    texts = [
        "I hear how exhausted you feel right now.",
        "It sounds like part of you wants relief and part of you feels stuck.",
        "You are carrying a lot of pressure.",
        "It makes sense that this has been overwhelming.",
        "You want change but you are scared of what comes next.",
        "Why did you do that?",
        "You should just make a plan.",
        "Here is some information about stress management.",
        "Can you tell me more?",
        "Try to calm down and focus.",
    ]
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    records = [
        {
            "predicted_code": "RE" if label == 1 else "NonRE",
            "predicted_subcode": "REC" if i % 2 == 0 and label == 1 else ("RES" if label == 1 else "QUC"),
            "rationale": f"rationale-{i}",
        }
        for i, label in enumerate(labels)
    ]

    features = np.zeros((len(texts), 12), dtype=np.float32)
    features[:5, 0] = np.array([3.0, 2.8, 2.5, 2.2, 2.0], dtype=np.float32)
    features[:5, 1] = np.array([2.0, 2.2, 1.8, 1.7, 1.5], dtype=np.float32)
    features[:5, 2] = np.array([1.5, 1.4, 1.3, 1.2, 1.1], dtype=np.float32)
    features[5:, 0] = np.array([0.2, 0.1, 0.0, 0.1, 0.2], dtype=np.float32)
    features[5:, 1] = np.array([0.2, 0.3, 0.2, 0.1, 0.0], dtype=np.float32)
    features[5:, 2] = np.array([0.1, 0.1, 0.2, 0.1, 0.1], dtype=np.float32)

    candidate_df = pd.DataFrame(
        {
            "latent_idx": list(range(features.shape[1])),
            "cohens_d": np.linspace(3.0, 0.1, features.shape[1]),
            "abs_cohens_d": np.linspace(3.0, 0.1, features.shape[1]),
            "auc": np.linspace(0.95, 0.5, features.shape[1]),
            "p_value": np.linspace(0.001, 0.3, features.shape[1]),
            "significant_fdr": [True] * features.shape[1],
        }
    )

    return export_judge_bundle(
        output_dir=tmpdir,
        candidate_df=candidate_df,
        utterance_features=features,
        texts=texts,
        labels=labels,
        records=records,
        aggregation="max",
        hook_point="blocks.19.hook_resid_post",
        model_name="dummy-model",
        sae_repo_id="dummy/repo",
        sae_subfolder="layer19",
        group_weights={
            "G1": [1.0],
            "G5": [0.35, 0.25, 0.2, 0.1, 0.1],
            "G20": [1.0 / 12] * 12,
        },
        top_latents=6,
        top_n=4,
        control_n=2,
    )


def test_compute_group_scores():
    from nlp_re_base.ai_re_judge import compute_group_scores

    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.0],
            [3.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    scores, weights = compute_group_scores(X, latent_ids=[0, 1], weights=[2.0, 1.0])
    assert scores.shape == (3,)
    assert abs(sum(weights) - 1.0) < 1e-6
    print("  PASS test_compute_group_scores")


def test_classify_latent_review():
    from nlp_re_base.ai_re_judge import classify_latent_review

    assert classify_latent_review(
        judge_re_rate=0.8,
        control_re_rate=0.2,
        avg_clarity_score=4.3,
        lexical_template_only_rate=0.1,
    ) == "clear_re_feature"
    assert classify_latent_review(
        judge_re_rate=0.45,
        control_re_rate=0.25,
        avg_clarity_score=3.0,
        lexical_template_only_rate=0.5,
    ) == "mixed_feature"
    print("  PASS test_classify_latent_review")


def test_run_ai_judge_dry_run():
    from nlp_re_base.ai_re_judge import run_ai_judge_pipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = _build_synthetic_bundle(tmpdir)
        output_dir = Path(tmpdir) / "ai_judge_dry"
        results = run_ai_judge_pipeline(
            input_dir=bundle_path,
            output_dir=output_dir,
            model="dry-run-model",
            top_latents=3,
            top_n=3,
            control_n=2,
            dry_run_prompts=True,
        )
        assert results["status"] == "dry_run"
        assert (output_dir / "prompts").exists()
        assert (output_dir / "report.md").exists()
        assert (output_dir / "dry_run_manifest.json").exists()
    print("  PASS test_run_ai_judge_dry_run")


class FakeJudgeClient:
    def chat(self, *, model: str, messages: list[dict[str, str]], temperature: float = 0.0):
        user_content = messages[-1]["content"]
        if "Utterance:\n" in user_content:
            text = user_content.split("Utterance:\n", 1)[1].strip().lower()
            positive = any(token in text for token in ["hear", "sounds like", "overwhelming", "pressure", "stuck"])
            payload = {
                "has_clear_re_feature": "yes" if positive else "no",
                "re_type": "complex" if "part of you" in text or "stuck" in text else ("simple" if positive else "non_re"),
                "clarity_score": 5 if positive else 1,
                "dimension_scores": {
                    "mirrors_client_meaning": 5 if positive else 1,
                    "adds_valid_meaning_or_empathy": 4 if positive else 1,
                    "non_directive_non_question": 5 if positive else 1,
                    "natural_therapeutic_language": 4 if positive else 1,
                },
                "evidence_spans": ["reflective wording"] if positive else [],
                "reason_zh": "呈现明显反映" if positive else "更像非反映性回应",
                "risk_flags": [] if positive else ["advice_like"],
            }
        else:
            payload = {
                "shared_feature_name": "RE-style empathy",
                "shared_feature_description_zh": "这一组样本大多在反映来访者感受或意义。",
                "common_positive_evidence": ["反映情绪", "非指令性语言"],
                "common_counterevidence": ["少数样本依赖模板句式"],
                "failure_modes": ["可能受到固定表达影响"],
            }
        raw = {"choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}]}
        return json.dumps(payload, ensure_ascii=False), raw


def test_run_ai_judge_with_mock_client():
    from nlp_re_base.ai_re_judge import run_ai_judge_pipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = _build_synthetic_bundle(tmpdir)
        output_dir = Path(tmpdir) / "ai_judge_live"
        results = run_ai_judge_pipeline(
            input_dir=bundle_path,
            output_dir=output_dir,
            model="mock-model",
            top_latents=3,
            top_n=3,
            control_n=2,
            client=FakeJudgeClient(),
        )
        assert results["status"] == "completed"
        assert (output_dir / "utterance_reviews.jsonl").exists()
        assert (output_dir / "latent_reviews.json").exists()
        assert (output_dir / "group_reviews.json").exists()
        assert (output_dir / "calibration.json").exists()

        latent_reviews = json.loads((output_dir / "latent_reviews.json").read_text(encoding="utf-8"))
        assert len(latent_reviews) == 3
        assert "final_latent_judgement" in latent_reviews[0]
    print("  PASS test_run_ai_judge_with_mock_client")


class _FakeHttpResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(
            {"choices": [{"message": {"content": json.dumps({"ok": True})}}]}
        ).encode("utf-8")


def test_openai_client_extra_body():
    from nlp_re_base.ai_re_judge import OpenAICompatibleChatClient

    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeHttpResponse()

    client = OpenAICompatibleChatClient(
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        extra_body={"enable_thinking": False},
    )
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        content, raw = client.chat(
            model="qwen3.5-plus",
            messages=[{"role": "user", "content": "Return JSON."}],
            temperature=0.0,
        )

    assert captured["url"].endswith("/chat/completions")
    assert captured["payload"]["model"] == "qwen3.5-plus"
    assert captured["payload"]["enable_thinking"] is False
    assert json.loads(content) == {"ok": True}
    assert "choices" in raw
    print("  PASS test_openai_client_extra_body")


def test_call_with_retry_timeout_then_success():
    from nlp_re_base.ai_re_judge import _call_with_retry

    class FlakyClient:
        def __init__(self):
            self.calls = 0

        def chat(self, *, model: str, messages: list[dict[str, str]], temperature: float = 0.0):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("simulated timeout")
            payload = {"choices": [{"message": {"content": json.dumps({"ok": True})}}]}
            return json.dumps({"ok": True}), payload

    with tempfile.TemporaryDirectory() as tmpdir:
        logs_dir = Path(tmpdir)
        raw = _call_with_retry(
            client=FlakyClient(),
            model="mock-model",
            messages=[{"role": "user", "content": "Return JSON."}],
            temperature=0.0,
            max_retries=2,
            log_prefix="timeout_retry",
            logs_dir=logs_dir,
        )
        assert raw == {"ok": True}
        assert (logs_dir / "timeout_retry_error_attempt_1.json").exists()
        assert (logs_dir / "timeout_retry_response_attempt_2.json").exists()
    print("  PASS test_call_with_retry_timeout_then_success")


def main() -> int:
    tests = [
        ("compute_group_scores", test_compute_group_scores),
        ("classify_latent_review", test_classify_latent_review),
        ("run_ai_judge_dry_run", test_run_ai_judge_dry_run),
        ("run_ai_judge_with_mock_client", test_run_ai_judge_with_mock_client),
        ("openai_client_extra_body", test_openai_client_extra_body),
        ("call_with_retry_timeout_then_success", test_call_with_retry_timeout_then_success),
    ]
    passed = 0
    failed = 0
    print("\n=== AI Judge Smoke Tests ===\n")
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as exc:  # pragma: no cover - manual smoke output
            failed += 1
            print(f"  FAIL {name}: {exc}")
            traceback.print_exc()
    print(f"\nResults: {passed} passed, {failed} failed, {len(tests)} total")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
