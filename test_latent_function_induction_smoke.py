from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from run_misc_latent_function_induction import run_latent_function_induction


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _safe_smoke_root() -> Path:
    root = (Path.cwd() / "outputs" / "_smoke_latent_function_induction").resolve()
    cwd = Path.cwd().resolve()
    if cwd not in root.parents:
        raise RuntimeError(f"Refusing to use smoke output outside repo: {root}")
    return root


def _packet(packet_id: str, label: str, latent_idx: int, alias: str) -> tuple[dict, dict, dict]:
    blind_examples = [
        {
            "packet_id": packet_id,
            "latent_alias": alias,
            "example_group": "top_activating",
            "rank_within_group": 1,
            "activation": 3.2,
            "unit_text": "can you tell me more about that",
            "normalized_text": "can you tell me more about that",
            "duplicate_text_within_packet": False,
        },
        {
            "packet_id": packet_id,
            "latent_alias": alias,
            "example_group": "high_non_target",
            "rank_within_group": 1,
            "activation": 2.4,
            "unit_text": "do you have any questions",
            "normalized_text": "do you have any questions",
            "duplicate_text_within_packet": False,
        },
        {
            "packet_id": packet_id,
            "latent_alias": alias,
            "example_group": "random_target",
            "rank_within_group": 1,
            "activation": 0.1,
            "unit_text": "what would you like to talk about",
            "normalized_text": "what would you like to talk about",
            "duplicate_text_within_packet": False,
        },
    ]
    labeled_examples = []
    for ex in blind_examples:
        item = dict(ex)
        item.update(
            {
                "target_label": label,
                "latent_idx": latent_idx,
                "rank_within_label": 1,
                "target_match": 1 if ex["example_group"] != "high_non_target" else 0,
                "active_labels": label,
                "record_id": f"r-{packet_id}",
                "file_id": "f1",
                "source_line": 1,
                "source_split": "high",
                "quality_label": "high",
                "predicted_code": label,
                "predicted_subcode": "",
                "confidence": 0.9,
            }
        )
        labeled_examples.append(item)

    blind_packet = {
        "packet_id": packet_id,
        "latent_alias": alias,
        "client_context_available": False,
        "context_note": "Only counselor current utterance is available in phase 1.",
        "summary": {"n_top_activating": 1, "n_high_non_target": 1, "n_random_target": 1},
        "examples": blind_examples,
    }
    labeled_packet = {
        "packet_id": packet_id,
        "target_label": label,
        "latent_idx": latent_idx,
        "rank_within_label": 1,
        "cohens_d": 1.0,
        "directional_auc": 0.8,
        "precision_at_50": 0.7,
        "client_context_available": False,
        "context_note": "Only counselor current utterance is available in phase 1.",
        "summary": {"top_activating_target_match_rate": 1.0},
        "examples": labeled_examples,
    }
    summary = {
        "packet_id": packet_id,
        "latent_alias": alias,
        "target_label": label,
        "latent_idx": latent_idx,
        "rank_within_label": 1,
        "top_activating_target_match_rate": 1.0,
        "active_label_counts_top_activating": f"{label}:1",
        "duplicate_text_row_count": 0,
        "unique_file_count": 1,
    }
    return blind_packet, labeled_packet, summary


class FakeClient:
    def __init__(self, *, invalid_on_call: int | None = None) -> None:
        self.calls: list[dict] = []
        self.invalid_on_call = invalid_on_call

    def chat(self, *, model: str, messages: list[dict[str, str]], temperature: float = 0.0):
        self.calls.append({"model": model, "messages": messages, "temperature": temperature})
        if self.invalid_on_call is not None and len(self.calls) == self.invalid_on_call:
            return "not json", {"bad": True}
        if len(self.calls) % 2 == 1:
            payload = {
                "tentative_interpretation": "This latent appears to be associated with question-like invitations.",
                "patterns": [
                    {
                        "pattern_name": "question wording",
                        "pattern_type": "surface_form",
                        "evidence": "Repeated can/what question forms.",
                    },
                    {
                        "pattern_name": "elaboration invitation",
                        "pattern_type": "dialogue_function",
                        "evidence": "Examples invite the client to expand.",
                    },
                ],
                "artifact_risks": ["May track question syntax."],
                "evidence_quality": "medium",
                "candidate_name": "question-form / elaboration invitation",
                "alternative_explanations": ["May track can/what words.", "May track short questions."],
                "recommended_followup_checks": ["Inspect triggering tokens."],
                "final_conclusion": "This is a candidate surface/function pattern.",
                "context_limitation_note": "Only counselor current utterance is available.",
            }
        else:
            payload = {
                "target_label_relationship": "The candidate pattern is partly consistent with the target label, but support is surface-form heavy.",
                "adjacent_label_risks": ["QUO vs QUC confusion."],
                "artifact_risks": ["Question syntax artifact."],
                "evidence_quality": "medium",
                "final_conclusion": "The latent may help explain label decodability, but not as causal mechanism evidence.",
                "context_limitation_note": "Only counselor current utterance is available; prior client context is unavailable.",
            }
        return json.dumps(payload), {"choices": [{"message": {"content": json.dumps(payload)}}]}


def test_dry_run_prompts_and_outputs() -> None:
    root = _safe_smoke_root()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        blind, labeled, summary = _packet("packet_0001", "QUO", 10, "latent_0001")
        _write_jsonl(root / "blind.jsonl", [blind])
        _write_jsonl(root / "labeled.jsonl", [labeled])
        pd.DataFrame([summary]).to_csv(root / "summary.csv", index=False)

        manifest = run_latent_function_induction(
            blind_packets_path=root / "blind.jsonl",
            labeled_packets_path=root / "labeled.jsonl",
            summary_path=root / "summary.csv",
            output_dir=root / "dry",
            dry_run_prompts=True,
            max_examples_per_group=1,
        )

        assert manifest["n_pending_dry_run"] == 1
        assert (root / "dry" / "latent_function_reviews.jsonl").exists()
        assert (root / "dry" / "latent_function_reviews.csv").exists()
        assert (root / "dry" / "latent_function_patterns.csv").exists()
        assert (root / "dry" / "label_function_cluster_summary.csv").exists()
        assert (root / "dry" / "latent_function_induction_report.md").exists()
        assert (root / "dry" / "manifest.json").exists()

        blind_prompt = (root / "dry" / "prompts" / "packet_0001_blind_prompt.json").read_text(encoding="utf-8")
        assert "target_label" not in blind_prompt
        assert "active_labels" not in blind_prompt
        assert "predicted_code" not in blind_prompt
        assert "rationale" not in blind_prompt
        assert "QUO" not in blind_prompt
        assert "contrast_high" in blind_prompt
        assert "comparison_random" in blind_prompt

        labeled_prompt = (root / "dry" / "prompts" / "packet_0001_labeled_prompt.json").read_text(encoding="utf-8")
        assert "target_label" in labeled_prompt
        assert "top_activating_target_match_rate" in labeled_prompt
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_api_parse_and_failed_review() -> None:
    root = _safe_smoke_root()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        blind1, labeled1, summary1 = _packet("packet_0001", "RE", 11, "latent_0001")
        blind2, labeled2, summary2 = _packet("packet_0002", "QUC", 12, "latent_0002")
        _write_jsonl(root / "blind.jsonl", [blind1, blind2])
        _write_jsonl(root / "labeled.jsonl", [labeled1, labeled2])
        pd.DataFrame([summary1, summary2]).to_csv(root / "summary.csv", index=False)

        manifest = run_latent_function_induction(
            blind_packets_path=root / "blind.jsonl",
            labeled_packets_path=root / "labeled.jsonl",
            summary_path=root / "summary.csv",
            output_dir=root / "api",
            dry_run_prompts=False,
            max_examples_per_group=1,
            client=FakeClient(invalid_on_call=3),
        )

        assert manifest["n_success"] == 1
        assert manifest["n_failed"] == 1
        reviews = pd.read_csv(root / "api" / "latent_function_reviews.csv")
        assert set(reviews["status"].tolist()) == {"ok", "failed"}
        ok = reviews[reviews["status"] == "ok"].iloc[0]
        assert ok["confidence"] == "medium"
        assert "client" in str(ok["context_limitation_note"]).lower()
        patterns = pd.read_csv(root / "api" / "latent_function_patterns.csv")
        assert "surface_form" in patterns["pattern_type"].tolist()
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_dry_run_prompts_and_outputs()
    test_api_parse_and_failed_review()
    print("test_latent_function_induction_smoke passed")
