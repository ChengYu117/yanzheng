from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from src.nlp_re_base.contrastive_quality_audit import audit_explainer_quality


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _canonical_hash(payload: dict) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_quality_and_provenance_gate(root: Path) -> None:
    prompt = """You are explaining an anonymous latent.
Samples:
- id=s001 tag=ACTIVE_HIGH activation=3.0
  text: please tell me what happened
- id=s002 tag=NONACTIVE_NEAR_MISS activation=0.0
  text: the weather is fine today
Return only one JSON object.
"""
    task = {
        "task_id": "ctli_0001_explainer_r01",
        "packet_id": "ctli_0001",
        "latent_idx": 7,
        "repeat": 1,
        "prompt": prompt,
        "expected_output_path": str(root / "raw" / "ctli_0001_explainer_r01.json"),
    }
    explanation = {
        "task_id": task["task_id"],
        "packet_id": task["packet_id"],
        "latent_idx": 7,
        "repeat": 1,
        "short_name": "client-directed inquiry",
        "main_hypothesis": "The latent responds to a client-directed inquiry that asks for an account of an event; it does not trigger on unrelated weather statements.",
        "positive_triggers": ["direct request for the client's account", "open inquiry about an event"],
        "explicit_exclusions": ["weather statements without a client-directed inquiry"],
        "possible_surface_confounds": ["question wording may be a surface correlate"],
        "feature_type": "pragmatic",
        "confidence": 0.65,
        "alternative_hypotheses": ["the latent may track question framing rather than inquiry function"],
        "key_evidence": ["s001", "s002"],
        "failure_modes": ["the two examples are too small to establish generality"],
    }
    raw_path = Path(task["expected_output_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps({k: v for k, v in explanation.items() if k not in {"task_id", "packet_id", "repeat"}}, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "task_id": task["task_id"],
        "model": "claude-test",
        "execution_mode": "claude_code_llm",
        "timestamp": "2026-01-01T00:00:00Z",
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "raw_output_sha256": _canonical_hash({k: v for k, v in explanation.items() if k not in {"task_id", "packet_id", "repeat"}}),
    }
    _write_jsonl(root / "tasks.jsonl", [task])
    _write_jsonl(root / "validated.jsonl", [explanation])
    _write_jsonl(root / "manifest.jsonl", [manifest])

    clean_repo = root / "clean_repo"
    clean_repo.mkdir()
    result = audit_explainer_quality(
        tasks_path=root / "tasks.jsonl",
        validated_explanations_path=root / "validated.jsonl",
        execution_manifest_path=root / "manifest.jsonl",
        output_dir=root / "audit",
        repo_root=clean_repo,
    )
    assert result["stage_gate_complete"] is True
    assert result["n_trusted_for_downstream"] == 1
    assert (root / "audit" / "trusted_explanations.jsonl").exists()


def test_contamination_and_bad_evidence_are_blocked(root: Path) -> None:
    prompt = """Samples:
- id=s001 tag=ACTIVE_HIGH activation=3
  text: an active example
- id=s002 tag=NONACTIVE_NEAR_MISS activation=0
  text: a near miss example
"""
    task = {
        "task_id": "ctli_0002_explainer_r01",
        "packet_id": "ctli_0002",
        "latent_idx": 8,
        "repeat": 1,
        "prompt": prompt,
        "expected_output_path": str(root / "raw" / "bad.json"),
    }
    bad = {
        "task_id": task["task_id"], "packet_id": task["packet_id"], "latent_idx": 8, "repeat": 1,
        "short_name": "bad", "main_hypothesis": "This is a sufficiently long but generic explanation for the latent.",
        "positive_triggers": ["a", "b"], "explicit_exclusions": ["none"],
        "possible_surface_confounds": ["length"], "feature_type": "unclear", "confidence": 0.9,
        "alternative_hypotheses": ["another possibility"], "key_evidence": ["not-in-prompt"],
        "failure_modes": "[\"serialized list\"]",
    }
    raw = {k: v for k, v in bad.items() if k not in {"task_id", "packet_id", "repeat"}}
    raw_path = Path(task["expected_output_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    _write_jsonl(root / "tasks.jsonl", [task])
    _write_jsonl(root / "validated.jsonl", [bad])
    _write_jsonl(root / "manifest.jsonl", [{
        "task_id": task["task_id"], "model": "claude-test",
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "raw_output_sha256": _canonical_hash(raw),
    }])
    repo = root / "repo"
    repo.mkdir()
    (repo / "process_explainer_tasks_fake.py").write_text(
        "def generate_explanation(): pass\nMODEL_NAME = \"claude-opus-test\"\n"
        "llm_execution_manifest.jsonl\ncontrastive_latent_interp\n",
        encoding="utf-8",
    )
    result = audit_explainer_quality(
        tasks_path=root / "tasks.jsonl", validated_explanations_path=root / "validated.jsonl",
        execution_manifest_path=root / "manifest.jsonl", output_dir=root / "audit", repo_root=repo,
    )
    assert result["contamination_detected"] is True
    assert result["stage_gate_complete"] is False
    assert result["n_trusted_for_downstream"] == 0


def main() -> None:
    root = (Path.cwd() / "outputs" / "_smoke_contrastive_quality_audit").resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        test_quality_and_provenance_gate(root / "good")
        test_contamination_and_bad_evidence_are_blocked(root / "bad")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("test_contrastive_quality_audit_smoke passed")


if __name__ == "__main__":
    main()
