from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import torch

from src.nlp_re_base.contrastive_evidence_pack import read_jsonl, write_jsonl
from src.nlp_re_base.deepseek_top50_induction import (
    DeepSeekTop50Config,
    build_deepseek_top50_tasks,
    run_deepseek_top50_tasks,
    validate_deepseek_top50_outputs,
)


class _FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _fake_request(*, url: str, headers: dict, payload: dict, timeout: float) -> _FakeResponse:
    assert url.endswith("/chat/completions")
    assert headers["Authorization"] == "Bearer test-key"
    assert payload["model"] == "deepseek-v4-flash"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["thinking"] == {"type": "disabled"}
    prompt = payload["messages"][1]["content"]
    latent_idx = int(re.search(r"latent_idx=(\d+)", prompt).group(1))
    ids = [f"s{index:03d}" for index in range(1, 51)]
    output = {
        "latent_idx": latent_idx,
        "short_name": "repeated candidate pattern",
        "candidate_explanation": "A synthetic repeated-content candidate pattern.",
        "surface_pattern": "A stable synthetic phrase template.",
        "surface_supporting_sample_ids": ids,
        "surface_outlier_sample_ids": [],
        "surface_representative_evidence_ids": ids[:2],
        "surface_confidence": 0.7,
        "semantic_pattern": "A repeated synthetic message content.",
        "semantic_supporting_sample_ids": ids,
        "semantic_outlier_sample_ids": [],
        "semantic_representative_evidence_ids": ids[2:4],
        "semantic_confidence": 0.7,
        "alternative_hypotheses": ["the synthetic prefix alone"],
        "failure_modes": ["this smoke payload has no real semantic evidence"],
    }
    if latent_idx == 3:
        output["semantic_pattern"] = "No synthetic semantic pattern is supported."
        output["semantic_supporting_sample_ids"] = []
        output["semantic_outlier_sample_ids"] = ids
        output["semantic_representative_evidence_ids"] = []
        output["semantic_confidence"] = 0.1
    return _FakeResponse(
        {
            "id": f"request-{latent_idx}",
            "model": "deepseek-v4-flash",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "choices": [{"message": {"content": json.dumps(output)}, "finish_reason": "stop"}],
        }
    )


def _write_inputs(root: Path) -> dict[str, Path]:
    records = [{"unit_text": f"synthetic sentence {index}"} for index in range(60)]
    records.extend({"unit_text": f"synthetic sentence {index}"} for index in range(4))
    records_path = root / "records.jsonl"
    write_jsonl(records_path, records)
    features = torch.zeros((64, 4), dtype=torch.float16)
    features[:, 1] = torch.linspace(5, 1, 64)
    features[60:, 1] = torch.tensor([6.0, 5.9, 5.8, 5.7])
    features[:, 3] = torch.linspace(4, 0, 64)
    feature_path = root / "features.pt"
    torch.save(features, feature_path)
    stable_path = root / "stable.csv"
    stable_path.write_text(
        "label,latent_idx,stable_set_role,inclusion_frequency,abs_cohens_d\n"
        "RE,1,stable_core,1.0,1.2\n"
        "REC,1,stable_core,1.0,1.1\n"
        "QU,3,stable_core,0.9,0.8\n",
        encoding="utf-8",
    )
    return {"records": records_path, "features": feature_path, "stable": stable_path}


def test_build_run_validate(root: Path) -> None:
    inputs = _write_inputs(root)
    output_dir = root / "output"
    config = DeepSeekTop50Config(concurrency=2, max_retries=0, timeout_seconds=1.0)
    build = build_deepseek_top50_tasks(
        stable_latents_path=inputs["stable"],
        feature_store_path=inputs["features"],
        records_path=inputs["records"],
        output_dir=output_dir,
        config=config,
    )
    assert build["n_stable_label_latent_rows"] == 3
    assert build["n_unique_latents"] == 2
    assert build["n_tasks"] == 2
    assert build["selection_unit"] == "unique_normalized_text"
    task_path = Path(build["outputs"]["tasks"])
    task_text = task_path.read_text(encoding="utf-8")
    assert "\"RE\"" not in task_text
    assert "\"REC\"" not in task_text
    task = read_jsonl(task_path)[0]
    visible = task["visible_samples"]
    assert len(visible) == 50
    assert len({sample["text"] for sample in visible}) == 50
    assert visible[0]["text"] == "synthetic sentence 0"
    assert all("duplicate_group" not in sample for sample in visible)
    run = run_deepseek_top50_tasks(
        tasks_path=task_path,
        output_dir=output_dir,
        api_key="test-key",
        config=config,
        request_fn=_fake_request,
    )
    assert run["counts"] == {"success": 2, "skipped": 0, "failed": 0}
    resumed = run_deepseek_top50_tasks(
        tasks_path=task_path,
        output_dir=output_dir,
        api_key="test-key",
        config=config,
        request_fn=_fake_request,
    )
    assert resumed["counts"] == {"success": 0, "skipped": 2, "failed": 0}
    validation = validate_deepseek_top50_outputs(
        tasks_path=task_path,
        execution_manifest_path=output_dir / "llm_execution_manifest.jsonl",
        output_dir=output_dir,
    )
    assert validation["n_tasks"] == 2
    assert validation["n_valid"] == 2
    assert validation["n_failed"] == 0
    Path(task["expected_output_path"]).write_text('{"broken": ', encoding="utf-8")
    malformed_validation = validate_deepseek_top50_outputs(
        tasks_path=task_path,
        execution_manifest_path=output_dir / "llm_execution_manifest.jsonl",
        output_dir=output_dir,
    )
    assert malformed_validation["n_valid"] == 1
    assert malformed_validation["n_failed"] == 1


def main() -> None:
    root = (Path.cwd() / "outputs" / "_smoke_deepseek_top50_induction").resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        test_build_run_validate(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("test_deepseek_top50_induction_smoke passed")


if __name__ == "__main__":
    main()
