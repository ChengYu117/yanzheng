from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import torch

from src.nlp_re_base.contrastive_evidence_pack import read_jsonl, write_jsonl
from src.nlp_re_base.deepseek_latent_cards import (
    LATENT_CARD_SYSTEM_PROMPT,
    build_latent_card_tasks,
    render_stable_core_top5_human_review_document,
    validate_latent_card_outputs,
)
from src.nlp_re_base.deepseek_top50_induction import DeepSeekTop50Config, run_deepseek_top50_tasks


class _Response:
    status_code = 200
    text = ""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def json(self) -> dict:
        return self.payload


def _request(*, url: str, headers: dict, payload: dict, timeout: float) -> _Response:
    assert payload["messages"][0]["content"] == LATENT_CARD_SYSTEM_PROMPT
    prompt = payload["messages"][1]["content"]
    assert "MISC" not in prompt
    assert "activation" not in prompt.lower()
    assert "comparison" not in prompt.lower()
    latent = int(re.search(r"Feature ID:\s*(\d+)", prompt).group(1))
    ids = [f"s{i:03d}" for i in range(1, 51)]
    card = {
        "latent_idx": latent,
        "short_name": "synthetic sentence pattern",
        "primary_explanation": "The sentences share a synthetic numbered template.",
        "candidate_behavioral_explanation": "The available sentences do not support a stable behavioral-function interpretation.",
        "explanation_type": "linguistic_structure",
        "supporting_sample_ids": ids,
        "outlier_sample_ids": [],
        "representative_evidence_ids": ids[:2],
        "linguistic_evidence": [
            {"sample_id": ids[0], "evidence": "A repeated numbered template is present.", "evidence_level": "lexical", "evidence_role": "support"}
        ],
        "alternative_explanations": ["The shared pattern may be topical."],
        "possible_confounds": ["Synthetic data wording."],
        "limitations": ["Only one sentence set is available."],
        "confidence": 4,
        "confidence_rationale": "The repeated structure covers the full set.",
    }
    return _Response({"id": f"r-{latent}", "model": "deepseek-v4-flash", "choices": [{"message": {"content": json.dumps(card)}, "finish_reason": "stop"}]})


def main() -> None:
    root = Path("outputs/_smoke_deepseek_latent_cards").resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    try:
        records = root / "records.jsonl"
        write_jsonl(records, [{"unit_text": f"synthetic sentence {i}"} for i in range(60)])
        features = root / "features.pt"
        tensor = torch.zeros((60, 2), dtype=torch.float16)
        tensor[:, 1] = torch.linspace(4, 0, 60)
        torch.save(tensor, features)
        stable = root / "stable.csv"
        stable.write_text("label,latent_idx,stable_set_role,inclusion_frequency,abs_cohens_d\nRE,1,stable_core,1,1\n", encoding="utf-8")
        cfg = DeepSeekTop50Config(concurrency=1, max_retries=0)
        build = build_latent_card_tasks(stable_latents_path=stable, feature_store_path=features, records_path=records, output_dir=root, config=cfg)
        tasks = read_jsonl(build["outputs"]["tasks"])
        assert set(tasks[0]["visible_samples"][0]) == {"id", "text"}
        run_deepseek_top50_tasks(tasks_path=build["outputs"]["tasks"], output_dir=root, api_key="test", config=cfg, system_prompt=LATENT_CARD_SYSTEM_PROMPT, request_fn=_request, analysis_name="deepseek_v4_flash_latent_cards", step_name="run-latent-card-generation")
        result = validate_latent_card_outputs(tasks_path=build["outputs"]["tasks"], execution_manifest_path=root / "llm_execution_manifest.jsonl", output_dir=root)
        assert result["n_valid"] == 1
        assert read_jsonl(result["outputs"]["validated_cards"])[0]["confidence"] == 4
        stable.write_text(
            "label,latent_idx,stable_set_role,rank_within_label,cohens_d\n"
            "RE,1,stable_core,1,1.0\n",
            encoding="utf-8",
        )
        review_path = render_stable_core_top5_human_review_document(
            output_dir=root,
            stable_latents_path=stable,
            labels=("RE",),
            latents_per_label=1,
        )
        review = review_path.read_text(encoding="utf-8")
        assert "## 标签 RE" in review
        assert "### Latent 1" in review
        assert review.count("| s") >= 50
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("test_deepseek_latent_cards_smoke passed")


if __name__ == "__main__":
    main()
