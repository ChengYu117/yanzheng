"""CPU smoke checks for sampled SAE/PCA contrastive prompts and schemas."""

import json
from pathlib import Path

from src.nlp_re_base.contrastive_faithfulness_v2 import SCORER_FROZEN_FIELDS, build_explainer_prompt, freeze_randomized_heldout_packet
from src.nlp_re_base import task5_sae_pca_contrastive_faithfulness as task5
from src.nlp_re_base.task5_sae_pca_contrastive_faithfulness import _scorer_prompt


def main() -> None:
    strong = [{"sample_id": f"A{i:03d}", "text": f"strong dialogue sentence {i}"} for i in range(1, 11)]
    weak = [{"sample_id": f"B{i:03d}", "text": f"weak dialogue sentence {i}"} for i in range(1, 11)]
    prompt = build_explainer_prompt("F001", strong, weak)
    lower = prompt.lower()
    for forbidden in ("sae", "latent", "pca", "misc", "activation"):
        assert forbidden not in lower
    assert all(f"A{i:03d}" in prompt and f"B{i:03d}" in prompt for i in range(1, 11))
    heldout = [{"sample_id": f"H{i:03d}", "text": f"held-out sentence {i}"} for i in range(1, 21)]
    explanation = {
        "short_name": "visible short name",
        "surface_or_linguistic_hypothesis": "visible surface hypothesis",
        "behavioral_or_discourse_hypothesis": "visible behavioral hypothesis",
        "primary_explanation": "visible primary explanation",
        "explanation_type": "behavioral_function",
        "contrastive_explanation": "FORBIDDEN_CONTRASTIVE",
        "necessary_or_characteristic_condition": "FORBIDDEN_NECESSARY",
        "insufficient_conditions": ["FORBIDDEN_INSUFFICIENT"],
        "possible_confounds": ["FORBIDDEN_CONFOUND"],
        "limitations": ["FORBIDDEN_LIMITATION"],
    }
    scorer = _scorer_prompt("F001", explanation, heldout)
    assert tuple(SCORER_FROZEN_FIELDS) == (
        "short_name", "surface_or_linguistic_hypothesis", "behavioral_or_discourse_hypothesis",
        "primary_explanation", "explanation_type",
    )
    for forbidden in ("FORBIDDEN_CONTRASTIVE", "FORBIDDEN_NECESSARY", "FORBIDDEN_INSUFFICIENT", "FORBIDDEN_CONFOUND", "FORBIDDEN_LIMITATION"):
        assert forbidden not in scorer
    assert "one exact contiguous substring" in scorer
    assert scorer.count("sample_id=H") == 20
    assert task5.freeze_randomized_heldout_packet is freeze_randomized_heldout_packet
    texts = [f"task5 held-out sentence {i}" for i in range(20)]
    selected = {"high": list(range(5)), "mid": list(range(5, 10)), "weak": list(range(10, 15)), "control": list(range(15, 20))}
    frozen = task5.freeze_randomized_heldout_packet(selected, texts, "F001", "task5-presentation-seed")
    assert [row["row_idx"] for row in frozen["private_truth"]] != list(range(20))
    assert all(set(row) == {"sample_id", "text"} for row in frozen["public_samples"])
    for path in ("config/contrastive_explainer_v2_schema.json", "config/contrastive_scorer_v2_schema.json"):
        json.loads(Path(path).read_text(encoding="utf-8"))
    print("test_task5_sae_pca_contrastive_faithfulness_smoke passed")


if __name__ == "__main__":
    main()
