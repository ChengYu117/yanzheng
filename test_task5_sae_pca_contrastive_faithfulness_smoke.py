"""CPU smoke checks for sampled SAE/PCA contrastive prompts and schemas."""

import json
from pathlib import Path

from src.nlp_re_base.contrastive_faithfulness_v2 import build_explainer_prompt


def main() -> None:
    strong = [{"sample_id": f"A{i:03d}", "text": f"strong dialogue sentence {i}"} for i in range(1, 11)]
    weak = [{"sample_id": f"B{i:03d}", "text": f"weak dialogue sentence {i}"} for i in range(1, 11)]
    prompt = build_explainer_prompt("F001", strong, weak)
    lower = prompt.lower()
    for forbidden in ("sae", "latent", "pca", "misc", "activation"):
        assert forbidden not in lower
    assert all(f"A{i:03d}" in prompt and f"B{i:03d}" in prompt for i in range(1, 11))
    for path in ("config/contrastive_explainer_v2_schema.json", "config/contrastive_scorer_v2_schema.json"):
        json.loads(Path(path).read_text(encoding="utf-8"))
    print("test_task5_sae_pca_contrastive_faithfulness_smoke passed")


if __name__ == "__main__":
    main()
