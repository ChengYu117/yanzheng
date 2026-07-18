"""CPU smoke checks for v2 prompt privacy and schemas."""
import json
from pathlib import Path
from src.nlp_re_base.contrastive_faithfulness_v2 import build_explainer_prompt

def main():
    strong=[{"sample_id":f"A{i:03d}","text":f"strong sentence {i}"} for i in range(1,11)]; weak=[{"sample_id":f"B{i:03d}","text":f"weak sentence {i}"} for i in range(1,11)]; prompt=build_explainer_prompt("F001",strong,weak); lower=prompt.lower()
    for forbidden in ("sae","latent","llama","token activation","misc label"): assert forbidden not in lower
    assert "spoken or transcribed dialogue" in lower and "narrowest" in lower
    for path in ("config/contrastive_explainer_v2_schema.json","config/contrastive_scorer_v2_schema.json"): json.loads(Path(path).read_text(encoding="utf-8"))
    print("test_contrastive_faithfulness_v2_smoke passed")
if __name__=="__main__": main()
