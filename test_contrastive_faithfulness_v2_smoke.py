"""CPU smoke checks for v2 prompt privacy and schemas."""
import json
import tempfile
from pathlib import Path
from src.nlp_re_base.contrastive_evidence_pack import normalise_text, read_jsonl, write_jsonl
from src.nlp_re_base.contrastive_faithfulness_v2 import (
    SCORER_FROZEN_FIELDS,
    align_scorer_predictions_by_id,
    build_explainer_prompt,
    build_scorer_prompt,
    freeze_randomized_heldout_packet,
    validate_scorer_and_score,
)

def main():
    strong=[{"sample_id":f"A{i:03d}","text":f"strong sentence {i}"} for i in range(1,11)]; weak=[{"sample_id":f"B{i:03d}","text":f"weak sentence {i}"} for i in range(1,11)]; prompt=build_explainer_prompt("F001",strong,weak); lower=prompt.lower()
    for forbidden in ("sae","latent","llama","token activation","misc label"): assert forbidden not in lower
    assert "spoken or transcribed dialogue" in lower and "narrowest" in lower
    heldout=[{"sample_id":f"H{i:03d}","text":f"held-out sentence {i}"} for i in range(1,21)]
    explanation={
        "short_name":"visible short name",
        "surface_or_linguistic_hypothesis":"visible surface hypothesis",
        "behavioral_or_discourse_hypothesis":"visible behavioral hypothesis",
        "primary_explanation":"visible primary explanation",
        "explanation_type":"behavioral_function",
        "contrastive_explanation":"FORBIDDEN_CONTRASTIVE",
        "necessary_or_characteristic_condition":"FORBIDDEN_NECESSARY",
        "insufficient_conditions":["FORBIDDEN_INSUFFICIENT"],
        "possible_confounds":["FORBIDDEN_CONFOUND"],
        "limitations":["FORBIDDEN_LIMITATION"],
    }
    scorer=build_scorer_prompt("F001",explanation,heldout)
    assert tuple(SCORER_FROZEN_FIELDS)==("short_name","surface_or_linguistic_hypothesis","behavioral_or_discourse_hypothesis","primary_explanation","explanation_type")
    for visible in ("visible short name","visible surface hypothesis","visible behavioral hypothesis","visible primary explanation","behavioral_function"): assert visible in scorer
    for forbidden in ("FORBIDDEN_CONTRASTIVE","FORBIDDEN_NECESSARY","FORBIDDEN_INSUFFICIENT","FORBIDDEN_CONFOUND","FORBIDDEN_LIMITATION"): assert forbidden not in scorer
    assert "one exact contiguous substring" in scorer
    assert "never normalize spacing" in scorer
    assert scorer.count("sample_id=H")==20

    texts=[f"held-out unique sentence {i}" for i in range(20)]
    selected={"high":list(range(0,5)),"mid":list(range(5,10)),"weak":list(range(10,15)),"zero":list(range(15,20))}
    frozen_a=freeze_randomized_heldout_packet(selected,texts,"F001","presentation-seed-a")
    frozen_a_repeat=freeze_randomized_heldout_packet(selected,texts,"F001","presentation-seed-a")
    frozen_b=freeze_randomized_heldout_packet(selected,texts,"F001","presentation-seed-b")
    assert frozen_a==frozen_a_repeat
    assert frozen_a["packet_sha256"]==frozen_a_repeat["packet_sha256"]
    assert frozen_a["selected_rows_sha256"]==frozen_b["selected_rows_sha256"]
    assert frozen_a["packet_sha256"]!=frozen_b["packet_sha256"]
    assert [row["row_idx"] for row in frozen_a["private_truth"]] != list(range(20))
    assert {row["public_sample_id"] for row in frozen_a["private_truth"]}=={f"H{i:03d}" for i in range(1,21)}
    assert all(set(row)=={"sample_id","text"} for row in frozen_a["public_samples"])
    assert {tag:sum(row["stratum"]==tag for row in frozen_a["private_truth"]) for tag in selected}=={tag:5 for tag in selected}
    row_by_public_id={row["public_sample_id"]:row["row_idx"] for row in frozen_a["private_truth"]}
    row_by_public_id_b={row["public_sample_id"]:row["row_idx"] for row in frozen_b["private_truth"]}
    assert row_by_public_id!=row_by_public_id_b

    discovery_rows=set(range(20,40)); heldout_rows=set(range(20))
    discovery_texts={normalise_text(f"discovery sentence {i}") for i in discovery_rows}
    heldout_texts={normalise_text(texts[i]) for i in heldout_rows}
    discovery_sources={f"discovery-file-{i//2}" for i in discovery_rows}
    heldout_sources={f"heldout-file-{i//2}" for i in heldout_rows}
    assert discovery_rows.isdisjoint(heldout_rows)
    assert discovery_texts.isdisjoint(heldout_texts)
    assert discovery_sources.isdisjoint(heldout_sources)

    truth=[]
    for row in frozen_a["private_truth"]:
        truth.append({**row,"true_activation":float(row["row_idx"]+1)})
    predictions=[
        {"sample_id":row["public_sample_id"],"predicted_feature_score":row["row_idx"]+1,"matching_evidence_span":""}
        for row in reversed(truth)
    ]
    aligned=align_scorer_predictions_by_id(public_samples=frozen_a["public_samples"],private_truth=truth,predictions=predictions,feature_id="F001")
    assert all(pred["predicted_feature_score"]==private["true_activation"] for pred,private,_ in aligned)

    with tempfile.TemporaryDirectory() as tmp:
        output=Path(tmp); raw=output/"scorer"/"raw"/"F001_scorer.json"; raw.parent.mkdir(parents=True)
        write_jsonl(output/"public_packets"/"scorer_packets.jsonl",[{"feature_id":"F001","samples":frozen_a["public_samples"]}])
        write_jsonl(output/"private"/"heldout_truth.jsonl",[{"feature_id":"F001","latent_idx":1,"samples":truth}])
        write_jsonl(output/"scorer"/"tasks.jsonl",[{"task_id":"F001_scorer","latent_idx":1,"feature_id":"F001","expected_output_path":str(raw)}])
        raw.write_text(json.dumps({"feature_id":"F001","predictions":predictions}),encoding="utf-8")
        validation=validate_scorer_and_score(output_dir=output)
        assert validation=={"n_tasks":1,"n_valid":1,"n_failed":0}
        validated=read_jsonl(output/"scorer"/"validated_predictions_private.jsonl")
        assert all(row["predicted_feature_score"]==row["true_activation"] for row in validated)
    for path in ("config/contrastive_explainer_v2_schema.json","config/contrastive_scorer_v2_schema.json"): json.loads(Path(path).read_text(encoding="utf-8"))
    print("test_contrastive_faithfulness_v2_smoke passed")
if __name__=="__main__": main()
