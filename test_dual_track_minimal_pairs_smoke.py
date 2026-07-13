from __future__ import annotations
import json, shutil
from pathlib import Path
from src.nlp_re_base.contrastive_evidence_pack import write_jsonl
from src.nlp_re_base.dual_track_minimal_pairs import DualTrackPairConfig, _prompt, validate_dual_track_pair_outputs

def main():
    explanation={"surface_pattern":"sentence-opening what form","surface_confidence":0.8,"surface_raw_support_fraction":0.7,"semantic_pattern":"open request for information","semantic_confidence":0.8,"semantic_raw_support_fraction":0.7,"candidate_explanation":"open questions often beginning with what","alternative_hypotheses":[],"failure_modes":[]}
    top5=["What helped you today?", "What would change?", "What comes next?", "What matters most?", "What did you notice?"]
    b_prompt=_prompt(explanation,top5,top5[0],"B_surface_minus_function_plus")
    c_prompt=_prompt(explanation,top5,top5[0],"C_surface_plus_function_minus")
    assert "Change the identified surface pattern substantially" in b_prompt
    assert "Change to a clearly different communicative-act category" in c_prompt
    assert "Communicative function to preserve:\nopen request for information" in b_prompt
    assert "Original communicative function to avoid:\nopen request for information" in c_prompt
    assert 'Change the surface feature "sentence-opening what form" while preserving the functional feature "open request for information"' in b_prompt
    assert 'Preserve the surface feature "sentence-opening what form" while changing away from the functional feature "open request for information"' in c_prompt
    for prompt in (b_prompt, c_prompt):
        assert "cell B" not in prompt and "cell C" not in prompt and "2x2" not in prompt
        assert "reference_utterances" not in prompt and "top5_activation_templates" not in prompt
        assert "within 5 words" in prompt
    assert "Could you describe" in b_prompt and "What meaningful changes" in c_prompt
    root=Path("outputs/_smoke_dual_track_pairs"); shutil.rmtree(root,ignore_errors=True); (root/"raw").mkdir(parents=True)
    task={"task_id":"dtm_00001","latent_idx":1,"associated_label":"AF","expected_blocks":1,"expected_output_path":str(root/"raw"/"dtm_00001.json")}
    write_jsonl(root/"tasks.jsonl",[task])
    block={"block_id":"b1","topic":"exercise","generation_status":"feasible","surface_anchor":"sounds like","semantic_function":"affirmation","s_plus_f_plus":"It sounds like you made a strong plan today.","s_minus_f_plus":"You made a strong and thoughtful plan today.","s_plus_f_minus":"It sounds like you missed the planned session today.","s_minus_f_minus":"You missed the scheduled session again today.","surface_changed":"anchor removed","semantic_changed":"affirmation to criticism","held_constant":"topic and length"}
    (root/"raw"/"dtm_00001.json").write_text(json.dumps({"blocks":[block]}),encoding="utf-8")
    result=validate_dual_track_pair_outputs(tasks_path=root/"tasks.jsonl",output_dir=root,config=DualTrackPairConfig(blocks_per_latent=1,max_length_ratio=2.0,min_token_jaccard=0.1))
    assert result["n_blocks"]==1 and result["n_errors"]==0
    shutil.rmtree(root,ignore_errors=True); print("test_dual_track_minimal_pairs_smoke passed")
if __name__=="__main__": main()
