"""Build, generate, validate, and activate dual-track 2x2 minimal-pair blocks."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from nlp_re_base.deepseek_top50_induction import DeepSeekTop50Config, run_deepseek_top50_tasks
from nlp_re_base.dual_track_minimal_pairs import DualTrackPairConfig, build_dual_track_pair_tasks, build_phase2_d_tasks, run_dual_track_activation_test, validate_dual_track_pair_outputs, validate_phase1_mutations, validate_phase2_and_assemble

BASE = Path("outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/deepseek_v4_flash_top50_dual_track_induction")
OUT = BASE / "minimal_pair_pilot"

def args():
    p=argparse.ArgumentParser(); p.add_argument("--action",required=True,choices=["build","generate","validate-phase1","build-d","generate-d","validate-d","activate"])
    p.add_argument("--output-dir",default=str(OUT)); p.add_argument("--mode",choices=["pilot","full"],default="pilot")
    p.add_argument("--blocks-per-latent",type=int,default=5); p.add_argument("--concurrency",type=int,default=18)
    p.add_argument("--tasks-path",default=None,help="Optional task JSONL override for generate actions")
    p.add_argument("--mutations-path",default=None,help="Optional validated phase-1 mutations for build-d")
    p.add_argument("--model-dir",default=None); p.add_argument("--device",default=None); p.add_argument("--batch-size",type=int,default=1)
    return p.parse_args()

def main():
    a=args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); cfg=DualTrackPairConfig(blocks_per_latent=a.blocks_per_latent,mode=a.mode)
    tasks=out/"llm_tasks"/"dual_track_minimal_pair_tasks.jsonl"
    if a.action=="build": result=build_dual_track_pair_tasks(explanations_path=BASE/"explainer_outputs"/"validated_explanations.jsonl",top50_tasks_path=BASE/"llm_tasks"/"deepseek_top50_induction_tasks.jsonl",stable_latents_path="outputs/cross_val/stable_topk_selection/stable_topk_latent_set.csv",output_dir=out,config=cfg)
    elif a.action=="generate":
        key=os.getenv("DEEPSEEK_API_KEY","");
        if not key: raise RuntimeError("DEEPSEEK_API_KEY missing")
        result=run_deepseek_top50_tasks(tasks_path=Path(a.tasks_path) if a.tasks_path else tasks,output_dir=out,api_key=key,config=DeepSeekTop50Config(concurrency=a.concurrency),resume=True)
    elif a.action=="validate-phase1": result=validate_phase1_mutations(tasks_path=Path(a.tasks_path) if a.tasks_path else tasks,output_dir=out,config=cfg)
    elif a.action=="build-d": result=build_phase2_d_tasks(mutations_path=Path(a.mutations_path) if a.mutations_path else out/"validated_phase1_mutations.jsonl",output_dir=out)
    elif a.action=="generate-d":
        key=os.getenv("DEEPSEEK_API_KEY","");
        if not key: raise RuntimeError("DEEPSEEK_API_KEY missing")
        result=run_deepseek_top50_tasks(tasks_path=Path(a.tasks_path) if a.tasks_path else out/"llm_tasks"/"phase2_d_tasks.jsonl",output_dir=out,api_key=key,config=DeepSeekTop50Config(concurrency=a.concurrency),resume=True)
    elif a.action=="validate-d": result=validate_phase2_and_assemble(d_tasks_path=Path(a.tasks_path) if a.tasks_path else out/"llm_tasks"/"phase2_d_tasks.jsonl",output_dir=out,config=cfg)
    else: result=run_dual_track_activation_test(blocks_path=out/"validated_2x2_blocks.jsonl",output_dir=out,sae_config_path="config/sae_config.json",model_config_path="config/model_config.json",model_dir=a.model_dir,feature_store_path="outputs/misc_full_sae_eval/feature_store/utterance_features.pt",stable_latents_path="outputs/cross_val/stable_topk_selection/stable_topk_latent_set.csv",device=a.device,batch_size=a.batch_size)
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
