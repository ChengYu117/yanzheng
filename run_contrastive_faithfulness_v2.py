"""Run the GPT-5.5-low contrastive faithfulness v2 workflow."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from src.nlp_re_base.contrastive_faithfulness_v2 import build_full_packets, build_pilot_packets, make_explainer_retry_tasks, make_scorer_tasks, run_stage, validate_explanations, validate_scorer_and_score

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("action",choices=("build","build-full","run-all","run-explainer","validate-explainer","make-explainer-retry","run-explainer-retry","make-scorer","run-scorer","validate-scorer")); p.add_argument("--output-dir",required=True); p.add_argument("--feature-store",default="outputs/rerun_new_dataset_20260716/min5_words/feature_store/utterance_features.pt"); p.add_argument("--records",default="outputs/rerun_new_dataset_20260716/min5_words/records.jsonl"); p.add_argument("--stable-latents",default="outputs/rerun_new_dataset_20260716/min5_words/cross_val/stable_topk_selection_n20_relaxed_leaf7/stable_topk_latent_set.csv"); p.add_argument("--model",default="gpt-5.5"); p.add_argument("--reasoning-effort",default="low"); p.add_argument("--concurrency",type=int,default=4); p.add_argument("--timeout-seconds",type=float,default=600); a=p.parse_args(); o=Path(a.output_dir)
    if a.action=="build": result=build_pilot_packets(feature_store_path=a.feature_store,records_path=a.records,output_dir=o)
    elif a.action=="build-full": result=build_full_packets(feature_store_path=a.feature_store,records_path=a.records,stable_latents_path=a.stable_latents,output_dir=o)
    elif a.action=="run-all":
        status={"stage":"starting"}; (o/"pipeline_status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        if not (o/"explainer"/"tasks.jsonl").exists():
            build_full_packets(feature_store_path=a.feature_store,records_path=a.records,stable_latents_path=a.stable_latents,output_dir=o)
        status={"stage":"explainer_running"}; (o/"pipeline_status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        run_stage(stage_dir=o/"explainer",tasks_path=o/"explainer/tasks.jsonl",schema_path="config/contrastive_explainer_v2_schema.json",instructions_path="config/contrastive_explainer_v2_base_instructions.txt",model=a.model,reasoning_effort=a.reasoning_effort,concurrency=a.concurrency,timeout_seconds=a.timeout_seconds)
        validation=validate_explanations(output_dir=o)
        if validation["n_failed"]:
            retry=make_explainer_retry_tasks(output_dir=o)
            if retry["n_retry_tasks"]:
                status={"stage":"explainer_retry_running","n_retry_tasks":retry["n_retry_tasks"]}; (o/"pipeline_status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
                run_stage(stage_dir=o/"explainer_retry",tasks_path=o/"explainer/retry_tasks.jsonl",schema_path="config/contrastive_explainer_v2_schema.json",instructions_path="config/contrastive_explainer_v2_base_instructions.txt",model=a.model,reasoning_effort=a.reasoning_effort,concurrency=a.concurrency,timeout_seconds=a.timeout_seconds)
                validation=validate_explanations(output_dir=o)
        status={"stage":"scorer_running","n_valid_explanations":validation["n_valid"],"n_failed_explanations":validation["n_failed"]}; (o/"pipeline_status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        make_scorer_tasks(output_dir=o)
        run_stage(stage_dir=o/"scorer",tasks_path=o/"scorer/tasks.jsonl",schema_path="config/contrastive_scorer_v2_schema.json",instructions_path="config/contrastive_scorer_v2_base_instructions.txt",model=a.model,reasoning_effort=a.reasoning_effort,concurrency=a.concurrency,timeout_seconds=a.timeout_seconds)
        result=validate_scorer_and_score(output_dir=o)
        status={"stage":"complete","scorer_validation":result}; (o/"pipeline_status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    elif a.action=="run-explainer": result=run_stage(stage_dir=o/"explainer",tasks_path=o/"explainer/tasks.jsonl",schema_path="config/contrastive_explainer_v2_schema.json",instructions_path="config/contrastive_explainer_v2_base_instructions.txt",model=a.model,reasoning_effort=a.reasoning_effort,concurrency=a.concurrency,timeout_seconds=a.timeout_seconds)
    elif a.action=="validate-explainer": result=validate_explanations(output_dir=o)
    elif a.action=="make-explainer-retry": result=make_explainer_retry_tasks(output_dir=o)
    elif a.action=="run-explainer-retry": result=run_stage(stage_dir=o/"explainer_retry",tasks_path=o/"explainer/retry_tasks.jsonl",schema_path="config/contrastive_explainer_v2_schema.json",instructions_path="config/contrastive_explainer_v2_base_instructions.txt",model=a.model,reasoning_effort=a.reasoning_effort,concurrency=a.concurrency,timeout_seconds=a.timeout_seconds)
    elif a.action=="make-scorer": result=make_scorer_tasks(output_dir=o)
    elif a.action=="run-scorer": result=run_stage(stage_dir=o/"scorer",tasks_path=o/"scorer/tasks.jsonl",schema_path="config/contrastive_scorer_v2_schema.json",instructions_path="config/contrastive_scorer_v2_base_instructions.txt",model=a.model,reasoning_effort=a.reasoning_effort,concurrency=a.concurrency,timeout_seconds=a.timeout_seconds)
    else: result=validate_scorer_and_score(output_dir=o)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
