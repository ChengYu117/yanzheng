"""
Extract and display tasks 121-180 from explainer_tasks.jsonl for analysis.
"""
import json
from pathlib import Path

TASKS_FILE = Path("outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/llm_tasks/explainer_tasks.jsonl")

with open(TASKS_FILE, "r", encoding="utf-8") as f:
    all_lines = f.readlines()

start_idx = 120
end_idx = min(180, len(all_lines))
task_lines = all_lines[start_idx:end_idx]

print(f"Total lines: {len(all_lines)}, processing {len(task_lines)} tasks (lines {start_idx+1}-{end_idx})")
print("=" * 80)

for i, line in enumerate(task_lines):
    task = json.loads(line.strip())
    task_id = task["task_id"]
    latent_idx = task["latent_idx"]
    prompt = task["prompt"]

    # Parse samples from prompt
    samples = []
    in_samples = False
    current_sample = None
    for pline in prompt.split("\n"):
        stripped = pline.strip()
        if stripped.startswith("Samples:"):
            in_samples = True
            continue
        if in_samples and stripped.startswith("- id="):
            if current_sample:
                samples.append(current_sample)
            # Parse metadata
            parts = stripped.split()
            sid = parts[0].replace("- id=", "") if len(parts) > 0 else ""
            stag = parts[1].replace("tag=", "") if len(parts) > 1 else ""
            sact = parts[2].replace("activation=", "") if len(parts) > 2 else ""
            stext_idx = stripped.find("text:")
            stext = stripped[stext_idx+5:].strip() if stext_idx >= 0 else ""
            current_sample = {"id": sid, "tag": stag, "activation": sact, "text": stext}
        elif in_samples and current_sample and stripped.startswith("text:"):
            current_sample["text"] = stripped[5:].strip()
    if current_sample:
        samples.append(current_sample)

    task_num = start_idx + i + 1
    print(f"\nTask {task_num}: {task_id} | latent={latent_idx} | {len(samples)} samples")
    for s in samples:
        tag_short = s["tag"].replace("ACTIVE_", "A_").replace("NONACTIVE_", "N_")
        print(f"  [{tag_short:8s}] act={s['activation']:>10s} | {s['text'][:80]}")
