#!/usr/bin/env python
"""
P3 Parallel LLM Runner
======================
Automates the step-by-step execution of P3 Task A and Task B using DashScope API in parallel.

Steps:
1. Back up existing explanations and predictions.
2. Clean outputs and batch directories.
3. Prepare Task A batch containing all 180 prompts.
4. Process Task A prompts in parallel using ThreadPoolExecutor.
5. Merge Task A outputs.
6. Prepare Task B batch containing all 360 scoring tasks.
7. Process Task B tasks in parallel.
8. Merge Task B outputs and compute metrics.
9. Run Task C (Manual Review Template).
10. Run Task D (Final Card Builder).
11. Run Validate.
"""

import os
import json
import urllib.request
import urllib.error
import pathlib
import subprocess
import sys
import time
import socket
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

ENV_FILE = pathlib.Path("deploy/local/bailian_qwen35plus.env")
BASE_DIR = pathlib.Path("outputs/misc_full_sae_eval/interpretability/p3_feature_cards")
BATCH_DIR = BASE_DIR / "agent_batches"

# Files to backup/clear
EXPLANATIONS_JSONL = BASE_DIR / "ai_reviews" / "p3_input_explanations.jsonl"
EXPLANATIONS_CSV = BASE_DIR / "ai_reviews" / "p3_input_explanations.csv"
PREDICTIONS_JSONL = BASE_DIR / "ai_scoring" / "p3_scoring_predictions.jsonl"

def load_env():
    """Load environment variables from the .env file."""
    if not ENV_FILE.exists():
        print(f"ERROR: env file {ENV_FILE} not found.")
        sys.exit(1)
    
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

load_env()

API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
MODEL = os.environ.get("OPENAI_MODEL", "qwen-plus")

if not API_KEY:
    print("ERROR: OPENAI_API_KEY not found in env.")
    sys.exit(1)

def backup_file(path: pathlib.Path):
    if path.exists():
        bak_path = path.with_suffix(path.suffix + ".bak")
        print(f"Backing up {path.name} -> {bak_path.name}")
        if bak_path.exists():
            bak_path.unlink()
        path.rename(bak_path)

def clean_file(path: pathlib.Path):
    if path.exists():
        print(f"Removing existing file: {path}")
        path.unlink()

def run_command(args):
    print(f"Running command: {' '.join(args)}")
    res = subprocess.run(
        [sys.executable] + args,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    if res.returncode != 0:
        print(f"Command failed with exit code {res.returncode}")
        print(f"STDOUT: {res.stdout}")
        print(f"STDERR: {res.stderr}")
        sys.exit(res.returncode)
    print(res.stdout)

def _extract_json_blob(raw_text: str) -> dict:
    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if "```" in text:
        chunks = [chunk.strip() for chunk in text.split("```") if chunk.strip()]
        for chunk in chunks:
            if chunk.startswith("json"):
                chunk = chunk[4:].strip()
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                continue

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("Could not parse JSON from model response.")

def call_chat_completion(messages, temperature=0.0):
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"}
    }
    req = urllib.request.Request(
        url=f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        },
        method="POST"
    )
    # Simple retry loop
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw_response = json.loads(resp.read().decode("utf-8"))
            content = raw_response["choices"][0]["message"]["content"]
            return _extract_json_blob(content)
        except Exception as e:
            print(f"API Attempt {attempt+1} failed: {e}. Retrying in {2 * attempt + 1}s...")
            time.sleep(2 * attempt + 1)
    raise RuntimeError("API request failed after 5 attempts.")

# Task A execution
def process_task_a_item(item):
    item_idx = item["item_index"]
    pid = item["packet_id"]
    lbl = item["target_label"]
    latent = item["latent_idx"]
    rank = item["rank_within_label"]
    messages = item["prompt_messages"]
    
    try:
        parsed_res = call_chat_completion(messages, temperature=0.0)
        result = {
            "item_index": item_idx,
            "packet_id": pid,
            "target_label": lbl,
            "latent_idx": latent,
            "rank_within_label": rank,
            **parsed_res,
            "parse_status": "ok"
        }
        print(f"  [Task A] Successfully processed {pid} (AF/rank {rank})")
        return result
    except Exception as e:
        print(f"  [Task A] ERROR processing {pid}: {e}")
        return {
            "item_index": item_idx,
            "packet_id": pid,
            "target_label": lbl,
            "latent_idx": latent,
            "rank_within_label": rank,
            "one_sentence_tentative_interpretation": f"Error: {e}",
            "parse_status": "error"
        }

# Task B execution
def process_task_b_item(item):
    item_idx = item["item_index"]
    pid = item["packet_id"]
    lbl = item["target_label"]
    latent = item["latent_idx"]
    tt = item["task_type"]
    instructions = item["instructions"]
    candidate_explanation = item["candidate_explanation"]
    candidate_feature_name = item["candidate_feature_name"]
    final_concise_conclusion = item["final_concise_conclusion"]
    examples = item["examples"]
    
    # Construct prompt messages
    sys_content = (
        f"You are an objective auditor evaluating a sparse autoencoder (SAE) latent's candidate explanation.\n"
        f"You are given:\n"
        f"- The target MISC category: {lbl}\n"
        f"- A candidate feature name: {candidate_feature_name}\n"
        f"- A candidate explanation: {candidate_explanation}\n"
        f"- The conclusion of the explanation: {final_concise_conclusion}\n"
        f"- The task type: {tt}\n\n"
        f"Your job is to read the explanation and evaluate several held-out counselor utterance examples.\n"
        f"For each example, output a prediction (1 or 0) indicating:\n"
        f"- If task_type is 'activation_prediction_task': Predict whether this utterance will activate the latent strongly (1) or not (0).\n"
        f"- If task_type is 'code_discrimination_task': Predict whether this utterance matches the target MISC label (1) or is a negative/contrastive instance (0).\n\n"
        f"You must output your predictions in JSON format matching the schema."
    )
    
    formatted_examples = []
    for ex in examples:
        row_idx = ex.get("row_idx")
        active_lbls = ex.get("active_labels")
        text = ex.get("text", "")
        formatted_examples.append(
            f"- Example row_idx={row_idx} (Active MISC labels: '{active_lbls}'): \"{text}\""
        )
    formatted_examples_list = "\n".join(formatted_examples)
    
    user_content = (
        f"Task Type: {tt}\n"
        f"Target MISC Label: {lbl}\n"
        f"Candidate Explanation: {candidate_explanation}\n"
        f"Candidate Feature Name: {candidate_feature_name}\n\n"
        f"Examples to evaluate:\n"
        f"{formatted_examples_list}\n\n"
        f"Provide a JSON object containing a 'predictions' array. Each item in the array must have:\n"
        f"- 'row_idx' (integer, matching the input example)\n"
        f"- 'predicted' (integer, 1 or 0)\n"
        f"- 'confidence' (float between 0 and 1)\n"
        f"- 'rationale_short' (string, a brief rationale)"
    )
    
    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_content}
    ]
    
    try:
        parsed_res = call_chat_completion(messages, temperature=0.0)
        result = {
            "item_index": item_idx,
            "packet_id": pid,
            "target_label": lbl,
            "latent_idx": latent,
            "task_type": tt,
            "predictions": parsed_res.get("predictions", [])
        }
        print(f"  [Task B] Successfully processed {pid} ({tt})")
        return result
    except Exception as e:
        print(f"  [Task B] ERROR processing {pid} ({tt}): {e}")
        # Fallback empty predictions
        fallback_preds = []
        for ex in examples:
            fallback_preds.append({
                "row_idx": ex.get("row_idx"),
                "predicted": 0,
                "confidence": 0.5,
                "rationale_short": f"Error: {e}"
            })
        return {
            "item_index": item_idx,
            "packet_id": pid,
            "target_label": lbl,
            "latent_idx": latent,
            "task_type": tt,
            "predictions": fallback_preds
        }

def clean_batches_dir():
    if BATCH_DIR.exists():
        print(f"Cleaning existing batch files in {BATCH_DIR}...")
        for f in BATCH_DIR.glob("*"):
            if f.is_file():
                f.unlink()

def main():
    print("=" * 60)
    print("P3 Step-by-Step Execution Runner")
    print("=" * 60)
    
    # 1. Backup existing results
    backup_file(EXPLANATIONS_JSONL)
    backup_file(EXPLANATIONS_CSV)
    backup_file(PREDICTIONS_JSONL)
    
    # 2. Clean batches dir
    clean_batches_dir()
    
    # 3. Task A: Prepare and run
    print("\n--- Running Task A: Input Explanations ---")
    run_command(["run_misc_p3_agent_orchestrator.py", "task-a", "prepare", "--limit", "180", "--force"])
    
    batch_input_file = BATCH_DIR / "task_a_batch_001_input.json"
    if not batch_input_file.exists():
        print(f"ERROR: Prepared batch input file {batch_input_file} not found.")
        sys.exit(1)
        
    batch_data = json.loads(batch_input_file.read_text(encoding="utf-8"))
    items = batch_data["items"]
    print(f"Loaded {len(items)} prompts for Task A.")
    
    task_a_results = []
    # Use ThreadPoolExecutor to run API calls in parallel
    print("Processing Task A prompts in parallel...")
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_task_a_item, item): item for item in items}
        for future in as_completed(futures):
            task_a_results.append(future.result())
            
    # Sort results by item_index
    task_a_results.sort(key=lambda x: x["item_index"])
    
    # Write batch output
    batch_output = {
        "batch_id": 1,
        "model": MODEL,
        "completed_at": datetime.datetime.now().isoformat(),
        "results": task_a_results
    }
    batch_output_file = BATCH_DIR / "task_a_batch_001_output.json"
    with open(batch_output_file, "w", encoding="utf-8") as f:
        json.dump(batch_output, f, indent=2, ensure_ascii=False)
        
    print(f"Saved batch output to {batch_output_file}")
    
    # Merge Task A
    run_command(["run_misc_p3_agent_orchestrator.py", "task-a", "merge"])
    run_command(["run_misc_p3_agent_orchestrator.py", "task-a", "validate"])
    
    # 4. Task B: Prepare and run
    print("\n--- Running Task B: Input Scoring ---")
    run_command(["run_misc_p3_agent_orchestrator.py", "task-b", "prepare", "--limit", "360", "--force"])
    
    batch_input_file_b = BATCH_DIR / "task_b_batch_001_input.json"
    if not batch_input_file_b.exists():
        print(f"ERROR: Prepared batch input file {batch_input_file_b} not found.")
        sys.exit(1)
        
    batch_data_b = json.loads(batch_input_file_b.read_text(encoding="utf-8"))
    items_b = batch_data_b["items"]
    print(f"Loaded {len(items_b)} tasks for Task B.")
    
    task_b_results = []
    print("Processing Task B tasks in parallel...")
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_task_b_item, item): item for item in items_b}
        for future in as_completed(futures):
            task_b_results.append(future.result())
            
    # Sort results by item_index
    task_b_results.sort(key=lambda x: x["item_index"])
    
    # Write batch output
    batch_output_b = {
        "batch_id": 1,
        "model": MODEL,
        "completed_at": datetime.datetime.now().isoformat(),
        "results": task_b_results
    }
    batch_output_file_b = BATCH_DIR / "task_b_batch_001_output.json"
    with open(batch_output_file_b, "w", encoding="utf-8") as f:
        json.dump(batch_output_b, f, indent=2, ensure_ascii=False)
        
    print(f"Saved batch output to {batch_output_file_b}")
    
    # Merge Task B
    run_command(["run_misc_p3_agent_orchestrator.py", "task-b", "merge"])
    run_command(["run_misc_p3_agent_orchestrator.py", "task-b", "metrics"])
    
    # 5. Task C: Manual Review Template
    print("\n--- Running Task C: Manual Review Template ---")
    run_command(["run_misc_p3_agent_orchestrator.py", "task-c", "run"])
    
    # 6. Task D: Final Feature Cards
    print("\n--- Running Task D: Final Feature Cards ---")
    run_command(["run_misc_p3_agent_orchestrator.py", "task-d", "run"])
    
    # 7. Final Validation
    print("\n--- Running Validation ---")
    run_command(["run_misc_p3_agent_orchestrator.py", "validate"])
    
    print("\n" + "=" * 60)
    print("P3 Execution Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
