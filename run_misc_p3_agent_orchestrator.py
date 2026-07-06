#!/usr/bin/env python
"""
P3 Agent Batch Orchestrator
============================
Manages batch processing of P3 evaluation tasks for a dialogue AI agent.

This script orchestrates the P3 feature card evaluation pipeline by:
1. Splitting large tasks into manageable batches
2. Preparing self-contained batch input files for the current AI agent to process
3. Merging batch outputs into the main result files
4. Computing evaluation metrics
5. Running pure-Python data processing tasks (Task C, D)

It does not call model APIs. Task A/B reasoning is performed by the current
dialogue model or another agent that reads the generated batch files and writes
the requested output JSON.

Usage:
    python run_misc_p3_agent_orchestrator.py 启动p3 [--batch-size 3]
    python run_misc_p3_agent_orchestrator.py start-p3 [--batch-size 3]
    python run_misc_p3_agent_orchestrator.py status
    python run_misc_p3_agent_orchestrator.py next-batch [--batch-size 3]
    python run_misc_p3_agent_orchestrator.py task-a prepare [--batch-size 5] [--label RE]
    python run_misc_p3_agent_orchestrator.py task-a merge
    python run_misc_p3_agent_orchestrator.py task-a validate
    python run_misc_p3_agent_orchestrator.py task-b prepare [--batch-size 10] [--label QUO]
    python run_misc_p3_agent_orchestrator.py task-b merge
    python run_misc_p3_agent_orchestrator.py task-b metrics
    python run_misc_p3_agent_orchestrator.py task-c run
    python run_misc_p3_agent_orchestrator.py task-d run
    python run_misc_p3_agent_orchestrator.py validate

Batch files (in agent_batches/):
    task_a_batch_{NNN}_input.json   - Prompts for the current agent to process
    task_a_batch_{NNN}_output.json  - Agent-generated explanations
    task_b_batch_{NNN}_input.json   - Scoring tasks for the current agent
    task_b_batch_{NNN}_output.json  - Agent-generated predictions
"""

import argparse
import json
import csv
import pathlib
import datetime
import sys
import math
from collections import defaultdict

# ============================================================
# Constants
# ============================================================

DEFAULT_BASE_DIR = pathlib.Path("outputs/misc_full_sae_eval/interpretability/p3_feature_cards_stable_core")
BASE_DIR = DEFAULT_BASE_DIR
BATCH_DIR = BASE_DIR / "agent_batches"
PROGRESS_JSON = BASE_DIR / "p3_agent_batch_progress.json"
VALIDATION_JSON = BASE_DIR / "p3_agent_validation.json"

# Task A paths
PROMPTS_DIR = BASE_DIR / "p3_input_explanation_prompts"
REVIEWS_DIR = BASE_DIR / "ai_reviews"
RAW_RESP_A_DIR = REVIEWS_DIR / "raw_responses"
EXPLANATIONS_JSONL = REVIEWS_DIR / "p3_input_explanations.jsonl"
EXPLANATIONS_CSV = REVIEWS_DIR / "p3_input_explanations.csv"
FAILED_A_JSONL = REVIEWS_DIR / "failed_responses.jsonl"
MANIFEST_A = REVIEWS_DIR / "manifest.json"

# Task B paths
SCORING_DIR = BASE_DIR / "ai_scoring"
RAW_RESP_B_DIR = SCORING_DIR / "raw_responses"
SCORING_TASKS_JSONL = BASE_DIR / "p3_scoring_tasks.jsonl"
PREDICTIONS_JSONL = SCORING_DIR / "p3_scoring_predictions.jsonl"
METRICS_BY_LATENT = SCORING_DIR / "p3_scoring_metrics_by_latent.csv"
METRICS_BY_LABEL = SCORING_DIR / "p3_scoring_metrics_by_label.csv"
METRICS_OVERALL = SCORING_DIR / "p3_scoring_metrics_overall.json"
MANIFEST_B = SCORING_DIR / "manifest.json"

# Task C paths
MANUAL_DIR = BASE_DIR / "manual_review"
REVIEW_TEMPLATE = MANUAL_DIR / "p3_mi_coder_review_template.csv"

# Task D paths
FINAL_DIR = BASE_DIR / "final_cards"
CARDS_JSONL = BASE_DIR / "p3_feature_card_packets.jsonl"
FINAL_JSONL = FINAL_DIR / "p3_final_feature_cards.jsonl"
FINAL_CSV = FINAL_DIR / "p3_final_feature_cards.csv"
FINAL_MD = FINAL_DIR / "p3_final_feature_cards.md"
LABEL_SUMMARY = FINAL_DIR / "p3_label_summary.csv"
CLUSTER_SEED = FINAL_DIR / "p3_concept_cluster_seed_table.csv"
MANIFEST_D = FINAL_DIR / "manifest.json"


def configure_paths(base_dir):
    """Configure all P3 artifact paths for the selected feature-card directory."""
    global BASE_DIR, BATCH_DIR, PROGRESS_JSON, VALIDATION_JSON
    global PROMPTS_DIR, REVIEWS_DIR, RAW_RESP_A_DIR, EXPLANATIONS_JSONL, EXPLANATIONS_CSV
    global FAILED_A_JSONL, MANIFEST_A, SCORING_DIR, RAW_RESP_B_DIR, SCORING_TASKS_JSONL
    global PREDICTIONS_JSONL, METRICS_BY_LATENT, METRICS_BY_LABEL, METRICS_OVERALL, MANIFEST_B
    global MANUAL_DIR, REVIEW_TEMPLATE, FINAL_DIR, CARDS_JSONL, FINAL_JSONL, FINAL_CSV
    global FINAL_MD, LABEL_SUMMARY, CLUSTER_SEED, MANIFEST_D

    BASE_DIR = pathlib.Path(base_dir)
    BATCH_DIR = BASE_DIR / "agent_batches"
    PROGRESS_JSON = BASE_DIR / "p3_agent_batch_progress.json"
    VALIDATION_JSON = BASE_DIR / "p3_agent_validation.json"

    PROMPTS_DIR = BASE_DIR / "p3_input_explanation_prompts"
    REVIEWS_DIR = BASE_DIR / "ai_reviews"
    RAW_RESP_A_DIR = REVIEWS_DIR / "raw_responses"
    EXPLANATIONS_JSONL = REVIEWS_DIR / "p3_input_explanations.jsonl"
    EXPLANATIONS_CSV = REVIEWS_DIR / "p3_input_explanations.csv"
    FAILED_A_JSONL = REVIEWS_DIR / "failed_responses.jsonl"
    MANIFEST_A = REVIEWS_DIR / "manifest.json"

    SCORING_DIR = BASE_DIR / "ai_scoring"
    RAW_RESP_B_DIR = SCORING_DIR / "raw_responses"
    SCORING_TASKS_JSONL = BASE_DIR / "p3_scoring_tasks.jsonl"
    PREDICTIONS_JSONL = SCORING_DIR / "p3_scoring_predictions.jsonl"
    METRICS_BY_LATENT = SCORING_DIR / "p3_scoring_metrics_by_latent.csv"
    METRICS_BY_LABEL = SCORING_DIR / "p3_scoring_metrics_by_label.csv"
    METRICS_OVERALL = SCORING_DIR / "p3_scoring_metrics_overall.json"
    MANIFEST_B = SCORING_DIR / "manifest.json"

    MANUAL_DIR = BASE_DIR / "manual_review"
    REVIEW_TEMPLATE = MANUAL_DIR / "p3_mi_coder_review_template.csv"

    FINAL_DIR = BASE_DIR / "final_cards"
    CARDS_JSONL = BASE_DIR / "p3_feature_card_packets.jsonl"
    FINAL_JSONL = FINAL_DIR / "p3_final_feature_cards.jsonl"
    FINAL_CSV = FINAL_DIR / "p3_final_feature_cards.csv"
    FINAL_MD = FINAL_DIR / "p3_final_feature_cards.md"
    LABEL_SUMMARY = FINAL_DIR / "p3_label_summary.csv"
    CLUSTER_SEED = FINAL_DIR / "p3_concept_cluster_seed_table.csv"
    MANIFEST_D = FINAL_DIR / "manifest.json"


# ============================================================
# Helper Functions
# ============================================================

def load_jsonl(path):
    """Load a JSONL file, return list of dicts."""
    if not path.exists():
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return items


def save_jsonl(items, path):
    """Save list of dicts to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_json(path):
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path, indent=2):
    """Save dict to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def get_prompt_files():
    """Get all prompt files sorted."""
    if not PROMPTS_DIR.exists():
        return []
    return sorted(PROMPTS_DIR.glob("*_input_prompt.json"))


def parse_prompt_filename(filename):
    """Parse label, rank, latent from prompt filename."""
    stem = pathlib.Path(filename).stem.replace("_input_prompt", "")
    parts = stem.split("_")
    label = parts[0]
    rank = int(parts[1].replace("rank", ""))
    latent = int(parts[2].replace("latent", ""))
    return label, rank, latent


def build_cards_index():
    """Build index from (label, latent, rank) -> packet_id using cards JSONL."""
    cards = load_jsonl(CARDS_JSONL)
    index = {}
    for card in cards:
        key = (card["target_label"], card["latent_idx"], card["rank_within_label"])
        index[key] = card["packet_id"]
    return index


def get_completed_packet_ids_from_jsonl(jsonl_path):
    """Get set of completed packet_ids from a JSONL file."""
    items = load_jsonl(jsonl_path)
    return {item.get("packet_id") for item in items if item.get("packet_id")}


def get_completed_from_batch_outputs(pattern):
    """Get set of completed packet_ids from batch output files."""
    completed = set()
    if BATCH_DIR.exists():
        for bf in sorted(BATCH_DIR.glob(pattern)):
            try:
                data = load_json(bf)
                for r in data.get("results", []):
                    pid = r.get("packet_id")
                    if pid:
                        completed.add(pid)
            except Exception:
                pass
    return completed


def count_csv_rows(path):
    """Count data rows in a CSV file, excluding header."""
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def expected_artifact_counts():
    """Derive validation expectations from the current P3 feature-card inputs."""
    cards = load_jsonl(CARDS_JSONL)
    scoring_tasks = load_jsonl(SCORING_TASKS_JSONL)
    return {
        "cards": len(cards),
        "prompts": len(get_prompt_files()),
        "expected_predictions": sum(len(t.get("examples", [])) for t in scoring_tasks),
        "metrics_by_latent_rows": len(
            {
                (t.get("packet_id", ""), t.get("task_type", ""))
                for t in scoring_tasks
                if t.get("packet_id") and t.get("task_type")
            }
        ),
        "metrics_by_label_rows": len({card.get("target_label", "") for card in cards if card.get("target_label")}),
    }


def parse_labels_arg(value):
    """Parse comma/space-separated label filters."""
    if not value:
        return set()
    if isinstance(value, (list, tuple)):
        raw = []
        for item in value:
            raw.extend(str(item).replace(",", " ").split())
    else:
        raw = str(value).replace(",", " ").split()
    return {item.strip().upper() for item in raw if item.strip()}


def next_batch_id(prefix):
    """Return next numeric batch id for a task prefix."""
    existing = sorted(BATCH_DIR.glob(f"{prefix}_batch_*_input.json")) if BATCH_DIR.exists() else []
    max_id = 0
    for path in existing:
        parts = path.stem.split("_")
        for part in parts:
            if part.isdigit():
                max_id = max(max_id, int(part))
    return max_id + 1


def apply_common_filters(items, args, *, label_getter, rank_getter=None):
    """Apply label/rank/start/limit filters to prepared candidate items."""
    labels = parse_labels_arg(getattr(args, "label", None))
    if labels:
        items = [item for item in items if str(label_getter(item)).upper() in labels]

    rank_min = getattr(args, "rank_min", None)
    rank_max = getattr(args, "rank_max", None)
    if rank_getter is not None and rank_min is not None:
        items = [item for item in items if int(rank_getter(item)) >= int(rank_min)]
    if rank_getter is not None and rank_max is not None:
        items = [item for item in items if int(rank_getter(item)) <= int(rank_max)]

    start = max(int(getattr(args, "start_index", 0) or 0), 0)
    limit = getattr(args, "limit", None)
    if limit is None:
        limit = getattr(args, "batch_size", None)
    if limit is None:
        return items[start:]
    return items[start : start + max(int(limit), 0)]


def trim_prompt_messages(prompt_messages, max_examples_per_group):
    """Return prompt messages with user payload examples truncated by group."""
    if max_examples_per_group is None:
        return prompt_messages
    max_examples_per_group = int(max_examples_per_group)
    trimmed = []
    for message in prompt_messages:
        if message.get("role") != "user":
            trimmed.append(message)
            continue
        content = message.get("content", "")
        try:
            payload = json.loads(content)
        except Exception:
            trimmed.append(message)
            continue
        examples = payload.get("examples", [])
        by_group = defaultdict(list)
        new_examples = []
        for example in examples:
            group = example.get("group", "")
            if len(by_group[group]) < max_examples_per_group:
                by_group[group].append(example)
                new_examples.append(example)
        payload["examples"] = new_examples
        payload["batch_truncation_note"] = (
            f"Examples were truncated to at most {max_examples_per_group} per group for this agent batch. "
            "Use the full prompt/card files for final audit if needed."
        )
        trimmed.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})
    return trimmed


def trim_task_examples(task, max_examples_per_task):
    """Return a scoring task copy with examples truncated."""
    if max_examples_per_task is None:
        return task
    out = dict(task)
    out["examples"] = list(task.get("examples", []))[: int(max_examples_per_task)]
    out["batch_truncation_note"] = (
        f"Examples were truncated to at most {int(max_examples_per_task)} for this agent batch. "
        "Do not use truncated batches as final full scoring unless this is intentional."
    )
    return out


def write_batch_instructions(batch_input, batch_path, *, task_label):
    """Write a human-readable Markdown instruction file for the current AI agent."""
    instruction_path = batch_path.with_name(batch_path.name.replace("_input.json", "_instructions.md"))
    info = batch_input.get("batch_info", {})
    output = batch_input.get("output_instructions", {})
    lines = [
        f"# {task_label} Batch {info.get('batch_id')}",
        "",
        "Read this file before processing the JSON batch. This batch is intentionally small so the current dialogue AI can handle it without losing context.",
        "",
        "No API runner is used here. Use the model in the current conversation to reason over the input and write the JSON output file.",
        "",
        "## Input",
        "",
        f"- JSON batch: `{batch_path.as_posix()}`",
        f"- Items: `{info.get('items_in_batch')}`",
        f"- Created at: `{info.get('created_at')}`",
        "",
        "## Output",
        "",
        f"- Write result JSON to: `{output.get('output_file')}`",
        "- Preserve `packet_id`, `target_label`, `latent_idx`, `rank_within_label`, and `item_index` exactly.",
        "- Do not stop the full project if one item fails; write a failed item with `parse_status` or a short error note.",
        "- Keep all conclusions bounded: candidate interpretation / scoring judgment, not causal proof.",
        "- Set `model` to a descriptive current-agent value such as `current_dialogue_agent`.",
        "",
        "## Items",
        "",
    ]
    for item in batch_input.get("items", []):
        task_type = item.get("task_type", "")
        extra = f", task={task_type}" if task_type else ""
        lines.append(
            f"- item {item.get('item_index')}: `{item.get('packet_id')}`, "
            f"{item.get('target_label')} rank {item.get('rank_within_label', '?')}, "
            f"latent {item.get('latent_idx')}{extra}"
        )
    lines.extend([
        "",
        "## Continuation",
        "",
        "After writing the output JSON, run the corresponding merge command:",
        "",
        "- Task A: `python run_misc_p3_agent_orchestrator.py task-a merge`",
        "- Task B: `python run_misc_p3_agent_orchestrator.py task-b merge`",
        "",
        "Then run:",
        "",
        "`python run_misc_p3_agent_orchestrator.py status`",
        "",
    ])
    instruction_path.write_text("\n".join(lines), encoding="utf-8")
    return instruction_path


def collect_progress():
    """Collect current P3 agent pipeline progress."""
    prompt_files = get_prompt_files()
    explanations = load_jsonl(EXPLANATIONS_JSONL)
    scoring_tasks = load_jsonl(SCORING_TASKS_JSONL)
    predictions = load_jsonl(PREDICTIONS_JSONL)
    expected_preds = sum(len(t.get("examples", [])) for t in scoring_tasks)
    final_cards = load_jsonl(FINAL_JSONL)

    ba_in = sorted(BATCH_DIR.glob("task_a_batch_*_input.json")) if BATCH_DIR.exists() else []
    ba_out = sorted(BATCH_DIR.glob("task_a_batch_*_output.json")) if BATCH_DIR.exists() else []
    bb_in = sorted(BATCH_DIR.glob("task_b_batch_*_input.json")) if BATCH_DIR.exists() else []
    bb_out = sorted(BATCH_DIR.glob("task_b_batch_*_output.json")) if BATCH_DIR.exists() else []

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "base_dir": str(BASE_DIR),
        "task_a": {
            "prompts": len(prompt_files),
            "explanations": len(explanations),
            "parse_ok": sum(1 for r in explanations if r.get("parse_status") == "ok"),
            "batch_inputs": len(ba_in),
            "batch_outputs": len(ba_out),
            "complete": len(prompt_files) > 0 and len(explanations) >= len(prompt_files),
        },
        "task_b": {
            "tasks": len(scoring_tasks),
            "expected_predictions": expected_preds,
            "predictions": len(predictions),
            "batch_inputs": len(bb_in),
            "batch_outputs": len(bb_out),
            "metrics_by_latent_rows": count_csv_rows(METRICS_BY_LATENT),
            "metrics_by_label_rows": count_csv_rows(METRICS_BY_LABEL),
            "complete": expected_preds > 0 and len(predictions) >= expected_preds,
        },
        "task_c": {
            "manual_review_template_rows": count_csv_rows(REVIEW_TEMPLATE),
            "complete": REVIEW_TEMPLATE.exists() and count_csv_rows(REVIEW_TEMPLATE) > 0,
        },
        "task_d": {
            "final_cards": len(final_cards),
            "markdown_exists": FINAL_MD.exists(),
            "label_summary_rows": count_csv_rows(LABEL_SUMMARY),
            "complete": len(final_cards) > 0 and FINAL_MD.exists(),
        },
    }


def write_progress_snapshot(note=None):
    """Persist current progress to p3_agent_batch_progress.json."""
    progress = collect_progress()
    if note:
        progress["note"] = note
    save_json(progress, PROGRESS_JSON)
    return progress


# ============================================================
# Metrics Computation
# ============================================================

def compute_binary_metrics(y_true, y_pred, y_scores=None):
    """Compute binary classification metrics without sklearn."""
    n = len(y_true)
    if n == 0:
        return {}

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    accuracy = (tp + tn) / n if n > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    n_pos = sum(y_true)
    n_neg = n - n_pos
    tpr = tp / n_pos if n_pos > 0 else 0
    tnr = tn / n_neg if n_neg > 0 else 0
    balanced_acc = (tpr + tnr) / 2

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n,
    }

    if y_scores is not None:
        auroc = compute_auroc(y_true, y_scores)
        metrics["auroc"] = round(auroc, 4) if auroc is not None else "not_available_binary_only"

    return metrics


def compute_auroc(y_true, y_scores):
    """Compute AUROC using trapezoidal rule on ROC curve."""
    unique_scores = set(y_scores)
    if len(unique_scores) <= 2:
        return None

    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    pairs = sorted(zip(y_scores, y_true), key=lambda x: -x[0])
    tp = fp = 0
    prev_score = None
    prev_tpr = prev_fpr = 0.0
    auc = 0.0

    for score, label in pairs:
        if score != prev_score and prev_score is not None:
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
            prev_tpr, prev_fpr = tpr, fpr
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score

    tpr = tp / n_pos
    fpr = fp / n_neg
    auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
    return auc


# ============================================================
# Task A: Input Explanations
# ============================================================

def cmd_task_a_prepare(args):
    """Prepare next batch of prompts for the current agent to process."""
    batch_size = args.batch_size
    prompt_files = get_prompt_files()
    if not prompt_files:
        print("ERROR: No prompt files found in", PROMPTS_DIR)
        return 1

    cards_index = build_cards_index()
    completed_main = get_completed_packet_ids_from_jsonl(EXPLANATIONS_JSONL)
    completed_batch = get_completed_from_batch_outputs("task_a_batch_*_output.json")
    all_completed = set() if getattr(args, "include_completed", False) else (completed_main | completed_batch)

    # Find unprocessed prompts
    unprocessed = []
    for pf in prompt_files:
        label, rank, latent = parse_prompt_filename(pf.name)
        pid = cards_index.get((label, latent, rank))
        if pid and pid not in all_completed:
            unprocessed.append((pf, pid, label, rank, latent))

    filtered = apply_common_filters(
        unprocessed,
        args,
        label_getter=lambda item: item[2],
        rank_getter=lambda item: item[3],
    )

    if not filtered:
        print(f"All {len(prompt_files)} prompts have been processed!")
        print(f"  In main JSONL: {len(completed_main)}")
        print(f"  In batch outputs: {len(completed_batch)}")
        print("Run 'task-a merge' to consolidate all results, or use --include-completed to regenerate a review batch.")
        return 0

    # Determine batch ID
    next_id = int(getattr(args, "batch_id", None) or next_batch_id("task_a"))
    batch_items = filtered

    # Build batch input file
    items = []
    for i, (pf, pid, label, rank, latent) in enumerate(batch_items):
        prompt_data = load_json(pf)
        items.append({
            "item_index": i,
            "prompt_filename": pf.name,
            "packet_id": pid,
            "target_label": label,
            "latent_idx": latent,
            "rank_within_label": rank,
            "prompt_messages": trim_prompt_messages(
                prompt_data.get("messages", []),
                getattr(args, "max_examples_per_group", None),
            ),
        })

    batch_input = {
        "batch_info": {
            "batch_id": next_id,
            "task_type": "input_explanation",
            "items_in_batch": len(items),
            "total_remaining": len(unprocessed),
            "filtered_remaining": len(filtered),
            "total_prompts": len(prompt_files),
            "total_completed": len(all_completed),
            "estimated_batches_remaining": math.ceil(len(unprocessed) / max(batch_size, 1)),
            "label_filter": sorted(parse_labels_arg(getattr(args, "label", None))),
            "rank_min": getattr(args, "rank_min", None),
            "rank_max": getattr(args, "rank_max", None),
            "start_index": getattr(args, "start_index", 0),
            "limit": getattr(args, "limit", None) or batch_size,
            "include_completed": bool(getattr(args, "include_completed", False)),
            "max_examples_per_group": getattr(args, "max_examples_per_group", None),
            "created_at": datetime.datetime.now().isoformat(),
        },
        "output_instructions": {
            "output_file": str(BATCH_DIR / f"task_a_batch_{next_id:03d}_output.json"),
            "description": (
                "The current dialogue agent should read each item's prompt_messages, "
                "analyze the SAE latent examples, and generate a structured explanation. "
                "Write the results array to the output file."
            ),
            "output_schema": {
                "batch_id": next_id,
                "model": "current_dialogue_agent",
                "completed_at": "ISO_TIMESTAMP",
                "results": [{
                    "item_index": 0,
                    "packet_id": "packet_XXXX",
                    "target_label": "LABEL",
                    "latent_idx": 0,
                    "rank_within_label": 0,
                    "one_sentence_tentative_interpretation": "...",
                    "main_patterns": [{
                        "pattern_name": "...",
                        "pattern_type": "surface_form|dialogue_function|context_relation|mi_principle|artifact|mixed_unclear",
                        "evidence": "...",
                    }],
                    "relationship_to_target_label": "...",
                    "artifact_risk": "low|medium|high|unclear",
                    "evidence_quality": "high|medium|low|uninterpretable",
                    "candidate_feature_name": "...",
                    "alternative_explanations": ["..."],
                    "recommended_followup_checks": ["..."],
                    "final_concise_conclusion": "...",
                    "parse_status": "ok|error",
                }],
            },
        },
        "items": items,
    }

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    batch_path = BATCH_DIR / f"task_a_batch_{next_id:03d}_input.json"
    if batch_path.exists() and not getattr(args, "force", False):
        print(f"ERROR: {batch_path} already exists. Use --batch-id with another id or --force.")
        return 1
    save_json(batch_input, batch_path)
    instruction_path = write_batch_instructions(batch_input, batch_path, task_label="Task A Input Explanation")
    write_progress_snapshot(f"prepared task-a batch {next_id}")

    print(f"=== Task A Batch {next_id} Prepared ===")
    print(f"  Batch file: {batch_path}")
    print(f"  Instructions: {instruction_path}")
    print(f"  Items in batch: {len(items)}")
    print(f"  Remaining after this: {max(len(unprocessed) - len(items), 0)}")
    print(f"  Total progress: {len(all_completed)}/{len(prompt_files)}")
    print()
    print("Prompts in this batch:")
    for item in items:
        print(f"  [{item['item_index']}] {item['prompt_filename']} "
              f"({item['target_label']} rank{item['rank_within_label']})")
    print()
    print(f"Output file: {batch_input['output_instructions']['output_file']}")
    return 0


def cmd_task_a_merge(args):
    """Merge all batch outputs into main explanations JSONL."""
    batch_outputs = sorted(BATCH_DIR.glob("task_a_batch_*_output.json")) if BATCH_DIR.exists() else []
    if not batch_outputs:
        print("No batch output files found. Nothing to merge.")
        return 1

    existing = load_jsonl(EXPLANATIONS_JSONL)
    by_pid = {e["packet_id"]: e for e in existing}
    n_new = n_updated = 0

    for bf in batch_outputs:
        try:
            data = load_json(bf)
            model = data.get("model", "current_dialogue_agent")
            for result in data.get("results", []):
                pid = result.get("packet_id")
                if not pid:
                    continue
                record = {
                    "packet_id": pid,
                    "target_label": result.get("target_label", ""),
                    "latent_idx": result.get("latent_idx", 0),
                    "rank_within_label": result.get("rank_within_label", 0),
                    "model": model,
                    "prompt_path": result.get("prompt_filename", ""),
                    "one_sentence_tentative_interpretation": result.get("one_sentence_tentative_interpretation", ""),
                    "main_patterns": result.get("main_patterns", []),
                    "relationship_to_target_label": result.get("relationship_to_target_label", ""),
                    "artifact_risk": result.get("artifact_risk", "unclear"),
                    "evidence_quality": result.get("evidence_quality", "low"),
                    "candidate_feature_name": result.get("candidate_feature_name", ""),
                    "alternative_explanations": result.get("alternative_explanations", []),
                    "recommended_followup_checks": result.get("recommended_followup_checks", []),
                    "final_concise_conclusion": result.get("final_concise_conclusion", ""),
                    "parse_status": result.get("parse_status", "ok"),
                }
                if pid in by_pid:
                    n_updated += 1
                else:
                    n_new += 1
                by_pid[pid] = record
        except Exception as e:
            print(f"Warning: Error reading {bf.name}: {e}")

    all_records = sorted(by_pid.values(), key=lambda r: (r.get("target_label", ""), r.get("rank_within_label", 0)))
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    save_jsonl(all_records, EXPLANATIONS_JSONL)

    # Save CSV
    csv_fields = [
        "packet_id", "target_label", "latent_idx", "rank_within_label", "model",
        "candidate_feature_name", "one_sentence_tentative_interpretation",
        "artifact_risk", "evidence_quality", "relationship_to_target_label",
        "final_concise_conclusion", "parse_status",
    ]
    with open(EXPLANATIONS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)

    # Update manifest
    save_json({
        "analysis_phase": "p3_explanations_runner",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "current_dialogue_agent",
        "n_prompts_found": len(get_prompt_files()),
        "n_success": sum(1 for r in all_records if r.get("parse_status") == "ok"),
        "n_failed": sum(1 for r in all_records if r.get("parse_status") != "ok"),
        "explanations_jsonl": str(EXPLANATIONS_JSONL),
        "explanations_csv": str(EXPLANATIONS_CSV),
        "failed_responses_jsonl": str(FAILED_A_JSONL),
        "raw_responses_dir": str(RAW_RESP_A_DIR),
    }, MANIFEST_A)
    write_progress_snapshot("merged task-a batch outputs")

    print(f"=== Task A Merge Complete ===")
    print(f"  New: {n_new}, Updated: {n_updated}, Total: {len(all_records)}")
    print(f"  Output: {EXPLANATIONS_JSONL}")
    return 0


def cmd_task_a_validate(args):
    """Validate Task A results."""
    prompt_files = get_prompt_files()
    explanations = load_jsonl(EXPLANATIONS_JSONL)
    cards_index = build_cards_index()

    prompt_pids = set()
    for pf in prompt_files:
        label, rank, latent = parse_prompt_filename(pf.name)
        pid = cards_index.get((label, latent, rank))
        if pid:
            prompt_pids.add(pid)

    exp_pids = {e["packet_id"] for e in explanations}
    ok = sum(1 for e in explanations if e.get("parse_status") == "ok")
    missing = prompt_pids - exp_pids

    print(f"=== Task A Validation ===")
    print(f"  Total prompts: {len(prompt_files)}")
    print(f"  Total explanations: {len(explanations)}")
    print(f"  parse_status=ok: {ok}")
    print(f"  parse_status!=ok: {len(explanations) - ok}")
    print(f"  Missing: {len(missing)}")

    if missing:
        for m in sorted(missing)[:10]:
            print(f"    {m}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
        print(f"  STATUS: INCOMPLETE")
    else:
        print(f"  STATUS: COMPLETE")
    write_progress_snapshot("validated task-a")
    return 0 if not missing else 1


# ============================================================
# Task B: Input Scoring
# ============================================================

def cmd_task_b_prepare(args):
    """Prepare next batch of scoring tasks for the current agent."""
    batch_size = args.batch_size  # number of latents per batch

    scoring_tasks = load_jsonl(SCORING_TASKS_JSONL)
    if not scoring_tasks:
        print("ERROR: No scoring tasks found at", SCORING_TASKS_JSONL)
        return 1

    explanations = load_jsonl(EXPLANATIONS_JSONL)
    exp_by_pid = {e["packet_id"]: e for e in explanations}
    if not explanations:
        print("ERROR: No explanations found. Complete Task A first.")
        return 1

    # Determine completed tasks
    completed_preds = load_jsonl(PREDICTIONS_JSONL)
    completed_keys = {(p.get("packet_id", ""), p.get("task_type", "")) for p in completed_preds}
    # Also check batch outputs
    if BATCH_DIR.exists():
        for bf in sorted(BATCH_DIR.glob("task_b_batch_*_output.json")):
            try:
                data = load_json(bf)
                for r in data.get("results", []):
                    completed_keys.add((r.get("packet_id", ""), r.get("task_type", "")))
            except Exception:
                pass

    if getattr(args, "include_completed", False):
        completed_keys = set()

    # Find unprocessed tasks (only those with explanations)
    unprocessed = []
    for task in scoring_tasks:
        key = (task.get("packet_id", ""), task.get("task_type", ""))
        if key not in completed_keys and task.get("packet_id", "") in exp_by_pid:
            unprocessed.append(task)

    if not unprocessed:
        print(f"All scoring tasks processed! Total: {len(scoring_tasks)}")
        print("Run 'task-b merge' then 'task-b metrics'.")
        return 0

    # Group by packet_id (latent), take batch_size/limit latents
    by_pid = defaultdict(list)
    for task in unprocessed:
        by_pid[task["packet_id"]].append(task)

    latent_items = []
    cards_by_pid = {card.get("packet_id"): card for card in load_jsonl(CARDS_JSONL)}
    for pid in by_pid:
        card = cards_by_pid.get(pid, {})
        first_task = by_pid[pid][0]
        latent_items.append({
            "packet_id": pid,
            "target_label": first_task.get("target_label", card.get("target_label", "")),
            "rank_within_label": card.get("rank_within_label", first_task.get("rank_within_label", 0) or 0),
        })
    filtered_latents = apply_common_filters(
        latent_items,
        args,
        label_getter=lambda item: item["target_label"],
        rank_getter=lambda item: item["rank_within_label"],
    )
    if not filtered_latents:
        print("No scoring tasks matched the current filters.")
        print("Try relaxing --label/--rank-min/--rank-max/--start-index/--limit, or use --include-completed.")
        return 0
    latent_pids = [item["packet_id"] for item in filtered_latents]
    batch_tasks = []
    for pid in latent_pids:
        batch_tasks.extend(by_pid[pid])

    next_id = int(getattr(args, "batch_id", None) or next_batch_id("task_b"))

    items = []
    for i, task in enumerate(batch_tasks):
        task = trim_task_examples(task, getattr(args, "max_examples_per_task", None))
        pid = task["packet_id"]
        exp = exp_by_pid.get(pid, {})
        items.append({
            "item_index": i,
            "packet_id": pid,
            "target_label": task.get("target_label", ""),
            "latent_idx": task.get("latent_idx", 0),
            "task_type": task.get("task_type", ""),
            "instructions": task.get("instructions", ""),
            "candidate_explanation": exp.get("one_sentence_tentative_interpretation", ""),
            "candidate_feature_name": exp.get("candidate_feature_name", ""),
            "artifact_risk": exp.get("artifact_risk", ""),
            "evidence_quality": exp.get("evidence_quality", ""),
            "final_concise_conclusion": exp.get("final_concise_conclusion", ""),
            "examples": task.get("examples", []),
        })

    batch_input = {
        "batch_info": {
            "batch_id": next_id,
            "task_type": "scoring",
            "items_in_batch": len(items),
            "latents_in_batch": len(latent_pids),
            "total_remaining_tasks": len(unprocessed),
            "filtered_latents": len(filtered_latents),
            "total_tasks": len(scoring_tasks),
            "estimated_batches_remaining": math.ceil(len(by_pid) / max(batch_size, 1)),
            "label_filter": sorted(parse_labels_arg(getattr(args, "label", None))),
            "rank_min": getattr(args, "rank_min", None),
            "rank_max": getattr(args, "rank_max", None),
            "start_index": getattr(args, "start_index", 0),
            "limit": getattr(args, "limit", None) or batch_size,
            "include_completed": bool(getattr(args, "include_completed", False)),
            "max_examples_per_task": getattr(args, "max_examples_per_task", None),
            "created_at": datetime.datetime.now().isoformat(),
        },
        "output_instructions": {
            "output_file": str(BATCH_DIR / f"task_b_batch_{next_id:03d}_output.json"),
            "description": (
                "For each item, read the candidate_explanation, "
                "then predict for each example whether it matches the expected outcome. "
                "For activation_prediction_task: predict high_activation (1) or not (0). "
                "For code_discrimination_task: predict target_match (1) or not (0)."
            ),
            "output_schema": {
                "batch_id": next_id,
                "model": "current_dialogue_agent",
                "completed_at": "ISO_TIMESTAMP",
                "results": [{
                    "item_index": 0,
                    "packet_id": "packet_XXXX",
                    "target_label": "LABEL",
                    "latent_idx": 0,
                    "task_type": "activation_prediction_task|code_discrimination_task",
                    "predictions": [{
                        "row_idx": 0,
                        "predicted": 1,
                        "confidence": 0.85,
                        "rationale_short": "...",
                    }],
                }],
            },
        },
        "items": items,
    }

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    batch_path = BATCH_DIR / f"task_b_batch_{next_id:03d}_input.json"
    if batch_path.exists() and not getattr(args, "force", False):
        print(f"ERROR: {batch_path} already exists. Use --batch-id with another id or --force.")
        return 1
    save_json(batch_input, batch_path)
    instruction_path = write_batch_instructions(batch_input, batch_path, task_label="Task B Input Scoring")
    write_progress_snapshot(f"prepared task-b batch {next_id}")

    n_examples = sum(len(item["examples"]) for item in items)
    print(f"=== Task B Batch {next_id} Prepared ===")
    print(f"  Batch file: {batch_path}")
    print(f"  Instructions: {instruction_path}")
    print(f"  Tasks: {len(items)} ({len(latent_pids)} latents)")
    print(f"  Examples to predict: {n_examples}")
    print(f"  Remaining: {len(unprocessed) - len(batch_tasks)} tasks")
    print()
    for pid in latent_pids:
        tasks = by_pid[pid]
        label = tasks[0].get("target_label", "?")
        print(f"  {pid} ({label}): {len(tasks)} tasks")
    return 0


def cmd_task_b_merge(args):
    """Merge all batch outputs into main predictions JSONL."""
    batch_outputs = sorted(BATCH_DIR.glob("task_b_batch_*_output.json")) if BATCH_DIR.exists() else []
    if not batch_outputs:
        print("No batch output files found.")
        return 1

    existing = load_jsonl(PREDICTIONS_JSONL)
    by_key = {}
    for p in existing:
        key = (p.get("packet_id", ""), p.get("task_type", ""), p.get("row_idx", 0))
        predicted = p.get("predicted", p.get("predicted_value", 0))
        confidence = p.get("confidence", 0.5)
        rationale_short = p.get("rationale_short", "")
        packet_id = p.get("packet_id", "")
        task_type = p.get("task_type", "")
        row_idx = p.get("row_idx", 0)
        target_label = p.get("target_label", "")
        latent_idx = p.get("latent_idx", 0)
        model = p.get("model", "historical")
        
        standard_record = {
            "packet_id": packet_id,
            "target_label": target_label,
            "latent_idx": latent_idx,
            "task_type": task_type,
            "row_idx": row_idx,
            "predicted": predicted,
            "confidence": confidence,
            "rationale_short": rationale_short,
            "model": model,
        }
        by_key[key] = standard_record

    n_new = 0
    for bf in batch_outputs:
        try:
            data = load_json(bf)
            model = data.get("model", "current_dialogue_agent")
            for result in data.get("results", []):
                pid = result.get("packet_id", "")
                tt = result.get("task_type", "")
                for pred in result.get("predictions", []):
                    key = (pid, tt, pred.get("row_idx", 0))
                    predicted = pred.get("predicted", pred.get("predicted_value", 0))
                    record = {
                        "packet_id": pid,
                        "target_label": result.get("target_label", ""),
                        "latent_idx": result.get("latent_idx", 0),
                        "task_type": tt,
                        "row_idx": pred.get("row_idx", 0),
                        "predicted": predicted,
                        "confidence": pred.get("confidence", 0.5),
                        "rationale_short": pred.get("rationale_short", ""),
                        "model": model,
                    }
                    if key not in by_key:
                        n_new += 1
                    by_key[key] = record
        except Exception as e:
            print(f"Warning: Error reading {bf.name}: {e}")

    all_records = sorted(by_key.values(),
                         key=lambda r: (r.get("packet_id", ""), r.get("task_type", ""), r.get("row_idx", 0)))
    SCORING_DIR.mkdir(parents=True, exist_ok=True)
    save_jsonl(all_records, PREDICTIONS_JSONL)

    save_json({
        "analysis_phase": "p3_scoring_runner_agent",
        "n_predictions": len(all_records),
        "n_latents_scored": len({r["packet_id"] for r in all_records}),
        "predictions_jsonl": str(PREDICTIONS_JSONL),
        "metrics_by_latent_csv": str(METRICS_BY_LATENT),
        "metrics_by_label_csv": str(METRICS_BY_LABEL),
        "metrics_overall_json": str(METRICS_OVERALL),
    }, MANIFEST_B)
    write_progress_snapshot("merged task-b batch outputs")

    print(f"=== Task B Merge Complete ===")
    print(f"  New: {n_new}, Total: {len(all_records)}")
    print("Run 'task-b metrics' to compute evaluation metrics.")
    return 0


def cmd_task_b_metrics(args):
    """Compute scoring metrics from all predictions."""
    predictions = load_jsonl(PREDICTIONS_JSONL)
    scoring_tasks = load_jsonl(SCORING_TASKS_JSONL)

    if not predictions:
        print("ERROR: No predictions found. Run Task B first.")
        return 1

    # Build expected values
    expected = {}
    for task in scoring_tasks:
        pid, tt = task.get("packet_id", ""), task.get("task_type", "")
        for ex in task.get("examples", []):
            rid = ex.get("row_idx", 0)
            if tt == "activation_prediction_task":
                expected[(pid, tt, rid)] = ex.get("expected_high_activation", 0)
            else:
                expected[(pid, tt, rid)] = ex.get("expected_target_match", 0)

    # Match predictions
    matched = []
    for pred in predictions:
        key = (pred.get("packet_id", ""), pred.get("task_type", ""), pred.get("row_idx", 0))
        if key in expected:
            matched.append({
                "packet_id": pred["packet_id"],
                "target_label": pred.get("target_label", ""),
                "latent_idx": pred.get("latent_idx", 0),
                "task_type": pred["task_type"],
                "row_idx": pred["row_idx"],
                "y_true": expected[key],
                "y_pred": pred.get("predicted", 0),
                "confidence": pred.get("confidence", 0.5),
            })

    if not matched:
        print("ERROR: No predictions matched scoring task examples.")
        return 1

    # --- By-latent metrics ---
    by_latent = defaultdict(list)
    for m in matched:
        by_latent[(m["packet_id"], m["target_label"], m["latent_idx"], m["task_type"])].append(m)

    latent_metrics = []
    for (pid, label, lidx, tt), items in sorted(by_latent.items()):
        y_true = [i["y_true"] for i in items]
        y_pred = [i["y_pred"] for i in items]
        y_scores = [i["confidence"] for i in items]
        metrics = compute_binary_metrics(y_true, y_pred, y_scores)
        metrics.update({"packet_id": pid, "target_label": label, "latent_idx": lidx, "task_type": tt})
        latent_metrics.append(metrics)

    fields = ["packet_id", "target_label", "latent_idx", "task_type",
              "accuracy", "f1", "precision", "recall", "balanced_accuracy", "auroc",
              "tp", "tn", "fp", "fn", "n"]
    with open(METRICS_BY_LATENT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(latent_metrics)

    # --- By-label metrics ---
    by_label = defaultdict(list)
    for m in matched:
        by_label[m["target_label"]].append(m)

    label_metrics = []
    for label, items in sorted(by_label.items()):
        y_true = [i["y_true"] for i in items]
        y_pred = [i["y_pred"] for i in items]
        y_scores = [i["confidence"] for i in items]
        metrics = compute_binary_metrics(y_true, y_pred, y_scores)
        metrics["target_label"] = label
        metrics["n_latents"] = len({i["packet_id"] for i in items})
        label_metrics.append(metrics)

    fields_label = ["target_label", "n_latents", "accuracy", "f1", "precision",
                     "recall", "balanced_accuracy", "auroc", "n"]
    with open(METRICS_BY_LABEL, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields_label, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(label_metrics)

    # --- Overall metrics ---
    y_true_all = [m["y_true"] for m in matched]
    y_pred_all = [m["y_pred"] for m in matched]
    y_scores_all = [m["confidence"] for m in matched]
    overall = compute_binary_metrics(y_true_all, y_pred_all, y_scores_all)
    overall["n_latents"] = len({m["packet_id"] for m in matched})
    overall["n_labels"] = len({m["target_label"] for m in matched})
    overall["n_predictions_matched"] = len(matched)
    save_json(overall, METRICS_OVERALL)

    print(f"=== Task B Metrics ===")
    print(f"  Matched predictions: {len(matched)}")
    print(f"  Overall accuracy: {overall.get('accuracy', 'N/A')}")
    print(f"  Overall F1: {overall.get('f1', 'N/A')}")
    print(f"  Overall AUROC: {overall.get('auroc', 'N/A')}")
    print(f"  Files: {METRICS_BY_LATENT}, {METRICS_BY_LABEL}, {METRICS_OVERALL}")
    write_progress_snapshot("computed task-b metrics")
    return 0


# ============================================================
# Task C: Manual Review Template
# ============================================================

def cmd_task_c_run(args):
    """Generate manual review template CSV (pure Python, no AI needed)."""
    cards = load_jsonl(CARDS_JSONL)
    if not cards:
        print("ERROR: No cards found at", CARDS_JSONL)
        return 1

    explanations = load_jsonl(EXPLANATIONS_JSONL)
    exp_by_pid = {e["packet_id"]: e for e in explanations}

    latent_metrics = {}
    if METRICS_BY_LATENT.exists():
        with open(METRICS_BY_LATENT, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                latent_metrics[(row.get("packet_id", ""), row.get("task_type", ""))] = row

    rows = []
    for card in cards:
        pid = card.get("packet_id", "")
        exp = exp_by_pid.get(pid, {})
        act_m = latent_metrics.get((pid, "activation_prediction_task"), {})
        code_m = latent_metrics.get((pid, "code_discrimination_task"), {})
        patterns = exp.get("main_patterns", [])

        rows.append({
            "packet_id": pid,
            "target_label": card.get("target_label", ""),
            "latent_idx": card.get("latent_idx", ""),
            "rank_within_label": card.get("rank_within_label", ""),
            "candidate_feature_name": exp.get("candidate_feature_name", ""),
            "one_sentence_tentative_interpretation": exp.get("one_sentence_tentative_interpretation", ""),
            "dominant_pattern_type": patterns[0].get("pattern_type", "") if patterns else "",
            "artifact_risk_ai": exp.get("artifact_risk", ""),
            "evidence_quality_ai": exp.get("evidence_quality", ""),
            "activation_prediction_auroc": act_m.get("auroc", ""),
            "activation_prediction_accuracy": act_m.get("accuracy", ""),
            "code_discrimination_accuracy": code_m.get("accuracy", ""),
            "code_discrimination_f1": code_m.get("f1", ""),
            "output_delta_target_logit": "",
            "mi_coder_label": "",
            "mi_coder_confidence": "",
            "mi_coder_artifact_risk": "",
            "final_status": "",
            "reviewer_notes": "",
        })

    rows.sort(key=lambda r: (r.get("target_label", ""), int(r.get("rank_within_label", 0) or 0)))
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(REVIEW_TEMPLATE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"=== Task C Complete ===")
    print(f"  Template: {REVIEW_TEMPLATE}")
    print(f"  Rows: {len(rows)}")
    write_progress_snapshot("generated task-c manual review template")
    return 0


# ============================================================
# Task D: Final Feature Cards
# ============================================================

def cmd_task_d_run(args):
    """Generate final feature cards (pure Python, no AI needed)."""
    cards = load_jsonl(CARDS_JSONL)
    if not cards:
        print("ERROR: No cards found at", CARDS_JSONL)
        return 1

    explanations = load_jsonl(EXPLANATIONS_JSONL)
    exp_by_pid = {e["packet_id"]: e for e in explanations}

    latent_metrics = {}
    if METRICS_BY_LATENT.exists():
        with open(METRICS_BY_LATENT, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                latent_metrics[(row.get("packet_id", ""), row.get("task_type", ""))] = row

    # Load manual review if available
    manual_review = {}
    manual_completed = MANUAL_DIR / "p3_mi_coder_review_completed.csv"
    review_file = manual_completed if manual_completed.exists() else REVIEW_TEMPLATE
    if review_file.exists():
        with open(review_file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                manual_review[row.get("packet_id", "")] = row

    final_cards = []
    label_counts = defaultdict(lambda: defaultdict(int))

    for card in cards:
        pid = card.get("packet_id", "")
        exp = exp_by_pid.get(pid, {})
        manual = manual_review.get(pid, {})
        act_m = latent_metrics.get((pid, "activation_prediction_task"), {})
        code_m = latent_metrics.get((pid, "code_discrimination_task"), {})

        final_status = manual.get("final_status", "")
        if not final_status:
            ar = exp.get("artifact_risk", "unclear")
            eq = exp.get("evidence_quality", "low")
            if ar == "high":
                final_status = "surface_artifact"
            elif eq in ("high", "medium") and ar == "low":
                final_status = "robust_code_candidate"
            else:
                final_status = "mixed_unclear"

        label = card.get("target_label", "")
        label_counts[label][final_status] += 1

        fc = {**card}
        fc.update({
            "input_centric_explanation": exp.get("one_sentence_tentative_interpretation", ""),
            "candidate_feature_name": exp.get("candidate_feature_name", ""),
            "main_patterns": exp.get("main_patterns", []),
            "artifact_risk": exp.get("artifact_risk", "unclear"),
            "evidence_quality": exp.get("evidence_quality", "low"),
            "relationship_to_target_label": exp.get("relationship_to_target_label", ""),
            "alternative_explanations": exp.get("alternative_explanations", []),
            "final_concise_conclusion": exp.get("final_concise_conclusion", ""),
            "activation_prediction_accuracy": act_m.get("accuracy", ""),
            "activation_prediction_auroc": act_m.get("auroc", ""),
            "code_discrimination_accuracy": code_m.get("accuracy", ""),
            "code_discrimination_f1": code_m.get("f1", ""),
            "output_centric_explanation": "",
            "output_effect": "",
            "mi_coder_judgment": manual.get("mi_coder_label", ""),
            "final_status": final_status,
            "limitations": card.get("context_limitation", ""),
        })
        fc["dry_run_explanation_slots"] = {
            "input_centric_explanation": "completed" if exp else "pending",
            "artifact_risk": exp.get("artifact_risk", "pending"),
            "mi_coder_judgment": manual.get("mi_coder_label", "pending"),
            "final_status": final_status,
        }
        fc["scoring_slots"] = {
            "activation_prediction_score": act_m.get("accuracy", "pending_score"),
            "code_discrimination_score": code_m.get("accuracy", "pending_score"),
        }
        final_cards.append(fc)

    final_cards.sort(key=lambda c: (c.get("target_label", ""), c.get("rank_within_label", 0)))

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    save_jsonl(final_cards, FINAL_JSONL)

    # CSV
    csv_fields = [
        "packet_id", "target_label", "latent_idx", "rank_within_label",
        "candidate_feature_name", "input_centric_explanation",
        "artifact_risk", "evidence_quality", "final_status",
        "activation_prediction_accuracy", "activation_prediction_auroc",
        "code_discrimination_accuracy", "code_discrimination_f1",
        "mi_coder_judgment", "limitations",
    ]
    with open(FINAL_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_cards)

    # Markdown
    by_label = defaultdict(list)
    for c in final_cards:
        by_label[c.get("target_label", "")].append(c)

    with open(FINAL_MD, "w", encoding="utf-8") as f:
        f.write(f"# P3 Final Feature Cards\n\nGenerated: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Total cards: {len(final_cards)}\n\n")
        for label in sorted(by_label):
            lc = by_label[label]
            f.write(f"## {label}\n\nCards: {len(lc)}\n\n")
            for c in lc:
                f.write(f"### {label} rank{c.get('rank_within_label','')} - latent {c.get('latent_idx','')}\n\n")
                f.write(f"- **Feature ID**: {c.get('packet_id','')}\n")
                lm = c.get("layer_metadata", {})
                f.write(f"- **Layer**: {lm.get('sae_canonical_layer','')}\n")
                f.write(f"- **Candidate name**: {c.get('candidate_feature_name','')}\n")
                f.write(f"- **Explanation**: {c.get('input_centric_explanation','')}\n")
                f.write(f"- **Artifact risk**: {c.get('artifact_risk','')}\n")
                f.write(f"- **Evidence quality**: {c.get('evidence_quality','')}\n")
                f.write(f"- **Act. pred. acc**: {c.get('activation_prediction_accuracy','')}\n")
                f.write(f"- **Code disc. acc**: {c.get('code_discrimination_accuracy','')}\n")
                f.write(f"- **Final status**: {c.get('final_status','')}\n")
                f.write(f"- **Conclusion**: {c.get('final_concise_conclusion','')}\n\n")

    # Label summary
    summary_rows = []
    for label in sorted(label_counts):
        counts = label_counts[label]
        summary_rows.append({
            "target_label": label,
            "total": sum(counts.values()),
            "robust_code_candidate": counts.get("robust_code_candidate", 0),
            "family_level_candidate": counts.get("family_level_candidate", 0),
            "subskill_candidate": counts.get("subskill_candidate", 0),
            "surface_artifact": counts.get("surface_artifact", 0),
            "mixed_unclear": counts.get("mixed_unclear", 0),
            "reject": counts.get("reject", 0),
        })

    with open(LABEL_SUMMARY, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(CLUSTER_SEED, "w", encoding="utf-8") as f:
        f.write("")

    review_status = "manual_completed" if manual_completed.exists() else "manual_draft"
    save_json({
        "analysis_phase": "p3_final_feature_cards_builder",
        "review_status": review_status,
        "n_cards_total": len(final_cards),
        "n_seeds_extracted": 0,
        "final_feature_cards_jsonl": str(FINAL_JSONL),
        "final_feature_cards_csv": str(FINAL_CSV),
        "final_feature_cards_md": str(FINAL_MD),
        "label_summary_csv": str(LABEL_SUMMARY),
        "concept_cluster_seed_table_csv": str(CLUSTER_SEED),
    }, MANIFEST_D)

    print(f"=== Task D Complete ===")
    print(f"  Cards: {len(final_cards)}")
    print(f"  JSONL: {FINAL_JSONL}")
    print(f"  CSV: {FINAL_CSV}")
    print(f"  MD: {FINAL_MD}")
    print()
    for row in summary_rows:
        print(f"  {row['target_label']}: {row['total']} total, "
              f"{row['robust_code_candidate']} robust, "
              f"{row['surface_artifact']} artifact, "
              f"{row['mixed_unclear']} mixed")
    write_progress_snapshot("generated task-d final cards")
    return 0


# ============================================================
# Status
# ============================================================

def cmd_status(args):
    """Show overall progress for all tasks."""
    print("=" * 60)
    print("P3 Agent Batch Orchestrator - Status")
    print("=" * 60)
    print()

    # Task A
    prompt_files = get_prompt_files()
    explanations = load_jsonl(EXPLANATIONS_JSONL)
    ba_in = sorted(BATCH_DIR.glob("task_a_batch_*_input.json")) if BATCH_DIR.exists() else []
    ba_out = sorted(BATCH_DIR.glob("task_a_batch_*_output.json")) if BATCH_DIR.exists() else []

    print(f"Task A: Input Explanations")
    print(f"  Prompts:          {len(prompt_files)}")
    print(f"  Explanations:     {len(explanations)}")
    print(f"  Batches in/out:   {len(ba_in)}/{len(ba_out)}")
    status_a = "COMPLETE" if len(explanations) >= len(prompt_files) > 0 else f"{len(explanations)}/{len(prompt_files)}"
    print(f"  Status:           {status_a}")
    print()

    # Task B
    scoring_tasks = load_jsonl(SCORING_TASKS_JSONL)
    predictions = load_jsonl(PREDICTIONS_JSONL)
    expected_preds = sum(len(t.get("examples", [])) for t in scoring_tasks)
    bb_in = sorted(BATCH_DIR.glob("task_b_batch_*_input.json")) if BATCH_DIR.exists() else []
    bb_out = sorted(BATCH_DIR.glob("task_b_batch_*_output.json")) if BATCH_DIR.exists() else []

    print(f"Task B: Input Scoring")
    print(f"  Tasks:            {len(scoring_tasks)}")
    print(f"  Expected preds:   {expected_preds}")
    print(f"  Predictions:      {len(predictions)}")
    print(f"  Batches in/out:   {len(bb_in)}/{len(bb_out)}")
    if METRICS_OVERALL.exists():
        ov = load_json(METRICS_OVERALL)
        print(f"  Accuracy/F1:      {ov.get('accuracy','N/A')}/{ov.get('f1','N/A')}")
    status_b = "COMPLETE" if len(predictions) >= expected_preds > 0 else f"{len(predictions)}/{expected_preds}"
    print(f"  Status:           {status_b}")
    print()

    # Task C
    print(f"Task C: Manual Review Template")
    if REVIEW_TEMPLATE.exists():
        with open(REVIEW_TEMPLATE, "r", encoding="utf-8") as f:
            n_rows = sum(1 for _ in f) - 1
        print(f"  Rows:             {n_rows}")
        print(f"  Status:           GENERATED")
    else:
        print(f"  Status:           NOT GENERATED")
    print()

    # Task D
    final_cards = load_jsonl(FINAL_JSONL)
    print(f"Task D: Final Feature Cards")
    print(f"  Cards:            {len(final_cards)}")
    md_ok = FINAL_MD.exists()
    print(f"  MD report:        {'yes' if md_ok else 'no'}")
    status_d = "GENERATED" if len(final_cards) > 0 else "NOT GENERATED"
    print(f"  Status:           {status_d}")
    print()

    if BATCH_DIR.exists():
        all_b = list(BATCH_DIR.glob("*.json"))
        total_kb = sum(f.stat().st_size for f in all_b) / 1024
        print(f"Batch dir: {BATCH_DIR} ({len(all_b)} files, {total_kb:.0f} KB)")

    progress = write_progress_snapshot("status")
    print(f"Progress JSON: {PROGRESS_JSON}")
    print(f"Pipeline complete: {all(progress[key].get('complete', False) for key in ('task_a', 'task_b', 'task_c', 'task_d'))}")
    return 0


def cmd_validate(args):
    """Validate the full P3 agent pipeline against expected artifact counts."""
    progress = collect_progress()
    expected = expected_artifact_counts()
    checks = []

    def add_check(name, passed, expected, observed, path=""):
        checks.append({
            "name": name,
            "passed": bool(passed),
            "expected": expected,
            "observed": observed,
            "path": str(path),
        })

    # Validate predictions schema
    predictions = load_jsonl(PREDICTIONS_JSONL)
    required_fields = {"predicted", "confidence", "packet_id", "task_type", "row_idx"}
    bad_pred_count = 0
    for p in predictions:
        if not required_fields.issubset(p.keys()):
            bad_pred_count += 1
    pred_fields_ok = (bad_pred_count == 0)

    add_check(
        "task_a_explanations_count",
        expected["prompts"] > 0 and progress["task_a"]["explanations"] == progress["task_a"]["prompts"] == expected["prompts"],
        f"{expected['prompts']} explanations for {expected['prompts']} prompts",
        f"{progress['task_a']['explanations']} / {progress['task_a']['prompts']}",
        EXPLANATIONS_JSONL,
    )
    add_check(
        "task_a_parse_ok_count",
        expected["prompts"] > 0 and progress["task_a"]["parse_ok"] == progress["task_a"]["prompts"] == expected["prompts"],
        f"{expected['prompts']} parse_status=ok explanations",
        progress["task_a"]["parse_ok"],
        EXPLANATIONS_JSONL,
    )
    add_check(
        "task_b_prediction_count",
        expected["expected_predictions"] > 0
        and progress["task_b"]["predictions"] == progress["task_b"]["expected_predictions"] == expected["expected_predictions"],
        f"{expected['expected_predictions']} predictions for all scoring examples",
        f"{progress['task_b']['predictions']} / {progress['task_b']['expected_predictions']}",
        PREDICTIONS_JSONL,
    )
    add_check(
        "task_b_prediction_schema",
        expected["expected_predictions"] > 0 and pred_fields_ok and len(predictions) == expected["expected_predictions"],
        f"All {expected['expected_predictions']} predictions have correct schema fields",
        f"Passed: {len(predictions) - bad_pred_count} / {len(predictions)}",
        PREDICTIONS_JSONL,
    )
    add_check(
        "task_b_metrics_by_latent",
        expected["metrics_by_latent_rows"] > 0
        and progress["task_b"]["metrics_by_latent_rows"] == expected["metrics_by_latent_rows"],
        f"{expected['metrics_by_latent_rows']} rows, one per latent x task_type",
        progress["task_b"]["metrics_by_latent_rows"],
        METRICS_BY_LATENT,
    )
    add_check(
        "task_b_metrics_by_label",
        expected["metrics_by_label_rows"] > 0
        and progress["task_b"]["metrics_by_label_rows"] == expected["metrics_by_label_rows"],
        f"{expected['metrics_by_label_rows']} label-level rows",
        progress["task_b"]["metrics_by_label_rows"],
        METRICS_BY_LABEL,
    )
    add_check(
        "task_c_manual_review_template",
        expected["cards"] > 0 and progress["task_c"]["manual_review_template_rows"] == expected["cards"],
        f"{expected['cards']} manual review template rows",
        progress["task_c"]["manual_review_template_rows"],
        REVIEW_TEMPLATE,
    )
    add_check(
        "task_d_final_cards",
        expected["cards"] > 0 and progress["task_d"]["final_cards"] == expected["cards"] and progress["task_d"]["markdown_exists"],
        f"{expected['cards']} final cards plus markdown report",
        f"{progress['task_d']['final_cards']} cards, md={progress['task_d']['markdown_exists']}",
        FINAL_JSONL,
    )

    validation = {
        "timestamp": datetime.datetime.now().isoformat(),
        "all_passed": all(check["passed"] for check in checks),
        "checks": checks,
        "progress": progress,
        "expected": expected,
    }
    save_json(validation, VALIDATION_JSON)
    write_progress_snapshot("validated full p3 agent pipeline")

    print("=" * 60)
    print("P3 Agent Pipeline Validation")
    print("=" * 60)
    for check in checks:
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"[{mark}] {check['name']}: observed={check['observed']} expected={check['expected']}")
    print(f"Validation JSON: {VALIDATION_JSON}")
    print(f"Overall: {'PASS' if validation['all_passed'] else 'FAIL'}")
    return 0 if validation["all_passed"] else 1


def cmd_next_batch(args):
    """Prepare the next actionable Task A/B batch for the current agent."""
    progress = collect_progress()
    if not progress["task_a"]["complete"]:
        print("Next stage: Task A input explanation")
        task_args = argparse.Namespace(
            batch_size=args.batch_size,
            limit=args.limit,
            label=args.label,
            rank_min=args.rank_min,
            rank_max=args.rank_max,
            start_index=args.start_index,
            batch_id=args.batch_id,
            include_completed=args.include_completed,
            max_examples_per_group=args.max_examples_per_group,
            force=args.force,
        )
        return cmd_task_a_prepare(task_args)

    if not progress["task_b"]["complete"]:
        print("Next stage: Task B input scoring")
        task_args = argparse.Namespace(
            batch_size=args.batch_size,
            limit=args.limit,
            label=args.label,
            rank_min=args.rank_min,
            rank_max=args.rank_max,
            start_index=args.start_index,
            batch_id=args.batch_id,
            include_completed=args.include_completed,
            max_examples_per_task=args.max_examples_per_task,
            force=args.force,
        )
        return cmd_task_b_prepare(task_args)

    print("No Task A/B batches are pending.")
    if not progress["task_c"]["complete"]:
        print("Next pure-Python step: python run_misc_p3_agent_orchestrator.py task-c run")
    elif not progress["task_d"]["complete"]:
        print("Next pure-Python step: python run_misc_p3_agent_orchestrator.py task-d run")
    else:
        print("Task A-D appear complete. Run validate for a final artifact count check.")
    write_progress_snapshot("next-batch found no pending Task A/B work")
    return 0


def _has_batch_outputs(pattern):
    return BATCH_DIR.exists() and any(BATCH_DIR.glob(pattern))


def _latest_unanswered_batch(prefix):
    if not BATCH_DIR.exists():
        return None
    inputs = sorted(BATCH_DIR.glob(f"{prefix}_batch_*_input.json"))
    for input_path in reversed(inputs):
        output_path = input_path.with_name(input_path.name.replace("_input.json", "_output.json"))
        if not output_path.exists():
            return input_path
    return None


def _print_agent_handoff():
    print()
    print("=" * 60)
    print("Current-agent handoff")
    print("=" * 60)
    print("No model API is called by this pipeline.")
    print("Use the current dialogue model to read the generated batch input JSON,")
    print("produce the requested structured judgments, and write the batch output JSON.")
    print("Then run this command again:")
    print()
    print("  python run_misc_p3_agent_orchestrator.py 启动p3")
    print()


def cmd_start_p3(args):
    """Start or continue P3 using file batches for the current dialogue agent."""
    print("=" * 60)
    print("P3 current-agent workflow")
    print("=" * 60)
    print("Mode: file-based agent batches; no API keys, no external model runner.")
    print()

    if not CARDS_JSONL.exists() or not SCORING_TASKS_JSONL.exists() or not PROMPTS_DIR.exists():
        print("P3 dry-run inputs are missing. Generate them first:")
        print()
        print("  python run_misc_p3_feature_cards.py")
        print()
        return 1

    progress = collect_progress()

    if not progress["task_a"]["complete"]:
        if _has_batch_outputs("task_a_batch_*_output.json"):
            print("Found Task A batch outputs; merging before preparing more work.")
            cmd_task_a_merge(argparse.Namespace())
            progress = collect_progress()

        if not progress["task_a"]["complete"]:
            pending = _latest_unanswered_batch("task_a")
            if pending is not None:
                print("Task A already has a pending batch input without output.")
                print(f"Batch input: {pending}")
                print(f"Instructions: {pending.with_name(pending.name.replace('_input.json', '_instructions.md'))}")
                print(f"Expected output: {pending.with_name(pending.name.replace('_input.json', '_output.json'))}")
                _print_agent_handoff()
                return 0
            print("Next required reasoning stage: Task A input explanations.")
            rc = cmd_next_batch(args)
            _print_agent_handoff()
            return rc

    if not progress["task_b"]["complete"]:
        if _has_batch_outputs("task_b_batch_*_output.json"):
            print("Found Task B batch outputs; merging and recomputing metrics before preparing more work.")
            cmd_task_b_merge(argparse.Namespace())
            cmd_task_b_metrics(argparse.Namespace())
            progress = collect_progress()

        if not progress["task_b"]["complete"]:
            pending = _latest_unanswered_batch("task_b")
            if pending is not None:
                print("Task B already has a pending batch input without output.")
                print(f"Batch input: {pending}")
                print(f"Instructions: {pending.with_name(pending.name.replace('_input.json', '_instructions.md'))}")
                print(f"Expected output: {pending.with_name(pending.name.replace('_input.json', '_output.json'))}")
                _print_agent_handoff()
                return 0
            print("Next required reasoning stage: Task B held-out scoring.")
            rc = cmd_next_batch(args)
            _print_agent_handoff()
            return rc

    if not progress["task_c"]["complete"]:
        print("Task A/B are complete. Generating Task C manual review template.")
        rc = cmd_task_c_run(argparse.Namespace())
        if rc:
            return rc
        progress = collect_progress()

    if not progress["task_d"]["complete"]:
        print("Generating Task D final feature cards.")
        rc = cmd_task_d_run(argparse.Namespace())
        if rc:
            return rc
        progress = collect_progress()

    print("P3 stages A-D appear complete. Running final validation.")
    return cmd_validate(argparse.Namespace())


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="P3 Agent Batch Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_misc_p3_agent_orchestrator.py 启动p3 --batch-size 3
  python run_misc_p3_agent_orchestrator.py start-p3 --batch-size 3
  python run_misc_p3_agent_orchestrator.py status
  python run_misc_p3_agent_orchestrator.py next-batch --batch-size 3 --label RE
  python run_misc_p3_agent_orchestrator.py task-a prepare --batch-size 5
  python run_misc_p3_agent_orchestrator.py task-a prepare --label RES --rank-min 4 --rank-max 10 --limit 3
  python run_misc_p3_agent_orchestrator.py task-a merge
  python run_misc_p3_agent_orchestrator.py task-a validate
  python run_misc_p3_agent_orchestrator.py task-b prepare --batch-size 10
  python run_misc_p3_agent_orchestrator.py task-b prepare --label QUO --limit 2 --max-examples-per-task 6
  python run_misc_p3_agent_orchestrator.py task-b merge
  python run_misc_p3_agent_orchestrator.py task-b metrics
  python run_misc_p3_agent_orchestrator.py task-c run
  python run_misc_p3_agent_orchestrator.py task-d run
  python run_misc_p3_agent_orchestrator.py validate
        """,
    )
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_BASE_DIR),
        help="P3 feature-card artifact directory. Defaults to the stable_core P3 output.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show overall progress").set_defaults(func=cmd_status)
    sub.add_parser("validate", help="Validate all P3 agent artifacts").set_defaults(func=cmd_validate)

    def add_start_parser(name, help_text):
        p_start = sub.add_parser(name, help=help_text)
        p_start.add_argument("--batch-size", type=int, default=3, help="Items/latents per batch")
        p_start.add_argument("--limit", type=int, default=None, help="Override batch-size after filters")
        p_start.add_argument("--label", nargs="*", default=None, help="Optional label filter, e.g. RE RES or RE,RES")
        p_start.add_argument("--rank-min", type=int, default=None)
        p_start.add_argument("--rank-max", type=int, default=None)
        p_start.add_argument("--start-index", type=int, default=0, help="Offset within filtered pending items")
        p_start.add_argument("--batch-id", type=int, default=None, help="Explicit batch id")
        p_start.add_argument("--include-completed", action="store_true", help="Regenerate already completed items")
        p_start.add_argument("--max-examples-per-group", type=int, default=None, help="Task A prompt example cap per group")
        p_start.add_argument("--max-examples-per-task", type=int, default=None, help="Task B example cap per task")
        p_start.add_argument("--force", action="store_true", help="Overwrite an existing batch input id")
        p_start.set_defaults(func=cmd_start_p3)

    add_start_parser("启动p3", "Start or continue P3 with the current dialogue agent")
    add_start_parser("start-p3", "Start or continue P3 with the current dialogue agent")

    p_next = sub.add_parser("next-batch", help="Prepare the next pending Task A/B batch")
    p_next.add_argument("--batch-size", type=int, default=3, help="Items/latents per batch")
    p_next.add_argument("--limit", type=int, default=None, help="Override batch-size after filters")
    p_next.add_argument("--label", nargs="*", default=None, help="Optional label filter, e.g. RE RES or RE,RES")
    p_next.add_argument("--rank-min", type=int, default=None)
    p_next.add_argument("--rank-max", type=int, default=None)
    p_next.add_argument("--start-index", type=int, default=0, help="Offset within filtered pending items")
    p_next.add_argument("--batch-id", type=int, default=None, help="Explicit batch id")
    p_next.add_argument("--include-completed", action="store_true", help="Regenerate already completed items")
    p_next.add_argument("--max-examples-per-group", type=int, default=None, help="Task A prompt example cap per group")
    p_next.add_argument("--max-examples-per-task", type=int, default=None, help="Task B example cap per task")
    p_next.add_argument("--force", action="store_true", help="Overwrite an existing batch input id")
    p_next.set_defaults(func=cmd_next_batch)

    # task-a
    sp_a = sub.add_parser("task-a", help="Task A: Input explanations")
    sa = sp_a.add_subparsers(dest="subcommand")
    p = sa.add_parser("prepare", help="Prepare next batch")
    p.add_argument("--batch-size", type=int, default=5, help="Prompts per batch (default: 5)")
    p.add_argument("--limit", type=int, default=None, help="Override batch-size after filters")
    p.add_argument("--label", nargs="*", default=None, help="Optional label filter, e.g. RE RES or RE,RES")
    p.add_argument("--rank-min", type=int, default=None)
    p.add_argument("--rank-max", type=int, default=None)
    p.add_argument("--start-index", type=int, default=0, help="Offset within filtered pending prompts")
    p.add_argument("--batch-id", type=int, default=None, help="Explicit batch id")
    p.add_argument("--include-completed", action="store_true", help="Regenerate already completed prompts")
    p.add_argument("--max-examples-per-group", type=int, default=None, help="Truncate prompt examples per group")
    p.add_argument("--force", action="store_true", help="Overwrite an existing batch input id")
    p.set_defaults(func=cmd_task_a_prepare)
    sa.add_parser("merge", help="Merge batch outputs").set_defaults(func=cmd_task_a_merge)
    sa.add_parser("validate", help="Validate results").set_defaults(func=cmd_task_a_validate)

    # task-b
    sp_b = sub.add_parser("task-b", help="Task B: Input scoring")
    sb = sp_b.add_subparsers(dest="subcommand")
    p = sb.add_parser("prepare", help="Prepare next batch")
    p.add_argument("--batch-size", type=int, default=10, help="Latents per batch (default: 10)")
    p.add_argument("--limit", type=int, default=None, help="Override batch-size after filters")
    p.add_argument("--label", nargs="*", default=None, help="Optional label filter, e.g. QUO QUC or QUO,QUC")
    p.add_argument("--rank-min", type=int, default=None)
    p.add_argument("--rank-max", type=int, default=None)
    p.add_argument("--start-index", type=int, default=0, help="Offset within filtered pending latents")
    p.add_argument("--batch-id", type=int, default=None, help="Explicit batch id")
    p.add_argument("--include-completed", action="store_true", help="Regenerate already completed scoring tasks")
    p.add_argument("--max-examples-per-task", type=int, default=None, help="Truncate examples per scoring task")
    p.add_argument("--force", action="store_true", help="Overwrite an existing batch input id")
    p.set_defaults(func=cmd_task_b_prepare)
    sb.add_parser("merge", help="Merge batch outputs").set_defaults(func=cmd_task_b_merge)
    sb.add_parser("metrics", help="Compute metrics").set_defaults(func=cmd_task_b_metrics)

    # task-c
    sp_c = sub.add_parser("task-c", help="Task C: Manual review template")
    sc = sp_c.add_subparsers(dest="subcommand")
    sc.add_parser("run", help="Generate template").set_defaults(func=cmd_task_c_run)

    # task-d
    sp_d = sub.add_parser("task-d", help="Task D: Final feature cards")
    sd = sp_d.add_subparsers(dest="subcommand")
    sd.add_parser("run", help="Generate final cards").set_defaults(func=cmd_task_d_run)

    args = parser.parse_args()
    configure_paths(args.base_dir)

    if not args.command:
        parser.print_help()
        return 1

    if hasattr(args, "func"):
        return args.func(args)

    # subcommand not given
    lookup = {"task-a": sp_a, "task-b": sp_b, "task-c": sp_c, "task-d": sp_d}
    if args.command in lookup:
        lookup[args.command].print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
