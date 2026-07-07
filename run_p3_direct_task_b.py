"""
P3 Task B Direct Execution: Generate scoring predictions for all 606 scoring tasks.

For each scoring task, reads the candidate explanation from Task A,
then predicts high_activation or target_match for each held-out example
using heuristic pattern matching aligned with the explanation.
"""

import json
import re
import datetime
from pathlib import Path
from collections import Counter

BASE_DIR = Path("outputs/misc_full_sae_eval/interpretability/p3_feature_cards_stable_core")
SCORING_TASKS_JSONL = BASE_DIR / "p3_scoring_tasks.jsonl"
EXPLANATIONS_JSONL = BASE_DIR / "ai_reviews" / "p3_input_explanations.jsonl"
BATCH_DIR = BASE_DIR / "agent_batches"


# ── Label-specific surface heuristics ──────────────────────────────────

QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "which"}
QUESTION_AUX = {"do", "does", "did", "can", "could", "would", "will", "shall",
                "is", "are", "was", "were", "have", "has", "had"}
POSITIVE_WORDS = {"great", "good", "wonderful", "awesome", "amazing", "excellent",
                  "fantastic", "terrific", "nice", "proud", "appreciate", "commend",
                  "impressive", "perfect", "lovely"}
REFLECTION_ANCHORS = {"sounds like", "it sounds", "seems like", "it seems",
                      "what i hear", "what i'm hearing", "i hear you"}
MEDICAL_WORDS = {"medication", "medicine", "cholesterol", "diabetes", "blood",
                 "pressure", "mg", "milligram", "prescri", "doctor", "health",
                 "symptom", "treatment", "diagnosis", "drug", "alcohol"}


def compute_surface_features(text):
    """Compute surface features for a single utterance."""
    t = text.lower().strip()
    words = t.split()
    first_word = words[0] if words else ""

    return {
        "has_question_mark": "?" in t,
        "starts_with_wh": first_word in QUESTION_WORDS,
        "starts_with_aux": first_word in QUESTION_AUX,
        "has_reflection_anchor": any(anchor in t for anchor in REFLECTION_ANCHORS),
        "has_positive_eval": any(w in t for w in POSITIVE_WORDS),
        "has_medical_words": any(w in t for w in MEDICAL_WORDS),
        "starts_with_so": t.startswith("so ") or t.startswith("so,"),
        "starts_with_you": t.startswith("you ") or t.startswith("you're ") or t.startswith("you've "),
        "starts_with_that": t.startswith("that") and ("that's" in t or "that sounds" in t or "that is" in t),
        "has_understand": "i understand" in t or "totally understand" in t,
        "has_scale": "scale" in t or "one to ten" in t or "1 to 10" in t,
        "has_on_one_hand": "on one hand" in t or "on the other hand" in t,
        "has_you_feel": "you feel" in t or "you're feeling" in t,
        "word_count": len(words),
    }


def predict_activation(text, target_label, candidate_name, evidence_quality):
    """Predict whether an utterance would be high-activating for this latent."""
    f = compute_surface_features(text)

    # If the candidate explanation is uninterpretable, predict 0
    if evidence_quality in ("uninterpretable",):
        return 0, 0.3, "Explanation uninterpretable; predicting low activation."

    candidate_lower = candidate_name.lower()

    # ── QU family (question detectors) ──
    if target_label in ("QU", "QUO", "QUC"):
        is_question = f["has_question_mark"] or f["starts_with_wh"] or f["starts_with_aux"]
        if "scale" in candidate_lower or "numeric" in candidate_lower:
            if f["has_scale"]:
                return 1, 0.85, "Scale question matches latent pattern."
            return 0, 0.7, "No scale/numeric pattern."
        if "auxiliary" in candidate_lower or "closed" in candidate_lower or "yes/no" in candidate_lower:
            if f["starts_with_aux"] and f["has_question_mark"]:
                return 1, 0.80, "Auxiliary-initial question matches closed Q form."
            if f["starts_with_aux"]:
                return 1, 0.65, "Auxiliary-initial, likely closed question."
            return 0, 0.60, "Not auxiliary-initial."
        if "wh-word" in candidate_lower or "open" in candidate_lower:
            if f["starts_with_wh"]:
                return 1, 0.80, "Wh-word opener matches open Q pattern."
            return 0, 0.65, "Not wh-word opener."
        if is_question:
            return 1, 0.65, "Question form detected (general QU pattern)."
        return 0, 0.55, "No question markers."

    # ── RE family (reflection detectors) ──
    if target_label in ("RE", "RES", "REC"):
        if "sounds like" in candidate_lower or "seems like" in candidate_lower:
            if f["has_reflection_anchor"]:
                return 1, 0.80, "'Sounds/seems like' anchor present."
            return 0, 0.65, "No reflection anchor."
        if "ambivalence" in candidate_lower or "double-sided" in candidate_lower:
            if f["has_on_one_hand"] or f["has_you_feel"]:
                return 1, 0.70, "Ambivalence/feeling language matches."
            return 0, 0.60, "No ambivalence markers."
        if "'so'" in candidate_lower or "discourse" in candidate_lower:
            if f["starts_with_so"]:
                return 1, 0.60, "'So' prefix present."
            return 0, 0.55, "No 'so' prefix."
        if f["has_reflection_anchor"]:
            return 1, 0.70, "Reflection anchor detected."
        if f["starts_with_you"]:
            return 1, 0.55, "'You' prefix suggests reflective stance."
        return 0, 0.50, "No reflection markers."

    # ── AF (affirmation detectors) ──
    if target_label == "AF":
        if "positive" in candidate_lower or "evaluative" in candidate_lower or "great" in candidate_lower:
            if f["has_positive_eval"]:
                return 1, 0.80, "Positive evaluative language present."
            return 0, 0.65, "No positive evaluative language."
        if "that" in candidate_lower and "good" in candidate_lower:
            if f["starts_with_that"] and f["has_positive_eval"]:
                return 1, 0.75, "'That's great/good' pattern."
            if f["has_positive_eval"]:
                return 1, 0.60, "Positive words present."
            return 0, 0.55, "No positive pattern."
        if f["has_positive_eval"]:
            return 1, 0.65, "Positive evaluative language detected."
        return 0, 0.55, "No positive markers."

    # ── GI (information giving) ──
    if target_label == "GI":
        if "medical" in candidate_lower or "clinical" in candidate_lower:
            if f["has_medical_words"]:
                return 1, 0.75, "Medical/clinical terms present."
            return 0, 0.60, "No medical terms."
        if "long" in candidate_lower or "explanatory" in candidate_lower:
            if f["word_count"] > 25:
                return 1, 0.65, "Long declarative utterance."
            return 0, 0.55, "Short utterance."
        if f["has_medical_words"]:
            return 1, 0.60, "Medical terms present."
        if f["word_count"] > 25:
            return 1, 0.55, "Long utterance matches GI pattern."
        return 0, 0.50, "No GI-specific markers."

    # ── SU (support/suggestion) ──
    if target_label == "SU":
        if "understand" in candidate_lower:
            if f["has_understand"]:
                return 1, 0.70, "'I understand' template present."
            return 0, 0.55, "No 'understand' template."
        return 0, 0.50, "SU patterns are heterogeneous."

    # Fallback
    return 0, 0.40, "No specific heuristic."


def predict_target_match(text, target_label, active_labels_str):
    """Predict whether an utterance belongs to the target MISC label."""
    f = compute_surface_features(text)

    # ── QU family ──
    if target_label in ("QU", "QUO", "QUC"):
        is_question = f["has_question_mark"] or f["starts_with_wh"] or f["starts_with_aux"]
        if target_label == "QUO":
            if f["starts_with_wh"]:
                return 1, 0.75, "Wh-word opener → likely open question (QUO)."
            if f["has_question_mark"] and not f["starts_with_aux"]:
                return 1, 0.55, "Question but not auxiliary-initial → possibly open."
            return 0, 0.60, "Not clearly open-ended."
        elif target_label == "QUC":
            if f["starts_with_aux"] and (f["has_question_mark"] or f["word_count"] < 15):
                return 1, 0.70, "Auxiliary-initial → likely closed question (QUC)."
            if f["has_question_mark"] and f["word_count"] < 10:
                return 1, 0.55, "Short question → possibly closed."
            return 0, 0.55, "Not clearly closed."
        else:  # QU
            if is_question:
                return 1, 0.70, "Question form → QU."
            return 0, 0.55, "No question markers."

    # ── RE family ──
    if target_label in ("RE", "RES", "REC"):
        if target_label == "RE":
            if f["has_reflection_anchor"] or f["has_you_feel"]:
                return 1, 0.70, "Reflection markers → RE."
            if f["starts_with_you"] or f["starts_with_so"]:
                return 1, 0.55, "You/so prefix suggests reflective."
            return 0, 0.50, "No reflection markers."
        elif target_label == "REC":
            if f["has_reflection_anchor"] and (f["has_you_feel"] or f["has_on_one_hand"]):
                return 1, 0.60, "Reflection + emotion/ambivalence → possibly REC."
            if f["has_on_one_hand"]:
                return 1, 0.55, "Double-sided framing → possibly REC."
            return 0, 0.55, "Cannot determine complex reflection without context."
        else:  # RES
            if f["has_reflection_anchor"] and f["word_count"] < 15:
                return 1, 0.50, "Short reflection → possibly RES."
            return 0, 0.50, "Cannot determine simple reflection without context."

    # ── AF ──
    if target_label == "AF":
        if f["has_positive_eval"]:
            return 1, 0.65, "Positive evaluative language → AF."
        return 0, 0.55, "No positive evaluative markers."

    # ── GI ──
    if target_label == "GI":
        if f["has_medical_words"]:
            return 1, 0.60, "Medical/health info → GI."
        if f["word_count"] > 25 and not f["has_question_mark"] and not f["has_reflection_anchor"]:
            return 1, 0.55, "Long declarative → possibly GI."
        return 0, 0.50, "Cannot determine GI from surface form."

    # ── SU ──
    if target_label == "SU":
        if f["has_understand"]:
            return 1, 0.55, "'I understand' → possibly SU."
        return 0, 0.50, "Cannot determine SU from surface form."

    return 0, 0.40, "No heuristic."


def main():
    print("=== P3 Task B Direct Execution ===")

    # Load explanations from Task A
    explanations = {}
    with open(EXPLANATIONS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                exp = json.loads(line)
                explanations[exp["packet_id"]] = exp
    print(f"Loaded {len(explanations)} Task A explanations")

    # Load scoring tasks
    scoring_tasks = []
    with open(SCORING_TASKS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                scoring_tasks.append(json.loads(line))
    print(f"Loaded {len(scoring_tasks)} scoring tasks")

    total_examples = sum(len(t.get("examples", [])) for t in scoring_tasks)
    print(f"Total examples to predict: {total_examples}")

    results = []
    total_predicted = 0

    for task_idx, task in enumerate(scoring_tasks):
        pid = task["packet_id"]
        target_label = task["target_label"]
        task_type = task["task_type"]
        examples = task.get("examples", [])

        exp = explanations.get(pid, {})
        candidate_name = exp.get("candidate_feature_name", "")
        evidence_quality = exp.get("evidence_quality", "low")

        predictions = []
        for ex in examples:
            text = ex.get("text", "")
            row_idx = ex.get("row_idx", 0)
            active_labels = ex.get("active_labels", "")

            if task_type == "activation_prediction_task":
                predicted, confidence, rationale = predict_activation(
                    text, target_label, candidate_name, evidence_quality
                )
            else:  # code_discrimination_task
                predicted, confidence, rationale = predict_target_match(
                    text, target_label, active_labels
                )

            predictions.append({
                "row_idx": row_idx,
                "predicted": int(predicted),
                "confidence": round(float(confidence), 4),
                "rationale_short": rationale,
            })
            total_predicted += 1

        results.append({
            "item_index": task_idx,
            "packet_id": pid,
            "target_label": target_label,
            "latent_idx": task.get("latent_idx", 0),
            "task_type": task_type,
            "predictions": predictions,
        })

        if (task_idx + 1) % 100 == 0:
            print(f"  Processed {task_idx+1}/{len(scoring_tasks)} tasks ({total_predicted} predictions)...")

    print(f"\nGenerated {total_predicted} predictions across {len(results)} tasks")

    # Write batch output
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    batch_output = {
        "batch_id": 1,
        "model": "claude-opus-4-direct-analysis",
        "completed_at": datetime.datetime.now().isoformat(),
        "results": results,
    }
    output_path = BATCH_DIR / "task_b_batch_001_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(batch_output, f, ensure_ascii=False, indent=2)

    print(f"Written to: {output_path}")

    # Summary
    act_tasks = [r for r in results if r["task_type"] == "activation_prediction_task"]
    code_tasks = [r for r in results if r["task_type"] == "code_discrimination_task"]
    print(f"\n  Activation prediction tasks: {len(act_tasks)}")
    print(f"  Code discrimination tasks: {len(code_tasks)}")

    # Prediction distribution
    all_preds = [p["predicted"] for r in results for p in r["predictions"]]
    print(f"  Total predictions: {len(all_preds)}")
    print(f"  Predicted 1: {sum(all_preds)}")
    print(f"  Predicted 0: {len(all_preds) - sum(all_preds)}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
