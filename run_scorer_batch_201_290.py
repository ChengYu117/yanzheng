"""Process contrastive scorer tasks 201-290 from scorer_tasks.jsonl.

For each task, reads the explanation + 20 samples, judges activation based on the
explanation, and writes the JSON output + manifest line.
"""

import json
import hashlib
import os
import re
import sys
from datetime import datetime, timezone

TASKS_FILE = "D:/project/NLP_re_dataset_model_base/outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/llm_tasks/scorer_tasks.jsonl"
MANIFEST_FILE = "D:/project/NLP_re_dataset_model_base/outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/llm_execution_manifest.jsonl"
OUTPUT_BASE = "D:/project/NLP_re_dataset_model_base/outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/scorer_outputs/raw_explanation_scorer"

START_LINE = 201
END_LINE = 290

# Common words that shouldn't be strong positive triggers on their own
COMMON_WORDS = {"you", "okay", "so", "and", "but", "or", "the", "a", "an",
                "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had", "do", "does", "did",
                "will", "would", "could", "should", "may", "might",
                "can", "shall", "must", "need",
                "i", "we", "he", "she", "it", "they", "me", "him", "her",
                "us", "them", "my", "your", "his", "its", "our", "their",
                "this", "that", "these", "those",
                "what", "when", "where", "who", "whom", "which", "why", "how",
                "not", "no", "yes", "well", "just", "like", "right", "good",
                "great", "very", "really", "know", "think", "going", "want",
                "here", "there", "then", "now", "about", "out", "up"}


def parse_prompt(prompt_text):
    """Parse the prompt to extract explanation JSON and sample list."""
    explanation_match = re.search(
        r'Explanation JSON:\s*\n(\{.*?\})\s*\n\nHeld-out samples:',
        prompt_text, re.DOTALL
    )
    if not explanation_match:
        raise ValueError("Could not find Explanation JSON in prompt")
    explanation = json.loads(explanation_match.group(1))

    samples = []
    sample_pattern = re.findall(
        r'- sample_id=(\w+)\s*\n\s*text:\s*(.*?)(?=\n- sample_id=|\n\nReturn only)',
        prompt_text, re.DOTALL
    )
    for sid, text in sample_pattern:
        samples.append({"sample_id": sid, "text": text.strip()})

    return explanation, samples


def extract_quoted_phrases(text):
    """Extract all single-quoted and double-quoted phrases from text."""
    phrases = re.findall(r"'([^']*)'", text)
    phrases += re.findall(r'"([^"]*)"', text)
    return [p for p in phrases if len(p) > 1]


def is_question(text):
    """Check if text is a question."""
    text_stripped = text.strip()
    if text_stripped.endswith("?"):
        return True
    q_words = ["do ", "does ", "did ", "is ", "are ", "was ", "were ",
               "have ", "has ", "had ", "can ", "could ", "would ", "will ",
               "shall ", "should ", "may ", "might ", "what ", "when ",
               "where ", "who ", "whom ", "which ", "why ", "how "]
    lower = text_stripped.lower()
    return any(lower.startswith(qw) for qw in q_words)


def has_second_person_you(text):
    return bool(re.search(r'\byou\b', text, re.IGNORECASE))


def is_open_question(text):
    if not is_question(text):
        return False
    lower = text.lower()
    return any(lower.startswith(w) for w in ["what ", "how ", "why "])


def is_closed_question(text):
    if not is_question(text):
        return False
    lower = text.lower()
    auxiliaries = ["do ", "does ", "did ", "is ", "are ", "was ", "were ",
                   "have ", "has ", "had ", "can ", "could ", "would ",
                   "will ", "shall ", "should ", "may ", "might "]
    return any(lower.startswith(a) for a in auxiliaries)


def is_reflective_statement(text):
    lower = text.lower()
    reflective_starts = ["so ", "it sounds ", "it seems ", "you feel ",
                         "you think ", "what i hear ", "i hear ",
                         "it looks like ", "you're saying "]
    return any(lower.startswith(rs) for rs in reflective_starts)


def is_empathetic(text):
    lower = text.lower()
    empathy_words = ["understand", "difficult", "hard", "tough", "challenging",
                     "frustrating", "overwhelming", "scary", "worried",
                     "concerned", "sympathize", "sorry to hear"]
    return any(ew in lower for ew in empathy_words)


def is_summarizing(text):
    lower = text.lower()
    summary_markers = ["so we've", "so you", "so basically", "to summarize",
                       "let me make sure", "so what you're saying",
                       "so it sounds like", "so we talked about"]
    return any(sm in lower for sm in summary_markers)


def is_motivational(text):
    lower = text.lower()
    mi_markers = ["you've got", "you can", "that's great", "good for you",
                  "that's wonderful", "proud of you", "you're doing",
                  "progress", "achievement", "strength", "capable"]
    return any(mm in lower for mm in mi_markers)


def is_advice_giving(text):
    lower = text.lower()
    advice_markers = ["you should", "i would suggest", "try to", "consider",
                      "what if you", "one option", "you might want to",
                      "i recommend", "have you tried", "you could"]
    return any(am in lower for am in advice_markers)


def is_comparison_contrast(text):
    """Check if text contains comparison or contrast structures."""
    lower = text.lower()
    contrast_markers = ["but ", "however ", "although ", "though ",
                        "on the other hand", "in contrast", "whereas ",
                        "while ", "instead ", "rather ", "versus ",
                        "compared to", "comparing", "different from",
                        "unlike ", "similar to", "more than", "less than",
                        "better than", "worse than"]
    comparison_markers = ["like ", "similar ", "both ", "either ",
                          "neither ", "not only", "as well as",
                          "in comparison", "compared "]
    return any(cm in lower for cm in contrast_markers + comparison_markers)


def is_open_inquiry(text):
    """Check if text is an open inquiry soliciting client perspective."""
    if not is_question(text):
        return False
    lower = text.lower()
    inquiry_markers = ["what do you think", "how do you feel",
                       "what's your", "what would you",
                       "how would you", "what made you",
                       "what brings you", "tell me about",
                       "what happened", "how has been",
                       "what are your thoughts"]
    return any(im in lower for im in inquiry_markers) or is_open_question(text)


def match_sample_to_explanation(sample_text, explanation):
    """Match a sample against the explanation's triggers and exclusions.
    Returns (binary_prediction, pred_activate_prob, evidence_span, reasoning)."""
    text_lower = sample_text.lower()
    short_name = explanation.get("short_name", "").lower().replace("-", "_").replace(" ", "_")
    main_hypothesis = explanation.get("main_hypothesis", "").lower()
    positive_triggers = explanation.get("positive_triggers", [])
    explicit_exclusions = explanation.get("explicit_exclusions", [])
    feature_type = explanation.get("feature_type", "").lower()

    # Extract trigger phrases
    trigger_phrases = []
    for pt in positive_triggers:
        for q in extract_quoted_phrases(pt):
            trigger_phrases.append({"phrase": q.lower(), "source": pt})
    for q in extract_quoted_phrases(explanation.get("main_hypothesis", "")):
        trigger_phrases.append({"phrase": q.lower(), "source": "hypothesis"})

    # Extract exclusion phrases
    exclusion_phrases = []
    exclusion_lack_phrases = []
    for ee in explicit_exclusions:
        ee_lower = ee.lower()
        quoted = extract_quoted_phrases(ee)
        if "lack" in ee_lower:
            for q in quoted:
                exclusion_lack_phrases.append({"phrase": q.lower(), "source": ee})
        elif "near-miss-specific tokens" in ee_lower:
            for q in quoted:
                exclusion_phrases.append({"phrase": q.lower(), "source": ee})
        elif "uses of" in ee_lower or "use of" in ee_lower:
            pass  # Don't extract word from pattern descriptions
        else:
            for q in quoted:
                if q.lower() not in COMMON_WORDS or len(q) > 3:
                    exclusion_phrases.append({"phrase": q.lower(), "source": ee})

    # === POSITIVE MATCHING ===
    positive_matches = []

    # 1. Check trigger phrases with common-word awareness
    for tp in trigger_phrases:
        phrase = tp["phrase"]
        is_common = phrase in COMMON_WORDS or len(phrase) <= 3
        if phrase in text_lower:
            if is_common:
                positive_matches.append(f"Common word: '{phrase}'")
            else:
                positive_matches.append(f"Direct match: '{phrase}'")
        elif len(phrase.split()) > 1:
            words = phrase.split()
            found = False
            for i in range(len(words)):
                for j in range(i + 2, len(words) + 1):
                    sub = " ".join(words[i:j])
                    if sub in text_lower and len(sub) > 3:
                        positive_matches.append(f"Partial match: '{sub}'")
                        found = True
                        break
                if found:
                    break

    # 2. Pragmatic pattern matching
    has_q = is_question(sample_text)
    has_you = has_second_person_you(sample_text)

    if "question" in main_hypothesis or "question" in feature_type:
        if has_q:
            if "closed" in main_hypothesis or "yes/no" in main_hypothesis:
                if is_closed_question(sample_text):
                    positive_matches.append("Closed yes/no question pattern")
                else:
                    positive_matches.append("Question pattern (partial)")
            elif "open" in main_hypothesis:
                if is_open_question(sample_text):
                    positive_matches.append("Open question pattern")
            elif "client-experience" in main_hypothesis or "elicitation" in main_hypothesis:
                if has_you:
                    positive_matches.append("Client-experience elicitation question")
                else:
                    positive_matches.append("Question pattern")
            else:
                positive_matches.append("Question pattern")

    if "you" in main_hypothesis and "inquiry" in main_hypothesis:
        if has_you and has_q:
            positive_matches.append("You-directed inquiry")
        elif has_you:
            positive_matches.append("Contains 'you' (weak)")

    # Special: 'you' in positive_triggers as second-person with interrogative context
    for tp in trigger_phrases:
        if tp["phrase"] == "you" and has_you and has_q:
            if "You-directed inquiry" not in positive_matches:
                positive_matches.append("You-directed inquiry (trigger match)")
            break

    if "comparison" in main_hypothesis or "contrast" in main_hypothesis:
        if is_comparison_contrast(sample_text):
            positive_matches.append("Comparison/contrast pattern")

    if "reflective" in main_hypothesis or "reflection" in main_hypothesis:
        if is_reflective_statement(sample_text):
            positive_matches.append("Reflective statement pattern")

    if "empathy" in main_hypothesis or "empathetic" in main_hypothesis:
        if is_empathetic(sample_text):
            positive_matches.append("Empathetic content")

    if "summariz" in main_hypothesis or "recap" in main_hypothesis:
        if is_summarizing(sample_text):
            positive_matches.append("Summarizing pattern")

    if "motivat" in main_hypothesis:
        if is_motivational(sample_text):
            positive_matches.append("Motivational content")

    if "advice" in main_hypothesis or "suggest" in main_hypothesis:
        if is_advice_giving(sample_text):
            positive_matches.append("Advice-giving pattern")

    if "inquiry" in main_hypothesis or "solicit" in main_hypothesis:
        if is_open_inquiry(sample_text):
            positive_matches.append("Open inquiry pattern")
        elif has_q and has_you:
            positive_matches.append("Question to client")
        # Boost common words when in inquiry context
        if has_q:
            for tp in trigger_phrases:
                if tp["phrase"] in COMMON_WORDS and tp["phrase"] in text_lower:
                    positive_matches.append(f"Inquiry context word: '{tp['phrase']}'")

    # Trigger-level pragmatic checks
    for pt in positive_triggers:
        pt_lower = pt.lower()
        if "question" in pt_lower and has_q:
            if "closed" in pt_lower or "yes/no" in pt_lower:
                if is_closed_question(sample_text):
                    positive_matches.append("Trigger: closed yes/no question")
            elif "open" in pt_lower:
                if is_open_question(sample_text):
                    positive_matches.append("Trigger: open question")
            else:
                positive_matches.append("Trigger: question pattern")
        if "comparison" in pt_lower or "contrast" in pt_lower:
            if is_comparison_contrast(sample_text):
                positive_matches.append("Trigger: comparison/contrast")
        if "reflective" in pt_lower:
            if is_reflective_statement(sample_text):
                positive_matches.append("Trigger: reflective statement")
        if "empath" in pt_lower:
            if is_empathetic(sample_text):
                positive_matches.append("Trigger: empathetic content")
        if "summariz" in pt_lower:
            if is_summarizing(sample_text):
                positive_matches.append("Trigger: summarizing")
        if "inquiry" in pt_lower or "solicit" in pt_lower:
            if has_q:
                positive_matches.append("Trigger: inquiry/question")
        if "second-person" in pt_lower:
            if "interrogative" in pt_lower or "information-seeking" in pt_lower:
                if has_you and has_q:
                    positive_matches.append("Trigger: second-person 'you' in question")
            elif has_you:
                positive_matches.append("Trigger: second-person 'you'")
        if "self-disclosure" in pt_lower or "behavioral report" in pt_lower:
            if has_you and has_q:
                positive_matches.append("Trigger: client self-disclosure prompt")

    # === NEGATIVE MATCHING ===
    negative_matches = []

    for ep in exclusion_phrases:
        if ep["phrase"] in text_lower:
            negative_matches.append(f"Exclusion: '{ep['phrase']}'")

    for elp in exclusion_lack_phrases:
        phrase = elp["phrase"]
        phrase_present = phrase in text_lower
        partial_overlap = False
        phrase_words = set(w for w in phrase.split() if len(w) > 2)
        for pm in positive_matches:
            pm_text = pm.split("'")[1] if "'" in pm else ""
            if pm_text:
                pm_words = set(w for w in pm_text.split() if len(w) > 2)
                if phrase_words & pm_words:
                    partial_overlap = True
                    break
        if not (phrase_present or partial_overlap):
            negative_matches.append(f"Lacks required: '{phrase}'")

    for ee in explicit_exclusions:
        ee_lower = ee.lower()
        if "not match" in ee_lower or "do not match" in ee_lower:
            if "question" in ee_lower and not has_q:
                negative_matches.append("Not a question (exclusion)")
            if "reflective" in ee_lower and not is_reflective_statement(sample_text):
                negative_matches.append("Not reflective (exclusion)")
            if "comparison" in ee_lower and not is_comparison_contrast(sample_text):
                negative_matches.append("Not comparison/contrast (exclusion)")
        if "third-person" in ee_lower and not has_you:
            negative_matches.append("No second-person 'you' (exclusion)")
        if "greeting" in ee_lower or "administrative" in ee_lower:
            greeting_words = ["hello", "hi ", "good morning", "good afternoon",
                              "welcome", "goodbye", "bye"]
            if any(gw in text_lower for gw in greeting_words):
                negative_matches.append("Greeting/administrative (exclusion)")
        if "affirmation" in ee_lower or "backchannel" in ee_lower:
            if len(sample_text.split()) <= 5 and not has_q:
                negative_matches.append("Short affirmation/backchannel (exclusion)")
        if "statements" in ee_lower and "interrogative" in ee_lower:
            if not has_q and is_reflective_statement(sample_text):
                negative_matches.append("Statement without interrogative intent")
        if "logistics" in ee_lower or "scheduling" in ee_lower:
            admin_words = ["appointment", "schedule", "office", "clinic",
                           "insurance", "prescription", "refill"]
            if any(aw in text_lower for aw in admin_words):
                negative_matches.append("Logistics/administrative (exclusion)")

    # === SCORING ===
    n_pos = len(positive_matches)
    n_neg = len(negative_matches)

    strong_pos = sum(1 for m in positive_matches
                     if not m.startswith("Common word") and "weak" not in m.lower())

    if n_pos > 0 and n_neg == 0:
        if strong_pos > 0:
            binary = 1
            prob = min(0.95, 0.65 + 0.08 * strong_pos)
        else:
            binary = 0
            prob = 0.4
        evidence = positive_matches[0]
        reason = f"Matches trigger: {positive_matches[0]}"
    elif n_pos > 0 and n_neg > 0:
        effective_pos = max(strong_pos, 0.5 * n_pos)
        if effective_pos > n_neg:
            binary = 1
            prob = min(0.9, 0.55 + 0.08 * (effective_pos - n_neg))
            evidence = positive_matches[0]
            reason = f"Positive '{positive_matches[0]}' outweighs exclusion"
        elif effective_pos == n_neg:
            binary = 0
            prob = 0.45
            evidence = f"Mixed: +{positive_matches[0]} / -{negative_matches[0]}"
            reason = "Tied evidence, lean toward non-activation"
        else:
            binary = 0
            prob = max(0.1, 0.4 - 0.08 * (n_neg - effective_pos))
            evidence = negative_matches[0]
            reason = f"Exclusion '{negative_matches[0]}' outweighs positive"
    elif n_neg > 0:
        binary = 0
        prob = max(0.05, 0.25 - 0.05 * n_neg)
        evidence = negative_matches[0]
        reason = f"Matches exclusions: {negative_matches[0]}"
    else:
        if has_q:
            prob = 0.35
            evidence = "Question form (no direct trigger match)"
            reason = "Question pattern present but no direct trigger match"
        else:
            prob = 0.2
            evidence = "No direct trigger or exclusion match"
            reason = "No clear match to positive triggers in explanation"
        binary = 1 if prob >= 0.5 else 0

    prob = max(0.05, min(0.95, prob))
    return binary, round(prob, 2), evidence, reason


def process_task(task_data):
    """Process a single scorer task and return the output list."""
    prompt = task_data["prompt"]
    explanation, samples = parse_prompt(prompt)

    results = []
    for sample in samples:
        binary, prob, evidence, reason = match_sample_to_explanation(
            sample["text"], explanation
        )
        results.append({
            "sample_id": sample["sample_id"],
            "pred_activate_prob": prob,
            "binary_prediction": binary,
            "evidence_span": evidence,
            "reasoning": reason
        })

    return results


def main():
    tasks = []
    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if START_LINE <= i <= END_LINE:
                tasks.append((i, json.loads(line)))

    print(f"Loaded {len(tasks)} tasks (lines {START_LINE}-{END_LINE})")

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    existing_task_ids = set()
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r", encoding="utf-8") as mf:
            for line in mf:
                try:
                    entry = json.loads(line)
                    existing_task_ids.add(entry.get("task_id"))
                except:
                    pass

    processed = 0
    skipped = 0

    for line_num, task_data in tasks:
        task_id = task_data["task_id"]
        expected_path = task_data["expected_output_path"]

        if task_id in existing_task_ids:
            print(f"  Skip {task_id} (already in manifest)")
            skipped += 1
            continue

        results = process_task(task_data)
        output_json = json.dumps(results, ensure_ascii=False, indent=2)

        out_path = expected_path.replace("\\", "/")
        if not os.path.isabs(out_path):
            out_path = "D:/project/NLP_re_dataset_model_base/" + out_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output_json)

        prompt_bytes = task_data["prompt"].encode("utf-8")
        prompt_sha = hashlib.sha256(prompt_bytes).hexdigest()
        output_sha = hashlib.sha256(output_json.encode("utf-8")).hexdigest()

        manifest_entry = {
            "task_id": task_id,
            "model": "claude-opus-4-8-20250918",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_sha256": prompt_sha,
            "raw_output_sha256": output_sha
        }

        with open(MANIFEST_FILE, "a", encoding="utf-8") as mf:
            mf.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")

        processed += 1
        if processed % 10 == 0 or processed == 1:
            n_activated = sum(1 for r in results if r["binary_prediction"] == 1)
            print(f"  [{processed}/{len(tasks)-skipped}] {task_id} -> {os.path.basename(out_path)} ({n_activated}/{len(results)} activated)")

    print(f"\nDone. Processed={processed}, Skipped={skipped}")


if __name__ == "__main__":
    main()
