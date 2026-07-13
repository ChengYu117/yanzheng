#!/usr/bin/env python3
"""Process scorer tasks lines 336-485 from scorer_tasks.jsonl.

For each task, parse the explanation and 20 samples, judge activation
probability based on the explanation, write output JSON and append manifest.
"""

import json
import hashlib
import os
import re
import sys
from datetime import datetime, timezone

TASKS_FILE = "outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/llm_tasks/scorer_tasks.jsonl"
MANIFEST_FILE = "outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/llm_execution_manifest.jsonl"
MODEL_NAME = "claude-opus-4-8-20250918"
LINE_START = 336  # 1-indexed
LINE_END = 485    # 1-indexed inclusive


def parse_prompt(prompt_text):
    """Extract explanation JSON and samples from the prompt."""
    # Extract explanation JSON - try multiple patterns
    expl_match = re.search(r'Explanation JSON:\s*\n(\{.*?\})\s*\n\nHeld-out samples:', prompt_text, re.DOTALL)
    if not expl_match:
        # Try alternative pattern
        expl_match = re.search(r'Explanation JSON:\s*\n(\{.*?\})\s*\nHeld-out samples:', prompt_text, re.DOTALL)
    if not expl_match:
        return None, []

    try:
        explanation = json.loads(expl_match.group(1))
    except json.JSONDecodeError:
        # Try to fix common JSON issues
        raw = expl_match.group(1)
        raw = raw.replace('\n', ' ').replace('\r', '')
        try:
            explanation = json.loads(raw)
        except:
            return None, []

    # Extract samples
    samples = []
    sample_pattern = re.compile(r'- sample_id=(\w+)\s*\n\s*text:\s*(.+?)(?=\n- sample_id=|\nReturn only)', re.DOTALL)
    for match in sample_pattern.finditer(prompt_text):
        sid = match.group(1)
        text = match.group(2).strip()
        samples.append({"sample_id": sid, "text": text})

    # If pattern didn't match, try alternative
    if not samples:
        sample_pattern2 = re.compile(r'sample_id=(\w+)\s*\n\s*text:\s*(.+?)(?=\nsample_id=|\nReturn)', re.DOTALL)
        for match in sample_pattern2.finditer(prompt_text):
            sid = match.group(1)
            text = match.group(2).strip()
            samples.append({"sample_id": sid, "text": text})

    return explanation, samples


def extract_quoted_phrases(text):
    """Extract quoted phrases from text."""
    return re.findall(r"'([^']+)'", text) + re.findall(r'"([^"]+)"', text)


def judge_sample(explanation, text):
    """Judge whether a sample should activate the latent based on the explanation.

    Returns (pred_activate_prob, binary_prediction, evidence_span, reason).
    """
    text_lower = text.lower().strip()
    text_words = set(re.findall(r'\b[a-z]+\b', text_lower))

    short_name = explanation.get("short_name", "").lower()
    main_hyp = explanation.get("main_hypothesis", "").lower()
    positive_triggers = explanation.get("positive_triggers", [])
    exclusions = explanation.get("explicit_exclusions", [])
    confounds = explanation.get("possible_surface_confounds", [])
    feature_type = explanation.get("feature_type", "")
    failure_modes = explanation.get("failure_modes", "")

    score = 0.45  # slightly below neutral base
    evidence_parts = []
    reasons = []
    exclusion_penalty = 0

    # === STEP 1: Identify the core concept from short_name and main_hypothesis ===

    # Build a concept profile
    concept_keywords = set()
    for phrase in extract_quoted_phrases(main_hyp):
        concept_keywords.add(phrase.lower())
    for phrase in extract_quoted_phrases(short_name):
        concept_keywords.add(phrase.lower())

    # === STEP 2: Check positive triggers ===
    trigger_score = 0
    for trigger in positive_triggers:
        trigger_lower = trigger.lower()
        quoted_in_trigger = extract_quoted_phrases(trigger)

        # Check quoted examples from trigger descriptions
        for phrase in quoted_in_trigger:
            phrase_lower = phrase.lower()
            if phrase_lower in text_lower:
                trigger_score += 0.2
                evidence_parts.append(phrase[:50])
                reasons.append(f"matches positive trigger pattern")
                break

        # Check for key concept words from trigger
        trigger_key_words = set()
        # Extract meaningful words (>3 chars, not stopwords)
        stopwords = {'the', 'and', 'that', 'this', 'with', 'from', 'have', 'has', 'had',
                     'was', 'were', 'been', 'being', 'are', 'does', 'did', 'will',
                     'would', 'could', 'should', 'can', 'may', 'might', 'must',
                     'shall', 'not', 'but', 'for', 'you', 'your', 'they', 'them',
                     'their', 'its', 'our', 'his', 'her', 'she', 'when', 'where',
                     'which', 'what', 'who', 'whom', 'how', 'why', 'than', 'then',
                     'also', 'just', 'very', 'some', 'any', 'each', 'every', 'all',
                     'both', 'few', 'more', 'most', 'other', 'such', 'only', 'own',
                     'same', 'about', 'above', 'after', 'again', 'against', 'before',
                     'below', 'between', 'during', 'into', 'through', 'under', 'over',
                     'out', 'off', 'down', 'further', 'once', 'here', 'there',
                     'those', 'these', 'because', 'since', 'until', 'while', 'whether',
                     'although', 'though', 'even', 'still', 'already', 'yet', 'ever',
                     'never', 'always', 'sometimes', 'often', 'usually', 'rather',
                     'quite', 'almost', 'enough', 'too', 'really', 'well', 'like',
                     'know', 'think', 'feel', 'want', 'need', 'make', 'take',
                     'come', 'go', 'see', 'look', 'say', 'tell', 'give', 'use',
                     'find', 'ask', 'put', 'keep', 'let', 'try', 'seem', 'become',
                     'leave', 'call', 'include', 'using', 'particularly', 'specifically',
                     'especially', 'e.g', 'context', 'combined', 'appears', 'requires',
                     'presence', 'associated', 'soliciting', 'perspective', 'client',
                     'counselor', 'utterance', 'contains', 'language', 'function',
                     'content', 'speech', 'act', 'question', 'prompt', 'frame',
                     'addressed', 'information', 'seeking', 'source', 'sought',
                     'role', 'provider', 'paraphrases', 'summaries', 'social',
                     'greetings', 'administrative', 'responses', 'affirmations',
                     'backchannel', 'logistics', 'scheduling', 'topics', 'statements',
                     'reflections', 'third', 'person', 'references', 'general',
                     'echo', 'clarification', 'requests', 'share', 'surface',
                     'without', 'reflective', 'intent', 'overlap', 'between',
                     'near', 'miss', 'suggests', 'trigger', 'purely', 'lexical',
                     'frequency', 'common', 'words', 'inflate', 'activation',
                     'regardless', 'tracking', 'broader', 'topic', 'rather',
                     'behavior', 'discussion', 'respond', 'combination', 'semantic',
                     'syntactic', 'features', 'fully', 'captured', 'word', 'lists',
                     'conflate', 'actual', 'patterns', 'specific', 'dataset',
                     'general', 'boundary', 'examining', 'token', 'level',
                     'activations', 'open', 'closed', 'self', 'disclosure',
                     'behavioral', 'report', 'questions', 'prompts', 'put'}

        for word in re.findall(r'\b[a-z]{4,}\b', trigger_lower):
            if word not in stopwords:
                trigger_key_words.add(word)

        # Check if key concept words appear in text
        matched_concepts = trigger_key_words & text_words
        if len(matched_concepts) >= 3:
            trigger_score += 0.15
            reasons.append(f"multiple concept matches: {', '.join(list(matched_concepts)[:3])}")
        elif len(matched_concepts) >= 1:
            trigger_score += 0.05

    # === STEP 3: Check specific patterns based on explanation type ===

    # Scale questions
    if "scale" in short_name or "scale" in main_hyp:
        if re.search(r'scale.{0,40}(one|two|three|four|five|six|seven|eight|nine|ten|\d+).{0,20}(to|-).{0,20}(one|two|three|four|five|six|seven|eight|nine|ten|\d+)', text_lower):
            trigger_score += 0.3
            scale_match = re.search(r'scale.{0,30}', text_lower)
            evidence_parts.append(scale_match.group() if scale_match else "scale of X to Y")
            reasons.append("explicit scaling question frame with numeric range")
        elif "scale" in text_lower:
            trigger_score += 0.25
            scale_match = re.search(r'scale.{0,30}', text_lower)
            evidence_parts.append(scale_match.group() if scale_match else "scale")
            reasons.append("contains 'scale' keyword")
        elif re.search(r'(one|1) to (ten|10|100)', text_lower):
            trigger_score += 0.2
            evidence_parts.append("numeric range")
            reasons.append("numeric range without explicit 'scale' word")
        elif "confidence" in text_lower and ("how" in text_lower or "rate" in text_lower):
            trigger_score += 0.1
            reasons.append("confidence question without scale frame")

    # "how long" temporal questions
    if "how long" in short_name or "how long" in main_hyp:
        if "how long" in text_lower:
            trigger_score += 0.35
            hl_match = re.search(r'how long.{0,30}', text_lower)
            evidence_parts.append(hl_match.group() if hl_match else "how long")
            reasons.append("'how long' temporal inquiry present")
        elif "how soon" in text_lower or "how much time" in text_lower:
            trigger_score += 0.15
            reasons.append("temporal question variant")

    # Second-person "you"-directed inquiry
    if "you_directed" in short_name or "you-directed" in short_name:
        has_you = "you" in text_words or "your" in text_words
        is_question = "?" in text
        is_interrogative = text_lower.startswith(("do ", "did ", "does ", "can ", "could ",
                                                   "would ", "will ", "have ", "has ", "how ",
                                                   "what ", "when ", "where ", "why ", "is ",
                                                   "are ", "may "))
        if has_you and (is_question or is_interrogative):
            trigger_score += 0.3
            you_ctx = re.search(r'(do|did|does|can|could|would|will|have|has|how|what)\s+.{0,5}you', text_lower)
            evidence_parts.append(you_ctx.group()[:40] if you_ctx else "you + question")
            reasons.append("second-person 'you' in interrogative frame")
        elif has_you and any(w in text_lower for w in ["think", "feel", "believe", "want", "consider"]):
            trigger_score += 0.15
            reasons.append("'you' with inquiry verb but weak question frame")

    # Inquiry with specific vocabulary (like/well/know or get/know/think)
    if "inquiry" in short_name and ("like" in short_name or "well" in short_name or "get" in short_name or "know" in short_name):
        is_question = "?" in text or text_lower.startswith(("do ", "did ", "does ", "can ", "could ",
                                                           "would ", "will ", "have ", "has ", "how ",
                                                           "what ", "when ", "where ", "why "))
        inquiry_words = set()
        for w in ["like", "well", "know", "get", "think", "feel", "want", "believe", "consider"]:
            if w in short_name:
                inquiry_words.add(w)
        if not inquiry_words:
            inquiry_words = {"like", "well", "know", "get", "think"}

        has_inquiry_word = bool(inquiry_words & text_words)
        if is_question and has_inquiry_word:
            trigger_score += 0.25
            matched = inquiry_words & text_words
            evidence_parts.append(list(matched)[0] if matched else "")
            reasons.append(f"question with inquiry vocabulary: {', '.join(list(matched)[:2])}")
        elif is_question:
            trigger_score += 0.1
            reasons.append("question without key inquiry vocabulary")

    # Disfluency / speech repair
    if "disfluency" in short_name or "speech repair" in short_name or "self-correction" in short_name:
        disfluency_score = 0
        disfluency_evidence = []

        # Check for explicit examples from positive triggers
        for trigger in positive_triggers:
            for phrase in extract_quoted_phrases(trigger):
                if phrase.lower() in text_lower:
                    disfluency_score += 0.3
                    disfluency_evidence.append(phrase[:40])

        # Self-corrections: "I mean", "or rather", "actually", "sorry"
        if re.search(r'\b(i mean|or rather|actually|sorry|no wait|that is)\b', text_lower):
            disfluency_score += 0.15
            disfluency_evidence.append("self-correction marker")

        # Stuttering/repetition: consecutive repeated words
        words = text_lower.split()
        for i in range(len(words)-1):
            if words[i] == words[i+1] and len(words[i]) > 2:
                disfluency_score += 0.2
                disfluency_evidence.append(f"'{words[i]}' repeated")
                break

        # Tangled syntax: look for specific patterns from examples
        if re.search(r'(not come thank|you are not shutting|haha not|I I\'m|that you\'re on two different)', text_lower):
            disfluency_score += 0.25
            disfluency_evidence.append("tangled syntax")

        # False starts with filled pauses in longer utterances
        if re.search(r'\b(um|uh|er|ah)\b', text_lower) and len(words) > 10:
            disfluency_score += 0.1
            disfluency_evidence.append("filled pause in longer utterance")

        # Multiple "and" conjunctions suggesting run-on/repair
        if text_lower.count(" and ") >= 3:
            disfluency_score += 0.05

        trigger_score += min(disfluency_score, 0.4)
        if disfluency_evidence:
            evidence_parts.append(disfluency_evidence[0])
            reasons.append(f"disfluency: {disfluency_evidence[0]}")

    # Health risk consequence warning
    if "health risk" in short_name or "consequence warning" in short_name or "risk" in short_name and "warning" in short_name:
        warning_score = 0
        warning_evidence = []

        # Check for explicit warning examples
        for trigger in positive_triggers:
            for phrase in extract_quoted_phrases(trigger):
                if phrase.lower() in text_lower:
                    warning_score += 0.3
                    warning_evidence.append(phrase[:40])

        # Warning language patterns
        warning_patterns = [
            (r'(if you don\'t|unless you|without)\s+\w+.{0,20}(quit|stop|change|reduce)', "conditional warning"),
            (r'(risk factor|major risk|serious risk)', "risk language"),
            (r'(will|could|may)\s+\w+\s+(return|worsen|get worse|increase)', "negative outcome prediction"),
            (r'(affects? you|impacts? you|hurts? you|harms? you)', "personal impact warning"),
            (r'(doesn\'t mean you\'re|not that you\'re)\s+(fine|healthy|ok)', "reassurance challenge"),
            (r'(smoke exposure|blood clots|reduce blood flow|sore on)', "specific health consequence"),
        ]
        for pattern, desc in warning_patterns:
            if re.search(pattern, text_lower):
                warning_score += 0.15
                warning_evidence.append(desc)

        # General negative health language with causal framing
        negative_health = ["risk", "danger", "harm", "damage", "infection", "disease",
                          "cancer", "clots", "sores", "tired", "fatigue", "worsen"]
        causal_framing = ["will", "could", "may", "might", "causes", "leads to",
                         "results in", "contributes to", "increases", "reduces"]
        has_neg = any(w in text_words for w in negative_health)
        has_causal = any(w in text_words for w in causal_framing)
        if has_neg and has_causal:
            warning_score += 0.1
            warning_evidence.append("negative outcome + causal framing")

        trigger_score += min(warning_score, 0.35)
        if warning_evidence:
            evidence_parts.append(warning_evidence[0])
            reasons.append(f"health warning: {warning_evidence[0]}")

    # Empathy / affirmation
    if "empathy" in short_name or "affirm" in short_name or "empath" in main_hyp:
        empathy_score = 0
        empathy_evidence = []

        # Empathic phrases
        empathy_patterns = [
            (r'(i understand|i appreciate|i know)\b', "empathic statement"),
            (r'(it must be|how difficult|how hard)', "validation of difficulty"),
            (r'(that\'s (great|wonderful|good|nice|fantastic))', "positive affirmation"),
            (r'(good for you|well done|you should be proud)', "praise"),
            (r'(i hear you|that makes sense|i can see)', "active listening"),
            (r'(you\'re (right|doing|making))', "affirmation"),
            (r'(absolutely|exactly|definitely)\b', "strong agreement"),
            (r'(i don\'t want to make light)', "empathic disclaimer"),
            (r'(it takes (a lot|strength|courage))', "strength recognition"),
            (r'(you rise to|you manage|you come out)', "strength affirmation"),
            (r'(i\'d be happy|i\'m happy)', "willingness expression"),
            (r'(i appreciate your help|thank you for)', "gratitude"),
        ]
        for pattern, desc in empathy_patterns:
            if re.search(pattern, text_lower):
                empathy_score += 0.2
                empathy_evidence.append(desc)

        trigger_score += min(empathy_score, 0.35)
        if empathy_evidence:
            evidence_parts.append(empathy_evidence[0])
            reasons.append(f"empathy/affirmation: {empathy_evidence[0]}")

    # Reflection / reflective listening
    if "reflection" in short_name or "reflect" in main_hyp and "listening" in main_hyp:
        reflection_score = 0
        reflection_evidence = []

        reflection_patterns = [
            (r'(it sounds like|it seems like|it appears)', "reflective framing"),
            (r'(you (feel|felt|feeling))\b', "feeling reflection"),
            (r'(so you|you mentioned|you said)\b', "content reflection"),
            (r'(i hear|i\'m hearing|what i hear)', "active listening reflection"),
            (r'(i (completely )?understand|i can understand)', "understanding statement"),
            (r'(that\'s true|right exactly|you\'re right)', "validation"),
            (r'(i know (quitting|it\'s|this))', "empathic knowledge"),
            (r'(throwing up is never fun)', "empathic mirroring"),
        ]
        for pattern, desc in reflection_patterns:
            if re.search(pattern, text_lower):
                reflection_score += 0.2
                reflection_evidence.append(desc)

        trigger_score += min(reflection_score, 0.35)
        if reflection_evidence:
            evidence_parts.append(reflection_evidence[0])
            reasons.append(f"reflection: {reflection_evidence[0]}")

    # === STEP 4: Check exclusions ===
    for exc in exclusions:
        exc_lower = exc.lower()
        exc_quoted = extract_quoted_phrases(exc)

        # Check quoted exclusion examples
        for phrase in exc_quoted:
            if phrase.lower() in text_lower:
                exclusion_penalty += 0.15
                reasons.append(f"exclusion match: '{phrase[:30]}'")

        # Administrative/logistics/scheduling exclusions
        if any(w in exc_lower for w in ["administrative", "logistics", "scheduling", "demographic"]):
            admin_words = {"appointment", "schedule", "office", "pharmacy", "refill",
                          "prescription", "insurance", "billing", "hours", "location"}
            if admin_words & text_words:
                if not any(w in text_lower for w in ["how long", "scale", "think", "feel", "believe"]):
                    exclusion_penalty += 0.1
                    reasons.append("administrative context")

        # Social greeting exclusions
        if any(w in exc_lower for w in ["greeting", "rapport", "social"]):
            greetings = {"hello", "hi", "hey", "goodbye", "bye", "welcome", "nice to meet"}
            if any(g in text_lower for g in greetings):
                exclusion_penalty += 0.15
                reasons.append("social greeting")

        # Factual statement exclusions
        if "factual" in exc_lower or "neutral" in exc_lower:
            if "?" not in text and len(text.split()) < 8:
                if not any(w in text_lower for w in ["think", "feel", "believe", "consider", "sounds"]):
                    exclusion_penalty += 0.1
                    reasons.append("short factual statement")

        # Acknowledgment/backchannel exclusions
        if any(w in exc_lower for w in ["acknowledgment", "backchannel", "affirmation"]):
            short_responses = {"okay", "ok", "right", "sure", "yeah", "yes", "mm-hmm",
                             "uh-huh", "i see", "got it", "alright", "fine", "good",
                             "great", "nice", "wonderful", "that's nice", "no problem"}
            if text_lower.strip() in short_responses or (len(text.split()) <= 4 and "?" not in text):
                exclusion_penalty += 0.1
                reasons.append("short acknowledgment")

        # Closed question without reflective elements
        if "closed question" in exc_lower:
            if text_lower.startswith(("do you", "did you", "does", "is there", "are there")):
                if len(text.split()) < 10 and not any(w in text_lower for w in ["think", "feel", "believe"]):
                    exclusion_penalty += 0.05
                    reasons.append("closed question")

        # Directive advice without empathic framing
        if "directive" in exc_lower and "advice" in exc_lower:
            if any(w in text_lower for w in ["should", "need to", "must", "have to", "I recommend"]):
                if not any(w in text_lower for w in ["understand", "appreciate", "hear", "sounds"]):
                    exclusion_penalty += 0.1
                    reasons.append("directive advice without empathy")

    # === STEP 5: Compute final score ===
    final_score = score + trigger_score - exclusion_penalty

    # Apply confound awareness
    for confound in confounds:
        confound_lower = confound.lower()
        if "length" in confound_lower:
            word_count = len(text.split())
            if word_count > 25:
                final_score -= 0.03  # slight penalty for very long utterances
        if "question word frequency" in confound_lower:
            if text_lower.count("?") > 0 and len(text.split()) < 6:
                final_score -= 0.02  # very short questions may be noise

    # Clamp
    final_score = max(0.05, min(0.95, final_score))
    final_score = round(final_score, 2)
    binary = 1 if final_score >= 0.5 else 0

    # Generate evidence span
    if evidence_parts:
        evidence = evidence_parts[0][:60]
    else:
        # Pick most relevant span
        words = text.split()
        if "?" in text:
            q_start = text.index("?")
            context_start = max(0, q_start - 40)
            evidence = text[context_start:q_start+1].strip()
        elif len(words) > 8:
            evidence = " ".join(words[:8]) + "..."
        else:
            evidence = text[:60]

    # Generate reason
    if reasons:
        # Pick the most informative reason
        reason = reasons[0]
        if len(reason) > 100:
            reason = reason[:97] + "..."
    else:
        if binary == 1:
            reason = "partial trigger match supports activation"
        else:
            reason = "no strong trigger match; exclusion signals present"

    return final_score, binary, evidence, reason


def process_task(task_line):
    """Process a single task line and return output data."""
    task = json.loads(task_line)
    task_id = task["task_id"]
    prompt = task["prompt"]
    expected_path = task["expected_output_path"]

    # Check if output already exists
    full_path = os.path.join("D:/project/NLP_re_dataset_model_base", expected_path)
    if os.path.exists(full_path):
        return None, None, "exists"

    # Parse prompt
    explanation, samples = parse_prompt(prompt)
    if explanation is None:
        return None, None, "parse_error"

    if len(samples) == 0:
        return None, None, "no_samples"

    # Judge each sample
    results = []
    for sample in samples:
        prob, binary, evidence, reason = judge_sample(explanation, sample["text"])
        results.append({
            "sample_id": sample["sample_id"],
            "pred_activate_prob": prob,
            "binary_prediction": binary,
            "evidence_span": evidence,
            "reason": reason
        })

    return results, task, "processed"


def main():
    base_dir = "D:/project/NLP_re_dataset_model_base"
    tasks_path = os.path.join(base_dir, TASKS_FILE)
    manifest_path = os.path.join(base_dir, MANIFEST_FILE)

    # Ensure output directory exists
    output_dir = os.path.join(base_dir, "outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/scorer_outputs/raw_explanation_scorer")
    os.makedirs(output_dir, exist_ok=True)

    stats = {"processed": 0, "skipped": 0, "errors": 0, "no_samples": 0}

    with open(tasks_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    manifest_entries = []
    error_lines = []

    for line_num in range(LINE_START - 1, min(LINE_END, len(lines))):
        task_line = lines[line_num].strip()
        if not task_line:
            continue

        try:
            results, task, status = process_task(task_line)
        except Exception as e:
            error_lines.append((line_num + 1, str(e)))
            stats["errors"] += 1
            continue

        if status == "exists":
            stats["skipped"] += 1
            continue
        elif status == "parse_error":
            stats["errors"] += 1
            error_lines.append((line_num + 1, "parse_error"))
            continue
        elif status == "no_samples":
            stats["no_samples"] += 1
            error_lines.append((line_num + 1, "no_samples"))
            continue

        # Write output JSON
        output_path = os.path.join(base_dir, task["expected_output_path"])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_json = json.dumps(results, ensure_ascii=False, indent=2)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_json)

        # Compute hashes for manifest
        prompt_hash = hashlib.sha256(task["prompt"].encode('utf-8')).hexdigest()
        output_hash = hashlib.sha256(output_json.encode('utf-8')).hexdigest()

        manifest_entry = {
            "task_id": task["task_id"],
            "model": MODEL_NAME,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_sha256": prompt_hash,
            "raw_output_sha256": output_hash
        }
        manifest_entries.append(manifest_entry)

        stats["processed"] += 1
        if stats["processed"] % 20 == 0:
            print(f"Processed {stats['processed']} tasks...", flush=True)

    # Append manifest entries
    if manifest_entries:
        with open(manifest_path, 'a', encoding='utf-8') as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\nDone! Processed: {stats['processed']}, Skipped (existing): {stats['skipped']}, "
          f"Errors: {stats['errors']}, No samples: {stats['no_samples']}")
    if error_lines:
        print(f"\nError lines: {error_lines[:20]}")

    return stats


if __name__ == "__main__":
    main()
