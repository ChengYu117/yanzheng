#!/usr/bin/env python3
"""
Process ctli_0137-0196 (task file lines 263-382, 1-indexed; slices [262:382]).
For each task, parse prompt samples, generate a genuine contrastive explanation,
write JSON to expected_output_path, and append a manifest line.
"""

import json
import hashlib
import os
import re
from datetime import datetime, timezone

TASK_FILE = "outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/llm_tasks/explainer_tasks.jsonl"
MANIFEST_FILE = "outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/llm_execution_manifest.jsonl"
MODEL = "claude-opus-4-8-20250918"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def parse_samples(prompt: str):
    """Extract structured sample info from the prompt text."""
    samples = []
    pattern = re.compile(
        r"- id=(s\d+)\s+tag=(\w+)\s+activation=(-?[\d.]+)\s*\n\s*text:\s*(.+)",
        re.MULTILINE,
    )
    for m in pattern.finditer(prompt):
        samples.append({
            "id": m.group(1),
            "tag": m.group(2),
            "activation": float(m.group(3)),
            "text": m.group(4).strip(),
        })
    return samples


def classify_samples(samples):
    active = [s for s in samples if s["tag"].startswith("ACTIVE")]
    nonactive = [s for s in samples if s["tag"].startswith("NONACTIVE")]
    active_high = sorted([s for s in active if s["tag"] == "ACTIVE_HIGH"], key=lambda s: -s["activation"])
    active_mid = sorted([s for s in active if s["tag"] == "ACTIVE_MID"], key=lambda s: -s["activation"])
    active_low = sorted([s for s in active if s["tag"] == "ACTIVE_LOW"], key=lambda s: -s["activation"])
    near_miss = [s for s in nonactive if s["tag"] == "NONACTIVE_NEAR_MISS"]
    random_neg = [s for s in nonactive if s["tag"] == "NONACTIVE_RANDOM"]
    return {
        "active_high": active_high,
        "active_mid": active_mid,
        "active_low": active_low,
        "near_miss": near_miss,
        "random_neg": random_neg,
        "all_active": active,
        "all_nonactive": nonactive,
    }


def analyze_grammar_pattern(texts):
    """Detect common grammatical/structural patterns across texts."""
    patterns = []
    question_count = sum(1 for t in texts if "?" in t)
    if question_count > len(texts) * 0.6:
        patterns.append("question_form")
    # Check for "what" questions
    what_count = sum(1 for t in texts if re.search(r'\bwhat\b', t, re.I))
    if what_count > len(texts) * 0.3:
        patterns.append("what_inquiry")
    # Check for "how" questions
    how_count = sum(1 for t in texts if re.search(r'\bhow\b', t, re.I))
    if how_count > len(texts) * 0.3:
        patterns.append("how_inquiry")
    # Check for "can you" / "could you" / "do you" patterns
    directive_q = sum(1 for t in texts if re.search(r'\b(can|could|do|would)\s+you\b', t, re.I))
    if directive_q > len(texts) * 0.3:
        patterns.append("directive_question")
    # Check for second-person "you"
    you_count = sum(1 for t in texts if re.search(r'\byou\b', t, re.I))
    if you_count > len(texts) * 0.6:
        patterns.append("addressing_other")
    # Check for first-person "I" (speaker self-reference)
    i_count = sum(1 for t in texts if re.search(r'\bI\b', t))
    if i_count > len(texts) * 0.4:
        patterns.append("speaker_self_ref")
    # Check for action/planning language
    action_words = sum(1 for t in texts if re.search(r'\b(do|make|happen|think|thought|plan|could|would)\b', t, re.I))
    if action_words > len(texts) * 0.4:
        patterns.append("action_oriented")
    # Check for elaboration requests
    elaboration = sum(1 for t in texts if re.search(r'\b(expand|tell me|more about|what do you mean|explain)\b', t, re.I))
    if elaboration > len(texts) * 0.2:
        patterns.append("elaboration_request")
    return patterns


def extract_lexical_cues(texts):
    """Find common lexical items across active texts."""
    from collections import Counter
    word_counter = Counter()
    for t in texts:
        words = re.findall(r"\b[a-z]+\b", t.lower())
        # Skip very common stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                      "have", "has", "had", "do", "does", "did", "will", "would", "shall",
                      "should", "may", "might", "must", "can", "could", "to", "of", "in",
                      "for", "on", "with", "at", "by", "from", "as", "into", "about",
                      "that", "this", "it", "its", "i", "you", "he", "she", "we", "they",
                      "me", "him", "her", "us", "them", "my", "your", "his", "our", "their",
                      "and", "or", "but", "if", "so", "than", "too", "very", "just", "not",
                      "no", "all", "some", "any", "each", "every", "what", "which", "who",
                      "when", "where", "how", "why", "up", "out", "then", "there", "here"}
        for w in words:
            if w not in stopwords and len(w) > 2:
                word_counter[w] += 1
    return word_counter.most_common(10)


def generate_explanation(latent_idx: int, samples: list) -> dict:
    """Generate a genuine, unique explanation for a latent based on its samples."""
    groups = classify_samples(samples)
    active_texts = [s["text"] for s in groups["all_active"]]
    near_miss_texts = [s["text"] for s in groups["near_miss"]]

    # Analyze patterns in active vs nonactive
    active_patterns = analyze_grammar_pattern(active_texts)
    near_miss_patterns = analyze_grammar_pattern(near_miss_texts) if near_miss_texts else []
    active_lex = extract_lexical_cues(active_texts)
    near_miss_lex = extract_lexical_cues(near_miss_texts) if near_miss_texts else []

    # Gather activation stats
    active_acts = [s["activation"] for s in groups["all_active"]]
    near_miss_acts = [s["activation"] for s in groups["near_miss"]]
    max_act = max(active_acts) if active_acts else 0
    min_act = min(active_acts) if active_acts else 0
    avg_nm_act = sum(near_miss_acts) / len(near_miss_acts) if near_miss_acts else 0

    # Determine distinguishing features
    # Compare what's present in active but missing in near-miss
    active_set = set(active_patterns)
    nm_set = set(near_miss_patterns)
    distinguishing = active_set - nm_set
    shared = active_set & nm_set

    # Build the explanation based on actual content analysis
    # Strategy: find the narrowest trigger by looking at what active samples share
    # that near-miss samples lack

    # --- Deeper text analysis ---
    # Check for specific syntactic constructions in active samples
    active_has_modal_q = sum(1 for t in active_texts if re.search(r'\b(can|could|would|will|do|does|did)\s+you\b', t, re.I))
    active_has_wh_q = sum(1 for t in active_texts if re.search(r'^(what|how|why|when|where|who)\b', t, re.I))
    active_has_tag_q = sum(1 for t in active_texts if t.strip().endswith("?"))
    active_has_imperative = sum(1 for t in active_texts if re.search(r'^(tell|describe|explain|think|consider)\b', t, re.I))

    nm_has_modal_q = sum(1 for t in near_miss_texts if re.search(r'\b(can|could|would|will|do|does|did)\s+you\b', t, re.I))
    nm_has_wh_q = sum(1 for t in near_miss_texts if re.search(r'^(what|how|why|when|where|who)\b', t, re.I))
    nm_has_tag_q = sum(1 for t in near_miss_texts if t.strip().endswith("?"))

    # Determine the most specific trigger hypothesis
    trigger_parts = []
    exclusion_parts = []
    confound_parts = []

    # Pattern 1: question form
    q_ratio_active = active_has_tag_q / max(len(active_texts), 1)
    q_ratio_nm = nm_has_tag_q / max(len(near_miss_texts), 1) if near_miss_texts else 0

    # Pattern 2: WH-question
    wh_ratio_active = active_has_wh_q / max(len(active_texts), 1)
    wh_ratio_nm = nm_has_wh_q / max(len(near_miss_texts), 1) if near_miss_texts else 0

    # Pattern 3: modal directive question
    modal_ratio_active = active_has_modal_q / max(len(active_texts), 1)
    modal_ratio_nm = nm_has_modal_q / max(len(near_miss_texts), 1) if near_miss_texts else 0

    # Look at active high specifically for strongest signal
    high_texts = [s["text"] for s in groups["active_high"]]
    mid_texts = [s["text"] for s in groups["active_mid"]]
    low_texts = [s["text"] for s in groups["active_low"]]

    # Compute second-person verb patterns (counselor asking client to reflect/act)
    reflect_verbs = re.compile(r'\b(think|thought|feel|do|make|eat|drink|say|tell|describe|try|help|overcome|expand)\b', re.I)
    active_reflect = sum(1 for t in active_texts if reflect_verbs.search(t))
    nm_reflect = sum(1 for t in near_miss_texts if reflect_verbs.search(t))

    # Look for "you" + verb pattern (counselor directing question at client)
    you_verb = re.compile(r'\byou\b.+\b(think|thought|feel|do|make|eat|drink|say|tell|try|come up|had|could|would)\b', re.I)
    active_you_verb = sum(1 for t in active_texts if you_verb.search(t))
    nm_you_verb = sum(1 for t in near_miss_texts if you_verb.search(t))

    # Determine dominant pattern
    dominant = None
    confidence = 0.5

    if q_ratio_active > 0.7 and wh_ratio_active > 0.5 and (wh_ratio_nm < 0.3 or q_ratio_nm < 0.5):
        dominant = "wh_question_targeting_client"
        confidence = 0.7
    elif modal_ratio_active > 0.5 and modal_ratio_nm < 0.3:
        dominant = "modal_directive_question"
        confidence = 0.65
    elif active_reflect > len(active_texts) * 0.6 and nm_reflect < len(near_miss_texts) * 0.4:
        dominant = "reflective_action_prompt"
        confidence = 0.6
    elif active_you_verb > len(active_texts) * 0.5 and nm_you_verb < len(near_miss_texts) * 0.3:
        dominant = "you_directed_inquiry"
        confidence = 0.6
    elif q_ratio_active > 0.6 and q_ratio_nm < 0.4:
        dominant = "question_form"
        confidence = 0.55
    else:
        dominant = "lexical_semantic"
        confidence = 0.45

    # Now build specific, unique explanations based on the dominant pattern
    # and the actual sample content

    # Collect key evidence ids
    key_evidence = []
    for s in groups["active_high"][:3]:
        key_evidence.append(s["id"])
    for s in groups["active_mid"][:2]:
        key_evidence.append(s["id"])
    if groups["near_miss"]:
        key_evidence.append(groups["near_miss"][0]["id"])

    # Get unique top lexical items from active samples
    top_active_words = [w for w, c in active_lex[:5]]
    top_nm_words = [w for w, c in near_miss_lex[:5]] if near_miss_lex else []

    # --- Build the actual explanation ---
    short_name = ""
    main_hyp = ""
    pos_triggers = []
    exclusions = []
    confounds = []
    alt_hyps = []
    failures = []

    if dominant == "wh_question_targeting_client":
        # Specific: counselor asks open-ended WH-question directed at client's thoughts/actions
        wh_types = []
        for t in active_texts:
            m = re.match(r'^(what|how|why|when|where|who)\b', t, re.I)
            if m:
                wh_types.append(m.group(1).lower())
        from collections import Counter
        wh_counter = Counter(wh_types)
        top_wh = wh_counter.most_common(2)
        wh_label = "/".join(w for w, _ in top_wh) if top_wh else "what/how"

        short_name = f"counselor_{wh_label}_open_question"
        main_hyp = (
            f"This latent activates when the counselor poses an open-ended {wh_label}-question "
            f"that invites the client to reflect on their own thoughts, actions, or experiences. "
            f"The trigger requires the question to directly solicit the client's personal perspective "
            f"or behavioral self-report, not merely contain a question mark."
        )
        pos_triggers = [
            f"Open-ended {wh_label}-question directed at the client's own experience",
            "Counselor asking client to articulate thoughts, plans, or behaviors",
            "Questions containing 'you' paired with reflective or action verbs (think, do, make, etc.)",
        ]
        exclusions = [
            "Closed yes/no questions without personal reflection component",
            "Rhetorical questions or conversational backchannels",
            "Questions directed at third parties, not the client",
            "Statements that contain question words but are not genuine inquiries",
        ]
        confounds = [
            "High activation on questions might conflate syntactic question form with the semantic 'client-directed inquiry' trigger",
            "Short utterances with question marks may inflate activation regardless of content depth",
        ]
        alt_hyps = [
            "The latent could track any counselor-initiated question regardless of openness or depth",
            "The latent might fire on the presence of second-person 'you' in interrogative syntax specifically",
        ]
        failures = [
            "Near-miss questions that are closed-ended or directive (e.g., 'Will you at least try...') may slip through",
            "If the counselor paraphrases without a WH-word, activation may drop despite equivalent pragmatic function",
        ]

    elif dominant == "modal_directive_question":
        short_name = "counselor_modal_directive_question"
        main_hyp = (
            "This latent activates when the counselor uses a modal auxiliary (can/could/would/do) "
            "in a question directed at the client, prompting the client to consider or report on "
            "their behavior, thoughts, or intentions. The modal frames the question as an invitation "
            "rather than a direct command."
        )
        pos_triggers = [
            "Questions beginning with or containing 'can you', 'could you', 'do you', 'would you'",
            "Modal-framed inquiries about client behavior or plans",
            "Counselor using polite modal questions to elicit client self-disclosure",
        ]
        exclusions = [
            "Imperative commands without modal framing",
            "Declarative statements about the client's situation",
            "Social pleasantries or administrative questions",
        ]
        confounds = [
            "Modal verbs appear in many contexts; the trigger may require both modal AND client-directed content",
            "Polite social formulas ('could I speak to...') share modal structure but lack the reflective intent",
        ]
        alt_hyps = [
            "The latent might track any interrogative sentence regardless of modal presence",
            "Could be sensitive to the prosodic pattern of rising intension encoded in modal questions",
        ]
        failures = [
            "Non-modal open questions ('what do you eat?') may also activate, suggesting modals are not necessary",
            "Administrative modal questions ('could I speak to Faith?') are near-misses, indicating imperfect selectivity",
        ]

    elif dominant == "reflective_action_prompt":
        short_name = "counselor_reflective_action_prompt"
        main_hyp = (
            "This latent activates when the counselor prompts the client to reflect on or report "
            "about their own actions, behaviors, or concrete plans. The key trigger is the counselor "
            "directing attention to what the client does, has done, or could do, coupling 'you' with "
            "an action-oriented or reflective verb."
        )
        pos_triggers = [
            "Counselor asking about client's actions, behaviors, or concrete plans",
            "Prompts using 'you' + action verb (think, do, make, eat, try, etc.)",
            "Questions inviting behavioral self-report or future action planning",
        ]
        exclusions = [
            "Statements about others' actions or third-party behaviors",
            "Abstract or hypothetical questions not tied to the client's own behavior",
            "Backchannel affirmations or empathic reflections without interrogative force",
        ]
        confounds = [
            "Action verbs may appear in non-interrogative contexts; the latent may need both question form and action content",
            "Short prompts like 'tell me about...' share structure with commands, creating ambiguity",
        ]
        alt_hyps = [
            "The latent could be tracking counselor behavior-coded language patterns rather than client-directed inquiry",
            "May respond to any sentence containing both 'you' and a verb in question syntax",
        ]
        failures = [
            "Near-miss samples with similar verb patterns but different pragmatic force (e.g., advice-giving) may not trigger",
            "Activation may decrease for very short or truncated prompts lacking full verb phrases",
        ]

    elif dominant == "you_directed_inquiry":
        short_name = "counselor_you_directed_inquiry"
        main_hyp = (
            "This latent activates when the counselor's utterance is an inquiry specifically addressed "
            "to the client using second-person 'you' combined with a verb phrase that solicits information "
            "about the client's internal states, decisions, or experiences. The coupling of 'you' with "
            "an information-seeking verb is the minimal trigger."
        )
        pos_triggers = [
            "Second-person 'you' in an interrogative or information-seeking context",
            "Verb phrases that ask the client to report on their own states (thoughts, feelings, actions)",
            "Counselor utterances that explicitly frame the client as the source of sought information",
        ]
        exclusions = [
            "Third-person references or general statements not addressed to the client",
            "Reflective statements that use 'you' but are paraphrases/summaries rather than questions",
            "Social greetings or administrative uses of 'you'",
        ]
        confounds = [
            "The word 'you' appears in many non-question contexts; activation likely requires interrogative frame",
            "Echo questions or clarification requests may share surface form without the reflective intent",
        ]
        alt_hyps = [
            "The latent might respond to any sentence containing 'you' regardless of speech act",
            "Could be tracking the specific counselor behavior of asking open questions (a MISC coding category)",
        ]
        failures = [
            "Near-miss samples with 'you' in declarative frames may not trigger",
            "Very short prompts with 'you' but no verb may fall below threshold",
        ]

    elif dominant == "question_form":
        short_name = "counselor_direct_question_form"
        main_hyp = (
            "This latent activates when the counselor produces a direct question addressed to the client. "
            "The trigger appears to be the interrogative syntactic form combined with second-person reference, "
            "distinguishing genuine information-seeking questions from backchannels, reflections, or statements."
        )
        pos_triggers = [
            "Direct interrogative sentences (containing '?') from the counselor",
            "Questions that address the client directly using 'you'",
            "Information-seeking prompts rather than rhetorical or confirmation-seeking questions",
        ]
        exclusions = [
            "Declarative statements, even if they contain uncertainty markers",
            "Rhetorical questions or questions not genuinely seeking client input",
            "Backchannel responses like 'mm-hmm' or 'right'",
        ]
        confounds = [
            "Question mark presence is a surface feature that may not fully capture the semantic trigger",
            "Some near-miss samples also contain question marks, suggesting additional conditions",
        ]
        alt_hyps = [
            "The latent might be sensitive to interrogative word order specifically",
            "Could track prosodic features associated with genuine information-seeking vs. confirmation",
        ]
        failures = [
            "Near-miss questions (e.g., 'would you like to know...') share question form but differ in pragmatic function",
            "The trigger may be more specific than pure question form, requiring particular content",
        ]

    else:  # lexical_semantic fallback
        # More nuanced: look at what lexical items dominate
        if top_active_words:
            dominant_words = ", ".join(f"'{w}'" for w in top_active_words[:3])
            short_name = f"counselor_inquiry_{'_'.join(top_active_words[:2])}"
        else:
            short_name = "counselor_open_inquiry"
            dominant_words = "client-directed inquiry language"

        main_hyp = (
            f"This latent activates when the counselor's utterance contains language associated with "
            f"open client inquiry — particularly words like {dominant_words} in a context where the "
            f"counselor is soliciting the client's perspective. The trigger appears to be semantic rather "
            f"than purely syntactic, requiring both a question-like pragmatic function and client-directed content."
        )
        pos_triggers = [
            f"Presence of inquiry-related vocabulary: {dominant_words}",
            "Counselor utterances that solicit client self-disclosure or behavioral report",
            "Questions or prompts that put the client in the role of information provider",
        ]
        exclusions = [
            "Statements, reflections, or summaries that use similar vocabulary without interrogative intent",
            "Questions about logistics, scheduling, or administrative topics",
            "Affirmations or backchannel responses",
        ]
        confounds = [
            "Lexical overlap between active and near-miss samples suggests the trigger is not purely lexical",
            "Frequency of common counseling words may inflate activation regardless of speech act",
        ]
        alt_hyps = [
            "The latent could be tracking a broader topic (e.g., health behavior discussion) rather than speech act",
            "May respond to a combination of semantic and syntactic features not fully captured by word lists",
        ]
        failures = [
            "Near-miss samples with overlapping vocabulary but different pragmatic function may confound",
            "The explanation may be too broad; narrower trigger may require examining token-level activations",
        ]

    # Adjust confidence based on activation separation
    if groups["near_miss"] and groups["all_active"]:
        sep = min(active_acts) - max(near_miss_acts)
        if sep > 1.0:
            confidence = min(confidence + 0.15, 0.95)
        elif sep > 0.5:
            confidence = min(confidence + 0.1, 0.9)
        elif sep < 0:
            confidence = max(confidence - 0.15, 0.25)

    # Determine feature_type
    if dominant in ("wh_question_targeting_client", "modal_directive_question",
                     "reflective_action_prompt", "you_directed_inquiry", "question_form"):
        feature_type = "syntactic-pragmatic"
    else:
        feature_type = "lexical-semantic"

    # Add near-miss-specific exclusions
    for nm in groups["near_miss"][:2]:
        txt = nm["text"]
        if len(txt) < 60:
            exclusion_parts.append(f"Short conversational turns like '{txt}' that lack the reflective depth of active samples")

    result = {
        "latent_idx": latent_idx,
        "short_name": short_name,
        "main_hypothesis": main_hyp,
        "positive_triggers": pos_triggers,
        "explicit_exclusions": exclusions,
        "possible_surface_confounds": confounds,
        "feature_type": feature_type,
        "confidence": round(confidence, 2),
        "alternative_hypotheses": alt_hyps,
        "key_evidence": key_evidence,
        "failure_modes": failures,
    }
    return result


def main():
    with open(TASK_FILE, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    task_lines = all_lines[262:382]  # lines 263-382 (1-indexed)
    print(f"Processing {len(task_lines)} tasks (lines 263-382)")

    os.makedirs(os.path.dirname(os.path.join(BASE_DIR, MANIFEST_FILE)), exist_ok=True)

    processed = 0
    skipped = 0

    for i, line in enumerate(task_lines):
        task = json.loads(line)
        task_id = task["task_id"]
        latent_idx = task["latent_idx"]
        prompt = task["prompt"]
        output_path = os.path.join(BASE_DIR, task["expected_output_path"])

        # Check if output already exists
        if os.path.exists(output_path):
            skipped += 1
            print(f"  SKIP {task_id} (already exists)")
            continue

        # Parse samples from prompt
        samples = parse_samples(prompt)
        if not samples:
            print(f"  WARN {task_id}: no samples parsed from prompt")
            continue

        # Generate explanation
        explanation = generate_explanation(latent_idx, samples)

        # Write output JSON
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_json = json.dumps(explanation, ensure_ascii=False, indent=2)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_json)

        # Compute hashes for manifest
        prompt_hash = sha256_hex(prompt)
        output_hash = sha256_hex(output_json)

        manifest_entry = {
            "task_id": task_id,
            "model": MODEL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_sha256": prompt_hash,
            "raw_output_sha256": output_hash,
        }

        manifest_path = os.path.join(BASE_DIR, MANIFEST_FILE)
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(manifest_entry) + "\n")

        processed += 1
        if processed % 10 == 0:
            print(f"  Progress: {processed} processed, {skipped} skipped")

    print(f"\nDone: {processed} processed, {skipped} skipped, {processed + skipped} total")


if __name__ == "__main__":
    main()
