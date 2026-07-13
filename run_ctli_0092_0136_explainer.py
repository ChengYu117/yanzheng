"""
Process contrastive latent explainer tasks ctli_0092 through ctli_0136
(lines 181-268 of explainer_tasks.jsonl, 0-indexed [180:268]).

For each task:
1. Parse JSON line for task_id, latent_idx, prompt, expected_output_path
2. Check if output already exists; skip if so
3. Analyze the ACTIVE vs NONACTIVE samples to generate a genuine explanation
4. Write the explanation JSON to expected_output_path
5. Append manifest line
"""

import json
import hashlib
import os
import re
from datetime import datetime, timezone
from collections import Counter

TASK_FILE = "outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/llm_tasks/explainer_tasks.jsonl"
MANIFEST_FILE = "outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/llm_execution_manifest.jsonl"
BASE_DIR = "D:/project/NLP_re_dataset_model_base"


def parse_samples_from_prompt(prompt: str) -> list[dict]:
    """Extract structured sample info from the prompt text.

    The prompt format has samples as two lines each:
        - id=sXXX tag=YYY activation=ZZZ
          text: the utterance text
    """
    samples = []
    lines = prompt.split("\n")
    i = 0
    while i < len(lines):
        line_s = lines[i].strip()
        if line_s.startswith("- id=") and "tag=" in line_s and "activation=" in line_s:
            tokens = line_s.split()
            # tokens are: ['-', 'id=sXXX', 'tag=YYY', 'activation=ZZZ']
            sid = tokens[1].replace("id=", "")
            tag = tokens[2].replace("tag=", "")
            act = float(tokens[3].replace("activation=", ""))
            text = ""
            if "  text:" in line_s:
                text = line_s.split("  text:", 1)[1].strip()
            elif i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith("text:"):
                    text = next_line[len("text:"):].strip()
                    i += 1
            samples.append({"id": sid, "tag": tag, "activation": act, "text": text})
        i += 1
    return samples


# ── Helpers ──────────────────────────────────────────────────────────────────

def simple_tok(text):
    return [w.lower().strip(".,;:!?()\"'`") for w in text.split() if w.strip(".,;:!?()\"'`")]

def extract_ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def substrings(text, min_len=3, max_len=30):
    """Extract contiguous word substrings from text."""
    words = text.lower().split()
    subs = []
    for length in range(min_len, min(max_len, len(words)+1)):
        for i in range(len(words) - length + 1):
            subs.append(" ".join(words[i:i+length]))
    return subs


# ── Core contrastive analysis engine ────────────────────────────────────────

def find_contrastive_features(all_active, all_nonactive, near_misses):
    """
    Find what's genuinely different between active and near-miss samples
    for THIS specific latent. Returns a ranked list of candidate hypotheses.
    """
    at = [s["text"] for s in all_active]
    nt = [s["text"] for s in near_misses] if near_misses else [s["text"] for s in all_nonactive]

    candidates = []

    # ── Strategy 1: Phrase-level analysis ──
    # Find multi-word phrases that appear in most active but not in near-miss
    # Use 2-5 word contiguous substrings

    # Count phrase occurrences across active samples
    active_phrase_counts = Counter()  # phrase -> number of active samples containing it
    near_phrase_counts = Counter()

    for text in at:
        words = simple_tok(text)
        seen = set()
        for n in range(2, min(6, len(words)+1)):
            for ng in extract_ngrams(words, n):
                seen.add(ng)
        for ng in seen:
            active_phrase_counts[ng] += 1

    for text in nt:
        words = simple_tok(text)
        seen = set()
        for n in range(2, min(6, len(words)+1)):
            for ng in extract_ngrams(words, n):
                seen.add(ng)
        for ng in seen:
            near_phrase_counts[ng] += 1

    # Find phrases present in >= 40% of active samples and < 20% of near-miss
    n_active = len(at)
    n_near = max(len(nt), 1)
    active_threshold = max(2, int(n_active * 0.35))
    near_threshold = max(1, int(n_near * 0.25))

    distinguishing_phrases = []
    for phrase, count in active_phrase_counts.most_common(200):
        if count < active_threshold:
            continue
        near_count = near_phrase_counts.get(phrase, 0)
        if near_count <= near_threshold:
            # Score: how many active have it vs how few near-miss have it
            score = count / max(near_count, 0.1)
            distinguishing_phrases.append((phrase, count, near_count, score))

    distinguishing_phrases.sort(key=lambda x: x[3], reverse=True)

    if distinguishing_phrases:
        best_phrase, best_act_count, best_near_count, best_score = distinguishing_phrases[0]
        candidates.append({
            "type": "phrase_pattern",
            "description": f"Presence of the phrase pattern '{best_phrase}' or close variants",
            "act_count": best_act_count,
            "near_count": best_near_count,
            "score": best_score,
            "phrase": best_phrase,
            "all_top_phrases": distinguishing_phrases[:5],
        })

    # ── Strategy 2: Semantic frame analysis ──
    # Look for specific speech-act frames

    frames = {
        "reflective_sounds_like": {
            "pattern": r"\bsounds? like\b",
            "description": "Reflective 'sounds like' framing",
        },
        "reflective_seems": {
            "pattern": r"\bseems? like\b|\bit seems\b",
            "description": "Reflective 'seems like' framing",
        },
        "reflective_looks": {
            "pattern": r"\blooks? like\b|\bit looks\b",
            "description": "Reflective 'looks like' framing",
        },
        "reflective_feels": {
            "pattern": r"\byou feel\b|\byou're feeling\b",
            "description": "Reflection on client's feeling state",
        },
        "reflective_hearing": {
            "pattern": r"\bwhat i'm hearing\b|\bi hear\b|\bi'm hearing\b",
            "description": "Explicit hearing/restatement frame",
        },
        "reflective_you_re_saying": {
            "pattern": r"\byou('re| are) saying\b|\byou said\b",
            "description": "Attribution of client's prior statement",
        },
        "question_how_often": {
            "pattern": r"\bhow often\b|\bhow long\b|\bhow much\b|\bhow many\b",
            "description": "Quantifying question about frequency/amount",
        },
        "question_tell_me": {
            "pattern": r"\btell me\b|\bcan you tell\b",
            "description": "Open-ended elicitation question",
        },
        "question_what_do": {
            "pattern": r"\bwhat do you\b|\bwhat would you\b|\bwhat did you\b",
            "description": "Client-experience elicitation question",
        },
        "question_do_you": {
            "pattern": r"\bdo you\b|\bare you\b|\bhave you\b|\bcan you\b",
            "description": "Closed yes/no question to client",
        },
        "question_is_there": {
            "pattern": r"\bis there\b|\bare there\b",
            "description": "Existential question about client's situation",
        },
        "directive_need_to": {
            "pattern": r"\byou need to\b|\byou should\b|\bit's important\b",
            "description": "Directive advising client action",
        },
        "directive_let_me": {
            "pattern": r"\blet me\b|\blet's\b|\bi want you to\b",
            "description": "Collaborative directive or instruction",
        },
        "directive_try_to": {
            "pattern": r"\btry to\b|\btry and\b|\bmake sure\b",
            "description": "Suggestion or encouragement to act",
        },
        "affirmation_positive": {
            "pattern": r"^(good|great|excellent|wonderful|awesome|fantastic|very good|that('s| is) great|that('s| is) good|nice|perfect|wonderful)\b",
            "description": "Brief positive affirmation/acknowledgment",
        },
        "open_so_reflective": {
            "pattern": r"^so\b.*\b(you|your|it)\b",
            "description": "So-prefaced reflective or transitional statement",
        },
        "open_okay_question": {
            "pattern": r"^(okay|ok|alright)\b.*\?",
            "description": "Acknowledgment followed by question",
        },
        "client_situation_naming": {
            "pattern": r"\b(your|you're|you are)\b.*(situation|issue|problem|concern|challenge|experience|condition)",
            "description": "Naming or labeling the client's situation",
        },
        "counselor_knowledge_claim": {
            "pattern": r"\bi (know|understand|see|hear|notice|think|believe)\b|\bwhat i know\b",
            "description": "Counselor asserting understanding or knowledge",
        },
        "conditional_reasoning": {
            "pattern": r"\bif you\b|\bwhen you\b|\bwhat if\b|\bsuppose\b",
            "description": "Conditional/hypothetical reasoning about client actions",
        },
        "temporal_reference": {
            "pattern": r"\bin the (future|past|last)\b|\bgoing forward\b|\bso far\b|\bback then\b",
            "description": "Temporal reference to past or future",
        },
        "comparison_contrast": {
            "pattern": r"\b(on the other hand|but|however|instead|rather than|compared to)\b",
            "description": "Comparison or contrast structure",
        },
        "enumeration_listing": {
            "pattern": r"\b(first|second|third|one thing|another|also|additionally|plus)\b",
            "description": "Enumerating or listing points",
        },
    }

    frame_scores = []
    for frame_name, frame_info in frames.items():
        act_count = sum(1 for t in at if re.search(frame_info["pattern"], t, re.IGNORECASE))
        near_count = sum(1 for t in nt if re.search(frame_info["pattern"], t, re.IGNORECASE))
        if act_count >= 2 and (act_count / max(near_count, 0.5)) > 1.3:
            score = act_count / max(near_count, 0.5) * act_count
            frame_scores.append((frame_name, frame_info["description"], act_count, near_count, score))

    frame_scores.sort(key=lambda x: x[4], reverse=True)

    for frame_name, desc, act_c, near_c, score in frame_scores[:3]:
        candidates.append({
            "type": "semantic_frame",
            "description": desc,
            "act_count": act_c,
            "near_count": near_c,
            "score": score,
            "frame": frame_name,
        })

    # ── Strategy 3: Opening word / discourse marker analysis ──
    active_openers = Counter()
    near_openers = Counter()
    for t in at:
        words = t.strip().split()
        if words:
            active_openers[words[0].lower()] += 1
    for t in nt:
        words = t.strip().split()
        if words:
            near_openers[words[0].lower()] += 1

    opener_diffs = []
    for w in set(list(active_openers.keys()) + list(near_openers.keys())):
        a = active_openers.get(w, 0)
        n = near_openers.get(w, 0)
        if a >= 2 and a > n * 1.5:
            opener_diffs.append((w, a, n))
    opener_diffs.sort(key=lambda x: x[1] - x[2], reverse=True)

    if opener_diffs and opener_diffs[0][1] >= 3:
        top_openers = opener_diffs[:3]
        opener_str = ", ".join(f"'{w}' ({a}/{n})" for w, a, n in top_openers)
        candidates.append({
            "type": "opener_pattern",
            "description": f"Discourse markers at utterance start: {opener_str}",
            "act_count": top_openers[0][1],
            "near_count": top_openers[0][2],
            "score": top_openers[0][1] / max(top_openers[0][2], 0.5),
        })

    # ── Strategy 4: Lexical content analysis ──
    # Find content words (not stopwords) enriched in active

    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                 "have", "has", "had", "do", "does", "did", "will", "would", "could",
                 "should", "may", "might", "shall", "can", "to", "of", "in", "for",
                 "on", "with", "at", "by", "from", "as", "into", "through", "during",
                 "before", "after", "above", "below", "between", "out", "off", "over",
                 "under", "again", "further", "then", "once", "here", "there", "when",
                 "where", "why", "how", "all", "both", "each", "few", "more", "most",
                 "other", "some", "such", "no", "nor", "not", "only", "own", "same",
                 "so", "than", "too", "very", "just", "and", "but", "or", "if", "while",
                 "about", "up", "it", "its", "i", "me", "my", "you", "your", "he", "him",
                 "his", "she", "her", "we", "us", "they", "them", "their", "that", "this",
                 "these", "those", "what", "which", "who", "whom", "get", "got", "go",
                 "going", "went", "come", "came", "say", "said", "told", "telling",
                 "okay", "well", "right", "like", "know", "think", "really", "thing",
                 "things", "yeah", "yes", "let", "also", "much", "many", "one", "two"}

    active_content = Counter()
    near_content = Counter()
    for t in at:
        words = [w.lower().strip(".,;:!?()\"'`") for w in t.split() if len(w) > 2]
        content = [w for w in words if w not in stopwords and len(w) > 2]
        active_content.update(content)
    for t in nt:
        words = [w.lower().strip(".,;:!?()\"'`") for w in t.split() if len(w) > 2]
        content = [w for w in words if w not in stopwords and len(w) > 2]
        near_content.update(content)

    n_act = len(at)
    n_near = max(len(nt), 1)
    content_diffs = []
    for w, c in active_content.most_common(40):
        n_c = near_content.get(w, 0)
        # Enriched if present in >= 3 active and ratio > 2x
        if c >= 3 and (c / n_act) > (n_c / n_near) * 2:
            content_diffs.append((w, c, n_c))
    content_diffs.sort(key=lambda x: x[1] / max(x[2], 0.5), reverse=True)

    if content_diffs:
        top_words = [w for w, _, _ in content_diffs[:5]]
        candidates.append({
            "type": "lexical_content",
            "description": f"Enriched content words: {', '.join(top_words)}",
            "act_count": content_diffs[0][1],
            "near_count": content_diffs[0][2],
            "score": content_diffs[0][1] / max(content_diffs[0][2], 0.5),
            "words": top_words,
        })

    # ── Strategy 5: Length / complexity analysis ──
    act_avg_len = sum(len(t.split()) for t in at) / max(len(at), 1)
    near_avg_len = sum(len(t.split()) for t in nt) / max(len(nt), 1)
    if act_avg_len > near_avg_len * 1.3 and act_avg_len > 12:
        candidates.append({
            "type": "complexity",
            "description": f"Longer utterances (active avg {act_avg_len:.0f} words vs near-miss {near_avg_len:.0f})",
            "act_count": int(act_avg_len),
            "near_count": int(near_avg_len),
            "score": act_avg_len / max(near_avg_len, 1),
        })

    # ── Strategy 6: What near-misses have that active doesn't ──
    # This helps define exclusions
    near_only_patterns = []
    for w, c in near_content.most_common(20):
        a_c = active_content.get(w, 0)
        if c >= 2 and (c / n_near) > (a_c / n_act) * 2:
            near_only_patterns.append((w, c, a_c))
    near_only_patterns.sort(key=lambda x: x[1] / max(x[2], 0.5), reverse=True)

    # ── Strategy 7: Look at what the highest-activating samples share ──
    # that the lowest-activating active samples DON'T
    high_active = [s for s in all_active if s["tag"] == "ACTIVE_HIGH"]
    low_active = [s for s in all_active if s["tag"] in ("ACTIVE_LOW", "ACTIVE_MID")]

    if len(high_active) >= 2 and low_active:
        high_words = Counter()
        low_words = Counter()
        for s in high_active:
            high_words.update(simple_tok(s["text"]))
        for s in low_active:
            low_words.update(simple_tok(s["text"]))
        # Words in high but not low
        high_specific = [(w, c) for w, c in high_words.most_common(30)
                        if w not in stopwords and len(w) > 2 and c >= 2
                        and low_words.get(w, 0) < c]
        if high_specific:
            candidates.append({
                "type": "activation_gradient",
                "description": f"High-activation specific words: {', '.join(w for w, _ in high_specific[:4])}",
                "act_count": high_specific[0][1],
                "near_count": 0,
                "score": high_specific[0][1] * 2,
            })

    # Sort candidates by score
    candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    return candidates, frame_scores, content_diffs, near_only_patterns, opener_diffs, distinguishing_phrases


def build_explanation(latent_idx, candidates, frame_scores, content_diffs, near_only_patterns,
                      opener_diffs, distinguishing_phrases, all_active, near_misses):
    """Build the full explanation from the analysis results."""

    at = [s["text"] for s in all_active]
    nt = [s["text"] for s in near_misses] if near_misses else []

    act_avg_len = sum(len(t.split()) for t in at) / max(len(at), 1)
    near_avg_len = sum(len(t.split()) for t in nt) / max(len(nt), 1)

    # Key evidence: highest active, 1 mid active, lowest active, highest near-miss
    act_sorted = sorted(all_active, key=lambda s: s["activation"], reverse=True)
    low_sorted = sorted(all_active, key=lambda s: s["activation"])
    near_sorted = sorted(near_misses, key=lambda s: s["activation"], reverse=True) if near_misses else []

    evidence_ids = []
    if act_sorted:
        evidence_ids.append(act_sorted[0]["id"])
    if len(act_sorted) >= 2:
        evidence_ids.append(act_sorted[1]["id"])
    if low_sorted and low_sorted[0]["id"] not in evidence_ids:
        evidence_ids.append(low_sorted[0]["id"])
    if near_sorted:
        evidence_ids.append(near_sorted[0]["id"])

    # Activation stats
    acts_active = [s["activation"] for s in all_active]
    acts_near = [s["activation"] for s in near_misses if s["activation"] > 0] if near_misses else []
    min_active = min(acts_active) if acts_active else 0
    max_near = max(acts_near) if acts_near else 0
    separation = min_active / max(max_near, 0.001) if max_near > 0 else float("inf")

    positive_triggers = []
    explicit_exclusions = []
    surface_confounds = []
    alt_hypotheses = []
    failure_modes = []

    # ── Build hypothesis from top candidates ──
    # Use the top 2-3 candidates to build a composite hypothesis

    primary = candidates[0] if candidates else None
    secondary = candidates[1] if len(candidates) > 1 else None
    tertiary = candidates[2] if len(candidates) > 2 else None

    # ── Determine short_name and feature_type first ──
    short_name = "latent-feature"
    feature_type = "mixed"

    if primary:
        if primary["type"] == "phrase_pattern":
            phrase = primary.get("phrase", "")
            # Classify the phrase
            if any(w in phrase for w in ["sounds like", "seems like", "looks like", "feels like"]):
                short_name = "reflective-framing"
                feature_type = "discourse-act"
            elif any(w in phrase for w in ["you need", "you should", "it's important", "make sure"]):
                short_name = "directive-statement"
                feature_type = "pragmatic"
            elif any(w in phrase for w in ["how often", "how long", "tell me", "what do"]):
                short_name = "elicitation-question"
                feature_type = "pragmatic"
            elif any(w in phrase for w in ["i want", "i need", "i think", "i know", "i hear"]):
                short_name = "counselor-stance"
                feature_type = "pragmatic"
            elif any(w in phrase for w in ["your", "you're", "you are"]):
                short_name = "client-state-reference"
                feature_type = "semantic"
            else:
                # Use the phrase itself
                words = phrase.split()
                if len(words) >= 3:
                    short_name = f"phrase-{words[0]}-{words[1]}-{words[2]}"
                else:
                    short_name = f"phrase-{phrase.replace(' ', '-')}"
                feature_type = "lexical"

        elif primary["type"] == "semantic_frame":
            frame = primary.get("frame", "")
            if "reflective" in frame:
                short_name = frame.replace("reflective_", "reflect-")
                feature_type = "discourse-act"
            elif "question" in frame:
                short_name = frame.replace("question_", "q-")
                feature_type = "pragmatic"
            elif "directive" in frame:
                short_name = frame.replace("directive_", "dir-")
                feature_type = "pragmatic"
            elif "affirmation" in frame:
                short_name = "brief-affirmation"
                feature_type = "pragmatic"
            else:
                short_name = frame.replace("_", "-")
                feature_type = "pragmatic"

        elif primary["type"] == "opener_pattern":
            desc = primary.get("description", "")
            # Extract the opener words
            openers = re.findall(r"'(\w+)'", desc)
            if openers:
                short_name = f"opener-{openers[0]}"
            else:
                short_name = "discourse-opener"
            feature_type = "structural"

        elif primary["type"] == "lexical_content":
            words = primary.get("words", [])
            if words:
                short_name = f"content-{words[0]}"
            feature_type = "semantic"

        elif primary["type"] == "complexity":
            short_name = "complex-utterance"
            feature_type = "structural"

    # ── Build positive triggers ──

    if primary:
        if primary["type"] == "phrase_pattern":
            phrase = primary.get("phrase", "")
            act_c = primary.get("act_count", 0)
            near_c = primary.get("near_count", 0)
            positive_triggers.append(
                f"Presence of the phrase pattern '{phrase}' (found in {act_c}/{len(at)} active samples, {near_c}/{len(nt)} near-miss samples)"
            )
            # Add secondary phrase if available
            top_phrases = primary.get("all_top_phrases", [])
            if len(top_phrases) >= 2:
                p2, c2, n2, _ = top_phrases[1]
                if c2 >= 2 and n2 <= 1:
                    positive_triggers.append(
                        f"Related pattern '{p2}' (active {c2}/{len(at)}, near-miss {n2}/{len(nt)})"
                    )

        elif primary["type"] == "semantic_frame":
            positive_triggers.append(
                f"Utterances matching the pattern: {primary['description']} "
                f"(active {primary['act_count']}/{len(at)}, near-miss {primary['near_count']}/{len(nt)})"
            )

        elif primary["type"] == "opener_pattern":
            positive_triggers.append(
                f"Utterances beginning with: {primary['description']}"
            )

        elif primary["type"] == "lexical_content":
            words = primary.get("words", [])
            positive_triggers.append(
                f"Utterances containing specific content words: {', '.join(words[:4])}"
            )

        elif primary["type"] == "complexity":
            positive_triggers.append(
                f"Longer, more elaborated counselor utterances (avg {act_avg_len:.0f} words vs {near_avg_len:.0f} for near-misses)"
            )

    # Add secondary trigger
    if secondary:
        if secondary["type"] == "semantic_frame" and (not primary or primary["type"] != "semantic_frame"):
            positive_triggers.append(
                f"Also involves: {secondary['description']} (active {secondary['act_count']}/{len(at)})"
            )
        elif secondary["type"] == "phrase_pattern" and (not primary or primary["type"] != "phrase_pattern"):
            phrase2 = secondary.get("phrase", "")
            positive_triggers.append(
                f"Additionally characterized by: '{phrase2}'"
            )
        elif secondary["type"] == "lexical_content" and (not primary or primary["type"] != "lexical_content"):
            words2 = secondary.get("words", [])
            positive_triggers.append(
                f"Enriched content vocabulary: {', '.join(words2[:3])}"
            )

    # ── Build explicit exclusions ──
    # Based on near-miss patterns

    if near_only_patterns:
        top_near_words = [w for w, _, _ in near_only_patterns[:3]]
        explicit_exclusions.append(
            f"Utterances enriched in near-miss-specific tokens: {', '.join(top_near_words)} "
            f"-- these words mark the non-triggering samples"
        )

    # Check if near-misses are mostly short affirmations
    near_affirm = sum(1 for t in nt if len(t.split()) <= 6 and any(
        t.lower().startswith(w) for w in ["good", "great", "okay", "well", "yes", "all right", "excellent"]
    ))
    if near_affirm >= 2:
        explicit_exclusions.append(
            f"Brief affirmations or acknowledgments ({near_affirm} near-miss samples are short positive statements)"
        )

    # Check if near-misses lack the primary pattern
    if primary and primary["type"] == "phrase_pattern":
        phrase = primary.get("phrase", "")
        explicit_exclusions.append(
            f"Utterances that lack the phrase '{phrase}' despite similar topic or tone"
        )
    elif primary and primary["type"] == "semantic_frame":
        explicit_exclusions.append(
            f"Utterances that do not match the {primary['description'].lower()} pattern"
        )

    # ── Surface confounds ──
    if act_avg_len > near_avg_len * 1.2:
        surface_confounds.append(
            f"Active samples are longer on average ({act_avg_len:.0f} vs {near_avg_len:.0f} words), "
            "which may partly drive activation"
        )

    # Check if opener pattern is just a length proxy
    if primary and primary["type"] == "opener_pattern":
        surface_confounds.append(
            "The discourse marker pattern may correlate with utterance length or turn position rather than being the true trigger"
        )

    if near_sorted and max_near > min_active * 0.5:
        surface_confounds.append(
            f"Some near-miss activations ({max_near:.2f}) overlap with active range ({min_active:.2f}-{max(acts_active):.2f}), "
            "suggesting the boundary is soft"
        )

    if content_diffs:
        top_content = [w for w, _, _ in content_diffs[:3]]
        surface_confounds.append(
            f"Content words like {', '.join(top_content)} may co-occur with the true trigger rather than being the trigger itself"
        )

    # ── Alternative hypotheses ──
    alt_hypotheses.append(
        "The latent may track utterance length or syntactic complexity rather than a specific semantic or pragmatic feature"
    )

    # Add a domain-specific alternative if we have content words
    if content_diffs:
        top_words = [w for w, _, _ in content_diffs[:3]]
        alt_hypotheses.append(
            f"Activation could be driven by topical content ({', '.join(top_words)}) rather than the speech-act pattern"
        )

    # Add frame-based alternative
    if frame_scores:
        second_frame = frame_scores[1] if len(frame_scores) > 1 else None
        if second_frame:
            alt_hypotheses.append(
                f"An alternative trigger could be: {second_frame[1]} "
                f"(active {second_frame[2]}/{len(at)}, near-miss {second_frame[3]}/{len(nt)})"
            )

    # ── Failure modes ──
    failure_modes.append(
        "Small sample size (20 utterances per task) limits the reliability of any pattern-based hypothesis"
    )

    if separation < 2.0:
        failure_modes.append(
            f"Activation overlap between active (min={min_active:.2f}) and near-miss (max={max_near:.2f}) "
            "makes the decision boundary uncertain"
        )

    if act_avg_len > near_avg_len * 1.3:
        failure_modes.append(
            "Length confound: active samples are substantially longer, making it hard to isolate semantic vs. structural triggers"
        )

    # Check if both repeats of this latent produce very different patterns
    if primary and primary.get("score", 0) < 3:
        failure_modes.append(
            "The distinguishing signal is weak (low contrast score); the hypothesis is provisional"
        )

    # ── Confidence ──
    if primary and primary.get("score", 0) > 10:
        confidence = 0.75
    elif primary and primary.get("score", 0) > 5:
        confidence = 0.65
    elif primary and primary.get("score", 0) > 3:
        confidence = 0.55
    else:
        confidence = 0.45

    if separation > 5:
        confidence = min(confidence + 0.1, 0.85)
    elif separation < 1.5:
        confidence = max(confidence - 0.1, 0.3)

    # ── Build main hypothesis ──
    if primary:
        if primary["type"] == "phrase_pattern":
            phrase = primary.get("phrase", "")
            main_hyp = (
                f"This latent activates when the counselor's utterance contains the phrase pattern '{phrase}' "
                f"or close variants. This pattern appears in {primary['act_count']}/{len(at)} active samples "
                f"but only {primary['near_count']}/{len(nt)} near-miss samples. "
                f"Near-miss samples that look similar but lack this specific phrasing do not trigger the latent."
            )
        elif primary["type"] == "semantic_frame":
            main_hyp = (
                f"This latent activates for counselor utterances that match the {primary['description'].lower()} pattern. "
                f"This frame is present in {primary['act_count']}/{len(at)} active samples versus "
                f"{primary['near_count']}/{len(nt)} near-miss samples. "
                "The trigger appears to be the specific pragmatic structure rather than topical content."
            )
        elif primary["type"] == "opener_pattern":
            main_hyp = (
                f"This latent activates for counselor utterances beginning with specific discourse markers. "
                f"{primary['description']}. "
                "The trigger appears to be the utterance-initial discourse position rather than the content that follows."
            )
        elif primary["type"] == "lexical_content":
            words = primary.get("words", [])
            main_hyp = (
                f"This latent activates when the counselor uses specific content vocabulary: {', '.join(words[:4])}. "
                f"These words are significantly enriched in active versus near-miss samples. "
                "The trigger appears to be the presence of this lexical cluster rather than a structural pattern."
            )
        elif primary["type"] == "complexity":
            main_hyp = (
                f"This latent activates for longer, more elaborated counselor utterances "
                f"(avg {act_avg_len:.0f} words vs {near_avg_len:.0f} for near-misses). "
                "The trigger appears to be utterance complexity or length rather than a specific semantic pattern."
            )
        else:
            main_hyp = (
                f"This latent activates for a specific pattern in counselor utterances: {primary['description']}."
            )
    else:
        main_hyp = (
            f"Latent {latent_idx} activates for a narrow pattern in counselor utterances. "
            "Insufficient contrastive signal was found to formulate a specific hypothesis."
        )

    # Ensure non-empty arrays
    if not positive_triggers:
        positive_triggers.append("A specific pattern in counselor utterance structure that distinguishes active from near-miss samples")
    if not explicit_exclusions:
        explicit_exclusions.append("Utterances that share surface vocabulary with active samples but differ in pragmatic function")
    if not surface_confounds:
        surface_confounds.append("Possible confounding with utterance length or topic")
    if not alt_hypotheses:
        alt_hypotheses.append("The latent may track a different feature than the one hypothesized")

    return {
        "short_name": short_name,
        "main_hypothesis": main_hyp,
        "positive_triggers": positive_triggers,
        "explicit_exclusions": explicit_exclusions,
        "surface_confounds": surface_confounds,
        "alt_hypotheses": alt_hypotheses,
        "failure_modes": failure_modes,
        "feature_type": feature_type,
        "confidence": confidence,
        "key_evidence": evidence_ids,
    }


# ── Main processing ─────────────────────────────────────────────────────────

def process_task(task: dict) -> dict:
    """Process a single task: parse samples, analyze, construct hypothesis."""
    latent_idx = task["latent_idx"]
    prompt = task["prompt"]
    samples = parse_samples_from_prompt(prompt)

    all_active = [s for s in samples if s["tag"].startswith("ACTIVE")]
    all_nonactive = [s for s in samples if s["tag"].startswith("NONACTIVE")]
    near_misses = [s for s in samples if s["tag"] == "NONACTIVE_NEAR_MISS"]

    candidates, frame_scores, content_diffs, near_only_patterns, opener_diffs, distinguishing_phrases = \
        find_contrastive_features(all_active, all_nonactive, near_misses)

    result = build_explanation(
        latent_idx, candidates, frame_scores, content_diffs, near_only_patterns,
        opener_diffs, distinguishing_phrases, all_active, near_misses
    )

    return {
        "latent_idx": latent_idx,
        "short_name": result["short_name"],
        "main_hypothesis": result["main_hypothesis"],
        "positive_triggers": result["positive_triggers"],
        "explicit_exclusions": result["explicit_exclusions"],
        "possible_surface_confounds": result["surface_confounds"],
        "feature_type": result["feature_type"],
        "confidence": result["confidence"],
        "alternative_hypotheses": result["alt_hypotheses"],
        "key_evidence": result["key_evidence"],
        "failure_modes": result["failure_modes"],
    }


def main():
    task_path = os.path.join(BASE_DIR, TASK_FILE)
    manifest_path = os.path.join(BASE_DIR, MANIFEST_FILE)

    with open(task_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    # ctli_0092-0136: lines 181-268 (1-indexed) = [180:268] (0-indexed)
    task_lines = all_lines[180:268]
    print(f"Total lines: {len(all_lines)}")
    print(f"Processing ctli_0092-0136 (lines 181-268, {len(task_lines)} tasks)")

    processed = 0
    skipped = 0
    errors = 0

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    for i, line in enumerate(task_lines):
        line = line.strip()
        if not line:
            continue

        try:
            task = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"  ERROR parsing line {181 + i}: {e}")
            errors += 1
            continue

        task_id = task.get("task_id", "unknown")
        latent_idx = task.get("latent_idx", -1)
        expected_output_path = task.get("expected_output_path", "")
        output_path = os.path.join(BASE_DIR, expected_output_path.replace("\\", "/"))

        if os.path.exists(output_path):
            print(f"  SKIP {task_id}: already exists")
            skipped += 1
            continue

        print(f"  [{181+i}] Processing {task_id} (latent {latent_idx})...")

        try:
            explanation = process_task(task)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            errors += 1
            continue

        # Validate
        required = ["latent_idx", "short_name", "main_hypothesis", "positive_triggers",
                     "explicit_exclusions", "possible_surface_confounds", "feature_type",
                     "confidence", "alternative_hypotheses", "key_evidence", "failure_modes"]
        for field in required:
            if field not in explanation:
                if field in ("positive_triggers", "explicit_exclusions", "possible_surface_confounds",
                             "alternative_hypotheses", "key_evidence", "failure_modes"):
                    explanation[field] = []
                elif field == "confidence":
                    explanation[field] = 0.5
                else:
                    explanation[field] = "unknown"

        for field in ["positive_triggers", "explicit_exclusions", "possible_surface_confounds",
                       "alternative_hypotheses", "key_evidence", "failure_modes"]:
            if not isinstance(explanation[field], list):
                explanation[field] = [explanation[field]]

        # Write output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_json = json.dumps(explanation, ensure_ascii=False, indent=2)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_json)

        # Manifest
        prompt_text = task.get("prompt", "")
        prompt_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        raw_output_sha = hashlib.sha256(output_json.encode("utf-8")).hexdigest()
        manifest_entry = {
            "task_id": task_id,
            "model": "claude-opus-4-8-20250918",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_sha256": prompt_sha,
            "raw_output_sha256": raw_output_sha,
        }
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")

        processed += 1
        print(f"    -> {explanation['short_name']} | conf={explanation['confidence']} | triggers={len(explanation['positive_triggers'])}")

    print(f"\nDone: {processed} processed, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
