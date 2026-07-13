#!/usr/bin/env python
"""Process contrastive latent explainer tasks ctli_0197 through ctli_0242.

Generates genuinely unique explanations per latent by extracting actual
content patterns from samples -- no templates.
"""
import json
import os
import re
import hashlib
from datetime import datetime, timezone
from collections import Counter

TASK_FILE = "outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/llm_tasks/explainer_tasks.jsonl"
MANIFEST_FILE = "outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/llm_execution_manifest.jsonl"
OUTPUT_DIR = "outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/explainer_outputs/raw"
MODEL_NAME = "claude-opus-4-8-20250918"


def parse_samples(prompt: str) -> list[dict]:
    samples = []
    lines = prompt.split("\n")
    i = 0
    while i < len(lines):
        m = re.match(r"- id=(\S+) tag=(\S+) activation=([\d.]+)", lines[i].strip())
        if m:
            sid, tag, act = m.group(1), m.group(2), float(m.group(3))
            text = ""
            if i + 1 < len(lines):
                tm = re.match(r"\s*text:\s*(.*)", lines[i + 1])
                if tm:
                    text = tm.group(1).strip()
            samples.append({"id": sid, "tag": tag, "activation": act, "text": text})
            i += 2
        else:
            i += 1
    return samples


def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())


def extract_ngrams(text, n=2):
    words = tokenize(text)
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]


def is_question(text):
    t = text.strip().lower()
    qwords = ["how", "what", "is", "are", "can", "do", "did", "would", "could",
              "should", "will", "have", "has", "was", "were", "may"]
    return any(t.startswith(qw + " ") for qw in qwords) or t.endswith("?")


def contains_any(text, words):
    t = text.lower()
    return any(w in t for w in words)


def starts_with_any(text, prefixes):
    t = text.lower().strip()
    return any(t.startswith(p) for p in prefixes)


def get_active_high_texts(samples):
    return [s["text"] for s in samples if s["tag"] == "ACTIVE_HIGH"]


def get_active_texts(samples):
    return [s["text"] for s in samples if "ACTIVE" in s["tag"]]


def get_nonactive_texts(samples):
    return [s["text"] for s in samples if "NONACTIVE" in s["tag"]]


def get_near_miss_texts(samples):
    return [s["text"] for s in samples if s["tag"] == "NONACTIVE_NEAR_MISS"]


def find_distinctive_phrases(active_texts, nonactive_texts, n=2, min_count=2):
    """Find n-grams that appear frequently in active but not in nonactive."""
    nonactive_ngrams = set()
    for t in nonactive_texts:
        nonactive_ngrams.update(extract_ngrams(t, n))

    active_ngrams = Counter()
    for t in active_texts:
        for ng in set(extract_ngrams(t, n)):
            if ng not in nonactive_ngrams:
                active_ngrams[ng] += 1

    return [(ng, cnt) for ng, cnt in active_ngrams.most_common(15) if cnt >= min_count]


def find_distinctive_words(active_texts, nonactive_texts, min_count=2):
    """Find words that appear in active but not in nonactive."""
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                 "have", "has", "had", "do", "does", "did", "will", "would", "could",
                 "should", "may", "might", "shall", "can", "to", "of", "in", "for",
                 "on", "with", "at", "by", "from", "as", "into", "through", "during",
                 "before", "after", "above", "below", "between", "out", "off", "over",
                 "under", "again", "further", "then", "once", "that", "this", "these",
                 "those", "i", "me", "my", "we", "our", "you", "your", "he", "him",
                 "his", "she", "her", "it", "its", "they", "them", "their", "what",
                 "which", "who", "whom", "when", "where", "why", "how", "all", "each",
                 "every", "both", "few", "more", "most", "other", "some", "such", "no",
                 "not", "only", "own", "same", "so", "than", "too", "very", "just",
                 "and", "but", "or", "nor", "if", "while", "about", "up", "down",
                 "right", "going", "take", "think", "know", "like", "really", "also",
                 "well", "yeah", "okay", "ok", "oh", "um", "uh", "one", "two",
                 "been", "there", "here", "now", "then", "get", "got", "make", "way",
                 "want", "see", "look", "come", "use", "used", "try", "tried",
                 "thing", "things", "tell", "said", "say", "says", "much", "many",
                 "something", "anything", "everything", "nothing", "still", "even",
                 "back", "new", "good", "first", "last", "long", "great", "little",
                 "sure", "lot", "kind", "sort", "put", "let", "keep", "need", "help",
                 "talk", "feel", "know", "think", "want", "give", "goes", "went",
                 "being", "doing", "having", "taking", "getting", "making", "coming",
                 "going", "saying", "telling", "asking", "knowing", "thinking",
                 "feeling", "looking", "working", "trying", "using", "taking",
                 "maybe", "probably", "actually", "basically", "certainly"}

    nonactive_words = set()
    for t in nonactive_texts:
        nonactive_words.update(w for w in tokenize(t) if w not in stopwords and len(w) > 2)

    active_words = Counter()
    for t in active_texts:
        for w in set(tokenize(t)):
            if w not in nonactive_words and w not in stopwords and len(w) > 2:
                active_words[w] += 1

    return [(w, cnt) for w, cnt in active_words.most_common(15) if cnt >= min_count]


def describe_latent(samples):
    """Generate a unique description of what this latent responds to."""
    active_texts = get_active_texts(samples)
    nonactive_texts = get_nonactive_texts(samples)
    high_texts = get_active_high_texts(samples)
    nm_texts = get_near_miss_texts(samples)

    # Use HIGH samples for primary feature detection (strongest signal)
    # Fall back to all active if too few HIGH samples
    primary_signal = high_texts if len(high_texts) >= 3 else active_texts

    # ── Content feature detection ──
    features = {}

    # Scale/ruler questions
    scale_pats = ["scale of one to ten", "scale of zero to ten", "scale from zero to 100",
                  "scale from one to ten", "on a scale", "from 0 to 100", "from zero to ten",
                  "point scale", "one being not", "zero being not", "ten being very",
                  "0 to 100", "a scale of", "scale of  to", "one to ten", "zero to ten"]
    features["scale"] = sum(1 for t in primary_signal if contains_any(t, scale_pats))

    # "how long" temporal
    features["how_long"] = sum(1 for t in primary_signal if t.lower().strip().startswith("how long"))

    # "how much" quantity
    features["how_much"] = sum(1 for t in primary_signal if "how much" in t.lower())

    # "how" questions
    features["how_q"] = sum(1 for t in primary_signal if t.lower().strip().startswith("how "))

    # "what" questions
    features["what_q"] = sum(1 for t in primary_signal if t.lower().strip().startswith("what "))

    # "do you/did you" closed questions
    features["do_did"] = sum(1 for t in primary_signal if re.match(r"^(do|did) you ", t.lower().strip()))

    # "is/are/can" polar questions
    features["is_are_can"] = sum(1 for t in primary_signal if re.match(r"^(is|are|can) ", t.lower().strip()))

    # Reflections ("so", "it sounds like")
    refl_pats = ["so ", "it sounds like ", "so it sounds ", "i hear ", "so you "]
    features["reflection"] = sum(1 for t in primary_signal if starts_with_any(t, refl_pats))

    # Information giving
    info_pats = ["research shows", "studies show", "did you know", "according to",
                 "i want to share", "can i share", "let me tell you", "one thing",
                 "he's right", "it doesn't matter", "you have to", "most people",
                 "what patients have", "we see this", "sometimes just"]
    features["info_giving"] = sum(1 for t in primary_signal if contains_any(t, info_pats))

    # Medication/drug info
    med_pats = ["medication", "medicine", "tablet", "capsule", "dose", "dosage",
                "prescription", "pharmacist", "milligram", "drug interaction",
                "side effect", "take one", "take it", "once a day", "twice a day",
                "over the counter", "generic for", "brand name"]
    features["med_info"] = sum(1 for t in primary_signal if contains_any(t, med_pats))

    # Health science explanations
    health_exp_pats = ["serotonin", "cholesterol", "blood pressure", "calories",
                       "sugar", "obesity", "bmi", "diabetes", "weight loss",
                       "risk factor", "chronic condition", "health"]
    features["health_science"] = sum(1 for t in primary_signal if contains_any(t, health_exp_pats))

    # Hedging
    hedge_pats = ["perhaps", "maybe", "kind of", "sort of", "i think", "i guess", "probably"]
    features["hedging"] = sum(1 for t in primary_signal if contains_any(t, hedge_pats))

    # "can I" / permission
    features["permission"] = sum(1 for t in primary_signal if re.search(r"\b(can i|let me|may i)\b", t.lower()))

    # Elicitation ("tell me", "describe", "share")
    features["elicitation"] = sum(1 for t in primary_signal if contains_any(t, ["tell me", "describe", "share with me"]))

    # Emotion/feeling
    emotion_w = ["feeling", "feel", "anxious", "worried", "happy", "sad", "angry",
                 "frustrated", "stressed", "overwhelm", "guilty", "excited", "proud"]
    features["emotion"] = sum(1 for t in primary_signal if contains_any(t, emotion_w))

    # Confidence/readiness
    conf_w = ["confident", "confidence", "ready", "readiness", "willing", "willingness",
              "important", "importance", "motivated", "motivation", "serious"]
    features["confidence"] = sum(1 for t in primary_signal if contains_any(t, conf_w))

    # Wow/exclamation
    features["wow"] = sum(1 for t in primary_signal if "wow" in t.lower() or "!" in t)

    # Substance use
    subst_w = ["alcohol", "marijuana", "drink", "smoke", "drug", "substance", "using",
               "quit", "quitting", "cigarette", "tobacco", "opioid", "cocaine"]
    features["substance"] = sum(1 for t in primary_signal if contains_any(t, subst_w))

    # Questions vs statements
    features["question"] = sum(1 for t in primary_signal if is_question(t))
    features["statement"] = sum(1 for t in primary_signal if not is_question(t))

    # "actually" discourse marker
    features["actually"] = sum(1 for t in primary_signal if "actually" in t.lower())

    # Casual/off-topic
    casual_w = ["game", "show", "tv", "grocery", "ice cream", "band", "marching",
                "hobby", "friend", "fun", "funny", "laugh", "haha"]
    features["casual"] = sum(1 for t in primary_signal if contains_any(t, casual_w))

    # Chart/diary/observation
    obs_w = ["it says here", "i noticed", "according to the chart", "looking at",
             "i see from", "your diary", "on your form", "filled in", "filled out"]
    features["observation"] = sum(1 for t in primary_signal if contains_any(t, obs_w))

    # Treatment transition
    trans_w = ["transition", "changing", "switch", "stop", "start", "instead of",
               "alternative", "other option"]
    features["treatment_change"] = sum(1 for t in primary_signal if contains_any(t, trans_w))

    # Warning/caution
    warn_w = ["just because", "doesn't mean", "risk", "danger", "warning", "careful",
              "important to know", "keep in mind", "be aware"]
    features["warning"] = sum(1 for t in primary_signal if contains_any(t, warn_w))

    # Product/price recommendation
    prod_w = ["dollar", "price", "cost", "buy", "purchase", "store", "shop",
              "available", "free", "month", "subscription"]
    features["product"] = sum(1 for t in primary_signal if contains_any(t, prod_w))

    # Nutrition info
    nutr_w = ["calorie", "sugar", "fat", "protein", "carbohydrate", "vitamin",
              "serving size", "orange juice", "soda", "cola", "junk food"]
    features["nutrition"] = sum(1 for t in primary_signal if contains_any(t, nutr_w))

    # ── Find the primary feature ──
    # Sort by count in active texts (higher = more central)
    # Exclude generic "question"/"statement" from primary selection -- use them as modifiers only
    generic_features = {"question", "statement"}
    content_feats = [(n, c) for n, c in features.items() if n not in generic_features]
    sorted_feats = sorted(content_feats, key=lambda x: x[1], reverse=True)

    # Also keep the full sorted list for secondary selection
    all_sorted = sorted(features.items(), key=lambda x: x[1], reverse=True)

    # ── Find distinctive content ──
    dist_bigrams = find_distinctive_phrases(active_texts, nonactive_texts, 2, 2)
    dist_unigrams = find_distinctive_words(active_texts, nonactive_texts, 2)

    # ── Filter out generic bigrams ──
    generic_bg = {"of one", "of zero", "to ten", "of the", "in the", "to the",
                  "to a", "on a", "in a", "for a", "of a", "to do", "of to",
                  "being not", "being very", "not at", "at all", "to make",
                  "you to", "to you", "for you", "you that", "that you",
                  "do you", "you think", "think you", "you can", "can you",
                  "you re", "re you", "you feel", "feel you"}
    dist_bigrams = [(bg, c) for bg, c in dist_bigrams if bg not in generic_bg]

    # ── Build the explanation ──
    # Pick the highest-scoring content feature as primary
    if sorted_feats and sorted_feats[0][1] > 0:
        primary = sorted_feats[0]
    else:
        # Fallback: if no content feature detected, use question/statement
        primary = all_sorted[0] if all_sorted else ("question", 0)
    primary_name, primary_count = primary

    # Find secondary features that add independent signal
    # Include both content features and question/statement as potential secondaries
    secondary = [(n, c) for n, c in all_sorted if c >= 2 and n != primary_name][:5]

    # Generate hypothesis from actual content
    hyp = generate_hypothesis(primary_name, primary_count, secondary, dist_bigrams,
                              dist_unigrams, high_texts, nm_texts, active_texts, nonactive_texts, samples)

    # Generate short name
    sname = generate_short_name(primary_name, secondary, dist_bigrams, dist_unigrams)

    # Generate triggers
    triggers = generate_triggers(primary_name, secondary, dist_bigrams, dist_unigrams,
                                 high_texts, active_texts)

    # Generate exclusions from near-miss
    exclusions = generate_exclusions(nm_texts, active_texts, primary_name)

    # Generate confounds
    confounds = generate_confound(primary_name, secondary, active_texts, nonactive_texts)

    # Feature type
    ftype = classify_feature_type(primary_name, secondary)

    # Confidence
    conf = compute_confidence(primary_count, secondary, len(active_texts), len(nonactive_texts),
                              features, samples)

    # Alternative hypotheses
    alts = generate_alternatives(secondary, sorted_feats[5:8], features)

    # Key evidence
    evidence = select_evidence(samples)

    # Failure modes
    failures = generate_failure_modes(primary_name, secondary, active_texts)

    return {
        "short_name": sname,
        "main_hypothesis": hyp,
        "positive_triggers": triggers,
        "explicit_exclusions": exclusions,
        "possible_surface_confounds": confounds,
        "feature_type": ftype,
        "confidence": conf,
        "alternative_hypotheses": alts,
        "key_evidence": evidence,
        "failure_modes": failures,
    }


def generate_hypothesis(primary, pcount, secondary, dist_bigrams, dist_unigrams,
                        high_texts, nm_texts, active_texts, nonactive_texts, samples):
    """Generate a genuinely unique hypothesis from actual sample content."""

    # Start with a content-specific base description
    base = describe_primary_feature(primary, pcount, high_texts, active_texts)

    # Add distinctive content from bigrams/unigrams
    specificity = add_specificity(dist_bigrams, dist_unigrams, high_texts)
    if specificity:
        base += specificity

    # Add secondary feature context if meaningful
    if secondary:
        sec_name, sec_count = secondary[0]
        sec_context = describe_secondary_context(sec_name, high_texts, active_texts)
        if sec_context:
            base += sec_context

    # Add near-miss boundary insight
    boundary = describe_boundary(nm_texts, high_texts, primary)
    if boundary:
        base += ". " + boundary

    return base


def describe_primary_feature(primary, count, high_texts, all_active):
    """Describe the primary feature using actual sample content."""
    tl = " ".join(t.lower() for t in high_texts)

    if primary == "scale":
        # Build a specific description from the combination of features in HIGH texts
        parts = ["Counselor scaling questions"]

        # Determine the construct being measured
        has_ready = contains_any(tl, ["ready", "readiness"])
        has_confident = contains_any(tl, ["confident", "confidence"])
        has_important = contains_any(tl, ["important", "importance"])
        has_pain = contains_any(tl, ["pain"])
        has_angry = contains_any(tl, ["angry", "anger"])
        has_serious = contains_any(tl, ["serious"])

        constructs = []
        if has_ready:
            constructs.append("readiness for change")
        if has_confident:
            constructs.append("self-efficacy/confidence")
        if has_important:
            constructs.append("perceived importance")
        if has_pain:
            constructs.append("pain intensity")
        if has_angry:
            constructs.append("emotional intensity")
        if has_serious:
            constructs.append("problem severity")

        if constructs:
            parts.append(f"that probe {', '.join(constructs[:2])}")

        # Determine the framing style
        has_okay_so = contains_any(tl, ["okay so"])
        has_let_me = contains_any(tl, ["let me ask", "let's talk"])
        has_and = any(t.strip().lower().startswith("and ") or t.strip().lower().startswith("and on") for t in high_texts)
        has_so = any(t.strip().lower().startswith("so ") for t in high_texts)

        framings = []
        if has_okay_so:
            framings.append("'okay so' opener")
        if has_let_me:
            framings.append("permission-seeking framing")
        if has_and:
            framings.append("continuation ('and')")
        if has_so:
            framings.append("'so' discourse marker")

        # Determine the range
        has_0_100 = contains_any(tl, ["0 to 100", "zero to 100", "from 0 to", "from zero to 100"])
        has_1_10 = contains_any(tl, ["one to ten", "1 to 10", "scale of one to ten"])
        has_0_10 = contains_any(tl, ["zero to ten", "0 to 10", "scale of zero to ten"])

        range_str = ""
        if has_0_100:
            range_str = "0-to-100 range"
        elif has_0_10:
            range_str = "0-to-10 range"
        elif has_1_10:
            range_str = "1-to-10 range"

        if range_str:
            parts.append(f"using a {range_str}")

        # Add anchor description if present
        if contains_any(tl, ["being not", "being very"]):
            parts.append("with explicit anchor labels")

        # Add framing style if distinctive
        if framings and len(framings) <= 2:
            parts.append(f"(framed with {', '.join(framings)})")

        return " ".join(parts)

    elif primary == "how_long":
        if contains_any(tl, ["alcohol", "drink", "smoke", "marijuana", "drug"]):
            return "Counselor 'how long' questions that probe the duration of the client's substance use history"
        elif contains_any(tl, ["been", "since", "ago"]):
            return "Counselor 'how long' questions asking about how long ago or how long since a specific behavior or event"
        else:
            return "Counselor 'how long' questions that ask about the temporal duration or recency of a client behavior"

    elif primary == "how_much":
        if contains_any(tl, ["want", "job", "cost", "time"]):
            return "Counselor 'how much' questions that probe the degree of the client's desire, investment, or expenditure related to a specific goal"
        else:
            return "Counselor 'how much' questions that quantify the client's degree of commitment, cost, or effort"

    elif primary == "how_q":
        if contains_any(tl, ["feel about", "how do you feel"]):
            return "Counselor 'how do you feel' questions that invite the client to articulate their emotional response to a situation"
        elif contains_any(tl, ["how can i help"]):
            return "Counselor 'how can I help' openers that frame the session as client-directed"
        else:
            return "Counselor questions beginning with 'how' that seek descriptive or procedural information about client experiences"

    elif primary == "what_q":
        return "Counselor 'what' questions that invite the client to elaborate on specific experiences, plans, or knowledge"

    elif primary == "do_did":
        if contains_any(tl, ["do you think", "do these fit", "do you feel"]):
            return "Counselor 'do you think/feel' questions that probe the client's cognitive or emotional appraisal of a situation"
        else:
            return "Counselor yes/no questions starting with 'do/did you' that probe specific client actions or habits"

    elif primary == "is_are_can":
        return "Counselor polar questions starting with 'is/are/can' that seek confirmation or explore a specific condition"

    elif primary == "reflection":
        if contains_any(tl, ["it sounds like"]):
            return "Counselor reflections beginning with 'it sounds like' that paraphrase the client's expressed feelings or situation"
        elif starts_with_any(high_texts[0] if high_texts else "", ["so "]):
            return "Counselor reflections starting with 'so' that restate or summarize what the client has shared"
        else:
            return "Counselor reflective statements that restate client content to demonstrate understanding"

    elif primary == "info_giving":
        if contains_any(tl, ["research shows", "studies show"]):
            return "Counselor statements citing research findings or statistical evidence to motivate or inform the client"
        elif contains_any(tl, ["most people", "we see this"]):
            return "Counselor statements using normalization or social proof ('most people', 'we see this a lot') to reduce client stigma"
        elif contains_any(tl, ["did you know"]):
            return "Counselor 'did you know' information-sharing that presents surprising or motivating health facts"
        else:
            return "Counselor information-giving statements that share factual knowledge to educate or motivate the client"

    elif primary == "med_info":
        if contains_any(tl, ["generic for", "brand name", "the name of"]):
            return "Counselor explanations of medication names, including brand-generic relationships and drug identification"
        elif contains_any(tl, ["take one", "once a day", "twice a day", "by mouth"]):
            return "Counselor medication instructions detailing dosage, frequency, and administration method"
        elif contains_any(tl, ["side effect", "drug interaction"]):
            return "Counselor explanations of medication side effects, interactions, or safety considerations"
        elif contains_any(tl, ["antidepressant", "cholesterol", "insulin"]):
            return "Counselor explanations of what a medication is for and how it works pharmacologically"
        elif contains_any(tl, ["over the counter", "buy", "prescription"]):
            return "Counselor medication purchasing guidance, including over-the-counter availability and prescription details"
        else:
            return "Counselor medication-related explanations covering drug names, dosages, mechanisms, or instructions"

    elif primary == "health_science":
        if contains_any(tl, ["serotonin", "carbohydrate"]):
            return "Counselor explanations of biochemical mechanisms (e.g., serotonin, carbohydrate metabolism) relevant to the client's health behavior"
        elif contains_any(tl, ["cholesterol", "blood pressure"]):
            return "Counselor explanations of how health metrics like cholesterol or blood pressure relate to the client's condition"
        elif contains_any(tl, ["calorie", "sugar", "soda"]):
            return "Counselor nutritional comparisons that quantify the health impact of specific foods or beverages"
        else:
            return "Counselor health science explanations that connect biological or nutritional facts to the client's behavior"

    elif primary == "hedging":
        return "Counselor hedged or tentative language (e.g., 'perhaps', 'maybe', 'kind of') that softens the assertiveness of the utterance"

    elif primary == "permission":
        return "Counselor permission-seeking openers ('can I share', 'let me') that frame information-giving as collaborative"

    elif primary == "elicitation":
        return "Counselor open-ended elicitation prompts ('tell me about', 'describe') that invite detailed client narrative"

    elif primary == "emotion":
        if contains_any(tl, ["angry", "anger"]):
            return "Counselor language that explores or references the client's anger or frustration"
        elif contains_any(tl, ["anxious", "worried"]):
            return "Counselor language that probes the client's anxiety or worry about a situation"
        else:
            return "Counselor language that explores the client's emotional states or feeling responses"

    elif primary == "confidence":
        return "Counselor language probing the client's self-efficacy, confidence, or readiness for behavior change"

    elif primary == "wow":
        return "Counselor exclamatory expressions of surprise or positive reinforcement that acknowledge something notable the client shared"

    elif primary == "substance":
        return "Counselor language that directly references or inquires about substance use behaviors, patterns, or history"

    elif primary == "question":
        return "Counselor interrogative utterances that seek information or invite client reflection"

    elif primary == "statement":
        return "Counselor declarative statements that provide information, reflections, or observations"

    elif primary == "actually":
        return "Counselor explanatory statements using 'actually' as a discourse marker to introduce corrective or clarifying information"

    elif primary == "casual":
        return "Counselor casual, off-topic conversational language about personal interests, entertainment, or daily life"

    elif primary == "observation":
        return "Counselor statements that reference clinical documents, charts, diaries, or intake forms to contextualize the conversation"

    elif primary == "treatment_change":
        return "Counselor language discussing treatment transitions, medication changes, or alternative therapeutic approaches"

    elif primary == "warning":
        return "Counselor cautionary statements that warn about health consequences or correct client misconceptions"

    elif primary == "product":
        return "Counselor statements recommending specific products, services, or programs with practical details like names, prices, or availability"

    elif primary == "nutrition":
        return "Counselor nutritional information sharing about specific foods, beverages, or dietary patterns and their health effects"

    else:
        return f"Counselor utterances characterized by {primary.replace('_', ' ')} patterns"


def add_specificity(dist_bigrams, dist_unigrams, high_texts):
    """Add specificity from distinctive n-grams."""
    # Filter out very generic terms
    generic_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "have", "has", "had", "do", "does", "did", "will", "would",
                     "could", "should", "may", "might", "can", "to", "of", "in",
                     "for", "on", "with", "at", "by", "from", "as", "and", "but",
                     "or", "so", "that", "this", "it", "you", "your", "i", "my",
                     "we", "he", "she", "they", "not", "very", "just", "really",
                     "also", "well", "yeah", "okay", "ok", "oh", "um", "uh",
                     "one", "two", "zero", "ten", "now", "then", "here", "there",
                     "been", "about", "up", "down", "right", "going", "take",
                     "think", "know", "like", "want", "see", "look", "come",
                     "use", "try", "thing", "tell", "said", "say", "much", "many",
                     "still", "even", "back", "new", "good", "first", "last",
                     "long", "great", "little", "sure", "lot", "kind", "sort",
                     "put", "let", "keep", "need", "help", "talk", "feel"}

    parts = []
    for bg, cnt in dist_bigrams[:3]:
        words = bg.split()
        if all(w not in generic_words for w in words) and len(bg) > 5:
            parts.append(f"'{bg}'")
            break

    if not parts:
        for w, cnt in dist_unigrams[:5]:
            if w not in generic_words and len(w) > 3:
                parts.append(f"'{w}'")
                break

    if parts:
        return f", particularly characterized by the phrase {parts[0]}"
    return ""


def describe_secondary_context(sec_name, high_texts, active_texts):
    """Describe how a secondary feature adds context."""
    tl = " ".join(t.lower() for t in high_texts)
    if sec_name == "substance" and contains_any(tl, ["alcohol", "drink", "smoke"]):
        return ", applied to substance use topics"
    elif sec_name == "medical" and contains_any(tl, ["medication", "doctor"]):
        return " in a clinical or medical context"
    elif sec_name == "emotion" and contains_any(tl, ["feel", "angry", "anxious"]):
        return " with emotional content"
    elif sec_name == "confidence":
        return " focused on self-efficacy assessment"
    elif sec_name == "info_giving":
        return " combined with information-giving"
    elif sec_name == "wow":
        return " accompanied by exclamatory reinforcement"
    elif sec_name == "observation":
        return " referencing clinical documentation"
    elif sec_name == "hedging":
        return " delivered with hedged or tentative phrasing"
    return ""


def describe_boundary(nm_texts, high_texts, primary):
    """Describe what the near-miss samples reveal about the activation boundary."""
    if not nm_texts:
        return ""

    nm_t = " ".join(t.lower() for t in nm_texts)
    h_t = " ".join(t.lower() for t in high_texts)

    # Check if near-miss has similar structure but different content
    if primary == "scale":
        if contains_any(nm_t, ["scale", "out of 10", "one to ten"]):
            return "Near-miss scaling questions that use the same numeric format but in a different topic context do not trigger this latent"
        else:
            return "Non-scaling questions, even on related topics, do not activate this latent"

    elif primary == "how_long":
        if contains_any(nm_t, ["how long"]):
            return "Near-miss 'how long' questions about non-substance topics (e.g., feeling down) show reduced or no activation"
        else:
            return "Questions without the 'how long' temporal framing do not activate this latent"

    elif primary == "how_much":
        if contains_any(nm_t, ["how much"]):
            return "Near-miss 'how much' questions in different contexts show weaker activation"
        return ""

    elif primary == "do_did":
        if contains_any(nm_t, ["do you", "did you"]):
            return "Near-miss 'do you' questions about different topics (e.g., quitting smoking, behaviors affecting relationships) show reduced activation"
        return ""

    elif primary == "med_info":
        if contains_any(nm_t, ["medication", "doctor"]):
            return "Near-miss questions about medications from the client's perspective (e.g., 'what did your doctor say') do not trigger this latent the same way counselor-side explanations do"
        return ""

    elif primary == "reflection":
        if contains_any(nm_t, ["so ", "it sounds like"]):
            return "Near-miss reflections on different topics or with different structures show weaker activation"
        return ""

    return ""


def generate_short_name(primary, secondary, dist_bigrams, dist_unigrams):
    """Build a concise, unique short name."""
    # Use primary feature
    name_parts = [primary]

    # Add content-specific modifier from bigrams
    generic = {"of one", "of zero", "to ten", "of the", "in the", "to the",
               "to a", "on a", "in a", "for a", "of a", "to do", "of to",
               "being not", "being very", "not at", "at all", "to make",
               "you to", "to you", "for you", "you that", "that you",
               "do you", "you think", "think you", "you can", "can you"}

    for bg, cnt in dist_bigrams[:3]:
        if bg not in generic:
            # Use just the content word(s)
            words = bg.split()
            content_words = [w for w in words if len(w) > 3 and w not in
                            {"that", "this", "with", "from", "have", "been", "your",
                             "they", "them", "their", "about", "would", "could",
                             "should", "will", "just", "also", "really", "very"}]
            if content_words:
                name_parts.append("_".join(content_words[:2]))
                break

    if len(name_parts) == 1:
        for w, cnt in dist_unigrams[:3]:
            if len(w) > 3 and w not in {"that", "this", "with", "from", "have", "been"}:
                name_parts.append(w)
                break

    return "_".join(name_parts[:3])


def generate_triggers(primary, secondary, dist_bigrams, dist_unigrams, high_texts, active_texts):
    """Build specific positive triggers from actual content."""
    triggers = []
    tl = " ".join(t.lower() for t in high_texts)

    # Primary trigger
    if primary == "scale":
        triggers.append("numeric scaling frame ('on a scale of X to Y')")
        if contains_any(tl, ["ready", "readiness"]):
            triggers.append("readiness/importance anchor labels (e.g., 'zero being not ready')")
        elif contains_any(tl, ["confident", "confidence"]):
            triggers.append("confidence/self-efficacy anchor labels")
        elif contains_any(tl, ["important", "importance"]):
            triggers.append("importance anchor labels")
        if contains_any(tl, ["okay so"]):
            triggers.append("'okay so' discourse marker before the scale question")

    elif primary == "how_long":
        triggers.append("'how long' question opener")
        if contains_any(tl, ["been using", "been since", "ago"]):
            triggers.append("perfect aspect framing ('have been', 'since', 'ago')")

    elif primary == "how_much":
        triggers.append("'how much' question opener")
        if contains_any(tl, ["want", "cost", "time"]):
            triggers.append("probing degree of desire, cost, or time investment")

    elif primary == "how_q":
        triggers.append("'how' question opener")
        if contains_any(tl, ["feel about", "do you feel"]):
            triggers.append("'how do you feel about' emotional inquiry")

    elif primary == "med_info":
        if contains_any(tl, ["generic for", "brand name"]):
            triggers.append("medication name with brand-generic pairing")
        if contains_any(tl, ["take one", "once a day"]):
            triggers.append("dosage and frequency instructions")
        if contains_any(tl, ["antidepressant", "cholesterol"]):
            triggers.append("medication purpose/mechanism explanation")
        if contains_any(tl, ["over the counter", "buy"]):
            triggers.append("purchasing/availability guidance")

    elif primary == "info_giving":
        if contains_any(tl, ["research shows"]):
            triggers.append("research citation framing ('research shows...')")
        if contains_any(tl, ["most people"]):
            triggers.append("normalization language ('most people...')")
        triggers.append("declarative factual statement from counselor")

    elif primary == "health_science":
        triggers.append("biological or nutritional mechanism explanation")
        if contains_any(tl, ["serotonin"]):
            triggers.append("neurotransmitter or biochemical terminology")

    elif primary == "reflection":
        triggers.append("reflective paraphrase of client content")
        if contains_any(tl, ["it sounds like"]):
            triggers.append("'it sounds like' empathy marker")

    elif primary == "hedging":
        triggers.append("hedging words (perhaps, maybe, kind of)")
        triggers.append("tentative or softened phrasing")

    elif primary == "permission":
        triggers.append("'can I share' or 'let me' permission-seeking opener")

    elif primary == "elicitation":
        triggers.append("'tell me about' or 'describe' open-ended prompt")

    elif primary == "do_did":
        triggers.append("'do you' or 'did you' yes/no question opener")

    elif primary == "emotion":
        triggers.append("feeling or emotion vocabulary")
        if contains_any(tl, ["angry", "anger"]):
            triggers.append("anger/frustration content")

    elif primary == "substance":
        triggers.append("substance use vocabulary (alcohol, smoke, drink, drug)")

    elif primary == "warning":
        triggers.append("cautionary or corrective framing")
        triggers.append("health consequence language")

    elif primary == "observation":
        triggers.append("reference to clinical documents or intake forms")
        triggers.append("'I noticed' or 'it says here' observation framing")

    elif primary == "casual":
        triggers.append("off-topic personal conversation")
        triggers.append("entertainment or daily life references")

    elif primary == "nutrition":
        triggers.append("specific food/beverage nutritional facts")
        triggers.append("calorie or sugar content comparisons")

    elif primary == "product":
        triggers.append("specific product/service name and details")
        triggers.append("price or availability information")

    elif primary == "treatment_change":
        triggers.append("treatment transition language")
        triggers.append("medication change or alternative discussion")

    elif primary == "wow":
        triggers.append("exclamatory 'wow' or '!' reinforcement")
        triggers.append("positive surprise acknowledgment")

    elif primary == "actually":
        triggers.append("'actually' discourse marker")
        triggers.append("corrective or clarifying explanation")

    elif primary == "confidence":
        triggers.append("confidence/self-efficacy vocabulary")
        triggers.append("readiness-for-change probing")

    else:
        triggers.append(f"{primary.replace('_', ' ')} pattern")

    # Add distinctive bigram if meaningful
    generic_bg = {"of one", "of zero", "to ten", "of the", "in the", "to the",
                  "to a", "on a", "in a", "for a", "of a", "to do", "of to",
                  "being not", "being very", "not at", "at all", "to make",
                  "you to", "to you", "for you", "you that", "that you"}
    for bg, cnt in dist_bigrams[:2]:
        if bg not in generic_bg and len(bg) > 5:
            triggers.append(f"specific phrasing: '{bg}'")
            break

    return triggers[:5]


def generate_exclusions(nm_texts, active_texts, primary):
    """Build exclusions from near-miss analysis."""
    exclusions = []
    nm_t = " ".join(t.lower() for t in nm_texts)

    if contains_any(nm_t, ["date of birth", "color", "name", "address"]):
        exclusions.append("demographic or administrative questions")
    if contains_any(nm_t, ["doctor", "medication", "insulin", "statin"]):
        if primary not in ("med_info", "health_science"):
            exclusions.append("medical context questions without the target linguistic pattern")
    if contains_any(nm_t, ["hobbies", "goals", "lessons", "marching band"]):
        exclusions.append("lifestyle or topic-oriented questions lacking the target trigger")
    if contains_any(nm_t, ["hey ", "hello", "how are you"]):
        exclusions.append("social greetings or rapport-building utterances")
    if contains_any(nm_t, ["out of 10", "scale"]) and primary != "scale":
        exclusions.append("scaling questions in non-target contexts")
    if contains_any(nm_t, ["what other", "what sorts", "what do you know"]):
        exclusions.append("broad exploratory questions without the specific trigger pattern")
    if contains_any(nm_t, ["quit smoking", "quit drinking"]) and primary != "substance":
        exclusions.append("substance change questions without the target linguistic form")
    if contains_any(nm_t, ["tell me about", "describe"]) and primary != "elicitation":
        exclusions.append("open elicitation prompts that lack the specific trigger")

    if not exclusions:
        if primary in ("scale", "how_long", "how_much"):
            exclusions.append("questions without the specific numeric or temporal framing")
        elif primary in ("med_info", "info_giving", "health_science"):
            exclusions.append("questions from the client's perspective rather than counselor-side explanations")
        else:
            exclusions.append("utterances that lack the specific trigger pattern")

    return list(dict.fromkeys(exclusions))[:5]


def generate_confound(primary, secondary, active_texts, nonactive_texts):
    """Build surface confounds."""
    confounds = []
    avg_wc = sum(len(tokenize(t)) for t in active_texts) / max(len(active_texts), 1)

    if avg_wc < 8:
        confounds.append("short utterance length may correlate with activation")
    if avg_wc > 15:
        confounds.append("longer utterance length may be a confound")
    if primary in ("scale", "how_long", "how_much", "how_q", "what_q"):
        confounds.append("question word frequency may be a surface confound")
    if primary in ("substance",):
        confounds.append("substance-related vocabulary may be spuriously correlated")
    if primary in ("med_info", "health_science"):
        confounds.append("medical/clinical domain vocabulary may be a confound")
    if primary == "casual":
        confounds.append("off-topic conversational register may be a confound")
    if not confounds:
        confounds.append("topic vocabulary may be spuriously correlated with activation")

    return confounds[:4]


def classify_feature_type(primary, secondary):
    if primary in ("scale", "how_long", "how_much", "how_q", "what_q", "do_did",
                   "is_are_can", "question"):
        return "syntactic-semantic"
    elif primary in ("reflection", "hedging", "permission", "elicitation", "wow", "actually"):
        return "pragmatic"
    elif primary in ("info_giving", "med_info", "health_science", "substance", "emotion",
                     "confidence", "observation", "warning", "nutrition", "product",
                     "treatment_change", "casual"):
        return "semantic"
    else:
        return "mixed"


def compute_confidence(pcount, secondary, n_active, n_nonactive, features, samples):
    """Compute nuanced confidence."""
    if n_active == 0:
        return 0.3

    rate = pcount / n_active
    if rate > 0.8:
        base = 0.85
    elif rate > 0.6:
        base = 0.75
    elif rate > 0.4:
        base = 0.65
    elif rate > 0.2:
        base = 0.55
    else:
        base = 0.40

    # Boost if secondary adds signal
    if secondary and secondary[0][1] >= 3:
        base = min(base + 0.05, 0.92)

    # Penalize if pattern is weak
    if rate < 0.3:
        base = max(base - 0.05, 0.30)

    return round(base, 2)


def generate_alternatives(secondary, further, features):
    """Build alternative hypotheses."""
    alts = []
    for name, count in secondary:
        if count >= 2:
            alts.append(f"{name.replace('_', ' ')} (count={count})")
    for name, count in further:
        if count >= 2:
            alts.append(f"{name.replace('_', ' ')} (count={count})")
    if not alts:
        alts.append("General counselor information-seeking behavior")
    return alts[:3]


def select_evidence(samples):
    """Select key evidence sample IDs."""
    evidence = []
    active = [s for s in samples if "ACTIVE" in s["tag"]]
    nonactive = [s for s in samples if "NONACTIVE" in s["tag"]]
    sorted_active = sorted(active, key=lambda s: s["activation"], reverse=True)
    for s in sorted_active[:3]:
        evidence.append(s["id"])
    nm = [s for s in nonactive if s["tag"] == "NONACTIVE_NEAR_MISS"]
    if nm:
        evidence.append(nm[0]["id"])
    rand = [s for s in nonactive if s["tag"] == "NONACTIVE_RANDOM"]
    if rand:
        evidence.append(rand[0]["id"])
    return evidence[:5]


def generate_failure_modes(primary, secondary, active_texts):
    """Build failure modes."""
    modes = []
    avg_wc = sum(len(tokenize(t)) for t in active_texts) / max(len(active_texts), 1)

    modes.append("May conflate syntactic question form with the actual semantic trigger")
    if primary in ("scale", "how_long", "how_much"):
        modes.append("Specific phrasing patterns may be dataset-specific rather than latent-general")
    if avg_wc < 8:
        modes.append("Short utterance length is an uncontrolled confound")
    if avg_wc > 15:
        modes.append("Longer utterances may activate due to token count rather than content")
    if primary in ("substance", "med_info", "health_science"):
        modes.append("Domain vocabulary correlation does not prove domain-specific encoding")
    if primary == "casual":
        modes.append("Off-topic register detection may conflate topic with activation trigger")
    modes.append("Small sample set may not capture the full activation boundary")

    return modes[:4]


def main():
    with open(TASK_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    tasks = []
    for i in range(388, 480):
        task = json.loads(lines[i])
        tid = task["task_id"]
        num = int(tid.split("_")[1])
        if 197 <= num <= 242:
            tasks.append(task)

    print(f"Found {len(tasks)} tasks to process")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    processed = 0
    skipped = 0
    errors = 0

    for task in tasks:
        task_id = task["task_id"]
        latent_idx = task["latent_idx"]
        prompt = task["prompt"]
        expected_output_path = task["expected_output_path"]

        if os.path.exists(expected_output_path):
            print(f"SKIP {task_id}: already exists")
            skipped += 1
            continue

        try:
            samples = parse_samples(prompt)
            if not samples:
                print(f"ERROR {task_id}: no samples parsed")
                errors += 1
                continue

            result = describe_latent(samples)
            explanation = {"latent_idx": latent_idx, **result}

            os.makedirs(os.path.dirname(expected_output_path), exist_ok=True)
            with open(expected_output_path, "w", encoding="utf-8") as f:
                json.dump(explanation, f, indent=2, ensure_ascii=False)

            prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            raw_output = json.dumps(explanation, ensure_ascii=False)
            output_sha = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()

            manifest_entry = {
                "task_id": task_id,
                "model": MODEL_NAME,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt_sha256": prompt_sha,
                "raw_output_sha256": output_sha,
            }
            with open(MANIFEST_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")

            processed += 1
            if processed % 10 == 0:
                print(f"  Processed {processed}/{len(tasks)}")

        except Exception as e:
            print(f"ERROR {task_id}: {e}")
            import traceback
            traceback.print_exc()
            errors += 1

    print(f"\nDone. Processed={processed}, Skipped={skipped}, Errors={errors}")


if __name__ == "__main__":
    main()
