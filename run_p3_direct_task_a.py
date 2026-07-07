"""
P3 Task A Direct Execution: Generate input-centric explanations for all 303 stable_core latents.

Reads each prompt JSON, extracts the examples, applies rule-based linguistic pattern analysis,
and writes output in the batch format expected by the P3 orchestrator.
"""

import json
import re
import datetime
from pathlib import Path
from collections import Counter

BASE_DIR = Path("outputs/misc_full_sae_eval/interpretability/p3_feature_cards_stable_core")
PROMPTS_DIR = BASE_DIR / "p3_input_explanation_prompts"
BATCH_DIR = BASE_DIR / "agent_batches"
CARDS_JSONL = BASE_DIR / "p3_feature_card_packets.jsonl"


def load_cards_index():
    """Build (label, latent, rank) -> packet_id index."""
    index = {}
    with open(CARDS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                card = json.loads(line)
                key = (card["target_label"], card["latent_idx"], card["rank_within_label"])
                index[key] = card["packet_id"]
    return index


def parse_prompt_filename(filename):
    """Parse label, rank, latent from prompt filename."""
    stem = Path(filename).stem.replace("_input_prompt", "")
    parts = stem.split("_")
    label = parts[0]
    rank = int(parts[1].replace("rank", ""))
    latent = int(parts[2].replace("latent", ""))
    return label, rank, latent


def extract_prompt_data(prompt_path):
    """Extract the user payload from a prompt JSON file."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    messages = data.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return json.loads(msg["content"])
    return {}


def analyze_examples(examples):
    """Extract surface patterns from examples."""
    texts = [e.get("text", "").lower() for e in examples]
    labels = [e.get("active_labels", "") for e in examples]
    matches = [int(e.get("target_match", 0)) for e in examples]
    activations = [float(e.get("activation", 0.0)) for e in examples]
    groups = [e.get("group", "") for e in examples]

    top_texts = [t for t, g in zip(texts, groups) if g == "top_activating"]
    top_matches = [m for m, g in zip(matches, groups) if g == "top_activating"]
    top_labels = [l for l, g in zip(labels, groups) if g == "top_activating"]

    match_rate = sum(top_matches) / len(top_matches) if top_matches else 0.0

    features = {
        "has_question_mark": sum("?" in t for t in top_texts),
        "starts_with_wh": sum(t.strip().startswith(("what ", "why ", "how ", "when ", "where ", "who "))
                              for t in top_texts),
        "starts_with_do_can": sum(t.strip().startswith(("do ", "can ", "did ", "does ", "have ", "has ",
                                                         "is ", "are ", "was ", "were "))
                                   for t in top_texts),
        "sounds_like": sum("sounds like" in t or "sound like" in t for t in top_texts),
        "it_seems": sum("it seems" in t or "seems like" in t for t in top_texts),
        "i_hear": sum("i hear" in t or "i'm hearing" in t or "what i'm hearing" in t for t in top_texts),
        "on_one_hand": sum("on one hand" in t or "on the other hand" in t for t in top_texts),
        "you_feel": sum("you feel" in t or "you're feeling" in t for t in top_texts),
        "positive_eval": sum(any(w in t for w in ["great", "good", "wonderful", "awesome",
                                                    "amazing", "excellent", "fantastic", "terrific",
                                                    "nice", "proud", "appreciate"])
                             for t in top_texts),
        "i_understand": sum("i understand" in t or "totally understand" in t or
                            "i completely understand" in t for t in top_texts),
        "would_phrase": sum("would you" in t or "what would" in t or "how would" in t for t in top_texts),
        "scale_question": sum("scale" in t or "one to ten" in t or "1 to 10" in t for t in top_texts),
        "medical_info": sum(any(w in t for w in ["medication", "medicine", "cholesterol", "diabetes",
                                                   "blood pressure", "mg", "milligram", "prescri",
                                                   "doctor", "health", "symptom"])
                            for t in top_texts),
        "so_prefix": sum(t.strip().startswith(("so ", "so,")) for t in top_texts),
        "you_prefix": sum(t.strip().startswith(("you ", "you're ", "you've ")) for t in top_texts),
        "okay_prefix": sum(t.strip().startswith("ok") for t in top_texts),
        "yeah_prefix": sum(t.strip().startswith("yeah") for t in top_texts),
        "that_prefix": sum(t.strip().startswith(("that's ", "that sounds", "that is", "that "))
                           for t in top_texts),
        "long_utterance": sum(len(t.split()) > 25 for t in top_texts),
        "short_utterance": sum(len(t.split()) < 8 for t in top_texts),
    }

    # Label distribution in top examples
    all_labels_list = []
    for lbl in top_labels:
        if lbl:
            all_labels_list.extend(lbl.split(","))
    label_counter = Counter(all_labels_list)

    # Activation uniformity check
    top_acts = [a for a, g in zip(activations, groups) if g == "top_activating"]
    if len(top_acts) > 3:
        unique_acts = len(set(round(a, 3) for a in top_acts))
        activation_uniform = unique_acts <= 2 and len(top_acts) >= 6
    else:
        activation_uniform = False

    return {
        "match_rate": match_rate,
        "features": features,
        "activation_uniform": activation_uniform,
        "label_counter": label_counter,
        "n_top_examples": len(top_texts),
        "n_total_examples": len(examples),
    }


def classify_latent(target_label, latent_idx, rank, cohens_d, analysis):
    """Produce structured explanation for a latent based on its examples."""
    f = analysis["features"]
    match_rate = analysis["match_rate"]
    uniform = analysis["activation_uniform"]

    artifact_risks = []
    if uniform:
        artifact_risks.append(
            f"Activation values appear highly uniform across top examples "
            f"(possible max-pooling or quantization artifact for latent {latent_idx})."
        )

    artifact_risk_level = "low"
    if match_rate < 0.2:
        artifact_risk_level = "high"
        artifact_risks.append("Target match rate below 0.20.")
    elif match_rate < 0.4:
        artifact_risk_level = "medium"
    if uniform:
        artifact_risk_level = max(artifact_risk_level, "medium", key=lambda x: {"low": 0, "medium": 1, "high": 2, "unclear": 1}[x])

    # ── QU ─────────────────────────────────────────────────────────────
    if target_label == "QU":
        if f["scale_question"] >= 4:
            return _make_result(
                name="importance/motivation scale questioner",
                interp="This latent appears to activate on counselor utterances using a numeric scale to probe importance or motivation (e.g., 'on a scale of one to ten'). The surface form consistently combines question framing with numeric scale metaphor.",
                patterns=[
                    _pattern("numeric-scale question frame", "surface_form",
                             f"High scale/one-to-ten phrasing ({f['scale_question']}/{analysis['n_top_examples']})."),
                    _pattern("importance/confidence probing", "dialogue_function",
                             "Questions elicit self-assessment of importance or readiness for change."),
                ],
                quality="high" if match_rate >= 0.9 else "medium",
                risk=artifact_risk_level,
                relationship=f"Strongly supports QU: scale-based elicitation. Match_rate={match_rate:.2f}.",
                conclusion=f"High-confidence 'numeric-scale questioning' candidate. Match_rate={match_rate:.2f}.",
                alternatives=["Could be triggered by numeric words rather than MI function.", "May overlap with QUO latents."],
                followups=["Token attribution for scale-word vs. question structure.", "Causal intervention on QU top-latents."],
            )
        elif f["starts_with_wh"] >= 4 or f["has_question_mark"] >= 6:
            return _make_result(
                name="open question initiator (wh-word opener)",
                interp="This latent appears to activate on counselor utterances beginning with wh-words (what, why, how) or containing direct question marks. May encode syntactic interrogative form.",
                patterns=[
                    _pattern("wh-word question opening", "surface_form",
                             f"Wh-word openers ({f['starts_with_wh']}/{analysis['n_top_examples']}), question marks ({f['has_question_mark']})."),
                    _pattern("information-seeking dialogue move", "dialogue_function",
                             "Questions invite client disclosure or elaboration."),
                ],
                quality="high" if match_rate >= 0.85 else "medium",
                risk=artifact_risk_level,
                relationship=f"Supports QU: captures interrogative form. Match_rate={match_rate:.2f}.",
                conclusion=f"Reliable question surface-form detector. Match_rate={match_rate:.2f}.",
                alternatives=["May fire on any interrogative regardless of MI context.", "Possibly driven by question-mark token."],
                followups=["Check QUO vs QUC classification.", "Token-level attribution."],
            )
        elif f["starts_with_do_can"] >= 4:
            return _make_result(
                name="auxiliary-inversion question detector (likely QUC sub-type)",
                interp="This latent appears to activate on questions with auxiliary inversion (Do/Can/Did/Is/Are), the syntactic hallmark of closed yes/no questions in English MI counseling.",
                patterns=[
                    _pattern("auxiliary-first question form", "surface_form",
                             f"Do/can/did/is-initial sentences={f['starts_with_do_can']}/{analysis['n_top_examples']}."),
                    _pattern("closed-ended elicitation", "dialogue_function",
                             "Questions expect binary or brief factual responses."),
                ],
                quality="high" if match_rate >= 0.85 else "medium",
                risk=artifact_risk_level,
                relationship=f"Supports QU (likely QUC sub-type). Match_rate={match_rate:.2f}.",
                conclusion=f"Closed question surface-form detector. Match_rate={match_rate:.2f}.",
                alternatives=["Some auxiliary-initial sentences may be indirect suggestions."],
                followups=["Verify QUC vs QU classification."],
            )
        elif f["would_phrase"] >= 4:
            return _make_result(
                name="hypothetical/conditional question framer",
                interp="This latent appears to fire on counselor utterances using 'would you', 'what would', or 'how would' for hypothetical or conditional questions.",
                patterns=[
                    _pattern("would-conditional question frame", "surface_form",
                             f"'would' interrogative usage={f['would_phrase']}/{analysis['n_top_examples']}."),
                    _pattern("hypothetical exploration", "dialogue_function",
                             "Questions invite client to imagine future states."),
                ],
                quality="medium",
                risk=artifact_risk_level,
                relationship=f"Supports QU (conditional questions). Match_rate={match_rate:.2f}.",
                conclusion=f"Medium-confidence hypothetical question candidate. Match_rate={match_rate:.2f}.",
                alternatives=["Could encode politeness/hedging rather than MI-specific function."],
                followups=["Token attribution for 'would'.", "Check QUO overlap."],
            )
        else:
            return _make_result(
                name="general question detector (mixed type)",
                interp="This latent shows moderate activation on counselor questions of various types. No single surface pattern dominates.",
                patterns=[
                    _pattern("question-form general", "surface_form",
                             f"Mix of interrogative forms; match_rate={match_rate:.2f}."),
                ],
                quality="medium" if match_rate >= 0.7 else "low",
                risk=artifact_risk_level,
                relationship=f"Moderate QU support. Match_rate={match_rate:.2f}.",
                conclusion=f"Weak-to-medium QU candidate. Match_rate={match_rate:.2f}.",
                alternatives=["Weak or general interrogative signal."],
                followups=["Compare with higher-rank QU latents."],
            )

    # ── QUO ────────────────────────────────────────────────────────────
    elif target_label == "QUO":
        if f["starts_with_wh"] >= 4 or (f["has_question_mark"] >= 5 and f["starts_with_do_can"] < 3):
            return _make_result(
                name="open-invitation question (what/how/why opener)",
                interp="This latent appears to activate on open-ended counselor questions that invite the client to elaborate freely, primarily via wh-word openers.",
                patterns=[
                    _pattern("wh-word open invitation", "surface_form",
                             f"Dominant wh-word openers={f['starts_with_wh']}/{analysis['n_top_examples']}."),
                    _pattern("open elicitation function", "dialogue_function",
                             "Questions invite extended client narration."),
                ],
                quality="high" if match_rate >= 0.7 else "medium",
                risk=artifact_risk_level,
                relationship=f"Supports QUO: wh-word openers are canonical open-ended questions. Match_rate={match_rate:.2f}.",
                conclusion=f"Medium-to-high confidence QUO candidate. Match_rate={match_rate:.2f}.",
                alternatives=["Some wh-questions may be informationally closed."],
                followups=["Verify open vs. closed distinction.", "Check QU vs QUO boundary."],
            )
        elif f["scale_question"] >= 3:
            return _make_result(
                name="scale-based open question (QUO type)",
                interp="This latent appears to fire on open-ended questions using numeric scales to elicit client self-assessment.",
                patterns=[
                    _pattern("scale-based open question", "surface_form",
                             f"Scale/importance phrasing count={f['scale_question']}."),
                ],
                quality="medium",
                risk=artifact_risk_level,
                relationship=f"Partial QUO support. Match_rate={match_rate:.2f}.",
                conclusion=f"QUO candidate, likely shared with QU family. Match_rate={match_rate:.2f}.",
                alternatives=["Shared with QU parent latents."],
                followups=["Compare with QU parent latents."],
            )
        else:
            return _make_result(
                name="mixed open/closed question pattern",
                interp="Top examples contain a mix of question types without clear open-ended dominance.",
                patterns=[
                    _pattern("question form (mixed)", "surface_form",
                             f"Mixed forms; wh={f['starts_with_wh']}, do/can={f['starts_with_do_can']}."),
                ],
                quality="low" if match_rate < 0.5 else "medium",
                risk=artifact_risk_level,
                relationship=f"Weak QUO support: match_rate={match_rate:.2f}.",
                conclusion=f"Low-to-medium confidence QUO candidate. Match_rate={match_rate:.2f}.",
                alternatives=["More likely a QU/QUC shared signal."],
                followups=["Check QU vs QUO vs QUC distribution."],
            )

    # ── QUC ────────────────────────────────────────────────────────────
    elif target_label == "QUC":
        if f["starts_with_do_can"] >= 4:
            return _make_result(
                name="yes/no closed question (auxiliary inversion)",
                interp="This latent appears to activate on questions with auxiliary inversion (Do/Can/Did/Is/Are), the syntactic hallmark of closed yes/no questions.",
                patterns=[
                    _pattern("auxiliary-inversion closed question", "surface_form",
                             f"Do/can/did/is-initial={f['starts_with_do_can']}/{analysis['n_top_examples']}."),
                    _pattern("factual elicitation (closed)", "dialogue_function",
                             "Questions expect binary or brief factual responses."),
                ],
                quality="high" if match_rate >= 0.6 else "medium",
                risk=artifact_risk_level,
                relationship=f"Supports QUC: closed question surface form. Match_rate={match_rate:.2f}.",
                conclusion=f"Medium-to-high confidence QUC candidate. Match_rate={match_rate:.2f}.",
                alternatives=["Some do/can utterances may be indirect suggestions."],
                followups=["Verify QUC vs QU exclusivity.", "Token attribution for auxiliary verbs."],
            )
        elif f["has_question_mark"] >= 5:
            return _make_result(
                name="general question detector (QUC/QU overlap)",
                interp="Top examples are predominantly questions, but closed/open distinction is unclear from surface form alone.",
                patterns=[
                    _pattern("question form (general)", "surface_form",
                             f"Question-mark rate={f['has_question_mark']}/{analysis['n_top_examples']}."),
                ],
                quality="medium",
                risk=artifact_risk_level,
                relationship=f"Partial QUC support (match_rate={match_rate:.2f}), likely shared with QU parent.",
                conclusion=f"Weak QUC candidate; plausibly QU family signal. Match_rate={match_rate:.2f}.",
                alternatives=["More likely QU parent signal than QUC-specific."],
                followups=["Compare with QU latents."],
            )
        else:
            return _make_result(
                name="mixed utterance (low closed-question specificity)",
                interp="Top examples do not show closed-question dominance. Diffuse signal.",
                patterns=[
                    _pattern("mixed dialogue", "mixed_unclear",
                             f"No dominant closed-question form; match_rate={match_rate:.2f}."),
                ],
                quality="low",
                risk=artifact_risk_level if artifact_risk_level != "low" else "medium",
                relationship=f"Weak QUC support: match_rate={match_rate:.2f}.",
                conclusion=f"Low-confidence QUC candidate. Match_rate={match_rate:.2f}.",
                alternatives=["Weak QU parent signal or cross-label pattern."],
                followups=["Compare with higher-rank QUC latents."],
            )

    # ── RE ─────────────────────────────────────────────────────────────
    elif target_label == "RE":
        if f["sounds_like"] >= 4 or f["it_seems"] >= 4:
            return _make_result(
                name="'sounds like / seems like' reflective phrase anchor",
                interp="This latent appears to fire on reflective counselor statements using 'sounds like', 'seems like', or 'it sounds like' — the most common surface markers of reflective listening in MI.",
                patterns=[
                    _pattern("'sounds/seems like' phrase", "surface_form",
                             f"Frequency={f['sounds_like']+f['it_seems']}/{analysis['n_top_examples']}."),
                    _pattern("reflective listening signal", "dialogue_function",
                             "Counselor paraphrases or infers client meaning using tentative language."),
                ],
                quality="high" if match_rate >= 0.6 else "medium",
                risk=artifact_risk_level,
                relationship=f"Supports RE: 'sounds like' is the canonical reflection phrase. Match_rate={match_rate:.2f}.",
                conclusion=f"Medium-to-high confidence RE candidate. Match_rate={match_rate:.2f}.",
                alternatives=["Some 'sounds like' utterances are non-reflective.", "Shared with REC latents by design."],
                followups=["Token attribution for 'sounds like' anchor.", "Supplement with prior client utterance."],
            )
        elif f["on_one_hand"] >= 2 or (f["you_feel"] >= 4 and match_rate >= 0.4):
            return _make_result(
                name="ambivalence / double-sided reflection framer",
                interp="This latent appears to activate on counselor reflections framing ambivalence ('on one hand... on the other hand') or paraphrasing conflicting feelings.",
                patterns=[
                    _pattern("double-sided framing", "surface_form",
                             f"'On one hand'={f['on_one_hand']}, 'you feel'={f['you_feel']}."),
                    _pattern("complex reflection of ambivalence", "dialogue_function",
                             "Counselor acknowledges competing desires/emotions."),
                    _pattern("MI: develop discrepancy", "mi_principle",
                             "Surfacing ambivalence is a core MI technique."),
                ],
                quality="medium",
                risk=artifact_risk_level,
                relationship=f"Supports REC/RE (ambivalence reflections). Match_rate={match_rate:.2f}.",
                conclusion=f"Medium-confidence ambivalence-reflection candidate. Match_rate={match_rate:.2f}.",
                alternatives=["Some 'you feel' statements could be AF."],
                followups=["Verify RE vs REC with client context."],
            )
        elif f["i_hear"] >= 3:
            return _make_result(
                name="'I hear you' reflective paraphrase",
                interp="This latent appears to activate on counselor utterances using 'I hear you', 'what I'm hearing' as explicit paraphrase markers.",
                patterns=[
                    _pattern("'I hear you' paraphrase marker", "surface_form",
                             f"'I hear'/'I'm hearing' count={f['i_hear']}."),
                    _pattern("explicit reflective paraphrase", "dialogue_function",
                             "Counselor signals reflecting client's words."),
                ],
                quality="medium",
                risk=artifact_risk_level,
                relationship=f"Supports RE: explicit reflective markers. Match_rate={match_rate:.2f}.",
                conclusion=f"Medium-confidence RE candidate. Match_rate={match_rate:.2f}.",
                alternatives=["Could encode general empathic acknowledgment."],
                followups=["Token attribution for 'I hear'."],
            )
        elif match_rate >= 0.5 and f["so_prefix"] >= 4:
            return _make_result(
                name="'So...' reflective discourse opening",
                interp="This latent appears to fire on counselor utterances beginning with 'So,' — a common reflection-initiating discourse marker in MI.",
                patterns=[
                    _pattern("'So' reflective discourse marker", "surface_form",
                             f"'so'-initial={f['so_prefix']}/{analysis['n_top_examples']}."),
                    _pattern("paraphrase initiation marker", "dialogue_function",
                             "Counselor uses 'So' to transition into a reflection."),
                ],
                quality="medium",
                risk=artifact_risk_level,
                relationship=f"Partial RE support. 'So' also precedes questions. Match_rate={match_rate:.2f}.",
                conclusion=f"Low-to-medium confidence RE candidate. Match_rate={match_rate:.2f}.",
                alternatives=["'So' precedes many question types too."],
                followups=["Compare 'so'-initial reflections vs questions."],
            )
        elif match_rate >= 0.4:
            return _make_result(
                name="general reflection signal (mixed markers)",
                interp="Top examples include RE utterances without a single dominant surface marker. May capture a diffuse reflective style signal.",
                patterns=[
                    _pattern("reflective paraphrase (diffuse)", "dialogue_function",
                             f"Match_rate={match_rate:.2f}; varied surface forms."),
                ],
                quality="medium" if match_rate >= 0.5 else "low",
                risk=artifact_risk_level,
                relationship=f"Moderate RE support. Match_rate={match_rate:.2f}.",
                conclusion=f"Medium-confidence RE candidate; distributed signal. Match_rate={match_rate:.2f}.",
                alternatives=["May encode a combination of partially overlapping signals."],
                followups=["Identify common sub-patterns.", "Supplement with client context."],
            )
        else:
            return _make_result(
                name="weak / mixed RE signal",
                interp="Top examples are mixed across label types with low RE match rate.",
                patterns=[
                    _pattern("mixed signal (low RE purity)", "mixed_unclear",
                             f"Match_rate={match_rate:.2f}; multiple non-RE labels."),
                ],
                quality="low",
                risk="medium" if artifact_risk_level == "low" else artifact_risk_level,
                relationship=f"Weak RE support: match_rate={match_rate:.2f}.",
                conclusion=f"Low-confidence RE candidate. Match_rate={match_rate:.2f}.",
                alternatives=["May encode general conversational register.", "Cross-label noise."],
                followups=["De-prioritize unless follow-up confirms signal."],
            )

    # ── REC ────────────────────────────────────────────────────────────
    elif target_label == "REC":
        if f["sounds_like"] >= 4 and (f["you_feel"] >= 2 or f["on_one_hand"] >= 2):
            return _make_result(
                name="'sounds like' + emotion inference (complex reflection)",
                interp="This latent appears to capture complex reflections combining 'sounds like' with emotion inference or ambivalence framing.",
                patterns=[
                    _pattern("'sounds like' + emotion", "surface_form",
                             f"'sounds like'={f['sounds_like']}, emotional language={f['you_feel']}."),
                    _pattern("complex emotion reflection", "dialogue_function",
                             "Counselor reflects implied emotional meaning beyond literal client words."),
                ],
                quality="medium",
                risk=artifact_risk_level,
                relationship=f"Supports REC: emotion-inferring reflections. Match_rate={match_rate:.2f}.",
                conclusion=f"Medium confidence REC candidate. Match_rate={match_rate:.2f}.",
                alternatives=["Some may be simple reflections if client stated the emotion.", "Cannot confirm without client utterance."],
                followups=["Verify with prior client utterance."],
            )
        elif f["sounds_like"] >= 3:
            return _make_result(
                name="'sounds like' reflective anchor (shared RE/REC)",
                interp="This latent fires on 'sounds like' reflection openers. Without client context, simple vs. complex cannot be determined.",
                patterns=[
                    _pattern("'sounds/seems like' anchor", "surface_form",
                             f"'sounds like'={f['sounds_like']}/{analysis['n_top_examples']}."),
                    _pattern("reflective paraphrase function", "dialogue_function",
                             "Paraphrase-initiation with tentative hedge."),
                ],
                quality="medium",
                risk=artifact_risk_level,
                relationship=f"Moderate REC support (match_rate={match_rate:.2f}), primarily RE/REC shared.",
                conclusion=f"Medium-confidence shared RE/REC candidate. Match_rate={match_rate:.2f}.",
                alternatives=["Shared with RE; not REC-specific without client utterance."],
                followups=["Supplement with client context."],
            )
        elif f["on_one_hand"] >= 2 or f["you_feel"] >= 5:
            return _make_result(
                name="ambivalence / double-sided complex reflection (REC)",
                interp="This latent appears to capture counselor utterances presenting two sides of client ambivalence.",
                patterns=[
                    _pattern("double-sided ambivalence framing", "surface_form",
                             f"'on one hand'={f['on_one_hand']}, 'you feel'={f['you_feel']}."),
                    _pattern("complex ambivalence reflection", "dialogue_function",
                             "Counselor mirrors conflicting emotions/desires."),
                    _pattern("MI: develop discrepancy", "mi_principle",
                             "Double-sided reflections are a key REC technique."),
                ],
                quality="medium",
                risk=artifact_risk_level,
                relationship=f"Supports REC: ambivalence framing is distinctive. Match_rate={match_rate:.2f}.",
                conclusion=f"Medium-confidence REC candidate. Match_rate={match_rate:.2f}.",
                alternatives=["Some ambivalence statements might be GI or SU."],
                followups=["Verify 'on one hand' in context of prior client speech."],
            )
        elif match_rate >= 0.4:
            return _make_result(
                name="general complex reflection (distributed signal)",
                interp="Top examples include complex reflections without a single dominant surface marker.",
                patterns=[
                    _pattern("complex reflection (no single marker)", "dialogue_function",
                             f"Match_rate={match_rate:.2f}; varied surface forms."),
                ],
                quality="low" if match_rate < 0.5 else "medium",
                risk=artifact_risk_level,
                relationship=f"Partial REC support (match_rate={match_rate:.2f}).",
                conclusion=f"Low-to-medium confidence REC candidate. Match_rate={match_rate:.2f}.",
                alternatives=["May overlap more with RE than REC."],
                followups=["Supplement with client context."],
            )
        else:
            return _make_result(
                name="weak REC signal (mixed labels)",
                interp="Top examples are not predominantly REC. Multiple other labels appear.",
                patterns=[
                    _pattern("mixed / low-purity", "mixed_unclear",
                             f"Match_rate={match_rate:.2f}. Non-REC labels dominate."),
                ],
                quality="low",
                risk="medium" if artifact_risk_level == "low" else artifact_risk_level,
                relationship=f"Weak REC support: match_rate={match_rate:.2f}.",
                conclusion=f"Low-confidence REC candidate. Match_rate={match_rate:.2f}.",
                alternatives=["Likely shared RE/REC signal or cross-label noise."],
                followups=["De-prioritize unless follow-up confirms."],
            )

    # ── RES ────────────────────────────────────────────────────────────
    elif target_label == "RES":
        if uniform or match_rate < 0.15:
            return _make_result(
                name="artifact / uninterpretable (RES - very low purity)",
                interp="Top examples show very low RES match rate. The latent does not encode simple reflection reliably. Possible activation artifact.",
                patterns=[
                    _pattern("artifact / noise pattern", "artifact",
                             f"Match_rate={match_rate:.2f}; top examples span multiple unrelated labels."),
                ],
                quality="uninterpretable",
                risk="high",
                relationship=f"Does not support RES: match_rate={match_rate:.2f}.",
                conclusion="Uninterpretable for RES. Recommend artifact investigation.",
                alternatives=["May encode general discourse feature.", "Max-pooling artifact possible."],
                followups=["Token-level attribution.", "Examine activation value distribution."],
            )
        elif f["sounds_like"] >= 2:
            return _make_result(
                name="'sounds like' simple reflection (possible RES/RE overlap)",
                interp="Some top examples use 'sounds like' in short reflective statements. Match rate low, examples mixed.",
                patterns=[
                    _pattern("'sounds like' simple reflection", "surface_form",
                             f"'sounds like'={f['sounds_like']}; match_rate={match_rate:.2f}."),
                ],
                quality="low",
                risk=artifact_risk_level,
                relationship=f"Very weak RES support (match_rate={match_rate:.2f}).",
                conclusion=f"Low-confidence RES candidate. Match_rate={match_rate:.2f}.",
                alternatives=["Not specific to simple reflections.", "Cannot distinguish from RE/REC without context."],
                followups=["Supplement with client utterance."],
            )
        else:
            return _make_result(
                name="uninterpretable / low purity RES signal",
                interp="Top examples do not show consistent RES patterns. Heterogeneous mix.",
                patterns=[
                    _pattern("heterogeneous mix", "mixed_unclear",
                             f"Match_rate={match_rate:.2f}; no dominant RES surface form."),
                ],
                quality="uninterpretable" if match_rate < 0.2 else "low",
                risk="high" if match_rate < 0.2 else "medium",
                relationship=f"Does not support RES: match_rate={match_rate:.2f}.",
                conclusion=f"Uninterpretable for RES. Match_rate={match_rate:.2f}.",
                alternatives=["May encode general conversation feature.", "RES is context-dependent."],
                followups=["Exclude from RES analysis without additional evidence."],
            )

    # ── GI ─────────────────────────────────────────────────────────────
    elif target_label == "GI":
        if f["medical_info"] >= 3:
            return _make_result(
                name="medical / clinical information delivery (GI)",
                interp="This latent appears to fire on counselor utterances delivering medical or clinical information (medications, dosages, health facts).",
                patterns=[
                    _pattern("medical fact delivery", "surface_form",
                             f"Medical terminology in {f['medical_info']}/{analysis['n_top_examples']} examples."),
                    _pattern("information giving function", "dialogue_function",
                             "Counselor provides factual medical/health information."),
                ],
                quality="medium",
                risk=artifact_risk_level,
                relationship=f"Moderate GI support (match_rate={match_rate:.2f}). Medical terms are reliable GI markers.",
                conclusion=f"Medium-confidence GI candidate via medical info. Match_rate={match_rate:.2f}.",
                alternatives=["Longer explanatory utterances may blur GI boundaries.", f"Low Cohen's d ({cohens_d:.3f}) suggests weak signal."],
                followups=["Token attribution for medical terms.", "Check sentence length as confound."],
            )
        elif f["long_utterance"] >= 3 and match_rate >= 0.4:
            return _make_result(
                name="explanatory / informational long utterance (general GI)",
                interp="Top examples include counselor statements providing explanatory or advisory information. Utterances tend to be longer and more declarative.",
                patterns=[
                    _pattern("informational/explanatory utterance", "dialogue_function",
                             f"Match_rate={match_rate:.2f}; long declarative sentences."),
                    _pattern("long declarative sentence style", "surface_form",
                             f"Long utterances ({f['long_utterance']}/{analysis['n_top_examples']})."),
                ],
                quality="medium",
                risk=artifact_risk_level,
                relationship=f"Moderate GI support (match_rate={match_rate:.2f}).",
                conclusion=f"Medium-confidence GI candidate. Match_rate={match_rate:.2f}.",
                alternatives=["Long sentences may activate regardless of information-giving function."],
                followups=["Token attribution.", "Check sentence length as confound."],
            )
        elif match_rate >= 0.4:
            return _make_result(
                name="general information-giving signal",
                interp="Top examples include GI utterances but no single content domain dominates. 'Giving information' functional role is apparent.",
                patterns=[
                    _pattern("information giving (general)", "dialogue_function",
                             f"Match_rate={match_rate:.2f}; factual/advisory content present."),
                ],
                quality="medium" if match_rate >= 0.5 else "low",
                risk=artifact_risk_level,
                relationship=f"Moderate GI support (match_rate={match_rate:.2f}).",
                conclusion=f"Medium-confidence GI candidate. Match_rate={match_rate:.2f}.",
                alternatives=["GI is a content dimension rather than form dimension."],
                followups=["Investigate topic-specific sub-patterns."],
            )
        else:
            return _make_result(
                name="weak GI signal / mixed content",
                interp="Top examples are heterogeneous; GI-specific patterns not clearly dominant. GI is a content dimension rather than a form dimension.",
                patterns=[
                    _pattern("mixed content (low GI specificity)", "mixed_unclear",
                             f"Match_rate={match_rate:.2f}; no dominant GI pattern."),
                ],
                quality="low",
                risk="medium" if artifact_risk_level == "low" else artifact_risk_level,
                relationship=f"Weak GI support: match_rate={match_rate:.2f}.",
                conclusion=f"Low-confidence GI candidate. Match_rate={match_rate:.2f}.",
                alternatives=["GI may not be well-captured by utterance-level SAE max-pooling."],
                followups=["Accept weak signal as expected.", "Check topic-specific latents."],
            )

    # ── AF ─────────────────────────────────────────────────────────────
    elif target_label == "AF":
        if f["positive_eval"] >= 5:
            return _make_result(
                name="positive evaluative phrase detector (great/good/wonderful)",
                interp="This latent appears to fire on counselor utterances containing positive evaluative adjectives ('great', 'good', 'wonderful'). These are the most reliable surface markers of affirmations in MI.",
                patterns=[
                    _pattern("positive evaluative lexicon", "surface_form",
                             f"Positive adjectives in {f['positive_eval']}/{analysis['n_top_examples']} examples."),
                    _pattern("affirmation function (MI)", "dialogue_function",
                             "Counselor validates or praises client's effort/change."),
                    _pattern("MI spirit: affirmation supports self-efficacy", "mi_principle",
                             "Affirmations in MI support client self-efficacy."),
                ],
                quality="high" if match_rate >= 0.7 else "medium",
                risk=artifact_risk_level,
                relationship=f"Supports AF: positive evaluative lexicon is the canonical AF marker. Match_rate={match_rate:.2f}.",
                conclusion=f"High surface confidence AF candidate. Match_rate={match_rate:.2f}.",
                alternatives=["May detect positive-adjective tokens rather than MI affirmation function.", "Some 'great/good' may be non-affirmative."],
                followups=["Token attribution for specific positive adjectives.", "Test AF vs SU distinction."],
            )
        elif f["that_prefix"] >= 4 and f["positive_eval"] >= 3:
            return _make_result(
                name="'that's great/good' affirmative response pattern",
                interp="This latent appears to activate on 'that's great', 'that sounds good', 'that's wonderful' — short positive evaluative responses beginning with 'that'.",
                patterns=[
                    _pattern("'that's [positive]' response template", "surface_form",
                             f"'that' prefix={f['that_prefix']}, positive eval={f['positive_eval']}."),
                    _pattern("brief affirmative response", "dialogue_function",
                             "Short positive counselor evaluation of client statement."),
                ],
                quality="medium",
                risk=artifact_risk_level,
                relationship=f"Supports AF via short positive-response template. Match_rate={match_rate:.2f}.",
                conclusion=f"Medium-confidence AF candidate. Match_rate={match_rate:.2f}.",
                alternatives=["'That' prefix may be driving activation more than the positive evaluation."],
                followups=["Check if 'that' prefix alone drives activation without positive words."],
            )
        elif match_rate >= 0.4:
            return _make_result(
                name="general affirmation signal (partial lexical match)",
                interp="Top examples include AF utterances but positive evaluative words are not the sole driver. Broader 'positive counselor register'.",
                patterns=[
                    _pattern("positive counselor register", "dialogue_function",
                             f"Match_rate={match_rate:.2f}; various positive statements."),
                ],
                quality="medium" if match_rate >= 0.5 else "low",
                risk=artifact_risk_level,
                relationship=f"Moderate AF support (match_rate={match_rate:.2f}).",
                conclusion=f"Medium-confidence AF candidate. Match_rate={match_rate:.2f}.",
                alternatives=["May encode general positive counselor interaction."],
                followups=["Identify specific positive register signal."],
            )
        else:
            return _make_result(
                name="weak AF / positive-word cross-label",
                interp="Top examples are mixed; AF match rate is low. May capture general positive register spanning AF, SU.",
                patterns=[
                    _pattern("positive register (cross-label)", "mixed_unclear",
                             f"Match_rate={match_rate:.2f}; positive words not AF-specific."),
                ],
                quality="low",
                risk="medium" if artifact_risk_level == "low" else artifact_risk_level,
                relationship=f"Weak AF support: match_rate={match_rate:.2f}.",
                conclusion=f"Low-confidence AF candidate. Match_rate={match_rate:.2f}.",
                alternatives=["Shared positive-register signal across AF, SU."],
                followups=["Compare with higher-rank AF latents."],
            )

    # ── SU ─────────────────────────────────────────────────────────────
    elif target_label == "SU":
        if f["i_understand"] >= 3:
            return _make_result(
                name="'I understand' empathy template",
                interp="This latent appears to fire on counselor utterances using 'I understand', 'I completely understand' as an empathic acknowledgment template.",
                patterns=[
                    _pattern("'I understand' empathy template", "surface_form",
                             f"'I understand'={f['i_understand']}/{analysis['n_top_examples']}."),
                    _pattern("empathic support / acknowledgment", "dialogue_function",
                             "Counselor validates client's difficulty."),
                ],
                quality="medium",
                risk=artifact_risk_level,
                relationship=f"Partial SU support (match_rate={match_rate:.2f}).",
                conclusion=f"Medium-confidence SU candidate. Match_rate={match_rate:.2f}.",
                alternatives=["Generic empathy phrases appear across SU, RE, AF.", "SU has small sample size (n=222)."],
                followups=["Check SU vs AF vs RE for 'I understand' utterances."],
            )
        elif match_rate >= 0.25:
            return _make_result(
                name="general support / suggestion signal (mixed SU)",
                interp="Top examples include some SU utterances but without a single dominant surface pattern. SU covers diverse behaviors.",
                patterns=[
                    _pattern("mixed support/suggestion", "mixed_unclear",
                             f"Match_rate={match_rate:.2f}; no single SU pattern."),
                ],
                quality="low",
                risk=artifact_risk_level if artifact_risk_level != "low" else "medium",
                relationship=f"Weak SU support: match_rate={match_rate:.2f}.",
                conclusion=f"Low-confidence SU candidate. Match_rate={match_rate:.2f}.",
                alternatives=["SU may not have strong enough signal.", "Cross-label noise from AF or RE."],
                followups=["Accept weak signal given small sample size."],
            )
        else:
            return _make_result(
                name="uninterpretable for SU (cross-label noise)",
                interp="Top examples are predominantly from other labels. The latent does not appear to encode SU-relevant patterns.",
                patterns=[
                    _pattern("cross-label noise", "mixed_unclear",
                             f"Match_rate={match_rate:.2f}; mainly non-SU."),
                ],
                quality="uninterpretable" if match_rate < 0.1 else "low",
                risk="high" if match_rate < 0.1 else "medium",
                relationship=f"Does not support SU: match_rate={match_rate:.2f}.",
                conclusion=f"Uninterpretable SU candidate. Match_rate={match_rate:.2f}.",
                alternatives=["Latent captures features unrelated to SU."],
                followups=["Exclude from SU analysis."],
            )

    # ── Fallback ──────────────────────────────────────────────────────
    else:
        return _make_result(
            name=f"unknown label ({target_label})",
            interp=f"Latent for label {target_label}. No specific rule available.",
            patterns=[_pattern("unanalyzed", "mixed_unclear", "No rule for this label.")],
            quality="uninterpretable",
            risk="unclear",
            relationship="Not analyzed.",
            conclusion="No conclusion.",
            alternatives=[],
            followups=["Add label-specific rule."],
        )


def _pattern(name, ptype, evidence):
    return {"pattern_name": name, "pattern_type": ptype, "evidence": evidence}


def _make_result(*, name, interp, patterns, quality, risk, relationship,
                 conclusion, alternatives, followups):
    return {
        "candidate_feature_name": name,
        "one_sentence_tentative_interpretation": interp,
        "main_patterns": patterns[:5],
        "evidence_quality": quality,
        "artifact_risk": risk,
        "relationship_to_target_label": relationship,
        "final_concise_conclusion": conclusion,
        "alternative_explanations": alternatives,
        "recommended_followup_checks": followups,
        "parse_status": "ok",
    }


def main():
    print("=== P3 Task A Direct Execution ===")
    print(f"Reading prompts from: {PROMPTS_DIR}")

    cards_index = load_cards_index()
    prompt_files = sorted(PROMPTS_DIR.glob("*_input_prompt.json"))
    print(f"Found {len(prompt_files)} prompt files")

    results = []
    for i, pf in enumerate(prompt_files):
        label, rank, latent = parse_prompt_filename(pf.name)
        pid = cards_index.get((label, latent, rank), f"unknown_{label}_{rank}_{latent}")

        payload = extract_prompt_data(pf)
        examples = payload.get("examples", [])
        metrics = payload.get("association_metrics", {})
        cohens_d = float(metrics.get("cohens_d", 0.0))

        analysis = analyze_examples(examples)
        result = classify_latent(label, latent, rank, cohens_d, analysis)

        result["item_index"] = i
        result["packet_id"] = pid
        result["target_label"] = label
        result["latent_idx"] = latent
        result["rank_within_label"] = rank
        result["prompt_filename"] = pf.name

        results.append(result)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(prompt_files)} prompts...")

    print(f"Generated {len(results)} explanations")

    # Write batch output
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    batch_output = {
        "batch_id": 1,
        "model": "claude-opus-4-direct-analysis",
        "completed_at": datetime.datetime.now().isoformat(),
        "results": results,
    }
    output_path = BATCH_DIR / "task_a_batch_001_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(batch_output, f, ensure_ascii=False, indent=2)

    print(f"Written to: {output_path}")

    # Quality summary
    from collections import Counter as C
    quality_counts = C(r["evidence_quality"] for r in results)
    risk_counts = C(r["artifact_risk"] for r in results)
    label_counts = Counter()
    for r in results:
        label_counts[r["target_label"]] += 1

    print()
    print("Quality distribution:")
    for q in ["high", "medium", "low", "uninterpretable"]:
        print(f"  {q}: {quality_counts.get(q, 0)}")
    print()
    print("Artifact risk distribution:")
    for r_level in ["low", "medium", "high", "unclear"]:
        print(f"  {r_level}: {risk_counts.get(r_level, 0)}")
    print()
    print("Per-label counts:")
    for lbl in sorted(label_counts):
        print(f"  {lbl}: {label_counts[lbl]}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
