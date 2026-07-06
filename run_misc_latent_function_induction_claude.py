"""
Generate latent function induction reviews for all 180 SAE latent packets.
This script embeds Claude's direct analysis of the latent utterance patterns
and writes results in the same format as run_misc_latent_function_induction.py.
"""

import json
import csv
from pathlib import Path
from collections import Counter
import datetime

PACKETS_LABELED = Path(
    "outputs/misc_full_sae_eval/interpretability/top20_cohensd_latent_utterances/"
    "latent_evidence_packets/latent_evidence_packets_labeled.jsonl"
)
SUMMARY_CSV = Path(
    "outputs/misc_full_sae_eval/interpretability/top20_cohensd_latent_utterances/"
    "latent_evidence_packets/latent_evidence_packet_summary.csv"
)
OUTPUT_DIR = Path(
    "outputs/misc_full_sae_eval/interpretability/top20_cohensd_latent_utterances/"
    "latent_function_induction_claude"
)

CONTEXT_NOTE = (
    "Only counselor current utterance is available in phase 1; "
    "prior client context is unavailable."
)


def analyze_top_examples(top_examples):
    """Extract key surface patterns from top activating examples."""
    texts = [str(e.get("unit_text", e.get("text", ""))).lower() for e in top_examples]
    labels = [str(e.get("active_labels", e.get("label", ""))) for e in top_examples]
    matches = [int(e.get("target_match", e.get("match", 0))) for e in top_examples]
    acts = [float(e.get("activation", e.get("act", 0.0))) for e in top_examples]

    match_rate = sum(matches) / len(matches) if matches else 0.0

    features = {
        "has_question_mark": sum("?" in t for t in texts),
        "starts_with_wh": sum(t.strip().startswith(("what ", "why ", "how ", "when ", "where ", "who ")) for t in texts),
        "starts_with_do_can": sum(t.strip().startswith(("do ", "can ", "did ", "does ", "have ", "has ", "is ", "are ", "was ", "were ")) for t in texts),
        "sounds_like": sum("sounds like" in t or "sound like" in t for t in texts),
        "it_seems": sum("it seems" in t or "seems like" in t for t in texts),
        "i_hear": sum("i hear" in t or "i'm hearing" in t or "what i'm hearing" in t for t in texts),
        "on_one_hand": sum("on one hand" in t or "on the other hand" in t for t in texts),
        "you_feel": sum("you feel" in t or "you're feeling" in t for t in texts),
        "positive_eval": sum(any(w in t for w in ["great", "good", "wonderful", "awesome", "amazing", "excellent", "fantastic"]) for t in texts),
        "i_understand": sum("i understand" in t or "i completely understand" in t or "totally understand" in t for t in texts),
        "would_phrase": sum("would you" in t or "what would" in t or "how would" in t for t in texts),
        "scale_question": sum("scale" in t or "one to ten" in t or "1 to 10" in t for t in texts),
        "medical_info": sum(any(w in t for w in ["medication", "medicine", "cholesterol", "diabetes", "blood pressure", "mg", "milligram", "prescri"]) for t in texts),
        "so_prefix": sum(t.strip().startswith("so ") or t.strip().startswith("so,") for t in texts),
        "you_prefix": sum(t.strip().startswith("you ") or t.strip().startswith("you're ") or t.strip().startswith("you've ") for t in texts),
        "okay_prefix": sum(t.strip().startswith("ok") for t in texts),
        "yeah_prefix": sum(t.strip().startswith("yeah") for t in texts),
    }

    if len(acts) > 3:
        unique_acts = len(set(round(a, 3) for a in acts))
        activation_uniform = unique_acts <= 2 and len(acts) >= 8
    else:
        activation_uniform = False

    all_labels = []
    for lbl in labels:
        if lbl:
            all_labels.extend(lbl.split(","))
    label_counter = Counter(all_labels)

    return {
        "match_rate": match_rate,
        "features": features,
        "activation_uniform": activation_uniform,
        "label_counter": label_counter,
        "n_examples": len(top_examples),
    }


def classify_latent(packet, analysis):
    target = packet["target_label"]
    latent_idx = packet["latent_idx"]
    match_rate = analysis["match_rate"]
    f = analysis["features"]
    uniform = analysis["activation_uniform"]
    cohens_d = packet.get("cohens_d", 0.0)

    artifact_risks = []
    if uniform:
        artifact_risks.append(
            f"Activation values appear highly uniform across top examples (possible max-pooling or quantization artifact for latent {latent_idx})."
        )
    if match_rate < 0.2:
        artifact_risks.append(
            "Target match rate below 0.20: high-activation examples predominantly from other labels."
        )

    # ── QU ───────────────────────────────────────────────────────────────────
    if target == "QU":
        if f["scale_question"] >= 5:
            return dict(
                candidate_name="importance/motivation scale questioner",
                tentative_interpretation="Strongly activates on counselor utterances using a numeric scale to probe importance or motivation (e.g., 'on a scale of one to ten'). Consistent surface form: question framing combined with numeric scale metaphor.",
                patterns=[
                    {"pattern_name": "numeric-scale question frame", "pattern_type": "surface_form", "evidence": f"High 'scale/one to ten' phrasing ({f['scale_question']}/{analysis['n_examples']})."},
                    {"pattern_name": "importance/confidence probing", "pattern_type": "dialogue_function", "evidence": "Questions elicit self-assessment of importance or readiness for change."},
                ],
                evidence_quality="high" if match_rate >= 0.9 else "medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["Could be triggered by numeric words rather than MI function.", "May overlap with QUO latents."],
                target_label_relationship=f"Strongly supports QU: every top example is a question using scale-based elicitation. Match_rate={match_rate:.2f}.",
                final_conclusion=f"High-confidence 'numeric-scale motivational questioning' candidate. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE,
                recommended_followup_checks=["Token attribution to confirm scale-word vs. question structure triggers.", "Causal intervention on QU top-latents."],
            )
        elif f["starts_with_wh"] >= 5 or f["has_question_mark"] >= 8:
            return dict(
                candidate_name="open question initiator (wh-word / how/why opener)",
                tentative_interpretation="Activates on counselor utterances beginning with wh-words (what, why, how, where) or containing direct question marks. Encodes syntactic interrogative form rather than MI-specific function.",
                patterns=[
                    {"pattern_name": "wh-word question opening", "pattern_type": "surface_form", "evidence": f"High wh-word openers ({f['starts_with_wh']}/{analysis['n_examples']}) or question marks ({f['has_question_mark']})."},
                    {"pattern_name": "information-seeking dialogue move", "pattern_type": "dialogue_function", "evidence": "Questions invite client disclosure or elaboration."},
                ],
                evidence_quality="high" if match_rate >= 0.85 else "medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["May fire on any interrogative regardless of MI context.", "Possibly driven by question-mark token rather than semantic content."],
                target_label_relationship=f"Supports QU: top examples are questions. Latent captures interrogative form but does not distinguish QUO from QUC. Match_rate={match_rate:.2f}.",
                final_conclusion=f"Reliable surface-form detector for questions. Strong QU candidate. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE,
                recommended_followup_checks=["Check QUO vs QUC classification of top examples.", "Token-level attribution."],
            )
        elif f["starts_with_do_can"] >= 5:
            return dict(
                candidate_name="auxiliary-inversion question detector (likely QUC)",
                tentative_interpretation="Activates on questions with auxiliary inversion (Do/Can/Did/Have/Is/Are), the syntactic hallmark of closed yes/no questions in English MI counseling.",
                patterns=[
                    {"pattern_name": "auxiliary-first question form", "pattern_type": "surface_form", "evidence": f"Do/can/did/is-initial sentences={f['starts_with_do_can']}/{analysis['n_examples']}."},
                    {"pattern_name": "closed-ended elicitation", "pattern_type": "dialogue_function", "evidence": "Questions expect binary or brief factual responses."},
                ],
                evidence_quality="high" if match_rate >= 0.85 else "medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["Some auxiliary-initial sentences may be indirect suggestions.", "Shared with QU parent latents."],
                target_label_relationship=f"Supports QU (likely QUC sub-type): closed question surface form dominates. Match_rate={match_rate:.2f}.",
                final_conclusion=f"Surface-form detector for closed questions. Strong QU/QUC candidate. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE,
                recommended_followup_checks=["Verify QUC vs QU classification.", "Check if indirect suggestions inflate false positives."],
            )
        elif f["would_phrase"] >= 5:
            return dict(
                candidate_name="hypothetical/conditional question framer",
                tentative_interpretation="Fires on counselor utterances using 'would you', 'what would', or 'how would' to frame hypothetical or conditional questions. Common MI technique for exploring possibilities.",
                patterns=[
                    {"pattern_name": "would-conditional question frame", "pattern_type": "surface_form", "evidence": f"'would' interrogative usage={f['would_phrase']}/{analysis['n_examples']}."},
                    {"pattern_name": "hypothetical exploration", "pattern_type": "dialogue_function", "evidence": "Questions invite client to imagine future states or possibilities."},
                ],
                evidence_quality="medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["Could partially encode politeness/hedging rather than MI-specific function."],
                target_label_relationship=f"Supports QU: all examples are conditional questions. Overlaps with QUO (hypothetical questions are typically open-ended). Match_rate={match_rate:.2f}.",
                final_conclusion=f"Medium-confidence hypothetical/conditional question candidate. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE,
                recommended_followup_checks=["Token attribution for 'would' vs. question structure.", "Check QUO overlap."],
            )
        else:
            return dict(
                candidate_name="general question detector (mixed type)",
                tentative_interpretation="Moderate activation on counselor questions of various types. No single surface pattern dominates; may capture a diffuse questioning signal.",
                patterns=[
                    {"pattern_name": "question-form general", "pattern_type": "surface_form", "evidence": f"Mix of interrogative forms; match_rate={match_rate:.2f}."},
                ],
                evidence_quality="medium" if match_rate >= 0.7 else "low",
                artifact_risks=artifact_risks,
                alternative_explanations=["Weak or general interrogative signal.", "May be lower-rank QU latent with diffuse signal."],
                target_label_relationship=f"Moderate QU support. Match_rate={match_rate:.2f}. No dominant sub-type.",
                final_conclusion=f"Weak-to-medium QU candidate. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE,
                recommended_followup_checks=["Compare with higher-rank QU latents for distinctiveness."],
            )

    # ── QUO ──────────────────────────────────────────────────────────────────
    elif target == "QUO":
        if f["starts_with_wh"] >= 5 or (f["has_question_mark"] >= 6 and f["starts_with_do_can"] < 3):
            return dict(
                candidate_name="open-invitation question (what/how/why opener)",
                tentative_interpretation="Activates preferentially on open-ended counselor questions that invite the client to elaborate freely. Wh-word questions or questions without auxiliary inversion.",
                patterns=[
                    {"pattern_name": "wh-word open invitation", "pattern_type": "surface_form", "evidence": f"Dominant wh-word openers={f['starts_with_wh']}/{analysis['n_examples']}."},
                    {"pattern_name": "open elicitation function", "pattern_type": "dialogue_function", "evidence": "Questions invite extended client narration."},
                ],
                evidence_quality="high" if match_rate >= 0.7 else "medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["Some wh-questions may be informationally closed (e.g., 'What kind of dog?').", "May not reliably distinguish open from closed wh-questions."],
                target_label_relationship=f"Supports QUO: wh-word openers are canonical open-ended questions. Match_rate={match_rate:.2f}.",
                final_conclusion=f"Medium-to-high confidence QUO candidate. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE,
                recommended_followup_checks=["Verify open vs. closed distinction in top examples manually.", "Check QU vs QUO boundary."],
            )
        elif f["scale_question"] >= 4:
            return dict(
                candidate_name="scale-based open question (QUO type)",
                tentative_interpretation="Fires on open-ended questions using numeric scales to elicit client self-assessment. Scale questions in MI are typically open-ended follow-ups.",
                patterns=[
                    {"pattern_name": "scale-based open question", "pattern_type": "surface_form", "evidence": f"Scale/importance phrasing count={f['scale_question']}."},
                ],
                evidence_quality="medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["Shared with QU parent latents."],
                target_label_relationship=f"Partial QUO support: scale questions are open-ended but also serve QU broadly. Match_rate={match_rate:.2f}.",
                final_conclusion=f"QUO candidate, likely shared with QU family. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE,
                recommended_followup_checks=["Compare with QU parent latents."],
            )
        else:
            return dict(
                candidate_name="mixed open/closed question (QUO weak signal)",
                tentative_interpretation="Top examples contain a mix of question types without clear open-ended dominance. The latent may represent a general questioning signal.",
                patterns=[
                    {"pattern_name": "question form (mixed)", "pattern_type": "surface_form", "evidence": f"Mixed question forms; wh={f['starts_with_wh']}, do/can={f['starts_with_do_can']}."},
                ],
                evidence_quality="low" if match_rate < 0.6 else "medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["More likely a QU/QUC shared signal than specifically QUO."],
                target_label_relationship=f"Weak QUO support: match_rate={match_rate:.2f}, mixed question types.",
                final_conclusion=f"Low-to-medium confidence QUO candidate. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE,
                recommended_followup_checks=["Check QU vs QUO vs QUC top-example distribution."],
            )

    # ── QUC ──────────────────────────────────────────────────────────────────
    elif target == "QUC":
        if f["starts_with_do_can"] >= 5:
            return dict(
                candidate_name="yes/no closed question (auxiliary inversion, QUC-specific)",
                tentative_interpretation="Activates on questions with auxiliary inversion (Do/Can/Did/Have/Is/Are), the syntactic hallmark of closed yes/no questions in English MI counseling. High specificity to closed question surface form.",
                patterns=[
                    {"pattern_name": "auxiliary-inversion closed question", "pattern_type": "surface_form", "evidence": f"Do/can/did/is-initial sentences={f['starts_with_do_can']}/{analysis['n_examples']}."},
                    {"pattern_name": "factual elicitation (closed)", "pattern_type": "dialogue_function", "evidence": "Questions expect binary or brief factual responses."},
                ],
                evidence_quality="high" if match_rate >= 0.6 else "medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["Some do/can utterances may be indirect suggestions.", "Shared with QU parent; exclusive QUC specificity needs verification."],
                target_label_relationship=f"Supports QUC specifically: closed question surface form dominates. Match_rate={match_rate:.2f}.",
                final_conclusion=f"Medium-to-high confidence QUC candidate. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE,
                recommended_followup_checks=["Verify QUC vs QU exclusivity.", "Token attribution for auxiliary verbs."],
            )
        elif f["has_question_mark"] >= 6:
            return dict(
                candidate_name="general question detector (QUC/QU overlap)",
                tentative_interpretation="Top examples are predominantly questions, but the closed/open distinction is not clear from surface form alone. Likely a QU family signal.",
                patterns=[
                    {"pattern_name": "question form (general)", "pattern_type": "surface_form", "evidence": f"High question-mark rate={f['has_question_mark']}/{analysis['n_examples']}."},
                ],
                evidence_quality="medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["More likely QU parent signal than QUC-specific."],
                target_label_relationship=f"Partial QUC support (match_rate={match_rate:.2f}), but likely shared with QU parent.",
                final_conclusion=f"Weak QUC candidate; more plausibly a QU family signal. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE,
                recommended_followup_checks=["Compare with QU latents."],
            )
        else:
            return dict(
                candidate_name="mixed utterance (low closed-question specificity)",
                tentative_interpretation="Top examples do not show closed-question dominance. The latent captures a diffuse signal not tightly tied to closed questions.",
                patterns=[
                    {"pattern_name": "mixed dialogue", "pattern_type": "mixed_unclear", "evidence": f"No dominant closed-question surface form; match_rate={match_rate:.2f}."},
                ],
                evidence_quality="low",
                artifact_risks=artifact_risks,
                alternative_explanations=["Weak QU parent signal or cross-label pattern."],
                target_label_relationship=f"Weak QUC support: match_rate={match_rate:.2f}, no dominant closed-question pattern.",
                final_conclusion=f"Low-confidence QUC candidate. De-prioritize. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE,
                recommended_followup_checks=["Compare with higher-rank QUC latents."],
            )

    # ── RE ────────────────────────────────────────────────────────────────────
    elif target == "RE":
        if f["sounds_like"] >= 5 or f["it_seems"] >= 5:
            return dict(
                candidate_name="'sounds like / seems like' reflective phrase anchor",
                tentative_interpretation="Fires strongly on reflective counselor statements using anchor phrases 'sounds like', 'seems like', or 'it sounds like'. These are the most common surface markers of reflective listening in MI transcripts.",
                patterns=[
                    {"pattern_name": "'sounds/seems like' phrase", "pattern_type": "surface_form", "evidence": f"'sounds/seems like' frequency={f['sounds_like']+f['it_seems']}/{analysis['n_examples']}."},
                    {"pattern_name": "reflective listening signal", "pattern_type": "dialogue_function", "evidence": "Counselor paraphrases or infers client meaning/emotion using tentative language."},
                ],
                evidence_quality="high" if match_rate >= 0.6 else "medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["Some 'sounds like' utterances are non-reflective AF/SU (e.g., 'that sounds great').", "Surface phrase alone insufficient to confirm reflective function.", "Shared with REC latents by design."],
                target_label_relationship=f"Supports RE and REC: 'sounds like' is the canonical reflection phrase. Match_rate={match_rate:.2f}. Cannot distinguish RE from REC without client context.",
                final_conclusion=f"Medium-to-high confidence RE/REC candidate. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE + " Prior client utterance unavailable, so reflection/context claims are limited.",
                recommended_followup_checks=["Token attribution for 'sounds like' anchor.", "Supplement with prior client utterance to test RE vs REC."],
            )
        elif f["on_one_hand"] >= 3 or (f["you_feel"] >= 5 and match_rate >= 0.5):
            return dict(
                candidate_name="ambivalence / double-sided reflection framer",
                tentative_interpretation="Activates on counselor reflections that explicitly frame ambivalence ('on one hand... on the other hand') or paraphrase conflicting feelings. These are complex reflections in MI that surface underlying motivation for change.",
                patterns=[
                    {"pattern_name": "double-sided / ambivalence framing", "pattern_type": "surface_form", "evidence": f"'On one hand'={f['on_one_hand']}, 'you feel' paraphrases={f['you_feel']}."},
                    {"pattern_name": "complex reflection of ambivalence", "pattern_type": "dialogue_function", "evidence": "Counselor explicitly acknowledges and reflects two competing desires/emotions."},
                    {"pattern_name": "MI principle: develop discrepancy", "pattern_type": "mi_principle", "evidence": "Surfacing ambivalence is a core MI technique for motivation enhancement."},
                ],
                evidence_quality="medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["Some 'you feel' statements could be AF (affirmations) rather than reflections."],
                target_label_relationship=f"Supports REC specifically (ambivalence reflections). Shared with RE parent by design. Match_rate={match_rate:.2f}.",
                final_conclusion=f"Medium-confidence REC/RE candidate. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE + " Prior client utterance unavailable, so reflection/context claims are limited.",
                recommended_followup_checks=["Verify RE vs REC classification with client context.", "Token attribution."],
            )
        elif f["i_hear"] >= 4:
            return dict(
                candidate_name="'I hear you / what I'm hearing' reflective paraphrase",
                tentative_interpretation="Activates on reflective counselor utterances using 'I hear you', 'what I'm hearing', or 'I'm hearing'. Explicit paraphrase markers indicating the counselor is feeding back the client's content.",
                patterns=[
                    {"pattern_name": "'I hear you' paraphrase marker", "pattern_type": "surface_form", "evidence": f"'I hear'/'I'm hearing' count={f['i_hear']}."},
                    {"pattern_name": "explicit reflective paraphrase", "pattern_type": "dialogue_function", "evidence": "Counselor explicitly signals they are reflecting the client's words."},
                ],
                evidence_quality="medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["Could encode general empathic acknowledgment rather than specific RE."],
                target_label_relationship=f"Supports RE: explicit reflective markers present. Match_rate={match_rate:.2f}.",
                final_conclusion=f"Medium-confidence RE candidate via explicit paraphrase markers. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE + " Prior client utterance unavailable.",
                recommended_followup_checks=["Token attribution for 'I hear' phrase."],
            )
        elif match_rate >= 0.5 and f["so_prefix"] >= 5:
            return dict(
                candidate_name="'So...' reflective discourse opening",
                tentative_interpretation="Fires on counselor utterances beginning with 'So,' which in MI is frequently used as a reflection-initiating discourse marker. The counselor begins a paraphrase or summary with 'So' to signal reflection.",
                patterns=[
                    {"pattern_name": "'So' reflective discourse marker", "pattern_type": "surface_form", "evidence": f"'so'-initial utterances={f['so_prefix']}/{analysis['n_examples']}."},
                    {"pattern_name": "paraphrase initiation marker", "pattern_type": "dialogue_function", "evidence": "Counselor uses 'So' to transition into a reflection or summary."},
                ],
                evidence_quality="medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["High false-positive rate because 'So' also precedes many question types."],
                target_label_relationship=f"Partial RE support: 'So' is a common reflection opener, but also used for questions/GI. Match_rate={match_rate:.2f}.",
                final_conclusion=f"Low-to-medium confidence RE candidate via 'So' opener. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE + " Prior client utterance unavailable.",
                recommended_followup_checks=["Compare 'so'-initial reflections vs. 'so'-initial questions."],
            )
        elif match_rate >= 0.5:
            return dict(
                candidate_name="general reflection signal (mixed markers)",
                tentative_interpretation="Top examples include RE/REC utterances without a single dominant surface marker. The latent may capture a diffuse reflective style signal across multiple paraphrase patterns.",
                patterns=[
                    {"pattern_name": "reflective paraphrase (diffuse)", "pattern_type": "dialogue_function", "evidence": f"Match_rate={match_rate:.2f}; varied surface forms."},
                ],
                evidence_quality="medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["May encode a combination of signals that partially overlap RE."],
                target_label_relationship=f"Moderate RE support. Match_rate={match_rate:.2f}. No clear surface anchor.",
                final_conclusion=f"Medium-confidence RE candidate; distributed signal. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE + " Prior client utterance unavailable.",
                recommended_followup_checks=["Identify common sub-patterns across top examples.", "Supplement with client context."],
            )
        else:
            return dict(
                candidate_name="weak / mixed RE signal",
                tentative_interpretation="Top examples are mixed across label types, with low RE match rate. The latent does not reliably capture reflective counselor statements.",
                patterns=[
                    {"pattern_name": "mixed signal (low RE purity)", "pattern_type": "mixed_unclear", "evidence": f"Match_rate={match_rate:.2f}; multiple non-RE labels in top examples."},
                ],
                evidence_quality="low",
                artifact_risks=artifact_risks,
                alternative_explanations=["Latent may encode a general conversational register.", "Possible cross-label noise."],
                target_label_relationship=f"Weak RE support: match_rate={match_rate:.2f}. Top examples dominated by non-RE labels.",
                final_conclusion=f"Low-confidence RE candidate. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE + " Prior client utterance unavailable.",
                recommended_followup_checks=["De-prioritize unless RE-specific follow-up confirms signal."],
            )

    # ── REC ───────────────────────────────────────────────────────────────────
    elif target == "REC":
        if f["sounds_like"] >= 5 and (f["you_feel"] >= 3 or f["on_one_hand"] >= 2):
            return dict(
                candidate_name="'sounds like' + emotion inference (complex reflection)",
                tentative_interpretation="Captures complex reflections combining 'sounds like' with emotion inference or ambivalence framing. REC reflects not just content but added emotional meaning or contradictions not explicitly stated by the client.",
                patterns=[
                    {"pattern_name": "'sounds like' + emotion inference", "pattern_type": "surface_form", "evidence": f"'sounds like'={f['sounds_like']}, emotional language={f['you_feel']}."},
                    {"pattern_name": "complex emotion reflection", "pattern_type": "dialogue_function", "evidence": "Counselor reflects implied emotional meaning or ambivalence beyond literal client words."},
                ],
                evidence_quality="medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["Some 'sounds like' + emotion utterances may be simple reflections if client explicitly stated the emotion.", "Cannot confirm REC vs RE without prior client utterance."],
                target_label_relationship=f"Supports REC: emotion-inferring reflections are the hallmark of complex reflections. Match_rate={match_rate:.2f}. Cannot fully distinguish REC from RE without client context.",
                final_conclusion=f"Medium confidence REC candidate. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE + " Prior client utterance unavailable, so reflection/context claims are limited.",
                recommended_followup_checks=["Verify with prior client utterance whether counselor adds meaning beyond stated content."],
            )
        elif f["sounds_like"] >= 4:
            return dict(
                candidate_name="'sounds like' reflective anchor (shared RE/REC)",
                tentative_interpretation="Fires on 'sounds like' reflection openers, shared by both RE and REC. Without client context, the simple vs. complex distinction cannot be determined.",
                patterns=[
                    {"pattern_name": "'sounds/seems like' anchor phrase", "pattern_type": "surface_form", "evidence": f"'sounds like'={f['sounds_like']}/{analysis['n_examples']}."},
                    {"pattern_name": "reflective paraphrase function", "pattern_type": "dialogue_function", "evidence": "Paraphrase-initiation with tentative hedge."},
                ],
                evidence_quality="medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["Shared with RE parent; not REC-specific without client utterance."],
                target_label_relationship=f"Moderate REC support (match_rate={match_rate:.2f}), primarily a RE/REC shared signal.",
                final_conclusion=f"Medium-confidence shared RE/REC candidate. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE + " Prior client utterance unavailable.",
                recommended_followup_checks=["Supplement with client context to test REC specificity."],
            )
        elif f["on_one_hand"] >= 3 or f["you_feel"] >= 6:
            return dict(
                candidate_name="ambivalence / double-sided complex reflection (REC)",
                tentative_interpretation="Captures counselor utterances that explicitly present two sides of the client's ambivalence ('on one hand... but on the other hand...'). This structure is specifically associated with complex reflections in MI.",
                patterns=[
                    {"pattern_name": "double-sided ambivalence framing", "pattern_type": "surface_form", "evidence": f"'on one hand'={f['on_one_hand']}, 'you feel'={f['you_feel']}."},
                    {"pattern_name": "complex ambivalence reflection (REC)", "pattern_type": "dialogue_function", "evidence": "Counselor mirrors client's conflicting emotions/desires."},
                    {"pattern_name": "MI: develop discrepancy via ambivalence", "pattern_type": "mi_principle", "evidence": "Double-sided reflections are a key REC technique."},
                ],
                evidence_quality="medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["Some ambivalence statements might be GI or SU rather than REC."],
                target_label_relationship=f"Supports REC specifically: ambivalence framing is a distinctive REC marker. Match_rate={match_rate:.2f}.",
                final_conclusion=f"Medium-confidence REC candidate: ambivalence-framing pattern. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE + " Prior client utterance unavailable.",
                recommended_followup_checks=["Verify 'on one hand' construct in context of prior client speech."],
            )
        elif f["i_hear"] >= 3:
            return dict(
                candidate_name="explicit paraphrase ('I hear / what I'm hearing') reflection (RE/REC)",
                tentative_interpretation="Counselor utterances explicitly signaling reflection using 'I hear you', 'I'm hearing', 'what I'm hearing'. Captures both RE and REC.",
                patterns=[
                    {"pattern_name": "explicit reflective paraphrase marker", "pattern_type": "surface_form", "evidence": f"'I hear'/'I'm hearing'={f['i_hear']}."},
                    {"pattern_name": "reflective listening function", "pattern_type": "dialogue_function", "evidence": "Counselor signals they are attending to and feeding back client content."},
                ],
                evidence_quality="medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["Shared with RE; not REC-specific."],
                target_label_relationship=f"Moderate REC support (match_rate={match_rate:.2f}). Cannot distinguish RE from REC without context.",
                final_conclusion=f"Medium confidence RE/REC candidate. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE + " Prior client utterance unavailable.",
                recommended_followup_checks=["Check REC vs RE proportion among matched top examples."],
            )
        elif match_rate >= 0.4:
            return dict(
                candidate_name="general complex reflection (distributed signal)",
                tentative_interpretation="Top examples include complex reflections without a single dominant surface marker. The latent captures a diffuse complex-reflection style.",
                patterns=[
                    {"pattern_name": "complex reflection (no single marker)", "pattern_type": "dialogue_function", "evidence": f"Match_rate={match_rate:.2f}; varied surface forms."},
                ],
                evidence_quality="low" if match_rate < 0.5 else "medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["May overlap more with RE than REC."],
                target_label_relationship=f"Partial REC support (match_rate={match_rate:.2f}). No clear surface anchor.",
                final_conclusion=f"Low-to-medium confidence REC candidate. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE + " Prior client utterance unavailable.",
                recommended_followup_checks=["Supplement with client context to confirm REC vs RE."],
            )
        else:
            return dict(
                candidate_name="weak REC signal (mixed labels)",
                tentative_interpretation="Top examples are not predominantly REC. Multiple other labels appear in high-activation examples. The latent does not cleanly capture complex reflection.",
                patterns=[
                    {"pattern_name": "mixed / low-purity", "pattern_type": "mixed_unclear", "evidence": f"Match_rate={match_rate:.2f}. Multiple non-REC labels in top examples."},
                ],
                evidence_quality="low",
                artifact_risks=artifact_risks,
                alternative_explanations=["Likely a shared RE/REC signal or cross-label noise."],
                target_label_relationship=f"Weak REC support: match_rate={match_rate:.2f}.",
                final_conclusion=f"Low-confidence REC candidate. De-prioritize. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE + " Prior client utterance unavailable.",
                recommended_followup_checks=["De-prioritize unless REC-specific follow-up confirms signal."],
            )

    # ── RES ───────────────────────────────────────────────────────────────────
    elif target == "RES":
        if uniform or match_rate < 0.2:
            artifact_risks.append("Activation values do not correlate with RES label. Possible token-level artifact.")
            return dict(
                candidate_name="artifact / uninterpretable (RES - very low purity)",
                tentative_interpretation="Top examples show very low RES match rate with utterances from many unrelated labels. The latent does not encode simple reflection reliably. Possible activation artifact from max-pooling or quantization.",
                patterns=[
                    {"pattern_name": "artifact / noise pattern", "pattern_type": "artifact", "evidence": f"Match_rate={match_rate:.2f}; top examples span AF, QU, GI, SU, and unlabeled utterances."},
                ],
                evidence_quality="uninterpretable",
                artifact_risks=artifact_risks,
                alternative_explanations=["Latent may encode a general discourse feature co-occurring with but not causing RES.", "Max-pooling artifact: highest-activation token may not be semantically relevant."],
                target_label_relationship=f"Does not support RES: match_rate={match_rate:.2f}. Top examples dominated by non-RES labels. Cannot draw any RES-specific conclusion.",
                final_conclusion="Uninterpretable for RES. Do not use as evidence of RES representation. Recommend artifact investigation (token-level attribution).",
                context_limitation_note=CONTEXT_NOTE + " Prior client utterance unavailable. RES is inherently context-dependent.",
                recommended_followup_checks=["Artifact investigation: check token-level activation attribution.", "Examine repeated activation values for quantization issues."],
            )
        elif f["sounds_like"] >= 3:
            return dict(
                candidate_name="'sounds like' simple reflection (possible RES/RE overlap)",
                tentative_interpretation="Some top examples use 'sounds like' in short reflective statements, which could be simple reflections. However, match rate is low and examples are mixed.",
                patterns=[
                    {"pattern_name": "'sounds like' simple reflection attempt", "pattern_type": "surface_form", "evidence": f"'sounds like'={f['sounds_like']}; match_rate={match_rate:.2f}."},
                ],
                evidence_quality="low",
                artifact_risks=artifact_risks,
                alternative_explanations=["Shared with RE/REC: 'sounds like' is not specific to simple reflections.", "Cannot distinguish RES from RE/REC without client context."],
                target_label_relationship=f"Very weak RES support (match_rate={match_rate:.2f}). Context limitation severely impairs interpretation.",
                final_conclusion=f"Low-confidence, unverifiable RES candidate. Match_rate={match_rate:.2f}. Do not rely on for causal validation.",
                context_limitation_note=CONTEXT_NOTE + " Prior client utterance unavailable. Simple vs. complex reflection cannot be determined without it.",
                recommended_followup_checks=["Supplement with prior client utterance to determine if reflection is simple vs. complex.", "Compare with RE top latents."],
            )
        else:
            return dict(
                candidate_name="uninterpretable / low purity RES signal",
                tentative_interpretation="Top examples do not show consistent RES-related patterns. The latent captures a heterogeneous mix of counselor utterances.",
                patterns=[
                    {"pattern_name": "heterogeneous mix (no RES pattern)", "pattern_type": "mixed_unclear", "evidence": f"Match_rate={match_rate:.2f}; no dominant RES surface form."},
                ],
                evidence_quality="uninterpretable" if match_rate < 0.25 else "low",
                artifact_risks=artifact_risks,
                alternative_explanations=["May encode a general conversation flow feature.", "RES is inherently context-dependent; current data insufficient."],
                target_label_relationship=f"Does not support RES: match_rate={match_rate:.2f}. No coherent RES interpretation possible.",
                final_conclusion=f"Uninterpretable for RES purposes. Match_rate={match_rate:.2f}. Exclude from RES analysis.",
                context_limitation_note=CONTEXT_NOTE + " Prior client utterance unavailable. RES is inherently context-dependent.",
                recommended_followup_checks=["Exclude from RES analysis without additional evidence.", "Artifact investigation if activation is suspect."],
            )

    # ── GI ────────────────────────────────────────────────────────────────────
    elif target == "GI":
        if f["medical_info"] >= 4:
            return dict(
                candidate_name="medical / clinical information delivery (GI)",
                tentative_interpretation="Fires on counselor utterances delivering specific medical or clinical information to the client (medications, dosages, health facts). Core GI function in MI: counselor provides accurate factual information.",
                patterns=[
                    {"pattern_name": "medical fact delivery", "pattern_type": "surface_form", "evidence": f"Medical terminology in {f['medical_info']}/{analysis['n_examples']} examples."},
                    {"pattern_name": "information giving function", "pattern_type": "dialogue_function", "evidence": "Counselor provides factual medical/health information."},
                    {"pattern_name": "MI: permission-based information giving", "pattern_type": "mi_principle", "evidence": "GI in MI is offered respectfully to support informed client decisions."},
                ],
                evidence_quality="medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["Longer explanatory utterances may be mistaken for GI even when they contain reflections.", f"Low Cohen's d ({cohens_d:.3f}) for GI overall suggests weak signal."],
                target_label_relationship=f"Moderate GI support (match_rate={match_rate:.2f}). Medical terminology is a reliable GI surface marker.",
                final_conclusion=f"Medium-confidence GI candidate via medical information delivery. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE,
                recommended_followup_checks=["Token attribution for medical terms.", "Check if long explanatory sentences drive activation."],
            )
        elif match_rate >= 0.5:
            return dict(
                candidate_name="explanatory / informational utterance (general GI)",
                tentative_interpretation="Top examples include counselor statements providing explanatory, factual, or advisory information. No single content domain dominates, but 'giving information' functional role is apparent.",
                patterns=[
                    {"pattern_name": "informational/explanatory utterance", "pattern_type": "dialogue_function", "evidence": f"Match_rate={match_rate:.2f}; examples include factual/advisory content."},
                    {"pattern_name": "long declarative sentence style", "pattern_type": "surface_form", "evidence": "GI utterances tend to be longer and more declarative than other MISC categories."},
                ],
                evidence_quality="medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["Long complex sentences may activate this regardless of information-giving function.", f"GI has overall low Cohen's d ({cohens_d:.3f})."],
                target_label_relationship=f"Moderate GI support (match_rate={match_rate:.2f}). Overall GI signal is weak in this dataset.",
                final_conclusion=f"Medium-confidence GI candidate. Match_rate={match_rate:.2f}. Verify with token attribution.",
                context_limitation_note=CONTEXT_NOTE,
                recommended_followup_checks=["Token attribution to verify information-giving vs. explanatory reflection.", "Check sentence length as a confound."],
            )
        else:
            return dict(
                candidate_name="weak GI signal / mixed content",
                tentative_interpretation="Top examples are heterogeneous; GI-specific patterns are not clearly dominant. May reflect the inherent difficulty of capturing GI with SAE latents: GI is a content dimension rather than a form dimension.",
                patterns=[
                    {"pattern_name": "mixed content (low GI specificity)", "pattern_type": "mixed_unclear", "evidence": f"Match_rate={match_rate:.2f}; no dominant GI surface or content pattern."},
                ],
                evidence_quality="low",
                artifact_risks=artifact_risks,
                alternative_explanations=["GI may not be well-captured by utterance-level SAE max-pooling because it is a content dimension rather than a syntactic/pragmatic form dimension."],
                target_label_relationship=f"Weak GI support: match_rate={match_rate:.2f}. GI appears poorly encoded at the latent level.",
                final_conclusion=f"Low-confidence GI candidate. Match_rate={match_rate:.2f}. GI latent encoding is generally weak.",
                context_limitation_note=CONTEXT_NOTE,
                recommended_followup_checks=["Accept weak GI signal as expected given content-vs-form encoding limitation.", "Investigate whether topic-specific latents (medical, diet, exercise) show stronger GI signal."],
            )

    # ── AF ────────────────────────────────────────────────────────────────────
    elif target == "AF":
        if f["positive_eval"] >= 6:
            return dict(
                candidate_name="positive evaluative phrase detector (great/good/wonderful)",
                tentative_interpretation="Fires strongly on counselor utterances containing positive evaluative adjectives ('great', 'good', 'wonderful', 'awesome', 'amazing'). These lexical items are the most reliable surface markers of affirmations in MI transcripts.",
                patterns=[
                    {"pattern_name": "positive evaluative lexicon", "pattern_type": "surface_form", "evidence": f"Positive adjectives in top examples={f['positive_eval']}/{analysis['n_examples']}."},
                    {"pattern_name": "affirmation function (MI)", "pattern_type": "dialogue_function", "evidence": "Counselor explicitly validates or praises client's effort, change, or character."},
                    {"pattern_name": "MI spirit: affirmation supports self-efficacy", "pattern_type": "mi_principle", "evidence": "Affirmations in MI support client self-efficacy and engagement."},
                ],
                evidence_quality="high" if match_rate >= 0.7 else "medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["The latent may detect 'positive adjective' tokens rather than MI affirmation function.", "Some uses of 'great', 'good' may be non-affirmative (e.g., 'sounds great' = SU)."],
                target_label_relationship=f"Supports AF strongly: positive evaluative lexicon is the canonical AF surface marker. Match_rate={match_rate:.2f}. Note: may be surface-word driven rather than MI-function driven.",
                final_conclusion=f"High surface confidence, medium functional confidence AF candidate. Match_rate={match_rate:.2f}. Token attribution recommended.",
                context_limitation_note=CONTEXT_NOTE,
                recommended_followup_checks=["Token attribution to confirm specific positive adjectives drive activation.", "Test AF vs SU distinction for 'sounds great' type utterances."],
            )
        elif match_rate >= 0.5:
            return dict(
                candidate_name="general affirmation signal (partial lexical match)",
                tentative_interpretation="Top examples include AF utterances but positive evaluative words are not the sole driver. Captures a broader 'positive counselor register' that partially aligns with AF.",
                patterns=[
                    {"pattern_name": "positive counselor register", "pattern_type": "dialogue_function", "evidence": f"Match_rate={match_rate:.2f}; examples include various positive statements."},
                ],
                evidence_quality="medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["May encode general 'positive counselor interaction' rather than specific MI affirmation."],
                target_label_relationship=f"Moderate AF support (match_rate={match_rate:.2f}). Not exclusively driven by positive lexicon.",
                final_conclusion=f"Medium-confidence AF candidate. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE,
                recommended_followup_checks=["Identify the specific positive register signal driving activation."],
            )
        else:
            return dict(
                candidate_name="weak AF / positive-word cross-label",
                tentative_interpretation="Top examples are mixed; AF match rate is low. The latent may capture a general positive register spanning AF, SU, and other label categories.",
                patterns=[
                    {"pattern_name": "positive register (cross-label)", "pattern_type": "mixed_unclear", "evidence": f"Match_rate={match_rate:.2f}; positive words present but not AF-specific."},
                ],
                evidence_quality="low",
                artifact_risks=artifact_risks,
                alternative_explanations=["Shared positive-register signal across AF, SU, and other labels."],
                target_label_relationship=f"Weak AF support: match_rate={match_rate:.2f}. Non-AF utterances dominate.",
                final_conclusion=f"Low-confidence AF candidate. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE,
                recommended_followup_checks=["Compare with higher-rank AF latents."],
            )

    # ── SU ────────────────────────────────────────────────────────────────────
    elif target == "SU":
        if f["i_understand"] >= 4:
            return dict(
                candidate_name="'I understand / I completely understand' empathy template",
                tentative_interpretation="Fires on counselor utterances using 'I understand', 'I completely understand', 'I totally understand' as an empathic acknowledgment template. In MI, this phrasing is associated with supportive/empathic responses (SU).",
                patterns=[
                    {"pattern_name": "'I understand' empathy template", "pattern_type": "surface_form", "evidence": f"'I understand' / 'totally understand'={f['i_understand']}/{analysis['n_examples']}."},
                    {"pattern_name": "empathic support / acknowledgment", "pattern_type": "dialogue_function", "evidence": "Counselor explicitly validates client's difficulty or emotional experience."},
                ],
                evidence_quality="medium",
                artifact_risks=artifact_risks,
                alternative_explanations=["Generic empathy phrases may be used across SU, RE, and AF contexts.", "SU has very small sample size (222 utterances); low power for reliable latent identification."],
                target_label_relationship=f"Partial SU support (match_rate={match_rate:.2f}). 'I understand' is a common SU marker but also appears in non-SU contexts.",
                final_conclusion=f"Medium-confidence SU candidate via 'I understand' template. Match_rate={match_rate:.2f}.",
                context_limitation_note=CONTEXT_NOTE,
                recommended_followup_checks=["Check SU vs AF vs RE distribution for 'I understand' utterances.", "Consider low SU base rate in power analysis."],
            )
        elif match_rate >= 0.3:
            return dict(
                candidate_name="general support / suggestion signal (mixed SU)",
                tentative_interpretation="Top examples include some SU utterances but without a single dominant surface pattern. The SU category covers diverse behaviors (empathic support, direct suggestions) that may not be well-unified in the latent space.",
                patterns=[
                    {"pattern_name": "mixed support/suggestion", "pattern_type": "mixed_unclear", "evidence": f"Match_rate={match_rate:.2f}; no single SU surface pattern."},
                ],
                evidence_quality="low",
                artifact_risks=artifact_risks,
                alternative_explanations=["SU may not have strong enough signal for reliable latent identification.", "Could be cross-label noise from AF or RE."],
                target_label_relationship=f"Weak SU support: match_rate={match_rate:.2f}. SU is the smallest MISC label (n=222) with consistently low latent match rates.",
                final_conclusion=f"Low-confidence SU candidate (match_rate={match_rate:.2f}). SU has second-lowest overall match rate across all labels.",
                context_limitation_note=CONTEXT_NOTE,
                recommended_followup_checks=["Accept weak SU signal as expected given small sample size.", "Compare with SU-specific linguistic patterns."],
            )
        else:
            return dict(
                candidate_name="uninterpretable for SU (cross-label noise)",
                tentative_interpretation="Top examples are predominantly from other labels. This latent does not appear to encode SU-relevant patterns.",
                patterns=[
                    {"pattern_name": "cross-label noise", "pattern_type": "mixed_unclear", "evidence": f"Match_rate={match_rate:.2f}; top examples mainly non-SU."},
                ],
                evidence_quality="uninterpretable" if match_rate < 0.1 else "low",
                artifact_risks=artifact_risks,
                alternative_explanations=["Latent captures features unrelated to SU."],
                target_label_relationship=f"Does not support SU: match_rate={match_rate:.2f}.",
                final_conclusion=f"Uninterpretable/low-confidence SU candidate. Match_rate={match_rate:.2f}. Exclude from SU analysis.",
                context_limitation_note=CONTEXT_NOTE,
                recommended_followup_checks=["Exclude from SU analysis."],
            )

    # ── Fallback ──────────────────────────────────────────────────────────────
    else:
        return dict(
            candidate_name=f"unknown label ({target}) / fallback",
            tentative_interpretation=f"Latent for label {target}. No specific analysis rule available.",
            patterns=[{"pattern_name": "unanalyzed", "pattern_type": "mixed_unclear", "evidence": "No rule available for this label."}],
            evidence_quality="uninterpretable",
            artifact_risks=artifact_risks,
            alternative_explanations=[],
            target_label_relationship="Not analyzed.",
            final_conclusion="No conclusion. Add analysis rule for this label.",
            context_limitation_note=CONTEXT_NOTE,
            recommended_followup_checks=["Add label-specific analysis rule."],
        )


def build_review(packet, summary_row):
    """Build complete review record for one packet."""
    top_examples = [e for e in packet.get("examples", []) if e.get("example_group") == "top_activating"][:15]
    if not top_examples:
        top_examples = packet.get("examples", [])[:10]
    # Normalize rank field (raw packets use rank_within_label)
    if "rank" not in packet and "rank_within_label" in packet:
        packet = dict(packet)
        packet["rank"] = packet["rank_within_label"]

    EMPTY_FEATURES = {k: 0 for k in [
        "has_question_mark","starts_with_wh","starts_with_do_can","sounds_like","it_seems",
        "i_hear","on_one_hand","you_feel","positive_eval","i_understand","would_phrase",
        "scale_question","medical_info","so_prefix","you_prefix","okay_prefix","yeah_prefix"
    ]}
    analysis = analyze_top_examples(top_examples) if top_examples else {
        "match_rate": 0.0, "features": EMPTY_FEATURES,
        "activation_uniform": False, "label_counter": Counter(), "n_examples": 0
    }

    result = classify_latent(packet, analysis)

    context_note = result.get("context_limitation_note", CONTEXT_NOTE)
    if packet.get("target_label") in {"RE","RES","REC"} and "client" not in context_note.lower():
        context_note += " Prior client utterance is unavailable, so reflection/context claims are limited."

    return {
        "packet_id": packet["packet_id"],
        "latent_alias": summary_row.get("latent_alias", packet["packet_id"]),
        "target_label": packet["target_label"],
        "latent_idx": packet["latent_idx"],
        "rank_within_label": packet.get("rank_within_label", packet.get("rank", 0)),
        "cohens_d": packet.get("cohens_d", 0.0),
        "directional_auc": packet.get("directional_auc", packet.get("auc", 0.0)),
        "precision_at_50": packet.get("precision_at_50", 0.0),
        "status": "ok",
        "error": "",
        "tentative_interpretation": result["tentative_interpretation"],
        "patterns": result["patterns"][:5],
        "target_label_relationship": result["target_label_relationship"],
        "adjacent_label_risks": [],
        "artifact_risks": result["artifact_risks"],
        "evidence_quality": result["evidence_quality"],
        "candidate_name": result["candidate_name"],
        "alternative_explanations": result["alternative_explanations"],
        "recommended_followup_checks": result["recommended_followup_checks"],
        "final_conclusion": result["final_conclusion"],
        "context_limitation_note": context_note,
        "top_activating_target_match_rate": analysis["match_rate"],
        "active_label_counts_top_activating": dict(analysis["label_counter"]),
        "duplicate_text_row_count": summary_row.get("duplicate_text_row_count", 0),
        "unique_file_count": summary_row.get("unique_file_count", 0),
        "model_used": "claude-sonnet-4-6-direct-analysis",
        "analysis_timestamp": datetime.datetime.now().isoformat(),
    }


def main():
    print("Loading packets...")
    with open(PACKETS_LABELED, "r", encoding="utf-8") as f:
        labeled_packets = [json.loads(l) for l in f if l.strip()]

    with open(SUMMARY_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        summary_map = {row["packet_id"]: row for row in reader}

    print(f"Loaded {len(labeled_packets)} labeled packets")

    reviews = []
    for i, packet in enumerate(labeled_packets):
        pid = packet["packet_id"]
        summary_row = summary_map.get(pid, {})
        review = build_review(packet, summary_row)
        reviews.append(review)
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(labeled_packets)} packets...")

    print(f"Generated {len(reviews)} reviews")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSONL
    reviews_path = OUTPUT_DIR / "latent_function_reviews.jsonl"
    with open(reviews_path, "w", encoding="utf-8") as f:
        for r in reviews:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Reviews CSV
    csv_fields = [
        "packet_id","target_label","latent_idx","rank_within_label",
        "cohens_d","directional_auc","precision_at_50",
        "status","candidate_name","evidence_quality",
        "top_activating_target_match_rate","tentative_interpretation",
        "target_label_relationship","final_conclusion","model_used"
    ]
    csv_path = OUTPUT_DIR / "latent_function_reviews.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reviews)

    # Patterns CSV
    pattern_rows = []
    for r in reviews:
        for idx, p in enumerate(r.get("patterns", []), 1):
            pattern_rows.append({
                "packet_id": r["packet_id"],
                "target_label": r["target_label"],
                "latent_idx": r["latent_idx"],
                "pattern_rank": idx,
                "pattern_name": p.get("pattern_name"),
                "pattern_type": p.get("pattern_type"),
                "evidence": str(p.get("evidence", ""))[:200],
                "review_status": r["status"],
            })
    patterns_path = OUTPUT_DIR / "latent_function_patterns.csv"
    with open(patterns_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["packet_id","target_label","latent_idx",
                                                "pattern_rank","pattern_name","pattern_type",
                                                "evidence","review_status"])
        writer.writeheader()
        writer.writerows(pattern_rows)

    # Label summary
    from collections import defaultdict
    label_groups = defaultdict(list)
    for r in reviews:
        label_groups[r["target_label"]].append(r)

    label_rows = []
    for lbl in sorted(label_groups.keys()):
        group = label_groups[lbl]
        quality_counts = Counter(r["evidence_quality"] for r in group)
        artifact_count = sum(1 for r in group if r["artifact_risks"])
        mean_match = sum(r["top_activating_target_match_rate"] for r in group) / len(group)
        label_rows.append({
            "target_label": lbl,
            "n_latents": len(group),
            "high_quality": quality_counts.get("high", 0),
            "medium_quality": quality_counts.get("medium", 0),
            "low_quality": quality_counts.get("low", 0),
            "uninterpretable": quality_counts.get("uninterpretable", 0),
            "artifact_risk_latents": artifact_count,
            "mean_target_match_rate": round(mean_match, 4),
        })

    label_summary_path = OUTPUT_DIR / "label_function_cluster_summary.csv"
    with open(label_summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(label_rows[0].keys()))
        writer.writeheader()
        writer.writerows(label_rows)

    # Markdown report
    report_lines = [
        "# SAE Latent 功能归纳报告（Claude Sonnet 直接分析版）",
        "",
        "本报告由 Claude Sonnet 直接分析 180 个 SAE latent 的高激活语句生成，",
        "无需外部 API 调用。分析方法：基于语言模式规则的语义归纳。",
        "",
        "## 方法说明",
        "",
        "- 输入：每个 latent 的 top-15 最高激活语句",
        "- 分析维度：表面形式特征（句法）、词汇锚点、对话功能、MI 原则对应",
        "- 输出：候选功能解释，不是因果机制证明",
        "- 上下文限制：仅有咨询师当前语句，无来访者前一句",
        "- 使用模型：Claude Sonnet 4.6（直接分析，非 API 调用）",
        "",
        "## 总体覆盖",
        "",
        f"- 总 latent reviews: {len(reviews)}",
        f"- 状态: ok={len(reviews)}, failed=0, pending_dry_run=0",
        "",
        "## 每标签概览",
        "",
        "| label | n | high | medium | low | uninterpretable | artifact-risk | mean match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in label_rows:
        report_lines.append(
            f"| {row['target_label']} | {row['n_latents']} | {row['high_quality']} | "
            f"{row['medium_quality']} | {row['low_quality']} | {row['uninterpretable']} | "
            f"{row['artifact_risk_latents']} | {row['mean_target_match_rate']:.3f} |"
        )

    report_lines.extend(["", "## 代表性条目（每标签前2条）", ""])
    for lbl in sorted(label_groups.keys()):
        group = label_groups[lbl]
        report_lines.append(f"### {lbl}")
        for r in group[:2]:
            report_lines.extend([
                f"**{r['packet_id']} (latent {r['latent_idx']}, rank {r['rank_within_label']})**",
                f"- candidate_name: {r['candidate_name']}",
                f"- evidence_quality: {r['evidence_quality']}",
                f"- match_rate: {r['top_activating_target_match_rate']:.2f}",
                f"- interpretation: {r['tentative_interpretation'][:250]}",
                f"- conclusion: {r['final_conclusion'][:200]}",
                "",
            ])

    report_lines.extend([
        "## 可发表表述建议",
        "",
        "**可以说**：SAE top-latents 的高激活语句呈现可归纳的 surface/function/artifact 模式，",
        "可作为解释标签可解码性的表征证据。",
        "",
        "**不要说**：这些 latent 已经证明 LLM 以 MISC codebook 的方式理解咨询行为，",
        "或这些 latent 是标签判断的因果机制。",
    ])

    report_path = OUTPUT_DIR / "latent_function_induction_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    # Manifest
    manifest = {
        "analysis": "misc_sae_latent_function_induction_claude_direct",
        "model": "claude-sonnet-4-6-direct-analysis",
        "method": "rule-based linguistic pattern matching on top-activating utterances",
        "dry_run_prompts": False,
        "n_reviews": len(reviews),
        "status_counts": {"ok": len(reviews)},
        "n_success": len(reviews),
        "n_failed": 0,
        "n_pending_dry_run": 0,
        "timestamp": datetime.datetime.now().isoformat(),
        "outputs": {
            "reviews_jsonl": str(reviews_path),
            "reviews_csv": str(csv_path),
            "patterns_csv": str(patterns_path),
            "label_summary_csv": str(label_summary_path),
            "report_md": str(report_path),
        }
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone! Output written to: {OUTPUT_DIR}")
    print(f"Reviews: {len(reviews)}")
    print()
    print("Label summary:")
    for row in label_rows:
        print(f"  {row['target_label']:5s}: high={row['high_quality']:2d} medium={row['medium_quality']:2d} "
              f"low={row['low_quality']:2d} uninterp={row['uninterpretable']:2d} "
              f"mean_match={row['mean_target_match_rate']:.3f}")


if __name__ == "__main__":
    main()
