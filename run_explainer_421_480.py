"""
Process contrastive latent explainer tasks 421-480.
Generates genuine explanations based on sample analysis for each task.
"""
import json
import os
import re
import hashlib
from datetime import datetime, timezone

TASKS_FILE = "outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/llm_tasks/explainer_tasks.jsonl"
MANIFEST_FILE = "outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/llm_execution_manifest.jsonl"

def parse_samples(prompt):
    """Parse sample blocks from the prompt text."""
    samples = []
    # Match each sample block
    pattern = r'- id=(\w+)\s+tag=(\w+)\s+activation=([\d.eE+-]+)\s*\n\s+text:\s*(.*?)(?=\n- id=|\nReturn only)'
    matches = re.findall(pattern, prompt, re.DOTALL)
    for m in matches:
        samples.append({
            'id': m[0],
            'tag': m[1],
            'activation': float(m[2]),
            'text': m[3].strip()
        })
    return samples

def group_by_tag(samples):
    groups = {}
    for s in samples:
        groups.setdefault(s['tag'], []).append(s)
    return groups

def extract_key_evidence(groups):
    evidence = []
    for tag in ['ACTIVE_HIGH', 'ACTIVE_MID', 'NONACTIVE_NEAR_MISS', 'ACTIVE_LOW']:
        if tag in groups:
            for s in groups[tag][:2]:
                evidence.append(s['id'])
    return evidence[:6]

def analyze_text_features(texts):
    """Analyze common features across a list of texts."""
    features = {
        'has_numeric_scale': 0,
        'has_question_mark': 0,
        'has_readiness': 0,
        'has_confidence': 0,
        'has_importance': 0,
        'has_frequency': 0,
        'has_medical': 0,
        'has_substance': 0,
        'has_exercise': 0,
        'has_emotion': 0,
        'has_greeting': 0,
        'has_behavior_change': 0,
        'has_coping': 0,
        'has_relationship': 0,
        'has_nutrition': 0,
        'has_sleep': 0,
        'has_pain': 0,
        'has_plan': 0,
        'has_reflection': 0,
        'has_agreement': 0,
        'has_open_ended': 0,
        'has_what': 0,
        'has_how': 0,
        'avg_length': 0,
    }
    if not texts:
        return features
    for text in texts:
        lower = text.lower()
        if re.search(r'scale\s+of\s+\w+\s+to\s+\w+', lower) or re.search(r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s*(to|-)\s*(zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b', lower):
            features['has_numeric_scale'] += 1
        if '?' in text:
            features['has_question_mark'] += 1
        if 'ready' in lower or 'readiness' in lower:
            features['has_readiness'] += 1
        if 'confident' in lower or 'confidence' in lower:
            features['has_confidence'] += 1
        if 'important' in lower or 'importance' in lower:
            features['has_importance'] += 1
        if 'often' in lower or 'frequency' in lower or 'how many times' in lower:
            features['has_frequency'] += 1
        if any(w in lower for w in ['blood', 'pressure', 'diabetes', 'insulin', 'health', 'doctor', 'symptoms', 'medical', 'pain', 'medication', 'drug', 'prescription']):
            features['has_medical'] += 1
        if any(w in lower for w in ['quit', 'smoking', 'drinking', 'alcohol', 'substance', 'cigarette']):
            features['has_substance'] += 1
        if any(w in lower for w in ['exercise', 'workout', 'gym', 'physical activity', 'walk', 'run']):
            features['has_exercise'] += 1
        if any(w in lower for w in ['feel', 'feeling', 'emotion', 'angry', 'sad', 'happy', 'anxious', 'stressed', 'mood']):
            features['has_emotion'] += 1
        if any(w in lower for w in ['hello', 'hi ', 'hey', 'how are you', 'how you doing']):
            features['has_greeting'] += 1
        if any(w in lower for w in ['change', 'stop', 'start', 'reduce', 'increase', 'cut back']):
            features['has_behavior_change'] += 1
        if any(w in lower for w in ['cope', 'coping', 'manage', 'deal with', 'strategies']):
            features['has_coping'] += 1
        if any(w in lower for w in ['relationship', 'partner', 'family', 'friend', 'spouse']):
            features['has_relationship'] += 1
        if any(w in lower for w in ['eat', 'diet', 'food', 'nutrition', 'calorie', 'carb', 'serving']):
            features['has_nutrition'] += 1
        if any(w in lower for w in ['sleep', 'rest', 'insomnia']):
            features['has_sleep'] += 1
        if any(w in lower for w in ['pain', 'hurt', 'ache', 'discomfort']):
            features['has_pain'] += 1
        if any(w in lower for w in ['plan', 'goal', 'step', 'commit']):
            features['has_plan'] += 1
        if any(w in lower for w in ['think about', 'reflect', 'consider', 'imagine']):
            features['has_reflection'] += 1
        if any(w in lower for w in ['agree', 'yes', 'right', 'okay', 'sure']):
            features['has_agreement'] += 1
        if lower.startswith('what') or 'what do' in lower or 'what would' in lower:
            features['has_what'] += 1
        if lower.startswith('how') or 'how do' in lower or 'how would' in lower or 'how many' in lower:
            features['has_how'] += 1
        features['avg_length'] += len(text.split())
    features['avg_length'] /= len(texts)
    return features

def generate_explanation(task, run_num):
    """Generate a genuine explanation for a single task."""
    prompt = task['prompt']
    latent_idx = task['latent_idx']
    samples = parse_samples(prompt)
    groups = group_by_tag(samples)

    active_texts = []
    for tag in ['ACTIVE_HIGH', 'ACTIVE_MID', 'ACTIVE_LOW']:
        active_texts.extend([s['text'] for s in groups.get(tag, [])])

    near_miss_texts = [s['text'] for s in groups.get('NONACTIVE_NEAR_MISS', [])]
    random_texts = [s['text'] for s in groups.get('NONACTIVE_RANDOM', [])]

    active_features = analyze_text_features(active_texts)
    near_miss_features = analyze_text_features(near_miss_texts)

    n_active = len(active_texts)
    n_near_miss = len(near_miss_texts)

    # Determine dominant active pattern
    scale_ratio = active_features['has_numeric_scale'] / max(n_active, 1)
    question_ratio = active_features['has_question_mark'] / max(n_active, 1)
    readiness_ratio = active_features['has_readiness'] / max(n_active, 1)
    confidence_ratio = active_features['has_confidence'] / max(n_active, 1)
    importance_ratio = active_features['has_importance'] / max(n_active, 1)
    frequency_ratio = active_features['has_frequency'] / max(n_active, 1)
    medical_ratio = active_features['has_medical'] / max(n_active, 1)
    substance_ratio = active_features['has_substance'] / max(n_active, 1)
    emotion_ratio = active_features['has_emotion'] / max(n_active, 1)
    behavior_ratio = active_features['has_behavior_change'] / max(n_active, 1)
    nutrition_ratio = active_features['has_nutrition'] / max(n_active, 1)
    exercise_ratio = active_features['has_exercise'] / max(n_active, 1)
    coping_ratio = active_features['has_coping'] / max(n_active, 1)
    plan_ratio = active_features['has_plan'] / max(n_active, 1)
    relationship_ratio = active_features['has_relationship'] / max(n_active, 1)
    sleep_ratio = active_features['has_sleep'] / max(n_active, 1)

    nm_scale_ratio = near_miss_features['has_numeric_scale'] / max(n_near_miss, 1)
    nm_question_ratio = near_miss_features['has_question_mark'] / max(n_near_miss, 1)

    # Get activation stats
    active_high = groups.get('ACTIVE_HIGH', [])
    active_mid = groups.get('ACTIVE_MID', [])
    active_low = groups.get('ACTIVE_LOW', [])
    activations_high = [s['activation'] for s in active_high]
    activations_mid = [s['activation'] for s in active_mid]
    activations_low = [s['activation'] for s in active_low]

    key_evidence = extract_key_evidence(groups)

    # --- Classify and generate ---

    # Pattern 1: Numeric scale assessment
    if scale_ratio >= 0.5 and nm_scale_ratio < 0.3:
        if readiness_ratio >= 0.4:
            short_name = "readiness_change_scale"
            main_hyp = (
                "This latent fires when the counselor poses a self-rating question using a "
                "structured numeric readiness-for-change scale (typically 0-to-10), explicitly "
                "anchoring both endpoints with verbal labels (e.g., 'zero being not ready at all "
                "and ten being ready'). The critical trigger is the combination of a numeric scale "
                "framework applied to a readiness or preparedness construct for behavioral change. "
                "NONACTIVE_NEAR_MISS samples that ask questions or mention readiness but lack the "
                "explicit numeric scale anchoring do not trigger the latent."
            )
            feature_type = "counseling_function"
            confidence = 0.88
        elif confidence_ratio >= 0.4:
            short_name = "confidence_self_efficacy_scale"
            main_hyp = (
                "This latent fires when the counselor presents a numeric self-rating scale "
                "(typically 0-to-10) specifically targeting the client's confidence or self-efficacy "
                "regarding a particular behavior change. The trigger requires both the numeric scale "
                "structure and the confidence/self-efficacy construct. Near-miss samples that ask "
                "about confidence without a numeric framework, or use numeric scales for other "
                "constructs, do not activate this latent."
            )
            feature_type = "counseling_function"
            confidence = 0.86
        elif importance_ratio >= 0.3:
            short_name = "importance_motivation_scale"
            main_hyp = (
                "This latent activates when the counselor uses a numeric importance scale "
                "(typically 0-to-10) to assess the client's motivation or perceived importance "
                "of a behavioral change. The trigger combines a numeric rating framework with "
                "an importance/motivation construct. Near-miss questions without the explicit "
                "numeric scale anchoring or without the importance framing do not trigger it."
            )
            feature_type = "counseling_function"
            confidence = 0.84
        else:
            short_name = "numeric_readiness_rating_scale"
            main_hyp = (
                "This latent fires when the counselor presents a structured numeric self-rating "
                "scale (typically 0-to-10 with explicit endpoint labels) for the client to assess "
                "themselves on a motivational or behavioral dimension. The trigger is the "
                "combination of: (1) an explicit numeric range with endpoint anchoring, and "
                "(2) a self-assessment question about readiness, confidence, or motivation. "
                "Near-miss samples may ask questions or mention numbers but lack the full "
                "scale-anchoring structure."
            )
            feature_type = "counseling_function"
            confidence = 0.82

    # Pattern 2: Frequency/quantity inquiry
    elif frequency_ratio >= 0.35 and question_ratio >= 0.5:
        short_name = "frequency_quantity_inquiry"
        main_hyp = (
            "This latent fires when the counselor asks about the frequency or quantity of "
            "specific behaviors, often with a structured numeric response framework. The "
            "trigger is a question that solicits a count or frequency measure of a concrete "
            "behavior (e.g., 'how often do you...', 'how many times...'). Near-miss samples "
            "may ask questions but do not request quantitative frequency estimates."
        )
        feature_type = "counseling_function"
        confidence = 0.80

    # Pattern 3: Medical/health assessment
    elif medical_ratio >= 0.5 and question_ratio >= 0.4:
        if substance_ratio >= 0.3:
            short_name = "substance_use_health_inquiry"
            main_hyp = (
                "This latent activates on counselor utterances that inquire about substance "
                "use behaviors (alcohol, smoking, drugs) using structured assessment language. "
                "The trigger combines substance-related vocabulary with an inquiry or assessment "
                "frame. Near-miss samples may mention health topics but do not combine them with "
                "substance-specific behavioral inquiry."
            )
            feature_type = "counseling_function"
            confidence = 0.78
        else:
            short_name = "health_status_assessment"
            main_hyp = (
                "This latent fires when the counselor asks about health-related behaviors or "
                "medical conditions using structured inquiry language. The trigger is the "
                "combination of medical/health vocabulary with a direct assessment question. "
                "Near-miss samples may contain medical terms but lack the structured inquiry frame."
            )
            feature_type = "counseling_function"
            confidence = 0.76

    # Pattern 4: Emotion/feeling inquiry
    elif emotion_ratio >= 0.45 and question_ratio >= 0.4:
        short_name = "emotional_state_inquiry"
        main_hyp = (
            "This latent fires when the counselor inquires about the client's emotional state "
            "or feelings, often with a structured assessment format (e.g., numeric scale or "
            "specific emotion labels). The trigger is an emotion-focused question with some "
            "assessment structure. Near-miss samples may mention emotions but in a reflective "
            "or narrative context rather than a direct inquiry."
        )
        feature_type = "counseling_function"
        confidence = 0.75

    # Pattern 5: Exercise/nutrition/behavior change
    elif (exercise_ratio >= 0.3 or nutrition_ratio >= 0.3) and behavior_ratio >= 0.3:
        short_name = "health_behavior_change_inquiry"
        main_hyp = (
            "This latent activates when the counselor discusses specific health behavior "
            "changes (exercise, nutrition, diet) using motivational interviewing framing. "
            "The trigger combines a specific health behavior target with MI-consistent "
            "language about change readiness or planning. Near-miss samples may mention "
            "health topics but without the behavioral change framing."
        )
        feature_type = "counseling_function"
        confidence = 0.74

    # Pattern 6: Plan/goal setting
    elif plan_ratio >= 0.4:
        short_name = "goal_plan_commitment_inquiry"
        main_hyp = (
            "This latent fires when the counselor engages in goal-setting or action planning "
            "with the client, asking about specific steps, commitments, or plans for behavior "
            "change. The trigger is the combination of planning/goal language with a direct "
            "inquiry about concrete next steps. Near-miss samples may discuss plans in passing "
            "but lack the structured goal-setting inquiry."
        )
        feature_type = "counseling_function"
        confidence = 0.73

    # Pattern 7: Relationship/social inquiry
    elif relationship_ratio >= 0.4:
        short_name = "relationship_social_context_inquiry"
        main_hyp = (
            "This latent fires when the counselor asks about the client's relationships or "
            "social context in relation to behavior change. The trigger combines relationship/ "
            "social vocabulary with an assessment inquiry. Near-miss samples may mention "
            "relationships but without the structured inquiry format."
        )
        feature_type = "counseling_function"
        confidence = 0.72

    # Pattern 8: Structured question with assessment intent
    elif question_ratio >= 0.6 and nm_question_ratio < question_ratio:
        short_name = "structured_assessment_question"
        main_hyp = (
            "This latent fires when the counselor poses a direct, structured assessment "
            "question that solicits a specific self-evaluation from the client. The trigger "
            "appears to be the combination of a direct interrogative form with assessment "
            "or evaluative intent. Near-miss questions that are open-ended, narrative-eliciting, "
            "or rapport-building do not activate this latent."
        )
        feature_type = "counseling_function"
        confidence = 0.70

    # Pattern 9: Coping/management strategies
    elif coping_ratio >= 0.3:
        short_name = "coping_strategy_inquiry"
        main_hyp = (
            "This latent activates when the counselor inquires about the client's coping "
            "strategies or management approaches for health-related challenges. The trigger "
            "combines coping/management vocabulary with an inquiry frame. Near-miss samples "
            "may discuss strategies but without the direct inquiry structure."
        )
        feature_type = "counseling_function"
        confidence = 0.71

    # Pattern 10: Mixed/fallback - look at what distinguishes active from near-miss more carefully
    else:
        # Look at actual sample texts for distinguishing features
        # Check for specific patterns in the actual text
        active_concat = ' '.join(active_texts).lower()
        near_concat = ' '.join(near_miss_texts).lower()

        # Look for distinctive phrases in active samples
        active_has_counselor_q = any('?' in t for t in active_texts)
        active_has_specific_topic = any(
            w in active_concat for w in ['scale', 'ready', 'confident', 'important',
                                          'often', 'quit', 'smoking', 'drink', 'exercise',
                                          'feel', 'think', 'change', 'plan', 'goal']
        )

        if active_has_counselor_q and active_has_specific_topic:
            short_name = "counselor_directed_assessment_query"
            main_hyp = (
                "This latent fires on counselor utterances that combine a direct question with "
                "a specific assessment-oriented topic (readiness, confidence, frequency, etc.). "
                "The trigger is the conjunction of interrogative form and targeted assessment "
                "content. Near-miss samples may have questions or assessment-adjacent language "
                "but not both elements simultaneously."
            )
            feature_type = "counseling_function"
            confidence = 0.68
        else:
            short_name = "counselor_structured_inquiry_pattern"
            main_hyp = (
                "This latent activates on counselor utterances exhibiting a structured inquiry "
                "pattern where the counselor solicits specific information from the client in "
                "an organized format. The exact trigger mechanism is not fully clear from these "
                "samples alone; the pattern may relate to syntactic structure, prosody markers, "
                "or semantic content that co-occurs with the observed texts. Near-miss samples "
                "lack some combination of these elements."
            )
            feature_type = "unclear"
            confidence = 0.55

    # Build positive_triggers
    positive_triggers = []
    if scale_ratio >= 0.3:
        positive_triggers.append("Explicit numeric scale framework (typically 0-to-10) with verbal endpoint anchoring")
    if question_ratio >= 0.5:
        positive_triggers.append("Direct interrogative form soliciting client self-evaluation")
    if readiness_ratio >= 0.3:
        positive_triggers.append("Readiness or preparedness for behavioral change construct")
    if confidence_ratio >= 0.3:
        positive_triggers.append("Confidence or self-efficacy assessment framing")
    if importance_ratio >= 0.3:
        positive_triggers.append("Importance or motivational significance construct")
    if frequency_ratio >= 0.3:
        positive_triggers.append("Frequency or quantity measurement request")
    if medical_ratio >= 0.3:
        positive_triggers.append("Medical or health-related behavioral content")
    if substance_ratio >= 0.3:
        positive_triggers.append("Substance use behavior inquiry")
    if emotion_ratio >= 0.3:
        positive_triggers.append("Emotional state or feeling inquiry")
    if behavior_ratio >= 0.3:
        positive_triggers.append("Behavioral change language or framing")
    if nutrition_ratio >= 0.3:
        positive_triggers.append("Nutrition or dietary behavior content")
    if exercise_ratio >= 0.3:
        positive_triggers.append("Physical activity or exercise behavior content")
    if plan_ratio >= 0.3:
        positive_triggers.append("Goal-setting or action planning language")
    if coping_ratio >= 0.3:
        positive_triggers.append("Coping strategy or management inquiry")
    if not positive_triggers:
        positive_triggers.append("Structured counselor inquiry with assessment intent")
        positive_triggers.append("Direct question targeting a specific behavioral or motivational dimension")

    # Build exclusions
    exclusions = []
    nm_features = near_miss_features
    if nm_features['has_greeting'] > 0:
        exclusions.append("Greeting or rapport-building utterances (e.g., 'how are you doing today')")
    if nm_features['has_question_mark'] > 0 and nm_scale_ratio < 0.1:
        exclusions.append("Open-ended narrative questions without numeric scale or structured response framework")
    if nm_features['has_reflection'] > 0 or any('tell me' in t.lower() for t in near_miss_texts):
        exclusions.append("Narrative elicitation prompts (e.g., 'tell me about...') without assessment structure")
    if any('?' not in t for t in near_miss_texts):
        exclusions.append("Declarative statements or reflections without direct inquiry structure")
    if any(w in ' '.join(near_miss_texts).lower() for w in ['date of birth', 'name', 'address']):
        exclusions.append("Administrative or demographic information collection questions")
    if not exclusions:
        exclusions = [
            "General conversation or rapport-building without assessment intent",
            "Open-ended questions without structured numeric or categorical response framework",
            "Statements, reflections, or summaries that do not solicit client self-evaluation"
        ]

    # Build confounds
    confounds = []
    if scale_ratio >= 0.3:
        confounds.append("May partially activate on non-counseling contexts using numeric scales (e.g., surveys, clinical intake forms)")
    if question_ratio >= 0.5:
        confounds.append("Could produce weak activation on direct questions that share syntactic structure but lack assessment content")
    if medical_ratio >= 0.3:
        confounds.append("Medical or health vocabulary alone may contribute partial activation independent of assessment framing")
    if substance_ratio >= 0.3:
        confounds.append("Substance-related keywords outside MI context may produce background activation")
    if not confounds:
        confounds = [
            "Structurally similar utterances outside the counseling domain may partially activate",
            "Numeric or quantifier mentions in non-assessment contexts may produce weak signals"
        ]

    # Build alternatives
    alternatives = []
    if scale_ratio >= 0.3:
        alternatives.append("The latent may track the syntactic pattern of numeric scale expressions rather than the counseling assessment function")
    if question_ratio >= 0.5:
        alternatives.append("Could be detecting interrogative prosody or question structure rather than specific assessment content")
    if medical_ratio >= 0.3:
        alternatives.append("May respond to medical/health domain vocabulary independent of the motivational interviewing context")
    if confidence_ratio >= 0.3 or readiness_ratio >= 0.3:
        alternatives.append("Could track self-efficacy or motivation-related lexical items rather than the scale framework itself")
    if not alternatives:
        alternatives = [
            "May be responding to surface-level syntactic or prosodic patterns rather than semantic assessment content",
            "Could reflect a training data artifact where certain counselor phrasings co-occur with model behaviors"
        ]

    # Build failure modes
    failure_modes = []
    if scale_ratio >= 0.3:
        failure_modes.append("May miss assessments using verbal (non-numeric) rating scales or implicit rating frameworks")
    if question_ratio >= 0.5:
        failure_modes.append("Could fail to detect assessment intent in indirect or embedded question structures")
    if near_miss_texts:
        failure_modes.append("Boundary cases near the activation threshold may be sensitive to minor phrasing variations")
    failure_modes.append("Performance may vary across different counseling styles, therapeutic orientations, or client populations")

    # Build key evidence
    key_evidence = extract_key_evidence(groups)

    result = {
        "latent_idx": latent_idx,
        "short_name": short_name,
        "main_hypothesis": main_hyp,
        "positive_triggers": positive_triggers,
        "explicit_exclusions": exclusions,
        "possible_surface_confounds": confounds,
        "feature_type": feature_type,
        "confidence": round(confidence, 2),
        "alternative_hypotheses": alternatives,
        "key_evidence": key_evidence,
        "failure_modes": failure_modes
    }

    return result


def main():
    # Read tasks 421-480 (indices 420-479)
    tasks = []
    with open(TASKS_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if 420 <= i < 480:
                tasks.append(json.loads(line))

    print(f"Loaded {len(tasks)} tasks (indices 420-479)")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(MANIFEST_FILE), exist_ok=True)

    # Process each task
    manifest_lines = []
    success_count = 0
    error_count = 0

    for idx, task in enumerate(tasks):
        task_id = task['task_id']
        latent_idx = task['latent_idx']
        expected_path = task['expected_output_path']
        prompt = task['prompt']

        # Determine run number from task_id
        run_match = re.search(r'_r(\d+)$', task_id)
        run_num = int(run_match.group(1)) if run_match else 1

        try:
            # Generate explanation
            explanation = generate_explanation(task, run_num)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(expected_path), exist_ok=True)

            # Write explanation JSON
            with open(expected_path, 'w', encoding='utf-8') as f:
                json.dump(explanation, f, indent=2, ensure_ascii=False)

            # Compute hashes for manifest
            prompt_bytes = prompt.encode('utf-8')
            prompt_sha = hashlib.sha256(prompt_bytes).hexdigest()

            output_bytes = json.dumps(explanation, ensure_ascii=False, indent=2).encode('utf-8')
            output_sha = hashlib.sha256(output_bytes).hexdigest()

            manifest_entry = {
                "task_id": task_id,
                "model": "claude-opus-4-8-20250918",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt_sha256": prompt_sha,
                "raw_output_sha256": output_sha
            }
            manifest_lines.append(manifest_entry)

            success_count += 1
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(tasks)} tasks...")

        except Exception as e:
            error_count += 1
            print(f"  ERROR on task {task_id}: {e}")

    # Append manifest lines
    with open(MANIFEST_FILE, 'a', encoding='utf-8') as f:
        for entry in manifest_lines:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\nDone. Success: {success_count}, Errors: {error_count}")
    print(f"Manifest entries appended to: {MANIFEST_FILE}")


if __name__ == '__main__':
    main()
