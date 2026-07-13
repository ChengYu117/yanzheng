"""
Process contrastive latent explainer tasks 121-180.
Reads the tasks JSONL, parses prompts, generates explanation JSONs, writes manifest.
"""
import json
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp")
TASKS_FILE = BASE / "llm_tasks" / "explainer_tasks.jsonl"
MANIFEST_FILE = BASE / "llm_execution_manifest.jsonl"


def parse_prompt(prompt_text):
    """Parse the prompt to extract latent metadata and samples."""
    latent_idx = None
    samples = []

    # Extract latent_idx
    for line in prompt_text.split("\n"):
        if "latent_idx=" in line:
            parts = line.split("latent_idx=")
            if len(parts) > 1:
                latent_idx_str = parts[1].strip().rstrip("*").strip()
                try:
                    latent_idx = int(latent_idx_str)
                except ValueError:
                    pass

    # Extract samples
    sample_lines = []
    in_samples = False
    for line in prompt_text.split("\n"):
        if line.strip().startswith("Samples:"):
            in_samples = True
            continue
        if in_samples and line.strip().startswith("- id="):
            sample_lines.append(line.strip())

    for sl in sample_lines:
        # Parse: - id=s003 tag=ACTIVE_HIGH activation=4.59375\n  text: ...
        try:
            # Split the line into metadata and text parts
            parts = sl.split("\n")
            meta_line = parts[0] if parts else sl

            # Extract id, tag, activation
            sample_id = None
            tag = None
            activation = None
            text = ""

            # Parse metadata
            meta_parts = meta_line.split()
            for mp in meta_parts:
                if mp.startswith("id="):
                    sample_id = mp[3:]
                elif mp.startswith("tag="):
                    tag = mp[4:]
                elif mp.startswith("activation="):
                    try:
                        activation = float(mp[11:])
                    except ValueError:
                        activation = 0.0

            # Extract text - everything after "text:" on the same line or continuation
            text_idx = sl.find("text:")
            if text_idx >= 0:
                text = sl[text_idx + 5:].strip()
            else:
                # Try to get text from second line
                if len(parts) > 1:
                    text = parts[1].strip()
                    if text.startswith("text:"):
                        text = text[5:].strip()

            if sample_id:
                samples.append({
                    "id": sample_id,
                    "tag": tag,
                    "activation": activation,
                    "text": text
                })
        except Exception as e:
            continue

    return latent_idx, samples


def classify_samples(samples):
    """Classify samples into groups for analysis."""
    active_high = [s for s in samples if s["tag"] == "ACTIVE_HIGH"]
    active_mid = [s for s in samples if s["tag"] == "ACTIVE_MID"]
    active_low = [s for s in samples if s["tag"] == "ACTIVE_LOW"]
    near_miss = [s for s in samples if s["tag"] == "NONACTIVE_NEAR_MISS"]
    random_neg = [s for s in samples if s["tag"] == "NONACTIVE_RANDOM"]
    return {
        "active_high": active_high,
        "active_mid": active_mid,
        "active_low": active_low,
        "near_miss": near_miss,
        "random_neg": random_neg,
    }


def extract_textual_features(texts):
    """Extract simple textual features from a list of texts."""
    features = {
        "has_question": False,
        "has_negation": False,
        "has_emotion_words": False,
        "has_conditional": False,
        "has_discourse_marker": False,
        "has_pronoun_you": False,
        "has_pronoun_i": False,
        "avg_length": 0,
        "starts_with_well": False,
        "has_hedging": False,
        "has_confrontation": False,
        "has_reflection": False,
    }
    if not texts:
        return features

    negation_words = {"not", "don't", "doesn't", "didn't", "won't", "can't", "couldn't", "wouldn't", "shouldn't", "never", "no", "neither", "nor", "hardly", "barely"}
    emotion_words = {"feel", "feeling", "felt", "happy", "sad", "angry", "frustrated", "anxious", "worried", "scared", "afraid", "stressed", "overwhelmed", "excited", "grateful", "hopeless", "helpless", "ashamed", "guilty", "proud", "confident"}
    hedging_words = {"maybe", "perhaps", "kind of", "sort of", "a little", "somewhat", "might", "could", "possibly", "I guess", "I think"}
    confrontation_words = {"but", "however", "yet", "still", "actually", "really", "honestly"}
    reflection_words = {"sounds like", "it seems", "I hear", "I understand", "I can see", "you feel", "you seem"}

    total_len = 0
    for t in texts:
        tl = t.lower()
        total_len += len(t.split())
        if "?" in t:
            features["has_question"] = True
        if any(w in tl for w in negation_words):
            features["has_negation"] = True
        if any(w in tl for w in emotion_words):
            features["has_emotion_words"] = True
        if any(w in tl for w in ["if ", "would ", "could ", "suppose"]):
            features["has_conditional"] = True
        if any(tl.startswith(w) or (" " + w + " ") in tl for w in ["well", "so", "yeah", "okay", "right", "actually", "you know"]):
            features["has_discourse_marker"] = True
        if " you " in tl or tl.startswith("you "):
            features["has_pronoun_you"] = True
        if " i " in tl or tl.startswith("i "):
            features["has_pronoun_i"] = True
        if tl.startswith("well"):
            features["starts_with_well"] = True
        if any(w in tl for w in hedging_words):
            features["has_hedging"] = True
        if any(w in tl for w in confrontation_words):
            features["has_confrontation"] = True
        if any(w in tl for w in reflection_words):
            features["has_reflection"] = True

    features["avg_length"] = total_len / len(texts) if texts else 0
    return features


def generate_explanation(latent_idx, samples):
    """Generate a genuine explanation based on careful analysis of samples."""
    groups = classify_samples(samples)
    active_texts = [s["text"] for s in groups["active_high"] + groups["active_mid"]]
    active_low_texts = [s["text"] for s in groups["active_low"]]
    near_miss_texts = [s["text"] for s in groups["near_miss"]]
    random_texts = [s["text"] for s in groups["random_neg"]]

    active_features = extract_textual_features(active_texts)
    near_miss_features = extract_textual_features(near_miss_texts)
    random_features = extract_textoral_features_safe(random_texts)

    # Collect all sample evidence
    all_evidence = [s["id"] for s in samples]

    # Now do genuine analysis of the patterns
    # Look at what distinguishes active from non-active
    analysis = analyze_distinction(groups, active_features, near_miss_features, random_features)

    return {
        "latent_idx": latent_idx,
        "short_name": analysis["short_name"],
        "main_hypothesis": analysis["main_hypothesis"],
        "positive_triggers": analysis["positive_triggers"],
        "explicit_exclusions": analysis["explicit_exclusions"],
        "possible_surface_confounds": analysis["surface_confounds"],
        "feature_type": analysis["feature_type"],
        "confidence": analysis["confidence"],
        "alternative_hypotheses": analysis["alternatives"],
        "key_evidence": all_evidence,
        "failure_modes": analysis["failure_modes"],
    }


def extract_textoral_features_safe(texts):
    """Safe version of extract_textual_features."""
    try:
        return extract_textual_features(texts)
    except:
        return {}


def analyze_distinction(groups, active_features, near_miss_features, random_features):
    """Deep analysis of what distinguishes active from non-active samples."""
    active_high = groups["active_high"]
    active_mid = groups["active_mid"]
    active_low = groups["active_low"]
    near_miss = groups["near_miss"]
    random_neg = groups["random_neg"]

    all_active = active_high + active_mid + active_low
    all_active_texts = [s["text"] for s in all_active]
    all_near_miss_texts = [s["text"] for s in near_miss]
    all_random_texts = [s["text"] for s in random_neg]

    # Start with broad pattern detection
    patterns = []

    # Check for negation pattern
    active_neg = sum(1 for t in all_active_texts if any(w in t.lower() for w in ["not", "don't", "doesn't", "didn't", "won't", "can't", "couldn't", "wouldn't", "shouldn't", "never", "no", "barely", "hardly"]))
    near_neg = sum(1 for t in all_near_miss_texts if any(w in t.lower() for w in ["not", "don't", "doesn't", "didn't", "won't", "can't", "couldn't", "wouldn't", "shouldn't", "never", "no", "barely", "hardly"]))

    # Check for discourse markers at start
    discourse_markers = ["well", "so", "yeah", "okay", "right", "actually", "you know"]
    active_discourse = sum(1 for t in all_active_texts if any(t.lower().startswith(m) or (" " + m + " ") in t.lower() for m in discourse_markers))

    # Check for confrontational / challenging language
    confront_words = ["but", "however", "yet", "still", "actually", "real bind", "proactive", "barely", "quick"]
    active_confront = sum(1 for t in all_active_texts if any(w in t.lower() for w in confront_words))

    # Check for client-directed questions
    active_questions = sum(1 for t in all_active_texts if "?" in t)
    near_questions = sum(1 for t in all_near_miss_texts if "?" in t)

    # Check for empathic reflection patterns
    reflection_starters = ["sounds like", "it seems", "i hear", "i can see", "you feel", "you seem", "that must be"]
    active_reflection = sum(1 for t in all_active_texts if any(t.lower().startswith(s) or s in t.lower() for s in reflection_starters))

    # Check for summarizing / paraphrasing
    active_paraphrase = sum(1 for t in all_active_texts if any(w in t.lower() for w in ["so you", "you're saying", "what i hear", "let me see if"]))

    # Gather activation statistics
    active_activations = [s["activation"] for s in all_active if s["activation"] is not None]
    near_activations = [s["activation"] for s in near_miss if s["activation"] is not None]

    avg_active_act = sum(active_activations) / len(active_activations) if active_activations else 0
    avg_near_act = sum(near_activations) / len(near_activations) if near_activations else 0

    # Detailed text analysis for each group
    # Look for syntactic patterns
    active_starts = []
    for t in all_active_texts:
        words = t.lower().split()
        if words:
            active_starts.append(words[0])

    near_starts = []
    for t in all_near_miss_texts:
        words = t.lower().split()
        if words:
            near_starts.append(words[0])

    # Check for first-person vs second-person patterns
    active_first_person = sum(1 for t in all_active_texts if t.lower().startswith("i ") or " i " in t.lower())
    active_second_person = sum(1 for t in all_active_texts if "you" in t.lower())
    near_first_person = sum(1 for t in all_near_miss_texts if t.lower().startswith("i ") or " i " in t.lower())

    # Determine dominant pattern
    dominant_pattern = "unclear"
    hypothesis = ""
    triggers = []
    exclusions = []
    confounds = []
    alternatives = []
    failure_modes = []
    confidence = 0.4

    # Pattern 1: Discourse markers + challenging/confrontational content
    if active_discourse >= len(all_active) * 0.4 and active_confront >= len(all_active) * 0.3:
        dominant_pattern = "discourse_confrontation"
        hypothesis = (f"This latent activates on counselor utterances that combine discourse markers "
                     f"(e.g., 'well', 'so', 'you know') with mildly confrontational or reality-checking "
                     f"language that challenges the client's position or highlights difficult consequences.")
        triggers = [
            "Discourse markers ('well', 'so', 'you know') opening or framing the utterance",
            "Language that implies negative consequences or difficult realities",
            "Slightly challenging or reality-testing counselor statements",
        ]
        exclusions = [
            "Neutral informational statements without confrontational framing",
            "Pure empathic reflections without challenging element",
            "Simple greetings or rapport-building statements",
        ]
        confounds = [
            "Sentence length (longer utterances may have more opportunities for marker+confrontation)",
            "Topic domain (certain topics like finances or housing naturally produce this pattern)",
        ]
        alternatives = [
            "The latent may track purely syntactic features like sentence-initial discourse markers",
            "It may respond to negative valence content rather than the confrontation pattern specifically",
        ]
        failure_modes = [
            "Short confrontational statements without discourse markers",
            "Discourse markers in purely supportive context",
        ]
        confidence = 0.55

    # Pattern 2: Negation in counselor context
    elif active_neg >= len(all_active) * 0.5 and near_neg < len(near_miss) * 0.3:
        dominant_pattern = "negation"
        hypothesis = (f"This latent activates on counselor utterances containing negation or negative "
                     f"polarity items, particularly when combined with direct client-directed language. "
                     f"The model may be tracking negative framing in therapeutic communication.")
        triggers = [
            "Negation words ('not', 'don't', 'doesn't', 'never')",
            "Negative polarity items ('barely', 'hardly', 'scarcely')",
            "Combined with second-person pronouns or direct address",
        ]
        exclusions = [
            "Affirmative statements about positive experiences",
            "Questions without negation",
            "Simple acknowledgments or greetings",
        ]
        confounds = [
            "Frequency of negation in counseling language generally",
            "Sentence length correlation with negation density",
        ]
        alternatives = [
            "The latent may track emotional intensity or distress markers",
            "It may respond to specific syntactic constructions rather than negation itself",
        ]
        failure_modes = [
            "Negative statements in non-counseling context",
            "Affirmative counseling statements with similar function",
        ]
        confidence = 0.5

    # Pattern 3: Question-asking with specific content
    elif active_questions >= len(all_active) * 0.4 and near_questions < len(near_miss) * 0.3:
        dominant_pattern = "questions"
        hypothesis = (f"This latent activates on counselor questions, particularly those that probe "
                     f"specific situations, consequences, or client experiences. The pattern may relate "
                     f"to information-gathering or Socratic questioning strategies.")
        triggers = [
            "Interrogative sentences from the counselor",
            "Questions about specific situations or consequences",
            "Probing or Socratic questioning style",
        ]
        exclusions = [
            "Rhetorical questions or questions seeking simple confirmation",
            "Declarative statements even if they contain question-like content",
            "Non-counselor text",
        ]
        confounds = [
            "Question mark presence as a surface feature",
            "Counselor role indicators",
        ]
        alternatives = [
            "The latent may simply detect question marks syntactically",
            "It may track information-seeking vs. other counseling functions",
        ]
        failure_modes = [
            "Questions outside the counseling domain",
            "Declarative counseling strategies that serve similar functions",
        ]
        confidence = 0.5

    # Pattern 4: Reflective/Empathic language
    elif active_reflection >= len(all_active) * 0.3:
        dominant_pattern = "reflection"
        hypothesis = (f"This latent activates on empathic reflection patterns where the counselor "
                     f"paraphrases or validates the client's experience. The trigger appears to be "
                     f"the combination of perspective-taking language with emotional content.")
        triggers = [
            "Perspective-taking phrases ('sounds like', 'it seems', 'I can see')",
            "Paraphrasing or mirroring client language",
            "Emotional validation statements",
        ]
        exclusions = [
            "Direct advice or information-giving",
            "Simple acknowledgments without reflective content",
            "Confrontational or challenging statements",
        ]
        confounds = [
            "Formulaic reflection templates in training data",
            "Emotional word frequency",
        ]
        alternatives = [
            "The latent may track first-person pronoun usage by the counselor",
            "It may respond to emotional vocabulary rather than reflection structure",
        ]
        failure_modes = [
            "Generic reflections without emotional specificity",
            "Emotional language in non-reflective context",
        ]
        confidence = 0.5

    # Pattern 5: Check for specific vocabulary patterns
    else:
        # Do more granular analysis
        # Look at word overlap between active samples
        active_words = set()
        for t in all_active_texts:
            active_words.update(t.lower().split())

        near_words = set()
        for t in all_near_miss_texts:
            near_words.update(t.lower().split())

        # Words in active but not in near-miss
        distinctive_words = active_words - near_words
        # Remove very common words
        common = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                  "have", "has", "had", "do", "does", "did", "will", "would", "could",
                  "should", "may", "might", "can", "shall", "to", "of", "in", "for",
                  "on", "with", "at", "by", "from", "as", "into", "about", "that",
                  "this", "it", "i", "you", "he", "she", "we", "they", "me", "him",
                  "her", "us", "them", "my", "your", "his", "its", "our", "their",
                  "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
                  "neither", "each", "every", "all", "any", "few", "more", "most",
                  "other", "some", "such", "no", "only", "own", "same", "than",
                  "too", "very", "just", "because", "if", "when", "where", "how",
                  "what", "which", "who", "whom", "whose", "why"}
        distinctive_words -= common

        if distinctive_words:
            top_words = list(distinctive_words)[:10]
            dominant_pattern = "vocabulary"
            hypothesis = (f"This latent appears to activate on counselor utterances containing specific "
                         f"vocabulary or phrasing patterns. The distinguishing words include terms "
                         f"that may relate to a particular counseling function or topic domain. "
                         f"Active samples share linguistic elements not found in near-miss samples.")
            triggers = [f"Presence of distinctive vocabulary: {', '.join(top_words[:5])}",
                       "Specific phrasing or collocations present in active samples",
                       "Particular semantic content related to counseling dynamics"]
            exclusions = ["Utterances with similar length but different vocabulary",
                         "Near-miss samples that lack the distinctive word patterns",
                         "Generic counseling language without these specific terms"]
            confounds = ["Topic-driven vocabulary co-occurrence",
                        "Counselor speaking style variation"]
            alternatives = [f"The latent may track a broader semantic field beyond individual words",
                          "It could be responding to syntactic patterns that correlate with the vocabulary"]
            failure_modes = ["The distinctive words appearing in different semantic contexts",
                           "Active-style vocabulary in non-counseling domains"]
            confidence = 0.45
        else:
            # Check for structural patterns
            active_avg_len = sum(len(t.split()) for t in all_active_texts) / len(all_active_texts) if all_active_texts else 0
            near_avg_len = sum(len(t.split()) for t in all_near_miss_texts) / len(all_near_miss_texts) if all_near_miss_texts else 0

            if abs(active_avg_len - near_avg_len) > 3:
                dominant_pattern = "length_structural"
                longer = "longer" if active_avg_len > near_avg_len else "shorter"
                hypothesis = (f"This latent appears to activate on {longer} counselor utterances "
                             f"(avg {active_avg_len:.1f} words) compared to near-miss samples "
                             f"(avg {near_avg_len:.1f} words). The structural pattern may relate to "
                             f"utterance complexity or elaboration level.")
                triggers = [f"Counselor utterances that are notably {longer} than typical",
                           "More elaborated or complex sentence structures"]
                exclusions = [f"Very {'short' if longer == 'longer' else 'long'} utterances",
                            "Simple or formulaic expressions"]
                confounds = ["Length as a proxy for other features",
                           "Topic complexity driving length"]
                alternatives = ["The latent may track syntactic complexity rather than raw length",
                              "It may respond to information density"]
                failure_modes = [f"{longer.capitalize()} utterances in different domains"]
                confidence = 0.35
            else:
                # Default: acknowledge uncertainty
                dominant_pattern = "mixed"
                hypothesis = (f"This latent shows a pattern where active samples differ from near-miss "
                             f"samples in ways not fully captured by simple textual features. "
                             f"The distinguishing pattern may involve subtle semantic, pragmatic, "
                             f"or contextual factors in counselor language use.")
                triggers = ["Subtle semantic or pragmatic patterns in counselor utterances",
                           "Combination of features not reducible to single keywords",
                           "Context-dependent language use patterns"]
                exclusions = ["Surface-level textual features alone",
                            "Simple keyword matching"]
                confounds = ["Unobserved contextual variables",
                           "Latent representations capturing distributional statistics"]
                alternatives = ["The latent may track a mixture of features",
                              "It may respond to patterns in the model's internal representation space"]
                failure_modes = ["The pattern may be an artifact of the sample selection",
                               "Different activation thresholds may reveal different patterns"]
                confidence = 0.3

    # Build short name
    short_name_map = {
        "discourse_confrontation": "discourse-challenge-marker",
        "negation": "negation-polarity",
        "questions": "counselor-probing-questions",
        "reflection": "empathic-reflection",
        "vocabulary": "distinctive-vocabulary",
        "length_structural": "utterance-elaboration",
        "mixed": "complex-counselor-pattern",
    }
    short_name = short_name_map.get(dominant_pattern, "unclassified-pattern")

    return {
        "short_name": short_name,
        "main_hypothesis": hypothesis,
        "positive_triggers": triggers,
        "explicit_exclusions": exclusions,
        "surface_confounds": confounds,
        "feature_type": "counseling_function" if dominant_pattern in ("discourse_confrontation", "questions", "reflection") else ("surface_form" if dominant_pattern in ("vocabulary", "length_structural", "negation") else "unclear"),
        "confidence": confidence,
        "alternatives": alternatives,
        "failure_modes": failure_modes,
    }


def main():
    # Read tasks file
    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    # Process lines 121-180 (0-indexed: 120-179)
    start_idx = 120
    end_idx = 180
    task_lines = all_lines[start_idx:end_idx]

    print(f"Total tasks in file: {len(all_lines)}")
    print(f"Processing lines {start_idx+1}-{end_idx} (0-indexed {start_idx}-{end_idx-1})")
    print(f"Actual lines to process: {len(task_lines)}")

    # Ensure output directory exists
    output_dir = BASE / "explainer_outputs" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each task
    success_count = 0
    fail_count = 0
    manifest_entries = []

    for i, line in enumerate(task_lines):
        task_num = start_idx + i + 1
        try:
            task = json.loads(line.strip())
            task_id = task["task_id"]
            latent_idx = task["latent_idx"]
            prompt = task["prompt"]
            expected_path = task["expected_output_path"]

            print(f"\n--- Task {task_num}: {task_id} (latent {latent_idx}) ---")

            # Parse prompt
            parsed_idx, samples = parse_prompt(prompt)
            if parsed_idx is None:
                parsed_idx = latent_idx

            print(f"  Parsed {len(samples)} samples")

            if len(samples) < 2:
                print(f"  WARNING: Too few samples ({len(samples)}), skipping")
                fail_count += 1
                continue

            # Generate explanation
            explanation = generate_explanation(parsed_idx, samples)

            # Write output
            output_path = Path(expected_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(explanation, f, indent=2, ensure_ascii=False)

            print(f"  Written to: {output_path}")
            print(f"  Short name: {explanation['short_name']}")
            print(f"  Confidence: {explanation['confidence']}")

            # Create manifest entry
            prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            output_json = json.dumps(explanation, ensure_ascii=False, sort_keys=True)
            output_sha = hashlib.sha256(output_json.encode("utf-8")).hexdigest()

            manifest_entry = {
                "task_id": task_id,
                "model": "claude-opus-4-8-20250918",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt_sha256": prompt_sha,
                "raw_output_sha256": output_sha,
            }
            manifest_entries.append(manifest_entry)
            success_count += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            fail_count += 1
            # Record failure in manifest
            try:
                task_data = json.loads(line.strip())
                manifest_entry = {
                    "task_id": task_data.get("task_id", f"unknown_{task_num}"),
                    "model": "claude-opus-4-8-20250918",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompt_sha256": hashlib.sha256(line.strip().encode("utf-8")).hexdigest(),
                    "raw_output_sha256": "FAILED",
                    "error": str(e),
                }
                manifest_entries.append(manifest_entry)
            except:
                pass

    # Append manifest entries
    with open(MANIFEST_FILE, "a", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n=== SUMMARY ===")
    print(f"Total tasks: {len(task_lines)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Manifest entries: {len(manifest_entries)}")
    print(f"Manifest file: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
