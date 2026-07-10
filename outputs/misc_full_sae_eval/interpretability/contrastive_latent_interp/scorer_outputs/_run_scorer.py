"""Batch scorer for tasks 0-9 in tasks_121_200.json.
Judges whether each sample sentence should activate a latent based on explanation.
"""
import json, os, hashlib
from datetime import datetime, timezone

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(base_dir, "..", "..", "..", "..", ".."))
tasks_path = os.path.join(base_dir, "tasks_121_200.json")
manifest_path = os.path.join(
    project_root, "outputs", "misc_full_sae_eval", "interpretability",
    "contrastive_latent_interp", "llm_execution_manifest.jsonl"
)

with open(tasks_path, 'r', encoding='utf-8') as f:
    all_tasks = json.load(f)

tasks = all_tasks[:10]

# ---------------------------------------------------------------
# Parse task prompts
# ---------------------------------------------------------------
parsed = []
for idx, task in enumerate(tasks):
    prompt = task['prompt']
    expl_start = prompt.index('Explanation JSON:\n') + len('Explanation JSON:\n')
    depth = 0
    json_start = expl_start
    json_end = expl_start
    for i, ch in enumerate(prompt[expl_start:], expl_start):
        if ch == '{':
            if depth == 0:
                json_start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                json_end = i + 1
                break
    explanation = json.loads(prompt[json_start:json_end])

    samples_start = prompt.index('Held-out samples:\n') + len('Held-out samples:\n')
    rest = prompt[samples_start:]
    samples = []
    lines_list = rest.strip().split('\n')
    i = 0
    while i < len(lines_list):
        line = lines_list[i].strip()
        if line.startswith('- sample_id='):
            sid = line.split('=')[1]
            if i + 1 < len(lines_list):
                text_line = lines_list[i+1].strip()
                if text_line.startswith('text:'):
                    text = text_line[len('text:'):].strip()
                    samples.append({'sample_id': sid, 'text': text})
                    i += 2
                    continue
        i += 1

    parsed.append({
        'task_idx': idx,
        'task': task,
        'explanation': explanation,
        'samples': samples
    })

# ---------------------------------------------------------------
# Judgment dictionaries per task
# Keys are sample_id -> (predicted_label, confidence, reasoning)
# ---------------------------------------------------------------

# TASK 0 & 1: composite_behavioral_directive (latent 32727)
# Same explanation, same samples for r01 and r02
behavioral_directive_judgments = {
    "u006": (0, 0.85, "Reflective summary of client's observed changes; no direction or suggestion for action."),
    "u008": (0, 0.8, "Exploratory question about cravings timing; asking for information, not directing action."),
    "u003": (0, 0.9, "Factual statement with no action verb, directive, or behavioral suggestion."),
    "u007": (1, 0.85, "Raises health risks (lung cancer, emphysema) to direct client toward awareness; risk language as secondary trigger."),
    "u009": (0, 0.75, "Positive reinforcement of past progress; acknowledging achievement rather than directing future action."),
    "u018": (0, 0.95, "Brief factual observation about physical appearance; no directive content."),
    "u012": (0, 0.9, "Client's narrative about a past scary event; not counselor directive."),
    "u013": (0, 0.85, "Exploratory/reflection about someone else's worry; no guidance given."),
    "u016": (0, 0.7, "Describes current behavior pattern; no directive to change it."),
    "u002": (0, 0.9, "Simple statement of fact; no directive or behavioral suggestion."),
    "u001": (0, 0.8, "Acknowledgment of client's intent; not a directive from the counselor."),
    "u015": (1, 0.8, "Encouragement directive: 'no reason you shouldn't be able to quit' guides client toward behavior change."),
    "u014": (1, 0.75, "Directive to maintain current positive behavior: 'continuing to do what you're doing'."),
    "u004": (0, 0.85, "Empathic reflection of client's overwhelmed feeling; no guidance or suggestions."),
    "u005": (1, 0.6, "Borderline: implies consequence of not seeing a dentist, but grammar is unclear; weak directive signal."),
    "u010": (1, 0.9, "Direct directive: 'you need to be looking at your whole lifestyle which is diet and exercise'."),
    "u019": (0, 0.9, "Validating acknowledgment; no directive content."),
    "u017": (0, 0.9, "Reflective summary of client's pain experience; no direction given."),
    "u020": (0, 0.85, "Client's own proposal for behavior change, not counselor directive."),
    "u011": (0, 0.85, "Reflective summary of client's surprise about patterns; no guidance given."),
}

# TASK 2 & 3: hypothetical_scenario_probe (latent 16320)
# Same explanation, same samples
hypothetical_scenario_judgments = {
    "u003": (0, 0.9, "Describes client's current reality of caregiving burden; no hypothetical framing."),
    "u007": (0, 0.9, "Opinion/statement about depression and Star Wars; no conditional or hypothetical structure."),
    "u006": (0, 0.85, "Describes actual relationship situation; no hypothetical framing."),
    "u004": (0, 0.95, "Factual question about what brought client here; no conditional construction."),
    "u002": (0, 0.9, "Statement about current breathing difficulty; present reality, not hypothetical."),
    "u001": (1, 0.65, "Explores hypothetical advantages of stopping/cutting down; conditional framing with 'advantages of actually stopping'."),
    "u005": (0, 0.85, "Statement about obligation ('you have to'); factual, not hypothetical exploration."),
    "u010": (0, 0.8, "Health risk statement; factual consequence, not hypothetical scenario framing."),
    "u011": (0, 0.8, "Past tense statement about steps already taken; actual events, not hypothetical."),
    "u018": (0, 0.95, "Brief factual observation; no conditional or hypothetical content."),
    "u017": (0, 0.9, "Past event description; explicitly about actual experience, not hypothetical."),
    "u015": (1, 0.65, "Borderline: 'you can do that' invites imagining ability/success; mild conditional."),
    "u020": (0, 0.8, "Statement about past attempts; actual events, not hypothetical framing."),
    "u008": (0, 0.9, "Descriptive statement about a scene; no hypothetical construction."),
    "u019": (0, 0.9, "Validating acknowledgment; no conditional or hypothetical structure."),
    "u016": (0, 0.8, "Describes actual behavior pattern with 'when' temporal clause; not hypothetical scenario."),
    "u013": (0, 0.85, "Statement about current impact; present reality, not imagined situation."),
    "u012": (0, 0.85, "Reflection on making sense of something; no hypothetical framing."),
    "u009": (0, 0.9, "Direct factual claim about different medications; no conditional construction."),
    "u014": (0, 0.85, "Past tense statement linking behavior; actual events, not hypothetical."),
}

# TASK 4 & 5: composite_hypothetical_scenario_probe (latent 29701)
# Same explanation, same samples
composite_hypothetical_judgments = {
    "u007": (0, 0.85, "Direct question about wanting to go home; no hypothetical framing."),
    "u003": (0, 0.85, "Open question about what brings client in; factual, not hypothetical."),
    "u010": (1, 0.8, "Hypothetical scenario: 'curling iron could start a fire' explores imagined consequences."),
    "u004": (0, 0.75, "Statement about health maintenance benefits; factual claim, not hypothetical framing."),
    "u013": (1, 0.7, "Explores hypothetical consequence: 'possibility of losing your apartment' frames imagined outcome."),
    "u017": (0, 0.9, "Past event description; actual experience, not hypothetical."),
    "u018": (0, 0.95, "Brief factual observation; no conditional content."),
    "u016": (0, 0.8, "Describes actual behavior with temporal clause; not hypothetical scenario."),
    "u009": (0, 0.9, "Direct question/comment about depression; no conditional framing."),
    "u014": (0, 0.85, "Assessment of readiness; no hypothetical scenario construction."),
    "u015": (1, 0.7, "Readiness assessment with implied hypothetical change: 'ready to make a change'."),
    "u008": (0, 0.8, "Question about wanting to think; exploratory, not hypothetical framing."),
    "u011": (1, 0.85, "Clear hypothetical: 'you could give a shot' with conditional ability framing."),
    "u012": (0, 0.85, "Describes current/recent state; no hypothetical construction."),
    "u006": (0, 0.85, "Fragment about wanting clarity; no hypothetical framing."),
    "u020": (0, 0.8, "Statement about past attempts; actual events."),
    "u001": (0, 0.9, "Compliment about past event; no conditional structure."),
    "u005": (0, 0.9, "Simple question; factual, no hypothetical."),
    "u002": (0, 0.9, "Routine procedural question; no hypothetical framing."),
    "u019": (0, 0.9, "Validating acknowledgment; no conditional content."),
}

# TASK 6 & 7: counselor_meta_framing (latent 26319)
# Same explanation, same samples
meta_framing_judgments = {
    "u009": (0, 0.9, "Reflective statement about client's anxiety; no meta-communicative framing of counselor's own agenda."),
    "u005": (0, 0.95, "Vague fragment; no meta-communicative framing."),
    "u004": (0, 0.9, "Third-party description; no counselor self-reference or process framing."),
    "u003": (1, 0.9, "Explicit meta-framing: 'let me first verify' announces counselor's conversational move and agenda."),
    "u019": (0, 0.85, "Reflective summary of client's activities; no process-framing of counselor's intent."),
    "u011": (0, 0.8, "Counselor assessment with 'I'd say'; borderline but no explicit process announcement."),
    "u012": (0, 0.85, "Reflective statement; no meta-communicative framing of counselor's agenda."),
    "u001": (0, 0.8, "Directive about getting control; no meta-framing of conversational process."),
    "u014": (0, 0.8, "Reflective summary; no explicit process-framing or agenda announcement."),
    "u008": (0, 0.85, "Reflective statement about eating habits; no meta-communicative framing."),
    "u013": (0, 0.85, "Reflective statement about steps taken; no meta-framing."),
    "u018": (0, 0.9, "Brief relational observation; no meta-communicative content."),
    "u017": (0, 0.9, "Direct question about pain; no meta-framing of counselor's move."),
    "u007": (0, 0.8, "Reflective statement about support group; no process-framing."),
    "u016": (0, 0.9, "Validating acknowledgment; no meta-communicative framing."),
    "u002": (1, 0.85, "Process-framing: 'I have a little handout here so we can talk about' announces counselor's agenda."),
    "u006": (1, 0.8, "Meta-framing of session management: 'you're going to take that home and you're going to look that over' frames counselor's plan."),
    "u015": (0, 0.8, "Reflective summary of workplace situation; no meta-framing."),
    "u010": (0, 0.8, "Reflective statement about past feelings; no meta-communicative framing."),
    "u020": (0, 0.9, "General descriptive statement; no meta-communicative content."),
}

# TASK 8 & 9: composite_health_concern_probe (latent 27859)
# Same explanation, same samples
health_concern_judgments = {
    "u020": (0, 0.8, "Client's proposal to cut back; not counselor raising health concern."),
    "u006": (1, 0.75, "Counselor flags obstacles to success ('not a 10') implying potential negative outcomes; concern-raising."),
    "u008": (0, 0.7, "Empowering statement about choice; positive framing, not risk/concern raising."),
    "u019": (0, 0.85, "Acknowledgment of past attempts; no health risk framing."),
    "u001": (0, 0.85, "Mysterious/unusual statement; no health risk or concern content."),
    "u007": (0, 0.85, "Reflective statement about someone's priorities; no health concern raised."),
    "u016": (0, 0.65, "Borderline: mentions blood sugars running high which relates to health monitoring, but framed as behavior description not counselor raising concern."),
    "u010": (1, 0.75, "Same pattern as u006: flags obstacles to success, implying potential negative outcomes."),
    "u002": (0, 0.8, "Financial advice; not health-related concern."),
    "u009": (0, 0.85, "Positive reinforcement; no risk or concern framing."),
    "u005": (0, 0.8, "Describes past events; no health concern raised."),
    "u013": (0, 0.8, "Reflective statement about worry; describes client's state, not counselor raising health concern."),
    "u017": (0, 0.85, "Reflective summary of pain; no explicit health risk or concern framing by counselor."),
    "u004": (1, 0.9, "Explicit health concern: medication side effects, 'obsessed Omikron or vomiting', 'nervous of shaking'."),
    "u014": (0, 0.8, "Assessment of enthusiasm; no health risk or concern raised."),
    "u015": (0, 0.8, "Assessment of ability; no health concern framing."),
    "u012": (0, 0.8, "Reflective statement about compliance; no health concern."),
    "u018": (0, 0.75, "Physical observation ('swollen') but brief factual comment, not raising concern."),
    "u003": (0, 0.8, "Abstract/unclear statement; no health concern content."),
    "u011": (0, 0.8, "Hypothetical about ability to try; no health risk or concern framing."),
}

# ---------------------------------------------------------------
# Map task_idx to judgment dict
# ---------------------------------------------------------------
task_judgment_map = {
    0: behavioral_directive_judgments,
    1: behavioral_directive_judgments,
    2: hypothetical_scenario_judgments,
    3: hypothetical_scenario_judgments,
    4: composite_hypothetical_judgments,
    5: composite_hypothetical_judgments,
    6: meta_framing_judgments,
    7: meta_framing_judgments,
    8: health_concern_judgments,
    9: health_concern_judgments,
}

# ---------------------------------------------------------------
# Process each task
# ---------------------------------------------------------------
model_name = "claude-opus-4-8-20250918"

for p in parsed:
    idx = p['task_idx']
    task = p['task']
    samples = p['samples']
    out_path = os.path.join(project_root, task['expected_output_path'])

    if os.path.exists(out_path):
        print(f"Task {idx} ({task['task_id']}): output already exists, skipping.")
        continue

    judgments = task_judgment_map[idx]
    results = []
    for s in samples:
        sid = s['sample_id']
        label, conf, reason = judgments[sid]
        results.append({
            "sample_id": sid,
            "predicted_label": label,
            "confidence": conf,
            "reasoning": reason
        })

    # Write output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    output_json = json.dumps(results, ensure_ascii=False, indent=2)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(output_json)
    print(f"Task {idx} ({task['task_id']}): wrote {len(results)} judgments to {out_path}")

    # Compute SHA256s
    prompt_sha = hashlib.sha256(task['prompt'].encode('utf-8')).hexdigest()
    output_sha = hashlib.sha256(output_json.encode('utf-8')).hexdigest()

    manifest_entry = {
        "task_id": task['task_id'],
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_sha256": prompt_sha,
        "raw_output_sha256": output_sha
    }

    with open(manifest_path, 'a', encoding='utf-8') as mf:
        mf.write(json.dumps(manifest_entry, ensure_ascii=False) + '\n')
    print(f"  -> manifest entry appended")

print("\nAll 10 tasks processed.")
