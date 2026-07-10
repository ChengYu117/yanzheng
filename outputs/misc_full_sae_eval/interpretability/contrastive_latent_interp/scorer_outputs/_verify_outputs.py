"""Verify all 10 output files exist and are valid."""
import json, os

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_explanation_scorer")
expected = [
    'ctli_0065_scorer_ctli_0065_explainer_r01.json',
    'ctli_0065_scorer_ctli_0065_explainer_r02.json',
    'ctli_0066_scorer_ctli_0066_explainer_r01.json',
    'ctli_0066_scorer_ctli_0066_explainer_r02.json',
    'ctli_0067_scorer_ctli_0067_explainer_r01.json',
    'ctli_0067_scorer_ctli_0067_explainer_r02.json',
    'ctli_0068_scorer_ctli_0068_explainer_r01.json',
    'ctli_0068_scorer_ctli_0068_explainer_r02.json',
    'ctli_0069_scorer_ctli_0069_explainer_r01.json',
    'ctli_0069_scorer_ctli_0069_explainer_r02.json',
]
for fn in expected:
    path = os.path.join(base, fn)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        pos = sum(1 for d in data if d['predicted_label'] == 1)
        neg = sum(1 for d in data if d['predicted_label'] == 0)
        avg_conf = sum(d['confidence'] for d in data) / len(data)
        # Validate fields
        for d in data:
            assert 'sample_id' in d
            assert 'predicted_label' in d
            assert 'confidence' in d
            assert 'reasoning' in d
            assert d['predicted_label'] in (0, 1)
            assert 0.0 <= d['confidence'] <= 1.0
        print(f'OK {fn}: {len(data)} samples, pos={pos}, neg={neg}, avg_conf={avg_conf:.2f}')
    else:
        print(f'MISSING {fn}')

# Verify manifest
manifest_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "llm_execution_manifest.jsonl"))
if os.path.exists(manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f'\nManifest has {len(lines)} entries (last 10):')
    for line in lines[-10:]:
        entry = json.loads(line)
        print(f'  {entry["task_id"]} | {entry["model"]} | {entry["timestamp"][:19]}')
else:
    print(f'\nManifest not found at {manifest_path}')
