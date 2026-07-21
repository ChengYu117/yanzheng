import json
import re
from pathlib import Path

ROOT = Path(r'D:\project\NLP_re_dataset_model_base')
base = ROOT / 'outputs/rerun_new_dataset_20260716/min5_words/interpretability'
raw_dir = base / 'stable_core_latent_cards_translation_review/raw_translation'
out_path = base / 'stable_core_latent_cards_translation_review/stable_core_latent_cards_每标签5个_中文最简审核版_20260718.md'


def section(markdown, number):
    heading = r'(?:候选|不充分|表层|行为|主要|逐条|备选|强组|弱组|Held-out)'
    match = re.search(
        rf'(?ms)^(?:##\s*)?{number}\.\s+{heading}.*?\n(.*?)(?=^(?:##\s*)?[0-9]+\.\s+{heading}|\Z)',
        markdown,
    )
    return match.group(1).strip() if match else ''


def sentences(block):
    result = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if 'text:' in line:
            line = line.split('text:', 1)[1].strip()
        else:
            line = re.sub(r'^\d+\.\s*', '', line)
            line = re.sub(r'^(?:sample_id:\s*\w+\s*;?\s*)', '', line)
            line = re.sub(r'^\w+:\s*', '', line)
        line = line.lstrip('- ').strip()
        if line:
            result.append(line)
    return result


def explanation(block, field_names):
    for name in field_names:
        match = re.search(rf'(?m)^\s*-?\s*{re.escape(name)}\s*[:：]\s*(.+)$', block)
        if match:
            return match.group(1).strip()
    return ''


items = []
for path in sorted(raw_dir.glob('*.json')):
    markdown = json.loads(path.read_text(encoding='utf-8'))['markdown']
    strong = sentences(section(markdown, 8))
    weak = sentences(section(markdown, 9))
    surface = section(markdown, 3)
    function = section(markdown, 4)
    summary = explanation(section(markdown, 5), ('primary_explanation', '主要解释'))
    if not (strong and weak and surface and function and summary):
        raise RuntimeError(f'Incomplete translated card: {path.name}')
    items.append((strong, weak, surface, function, summary))

out = [
    '# Stable-core latent card 中文最简审核版',
    '',
    '仅保留被解释句子簇、表面解释、功能解释和总概括。',
    '',
]
for index, (strong, weak, surface, function, summary) in enumerate(items, 1):
    out += [
        f'## 审核条目 {index}',
        '',
        '### 被解释的句子簇',
        '',
        '#### 强响应句子',
        '',
        *[f'- {sentence}' for sentence in strong],
        '',
        '#### 弱响应句子',
        '',
        *[f'- {sentence}' for sentence in weak],
        '',
        '### 表面解释',
        '',
        surface,
        '',
        '### 功能解释',
        '',
        function,
        '',
        '### 总概括',
        '',
        summary,
        '',
        '---',
        '',
    ]

out_path.write_text('\n'.join(out), encoding='utf-8')
print(f'generated {out_path} ({len(items)} items)')
