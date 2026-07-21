import json
import re
from pathlib import Path

ROOT = Path(r'D:\project\NLP_re_dataset_model_base')
base = ROOT / 'outputs/rerun_new_dataset_20260716/min5_words/interpretability'
raw_dir = base / 'stable_core_latent_cards_translation_review/raw_translation'
metrics_path = base / 'contrastive_latent_faithfulness_v2_gpt55_low_full218_scorer_deanchored_20260718/scorer/faithfulness_metrics.csv'
out_path = base / 'stable_core_latent_cards_translation_review/stable_core_latent_cards_每标签5个_中文审核易读版_20260718_deanchored.md'

def section(md, title, next_titles):
    pat = rf'## {re.escape(title)}（.*?）\n(.*?)(?=\n## (?:' + '|'.join(map(re.escape, next_titles)) + r')（|\Z)'
    m = re.search(pat, md, re.S)
    return m.group(1).strip() if m else ''

def lines(block):
    return block.splitlines()

metrics = {}
for line in metrics_path.read_text(encoding='utf-8-sig').splitlines()[1:]:
    p = line.split(',')
    if len(p) >= 6:
        metrics[p[0]] = dict(spearman=p[2], pearson=p[3], auroc=p[4], pair=p[5])

items = []
for path in sorted(raw_dir.glob('*.json')):
    obj = json.loads(path.read_text(encoding='utf-8'))
    md = obj['markdown']
    def field(name):
        m = re.search(rf'^-?\s*{re.escape(name)}:\s*(\S+)', md, re.M)
        return m.group(1) if m else None
    fid = field('feature_id')
    latent = field('latent_idx')
    label = field('stable_core_label_for_audit')
    rank = field('rank_within_label')
    if not all((fid, latent, label, rank)):
        print('skip malformed', path)
        continue
    h8m = re.search(r'^\s*(?:##\s*)?8\.\s*强组句子', md, re.M)
    if not h8m:
        print('skip missing sentence sections', path)
        continue
    h8 = h8m.start()
    explanation = md[:h8].split('\n', 1)[1].strip()
    strong = re.search(r'^\s*(?:##\s*)?8\..*?\n(.*?)(?=^\s*(?:##\s*)?9\.)', md, re.S | re.M).group(1).strip()
    weak = re.search(r'^\s*(?:##\s*)?9\..*?\n(.*?)(?=^\s*(?:##\s*)?10\.)', md, re.S | re.M).group(1).strip()
    held = re.search(r'^\s*(?:##\s*)?10\..*?\n(.*)$', md, re.S | re.M).group(1).strip()
    items.append((label, int(rank), fid, latent, explanation, strong, weak, held))

items.sort(key=lambda x: (x[0], x[1]))
out = ['# Stable-core latent 中文审核易读版（2026-07-18 de-anchored Scorer）', '', '本文件按“待评估句子 → 模型生成的中文解释 → held-out 预测准确率”组织，供人工审核。每个标签抽取5个 latent。', '', '来源：最新 de-anchored GPT-5.5-low Scorer 评估；本文件仅重新排版并复用已完成的中文翻译，不重新生成解释或评估。主评估为 207/207 个有效 latent；clean retry 仅补跑其中2个重试任务。', '', '重要边界：这里的 card 是候选解释，held-out 指标是解释忠实度/预测一致性证据；不等同于 MISC 标签、心理机制或因果证明。', '']
for label, rank, fid, latent, explanation, strong, weak, held in items:
    out += [f'## 标签 {label} / Latent {latent} / {fid}', '', f'标签内抽样序号：{rank}', '', '### 一、待评估句子', '', '#### 1. 强组句子', '', strong, '', '#### 2. 弱组句子', '', weak, '', '#### 3. Held-out句子', '', held, '', '### 二、模型生成的中文解释', '', explanation, '', '### 三、Held-out预测准确率', '']
    m = metrics.get(fid, {})
    out += [f"- Spearman 相关系数：{m.get('spearman', '缺失')}", f"- Pearson（对数激活）：{m.get('pearson', '缺失')}", f"- 正激活与零激活 AUROC：{m.get('auroc', '缺失')}", f"- 强组–弱组排序准确率：{m.get('pair', '缺失')}", '', '---', '']
out_path.write_text('\n'.join(out), encoding='utf-8')
print(f'generated {out_path} ({len(items)} items)')
