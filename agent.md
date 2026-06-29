# AI Agent Context

本文件给自动化代理和其他 AI 使用，帮助它们在本仓库中正确读取 MISC 数据、理解标签分布，并避免常见错误。更完整的运行规则见 `CLAUDE.md`。

## 项目任务

本项目研究 MISC 心理咨询行为标签与 Llama/Gemma hidden states、SAE latents 之间的可解释关系。多数输出位于：

```text
outputs/misc_full_sae_eval
```

核心问题不是“某个 latent 是否等于某个标签”，而是：

```text
人工 MISC 标签能否从 LLM 内部表征中被识别？
哪些 SAE latents 或 latent 子空间与标签相关？
这些 latent 更像行为功能、表面形式、上下文关系，还是数据 artifact？
```

## Canonical 数据入口

优先使用已合并产物：

```text
outputs/misc_full_sae_eval/records.jsonl
outputs/misc_full_sae_eval/label_matrix.csv
```

原始数据根目录：

```text
data/mi_quality_counseling_misc
```

原始目录结构：

```text
metadata/labels.csv
raw_transcripts/high/*.txt
raw_transcripts/low/*.txt
counselor_utterances/high/*.jsonl
counselor_utterances/low/*.jsonl
misc_annotations/high/*.jsonl
misc_annotations/low/*.jsonl
derived/re_nonre/re_dataset.jsonl
derived/re_nonre/nonre_dataset.jsonl
```

不要默认使用 `data/mi_re`。它是 legacy RE/Non-RE 兼容数据，只有用户或脚本显式指定时才用。

## 合并数据格式

`records.jsonl` 每行是一个 counselor 当前 utterance 行为单元，典型字段：

```json
{
  "file_id": "high_001",
  "unit_text": "hey Monica how are you doing today",
  "predicted_code": "QU",
  "predicted_subcode": "QUO",
  "rationale": "short annotation rationale",
  "confidence": 0.86,
  "record_id": "high_001:0001",
  "quality_label": "high",
  "labels": ["QU", "QUO"],
  "source_split": "high",
  "source_file": "high_001.jsonl",
  "source_line": 1
}
```

`label_matrix.csv` 是机器学习和探针脚本的主表，形状为：

```text
6194 rows × 19 columns
```

列：

```text
row_idx
record_id
file_id
source_split
source_file
predicted_code
predicted_subcode
confidence
unit_text
RE RES REC QU QUO QUC GI SU AF OTHER
```

行对齐规则：

```text
label_matrix.csv row_idx
records.jsonl line index
utterance_features.pt first dimension
utterance_activations.pt first dimension
```

这些必须一一对应。不要打乱行顺序后直接复用 feature tensor。

## 标签体系

9 个核心标签：

| 标签 | 含义 | 角色 |
|---|---|---|
| `RE` | Reflection 父类 | 父标签 |
| `RES` | Simple reflection | leaf |
| `REC` | Complex reflection | leaf |
| `QU` | Question 父类 | 父标签 |
| `QUO` | Open question | leaf |
| `QUC` | Closed question | leaf |
| `GI` | Giving information | leaf |
| `SU` | Support | leaf |
| `AF` | Affirmation | leaf |

重要层级关系：

```text
RE == RES OR REC
QU == QUO OR QUC
```

因此 `RE-REC`、`RE-RES`、`QU-QUO`、`QU-QUC` 的共现主要是标签层级结构，不是模型独立发现的跨标签共享。

`OTHER` 表示非核心标签集合。`predicted_code` 中这些非核心 MISC/会话代码会进入 `OTHER` 口径：

```text
FA ST AD FI DI NR RC CO WA EC RF
```

## 当前分布

总样本数：

```text
6194 counselor utterance units
```

来源 split：

| source_split | rows | files |
|---|---:|---:|
| high | 4169 | 153 |
| low | 2025 | 99 |

`source_split` 是原会话质量来源，不是训练/测试划分。

核心标签计数：

| label | count | prevalence |
|---|---:|---:|
| QU | 1974 | 31.9% |
| RE | 1358 | 21.9% |
| QUO | 1206 | 19.5% |
| REC | 842 | 13.6% |
| QUC | 768 | 12.4% |
| GI | 681 | 11.0% |
| RES | 516 | 8.3% |
| AF | 349 | 5.6% |
| SU | 222 | 3.6% |
| OTHER | 1610 | 26.0% |

每行核心标签激活数：

| active core labels | rows | 说明 |
|---:|---:|---|
| 0 | 1610 | 通常是 OTHER |
| 1 | 1252 | GI/SU/AF 等单标签，或不带子类的口径 |
| 2 | 3332 | 主要是父子标签共现，例如 QU+QUO、RE+REC |

按 high/low 的标签计数：

| split | RE | RES | REC | QU | QUO | QUC | GI | SU | AF | OTHER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| high | 1139 | 395 | 744 | 1417 | 941 | 476 | 329 | 121 | 248 | 915 |
| low | 219 | 121 | 98 | 557 | 265 | 292 | 352 | 101 | 101 | 695 |

分布模式：

- high 会话中 `RE`, `REC`, `QUO`, `AF` 更常见。
- low 会话中 `GI`, `QUC`, `OTHER` 相对更多。
- `SU` 和 `AF` 是低频标签，训练探针和解释时需要注意类别不均衡。
- `OTHER` 占 26%，但不是一个单一行为概念，而是多个非核心代码的集合。

## 文本和标注质量特征

`unit_text` 是 counselor 当前 utterance，不含前一句 client context。不要凭当前行强判 RES/REC 是否真正复述了 client 内容。

文本长度：

```text
mean words ≈ 13.1
median words = 10
max words = 50
empty text = 0
```

标注置信度：

```text
mean confidence ≈ 0.862
median confidence = 0.90
min = 0.20
max = 0.99
```

重复文本：

```text
unique normalized unit_text ≈ 5392
duplicate-text rows ≈ 1214
```

重复文本不能简单忽略。它可能反映常见模板、转录重复、固定咨询话术或 artifact。做 top activating utterance 审阅时应显式标注 duplicate risk。

## Feature tensor 对齐

当前 Llama 主线：

```text
outputs/misc_full_sae_eval/feature_store/utterance_features.pt
shape = [6194, 32768]
```

raw hidden 聚合：

```text
outputs/misc_full_sae_eval/feature_store/utterance_activations.pt
shape = [6194, 4096]
```

metadata：

```text
outputs/misc_full_sae_eval/feature_store/feature_metadata.json
```

关键设置：

```text
hook_point = blocks.19.hook_resid_post
aggregation = max
max_seq_len = 128
save_token_topk = false
```

当前没有保存完整 token-level latent 激活。严格的 token-position 机制结论或 token attribution 需要新增导出，不要从 utterance-level max 激活中硬推。

## 解释和报告边界

可以说：

```text
人工标签可从 raw hidden / PCA / SAE feature space 中被探针识别。
某些 SAE latents 与标签呈相关或预测充分关系。
top activating utterances 支持候选功能解释。
```

不能说：

```text
某个 SAE latent 就是某个 MISC 标签。
top activating examples 证明模型理解了该咨询技能。
当前结果已经证明因果机制。
```

对 RES/REC/RE 的固定 caveat：

```text
当前合并数据没有 previous client utterance。
只能分析 counselor utterance 内部的 reflective phrasing 或表面形式。
不能单凭当前句证明它复述/改写了 client 内容。
```

## 实用命令

快速检查数据分布：

```powershell
C:\Users\chengyu\AppData\Local\miniconda3\envs\qwen-env-py311\python.exe -c "import pandas as pd; lm=pd.read_csv('outputs/misc_full_sae_eval/label_matrix.csv'); labels=['RE','RES','REC','QU','QUO','QUC','GI','SU','AF']; print(lm.shape); print(lm[labels+['OTHER']].sum().astype(int).to_dict()); print(lm['source_split'].value_counts().to_dict())"
```

快速检查 feature shape：

```powershell
Get-Content -Raw -Encoding UTF8 outputs\misc_full_sae_eval\feature_store\feature_metadata.json
```

运行 Python 脚本时优先使用：

```powershell
C:\Users\chengyu\AppData\Local\miniconda3\envs\qwen-env-py311\python.exe <script>.py
```

