# CACTUS 数据集处理说明

## 概述

本模块将 [LangAGI-Lab/cactus](https://huggingface.co/datasets/LangAGI-Lab/cactus) 数据集处理为用于 SAE 可解释性验证的小规模标注数据集（1500 条），三类各 500 条：

| 类别 | 含义 |
|------|------|
| `RE` | 反映式倾听（therapist 主要在反映/镜映 client 体验） |
| `NonRE_CBT` | CBT 技术动作（evidence checking, reframing, homework 等） |
| `NonTech_Process` | 流程/寒暄/会谈管理 |

> **原始数据集保留不变**：`data/mi_re/` 目录中的原 MI-RE 数据集不受影响。

---

## 文件结构

```
build_cactus_dataset.py   ← 主处理脚本
verify_cactus_pipeline.py ← 端到端链路验证脚本
data/
  mi_re/                  ← 原始 MI 数据集（不动）
  cactus/                 ← 新生成（运行后自动创建）
    cactus_re_small_1500.jsonl
    cactus_re_small_1500_stats.json
    cactus_re_small_1500_preview.csv
```

---

## 如何运行

### 步骤 1：安装依赖
```bash
pip install datasets
```

### 步骤 2：构建数据集
```bash
conda activate qwen-env
python build_cactus_dataset.py
```

可选参数：
```
--target-per-class 500   每类样本数（默认500）
--seed 42                随机种子（默认42）
--output-dir data/cactus 输出目录（默认data/cactus）
```

### 步骤 3：验证端到端链路（需要 GPU + 已下载模型）
```bash
python verify_cactus_pipeline.py --n-samples 10 --topk-show 10
```

---

## 输出文件说明

### `cactus_re_small_1500.jsonl`
每行一条 JSON 样本，字段如下：

| 字段 | 说明 |
|------|------|
| `sample_id` | 唯一 ID，格式 `cactus_XXXXXX_tYYY` |
| `label` | `RE` / `NonRE_CBT` / `NonTech_Process` |
| `client_prev` | 前一轮 client 发言 |
| `therapist_curr` | 当前 therapist 发言 |
| `formatted_text` | XML 模板化完整文本，用于输入模型 |
| `therapist_char_start` | therapist 内容在 `formatted_text` 中的字符起始位置 |
| `therapist_char_end` | therapist 内容在 `formatted_text` 中的字符结束位置 |
| `source_cbt_technique` | 来自 CACTUS 原始字段 |

### `cactus_re_small_1500_stats.json`
每类样本统计（数量、平均词数、问句占比、technique 分布）。

### `cactus_re_small_1500_preview.csv`
可直接用 Excel/Numbers 打开的人工抽查文件。

---

## 规则筛选逻辑

标签通过规则匹配 `therapist_curr` 文本自动分配：

### RE 规则
**包含**（至少 1 条）：
- `it sounds like`, `sounds like you`, `you feel`, `i hear that`, `what i'm hearing`, `so you`, `it's as though` 等反映式表达

**排除**（不含）：
- 以 `?` 结尾的疑问句
- 含有 `let's look at`, `can you try`, `homework`, `experiment` 等技术推进短语

### NonRE_CBT 规则
**包含**：`what evidence`, `is there another way`, `let's try`, `behavioral plan`, `thought record`, `reframe` 等 CBT 技术动作

**排除**：寒暄/流程类开头

### NonTech_Process 规则
**包含**：`hi`, `nice to meet`, `thank you for sharing`, `let's start`, `see you next` 等流程句

**排除**：CBT 技术动作

若一条样本同时匹配多个类别（混合句），直接丢弃。

---

## Therapist Token Span 定位方式

1. 用 XML 模板生成 `formatted_text`：
   ```
   <client>
   {client_prev}
   </client>
   <therapist>
   {therapist_curr}
   </therapist>
   ```
2. 用字符串查找定位 `<therapist>\n` 后 therapist 内容的字符区间
   → 存入 `therapist_char_start` / `therapist_char_end`
3. 后续 tokenize 时开启 `return_offsets_mapping=True`，用字符区间映射找到 token span
4. **只在 therapist token span 内聚合 SAE latent activation**（不混入 client tokens）
