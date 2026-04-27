# MISC 数据集 SAE 表征映射执行大纲

> 生成日期：2026-04-27  
> 依据材料：`doc/project summary.docx` 导师指导、当前项目代码、`data/mi_quality_counseling_misc` 完整 MISC 数据集。

## 1. 核心研究定位

本项目不应被定位成“训练一个 RE 分类器”，而应定位成：

> 分析咨询行为标注空间与大语言模型 SAE 表征空间之间的结构化错配。

导师指导中的中心判断是：

> 行为标签与模型内部表征之间不是简单的一一对应关系，而是存在有规律的多对多映射。不同咨询行为在 SAE latent 空间中呈现不同程度的碎片化、重叠和不对称结构。

因此，后续执行目标不是只证明“某些 latent 能区分 RE/NonRE”，而是要构建并解释：

- 一个 `Latent × Label` 关联矩阵
- 每个标签对应多少 latent 的碎片化结构
- 每个 latent 同时关联多少标签的重叠结构
- RE、QU、RES、REC、QUO、QUC 等行为之间的表征不对称性
- 这些结构对行为评估和解释性的影响

## 2. 当前数据状态

当前项目中完整 MISC 数据集实际路径为：

```text
data/mi_quality_counseling_misc
```

不是独立的 `data/misc` 目录。该目录已经包含可直接使用的数据资产：

| 数据资产 | 路径 | 用途 |
|---|---|---|
| 会话级 high/low 标签 | `metadata/labels.csv` | 质量分组、后续 high vs low 对照 |
| 原始咨询转录 | `raw_transcripts/high`, `raw_transcripts/low` | 语境追溯 |
| 咨询师话语 | `counselor_utterances/high`, `counselor_utterances/low` | 构造输入文本 |
| MISC 标注结果 | `misc_annotations/high`, `misc_annotations/low` | 多标签行为映射主数据 |
| RE/NonRE 平衡集 | `derived/re_nonre/re_dataset.jsonl`, `nonre_dataset.jsonl` | 当前主流程可直接运行 |
| 数据统计 | `derived/re_nonre/result_summary.csv` | 标签分布、质量检查 |

已知总量：

| 项目 | 数量 |
|---|---:|
| conversation labels | 259 |
| raw transcripts | 258 |
| counselor utterance files | 258 |
| MISC annotation files | 252 |
| MISC 标注行为单元 | 6194 |
| RE 总数 | 1358 |
| RES | 516 |
| REC | 842 |
| QU 总数 | 1974 |
| QUO | 1206 |
| QUC | 768 |
| 当前平衡 RE 集 | 799 |
| 当前平衡 NonRE 集 | 799 |

注意：当前主流程可直接读取的是 `derived/re_nonre` 下的二分类 split；完整多标签结构分析应读取 `misc_annotations` 下所有 MISC 标注。

## 3. 当前项目已实现能力

| 研究需要 | 当前实现 | 主要输出 |
|---|---|---|
| 加载 Llama-3.1-8B 与 OpenMOSS SAE | `src/nlp_re_base/model.py`, `src/nlp_re_base/sae.py` | 官方 `lm-saes` SAE 前向 |
| 抽取 layer 19 residual 并运行 SAE | `src/nlp_re_base/activations.py` | utterance-level SAE features |
| RE/NonRE 功能指标 | `src/nlp_re_base/eval_functional.py` | `candidate_latents.csv`, `metrics_functional.json` |
| 结构保真指标 | `src/nlp_re_base/eval_structural.py`, `diagnostics.py` | `metrics_structural.json` |
| latent 高激活样本卡片 | `maxact_analysis`, `latent_cards/` | 每个候选 latent 的 top activation 文本 |
| AI 评审 bundle | `src/nlp_re_base/ai_re_judge.py` | `judge_bundle/` |
| 因果验证 | `causal/run_experiment.py` | necessity, sufficiency, selectivity, group results |
| 进度与错误日志 | `causal/run_experiment.py` | `run_status.json`, `run_events.jsonl`, `causal_run.log`, `fatal_traceback.log` |

这些能力已经足够支撑第一阶段 RE/NonRE 实验和因果验证；但导师指导中要求的完整 `Latent × Label` 多标签矩阵，目前还需要在现有功能指标逻辑上做一次扩展。

## 4. 执行路线总览

建议把后续工作分为三条线：

1. **主线 A：用完整 MISC 派生出的 RE/NonRE 平衡集复现实验**
   - 目标：复用当前代码，先得到稳定的 RE 相关 latent、结构指标和因果验证结果。
   - 数据：`data/mi_quality_counseling_misc/derived/re_nonre`
   - 结论定位：证明 SAE 空间中存在 RE 相关候选 latent 子空间。

2. **主线 B：构建 MISC 多标签 Latent × Label 矩阵**
   - 目标：从 `misc_annotations` 读取完整 MISC 行为标签，计算每个 latent 与每个标签的关联。
   - 数据：`data/mi_quality_counseling_misc/misc_annotations`
   - 结论定位：回答“标签与表征是否多对多、碎片化、重叠、行为间不对称”。

3. **主线 C：面向论文叙事的结构解释与因果补强**
   - 目标：把矩阵结构、latent 卡片、因果验证和 AI 评审合并成可写入报告/论文的证据链。
   - 结论定位：从“RE latent 有效”推进到“行为标签与内部表征存在结构化错配”。

## 5. 主线 A：当前代码可直接执行的 RE/NonRE 实验

### 5.1 运行 SAE 主评估

本地 PowerShell 推荐命令：

```powershell
conda activate qwen-env-py311; `
$env:MODEL_DIR="D:\project\NLP_v3\NLP_data\Llama-3.1-8B"; `
python run_sae_evaluation.py `
  --model-dir "$env:MODEL_DIR" `
  --device cuda `
  --data-dir data/mi_quality_counseling_misc/derived/re_nonre `
  --batch-size 4 `
  --max-seq-len 128 `
  --full-structural `
  --checkpoint-topk-semantics hard `
  --output-dir outputs/misc_re_nonre_sae_eval
```

重点检查输出：

```text
outputs/misc_re_nonre_sae_eval/metrics_structural.json
outputs/misc_re_nonre_sae_eval/metrics_functional.json
outputs/misc_re_nonre_sae_eval/candidate_latents.csv
outputs/misc_re_nonre_sae_eval/latent_cards/
outputs/misc_re_nonre_sae_eval/judge_bundle/
```

### 5.2 运行因果验证

```powershell
python causal/run_experiment.py `
  --model-dir "$env:MODEL_DIR" `
  --device cuda `
  --data-dir data/mi_quality_counseling_misc/derived/re_nonre `
  --candidate-csv outputs/misc_re_nonre_sae_eval/candidate_latents.csv `
  --checkpoint-topk-semantics hard `
  --output-dir outputs/misc_re_nonre_causal_validation
```

重点检查输出：

```text
outputs/misc_re_nonre_causal_validation/results_necessity.json
outputs/misc_re_nonre_causal_validation/results_sufficiency.json
outputs/misc_re_nonre_causal_validation/results_selectivity.json
outputs/misc_re_nonre_causal_validation/results_group.json
outputs/misc_re_nonre_causal_validation/run_status.json
outputs/misc_re_nonre_causal_validation/causal_run.log
```

### 5.3 这一阶段可以回答的问题

- SAE 中是否存在与 RE/NonRE 显著相关的 latent？
- top-k sparse probe 是否能有效区分 RE？
- 消融 G1/G5/G10/G20 是否会削弱 RE 方向？
- 注入 G1/G5/G10/G20 是否会增强 RE 方向？
- 随机 latent、bottom latent、正交方向是否明显弱于目标 latent 组？

### 5.4 这一阶段不能单独回答的问题

- RE 与 QU、GI、SU、AF 等标签之间是否存在表征重叠？
- RES 与 REC 是否对应不同 latent 子结构？
- QUO 与 QUC 是否比 RE 更紧凑或更碎片化？
- 一个 latent 是否同时对应多个 MISC 行为？

这些问题需要主线 B 的多标签矩阵。

## 6. 主线 B：MISC 多标签 Latent × Label 矩阵

### 6.1 目标

从 `misc_annotations/high` 和 `misc_annotations/low` 中读取完整行为标注，形成：

```text
utterance -> predicted_code / predicted_subcode / confidence
utterance -> SAE latent activations
```

然后对每个标签和每个 latent 计算关联指标：

```text
label in {RE, RES, REC, QU, QUO, QUC, GI, SU, AF, ...}
latent in {0 ... 32767}
metric in {AUC, Cohen's d, p-value, FDR, precision@k, top activation purity}
```

核心输出应是：

```text
outputs/misc_label_mapping/latent_label_matrix.csv
outputs/misc_label_mapping/label_fragmentation.json
outputs/misc_label_mapping/latent_overlap.json
outputs/misc_label_mapping/behavior_asymmetry.md
outputs/misc_label_mapping/top_latents_by_label/
```

### 6.2 可复用的现有代码逻辑

当前 `eval_functional.py` 已经实现了二分类版本的：

- Cohen's d
- AUC
- p-value
- BH-FDR
- sparse probe
- MaxAct
- feature absorption
- feature geometry
- TPP

多标签矩阵不需要重写这些统计思想，只需要把输入从：

```text
RE features vs NonRE features
```

扩展为：

```text
label-positive features vs label-negative features
```

即对每个 MISC 标签循环执行一次 univariate association。

### 6.3 标签定义建议

建议至少建立三层标签：

| 层级 | 标签 |
|---|---|
| 粗粒度行为 | `RE`, `QU`, `GI`, `SU`, `AF`, `OTHER` |
| RE 子类 | `RES`, `REC` |
| QU 子类 | `QUO`, `QUC` |

如果原始 annotation 中还存在更多 MISC code，应从 `predicted_code` 和 `predicted_subcode` 自动枚举，不要只写死 RE/QU。

### 6.4 矩阵分析指标

围绕导师指导，矩阵分析应对应三组研究问题。

#### R1. Mapping Structure

问题：标签与 latent 的基本映射形态是什么？

建议指标：

- 每个 label 的显著 latent 数量
- 每个 label 的 top latent AUC / Cohen's d 分布
- 每个 latent 关联的 label 数量
- 单标签 latent、双标签 latent、多标签 latent 的比例
- label-latent 二部图稀疏度

对应结论：

> 行为标签与 SAE 表征不是一一对应，而是多对多映射。

#### R2. Asymmetry Across Behaviors

问题：不同咨询行为的错配方式是否相同？

建议指标：

- `RE` vs `QU` 的显著 latent 数量差异
- `RES` vs `REC` 的 latent 重叠率
- `QUO` vs `QUC` 的 latent 重叠率
- 每个 label 的 top latent 集合 Jaccard 相似度
- 每个 label 的碎片化指数
- 每个 label 的重叠指数

对应结论：

> 不同行为在表征空间中的结构不同，有的更碎片化，有的更紧凑，有的与其他行为高度重叠。

#### R3. Implications

问题：这种结构对评估和解释有什么影响？

建议分析：

- 离散标签是否压缩了模型内部更细的区别？
- 某些 latent 是否跨越多个人工标签？
- RE 相关 latent 是否混入 question/advice/support 等行为信号？
- 如果标签之间共享 latent，单纯用标签评价模型行为是否会误导？

对应结论：

> 标注体系并不能完全反映模型内部如何编码咨询行为。

## 7. 主线 C：结构解释、因果补强与报告产出

### 7.1 latent 卡片解释

对每个关键标签输出 top latent 后，需要抽样检查：

- top activation 文本是否语义一致？
- latent 是否更像某个明确行为，还是混合了多个行为？
- RE latent 是偏“复述内容”、偏“情绪反映”、还是偏“咨询对话模板”？
- QU latent 是偏“开放问题”、偏“封闭问题”、还是偏一般疑问句结构？

当前 `latent_cards/` 和 `judge_bundle/` 可以直接复用。

### 7.2 AI 评审

AI 评审不应只问“是不是 RE”，还应扩展为：

- 该 latent 的 top examples 是否表现出一致的 MISC 行为？
- 是否同时包含多个 MISC 标签？
- 是否更像语言形式特征，而不是咨询行为特征？
- top examples 与 control examples 的差异是否明显？

### 7.3 因果验证

当前因果验证可先用于 RE/NonRE 主线。多标签版本可逐步扩展：

- 对 RE top latent 继续跑 necessity/sufficiency
- 对 QU top latent 做同样验证
- 检查 RE latent 注入是否同时提高 QU 或其他行为方向
- 检查多标签 overlap latent 是否比单标签 latent 更容易产生副作用

## 8. 最小可执行版本

为了尽快形成一版结果，建议先按以下顺序执行：

1. 用 `derived/re_nonre` 跑通当前 `run_sae_evaluation.py`
2. 用同一输出跑通 `causal/run_experiment.py`
3. 从 `misc_annotations` 构建完整 utterance-level 多标签表
4. 复用 SAE 抽取逻辑，为所有 MISC utterance 生成 latent features
5. 复用二分类 univariate 逻辑，循环每个 label 生成 `latent_label_matrix.csv`
6. 汇总 `label_fragmentation.json` 和 `latent_overlap.json`
7. 针对 RE、RES、REC、QU、QUO、QUC 写第一版结构分析报告

第一版报告的核心表格建议包括：

| 表格 | 内容 |
|---|---|
| Table 1 | MISC 标签分布 |
| Table 2 | 每个标签的显著 latent 数、top AUC、top Cohen's d |
| Table 3 | 标签之间 top-k latent Jaccard 相似度 |
| Table 4 | 单标签 latent vs 多标签 latent 数量 |
| Table 5 | RE/QU 子标签碎片化与重叠对比 |

## 9. 预期论文叙事

如果执行顺利，论文或总报告可以按下面逻辑组织：

1. **数据与标注空间**
   - MISC 行为单元
   - RE/QU 及其子标签
   - high/low counseling conversation 背景

2. **SAE 表征空间**
   - Llama-3.1-8B layer 19 residual
   - OpenMOSS SAE
   - 32768 sparse latents

3. **二分类主结果**
   - RE/NonRE 可以被 SAE latent 区分
   - top-k latent 有明显统计关联
   - 因果干预显示目标 latent 组有非随机效应

4. **多标签映射结构**
   - label 到 latent 是一对多
   - latent 到 label 是多对多
   - 不同行为呈现不同碎片化与重叠模式

5. **解释与影响**
   - 人工标签压缩了模型内部表征差异
   - 离散行为评估可能掩盖共享机制和混合信号
   - SAE 可作为审计咨询行为表征结构的工具

## 10. 风险与注意事项

- 当前 `run_sae_evaluation.py` 默认是二分类接口，不能直接产出完整多标签矩阵。
- `derived/re_nonre` 是高置信平衡集，适合跑 RE 主线，但不代表完整 MISC 分布。
- `misc_annotations` 中标签置信度不完全一致，多标签矩阵应记录并可选过滤 `confidence`。
- 部分 MISC 文件缺失，应在报告中说明，不要假设 259 个会话全部有 annotation。
- 如果一个 latent 同时关联多个标签，不应急着判定为“噪声”，它可能正是导师所说的结构化重叠。
- 因果验证结果应和矩阵结构一起解释，避免把“能推动 RE probe”直接写成“发现 RE 概念机制”。

## 11. 当前结论边界

当前项目已经具备证明“RE 相关 latent 子空间存在”的实验基础；新的 MISC 数据集使项目可以进一步升级为导师建议的核心问题：

> 咨询行为标签与 LLM 内部 SAE 表征之间存在结构化、多对多、行为依赖的映射关系。

下一步最关键的新增产物不是新的模型，而是：

```text
Latent × MISC Label association matrix
```

只要该矩阵和后续结构分析完成，项目叙事就可以从“RE 特征发现”推进到“咨询行为标注体系与模型内部表征空间的结构性错配分析”。

## 12. 已实现的多标签矩阵运行入口

当前已经新增脚本：

```text
run_misc_label_mapping.py
```

该脚本支持两种运行方式。

### 12.1 直接抽取 SAE features 并生成矩阵

适合正式运行：

```powershell
conda activate qwen-env-py311; `
$env:MODEL_DIR="D:\project\NLP_v3\NLP_data\Llama-3.1-8B"; `
python run_misc_label_mapping.py `
  --model-dir "$env:MODEL_DIR" `
  --device cuda `
  --data-dir data/mi_quality_counseling_misc `
  --batch-size 4 `
  --max-seq-len 128 `
  --checkpoint-topk-semantics hard `
  --output-dir outputs/misc_label_mapping_full
```

### 12.2 复用已抽取 features 生成矩阵

适合调试、重复分析、调整标签阈值：

```powershell
python run_misc_label_mapping.py `
  --data-dir data/mi_quality_counseling_misc `
  --features-path outputs/misc_label_mapping_full/utterance_features.pt `
  --output-dir outputs/misc_label_mapping_full_reanalysis `
  --labels RE RES REC QU QUO QUC GI SU AF `
  --min-positive 10 `
  --min-negative 10 `
  --precision-k 10 50 `
  --top-k-per-label 50
```

### 12.3 主要输出

```text
latent_label_matrix.csv          # Latent × Label 长表
label_summary.json               # 每个标签的样本量和占比
label_fragmentation.json         # 每个标签对应多少显著 latent
latent_overlap.json              # 每个 latent 关联多少标签
label_topk_jaccard.json          # 标签之间 top-k latent 重叠
behavior_asymmetry.md            # 自动生成的结构分析摘要
top_latents_by_label/            # 每个标签的 top latent CSV
top_examples_by_label/           # 每个标签 top latent 的高激活样本卡片
annotation_records.jsonl         # 实际参与分析的 MISC 记录
label_indicator_matrix.csv       # utterance × label 指示矩阵
```

### 12.4 已验证的测试入口

不加载大模型的 smoke test：

```powershell
conda run -n qwen-env-py311 python test_misc_label_mapping_smoke.py
```

原有 SAE-RE smoke test：

```powershell
conda run -n qwen-env-py311 python test_pipeline_smoke.py
```
