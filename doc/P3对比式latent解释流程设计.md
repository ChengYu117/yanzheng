# P3 对比式 LLM 辅助 SAE Latent 解释流程设计

> Contrastive LLM-assisted SAE Latent Interpretation
> 替换旧 P3 Task A/B/C/D 启发式流程，基于 stable_core latent 集合重建。
> 更新：2026-07-09

---

## 1. 研究目标与流程总览

### 1.1 三个研究问题的对应关系

| RQ | 问题 | 本流程对应步骤 |
|---|---|---|
| **RQ1** | 稀疏 SAE 表征是否能忠实保留心理咨询行为信息？ | Step 6 下游 probe 验证 |
| **RQ2** | 被选中的稀疏特征是否能在语义上被解释为有意义的心理咨询行为？ | Step 2 Explainer + Step 3 Scorer + Step 4 Minimal Pair |
| **RQ3** | 这些被解释的特征揭示了 LLM 内部心理咨询行为的何种组织方式？ | Step 5 子概念聚合 |

### 1.2 完整流程概览

```
stable_core latent 集合（`stable_set_role == stable_core`，303 条；完整 CSV 共 455 行）
    ↓
Step 1：构建对比式样本包（ACTIVE_HIGH / ACTIVE_MID / ACTIVE_LOW / NONACTIVE_NEAR_MISS / NONACTIVE_RANDOM）
    ↓
Step 2：Explainer LLM 生成子概念假设（label-blind，结构化 JSON）
    ↓
Step 3：Scorer LLM 在 held-out 激活检测样本上验证（AUROC 阈值 0.70/0.60）+ label-definition baseline
    ↓
Step 4：代表 latent 的 Minimal Pair Activation Test（SAE 前向推理）
    ↓
Step 5：子概念聚合（MISC 标签 → 子概念 → 代表 latents）
    ↓
Step 6：下游验证（复用 probe 对比 + 新增 probe-space feature ablation）
```

### 1.3 核心设计原则

**不做**：不解释全部 32768 个 latent；不做复杂 agent；不把 LLM 输出当最终证据；不再用启发式/规则函数替代 LLM。

**要做**：用对比式样本包逼迫 AI 提出**可被反例推翻的子概念假设**，再用 held-out detection、minimal pairs、下游 probe/ablation 验证。

---

## 2. 范围与分级

### 2.1 Latent 集合

来源：`outputs/cross_val/stable_topk_selection/stable_topk_latent_set.csv`。

注意：该 CSV 当前共 455 行；本流程只使用其中 `stable_set_role == "stable_core"` 的 303 条。其他 `boundary_candidate`、`unstable_inclusion`、`ci_not_supported` 行只作为审计背景，不进入主解释对象。

| Tier | 标签 | stable_core 数 | label_selection_status | 处理策略 |
|---|---|---:|---|---|
| **主层级** | QU | 26 | stable_topk_found | 完整 Step 1-6 |
| **主层级** | QUO | 32 | stable_topk_found | 完整 Step 1-6 |
| **主层级** | QUC | 45 | stable_topk_found | 完整 Step 1-6 |
| **主层级** | RE | 54 | stable_topk_found | 完整 Step 1-6 |
| **主层级** | REC | 47 | stable_topk_found | 完整 Step 1-6 |
| **主层级** | AF | 27 | stable_topk_found | 完整 Step 1-6 |
| **副层级** | SU | 31 | performance_only_unstable | Step 1-3，不做 Minimal Pair，子概念标"待验证" |
| **副层级** | GI | 26 | performance_only_unstable | 同上 |
| **副层级** | RES | 15 | performance_only_unstable | 同上 + 声明缺 client context 限制 |

主层级 231 个 latent，副层级 72 个，总计 303 个。

### 2.2 底层模型与 SAE 信息

| 项目 | 值 |
|---|---|
| 基础模型 | Llama-3.1-8B |
| Hook 点 | `blocks.19.hook_resid_post` |
| SAE 维度 | 32768 |
| 激活聚合 | utterance-level max pooling |
| 特征文件 | `outputs/misc_full_sae_eval/feature_store/utterance_features.pt` [6194, 32768] |
| 原始 hidden | `outputs/misc_full_sae_eval/feature_store/utterance_activations.pt` [6194, 4096] |

### 2.3 输出根目录

```
outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/
├── evidence_packs/       # Step 1
├── explainer_outputs/    # Step 2
├── scorer_outputs/       # Step 3
├── minimal_pairs/        # Step 4
├── subconcepts/          # Step 5
├── ablation/             # Step 6
└── manifest.json
```

---

## 3. Step 1：构建对比式样本包

### 3.1 设计目的

旧流程只给 AI 看 top 激活样本，导致 AI 只总结表面形式（如"有问号 = 问题 latent"）。对比式样本包加入**中等激活、边界低激活、近邻非激活样本、随机非激活样本**，逼迫 AI 区分"真正触发 latent 的整合模式"与"表面相似但不触发的文本"。

### 3.2 样本定义

样本包内部保留 MISC 标签、target_match、active_labels 等元数据；但传给 Explainer 的序列化版本必须只暴露中性 tag，不能暴露目标标签名或"同标签"关系。

| 内部类型 | Explainer 可见 Tag | 数量 | 采样逻辑 |
|---|---|---:|---|
| 高激活样本 | `ACTIVE_HIGH` | 5 | 该 latent 激活值 top |
| 中高激活样本 | `ACTIVE_MID` | 4 | 有激活样本的 50-75th 百分位，随机取 |
| 低激活边界 | `ACTIVE_LOW` | 2 | 有激活样本的 5-20th 百分位（边界案例） |
| 表面/邻近非激活 | `NONACTIVE_NEAR_MISS` | 4 | 表面形式、父子/兄弟标签或语义相近，但该 latent 激活低于检测阈值 |
| 目标标签正例但不激活 | `NONACTIVE_NEAR_MISS` | 3 | 目标标签正例但该 latent 激活在底 20%；只作为内部来源，不把"目标标签正例"告诉 Explainer |
| 随机非激活 | `NONACTIVE_RANDOM` | 2 | 随机样本，激活≈0 |

每个 latent 约 20 条。

### 3.3 Held-out 保留（供 Step 3 使用，与样本包不重叠）

关键：Step 1 构建样本包时**同时预留** Step 3 的 held-out 集，两者行索引严格不相交，运行时校验每个声明的 held-out tag 都有对应保留逻辑。

| Held-out Tag | 数量 | 激活检测 ground truth | 来源 |
|---|---|---|---|
| `ACTIVE_HIGH` | 5 | 1 | top 激活池的保留部分，激活值高于检测阈值 |
| `ACTIVE_MID` | 5 | 1 | 中高激活保留部分，激活值高于检测阈值 |
| `NONACTIVE_NEAR_MISS` | 5 | 0 | 表面/邻近相似但激活值低于检测阈值 |
| `NONACTIVE_LABEL_MATCH` | 5 | 0 | 目标标签正例但激活底 20%；Scorer 可见 tag 仍改写为 `NONACTIVE_NEAR_MISS` |

检测阈值固定为每个 latent 在训练池中非零激活值的中位数（`q50_nonzero_activation`）。Step 3 的 ground truth 只回答"这条文本是否应触发该 latent"，不回答"这条文本是否属于目标 MISC 标签"。

> 说明：`NONACTIVE_LABEL_MATCH` 是为了测试"标签对但 latent 不触发"的反例，但这个来源只写入内部审计字段，不能暴露给 Explainer，否则会破坏 label-blind 设定。

### 3.4 样本包格式

```json
{
  "latent_idx": 12345,
  "target_label": "QUO",
  "rank_within_label": 3,
  "stable_set_role": "stable_core",
  "auc": 0.921, "cohens_d": 1.43,
  "inclusion_frequency": 0.84,
  "tier": "primary",
  "samples_internal": [
    {"id":"s001","tag":"ACTIVE_HIGH","activation":4.23,
     "text":"What would make it easier for you to take that first step?",
      "target_match":1,"active_labels":"QU,QUO","source_split":"high",
      "internal_source":"top_activation"}
  ],
  "samples_for_explainer": [
    {"id":"s001","tag":"ACTIVE_HIGH","activation":4.23,
     "text":"What would make it easier for you to take that first step?"}
  ],
  "heldout_internal_by_tag": {"ACTIVE_HIGH":[...],"ACTIVE_MID":[...],"NONACTIVE_NEAR_MISS":[...],"NONACTIVE_LABEL_MATCH":[...]}
}
```

### 3.5 实现入口

计划入口：

```
src/nlp_re_base/contrastive_evidence_pack.py
run_misc_contrastive_latent_interp.py --step build-packs
```

> 此步骤**不需要 GPU**，全部从 `utterance_features.pt` + `label_matrix.csv` 计算。当前仓库尚未落地上述入口；文档中的 303 latent 是已核对的 stable_core 目标规模，不代表 Step 1 已生成完成。

---

## 4. Step 2：Explainer LLM 生成子概念假设

### 4.1 设计原则

- **Label-blind**：提示词**不得包含目标 MISC 标签名**。序列化层做投影，只输出 `id / tag / activation / text`，过滤掉 `target_match`、`active_labels`、包级 `target_label`。发送前用整词正则守卫检测 MISC 标签 token，命中则报错拦截。
- **可证伪假设**：不是"总结共同主题"，而是"提出最窄的、可被反例推翻的子概念假设"。重点解释"为什么 `NONACTIVE_NEAR_MISS` 没触发"。
- **多次采样**：每个 latent 至少 2 次，报告一致性（short_name / feature_type 是否一致）。
- **LLM 执行方式**：使用 Antigravity IDE 中的 Gemini 3.5。当前不假设存在可脚本调用的 API；代码只生成 prompt 任务文件、校验 Gemini 返回 JSON、汇总指标。

### 4.2 输出 JSON Schema（10 字段）

```json
{
  "latent_idx": 12345,
  "short_name": "开放式探索问题",
  "main_hypothesis": "该 latent 在提出开放式、邀请展开自我叙述的问题时激活。",
  "positive_triggers": ["以 what/how/why 引导回答","面向内在感受/原因/选择","非简单事实确认"],
  "explicit_exclusions": ["封闭式 yes/no 问题","只有问号但无探索功能","泛泛支持性回应"],
  "possible_surface_confounds": ["question mark","second-person pronoun","sentence length"],
  "feature_type": "discourse/pragmatic",
  "confidence": 0.76,
  "alternative_hypotheses": ["第二人称咨询式提问","开放式问题模板"],
  "key_evidence": ["s001","s003","s007"],
  "failure_modes": "how/what 开头的陈述句可能误判"
}
```

### 4.3 提示词模板

```
你正在解释一个 LLM 内部的 SAE latent（稀疏自编码器特征单元）。

任务：
- 不是总结样本的共同主题，而是提出一个【可被反例推翻的、尽量窄的触发假设】。
- 重点区分：为什么 NONACTIVE_NEAR_MISS 样本没有触发这个 latent？
- 你的解释应明确"什么样的文本一定不触发"。

样本标记说明：
[H] ACTIVE_HIGH：激活最高，latent 明确触发
[M] ACTIVE_MID：中高激活，latent 触发但不极端
[L] ACTIVE_LOW：低激活边界，接近不触发
[N] NONACTIVE_NEAR_MISS：表面或语义相近但不触发——关键反例
[R] NONACTIVE_RANDOM：随机非激活样本

样本：
{formatted_samples}

请输出 JSON，字段：short_name(≤8字,不含MISC标签名)/main_hypothesis/positive_triggers/
explicit_exclusions/possible_surface_confounds/feature_type/confidence/
alternative_hypotheses/key_evidence/failure_modes
```

### 4.4 Antigravity / Gemini 3.5 执行约定

由于 Gemini 3.5 位于 Antigravity IDE 内，本流程采用**文件队列 + IDE 执行 + 本地校验**，而不是程序直接调用 LLM API。

1. 本地脚本生成 `ide_tasks/explainer_tasks.jsonl`，每行包含 `task_id`、`latent_idx`、`sample_id_set`、完整 prompt、预期输出路径。
2. 在 Antigravity IDE 中打开任务文件，逐条或分批交给 Gemini 3.5 执行，要求只返回 JSON。
3. 将 Gemini 输出保存到 `explainer_outputs/raw/{task_id}.json`。不要手动改写字段名；解析失败的原始输出也保留。
4. 本地脚本运行 `--step validate-explainer`，执行 JSON schema 校验、MISC 标签泄漏检查、字段完整性检查和 confidence 范围检查。
5. 校验失败任务写入 `explainer_outputs/retry_tasks.jsonl`，在 Antigravity 中重跑；校验通过结果写入 `explainer_outputs/validated_explanations.jsonl`。

这种设计保留 Antigravity Gemini 3.5 的实际使用方式，同时让研究结果仍可审计到"输入 prompt / 原始输出 / 校验后结构化结果"三级证据。

### 4.5 一致性评估

同一 latent 两次采样，计算 `short_name_match` / `feature_type_match` / `confidence_gap`。若两次 short_name 完全不同且 confidence 均 < 0.60，标 `low_consistency`，不进主结论。

---

## 5. Step 3：Scorer LLM 在 Held-out 样本上验证

### 5.1 设计目的

解释生成后不能直接信。Scorer 在与 Explainer **不相交**的 held-out 样本上验证"解释是否能预测 latent 激活"。同时运行 **label-definition baseline**（不给 latent 解释，只给标签定义），判断分数里有多少来自 latent 解释的增量，而非 Gemini 对 MISC 标签或咨询语言的先验。

### 5.2 Held-out 样本构成（来自 Step 1 预留，20 条）

| 类型 | 数量 | ground truth |
|---|---|---|
| ACTIVE_HIGH held-out | 5 | 1 |
| ACTIVE_MID held-out | 5 | 1 |
| NONACTIVE_NEAR_MISS held-out | 5 | 0 |
| NONACTIVE_LABEL_MATCH held-out | 5 | 0 |

Ground truth：激活值是否高于 Step 1 固定的 `q50_nonzero_activation` 检测阈值。`NONACTIVE_LABEL_MATCH` 在发送给 Scorer 时也改写为 `NONACTIVE_NEAR_MISS`，避免提示中出现"目标标签正例但不激活"这类标签成员关系泄漏。

### 5.3 Scorer Prompt

```
基于以下解释，判断每条句子是否应激活该 latent。
解释：{explanation_json}
注意：必须基于解释中的触发/排除条件，不得用其他知识。
待判断：{formatted_holdout_samples}
输出 JSON 列表：{"sample_id","pred_activate_prob","binary_prediction","evidence_span","reason"}
```

### 5.4 Label-Definition Baseline

同 20 条 held-out，prompt 改为"请判断是否属于以下 MISC 类别：{label_definition}，仅凭定义作判断"。该 baseline 的预测仍然与 latent 激活 ground truth 比较，不与 MISC 标签本身比较；其作用是估计 Gemini 的标签先验是否已经足以解释 held-out 激活检测表现。

### 5.5 Antigravity 执行方式

Step 3 不直接调用 Gemini API。脚本生成 `ide_tasks/scorer_tasks.jsonl` 和 `ide_tasks/baseline_tasks.jsonl`，在 Antigravity IDE 中由 Gemini 3.5 执行。原始输出分别保存到：

- `scorer_outputs/raw_explanation_scorer/{task_id}.json`
- `scorer_outputs/raw_label_baseline/{task_id}.json`

本地 `--step validate-scorer` 负责解析、schema 校验、概率范围校验、AUROC/Accuracy/Precision/Specificity 计算，以及 `latent_gap` 汇总。

### 5.6 评估指标与阈值

指标：AUROC、Accuracy、Precision、Specificity、`latent_gap = AUROC_explanation − AUROC_baseline`。

| AUROC | latent_gap | 状态 |
|---|---|---|
| ≥ 0.70 | > 0 | ✅ accepted |
| 0.60–0.70 | 任意 | ⚠️ ambiguous |
| < 0.60 | 任意 | ❌ rejected |
| 任意 | ≤ 0 | ❌ no_latent_contribution（解释无增量） |

> 注意：**永远不要把 activation_prediction 和 code_discrimination 合并报告**（旧流程教训）。这里只做 detection scoring，不做旧的 code_discrimination（那本质是标签分类，被 label baseline 取代）。confidence 必须是连续概率，不接受固定值。

---

## 6. Step 4：Minimal Pair Activation Test

### 6.1 范围

仅对**主层级**每个标签前 3 个代表 latent（按 `inclusion_frequency` 排名）做，共 6×3=18 个 latent。副层级不做。

### 6.2 流程

```
Designer LLM（Antigravity Gemini 3.5）根据 Step 2 解释生成 3-5 对 minimal pair
    ↓
每对：positive（应激活）vs negative（只改一个功能因素，应不激活）
    ↓
新句子跑 Llama-3.1-8B + hook blocks.19.hook_resid_post → SAE → 取该 latent 激活
    ↓
minimal_pair_gap = mean(act_positive − act_negative)
```

### 6.3 示例

```
开放式探索问题 latent：
  Positive: What do you think would make this easier for you?
  Negative: Do you think this would make it easier for you?   （改为封闭式）

情绪反映 latent：
  Positive: You feel disappointed because you tried hard and nothing changed.
  Negative: You tried hard and nothing changed.               （去掉情绪整合）
```

### 6.4 判读

- `gap > 0` 且多数 pair 成立 → 保留解释，标 `minimal_pair_confirmed`
- positive/negative 都高 → 可能只是表面检测器（如"question detector"），标 `surface_confound`
- 都低 → 解释可能错，标 `minimal_pair_failed`

报告：平均 gap + 通过比例（gap>0 的 pair 数 / 总 pair 数）。不做复杂统计检验。

### 6.5 实现

设计任务由脚本写入 `ide_tasks/minimal_pair_designer_tasks.jsonl`，在 Antigravity 中由 Gemini 3.5 生成 minimal pairs，原始输出保存到 `minimal_pairs/raw_designer_outputs/`。本地脚本先用 `--step validate-minimal-pairs` 做 schema 校验和去重，再用 `--step run-minimal-pairs` 复用 `activations.py` 的前向逻辑 + SAE loader 计算新句子的 SAE latent 激活（需 GPU）。任务生成入口是 `--step make-minimal-pair-tasks`。

---

## 7. Step 5：子概念聚合

### 7.1 流程

```
收集所有 accepted latent 的 explanation
    ↓
Antigravity Gemini 3.5 辅助语义聚类（合并同义 short_name）
    ↓
人工验证：每标签 3-6 个子概念，每子概念 1-3 个代表 latent
    ↓
输出：MISC 标签 → 子概念 → 代表 latents → 说明 + 平均 Scorer 指标
```

### 7.2 输出表格式

| MISC 标签 | 子概念 | 代表 latents | 说明 | 平均 AUROC |
|---|---|---|---|---|
| QUO | 开放式探索 | L123, L304 | 邀请展开想法/计划/感受 | 0.78 |
| QUO | 封闭式确认 | L889 | yes/no 型事实确认 | 0.74 |
| RE | 情绪反映 | L456, L777 | 复述情绪状态 | 0.76 |
| RE | 原因整合反映 | L332 | 情绪与原因/经历连接 | 0.73 |

### 7.3 核心叙事（RQ3 答案）

> MISC 标签不是单一 monolithic SAE latent，而是由多个低层语义、语用、句式、话语功能子概念组合而成。不同标签的子概念数量和稳定性不同。

副层级标签（SU/GI/RES）的子概念一律标 `tentative`，RES 额外声明缺 client context。

---

## 8. Step 6：下游验证

### 8.1 Probe 对比（复用已有结果）

直接引用 `表示方式probing对比实验结果分析报告.md`：

| 表示 | macro AUC | 说明 |
|---|---|---|
| Hidden State | 0.900 | 不可解释 |
| Full SAE | 0.890 | 稀疏化无大损失 |
| Stable Core SAE (~34特征) | 0.869 | 接近 Top-n SAE n=50-100 |
| Random SAE-n | 0.55-0.75 | 确认排名有信息 |

**论点**：SAE 的价值不在预测性能（PCA 更高），而在于提供 PCA/raw hidden 无法给出的**可解释、可命名的子概念证据**。

### 8.2 Set-level Ablation（新实现）

```
对每个标签的 stable_core latent 集合做 zero-ablation：
  ablate 目标标签 core latents → 测 target probe AUC 下降
  ablate random same-size latents → baseline
指标：target_drop / non_target_preservation / vs_random_baseline
```

预期：ablate 目标 core 后目标 probe 选择性下降，非目标 probe 基本保留 → 支持这些 latent 对 probe-space 标签预测具有选择性依赖关系。该结果**不能**写成目标概念的因果机制证明，只能作为表示层依赖/预测贡献证据。入口 `--step ablation`。

---

## 9. 结论边界与表达规范

**允许声明**：
- ✅ "该 latent 表现出候选输入选择性模式（candidate input-selective pattern）"
- ✅ "在 held-out 样本上有预测充分性（predictive sufficiency）"
- ✅ "accepted 解释在 minimal pair 上的功能敏感性得到确认"
- ✅ "稀疏特征集合提供了可检查的内部证据（auditable evidence）"

**禁止声明**：
- ❌ "该 latent 就是某个 MI 概念"
- ❌ "因果机制证明"（需 P5 级 intervention/patching，本流程 Step 6 只到 probe-space 选择性依赖）
- ❌ 把 LLM 解释当最终结论

**副层级特别声明**：SU/GI/RES 所有子概念标 `tentative`；RES 声明"仅 counselor 当前 utterance，缺 client context"。

---

## 10. 产出物清单

| 步骤 | 路径 | 内容 |
|---|---|---|
| Step 1 | `evidence_packs/*.jsonl` | 303 个对比式样本包 + held-out 预留 |
| Step 2 | `explainer_outputs/*.jsonl` | 每 latent 2 次采样的解释 JSON + 一致性 |
| Step 2 | `ide_tasks/explainer_tasks.jsonl` | 给 Antigravity Gemini 3.5 的解释任务队列 |
| Step 2 | `explainer_outputs/raw/*.json` | Gemini 原始解释输出 |
| Step 2 | `explainer_outputs/validated_explanations.jsonl` | schema 校验后的解释 JSON |
| Step 3 | `ide_tasks/scorer_tasks.jsonl` | 给 Gemini 的 held-out activation detection 任务 |
| Step 3 | `ide_tasks/baseline_tasks.jsonl` | 给 Gemini 的 label-definition baseline 任务 |
| Step 3 | `scorer_outputs/scorer_metrics.csv` | AUROC/Acc/Prec/Spec + latent_gap + 状态 |
| Step 3 | `scorer_outputs/baseline_metrics.csv` | label-definition baseline |
| Step 4 | `ide_tasks/minimal_pair_designer_tasks.jsonl` | 给 Gemini 的 minimal pair 生成任务 |
| Step 4 | `minimal_pairs/*.jsonl` | pair 输入 + activation 结果 + gap |
| Step 5 | `subconcepts/subconcept_table.csv` | 子概念聚合表 |
| Step 6 | `ablation/set_ablation.csv` | set-level ablation 结果 |
| 全局 | `manifest.json` | 版本/时间戳/各步路径/参数 |

---

## 11. 代码实现路线

| 文件 | 说明 | 状态 |
|---|---|---|
| `src/nlp_re_base/contrastive_evidence_pack.py` | Step 1 样本包 + held-out 预留 | 已实现并 smoke 通过 |
| `src/nlp_re_base/contrastive_explainer.py` | Step 2 label-blind 投影 + 泄漏守卫 + prompt 任务组装 | 已实现并 smoke 通过 |
| `src/nlp_re_base/contrastive_gemini_io.py` | Antigravity Gemini 3.5 原始输出导入、schema 校验、retry task 生成 | 已实现并 smoke 通过 |
| `src/nlp_re_base/contrastive_scorer.py` | Step 3 scorer/baseline 指标计算 | 已实现并 smoke 通过；等待 Gemini raw outputs 后运行 |
| `src/nlp_re_base/contrastive_minimal_pairs.py` | Step 4 minimal pair 校验 + SAE activation 测试 | 已实现；`run-minimal-pairs` 需 GPU + 本地模型/SAE |
| `src/nlp_re_base/contrastive_subconcepts.py` | Step 5 子概念聚类任务 + 本地 fallback 表 | 已实现；等待 Gemini cluster outputs 后运行 |
| `src/nlp_re_base/contrastive_ablation.py` | Step 6 probe-space stable-core set ablation | 已实现并已在真实数据上运行 |
| `run_misc_contrastive_latent_interp.py` | 主 runner，分 step 生成任务、校验输出、汇总报告 | 已实现 |
| `test_contrastive_evidence_pack_smoke.py` | Step 1 smoke（验证 tag 齐全、held-out 不重叠、确定性） | 已实现并通过 |
| `test_contrastive_gemini_io_smoke.py` | Step 2/3 JSON schema、label-blind 泄漏守卫、retry 生成 | 已实现并通过 |
| `prompts/p3_contrastive/` | explainer / scorer / baseline / minimal_pair 提示词模板 | 未单独建目录；模板内置在对应模块中，任务 JSONL 保留完整 prompt |

### 11.1 当前真实运行状态

已在真实 stable_core 输入上完成本地可执行部分：

- Step 1 evidence packs：`outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/evidence_packs/contrastive_evidence_packs.jsonl`，303 行。
- Step 2 explainer task queue：`outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/ide_tasks/explainer_tasks.jsonl`，606 行（每 latent 2 次）。
- Step 6 probe-space ablation：`outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/ablation/set_ablation.csv`。
- 总报告：`outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/contrastive_latent_interp_report.md`。
- Antigravity Gemini 3.5 操作指南：`outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/gemini_antigravity_execution_guide.md`。

当前还没有 Gemini raw output，因此 Step 2 validation、Step 3 scorer、Step 4 minimal pair、Step 5 子概念聚合仍处于"任务已生成/入口已实现，等待 Antigravity Gemini 输出"阶段。

### 各 step 计算需求

| Step | 需 GPU | 需 Antigravity Gemini 3.5 | 本地脚本职责 |
|---|---|---|---|
| 1 build-packs | 否 | 否 | 生成 evidence packs + held-out |
| 2 explainer | 否 | 是（IDE 中执行任务队列） | 生成 prompt、导入/校验 JSON |
| 3 scorer | 否 | 是（IDE 中执行任务队列） | 生成 scorer/baseline prompt、计算指标 |
| 4 minimal-pairs | **是** | 是（Designer） | 生成 pair 任务、校验 pair、跑 SAE activation |
| 5 subconcepts | 否 | 是（聚类辅助，可人工修订） | 生成聚类任务、汇总表 |
| 6 ablation | 是（probe/feature ablation） | 否 | 计算 probe-space 依赖指标 |

> LLM 固定为 Antigravity IDE 中的 Gemini 3.5。若未来可获得 API，再新增 client；当前主设计必须以文件任务队列为准。

---

## 12. 详细实施流程

### 12.1 前置检查

1. 确认运行环境为 `qwen-env-py311`。
2. 确认输入文件存在：
   - `outputs/cross_val/stable_topk_selection/stable_topk_latent_set.csv`
   - `outputs/misc_full_sae_eval/feature_store/utterance_features.pt`
   - `outputs/misc_full_sae_eval/label_matrix.csv`
   - `outputs/misc_full_sae_eval/records.jsonl`
3. 确认目标 latent 数为 `stable_set_role == "stable_core"` 的 303 条；若不是 303，先停止并记录数据版本差异。
4. 建立输出目录 `outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/`。

### 12.2 Step 1：本地生成 evidence packs

运行计划命令：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step build-packs
```

本地脚本应完成：

- 读取 303 个 stable_core latent。
- 为每个 latent 构造 `samples_internal`、`samples_for_explainer`、`heldout_internal_by_tag`。
- 校验 `samples_for_explainer` 不包含 `target_label`、`target_match`、`active_labels`、MISC 标签名。
- 校验 evidence pack 与 held-out 行索引不重叠。
- 写出 `evidence_packs/contrastive_evidence_packs.jsonl` 和 `manifest.json`。

验收标准：

- pack 数为 303。
- 每个 pack 至少包含 `ACTIVE_HIGH`、`ACTIVE_MID`、`ACTIVE_LOW`、`NONACTIVE_NEAR_MISS`、`NONACTIVE_RANDOM`。
- 每个 held-out 集包含 20 条，且正负各 10 条。

### 12.3 Step 2：生成 Gemini explainer 任务并在 Antigravity 执行

本地生成任务：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step make-explainer-tasks
```

输出：

- `ide_tasks/explainer_tasks.jsonl`
- 每个 latent 两条任务，使用不同 sample order 或不同 temperature 指令。

Antigravity 操作：

1. 在 IDE 中打开 `ide_tasks/explainer_tasks.jsonl`。
2. 将每条任务的 prompt 交给 Gemini 3.5。
3. 要求 Gemini 只输出 JSON，不输出解释性散文。
4. 把结果保存为 `explainer_outputs/raw/{task_id}.json`。
5. 如果 Gemini 输出 Markdown code fence，原样保存，不要人工清洗；由本地校验脚本清洗和记录。

本地校验：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step validate-explainer
```

验收标准：

- 每个 latent 至少 1 条 schema-valid explanation。
- `short_name` 不包含 MISC 标签名。
- `confidence` 为 0-1 连续数值。
- 失败任务进入 `explainer_outputs/retry_tasks.jsonl`。

### 12.4 Step 3：生成 scorer/baseline 任务并计算解释有效性

本地生成任务：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step make-scorer-tasks
```

输出：

- `ide_tasks/scorer_tasks.jsonl`
- `ide_tasks/baseline_tasks.jsonl`

Antigravity 操作：

1. 用 Gemini 3.5 执行 scorer 任务，保存到 `scorer_outputs/raw_explanation_scorer/{task_id}.json`。
2. 用 Gemini 3.5 执行 baseline 任务，保存到 `scorer_outputs/raw_label_baseline/{task_id}.json`。
3. 不允许人工改写概率或 binary prediction。

本地校验和指标计算：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step validate-scorer
```

验收标准：

- `scorer_metrics.csv` 包含每个 latent 的 AUROC、Accuracy、Precision、Specificity、latent_gap、status。
- `accepted` 条件：AUROC >= 0.70 且 latent_gap > 0。
- `no_latent_contribution` 条件：latent_gap <= 0，即 Gemini 的标签先验不弱于 latent 解释。

### 12.5 Step 4：主层级代表 latent 的 minimal pair activation test

本地生成 Designer 任务：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step make-minimal-pair-tasks
```

Antigravity 操作：

1. 用 Gemini 3.5 为每个代表 latent 生成 3-5 对 minimal pairs。
2. 输出保存到 `minimal_pairs/raw_designer_outputs/{task_id}.json`。
3. pair 必须只改变一个功能因素，不能同时改写主题、长度、语气和句式。

本地 GPU activation 测试：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step run-minimal-pairs
```

验收标准：

- 只对主层级 18 个代表 latent 执行。
- 每个 latent 至少 3 对有效 minimal pairs。
- 输出 `minimal_pairs/minimal_pair_results.csv`，包含 positive_activation、negative_activation、gap、pass_rate。

### 12.6 Step 5：子概念聚合

本地生成聚类输入：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step make-subconcept-tasks
```

Antigravity 操作：

- 用 Gemini 3.5 合并同义 short_name，给出每个标签 3-6 个候选子概念。
- 结果保存到 `subconcepts/raw_cluster_outputs/`。

本地汇总：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step build-subconcept-table
```

验收标准：

- `subconcepts/subconcept_table.csv` 每行包含 label、subconcept、representative_latents、mean_auroc、mean_latent_gap、status。
- SU/GI/RES 的所有子概念标 `tentative`。
- RES/REC/RE 报告中明确当前数据缺少前文 client context。

### 12.7 Step 6：probe-space feature ablation

运行：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step ablation
```

验收标准：

- `ablation/set_ablation.csv` 包含 target_drop、non_target_preservation、random_baseline_drop。
- 报告只写"probe-space 选择性依赖"，不写"因果机制证明"。

### 12.8 最终汇总报告

运行：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step report
```

最终报告必须回答：

- 哪些 latent 的解释通过 held-out activation detection。
- 哪些解释只是标签先验或表面模板。
- 每个 MISC 标签下形成了哪些候选子概念。
- 哪些子概念有 minimal pair 支持。
- stable_core set 是否对 probe-space 标签预测有选择性贡献。
- 所有结论是否遵守"候选解释，不等同概念，不证明因果机制"边界。


