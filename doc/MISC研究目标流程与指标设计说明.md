# MISC 研究目标、实验流程与指标设计说明

生成日期：2026-05-03
对应项目：MISC 全量数据驱动的 SAE 表征映射与可解释性分析
当前正式口径：全量矩阵用于排序，每个核心 MISC 标签的 Top20 latent 用于正式可解释性分析。

## 1. 导师的核心目的

导师文档的核心目标不是让项目变成一个新的行为分类器，也不是单纯证明某些 latent 能区分 RE/NonRE，而是要回答一个更有研究价值的问题：

> 人工标注的咨询行为标签空间，与大语言模型内部的 SAE latent 表征空间之间，到底是什么关系？

导师强调的主张可以概括为一句话：

> The mismatch between labels and representations is structured, not random.

也就是说，行为标签和模型内部表征之间不应被假设成简单的一一对应。更重要的是证明：

1. 一个行为标签可能对应多个 latent。
2. 一个 latent 也可能同时服务多个行为标签。
3. 不同行为标签的碎片化、共享程度和边界清晰度不同。
4. 这种错位不是噪声，而是有结构、有规律、有跨行为差异。
5. 这种结构性错位会影响我们如何评价、解释和干预模型中的咨询行为能力。

因此，本项目的研究定位应是：

> 使用 SAE 分析 LLM 内部如何表示咨询行为，并证明人工 MISC 标签与模型内部表征之间存在结构化、多对多、行为依赖的映射关系。

这也是为什么当前论文或报告不应写成“我们训练了一个 RE 分类器”，而应写成“我们用 SAE 揭示了 MISC 行为标注体系和 LLM 内部表征之间的结构性错位”。

## 2. 导师要求的三个分析层次

导师文档将矩阵生成后的核心工作分成三层。

### 2.1 R1：Mapping Structure

要回答的问题：

> 标签和 latent 到底怎么对应？

分析目的：

- 判断是否存在 one label -> multiple latents。
- 判断是否存在 one latent -> multiple labels。
- 判断 MISC 标签与 SAE latent 是否是一一对应，还是多对多映射。

对应到本项目：

- 先生成 `Latent x Label matrix`。
- 再从矩阵中为每个标签筛选 Top20 latent。
- 在 Top20 子空间中计算标签碎片化、latent overlap、标签间 Jaccard、latent role taxonomy。

### 2.2 R2：Asymmetry Across Behaviors

要回答的问题：

> 不同行为标签的不对应方式是否不同？

分析目的：

- 比较 RE、QU、GI、SU、AF 等标签的表征形态。
- 判断哪些标签更 compact，哪些更 fragmented，哪些更 shared。
- 找出行为之间的结构差异，而不是只给出全局统计。

对应到本项目：

- 比较每个标签 Top20 中 shared/exclusive latent 数量。
- 比较每个标签 top latent 的 effect size、AUC 和 case purity。
- 比较 `RE/REC/RES` 与 `QU/QUO/QUC` 的父子标签结构恢复情况。

### 2.3 R3：Implications

要回答的问题：

> 这种结构性错位为什么重要？

分析目的：

- 说明人工标签压缩了模型内部更细粒度、更连续、更重叠的行为表征。
- 说明只用离散标签评价模型，可能遮蔽模型内部的共享机制和混合行为。
- 说明后续做因果干预时，不能把“相关 latent”直接等价成“行为机制 latent”。

对应到本项目：

- 通过 case cards 展示 latent 的语言模式。
- 通过 G1/G5/G10/G20 因果候选组，把相关性分析推进到后续干预验证。
- 在报告中明确区分“候选解释证据”和“因果机制证明”。

## 3. 我们的总体设计

我们把导师目标拆成一个从数据到结论的闭环：

```text
MISC unit_text
  -> 数据标准化与多标签矩阵
  -> Llama hidden states
  -> SAE latent features
  -> Latent x Label association matrix
  -> 每标签 Top20 latent 候选
  -> Mapping Structure 分析
  -> 行为不对称性分析
  -> latent case cards
  -> 因果候选组 G1/G5/G10/G20
  -> 后续因果验证与论文结论
```

其中最重要的设计选择是：

> 全量 SAE latent space 只用于排序和候选发现；正式可解释性分析限定在每个标签的 Top20 latent。

这样设计的原因是：

- 避免把全量显著 latent 数当成夸大的解释性证据。
- 保持候选规模可人工审查。
- 和后续 G1/G5/G10/G20 因果验证链路自然衔接。
- 使论文结论更保守、更容易被审稿人接受。

## 4. 过程一：数据与标签标准化

### 4.1 这个过程是为了什么

目的是把 MISC 标注结果转换成模型可以处理、指标可以对齐的样本表。

导师关心的是行为标签和内部表征的关系，因此样本单位必须和行为标签单位一致。本项目选择：

```text
一条 misc_annotations JSONL 行 = 一个 unit_text = 一个 SAE 推理样本
```

### 4.2 我们如何设计

正式数据目录：

```text
data/mi_quality_counseling_misc
```

主要读取：

```text
data/mi_quality_counseling_misc/misc_annotations/high/*.jsonl
data/mi_quality_counseling_misc/misc_annotations/low/*.jsonl
data/mi_quality_counseling_misc/metadata/labels.csv
```

每条记录标准化为：

| 字段 | 作用 |
|---|---|
| `sample_id` | 稳定样本 ID，保证 records、label matrix、features 行号对齐 |
| `file_id` | 原始会话 ID |
| `quality_label` | high/low 会话质量标签 |
| `text` / `unit_text` | 输入 Llama 的文本 |
| `predicted_code` | 原始 MISC 粗标签 |
| `predicted_subcode` | 原始 MISC 子标签 |
| `labels` | 标准化后的多标签集合 |
| `label_re` | RE vs NonRE 二分类标签 |
| `confidence` | 标注置信度 |
| `rationale` | 标注解释，仅用于审查，不输入模型 |

核心标签：

```text
RE, RES, REC, QU, QUO, QUC, GI, SU, AF, OTHER
```

`label_re=True` 的规则：

```text
predicted_code == "RE" or predicted_subcode in {"RES", "REC"}
```

### 4.3 对应指标

| 指标 | 当前结果 | 目的 |
|---|---:|---|
| 样本数 | 6194 | 确认全量 MISC 行为单元进入实验 |
| RE 样本数 | 1358 | 二分类 RE/NonRE 任务基础 |
| NonRE 样本数 | 4836 | 二分类对照基础 |
| high quality 样本数 | 4169 | 后续质量差异分析 |
| low quality 样本数 | 2025 | 后续质量差异分析 |
| 标签计数 | RE 1358, RES 516, REC 842, QU 1974, QUO 1206, QUC 768, GI 681, SU 222, AF 349, OTHER 1610 | 判断标签分布和稀疏标签风险 |

对应输出：

```text
outputs/misc_full_sae_eval/dataset_summary.json
outputs/misc_full_sae_eval/records.jsonl
outputs/misc_full_sae_eval/label_matrix.csv
```

## 5. 过程二：SAE 推理、结构指标与特征保存

### 5.1 这个过程是为了什么

目的是确认当前 SAE 是否能在 MISC 数据上提供可用的内部表征，并为后续矩阵分析保存每条样本的 latent features。

导师目标要求我们连接两个空间：

- 外部 annotation space：MISC 行为标签。
- 内部 representation space：SAE latent activations。

这一阶段就是生成内部 representation space。

### 5.2 我们如何设计

模型与 SAE：

- Base model：Llama-3.1-8B。
- SAE：OpenMOSS / Llama Scope 风格 SAE。
- 输入单位：每条 `unit_text`。
- hidden activation：指定 transformer 层的 residual activation。
- SAE feature：对 token-level latent activation 做 utterance-level 聚合，当前默认使用 max pooling。

默认保存：

```text
feature_store/utterance_features.pt
feature_store/utterance_activations.pt
records.jsonl
label_matrix.csv
feature_metadata.json
```

不默认保存全量 token-level latent，以避免文件膨胀。

### 5.3 对应指标

结构指标用于回答：

> SAE 在该数据上是否具有基本重构能力？后续 latent 分析是否有基础？

当前关键结果：

| 指标 | 当前结果 | 解释 |
|---|---:|---|
| `n_tokens` | 93636 | 实际参与结构评估的 token 数 |
| `mse` | 7.4456 | 重构误差 |
| `cosine_similarity` | 0.8564 | 原 activation 与重构 activation 的方向一致性 |
| `ev_openmoss_legacy` | 0.5080 | 当前论文口径优先参考的 official legacy EV |
| `l0_mean` | 42.94 | 平均每个 token 激活的 latent 数 |
| `dead_ratio` | 0.3231 | 在该数据上几乎不激活的 latent 比例 |
| `ce_loss_delta` | 1.3528 | SAE 重构后语言模型 loss 增量 |
| `kl_divergence` | 2.4593 | 原模型和 SAE 重构模型输出分布差异 |

这些指标的结论边界：

- `cosine_similarity` 和 `ev_openmoss_legacy` 说明 SAE 有可用的重构基础。
- `ce_loss_delta` 和 `kl_divergence` 提醒我们不能把 SAE 重构当作完全无损替代。
- 因此后续解释应定位为候选表征分析，而不是高保真机制完全还原。

## 6. 过程三：RE/NonRE 功能基线

### 6.1 这个过程是为了什么

导师目标虽然不是做分类器，但 RE 是项目最早的主线。我们保留 RE/NonRE 功能基线，是为了回答：

> SAE latent 空间中是否存在与 RE 行为相关的可检测信号？

这一步为后续多标签分析和因果验证提供一个稳定锚点。

### 6.2 我们如何设计

使用 `label_re` 把全量 MISC 样本分成：

```text
RE vs NonRE
```

对每个 latent 计算 RE-positive 与 RE-negative 的单变量关联，然后训练 probe：

- `sparse_probe_k1`
- `sparse_probe_k5`
- `sparse_probe_k20`
- `dense_probe`

### 6.3 对应指标

| 指标 | 当前结果 | 目的 |
|---|---:|---|
| `total_latents` | 32768 | SAE latent 总维度 |
| `significant_fdr` | 3461 | RE/NonRE 相关候选是否广泛存在 |
| Top latent | 29759 | RE 最强单变量候选 |
| Top latent Cohen's d | 0.847 | RE 与 NonRE 激活差异大小 |
| Top latent AUC | 0.694 | 单 latent 区分能力 |
| `sparse_probe_k20 AUC` | 0.8245 | 20 个候选 latent 的 RE 判别能力 |
| `dense_probe AUC` | 0.8489 | 全空间线性探针上限参考 |

这一步支持的结论：

> RE 信号可以在 SAE latent 空间中被捕获，但 RE 不是单一 latent 即可完全解释的行为。

这一步不支持的结论：

> 某个 latent 就是 RE 概念本身。

## 7. 过程四：Latent x Label 矩阵生成

### 7.1 这个过程是为了什么

这是导师文档中的核心中间产物。它的作用是把每个 MISC 标签和每个 SAE latent 的关系系统化，而不是只看 RE。

要回答的问题：

> 每个 latent 和每个行为标签之间是否存在统计关联？这种关联方向和强度如何？

### 7.2 我们如何设计

对每个标签和每个 latent 计算 association：

```text
label in {RE, RES, REC, QU, QUO, QUC, GI, SU, AF, OTHER}
latent in {0 ... 32767}
```

得到长表：

```text
latent_label_matrix.csv
```

当前正式解释策略：

```text
全量 latent-label matrix
  -> 仅作为排序来源
  -> 每个核心标签取 Top20
  -> Top20 子空间进入正式可解释性分析
```

### 7.3 对应指标

矩阵中每个 latent-label pair 的指标包括：

| 指标 | 含义 | 用途 |
|---|---|---|
| `cohens_d` | 标签正负样本激活差异 | 衡量 effect size |
| `abs_cohens_d` | 绝对效应量 | 排序候选 |
| `auc` / `directional_auc` | 单 latent 区分该标签的能力 | 判断标签选择性 |
| `p_value` | 显著性检验 p 值 | 统计筛选 |
| `p_fdr` | BH-FDR 校正后 p 值 | 控制多重比较 |
| `significant_fdr` | 是否通过 FDR | 内部排序参考 |
| `precision_at_10` / `precision_at_50` | top activation 样本中目标标签纯度 | 判断可解释性纯度 |

需要特别强调：

> FDR 和全量矩阵用于候选发现，不作为最终解释性结论。最终正式分析对象是每个标签 Top20。

## 8. 过程五：Top20 Mapping Structure 分析

### 8.1 这个过程是为了什么

这是对导师 R1 的直接回答：

> MISC 标签与 SAE latent 是否是一一对应？

我们通过 Top20 子空间证明：

- 一个标签的 Top20 里有多个 latent。
- 一个 latent 可以进入多个标签的 Top20。
- 标签之间共享结构集中在有理论意义的父子标签或行为家族中。

### 8.2 我们如何设计

正式分析对象：

```text
RE, RES, REC, QU, QUO, QUC, GI, SU, AF
```

每个标签取 20 个 latent：

```text
9 labels x 20 latents = 180 label-latent candidate edges
```

优先读取：

```text
functional/misc_label_mapping/top_latents_by_label/<LABEL>.csv
```

如果不存在，则从 `latent_label_matrix.csv` 按以下顺序排序：

```text
abs_cohens_d desc
directional_auc desc
latent_idx asc
```

### 8.3 对应指标

#### 8.3.1 Many-to-many 指标

| 指标 | 当前结果 | 解释 |
|---|---:|---|
| `topk_latent_label_edges` | 180 | 每标签 Top20 的总边数 |
| `topk_unique_latents` | 136 | 去重后的 latent 数 |
| `topk_single_label_latents` | 103 | 只进入一个标签 Top20 的 latent |
| `topk_multi_label_latents` | 33 | 进入多个标签 Top20 的 latent |
| `topk_multi_label_latent_share` | 0.243 | Top20 子空间中的共享 latent 比例 |

科研含义：

> 即使只看每个标签最强的 20 个候选，也仍然存在多标签共享结构，说明 many-to-many 不是全量显著扫描造成的假象。

#### 8.3.2 Label fragmentation / sharing 指标

| 标签 | Shared / 20 | Exclusive / 20 | Top d | Top AUC | 解释 |
|---|---:|---:|---:|---:|---|
| QU | 19 | 1 | 2.584 | 0.925 | 最强共享型标签 |
| QUO | 14 | 6 | 1.762 | 0.800 | 与 QU 高重叠 |
| RE | 14 | 6 | 0.847 | 0.694 | RE/REC 共享明显 |
| REC | 13 | 7 | 0.965 | 0.611 | 反映家族共享 |
| QUC | 7 | 13 | 1.461 | 0.740 | 问题子类中更独立 |
| GI | 5 | 15 | 0.698 | 0.682 | 混合且有负向边 |
| RES | 4 | 16 | 0.701 | 0.626 | 当前解释质量较弱 |
| SU | 1 | 19 | 1.063 | 0.571 | 较独立但区分性弱 |
| AF | 0 | 20 | 2.237 | 0.800 | 最 compact、最独立 |

这里的核心不是“哪个标签 latent 更多”，因为每个标签固定 20 个；核心是 Top20 内部的共享结构不同。

#### 8.3.3 Latent role taxonomy

| Role | 数量 | 占比 | 含义 |
|---|---:|---:|---|
| exclusive | 103 | 0.757 | 只服务一个标签 |
| family_shared | 27 | 0.199 | 在同一行为家族内共享 |
| cross_family | 4 | 0.029 | 跨行为家族共享 |
| global | 2 | 0.015 | 进入 5 个及以上标签 Top20 |

科研含义：

> Top20 子空间以专属 latent 为主，但仍保留清晰的家族共享和少量跨家族共享结构。

#### 8.3.4 Label pair similarity

| 标签对 | Top20 intersection | Jaccard | 含义 |
|---|---:|---:|---|
| QU - QUO | 14 | 0.538 | QU 家族结构恢复最明显 |
| RE - REC | 13 | 0.481 | RE 与复杂反映共享明显 |
| QU - QUC | 7 | 0.212 | QU 与封闭式问题部分共享 |
| QUO - GI | 5 | 0.143 | 问题/信息相关行为有少量跨标签重叠 |
| QU - GI | 5 | 0.143 | 问题与信息给出存在弱共享 |

#### 8.3.5 Hierarchy alignment

| 层级关系 | Jaccard | Parent decomposition | Child coverage | 判断 |
|---|---:|---:|---:|---|
| QU -> QUO/QUC | 0.487 | 0.950 | 0.500 | 强恢复 |
| QU -> QUO | 0.538 | 0.700 | 0.700 | 最强父子证据 |
| QU -> QUC | 0.212 | 0.350 | 0.350 | 部分恢复 |
| RE -> RES/REC | 0.277 | 0.650 | 0.325 | 部分恢复 |
| RE -> REC | 0.481 | 0.650 | 0.650 | RE 家族主要由 REC 支撑 |
| RE -> RES | 0.000 | 0.000 | 0.000 | 当前未恢复 |

科研含义：

> QU 家族层级恢复较好；RE 家族只部分恢复，主要来自 REC，RES 是当前薄弱环节。

## 9. 过程六：Asymmetry Across Behaviors 分析

### 9.1 这个过程是为了什么

这是对导师 R2 的直接回答：

> 不同行为标签在 SAE 空间里的表征形态是否不同？

如果只证明 many-to-many，还不够。导师希望看到的是：

- 哪些行为更碎片化。
- 哪些行为更 compact。
- 哪些行为和其他标签共享更多。
- 哪些行为的解释质量更强或更弱。

### 9.2 我们如何设计

基于 Top20 候选，按标签统计：

- shared ratio
- exclusive latent 数
- family-shared latent 数
- cross-family latent 数
- global latent 数
- top latent AUC / Cohen's d
- Top20 group score 在标签正负样本中的差异
- high/low quality 会话内的 activation shift

### 9.3 对应指标

当前行为模式分类：

| Pattern | 标签数 | 解释 |
|---|---:|---|
| `mixed_distributed` | 7 | 大多数标签存在混合、分布式候选 |
| `compact_strong` | 1 | AF 最接近 compact strong |
| `shared_distributed` | 1 | QU 最明显共享分布式 |

代表性标签：

| 标签 | Pattern | 关键证据 |
|---|---|---|
| QU | shared_distributed | shared ratio 0.95，Top AUC 0.925 |
| AF | compact_strong | shared ratio 0.00，Top AUC 0.800，Top d 2.237 |
| RE | mixed_distributed | shared ratio 0.70，与 REC Jaccard 0.481 |
| REC | mixed_distributed | 与 RE 共享明显，但自身 AUC 较弱 |
| RES | mixed_distributed / weak | case purity 很弱，需要复核 |

科研含义：

> 不同行为不是以同一种方式映射到 SAE latent。QU 更像家族共享结构，AF 更像独立紧凑结构，RE/REC 更像混合分布结构，RES 当前不稳定。

## 10. 过程七：Latent case cards 与质性解释

### 10.1 这个过程是为了什么

矩阵和统计指标只能说明结构存在，但不能说明 latent 具体表达了什么语言模式或心理功能。

case cards 用来回答：

> 这些 Top20 latent 的高激活样本，是否真的呈现出与目标 MISC 标签相关的语言行为？

这是从数字回到语言材料的一步。

### 10.2 我们如何设计

对每个标签 Top20 latent 生成 case cards：

- 抽取该 latent 的 top activating examples。
- 统计 top examples 中目标标签比例。
- 统计 dominant labels。
- 统计 high/low quality 分布。
- 记录常见 tokens、平均词数、问号率等语言特征。
- 将 latent 标成：
  - `high_purity_candidate`
  - `mixed_but_label_relevant`
  - `low_purity_review_required`

### 10.3 对应指标

| 指标 | 当前结果 | 含义 |
|---|---:|---|
| case cards 总数 | 180 | 每标签 Top20 |
| high purity candidate | 64 | 可优先人工命名 |
| mixed but label relevant | 63 | 支持共享/混合解释 |
| low purity review required | 53 | 需要人工复核 |
| mean target purity | 0.546 | 整体 top examples 目标标签纯度 |

按标签看：

| 标签 | high purity | mixed | low purity | 平均 purity |
|---|---:|---:|---:|---:|
| QU | 19 | 1 | 0 | 0.937 |
| QUO | 9 | 8 | 3 | 0.671 |
| AF | 7 | 11 | 2 | 0.612 |
| QUC | 9 | 7 | 4 | 0.600 |
| RE | 6 | 9 | 5 | 0.533 |
| REC | 4 | 10 | 6 | 0.513 |
| GI | 7 | 7 | 6 | 0.513 |
| SU | 3 | 10 | 7 | 0.433 |
| RES | 0 | 0 | 20 | 0.100 |

科研含义：

- `QU` 的解释质量很强，适合作为正例。
- `QUO/QUC/AF` 有较好的候选解释空间。
- `RE/REC` 更偏混合相关，支持“反映行为是共享式/分布式表征”的论点。
- `RES` 暂时不适合作为强解释性结论。

## 11. 过程八：因果候选组导出

### 11.1 这个过程是为了什么

导师目标中的矩阵分析本质上是相关性分析。为了推进到更强的机制解释，需要后续因果验证。

因果候选导出的作用是：

> 把 Top20 相关候选组织成可干预的 latent group。

这样后续可以检验：

- ablation 这些 latent 是否削弱目标行为倾向。
- steering 这些 latent 是否增强目标行为倾向。
- shared latent 是否产生跨标签副作用。

### 11.2 我们如何设计

每个标签输出：

```text
G1, G5, G10, G20
```

其中：

- `G1`：Top20 中排序第 1 的 latent。
- `G5`：Top20 前 5。
- `G10`：Top20 前 10。
- `G20`：该标签正式 Top20 候选全集。

### 11.3 对应指标

| 标签 | G20 候选数 | high purity in G20 | Pattern |
|---|---:|---:|---|
| QU | 20 | 19 | shared_distributed |
| QUO | 20 | 9 | mixed_distributed |
| QUC | 20 | 9 | mixed_distributed |
| AF | 20 | 7 | compact_strong |
| GI | 20 | 7 | mixed_distributed |
| RE | 20 | 6 | mixed_distributed |
| REC | 20 | 4 | mixed_distributed |
| SU | 20 | 3 | mixed_distributed |
| RES | 20 | 0 | mixed_distributed |

科研含义：

- `QU` 最适合作为第一批因果验证对象。
- `RE/REC` 是项目主线，应继续做因果验证，但预期是混合/共享机制。
- `AF` 适合作为 compact 对照。
- `RES` 不建议作为第一批强因果结论对象。

## 12. 过程九：因果验证

### 12.1 这个过程是为了什么

因果验证用于回答：

> 统计相关的 latent 是否真的会影响模型产生某类咨询行为的倾向？

它把前面的 association 推进到 intervention。

### 12.2 我们如何设计

当前因果验证第一版仍以 `RE vs NonRE` 为主任务，使用从 Top20 候选链路导出的 RE candidate latents。

核心干预方式：

- necessity / ablation：压低或移除目标 latent，观察 RE 方向是否下降。
- sufficiency / steering：增强目标 latent，观察 RE 方向是否上升。
- selectivity：检查干预是否只影响目标方向，而不是无差别改变所有输出。
- group comparison：比较 G1/G5/G10/G20、random、bottom control。

### 12.3 对应指标

因果阶段应记录：

| 指标 | 目的 |
|---|---|
| `necessity_delta` | 目标 latent 被削弱后，目标行为分数是否下降 |
| `sufficiency_delta` | 目标 latent 被增强后，目标行为分数是否上升 |
| `selectivity_score` | 干预是否相对特异，而不是全局扰动 |
| `group_effect_G1/G5/G10/G20` | 判断单 latent 和组合 latent 的干预强度 |
| `random_control_effect` | 排除随机 latent 也有效的可能 |
| `bottom_control_effect` | 排除低相关 latent 也有效的可能 |
| `run_status.json` | 记录运行进度和失败阶段 |
| `fatal_traceback.log` | 记录长任务崩溃原因 |

论文中的理想结论形式不是：

> 这些 latent 就是 RE。

而是：

> 在 RE-associated Top20 候选中，只有一部分 latent 具有可观的因果干预效果；这些 latent 可能是 RE-specific 或 behavior-shared causal candidates。

## 13. 过程十：AI / 人工评审

### 13.1 这个过程是为了什么

SAE latent 的统计相关性不等于语义解释。AI 或人工评审用于补足：

> 这些 high-activation examples 是否真的构成一致的咨询行为模式？

### 13.2 我们如何设计

评审对象不是所有 latent，而是 Top20 中更高优先级的候选：

- high purity candidate
- RE/REC shared candidate
- QU/QUO strong candidate
- AF compact candidate
- RES low-purity review candidate

### 13.3 对应指标

| 指标 | 目的 |
|---|---|
| semantic consistency | top examples 是否表达同一语言/行为模式 |
| target label match | 是否符合目标 MISC 标签 |
| mixed-label evidence | 是否同时体现多个标签 |
| linguistic pattern | 是语言形式特征还是心理咨询行为特征 |
| human review decision | accept / revise / reject latent interpretation |

这一步的作用是把 latent 从“编号”推进到“可命名解释”。

## 14. 指标与导师问题的总对应表

| 导师问题 | 我们的过程 | 指标 | 当前可以支持的结论 |
|---|---|---|---|
| label 与 latent 是否一一对应？ | Top20 Mapping Structure | `topk_unique_latents`, `topk_multi_label_latents`, `latent_role_summary` | 不是一一对应，Top20 中有 33 个多标签 latent |
| 一个 label 是否对应多个 latent？ | label fragmentation | 每标签 Top20、`topk_abs_effect_sum`, `topk_shared_count` | 每个标签都有多个候选 latent，内部结构不同 |
| 一个 latent 是否对应多个 label？ | latent overlap distribution | labels per latent, single/multi/global role | 24.3% unique latent 被多个标签共享 |
| 哪些标签更接近？ | label pair similarity | Top20 intersection, Jaccard, Cohen's d correlation | `QU-QUO`、`RE-REC` 最清晰 |
| 父子标签是否被恢复？ | hierarchy alignment | parent decomposition, child coverage, sibling separation | QU 家族恢复较好，RE 家族部分恢复 |
| 不同行为是否有不同结构？ | behavior asymmetry | shared ratio, exclusive count, pattern type | QU 共享，AF 独立，RE/REC 混合，RES 薄弱 |
| latent 是否可解释？ | case cards | target purity, dominant labels, common tokens | QU 最纯，RE/REC 混合，RES 需复核 |
| latent 是否有因果作用？ | causal validation | necessity, sufficiency, selectivity, group effect | 需要后续干预验证，当前已完成候选组导出 |
| 这些发现为什么重要？ | implications | 行为差异、共享结构、case evidence | 人工标签压缩了更细粒度、更重叠的模型内部表征 |

## 15. 当前研究结论边界

当前已经可以较稳妥地说：

1. 全量 MISC 数据已经接入 SAE 主流程。
2. SAE 在 MISC 数据上有基本可用的重构与表征能力。
3. RE 信号可以被 SAE latent 捕获。
4. MISC 标签与 SAE latent 在 Top20 候选空间中呈现多对多结构。
5. 不同行为标签的 Top20 结构存在明显差异。
6. `QU/QUO/QUC` 的层级结构恢复较好。
7. `RE/REC` 共享明显，但 `RES` 仍是薄弱环节。
8. case cards 已经能支持部分 qualitative interpretation。
9. 因果候选组已经生成，可进入后续 intervention。

当前不应过度声称：

1. 不能说单个 latent 等价于某个 MISC 标签。
2. 不能说 Top20 已经证明因果机制。
3. 不能说所有 MISC 标签都被 SAE 高质量解释。
4. 不能把全量显著 latent 数作为正式可解释性结论。
5. 不能忽略 `RES` 和低纯度候选带来的解释风险。

## 16. 推荐论文式叙事

推荐将项目主线写成：

> We analyze how counseling behavior annotations map onto SAE features in a large language model. Instead of assuming a one-to-one correspondence between MISC labels and internal representations, we first compute a latent-label association matrix over full MISC utterances, then analyze the Top20 candidate latents for each behavior label. The resulting structure shows a behavior-dependent many-to-many mapping: some behaviors such as QU exhibit highly shared family-level representations, AF appears more compact and exclusive, while RE/REC show mixed and distributed representations. These findings suggest that discrete counseling behavior labels compress richer, overlapping internal representations learned by the model.

中文表述：

> 本研究并不把 SAE latent 直接等同于 MISC 标签，而是把每个标签的 Top20 latent 作为可审查候选空间，分析人工行为标签和模型内部表征之间的结构化错位。结果显示，MISC 标签与 SAE latent 存在行为依赖的多对多映射：QU 家族呈现强共享结构，AF 更紧凑独立，RE/REC 更混合分布，说明人工标签压缩了模型内部更细粒度、更重叠的咨询行为表征。

## 17. 后续执行优先级

1. 优先对 `QU`、`RE`、`REC`、`AF` 做因果干预。
2. 对 `RE` Top20 做人工语义复核，区分 RE-specific、RE+REC shared、RE+QU mixed 和低纯度候选。
3. 单独排查 `RES`，确认是标签定义、样本量、排序标准还是模型表征导致弱结果。
4. 将 case cards 中 high-purity latent 人工命名，形成论文中的 qualitative examples。
5. 后续所有解释性报告继续坚持 Top20 口径。
