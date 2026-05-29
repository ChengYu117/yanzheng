# MISC 可解释性工作简明汇报

> 面向导师阅读的阶段性汇报。本文只整理已有全量实验结果，不重新跑模型。

## 1. 研究问题

本阶段想回答的问题是：

**MISC 人工行为标签和大语言模型内部表征是否是一一对应的？**

我们当前的判断是：不是一一对应。更准确地说，MISC 标签和 SAE latent 之间存在一种**结构化的多对多关系**。

可以用三句话概括：

1. **人工标注标签并不等价于模型内部的单个特征。**
2. **同一行为标签往往由多个 latent 共同表达。**
3. **同一 latent 也可能参与多个行为标签，说明模型内部表征比人工标签更连续、更混合。**

这里的 SAE 指稀疏自编码器。它的作用不是训练一个分类器，而是把 Llama 中间层的 hidden states 分解成一组稀疏 latent features，用来观察模型内部可能存在的行为表征。

## 2. 实验流程

本次使用全量 MISC 数据，共 `6194` 条咨询行为单元。每条样本是一句 `unit_text`，例如咨询师的一句提问、反映、建议或支持。

整体流程如下：

```text
6194 条 MISC utterance
  -> 输入 Llama
  -> 抽取 hidden states
  -> 输入 SAE 得到 latent features
  -> 构建 latent x MISC label 矩阵
  -> 分析标签和内部表征之间的结构关系
```

当前正式解释对象是：**每个核心 MISC 标签下排名前 20 的候选 latent**。全量矩阵用于排序和筛选，Top20 候选空间用于生成可解释性结论和案例卡片。

## 3. 数据与标签

本次数据来自 `data/mi_quality_counseling_misc`，共 `6194` 条样本，其中高质量会话样本 `4169` 条，低质量会话样本 `2025` 条。

标签不是完全互斥的。例如 `RE` 是反映性倾听父类，包含 `RES` 和 `REC`；`QU` 是提问父类，包含 `QUO` 和 `QUC`。

| 标签 | 含义 | 样本数 | 占比 |
|---|---|---:|---:|
| RE | Reflective Listening，反映性倾听 | 1358 | 21.9% |
| RES | Simple Reflection，简单反映 | 516 | 8.3% |
| REC | Complex Reflection，复杂反映 | 842 | 13.6% |
| QU | Question，提问 | 1974 | 31.9% |
| QUO | Open Question，开放式问题 | 1206 | 19.5% |
| QUC | Closed Question，封闭式问题 | 768 | 12.4% |
| GI | Giving Information，提供信息 | 681 | 11.0% |
| SU | Support，支持 | 222 | 3.6% |
| AF | Affirm，肯定 | 349 | 5.6% |
| OTHER | 其他行为 | 1610 | 26.0% |

## 4. 核心结果：标签和 latent 是多对多关系

在每个核心标签的 Top20 候选 latent 中，共有：

| 指标 | 数值 |
|---|---:|
| 核心标签数 | 9 |
| 每个标签候选 latent 数 | 20 |
| latent-label 候选边总数 | 180 |
| 去重后的 latent 数 | 136 |
| 只进入一个标签 Top20 的 latent | 103 |
| 进入多个标签 Top20 的 latent | 33 |
| 多标签共享 latent 占比 | 24.3% |

这说明：即使只看最靠前的 Top20 候选 latent，也已经能看到明显的共享结构。也就是说，模型内部不是给每个 MISC 标签分配一个单独开关，而是通过多个 latent 的组合来表达行为。

Top20 结果如下：

| 标签 | 共享比例 | Top latent | Top AUC | Cohen's d |
|---|---:|---:|---:|---:|
| QU | 95.0% | 13430 | 0.925 | 2.584 |
| QUO | 70.0% | 9959 | 0.800 | 1.762 |
| RE | 70.0% | 29759 | 0.694 | 0.847 |
| REC | 65.0% | 31133 | 0.611 | 0.965 |
| QUC | 35.0% | 21935 | 0.740 | 1.461 |
| GI | 25.0% | 13430 | 0.682 | -0.698 |
| RES | 20.0% | 20808 | 0.626 | 0.701 |
| SU | 5.0% | 24760 | 0.571 | 1.063 |
| AF | 0.0% | 23464 | 0.800 | 2.237 |

说明：

- 共享比例越高，说明该标签的 Top20 latent 越多地也被其他标签使用。
- AUC 越高，说明该 latent 对该标签和非该标签样本的区分越明显。
- Cohen's d 表示效应大小；负数表示该 latent 更像该标签的边界或排除信号。

## 5. 行为差异：不同 MISC 标签的内部结构不同

结果不仅说明“不是一一对应”，还说明这种错配是有结构的。

**父子标签重叠不能作为独立结论。**
`QU` 与 `QUO/QUC`、`RE` 与 `RES/REC` 的重叠部分来自 MISC 标签体系的父子包含关系。它可以作为标签层级一致性的检查，但不能当作“模型独立发现了共享表征”的核心证据。

**同父子类区分更关键。**
`QUO` 与 `QUC` 的 Top20 只共享 `2` 个 latent，Jaccard 为 `0.053`；`RES` 与 `REC` 的 Top20 共享数为 `0`，Jaccard 为 `0.000`。这说明在去掉父标签重复计数后，同父子类之间并不是简单复用同一批 latent。

**AF/SU 更独立。**
`AF` 的 Top20 shared ratio 为 `0.0%`，`SU` 为 `5.0%`。这说明肯定和支持类行为在 Top20 候选空间中更像相对独立的表征。但由于 `AF` 和 `SU` 样本量较小，后续仍需要人工语义复核。

**RES 相对更难解释。**
`RES` 的 shared ratio 为 `20.0%`，Top AUC 为 `0.626`。它既不是非常集中的标签，也不像 `RE/REC` 那样有强共享结构，因此后续需要结合 top examples 做更细的人工分析。

### 跨标签/跨家族共享分析

除了父子标签的包含关系，我们也检查了同父子类之间以及不同 MISC 行为家族之间是否共享 latent。结果显示：父子重叠不能作为独立发现；同父子类共享较弱；跨家族共享不是普遍发生，而是集中在少数标签对上。

| 标签对 | 关系类型 | Top20 共享 latent 数 | Top20 Jaccard | 解释 |
|---|---|---:|---:|---|
| QUO-QUC | 同父子类 | 2 | 0.053 | 开放式/封闭式问题共享很少 |
| RES-REC | 同父子类 | 0 | 0.000 | 简单/复杂反映在 Top20 中分离 |
| QUO-GI | 跨家族 leaf 标签 | 5 | 0.143 | 开放式问题和提供信息有少量共享/边界 latent |
| RES-QUO | 跨家族 leaf 标签 | 4 | 0.111 | 简单反映和开放式问题存在弱共享 |
| RES-GI | 跨家族 leaf 标签 | 4 | 0.111 | 简单反映和提供信息存在弱共享 |
| RES-QUC | 跨家族 leaf 标签 | 2 | 0.053 | 简单反映和封闭式问题存在少量共享 |
| QUC-GI | 跨家族 leaf 标签 | 2 | 0.053 | 封闭式问题和提供信息存在少量共享 |
| REC-QUO | 跨家族 leaf 标签 | 0 | 0.000 | 复杂反映和开放式问题在 Top20 中分离 |
| REC-QUC | 跨家族 leaf 标签 | 0 | 0.000 | 复杂反映和封闭式问题在 Top20 中分离 |

因此，`RE` 和 `QU` 这种不同大类并不是当前最强的共享关系。当前最清晰的结构是：

- 父子标签重叠较高，但这是标签体系自带的包含关系，不能作为独立发现。
- 同父子类共享较弱：`QUO-QUC` 只有 `2` 个共享 latent，`RES-REC` 为 `0`。
- 跨家族共享较弱且集中：主要出现在 `QUO/GI`、`RES/GI`、`RES/QUO` 相关边上。
- 用 leaf-label family union 重新计算后，`QUO/QUC` 家族与 `GI` 的 Jaccard 为 `0.094`，`RES/REC` 家族与 `GI` 为 `0.071`，`RES/REC` 家族与 `QUO/QUC` 家族为 `0.054`。

从 latent 角色看，Top20 候选空间中共有 `136` 个去重 latent，其中 `103` 个是单标签 latent，`27` 个是同家族共享 latent，`4` 个是跨家族共享 latent，`2` 个是全局共享 latent。这说明跨标签共享确实存在，但它不是均匀扩散的，而是有明确结构。

我们还生成了 `180` 张 latent case card，用每个候选 latent 的高激活样本辅助解释。其中：

| case card 类型 | 数量 |
|---|---:|
| high-purity candidate | 64 |
| mixed but label-relevant | 63 |
| low-purity review required | 53 |

这说明当前候选 latent 中确实有一批语义较清晰的对象，但也有相当一部分需要人工复核，不能直接把单个 latent 命名为某个 MISC 标签。

## 6. 结论与限制

本阶段可以支持的结论是：

- MISC 标签和 SAE latent 不是一一对应关系，而是结构化多对多关系。
- 不同咨询行为的表征结构不同：提问类更集中，反映类更分布式，肯定和支持更独立。
- `QU -> QUO/QUC` 与 `RE -> RES/REC` 的父子重叠只能作为标签层级一致性检查；独立结论应更多依赖同父子类分离和跨家族有限共享。
- 人工标签压缩了模型内部更连续、更混合的行为表征，因此仅用离散标签评价模型行为可能会遗漏内部差异。

### 6.1 层级感知结构关系 v2

已新增层级感知结构关系分析，输出目录：

- `outputs/misc_full_sae_eval/interpretability/structural_relations_v2`

这版分析固定每个标签 Top20，但在 Top20 内增加 support edge 门控：FDR 显著、`abs_cohens_d >= 0.5`、`directional_auc >= 0.60`，正向 latent 还需要 `Precision@50 lift >= 0.10`。父标签 `RE/QU` 只做一致性检查，不进入主 overlap 结论。

**Fragmentation v2：**

| 标签 | 类型 | support latent 数 | effective n | 解释 |
|---|---|---:|---:|---|
| QUO | distributed | 20 | 19.696 | 开放式问题由大量支持 latent 分散表示 |
| REC | distributed | 15 | 14.982 | 复杂反映是分布式表征 |
| AF | distributed | 12 | 11.679 | 肯定类行为有多个较清晰候选 |
| QUC | distributed | 12 | 11.625 | 封闭式问题也较分散 |
| SU | compact | 2 | 1.996 | 当前门控下只有少数强支持 latent |
| GI | compact | 2 | 1.986 | 信息类行为的强支持 latent 很少 |
| RES | compact | 2 | 1.975 | 简单反映在当前 Top20 中多数为弱候选或边界候选 |

**Overlap v2：**

- `QUO-QUC` support Jaccard 为 `0.032`，raw Jaccard 为 `0.053`，同父子类共享很低。
- `RES-REC` support/raw Jaccard 均为 `0.000`，同父子类在 Top20 中分离。
- leaf-level 最强 support overlap 是 `RES-GI=0.333`，但只共享 `1` 个 support latent，因此需要谨慎解释。
- family-union 层面全部为 low overlap：`QU_family-GI=0.065`，`RE_family-GI=0.056`，`RE_family-QU_family=0.021`。
- `GI/SU/AF` 作为建议/信息/支持类 block 与 `RE_family` 的 support Jaccard 为 `0.031`，与 `QU_family` 为 `0.044`。

**Polysemanticity v2：**

| role | 数量 | 解释 |
|---|---:|---|
| label_specific | 59 | 只支持一个 leaf 标签 |
| cross_family | 1 | 支持两个 family、2-3 个 leaf 标签 |
| generalized | 1 | 支持 3 个及以上 family 或 4 个及以上 leaf 标签 |

修正后的论文表述应强调：父子标签 overlap 符合标注体系；真正层级去重后，同父子类共享很低，跨家族共享有限，polysemantic support latent 只集中在极少数候选上。

### 6.2 Step 5 固定结论：跨标签结构

本步骤固定回答四个问题：哪些标签紧凑，哪些标签碎片化，哪些标签对共享 latent，以及共享是否符合 MISC 层级结构。

**1. 哪些标签紧凑？**

`SU`、`GI`、`RES` 在 support edge 口径下最紧凑。三者在 Top20 候选中都只有 2 个 support latent，effective n 也接近 2。这说明在当前阈值下，它们的强关联候选较少，表征更集中。

| 标签 | support latent 数 | effective n | 结论 |
|---|---:|---:|---|
| SU | 2 | 1.996 | 紧凑 |
| GI | 2 | 1.986 | 紧凑 |
| RES | 2 | 1.975 | 紧凑 |

**2. 哪些标签碎片化？**

`QUO`、`REC`、`AF`、`QUC` 更碎片化。它们不仅 support latent 数较多，effective n 也较高，说明不是由一个 top latent 主导，而是多个 latent 共同承担表征。

| 标签 | support latent 数 | effective n | 结论 |
|---|---:|---:|---|
| QUO | 20 | 19.696 | 最分散 |
| REC | 15 | 14.982 | 分散 |
| AF | 12 | 11.679 | 分散 |
| QUC | 12 | 11.625 | 分散 |

**3. 哪些标签对存在重叠？**

父子标签存在较高 overlap，但这是 MISC 标签体系自带的包含关系。`QU-QUO` 和 `RE-REC` 只能解释为 parent-child consistency，不能作为模型独立发现。

| 比较 | support Jaccard | 解释 |
|---|---:|---|
| RE vs REC | 0.350 | 父子一致性检查 |
| QU vs QUO | 0.560 | 父子一致性检查 |
| QUO vs QUC | 0.032 | 同父问题子类共享很低 |
| RES vs REC | 0.000 | 同父反映子类分离 |

**4. 跨家族结构如何？**

排除父标签后，跨家族 overlap 整体较低。`RE_family` 与 `QU_family` 的 support Jaccard 为 0.021；`GI/SU/AF` block 与 `RE_family` 的 support Jaccard 为 0.031。`SU` 和 `AF` 与反映类几乎没有 Top20 共享，说明支持/肯定类在当前候选空间中更独立。

| 比较 | support Jaccard | 结论 |
|---|---:|---|
| RE_family vs QU_family | 0.021 | 反映类与问题类整体共享很低 |
| GI vs RE_family | 0.056 | 信息类与反映类共享较弱 |
| GI vs QU_family | 0.065 | 信息类与问题类有少量共享 |
| GI/SU/AF block vs RE_family | 0.031 | 建议/信息/支持类与反映类 overlap 很低 |
| SU vs RE_family | 0.000 | 支持类与反映类分离 |
| AF vs RE_family | 0.000 | 肯定类与反映类分离 |

固定表述：

> 不同的 MISC 行为标签表现出不同的 SAE 表征结构。`SU`、`GI` 和 `RES` 在 support-gated Top20 候选中只需要少量 latent 表示，呈现较紧凑结构；相比之下，`QUO`、`REC`、`AF` 和 `QUC` 需要更多 latent，表现为更分散的表征。标签间 overlap 也具有明显层级特征：`QU-QUO` 和 `RE-REC` 的高重叠主要反映 MISC 父子标签的标注包含关系，应作为层级一致性检查，而非独立发现。去除父标签后，同父子类之间的共享较弱，`QUO-QUC` 和 `RES-REC` 均表现出较低 overlap；跨家族比较中，反映类与问题类、信息/支持/肯定类与反映类之间的 overlap 整体较低，仅有少数 latent 承担跨标签共享作用。

需要谨慎的地方：

- 当前分析是相关性和结构分析，不是因果验证。
- Top20 是候选解释空间，不代表完整 SAE latent space 的全部结论。
- 单个 latent 不能直接等价于某个 MISC 标签。
- 下一步需要做 latent 干预实验，并结合人工或专家模型对 case cards 做语义审查。

可作为后续报告或论文中的核心表述：

> 本研究发现，MISC 行为标签与 LLM 内部 SAE 表征之间并不存在简单的一一对应关系。相反，同一标签通常对应多个 latent，同一 latent 也可能参与多个标签，且这种多对多关系呈现出稳定的行为差异和层级结构。这说明人工标注体系和模型内部行为表征之间存在结构化错配。

## 附：指标全称与说明

### 基础术语

| 缩写/术语 | 全称 | 本报告中的含义 |
|---|---|---|
| hidden states | Hidden states | LLM 每一层内部产生的向量表征 |
| latent | Latent feature | SAE 输出的一个稀疏特征，可理解为模型内部表征空间中的一个候选方向 |
| utterance | Utterance | 一条咨询行为单元；本实验中每条 `unit_text` 是一个样本 |
| latent x label matrix | Latent-label association matrix | 每个 latent 和每个 MISC 标签之间的统计关联矩阵 |


### 表格指标

| 指标 | 全称 | 计算/解释方式 | 如何解读 |
|---|---|---|---|
| Top20 | Top 20 candidate latents | 对某个标签，先计算每个 latent 与该标签的关联强度，再选出排名前 20 的 latent | 本报告的正式解释对象；不是完整 SAE 空间 |
| Top latent | Top-ranked latent | 在某个标签的 Top20 中排名第 1 的 latent 编号 | 用于定位最强候选特征，但不能直接等价于标签 |
| latent-label edge | Latent-label candidate edge | 一个 latent 和一个标签形成一条候选关联；9 个标签各取 20 个，所以共有 `9 x 20 = 180` 条边 | 边越多表示候选关系越多，不代表全部都可直接解释 |
| unique latent | Unique latent | 把 180 条候选边里的 latent 编号去重后计数 | 如果去重后少于 180，说明多个标签复用了同一批 latent |
| single-label latent | Single-label latent | 只出现在一个标签 Top20 中的 latent | 越多说明标签更独立 |
| multi-label latent | Multi-label latent | 同时出现在两个或更多标签 Top20 中的 latent | 越多说明标签之间共享内部表征 |
| shared ratio | Shared-latent ratio | 某标签 Top20 中“也出现在其他标签 Top20 中”的 latent 数 / 20 | 越高说明该标签越依赖共享表征 |
| Top20 shared latent count | Top20 intersection count | 两个标签的 Top20 latent 集合取交集，统计共同 latent 数 | 例如 `QUO-GI=5` 表示二者共享 5 个候选 latent |
| Jaccard | Jaccard similarity | 两个标签 Top20 集合的交集数 / 并集数；例如交集 5、并集 35，则 `5/35=0.143` | 越接近 1 说明两个标签的候选 latent 越重合；父子标签需单独解释 |
| AUC | Area Under the ROC Curve | 把某个 latent 的激活值当作打分，检查它能否把“该标签样本”排在“非该标签样本”前面 | `0.5` 约等于随机；越接近 `1.0` 区分能力越强 |
| Cohen's d | Cohen's d standardized effect size | 计算该标签样本和非该标签样本在某个 latent 激活值上的平均差异，并用组内波动做标准化 | 绝对值越大效应越强；负数表示更像排除/边界信号 |
| case card | Latent case card | 对某个候选 latent，列出激活最高的一批真实样本及其标签 | 用于人工判断该 latent 是否有稳定语义 |
| high-purity candidate | High-purity candidate | 某个 latent 的高激活样本大多属于目标标签 | 优先用于人工命名和论文案例 |
| mixed but label-relevant | Mixed but label-relevant | 某个 latent 的高激活样本不完全属于目标标签，但仍与该行为相关 | 适合说明共享/混合表征 |
| low-purity review required | Low-purity review required | 某个 latent 的高激活样本语义不够稳定，或目标标签占比较低 | 需要人工复核，不宜直接作为强证据 |

### 关键指标的直观计算例子

**shared ratio 怎么算？**
如果 `QU` 的 Top20 latent 中，有 19 个也进入了其他标签的 Top20，那么 `QU` 的 shared ratio 就是 `19 / 20 = 95.0%`。这说明 `QU` 的候选表征很大程度上和其他标签共享。

**Jaccard 怎么算？**
以 `QUO-GI` 为例，`QUO` 和 `GI` 各有 20 个候选 latent，其中 5 个是共同的。两个集合合起来去重后一共有 35 个 latent，所以 Jaccard 是 `5 / 35 = 0.143`。如果比较的是 `QU-QUO` 这类父子标签，Jaccard 高更可能反映标签体系的包含关系，需要单独降级解释。

**AUC 怎么算？**
以某个 `QU` latent 为例，把所有样本按这个 latent 的激活值从高到低排序。如果真正的 `QU` 样本通常排在非 `QU` 样本前面，AUC 就会高。`0.925` 表示这个 latent 对 `QU` 和非 `QU` 的区分能力很强。

**Cohen's d 怎么算？**
还是以某个标签为例，先算该标签样本在某个 latent 上的平均激活，再算非该标签样本的平均激活，两者相减后再除以整体波动程度。它衡量的是“差异有多大”，而不只是“有没有差异”。

**multi-label latent 怎么判断？**
如果 latent `13430` 同时进入 `QU`、`QUO`、`QUC`、`GI`、`RES` 等多个标签的 Top20，它就不是单标签 latent，而是 multi-label 或 global latent。这类 latent 是“标签边界和模型内部表征边界不完全一致”的直接证据。

### latent 角色分类

| 角色 | 全称 | 含义 |
|---|---|---|
| exclusive | Exclusive latent | 只服务于一个标签的候选 latent |
| family-shared | Family-shared latent | 在同一标签家族内共享，例如 `RE/REC` 或 `QU/QUO` |
| cross-family | Cross-family latent | 跨不同 MISC 行为家族共享，例如 `QU/GI/RES` |
| global | Global latent | 同时进入多个不同标签的 Top20，可能反映更通用的语言或对话特征 |

### 需要避免的误读

- AUC 或 Cohen's d 高，只说明统计关联强，不等于已经证明因果机制。
- Top latent 是“最强候选特征”，不是“该标签本身”。
- Jaccard 为 0 不代表两个行为完全无关，只代表在当前 Top20 候选空间中没有共享 latent。
- multi-label latent 不是坏结果，反而是本研究的重要证据：它说明模型内部表征比人工标签更混合。
- case card 是解释性证据入口，最终语义命名还需要人工或专家模型复核。

## 附：已有图表位置

如果需要做 PPT 或 Word 汇报，可引用以下已有图片：

- `outputs/misc_full_sae_eval/interpretability/mapping_structure/figures/fragmentation_bar.png`
- `outputs/misc_full_sae_eval/interpretability/mapping_structure/figures/latent_polysemanticity_histogram.png`
- `outputs/misc_full_sae_eval/interpretability/mapping_structure/figures/label_jaccard_heatmap.png`
- `outputs/misc_full_sae_eval/interpretability/mapping_structure/figures/hierarchy_alignment_heatmap.png`
