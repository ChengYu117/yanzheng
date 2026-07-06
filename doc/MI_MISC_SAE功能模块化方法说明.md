# MI/MISC SAE 分析功能模块化方法说明

## 1. 文档目的

本文档说明当前 MI/MISC SAE 分析系统中已经形成的主要功能模块。它不列具体运行入口，而是用自然语言解释每个模块为什么存在、解决什么研究问题、如何设计、产出什么证据，以及这些证据能支持什么结论、不能支持什么结论。

整个系统的核心目标不是做一个普通的 MI/MISC 分类器，而是回答一个更细的问题：

> 人类专家定义的 MI/MISC 咨询行为标签，是否能在 LLM 的内部表征和 SAE latent 空间中被读出、拆分、比较和解释？

因此，当前分析链条被设计成一个逐步降低不确定性的研究流程：

1. 先判断模型 hidden representation 中是否存在可线性读出的 MI/MISC 标签信息。
2. 再判断 SAE 表征在 MI 数据集上是否质量足够、稀疏性是否合理。
3. 接着过滤掉明显不适合作解释的 latent，形成 filtered 初筛候选池。
4. 在 filtered 池内重新计算 label-latent 关联指标。
5. 对这些关联指标做交叉验证审计：先用 AUC@K 确定每个标签的候选预算 `K*`，再用 repeated split-half inclusion frequency 和 grouped bootstrap CI 筛出 `stable_core` latent set。
6. 分析标签和 stable core latent 之间的统计结构，包括重叠、碎片化和最小预测充分子空间。
7. 以后续默认 latent set，即 `stable_core`，为主对象做 top activation 功能解释，区分 MI 概念、咨询功能、表层模板和数据 artifact。

这条证据链刻意把 decodability、predictive utility、interpretability 和 causality 分开。一个 latent 能预测某个标签，并不等于它就是该标签的临床概念；一个 top activation example 看起来像某种 MI 技术，也不等于模型已经具备因果意义上的咨询理解。

## 2. 总体研究结构

当前功能可以理解为七个主模块：

| 模块 | 核心问题 | 主要证据 |
|---|---|---|
| 线性探针与选层 | 哪一层 hidden representation 最适合读出 MI/MISC 标签？ | 各层 probe AUC、macro AUC、标签级表现 |
| SAE 表征质量评估 | SAE 在 MI 数据集上的表示是否可用？ | EV、重构质量、稀疏性、激活分布 |
| 初筛 latent 池构建 | 哪些 latent 不适合作为解释候选？ | activation rate、dead/rare/always-active、异常激活过滤 |
| label-latent 关联指标 | 哪些 latent 与哪些标签统计相关？ | AUC、Cohen's d、directional AUC、FDR、Precision@K |
| label-latent 交叉验证审计 | 上述关联候选是否可复现、是否有置信区间、每个标签应使用多大的候选预算？ | AUC@K、per-label K*、50 次 split-half inclusion frequency、bootstrap CI、stable_core / boundary_candidate |
| 统计结构与最小充分子空间 | 标签是 compact 还是 distributed？标签之间 stable core latent 如何重叠？ | stable_core overlap、shared/exclusive、minimal K、fold stability |
| top latent 功能解释 | stable core 的高激活语句体现 MI 概念还是某种模式？ | top utterances、pattern taxonomy、candidate interpretation |

这些模块不是平行关系，而是从粗到细的证据递进：

```text
hidden layer decodability
  -> SAE representation quality
  -> filtered latent candidate pool
  -> label-latent association matrix
  -> cross-validation audit of association candidates
  -> statistical structure and minimal subspaces
  -> top-latent utterance interpretation
```

## 3. 数据与分析对象

当前主分析对象是 utterance-level 的 MI/MISC counselor behavior 数据。每一行对应一条咨询师语句，并带有模型或标注流程生成的 MISC 行为标签，例如 `RE`, `RES`, `REC`, `QU`, `QUO`, `QUC`, `GI`, `SU`, `AF`。

分析中有三个对齐对象：

| 对象 | 含义 |
|---|---|
| utterance text | 咨询师当前语句，用于 top activation 解释 |
| label matrix | 每条语句对应的 MI/MISC 标签，多标签层级可同时存在，例如 `QU` 和 `QUO` |
| SAE latent activation | 每条语句经过 LLM 和 SAE 后得到的 sparse latent activation 向量 |

当前数据是 utterance-level，不包含完整的前一句 client context。因此，对于 `RE`, `RES`, `REC` 这类依赖前文的反映式倾听标签，解释必须谨慎。仅凭咨询师当前语句可以判断其表面上是否像 reflection，但不能强断言它真正复述了 client 的内容。

## 4. 模块一：简单线性探针与选层

### 4.1 设计目的

线性探针模块用于回答最基础的问题：

> MI/MISC 标签信息是否存在于 LLM hidden representation 中？如果存在，在哪一层最容易被简单读出？

这个模块不追求构建最强分类模型，而是故意使用简单 probe。原因是：如果一个简单线性模型就能从某层 hidden state 中读出标签，说明该层表征已经以相对线性的方式包含了标签相关信息。反过来，如果需要复杂模型才能读出，则很难判断是 hidden representation 本身有结构，还是分类器学出了复杂捷径。

### 4.2 方法设计

方法上，对同一批 MI/MISC utterances，在不同 transformer 层抽取 hidden representation。然后对每层分别训练简单线性 probe，预测每个 MISC 标签。

评估时使用统一标签集合和统一数据切分，避免不同层之间因为样本或标签口径不同而不可比。每层 probe 的表现用 AUC 或 macro AUC 进行比较。

这个设计有三个关键点：

1. 使用简单线性模型，强调可读性而不是最大性能。
2. 每层使用相同数据和标签口径，确保层间比较公平。
3. 结果只用于选择后续 SAE 分析的优先层，不直接作为语义解释。

### 4.3 产出证据

该模块主要产出：

- 每一层对每个标签的 AUC。
- 每一层的 macro AUC。
- raw hidden、PCA、SAE 表征之间的 baseline 对比。
- 推荐后续 SAE 重点分析的层。

### 4.4 能支持的结论

可以支持：

- 某层 hidden representation 对 MI/MISC 标签更可解码。
- MI/MISC 标签信息在 LLM 内部不是完全不可读。
- 后续 SAE 选层有经验依据。

不能支持：

- 某层就是 MI 概念所在层。
- probe 成功等于模型理解了 MI。
- probe 权重大就等于对应特征是因果机制。

## 5. 模块二：SAE 表征质量与 MI 数据集适配

### 5.1 设计目的

SAE 表征质量模块用于回答：

> 这个 SAE 在 MI/MISC 数据集上是否有足够好的重构质量和稀疏激活结构，值得进一步解释？

SAE 的 latent 是否可解释，首先依赖两个基础条件：

1. SAE 能较好重构原始 hidden activation。
2. latent activation 不是全部死亡、全部激活或被异常样本支配。

如果这两个条件不满足，后续即使计算出某些 AUC 或 Cohen's d，也可能只是噪声、重构失败或激活异常造成的假象。

### 5.2 方法设计

该模块将 MI/MISC utterances 输入目标 LLM，抽取目标层 hidden activation，再经过 SAE 编码和解码。随后计算 reconstruction 和 activation distribution 两类指标。

第一类是重构质量指标，例如 EV。它衡量 SAE reconstruction 能解释多少原始 hidden activation 的方差。EV 越高，说明 SAE 保留原始表征信息越多。

第二类是稀疏激活指标。它检查每个 latent 在数据集上的激活频率、激活强度分布、是否有极端 outlier、是否被少数样本主导。

### 5.3 重点指标

| 指标 | 作用 |
|---|---|
| EV / explained variance | 衡量 SAE 对 hidden activation 的重构能力 |
| activation rate | 衡量某个 latent 在多少比例样本上激活 |
| active count | 衡量 latent 是否有足够样本支撑 |
| activation variance | 判断 latent 是否几乎不变化 |
| top activation share | 判断是否被少数异常样本主导 |
| outlier fraction | 判断异常高激活是否过多 |

### 5.4 能支持的结论

可以支持：

- SAE 在当前 MI/MISC 数据上的重构表现是否可接受。
- latent activation 是否稀疏且可分析。
- 哪些 latent 明显不适合进入解释候选池。

不能支持：

- EV 高就说明每个 latent 都可解释。
- 稀疏性好就说明 latent 是人类可读概念。
- SAE 表征质量指标本身能证明 MI 概念存在。

## 6. 模块三：filtered 初筛 latent 池构建

### 6.1 设计目的

filtered 初筛池的目的，是在做标签关联分析之前，先去掉明显不稳定或不适合作解释的 latent。

如果直接在全部 SAE latent 上计算标签关联，会遇到几个问题：

- 大量 latent 几乎从不激活，统计指标不稳定。
- 有些 latent 几乎总是激活，缺乏区分能力。
- 有些 latent 被少数极端样本支配，top examples 容易误导解释。
- 方差接近 0 的 latent 即使出现显著性，也缺乏实际解释价值。

因此，filtered pool 是一个候选池定义，不是标签选择结果。它先用 activation quality 筛 latent，再在保留下来的 latent 上重新计算标签关联指标。

### 6.2 方法设计

初筛逻辑基于 latent 自身的激活分布，而不是基于它对某个标签的预测表现。主要检查：

1. 是否有足够多样本激活。
2. 是否存在非零方差。
3. 是否不是几乎所有样本都激活。
4. 是否没有被极少数 outlier 支配。
5. top activation 是否没有过度集中在少数样本。

通过这些规则后，latent 才进入 filtered candidate pool。

### 6.3 当前 filtered pool 口径

当前主分析的 filtered pool 以 `FeatureFilterConfig keep=True` 为准。已固化的候选池统计为：

| 项目 | 数值 |
|---|---:|
| 原始 SAE latent 数 | 32768 |
| keep=True latent 数 | 12758 |
| dropped latent 数 | 20010 |
| 标签数 | 9 |
| filtered label-latent metric rows | 114822 |

这里的 `114822` 等于 `12758 × 9`，说明 filtered association matrix 是在 keep pool 上重新计算的完整矩阵。

### 6.4 能支持的结论

可以支持：

- 后续指标是在较稳定的候选 latent 池上计算。
- dropped latent 不参与 filtered-pool 结论。
- top latent 分析不被明显 dead 或 always-active latent 主导。

不能支持：

- 被 drop 的 latent 一定完全无意义。
- keep=True latent 一定可解释。
- filtered pool 是唯一合理候选池。它是当前固定实验口径。

## 7. 模块四：label-latent 关联指标

### 7.1 设计目的

该模块回答：

> 在 filtered latent pool 中，哪些 latent 和哪些 MI/MISC 标签存在稳定统计关联？

这是从 SAE latent 到 MISC 标签的第一层映射。它不解释 latent 的语义，只衡量 activation 与标签之间的统计关系。

### 7.2 方法设计

对每个 MISC 标签做 one-vs-rest 分析。也就是说，对于标签 `L`：

- 正例是所有带有标签 `L` 的 utterance。
- 负例是所有不带标签 `L` 的 utterance。
- 每个 latent 的 activation 是用于区分正负例的单变量信号。

对每个 label-latent pair 分别计算效应量、排序能力、显著性和 top activation 命中率。

这个设计有两个好处：

1. 每个 latent 都可以被独立评价，便于生成可解释候选。
2. 标签之间可以比较同一个 latent 的方向和强度，便于分析 overlap。

### 7.3 重点指标

| 指标 | 含义 | 解释方式 |
|---|---|---|
| AUC | latent activation 区分正负样本的能力 | 0.5 约等于随机，越高越能正向区分 |
| directional AUC | 把正向和负向区分统一成强度指标 | 大于 0.5 表示有方向性区分能力 |
| Cohen's d | 正例和负例 activation 均值差的标准化效应量 | 正值表示正例更高，负值表示负例更高 |
| p-value | 统计显著性 | 单独看容易受多重检验影响 |
| FDR | 多重检验校正后的显著性 | 更适合大规模 label-latent 检验 |
| Precision@10 / @50 | activation 最高的样本中有多少是目标标签 | 适合判断 top activation examples 是否标签纯 |

### 7.4 正向 latent 与负向 boundary latent

一个重要设计点是保留方向。

如果某 latent 对 `QUO` 的 Cohen's d 为正，说明它在 `QUO` 正例中更高，可能是 `QUO` 正向候选。如果它对 `RES` 的 Cohen's d 为负，则说明它更像 `RES` 的反向边界信号，而不是 `RES` 概念 latent。

这对解释非常关键。很多跨标签 overlap latent 并不是多个标签共同正向共享，而是对一个标签正向、对另一个标签负向。它们更适合解释标签边界，而不是解释共同概念。

### 7.5 能支持的结论

可以支持：

- 某 latent 与某标签存在统计关联。
- 某 latent 是正向候选还是负向 boundary signal。
- 某标签的 top candidate latent 集合。
- 哪些 latent 值得进入 top utterance 人工解释。

不能支持：

- 某 latent 就是该标签的语义概念。
- 单变量 AUC 高就说明 latent 有因果作用。
- FDR 显著就说明该 latent 的 top examples 一定语义纯。

### 7.6 关联指标的交叉验证审计

在 filtered label-latent association matrix 生成之后，当前系统又补充了一组交叉验证实验，用来形成每标签自适应的稳定 latent set。这一步的目的不是重新解释 latent，而是确定后续分析应该使用哪一批更稳健的统计候选。

这部分实验不重新定义新的语义解释模块，而是作为 label-latent 关联指标模块的稳健性审计。它回答的是：

> 对每个 MISC 标签，应使用多大的 TopK 候选预算？在这个预算内，哪些 latent 既反复出现，又有 positive Cohen's d 的 bootstrap CI 支持？

交叉实验现在分成五个部分：

| 实验 | 做了什么 | 验证的问题 | 主要输出 |
|---|---|---|---|
| E0 分组与去重地基 | 为 6194 条 utterance 建立 `source_file` 和去重文本组映射 | 后续 split / bootstrap 是否能避免重复文本和同源文件相关性 | `outputs/cross_val/dedup_group_mapping.csv` |
| E1 repeated split-half TopK grid | 按 `source_file` 用 50 个随机种子重复二分，每次分别重算 filtered label-latent matrix，并对 `K=1..100` 统计 TopK overlap 和 latent inclusion frequency | 候选 latent 是否在不同随机二分中反复出现，TopK 集合是否稳定 | `repeated_split_topk_grid_summary.csv`, `topk_inclusion_frequency.csv`, `topk_jaccard_vs_k.png` |
| E2 grouped bootstrap CI | 对每标签 Top100 positive Cohen's d latent 做 `source_file` grouped bootstrap | AUC / Cohen's d 是否有稳定置信区间，而不是只有点估计 | `bootstrap_ci_by_label_latent.csv` |
| E3 cross-quality validation | high-quality 上选 Top100 去 low-quality 验证，反向也做 | 候选是否可能受 high/low 数据质量或来源差异影响 | `cross_quality_auc_comparison.csv`, `cross_quality_summary.csv` |
| E4 stable TopK selection | 先用 AUC@K 找性能平台 `K_auc`，再用 E1 稳定性找 `K_stab`，最终在 full-data TopK* 内筛 `stable_core` 和 `boundary_candidate` | 如何形成每标签的稳定 latent set | `stable_k_by_label.csv`, `stable_topk_latent_set.csv`, `stable_topk_global_union.csv` |

当前交叉实验的主输入固定为 filtered pool：

- `outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered/latent_label_matrix.csv`
- `outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered/feature_filter_audit.csv`
- `outputs/misc_full_sae_eval/feature_store/utterance_features.pt`
- `outputs/misc_full_sae_eval/label_matrix.csv`

最终合并输出是：

- `outputs/cross_val/filtered_pool_association_matrix_with_cv.csv`
- `outputs/cross_val/cross_val_summary.md`
- `outputs/misc_full_sae_eval/interpretability/ranked_sae_subspace_probe_k001_100/auc_by_k_curve_0_100.csv`
- `outputs/cross_val/stable_topk_selection/stable_k_by_label.csv`
- `outputs/cross_val/stable_topk_selection/stable_topk_latent_set.csv`
- `outputs/cross_val/stable_topk_selection/stable_topk_analysis_report_20260706.md`

当前结果显示，merged matrix 仍为 `114822` 行，和原始 filtered association matrix 对齐。后续主线使用 `stable_core` 作为默认 latent set。

`K*` 的确定规则是：

1. 对每个标签用 positive Cohen's d 排序的 AUC@K 曲线找性能平台 `K_auc`。
2. 从 `K >= K_auc` 中找满足 repeated split-half 稳定平台的 `K_stab`。
3. 若有 `K_stab`，则 `K* = K_stab`；若没有，则 `K* = K_auc`，并把标签标记为 `performance_only_unstable`。
4. 在 full-data TopK* 内筛 latent-level `stable_core` 和 `boundary_candidate`。

当前 `stable_core` 的简化判定规则是：

- `full_data_rank <= K*`
- `inclusion_frequency >= 0.70`
- `cohens_d_ci_lo > 0`

`boundary_candidate` 的判定规则是：

- `full_data_rank <= K*`
- `0.40 <= inclusion_frequency < 0.70`
- `cohens_d_ci_lo > 0`

cross-quality 结果继续保留为风险审计字段，但不再作为进入 `stable_core` 的硬门槛。原因是本项目的后续目标是寻找与 MI code 稳定相关、可进入人工解释的 latent set；因此“反复出现 + positive Cohen's d CI 支持”比“跨质量 split 不掉点”更直接对应当前筛选目的。cross-quality 更适合作为后续解释时的风险提示，而不是第一层候选筛选条件。

当前每标签 `K*` 和 stable set 结果如下：

| label | K_auc | K_stab | K* | status | stable_core | boundary_candidate |
|---|---:|---:|---:|---|---:|---:|
| RE | 66 | 66 | 66 | stable_topk_found | 54 | 12 |
| RES | 45 | - | 45 | performance_only_unstable | 15 | 16 |
| REC | 56 | 56 | 56 | stable_topk_found | 47 | 9 |
| QU | 28 | 28 | 28 | stable_topk_found | 26 | 2 |
| QUO | 36 | 36 | 36 | stable_topk_found | 32 | 4 |
| QUC | 26 | 59 | 59 | stable_topk_found | 45 | 14 |
| GI | 72 | - | 72 | performance_only_unstable | 26 | 37 |
| SU | 60 | - | 60 | performance_only_unstable | 31 | 28 |
| AF | 33 | 33 | 33 | stable_topk_found | 27 | 6 |

总计得到 303 条 label-latent 层面的 `stable_core`，去重后为 225 个 unique latent。后续 top utterance、P3 解释审计、人工审核、case card 和 ablation 的默认 latent set 应使用 `stable_core`。

这部分能支持：

- 原始 filtered-pool AUC / Cohen's d 结果不是单次快照噪声。
- 每个标签可以得到有证据支持的 `K*` 候选预算。
- 一部分 latent 可以被标记为 `stable_core`，作为后续默认 latent set。

不能支持：

- 交叉验证通过就说明 latent 是临床概念。
- `stable_core` 就是完整 MI 概念本体。
- `K*` 是因果最小充分集合。
- bootstrap CI 不跨 0 就说明存在因果机制。
- cross-quality stable 就可以跳过人工语义审核。

## 8. 模块五：统计结构分析

### 8.1 设计目的

统计结构模块回答：

> MISC 标签和 SAE latent 的关系是一对一、少数 latent 支撑，还是多对多、分布式支撑？

单个 label-latent 指标只能说明某个 pair 的关联。统计结构分析则看整个标签系统的形状，包括：

- 每个标签的 top latent 是否集中。
- 不同标签之间是否共享 latent。
- 共享 latent 是同家族共享还是跨家族共享。
- 哪些 latent 经常出现在多个标签中。
- 父子标签的重叠是否主导了整体 overlap。

### 8.2 label fragmentation

label fragmentation 衡量一个标签的候选表示是否分散。

如果一个标签只需要少数强 latent 就能覆盖主要预测信号，可以称为更 compact。反之，如果一个标签需要很多 latent 才能接近完整候选池表现，则说明它更 distributed。

fragmentation 不只看 stable core 的数量，还要结合：

- stable core 与 boundary candidate 的比例。
- stable core shared/exclusive 比例。
- minimal sufficient K。
- fold stability。
- redundancy。

### 8.3 latent overlap

latent overlap 现在优先衡量不同标签的 stable core latent 集合是否相交。

常用方法是对每个标签取 `stable_core`，然后计算任意两个标签之间：

```text
intersection = 两个 stable_core 集合共同出现的 latent 数
union = 两个 stable_core 集合合并后的 latent 数
Jaccard = intersection / union
```

Jaccard 越高，说明两个标签在稳定统计候选 latent 上越相似。

### 8.4 父子标签与层级重叠

MISC 标签存在层级结构，例如：

- `QU` 是 question parent，`QUO` 和 `QUC` 是子类。
- `RE` 是 reflection parent，`RES` 和 `REC` 是子类。

父子标签天然会共享数据样本和标签定义。例如 `QUO` 样本通常也属于 `QU`。因此，`QU-QUO` 或 `RE-REC` 的高 overlap 不能直接解释为模型发现了新的跨行为结构，它很可能只是标签层级导致的重叠。

因此，当前分析会区分：

1. 包含父子标签的 overlap。
2. 排除父子标签后的 overlap。
3. 进一步只看 cross-family overlap。

这样可以避免把标签体系自带的层级重叠误读成新的 SAE 结构发现。

### 8.5 经常重叠的 latent

经常重叠的 latent 是那些进入多个标签 stable core 的 latent。它们需要进一步区分角色：

| 角色 | 含义 |
|---|---|
| exclusive | 只进入一个标签的 stable core |
| family_shared | 在同一标签家族内部共享，例如 `QU` 与 `QUO` |
| cross_family | 跨不同行为家族共享 |
| global | 出现在多个家族和多个标签中 |

当前 stable core 结构中，共有 303 条 label-latent 层面的稳定候选边，去重后为 225 个 unique latent。这说明标签和 latent 之间不是一对一映射，而是多对多结构。后续结构分析应以 stable core 为主。

### 8.6 能支持的结论

可以支持：

- MISC 标签在 SAE 空间中呈多对多结构。
- 某些标签更共享，某些标签更独立。
- 父子标签重叠和跨家族重叠可以被分开分析。
- 经常重叠的 latent 更适合做边界分析或单 latent case study。

不能支持：

- overlap latent 一定表示共同语义。
- stable core Jaccard 高就说明两个标签临床意义相同。
- shared latent 就一定比 exclusive latent 更重要。

## 9. 模块六：最小预测充分子空间

### 9.1 设计目的

最小充分子空间模块回答：

> 对每个 MI/MISC 标签，需要多少个 filtered candidate latent，才能接近完整候选池的预测表现？

这里的“充分”是 predictive sufficiency，也就是在 probe-space 中足够预测标签。它不是 causal sufficiency，不说明这些 latent 被干预时一定会导致模型行为变化。

### 9.2 方法设计

对每个标签，先从 filtered association matrix 中选出该标签的 Top candidate pool。当前固定口径是每标签最多 Top100。

然后分两步：

1. 使用完整 candidate pool 训练 probe，得到 full-candidate performance。
2. 逐步寻找更小的 latent 子集，直到这个子集的表现足够接近 full-candidate probe。

判断“足够接近”时，不只看 AUC，还同时看 AUPRC 和 Precision@50 lift。这样可以避免某个小子集只在 AUC 上接近，但在低基率标签或 top-ranked retrieval 上表现明显变差。

### 9.3 minimal K 的含义

minimal K 表示在当前候选池和当前阈值下，达到接近 full-candidate performance 所需的 latent 数量。

如果 minimal K 很小，说明标签可能有相对 compact 的 SAE 子空间。如果 minimal K 很大，说明标签更 distributed，需要多个 latent 协同预测。

当前 filtered 结果中，leaf 标签都能在 filtered Top100 候选池下恢复，但 minimal K 普遍不是 1 或 2，而是 10 到 24 左右。这更支持 distributed predictive subspace，而不是单 latent 标签表示。

### 9.4 fold stability 与 redundancy

minimal K 不是唯一重要结果。还要看：

- 不同 fold 选出的 latent 是否一致。
- selected latent 或 stable core latent 之间是否冗余。
- full-candidate performance 是否稳定。
- recoverable 标签是否有非空 selected latent set；解释审计阶段则优先看 stable core latent set。

如果 minimal K 看起来不大，但不同 fold 选择的 latent 差异很大，说明具体 latent 组合不稳定，结论应该写成“存在一组可预测子空间”，而不是“固定这几个 latent 构成标签机制”。

### 9.5 能支持的结论

可以支持：

- 某标签在 filtered SAE latent space 中可预测恢复。
- 某标签更 compact 或更 distributed。
- 某标签大概需要多少 latent 才能达到预测充分。
- minimal selected latent groups 可作为预测充分性分析的候选；人工审查和解释链路优先使用 stable core latent set。

不能支持：

- minimal selected latents 是唯一真实机制。
- predictive sufficiency 等于 causal sufficiency。
- minimal K 就是人类概念的真实维度。

## 10. 模块七：top latent AI 候选解释与人工审核

### 10.1 设计目的

top latent 功能解释与人工审核模块回答：

> 某个候选 latent 的最高激活语句，到底体现了 MI 概念、咨询功能、表层语言模板，还是数据 artifact？

统计指标只能说明“相关”和“可预测”。要判断 latent 是否可解释，必须回到实际 utterances。

本模块中，AI 生成的解释只作为 candidate interpretation。解释是否可以进入研究结论，由人工审核决定，而不是由原先的自动“三层验证”直接判定。人工审核是 AI 解释之后的主验证环节，也是回答 RQ2 时判断某个 latent 是否可解释、是否更像表面模式或核心咨询功能的关键依据。

### 10.2 方法设计

对一个 stable core latent，从全 MI/MISC 数据集中按 activation 排序，取最高激活的若干 utterances。AI 可以先基于这些语句生成候选解释，但该解释必须进入人工审核流程。

人工审核时不直接问“这个 latent 是不是 QUO/RES/MI 概念”，而是先归类它捕捉的模式：

| 类型 | 说明 |
|---|---|
| surface_form | 问号、what/how、固定短语、数字量尺、句法模板 |
| dialogue_function | 开放式提问、信息给予、反映、支持、建议 |
| context_relation | 与前文 client 内容的复述、改写或情感推断 |
| mi_principle | 自主支持、合作、非评判、evocation、重要性/信心量尺 |
| artifact | ASR 错误、重复模板、转录格式、来源文件偏差 |
| mixed_unclear | 多种模式混合，无法稳定命名 |

这个分类体系的目的，是避免把表面句式误命名为临床概念。例如一个 latent 可能高度激活 `what would you...` 问句，但它捕捉的可能只是固定问句模板，不一定是“改变计划”这个 MI 概念。

人工审核至少检查四类证据：

1. top activating utterances 是否真的共享稳定模式。
2. high-non-target utterances 是否显示相邻标签混淆或 artifact。
3. random target utterances 是否说明该 latent 覆盖目标标签的代表性不足。
4. 对 RE/RES/REC 等 context-dependent 标签，是否因为缺少 client 前文而必须降级解释。

审核输出不应只给“通过/不通过”，而应给出结构化判断：dominant pattern type、artifact risk、target-label support、adjacent-label confusion、evidence quality、final status 和 reviewer notes。

### 10.3 MI 概念 vs 模式

top latent 解释中最重要的边界是：

| 层级 | 例子 | 解释强度 |
|---|---|---|
| 表层模式 | question form, how do you, what would you, scale words | 最弱，可能只是语言模板 |
| 咨询功能 | open question, information giving, reflection-like statement | 中等，接近 MISC 行为功能 |
| MI 技术 | importance/confidence ruler, autonomy permission prompt, change-talk elicitation | 较强，但仍需验证 |
| MI 原则 | autonomy support, collaboration, evocation, nonjudgment | 最强，但最难仅凭 utterance-level top examples 证明 |

当前 single-latent review 的一个例子是：某些 overlap latent 的 top utterances 高度集中在 `on a scale of one to ten`, `how important`, `how confident`, `how ready` 这类语句上。它们可以谨慎解释为 importance/confidence/readiness ruler 模式。这比普通 question-form 更接近 MI 技术，但仍不能直接说这个 latent 就是 MI 概念本身。

### 10.4 人工审核后的证据等级

AI 解释经过人工审核后，建议分成四个证据等级：

| 等级 | 特征 |
|---|---|
| high | 人工审核确认 top utterances 高度一致，语义和功能都稳定，artifact 风险较低 |
| medium | 人工审核认为有稳定模式，但可能混合表层模板和咨询功能 |
| low | 人工审核认为模式弱或标签混杂，解释依赖较多假设 |
| uninterpretable | 人工审核认为 top utterances 无稳定共性，不能命名 |

对 `RE/RES/REC` 这类 context-dependent 标签，即使 utterance 看起来像 reflection，也要降低解释强度，因为缺少 client 前文会限制功能判断。

建议人工审核的 final status 使用：

| final status | 含义 |
|---|---|
| robust_candidate | 审核者认为该 latent 有稳定、可描述的候选模式 |
| surface_form_candidate | 主要是表面语言形式或固定句式 |
| function_candidate | 更接近咨询行为功能，但仍是候选解释 |
| mi_principle_candidate | 可能体现 MI 原则，但需要更强证据支撑 |
| artifact_risk | 高激活主要可能来自格式、重复、ASR 或来源偏差 |
| mixed_unclear | 多模式混合或证据不足 |
| reject | 不采纳该 AI 候选解释 |

### 10.5 能支持的结论

可以支持：

- 经人工审核后，某 latent appears associated with 某类语言或咨询模式。
- 经人工审核后，某 latent 的 top activation examples 是否语义稳定。
- 经人工审核后，某 latent 更像 MI 技术、MISC 行为功能、表层模板还是 artifact。
- 哪些 latent 在人工审核后值得进入 case card 或后续扩展验证。

不能支持：

- top examples 好看就证明 latent 是临床概念。
- 一个 latent 的名字可以直接等于一个 MISC 标签。
- AI 自动解释未经人工审核就作为研究结论。
- 没有 token-level 和 intervention 检查时，不能写成机制发现。

## 11. 模块之间的依赖关系

这些模块之间有明确依赖：

1. 线性探针和选层决定后续使用哪层表征更合理。
2. SAE 表征质量评估决定 SAE 是否值得做可解释性分析。
3. 初筛 latent 池决定哪些 latent 进入 filtered-pool 统计。
4. label-latent 指标提供 positive Cohen's d 排序、方向性信息和每个 latent 的单变量关联证据。
5. 交叉验证审计先用 AUC@K 和 split-half 稳定性确定每标签 `K*`，再用 repeated inclusion 和 bootstrap CI 筛出 `stable_core`。
6. 统计结构分析默认使用 `stable_core`，判断标签是否 distributed、是否重叠，以及哪些 stable core latent 被多个标签共享。
7. minimal sufficient subspace 使用候选池估计每个标签需要多少 latent，但不替代 `stable_core` 作为解释对象。
8. top latent 功能解释优先使用 `stable_core`，回到真实 utterance 做模式审查。

如果前面的模块没有固定，后面的解释就容易漂移。例如，如果不先过滤 dead 或 always-active latent，top examples 可能被异常激活误导。如果不先区分父子标签 overlap，`QU-QUO` 的重叠可能被误写成新的跨标签共享结构。

## 12. 研究结论应如何表述

当前证据最适合支持以下类型的表述：

- MI/MISC 标签信息可以从 LLM hidden representation 和 SAE latent space 中被预测性读出。
- filtered SAE latent pool 中存在与 MISC 标签相关、反复出现且 positive Cohen's d CI 支持的 `stable_core` 候选特征。
- MISC 标签与 SAE latent 呈多对多映射，而不是一标签一 latent。
- 多数 leaf 标签更像 distributed predictive subspace，需要多个 latent 协同表示。
- 部分 stable core latent 的高激活语句呈现稳定的咨询行为模式或 MI 技术模板。
- top latent 解释应被视为 candidate interpretation，需要人工审核后才能进入研究结论。

不应表述为：

- SAE 已经发现了真正的 MI 概念机制。
- 某个单 latent 就是 `QUO`、`RES` 或 `MI` 概念。
- predictive sufficiency 等于 causal sufficiency。
- 高 AUC 或高 Cohen's d 自动意味着临床语义纯。
- top activation examples 足以证明模型理解 MI。

## 13. 推荐的项目阶段门

从 research-project management 的角度，可以把当前项目分成四个阶段门。

### 13.1 Stage 1：decodability

目标：证明 MI/MISC 标签可以从 LLM 表征中被简单读出。

通过条件：

- layer probe 有稳定高于随机的 AUC。
- raw hidden、PCA、SAE 等 baseline 结果可比较。
- 选层依据清楚。

### 13.2 Stage 2：filtered candidate discovery

目标：得到质量可控的 SAE latent 候选池和 label-latent association matrix。

通过条件：

- dead/rare/always-active/outlier-dominated latent 被过滤。
- filtered matrix 行数和 keep pool 对齐。
- AUC、Cohen's d、FDR、Precision@K 等指标重新在 filtered pool 上计算。
- 已补充交叉验证审计：AUC@K 曲线、50 次 source-file split-half TopK grid、grouped bootstrap CI 和 high/low cross-quality validation。
- 已生成每标签 `K*`、`stable_core`、`boundary_candidate` 和 global union；后续默认 latent set 使用 `stable_core`。

### 13.3 Stage 3：structure and sufficiency

目标：判断标签表征结构是 compact 还是 distributed，并定位共享 latent。

通过条件：

- stable core overlap、Jaccard、shared/exclusive role taxonomy 已生成。
- 父子标签 overlap 与 cross-family overlap 分开分析。
- minimal sufficient subspace 给出每个标签的 minimal K、stability 和 redundancy。

### 13.4 Stage 4：AI candidate explanation and human audit

目标：先由 AI 对 top latent 的最高激活语句生成候选功能解释，再由人工审核决定该解释是否可采纳、应降级还是应拒绝。

通过条件：

- 每个 stable core latent 有 top activation utterances。
- AI 候选解释区分 surface_form、dialogue_function、context_relation、mi_principle、artifact、mixed_unclear。
- 人工审核记录 dominant pattern type、artifact risk、target-label support、adjacent-label confusion、evidence quality 和 final status。
- 对 RE/RES/REC 明确标注缺少 client context 的限制，并由人工审核决定是否降级。
- RQ2 的回答以人工审核后的 final status 为准，而不是以 AI 自动解释或原先的三层自动验证为准。
- 输出可以支持人工确认后的 case card 或后续 intervention 设计。

## 14. 人工审核后的扩展验证方向

当前系统已经完成从 decodability 到 filtered-pool 统计结构和 top-utterance candidate interpretation 的主要链条。AI 解释之后的主验证环节应先改为人工审核。人工审核完成前，相关结果只能写作候选解释，不能写作已经验证的 latent 功能。

人工审核的基本内容包括：

1. 审查 AI candidate feature name 是否过度命名。
2. 审查 top activating utterances 是否支持该候选解释。
3. 审查 high-non-target utterances 是否显示 sibling-label confusion、surface-form trigger 或 artifact。
4. 审查该 latent 更像 surface_form、dialogue_function、mi_principle、context_relation、artifact 还是 mixed_unclear。
5. 审查 RE/RES/REC 是否因为缺少 client context 而不能支持 context-relation 结论。
6. 给出 final status：robust_candidate、surface_form_candidate、function_candidate、mi_principle_candidate、artifact_risk、mixed_unclear 或 reject。

人工审核之后，若要进一步提高 claim 强度，可以再做扩展验证：

1. token-level attribution：确认 latent 激活由哪些 token 或短语触发。
2. contrastive minimal pairs：构造表层相似但 MI 功能不同的语句，测试 latent 是否只追踪模板。
3. context-aware RE/REC analysis：加入前一句 client utterance，判断 reflection 是否真正复述或改写 client 内容。
4. held-out source validation：检查人工采纳的解释是否跨文件、跨质量 split、跨主题稳定。
5. intervention or steering：优先对 stable core latent 或 stable core latent group 做 ablation、clamping 或 steering，观察下游标签判断是否按预期变化。

人工审核会把项目从“AI 候选解释”推进到“人工采纳的解释材料”。token attribution、minimal pairs 和 intervention 等扩展验证完成后，才可以进一步讨论更强的解释可靠性或潜在机制证据。在这些验证完成前，论文和报告中应使用候选、关联、可解码、人工审核采纳的模式解释等措辞，避免直接使用因果机制或真实概念表示这样的强表述。

## 15. 总结

当前 MI/MISC SAE 分析系统的设计核心，是把一个复杂的概念解释问题拆成可审计的证据链：

1. 用线性探针确认 MI/MISC 信息是否可读。
2. 用 EV 和稀疏性指标确认 SAE 表征是否可用。
3. 用 filtered pool 排除明显不适合解释的 latent。
4. 用 AUC、Cohen's d、directional AUC、FDR 和 Precision@K 建立 label-latent 关联矩阵。
5. 用 AUC@K、50 次 source-file split-half、grouped bootstrap CI 和 cross-quality validation 审计关联候选，并筛出后续默认使用的 `stable_core` latent set。
6. 用 stable core overlap、fragmentation 和 minimal sufficient subspace 判断标签在 SAE 空间中的统计结构。
7. 以 `stable_core` 为主对象生成 top activation 候选解释，并通过人工审核区分 MI 概念、咨询功能、表层模板和 artifact。

这套设计的优势是层次清楚、证据边界明确。它能支持“MI/MISC 标签在 SAE 空间中有可预测、可审查、分布式的候选表征结构”，但不会过度声称“已经证明模型内部存在清晰因果 MI 概念机制”。
