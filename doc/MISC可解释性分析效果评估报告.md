# MISC Top20 可解释性分析效果评估报告

> 本报告评估对象已经改为各标签 Top20 latent。全量矩阵只负责排序与候选生成，不作为正式可解释性结论。

## 1. 总体判断

当前 Top20 可解释性结果是可用的，可以支撑一个科研工作中的结构性可解释性分析，但仍需要后续因果干预来把“相关性解释”推进为“机制性解释”。

最重要的变化是：研究对象不再是全 SAE 空间，也不是所有 FDR 显著 latent，而是每个核心 MISC 标签的 Top20 候选 latent。这个口径更清晰、更容易人工审查，也更适合接入 `G1/G5/G10/G20` 因果验证。

## 2. Top20 结构证据

| 指标 | 数值 | 解释 |
|---|---:|---|
| Top20 latent-label 边 | 180 | 9 个核心标签各最多 20 个候选 |
| Top20 去重 latent | 136 | 不同标签候选存在共享 |
| Top20 单标签 latent | 103 | 多数候选仍较专属 |
| Top20 多标签 latent | 33 | 共享结构明确存在 |
| Top20 多标签占比 | 0.243 | 多对多存在，但没有被全量显著统计夸大 |

该结果比旧的全量显著口径更适合论文表达：它证明了在最强候选层面仍能观察到结构化重叠，而不是依赖大规模显著性扫描制造 overlap。

## 3. 行为差异

Top20 结果显示不同 MISC 行为的表征形态不同：

| 标签 | 结构判断 |
|---|---|
| `QU` | 最强共享型标签，Top20 中 19 个为共享 latent，且 top AUC 达到 0.925 |
| `QUO` | 与父标签 `QU` 高度重叠，但这属于层级一致性；与同父子类 `QUC` 的共享较低 |
| `RE` | Top20 中 14 个共享、6 个专属，说明 RE 同时具有专属和共享表征 |
| `REC` | 与 `RE` 重叠强，适合作为 RE 子类衔接分析 |
| `QUC` | 有共享但更分散，弱于 `QUO` |
| `AF/SU/RES` | Top20 更偏专属或边界型，需要谨慎解释 |

这支持导师文档中的关键点：不同咨询行为的不对应方式不同，不能只说“不是一一对应”，而要说明这种错位有结构、有差异。

## 4. 质量评估

当前结果可以接受为科研工作的原因：

- 研究对象明确：每个标签 Top20 latent，规模可控，可人工复核。
- 结构证据需要层级去重解释：`QU-QUO`、`RE-REC` 的 Top20 Jaccard 高，但这是父子标签包含关系，不能作为独立发现；更稳健的证据来自同父子类分离和跨家族有限共享。
- 案例解释已接入：Top20 候选已生成 latent case cards，可继续人工命名。
- 因果验证入口已生成：每个标签已有 `G1/G5/G10/G20` 候选组。

当前仍然不足的地方：

- Top20 仍是相关性候选，不是因果证明。
- `RE` 的 top AUC 不如 `QU`，说明 RE 更像复合行为，不适合写成单一强 latent。
- `AF/SU/RES` 的 Top20 解释需要人工审查，避免把边界型信号写成清晰正向概念。

## 5. 论文写作建议

可以写：

> The label-latent mapping remains structured even when restricted to the top-20 latents per behavior label. Parent-child overlaps are treated as hierarchy checks rather than independent evidence; sibling labels such as QUO/QUC and RES/REC show limited overlap, while cross-family sharing is modest and concentrated in a few behavior pairs.

不建议写：

> 大量 SAE latent 都显著关联 MISC 标签，因此标签和 latent 是多对多关系。

更准确的中文表述是：

> 在每个 MISC 标签最强的 Top20 SAE 候选中，父子标签重叠应被视为标签层级一致性检查，而不是独立发现；更可靠的结构证据是同父子类共享有限、跨家族共享较弱且集中，因此可解释性发现应在层级去重后的候选子空间中表述。

## 6. Latent × Label 热力图输出

已基于 `latent_label_matrix.csv` 生成 SAE latent 与 MISC label 的热力图。输出目录：

- `outputs/misc_full_sae_eval/interpretability/latent_label_heatmaps`

已生成 4 类 latent 子集：

| 子集 | latent 数 | 用途 |
|---|---:|---|
| `top20_union` | 136 | 论文主分析推荐图，对应每个标签 Top20 的并集 |
| `top30_union` | 203 | 检查 Top20 边界外的近邻候选 |
| `adaptive_pool` | 117 | 基于 cutoff audit 的质量门控候选池，适合人工命名和因果候选 |
| `full_all_latents` | 32768 | 全 SAE latent 压缩视图，用于完整性留档，不适合逐行阅读 |

每个子集都生成了 4 个指标热力图：

| 指标图 | 说明 |
|---|---|
| `cohens_d_heatmap.png` | signed Cohen's d，显示正向/负向关联 |
| `directional_auc_heatmap.png` | latent 激活对标签/非标签样本的排序区分能力 |
| `precision_at_50_heatmap.png` | 激活最高 50 条样本中的目标标签比例 |
| `precision_lift_at_50_heatmap.png` | Precision@50 相对标签基线频率的提升，更适合跨标签比较 |

索引文件：

- `outputs/misc_full_sae_eval/interpretability/latent_label_heatmaps/latent_label_heatmap_report.md`
- `outputs/misc_full_sae_eval/interpretability/latent_label_heatmaps/latent_label_heatmap_manifest.csv`

当前矩阵中已有 `Precision@50`，没有正式保存 `Precision@100`，因此本轮热力图采用 `Precision@50` 与 `Precision@50 lift`。

## 7. Fragmentation / Overlap / Polysemanticity 指标状态

这三类指标已经完成计算，并且已经形成结果文件和图：

| 指标 | 已有输出 | 当前口径 | 是否需要重算 |
|---|---|---|---|
| Fragmentation | `outputs/misc_full_sae_eval/interpretability/mapping_structure/label_fragmentation_rank.csv`、`figures/fragmentation_bar.png` | 每个标签 Top20 中 exclusive/shared latent 的数量与比例 | 不需要 |
| Overlap | `outputs/misc_full_sae_eval/interpretability/mapping_structure/label_pair_similarity.csv`、`figures/label_jaccard_heatmap.png` | 两个标签 Top20 latent 集合的 Jaccard 相似度 | 不需要 |
| Polysemanticity | `outputs/misc_full_sae_eval/interpretability/mapping_structure/latent_overlap_distribution.csv`、`latent_role_summary.csv`、`latent_role_assignments.csv`、`figures/latent_polysemanticity_histogram.png` | 一个 latent 同时进入多少个标签 Top20，并据此划分 role | 不需要 |

当前主要数值：

| 指标组 | 关键结果 |
|---|---|
| Fragmentation | `QU` shared ratio 为 `0.95`，`RE/QUO` 为 `0.70`，`REC` 为 `0.65`；`AF` 为 `0.00`，说明 AF 在 Top20 中最专属 |
| Overlap | 原始 Top20 Jaccard 最高的是父子标签对，但不作为独立结论；层级去重后，`QUO-QUC=0.053`、`RES-REC=0.000`，最强 leaf-level 跨家族对为 `QUO-GI=0.143` |
| Polysemanticity | Top20 去重后 136 个 latent；103 个只进入 1 个标签，28 个进入 2 个标签，1 个进入 3 个标签，2 个进入 4 个标签，2 个进入 5 个标签 |
| Role taxonomy | exclusive 103 个，family_shared 27 个，cross_family 4 个，global 2 个 |

当前 role 判断口径：

| role | 当前定义 |
|---|---|
| `exclusive` | latent 只进入 1 个标签的 Top20 |
| `family_shared` | latent 进入多个标签，但这些标签属于同一 MISC 家族，例如 `RE/REC` 或 `QU/QUO/QUC` |
| `cross_family` | latent 进入多个不同 MISC 家族，但进入标签数未达到 global 门槛 |
| `global` | latent 进入至少 5 个标签的 Top20 |

注意：这里的 `global >= 5 labels` 是当前项目实现中的分析门槛，不是 SAE 领域统一公认阈值。SAE 研究里更常见的是根据具体任务设定 TopK、激活频率、解释纯度或人审一致性门槛，并没有一个所有任务通用的 fragmentation / overlap / polysemanticity 阈值。

如果后续要新增“高/中/低 fragmentation”“强/弱 overlap”“低/高 polysemanticity”的分档结论，需要单独确认阈值后再生成最终版本。

## 8. 层级去重后的 overlap 修正

由于 `QU` 是 `QUO/QUC` 的父标签，`RE` 是 `RES/REC` 的父标签，父子标签的 Top20 overlap 不能作为独立的模型表征发现。已新增层级感知 overlap 审计：

- `outputs/misc_full_sae_eval/interpretability/mapping_structure/hierarchy_aware_overlap/hierarchy_aware_overlap_report.md`

修正后的结果：

| 比较对象 | 共享 latent 数 | Jaccard | 解释 |
|---|---:|---:|---|
| `QU` vs `QUO/QUC` child union | 19 | 0.487 | 父标签被子标签覆盖较多，这是标签层级一致性，不作为独立结论 |
| `RE` vs `RES/REC` child union | 13 | 0.277 | 父标签和子标签有覆盖，但弱于 QU 家族 |
| `QUO` vs `QUC` | 2 | 0.053 | 同父子类共享很少 |
| `RES` vs `REC` | 0 | 0.000 | 同父子类在 Top20 中分离 |
| `QUO` vs `GI` | 5 | 0.143 | 最强 leaf-level 跨家族共享 |
| `RES` vs `GI` | 4 | 0.111 | 弱跨家族共享 |
| `RES` vs `QUO` | 4 | 0.111 | 弱跨家族共享 |

使用 leaf-label family union 排除父标签后，跨父类家族 overlap 较弱：

| 家族比较 | 共享 latent 数 | Jaccard |
|---|---:|---:|
| `QUO/QUC` family vs `GI` | 5 | 0.094 |
| `RES/REC` family vs `GI` | 4 | 0.071 |
| `RES/REC` family vs `QUO/QUC` family | 4 | 0.054 |

因此，后续论文中不应再把 `QU-QUO` 或 `RE-REC` 的高重叠作为主要发现。更准确的结论是：父子重叠符合标注体系；同父子类多保持区分；跨家族共享存在但较弱，并集中在少数与 `GI`、`RES`、`QUO` 相关的 latent 上。

## 9. 结构关系 v2 实验结果

已按层级感知口径重新设计并执行 Fragmentation / Overlap / Polysemanticity 分析。输出目录：

- `outputs/misc_full_sae_eval/interpretability/structural_relations_v2`

核心输出：

| 文件 | 内容 |
|---|---|
| `label_fragmentation_v2.csv` | 每个 leaf 标签的 support latent 数、effective n、紧凑/分散分档 |
| `leaf_pair_overlap_v2.csv` | leaf-label pair 的 raw/support Jaccard |
| `family_union_overlap_v2.csv` | `RE_family`、`QU_family`、`GI/SU/AF` 等 family union 的 overlap |
| `parent_child_consistency_v2.csv` | `RE/QU` 父标签与子标签的 coverage，一律解释为一致性检查 |
| `latent_polysemanticity_v2.csv` | latent 支持 leaf 标签数量的分布 |
| `latent_role_assignments_v2.csv` | label_specific / cross_family / generalized 等 role 分配 |
| `structural_relation_report.md` | 自动生成的 v2 总报告 |

### 9.1 Fragmentation v2

| 标签 | 分档 | support latent 数 | effective n | 判断 |
|---|---|---:|---:|---|
| QUO | distributed | 20 | 19.696 | 最分散 |
| REC | distributed | 15 | 14.982 | 分散 |
| AF | distributed | 12 | 11.679 | 分散 |
| QUC | distributed | 12 | 11.625 | 分散 |
| SU | compact | 2 | 1.996 | 紧凑/当前强支持少 |
| GI | compact | 2 | 1.986 | 紧凑/当前强支持少 |
| RES | compact | 2 | 1.975 | 紧凑/多数候选弱或边界型 |

### 9.2 Overlap v2

主结论以 leaf-label 和 family-union 为准，不把父子标签 overlap 当独立发现。

| 比较 | support Jaccard | raw Jaccard | 分档 | 解释 |
|---|---:|---:|---|---|
| RES-GI | 0.333 | 0.111 | high | 只共享 1 个 support latent，因 support union 很小，需谨慎解释 |
| QUO-GI | 0.100 | 0.143 | moderate | 信息类与开放式问题有少量共享 |
| QUO-QUC | 0.032 | 0.053 | low | 同父问题子类共享很低 |
| RES-REC | 0.000 | 0.000 | low | 同父反映子类在 Top20 中分离 |
| RE_family-QU_family | 0.021 | 0.054 | low | 反映类与问题类整体 overlap 很低 |
| GI/SU/AF block vs RE_family | 0.031 | 0.042 | low | 建议/信息/支持类与反映类 overlap 很低 |

### 9.3 Polysemanticity v2

在 support edge 口径下，polysemantic latent 很少：

| role | latent 数 | 解释 |
|---|---:|---|
| label_specific | 59 | 只支持一个 leaf 标签 |
| cross_family | 1 | 支持两个 family、2-3 个 leaf 标签 |
| generalized | 1 | 支持至少 3 个 family 或更广泛 leaf 标签 |

因此，层级去重后的结论不是“大量 latent 泛化共享”，而是“多数强支持 latent 更接近标签特异，少数 latent 承担跨家族或泛化作用”。

### 9.4 Step 5：跨标签结构固定结论

本步骤回答的问题是：不同 MISC 行为标签在 SAE latent 空间中是否表现出不同结构；哪些标签紧凑，哪些标签碎片化；哪些标签之间共享 latent；这种共享是否符合 MISC 的父子层级。

**结论 1：不同 MISC 标签表现出不同的表征结构。**

按 support edge 门控后的 `label_fragmentation_v2.csv`，`SU`、`GI`、`RES` 属于紧凑标签，Top20 中只有 2 个 support latent，effective n 约为 2；`QUO`、`REC`、`AF`、`QUC` 属于分散标签，需要 12-20 个 support latent，effective n 也接近 support latent 数。这说明 MISC 标签不是统一地由单个 latent 或同样规模的 latent 集合表示，而是不同咨询行为对应不同程度的表征碎片化。

| 类型 | 标签 | 证据 |
|---|---|---|
| 紧凑 | `SU`, `GI`, `RES` | `n_support_latents=2`，`effective_n≈2` |
| 分散 | `QUO`, `REC`, `AF`, `QUC` | `n_support_latents=12-20`，`effective_n=11.625-19.696` |

**结论 2：父子标签重叠符合 MISC 层级结构，但不能作为独立发现。**

`RE` 是 `RES/REC` 的父标签，`QU` 是 `QUO/QUC` 的父标签，因此 `RE-REC`、`QU-QUO` 的高 overlap 主要反映标注体系中的包含关系。该结果可以作为 parent-child consistency 检查，但不能写成“模型独立发现了两个并列标签共享表征”。

| 比较 | support Jaccard | 解释口径 |
|---|---:|---|
| `RE` vs `REC` | 0.350 | 父子一致性检查，不作为独立 overlap 发现 |
| `QU` vs `QUO` | 0.560 | 父子一致性检查，不作为独立 overlap 发现 |
| `RE` vs `RES/REC` union | 0.318 | 父标签被子标签部分覆盖 |
| `QU` vs `QUO/QUC` union | 0.515 | 父标签被子标签高度覆盖 |

**结论 3：去除父标签后，同父子类之间共享很低。**

真正更干净的比较是 sibling pair，即 `QUO` vs `QUC`、`RES` vs `REC`。结果显示，这些子类并不是简单复用同一组 latent。

| 比较 | support Jaccard | raw Jaccard | 结论 |
|---|---:|---:|---|
| `QUO` vs `QUC` | 0.032 | 0.053 | 开放式/封闭式问题共享很低 |
| `RES` vs `REC` | 0.000 | 0.000 | 简单/复杂反映在 Top20 中分离 |

**结论 4：跨家族共享整体较弱，仅集中在少数标签对。**

反映类与问题类、信息/支持/肯定类与反映类之间的 overlap 整体较低。`RES-GI` 的 support Jaccard 虽为 0.333，但只共享 1 个 support latent，因 support union 很小，应谨慎解释；更稳妥的说法是跨家族共享有限，并集中在 `GI`、`RES`、`QUO` 相关边上。

| 比较 | support Jaccard | raw Jaccard | 解释 |
|---|---:|---:|---|
| `RE_family` vs `QU_family` | 0.021 | 0.054 | 反映类与问题类整体 overlap 很低 |
| `GI` vs `RE_family` | 0.056 | 0.071 | 信息类与反映类共享较弱 |
| `GI` vs `QU_family` | 0.065 | 0.094 | 信息类与问题类有少量共享 |
| `GI/SU/AF` block vs `RE_family` | 0.031 | 0.042 | 建议/信息/支持类与反映类 overlap 很低 |
| `SU` vs `RE_family` | 0.000 | 0.000 | 支持类与反映类在 Top20 中分离 |
| `AF` vs `RE_family` | 0.000 | 0.000 | 肯定类与反映类在 Top20 中分离 |

**可固定写入论文的结果段落：**

> 不同的 MISC 行为标签表现出不同的 SAE 表征结构。`SU`、`GI` 和 `RES` 在 support-gated Top20 候选中只需要少量 latent 表示，呈现较紧凑结构；相比之下，`QUO`、`REC`、`AF` 和 `QUC` 需要更多 latent，表现为更分散的表征。标签间 overlap 也具有明显层级特征：`QU-QUO` 和 `RE-REC` 的高重叠主要反映 MISC 父子标签的标注包含关系，应作为层级一致性检查，而非独立发现。去除父标签后，同父子类之间的共享较弱，`QUO-QUC` 和 `RES-REC` 均表现出较低 overlap；跨家族比较中，反映类与问题类、信息/支持/肯定类与反映类之间的 overlap 整体较低，仅有少数 latent 承担跨标签共享作用。

该段落应作为当前 Step 5 的固定结论。后续如新增因果干预结果，可以在此基础上补充 causal evidence，但不应再把父子标签 overlap 当成独立结构发现。
