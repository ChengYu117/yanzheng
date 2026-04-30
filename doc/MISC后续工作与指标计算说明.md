# MISC 后续工作与指标计算说明

> 说明对象：本文接续 `doc/MISC数据集代码对接说明.md`，总结 MISC 全量数据接入后已经完成的实验与解释性分析工作，并说明每一步计算了哪些指标、输出了哪些文件、这些指标可以支持什么结论。

## 1. 总览

在完成 MISC 数据集读取和样本标准化之后，项目已经从“能读取数据”推进到一条完整的 SAE 解释性分析链路：

```text
MISC unit_text 样本
  -> Llama layer-19 hidden activation
  -> OpenMOSS SAE latent features
  -> 结构指标与 RE/NonRE 功能指标
  -> latent x MISC label 矩阵
  -> Mapping Structure 分析
  -> 行为差异与 latent 案例解释
  -> 后续因果验证候选组导出
```

当前正式输入目录为：

```text
data/mi_quality_counseling_misc
```

当前正式输出目录为：

```text
outputs/misc_full_sae_eval
```

全量数据口径：

| 项目 | 数值 |
|---|---:|
| 样本单位 | 每条 `misc_annotations` 中的 `unit_text` |
| 样本总数 | 6194 |
| RE 样本 | 1358 |
| NonRE 样本 | 4836 |
| high quality 样本 | 4169 |
| low quality 样本 | 2025 |
| MISC 标签 | `RE/RES/REC/QU/QUO/QUC/GI/SU/AF/OTHER` |
| SAE latent 维度 | 32768 |
| 模型句向量维度 | 4096 |
| full structural token 数 | 93636 |

标签分布：

| Label | Count | Prevalence |
|---|---:|---:|
| RE | 1358 | 0.219 |
| RES | 516 | 0.083 |
| REC | 842 | 0.136 |
| QU | 1974 | 0.319 |
| QUO | 1206 | 0.195 |
| QUC | 768 | 0.124 |
| GI | 681 | 0.110 |
| SU | 222 | 0.036 |
| AF | 349 | 0.056 |
| OTHER | 1610 | 0.260 |

## 2. 第一阶段：全量 SAE 推理与特征保存

### 2.1 已完成工作

主流程使用 `run_sae_evaluation.py` 读取全量 MISC 样本，对每条 `unit_text` 做 Llama 推理，并在指定 hook 点抽取 activation：

```text
blocks.19.hook_resid_post
```

对应 HuggingFace Llama 的第 19 层 residual hidden state，token-level activation 形状为：

```text
[B, T, 4096]
```

随后把该 activation 输入 OpenMOSS SAE，得到 token-level SAE latent：

```text
[B, T, 32768]
```

默认不保存全量 token-level latent，而是对非 padding token 做 max pooling，保存 utterance-level 特征：

```text
[6194, 32768]
```

### 2.2 主要输出

| 文件 | 作用 |
|---|---|
| `outputs/misc_full_sae_eval/dataset_summary.json` | 数据集统计、标签统计、样本示例 |
| `outputs/misc_full_sae_eval/records.jsonl` | 标准化后的每条 MISC 样本 |
| `outputs/misc_full_sae_eval/label_matrix.csv` | 每条样本的多标签 one-hot 矩阵 |
| `outputs/misc_full_sae_eval/feature_store/utterance_features.pt` | SAE 聚合 latent 特征，形状 `[6194, 32768]` |
| `outputs/misc_full_sae_eval/feature_store/utterance_activations.pt` | 模型聚合 hidden activation，形状 `[6194, 4096]` |
| `outputs/misc_full_sae_eval/feature_store/feature_metadata.json` | 特征保存的元信息 |

### 2.3 这一阶段的意义

这一阶段把 MISC 数据从文本和标签转化为可解释性分析需要的两个核心矩阵：

```text
样本 x SAE latent
样本 x MISC label
```

后面的所有功能指标、Mapping Structure、行为差异和候选组导出，都复用这批矩阵，不需要重复加载 Llama 或重复抽取 SAE features。

## 3. 第二阶段：结构指标计算

### 3.1 已完成工作

结构指标回答的问题是：

```text
SAE 重构后的 activation 是否接近原始模型 activation？
```

本项目在 `--full-structural` 模式下，使用全量非 padding token 计算结构指标，而不是只看 utterance-level pooled features。

### 3.2 已计算指标

| 指标 | 当前数值 | 含义 |
|---|---:|---|
| `n_tokens` | 93636 | 参与 full structural 统计的非 padding token 数 |
| `mse` | 7.4456 | 原 activation 与 SAE reconstruction 的均方误差 |
| `cosine_similarity` | 0.8564 | 原 activation 与重构 activation 的方向相似度 |
| `ev_openmoss_legacy` | 0.5080 | OpenMOSS legacy EV，当前论文口径主指标 |
| `ev_openmoss_aligned` | -1.1485 | official aligned EV，严格口径下为负 |
| `ev_centered_legacy` | -0.9751 | centered 口径 EV |
| `l0_mean` | 42.94 | 每个 token 平均激活 latent 数 |
| `dead_ratio` | 0.3231 | 在本数据集上基本不激活的 latent 占比 |
| `alive_count` | 22182 | 至少有一定激活的 latent 数 |
| `dead_count` | 10586 | 低激活或不激活 latent 数 |
| `ce_loss_orig` | 5.4521 | 原模型下一词预测 CE |
| `ce_loss_sae` | 6.8049 | 用 SAE reconstruction 替换 activation 后的 CE |
| `ce_loss_delta` | 1.3528 | SAE 替换导致的 CE 增量 |
| `kl_divergence` | 2.4593 | 原模型输出分布与 SAE 替换后输出分布的 KL |

### 3.3 指标解释

`ev_openmoss_legacy=0.5080` 和 `cosine_similarity=0.8564` 说明 SAE 在 MISC 数据上保留了一部分结构信息，可以作为候选表征分析工具。

但 `ce_loss_delta=1.3528`、`kl_divergence=2.4593`，以及 aligned/centered EV 为负，说明当前 SAE 还不能被描述为高保真 activation 替代模型。它适合用于候选 latent 发现、标签映射和后续因果验证入口，不适合直接宣称“已经完整复原模型内部机制”。

### 3.4 输出位置

```text
outputs/misc_full_sae_eval/metrics_structural.json
outputs/misc_full_sae_eval/metrics_ce_kl.json
```

## 4. 第三阶段：RE/NonRE 功能指标

### 4.1 已完成工作

功能指标回答的问题是：

```text
SAE latent 空间中是否存在可区分 RE 与 NonRE 的信号？
```

该阶段使用二分类标签：

```text
label_re = predicted_code == RE or predicted_subcode in {RES, REC}
```

输入为：

```text
utterance_features: [6194, 32768]
```

### 4.2 已计算指标

| 指标 | 当前数值 | 含义 |
|---|---:|---|
| `total_latents` | 32768 | 检验的 SAE latent 总数 |
| `significant_fdr` | 3461 | BH-FDR 后与 RE/NonRE 显著相关的 latent 数 |
| top latent | 29759 | RE/NonRE 单变量最强 latent |
| top latent Cohen's d | 0.8470 | top latent 在 RE 与 NonRE 间的标准化均值差 |
| top latent AUC | 0.6936 | top latent 单独区分 RE/NonRE 的能力 |
| sparse probe k=1 AUC | 0.6936 | 只用 top-1 latent 的 probe AUC |
| sparse probe k=5 AUC | 0.7653 | 使用 top-5 latent 的 probe AUC |
| sparse probe k=20 AUC | 0.8245 | 使用 top-20 latent 的 probe AUC |
| dense probe AUC | 0.8489 | 使用全部 latent 的 dense probe AUC |
| diffmean AUC | 0.7636 | RE 与 NonRE 均值差方向的 AUC |
| max-act 平均 RE purity | 0.4640 | top activation 样本中 RE 比例均值 |

### 4.3 还计算了哪些辅助指标

除主 probe 指标外，还计算了：

| 指标组 | 作用 |
|---|---|
| `feature_absorption` | 检查 top latent 的信息是否被相近 latent 吸收或替代 |
| `tpp` | 对 top latent 做扰动，观察 probe accuracy 是否下降 |
| `feature_geometry` | 计算 top latent 之间 decoder 或特征方向相似度 |
| `latent_cards` | 为 RE 候选 latent 生成示例卡片 |
| `judge_bundle` | 为 AI 或人工专家评审准备候选 latent 示例包 |

TPP 中最大的正向 accuracy drop 约为 `0.0305`，说明个别 latent 对 RE probe 有可见影响，但整体功能贡献仍是分布式的。

### 4.4 结论

RE/NonRE 信号在 SAE 空间中可被捕获，少量 top latent 组合已经有较强判别能力。但单个 latent 的纯度和 AUC 仍有限，因此合理表述应是：

```text
RE 相关信息存在于 SAE latent 子空间中，而不是由单个纯净 latent 独立承载。
```

### 4.5 输出位置

```text
outputs/misc_full_sae_eval/metrics_functional.json
outputs/misc_full_sae_eval/functional/re_binary/metrics_functional.json
outputs/misc_full_sae_eval/functional/re_binary/candidate_latents.csv
outputs/misc_full_sae_eval/functional/re_binary/latent_cards/
```

## 5. 第四阶段：MISC latent-label 矩阵生成

### 5.1 已完成工作

该阶段把二分类 RE/NonRE 扩展为 MISC 多标签分析，计算每个 SAE latent 与每个 MISC 标签之间的统计关联。

矩阵规模：

```text
32768 latents x 10 labels = 327680 rows
```

分析标签：

```text
RE, RES, REC, QU, QUO, QUC, GI, SU, AF, OTHER
```

### 5.2 每个 latent-label 对计算的指标

| 指标 | 含义 |
|---|---|
| `label` | 当前 MISC 标签 |
| `latent_idx` | SAE latent 编号 |
| `n_positive` | 该标签正例样本数 |
| `n_negative` | 该标签负例样本数 |
| `positive_mean` | 标签正例上的 latent 平均激活 |
| `negative_mean` | 标签负例上的 latent 平均激活 |
| `cohens_d` | 正负样本之间的标准化均值差 |
| `abs_cohens_d` | 绝对效应量 |
| `auc` | latent 激活区分该标签正负例的 AUC |
| `directional_auc` | 不区分正负方向后的判别强度 |
| `p_value` | 单变量统计检验 p 值 |
| `p_fdr` | BH-FDR 校正后的 p 值 |
| `significant_fdr` | 是否通过 FDR 显著性筛选 |
| `precision_at_10` | top-10 高激活样本中该标签比例 |
| `precision_at_50` | top-50 高激活样本中该标签比例 |

### 5.3 关键全量结果

| 指标 | 数值 |
|---|---:|
| 矩阵行数 | 327680 |
| 显著 latent-label 边 | 44308 |
| 至少关联一个标签的 latent | 13052 |
| 单标签 latent | 2147 |
| 多标签 latent | 10905 |
| 多标签 latent 占比 | 0.836 |

### 5.4 各标签最强正向 latent

| Label | Top latent | Cohen's d | AUC | P@10 | P@50 |
|---|---:|---:|---:|---:|---:|
| AF | 23464 | 2.237 | 0.800 | 0.90 | 0.76 |
| GI | 16345 | 0.617 | 0.544 | 0.70 | 0.66 |
| OTHER | 29191 | 0.477 | 0.567 | 0.90 | 0.78 |
| QU | 13430 | 2.584 | 0.925 | 0.70 | 0.94 |
| QUC | 21935 | 1.461 | 0.740 | 0.50 | 0.56 |
| QUO | 9959 | 1.762 | 0.800 | 0.50 | 0.66 |
| RE | 29759 | 0.847 | 0.694 | 0.40 | 0.54 |
| REC | 31133 | 0.965 | 0.611 | 0.80 | 0.58 |
| RES | 20808 | 0.701 | 0.626 | 0.30 | 0.24 |
| SU | 24760 | 1.063 | 0.571 | 0.80 | 0.40 |

### 5.5 输出位置

```text
outputs/misc_full_sae_eval/functional/misc_label_mapping/latent_label_matrix.csv
outputs/misc_full_sae_eval/functional/misc_label_mapping/label_summary.json
outputs/misc_full_sae_eval/functional/misc_label_mapping/label_indicator_matrix.csv
outputs/misc_full_sae_eval/functional/misc_label_mapping/label_fragmentation.json
outputs/misc_full_sae_eval/functional/misc_label_mapping/latent_overlap.json
outputs/misc_full_sae_eval/functional/misc_label_mapping/label_topk_jaccard.json
outputs/misc_full_sae_eval/functional/misc_label_mapping/top_latents_by_label/
outputs/misc_full_sae_eval/functional/misc_label_mapping/top_examples_by_label/
```

## 6. 第五阶段：Mapping Structure 分析

### 6.1 已完成工作

该阶段对应导师文档中的 R1 问题：

```text
标签空间与 SAE 表征空间是否是一一对应，还是结构化多对多映射？
```

新增入口：

```text
run_misc_mapping_structure_analysis.py
```

新增模块：

```text
src/nlp_re_base/mapping_structure.py
```

这一阶段不加载 Llama，不加载 SAE，只读取已经生成的矩阵：

```text
outputs/misc_full_sae_eval/functional/misc_label_mapping/latent_label_matrix.csv
```

### 6.2 Label Fragmentation 指标

Label fragmentation 衡量：

```text
一个 MISC 标签被分散到了多少个 SAE latent 上。
```

主要指标：

| 指标 | 定义 |
|---|---|
| `n_significant_latents` | 与该标签显著相关的 latent 数 |
| `n_positive_effect_significant` | 正向显著 latent 数 |
| `n_negative_effect_significant` | 负向显著 latent 数 |
| `fragmentation_ratio` | `n_significant_latents / 32768` |
| `top_abs_cohens_d` | 该标签最强 latent 的绝对效应量 |
| `top_directional_auc` | 该标签最强 latent 的方向无关 AUC |
| `top10/top50_abs_effect_share` | top-k latent 承载的效应量集中度 |

当前 fragmentation 排名：

| Label | Sig. Latents | Pos. | Neg. | Frag. Ratio | Top Latent | Top d | Top AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| SU | 8691 | 234 | 8457 | 0.265 | 24760 | 1.063 | 0.571 |
| AF | 6891 | 342 | 6549 | 0.210 | 23464 | 2.237 | 0.800 |
| RES | 4470 | 99 | 4371 | 0.136 | 20808 | 0.701 | 0.626 |
| OTHER | 4467 | 513 | 3954 | 0.136 | 13430 | 0.790 | 0.714 |
| REC | 4192 | 2592 | 1600 | 0.128 | 31133 | 0.965 | 0.611 |
| QUO | 3872 | 972 | 2900 | 0.118 | 9959 | 1.762 | 0.800 |
| QU | 3522 | 1260 | 2262 | 0.107 | 13430 | 2.584 | 0.925 |
| RE | 3462 | 2012 | 1450 | 0.106 | 29759 | 0.847 | 0.694 |
| QUC | 2868 | 540 | 2328 | 0.088 | 21935 | 1.461 | 0.740 |
| GI | 1873 | 774 | 1099 | 0.057 | 13430 | 0.698 | 0.682 |

解释：`SU/AF/RES` 的 significant latent 数很多，但大量为负向边界信号；`QU/QUO` 的 top AUC 更强，说明问题类行为有更紧凑、更可识别的表征。

### 6.3 Latent Overlap 指标

Latent overlap 衡量：

```text
一个 SAE latent 同时关联多少个 MISC 标签。
```

主要指标：

| 指标 | 定义 |
|---|---|
| `labels_per_latent` | 每个 latent 显著关联的标签数 |
| `single_label_latents` | 只关联一个标签的 latent |
| `multi_label_latents` | 关联两个或更多标签的 latent |
| `positive_only` | 关联标签方向全为正 |
| `negative_only` | 关联标签方向全为负 |
| `mixed_direction` | 同一 latent 对不同标签方向有正有负 |
| `global_latent` | 关联 5 个及以上标签的 latent |

当前结果：

| Labels per latent | Latents | Positive-only | Negative-only | Mixed direction |
|---:|---:|---:|---:|---:|
| 1 | 2147 | 65 | 2082 | 0 |
| 2 | 3138 | 33 | 2800 | 305 |
| 3 | 2603 | 0 | 1842 | 761 |
| 4 | 1836 | 0 | 738 | 1098 |
| 5 | 1306 | 0 | 295 | 1011 |
| 6 | 872 | 0 | 134 | 738 |
| 7 | 576 | 0 | 16 | 560 |
| 8 | 319 | 0 | 14 | 305 |
| 9 | 164 | 0 | 0 | 164 |
| 10 | 91 | 0 | 0 | 91 |

解释：多标签 latent 远多于单标签 latent，是“标签空间与 SAE 表征空间不是一一对应”的直接证据。

### 6.4 Label-pair Similarity 指标

Label-pair similarity 衡量标签之间是否共享 latent 表征。

已计算：

| 指标 | 含义 |
|---|---|
| `topk_jaccard` | 两个标签 top-k latent 集合的 Jaccard |
| `significant_jaccard` | 两个标签全部 FDR 显著 latent 集合的 Jaccard |
| `pearson_cohens_d` | 两个标签 Cohen's d 向量的 Pearson 相关 |
| `spearman_cohens_d` | 两个标签 Cohen's d 排名相关 |

Top-50 Jaccard 最强关系：

| Label pair | Top-k Jaccard | Full Sig. Jaccard | Pearson d |
|---|---:|---:|---:|
| QU - QUO | 0.408 | 0.532 | 0.877 |
| RE - REC | 0.389 | 0.645 | 0.937 |
| RES - REC | 0.220 | 0.234 | -0.094 |
| QU - QUC | 0.205 | 0.373 | 0.634 |
| QU - OTHER | 0.176 | 0.302 | -0.397 |
| RE - OTHER | 0.149 | 0.359 | -0.464 |

解释：`QU-QUO` 和 `RE-REC` 的共享最明显，符合 MISC 父子标签关系。`OTHER` 与多个标签有重叠，但方向相关常为负，说明它更像异质剩余类，不应作为核心心理咨询行为结论。

### 6.5 Hierarchy Alignment 指标

Hierarchy alignment 检查 MISC 父子标签结构是否在 SAE 空间中被部分恢复。

默认检查：

```text
RE -> RES, REC
QU -> QUO, QUC
```

主要指标：

| 指标 | 定义 |
|---|---|
| `parent_child_jaccard` | parent 与 child 显著 latent 集合重合度 |
| `parent_decomposition` | parent significant latent 中有多少能被 child 覆盖 |
| `child_coverage_by_parent` | child significant latent 中有多少被 parent 覆盖 |
| `sibling_separation` | 兄弟标签之间未重叠的程度 |

当前结果：

| Relation | Jaccard | Parent decomposition | Child coverage | Sibling separation |
|---|---:|---:|---:|---:|
| RE -> all children | 0.463 | 0.958 | 0.473 |  |
| RE -> RES | 0.253 | 0.463 | 0.359 |  |
| RE -> REC | 0.645 | 0.867 | 0.716 |  |
| RES vs REC | 0.234 |  |  | 0.766 |
| QU -> all children | 0.545 | 0.906 | 0.578 |  |
| QU -> QUO | 0.532 | 0.729 | 0.663 |  |
| QU -> QUC | 0.373 | 0.493 | 0.606 |  |
| QUO vs QUC | 0.222 |  |  | 0.778 |

解释：父标签和子标签共享明显，但兄弟标签仍有分离，说明 SAE 空间部分恢复了 MISC 层级，而不是把父子标签完全混成一团。

### 6.6 Latent Role Taxonomy 指标

每个显著 latent 被划分为：

| Role | 定义 |
|---|---|
| `exclusive` | 只关联一个核心标签 |
| `family_shared` | 只在同一标签家族内共享，如 `RE/RES/REC` |
| `cross_family` | 跨标签家族共享 |
| `global` | 关联 5 个及以上标签 |
| `auxiliary_only` | 只关联辅助或非核心标签 |

当前结果：

| Role | Latents | Share |
|---|---:|---:|
| cross_family | 6594 | 0.505 |
| global | 3328 | 0.255 |
| exclusive | 2416 | 0.185 |
| family_shared | 551 | 0.042 |
| auxiliary_only | 163 | 0.012 |

解释：cross-family 和 global latent 占比很高，说明模型内部表征更像连续、混合的行为子空间，而不是人工标签的离散字典。

### 6.7 输出位置

```text
outputs/misc_full_sae_eval/interpretability/mapping_structure/mapping_structure_metrics.json
outputs/misc_full_sae_eval/interpretability/mapping_structure/label_fragmentation_rank.csv
outputs/misc_full_sae_eval/interpretability/mapping_structure/latent_overlap_distribution.csv
outputs/misc_full_sae_eval/interpretability/mapping_structure/label_pair_similarity.csv
outputs/misc_full_sae_eval/interpretability/mapping_structure/hierarchy_alignment.csv
outputs/misc_full_sae_eval/interpretability/mapping_structure/latent_role_summary.csv
outputs/misc_full_sae_eval/interpretability/mapping_structure/mapping_structure_report.md
outputs/misc_full_sae_eval/interpretability/mapping_structure/figures/
doc/MISC Mapping Structure分析报告.md
```

## 7. 第六阶段：行为差异与后续可解释性分析

### 7.1 已完成工作

该阶段对应 R2 方向：

```text
不同行为标签在 SAE 空间中的表征模式是否不同？
```

新增入口：

```text
run_misc_interpretability_analysis.py
```

新增模块：

```text
src/nlp_re_base/behavior_interpretability.py
```

该阶段读取 Mapping Structure 结果、原始 records 和 feature store，不重新抽模型。

### 7.2 行为模式分类

每个核心标签被分为三类：

| Pattern | 含义 |
|---|---|
| `compact_strong` | top latent 较强，表征相对集中 |
| `shared_distributed` | 表征分散且与其他标签共享较多 |
| `negative_boundary` | 显著信号以负向或边界分离为主 |

当前分布：

| Pattern | 标签数 |
|---|---:|
| shared_distributed | 4 |
| negative_boundary | 3 |
| compact_strong | 2 |

主要判断：

| 标签类型 | 当前观察 |
|---|---|
| `QU/QUO` | 更接近 `compact_strong`，问题类行为表征更集中 |
| `RE/REC/QUC/GI` | 更接近 `shared_distributed`，需要组合 latent 解释 |
| `AF/SU/RES` | 更接近 `negative_boundary`，很多显著边是负向分离 |

### 7.3 行为差异指标

| 指标 | 含义 |
|---|---|
| `group_score_pos_mean` | 标签正例在 top-positive latent 组上的平均分 |
| `group_score_neg_mean` | 标签负例在 top-positive latent 组上的平均分 |
| `group_score_label_cohens_d` | 标签正负例组分数差异效应量 |
| `exclusive_latents` | 该标签独占 latent 数 |
| `family_shared_latents` | 标签家族内部共享 latent 数 |
| `cross_family_latents` | 跨家族共享 latent 数 |
| `global_latents` | 关联 5 个及以上标签的 latent 数 |
| `nearest_labels_top50` | top-50 latent 重叠最高的近邻标签 |
| `nearest_jaccard_top50` | 与近邻标签的 top-50 Jaccard |

几个代表性结果：

| Label | Pattern | Sig. Latents | Top AUC | Group score d | Nearest labels |
|---|---|---:|---:|---:|---|
| QU | compact_strong | 3522 | 0.925 | 2.530 | QUO, QUC, OTHER |
| QUO | compact_strong | 3872 | 0.800 | 2.370 | QU, OTHER, QUC |
| RE | shared_distributed | 3462 | 0.694 | 1.321 | REC, OTHER, RES |
| REC | shared_distributed | 4192 | 0.611 | 1.847 | RE, RES, OTHER |
| QUC | shared_distributed | 2868 | 0.740 | 1.892 | QU, OTHER, QUO |
| AF | negative_boundary | 6891 | 0.800 | 3.140 | SU, RES, OTHER |
| SU | negative_boundary | 8691 | 0.571 | 1.920 | AF, OTHER, RE |
| RES | negative_boundary | 4470 | 0.626 | 0.466 | REC, OTHER, RE |

### 7.4 质量分层指标

该阶段还比较了 high quality 与 low quality 会话中同一标签的 activation shift。

| 指标 | 含义 |
|---|---|
| `within_label_high_mean` | high quality 中该标签样本的 top-latent 组分数 |
| `within_label_low_mean` | low quality 中该标签样本的 top-latent 组分数 |
| `within_label_high_low_diff` | high 与 low 的均值差 |
| `within_label_high_low_d` | high 与 low 的 Cohen's d |

代表性结果：

| Label | High mean | Low mean | Diff | Cohen's d |
|---|---:|---:|---:|---:|
| RE | 0.600 | 0.473 | 0.127 | 0.348 |
| REC | 0.550 | 0.441 | 0.109 | 0.292 |
| AF | 0.436 | 0.362 | 0.074 | 0.242 |
| QU | 0.542 | 0.563 | -0.021 | -0.066 |
| QUC | 0.402 | 0.561 | -0.159 | -0.496 |

解释：`RE/REC` 在 high quality 对话中激活更强，和高质量咨询中更多反映性倾听的预期一致。但这只是相关分析，不等于因果结论。

### 7.5 输出位置

```text
outputs/misc_full_sae_eval/interpretability/followup_analysis/followup_interpretability_metrics.json
outputs/misc_full_sae_eval/interpretability/followup_analysis/behavior_asymmetry/behavior_asymmetry_summary.csv
outputs/misc_full_sae_eval/interpretability/followup_analysis/behavior_asymmetry/quality_label_activation_shift.csv
outputs/misc_full_sae_eval/interpretability/followup_analysis/behavior_asymmetry/family_asymmetry_summary.csv
outputs/misc_full_sae_eval/interpretability/followup_analysis/behavior_asymmetry/behavior_asymmetry_report.md
doc/MISC后续可解释性阶段分析报告.md
```

## 8. 第七阶段：latent 案例解释

### 8.1 已完成工作

为了避免只看统计量，该阶段为每个核心标签选取 top latent，并输出高激活样本卡片。

分析设置：

| 项目 | 数值 |
|---|---:|
| 标签数 | 9 |
| 每标签 top latent 数 | 5 |
| 每个 latent top examples | 12 |
| case cards 总数 | 45 |

### 8.2 已计算指标

| 指标 | 当前数值 | 含义 |
|---|---:|---|
| `n_case_cards` | 45 | 生成的 latent 案例卡片数 |
| `mean_target_purity` | 0.604 | top examples 中目标标签平均占比 |
| `high_purity_candidate` | 20 | 高纯度候选 latent 数 |
| `mixed_but_label_relevant` | 14 | 混合但仍和标签相关的 latent 数 |
| `low_purity_review_required` | 11 | 需要人工复核的 latent 数 |

### 8.3 案例卡片包含什么

每张 latent case card 通常包含：

| 内容 | 作用 |
|---|---|
| latent id | 可追踪到矩阵和候选组 |
| 所属 label | 当前解释标签 |
| top activation examples | 高激活文本样本 |
| target purity | top examples 中目标标签比例 |
| related labels | 共同出现或混合标签 |
| common tokens | 高频词线索 |
| 自动状态 | 是否适合进入人工命名或因果验证 |

### 8.4 输出位置

```text
outputs/misc_full_sae_eval/interpretability/followup_analysis/latent_cases/latent_case_metrics.json
outputs/misc_full_sae_eval/interpretability/followup_analysis/latent_cases/latent_case_summary.csv
outputs/misc_full_sae_eval/interpretability/followup_analysis/latent_cases/latent_case_report.md
outputs/misc_full_sae_eval/interpretability/followup_analysis/latent_cases/latent_case_cards/
```

## 9. 第八阶段：因果验证候选组导出

### 9.1 已完成工作

该阶段不是直接跑因果干预，而是把前面筛出来的 MISC label latent 转换成后续因果验证可以直接使用的候选组。

新增入口：

```text
run_misc_causal_candidate_export.py
```

新增模块：

```text
src/nlp_re_base/causal_candidates.py
```

### 9.2 候选组设计

对每个核心标签导出：

| 组别 | 含义 |
|---|---|
| `G1` | 排名最高的 1 个 latent |
| `G5` | 排名前 5 个 latent |
| `G10` | 排名前 10 个 latent |
| `G20` | 排名前 20 个 latent |
| `bottom_control` | 低排名控制组 |
| `random_control` | 随机控制组 |

候选排序综合考虑：

```text
Cohen's d
directional AUC
precision@50
case card purity
behavior asymmetry top-positive 排名
```

### 9.3 当前候选组摘要

| Label | Pattern | Candidates | High-purity in G20 | G1 |
|---|---|---:|---:|---:|
| RE | shared_distributed | 2012 | 1 | 19435 |
| RES | negative_boundary | 99 | 0 | 20808 |
| REC | shared_distributed | 2592 | 1 | 20436 |
| QU | compact_strong | 1260 | 5 | 13430 |
| QUO | compact_strong | 972 | 1 | 9959 |
| QUC | shared_distributed | 540 | 2 | 14014 |
| GI | shared_distributed | 774 | 4 | 16345 |
| SU | negative_boundary | 234 | 2 | 24760 |
| AF | negative_boundary | 342 | 4 | 23464 |

推荐优先级：

| 用途 | 建议 |
|---|---|
| RE/NonRE 因果验证 | 优先用 `RE` 的 `G20`，再比较 `G5/G10` |
| 紧凑标签对照 | 优先用 `QU`、`QUO`，因为它们 top signal 更强 |
| RE 子类分析 | 比较 `REC` 与 `RES`，注意 `RES` 候选更少且负向边界更明显 |
| 解释风险较高标签 | `AF/SU/RES` 应优先看 necessity，不宜直接宣称 sufficiency |

### 9.4 输出位置

```text
outputs/misc_full_sae_eval/interpretability/causal_candidates/causal_candidate_metrics.json
outputs/misc_full_sae_eval/interpretability/causal_candidates/candidate_group_summary.csv
outputs/misc_full_sae_eval/interpretability/causal_candidates/causal_candidate_groups.json
outputs/misc_full_sae_eval/interpretability/causal_candidates/label_candidates/
outputs/misc_full_sae_eval/interpretability/causal_candidates/causal_candidate_report.md
doc/MISC因果验证候选组说明.md
```

## 10. 测试与验证

已补充并运行过的测试包括：

| 测试 | 作用 |
|---|---|
| `test_dataset_loader_smoke.py` | 验证 MISC/legacy 数据读取 |
| `test_misc_label_mapping_smoke.py` | 验证 latent-label 矩阵生成 |
| `test_mapping_structure_analysis.py` | 验证 fragmentation、overlap、Jaccard、hierarchy 指标 |
| `test_behavior_interpretability.py` | 验证行为差异和 case card 分析 |
| `test_causal_candidate_export.py` | 验证候选组导出 |
| `test_causal_smoke.py` | 验证既有因果验证基础流程 |

最近日志记录的验证结果：

```text
test_mapping_structure_analysis.py: 2 passed
test_behavior_interpretability.py: 2 passed
test_causal_candidate_export.py: 2 passed
test_causal_smoke.py: 22 tests OK
```

此外，新增模块均做过 `py_compile` 语法检查。

## 11. 当前证据链可以支持什么结论

可以支持：

1. MISC 全量数据已经成功接入 SAE 主流程。
2. 每条 MISC 行为单元已经生成稳定的 SAE utterance feature。
3. SAE 在 MISC 数据上具有研究可用性，适合作为候选表征发现工具。
4. RE/NonRE 信号在 SAE latent 空间中存在，top-k sparse latent 可达到较好的 probe AUC。
5. MISC 标签和 SAE latent 不是一一对应，而是结构化多对多映射。
6. `RE/REC/RES` 和 `QU/QUO/QUC` 的父子标签关系在 SAE 空间中被部分恢复。
7. 不同行为标签的表征模式不同，`QU/QUO` 更紧凑，`RE/REC` 更分布式，`AF/SU/RES` 更偏边界信号。
8. 已经形成可进入后续因果验证的候选 latent 组。

暂时不能支持：

1. 不能说某个单一 latent 就等价于某个 MISC 标签。
2. 不能说当前 SAE 是高保真 activation 替代模型。
3. 不能把统计显著关联直接写成因果机制。
4. 不能把 `OTHER` 当作单一心理咨询行为解释。
5. 不能只凭 top example purity 就完成最终解释命名，仍需要人工审查。

## 12. 当前文档与报告索引

| 文档 | 内容 |
|---|---|
| `doc/MISC数据集代码对接说明.md` | 数据如何读取、样本如何构造、标签如何映射 |
| `doc/MISC全量SAE可用性分析报告.md` | 全量 SAE 结果可用性判断 |
| `doc/MISC矩阵生成结果评测报告.md` | 矩阵生成链路 smoke 评测 |
| `doc/MISC Mapping Structure分析报告.md` | R1 多对多映射结构证据 |
| `doc/MISC后续可解释性阶段分析报告.md` | R2 行为差异与 latent 案例分析 |
| `doc/MISC因果验证候选组说明.md` | 后续因果验证候选组 |
| `doc/MISC后续工作与指标计算说明.md` | 本文，汇总后续工作与指标口径 |

## 13. 下一步建议

后续工作建议按以下顺序推进：

1. 对 `RE/REC/QU/QUO/AF` 的高优先级 case cards 做人工命名和语义审查。
2. 使用 `outputs/misc_full_sae_eval/interpretability/causal_candidates` 中的 `G1/G5/G10/G20` 运行因果干预。
3. 对比 top candidate、bottom control、random control 的干预结果，区分真实行为贡献和统计相关。
4. 若需要更细粒度解释，再为少数候选 latent 开启 token-level top-k 保存，检查激活落在具体哪些词或短语上。
5. 将最终写作口径控制为“候选表征子空间与标签结构对齐”，避免写成“单 latent 对应单标签概念”。
