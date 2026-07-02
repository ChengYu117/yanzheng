## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-02T12:57:55+08:00
- Verification Status: ANALYZED
- Version Label: filtered_pool_validation_v1
- Source Repository: `D:\project\NLP_re_dataset_model_base`
- Git HEAD: `c80d1205f7bfd5be3ca43082e1110c0b20ead286`
- Python: `C:\Users\chengyu\AppData\Local\miniconda3\envs\qwen-env-py311\python.exe`

# Filtered 初筛池 MISC 表征结构与最小充分子空间分析报告

## 1. 固定实验记录

### 1.1 实验目标

本实验将原先面向全 SAE latent 空间或旧候选集的统计分析，固定到 `FeatureFilterConfig keep=True` 后的 filtered 初筛候选池上。目标是重新计算并固化两类指标：

1. MISC 标签与 SAE latent 的统计结构：fragmentation、latent overlap、Top-k Jaccard、latent role taxonomy。
2. 每个 MISC 标签的最小预测充分 latent 子空间：full-candidate probe、minimal K、fold stability、redundancy、selected latents。

本报告中的“充分”只指 probe-space predictive sufficiency，不等同于因果充分性。

### 1.2 输入与候选池口径

| 项目 | 固定值 |
|---|---|
| 主数据目录 | `outputs/misc_full_sae_eval` |
| filtered association | `outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered/latent_label_matrix.csv` |
| feature filter audit | `outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered/feature_filter_audit.csv` |
| feature store | `outputs/misc_full_sae_eval/feature_store/utterance_features.pt` |
| label matrix | `outputs/misc_full_sae_eval/label_matrix.csv` |
| records | `outputs/misc_full_sae_eval/records.jsonl` |
| candidate policy | `filtered_topk_only` |
| candidate TopK | 100 per label for minimal sufficient subspace |
| structure TopK | 20 per label for mapping-structure analysis |
| thresholded / stable-edge seeds | not used |

### 1.3 执行命令

```powershell
& 'C:\Users\chengyu\AppData\Local\miniconda3\envs\qwen-env-py311\python.exe' run_misc_filtered_label_mapping.py
& 'C:\Users\chengyu\AppData\Local\miniconda3\envs\qwen-env-py311\python.exe' run_misc_top20_cohensd_latent_utterances.py
& 'C:\Users\chengyu\AppData\Local\miniconda3\envs\qwen-env-py311\python.exe' run_misc_mapping_structure_filtered.py
& 'C:\Users\chengyu\AppData\Local\miniconda3\envs\qwen-env-py311\python.exe' run_misc_minimal_sufficient_subspace_v2.py
```

### 1.4 输出目录

| 分析 | 输出目录 |
|---|---|
| filtered label mapping | `outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered` |
| filtered Top20 utterances | `outputs/misc_full_sae_eval/interpretability/filtered_top20_cohensd_latent_utterances` |
| filtered mapping structure | `outputs/misc_full_sae_eval/interpretability/mapping_structure_filtered` |
| filtered minimal sufficient subspace | `outputs/misc_full_sae_eval/interpretability/minimal_sufficient_subspace_v2_filtered` |

旧目录 `minimal_sufficient_subspace_v2`、`mapping_structure`、`misc_label_mapping` 未作为本次 filtered 结论来源。

## 2. 数据与候选池校验

| 校验项 | 结果 |
|---|---:|
| records / samples | 6194 |
| original SAE latents | 32768 |
| keep=True latents | 12758 |
| dropped latents | 20010 |
| keep rate | 0.389 |
| labels | 9 |
| filtered metric rows expected | 114822 |
| filtered metric rows actual | 114822 |
| matrix latent set equals keep=True | true |
| minimal candidate rows | 900 |
| minimal candidate max per label | 100 |
| all minimal candidates keep=True | true |

Drop reason counts:

| reason | count |
|---|---:|
| rarely_active | 19958 |
| zero_or_near_zero_variance | 10606 |
| almost_always_active | 52 |

## 3. Mapping Structure 结果

### 3.1 Top20 多对多结构

| 指标 | 数值 |
|---|---:|
| analysis TopK | 20 |
| TopK latent-label edges | 180 |
| TopK unique latents | 131 |
| TopK single-label latents | 97 |
| TopK multi-label latents | 34 |
| multi-label share | 0.260 |

解释：filtered 初筛池中，每标签 Top20 仍不是一对一结构。180 条 label-latent 边压缩到 131 个 unique latents，其中 34 个 latent 进入多个标签 Top20。该结果支持“候选子空间是多对多映射”的表述。

### 3.2 Latent role taxonomy

| role | latents | share | mean label count | max abs Cohen's d |
|---|---:|---:|---:|---:|
| exclusive | 97 | 0.740 | 1.000 | 2.237 |
| family_shared | 26 | 0.198 | 2.077 | 1.461 |
| cross_family | 6 | 0.046 | 3.167 | 1.762 |
| global | 2 | 0.015 | 5.000 | 2.584 |

解释：exclusive latent 是多数，但 shared latent 仍足够多，特别是 QU/QUO、RE/REC 这类层级或语义邻近标签有明显重叠。

### 3.3 各标签 Top20 fragmentation

| label | positive | negative | shared | exclusive | shared ratio | top latent | top abs d | top directional AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QU | 20 | 0 | 19 | 1 | 0.950 | 13430 | 2.584 | 0.925 |
| RE | 20 | 0 | 15 | 5 | 0.750 | 29759 | 0.847 | 0.694 |
| QUO | 20 | 0 | 14 | 6 | 0.700 | 9959 | 1.762 | 0.800 |
| REC | 20 | 0 | 13 | 7 | 0.650 | 31133 | 0.965 | 0.611 |
| RES | 9 | 11 | 9 | 11 | 0.450 | 20808 | 0.701 | 0.626 |
| QUC | 20 | 0 | 7 | 13 | 0.350 | 21935 | 1.461 | 0.740 |
| GI | 15 | 5 | 5 | 15 | 0.250 | 13430 | 0.698 | 0.682 |
| SU | 20 | 0 | 1 | 19 | 0.050 | 24760 | 1.063 | 0.571 |
| AF | 20 | 0 | 0 | 20 | 0.000 | 23464 | 2.237 | 0.800 |

解释：

- `QU` 与 `QUO` 的 Top20 共享比例最高，符合父子标签结构。
- `RE` 与 `REC` 共享比例也较高，说明复杂反映与反映父类候选子空间紧密相关。
- `AF` 在 Top20 上完全 exclusive，表现为更清晰的局部候选组。
- `RES` 的 Top20 中负向 boundary latent 较多，说明简单反映的 Top20 统计边界并不全是正向语义 exemplar。

### 3.4 标签对 Top20 Jaccard

| label pair | Top20 Jaccard |
|---|---:|
| QU - QUO | 0.538 |
| RE - REC | 0.481 |
| QU - QUC | 0.212 |
| RES - QUO | 0.176 |
| RES - QU | 0.176 |
| QUO - GI | 0.143 |
| QU - GI | 0.143 |
| RES - GI | 0.111 |
| RE - RES | 0.081 |
| RES - REC | 0.053 |

解释：最高重叠主要出现在父子或同族标签中。跨族共享存在，但弱于 QU/QUO 与 RE/REC。

## 4. Minimal Sufficient Subspace 结果

### 4.1 汇总表

| label | role | status | class | full AUC | minimal K median | stability Jaccard | predictive redundancy | final K |
|---|---|---|---|---:|---:|---:|---:|---:|
| RE | parent_consistency_only | minimal_sufficient_found | parent_consistency_only | 0.898 | 19 | 0.204 | 0.526 | 19 |
| RES | leaf_or_atomic | minimal_sufficient_found | distributed | 0.862 | 21 | 0.166 | 0.429 | 21 |
| REC | leaf_or_atomic | minimal_sufficient_found | distributed | 0.907 | 14 | 0.189 | 0.143 | 14 |
| QU | parent_consistency_only | minimal_sufficient_found | parent_consistency_only | 0.975 | 6 | 0.474 | 0.167 | 6 |
| QUO | leaf_or_atomic | minimal_sufficient_found | distributed | 0.961 | 10 | 0.470 | 0.100 | 10 |
| QUC | leaf_or_atomic | minimal_sufficient_found | distributed | 0.927 | 15 | 0.352 | 0.133 | 15 |
| GI | leaf_or_atomic | minimal_sufficient_found | distributed | 0.851 | 17 | 0.282 | 0.059 | 17 |
| SU | leaf_or_atomic | minimal_sufficient_found | distributed | 0.895 | 24 | 0.145 | 0.208 | 24 |
| AF | leaf_or_atomic | minimal_sufficient_found | distributed | 0.930 | 12 | 0.245 | 0.083 | 12 |

### 4.2 Leaf 标签解释

| leaf label | minimal K median | full AUC | 解释 |
|---|---:|---:|---|
| QUO | 10 | 0.961 | 开放式问题的预测信号最强之一，子空间相对较紧凑，但仍不是单 latent 充分。 |
| AF | 12 | 0.930 | 肯定类行为有清晰强 latent 入口，但完整预测仍需约 12 个 latent。 |
| REC | 14 | 0.907 | 复杂反映需要多 latent 组合，且与 RE 父类有明显 overlap。 |
| QUC | 15 | 0.927 | 封闭式问题信号强，但充分表达仍为分布式子空间。 |
| GI | 17 | 0.851 | 信息给予可恢复，但最小充分子空间较分散。 |
| RES | 21 | 0.862 | 简单反映在 filtered pool 中需要较多 latent，且 Top20 边界含负向 latent。 |
| SU | 24 | 0.895 | 支持类行为在本口径下需要最多 latent，fold 稳定性也最低。 |

主结论：在 filtered 初筛候选池上，所有 leaf/atomic MISC 标签仍属于 distributed predictive subspace。即使 `QUO`、`AF` 这类较强标签，也需要约 10 个以上 latent 才能接近 Top100 full-candidate probe 表现。

### 4.3 与旧报告口径的关键差异

- 本次 minimal sufficient 不使用旧 `latent_space_search_v2` 的 stable-edge 或 thresholded seed。
- 所有候选均来自 filtered association per-label Top100。
- 因为候选池变化，`minimal_k_median` 不能与旧报告逐项直接视为同一实验复现；更准确的比较是“同一算法，在 filtered-only 候选池上的新结果”。
- `SU` 的 full AUC 从旧报告约 0.876 提升到本次 0.895，minimal K 中位数变为 24；这说明 filtered Top100 提供了更强 full probe 上限，但达到该上限需要更大的组合子空间。

## 5. Statistical Findings

| Metric | Test / Procedure | Value | Effect Size / Class | Confidence |
|---|---|---:|---|---|
| filtered metric matrix rows | deterministic row-count check | 114822 / 114822 | exact match | SOLID |
| keep pool membership | set inclusion audit | all candidates keep=True | 12758 keep latents | SOLID |
| mapping Top20 edges | TopK structural count | 180 | 9 labels x 20 | SOLID |
| mapping unique latents | TopK structural count | 131 | many-to-many evidence | SOLID |
| mapping multi-label latents | TopK structural count | 34 | 26.0% of Top20 unique latents | SOLID |
| highest label-pair overlap | Top20 Jaccard | QU-QUO = 0.538 | strong family overlap | SOLID |
| second label-pair overlap | Top20 Jaccard | RE-REC = 0.481 | strong family overlap | SOLID |
| leaf minimal K range | 5-fold greedy sufficiency | 10-24 | distributed subspace | SOLID |
| leaf full AUC range | 5-fold full-candidate probe | 0.851-0.961 | recoverable under filtered Top100 | SOLID |
| stability Jaccard range | fold selected-set Jaccard | 0.145-0.470 | unstable to moderately stable | CAUTION |

## 6. Warnings

| Type | Detail | Affected |
|---|---|---|
| Causal overclaim risk | 本实验是 association/probe-space sufficiency，不是 ablation、patching 或 steering。 | 所有 selected latents 与 minimal K |
| Multiple testing burden | filtered association 涉及 9 x 12758 = 114822 label-latent tests；已使用 FDR 字段，但后续解释仍应避免只挑显著结果叙事。 | label fragmentation / overlap |
| Candidate-pool dependence | minimal K 是 filtered Top100 候选池下的结果；换候选池会改变 full AUC 与 minimal K。 | minimal sufficient subspace |
| Fold instability | 多数 leaf 的 selected-set Jaccard < 0.40，说明具体 latent 组合不稳定，结论应偏向“分布式子空间”而非固定单组机制。 | RES, REC, QUC, GI, SU, AF |
| Boundary latent interpretation | RES Top20 中有 11 个负向 latent；这些 latent 更适合解释为边界/反向判别信号，不宜作为 RES 正向语义 exemplar。 | RES |

## 7. Fallacy Scan

Coverage: 11/11 checked.

| Fallacy | Severity | Detail | Recommendation |
|---|---|---|---|
| Simpson's Paradox | NOTE | 当前报告未按额外 grouping variable 分层，无法检测总体-分组方向反转。 | 若后续按 high/low、file_id 或模型分层，应重新检查方向一致性。 |
| Ecological Fallacy | NOTE | 分析单位是 utterance-level rows 与 latent activations，没有从群体均值推个体心理属性。 | 保持结论在 utterance / representation 层面。 |
| Berkson's Paradox | CAUTION | filtered pool 是经质量规则筛选后的 latent 子集，选择过程可能改变 overlap 与 minimal K。 | 明确写作 “filtered candidate pool under FeatureFilterConfig”。 |
| Collider Bias | NOTE | 本分析未加入控制变量模型，不涉及控制共同后果变量。 | 若后续加协变量 probe，需要重新评估。 |
| Base Rate Neglect | CAUTION | 标签 prevalence 差异明显，SU/AF 等低基率标签的 Precision@K 和 AUPRC 解释必须带 prevalence 背景。 | 报告 precision lift 而非只报告 precision。 |
| Regression to the Mean | NOTE | 没有 pre-post 或基于极端值入组的设计。 | 不适用。 |
| Survivorship Bias | NOTE | 没有 longitudinal attrition；但 feature filter 会排除 rarely-active/dead latent。 | 将 filter 视为候选池定义，不外推到全 SAE 空间。 |
| Look-Elsewhere Effect | CAUTION | 114822 个 label-latent 统计检验和多个 TopK 视角存在探索性选择风险。 | 使用 FDR、固定 TopK，并报告完整输出路径。 |
| Garden of Forking Paths | CAUTION | TopK、minimal sufficiency 阈值和 filter thresholds 是分析选择；本报告固定这些参数但不是预注册实验。 | 在论文中列明参数，避免事后调整叙事。 |
| Correlation != Causation | CAUTION | Cohen's d、AUC、minimal K 均为预测/关联证据。 | 不写成 causal mechanism；因果声明需另做 ablation/steering。 |
| Reverse Causality | NOTE | 这里不是时间因果模型，reverse causality 不直接适用。 | 若后续将 latent 激活解释为行为生成原因，需要因果干预支持。 |

## 8. Reproducibility

- Method: re-run environment-sensitive code experiment plus artifact/invariant checks.
- Verdict: PARTIALLY_REPRODUCIBLE.
- Reason: 生产命令已在同一工作区用 `qwen-env-py311` 执行并通过结构验收；未进行旧输出与新输出的 byte-for-byte diff，因此不标记为完全 deterministic reproducible。

| Metric | Expected | Observed | Status |
|---|---:|---:|---|
| filtered metric rows | 114822 | 114822 | MATCH |
| keep=True latents | 12758 | 12758 | MATCH |
| minimal labels | 9 | 9 | MATCH |
| minimal candidate max per label | <=100 | 100 | MATCH |
| all minimal candidates keep=True | true | true | MATCH |
| mapping Top20 edges | 180 | 180 | MATCH |
| mapping unique latents | recorded | 131 | ANALYZED |
| mapping multi-label latents | recorded | 34 | ANALYZED |

## 9. 固定产物索引

| Artifact | Path |
|---|---|
| filtered association | `outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered/latent_label_matrix.csv` |
| filtered behavior asymmetry | `outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered/behavior_asymmetry.md` |
| mapping structure report | `outputs/misc_full_sae_eval/interpretability/mapping_structure_filtered/mapping_structure_report.md` |
| mapping metrics | `outputs/misc_full_sae_eval/interpretability/mapping_structure_filtered/mapping_structure_metrics.json` |
| label fragmentation | `outputs/misc_full_sae_eval/interpretability/mapping_structure_filtered/label_fragmentation_rank.csv` |
| label pair similarity | `outputs/misc_full_sae_eval/interpretability/mapping_structure_filtered/label_pair_similarity.csv` |
| latent role summary | `outputs/misc_full_sae_eval/interpretability/mapping_structure_filtered/latent_role_summary.csv` |
| minimal sufficient report | `outputs/misc_full_sae_eval/interpretability/minimal_sufficient_subspace_v2_filtered/minimal_sufficient_subspace_report.md` |
| minimal sufficient summary | `outputs/misc_full_sae_eval/interpretability/minimal_sufficient_subspace_v2_filtered/minimal_sufficient_summary_v2.csv` |
| minimal candidate pool | `outputs/misc_full_sae_eval/interpretability/minimal_sufficient_subspace_v2_filtered/candidate_pool_v2.csv` |
| minimal summary JSON | `outputs/misc_full_sae_eval/interpretability/minimal_sufficient_subspace_v2_filtered/minimal_sufficient_summary.json` |
| top20 utterances | `outputs/misc_full_sae_eval/interpretability/filtered_top20_cohensd_latent_utterances/top20_utterances_by_top20_cohensd_latents.csv` |

## 10. 可写入论文的审慎表述

在 filtered 初筛候选池上，MISC 行为标签与 SAE latent 呈现多对多映射：每标签 Top20 候选共 180 条 label-latent 边，去重后为 131 个 latent，其中 34 个进入多个标签 Top20。进一步的 5-fold predictive sufficiency 分析显示，所有 leaf/atomic 标签均可由 filtered Top100 候选池恢复到较高 full-candidate AUC，但达到接近完整候选池表现通常需要 10-24 个 latent。该结果支持 MISC 标签在 SAE 空间中主要表现为 distributed predictive subspaces，而不是一对一 latent 映射。

不建议写作：这些 selected latents 已经证明是因果机制。更准确的写法是：这些 filtered-pool 子空间是后续 ablation、steering 与人工语义审查的优先候选。
