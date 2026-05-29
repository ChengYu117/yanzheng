# MISC Top20 矩阵生成结果评测报告

> 本报告采用新的研究口径：`Latent × Label matrix` 只作为候选排序来源，正式可解释性评估对象是每个核心 MISC 标签下排名前 20 的 SAE latent。

## 1. 结论

矩阵生成阶段已经完成，并且可以稳定导出每个标签的 Top20 候选 latent。后续 Mapping Structure、行为差异、case cards、AI/人工审查和因果候选组都应只围绕这些 Top20 候选展开，而不是解释整个 SAE latent space。

本次 Top20 子空间的核心结果：

| 指标 | 数值 |
|---|---:|
| 核心标签数 | 9 |
| 每标签候选上限 | 20 |
| Top20 latent-label 边 | 180 |
| Top20 去重 latent 数 | 136 |
| Top20 单标签 latent | 103 |
| Top20 多标签 latent | 33 |
| Top20 多标签占比 | 0.243 |

这些数值说明：即使只看每个标签最强的 20 个候选，也已经能观察到多标签共享结构，但共享比例不会被全量显著 latent 放大。

## 2. 矩阵的角色

全量 `latent_label_matrix.csv` 的作用是：

```text
全量 latent-label 统计扫描
  -> 每个标签按效应量、AUC、precision 排序
  -> 导出 top_latents_by_label/<LABEL>.csv
  -> 截取每个标签 Top20
  -> 形成 Mapping Structure / case cards / G1-G20 因果候选
```

因此，矩阵不是最终解释结论。正式写作中不应使用全量显著 latent 数作为可解释性证据。

## 3. Top20 标签结构

| Label | TopK | Pos. | Neg. | Shared | Exclusive | Shared Ratio | Top Latent | Top d | Top AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QU | 20 | 20 | 0 | 19 | 1 | 0.950 | 13430 | 2.584 | 0.925 |
| QUO | 20 | 20 | 0 | 14 | 6 | 0.700 | 9959 | 1.762 | 0.800 |
| RE | 20 | 20 | 0 | 14 | 6 | 0.700 | 29759 | 0.847 | 0.694 |
| REC | 20 | 20 | 0 | 13 | 7 | 0.650 | 31133 | 0.965 | 0.611 |
| QUC | 20 | 20 | 0 | 7 | 13 | 0.350 | 21935 | 1.461 | 0.740 |
| GI | 20 | 15 | 5 | 5 | 15 | 0.250 | 13430 | 0.698 | 0.682 |
| RES | 20 | 12 | 8 | 4 | 16 | 0.200 | 20808 | 0.701 | 0.626 |
| SU | 20 | 20 | 0 | 1 | 19 | 0.050 | 24760 | 1.063 | 0.571 |
| AF | 20 | 20 | 0 | 0 | 20 | 0.000 | 23464 | 2.237 | 0.800 |

评测判断：

- `QU/QUO/RE/REC` 的 Top20 中共享 latent 明显较多，是 Mapping Structure 的主要证据来源。
- `AF/SU/RES` 在 Top20 中更偏专属或边界型，不应再用全量显著数量描述其“高度碎片化”。
- `QU` 的 Top20 最强，AUC 和共享比例都高，适合作为结构恢复的正例。
- `RE` 的 Top20 同时包含专属与共享候选，适合衔接后续 RE 因果验证。

## 4. 标签间 Top20 重叠

| Label A | Label B | Top20 Jaccard |
|---|---|---:|
| QU | QUO | 0.538 |
| RE | REC | 0.481 |
| QU | QUC | 0.212 |
| QUO | GI | 0.143 |
| QU | GI | 0.143 |
| RES | GI | 0.111 |
| RES | QUO | 0.111 |
| RES | QU | 0.111 |

这说明父子标签关系在 Top20 SAE 子空间中得到部分恢复：`QU-QUO` 和 `RE-REC` 是最强的两组重叠。

## 5. 后续使用

后续正式分析只使用以下文件：

```text
outputs/misc_full_sae_eval/functional/misc_label_mapping/top_latents_by_label/
outputs/misc_full_sae_eval/interpretability/mapping_structure/topk_candidate_matrix.csv
outputs/misc_full_sae_eval/interpretability/followup_analysis/latent_cases/
outputs/misc_full_sae_eval/interpretability/causal_candidates/label_candidates/
```

写作口径应为：

> 我们先用全量矩阵完成候选排序，然后仅对每个标签 Top20 latent 子空间进行结构分析、案例解释和因果候选验证。
