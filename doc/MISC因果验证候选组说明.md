# MISC Causal Candidate Groups

本阶段只做候选组整理，不执行模型干预实验。它把已经得到的 MISC latent-label 矩阵、Mapping Structure 角色分类和 latent case card 汇总为后续因果验证可直接读取的 G1/G5/G10/G20 latent 组。

## 输出文件

- 候选目录：`outputs\misc_full_sae_eval\interpretability\causal_candidates`
- `causal_candidate_groups.json`：每个标签的候选组、bottom control 和 random control。
- `candidate_group_summary.csv`：每个标签的组摘要。
- `label_candidates/<LABEL>_candidate_latents.csv`：每个标签的完整排序候选池。

## 选择口径

- 优先选择 FDR 显著且 `cohens_d > 0` 的 latent，代表该标签正向激活更高。
- 若某标签没有正向显著 latent，则回退为绝对效应排序，并在 `selection_direction` 中标记。
- 排序综合 `abs_cohens_d`、`directional_auc`、`precision@50`、case card 纯度和行为差异分析中的 top-positive 排名。
- 这些组仍是候选解释对象，不能直接写成因果结论；真正因果性需要后续 `causal/run_experiment.py` 进行 ablation/steering 验证。

## 标签候选组摘要

| Label | Pattern | Direction | Candidates | G20 | High-purity |
| --- | --- | --- | --- | --- | --- |
| RE | shared_distributed | positive | 2012 | 19435,29759,30224,20436,31930,10181,1211,22558,21800,26800,17861,16292,15068,28269,31133,3993,12852,26319,29874,4625 | 1 |
| RES | negative_boundary | positive | 99 | 20808,28269,11435,3801,6982,13948,1092,20776,22044,23201,24977,31315,1850,3145,5923,16546,29942,8313,22294,13966 | 0 |
| REC | shared_distributed | positive | 2592 | 20436,31133,26800,30224,19435,16292,3993,10181,29759,15068,21800,1211,3805,29874,12852,23670,22558,1455,19005,3673 | 1 |
| QU | compact_strong | positive | 1260 | 13430,664,9959,11660,26485,10916,12340,27061,29590,21125,21935,24943,24761,26144,24744,21859,22358,8658,16969,32508 | 5 |
| QUO | compact_strong | positive | 972 | 9959,26485,664,13430,24761,23183,24744,21125,27061,11660,21203,26144,18310,14833,32508,6129,19840,13311,3887,3459 | 1 |
| QUC | shared_distributed | positive | 540 | 14014,21935,20869,13430,22358,4998,18646,20463,12340,28816,12555,8969,2504,7037,16969,27857,12887,6639,736,9827 | 2 |
| GI | shared_distributed | positive | 774 | 16345,8294,26879,13751,16515,17827,21634,18490,26236,9893,24876,1713,17685,27515,8166,17556,7148,16131,5582,2055 | 4 |
| SU | negative_boundary | positive | 234 | 24760,29825,4756,16736,9720,11948,4512,7389,16190,19935,30223,9109,28603,28795,19359,7382,11872,9048,10539,22730 | 2 |
| AF | negative_boundary | positive | 342 | 23464,7143,30870,17793,18492,6676,28724,32764,2434,24206,28469,22411,9869,24856,22475,29908,25532,453,31149,19654 | 4 |

## 后续运行建议

- 第一轮建议优先跑 `RE`，因为它与既有 RE/NonRE 因果流程兼容。
- 第二轮建议跑 `QU` 或 `QUO`，因为它们在当前 SAE 空间中最紧凑，适合作为正例对照。
- `AF/SU/RES` 更像 negative-boundary 或稀疏边界信号，干预解释要更谨慎，优先观察 necessity 而不是强 sufficiency。
- 当前导出的组大小：`G1, G5, G10, G20`。