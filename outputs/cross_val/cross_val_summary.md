# AUC / Cohen's d 交叉验证实验汇总

## 运行口径

- Candidate pool: `misc_label_mapping_filtered` 的 `keep=True` latents，共 12758 个。
- E1 repeated split-half: 按 `source_file` 分组二分，使用 50 个随机种子重复计算 split A/B 的 filtered label-latent matrix。
- E2 bootstrap CI: 按 `source_file` grouped bootstrap，Top100 positive Cohen's d latents per label，2000 次。
- E3 cross-quality: high -> low 与 low -> high，Top20 positive Cohen's d latents per label。

## E1 repeated split-half Top20 stability

所有 label / metric 在 50 次 repeated split-half 中均通过随机 Top20 null 检验（`passes_null_2sd_rate = 1.000`）。下表报告稳定性分布，而不是单次 split 的点估计。

| label | ranking_metric | Jaccard mean | Jaccard p05..p95 | Spearman mean | rho>=0.60 rate | full Top20 union coverage mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AF | cohens_d | 0.573 | 0.379..0.667 | 0.503 | 0.000 | 0.967 |
| AF | directional_auc | 0.523 | 0.429..0.600 | 0.518 | 0.000 | 0.998 |
| GI | cohens_d | 0.192 | 0.095..0.290 | 0.510 | 0.000 | 0.921 |
| GI | directional_auc | 0.418 | 0.310..0.538 | 0.444 | 0.000 | 0.973 |
| QU | cohens_d | 0.800 | 0.739..0.818 | 0.657 | 1.000 | 0.995 |
| QU | directional_auc | 0.799 | 0.739..0.905 | 0.567 | 0.000 | 1.000 |
| QUC | cohens_d | 0.578 | 0.401..0.739 | 0.485 | 0.000 | 0.990 |
| QUC | directional_auc | 0.637 | 0.481..0.739 | 0.450 | 0.000 | 0.996 |
| QUO | cohens_d | 0.831 | 0.739..0.957 | 0.607 | 0.640 | 0.996 |
| QUO | directional_auc | 0.852 | 0.739..0.905 | 0.531 | 0.000 | 0.998 |
| RE | cohens_d | 0.490 | 0.333..0.637 | 0.669 | 1.000 | 0.976 |
| RE | directional_auc | 0.766 | 0.600..0.905 | 0.593 | 0.320 | 0.998 |
| REC | cohens_d | 0.424 | 0.310..0.538 | 0.689 | 1.000 | 0.989 |
| REC | directional_auc | 0.536 | 0.429..0.667 | 0.641 | 1.000 | 1.000 |
| RES | cohens_d | 0.223 | 0.143..0.314 | 0.423 | 0.000 | 0.911 |
| RES | directional_auc | 0.548 | 0.452..0.600 | 0.448 | 0.000 | 0.942 |
| SU | cohens_d | 0.184 | 0.026..0.379 | 0.512 | 0.000 | 0.956 |
| SU | directional_auc | 0.342 | 0.193..0.429 | 0.557 | 0.000 | 0.988 |

## E3 跨质量稳定性

| label | stable_fraction | mean_auc_cross_drop | max_auc_cross_drop | high_low_rank_spearman |
| --- | ---: | ---: | ---: | ---: |
| AF | 0.975 | 0.007 | 0.078 | 0.413 |
| GI | 0.925 | 0.018 | 0.089 | 0.409 |
| QU | 0.775 | 0.029 | 0.212 | 0.456 |
| QUC | 0.650 | 0.045 | 0.230 | 0.352 |
| QUO | 0.950 | 0.004 | 0.077 | 0.415 |
| RE | 0.825 | 0.013 | 0.070 | 0.442 |
| REC | 0.875 | 0.022 | 0.088 | 0.509 |
| RES | 0.850 | 0.015 | 0.117 | 0.278 |
| SU | 0.325 | 0.065 | 0.151 | 0.368 |

## CV candidate status counts

| status | count |
| --- | ---: |
| ci_not_available_or_includes_zero | 113927 |
| ci_supported_not_cross_quality_tested | 625 |
| reproducible_candidate | 199 |
| cross_quality_risk | 71 |

## Top100 bootstrapped rows by label/status

| label | ci_not_available_or_includes_zero | ci_supported_not_cross_quality_tested | cross_quality_risk | reproducible_candidate |
| --- | ---: | ---: | ---: | ---: |
| AF | 0 | 74 | 1 | 25 |
| GI | 0 | 64 | 1 | 35 |
| QU | 0 | 74 | 8 | 18 |
| QUC | 0 | 69 | 14 | 17 |
| QUO | 0 | 77 | 2 | 21 |
| RE | 0 | 71 | 7 | 22 |
| REC | 0 | 71 | 5 | 24 |
| RES | 5 | 65 | 6 | 24 |
| SU | 0 | 60 | 27 | 13 |

## 输出文件

- 合并矩阵: `outputs/cross_val/filtered_pool_association_matrix_with_cv.csv`
- E1: `outputs/cross_val/topk_reproducibility/`
- E2: `outputs/cross_val/bootstrap_ci/`
- E3: `outputs/cross_val/cross_quality_validation/`
