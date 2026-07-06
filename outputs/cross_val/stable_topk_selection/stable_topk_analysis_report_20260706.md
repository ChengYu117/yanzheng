# Stable Top-K Latent Set 与简化 Stable Core 分析报告

生成日期：2026-07-06

## 1. 口径调整

原先的 `stable_core` 规则偏保守：要求 full-data TopK*、反复出现、bootstrap CI 为正、cross-quality drop 小，并且标签整体通过 split-half 稳定平台。这个设计适合非常保守的论文主张，但会把很多有统计支持的 latent 排除掉。

现在改为简化口径：

- `stable_core`：`full_data_rank <= K*`，`inclusion_frequency >= 0.70`，且 `cohens_d_ci_lo > 0`。
- `boundary_candidate`：`full_data_rank <= K*`，`0.40 <= inclusion_frequency < 0.70`，且 `cohens_d_ci_lo > 0`。
- cross-quality 结果继续保留在 CSV 中作为风险审计字段，但不再作为进入 `stable_core` 的硬门槛。
- `selection_status=performance_only_unstable` 仍然保留，表示该标签整体 TopK 集合的 Jaccard 稳定平台没过；但它不再阻止单个 latent 按“反复出现 + CI 为正”进入 `stable_core`。

一句话：现在的 `stable_core` 只回答“这个 latent 是否在 TopK* 内反复出现，且正向 Cohen's d 有 bootstrap CI 支持”。

## 2. K* 选择仍保持不变

`K*` 仍按原流程确定：

1. 用 AUC@K 曲线找每个标签的性能平台 `K_auc`。
2. 从 `K >= K_auc` 中找满足 split-half Jaccard 稳定平台的 `K_stab`。
3. 若有 `K_stab`，则 `K* = K_stab`；否则 `K* = K_auc`，标签标记为 `performance_only_unstable`。

因此，简化的是 `stable_core` 的成员筛选规则，不是 AUC@K 和 K* 选择流程。

## 3. 当前结果

| label | K_auc | K_stab | K* | status | stable core | boundary | candidate rows |
|---|---:|---:|---:|---|---:|---:|---:|
| RE | 66 | 66 | 66 | stable_topk_found | 54 | 12 | 66 |
| RES | 45 | - | 45 | performance_only_unstable | 15 | 16 | 45 |
| REC | 56 | 56 | 56 | stable_topk_found | 47 | 9 | 56 |
| QU | 28 | 28 | 28 | stable_topk_found | 26 | 2 | 28 |
| QUO | 36 | 36 | 36 | stable_topk_found | 32 | 4 | 36 |
| QUC | 26 | 59 | 59 | stable_topk_found | 45 | 14 | 59 |
| GI | 72 | - | 72 | performance_only_unstable | 26 | 37 | 72 |
| SU | 60 | - | 60 | performance_only_unstable | 31 | 28 | 60 |
| AF | 33 | 33 | 33 | stable_topk_found | 27 | 6 | 33 |

总计：

- label-latent 层面的 `stable_core`：303 行。
- label-latent 层面的 `boundary_candidate`：128 行。
- 去重后的 `stable_core` latent：225 个。
- 去重后的 `stable_core + boundary_candidate` latent：335 个。

## 4. RES/GI/SU 的解释

按当前简化规则，`RES/GI/SU` 的结果应这样理解：

- 它们有不少 latent 满足“反复出现 + CI 为正”。
- 但它们的标签级 TopK 集合稳定性仍然不够，`selection_status` 仍为 `performance_only_unstable`。

这两个结论不矛盾：

- `stable_core_count > 0` 表示存在单个 latent 层面的稳定统计候选。
- `performance_only_unstable` 表示整个 TopK 集合在不同随机二分中替换较多，不宜声称该标签有一个强稳定的完整 TopK set。

因此推荐论文写法是：

- 主结果可以报告简化 `stable_core`，因为它直接对应“反复出现 + CI 为正”。
- 对 `RES/GI/SU` 额外标注：这些标签存在 individual stable candidates，但 label-level TopK set stability 不足。

## 5. 输出文件

主输出：

- `outputs/cross_val/stable_topk_selection/stable_k_by_label.csv`
- `outputs/cross_val/stable_topk_selection/stable_topk_latent_set.csv`
- `outputs/cross_val/stable_topk_selection/stable_topk_global_union.csv`
- `outputs/cross_val/stable_topk_selection/stable_topk_selection_report.md`

配套曲线与 split-half 结果：

- `outputs/misc_full_sae_eval/interpretability/ranked_sae_subspace_probe_k001_100/auc_by_k_curve_0_100.csv`
- `outputs/cross_val/topk_reproducibility/repeated_split_topk_grid_summary.csv`
- `outputs/cross_val/topk_reproducibility/topk_inclusion_frequency.csv`

## 6. 检查结果

已检查简化规则下的 stable/boundary 行：

- `stable_core` 均满足 `inclusion_frequency >= 0.70` 且 `cohens_d_ci_lo > 0`。
- `boundary_candidate` 均满足 `0.40 <= inclusion_frequency < 0.70` 且 `cohens_d_ci_lo > 0`。
- 没有发现违反简化规则的行。
