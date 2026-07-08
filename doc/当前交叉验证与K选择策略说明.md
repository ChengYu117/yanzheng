# 当前交叉验证与 K 选择策略说明

生成日期：2026-07-07

本文档说明当前项目中 SAE latent（稀疏自编码器的一个激活维度，可理解为模型内部的一类候选语义/行为特征）交叉验证、AUC@K 曲线（只使用前 K 个 latent 训练 probe 后得到的验证集 AUC）、每标签 K 选择，以及 `stable_core` latent set（最终用于后续解释的稳定 latent 集合）的生成策略。当前版本的主目标是：在 filtered SAE latent pool（经过激活频率、方差等质量过滤后保留下来的 latent 候选池）中，为每个 MISC 标签（心理咨询行为编码标签）找到一组与 MI code（动机性访谈行为编码）稳定相关的 latent，作为后续解释和 P3 流程（后续 latent 文本证据、功能卡片和人工审查流程）的主输入。

## 1. 当前总口径

当前流程对每个标签单独选择 `K*`（该标签最终保留多少个候选 latent 的预算）。`K*` 先由 AUC@K 曲线确定性能平台（AUC 增长基本变平的最小 K），再用 repeated split-half（多次随机二分数据）检查 TopK 集合稳定性。

最终主 latent set 使用 `stable_core`：

- `stable_core` 只要求 latent 在 full-data TopK*（全量数据按排序指标得到的前 K* 个 latent）内反复出现，并且 positive Cohen's d（正向标准化均值差，表示目标标签样本中的激活更高）的 bootstrap CI（自举置信区间）支持为正。
- `cross-quality`（high/low 数据源之间的稳健性检查）和 label-level stability status（整个标签 TopK 集合是否稳定的状态）保留为审计字段，默认不是进入 `stable_core` 的硬门槛。
- `boundary_candidate`（边界候选，纳入频率中等但 CI 仍支持为正）只是审计标签，不作为当前后续主 latent set。

当前主排序指标是 positive `cohens_d`（Cohen's d 为正时才认为 latent 与目标标签正相关）。`directional_auc`（把 AUC 和 `1-AUC` 中较大者作为方向无关区分度）仍然在 E0/E1 中输出，但只作为 sensitivity 检查（敏感性分析，用来确认结论是否依赖某个单一排序指标）。

## 2. 输入数据与候选池

主要输入：

- SAE 特征：`outputs/misc_full_sae_eval/feature_store/utterance_features.pt`
- 标签矩阵：`outputs/misc_full_sae_eval/label_matrix.csv`
- filtered latent pool 审计：`outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered/feature_filter_audit.csv`
- full-data filtered association matrix：`outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered/latent_label_matrix.csv`

当前正式结果使用 filtered latent pool。E1/E2/E3 的 manifest（一次实验运行的配置和产物清单）显示共有 `6194` 条样本、`12758` 个 filtered latents。

## 3. E0: AUC@K 曲线

E0 的作用是回答：对某个标签，只使用按训练折内 positive `cohens_d` 排序的前 K 个 latent，线性 probe（简单线性分类器，用来测试 latent 子空间中是否含有标签信息）的 held-out AUC（未参与训练的验证折 AUC）会在多大的 K 附近进入平台？

方法：

1. 使用 `stratified-group-kfold` 做 5 折交叉验证；它同时保持标签正负例比例尽量接近，并让同一 group 不跨 train/test。
2. 分组列为 `file_id`，避免同一文件内样本泄漏到 train/test 两侧；这里的泄漏指相似或重复语境同时出现在训练和验证中，导致性能偏高。
3. 对每个 label（一个 MISC 行为标签）、每个 fold（交叉验证中的一折），仅在训练折内重新计算 latent 与标签的关联指标。
4. 在训练折内按 `cohens_d` 或 `directional_auc` 排序，取 `K=0..100`；TopK 指排序前 K 个 latent。
5. 用训练折的 TopK latent 训练 logistic linear probe（逻辑回归线性探针），在 held-out fold 上计算 AUC。
6. `K=0` 作为无 latent baseline（没有任何 latent 信息的基线），AUC 记为 `0.5`；AUC=0.5 约等于随机排序，AUC 越接近 1 表示区分越好。

关键点：TopK 不是先从全量数据中定好再验证，而是在每个训练折内重新排序、重新取 TopK。

输出：

- `outputs/misc_full_sae_eval/interpretability/ranked_sae_subspace_probe_k001_100/auc_by_k_curve_0_100.csv`
- `outputs/misc_full_sae_eval/interpretability/ranked_sae_subspace_probe_k001_100/figures/macro_auc_vs_k_0_100.png`
- `outputs/misc_full_sae_eval/interpretability/ranked_sae_subspace_probe_k001_100/figures/label_auc_vs_k_0_100.png`

## 4. E1: Repeated Split-Half TopK 稳定性

E1 的作用是回答：在不同随机二分数据下，同一个标签的 TopK latent set（前 K 个 latent 构成的集合）是否会反复出现？

方法：

1. 按 `source_file` 分组，把数据随机二分为 Split A 和 Split B；`source_file` 通常对应一个会话/来源文件，用它分组可以降低同源文本泄漏。
2. 重复二分 `50` 次，随机种子为 `42..91`。
3. 每次二分中，Split A 和 Split B 各自独立计算 association matrix（label-latent 关联矩阵，记录每个标签和每个 latent 的 AUC、Cohen's d 等）。
4. 对每个 label、每个 ranking metric（排序指标）、每个 `K=1..100`，分别取两半的 TopK latent set。
5. 计算两半 TopK set 的 Jaccard overlap；Jaccard 是两个集合交集除以并集，越高表示两半选出的 latent 越相似。
6. 用随机 TopK 的 null distribution（随机抽 latent 时应有的重叠分布）做对照，记录 `passes_null_2sd_rate`。
7. 同时统计每个 latent 在 50 次 split-half 的 100 个 half-runs（每次二分产生 A/B 两个半数据运行）中进入 TopK 的频率，即 `inclusion_frequency`。

用于 K 选择的两个稳定条件：

- `passes_null_2sd_rate >= 0.95`：50 次重复中，至少 95% 的 observed Jaccard 要高于随机 null 均值加 2 个标准差。
- `observed_jaccard_p05 >= 0.40`：50 次重复的 Jaccard 第 5 百分位数至少为 0.40，表示较差随机划分下仍有基本集合重叠。

输出：

- `outputs/cross_val/topk_reproducibility/repeated_split_topk_grid_summary.csv`
- `outputs/cross_val/topk_reproducibility/topk_inclusion_frequency.csv`
- `outputs/cross_val/topk_reproducibility/figures/topk_jaccard_vs_k.png`
- `outputs/cross_val/topk_reproducibility/figures/topk_stable_count_vs_k.png`

## 5. E2: Grouped Bootstrap CI

E2 的作用是回答：某个 label-latent（某标签与某 latent 的一条关联边）的 positive Cohen's d 是否不仅是点估计为正，而且在 bootstrap 置信区间上也支持为正？

方法：

1. 对每个标签取 full-data filtered association matrix 中 positive `cohens_d` 的 Top100 latents。
2. 以 `source_file` 为 bootstrap group 做有放回重采样；grouped bootstrap 表示一次抽取整个来源文件，而不是单条 utterance，以保留文件内相关性。
3. 当前运行使用 `n_bootstrap=2000`，即重复 2000 次自举重采样。
4. 每次 bootstrap 中重新计算该 latent 的 AUC 和 Cohen's d。
5. 输出 AUC CI 与 Cohen's d CI；CI 是置信区间，表示估计值在重采样下的不确定性范围。

进入 `stable_core` 的 CI 条件是：

```text
cohens_d_ci_lo > 0
```

输出：

- `outputs/cross_val/bootstrap_ci/bootstrap_ci_by_label_latent.csv`

## 6. E3: Cross-Quality Validation

E3 的作用是检查 high/low source split（高质量来源与低质量来源数据划分）之间是否存在明显质量源混淆。当前它是审计信息，不是 `stable_core` 的默认硬门槛。

方法：

1. 将数据按 `source_split` 分为 high 与 low。
2. 在 high、low 和 full reference 中取 positive `cohens_d` Top100。
3. 比较同一 latent 在另一个 split 中的 directional AUC；这里关注“在一个来源上强的 latent，到另一个来源是否仍然有区分度”。
4. 输出 `auc_cross_drop`、`stable_cross_quality` 等字段；`auc_cross_drop` 是跨来源后的 AUC 下降，`stable_cross_quality` 表示下降是否低于预设风险阈值。

输出：

- `outputs/cross_val/cross_quality_validation/cross_quality_auc_comparison.csv`
- `outputs/cross_val/cross_quality_validation/cross_quality_summary.csv`

## 7. K 选择规则

每个 label 单独选择 `K*`，不使用全局统一 K。这样做是因为不同 MISC 标签的信息集中程度不同：有些标签少量 latent 就进入平台，有些标签需要更大的候选预算。

第一步，从 AUC@K 曲线选择 `K_auc`：

```text
best_auc = max AUC(k), k <= 100
best_k = best_auc 对应的最小 k
best_auc_se = auc_std(best_k) / sqrt(n_folds)
target_auc = best_auc - max(0.01, best_auc_se)
K_auc = 最小的 k，使 AUC(k) >= target_auc
```

其中，`best_auc` 是 0..100 内最好的平均 AUC，`best_k` 是达到该 AUC 的最小 K，`auc_std` 是不同 fold 上 AUC 的标准差，`best_auc_se` 是 best AUC 的标准误，`target_auc` 是“足够接近最优”的平台阈值，`K_auc` 是达到该平台阈值的最小 K。

这一步寻找“已经接近最佳 AUC 的最小 K”，避免为了很小的 AUC 增益引入过多 latent。

第二步，从 repeated split-half 选择 `K_stab`：

```text
K_stab = 最小的 k，满足：
  k >= K_auc
  k <= 100
  passes_null_2sd_rate >= 0.95
  observed_jaccard_p05 >= 0.40
```

其中，`K_stab` 是集合稳定性达标的最小 K。它要求 K 不能小于 `K_auc`，因为最终候选集至少要保留接近最佳预测性能所需的 latent 数量。

第三步，确定最终 `K*`：

```text
如果存在 K_stab:
  K* = K_stab
  selection_status = stable_topk_found
否则:
  K* = K_auc
  selection_status = performance_only_unstable
```

`performance_only_unstable` 表示 AUC 平台可以确定，但在当前阈值下没有找到 label-level TopK set 的稳定平台。它不等于“没有稳定单个 latent”，只表示整个 TopK 集合的成员替换较多。

## 8. Stable Core 选择规则

确定 `K*` 后，对每个 label 从 full-data positive `cohens_d` TopK* 中筛选 `stable_core`。这里的 full-data 表示用全量样本重新计算最终排序；前面的交叉验证已经用于确定 K 和稳定性审计。

`stable_core` 条件：

```text
full_data_rank <= K*
inclusion_frequency >= 0.70
cohens_d_ci_lo > 0
```

解释：`full_data_rank <= K*` 表示 latent 位于该标签最终候选预算内；`inclusion_frequency >= 0.70` 表示它在 100 个 half-runs 中至少 70% 被选入 TopK*；`cohens_d_ci_lo > 0` 表示 Cohen's d 置信区间下界仍为正，正向效应有统计支持。

`boundary_candidate` 条件：

```text
full_data_rank <= K*
0.40 <= inclusion_frequency < 0.70
cohens_d_ci_lo > 0
```

解释：`boundary_candidate` 与 `stable_core` 一样要求排名在 TopK* 内且 CI 为正，但纳入频率只有 40% 到 70%，因此只作为边界审计，不作为当前主 latent set。

其他常见状态：

- `ci_not_supported`：Cohen's d bootstrap CI 下界不大于 0。
- `unstable_inclusion`：纳入频率低于 `0.40`。

当前后续主流程使用 `stable_core`，而不是 `stable_core + boundary_candidate`。

## 9. 当前 K 选择结果

当前结果来自：

- `outputs/cross_val/stable_topk_selection/stable_k_by_label.csv`
- `outputs/cross_val/stable_topk_selection/stable_topk_latent_set.csv`
- `outputs/cross_val/stable_topk_selection/stable_topk_global_union.csv`

表中 `K_auc` 是 AUC 平台给出的最小 K，`K_stab` 是 repeated split-half 稳定平台给出的最小 K，`K*` 是最终采用的 K；`status` 是标签级选择状态，`stable_core` 和 `boundary_candidate` 是 label-latent 行数。

| label | K_auc | K_stab | K* | status | stable_core | boundary_candidate |
|---|---:|---:|---:|---|---:|---:|
| RE | 66 | 66 | 66 | `stable_topk_found` | 54 | 12 |
| RES | 45 | - | 45 | `performance_only_unstable` | 15 | 16 |
| REC | 56 | 56 | 56 | `stable_topk_found` | 47 | 9 |
| QU | 28 | 28 | 28 | `stable_topk_found` | 26 | 2 |
| QUO | 36 | 36 | 36 | `stable_topk_found` | 32 | 4 |
| QUC | 26 | 59 | 59 | `stable_topk_found` | 45 | 14 |
| GI | 72 | - | 72 | `performance_only_unstable` | 26 | 37 |
| SU | 60 | - | 60 | `performance_only_unstable` | 31 | 28 |
| AF | 33 | 33 | 33 | `stable_topk_found` | 27 | 6 |

汇总：

- label-latent 层面的 `stable_core`：303 行。
- label-latent 层面的 `boundary_candidate`：128 行。
- 去重后的 `stable_core` latent：225 个。

## 10. 指标分工

`Cohen's d` 与 AUC 在当前流程中承担不同职责：

- `Cohen's d` 用于单个 latent 的效应方向和效应大小排序，回答“这个 latent 是否在目标标签样本中更强激活”。
- `AUC@K` 用于评估 TopK latent 子空间的线性可解码信息量，回答“前 K 个 latent 合起来是否足够支持标签识别，并在多大 K 附近进入平台”。
- `inclusion_frequency` 用于判断 latent 是否在不同随机二分下反复被选中。
- `cohens_d_ci_lo > 0` 用于判断该 latent 的正向效应是否有 bootstrap 统计支持。

因此，当前策略不是用 AUC 直接挑单个 latent，也不是只看 Cohen's d 点估计。AUC 负责定每个标签的候选预算 `K*`，Cohen's d 与 repeated split inclusion 负责确定最终进入 `stable_core` 的具体 latent。

## 11. 复现命令

常用执行顺序如下：

```powershell
python run_misc_full_representation_probe.py `
  --include-sae-ranked-subspaces `
  --subspace-max-n 100 `
  --subspace-step 1 `
  --subspace-rankings cohens_d directional_auc `
  --subspace-n-jobs 8

python run_cross_val_topk_reproducibility.py `
  --top-k-grid-max 100 `
  --n-repeats 50

python run_cross_val_bootstrap_ci.py `
  --top-k 100 `
  --n-bootstrap 2000

python run_cross_val_quality_split.py `
  --top-k 100

python run_cross_val_stable_topk_selection.py
```

## 12. 报告表述边界

当前实验支持的结论是：这些 SAE latent 与 MISC/MI code 存在稳定统计关联，并且在 repeated split-half 与 bootstrap CI 下具备一定复现性。

当前实验不支持直接写成：

- 单个 latent 等同于某个临床或咨询概念。
- latent 对行为标签具有因果机制。
- `performance_only_unstable` 标签没有可用 latent 表征。

后续若要提出更强机制性结论，需要结合 latent 激活文本审查、P3 feature cards、ablation 或 steering 等额外证据。
