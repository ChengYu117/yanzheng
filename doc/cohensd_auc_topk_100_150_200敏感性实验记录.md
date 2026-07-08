# Cohen's d Top-K 上限敏感性实验记录

**日期**：2026-07-08  
**实验问题**：原 stable-core 流程中将 Cohen's d 排序的 AUC@K 搜索上限固定为 `K<=100`，是否会限制经验 AUC 上限？若扩展到 `K<=150` 或 `K<=200`，AUC 上限与 `K_auc` 是否保持一致？

---

## 1. 实验设计

本实验只检查 **AUC 上限敏感性**，不重新执行 split-half 稳定性筛选，也不重新生成最终 `stable_core`。

实验口径：

- 候选池：沿用原流程的 **filtered train-fold SAE latent pool**（训练折内激活质量过滤后的 SAE latent 候选池）。
- 排序指标：`Cohen's d`（标准化均值差，用于度量某 latent 在标签阳性样本与阴性样本之间的激活均值差异）。
- 交叉验证：`stratified-group-kfold`，`folds=5`，`group_column=file_id`。
- 原始基线：复用 `outputs/misc_full_sae_eval/interpretability/ranked_sae_subspace_probe_k001_100/auc_by_k_curve_0_100.csv` 中的 `K=0..100`。
- 增量计算：新增计算 `K=101..200`。
- 比较截断：`max_k=100`、`max_k=150`、`max_k=200`。

运行脚本：

```powershell
C:\Users\chengyu\AppData\Local\miniconda3\envs\qwen-env-py311\python.exe -u run_cohensd_auc_k_sensitivity.py --quiet
```

核心输出目录：

```text
outputs/misc_full_sae_eval/interpretability/cohensd_auc_k_sensitivity_100_150_200/
```

关键结果文件：

- `auc_by_k_0_200_combined.csv`：合并后的 `K=0..200` Cohen's d AUC@K 曲线。
- `auc_upper_bound_100_150_200.csv`：三个上限下的经验 best AUC 与 best K。
- `k_auc_rule_100_150_200.csv`：按正式 `K_auc` 规则重算的结果。

---

## 2. 判定规则

对每个标签和每个 `max_k`，先在 `K<=max_k` 范围内寻找最高平均 AUC：

```text
best_auc(max_k) = max AUC@K, K <= max_k
```

再沿用 stable-core 流程中的性能平台规则：

```text
target_auc = best_auc - max(0.01, best_auc_se)
```

其中 `best_auc_se`（最佳 AUC 标准误）为：

```text
best_auc_std / sqrt(n_folds)
```

然后选择最小的满足：

```text
AUC@K >= target_auc
```

的 K，作为 `K_auc`（性能平台截断点）。

---

## 3. AUC 上限比较

| 标签 | best AUC ≤100 | best K ≤100 | best AUC ≤150 | best K ≤150 | best AUC ≤200 | best K ≤200 | ΔAUC 150-100 | ΔAUC 200-100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AF | 0.925446 | 84 | 0.925446 | 84 | 0.925446 | 84 | 0.000000 | 0.000000 |
| GI | 0.765622 | 95 | 0.779260 | 143 | 0.779260 | 143 | 0.013638 | 0.013638 |
| QU | 0.975686 | 96 | 0.975686 | 96 | 0.975686 | 96 | 0.000000 | 0.000000 |
| QUC | 0.926130 | 95 | 0.926576 | 114 | 0.926576 | 114 | 0.000446 | 0.000446 |
| QUO | 0.959473 | 100 | 0.960874 | 115 | 0.960874 | 115 | 0.001401 | 0.001401 |
| RE | 0.858289 | 98 | 0.865225 | 150 | 0.867420 | 199 | 0.006936 | 0.009131 |
| REC | 0.910814 | 96 | 0.917286 | 150 | 0.917444 | 158 | 0.006472 | 0.006630 |
| RES | 0.793923 | 62 | 0.795255 | 127 | 0.795699 | 158 | 0.001332 | 0.001776 |
| SU | 0.859822 | 85 | 0.859822 | 85 | 0.868283 | 183 | 0.000000 | 0.008461 |

---

## 4. 正式 K_auc 规则下的变化

| 标签 | K_auc ≤100 | K_auc ≤150 | K_auc ≤200 | 解读 |
|---|---:|---:|---:|---|
| AF | 33 | 33 | 33 | 完全不受 100 上限影响。 |
| GI | 72 | 95 | 95 | 扩展到 150 后出现更高 AUC 平台，100 上限会低估性能平台位置。 |
| QU | 28 | 28 | 28 | 完全不受 100 上限影响。 |
| QUC | 26 | 26 | 26 | best AUC 微升，但平台 K 不变。 |
| QUO | 36 | 49 | 49 | best AUC 仅小幅提升，但 target_auc 随之提高，使平台 K 从 36 移到 49。 |
| RE | 66 | 79 | 98 | AUC 上限持续到接近 200 才达到最高，100 上限会截断 RE 的上升趋势。 |
| REC | 56 | 74 | 74 | 扩展后 best AUC 提升，平台 K 后移。 |
| RES | 45 | 48 | 50 | best AUC 小幅提升，平台 K 轻微后移。 |
| SU | 60 | 60 | 70 | 150 不变，但 200 出现更高 AUC 点，平台 K 后移。 |

---

## 5. 固定结论

本实验显示，`K<=100` 不是对所有标签都完全无影响。

可以固定以下结论：

1. **100 上限足够稳定的标签**：`AF`、`QU`、`QUC`。  
   这些标签在 `K<=150/200` 下的 best AUC 或 `K_auc` 基本不变。

2. **100 上限基本不改变 AUC 上限、但会轻微改变平台规则的标签**：`QUO`、`RES`。  
   `QUO` 的 best AUC 只增加 0.001401，但正式平台 K 从 36 变为 49；`RES` 的 best AUC 只增加 0.001776，平台 K 从 45 变为 50。

3. **100 上限确实可能截断 AUC 上升空间的标签**：`GI`、`RE`、`REC`、`SU`。  
   这些标签在 `K>100` 后仍出现更高 best AUC，其中 `GI` 增幅最大（+0.013638），`RE` 和 `SU` 也接近 0.01 的方法阈值。

4. **方法决策**：如果研究目标是保守估计“Cohen's d Top-K 子空间的 AUC 上限”，后续建议将 AUC@K 探索上限从 `100` 固定扩展到 `200`。  
   如果研究目标是控制解释卡片数量和保持稳定核心紧凑性，`100` 仍可作为工程成本约束，但需要在方法局限中声明：`GI/RE/REC/SU` 的 AUC 上限可能被低估。

5. **下一步 stage gate**：若要把 `max_k=200` 正式替代原 stable-core 流程，还需要同步重跑 `repeated split-half`、`bootstrap CI` 与 `cross-quality validation` 的 `top_k_grid/top_k` 到 200；仅本实验不足以直接更新最终 `stable_core`，它只固定了 AUC 上限敏感性证据。

