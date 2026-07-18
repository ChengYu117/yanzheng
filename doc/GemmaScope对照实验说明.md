# Gemma3-4B + GemmaScope Layer-18 SAE 对照实验说明

> Status: Superseded
>
> Replaced by: `docs/current/experiment_workflow.md`
>
> Do not use this document for implementation or experiment decisions. The current experiment does not run Gemma.

本文档仅保留旧 Gemma 对照实验记录。

## 1. 实验口径

| 项目 | 当前固定值 |
|---|---|
| 基础模型 | `models/gemma-3-4b-pt`，对应 `google/gemma-3-4b-pt` |
| SAE repo | `google/gemma-scope-2-4b-pt` |
| SAE checkpoint | `resid_post_all/layer_18_width_16k_l0_small` |
| 主层 | `layer_idx=18`，0-based，第 19 个 decoder block |
| 选择原因 | 全量 MISC Gemma layer probe 中 `RE` 父标签最优层为 18 |
| 注意事项 | `QU` 父标签最优层为 10，因此本实验中的 QU 结论是 layer-18 主层下的稳健性观察 |
| 数据 | `data/mi_quality_counseling_misc` 全量 MISC，6194 条 |
| 标签 | `RE, RES, REC, QU, QUO, QUC, GI, SU, AF`，不把 `OTHER` 写入主结论 |
| 输出目录 | `outputs/gemma3_l18_gemmascope_sae_eval/` |

## 2. 代码入口

主评估入口：

```powershell
C:\Users\chengyu\AppData\Local\miniconda3\envs\qwen-env-py311\python.exe run_gemma_scope_sae_evaluation.py --layer-idx 18 --batch-size 2 --output-dir outputs/gemma3_l18_gemmascope_sae_eval
```

最小充分子空间：

```powershell
C:\Users\chengyu\AppData\Local\miniconda3\envs\qwen-env-py311\python.exe run_misc_minimal_sufficient_subspace_v2.py --feature-store outputs/gemma3_l18_gemmascope_sae_eval/feature_store/utterance_features.pt --label-matrix outputs/gemma3_l18_gemmascope_sae_eval/label_matrix.csv --output-dir outputs/gemma3_l18_gemmascope_sae_eval/interpretability/minimal_sufficient_subspace_v2
```

跨模型汇总：

```powershell
C:\Users\chengyu\AppData\Local\miniconda3\envs\qwen-env-py311\python.exe run_cross_model_sae_comparison.py --llama-root outputs/misc_full_sae_eval --gemma-root outputs/gemma3_l18_gemmascope_sae_eval --output-dir outputs/gemma3_l18_gemmascope_sae_eval/interpretability/model_specificity_comparison
```

闭环验收：

```powershell
C:\Users\chengyu\AppData\Local\miniconda3\envs\qwen-env-py311\python.exe run_gemma_scope_validation.py --root outputs/gemma3_l18_gemmascope_sae_eval
```

验收报告：

```text
outputs/gemma3_l18_gemmascope_sae_eval/validation/gemma_scope_validation_summary.json
outputs/gemma3_l18_gemmascope_sae_eval/validation/gemma_scope_validation_report.md
```

## 3. 生成文件

| 文件 / 目录 | 用途 |
|---|---|
| `metrics_structural.json` | GemmaScope SAE 重构与稀疏度指标 |
| `feature_store/utterance_features.pt` | utterance-level GemmaScope SAE latent features，形状 `[6194, 16384]` |
| `feature_store/utterance_activations.pt` | Gemma layer-18 raw hidden features，形状 `[6194, 2560]` |
| `functional/misc_label_mapping/latent_label_matrix.csv` | GemmaScope latent × MISC label 关联矩阵 |
| `interpretability/minimal_sufficient_subspace_v2/` | 每标签最小充分 latent 子空间 |
| `interpretability/baseline_comparison_step6/` | GemmaScope SAE vs PCA vs raw hidden 对照 |
| `interpretability/model_specificity_comparison/` | Llama SAE vs GemmaScope SAE 跨模型汇总 |
| `validation/` | artifact 完整性、shape、finite metric、标签覆盖和指标范围验收 |

## 4. 当前全量结果

### 4.1 结构指标

| 指标 | 数值 |
|---|---:|
| valid tokens | 97055 |
| invalid tokens skipped | 0 |
| explained variance | 0.9990 |
| cosine similarity | 0.9984 |
| MSE | 2145.5451 |
| MAE | 27.7285 |
| L0 mean | 22.8248 |
| L0 std | 16.8719 |

说明：最初使用 float16 SAE 解码时结构指标出现 NaN，原因是重构值溢出。当前正式结果已改为 float32 SAE 前向，并确认没有 invalid token 被跳过。

### 4.2 最小充分子空间结果

| 标签 | full-candidate AUC | median minimal K | 分档 |
|---|---:|---:|---|
| RE | 0.906 | 18 | parent consistency only |
| RES | 0.870 | 18 | distributed |
| REC | 0.918 | 15 | distributed |
| QU | 0.974 | 9 | parent consistency only |
| QUO | 0.961 | 14 | distributed |
| QUC | 0.919 | 10 | distributed |
| GI | 0.857 | 20 | distributed |
| SU | 0.921 | 17 | distributed |
| AF | 0.944 | 7 | moderate |

固定解释：GemmaScope layer-18 和 Llama 主结果一致地支持“多数 MISC leaf 标签不是由 1-3 个 latent 稳定充分表示，而是需要多 latent 子空间”。其中 `AF` 在 Gemma 中相对更紧，median K=7。

### 4.3 Step6 对照结果

| representation | mean label AUC | macro probe AUC | macro F1 | mean effective n | mean polysemanticity |
|---|---:|---:|---:|---:|---:|
| GemmaScope SAE latents | 0.768 | 0.917 | 0.594 | 1054.254 | 1.422 |
| PCA components | 0.762 | 0.931 | 0.664 | 364.122 | 2.000 |
| raw hidden dims | 0.751 | 0.896 | 0.547 | 1795.417 | 1.322 |

固定解释：GemmaScope SAE 的预测指标不总是优于 PCA，但它提供了原生稀疏 latent 空间，并在 label-level 结构分析中保留了更可解释的候选单位。PCA/raw hidden 仍作为 predictive control，不作为主要解释性结论。

## 5. 跨模型结论边界

当前跨模型比较显示：

1. `REC`、`QU`、`QUO` 在 Llama 与 Gemma 中都具有较清晰的可恢复预测信号。
2. `RES` 与 `GI` 在两个模型中都更分散，说明该现象不是单一模型特例。
3. `RE` 在 Gemma layer-18 与 Llama 主结果之间存在差异，应写作模型/层选择相关差异，不写成普遍机制。
4. minimal sufficient subspace 层面，两种模型都显示多标签需要分布式 latent 子空间，支持“非一对一映射”的稳健结论。
5. 本实验仍是相关性、结构性和预测充分性证据，不直接宣称因果机制。

## 6. 自动验收结果

`run_gemma_scope_validation.py` 已执行并通过，状态为 `PASSED`：

| 验收项 | 结果 |
|---|---:|
| checks | 24 |
| pass | 24 |
| warn | 0 |
| fail | 0 |

验收覆盖：

1. 根目录和 21 个核心 artifact 是否存在。
2. `utterance_features.pt` 与 `utterance_activations.pt` 的 shape 与 finite 值。
3. `label_matrix.csv` 是否覆盖 9 个核心标签且不含 `OTHER`。
4. structural metrics 是否 finite，且 `n_invalid_tokens_skipped=0`。
5. latent-label matrix 是否覆盖 9 标签、行数是否为 `16384 × 9`，AUC / precision 等指标是否在合法范围。
6. `fragmentation_v2`、`overlap_thresholded_v2`、`overlap_weighted_v2`、`polysemanticity_v2` 是否满足基本一致性与范围约束。
7. 最小充分子空间是否覆盖全部标签，recoverable 标签的 minimal K 是否不超过候选池。
8. Step6 对照是否包含 SAE / PCA / raw hidden 三类表征。
9. 跨模型汇总是否覆盖 Llama / Gemma 与全部核心标签。

本轮自动验收中曾发现两个脚本口径问题：`top20_jaccard` 的 family-union 不适用行存在空值，以及 `polysemanticity_v2` 当前列名为 `n_thresholded_labels/n_families`。这两个问题属于 validation 脚本误判，已修正后重跑通过。
