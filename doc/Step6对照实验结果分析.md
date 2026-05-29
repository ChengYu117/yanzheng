# Step 6 对照实验结果分析

## 1. 实验目的

本实验用于回答一个结构可解释性问题：在同一批 MISC 咨询师话语、同一组行为标签和同一层模型表征下，SAE latents 是否比 PCA components 和 raw hidden-state dimensions 暴露出更清晰、更稀疏、更符合 MISC 层级结构的标签-表征关系。

因此，本实验不是单纯的分类性能排行榜。分类性能只作为辅助指标，用来确认不同表征都保留了可预测的 MISC 行为信息；主要结论应来自结构指标，包括 compactness、association concentration、label overlap、polysemanticity 和 family hierarchy recovery。

## 2. 实验设置

- 数据：复用主分析中的 MISC utterance-level 表征和标签矩阵。
- 标签：`RE`, `REC`, `QU`, `QUO`, `GI`, `AF`, `SU`。
- 对照表征：
  - `sae_latents`：SAE latent activation，32768 维。
  - `pca_components`：在训练集 raw hidden states 上拟合 PCA，1024 维，解释方差比例为 0.896。
  - `raw_hidden_dims`：原始 hidden-state 坐标，4096 维。
- 特征-标签关联分数：`association_score = max(directional_auc - 0.5, 0)`。
- 阈值特征集合：`directional_auc >= 0.65`。
- 二级分类器：每个标签取训练集上关联最高的前 200 个特征，训练 one-vs-rest logistic regression。

注意：`RE-REC`、`QU-QUO` 在 MISC 语义上包含父子/同家族关系，因此在本实验中只作为 hierarchy consistency evidence，不作为“两个独立标签自然重叠”的发现来写。

## 3. 指标口径

### 3.1 Predictive Association

`Mean Label AUC` 是每个标签最强单一特征的 `top1_directional_auc` 的平均值。它衡量该表征空间中是否存在单个 feature/component/dimension 能够较强地区分某个标签。

该指标越高，说明该表征中更容易找到与标签强相关的单元。但它不能单独说明可解释性，因为 PCA component 或 raw hidden dimension 也可能有预测性，却未必是人类可解释的语义单元。

### 3.2 Secondary Classification

`Macro F1` 和 `Macro Probe AUC` 来自每个标签的二级 logistic probe。它们衡量使用该表征做监督分类时的预测能力。

这部分是辅助证据，不是主要判断依据。若 PCA 的分类性能更高，只能说明 PCA components 在监督分类中保留了强预测信息，不能直接说明它比 SAE 更可解释。

### 3.3 Compactness / Fragmentation

`Mean N_eff` 是基于所有特征的非负关联分数计算的有效特征数。直观上，它回答“一个标签的关联质量分散在多少个特征上”。

由于三种表征的维度不同，绝对 `N_eff` 不能直接比较。主比较应使用：

```text
N_eff / D
```

其中 `D` 是该表征的总维度。该值越低，说明标签关联集中在更小比例的特征中，结构更紧凑。

### 3.4 Association Concentration

`Concentration@20` 表示前 20 个特征捕获了多少总关联质量。由于 PCA 只有 1024 维，而 SAE 有 32768 维，固定 Top20 对 PCA 更有利。因此 `Concentration@20` 只能作为补充指标。

主比较采用：

```text
Concentration@1%
```

即每种表征取自身前 1% 特征，比较它们捕获的关联质量。该指标能更公平地比较不同维度的表征空间。

### 3.5 Label Overlap and Hierarchy Recovery

`Mean Label Overlap` 使用阈值特征集合的 Jaccard 相似度。它回答两个标签是否共享同一批强关联特征。

`Weighted Overlap` 使用完整 association profile 的 cosine similarity。它比 thresholded Jaccard 更平滑，能反映两个标签整体关联分布是否相似。

`Family Contrast` 定义为：

```text
mean(weighted overlap of same-family label pairs)
-
mean(weighted overlap of different-family label pairs)
```

该值越高，说明同家族标签比跨家族标签更相似，更符合 MISC 层级结构。

`Pair Recovery` 检查期望接近的标签对是否出现在最高相似度标签对中。本次三种表征均为 0.667，因此该指标支持“都能恢复一部分层级结构”，但不能区分 SAE 和 baseline。

### 3.6 Polysemanticity

`Mean Polysemanticity` 衡量一个被阈值选中的 feature 平均支持多少个标签。越高说明 feature 更容易跨多个标签复用，可能更泛化或更混合；越低说明 feature 更标签特异。

同时记录 role taxonomy：

- `label_specific`：只支持一个标签。
- `family_shared`：支持同一 MISC family 内多个标签。
- `cross_family`：跨 MISC family 支持多个标签。

对于可解释性而言，理想情况不是完全没有共享，而是 label-specific 和 family-shared 占主体，cross-family 共享较少。

## 4. 总体结果

| Representation | Mean Label AUC | Macro F1 | Macro Probe AUC | Mean N_eff | N_eff / D | Concentration@20 | Concentration@1% | Mean Jaccard | Weighted Overlap | Family Contrast | Pair Recovery | Mean Polysemanticity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sae_latents | 0.769 | 0.606 | 0.918 | 2771.3 | 0.085 | 0.039 | 0.266 | 0.074 | 0.668 | 0.241 | 0.667 | 1.29 |
| pca_components | 0.762 | 0.670 | 0.932 | 357.8 | 0.349 | 0.159 | 0.116 | 0.252 | 0.645 | 0.195 | 0.667 | 2.08 |
| raw_hidden_dims | 0.741 | 0.539 | 0.879 | 2891.5 | 0.706 | 0.016 | 0.030 | 0.052 | 0.736 | 0.196 | 0.667 | 1.22 |

总体上，SAE 的 `Mean Label AUC` 最高，说明在 SAE latent 空间中更容易找到与 MISC 标签强相关的单个 latent。PCA 的二级分类性能最高，说明 PCA 是强预测 baseline；但 PCA 的结构指标显示它更像密集预测方向，而不是稀疏可解释单元。Raw hidden states 的部分相似度很高，但 compactness 和 concentration 明显较弱，说明原始坐标中存在可预测信息，但信息更分散，语义单元边界不清晰。

## 5. 预测关联与分类性能分析

从 `Mean Label AUC` 看，三种表征均能捕获 MISC 标签信息：

- SAE：0.769
- PCA：0.762
- Raw hidden：0.741

SAE 相对 PCA 只高 0.007，说明它在最强单特征关联上不是压倒性优势；但相对 raw hidden 高 0.028，说明 SAE 编码后更容易出现和标签对齐的单元。

从二级分类看，PCA 更强：

- PCA `Macro F1 = 0.670`，`Macro Probe AUC = 0.932`
- SAE `Macro F1 = 0.606`，`Macro Probe AUC = 0.918`
- Raw hidden `Macro F1 = 0.539`，`Macro Probe AUC = 0.879`

这说明 PCA components 作为 dense linear basis 对监督分类非常有效。该结果不削弱 SAE 的可解释性结论，因为本实验的核心问题不是“哪个表征分类最高”，而是“哪个表征把标签关系暴露为更稀疏、更局部、更符合层级结构的 feature organization”。

逐标签看，SAE 在 `QU` 上的 F1 最高，为 0.895；在 `QUO` 上也与 PCA 接近，SAE 为 0.779，PCA 为 0.789。PCA 在 `RE`, `REC`, `GI`, `AF`, `SU` 的 F1 上更高。Raw hidden 在多数标签上低于 SAE 和 PCA。

结论：PCA 是更强的监督分类 baseline，SAE 是更合适的结构可解释性 baseline。论文中应避免写成“SAE 分类性能最好”，而应写成“SAE 在保持可比预测关联的同时，提供了更紧凑和更层级一致的标签结构”。

## 6. Compactness / Fragmentation 分析

### 6.1 归一化有效特征数

`N_eff / D` 是最关键的 compactness 指标：

- SAE：0.085
- PCA：0.349
- Raw hidden：0.706

SAE 的归一化有效特征数只有 PCA 的 24.2%，只有 raw hidden 的 12.0%。这说明虽然 SAE 绝对维度最多，但每个标签的关联质量只集中在很小比例的 latents 中。

逐标签的 `N_eff / D` 如下：

| Label | SAE | PCA | Raw hidden |
|---|---:|---:|---:|
| QUO | 0.058 | 0.229 | 0.611 |
| QU | 0.059 | 0.179 | 0.624 |
| REC | 0.076 | 0.334 | 0.869 |
| RE | 0.079 | 0.311 | 0.770 |
| SU | 0.093 | 0.520 | 0.686 |
| AF | 0.097 | 0.450 | 0.669 |
| GI | 0.130 | 0.422 | 0.712 |

所有 7 个标签上，SAE 的归一化有效特征数都是最低的。这是本实验支持 SAE 的最稳定证据。

### 6.2 前 1% 关联质量集中度

`Concentration@1%` 进一步支持 SAE 的 compactness：

- SAE：0.266
- PCA：0.116
- Raw hidden：0.030

SAE 的前 1% latent 捕获的关联质量约为 PCA 的 2.29 倍，raw hidden 的 8.81 倍。

逐标签看，SAE 在所有标签上的 `Concentration@1%` 都高于 PCA 和 raw hidden：

| Label | SAE | PCA | Raw hidden |
|---|---:|---:|---:|
| RE | 0.278 | 0.126 | 0.025 |
| REC | 0.285 | 0.120 | 0.020 |
| QU | 0.304 | 0.174 | 0.038 |
| QUO | 0.312 | 0.152 | 0.039 |
| GI | 0.195 | 0.089 | 0.028 |
| AF | 0.235 | 0.088 | 0.032 |
| SU | 0.252 | 0.064 | 0.030 |

这说明 SAE 的强关联 latent 更集中，尤其在 `QU` 和 `QUO` 上最明显。

### 6.3 阈值 fragmentation

按 `directional_auc >= 0.65` 选出的强关联特征数量如下：

| Label | SAE | PCA | Raw hidden |
|---|---:|---:|---:|
| GI | 1 | 2 | 1 |
| AF | 6 | 6 | 72 |
| RE | 8 | 4 | 135 |
| SU | 8 | 2 | 169 |
| QU | 13 | 4 | 107 |
| QUO | 22 | 3 | 139 |
| REC | 63 | 4 | 1310 |

该表不能孤立解读，因为 PCA 维度较少，raw hidden 维度居中，SAE 维度最多。但它仍然提供一个有用观察：SAE 中 `GI`, `AF`, `RE`, `SU` 的强关联 latent 数量较少，`QU` 和 `QUO` 中等，`REC` 明显更分散。

因此，从 SAE 角度看：

- 较紧凑标签：`GI`, `AF`, `RE`, `SU`
- 中等分散标签：`QU`, `QUO`
- 较分散标签：`REC`

`REC` 的分散性可能来自复杂反映本身语义异质性更高，包含更长、更复杂的咨询师回应模式；这与 MISC 行为定义是相符的。

## 7. Overlap 与层级恢复分析

### 7.1 Same-family vs different-family

SAE 的 family contrast 最高：

- SAE：0.241
- PCA：0.195
- Raw hidden：0.196

具体地，SAE 的 same-family weighted overlap 平均为 0.875，different-family weighted overlap 平均为 0.634，差值为 0.241。PCA 的差值为 0.195，raw hidden 为 0.196。

这说明 SAE 更能把同家族标签拉近，同时把跨家族标签相对分开。该指标是本实验支持“SAE 更符合 MISC 层级结构”的主要证据。

### 7.2 关键标签对

SAE 中 Jaccard 最高的标签对是：

| Pair | Same family | Jaccard | Weighted overlap |
|---|---:|---:|---:|
| QU - QUO | yes | 0.522 | 0.944 |
| GI - AF | no | 0.167 | 0.661 |
| RE - REC | yes | 0.127 | 0.954 |
| GI - SU | no | 0.125 | 0.754 |
| QUO - SU | no | 0.111 | 0.624 |
| QU - SU | no | 0.105 | 0.647 |
| AF - SU | yes | 0.077 | 0.725 |

`QU-QUO` 的 thresholded Jaccard 和 weighted overlap 都很高，说明问题类标签在 SAE 空间中共享了大量强关联 latent。这一结果符合 MISC 层级结构，但应写作 parent-child / same-family consistency，而不是写作两个独立标签之间的意外重叠。

`RE-REC` 的 Jaccard 不高，但 weighted overlap 很高，说明二者最强阈值 latent 集合并不完全相同，但整体关联 profile 非常相似。这适合解释为：反映类标签共享整体表征方向，但复杂反映 `REC` 需要额外 latents 承载更细的语义变化。

`AF-SU` 的 Jaccard 较低，weighted overlap 中等偏高，说明支持/肯定类在整体 profile 上有关联，但共享的强阈值 latent 较少。

### 7.3 与 PCA 和 raw hidden 的对比

PCA 的 same-family Jaccard 很高，达到 0.498，但 different-family Jaccard 也高，达到 0.211。其 top overlap 中出现了明显的跨家族混合，例如：

- `QUO-SU` Jaccard = 0.667
- `QU-SU` Jaccard = 0.500
- `RE-QUO` Jaccard = 0.400

这说明 PCA 的少数 dense components 会被多个标签共同使用，能够预测标签，但标签边界不够清楚。

Raw hidden 的 Jaccard 较低，但 weighted overlap 很高，same-family 为 0.904，different-family 也达到 0.708。也就是说，raw hidden 中很多标签的整体关联 profile 都相似，跨家族也不够分离。这更像是原始坐标中存在广泛分布的信息，而不是清晰可解释的标签单元。

相比之下，SAE 的 same-family Jaccard 为 0.242，different-family Jaccard 为 0.045，跨家族共享更少；同时 family contrast 最高。因此 SAE 的重叠模式更符合“同家族共享、跨家族分离”的结构解释。

## 8. Polysemanticity 分析

阈值选中特征的 polysemanticity 结果如下：

| Representation | Supported features | Label-specific | Family-shared | Cross-family | Mean polysemanticity | Max |
|---|---:|---:|---:|---:|---:|---:|
| SAE | 94 | 72 | 17 | 5 | 1.29 | 5 |
| PCA | 12 | 7 | 1 | 4 | 2.08 | 5 |
| Raw hidden | 1590 | 1288 | 194 | 108 | 1.22 | 5 |

SAE 中 94 个被阈值选中的 latents 里，72 个是 label-specific，17 个是 family-shared，只有 5 个是 cross-family。这说明 SAE 的强关联 latent 大多是标签特异或家族内共享，跨家族混合较少。

PCA 只有 12 个 components 被阈值选中，但其中 4 个是 cross-family，平均 polysemanticity 为 2.08，明显高于 SAE。这支持 PCA component 更容易成为混合型预测方向，而不是行为标签特异单元。

Raw hidden 的平均 polysemanticity 为 1.22，表面上最低，但它有 1590 个被阈值选中的维度，且 `N_eff / D` 和 `Concentration@1%` 都很差。因此 raw hidden 的低 polysemanticity 不能直接解释为更好可解释性；它更可能表示大量坐标各自携带弱而分散的关联。

结论：SAE 在 polysemanticity 上呈现更合理的结构，即多数 latent 标签特异，少量 latent 在 family 内共享，跨家族 latent 数量较少。

## 9. 综合判断

本实验的结果不是“SAE 在所有指标上都最好”。更准确的判断是：

1. PCA 在监督分类指标上最好，说明 dense PCA components 保留了强预测信息。
2. SAE 在单特征标签关联上略优，说明 SAE latent 空间中存在更清晰的标签相关单元。
3. SAE 在 `N_eff / D` 上显著最好，说明每个标签的关联质量集中在更小比例的 latents 中。
4. SAE 在 `Concentration@1%` 上显著最好，说明 top latent subset 捕获了更多标签关联质量。
5. SAE 的 `Family Contrast` 最高，说明它更符合 MISC 层级结构。
6. SAE 的 cross-family polysemanticity 更低，说明强关联 latent 更少跨家族混合。

因此，Step 6 的核心结论应写为：

> Compared with PCA components and raw hidden-state dimensions, SAE latents do not necessarily maximize supervised classification performance, but they expose a more compact and hierarchy-consistent MISC label-representation structure. Label associations are concentrated in a smaller fraction of the feature space, same-family labels show stronger profile similarity than different-family labels, and most selected SAE latents are label-specific or family-shared rather than broadly cross-family.

中文论文表述可以写为：

> 与 PCA components 和原始 hidden-state 坐标相比，SAE latents 并不一定取得最高的监督分类 F1，但它们更清晰地暴露了 MISC 标签在模型内部表征中的结构关系。具体而言，SAE 在归一化有效特征数、前 1% 关联质量集中度和 family contrast 上均优于对照表征，说明 MISC 行为标签在 SAE 空间中表现为更稀疏、更局部、且更符合标签层级结构的 latent 组织形式。

## 10. 可写入论文 Result 的结论

本节结果支持以下论文结论：

1. MISC 行为标签在 Gemma 表征中具有可测的线性关联，不同表征均能捕获一定标签信息。
2. PCA components 是强预测 baseline，但其标签共享模式更混合，跨家族 Jaccard 较高，说明它更适合作为 dense predictive control，而不是可解释单元基线。
3. Raw hidden-state dimensions 保留可预测信息，但标签关联较分散，前 1% 坐标只能捕获很少关联质量，说明原始坐标不直接提供紧凑的行为语义结构。
4. SAE latents 在结构指标上最优：归一化有效特征数最低，前 1% 关联质量最高，family contrast 最高，cross-family polysemanticity 较低。
5. 因此，后续对 MISC 标签结构、latent overlap、polysemanticity 和 causal ablation 的解释，应优先基于 SAE latent 空间，而不是 PCA 或 raw hidden 坐标。

## 11. 限制与注意事项

- PCA 当前使用 1024 components，解释方差为 0.896。由于 PCA 维度小于 raw hidden 和 SAE，固定 `Concentration@20` 对 PCA 有利，因此主结论应依赖 `N_eff / D` 和 `Concentration@1%`。
- `directional_auc >= 0.65` 是本项目的 operational threshold，不应写成 SAE 领域统一阈值。
- `RE-REC`、`QU-QUO` 包含父子/同家族关系，不能作为独立标签之间“自然发现的重叠”来夸大，应写作层级一致性证据。
- 本实验比较的是结构指标，不包含人工语义命名质量。SAE latent 是否具有稳定的人类语义解释，还需要结合 top activating examples 和 causal ablation 结果。
- `Pair Recovery` 三种表征均为 0.667，因此它不是区分 SAE 与 baseline 的关键指标，只能作为辅助证据。

## 12. 结果文件

- Paper-facing table: `outputs/misc_full_sae_eval/interpretability/baseline_comparison_step6/table4_baseline_comparison.md`
- Full table: `outputs/misc_full_sae_eval/interpretability/baseline_comparison_step6/table4_baseline_comparison.csv`
- Label-level compactness: `outputs/misc_full_sae_eval/interpretability/baseline_comparison_step6/label_structural_metrics_by_representation.csv`
- Label overlap: `outputs/misc_full_sae_eval/interpretability/baseline_comparison_step6/label_overlap_all_representations.csv`
- Polysemanticity: `outputs/misc_full_sae_eval/interpretability/baseline_comparison_step6/polysemanticity_all_representations.csv`
- Classification: `outputs/misc_full_sae_eval/interpretability/baseline_comparison_step6/classification_by_label.csv`
- Summary JSON: `outputs/misc_full_sae_eval/interpretability/baseline_comparison_step6/baseline_comparison_summary.json`
- Figures: `outputs/misc_full_sae_eval/interpretability/baseline_comparison_step6/figures/`
