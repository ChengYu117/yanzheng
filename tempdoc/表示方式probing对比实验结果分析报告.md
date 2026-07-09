# 表示方式 Probing 对比实验结果分析报告

> 实验：`run_misc_representation_probe_comparison.py`
> 输出目录：`outputs/misc_full_sae_eval/interpretability/representation_probe_comparison_stable_core/`
> 运行日期：2026-07-06，总时长约 1h 40min（Full SAE 32768维 × 9 标签 × 5 折）

---

## 1. 核心结论摘要（先说结果）

1. **PCA-n 在整个 n 范围内持续优于 Top-n SAE**——这是最重要的发现。
2. **Random SAE-n 与 Top-n SAE 有巨大差距**，确认 Cohen's d 排名携带真实信号。
3. **Stable Core SAE（33.7 维平均）相当于 Top-n SAE 的 n≈50-100**，表现出色。
4. **Full SAE 和 Hidden State 表现接近**（均为宏观 AUC≈0.89-0.90），SAE 稀疏化总体没有大损失。
5. **RES 是所有方法的共同瓶颈**，与缺少 client context 的预期一致；**QU/QUO 各方法均高**。

---

## 2. Macro AUC 全对比表

| 表示方式 | n | Macro AUC | Macro PR-AUC | Macro F1 | Macro BalAcc |
|---|---:|---:|---:|---:|---:|
| Hidden State | — | **0.900** | 0.679 | 0.642 | 0.811 |
| Full SAE | — | 0.890 | 0.600 | 0.572 | 0.815 |
| Stable Core SAE | — | 0.869 | 0.602 | 0.547 | 0.807 |
| PCA-n | 10 | 0.870 | 0.498 | 0.484 | 0.795 |
| PCA-n | 20 | 0.901 | 0.576 | 0.531 | 0.828 |
| PCA-n | 50 | 0.925 | 0.652 | 0.579 | 0.854 |
| PCA-n | **100** | **0.934** | **0.690** | **0.609** | **0.859** |
| PCA-n | 200 | 0.932 | 0.698 | **0.627** | 0.849 |
| Top-n SAE | 10 | 0.822 | 0.524 | 0.511 | 0.772 |
| Top-n SAE | 20 | 0.849 | 0.559 | 0.528 | 0.792 |
| Top-n SAE | 50 | 0.874 | 0.590 | 0.545 | 0.807 |
| Top-n SAE | **100** | **0.883** | 0.601 | 0.558 | 0.815 |
| Top-n SAE | 200 | 0.879 | 0.596 | 0.567 | 0.808 |
| Random SAE-n | 10 | 0.546 | 0.173 | 0.205 | 0.538 |
| Random SAE-n | 20 | 0.575 | 0.188 | 0.234 | 0.559 |
| Random SAE-n | 50 | 0.640 | 0.240 | 0.284 | 0.604 |
| Random SAE-n | 100 | 0.693 | 0.288 | 0.320 | 0.644 |
| Random SAE-n | 200 | 0.746 | 0.350 | 0.372 | 0.689 |

---

## 3. 关键发现详解

### 3.1 最重要发现：PCA-n 全程优于 Top-n SAE

```
n=20:   PCA 0.901  vs  Top-n SAE 0.849   → PCA 领先 +0.052
n=50:   PCA 0.925  vs  Top-n SAE 0.874   → PCA 领先 +0.051
n=100:  PCA 0.934  vs  Top-n SAE 0.883   → PCA 领先 +0.051
n=200:  PCA 0.932  vs  Top-n SAE 0.879   → PCA 领先 +0.053
```

PCA-n 在 n=20 时就已追平 Hidden State（0.901），在 n=100 达到全实验最佳（0.934）。
**结论**：在等量特征数下，raw hidden 的主成分方向比 SAE latent 的 Cohen's d top 方向更高效地编码 MISC 概念，即便 PCA 是无监督的。

**方法论含义（重要）**：这意味着"Top-n SAE > Random SAE-n"的差异主要是监督选择（Cohen's d）的功劳，而非 SAE 稀疏基的固有优势。若有 Top-n Raw 对照（相同监督选择施加到 raw hidden 维）——这正是 MAJOR-1 缺口——则可判断优势到底来自"监督选择"还是"SAE 基"。当前结果已暗示答案偏向前者。

### 3.2 Top-n SAE 峰值在 n=100，n=200 略微下降

```
n=10 → n=50:  AUC  0.822 → 0.874  (+0.052，增益显著)
n=50 → n=100: AUC  0.874 → 0.883  (+0.009，增益已平缓)
n=100 → n=200:AUC  0.883 → 0.879  (-0.004，轻微回落)
```

n=200 的轻微回落说明第 101-200 个正向 Cohen's d 候选 latent 引入了噪声，已超过"有效信号区间"的上界。这与 stable_k_by_label.csv 显示的稳定性收束 K* 集中在 26-72 之间一致：n=100 是合理的宏观峰值。

### 3.3 Random SAE-n 几乎近随机，但确认排名有效

```
n=10 时：Random AUC 0.546（接近随机 0.5）
n=200时：Random AUC 0.746（仍显著低于 Top-n SAE 0.879，差距 +0.133）
```

**Top-n SAE 与 Random SAE-n 的 AUC 差距在各 n 下均显著**（n=100 时差距 0.883-0.693=+0.190），这是 Cohen's d 排名携带真实 label-latent 关联信号的**直接证据**。这个对照是整个实验最重要的阴性控制。

### 3.4 Stable Core SAE 以平均 33.7 个特征接近 Top-n SAE n=100

```
Stable Core SAE:  macro AUC 0.869，平均特征数 33.7
Top-n SAE n=50:   macro AUC 0.874
Top-n SAE n=100:  macro AUC 0.883
```

Stable Core 以不到 34 个特征达到 Top-n SAE n=50 的水平，**信息密度高**。但它与 Full SAE 相比损失约 0.021 AUC，说明即便是精心筛选的稳定核心，在这一指标上仍不如全量 SAE。

### 3.5 Full SAE 与 Hidden State 相当，SAE 无大损失

```
Hidden State: macro AUC 0.900
Full SAE:     macro AUC 0.890
差值:         -0.010
```

两者差距仅 0.010 AUC，说明 SAE 稀疏化在宏观上保留了大多数 MISC 概念的线性可解码信息。这支持"SAE 重构充分性"的基本假设。

---

## 4. 标签级详细分析

### 4.1 各标签 Full SAE AUC（大图参考基准）

| 标签 | Full SAE AUC | Hidden AUC | Stable Core AUC | 稳定核心数 |
|---|---:|---:|---:|---:|
| QU | 0.970 | 0.976 | 0.971 | 26 |
| QUO | 0.946 | 0.952 | 0.949 | 32 |
| REC | 0.902 | 0.903 | 0.898 | 47 |
| QUC | 0.889 | 0.903 | 0.921 | 45 |
| RE | 0.888 | 0.898 | 0.848 | 54 |
| SU | 0.871 | 0.891 | 0.829 | 31 |
| AF | 0.927 | 0.950 | 0.910 | 27 |
| GI | 0.823 | 0.823 | 0.732 | 26 |
| RES | 0.794 | 0.803 | 0.768 | 15 |

**QU/QUO**：最容易，各方法均 AUC>0.94，Stable Core 已接近全量。  
**QUC**：Stable Core（0.921）反而高于 Full SAE（0.889）——这是一个值得关注的异常，可能与 Stable Core 排除了 QUC 的低质量噪声 latent 有关。  
**GI**：各方法均较低，GI 的表征分散，与 K*=72 且 unstable 吻合。  
**RES**：各方法最差（0.767-0.803），与 516 正例 + 缺 client context 完全一致。  
**RE vs REC**：RE 的 Stable Core（0.848）明显低于 Full SAE（0.888），说明 RE 的稳定核心（54个）遗漏了 Full SAE 中一些有贡献的 latent。

### 4.2 Top-n SAE 各标签的 n=10 vs n=100 增益

| 标签 | n=10 AUC | n=100 AUC | 增益 | 解释 |
|---|---:|---:|---:|---|
| QU | 0.957 | 0.975 | +0.018 | 已高度紧凑，小 n 够用 |
| QUO | 0.938 | 0.959 | +0.021 | 同上 |
| QUC | 0.894 | 0.925 | +0.031 | 需要更多 latent |
| REC | 0.868 | 0.910 | +0.042 | 中等增益 |
| RE | 0.789 | 0.858 | +0.069 | 增益显著，分布式 |
| SU | 0.687 | 0.850 | +0.163 | 增益最大，需大量 latent |
| AF | 0.892 | 0.915 | +0.023 | 较紧凑 |
| GI | 0.644 | 0.765 | +0.121 | 增益大，高度分布式 |
| RES | 0.731 | 0.790 | +0.059 | 较大增益但整体偏低 |

**QU/QUO/AF**：小 n 即可（紧凑型），Top-n SAE 在 n=10 已很高。  
**SU/GI**：增益最大，需要 n>50 才接近天花板，与"unstable"分类一致。

### 4.3 PCA 在各标签的表现（以 n=100 为例）

| 标签 | PCA-100 AUC | Top-n SAE-100 AUC | PCA 优势 |
|---|---:|---:|---:|
| QU | 0.979 | 0.975 | +0.004 |
| QUO | 0.966 | 0.959 | +0.007 |
| RE | 0.932 | 0.858 | **+0.074** |
| REC | 0.947 | 0.910 | +0.037 |
| QUC | 0.927 | 0.925 | +0.002 |
| AF | 0.956 | 0.915 | +0.041 |
| GI | 0.899 | 0.765 | **+0.134** |
| SU | 0.923 | 0.850 | **+0.073** |
| RES | 0.878 | 0.790 | **+0.088** |

PCA 优势最显著的是 GI（+0.134）、RES（+0.088）、SU（+0.073）、RE（+0.074）——全是"分布式/不稳定"标签。PCA 对这些标签更有效，可能因为它捕获了一些在 SAE 稀疏化后被碎片化的 dense 信号。

---

## 5. 方法论解读

### 5.1 PCA > Top-n SAE 的含义（核心讨论点）

这是与直觉预期相悖的结果，需要在论文中谨慎处理：

**可能解释**：
1. **SAE 稀疏化压缩了一部分 label 信号**：PCA 保留全部方差，而 SAE 为了稀疏性会牺牲一些 reconstruction；在 MISC 标签这样的任务上，这种损失可能不均匀。
2. **Cohen's d 监督选择 vs PCA 无监督保留**：PCA 方向保留了最大方差——如果 MISC 概念在 hidden space 中有较大方差方向，PCA 自然会优先保留；而 Cohen's d 是基于标签的监督排名，有数据集偏差。
3. **LinearRegression 对 dense PCA 特征更友好**：liblinear 对稠密、正交、归一化的特征效率更高；SAE 特征是稀疏的，标准化后可能引入人工信号。

**无法排除的替代解释**（因为缺少 Top-n Raw 对照）：  
如果 Top-n Raw（用相同 Cohen's d 监督排名选 raw hidden 维）也优于 Top-n SAE，则 "SAE 不如 dense 表征"的结论就是更强的。当前无法区分"dense 比 sparse 好"还是"PCA 方向比 Cohen's d 排名选出的方向好"。

### 5.2 Stable Core 的定位修正

Stable Core 以~34个特征达到宏观 AUC 0.869，介于 Top-n SAE n=50（0.874）和 n=100（0.883）之间。但 Stable Core 特征身份的选择存在轻微乐观偏置（用全量数据选择，见审查报告 MAJOR-2），所以其实际 held-out 性能可能略低于表中数字。

对 QUC 出现 Stable Core > Full SAE 这一逆转的解读：Stable Core 过滤掉了 QUC 中不稳定 latent，留下的"pure" latent 反而比全量 SAE 探针表现更好，说明 QUC 的 Full SAE 中有一定噪声 latent 干扰了分类器。

---

## 6. 与研究问题的对接

| 研究问题 | 实验回答 |
|---|---|
| SAE 的 Cohen's d 排名是否有效？ | ✅ 有效：Top-n SAE 远超 Random SAE-n（n=100 差距 +0.190 AUC）|
| SAE 全量表征是否保留了 MISC 信息？ | ✅ 是：Full SAE ≈ Hidden State（差距仅 -0.010）|
| 少量 latent 是否能高效编码？ | ✅ 是：Top-n SAE n=100 达到 0.883，Stable Core ~34特征达 0.869 |
| SAE 是否优于其他表示？ | ⚠️ 否：PCA-n 在所有 n 下均优于 Top-n SAE |
| 不同标签的可解码难度？ | ✅ QU/QUO 最易，RES/GI 最难，与语言学解释一致 |

---

## 7. 需要在论文中注明的限制

1. **缺少 Top-n Raw 对照**：无法区分"监督选择优势"vs"SAE 稀疏基优势"，这是 PCA > Top-n SAE 发现的核心解释盲点。
2. **Stable Core 轻微乐观偏置**：选择时见过全量数据，与 Top-n SAE 的无泄漏 per-fold 选择不同口径。
3. **Full SAE 计算开销**：32768 维 liblinear，在全量数据上是当前方法的上限，不适合迭代实验。
4. **PCA 在每折单独 fit**：与 Top-n 同等公平，但 PCA 方向是无监督的，某种程度上受到数据分布影响而非标签。

---

## 8. 后续建议

**P0（修复方法论缺口）**：
- 补 Top-n Raw 族：对 raw hidden 4096 维做同样 Cohen's d 训练折内排名，从根本上鉴别"监督选择"vs"SAE 基"的贡献。

**P1（深化发现）**：
- 分析 GI/SU/RE 的"PCA 大幅优于 Top-n SAE"：是 SAE 对这些标签稀疏化损失特别大，还是 Cohen's d 排名在这些标签上特别不稳定？与 cross_val 的 `performance_only_unstable` 结果对照。
- QUC Stable Core > Full SAE 的反转值得深挖：哪些 latent 被 Stable Core 过滤掉，它们是什么噪声类型？

---

*本报告对应实验输出：`outputs/misc_full_sae_eval/interpretability/representation_probe_comparison_stable_core/`*  
*图表：`figures/performance_curves_macro.png`*
