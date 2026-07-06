# Top-K 选择的依附分析与收束实验设计

> 当前 K=20 是预设值，缺乏数据依据。本文档分析哪些流程步骤依附于该值，并设计使 K 成为数据驱动收束值的实验方案。
> 更新：2026-07-05

---

## 1. 哪些流程步骤依附于 Top 20

按模块化方法文档（MI_MISC_SAE功能模块化方法说明.md）梳理，K=20 在三个不同"角色"上被使用：

| 使用位置 | Top 20 的角色 | 若 K 选错的后果 |
|---|---|---|
| **模块四 label-latent 关联** | 每标签的候选集截断点 | 截太窄漏掉真信号；截太宽引入噪声 |
| **模块五统计结构**（Overlap / Jaccard / 碎片化） | Jaccard 计算的集合边界 | K 不同则 Jaccard 数字完全不可比 |
| **P3 feature cards**（9×20=180 张） | 进入解释流程的 latent 总量 | 太多则人工/AI 审核难以闭环 |
| **最小充分子空间的上界**（候选池 Top 100）| Top 20 是进入 P3 的门槛，100 是搜索上界 | 若 20 以外有强候选，minimal K 会被低估 |

**核心问题**：K=20 事实上是"解释质量管道入口"的人工截断，而非统计判据。但它隐式影响了所有下游结构分析的数字，且对不同标签应该不同，目前被强行一致了。

---

## 2. K 在流程中扮演的两个功能

K=20 实际上扮演了两个不同角色，每个角色对应不同的收束方法：

| 功能 | 描述 | 收束方式 |
|---|---|---|
| **功能 A：统计筛选** | 多少个 latent 是真实信号？ | 统计判据（FDR、permutation null）|
| **功能 B：解释候选** | 多少个 latent 值得进 P3？ | 稳定性判据（split-half Jaccard）|

这两个功能的最优 K 不一定相同，可以分开报告，也可以取两者的交集作为最终 K。

---

## 3. 实验一：AUC 稳定性收束曲线（最高优先）

### 3.1 核心思路

通过 split-half 可复现性随 K 变化的曲线，找到"稳定信号区域"的边界。

### 3.2 操作

```
对每个 MISC 标签，对 K = 1, 2, 5, 10, 15, 20, 30, 50, 100：
  1. 按 source_file GroupKFold(n_splits=2) 分成独立两半 A / B
  2. 在 A 上按 AUC 排名取 Top-K latent
  3. 计算 Jaccard(Top-K_A, Top-K_B)
  4. 与随机 Top-K 的 null Jaccard 期望比较

画出：Jaccard(K) vs K 的曲线（每标签一条）
```

### 3.3 预期形状

```
高 ■■
   ■  ■ ■ ← 平台期（稳定信号区域）
Jaccard       ■ ■
              null ──────────────
低
   1  5  10  15  20  30  50  K
```

### 3.4 如何读 K

- 曲线在某个 K* 之前显著高于 null 且相对平坦 → 稳定信号区域
- K > K* 后 Jaccard 开始贴近 null → 已进入噪声区
- **K* 就是数据给出的收束 K**

### 3.5 对不同标签的预期

对不同标签，K* 会不同：
- QUO 可能 K*≈8（句法清晰，信号集中）
- GI 可能 K*≈25（功能分散，需要更多 latent）
- SU 可能 K*≈3-5（222 正例，信号弱）
- RES 可能 K*≈0-5（516 正例 + 缺 client context）

这本身是可报告的发现：**不同 MISC 标签的可稳定识别 latent 数量不同**，反映了各标签在 SAE 空间中的集中度差异。

---

## 4. 实验二：FDR 控制的数据驱动 K

### 4.1 核心思路

用统计显著性替代预设截断，让"有多少 latent 真实存在"来决定 K。

### 4.2 操作

```
对全部 filtered pool（12758 latent）× 9 标签：
  1. 计算每个 label-latent 的 AUC
  2. 用 permutation（打乱标签，重复 1000 次）构建 null 分布
  3. 计算每个 pair 的 permutation p-value
  4. BH-FDR 校正（α=0.05）
  5. 每个标签存活的 latent 数量 = 该标签的 FDR-K
```

### 4.3 预期结果形式

| 标签 | 预期 FDR-K | 与 Top20 对比 |
|---|---|---|
| QUO | 30–80 | 20 可能截断了真信号 |
| AF | 5–15 | 20 可能引入了噪声 |
| SU | 0–5 | 20 几乎全是 winner's curse |
| GI | 15–40 | 20 大致合理 |
| RES | 0–8 | 20 引入了大量噪声 latent |

### 4.4 意义

- 若某标签 FDR-K < 20 → 当前候选集里有噪声 latent，不应进 P3
- 若某标签 FDR-K > 20 → 当前候选集截断了真信号，P3 可能遗漏重要 latent
- SU/RES 若 FDR-K ≈ 0 → 这两个标签的"无贡献"结论有统计支撑

---

## 5. 实验三：AUC 增量收益曲线（Elbow 方法）

### 5.1 核心思路

在最小充分子空间逻辑的基础上，从 K=1 向上扩展，找到预测信息的"饱和点"。

### 5.2 操作

```
对每个 MISC 标签，按 AUC 排序后逐步扩大候选集：
  K = 1, 2, 3, 5, 10, 15, 20, 30, 50, 100

  对每个 K：
    取 Top-K latent 作为特征
    训练线性探针（5-fold grouped CV）
    记录 probe AUC

计算边际增益：ΔAUC(K) = AUC(K) - AUC(K-1)
找到 ΔAUC < ε（建议 ε=0.005）的第一个 K = 信息饱和点
```

### 5.3 与现有最小充分子空间的区别

| 方向 | 现有代码（minimal_sufficient_subspace_v2）| 本实验 |
|---|---|---|
| 搜索方向 | Top 100 内**向下**（找最小充分集）| 从 K=1 **向上**（找饱和点）|
| 问题 | 最少需要几个？| 多少个之后信息不再增加？|
| 输出 | minimal K（10-24）| saturation K（上界）|

**两者合起来确定"信号窗口"**：minimal K（下界）~ saturation K（上界）。

---

## 6. 实验四：K 敏感性扫描（论文防御性验证）

### 6.1 核心思路

证明 K=20 的主要结构性发现在合理 K 范围内是稳健的。

### 6.2 操作

```
对 K ∈ {10, 15, 20, 30}：
  重跑模块五：
    - Overlap Jaccard 矩阵（标签间相似度）
    - 父子标签 overlap vs 跨家族 overlap
    - exclusive / family_shared / cross_family latent 数量
    - 碎片化指标（minimal K 随 K 的变化）

比较各 K 下关键结论是否一致
```

### 6.3 判读

- 若 K=10 到 K=30 内，结论基本一致 → K=20 在稳定区，是合理选择
- 若结论随 K 变化明显 → 必须用实验一或二给出数据驱动的 K

---

## 7. 整合策略：如何形成论文中有说服力的 K 选择

### 7.1 执行顺序

```
实验一（稳定性曲线，必做）
  ↓
实验二（FDR-K，应做）
  ↓  比较：K*（实验一）≈ FDR-K（实验二）？
  ↓
实验四（敏感性扫描，建议做）
  ↓
更新 K 定义：每标签独立 K，或统一取合理范围
  ↓
若实验一的 K* 接近 20：
  保留 K=20 但补充"K=20 落在稳定区的证据"
若 K* 与 20 差距大：
  重新确定 K 并调整 P3 规模
```

### 7.2 实验三（Elbow）的补充作用

实验三的 saturation K 补充了实验一的 K*：
- 若 K* ≈ saturation K → 信号窗口很窄，latent 集中
- 若 K* ≪ saturation K → 有一批"弱但真实"的 latent 在稳定区外

### 7.3 论文最终表述模板

```
"We set K=20 as our primary analysis parameter.
 This value falls within the stable reproducibility region
 (Experiment 1: K* = [range] across labels),
 is confirmed by FDR analysis to cover the majority of
 statistically significant associations (Experiment 2: FDR-K median = [X]),
 and sensitivity analyses (K=10–30) show structural findings are robust
 to this choice (Experiment 4)."
```

若不同标签的 K* 差异大，改为：
```
"Per-label analysis reveals heterogeneous signal concentration:
 [QUO/QUC] have compact representations (K*≈[X]),
 while [GI/SU] require larger candidate sets (K*≈[Y]).
 We report results both at K=20 (for comparability) and at label-specific K* values."
```

---

## 8. 预期结论价值

除了为 K 选择提供依据，这组实验本身会产出新的可报告发现：

| 发现 | 意义 |
|---|---|
| 各标签 K* 不同 | 不同 MISC 行为在 SAE 空间中的集中度不同，是一个结构性发现 |
| SU/RES 的 K* 极小或 FDR-K≈0 | 这两个标签的可解释性天花板低，有数据支撑 |
| K* < 20 的标签（如 AF）| 说明 P3 里有噪声 latent 混入，可以精简 |
| 信号窗口（minimal K ~ saturation K）| 量化了每个标签的"有效 SAE 表征维度数" |

---

## 9. 代码实现计划（三件套）

| 文件 | 路径 | 说明 |
|---|---|---|
| `k_stability_curve` | `src/nlp_re_base/k_stability_curve.py` | 实验一：K 扫描 + Jaccard + null model |
| `fdr_driven_k` | `src/nlp_re_base/fdr_driven_k.py` | 实验二：permutation p-value + BH-FDR |
| `auc_saturation_curve` | `src/nlp_re_base/auc_saturation_curve.py` | 实验三：elbow 方法 |
| `run_k_convergence.py` | 项目根目录 | 统一入口，调用三个实验 |
| `test_k_convergence_smoke.py` | 项目根目录 | 合成数据 smoke test |

所有实验依赖 E0（GroupKFold框架）和已有 `utterance_features.pt` + `label_matrix.csv`，无需重跑推理，全程 CPU 可完成。

---

*本文档对应的执行日志记录至 `doc/日志.md`，最终结论写入 `doc/P3研究设计与方法规则.md` 对应章节。*
