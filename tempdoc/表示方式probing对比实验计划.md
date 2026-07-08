# 表示方式 Probing 对比实验计划

> 研究目标：公平比较不同表示方式对 MI/MISC 概念的线性可解码程度，探索并对比 SAE 对 MI 概念的"理解"程度。
> 对比表示：Hidden Representation、Full SAE、Top-n SAE、PCA-n、Random SAE-n，在 n ∈ {10,20,50,100,200} 画性能曲线。
> 更新：2026-07-06

---

## 1. 研究问题

**核心 RQ**：
> MISC 标签信息在不同表示空间中的线性可解码程度如何？稀疏的 SAE 特征基（Top-n SAE）是否比稠密基线（PCA-n）、随机基线（Random SAE-n）更高效地编码 MISC 概念？

这个实验回答的是"SAE 是否是一个比原始 hidden / PCA 更好的 MI 概念读出接口"，为 SAE 可解释性的合理性提供实证依据。

---

## 2. 现有基础设施盘点（关键：约 70% 已实现）

经代码勘查，`src/nlp_re_base/full_representation_probe.py` + `run_misc_full_representation_probe.py` 已经通过**同一条防泄漏代码路径**实现了大部分表示对比：

| 用户想要的表示族 | 现状 | 说明 |
|---|---|---|
| Hidden Representation (`raw_hidden`) | ✅ 已实现 | 完整 hidden 向量探针 |
| Full SAE (`full_sae_latents`) | ✅ 已实现 | 完整 SAE 特征向量探针 |
| Top-n SAE | ✅ 已实现 | 按 `cohens_d`/`directional_auc` 训练折内排名选择 |
| PCA-n | ⚠️ 部分 | 仅有 full-rank PCA（等价标准化 raw），**无 n 扫描** |
| Random SAE-n | ❌ 缺失 | 需新增 |

**已有的公平性保证**（三个表示共用）：
- `StratifiedGroupKFold`（按 `file_id` 分组，避免会话内泄漏）
- 每个训练折单独标准化（StandardScaler fit on train only）
- 相同 `LogisticRegression`（C=1.0, liblinear, class_weight=balanced）
- Top-n 特征选择只用训练折标签和训练折特征

**已有的曲线产物**：
- `auc_by_k_curve_0_100.csv`（AUC vs K）
- `figures/macro_auc_vs_k_0_100.png`、`label_auc_vs_k_0_100.png`

**结论**：这是一次**扩展**，不是从零构建。

---

## 3. 需要补的三个部分

### 3.1 PCA-n 扫描（必须）
当前只有 full-rank PCA。需要在 n ∈ {10,20,50,100,200} 各拟合一次 PCA（train fold only），取前 n 个主成分训练探针。

### 3.2 Random SAE-n 基线（必须，最关键的对照）
从与 Top-n SAE **相同的训练折候选池**中随机抽取 n 个特征，多个随机种子平均（捕捉方差）。

**为什么关键**：这直接回答"Cohen's d / AUC 排名选择是否真的优于等量随机选择"。如果 Top-n SAE ≈ Random SAE-n，说明排名没有信息价值，SAE 的"理解"存疑。

### 3.3 Top-n Raw 维度族（可选但推荐）
把相同的监督排名逻辑应用到原始 hidden 维度上。

**为什么推荐**：控制一个不对称性——Top-n SAE 是**监督选择**，而 PCA/Random 是**无监督/随机选择**。加入 `top_raw_n` 后，"稀疏 SAE 基优于稠密/随机基"的论证才严密，直接堵住审稿人最容易提的反对意见。

---

## 4. 统一实验设计（公平比较的核心）

### 4.1 对齐的 n 网格
所有表示族共用 **n = {10, 20, 50, 100, 200}**，保证每条曲线在同一张图上可比。

### 4.2 表示族与曲线角色

| 表示族 | 曲线类型 | 选择方式 |
|---|---|---|
| Top-n SAE | 随 n 变化曲线 | 监督（Cohen's d / directional AUC）|
| PCA-n | 随 n 变化曲线 | 无监督（方差）|
| Random SAE-n | 随 n 变化曲线（多种子均值±std）| 随机 |
| Top-n Raw | 随 n 变化曲线 | 监督（对照）|
| Full SAE | **水平参考线** | 全部 latent |
| Hidden Representation | **水平参考线** | 全部 hidden 维 |

### 4.3 公平性保证清单
- ✅ 相同 folds（StratifiedGroupKFold by file_id）
- ✅ 相同标准化（train fold only）
- ✅ 相同分类器与超参
- ✅ 所有特征选择（Top-n、PCA 拟合、随机抽样）都在训练折内完成
- ✅ 随机基线多种子平均
- ✅ 监督选择不对称性由 `top_raw_n` 控制

---

## 5. 实现计划（三件套，仅设计不写码）

### 5.1 扩展模块 `full_representation_probe.py`

向后兼容（新开关默认关闭），新增：

**Config 字段**：
```
include_pca_n_sweep: bool = False
include_random_sae_n_sweep: bool = False
include_top_raw_n_sweep: bool = False
random_sae_n_seeds: tuple[int,...] = (0,1,2,3,4)
comparison_ns: tuple[int,...] = (10,20,50,100,200)
```

**Helper 函数**：
- `_fit_pca_fold_n(...)`：在训练折上按显式 n 拟合 PCA
- `_random_subset_indices(...)`：确定性 RNG（key = random_state, label, fold, seed），从训练折候选池抽取
- 复用现有 `(subspace_ranking, top_n)` 行 schema，使 `_build_auc_by_k_curve` 自动识别

**家族标签**：`pca_n` / `random_sae_n` / `top_raw_n`（随机种子行经 `_summarize_by_label` 折叠为 per-representation 均值±std）

**图增强**：`_write_auc_k_figures` 增加所有家族对比 + Full-SAE / Raw-hidden 水平参考线

### 5.2 新增 Runner `run_misc_representation_comparison.py`
- 预设 5 族完整对比 + 显式 n 网格 + 种子
- 使用 **stable-core latents 作为 SAE 候选池**（对齐最新决定）
- 输出到 `outputs/misc_full_sae_eval/interpretability/representation_comparison/`

### 5.3 新增 Smoke 测试 `test_representation_comparison_smoke.py`
- 合成小数据，网格 {5,10}，2 种子
- 断言曲线 CSV 含 `pca_n` / `random_sae_n` / `top_raw_n` 行
- 断言 macro 图已写出（matplotlib 缺失则跳过）
- 断言无泄漏（选择只用 train 索引）

---

## 6. 验证方案
- `python -m py_compile` 两个文件
- `conda run -n qwen-env-py311 python test_representation_comparison_smoke.py`

---

## 7. 最终交付物

**核心图**：macro AUC vs n ∈ {10,20,50,100,200}
- 曲线：{Top-n SAE, PCA-n, Random SAE-n, Top-n Raw}
- 水平线：Full-SAE ceiling, Raw-hidden ceiling

**支撑 CSV**：per-label + macro 的 AUC-by-K 曲线数据

**可解读的结论形式**：
| 观测 | 解读 |
|---|---|
| Top-n SAE > Random SAE-n | SAE 排名有信息价值 |
| Top-n SAE ≈ Full SAE（小 n）| SAE 概念表征紧凑 |
| Top-n SAE > PCA-n | 稀疏基比稠密基更高效编码 MISC |
| Top-n SAE ≈ Top-n Raw | 优势来自监督选择而非 SAE 本身 |
| Top-n SAE > Top-n Raw | SAE 稀疏基确实优于原始维度 |

---

## 8. 范围边界
- 不重跑模型推理（复用现有 feature store）
- 不做 token-level 分析
- 不做 AI 解释（独立轨道）

---

*本文档为实验设计规范，不含实现代码。执行时按三件套规范落地。*
