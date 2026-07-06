# SAE 可解释性研究的交叉验证：文献综述与本项目实验设计

> 基于 SAE 可解释性、auto-interpretability、线性探针方法论文献，梳理领域标准做法，对应本项目设计交叉验证实验。
> 更新：2026-07-03

---

## 1. 背景：为什么 SAE 可解释性研究必须做交叉验证

SAE 可解释性研究的核心声明通常是：
> "这个 SAE latent 捕捉到了某个人类可理解的概念"

这类声明面临两个根本威胁：

**威胁一（Winner's Curse）**：在大量 latent（如 32768 个）上找 top-k，几乎必然选出噪声中的最高峰。缺少独立验证集，top 候选的"相关性"不可信。

**威胁二（Faithfulness ≠ Correlation）**：即使解释文字说"这个 latent 代表开放式提问"，也不能排除它只在追踪问号或某个词汇模板。只有受控的反事实实验才能区分。

SAE 领域主流论文（Cunningham et al. 2023, Bills et al. 2023, Anthropic scaling 2024, Lindsey et al. 2025）已经形成了一套标准验证框架，但当前本项目 P3 流程只完成了其中约三分之一。

---

## 2. 领域标准验证框架（三层）

### 2.1 Layer 1：观测性忠实度验证（Observational Faithfulness）

这是最基础的一层，问题是：**解释能否准确预测 latent 的激活行为？**

**标准做法（来源：Bills et al. 2023, Cunningham et al. 2023）**：

1. **Activation Prediction（激活预测）**：
   - 让解释器（LLM）看 latent 的 top 激活样例，生成候选解释
   - 在**独立 held-out 样例**上，用解释预测该样例是否会激活该 latent
   - 报告 Precision / Recall / F1（与真实激活对比）

2. **"Top-and-Random" 协议**（Cunningham et al.）：
   - 评估样例中一半是 top 激活样例，另一半是随机采样
   - 防止模型只学"高激活vs低激活"的简单判别而不是真正用解释

3. **相关性分数**（Bills et al.）：
   - 用 GPT 模拟预测激活强度，计算与真实激活强度的 Pearson/Spearman 相关

**本项目现状**：Task B 的 activation_prediction 任务在做这件事，但：
- 评估样例构造是否遵循"top-and-random"协议？需要确认
- confidence 坍缩在 0.85，说明激活强度的相关性信号丢失
- 缺少与 shuffled-latent baseline 的比较

---

### 2.2 Layer 2：统计可复现性验证（Statistical Reproducibility）

这一层问题是：**在独立数据上，top latent 的排名是否稳定？**

这是现有文献中**最容易被忽略、却在审稿时最常被追问**的一层。

**标准做法（来源：Templeton et al. 2024, SAE feature consistency position paper 2025）**：

1. **Split-Half 可复现性**：
   - 在独立两半数据上各自排名，计算 Spearman ρ
   - 报告 Top-K 集合的 Jaccard 系数，与随机基线比较

2. **Cross-Seed 稳定性**（多 run 训练 SAE 时）：
   - 不同随机种子训练的 SAE，是否收敛到相同的特征集
   - "Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs"（2025）专门把这点立为首要标准

3. **Bootstrap 置信区间**：
   - 对效应量（AUC, Cohen's d）报告 bootstrap CI
   - CI 跨零的 latent 不能作为候选

4. **Permutation Null**：
   - 打乱 latent-utterance 对应关系后重算指标
   - 若 shuffled 基线与原始指标相近，说明测量的是噪声

**本项目现状**：

- **缺失**：没有 split-half 验证
- **缺失**：没有 permutation null
- **缺失**：没有 bootstrap CI
- **风险**：SU(222 正例) / RES(516 正例) 极易出现 winner's curse

---



---

## 3. 本项目必须补的交叉验证实验

结合以上框架，本项目在 Layer 1 和 Layer 2 有以下具体缺口：

### 3.1 缺口列表（按优先级）

| 缺口 | 对应领域标准 | 后果（如不补） |
|---|---|---|
| 无 shuffled-latent baseline | Bills et al. / Cunningham et al. 阴性对照标配 | 无法排除"latent 无贡献"零假设 |
| 无 split-half TopK 可复现性 | Feature consistency 领域共识 | Top20 候选可能是 winner's curse |
| 无 bootstrap CI for AUC/d | 统计严谨性基线 | SU/RES 小样本结论不可信 |
| 无跨质量 split 验证 | Cross-distribution generalization 标准 | 无法排除"只是在区分会话质量"的替代解释 |
| Task B 无 label-only baseline | Bills et al. auto-interp 标准对照 | code_discrimination 无法归因于 latent |
| "Top-and-Random" 未确认 | Cunningham et al. 评估协议 | activation_prediction 可能虚高 |

### 3.2 各缺口的文献支撑与实验设计对应

下面逐一说明每个缺口在领域里的位置，以及本项目的具体操作。

---

## 4. 实验一：Shuffled-Latent Baseline（阴性对照，最高优先）

### 文献依据

Bills et al. (2023) §6 "Caveats and Limitations" 指出：
> "Future work should verify that explanation quality is above what would be expected from random neuron-explanation pairing."

Cunningham et al. (2023) 明确比较 SAE 与 PCA/ICA/random directions。

"Towards Principled Evaluations of SAEs" (2405.08366) 提出：supervised feature dictionaries 作为"上界参照"，random features 作为"下界参照"，任何 claim 必须在这两者之间。

### 本项目设计

```
操作：
  1. 对 utterance_features.pt[6194, 32768]，
     在 utterance 维度做随机打乱（Fisher-Yates shuffle）
  2. 打乱后重算所有 label-latent 的 AUC 和 Cohen's d
  3. 重复 N_permutations=1000 次
  4. 构建 null 分布

报告：
  - 每个 label 的 Top20 latent AUC 与 null 分布的 z-score
  - "显著高于 shuffled"（p < 0.05 / FDR）才能列为候选
  - 未通过的标签（如 SU / RES）必须在结论中降级
```

**判读意义**：这是"latent 是否携带有效信息"的最直接检验。若 shuffled 基线 AUC ≈ 真实 AUC，则所有后续解释全部撤回。

---

## 5. 实验二：Split-Half TopK 可复现性

### 文献依据

"Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs"（2505.20254）明确列出：

> "Feature consistency — the reliable convergence to equivalent feature sets across independent runs — should be treated as a first-class evaluation criterion."

Templeton et al. (2024) "Scaling Monosemanticity" 在附录报告了 latent 解释跨数据集的稳定性。

### 本项目设计

```
操作：
  1. 按 source_file GroupKFold(n_splits=2) 将 6194 条分成 A / B 两半
     （每半约 3097 条，按文件分组以避免会话内泄漏）
  2. 在 A 上独立计算每个 latent 对每个标签的 AUC
  3. 在 B 上独立计算同样的指标
  4. 对每个标签：
     - Spearman ρ(rank_A, rank_B)：整体排名稳定性
     - Jaccard(Top20_A, Top20_B)：候选集稳定性
     - 与 null（随机 Top20）的 Jaccard 期望做比较

报告：
  - 各标签 Spearman ρ（目标 ≥ 0.60）
  - 各标签 Top20 Jaccard（目标显著 > null + 2σ）
  - SU / RES 单独报告，预期可能失败
```

**结论规则**：
- Jaccard 通过 → 候选集稳定 → 可报为 reproducible_candidate
- Jaccard ≈ null → 候选集不稳定 → 不得进主结论（降级为 unstable_candidate）

---

## 6. 实验三：Bootstrap 置信区间

### 文献依据

"Faithful and Stable Neuron Explanations for Trustworthy Mechanistic Interpretability"（2512.18092）提出：

> "We derive generalization bounds for widely used similarity metrics (e.g. accuracy, AUROC, IoU) to guarantee faithfulness, and propose a bootstrap ensemble procedure that quantifies stability."

这是把 bootstrap CI 与 faithfulness 保证绑定的直接文献依据。

### 本项目设计

```
操作：
  1. 以 source_file 为单位做 grouped bootstrap（n=2000）
  2. 每次 bootstrap 重采样文件集合（有放回），保留会话内结构
  3. 计算每次 bootstrap 下每个 label-latent 的 AUC 和 Cohen's d
  4. 取 2.5 和 97.5 百分位作为 95% CI

报告：
  - 每个 Top20 latent 的 (AUC, AUC_CI, d, d_CI)
  - CI 跨零的 Cohen's d 不得报告为"正向候选"
  - 将 CI 宽度作为 SU / RES 小样本警告的量化依据
```

---

## 7. 实验四：跨质量 Split 验证（Cross-Distribution）

### 文献依据

Belinkov (2022) 综述指出：
> "Probing classifiers trained on one distribution may not generalize to another. Cross-topic and cross-domain generalization should be standard tests."

"Probing the Gap between In- and Cross-Topic Generalization"（EACL 2024）专门量化了这类分布迁移对探针解释的影响。

### 本项目设计

```
操作：
  1. 在 high split(4169 条) 上选 Top20 latent（按 AUC）
  2. 在 low split(2025 条) 上评估同一批 latent 的 AUC
  3. 计算 cross_split_AUC_drop = AUC_high - AUC_low
  4. 反向也做：low → high

报告：
  - 各标签 Top20 latent 的 cross-split drop 分布
  - drop ≥ 0.05 → 标注"质量源混淆风险"
  - 在 high 显著但 low 接近随机 → 不得进主结论
```

**意义**：直接排除"latent 区分的是会话质量风格，而非 MISC 行为模式"这个替代解释。这是审稿人最容易提出的混淆变量质疑。

---

## 8. 实验五：Task B 的 Label-Only Baseline（针对 code_discrimination）

### 文献依据

"Rigorously Assessing Natural Language Explanations of Neurons"（2309.10312）Appendix 专门测试：

> "Token-activation correlation baseline: a classifier that selects neurons whose activations correlate with concept-matching tokens — provides a stronger structural baseline than random."

类比到本项目：一个只拿标签定义（不拿 latent 派生解释）来做 code_discrimination 的分类器，应该作为 baseline。若 latent 解释没有超过这个 baseline，则 code_discrimination 指标的信息价值为零。

### 本项目设计

```
操作：
  在 Task B code_discrimination 的相同测试集上：
  1. 运行 Baseline A（Label-Only）：
     只给 AI 看 MISC 标签定义（如"QUO = Open-ended question"）
     不给任何 latent 相关信息
     让 AI 预测句子是否属于该标签
  2. 运行 Baseline B（Random-Latent）：
     随机选一个非目标 latent 的解释替换目标解释
     测量 code_discrimination 是否因此下降

报告：
  对每个标签报告：
    baseline_a_accuracy    ← 仅标签定义
    latent_accuracy        ← latent 派生解释（现有 Task B）
    delta = latent - baseline_a
  若 delta ≤ 0，说明 latent 解释对分类无增量贡献
```

---

## 9. 实验六：Top-and-Random 评估协议确认（Task B）

### 文献依据

Cunningham et al. (2023) §4.1 明确说：

> "We use a 'top-and-random' scoring procedure: half the evaluation fragments come from the top-activating examples, and half are sampled randomly from the corpus. This prevents the scorer from exploiting trivial high/low discrimination."

### 本项目现状与修复

Task B 的 `activation_prediction_task` 的评估样例如何构造需要确认：
- 若全部是 top 激活 vs low 激活的二分类 → 模型可以用激活强度高低作捷径
- 应改为"随机正例 vs 随机负例"的混合集合

**行动**：检查 `p3_scoring_tasks.jsonl` 中 `activation_prediction_task` 的样例分布，确认是否混入了随机样例（`random_target` 和 `low_activation` 两类）。若没有，重新构造评估集。

---

## 10. 文献来源汇总

| 文献 | 主要贡献 | 本项目应用 |
|---|---|---|
| Bills et al. (2023) | 自动 neuron 解释 + random baseline | Shuffled baseline、Label-Only baseline |
| [Cunningham et al. (2023)](https://ar5iv.labs.arxiv.org/html/2309.08600) | SAE 可解释性 + Top-and-Random + IOI causal test | Task B 评估协议、因果验证 |
| [Templeton et al. (2024)](https://arxiv.org/html/2406.04093v1) | SAE 扩展评估：probe loss, N2G, ablation sparsity | Probe-on-SAE-recon 充分性检验 |
| [Lindsey et al. (2405.08366)](https://arxiv.org/abs/2405.08366) | SAE 评估框架：近似/控制/可解释三轴 + supervised upper bound | 评估轴设计 |
| [Huang et al. (2309.10312)](https://arxiv.org/html/2309.10312v1) | 严格评估 neuron 解释：Precision/Recall/IIA | Task B 指标改进、IIA 设计 |
| [Cai et al. (2512.18092)](https://arxiv.org/html/2512.18092v1) | Bootstrap CI + generalization bounds for faithfulness | Bootstrap CI 实验 |
| [SAE consistency (2505.20254)](https://arxiv.org/html/2505.20254v1) | Feature consistency as first-class criterion | Split-half 可复现性实验 |
| [Belinkov (2022)](https://arxiv.org/abs/2102.12452) | 探针综述：cross-distribution generalization 标准 | 跨质量 split 验证 |
| [EACL 2024 cross-topic probing](https://ar5iv.labs.arxiv.org/html/2402.01375) | In- vs cross-topic generalization gap | 跨质量 split 验证 |

---

## 11. 实验优先级与执行顺序

```
必须做（P0）— 缺失则核心结论不可信：
  E1. Shuffled-Latent Baseline（排除 latent 无贡献零假设）
  E2. Split-Half TopK 可复现性（排除 winner's curse）
  E5. Label-Only Baseline for code_discrimination（归因 latent 贡献）

应该做（P1）— 决定结论强度：
  E3. Bootstrap CI（给效应量加置信区间）
  E4. 跨质量 Split 验证（排除源混淆）
  E6. Top-and-Random 评估协议确认（防止 Task B 虚高）

建议做（P2）— 增强可信度：
  — 多模型 / 多种子解释稳定性（Task A 跨模型对比）
  — Per-label Precision@K 在独立 split 上验证
```

---

*本文档是文献综述与实验设计，执行代码见 `doc/交叉验证实验计划.md` 和对应的 `run_*.py` 三件套。*
