# SAE Fragmentation / Downstream Effect Sparsity 参考文献统计

用途：为后续论文中 Fragmentation 指标、候选 latent 筛选、downstream effect sparsity、minimal sufficient latent count 等方法设计提供引用依据。

## 1. 参考文献总表

| 序号 | 文献 | 年份 | 来源 | 与本项目的关系 | 建议引用位置 |
|---:|---|---:|---|---|---|
| 1 | Elhage et al., *Toy Models of Superposition* | 2022 | arXiv / Transformer Circuits | 解释 polysemanticity 和 superposition，说明一个 neuron/方向可能混合多个概念，支持使用 SAE features/latents 分解表征。 | 理论背景：为什么需要 SAE |
| 2 | Bricken et al., *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* | 2023 | Anthropic / Transformer Circuits | 提出用 dictionary learning / SAE-like 方法把神经元激活分解为更可解释的 features；说明 features 可作为比 neuron 更合适的分析单元。 | SAE 方法背景 |
| 3 | Cunningham et al., *Sparse Autoencoders Find Highly Interpretable Features in Language Models* | 2023 / ICLR 2024 | arXiv / ICLR | 证明 SAE features 比 neurons、PCA、ICA 等方向更可解释，并能更细粒度定位任务相关 features。 | SAE features 作为可解释 latent 单元 |
| 4 | Anthropic, *Mapping the Mind of a Large Language Model* / Templeton et al., *Scaling Monosemanticity* | 2024 | Anthropic / Transformer Circuits | Claude 3 Sonnet 中抽取大规模 features，并通过 feature steering 验证部分 features 会影响模型行为；也提示 feature set 仍不完整。 | Claude 风格 feature identification；说明本项目不能简单照搬 token/topic feature counting |
| 5 | Gao et al., *Scaling and evaluating sparse autoencoders* | 2024 | arXiv / OpenAI | 提出 SAE 评价不应只看重构和稀疏度，还应看 hypothesized feature recovery、activation explainability、downstream effect sparsity。 | downstream effect sparsity 方法依据 |
| 6 | Marks et al., *Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models* | 2024 / ICLR 2025 | arXiv / ICLR | 用 SAE features 构建 sparse feature circuits，并通过 ablation / editing 评估 feature 对下游任务或行为的影响。 | downstream effect ablation / feature contribution |
| 7 | Chanin et al., *A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders* | 2024 / NeurIPS 2025 | arXiv / NeurIPS | 讨论 feature splitting 和 absorption，说明概念可能被拆分到多个 SAE features，支持“标签碎片化”问题的合理性。 | Fragmentation 理论动机 |
| 8 | Karvonen et al., *SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability* | 2025 / ICML 2025 | arXiv / ICML | 强调 SAE 评价需要多维 benchmark，不能只依赖无监督 proxy；支持在本项目中加入下游预测/干预导向指标。 | SAE 评价方法学依据 |

## 2. 与本项目方法设计的对应关系

| 本项目设计问题 | 推荐引用 | 作用 |
|---|---|---|
| 为什么不用 neuron，而用 SAE latent/features 分析标签表征？ | Elhage et al. 2022; Bricken et al. 2023; Cunningham et al. 2023 | superposition / polysemanticity 背景；SAE features 更可解释 |
| 为什么 Claude 风格 simple feature counting 不能直接照搬？ | Anthropic 2024; Bricken et al. 2023 | Claude 多基于 token/topic/entity features；本项目标签是 MISC 行为概念标签 |
| 为什么 Fragmentation 可以理解为概念由多少 features/latents 支持？ | Chanin et al. 2024; Gao et al. 2024 | feature splitting / downstream effect sparsity 支持“概念可能分散在多个 features” |
| 为什么不能对所有 latents 做下游效应评估？ | Gao et al. 2024; Karvonen et al. 2025 | SAE 评价需要任务相关指标，但也需要可计算、可解释的候选筛选 |
| 为什么使用候选召回池再做下游任务评估？ | Cunningham et al. 2023; Gao et al. 2024; Karvonen et al. 2025 | 先用关联指标召回候选，再用下游任务指标确认贡献 |
| 为什么 `minimal sufficient latent count` 适合作为 Fragmentation 主指标？ | Gao et al. 2024; Marks et al. 2024; Karvonen et al. 2025 | 与 downstream effect sparsity、feature contribution、task-oriented SAE evaluation 一致 |
| 为什么必须声明不是 causal sufficiency？ | Marks et al. 2024; Anthropic 2024 | 真正因果声明需要 ablation / patching / steering；probe-space sufficiency 只能说明预测充分 |

## 3. 建议写入论文的方法表述

### 3.1 Fragmentation 主指标

```text
Because MISC labels are concept-level behavioral annotations rather than token-level categories, we operationalize fragmentation as predictive sufficiency rather than simple top-activating-example feature counting. Specifically, fragmentation is measured as the minimal number of SAE latents required to recover the full-candidate label-probe performance within a predefined tolerance.
```

中文含义：

```text
由于 MISC 标签是概念级行为标注，而不是 token 级类别，我们不使用简单的 top-activating-example feature counting，而将 Fragmentation 操作化为预测充分性指标：恢复完整候选池标签预测表现所需的最小 SAE latent 数量。
```

### 3.2 候选 latent 筛选

```text
We first construct a candidate latent pool for each label using precomputed label-associated candidates and the Top100 association-ranked backup latents. This candidate-recall stage limits computation while preserving both strong and distributed medium-strength label signals. Downstream effect sparsity is then evaluated only within this candidate pool.
```

中文含义：

```text
我们先为每个标签构建候选 latent 池，包括预先生成的标签关联候选和 association-rank Top100 backup latents。该候选召回阶段在控制计算成本的同时保留强信号和中等强度的分布式信号。随后只在候选池内评估 downstream effect sparsity。
```

### 3.3 因果边界

```text
The minimal sufficient latent count should be interpreted as probe-space predictive sufficiency, not causal sufficiency. Causal claims would require separate ablation, patching, or steering experiments.
```

中文含义：

```text
最小充分 latent 数量应解释为预测空间中的充分性，而不是因果充分性。若要提出因果结论，需要额外的 ablation、patching 或 steering 实验。
```

## 4. APA 风格参考文献草稿

Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C., Grosse, R., McCandlish, S., Kaplan, J., Amodei, D., Wattenberg, M., & Olah, C. (2022). *Toy models of superposition*. arXiv. https://arxiv.org/abs/2209.10652

Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Hatfield-Dodds, Z., Tamkin, A., Nguyen, K., McLean, B., Burke, J. E., Hume, T., Carter, S., Henighan, T., & Olah, C. (2023). *Towards monosemanticity: Decomposing language models with dictionary learning*. Transformer Circuits Thread. https://transformer-circuits.pub/2023/monosemantic-features/

Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). *Sparse autoencoders find highly interpretable features in language models*. arXiv. https://arxiv.org/abs/2309.08600

Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B., Pearce, A., Citro, C., Ameisen, E., Jones, A., Cunningham, H., Turner, N. L., McDougall, C., MacDiarmid, M., Freeman, C. D., Sumers, T. R., Rees, E., Batson, J., Jermyn, A., Carter, S., Olah, C., & Henighan, T. (2024). *Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet*. Transformer Circuits Thread. https://transformer-circuits.pub/2024/scaling-monosemanticity/

Gao, L., Dupré la Tour, T., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., & Wu, J. (2024). *Scaling and evaluating sparse autoencoders*. arXiv. https://arxiv.org/abs/2406.04093

Marks, S., Rager, C., Michaud, E. J., Belinkov, Y., Bau, D., & Mueller, A. (2024). *Sparse feature circuits: Discovering and editing interpretable causal graphs in language models*. arXiv. https://arxiv.org/abs/2403.19647

Chanin, D., Wilken-Smith, J., Dulka, T., Bhatnagar, H., Golechha, S., & Bloom, J. (2024). *A is for absorption: Studying feature splitting and absorption in sparse autoencoders*. arXiv. https://arxiv.org/abs/2409.14507

Karvonen, A., Rager, C., Lin, J., Tigges, C., Bloom, J., Chanin, D., Lau, Y.-T., Farrell, E., McDougall, C., Ayonrinde, K., Till, D., Wearden, M., Conmy, A., Marks, S., & Nanda, N. (2025). *SAEBench: A comprehensive benchmark for sparse autoencoders in language model interpretability*. arXiv. https://arxiv.org/abs/2503.09532

## 5. 检索与核验状态

| 文献 | 核验状态 | 备注 |
|---|---|---|
| Elhage et al. 2022 | 已核验 arXiv 页面 | arXiv:2209.10652 |
| Bricken et al. 2023 | 已核验 Anthropic 页面 | Anthropic 页面说明 512 neurons 分解为 4000+ features |
| Cunningham et al. 2023 | 已核验 arXiv 页面 | arXiv:2309.08600 |
| Anthropic / Templeton et al. 2024 | 已核验 Anthropic 页面 | Anthropic 页面说明 Claude Sonnet 中提取 millions of features 并做 feature manipulation |
| Gao et al. 2024 | 已核验 arXiv 页面 | arXiv:2406.04093；摘要明确提到 downstream effects sparsity |
| Marks et al. 2024 | 已核验 arXiv 页面 | arXiv:2403.19647；摘要明确提到 feature ablation / downstream classifier |
| Chanin et al. 2024 | 已核验 arXiv 页面 | arXiv:2409.14507；摘要明确提到 feature splitting / absorption |
| Karvonen et al. 2025 | 已核验 arXiv 页面 | arXiv:2503.09532；ICML 2025 |
