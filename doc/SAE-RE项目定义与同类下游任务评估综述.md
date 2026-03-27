# SAE-RE项目定义与同类下游任务评估综述

> 用途：
> 1. 从本项目现有文档中抽取稳定的研究定义。
> 2. 整理与本项目同类的 SAE-LLM 下游分析/控制任务论文。
> 3. 总结这类研究通常怎么评估、什么证据算“最低合格”、什么证据才接近“强结论”。
>
> 说明：
> - 本文档采用“混合口径”纳入文献：既收录具体概念/行为/属性分析论文，也收录直接定义这类任务评估口径的 benchmark / evaluation 论文。
> - 中文标题为阅读方便的对照翻译，不是官方译名。
> - 主 corpus 优先使用一手来源：arXiv、OpenReview、ACL Anthology、PMLR、Transformer Circuits、Anthropic/OpenAI 官方研究页。

---

## 1. 本项目的研究目标与自然语言定义

### 1.1 从现有项目文档抽取出的目标

结合本仓库现有报告、技术流程说明和论文初稿，项目的真实目标不是“训练一个 RE 分类器”，而是：

1. 在预训练 SAE 分解后的 LLM 中间表示里，定位与 `RE / NonRE` 区分显著相关的候选 latent 或小子空间。
2. 用多种证据判断这些 latent 是否只是“统计上相关”，还是已经接近“可解释、可复核、可扰动验证”的 RE 概念表示。
3. 把 SAE 当作连接“模型内部状态”和“目标概念研究问题”的中间接口，而不是把 SAE 当作最终任务模型。

### 1.2 一句话核心定义

> 本项目属于一种“目标概念/行为导向的 SAE latent 发现与验证任务”：给定某个概念的正负样本或对照样本，在预训练 SAE 中定位与该概念相关的单 latent 或子空间，并用统计、预测、解释、冗余与干预证据验证其研究价值。

### 1.3 扩展定义

更完整地说，这类任务有 5 个共同组成部分：

1. 有一个**外部目标概念/行为/属性**，如 RE、语言身份、拒答、安全相关特征、指令跟随、sycophancy、欺骗、政治倾向等。
2. 有一组**正负样本、对照样本或可检索提示集**，用于把这个概念映射回模型内部表示。
3. 在预训练 SAE 或公开 SAE 字典上，寻找与该概念对齐的**单 latent、latent 组、稀疏子空间或可操作 feature 集**。
4. 评估不是只看“能不能分”，而是同时看：
   - fidelity / approximation
   - task alignment / predictivity
   - interpretability / disentanglement
   - control / causality / robustness
5. 研究目标通常分成两层：
   - **候选特征发现**：证明模型内部确实有与目标概念相关的信号。
   - **概念表示确认**：证明这些信号是较干净、较稳健、较可干预、较可复核的概念表示。

### 1.4 这类任务与普通分类任务的区别

它和普通文本分类最关键的区别是：

- 普通分类任务关注“输出对不对”。
- 这里关注“模型内部是否存在和目标概念对齐的 feature，以及这些 feature 是否有研究意义”。

因此，这类研究不能只用一个最终分类分数收尾，通常必须额外回答：

- 这些 feature 是否可解释？
- 是否冗余或吸收？
- 是否只是 proxy / keyword / surface pattern？
- 拿掉或增强这些 feature 会不会影响行为？
- 这种影响是否能跨数据、跨模板、跨设置复现？

---

## 2. 文献纳入标准与主 corpus

### 2.1 纳入标准

本文档纳入的论文满足下列至少一条：

1. 直接把 SAE 用于某个具体概念/行为/属性的发现、解释、steering、ablation 或控制。
2. 直接提供这类 SAE 下游任务的系统 benchmark / evaluation 框架。
3. 不是标准 SAE benchmark，但直接提供与 SAE 下游任务共用的“稀疏内部表示 steering / behavior-control”评估轴，可作为近邻参照。

不计入主 corpus 的内容：

- 只讲 SAE 基础理论、但不落到下游分析/控制任务的背景论文。
- 只讲一般 probing 方法、但不涉及 SAE 的论文。
- 只讲 RE 领域概念而不涉及 SAE-LLM 内部分析的论文。

### 2.2 主 corpus 总表

本文主 corpus 共 12 篇。

| # | 论文 | 类型 | 目标概念/任务 | 一手来源 |
|---|---|---|---|---|
| 1 | Sparse Autoencoders Find Highly Interpretable Features in Language Models | 任务型 | IOI 相关 feature 发现与因果定位 | arXiv / OpenReview |
| 2 | Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet | 任务型 | 大规模 feature 提取、安全相关 feature、行为操纵 | Transformer Circuits / Anthropic |
| 3 | Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders | 资源+评估型 | 大规模公开 SAE 套件与多维评估 | arXiv |
| 4 | SAIF: A Sparse Autoencoder Framework for Interpreting and Steering Instruction Following of Language Models | 任务型 | 指令跟随机理解释与 steering | arXiv |
| 5 | SAEs Are Good for Steering -- If You Select the Right Features | 任务型 | 概念 steering，区分 input/output feature | arXiv |
| 6 | Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders | 任务型 | 语言特异 feature、ablation、steering | ACL Anthology |
| 7 | Sparse Activation Editing for Reliable Instruction Following in Narratives | 任务型 | narrative 场景下的 instruction following | arXiv |
| 8 | Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control | 评估型 | 面向具体任务的近“有监督真值” SAE 评测框架 | arXiv |
| 9 | SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability | benchmark 型 | 概念检测、解释、重构、解缠、unlearning | arXiv |
|10 | AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders | benchmark 型 | steering 与 concept detection | OpenReview / PMLR |
|11 | Revisiting End-To-End Sparse Autoencoder Training: A Short Finetune Is All You Need | 评估型 | CE/KL fidelity 改进与下游权衡 | arXiv |
|12 | Evaluating Adversarial Robustness of Concept Representations in Sparse Autoencoders | 评估型 | 对抗鲁棒性与概念表示脆弱性 | arXiv |

---

## 3. 论文卡片：这 12 篇论文分别在做什么、怎么评估

以下每篇都按同一模板整理：

- 英文标题 + 中文标题
- 论文定位
- 任务 formulation
- 数据与评估协议
- 核心结果指标
- 论文把什么当作“成功”
- 与本项目的对应关系

### 3.1 Sparse Autoencoders Find Highly Interpretable Features in Language Models（《稀疏自编码器在语言模型中发现高可解释特征》）

- 作者/年份/来源：Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, Lee Sharkey. 2023. arXiv:2309.08600. https://arxiv.org/abs/2309.08600
- 论文定位：早期代表性任务型 SAE 论文。
- 下游任务类型：`发现 + 自动解释 + 局部因果验证`
- 目标概念/行为/属性：polysemanticity 解除、可解释 feature、IOI 任务中的 counterfactual feature。
- 基础模型与 SAE：语言模型内部激活上的 SAE；强调无监督地找到比其他分解更单义、更可解释的方向。
- 数据与评估协议：
  - 自动 interpretability 评估；
  - 在 IOI 任务上做 counterfactual behavior 对应 feature 的 finer-grained causal pinpointing。
- 核心结果指标：
  - `automated interpretability`
  - `monosemanticity`
  - `counterfactual causal responsibility`
- 论文如何判断成功：
  - SAE 学到的 feature 比替代分解更可解释、更单义；
  - 能在具体任务中更细粒度地定位对行为负责的 feature。
- 与本项目的对应关系：
  - 对应你们的“candidate latent + MaxAct/解释 + TPP”链条；
  - 说明“单 latent 相关性”之外，最好还要有某种局部因果证据。

### 3.2 Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet（《扩展单义性：从 Claude 3 Sonnet 中提取可解释特征》）

- 作者/年份/来源：Adly Templeton 等. 2024. Transformer Circuits Thread. https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
- 论文定位：大规模前沿模型 SAE feature 提取与安全相关 feature 研究。
- 下游任务类型：`发现 + 自动解释 + feature neighborhood + ablation/steering + safety case study`
- 目标概念/行为/属性：
  - 普通语义概念；
  - 代码错误；
  - sycophancy；
  - scam emails；
  - manipulation / deception / power-seeking 等安全相关特征。
- 基础模型与 SAE：Claude 3 Sonnet 中层 residual stream 上的百万到千万级 feature SAE。
- 数据与评估协议：
  - 最大激活样本与 activation buckets；
  - 自动 interpretability；
  - specificity 分析；
  - feature neighbors；
  - ablation / attribution / steering；
  - safety-relevant case studies。
- 核心结果指标：
  - `#features`: 1M / 4M / 34M
  - `active features per token`
  - `explained variance`
  - `dead feature ratio`
  - `auto-interpretability`
  - `specificity`
  - `ablation / attribution / steering effect`
- 已公开的代表性结果：
  - 三个 SAE 的每个 token 平均激活 feature 数都少于 `300`；
  - SAE 重建解释了至少 `65%` 的激活方差；
  - dead feature 比例约为 `2% / 35% / 65%`（随字典规模变化）；
  - 人工激活某些 feature 会显著改变模型行为，如 scam email、sycophancy 等。
- 论文如何判断成功：
  - 不只要可解释，还要能在行为上被操纵和验证；
  - 安全相关 feature 需要有可追踪、可干预、可复现的行为效应。
- 与本项目的对应关系：
  - 对应你们的 `MaxAct + Geometry + TPP + AI judge` 方向；
  - 也说明安全/行为概念研究通常不会只停在 probe/AUC。

### 3.3 Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders（《Llama Scope：用稀疏自编码器从 Llama-3.1-8B 中提取数百万特征》）

- 作者/年份/来源：Zhengfu He 等. 2024. arXiv:2410.20526. https://arxiv.org/abs/2410.20526
- 论文定位：公开 SAE 套件 + 多维评估框架。
- 下游任务类型：`资源建设 + 评估`
- 目标概念/行为/属性：不是单一任务，而是给研究者提供在 Llama-3.1-8B 全层/全子层开展下游概念分析的基础设施。
- 基础模型与 SAE：
  - Llama-3.1-8B-Base；
  - 256 个 SAE；
  - 每层/子层；
  - 32K 与 128K feature 宽度。
- 数据与评估协议：
  - held-out `50M` token；
  - base -> longer context / instruction-tuned generalization；
  - geometry 分析；
  - feature interpretability 自动评分。
- 核心结果指标：
  - `L0-norm`
  - `explained variance`
  - `Delta LM loss`
  - `MSE`
  - `latent firing frequency`
  - `monosemanticity score`（1-5）
  - `geometry`
  - `generalizability`
- 已公开的代表性结果：
  - 用 `L0`、`explained variance`、`Delta LM loss` 作为三大主指标；
  - TopK SAE 在若干层上把 `L0` 从约 `150` 压到约 `50`，同时维持或改善 `EV` 与 `Delta LM loss`；
  - wider SAE 通常有更低 `Delta LM loss`、更高 `EV`；
  - interpretability 用 GPT-4o 进行 `1-5` monosemanticity 评分。
- 论文如何判断成功：
  - 既要在重构-稀疏 tradeoff 上表现好，也要能在 interpretability、geometry、OOD generalization 上站得住。
- 与本项目的对应关系：
  - 这是你们当前公开 SAE 来源和评价口径最直接的外部参照。

### 3.4 SAIF: A Sparse Autoencoder Framework for Interpreting and Steering Instruction Following of Language Models（《SAIF：用于解释与操控语言模型指令跟随能力的稀疏自编码器框架》）

- 作者/年份/来源：Zirui He 等. 2025. arXiv:2502.11356. https://arxiv.org/abs/2502.11356
- 论文定位：以“指令跟随”作为目标行为的 SAE 下游任务论文。
- 下游任务类型：`解释 + steering + causal check`
- 目标概念/行为/属性：instruction-following capability。
- 基础模型与 SAE：在 LLM latent activation 上用 SAE 找 instruction-relevant latents。
- 数据与评估协议：
  - 识别与指令跟随相关的 latent；
  - 比较不同层位置；
  - 测试不同指令放置位置；
  - 做 steering 并观察输出是否更符合指令。
- 核心结果指标：
  - `semantic proximity to instruction`
  - `steering performance`
  - `causal effect on behavior`
  - `cross-SAE / cross-LLM scalability`
- 论文如何判断成功：
  - 找到的 latent 既要语义上贴近 instruction，又要在行为上可操控；
  - steering 效果必须随着 feature 选择更准确而提升。
- 与本项目的对应关系：
  - 和你们的 RE 任务最像的一点是：目标不是一般分类，而是“某种高层行为能力在 SAE 空间中的对齐 feature”。

### 3.5 SAEs Are Good for Steering -- If You Select the Right Features（《如果选对特征，SAE 其实适合做 steering》）

- 作者/年份/来源：Dana Arad, Aaron Mueller, Yonatan Belinkov. 2025. arXiv:2505.20063. https://arxiv.org/abs/2505.20063
- 论文定位：回应“SAE steering 不好用”的负面结果，强调 feature selection 才是关键。
- 下游任务类型：`steering + feature scoring`
- 目标概念/行为/属性：各种需要 steering 的目标概念。
- 基础模型与 SAE：以 SAE feature 为 steering basis，但区分 input feature 和 output feature。
- 数据与评估协议：
  - 先给 feature 打分；
  - 再过滤低 output-score feature；
  - 对比 steering 成效。
- 核心结果指标：
  - `input score`
  - `output score`
  - `steering improvement`
- 已公开的代表性结果：
  - 论文报告过滤掉低 output-score feature 后，steering 可获得约 `2-3x` 改善。
- 论文如何判断成功：
  - 不是“任何激活很强的 feature 都能 steering”；
  - 成功标准是：挑出的 feature 既与输入概念相关，也真正影响输出。
- 与本项目的对应关系：
  - 对你们的启发是：`MaxAct` 高不等于 `TPP` 或行为干预会强；
  - 后续如果做更强 RE steering，最好补“input-like / output-like feature”区分。

### 3.6 Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders（《通过稀疏自编码器揭示大语言模型中的语言特异特征》）

- 作者/年份/来源：Boyi Deng 等. 2025. ACL 2025. https://aclanthology.org/2025.acl-long.229/
- 论文定位：语言身份/多语言能力分析的典型下游任务论文。
- 下游任务类型：`发现 + 新指标 + ablation + steering`
- 目标概念/行为/属性：language-specific features。
- 基础模型与 SAE：在 LLM activation 上做 SAE 分解，找单语言或强语言偏向 feature。
- 数据与评估协议：
  - 提出 `monolinguality` 新指标；
  - 做单 feature 与多 feature synergy ablation；
  - 用 SAE feature 增强 steering vector；
  - 观察是否只影响某一种语言能力。
- 核心结果指标：
  - `monolinguality`
  - `single-feature ablation effect`
  - `multi-feature synergy`
  - `language steering effectiveness`
- 论文如何判断成功：
  - 真正的 language-specific feature 应表现出“语言选择性 + 语言内能力下降 + 跨语言副作用小”。
- 与本项目的对应关系：
  - 和你们的 `Feature Absorption / Geometry / TPP` 都高度相关；
  - 它说明“一个 feature 只对一个目标概念/属性造成局部影响”是非常重要的成功标准。

### 3.7 Sparse Activation Editing for Reliable Instruction Following in Narratives（《用于叙事场景中可靠指令跟随的稀疏激活编辑》）

- 作者/年份/来源：Runcong Zhao 等. 2025. arXiv:2505.16505. https://arxiv.org/abs/2505.16505
- 论文定位：更接近真实复杂输入场景的 instruction-following steering 论文。
- 纳入说明：它不是典型“公开 SAE 字典评测论文”，但与 SAE steering 工作共享几乎同一组行为控制与副作用评估轴，因此作为近邻参照纳入。
- 下游任务类型：`steering + benchmark construction`
- 目标概念/行为/属性：复杂 narrative context 中的 instruction adherence。
- 基础模型与 SAE：利用 instruction 找 instruction-relevant neurons / sparse directions，再做 activation editing。
- 数据与评估协议：
  - 构建 `FreeInstruct` benchmark，`1212` 个样本；
  - 比较 instruction adherence 与 generation quality；
  - 强调无标注、training-free。
- 核心结果指标：
  - `instruction adherence`
  - `generation quality`
  - `benchmark performance on FreeInstruct`
- 论文如何判断成功：
  - steering 不能只让模型“更听话”，还要保持 generation quality 不明显下降。
- 与本项目的对应关系：
  - 对你们的启发是：若后续做 RE steering 或模型层干预，必须同时报告“目标行为提升”和“输出质量副作用”。

### 3.8 Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control（《迈向面向可解释性与可控性的稀疏自编码器原则化评测》）

- 作者/年份/来源：Aleksandar Makelov, George Lange, Neel Nanda. 2024. arXiv:2405.08366. https://arxiv.org/abs/2405.08366
- 论文定位：任务锚定的评估论文。
- 下游任务类型：`evaluation framework`
- 目标概念/行为/属性：IOI 任务上的 feature dictionaries。
- 基础模型与 SAE：
  - GPT-2 Small；
  - 比较 supervised dictionaries 与 unsupervised SAEs；
  - 在 approximation / control / interpretability 三轴上对照。
- 数据与评估协议：
  - 使用 supervised dictionary 作为近“真值”参照；
  - 再衡量无监督 SAE 在相同任务轴上的表现。
- 核心结果指标：
  - `approximation`
  - `control`
  - `interpretability`
  - 以及两种失败模式：`feature occlusion`、`feature over-splitting`
- 论文如何判断成功：
  - 不是只要 feature 看起来可解释；
  - 还要能有效控制任务行为，并与更强参照字典相比不过度退化。
- 与本项目的对应关系：
  - 它是你们“候选发现 vs 概念确认”区分的最好方法论依据之一。

### 3.9 SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability（《SAEBench：面向语言模型可解释性的稀疏自编码器综合基准》）

- 作者/年份/来源：Adam Karvonen 等. 2025. arXiv:2503.09532. https://arxiv.org/abs/2503.09532
- 论文定位：当前最重要的 SAE 综合 benchmark 之一。
- 下游任务类型：`benchmark`
- 目标概念/行为/属性：不是单一概念，而是系统比较 SAE 是否对下游可解释性任务真的有用。
- 数据与评估协议：
  - 跨 `200+` SAEs、`8` 种架构/训练方法；
  - 组织成四个能力轴：
    - concept detection
    - interpretability
    - reconstruction
    - feature disentanglement
- 核心结果指标：
  - `sparsity-fidelity curve`
  - `loss recovered`
  - `automated interpretability`
  - `sparse probing`
  - `feature absorption`
  - `RAVEL`
  - `unlearning`
  - `spurious correlation removal`
  - `targeted probe perturbation`
- 已公开的关键结论：
  - proxy metrics 的提升并不稳定转化为下游实用表现；
  - Matryoshka SAE 在 proxy 上未必最强，但在 disentanglement 上可能更强。
- 论文如何判断成功：
  - 好 SAE 必须在多轴上同时站得住，不能只靠单一 unsupervised proxy。
- 与本项目的对应关系：
  - 你们当前的 `probe + MaxAct + absorption + geometry + TPP + CE/KL` 框架，和 SAEBench 的理念高度一致。

### 3.10 AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders（《AxBench：操控大语言模型？连简单基线都优于稀疏自编码器》）

- 作者/年份/来源：Zhengxuan Wu 等. 2025. ICML 2025 Spotlight. OpenReview: https://openreview.net/forum?id=K2CckZjNy0
- 论文定位：representation-based control 的统一 benchmark。
- 下游任务类型：`benchmark`
- 目标概念/行为/属性：steering 与 concept detection。
- 基础模型与 SAE：
  - Gemma-2 `2B` / `9B`
  - 对比 prompting、finetuning、SAE、LAT、probe、DiffMean、ReFT-r1 等。
- 数据与评估协议：
  - 两大轴：
    - `concept detection`
    - `model steering`
  - 提供 benchmark harness 与 leaderboard。
- 核心结果指标：
  - `steering score`
  - `concept detection score`
  - 可扩展到 imbalanced concept detection
- 已公开的代表性结果：
  - steering 上 prompting 最强，finetuning 次之；
  - concept detection 上 representation-based 方法（如 DiffMean）表现最佳；
  - GitHub leaderboard 中 SAE 的 steering 平均分约 `0.165`，过滤 feature 后约 `0.508`，Prompt 约 `0.894`。
- 论文如何判断成功：
  - 真正实用的表示方法必须在 steering 和 detection 两轴都经得住比较。
- 与本项目的对应关系：
  - 直接提醒你们：`DiffMean` 很强时，不能轻易把 SAE probe 的高分解释成“概念已被 cleanly isolated”。

### 3.11 Revisiting End-To-End Sparse Autoencoder Training: A Short Finetune Is All You Need（《重新审视端到端稀疏自编码器训练：短时微调已足够》）

- 作者/年份/来源：Adam Karvonen. 2025. arXiv:2503.17272. https://arxiv.org/abs/2503.17272
- 论文定位：fidelity/behavior-preservation 方向的关键评估论文。
- 下游任务类型：`evaluation`
- 目标概念/行为/属性：不是特定外部概念，而是“SAE 重建插回模型后，行为损失是否下降”。
- 数据与评估协议：
  - 比较单纯 MSE 训练与 KL+MSE end-to-end 或短时 finetune；
  - 看 cross-entropy loss gap 和 SAEBench 下游表现。
- 核心结果指标：
  - `cross-entropy loss gap`
  - `KL+MSE finetune effect`
  - `supervised SAEBench metrics`
- 已公开的代表性结果：
  - 简短 KL+MSE finetune 可把 `CE loss gap` 降低约 `20-50%`；
  - 但 SAEBench supervised metrics 改善是 mixed results，而不是全线提升。
- 论文如何判断成功：
  - 更好的 fidelity 值得要，但它不自动等于更好的 interpretability；
  - 结构指标改进后仍要回到下游任务指标验证。
- 与本项目的对应关系：
  - 这正对应你们报告里“CE/KL 很重要，但它也不能单独证明概念解释成功”。

### 3.12 Evaluating Adversarial Robustness of Concept Representations in Sparse Autoencoders（《评估稀疏自编码器中概念表征的对抗鲁棒性》）

- 作者/年份/来源：Aaron J. Li, Suraj Srinivas, Usha Bhalla, Himabindu Lakkaraju. 2025/2026 revision. arXiv:2505.16004. https://arxiv.org/abs/2505.16004
- 论文定位：鲁棒性挑战论文。
- 备注：项目内部早期文档曾以“Interpretability Illusions ...”表述其方向；当前 arXiv 标题已更新为上面的正式标题。
- 下游任务类型：`robustness evaluation`
- 目标概念/行为/属性：SAE concept representation 对输入扰动的稳定性。
- 数据与评估协议：
  - 把 robustness 定义成输入空间优化问题；
  - 设计 realistic adversarial perturbations 操纵 SAE 表征；
  - 同时观察 base LLM activation 是否明显变化。
- 核心结果指标：
  - `adversarial robustness`
  - `concept representation fragility`
  - `base activation preservation under perturbation`
- 已公开的关键结论：
  - 很小的输入扰动就能操纵 SAE concept interpretation；
  - 但 base LLM activation 可能几乎不明显变化；
  - 说明仅凭可解释外观做监控/监督并不稳。
- 论文如何判断成功：
  - 好的 concept representation 必须对合理输入扰动有一定稳健性；
  - 否则只能算“脆弱解释”，不适合模型监督与安全用途。
- 与本项目的对应关系：
  - 直接对应你们报告中“还缺少改写稳健性/held-out 因果验证”的弱点。

---

## 4. 这类下游任务常见的结果指标，到底在测什么

这一节不再按论文顺序，而按 4 条评价主线来整理。

### 4.1 Fidelity / Approximation

| 指标 | 一般是什么意思 | 常见参数/口径 | 高低如何解释 | 适用场景 |
|---|---|---|---|---|
| `MSE` | SAE 重建与原激活的均方误差 | token-level 或 activation-level | 越低越好，但不等于可解释更强 | 几乎所有 SAE 论文 |
| `explained_variance` / `EV` | 重建保留了多少原始方差 | 常与 `L0` 一起报告 | 越高越好 | Llama Scope、Scaling 等 |
| `FVU` | 未解释方差比例 | 与 EV 互补 | 越低越好 | 结构评估 |
| `Delta LM loss` / `loss delta` | 把 SAE 重建插回模型后，语言模型损失涨多少 | token CE、held-out text | 越低越好 | Llama Scope、你们项目 |
| `CE loss gap` / `loss recovered` | SAE 重建后相对原模型的行为损失偏移 | 常和 zero ablation 比 | 越接近原模型越好 | Revisiting、SAEBench |
| `KL divergence` | 原 logits 分布与 SAE 重建后 logits 的偏移 | logit-level | 越低越好 | 你们项目、end-to-end SAE 方向 |

这类指标一般怎么评估：

1. 用 held-out token 或 held-out prompt。
2. 插入 SAE reconstruction，而不是只看静态重建误差。
3. 同时报告 `sparsity`，避免拿近乎 dense 的表示换 fidelity。

这一轴最低合格的标准：

- 至少有一个“模型行为层”的 fidelity 指标，而不只是静态 MSE。

更强的标准：

- 有 held-out `CE/KL/loss recovered`；
- 并且 fidelity 提升不会显著牺牲解释性或下游 task alignment。

### 4.2 Task Alignment / Predictivity

| 指标 | 一般是什么意思 | 常见参数/口径 | 高低如何解释 | 适用场景 |
|---|---|---|---|---|
| `accuracy` | 用 top latent/子空间能否区分目标概念 | 常见于 sparse probe、concept detection | 越高越强，但可能只是表面 proxy | 你们项目、SAEBench、AxBench |
| `f1` | 在类别不平衡时更稳健的分类指标 | binary / macro / micro | 越高越好 | 稀疏 probe |
| `AUC` | 排序式区分能力 | binary AUC | 越高越强，但仍是相关性证据 | 你们项目 |
| `sparse probing` | 少量 latent 是否已携带主要任务信息 | `top-k`, probe accuracy/AUC | probe 强说明信息集中 | 你们项目、SAEBench |
| `concept detection score` | 某种 feature/direction 是否检测到目标概念 | 依 benchmark 而定 | 越高越好 | AxBench |
| `DiffMean baseline` | 不训练复杂 probe，只看平均差方向 | 单方向基线 | 强则说明任务可能本来就容易线性分开 | 你们项目、AxBench |

这类指标一般怎么评估：

1. 先有明确的正负样本或概念集合。
2. 再比较：
   - sparse probe
   - dense baseline
   - simple baseline（如 DiffMean）
3. 更好的论文还会比较不同 `k`、不同层、不同池化、不同数据切分。

这一轴最低合格的标准：

- 至少证明一小组 latent 比随机/弱基线更有信息量。

更强的标准：

- 与 dense baseline 的差距可解释；
- 与 simple baseline 的差距可解释；
- 有 held-out 或 OOD setting 下的稳定性。

必须明确的限制：

- 仅有 `probe/AUC/accuracy`，不足以确认“概念已经被 cleanly 表示出来”。

### 4.3 Interpretability / Disentanglement

| 指标 | 一般是什么意思 | 常见参数/口径 | 高低如何解释 | 适用场景 |
|---|---|---|---|---|
| `MaxAct` / top activating examples | 某 latent 高激活时到底在看什么 | top-N contexts | 便于人工判断，但易被 surface pattern 误导 | 你们项目、几乎所有 SAE 分析 |
| `monosemanticity score` | feature 是否单义、稳定、可描述 | 常见 1-5 rubric | 越高越“像概念” | Llama Scope、Anthropic 系列 |
| `auto-interpretability` | LLM judge 是否能用 feature description 正确预测激活样本 | judge accuracy / score | 越高越好 | SAEBench、OpenAI/Anthropic 路线 |
| `Feature Absorption` | 目标特征不亮时，近邻 feature 是否代偿 | mean/full absorption | 越低越好 | 你们项目、SAEBench |
| `Feature Geometry` | decoder 向量是否彼此重合 | cosine / \|cosine\| | 低表示更分散，高表示可能冗余 | 你们项目、Llama Scope |
| `RAVEL` | 干预某属性时，其他属性是否被污染 | cause + isolation | 越高越好 | SAEBench |
| `monolinguality` | 某 feature 是否偏向单语言 | 新指标 | 越高越“语言特异” | ACL 2025 multilingual SAE |

这类指标一般怎么评估：

1. 先做 feature-level qualitative inspection。
2. 再用自动 judge 或 benchmark 指标量化。
3. 最后检查冗余、吸收、过度拆分、邻域重叠。

这一轴最低合格的标准：

- 至少能给出一批可读样本，并说明这些样本不是完全随机噪声。

更强的标准：

- 有自动 interpretability / judge 量化；
- 有 disentanglement 指标；
- 有 feature-level failure mode 分析。

必须明确的限制：

- 仅有 `MaxAct` 或自动解释，不足以确认概念真的忠实。

### 4.4 Control / Causality / Robustness

| 指标 | 一般是什么意思 | 常见参数/口径 | 高低如何解释 | 适用场景 |
|---|---|---|---|---|
| `TPP` / `accuracy_drop` | 去掉某 latent 后任务表现降多少 | probe-space / model-space | 掉得越多，说明贡献越大 | 你们项目、SAEBench |
| `ablation effect` | feature 被置零后，目标行为是否下降 | single / multi-feature | 越有选择性越好 | Scaling、multilingual SAE |
| `steering effect` | 激活某 feature 后，输出是否朝目标方向变化 | adherence, steering score | 越强越好，但要看副作用 | SAIF、AxBench、Activation Editing |
| `side-effect / isolation` | 操纵某属性时，其他能力是否被污染 | isolation, quality retention | 越小越好 | RAVEL、narrative steering |
| `generalizability` | 换上下文长度、换模型设定后是否仍有效 | base -> instruct, short -> long | 越稳定越好 | Llama Scope |
| `adversarial robustness` | 微小输入扰动能否骗过概念解释 | attack success / fragility | 越稳越好 | adversarial robustness paper |

这类指标一般怎么评估：

1. 做 single-feature 或 group-level 干预。
2. 同时看目标指标和副作用。
3. 最好在 held-out、OOD 或 adversarial setting 下复查。

这一轴最低合格的标准：

- 至少有某种局部干预证据。

更强的标准：

- 干预在 held-out 数据上有效；
- 干预不只是 in-sample probe-space；
- 同时报告副作用、质量退化和稳健性。

必须明确的限制：

- 仅有 in-sample 干预，不足以支撑强因果结论。

---

## 5. 一般如何评估这种下游任务的结果

综合这些论文，可以把评估流程概括为 5 步：

1. **先证明有信号**
   - 单 latent 统计差异
   - concept detection
   - sparse probe
2. **再证明信号不是完全表面**
   - dense baseline / simple baseline
   - MaxAct / judge
   - monosemanticity / monolinguality
3. **再证明 feature 不是乱挤在一起**
   - absorption
   - geometry
   - RAVEL / disentanglement
4. **再证明 feature 对行为真的有贡献**
   - TPP
   - ablation
   - steering
5. **最后证明这些结果稳健**
   - held-out
   - OOD
   - longer contexts
   - adversarial robustness
   - side-effect / generation quality

如果一项研究只停在第 1 步，通常只能说：

> “模型内部存在和目标概念相关的候选特征。”

如果做到第 2-3 步，可以说：

> “这些候选特征开始具有解释价值，但还未必构成稳定概念表示。”

如果做到第 4-5 步，才更接近：

> “这是一组强效、稳健、可干预、较可信的概念表示。”

---

## 6. 什么样的结果才算“合格的研究”

这里给出一个三档判断框架。

### 6.1 最低合格：候选特征发现型研究

满足下列大部分条件即可：

- 有明确目标概念和对照样本
- 有 sparse probe / concept detection / 单 latent 差异证据
- 有至少一种 baseline
- 有 MaxAct 或自动解释
- 能证明“确实不是完全随机噪声”

这类研究可以支持的结论：

> “我们发现了和目标概念相关的 SAE 候选 latent / 子空间。”

不能支持的结论：

> “我们已经确认了强概念表示。”

### 6.2 较强研究：概念表示候选型研究

除了上面的条件，还应尽量满足：

- 有 `absorption / geometry / disentanglement` 一类指标
- 有局部干预，如 TPP、ablation 或 steering
- 有 held-out 或 cross-setting 复查
- 有对 baseline 的明确解释：为什么不是 prompt、DiffMean 或 dense activation 在起主要作用

这类研究可以支持的结论：

> “我们找到了一组较强、较可解释、较可干预的概念候选 feature。”

### 6.3 强结论研究：稳健概念表示确认型研究

除了上面条件，通常还要再满足：

- 干预不只是 in-sample，而是 held-out / OOD 有效
- 报告 steering / ablation 的副作用和隔离性
- 对抗扰动、模板改写、语言改写或数据切分后仍然站得住
- 最好有更接近 ground-truth 的任务锚点或监督参照

这类研究才更接近支持：

> “我们已确认一组强效、典型、稳健、因果上较可信的概念表示。”

---

## 7. 用这套标准回看本项目，当前最接近哪一档

把本项目现有结论放回上述框架中：

- 你们已经有：
  - 单 latent 显著性筛选
  - sparse probe / dense probe / diffmean
  - MaxAct
  - absorption
  - geometry
  - TPP
  - CE/KL
- 你们还欠缺：
  - held-out 干预
  - 改写稳健性
  - 输出层 steering / side-effect 评估
  - 更强的概念纯度或 judge-based 忠实性验证

因此本项目当前最稳妥的定位是：

> 已达到“候选特征发现型研究”的上沿，开始进入“概念表示候选型研究”，但还没有达到“强结论研究”。

换句话说：

- 可以合理说：`RE 信息存在于 SAE latent 空间中，并且一组候选特征能够较强地捕捉它`
- 暂不宜说：`我们已经确认了强效、典型、稳健、因果可信的 RE 概念表示`

---

## 8. 对本项目最有借鉴价值的外部结论

### 8.1 对“定义”最有帮助的结论

- `Towards Principled Evaluations...`
  - 强调要把“概念发现”和“概念控制/确认”分开。
- `SAEBench`
  - 强调 proxy metric 不等于下游解释任务表现。

### 8.2 对“指标设计”最有帮助的结论

- `Llama Scope`
  - 提供 `L0 + EV + Delta LM loss + interpretability + geometry + generalization` 的组合。
- `SAEBench`
  - 提供 `sparse probing + automated interpretability + feature absorption + RAVEL + unlearning + TPP` 的组合。

### 8.3 对“不要过度解读”最有帮助的结论

- `AxBench`
  - 简单 baseline 可能比 SAE 更强。
- `Evaluating Adversarial Robustness...`
  - 看起来像概念的解释，可能对输入微扰很脆弱。
- `SAEs Are Good for Steering...`
  - 激活强不等于 steering 有用，feature 选择非常关键。

---

## 9. 补充参考：不计入主 corpus，但对本项目仍重要

这些论文不计入上面的 12 篇主 corpus，但对写作和方法判断仍有帮助：

1. `Designing and Interpreting Probes with Control Tasks`
   - 用来约束如何解释 probe 结果。
2. `RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations`
   - 用来理解 disentanglement 的更强评估口径。
3. `Language Models Can Explain Neurons in Language Models`
   - 用来理解自动解释器和 LLM-as-judge 路线。
4. `Toward a Theory of Motivational Interviewing`
5. `Reflective Listening in Counseling`
   - 这两篇是 RE 概念本体背景，不是 SAE 任务论文，但对定义 RE 子维度很重要。

---

## 10. 一句话结论

> 与本项目最相似的 SAE-LLM 下游任务，不是一般分类任务，而是一类“目标概念/行为导向的内部表示发现与验证任务”。在这类任务里，真正合格的研究通常至少要同时交代：信号是否存在、feature 是否可解释、是否冗余、是否能被干预、以及这种干预是否稳健。
