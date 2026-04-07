# SAE-RE相关参考文献

> 说明：
> 1. 本文档面向当前仓库的 `SAE + RE/NonRE 概念发现` 任务整理。
> 2. 各条目统一采用同一模板：`功能描述 / 核心贡献 / 参考格式 / 来源 / 可访问链接 / 下载链接 / 与本项目关系`。
> 3. 若来源站点未提供官方独立 PDF，则“下载链接”字段会回退到稳定在线阅读页。
> 4. 中文标题为内部阅读用对照翻译，不是官方译名。

---

## 零、综述优先文献

### 1. A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models（《稀疏自编码器综述：解释大语言模型的内部机制》）

- 功能描述：系统综述 SAE 原理、训练策略、评估框架、常见局限与改进方向。
- 核心贡献：把 SAE 研究中的 structural / functional / robustness 三类问题梳理到同一张地图上，是总览型参考文献。
- 参考格式：Shu, Dong, Xuansheng Wu, Haiyan Zhao, et al. 2025. *A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models*. arXiv:2503.05613.
- 来源：arXiv
- 可访问链接：https://arxiv.org/abs/2503.05613
- 下载链接：https://arxiv.org/pdf/2503.05613.pdf
- 与本项目关系：适合优先建立总览框架，再回头细看结构、功能、鲁棒性和 RE 背景文献。

---

## 一、SAE 与单义性基础文献

### 2. Toy Models of Superposition（《叠加表征的玩具模型》）

- 功能描述：为 polysemanticity / superposition 提供理论起点，解释为什么神经网络内部会把多个概念压进同一组参数或方向里。
- 核心贡献：提出一套玩具模型来刻画 superposition 现象，为后续用 SAE 将混叠表示拆成更可解释特征提供理论背景。
- 参考格式：Elhage, Nelson, Tristan Hume, Catherine Olsson, et al. 2022. *Toy Models of Superposition*.
- 来源：arXiv / Anthropic
- 可访问链接：https://arxiv.org/abs/2209.10652
- 下载链接：https://arxiv.org/pdf/2209.10652.pdf
- 与本项目关系：解释了为什么 RE 相关线索不一定对应单一 neuron，而更可能需要在 feature / latent 空间里寻找。

### 3. Towards Monosemanticity: Decomposing Language Models With Dictionary Learning（《迈向单义性：用字典学习分解语言模型》）

- 功能描述：是 SAE 用于语言模型特征分解的代表性起点工作，展示如何把 language model activations 分解成更接近单义的 feature。
- 核心贡献：明确提出“用 feature 替代 neuron 作为分析单位”的研究路径，并通过 dictionary learning 展示 monosemantic feature discovery 的可行性。
- 参考格式：Bricken, Trenton, Adly Templeton, Joshua Batson, et al. 2023. *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*.
- 来源：Transformer Circuits / Anthropic
- 可访问链接：https://transformer-circuits.pub/2023/monosemantic-features/
- 下载链接：https://transformer-circuits.pub/2023/monosemantic-features/
- 与本项目关系：是你们把 `RE` 问题转成“候选 latent / feature 发现问题”的直接方法论来源。

### 4. Sparse Autoencoders Find Highly Interpretable Features in Language Models（《稀疏自编码器在语言模型中发现高可解释特征》）

- 功能描述：系统展示 SAE 在语言模型 residual stream 中可以学到较高可解释性的 feature，并尝试把这些 feature 用于更细粒度的机制定位。
- 核心贡献：把 SAE 从“能做重构”推进到“能找到可解释 feature”，并引入自动解释评分、替换干预等思路来评估 feature 质量。
- 参考格式：Huben, Robert, Hoagy Cunningham, Logan Riggs Smith, Aidan Ewart, and Lee Sharkey. 2024. *Sparse Autoencoders Find Highly Interpretable Features in Language Models*.
- 来源：ICLR 2024 / OpenReview
- 可访问链接：https://openreview.net/forum?id=F76bwRSLeK
- 下载链接：https://openreview.net/pdf?id=F76bwRSLeK
- 与本项目关系：和你们现在的“单 latent 可分性 + 解释性 + 后续干预潜力”目标最接近。

### 5. Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet（《扩展单义性：从 Claude 3 Sonnet 中提取可解释特征》）

- 功能描述：展示 SAE 路线可以扩展到更大模型，并将 feature 解释、局部因果验证和 steering 结合起来。
- 核心贡献：把 SAE 研究从“小模型可解释 feature”推进到“前沿模型上的大规模 feature 提取与验证”，并系统展示多维评估思路。
- 参考格式：Templeton, Adly, Tom Conerly, Jonathan Marcus, et al. 2024. *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*.
- 来源：Transformer Circuits / Anthropic
- 可访问链接：https://transformer-circuits.pub/2024/scaling-monosemanticity/
- 下载链接：https://transformer-circuits.pub/2024/scaling-monosemanticity/
- 与本项目关系：是“feature 提取 + 定性解释 + 干预验证”三位一体路线的典型参照。

### 6. Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders（《Llama Scope：用稀疏自编码器从 Llama-3.1-8B 中提取数百万特征》）

- 功能描述：提供大规模公开 Llama SAE 资源和配套评估框架，是当前 Llama 系列 SAE 实验的重要基础设施。
- 核心贡献：在 Llama-3.1-8B 上系统训练并发布大量 SAE，为下游概念发现、latent 筛选和 feature 研究提供现成模型与生态。
- 参考格式：He, Zhengfu, Wentao Shu, Xuyang Ge, et al. 2024. *Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders*. arXiv:2410.20526.
- 来源：arXiv
- 可访问链接：https://arxiv.org/abs/2410.20526
- 下载链接：https://arxiv.org/pdf/2410.20526.pdf
- 与本项目关系：与你们当前使用的公开 SAE 生态和 Llama 路线最直接相关。

---

## 二、SAE 评估、基准与局限文献

### 7. Scaling and Evaluating Sparse Autoencoders（《稀疏自编码器的扩展与评估》）

- 功能描述：讨论如何把 SAE 训练到极大规模，并提出一套更标准化的 feature quality / fidelity 评估框架。
- 核心贡献：提出 k-sparse SAE、规模律分析和多种 feature quality 指标，并展示极大 SAE 可以在更少 dead latent 的条件下扩展到更强模型激活。
- 参考格式：Gao, Leo, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. 2024. *Scaling and Evaluating Sparse Autoencoders*. arXiv:2406.04093.
- 来源：arXiv / OpenReview
- 可访问链接：https://arxiv.org/abs/2406.04093
- 下载链接：https://arxiv.org/pdf/2406.04093.pdf
- 与本项目关系：直接对应你们关心的 `dead_ratio`、fidelity、解释质量和大规模 SAE 评估口径。

### 8. Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control（《迈向面向可解释性与可控性的稀疏自编码器原则化评测》）

- 功能描述：专门讨论“应该如何更原则地评估 SAE”，区分 fidelity、interpretability、control 等不同维度。
- 核心贡献：指出单一 proxy metric 不足以定义 SAE 好坏，强调需要把结构指标、解释指标和控制/干预指标一起看。
- 参考格式：Makelov, Aleksandar, George Lange, and Neel Nanda. 2024. *Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control*. arXiv:2405.08366.
- 来源：arXiv
- 可访问链接：https://arxiv.org/abs/2405.08366
- 下载链接：https://arxiv.org/pdf/2405.08366.pdf
- 与本项目关系：非常适合校准你们当前“结构侧偏弱但功能侧仍有信号”的结论边界。

### 9. SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability（《SAEBench：面向语言模型可解释性的稀疏自编码器综合基准》）

- 功能描述：提供系统 benchmark，比较不同 SAE 在多种 interpretability / control / fidelity 任务上的表现。
- 核心贡献：把 SAE 评估从“各论文各自报分”推进到更统一的 benchmark 化阶段，强调不同任务目标下的 SAE 评价可能不一致。
- 参考格式：Karvonen, Adam, Can Rager, Johnny Lin, et al. 2025. *SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability*. arXiv:2503.09532.
- 来源：arXiv
- 可访问链接：https://arxiv.org/abs/2503.09532
- 下载链接：https://arxiv.org/pdf/2503.09532.pdf
- 与本项目关系：是你们后续判断“当前 SAE-RE 管线达到什么研究级别”的重要参照。

### 10. Revisiting End-To-End Sparse Autoencoder Training: A Short Finetune Is All You Need（《重新审视端到端稀疏自编码器训练：短时微调已足够》）

- 功能描述：聚焦 SAE 的行为层 fidelity 改进，研究短时 finetune 对 CE / KL 等指标的改善效果。
- 核心贡献：表明无需完全重训 SAE，也可以通过较短 finetune 改善行为保真度，为“结构弱但可修”提供实证支持。
- 参考格式：Karvonen, Adam. 2025. *Revisiting End-To-End Sparse Autoencoder Training: A Short Finetune Is All You Need*. arXiv:2503.17272.
- 来源：arXiv
- 可访问链接：https://arxiv.org/abs/2503.17272
- 下载链接：https://arxiv.org/pdf/2503.17272.pdf
- 与本项目关系：直接对应你们现在关心的 `CE loss delta` 和 `KL divergence`。

### 11. Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders（《跃迁向前：用 JumpReLU 稀疏自编码器提升重构保真度》）

- 功能描述：提出 JumpReLU SAE，重点解决“稀疏性和重构保真度之间的张力”。
- 核心贡献：用 JumpReLU 与直接 L0 稀疏训练改善 reconstruction fidelity，同时保持较强解释性，是结构指标研究中的代表性改进工作。
- 参考格式：Rajamanoharan, Senthooran, Tom Lieberum, Nicolas Sonnerat, Arthur Conmy, Vikrant Varma, János Kramár, and Neel Nanda. 2024. *Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders*. arXiv:2407.14435.
- 来源：arXiv
- 可访问链接：https://arxiv.org/abs/2407.14435
- 下载链接：https://arxiv.org/pdf/2407.14435.pdf
- 与本项目关系：与你们当前使用的 `JumpReLU` 架构直接相关，尤其适合解释重构保真和 dead feature 问题。

### 12. Empirical Evaluation of Progressive Coding for Sparse Autoencoders（《稀疏自编码器渐进编码的经验评估》）

- 功能描述：比较 progressive coding、Matryoshka SAE 与剪枝 vanilla SAE，讨论多尺度 SAE 的重构、相似性与解释性权衡。
- 核心贡献：指出不同 SAE 构造在 reconstruction、recaptured LM loss、RSA 和 interpretability 之间存在显著 trade-off，不存在单一最优方案。
- 参考格式：Peter, Hans, and Anders Søgaard. 2025. *Empirical Evaluation of Progressive Coding for Sparse Autoencoders*. arXiv:2505.00190.
- 来源：arXiv
- 可访问链接：https://arxiv.org/abs/2505.00190
- 下载链接：https://arxiv.org/pdf/2505.00190.pdf
- 与本项目关系：适合用来说明“结构指标好”和“解释性好”未必完全同向。

### 13. Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning（《用端到端稀疏字典学习识别功能上重要的特征》）

- 功能描述：不再只优化 activation reconstruction，而是直接把“功能重要性”纳入字典学习目标。
- 核心贡献：提出 end-to-end sparse dictionary learning 路线，强调 feature 应该对模型功能有贡献，而不仅仅是便于重构。
- 参考格式：Braun, Dan, Jordan Taylor, Nicholas Goldowsky-Dill, and Lee Sharkey. 2024. *Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning*. NeurIPS 2024.
- 来源：OpenReview / NeurIPS 2024
- 可访问链接：https://openreview.net/forum?id=7txPaUpUnc
- 下载链接：https://openreview.net/pdf?id=7txPaUpUnc
- 与本项目关系：非常适合用来对照你们当前“功能侧有信号，但结构侧偏弱”的状态。

### 14. Interpretability Illusions with Sparse Autoencoders: Evaluating Robustness of Concept Representations（《稀疏自编码器中的可解释性幻觉：概念表征鲁棒性评估》）
### 15. AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders（《AxBench：操控大语言模型？连简单基线都优于稀疏自编码器》）

- 功能描述：评估 SAE 在 concept detection 和 steering 任务上是否真的优于简单 baseline。
- 核心贡献：对“SAE 一定优于简单方向/简单基线”的直觉提出强挑战，是 SAE 功能侧评估的重要负向参照。
- 参考格式：Wu, Zhengxuan, Aryaman Arora, Atticus Geiger, et al. 2025. *AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders*. ICML 2025.
- 来源：PMLR / ICML 2025
- 可访问链接：https://proceedings.mlr.press/v267/wu25a.html
- 下载链接：https://proceedings.mlr.press/v267/wu25a/wu25a.pdf
- 与本项目关系：与你们当前 `diffmean`、`dense_probe`、`sparse_probe` 的比较逻辑直接相关。

---

## 三、探针、解缠评测与自动解释文献

### 16. Designing and Interpreting Probes with Control Tasks（《用控制任务设计并解释探针》）

- 功能描述：是 probe 文献中的经典工作，讨论 probe 结果应该如何解释，以及怎样避免把“可分”误当成“真正学到了概念”。
- 核心贡献：提出 control task 思路，系统提醒研究者 probe 是证据，但不是概念因果确认本身。
- 参考格式：Hewitt, John, and Percy Liang. 2019. *Designing and Interpreting Probes with Control Tasks*. EMNLP-IJCNLP 2019.
- 来源：ACL Anthology
- 可访问链接：https://aclanthology.org/D19-1275/
- 下载链接：https://aclanthology.org/D19-1275.pdf
- 与本项目关系：是解释你们 `sparse_probe` 与 `dense_probe` 结果时必须引用的约束文献。

### 17. RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations（《RAVEL：评估语言模型表征解缠中的可解释性方法》）

- 功能描述：提供更接近 benchmark 的 disentanglement 评测视角，用来判断表征是否真的被更干净地拆开。
- 核心贡献：把 interpretability 方法放到更严格的解缠任务上比较，为“这组 latent 是否真的更干净”提供参照框架。
- 参考格式：Huang, Jing, Zhengxuan Wu, Christopher Potts, Mor Geva, and Atticus Geiger. 2024. *RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations*. ACL 2024.
- 来源：ACL Anthology
- 可访问链接：https://aclanthology.org/2024.acl-long.470/
- 下载链接：https://aclanthology.org/2024.acl-long.470.pdf
- 与本项目关系：适合后续论证“候选 RE latent 子集是否真的在解缠相关概念”。

### 18. Language Models Can Explain Neurons in Language Models（《语言模型可以解释语言模型中的神经元》）

- 功能描述：代表自动解释器 / LLM-as-explainer 路线，展示如何让更强模型自动解释并评分较弱模型内部单元。
- 核心贡献：提出 explanation-simulation-scoring 流程，把自动解释从“看例子写标签”推进到“生成解释并验证解释是否能预测激活”。
- 参考格式：Bills, Steven, Nick Cammarata, Dan Mossing, Henk Tillman, Leo Gao, Gabriel Goh, Ilya Sutskever, Jan Leike, Jeff Wu, and William Saunders. 2023. *Language Models Can Explain Neurons in Language Models*.
- 来源：OpenAI Publication
- 可访问链接：https://openai.com/index/language-models-can-explain-neurons-in-language-models/
- 下载链接：https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html
- 与本项目关系：和你们仓库里后续 AI 代理评审、latent card、LLM-as-judge 路线高度相关。

---

## 四、RE / 动机式访谈（Motivational Interviewing）背景文献

### 19. Toward a Theory of Motivational Interviewing（《迈向动机式访谈的理论》）

- 功能描述：从理论层面讨论 MI 的核心机制，是理解 RE 在咨询语境中角色的重要背景文献。
- 核心贡献：系统阐释动机式访谈的作用机制，强调关系性因素、change talk 和咨询回应风格对行为改变的作用。
- 参考格式：Miller, William R., and Gary S. Rose. 2009. *Toward a Theory of Motivational Interviewing*. *American Psychologist* 64(6): 527-537.
- 来源：PubMed / PMC
- 可访问链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC2759607/
- 下载链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC2759607/
- 与本项目关系：为“RE 到底是什么”提供更稳的理论对照，避免把表层措辞误当 RE 本体。

### 20. Reflective Listening in Counseling: Effects of Training Time and Evaluator Social Skills（《咨询中的反映式倾听：训练时长与评估者社会技能的影响》）

- 功能描述：是与 reflective listening 技能本身更直接相关的实证研究，可用于补充 RE 训练与评估背景。
- 核心贡献：从训练时长和评估者因素出发，讨论 reflective listening 技能如何被观察、训练和评价。
- 参考格式：Rautalinko, Erik, Hans-Olof Lisper, and Bo Ekehammar. 2007. *Reflective Listening in Counseling: Effects of Training Time and Evaluator Social Skills*. *American Journal of Psychotherapy* 61(2): 191-209.
- 来源：PubMed / DOI
- 可访问链接：https://pubmed.ncbi.nlm.nih.gov/17760322/
- 下载链接：https://doi.org/10.1176/appi.psychotherapy.2007.61.2.191
- 与本项目关系：可作为解释 RE 为什么不是单一话术模式、而是受咨询场景和评估方式影响的背景依据。

---

## 五、建议优先阅读顺序

如果目标是尽快服务当前 SAE-RE 项目，建议按下面顺序阅读：

1. `Llama Scope`
2. `A Survey on Sparse Autoencoders`
3. `Scaling and Evaluating Sparse Autoencoders`
4. `Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control`
5. `SAEBench`
6. `Interpretability Illusions with Sparse Autoencoders`
7. `AxBench`
8. `Designing and Interpreting Probes with Control Tasks`
9. `Toward a Theory of Motivational Interviewing`
10. `Reflective Listening in Counseling`

---

## 六、与本项目最直接对应的映射

- 对应 SAE 来源与工程实现：
  - `Llama Scope`
  - `Scaling Monosemanticity`
  - `Towards Monosemanticity`
  - `Scaling and Evaluating Sparse Autoencoders`

- 对应结构与功能评估框架：
  - `Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control`
  - `SAEBench`
  - `Revisiting End-To-End Sparse Autoencoder Training`
  - `Jumping Ahead`
  - `Interpretability Illusions with Sparse Autoencoders`
  - `A Survey on Sparse Autoencoders`

- 对应 probe / disentanglement / 自动解释：
  - `Designing and Interpreting Probes with Control Tasks`
  - `RAVEL`
  - `Language Models Can Explain Neurons in Language Models`

- 对应 RE 概念背景：
  - `Toward a Theory of Motivational Interviewing`
  - `Reflective Listening in Counseling`

---

## 七、后续可继续补充的文献方向

- 模型层级因果干预：activation patching、causal tracing、steering
- 特征自动解释：feature labeling、LLM-as-judge、解释忠实度评测
- RE 数据与标注：MI 编码体系、咨询对话 RE 子维度、治疗联盟与 change talk 文献
