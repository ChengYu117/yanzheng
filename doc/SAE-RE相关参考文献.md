# SAE-RE 相关参考文献

> 说明：
> 1. 本文档面向当前仓库的 `SAE + RE/NonRE 概念发现` 任务整理。
> 2. 中文标题为便于内部阅读和写作的对照翻译，不是官方译名。
> 3. 参考文献优先选用原始来源页面（arXiv、OpenReview、ACL Anthology、Transformer Circuits、OpenAI、PubMed、PMC）。

---

## 一、SAE 与单义性（Monosemanticity）基础文献

### 1. Toy Models of Superposition（《叠加表征的玩具模型》）

- 参考格式：Elhage, Nelson, Tristan Hume, Catherine Olsson, et al. 2022. *Toy Models of Superposition*.
- 来源：arXiv 2209.10652 / Anthropic Research
- 相关性：解释了为什么神经网络会出现 polysemanticity / superposition，是后续用 SAE 做特征分解的理论起点。
- 链接：https://arxiv.org/abs/2209.10652

### 2. Towards Monosemanticity: Decomposing Language Models With Dictionary Learning（《迈向单义性：用字典学习分解语言模型》）

- 参考格式：Bricken, Trenton, Adly Templeton, Joshua Batson, et al. 2023. *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*.
- 来源：Transformer Circuits Thread / Anthropic
- 相关性：SAE 在语言模型里做 dictionary learning 的代表性起点工作，直接奠定“用 feature 替代 neuron 作为分析单位”的思路。
- 链接：https://transformer-circuits.pub/2023/monosemantic-features/

### 3. Sparse Autoencoders Find Highly Interpretable Features in Language Models（《稀疏自编码器在语言模型中发现高可解释特征》）

- 参考格式：Huben, Robert, Hoagy Cunningham, Logan Riggs Smith, Aidan Ewart, and Lee Sharkey. 2024. *Sparse Autoencoders Find Highly Interpretable Features in Language Models*.
- 来源：ICLR 2024 Poster / OpenReview
- 相关性：是 SAE 用于 LM residual stream 特征发现的核心实证论文之一，也和你们项目中的“单 latent 可解释性 + 因果定位”目标最接近。
- 链接：https://openreview.net/forum?id=F76bwRSLeK

### 4. Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet（《扩展单义性：从 Claude 3 Sonnet 中提取可解释特征》）

- 参考格式：Templeton, Adly, Tom Conerly, Jonathan Marcus, et al. 2024. *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*.
- 来源：Transformer Circuits Thread / Anthropic
- 相关性：证明 SAE 方法可以扩展到更大的前沿模型，并把“特征解释 + steering + 局部因果验证”结合起来。
- 链接：https://transformer-circuits.pub/2024/scaling-monosemanticity/

### 5. Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders（《Llama Scope：用稀疏自编码器从 Llama-3.1-8B 中提取数百万特征》）

- 参考格式：He, Zhengfu, Wentao Shu, Xuyang Ge, et al. 2024. *Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders*. arXiv:2410.20526.
- 来源：arXiv
- 相关性：与你们当前仓库最直接相关，因为项目使用的公开 SAE 生态和 Llama 系列 SAE 分析框架与这篇工作同一脉络。
- 链接：https://arxiv.org/abs/2410.20526

---

## 二、SAE 评估、基准与局限性文献

### 6. Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control（《迈向面向可解释性与可控性的稀疏自编码器原则化评测》）

- 参考格式：Makelov, Aleksandar, George Lange, and Neel Nanda. 2024. *Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control*. arXiv:2405.08366.
- 来源：arXiv
- 相关性：直接讨论“如何更原则地评估 SAE”，对你们现在关心的 fidelity、interpretability、control 这三类问题非常关键。
- 链接：https://arxiv.org/abs/2405.08366

### 7. SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability（《SAEBench：面向语言模型可解释性的稀疏自编码器综合基准》）

- 参考格式：Karvonen, Adam, Can Rager, Johnny Lin, et al. 2025. *SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability*. arXiv:2503.09532.
- 来源：arXiv
- 相关性：这是当前最直接的 SAE benchmark 之一，强调不同 proxy metric 和实际用途之间可能不一致，非常适合拿来校准你们项目的“达标”判断标准。
- 链接：https://arxiv.org/abs/2503.09532

### 8. Revisiting End-To-End Sparse Autoencoder Training: A Short Finetune Is All You Need（《重新审视端到端稀疏自编码器训练：短时微调已足够》）

- 参考格式：Karvonen, Adam. 2025. *Revisiting End-To-End Sparse Autoencoder Training: A Short Finetune Is All You Need*. arXiv:2503.17272.
- 来源：arXiv
- 相关性：直接关注 SAE 的 CE/KL 重构质量改进，对你们结构指标里 `CE loss / KL divergence` 的意义判断很重要。
- 链接：https://arxiv.org/abs/2503.17272

### 9. Interpretability Illusions with Sparse Autoencoders: Evaluating Robustness of Concept Representations（《稀疏自编码器中的可解释性幻觉：概念表征鲁棒性评估》）

- 参考格式：Li, Aaron J., Suraj Srinivas, Usha Bhalla, and Himabindu Lakkaraju. 2025. *Interpretability Illusions with Sparse Autoencoders: Evaluating Robustness of Concept Representations*. arXiv:2505.16004.
- 来源：arXiv
- 相关性：这篇论文非常适合给你们的报告加“反方约束”，因为它提醒我们：高可解释外观不等于鲁棒、可靠的概念表示。
- 链接：https://arxiv.org/abs/2505.16004

### 10. A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models（《稀疏自编码器综述：解释大语言模型的内部机制》）

- 参考格式：Shu, Dong, Xuansheng Wu, Haiyan Zhao, et al. 2025. *A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models*. arXiv:2503.05613.
- 来源：arXiv
- 相关性：适合作为总览文献，尤其是它对 structural / functional / robustness 三类评估框架做了系统整理。
- 链接：https://arxiv.org/abs/2503.05613

### 11. AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders（《AxBench：操控大语言模型？连简单基线都优于稀疏自编码器》）

- 参考格式：Wu, Zhengxuan, Aryaman Arora, Atticus Geiger, et al. 2025. *AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders*. ICML 2025.
- 来源：PMLR / ICML 2025
- 相关性：这篇论文对“SAE 是否真的优于简单 baseline”提出了很强挑战，与你们项目中的 `diffmean`、`dense_probe`、`sparse_probe` 对比逻辑直接相关。
- 链接：https://proceedings.mlr.press/v267/wu25a.html

---

## 三、探针、解缠评测与自动解释文献

### 12. Designing and Interpreting Probes with Control Tasks（《用控制任务设计并解释探针》）

- 参考格式：Hewitt, John, and Percy Liang. 2019. *Designing and Interpreting Probes with Control Tasks*. EMNLP-IJCNLP 2019.
- 来源：ACL Anthology
- 相关性：这是 probe 文献里的经典论文。你们项目里的 sparse probe 属于“probe-style evidence”，这篇论文提醒要谨慎解释 probe 结果，不能把“可分”直接当“概念已被定位”。
- 链接：https://aclanthology.org/D19-1275/

### 13. RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations（《RAVEL：评估语言模型表征解缠中的可解释性方法》）

- 参考格式：Huang, Jing, Zhengxuan Wu, Christopher Potts, Mor Geva, and Atticus Geiger. 2024. *RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations*. ACL 2024.
- 来源：ACL Anthology
- 相关性：如果你们后续要论证“这组 latent 是否真的在解缠 RE 相关概念”，RAVEL 提供了更接近 benchmark 思路的评估参照。
- 链接：https://aclanthology.org/2024.acl-long.470/

### 14. Language Models Can Explain Neurons in Language Models（《语言模型可以解释语言模型中的神经元》）

- 参考格式：Bills, Steven, Nick Cammarata, Dan Mossing, et al. 2023. *Language Models Can Explain Neurons in Language Models*.
- 来源：OpenAI Publication
- 相关性：这篇工作代表“自动解释器/模型代理人”路线，和你们仓库里后续 AI 专家代理评审管线的思路高度相关。
- 链接：https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html

---

## 四、RE / 动机式访谈（Motivational Interviewing）概念背景文献

### 15. Toward a Theory of Motivational Interviewing（《迈向动机式访谈的理论》）

- 参考格式：Miller, William R., and Gary S. Rose. 2009. *Toward a Theory of Motivational Interviewing*. *American Psychologist* 64(6): 527-537.
- 来源：PubMed / PMC
- 相关性：如果你们要把 SAE latent 与“RE 是什么”建立更稳的概念对照，这篇论文提供了 MI 机制层面的理论背景，尤其强调关系性成分与 change talk。
- 链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC2759607/

### 16. Reflective Listening in Counseling: Effects of Training Time and Evaluator Social Skills（《咨询中的反映式倾听：训练时长与评估者社会技能的影响》）

- 参考格式：Rautalinko, Erik, Hans-Olof Lisper, and Bo Ekehammar. 2007. *Reflective Listening in Counseling: Effects of Training Time and Evaluator Social Skills*. *American Journal of Psychotherapy* 61(2): 191-209.
- 来源：PubMed
- 相关性：这是与 RE 技能本身更直接相关的实证研究，可作为你们解释“RE 样本为何不是单一词面模式，而是会影响关系质量与情绪披露”的领域背景依据。
- 链接：https://pubmed.ncbi.nlm.nih.gov/17760322/

---

## 五、建议优先阅读顺序

如果目标是尽快服务当前项目，建议按下面顺序读：

1. `Llama Scope`
2. `Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control`
3. `SAEBench`
4. `Interpretability Illusions with Sparse Autoencoders`
5. `AxBench`
6. `Designing and Interpreting Probes with Control Tasks`
7. `Toward a Theory of Motivational Interviewing`
8. `Reflective Listening in Counseling`

---

## 六、与本项目最直接对应的映射

- 对应 SAE 来源与工程实现：
  - `Llama Scope`
  - `Scaling Monosemanticity`
  - `Towards Monosemanticity`
- 对应评估框架与“达标”判断：
  - `Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control`
  - `SAEBench`
  - `A Survey on Sparse Autoencoders`
  - `AxBench`
- 对应 probe / disentanglement / 自动解释：
  - `Designing and Interpreting Probes with Control Tasks`
  - `RAVEL`
  - `Language Models Can Explain Neurons in Language Models`
- 对应 RE 概念本身：
  - `Toward a Theory of Motivational Interviewing`
  - `Reflective Listening in Counseling`

---

## 七、后续可扩展

如果你后面要继续写正式论文或综述，下一步可以继续补三类文献：

- 模型层级因果干预：activation patching、causal tracing、steering
- 自动特征解释：feature labeling、LLM-as-judge、解释忠实度评测
- RE 数据与标注：动机式访谈编码体系、咨询对话 RE 子维度标注、治疗同盟与 change talk 相关研究
