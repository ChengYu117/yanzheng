# 面向反射性倾听概念发现的 SAE-RE 评估管线：工程实现与实验设计初稿

## 摘要

反射性倾听（Reflective Listening, RE）是动机式访谈与心理支持场景中的关键对话能力，但现有研究通常从分类性能或生成质量出发，较少直接分析大语言模型内部是否存在与 RE 概念对齐的可解释特征。本文围绕这一问题，设计并实现了一条面向概念发现的 SAE-RE 评估管线。该管线以本地部署的 `Meta-Llama-3.1-8B` 为基底模型，以 `OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x` 中 `Llama3_1-8B-Base-L19R-8x` 的预训练稀疏自编码器（SAE）为特征分解器，在 layer 19 residual stream 上对 `re_dataset.jsonl` 与 `nonre_dataset.jsonl` 进行端到端分析。工程上，本文重点解决了跨仓库模型整合、远程 SAE 权重映射、dtype 对齐、长批次激活流式提取、token 到 utterance 的特征聚合，以及结构性与功能性评估指标的统一编排问题。实验设计上，本文构建了以重构保真度、稀疏性、单 latent 区分度、稀疏 probe、MaxAct、feature absorption、feature geometry 和 targeted probe perturbation 为核心的多阶段评估框架。基于当前稳定运行的实验结果，系统在 1598 条 utterance 上完成了结构性评估与候选 latent 排序，得到 `MSE=4.58`、`cosine similarity=0.8088`、`dead ratio=90.05%`，并筛出 1540 个在 BH-FDR 条件下显著的候选 latent。结果表明，该 SAE 中确实存在一批与 RE/NonRE 区分显著相关的稀疏特征；同时，当前工程实现也揭示出高稀疏与低重构保真之间的明显张力。本文的贡献不在于宣称已经完成最终的心理语言学结论，而在于提供一条可复用、可审计、可逐步扩展的 SAE 概念分析工程方案，并给出围绕 RE 概念开展后续机制解释研究的实验设计基础。

**关键词**：稀疏自编码器；反射性倾听；大语言模型解释；机制可解释性；工程管线；实验设计

## 1. 引言

随着大语言模型在心理咨询、对话支持与教育反馈等场景中的应用不断扩展，研究者越来越关心模型是否真正形成了可解释、可干预、可验证的高层语义特征，而不仅仅是输出层面的表面拟合。对于反射性倾听这类具有明确交互目标和话语风格的概念，仅通过黑盒分类结果很难回答一个更深层的问题：模型内部是否存在与该概念稳定对齐的表示单元。

稀疏自编码器（Sparse Autoencoders, SAEs）为这一问题提供了可行路径。已有工作表明，SAE 可以将模型中高维、混叠的激活分解为更稀疏、更具可分析性的 latent feature，从而为概念级解释、特征检索和机制定位提供中间层接口 [1]。围绕 Llama 3.1 系列模型，Llama Scope 项目进一步提供了覆盖多个层与子层位置的大规模预训练 SAE，为研究者在开源模型上进行特征分析提供了工程基础 [3]。

本文以 RE 概念为例，面向一个实际研究项目回答两个问题。第一，如何把本地基础模型、远程 SAE 参数、领域数据集和多指标评估组合成一条可重复执行的分析管线。第二，在这一管线下，如何设计实验以筛出可能与 RE 对齐的候选 latent，并为后续定性解释与因果验证留下接口。

与以往直接讨论 SAE 理论或单一评估指标的工作不同，本文的重点在于**整个工程系统如何被构建出来，以及整个实验如何被组织与约束**。围绕这一目标，本文的主要贡献可以概括为三点：

1. 设计并实现了一条从本地 `Llama-3.1-8B` 激活提取、远程 SAE 对接到 RE 概念筛选的端到端工程管线。
2. 给出一套面向 RE 概念发现的多阶段实验设计，将结构性指标与功能性指标统一到同一执行框架中。
3. 基于当前稳定产出的实验结果，展示该系统已能完成结构评估与候选 latent 检索，并据此分析系统的能力边界与后续优化方向。

## 2. 研究问题与总体思路

本文关注的核心问题不是“模型能否分类 RE 与 NonRE”，而是“模型第 19 层 residual stream 经 SAE 分解后，是否出现与 RE 概念稳定相关的典型稀疏特征”。为回答该问题，我们将整体研究拆分为四个层级：

1. **表示层**：从本地基底模型提取指定层的 residual activation。
2. **分解层**：使用预训练 SAE 将 dense activation 编码为 sparse latent。
3. **聚合层**：将 token-level latent 转换为 utterance-level 表示，使其与样本标签对齐。
4. **评估层**：通过结构性与功能性指标筛选、解释并验证与 RE 相关的候选 latent。

这样的设计使项目不再把 SAE 当作一个独立模型，而是把它作为连接“模型内部状态”与“概念级研究问题”的桥梁。具体来说，研究对象被限定为数据中的 `unit_text`，因此本文研究的是**utterance 级别的 RE 风格与语义线索**，而不是更复杂的多轮对话策略或会话级机制。

## 3. 系统架构与工程实现

### 3.1 基础模型与 SAE 配置

系统使用本地 `Meta-Llama-3.1-8B` 作为基础模型，配置中采用 `float16` 与 `device_map=auto`。与之配套的 SAE 来自 `OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x` 仓库下的 `Llama3_1-8B-Base-L19R-8x` 子目录，对应 layer 19 residual post 位置。当前 SAE 配置的关键超参数如下：

| 配置项 | 数值 |
|---|---:|
| hook point | `blocks.19.hook_resid_post` |
| `d_model` | 4096 |
| `d_sae` | 32768 |
| activation | JumpReLU |
| threshold | 0.52734375 |
| norm | dataset-wise |
| `max_seq_len` | 128 |
| `batch_size` | 8 |
| aggregation | max |
| `fdr_alpha` | 0.05 |
| `probe_k_values` | [1, 5, 20] |
| `top_k_candidates` | 50 |

这一配置体现了本文的两个核心判断。其一，研究以中后层 residual stream 为主，因为该位置更可能承载语义与风格混合表征。其二，选择 8x expansion、32K latent 的 SAE，是在解释粒度和运行成本之间取得平衡。

### 3.2 数据入口与任务单位

实验数据来自两个 JSONL 文件：`re_dataset.jsonl` 与 `nonre_dataset.jsonl`。当前实现中仅取每条记录的 `unit_text` 作为模型输入，不引入更长的对话上下文、人工 rationale 或额外行为标签。两类样本规模均为 799 条，总计 1598 条 utterance。

这种数据设计有明显的工程优势：输入简单、标签清晰、便于与 utterance-level latent 对齐；但同时也带来研究边界，即系统分析的是“单句文本是否呈现 RE 特征”，而不是“完整对话中 RE 策略如何逐步展开”。

### 3.3 远程 SAE 权重接入

为了复用 OpenMOSS 发布的 SAE 参数，系统实现了一个独立的 SAE 加载模块。该模块负责：

1. 下载 `hyperparams.json`；
2. 下载并合并 `safetensors` 权重；
3. 将 checkpoint 键名映射到本地 SAE 模型的参数名；
4. 对 `W_enc` 与 `W_dec` 做 shape 检查与必要转置；
5. 使用 `strict=True` 加载 state dict。

这一策略的意义在于把“能跑起来”与“加载正确”区分开。对于解释性研究而言，部分随机初始化或静默缺失的 SAE 权重会直接污染后续 latent 分析，因此工程上必须以严格加载取代宽松容错。

### 3.4 Hook 与流式激活提取

系统通过 forward hook 从基础模型第 19 层获取 residual post activation。由于全量 token-level latent 的张量规模极大，如果直接保留 `[N, T, d_sae]` 级别的结果，会迅速超过单卡显存与主存可承受范围。因此，本文采用**流式处理（streaming）**设计：

1. 按 batch 对文本做 tokenize；
2. 运行基础模型并获取当前 batch 的 residual activation；
3. 立即将 activation cast 到 SAE 的 dtype；
4. 立即执行 SAE 编码与重建；
5. 立即在当前 batch 内完成 token-to-utterance 聚合；
6. 仅保存 utterance-level SAE features、utterance-level raw activations 与少量结构评估样本。

这一路径将内存中常驻的结果压缩到 `[N, d_sae]` 和 `[N, d_model]` 两个层级，从而使 8B 量级基础模型与 32K latent SAE 的联动分析在单机环境中成为可能。

### 3.5 dtype 对齐与数值稳定性

跨仓库集成时最容易被忽略的问题之一是 dtype 不一致。基础模型按 `float16` 加载，而 SAE 可能依据权重和配置在 `bfloat16` 或其他 dtype 上运行。若 residual activation 直接送入 SAE，很容易在首个线性层发生类型不匹配或隐式转换带来的不稳定。因此，系统在激活提取与干预计算路径中显式执行了两步对齐：

1. 从 SAE 读取其实际 dtype；
2. 在送入 SAE 前将 residual activation cast 到相同 dtype。

这一步是整条管线稳定运行的前提条件，也是工程实现区别于“概念演示代码”的关键细节之一。

### 3.6 Token 到 Utterance 的聚合

由于标签位于 utterance 级别，而 SAE 输出位于 token 级别，系统必须完成粒度对齐。当前实现提供 `max` 与 `mean` 两种聚合方式，默认使用 `max`。对于每个 latent，系统取该 utterance 中所有有效 token 的最大激活值作为句级特征。这样做的直觉是：只要某个句子中出现一次强 RE 相关模式，该 latent 就应在句级表示中被保留下来。

对应地，系统会生成两类中间产物：

1. `utterance_features`：维度 `[N, d_sae]`，用于功能性评估；
2. `utterance_activations`：维度 `[N, d_model]`，用于 dense probe baseline。

## 4. 实验设计

### 4.1 设计原则

本文的实验设计遵循三个原则。

第一，**先验证 SAE 是否按预期工作，再讨论 latent 是否可解释**。因此结构性指标必须先于功能性指标。

第二，**先做宽筛选，再做深分析**。在 32768 个 latent 中直接开展逐一人工解释是不现实的，因此需要用单 latent 统计与 probe 先缩小候选空间。

第三，**同时保留正向 RE 特征与反向 NonRE 特征**。一个有意义的概念空间不只包含“支持 RE”的 latent，也可能包含“违背 RE”的对偶 latent。

### 4.2 结构性评估

结构性评估用于回答“SAE 是否保留了基础模型该层的主要信息”。当前设计包含以下指标：

1. **重构保真度**：MSE、cosine similarity、explained variance、FVU；
2. **稀疏性**：L0 mean、L0 std；
3. **特征使用情况**：dead feature count、dead ratio、top firing features；
4. **输出分布偏移**：CE loss delta 与 KL divergence（设计中包含，但本次稳定实验使用 `--skip-ce-kl` 暂未执行）。

出于内存控制考虑，结构性指标目前采用 sample-based 估计，只保留前若干 batch 的 token-level 样本用于统计。因此这些数字更适合被解释为“结构状态的近似观察”，而不是全量 token 的严格总体估计。

### 4.3 功能性评估

功能性评估用于回答“哪些 latent 与 RE 概念相关，以及这种相关性是否具有解释与验证价值”。本文设计了六个功能模块：

1. **单 latent 统计筛选**：对每个 latent 计算 Cohen’s d、ROC-AUC、t-test 与 BH-FDR，形成 `candidate_latents.csv`。
2. **Sparse Probing**：按 `|Cohen’s d|` 排序选取 top-k latent，训练稀疏线性 probe，并与 dense probe 与 diff-mean baseline 比较。
3. **MaxAct Analysis**：抽取候选 latent 的高激活 utterance，用于人工判断该 latent 的语义内涵。
4. **Feature Absorption**：检测目标 latent 不激活时，邻近 latent 是否替代性激活，从而判断概念是否被分散编码。
5. **Feature Geometry**：比较候选 latent 在 decoder 空间中的余弦相似度，检测冗余或 feature splitting。
6. **TPP**：在 probe 空间逐个置零候选 latent，考察 probe 精度下降幅度，获得局部因果证据。

这套设计兼顾了统计显著性、任务有效性、定性可解释性与局部因果验证四个层面。对 RE 概念而言，它既能回答“是否有相关 latent”，也能回答“这些 latent 是否真在帮助区分 RE”。

### 4.4 输出物设计

为了支持重复实验与后续人工分析，系统将结果拆成多类产物：

1. `metrics_structural.json`：结构性指标汇总；
2. `candidate_latents.csv`：候选 latent 排名表；
3. `metrics_functional.json`：功能性指标汇总；
4. `latent_cards/*.md`：MaxAct 卡片，便于研究者逐个审阅。

这种设计避免把所有分析逻辑锁死在一个脚本输出中，而是把“数值评估”“候选检索”“人工解释”解耦成可以独立复查的对象。

## 5. 当前实验结果

### 5.1 运行设置

当前稳定跑通的主实验在 `qwen-env` 环境下执行，使用单卡 CUDA 与 `run_sae_evaluation.py --skip-ce-kl --output-dir outputs/sae_eval_20260310_skipcekl`。该实验完成了以下阶段：

1. 本地基础模型加载；
2. SAE 下载与权重接入；
3. streaming 激活提取；
4. utterance-level 聚合；
5. 结构性评估；
6. 单 latent 统计筛选。

目前尚未稳定完成全部功能性模块，因此本文将这批结果定位为**工程验证与候选发现阶段的初步实验结果**。

### 5.2 结构性结果

当前 `metrics_structural.json` 给出的主要结果如下：

| 指标 | 数值 |
|---|---:|
| MSE | 4.5804 |
| cosine similarity | 0.8088 |
| explained variance | 0.0682 |
| FVU | 0.9318 |
| L0 mean | 172.58 |
| L0 std | 506.14 |
| dead count | 29507 |
| dead ratio | 90.05% |
| alive count | 3261 |

这些结果反映出两个明显特征。首先，该 SAE 的表示非常稀疏，平均仅有少量 latent 在样本中活跃，且绝大部分 latent 在当前样本子集上处于不活跃状态。其次，尽管重构方向相似度较高，方差解释率仍然偏低，说明该 SAE 更像是在保留一部分稳定方向信息，而不是高保真复现原始 residual stream。

对于本文的研究目标，这一现象既是机会也是约束。机会在于，强稀疏性意味着 latent 更容易被用作候选概念单元；约束在于，较低的重构保真提示我们不能把 SAE 重构后的表示简单视作原始层表示的等价替代。

### 5.3 候选 latent 结果

在 32768 个 latent 中，单 latent 统计筛选得到 1540 个通过 BH-FDR 的显著候选。排名靠前的 latent 如表所示：

| 排名 | latent idx | Cohen’s d | AUC | 解释方向 |
|---|---:|---:|---:|---|
| 1 | 19435 | 0.8991 | 0.7003 | 强正向 RE 候选 |
| 2 | 13430 | -0.8887 | 0.3279 | 强反向 NonRE 候选 |
| 3 | 31930 | 0.8796 | 0.6966 | 强正向 RE 候选 |
| 4 | 5663 | 0.8470 | 0.6989 | 强正向 RE 候选 |
| 5 | 29759 | 0.8232 | 0.6722 | 中高强度正向候选 |
| 6 | 1516 | 0.7652 | 0.6802 | 稳定正向候选 |
| 7 | 1211 | 0.7526 | 0.6937 | 稳定正向候选 |
| 8 | 26681 | 0.7526 | 0.7500 | 高区分度候选 |

这一结果说明，尽管 SAE 的整体重构保真有限，但在概念筛选层面，它已经提供了相当丰富的候选空间。更重要的是，候选 latent 同时包含正向与反向区分特征，意味着 RE/NonRE 的分离不是由单侧模式主导，而可能由多种互补模式共同构成。

## 6. 讨论

### 6.1 工程意义

从工程角度看，本文系统的核心价值不在于某一个单独指标，而在于把原本分散的机制解释工作流收束成了同一条可执行管线。基础模型、远程 SAE、数据输入、hook、聚合、结构评估、功能评估与结果输出均由统一入口协调，这使后续研究能够在同一框架下替换模型、数据或指标，而不必重写主流程。

此外，严格权重加载、dtype 对齐和 streaming 设计使项目具备了真正面向真实大模型分析的可操作性。如果没有这些工程细节，许多解释性实验会停留在概念层面，而难以在单机环境里稳定复现。

### 6.2 实验设计意义

从实验设计看，本文并未把 RE 研究压缩成单一分类任务，而是将其拆成“候选发现—候选解释—候选验证”的序列。这样的设计更符合机制解释研究的逻辑：先用统计方法缩小搜索空间，再用 probe 与 MaxAct 提供证据，最后再用 absorption、geometry 与 TPP 检查候选是否稳健、是否冗余、是否具有局部因果重要性。

对 RE 这类具有语用与风格属性的概念而言，这种层次化设计比单纯报告 accuracy 更有研究价值，因为它允许我们追问“模型究竟是依赖哪些内部特征在识别 RE”。

### 6.3 局限性与效度威胁

本文也有若干必须明确承认的局限。

第一，当前输入仅使用 `unit_text`，没有引入多轮对话上下文。因此本文识别的是 utterance 级 RE 信号，而不是会话级反射性倾听策略。

第二，结构性指标当前基于部分 token 样本而非全量 token，因此数值应理解为近似估计。

第三，本次稳定实验跳过了 CE/KL 干预路径，因此尚不能从输出分布层面判断 SAE 重构对基础模型行为的影响。

第四，功能性评估模块虽然在设计上已经完整实现，但当前稳定产物主要覆盖单 latent 筛选阶段，后续 MaxAct、absorption、geometry 与 TPP 仍需要在运行环境层面继续做工程稳定化。

换言之，本文更准确的定位是一篇**工程与实验设计导向的技术初稿**，而不是一篇已经完成所有解释性验证的最终结论论文。

## 7. 结论

本文围绕 RE 概念发现问题，设计并实现了一条连接本地 `Llama-3.1-8B`、远程预训练 SAE 与 RE/NonRE 数据集的端到端 SAE-RE 评估管线。系统在工程上完成了模型接入、权重映射、dtype 对齐、streaming 激活处理、句级聚合与多阶段评估框架集成；在实验设计上，构建了由结构性评估、候选 latent 排序、稀疏 probe、MaxAct、feature absorption、feature geometry 与 TPP 组成的分析方案。

当前稳定实验结果表明，该 SAE 虽然整体重构保真有限，但已经能够在 32768 个 latent 中筛出 1540 个与 RE/NonRE 显著相关的候选特征。这说明 RE 概念确实有可能在模型中被部分稀疏化编码。对后续研究而言，更重要的工作不再是“是否存在候选 latent”，而是“这些 latent 的语义是否稳定、是否可复核、是否能通过更强的因果干预得到支持”。本文提供的工程管线与实验设计，正是为这一阶段服务的基础设施。

## 参考文献（草稿版）

[1] Anthropic. *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*. 2023.  
https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning

[2] Aaron Grattafiori et al. *The Llama 3 Herd of Models*. arXiv:2407.21783, 2024.  
https://arxiv.org/abs/2407.21783

[3] Zhengfu He et al. *Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders*. arXiv:2410.20526, 2024.  
https://arxiv.org/abs/2410.20526

[4] OpenMOSS. *Language-Model-SAEs*. GitHub repository, 2024.  
https://github.com/OpenMOSS/Language-Model-SAEs
