# PCA 可解释性文献审查与当前 Task 5 设计核验

> Status: Superseded
>
> Replaced by: `docs/current/experiment_workflow.md`
>
> Do not use this document for implementation or experiment decisions. It is retained only as a literature review record.

## 审查范围

本审查暂不判断 SAE 本身是否更可解释，只回答以下问题：

1. 既有研究如何解释和评价 PCA（主成分分析）方向；
2. 在 SAE–PCA 可解释性比较中，PCA 通常应如何处理；
3. 当前 Task 5 的 PCA Top-Card 设计是否准确测量了 PCA 可解释性；
4. 按标签 AUC（受试者工作特征曲线下面积）翻转 PCA 负方向是否错误。

审查对象主要是 `src/nlp_re_base/task5_matched_human_eval.py`，并核对 Cunningham et al. (2023) 的论文和公开代码实现。

## 编辑结论

**结论：当前实现没有明显的 PCA 代数错误，但实验估计对象被改成了“经标签监督定向、与标签相关的 PCA 半轴”，而不是“无监督 PCA 主成分轴的整体可解释性”。因此，现有结果不能直接支持关于 PCA 表征整体可解释性的结论。**

按训练折 AUC 翻转符号在数学上成立，也避免了测试集泄漏；错误不在“乘以 -1”本身，而在于如果不明确报告这一监督定向，就会把人为选择的标签正相关端误写成 PCA 自身具有的正方向。更重要的是，当前 Card 只展示定向后的正端和中间分数样本，没有展示另一极端，因而遗漏了 PCA 作为双极轴的基本结构。

建议将当前实验改称：

> **label-oriented PCA half-axis card evaluation（标签定向的 PCA 半轴卡片评价）**

在补充双端轴评价、全分布留出验证和固定 PCA 基底前，不应称为一般性的 PCA unit interpretability（PCA 单元可解释性）评价。

## 文献中的实际做法

### 1. 传统 PCA：解释的是轴的两端及其对比

PCA 得到的是中心化数据空间中的有符号投影轴。常规解释同时依赖：

- loadings（载荷，即原始维度对主成分方向的权重）；
- scores（得分，即样本在该主成分轴上的有符号投影）；
- 正、负两端分别有哪些高绝对值得分样本；
- 两端之间形成什么对比，而不是只给正端命名。

Jolliffe & Cadima (2016) 的 PCA 综述将载荷和得分作为主成分解释的基本对象。Zou, Hastie & Tibshirani (2006) 提出 sparse PCA（稀疏主成分分析）的直接动机之一，正是普通 PCA 的稠密载荷通常难以解释。这说明“最大方差”与“容易赋予概念名称”不是同一个优化目标。

Bro, Acar & Kolda (2008) 进一步指出 SVD（奇异值分解）和 PCA 的向量符号具有内在不确定性：`v` 与 `-v` 表示同一条轴。算法输出的正负号本身没有数据语义，符号约定只能用于保持跨运行一致或方便解释。

### 2. Cunningham et al.：不只观察 Top 样例，而是预测留出激活

Cunningham et al. (2023) 是目前最直接的 SAE–PCA 自动可解释性比较先例。其流程是：

1. 从约 50,000 个文本片段计算方向得分；
2. 用高得分片段生成自然语言解释；
3. 在另一批高得分片段和随机片段上，让 LLM（大语言模型）预测真实方向得分；
4. 以预测得分与真实得分的相关性作为自动可解释性分数；
5. 分别报告 top-and-random（高分与随机混合）和 random-only（仅随机）结果；
6. 每类方向评价约 150 个特征，并报告均值和 95% 置信区间。

因此，该论文中的 Top 样例主要承担 explanation discovery（解释发现）功能，而不是单独作为“该方向可解释”的最终证据。

公开代码还显示，Cunningham 的 PCA 处理不是单一形式：

- 普通 `PCAEncoder` 保留有符号投影，并按投影绝对值选择 Top-k；论文基线令 `k` 等于原维度，因此实际保留全部分量；
- `pca_topk` 将每个主成分显式扩展为 `+v` 和 `-v` 两个方向，再经过非负 Top-k 编码；
- 普通 PCA 自动解释路径按最大正得分选片段，因此仍受到任意符号的影响；这应视为该早期基线的局限，而不是应当照搬的规范。

论文附录在解释 residual stream（残差流）基方向时也明确检查最正和最负方向。这支持“有符号方向应检查两端”的方法学判断。

公开代码还存在两个复现层面的限制。第一，在线 PCA 估计了训练均值，但普通 `PCAEncoder.encode` 直接计算 `Vx`，没有计算标准 PCA 得分 `V(x-mean)`。第二，`pca_topk` 虽将字典扩展为 `[+V; -V]`，解释程序却固定从字典前部取前 150 个单元；在原维度大于 150 时，这批单元全部落在 `+V` 半边。论文文本没有充分披露这两个实现细节。因此 Cunningham 是重要的直接比较先例，但其 PCA 实现不应被当作无缺陷的评价标准。

### 3. 后续工作：Top 样例会产生可解释性幻觉

Gao et al. (2024) 指出，只看高激活样例可能产生 illusion of interpretability（可解释性幻觉）：一个过宽的解释可能覆盖正例，却在大量非目标文本上同样预测激活。其方法学重点是同时评价 precision（精确率）和 recall（召回率），而不能只验证解释是否覆盖 Top 样例。

Bills et al. (2023) 的自动解释框架以及 Cunningham 的改版，都要求解释去预测未参与解释生成的激活记录。Gurnee et al. (2023) 同样警告，前若干最大激活样例可能掩盖单元在其余分布上的多义性。

Makelov, Lange & Nanda (2024) 则强调：字典可解释性评价应绑定明确任务，并区分 approximation（近似/重构）、control（控制）和 interpretation faithfulness（解释忠实性）。单个 Top-Card 的视觉连贯性不能替代这些不同层面的证据。

## 对当前实现的逐项核验

### 主要问题 1：只解释标签定向后的正半轴

当前代码先计算每个 PC（主成分）对目标标签的训练折 AUC；若 AUC 小于 0.5，则设置 `direction_sign=-1`，并将分数乘以该符号。之后所有 discovery 样例都按翻转后的分数从高到低选取。

这相当于定义：

```text
oriented_score = sign(AUC - 0.5) * PCA_score
```

它回答的是“这个 PC 的哪一端更接近目标标签”，而不是“这个 PC 轴整体表达了什么对比”。当前 held-out control（留出对照）来自分数中间 35%–65% 分位区间，也没有包含反向极端样例。因此审查者无法判断：

- 正端和负端是否构成清晰的语义对比；
- 所谓正端模式是否只是一个孤立表面词；
- 同一解释是否错误覆盖负端；
- 该 PC 是单一连续轴、两个不相关极端，还是没有稳定模式。

这是当前 PCA Card 设计最关键的构念效度问题。

### 主要问题 2：每个标签重新拟合一套 PCA

当前 `_split_for_label` 为每个标签分别构造分层分组划分，随后在标签循环内重新拟合 PCA。由于各标签的训练行集合不同，REC、QUO、GI、AF 实际使用的不是同一套 PCA 基底。

这没有直接标签泄漏，因为 PCA 拟合本身未读取标签值；但它导致“PC1”“PC2”等不再是跨标签共享的固定表示单元。若研究问题是“大模型中 PCA 单元如何组织多个行为标签”，该设计不成立。更合适的做法是先冻结一个与标签无关的公共训练集合，在该集合上只拟合一次 PCA，再对所有标签计算关联。

### 主要问题 3：当前是 correlation-PCA，而非通常的 covariance-PCA

代码在 PCA 前使用 `StandardScaler` 对每个 hidden dimension（隐藏维度）做单位方差标准化。这使 PCA 最大化的是标准化坐标中的方差，即接近 correlation-PCA（相关矩阵 PCA），而不是对原 hidden state（隐藏状态）仅中心化后的 covariance-PCA（协方差 PCA）。

这不一定错误，但会实质改变主成分方向。对于神经网络隐藏维度，各维度本来处于同一表示空间，方差大小可能就是模型几何的一部分。若目标是提供普通 PCA 基线，应优先使用训练折均值中心化；若保留输入标准化，必须将方法明确命名为 standardized-input PCA，并说明选择依据。

PCA 后的第二次 `StandardScaler` 影响较小：它只按正数尺度缩放每个 PC，不改变单个 PC 的样本排序、AUC、Cohen's d（标准化均值差）或 Top 样例。它可用于统一显示尺度，但不能解决 Card 中观察到的表面模式问题。

### 主要问题 4：限制在前 50/100 个高方差 PC，不等于抽样 PCA 可解释性

前 50/100 个 PC 是最大化解释方差得到的方向，不是最大化标签关联或人类可解释性得到的方向。如果随后只在这批方向内按标签 AUC 选最优单元，实验得到的是“高方差子空间中最能预测标签的半轴”。

这种选择适合回答受限的标签解码问题，却不能外推到全部 PCA 单元。要评价 PCA 本身，至少应报告不同 explained-variance strata（解释方差分层）中的随机 PC，或者预先固定所评价的 PC 范围并把结论限制在该范围。

### 主要问题 5：Top-Card 连贯性不是留出解释性能

当前 Stage 1 给审查者 10 条训练折 Top 样例，要求形成一句解释；Stage 2 在 10 条测试折 Top 样例与 10 条中间分数对照上判断解释是否成立。该设计已经优于只看训练 Top 样例，但仍有三个限制：

- Stage 2 正例仍按极端得分挑选，不代表完整得分分布；
- 对照主要来自中间区间，不包含困难负端和表面相似反例；
- 二元“是否符合解释”没有检验解释能否预测连续 PCA score（PCA 得分）的相对大小。

因此它可以作为人工 discovery/verification（发现/验证）试验，但不能单独构成整体可解释性指标。

### 次要问题：保存模型与定向分数的符号需要一致记录

当前生成 Card 时同时翻转训练和测试分数，做法内部一致；但保存的 `pca_components` 仍是 sklearn 原始方向。报告中“negative directions are multiplied by -1”实际只落实在 Card 分数上，并未改写保存的 component/loading（成分向量/载荷）。

未来若使用保存模型做重构、干预或载荷展示，必须同时应用 `direction_sign`，否则 Card 解释会与方向向量相反。建议保存 `oriented_component = direction_sign * component`，同时保留原 component 和 sign 以便审计。

## “把负方向拨正”是否错误

### 数学判断：不是错误

对任一 PCA 成分，以下变换完全等价：

```text
component' = -component
score'     = -score
```

解释方差、子空间和重构结果不变。当前代码也只使用训练折 AUC 决定方向，并把同一个符号应用于训练、测试分数，因此不存在用测试标签选择符号的数据泄漏。

### 方法学判断：改变了估计对象

用标签 AUC 决定符号不是无监督的符号消歧，而是 supervised orientation（监督定向）。它使每个 PC 的“正端”都被定义为目标标签更常出现的一端。只要论文明确说明，作为标签关联分析是可接受的；但它不能被描述为 PCA 天然的正方向，也不能用于证明 PCA 轴本身单义。

### 何时应保留，何时应删除

- **保留 AUC 定向**：研究问题明确是“与某标签正相关的 PCA 半轴是否可解释”；方向只能在训练折确定，并在所有留出数据上冻结。
- **删除 AUC 定向**：研究问题是无监督 PCA 轴本身是否可解释；此时应使用标签无关的确定性符号约定，并同时展示正负两端。
- **更公平的半轴方案**：把每个 PC 显式拆成 `max(score, 0)` 和 `max(-score, 0)` 两个非负半轴，作为两个候选单元分别评价。此时必须把维度预算记为 `2d` 个半轴，而不能仍称为 `d` 个独立单元。

## 推荐的 PCA 评价协议

### 协议 A：轴级解释，回答“PCA 轴是否形成清晰对比”

1. 在一个公共、与标签无关的训练集合上只拟合一次 PCA。
2. 预先固定 covariance-PCA 或 standardized-input PCA，不根据 Card 结果切换。
3. 对每个 PC 提供正端 Top、负端 Top 和中间/随机样例。
4. 要求解释写成双极对比：`positive pole（正极） vs negative pole（负极）`；允许输出“无稳定对比”。
5. 冻结解释后，在未参与解释生成的样本上预测连续有符号得分或分位箱。
6. 报告 Pearson/Spearman correlation（皮尔逊/斯皮尔曼相关）、分位排序准确率，以及正负端识别 balanced accuracy（平衡准确率）。
7. 在预先定义的 PC 范围内随机抽样，并按解释方差分层报告，不只选择标签 AUC 最优方向。

### 协议 B：半轴级解释，回答“PCA 半轴能否像非负特征一样被命名”

1. 将每个 PC 拆成正、负两个非负半轴。
2. 两个半轴使用相同的 Top、随机、近失误和留出评价流程。
3. 选择方向时不使用测试数据；如按标签筛选，明确称为 label-associated half-axis（标签关联半轴）。
4. 匹配时除标签 AUC 外，至少分层或调整 activation coverage（激活覆盖率）、tail concentration（尾部集中度）、句长和样本重复率。
5. 以半轴为抽样单位；PCA-d 对应最多 2d 个半轴，报告时不能忽略这一预算变化。

两套协议回答不同问题，不应混合成同一个总分。协议 A 更忠实于 PCA 的数学结构；协议 B 更适合与非负、稀疏单元做材料形式相近的人工 Card 比较。

## 对现有结果可采用的安全表述

现有实验最多支持：

> 在前 50/100 个、经输入标准化后得到的 PCA 主成分中，按训练折标签 AUC 选择并定向的若干 PCA 半轴，其极端正得分语句经自动或人工归纳后，主要呈现某些语言结构、词汇或简单语义模式。

现有实验不支持：

- PCA 表征整体缺乏可解释性；
- PCA 单元本质上只编码表面形式；
- PCA 比另一种表示更不理解咨询行为；
- 负方向翻转后得到的是 PCA 的自然语义方向；
- 单个或少量标签相关 PC Card 可以代表全部 PCA 空间。

## 参考文献

1. Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600). arXiv:2309.08600.
2. Bills, S., et al. (2023). [Language Models Can Explain Neurons in Language Models](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html).
3. Gao, L., et al. (2024). [Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093). arXiv:2406.04093.
4. Gurnee, W., Nanda, N., Pauly, M., Harvey, K., Troitskii, D., & Bertsimas, D. (2023). [Finding Neurons in a Haystack: Case Studies with Sparse Probing](https://arxiv.org/abs/2305.01610). arXiv:2305.01610.
5. Makelov, A., Lange, G., & Nanda, N. (2024). [Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control](https://arxiv.org/abs/2405.08366). arXiv:2405.08366.
6. Jolliffe, I. T., & Cadima, J. (2016). [Principal component analysis: a review and recent developments](https://doi.org/10.1098/rsta.2015.0202). Philosophical Transactions of the Royal Society A, 374, 20150202.
7. Zou, H., Hastie, T., & Tibshirani, R. (2006). [Sparse Principal Component Analysis](https://doi.org/10.1198/106186006X113430). Journal of Computational and Graphical Statistics, 15(2), 265–286.
8. Bro, R., Acar, E., & Kolda, T. G. (2008). [Resolving the sign ambiguity in the singular value decomposition](https://doi.org/10.1002/cem.1122). Journal of Chemometrics, 22(2), 135–140.

## 代码核验位置

- 当前 PCA 训练与双重标准化：`src/nlp_re_base/task5_matched_human_eval.py:234`
- 当前 AUC 定向：`src/nlp_re_base/task5_matched_human_eval.py:270`
- 当前 Top 与中间分位对照抽样：`src/nlp_re_base/task5_matched_human_eval.py:399`
- 当前按标签分别拟合 PCA：`src/nlp_re_base/task5_matched_human_eval.py:703`
- Cunningham PCA 有符号编码与 `+v/-v` 扩展：`outputs/literature_audit_sparse_coding/autoencoders/pca.py:84`
- Cunningham Top-k 非负编码：`outputs/literature_audit_sparse_coding/autoencoders/topk_encoder.py:19`
- Cunningham 自动解释的 Top/随机留出评分：`outputs/literature_audit_sparse_coding/interpret.py:265`
