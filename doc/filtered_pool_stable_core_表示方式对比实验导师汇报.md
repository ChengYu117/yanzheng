# MISC 标签表征方式对比实验：Filtered-Pool / Stable-Core 导师汇报

## 1. 一页结论

本报告只汇总 `filtered-pool/stable-core` 对比实验，不包含后续取消初筛的 full-pool 复跑。研究对象是 Llama-3.1-8B 第 19 层、经 utterance-level max pooling 得到的咨询师当前发言表征；任务是用同一线性分类器预测 9 个 MISC 标签的可解码性。

最重要的三个结果如下。

1. **预测性能的最高点来自 PCA-n，而非 SAE。** PCA-100 的 macro AUC 为 `0.934`，高于原始激活 `0.900`、完整 SAE `0.890` 与 Top-100 SAE `0.883`。这说明在当前有限样本、线性 probe 设置下，低维 PCA 压缩具有很强的去噪/正则化效果；它不意味着 PCA 的成分比 SAE latent 更容易给出可审计的语义解释。
2. **筛选出的稀疏 SAE 特征保留了大量标签信号。** 每标签约 `33.7` 个 stable-core latent 的 macro AUC 为 `0.869`；训练折内按正向 Cohen's d 选取 `100` 个 Top-n SAE latent 时，macro AUC 为 `0.883`，距完整 SAE 的 `0.890` 仅差 `0.007`，但使用的特征数从 `32,768` 降至 `100`。
3. **被选择的 SAE latent 不是任意子集。** Random SAE-100 的 macro AUC 仅为 `0.693`，比 Top-100 SAE 低 `0.190`。因此，Top-n 与 stable core 的表现不能由“只要抽一些 SAE latent 就够了”解释；筛选过程确实找到了信号密度更高的稀疏子空间。

适合向导师概括的主张是：**原始 hidden state 和 PCA 在纯预测任务上更强；但经过稳定性与关联性筛选的 SAE 子空间能以很少的、可逐一审阅的 latent 保留大部分线性可解码信号。SAE 的研究价值主要在稀疏分解、候选证据审计与后续机制假设生成，而不是在本实验中取得最高分类 AUC。**

## 2. 研究问题与比较对象

本实验回答的是：对于 MISC 咨询师行为标签，标签信息能否从不同内部表征中由同一种简单线性分类器读出？其中 SAE 子空间是否比随机 SAE 子空间更集中地保留此类信息？

比较的表征如下。

| 表征 | 维度/选择方式 | 在本实验中的作用 |
| --- | ---: | --- |
| Hidden State | 4,096 个原始激活维度 | 模型原生表征 baseline |
| Full SAE | 32,768 个 SAE latent | 完整稀疏编码 baseline |
| Stable Core SAE | 每标签独立的稳定 latent 集，平均 33.7 个 | 可解释候选的紧凑 baseline |
| Top-n SAE | 每训练折内按**正向** Cohen's d 选择 n 个 filtered-pool latent | 关联最强的稀疏子空间曲线 |
| PCA-n | 每训练折内仅在训练数据上拟合 n 个 PCA 成分 | 紧凑、稠密的预测性压缩对照 |
| Random SAE-n | 从同一 filtered keep pool 随机抽 n 个 latent，20 个种子 | 验证 SAE 选择是否优于随机 |

`Stable Core SAE` 与 `Top-n SAE` 不相同。前者来自已有的交叉验证稳定筛选结果，只保留 `stable_set_role == stable_core` 的 latent；后者是每个训练折按该标签的正向 Cohen's d 临时排名取前 n，故它是“性能导向的关联子空间”，不等于稳定、可解释或因果的 latent 集。

## 3. 数据与公平比较设置

- 数据：`6,194` 条咨询师当前 utterance，来自 `252` 个源文件；不含前一句来访者发言。
- 标签：`RE, RES, REC, QU, QUO, QUC, GI, SU, AF`。正例比例从 `SU=3.6%` 到 `QU=31.9%`，存在明显类别不均衡。
- 划分：`StratifiedGroupKFold(n_splits=5)`，以 `file_id` 分组，避免同一源文件的相近话语同时进入训练和测试。
- 分类器：每个标签一个 `LogisticRegression(class_weight="balanced", C=1.0, solver="liblinear")`。
- 预处理：标准化器只在训练折拟合；PCA 也只在训练折拟合并变换测试折。因此 PCA-n 没有利用测试折统计量。
- Top-n SAE：只从经过初筛的 latent 候选池选择；Cohen's d 排名在每一个训练折重新计算，避免根据测试标签选择 feature。
- Random SAE-n：从同一个 `keep=True` filtered pool 抽取，使用 20 个随机种子；其报告值是随机选择的平均表现。

这里的“线性 probe”只测量信息是否可以被一个简单线性读出器从表征中恢复，不能直接证明模型理解了咨询行为的功能、MI 原则或因果机制。特别是数据没有 client context，因此 RE/RES/REC 的深层反射关系不能由本实验单独判定。

## 4. 宏观结果

下表为 9 个标签的非加权平均（macro mean）。同一行的四个指标应一起阅读，不建议只凭 F1 或 accuracy 下结论。

| 表征 | n | AUC | PR-AUC | F1 | Balanced Accuracy | 平均特征数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hidden State | baseline | 0.900 | 0.679 | 0.642 | 0.811 | 4,096 |
| Full SAE | baseline | 0.890 | 0.600 | 0.572 | 0.815 | 32,768 |
| Stable Core SAE | baseline | 0.869 | 0.602 | 0.547 | 0.807 | 33.7 |
| Top-n SAE | 10 | 0.822 | 0.524 | 0.511 | 0.772 | 10 |
| Top-n SAE | 20 | 0.849 | 0.559 | 0.528 | 0.792 | 20 |
| Top-n SAE | 50 | 0.874 | 0.590 | 0.545 | 0.807 | 50 |
| Top-n SAE | 100 | 0.883 | 0.601 | 0.558 | 0.815 | 100 |
| Top-n SAE | 200 | 0.879 | 0.596 | 0.567 | 0.808 | 200 |
| PCA-n | 10 | 0.870 | 0.498 | 0.484 | 0.795 | 10 |
| PCA-n | 20 | 0.901 | 0.576 | 0.531 | 0.828 | 20 |
| PCA-n | 50 | 0.925 | 0.652 | 0.579 | 0.854 | 50 |
| PCA-n | 100 | **0.934** | 0.690 | 0.609 | **0.859** | 100 |
| PCA-n | 200 | 0.932 | **0.698** | 0.627 | 0.849 | 200 |
| Random SAE-n | 10 | 0.546 | 0.173 | 0.205 | 0.538 | 10 |
| Random SAE-n | 20 | 0.575 | 0.188 | 0.234 | 0.559 | 20 |
| Random SAE-n | 50 | 0.640 | 0.240 | 0.284 | 0.604 | 50 |
| Random SAE-n | 100 | 0.693 | 0.288 | 0.320 | 0.644 | 100 |
| Random SAE-n | 200 | 0.746 | 0.350 | 0.372 | 0.689 | 200 |

### 对宏观结果的解读

- **原始激活 vs 完整 SAE：** Full SAE 的 AUC 比 Hidden State 低 `0.010`，PR-AUC 低 `0.079`，F1 低 `0.069`；Balanced Accuracy 则高 `0.004`。这与 SAE 是近似重构而非无损复制一致，也表明不能声称 SAE 在整体预测上超过原始激活。
- **Top-n SAE 的收敛：** AUC 从 Top-10 的 `0.822` 上升到 Top-100 的 `0.883`，随后在 Top-200 略降至 `0.879`。在本组 n 中，约 100 个关联最强的 latent 已接近完整 SAE 的 AUC，但没有超过原始激活或 PCA-n。
- **Stable core 的位置：** Stable Core 的 AUC 为 `0.869`，比 Full SAE 低 `0.021`、比 Top-100 SAE 低 `0.014`，但只需约 `33.7` 个 latent。这是后续逐 latent 审阅最实际的候选集合：它牺牲一部分预测性能，换取跨重采样稳定和人工可审计规模。
- **PCA-n 的位置：** PCA-100 比 Hidden State 高 `0.034` AUC、比 Full SAE 高 `0.044` AUC。PCA-200 的 PR-AUC 和 F1 更高，但 AUC 与 Balanced Accuracy 略低于 PCA-100，说明最佳维数依赖所强调的指标。
- **随机对照：** Random SAE-200 仍只有 `0.746` AUC，远低于经过选择的 Top-100 SAE（`0.883`）和 Stable Core（`0.869`）。这为“标签相关信息在少数 SAE latent 中集中”提供预测性证据，但不是 latent 单义性或因果性的证明。

## 5. 逐标签 AUC：最佳子空间与 baseline

下表选择每个标签下 AUC 最好的 Top-n SAE 与 PCA-n 配置；因此不同标签的 n 可以不同。AUC 用于展示区分排序能力，具体阈值下的 PR-AUC、F1 和 Balanced Accuracy 应回查原始逐标签表，不应由此表替代。

| 标签 | Hidden State | Full SAE | Stable Core SAE | 最佳 Top-n SAE | 最佳 PCA-n | AUC 最高表征 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| RE | 0.898 | 0.888 | 0.848 | n=200, 0.867 | n=200, 0.933 | PCA-n |
| RES | 0.803 | 0.794 | 0.768 | n=100, 0.790 | n=100, 0.878 | PCA-n |
| REC | 0.903 | 0.902 | 0.898 | n=200, 0.915 | n=100, 0.947 | PCA-n |
| QU | 0.976 | 0.970 | 0.971 | n=100, 0.975 | n=200, 0.979 | PCA-n |
| QUO | 0.952 | 0.946 | 0.949 | n=100, 0.959 | n=100, 0.966 | PCA-n |
| QUC | 0.903 | 0.889 | 0.921 | n=100, 0.925 | n=200, 0.931 | PCA-n |
| GI | 0.823 | 0.823 | 0.732 | n=200, 0.771 | n=200, 0.899 | PCA-n |
| SU | 0.891 | 0.871 | 0.829 | n=200, 0.850 | n=100, 0.923 | PCA-n |
| AF | 0.950 | 0.927 | 0.910 | n=50, 0.920 | n=200, 0.959 | PCA-n |

逐标签可以提出三点更细的观察。

1. **问题类标签最易线性解码。** `QU` 的 AUC 在各类表征下均接近或超过 `0.97`；`QUO`、`QUC` 也较高，符合问题句式中可能存在可直接读取的形式线索这一现象。但这不是“模型理解开放式提问功能”的充分证据，仍需 minimal pairs 等后续验证区分表面问句形式与对话功能。
2. **Stable core 的保真度具有标签差异。** 对 `REC`、`QU`、`QUO` 和尤其 `QUC`，Stable Core AUC 接近或超过原始激活/完整 SAE；对 `GI`、`RE`、`SU`，压缩损失更明显。这说明“一个统一规模的 stable latent 集”不一定适合所有 MISC 行为，应保留逐标签解释而不是只报宏观平均。
3. **低频标签的阈值指标需要谨慎。** `SU` 仅占 3.6%，虽然 PCA-100 的 AUC 为 `0.923`，但其 F1 为 `0.332`，低于 Raw Hidden 的 `0.495`。这是排序能力（AUC）和固定阈值下正例判定（F1）可以分离的实例，后续应补充阈值校准、PR 曲线和按标签的误差审阅。

`RE` 与 `QU` 是层级父标签，分别与反射类和问题类子标签存在结构共现。它们保留在比较中以检查层级一致性；论文主结论应优先讨论叶节点标签（如 RES、REC、QUO、QUC、GI、SU、AF）。

### 5.1 各标签的单独情况概述

以下数字均来自同一 filtered-pool/stable-core 五折分组交叉验证。Top-n 与 PCA-n 的 `n` 选择为该标签下 AUC 最好的值；它方便展示最优子空间，但不应替代完整 n 曲线。

- **RE（1,358 条正例，21.9%）：** Raw Hidden 的 AUC 为 `0.898`，Full SAE 为 `0.888`，Stable Core 降至 `0.848`；Top-200 SAE 也只有 `0.867`，而 PCA-200 为 `0.933`。这表示当前稳定 sparse 子集没有充分覆盖 RE 的全部可解码信息。由于 RE 是父标签且没有 client context，不能把这一落差解释成“模型不理解反射”；更合适的后续工作是检查其 stable latents 是否只覆盖 RES/REC 的一部分表面模板。
- **RES（516 条，8.3%）：** 这是较困难的反射子标签：Raw/Full SAE/Stable Core 的 AUC 分别为 `0.803/0.794/0.768`，Top-100 SAE 为 `0.790`，PCA-100 提高到 `0.878`。PCA 的 Balanced Accuracy 为 `0.801`，高于 Raw 的 `0.695`，但 F1 为 `0.405`，与 Raw 的 `0.408` 几乎相同。说明排序与两类召回改善，并未自动转化为默认阈值下的更好正例检测；同时它高度依赖缺失的来访者前文，必须保持解释克制。
- **REC（842 条，13.6%）：** Stable Core 的 AUC `0.898` 已非常接近 Raw `0.903` 和 Full SAE `0.902`；Top-200 SAE 达到 `0.915`，PCA-100 为 `0.947`。这是 stable SAE 保留反射相关信号较好的标签之一。不过 REC 的定义包含对来访者内容、意义或情绪的复杂改写，当前 utterance-only 数据只能支持“与 REC 标签可解码线索相关”，不能支持真正的上下文反射机制判断。
- **QU（1,974 条，31.9%，父标签）：** 所有表征的 AUC 都很高：Raw `0.976`、Stable Core `0.971`、Top-100 `0.975`、PCA-200 `0.979`。这说明问题类标签的线性线索高度集中，但也最可能包含问号、疑问词、固定问句框架等表面形式。它适合作为 minimal-pair 验证的优先对象，而不适合作为“已证明理解提问功能”的直接证据。
- **QUO（1,206 条，19.5%）：** Stable Core `0.949` 几乎保留了 Raw `0.952` 的 AUC，Top-100 为 `0.959`，PCA-100 为 `0.966`。相比 Raw 的 F1 `0.812`，Stable Core 与 PCA 的 F1 分别为 `0.753/0.790`，提示即使排序能力很强，固定阈值下的正例决策仍存在损失。后续应重点区分开放式追问的对话功能与 `what/how` 等疑问形式。
- **QUC（768 条，12.4%）：** 这是最值得关注的 stable-core 正向案例：Stable Core AUC `0.921` 高于 Raw `0.903` 和 Full SAE `0.889`，Top-100 为 `0.925`，PCA-200 为 `0.931`。它说明少量稳定 SAE latent 已能形成很有竞争力的封闭式问题子空间；但其 F1 `0.612` 仍低于 Raw 的 `0.653`，应检查默认阈值与 yes/no、do you 等表面模板是否主导了排名优势。
- **GI（681 条，11.0%）：** Raw 与 Full SAE 的 AUC 都为 `0.823`，但 Stable Core 只有 `0.732`，Top-200 SAE 也仅 `0.771`；PCA-200 则达到 `0.899`。GI 的信息在当前 stable sparse 子集中损失最大之一，可能意味着该标签依赖较分布式的内容、主题或句法线索，也可能反映标签边界混杂。该标签应优先审阅 Stable Core 的遗漏和高激活非命中样例，而不是急于给 latent 命名。
- **SU（222 条，3.6%）：** 这是最稀有的核心标签。AUC 从 Raw `0.891`、Full SAE `0.871`、Stable Core `0.829` 到 Top-200 `0.850`，PCA-100 为 `0.923`；但 PCA 的 F1 仅 `0.332`，低于 Raw 的 `0.495`，PR-AUC 也低于 Raw（`0.428` vs `0.513`）。因此不能把 PCA 的高 AUC 解读为 SU 在实际正例检出上全面更好；应先做阈值选择、PR 曲线和正例错误审阅，再比较 latent 的解释价值。
- **AF（349 条，5.6%）：** Raw 的 AUC/F1 为 `0.950/0.697`，Full SAE 为 `0.927/0.509`，Stable Core 为 `0.910/0.516`，Top-50 SAE 为 `0.920/0.518`，PCA-200 的 AUC 最高（`0.959`）但 F1 为 `0.622`。这同样显示高 AUC 与最佳默认阈值 F1 并不等价。AF 的 stable latent 可以作为候选解释材料，但其紧凑表示尚未完整保留 Raw Hidden 下的决策信息。

跨标签来看，可将后续工作分成三类：`REC/QUO/QUC` 是 stable-core 保真度较高、适合优先做对比式解释的对象；`RES/GI/SU/AF` 应先进行错误模式与阈值审阅；`QU` 则是检验 surface form 与 dialogue function 能否分离的关键压力测试标签。所有这些分类都来自表征的预测行为，而不是对单个 latent 的语义定论。

## 6. 为什么 PCA-n 可以比 SAE 和原始激活更高

这不是异常，也不等于 PCA “更懂”心理咨询。

1. **PCA-n 是压缩与正则化。** 原始激活有 4,096 维，样本量为 6,194。线性分类器面对很多弱相关或噪声维度时可能过拟合；PCA 保留训练折中方差最大的方向，丢弃大量低方差方向，相当于限制 probe 的自由度。
2. **SAE 的目标不同。** SAE 试图把激活重构为稀疏、可分解的 latent 字典。完整 SAE 是近似重构，存在信息损失；其 32,768 维稀疏坐标也不必是最适合有限样本线性分类的坐标。
3. **PCA 成分是稠密混合，SAE latent 是可审计候选。** PCA 的一个成分通常混合许多原始维度，难以将其直接对应到一组语句证据。SAE latent 可以按激活话语、标签关联、稳定性与潜在 artifact 风险逐个审查。因此“PCA 的 probe 更高”与“SAE 更适合做 latent-level 解释”并不矛盾。
4. **Full PCA 与 PCA-n 需要区分。** 在正确的标准化口径下，full-rank PCA 只是原始激活空间的正交旋转，对带 L2 正则的线性 probe 应与 Raw Hidden 等价；本项目已验证二者 macro AUC 都为 `0.900`。PCA-n 的较高表现来自降维，而不是全维 PCA 创造了新信息。

## 7. 术语速查

| 术语 | 本项目中的含义 | 应如何解读 |
| --- | --- | --- |
| Hidden State / 原始激活 | Llama 第 19 层的 4,096 维 contextual activation，经 utterance max pooling | 模型原生信息的预测 baseline；不是人工可读概念列表 |
| SAE latent | SAE 将原始激活编码后的稀疏坐标 | latent 的高激活样例只能产生候选解释，不能直接命名为某个 MISC 概念 |
| Full SAE | 使用全部 32,768 个 latent 训练 probe | 检查 SAE 总体是否保留标签信息，不代表单个 latent 有语义 |
| Filtered pool | 在关联分析前排除几乎从不激活、几乎总激活或明显不稳定/异常的 latent 后的候选池 | 目的是提高候选解释的质量与审阅可行性；不是“保证无 artifact” |
| Stable core | 经交叉验证稳定性筛选后、被标记为 `stable_core` 的每标签 latent 集 | 更适合作为人工审阅优先级，但不必然是预测 AUC 最高的集合 |
| Top-n SAE | 每训练折取该标签正向 Cohen's d 最高的 n 个 latent | 用于测量最相关子空间的预测充分性；不同标签选择不同 latent |
| PCA-n | 在训练折把 Raw Hidden 压缩为 n 个主成分 | 强预测压缩 baseline；PCA 成分是稠密组合，不等同于可命名特征 |
| Random SAE-n | 从同一 filtered pool 随机选择 n 个 latent | 用于检验选择策略是否优于随机抽取 |
| Cohen's d | 正例与负例中某 latent 激活均值的标准化差异 | `d > 0` 表示该 latent 在标签正例中平均更活跃；它是关联/排序量，不是因果效应 |
| AUC | ROC 曲线下面积，衡量模型把随机正例排在随机负例之前的能力 | 阈值无关；`0.5` 近似随机，越高越好；适合主比较 |
| PR-AUC / Average Precision | Precision-Recall 曲线概括，关注正例识别质量 | 类别不均衡时尤其重要；不同标签的基线受正例率影响，跨标签直接比较要谨慎 |
| F1 | Precision 与 Recall 的调和平均 | 依赖分类阈值；适合评估当前决策规则下的正例检测，不等同于 AUC |
| Balanced Accuracy | 正例召回率与负例召回率的平均 | 抵消多数类带来的普通 accuracy 虚高；`0.5` 近似随机 |
| Macro mean | 先计算每个标签指标，再对 9 个标签等权平均 | 防止高频标签主导总体结论，但不表示样本量加权的总体病人/语句表现 |

## 8. 推荐的导师汇报口径

可以使用如下表述：

> 在按会话文件分组的五折交叉验证中，原始 hidden state 与低维 PCA 是标签预测最强的表征；PCA-100 的 macro AUC 为 0.934。完整 SAE 的 AUC 为 0.890，说明稀疏重构保留了大部分、但并非全部原始标签信息。更关键的是，每标签约 34 个稳定 SAE latent 已达到 0.869 AUC，训练折内按关联强度选择 100 个 latent 达到 0.883，并在描述性指标上明显优于随机 SAE 子空间。这支持我们把稳定 SAE latent 作为后续可审计候选单元，但不支持把单个 latent 直接解释为完整 MISC 概念，也不支持因果机制主张。

不建议使用如下表述：

- “SAE 比 PCA 更好。”本实验的预测指标不支持这一句。
- “AUC 高说明模型真正理解了 MI 概念。”本实验只说明标签在当前表征中可被线性读出。
- “Top-n latent 就是该标签的神经机制。”Cohen's d 是统计关联；仍可能受表面形式、模板、数据来源或类别共现影响。
- “PCA-n 的优势证明 SAE 无用。”PCA 缺乏 latent 级可审计性，而 SAE 的研究目标包含解释与后续验证，不只追求分类分数。

## 9. 下一步建议

1. 将 `stable_core` 作为优先解释集合，并报告逐标签而非只报宏观均值；优先审计 Stable Core 保真度较高的 `REC/QU/QUO/QUC`，同时把 `GI/RE/SU` 作为压缩失败或混合表征的反例。
2. 针对 `SU`、`RES` 等低频或指标分歧标签，报告 PR 曲线、阈值校准后的 F1，并进行错误样例审阅，避免只用 AUC 夸大实际检出能力。
3. 在 latent 解释阶段使用对比式 evidence packet、minimal pairs 与 triggering-token 检查，以区分表面形式、对话功能、MI 原则和数据 artifact。现有数据没有 client context，因此不能只据当前 counselor utterance 声称 RES/REC 的上下文反射机制。
4. 若要比较方法差异是否稳定，补充同一 test fold 上的配对不确定性分析（例如跨折差值、bootstrap CI 或预注册的显著性检验）；目前宏观差值是描述性结果，不应写成统计显著性结论。

## 10. 可复核产物

本报告只使用下列 filtered-pool/stable-core 产物。

- 总体结果：[probe_macro_summary.csv](/D:/project/NLP_re_dataset_model_base/outputs/misc_full_sae_eval/interpretability/representation_probe_comparison_stable_core/probe_macro_summary.csv)
- 逐标签均值、标准差与标准误：[probe_summary_by_label.csv](/D:/project/NLP_re_dataset_model_base/outputs/misc_full_sae_eval/interpretability/representation_probe_comparison_stable_core/probe_summary_by_label.csv)
- 折级结果：[probe_fold_metrics.csv](/D:/project/NLP_re_dataset_model_base/outputs/misc_full_sae_eval/interpretability/representation_probe_comparison_stable_core/probe_fold_metrics.csv)
- 每标签特征选择记录：[selected_latents_by_label_n.csv](/D:/project/NLP_re_dataset_model_base/outputs/misc_full_sae_eval/interpretability/representation_probe_comparison_stable_core/selected_latents_by_label_n.csv)
- 宏观性能曲线：[performance_curves_macro.png](/D:/project/NLP_re_dataset_model_base/outputs/misc_full_sae_eval/interpretability/representation_probe_comparison_stable_core/figures/performance_curves_macro.png)
- 实验配置与输入摘要：[manifest.json](/D:/project/NLP_re_dataset_model_base/outputs/misc_full_sae_eval/interpretability/representation_probe_comparison_stable_core/manifest.json)
- Stable core 定义来源：[stable_topk_latent_set.csv](/D:/project/NLP_re_dataset_model_base/outputs/cross_val/stable_topk_selection/stable_topk_latent_set.csv)

报告生成日期：2026-07-10。统计口径：filtered-pool/stable-core 对比实验，5 折分层分组交叉验证。
