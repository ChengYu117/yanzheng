# AAAI 2027 HAI 研究说明报告：使用稀疏自编码器理解 LLM 中的心理咨询行为表征

## 一、项目概述

本项目拟面向 AAAI 2027 HAI（Explainable AI for Human Understanding）方向，研究大型语言模型内部如何表征由人类定义的心理咨询行为概念。具体而言，我们以 Motivational Interviewing Skill Code（MISC）中的咨询师行为标签为外部概念体系，使用 Sparse Autoencoder（SAE，稀疏自编码器）将 Llama-3.1-8B 的中间层 hidden state 分解为稀疏 latent（潜在特征），并检验这些 latent 是否能够形成紧凑、稳定且可供人检查的行为证据。

项目关注的不是构建一个更高准确率的 MISC 分类器，而是回答一个面向人类理解的问题：当语言模型能够区分提问、反映、建议、肯定等咨询行为时，模型内部是否存在较稀疏的特征集合来支持这些判断；如果存在，这些特征反映的是心理咨询功能、通用语言形式、主题词，还是多种因素的混合。

拟议英文标题可暂定为：

> **Sparse Internal Evidence for Human-Defined Counseling Behaviors in Large Language Models**

目标 Track 和题目仍需在 AAAI 2027 正式 CFP 发布后核对，不在当前阶段视为已确认投稿类别。

## 二、研究背景与问题

MISC 标签由心理咨询研究者从外部定义，例如开放式问题、封闭式问题、简单反映、复杂反映、信息提供、建议和肯定。这些标签对人类有明确的操作性含义，但语言模型内部用于支持标签判断的证据通常不可见。

常规 probing 可以回答某层 hidden state 是否包含标签信息，却不能直接说明信息由哪些内部模式承载。PCA 等通用降维方法可以压缩表示，但主成分是为总体方差优化的，不天然对应人类可读概念。SAE 则把高维 hidden state 分解为稀疏激活特征，为建立“模型内部特征—文本证据—人类行为概念”之间的可检查联系提供了可能。

本研究由此提出三个核心问题：

### RQ1：SAE 表征是否忠实保留心理咨询行为信息？

检验 Full SAE、Top-n SAE 和稳定特征子集能否恢复 MISC 行为标签，并与原始 hidden state、PCA 和随机 SAE 特征比较。

### RQ2：被选中的 SAE 特征能否被解释为有意义的咨询行为模式？

对稳定 latent 的高激活文本进行归纳，区分表面形式与语义/话语功能，记录支持证据、反例、替代解释和置信度。

### RQ3：这些可解释特征整体揭示了怎样的内部组织结构？

分析每个标签所需的最小充分特征数、latent 跨标签共享、标签独占特征、表面形式与功能模式的比例，以及标签与模型证据的不一致案例。

## 三、数据、模型与表示

### 3.1 数据

当前主数据为 `data/mi_quality_counseling_misc`，合并后包含 6,194 条 counselor behavior unit，来自 252 个会话文件：

| 数据来源 | 样本数 | 文件数 |
|---|---:|---:|
| high-quality split | 4,169 | 153 |
| low-quality split | 2,025 | 99 |
| 合计 | 6,194 | 252 |

核心标签为 `RE, RES, REC, QU, QUO, QUC, GI, SU, AF`。其中 `RE` 和 `QU` 是父标签，分别与 `RES/REC` 和 `QUO/QUC` 存在层级关系，不应与 leaf label 完全等同地解释。

当前标签矩阵来源于 LLM 分段后的 MISC 标注产物，尚未被独立确认为人工 gold annotation。因此论文在获得人工复核前应使用“reference labels（参考标签）”，不能直接称为“human gold labels（人工金标准）”。

另一个重要限制是：当前 `unit_text` 只有咨询师当前行为单元，没有前一句 client utterance。对 RES、REC 和 RE 的上下文关系判断必须保持保守。

### 3.2 模型与 SAE

- 基础模型：Llama-3.1-8B；
- SAE hook：`blocks.19.hook_resid_post`；
- hidden state 维度：4,096；
- SAE latent 维度：32,768；
- utterance 聚合：token 维度 max pooling；
- 当前没有保存完整 token-level latent attribution。

因此，当前证据是 utterance-level 的结构性与预测性证据，不是 token-level 机制定位，也不是因果干预证明。

## 四、实验一：表征验证

### 4.1 实验设计

比较以下表示：

1. Hidden State：原始 4,096 维 hidden state；
2. Full SAE：全部 32,768 个 SAE latent；
3. Top-n SAE：按训练数据中的正向 Cohen's d 选择 n 个 latent；
4. Stable Core SAE：跨重采样、置信区间和跨质量检查后保留的稳定 latent；
5. PCA-n：对 hidden state 做 PCA，并在 PCA 后再次标准化；
6. Random SAE-n：从过滤后的 SAE 候选池随机选择 n 个 latent。

对 Top-n、PCA 和 Random SAE 使用 `n = 10, 20, 50, 100, 200`。所有方法使用一致的 `StratifiedGroupKFold`，按 `file_id` 分组，训练折内拟合标准化器和分类器。分类器为 balanced logistic regression，评价指标包括 AUC、PR-AUC、F1 和 Balanced Accuracy。

### 4.2 当前主要结果

现有宏平均结果如下：

| 表示 | n | Macro AUC | Macro PR-AUC | Macro F1 | Macro Balanced Acc. |
|---|---:|---:|---:|---:|---:|
| Hidden State | 4096 | 0.900 | 0.679 | 0.642 | 0.811 |
| Full SAE | 32768 | 0.890 | 0.600 | 0.572 | 0.815 |
| Stable Core SAE | 平均 33.7 | 0.869 | 0.602 | 0.547 | 0.807 |
| Top-n SAE | 10 | 0.822 | 0.524 | 0.511 | 0.772 |
| Top-n SAE | 20 | 0.849 | 0.559 | 0.528 | 0.792 |
| Top-n SAE | 50 | 0.874 | 0.590 | 0.545 | 0.807 |
| Top-n SAE | 100 | 0.883 | 0.601 | 0.558 | 0.815 |
| Top-n SAE | 200 | 0.879 | 0.596 | 0.567 | 0.808 |
| PCA-n | 10 | 0.870 | 0.498 | 0.484 | 0.795 |
| PCA-n | 20 | 0.901 | 0.576 | 0.531 | 0.828 |
| PCA-n | 50 | 0.925 | 0.652 | 0.579 | 0.854 |
| PCA-n | 100 | 0.934 | 0.691 | 0.610 | 0.860 |
| PCA-n | 200 | 0.932 | 0.699 | 0.628 | 0.850 |
| Random SAE-n | 10 | 0.546 | 0.173 | 0.205 | 0.538 |
| Random SAE-n | 20 | 0.575 | 0.188 | 0.234 | 0.559 |
| Random SAE-n | 50 | 0.640 | 0.240 | 0.284 | 0.604 |
| Random SAE-n | 100 | 0.693 | 0.288 | 0.320 | 0.644 |
| Random SAE-n | 200 | 0.746 | 0.350 | 0.372 | 0.689 |

### 4.3 对 RQ1 的当前回答

当前结果支持以下判断：

1. **SAE 保留了大量行为信息。** Full SAE 的 macro AUC 为 0.890，接近 hidden state 的 0.900。
2. **有针对性选择的 SAE 特征明显优于随机 SAE 子集。** 例如 Top-100 SAE 的 macro AUC 为 0.883，而 Random SAE-100 为 0.693。
3. **行为信息可以由较紧凑的 stable-core 子集恢复。** Stable Core SAE 平均仅使用 33.7 个 latent，macro AUC 达到 0.869。
4. **当前结果不支持“SAE 优于 PCA”的主张。** PCA-100 的 macro AUC 为 0.934，明显高于 Top-100 SAE 的 0.883。PCA-20 已达到 0.901，也高于 Top-20 SAE 的 0.849。

第四点必须在论文中如实报告。性能上 PCA 更强并不自动否定 SAE 的解释价值，但意味着论文不能把“预测性能优于通用降维”作为主要贡献。更合理的论证是：PCA 提供较强的低维预测基线，而 SAE 的潜在价值在于可将证据映射回稀疏、可单独检查的激活特征。这个解释价值必须由实验二和实验三提供独立证据，不能从 probe AUC 推导。

## 五、实验二：稳定特征解释与 Feature Cards

### 5.1 候选集合

稳定筛选产生 303 个 label-latent 关联，对应 225 个 unique latent。各标签 stable-core 数量为：

| 标签 | Stable-core 数量 |
|---|---:|
| RE | 54 |
| RES | 15 |
| REC | 47 |
| QU | 26 |
| QUO | 32 |
| QUC | 45 |
| GI | 26 |
| SU | 31 |
| AF | 27 |
| 合计 | 303 |

### 5.2 当前解释流程

对每个 unique latent：

1. 从 6,194 条语句中按 latent activation 排序；
2. 在调用解释模型前按规范化文本去重，每个重复文本只保留最高激活实例；
3. 向解释模型提供 Top-50 unique activating utterances；
4. 分别归纳 surface pattern（词汇、句法、话语标记和模板）与 semantic pattern（语义或话语功能）；
5. 输出支持样本、outlier、代表证据、替代解释、failure mode 和校准后的置信度。

当前 225 个 latent 均已产生结构有效的双轨解释：

| 归纳状态 | Latent 数量 |
|---|---:|
| 表面与语义均满足多数条件 | 188 |
| 仅语义满足多数条件 | 28 |
| 仅表面满足多数条件 | 8 |
| 两者均不满足 | 1 |

表面模式多数门通过 196 个 latent，语义模式多数门通过 216 个 latent。

### 5.3 Feature Card 设计

每张最终 card 应包含：

- Feature ID 和相关标签；
- Top activating examples；
- 近邻、表面匹配和非目标对照；
- 表面模式与语义/功能模式；
- 支持证据及代表样本；
- 替代解释与潜在 artifact；
- 解释置信度；
- 当前证据边界。

论文只展示少量代表性 cards，完整 cards 放入 supplement 或 artifact。代表案例应覆盖：

1. 明确语义/功能型 latent；
2. 明确表面形式型 latent；
3. 表面与功能纠缠的 latent；
4. 跨标签共享 latent；
5. 解释失败或高度不确定的 latent。

### 5.4 对 RQ2 的当前回答边界

当前结果说明大量 latent 的 Top-50 文本存在可归纳模式，但“结构有效的 LLM 输出”不等于“已由人类确认的正确解释”。现有解释仍可能把固定短语、主题词或咨询语体误写成心理咨询功能。尤其是 RES、REC 和 RE，由于缺少 client 前文，强功能解释必须降级。

因此，RQ2 目前只能写为：

> 初步证据显示，多数 stable-core latent 的高激活文本具有可描述的表面或语义规律；其中哪些规律能够被人类稳定识别为心理咨询行为，仍需人工抽样复核和严格对照验证。

在人工复核冻结前，不建议写成“225 个 latent 均已获得可靠的人类可解释概念”。

## 六、实验三：表征结构分析

### 6.1 最小充分特征数量

`minimal_sufficient_subspace_v2` 衡量的是 probe-space predictive sufficiency，即最少多少 latent 可以接近完整候选池的预测表现。它不证明这些 latent 构成因果机制。

当前五折 selected-k 统计如下：

| 标签 | 均值 k | 中位数 k | 最小–最大 |
|---|---:|---:|---:|
| AF | 12.2 | 12 | 10–16 |
| GI | 18.2 | 17 | 15–24 |
| QU | 6.4 | 6 | 4–10 |
| QUC | 16.8 | 15 | 12–25 |
| QUO | 10.6 | 10 | 8–14 |
| RE | 27.0 | 19 | 11–49 |
| REC | 17.2 | 14 | 9–29 |
| RES | 28.2 | 21 | 14–47 |
| SU | 26.4 | 24 | 8–43 |

这一结果提供了一个有意义的结构假设：提问类标签，尤其 QU，可能由较少且较一致的形式/功能特征支持；RES、RE 和 SU 需要更多特征，可能反映行为定义更依赖语境、内部结构更分散，或标签噪声更高。该解释需要结合 feature cards 与上下文限制验证。

### 6.2 共享与独占结构

303 个 label-latent 关联对应 225 个 unique latent：

- 158 个 latent 仅进入一个标签的 stable core；
- 67 个 latent 被两个或三个标签共享；
- 单个 latent 最多关联 3 个核心标签。

共享并不一定表示语义混淆。部分共享来自标签层级，例如 RE 与 RES/REC、QU 与 QUO/QUC；部分可能表示跨行为的通用话语形式，例如第二人称、疑问结构、反映式模板或咨询语体。Fig. 4 应将父子层级共享与非层级共享分开，否则会夸大“跨标签复用”。

### 6.3 标注不一致审查

**结论先行：** SAE 线性探针与当前 reference labels 的不一致不是单一来源的随机误差，而是呈现出清晰的标签特异性结构。

1. **GI、SU 和 RES 的主要问题是漏检与覆盖不足。** 三者的 FN 率分别达到 47.1%、34.7% 和 33.1%；但 GI 和 SU 几乎没有高模型分数 FN，说明多数漏检位于表示覆盖或决策边界附近，而不是 probe 对这些正例作出强烈否定。
2. **高模型分数不一致主要来自 FP。** 七个叶标签共有 64 个高分 FN、841 个高分 FP。AF、QUO、QUC、REC 和 SU 的高分 FP 较多，表明 probe 会将积极词汇、问句形式、反映模板或帮助表达过度泛化为相应行为标签。
3. **不一致不能直接解释为标注错误。** RES/REC 受到 client 前文缺失和兄弟标签边界影响，QUO/QUC 存在开放/封闭及反映式问句混淆；对28条代表案例的独立质检还发现，AI辅助审查会过度使用“可能标注错误”判断。当前最稳健的结论是：这些案例揭示了标签边界、混合行为、表面触发和模型覆盖局限，需要独立 MISC coder 作最终裁决。

已使用 stable-core SAE probe 生成严格 OOF 预测：

- 6,194 个样本 × 9 个标签，共 55,746 条 OOF 预测；
- 所有预测来自测试折，45 个 fold 的 `file_id` overlap 均为 0；
- 共识别 8,194 个参考标签与 SAE 阈值预测不一致的 label-case；
- 其中 1,180 条属于高模型分数不一致，即 FN 的 probability ≤ 0.1 或 FP 的 probability ≥ 0.9；
- 已分层抽取 270 条案例，每个标签 30 条，覆盖高模型分数与阈值附近的 FP/FN，并完成首轮 AI 辅助语义审查；
- 每条抽样案例均重建原 OOF probe 的 latent 贡献，并关联已有 DeepSeek feature-card 解释；重建 probability 与原始结果的误差小于 `1e-6`。

叶标签的全体 OOF 不一致如下。RE 与 QU 是父标签，只用于层级一致性审查，不与叶标签重复解释。

| 标签 | FN率 | FP率 | 高分 FN | 高分 FP |
|---|---:|---:|---:|---:|
| RES | 33.1% | 25.6% | 1 | 73 |
| REC | 20.9% | 15.9% | 10 | 145 |
| QUO | 13.2% | 10.6% | 26 | 133 |
| QUC | 18.4% | 12.0% | 24 | 178 |
| GI | 47.1% | 14.2% | 0 | 69 |
| SU | 34.7% | 12.4% | 1 | 111 |
| AF | 22.9% | 7.3% | 2 | 132 |

![各叶标签的FN与FP率](../outputs/misc_full_sae_eval/interpretability/sae_annotation_audit_stable_core/semantic_case_analysis/figures/disagreement_rates_by_label.png)

*图：各叶标签的 OOF FN率与FP率。GI、SU和RES的FN率最突出，RES同时具有较高FP率。*

![各叶标签的高模型分数不一致](../outputs/misc_full_sae_eval/interpretability/sae_annotation_audit_stable_core/semantic_case_analysis/figures/high_score_disagreements_by_label.png)

*图：高模型分数不一致数量。叶标签中高分FP显著多于高分FN，提示表面模式或相邻功能的过度泛化。*

结果呈现出不同错误形态。GI 和 SU 的 FN 率较高，但几乎没有高分 FN，说明更多正例位于 probe 的覆盖边界，而不是被模型以极低分强烈排斥。RES 同时具有较高 FN 与 FP，并明显受到缺少 client 前文、反映式问句和第二人称改述模板影响。AF 的总体 FP 率最低，但 428 个 FP 中有 132 个达到高模型分数，代表案例显示 probe 会把 `good/great/perfect` 等一般积极评价误判为对来访者优势或努力的肯定。QUO/QUC 的疑问形式较容易识别，但开放/封闭边界及反映式问句仍产生混淆。

270 条分层样本的首轮 AI 辅助主诊断为：可能的 reference-label 问题 116 条、模型局限 79 条、上下文不足 42 条、阈值边界 19 条、标签歧义 10 条、表面 artifact 4 条。允许多标签诊断后，共发现 6 条明确混合行为候选，主要涉及反映与问句、GI 与反映、SU 与反映共存。由于该样本按标签、方向和 margin 分层抽取，这些诊断比例不能外推到全部 8,194 条不一致。进一步抽查 28 条代表案例发现，AI 对 `probable_reference_label_error` 存在明显过判，例如把 REC/RES 兄弟标签边界误写成标签矛盾，或把反映式问句直接判成漏标。因此 116 条只表示人工复核队列规模，不能作为标注错误数量。

![分层审查样本的诊断标志](../outputs/misc_full_sae_eval/interpretability/sae_annotation_audit_stable_core/semantic_case_analysis/figures/sampled_diagnosis_flags.png)

*图：270条分层样本中的多标签诊断标志。一条案例可以同时具有歧义、上下文不足、模型局限或表面artifact，因此各项数量不可相加为270。*

这 8,194 条不能解释为 8,194 个标注错误。当前 `label_matrix.csv` 的 reference labels 来自 LLM 分段后的 MISC 标注，并非独立人工 gold；`probable_reference_label_error` 只表示待人工复核候选。首轮 AI 审查也不能替代独立 MISC coder。详细结果见 `outputs/misc_full_sae_eval/interpretability/sae_annotation_audit_stable_core/semantic_case_analysis/semantic_disagreement_analysis_report.md`，28 条叶标签代表案例见同目录的 `representative_cases.csv`。

## 七、建议修订的科学主张

根据当前证据，建议将原蓝图中的主张调整如下。

### 可以保留的主张

1. **行为标签信息可以从 SAE 表征中恢复。** Full SAE 与 hidden state 的 AUC 接近。
2. **行为信息集中在非随机的稀疏特征子集中。** Top-n 和 stable-core SAE 明显优于同规模随机子集。
3. **不同标签表现出不同的稀疏结构。** 最小充分 k 和 stable-core 大小在标签之间差异明显。
4. **大量稳定 latent 的高激活文本存在可描述模式。** 这些模式包括表面形式、语义功能及两者的纠缠。

### 必须降级或删除的主张

1. **“被选中的 SAE 特征优于 PCA”应删除。** 当前结果相反。
2. **“单个 latent 对应一个 MISC 标签”不成立。** 存在共享、碎片化、层级重叠和表面 artifact。
3. **“已证明模型内部因果机制”不成立。** 当前是相关性、结构性和预测充分性证据。
4. **“全部 feature cards 已获人类验证”尚不成立。** 当前主要是自动归纳结果，人工质量确认仍需完成。

### 推荐的中心主张

> SAE 没有提供性能最优的低维表示，但它提供了一组非随机、稳定且可映射回具体文本证据的稀疏特征。通过同时分析预测充分性、特征稳定性、可解释模式与跨标签组织，本研究展示了如何把 LLM 中的心理咨询行为信息转化为可由人类检查的内部证据，同时揭示表面形式、咨询功能和标签层级之间的纠缠。

这个中心主张与 HAI 的“human understanding”定位更一致，也能容纳 PCA 性能更高这一负结果。

## 八、论文叙事结构

建议论文按以下逻辑展开：

1. **问题：** 外部 MISC 标签可定义，但 LLM 内部证据不可见；
2. **方法：** 用 SAE 建立稀疏特征，并通过稳定筛选得到可复现候选；
3. **量化有效性：** SAE 保留行为信息，紧凑特征显著优于随机；
4. **诚实对照：** PCA 的预测性能更强，说明稀疏可解释性与最优压缩性能不是同一目标；
5. **人类可检查证据：** feature cards 将 latent 与具体激活文本、对照和替代解释连接；
6. **科学洞察：** 标签之间存在紧凑性差异、共享、独占、层级重叠及表面/功能纠缠；
7. **边界：** 当前证据不是因果解释，且参考标签与 feature cards 仍需人工复核。

## 九、计划图表

### Figure 1：整体研究 Pipeline

展示 hidden state 提取、SAE 编码、稳定筛选、probe 验证、Top-50 证据归纳、feature cards 和结构汇总。图中应明确区分“模型计算”和“人类审查”。

### Figure 2：表征验证

主图绘制 Top-n SAE、PCA-n 和 Random SAE-n 的性能曲线；Hidden State、Full SAE 和 Stable Core SAE 作为水平参考。至少展示 AUC 和 PR-AUC，F1 与 Balanced Accuracy 可放附录。

### Figure 3：代表性 Feature Cards

建议展示 4–6 个案例，不只展示最漂亮的语义 latent，还应包含一个表面形式 latent 和一个纠缠/失败案例，以避免 cherry-picking。

### Figure 4：内部组织结构

候选形式包括 label-latent 二部图、共享矩阵或 UpSet plot。父子标签共享必须与非层级共享分开标记。

### Figure 5：标注审查案例（可选）

展示少量经人工复核的高模型分数不一致案例，分别代表疑似漏标、混合行为、上下文不足和模型表面误触发。

### 计划表格

- Table 1：数据集、标签频率和会话分布；
- Table 2：Hidden / Full SAE / Top-n / Stable Core / PCA / Random 的主结果；
- Table 3：每标签 stable-core 数量、最小充分 k、共享与独占统计；
- Table 4：经人工复核的代表性 feature cards；
- Appendix Table：逐标签性能、全部 card 索引和标注审查汇总。

## 十、剩余工作与冻结条件

### 优先级 1：冻结实验一

1. 确认最终使用 post-PCA 标准化版本；
2. 为主要比较计算 fold-level 配对差异和置信区间；
3. 固定随机种子、环境、输入文件和结果目录；
4. 明确 Top-n 的训练折内选择，避免数据泄漏；
5. 冻结 Table 2 和 Fig. 2。

### 优先级 2：完成人工 feature-card 审查

1. 从 225 个 unique latent 中制定分层抽样；
2. 覆盖 9 个标签、表面/语义/混合模式和不同置信度；
3. 至少对论文展示 cards 做双人复核；
4. 记录解释正确、部分正确、表面 artifact、无稳定概念和上下文不足；
5. 冻结 Fig. 3 与 Table 4。

### 优先级 3：冻结结构分析

1. 汇总 stable-core、最小充分 k 和共享矩阵；
2. 区分父子标签共享与真正跨功能共享；
3. 将 feature-card 类型映射到共享结构；
4. 只在结果确有信息量时保留 Fig. 4。

### 优先级 4：可选标注审查

1. 270 条案例的 AI 辅助首轮审查已完成，但不计作人工裁决；
2. 优先人工复核 28 条代表案例、全部 6 条混合行为候选，以及高模型分数的 probable reference-label error；
3. 对至少 20% 案例执行第二名独立 MISC coder 盲审，并报告一致率；
4. 所有 probable reference-label error 必须经人工复核后才能计入标注问题；
5. 缺少 client context 的 RES/REC 案例不得直接判错；
6. 仅在人工审查形成清晰且可复现的类别时保留 Fig. 5。

### 结果冻结门槛

只有满足以下条件后才转向最终写作：

- probing 主表和图可由固定命令复现；
- 主要比较的统计口径一致；
- 代表性 feature cards 经过人工确认；
- 所有论文主张均有对应结果文件；
- PCA 负结果和上下文限制被明确写入；
- 不再把自动参考标签写成人工 gold labels；
- 不使用因果性措辞描述 probe、Top-activation 或最小充分子空间结果。

## 十一、主要风险与应对

### 风险 1：PCA 性能明显强于 SAE

**应对：** 不把性能胜出作为 SAE 贡献；强调随机基线优势、稳定稀疏证据、文本可追溯性和结构发现。增加对 PCA 主成分难以逐特征解释的实证比较，而不是仅作概念性陈述。

### 风险 2：解释模型把表面模式误写为心理功能

**应对：** feature card 强制分开 surface 与 semantic，展示对照、替代解释和失败案例，并引入人工审核。

### 风险 3：标签不是人工金标准

**应对：** 全文使用 reference annotation；补充人工抽样审查，或在条件允许时引入 MISC 专业编码者。

### 风险 4：缺少 client context

**应对：** 限制 RES/REC/RE 的强结论，将其作为数据与任务边界；优先展示不高度依赖前文的 QUO、QUC、GI、AF 等案例。

### 风险 5：可解释性证据仍是相关性的

**应对：** 明确当前研究回答的是 representation structure 和 human-inspectable evidence。若后续资源允许，再对少量代表 latent 做激活干预或 token-level attribution，作为增强实验而非当前主张前提。

## 十二、需要导师确认的关键决策

1. 是否接受将论文核心从“SAE 性能优于 PCA”调整为“SAE 提供不同于 PCA 的人类可检查稀疏证据”；
2. 是否能够获得 MISC 专业编码者参与 feature-card 与标注不一致案例复核；
3. 是否将 RE、QU 父标签保留在主结果，还是只放层级一致性分析；
4. Fig. 5 标注审查是否作为主文实验，取决于人工审核完成度；
5. 是否投入额外资源进行少量 latent intervention，以增强忠实性证据；
6. 在 AAAI 2027 正式 CFP 发布后，重新确认 HAI Track 的主题、篇幅和 artifact 要求。

## 十三、当前阶段结论

本项目已经建立了较完整的量化与结构分析基础。现有结果表明，Llama-3.1-8B 的 SAE 表征保留了显著的心理咨询行为信息，且行为信息集中在非随机的稳定稀疏子集中。不同标签需要的特征数量不同，并存在独占、共享和层级重叠结构。与此同时，PCA 在纯预测性能上明显优于 Top-n SAE，自动生成的 latent 解释也尚未完全转化为人类确认的 feature cards。

因此，最稳健且有研究价值的论文方向不是宣称 SAE 是最佳低维分类表示，而是展示一套将内部稀疏特征转化为人类可检查证据的完整方法，并利用这套方法揭示心理咨询行为在 LLM 内部的紧凑性、共享性、碎片化及表面/功能纠缠。下一阶段应停止继续扩展无关实验，优先冻结 probing 结果、完成人工 feature-card 审核和结构图表，再进入论文写作。
