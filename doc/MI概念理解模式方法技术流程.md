# LLM 对 MI/MISC 概念理解模式的方法技术流程文档

## 1. 文档目的

本文档定义一个完整的方法技术流程，用于研究以下核心问题：

> LLM 在处理 Motivational Interviewing (MI) / MISC 咨询师行为标签时，其内部表征更像是在捕捉表面语言形式，还是捕捉更接近咨询行为功能和 MI 原则的核心内容？

这里的“理解”不直接等同于人类临床理解，也不直接等同于因果机制。本文将其操作化为一组可验证问题：

1. 人工 MISC 标签能否从 LLM 内部表征中被稳定解码出来。
2. 与标签相关的 SAE latents 是否具有可审计的高激活证据。
3. 这些高激活证据更像 surface-form、dialogue-function、MI-principle、context-relation，还是 artifact。
4. 在控制表面形式和咨询功能的 minimal pairs 中，latent 激活是否支持“功能驱动”而非“表面驱动”解释。
5. triggering-token 检查是否显示 latent 主要由局部词、短语、标点或模板触发。

本文档不提供代码实现，而是给出研究问题、子问题、变量定义、证据类型、判定规则、流程阶段、输出产物和结论边界。

## 2. 当前项目对接范围

当前项目目录：

`D:/project/NLP_re_dataset_model_base`

当前已有主数据与结果位置：

- 行级数据：`outputs/misc_full_sae_eval/records.jsonl`
- 标签矩阵：`outputs/misc_full_sae_eval/label_matrix.csv`
- SAE feature store：`outputs/misc_full_sae_eval/feature_store/utterance_features.pt`
- Top20 Cohen's d latents：`outputs/misc_full_sae_eval/interpretability/top20_cohensd_latent_utterances/top20_cohensd_latents_by_label.csv`
- P3 feature cards：`outputs/misc_full_sae_eval/interpretability/p3_feature_cards`
- P3 中文说明：`doc/P3流程详解.md`
- P3 自动化指南：`doc/P3_AI流水线评估与自动化实现指南.md`

当前核心标签范围：

`RE, RES, REC, QU, QUO, QUC, GI, SU, AF`

当前 P3 已有状态：

- 每个标签 20 个正向 Cohen's d latents。
- 总计 180 个 feature cards。
- 当前 evidence 只包含 counselor current utterance。
- 缺少前一句 client utterance，因此 RES、REC、RE 的 context-relation 结论必须降级。
- 当前 P3 是候选解释与审阅材料，不是因果机制证明。

## 3. 核心研究问题

### 3.1 主研究问题

LLM 对 MI/MISC 咨询师行为标签的内部表征模式是什么？

更具体地说：

> 与 MISC 标签相关的 SAE latents 和表征空间，主要编码的是表面形式线索，还是更接近咨询行为功能、MI 原则和上下文关系的核心内容？

### 3.2 关键概念边界

本文中的“MI 概念”指 MISC 咨询师行为标签所对应的可编码行为类别，例如：

- `QUO`：开放式提问。
- `QUC`：封闭式提问。
- `RES`：简单反射。
- `REC`：复杂反射。
- `GI`：提供信息。
- `SU`：支持。
- `AF`：肯定。
- `RE`：反射类更宽泛标签。
- `QU`：提问类更宽泛标签。

本文中的“核心内容”不等同于抽象心理状态本身，而是指比词形和模板更接近 MISC codebook 功能定义的证据，例如：

- 是否邀请来访者展开。
- 是否限制来访者只能回答是/否或短答案。
- 是否反映来访者已经表达的内容。
- 是否提供事实信息、建议或解释。
- 是否表达支持、肯定、合作、非评判或自主性支持。

本文中的“表面形式”指当前 utterance 内可直接观察到的语言形态，例如：

- 问号。
- `what/how/why/do you/can you` 等疑问词或助动词结构。
- `sounds like/you feel/I understand` 等固定短语。
- 句长、标点、大小写、转录格式。
- 高频模板化表达。

## 4. 子问题定义

### RQ1：标签是否可从内部表征中解码？

问题：

> MISC 人工标签能否从 LLM hidden states、PCA 表征和 SAE latent activations 中被稳定识别？

用途：

该问题回答“表征中是否存在与标签相关的信息”。它不能回答“模型是否真正理解概念”，但能建立后续 latent 分析的必要前提。

需要比较的表征：

- Raw hidden activation。
- Full-rank PCA hidden representation。
- Full SAE latent representation。
- 经过 feature filtering 后的 SAE candidate pool。
- Top-n SAE latent subspace。

主要指标：

- AUC：标签正例与负例的可分性。
- Macro F1：多标签或多类别下各标签 F1 的平均。
- Balanced accuracy：类别不平衡时的平均召回能力。
- Per-label AUC：每个标签独立的可解码程度。

判定逻辑：

- 若 raw hidden、PCA、SAE 都能高 AUC 解码标签，说明标签信息在该层表征中存在。
- 若 SAE 接近 raw hidden，说明 SAE 重构后的稀疏表示保留了标签相关信息。
- 若 top-n SAE subspace 接近 full SAE，说明少量 latents 可能承载了大部分可解码标签信息。
- 若 top-n 明显超过 full SAE 或 raw hidden，需要检查特征选择泄漏、交叉验证方式、标签不平衡和数据重复。

### RQ2：哪些 SAE latents 与 MISC 标签相关？

问题：

> 在过滤明显无用或不稳定 features 后，哪些 SAE latents 与每个 MISC 标签有稳定统计关联？

用途：

该问题为后续解释阶段提供候选 latent，而不是直接给 latent 命名。

候选池过滤应先去除：

1. 几乎从不激活的 features。
2. 几乎所有样本都激活的 features。
3. 激活极不稳定或主要由少数异常值支配的 features。

建议定义：

- activation rate：某 latent 在全数据集上激活值大于阈值的比例。
- active count：激活样本数。
- concentration@1%：前 1% 样本贡献的总激活占比。
- outlier dominance：最大或极高分位激活相对中位激活的支配程度。
- finite validity：是否存在 NaN、Inf 或异常存储值。

建议默认过滤口径：

- activation rate 过低：低于 0.5% 或 active count 低于最小可分析样本数。
- activation rate 过高：高于 95%，且区分度弱。
- concentration@1% 过高：说明 latent 可能只由少数异常样本驱动。
- 非有限值、全零、常数列直接排除。

关联指标：

- Cohen's d：目标标签正例与负例之间的平均激活差异。
- directional AUC：该 latent 单独区分目标标签的方向性 AUC。
- precision@k：激活最高的 k 条样本中，目标标签正例的比例。
- p-value / FDR：统计显著性控制，但不能替代效果量。

判定逻辑：

- 先用过滤后的 candidate pool。
- 每个标签按正向 Cohen's d 排序。
- 只把 top latents 当作候选证据入口。
- 不把单个 latent 的高 Cohen's d 解释为“该 latent 就是某标签”。

### RQ3：latent 高激活证据属于哪类模式？

问题：

> 每个标签相关 latent 的 top activating utterances 更像表面形式、对话功能、MI 原则、上下文关系，还是数据 artifact？

用途：

该问题是本文的核心解释任务。

模式类型定义如下。

#### Surface-form pattern

定义：

当前 counselor utterance 内可直接观察的词形、句法、标点或模板线索。

例子：

- 问号。
- `what/how/why`。
- `do you/can you/would you`。
- `sounds like/you feel`。
- 句子很短或很长。
- 固定开头或固定结尾。

判定标准：

- 即使不理解咨询功能，也能凭文本形态识别。
- 改写表面形式后，latent 激活可能明显下降。
- 可能跨多个 MISC 标签激活，因为它识别的是形式而不是 code 功能。

#### Dialogue-function pattern

定义：

utterance 在咨询对话中执行的行为功能。

例子：

- 邀请来访者展开。
- 追问细节。
- 封闭式确认。
- 反射或改写来访者内容。
- 提供信息。
- 表达肯定或支持。
- 提出建议。

判定标准：

- 多种表面表达能实现同一功能，latent 对这些改写保持较稳定激活。
- 高激活样例在功能上相似，而不只是共享词形。
- high_non_target 样例能揭示它是否跨标签族激活。

#### MI-principle pattern

定义：

与 MI 精神或原则相关的互动取向。

例子：

- 支持自主性。
- 合作而非命令。
- 非评判。
- 同理。
- 肯定努力。
- 控制性建议。
- 对抗性或羞辱性表达。

判定标准：

- 需要超出局部词形的语用判断。
- 可能跨多个 MISC 行为标签出现。
- 需要人工 MI coder 或更强对照样本辅助判断。

#### Context-relation pattern

定义：

counselor utterance 与前一句或前几句 client utterance 的关系。

例子：

- 复述 client 的内容。
- 改写 client 的意义。
- 推断 client 的情绪。
- 总结前文。

当前限制：

当前第一阶段 evidence packet 没有前一句 client utterance。因此对 RES、REC、RE 的 context-relation 只能记为“待验证”，不能强判。

#### Artifact pattern

定义：

与咨询功能无关，但可能被模型或 SAE 捕捉到的数据、格式或采集痕迹。

例子：

- 转录模板。
- ASR 错误。
- 重复句。
- 来源文件格式。
- speaker marker。
- 非自然标点。
- 极端长度。
- 某个数据源特有话术。

判定标准：

- 高激活样例共享格式或来源，而非咨询功能。
- 重复文本过多。
- 改写或换数据源后激活不稳定。
- high_non_target 也大量激活。

#### Mixed / unclear pattern

定义：

多个模式混合，或证据不足以给出稳定解释。

判定标准：

- top activating examples 之间缺少稳定共同点。
- surface、function 和 artifact 解释都可能成立。
- 需要 minimal pairs、triggering tokens 或人工审核进一步判定。

### RQ4：如何区分表面驱动和功能驱动？

问题：

> 如果一个 latent 高激活于某个标签，它到底是被表面形式触发，还是被咨询功能触发？

核心方法：

构造 minimal pairs，分别控制表面形式 `S` 和咨询功能 `F`。

变量定义：

- `S+`：保留目标表面形式。
- `S-`：改变目标表面形式。
- `F+`：保留目标咨询功能。
- `F-`：改变目标咨询功能。

四类样本：

| 条件 | 表面形式 | 咨询功能 | 用途 |
|---|---|---|---|
| `S+F+` | 保留 | 保留 | 原始目标模式或理想正例 |
| `S+F-` | 保留 | 改变 | 测试是否只是表面驱动 |
| `S-F+` | 改变 | 保留 | 测试是否能跨表面形式识别功能 |
| `S-F-` | 改变 | 改变 | 负对照 |

判定逻辑：

- 若 `S+F-` 仍高激活，而 `S-F+` 明显下降，则更支持 surface-driven。
- 若 `S-F+` 仍高激活，而 `S+F-` 明显下降，则更支持 function-driven。
- 若 `S+F+` 最高，`S+F-` 与 `S-F+` 都中等，则说明 surface 和 function 可能共同贡献。
- 若四类差异不稳定，则标记为 mixed / unclear。
- 若异常格式或模板一出现就高激活，则标记 artifact risk。

### RQ5：triggering tokens 是否解释 latent 激活？

问题：

> latent 激活是否集中由局部 token、短语、标点或模板触发？

用途：

triggering-token 检查用于区分：

- token/phrase-level surface feature。
- phrase template artifact。
- 更分布式的 dialogue-function representation。
- 更稳定的 MI-principle pattern。

检查对象：

- `what/how/why`。
- `do you/can you/would you`。
- `sounds like/you feel/you are`。
- `I understand/I appreciate`。
- `recommend/should/need to/because`。
- 问号、冒号、转录符号。
- 重复模板短语。

建议证据类型：

- Token-level SAE activation trace。
- Local phrase occlusion。
- Phrase substitution。
- Paraphrase robustness。
- Counterfactual deletion。

判定逻辑：

- 若删除或替换少数 token 后激活大幅下降，则支持 surface/token-triggered 解释。
- 若换掉具体词形但保留功能后激活稳定，则支持 function-driven 解释。
- 若激活集中在标点、speaker marker 或格式字符上，则支持 artifact 解释。
- 若激活分布在多个语义相关 token 上，且跨改写稳定，则支持更高层功能解释。

### RQ6：这些 latent 是否影响模型行为？

问题：

> 与 MISC 标签相关的 latent 是否不仅可解释，而且对模型输出或 probe 预测有可干预影响？

当前状态：

当前 P3 主要是 utterance-level activation 和 input-centric evidence，不足以证明因果机制。

后续需要的验证：

- SAE decoder direction stimulation。
- latent ablation。
- target label probe logit change。
- sibling label logit change。
- 生成式输出变化。

结论边界：

- 若仅有 top activating evidence，只能说 candidate interpretation。
- 若 minimal pairs 和 triggering-token 检查通过，可说 contrastively supported interpretation。
- 若 intervention 影响 label probe 或输出，可说 intervention evidence for model behavior。
- 即便有 intervention，也应避免直接写“LLM 真正理解了 MI”。

## 5. 总体证据等级

建议把证据分为六级。

| 级别 | 名称 | 能支持的说法 | 不能支持的说法 |
|---|---|---|---|
| L0 | 数据与标签可用 | 数据可对齐、标签可分析 | 模型表征含有该概念 |
| L1 | Representation decodability | 标签可从表征中解码 | 单个 latent 有语义 |
| L2 | Latent association | 某 latent 与标签统计相关 | latent 就是该标签 |
| L3 | Evidence packet interpretation | top examples 支持候选模式 | 因果机制 |
| L4 | Contrastive validation | surface/function 归因更可信 | 完整临床理解 |
| L5 | Intervention evidence | latent 对 probe/输出有影响 | 人类式 MI 理解证明 |

当前项目的 P3 主要处于 L2-L3。Minimal pairs 与 triggering-token 检查可以把部分 latent 推到 L4。真正的 causal ablation / steering 才可能进入 L5。

## 6. 方法技术流程

### Phase 0：数据契约与标签定义

目标：

保证所有后续分析基于同一行级数据。

输入：

- `records.jsonl`
- `label_matrix.csv`
- feature store
- label definitions / MISC codebook 摘要

需要固定：

- 每一行对应一个 counselor utterance。
- row index 在 records、label matrix、SAE feature store 中一致。
- 当前是否有 client context。
- 每个标签的正例、负例和重叠标签情况。
- 是否允许多标签共存。

输出：

- 数据合同说明。
- 标签分布摘要。
- 缺失值、重复文本、来源 split 摘要。
- 当前阶段 context limitation 声明。

当前项目特别限制：

当前第一阶段没有前一句 client utterance，因此 context-relation 类判断不能作为主结论。

### Phase 1：层与表征空间固定

目标：

明确分析哪个 LLM 层、哪个表征空间。

当前主线：

- Llama/OpenMOSS 主线 SAE 对应 layer 19。
- hook point：`blocks.19.hook_resid_post`。
- 当前 SAE 分析应严格对应这一层。

需要比较的表征：

- raw hidden。
- full PCA。
- full SAE。
- filtered SAE candidate pool。
- top-n SAE subspace。

原则：

- full-rank PCA 应视为标准化 hidden space 的旋转等价或近似等价基线，而不是天然更弱的压缩基线。
- 若使用 truncated PCA，必须明确维度和解释目的。
- SAE 有重构损失，因此 full SAE 不一定超过 raw hidden；若超过，需要检查评估流程是否有泄漏或正则化差异。

输出：

- 层选择说明。
- 表征空间比较报告。
- 每个标签的 decoding 指标。

### Phase 2：全表征 probe 基线

目标：

回答 RQ1：人工标签能否从内部表征中被识别出来。

设计：

- 对每个标签训练二分类 probe。
- 输入分别为 raw hidden、full PCA、full SAE。
- 使用一致的 train/validation/test split。
- 保证所有 preprocessing 只在训练集 fit。
- 对每个标签报告 AUC、F1、balanced accuracy。

关键检查：

- PCA 是否在全数据上 fit。若是，则存在数据泄漏风险。
- 特征选择是否使用了 test label。若是，则 top-n 结果不能作为独立泛化证据。
- 数据重复是否跨 train/test 泄漏。
- 类别不平衡是否导致 accuracy 虚高。

结论口径：

可以说：

> MISC 标签信息在该层内部表征中可被线性 probe 解码。

不能说：

> 模型以人类方式理解了 MISC。

### Phase 3：SAE feature candidate pool 过滤

目标：

先排除明显无分析价值或高风险 features，再做 top latent 选择。

过滤类别：

1. Dead or near-dead features。
2. Ubiquitous features。
3. Outlier-dominated unstable features。

建议每个 latent 计算：

- activation rate。
- nonzero mean activation。
- active sample count。
- concentration@1%。
- top activation / median active activation。
- finite value ratio。
- label coverage count。

输出：

- feature_filter_manifest。
- retained feature list。
- dropped feature list 与原因。
- 过滤前后 latent-label association 的比较。

结论口径：

过滤不是为了提高结果，而是为了减少 artifact、异常值和无效 feature 对解释阶段的污染。

### Phase 4：latent-label association

目标：

回答 RQ2：哪些 latents 与每个标签正向相关。

设计：

- 对每个目标标签，把样本分为正例和负例。
- 对每个 retained latent 计算 Cohen's d、directional AUC、precision@10、precision@50、p-value、FDR。
- 每个标签选取正向 Cohen's d 最高的 top-k latents。

建议主口径：

- top-k = 20。
- 排序主指标为正向 Cohen's d。
- 平局使用 directional AUC、precision@50、latent_idx 稳定排序。
- 不使用 abs(Cohen's d) 作为主口径，因为负向关联 latent 与“目标标签高激活”含义不同。

输出：

- 每标签 top latent 表。
- latent-label matrix。
- 候选 latent 审计报告。

结论口径：

这些 latents 是“与标签相关的候选 features”，不是“标签概念本身”。

### Phase 5：latent evidence packet 生成

目标：

为每个候选 latent 收集可审计证据。

每个 packet 应包含：

- top_activating：全数据集中 activation 最高的样例。
- high_non_target：目标标签负例中 activation 最高的样例。
- random_target：目标标签正例随机样例。
- sibling_code_contrast：相邻或易混标签的对照样例。
- surface_matched_contrast：表面形式相似但目标标签不同的样例。
- low_activation_scoring：低激活样例。

每条样例至少包含：

- row_idx。
- activation。
- unit_text。
- target_match。
- active_labels。
- source metadata。
- duplicate flag。

当前阶段限制：

只基于 counselor current utterance。不要推断缺失的 client context。

输出：

- labeled evidence packets。
- blind evidence packets。
- examples CSV。
- summary CSV。
- manifest。

用途：

- blind 版本用于减少先入为主。
- labeled 版本用于后续标签对齐审计。

### Phase 6：P3 谨慎功能归纳

目标：

回答 RQ3：每个 latent 的高激活证据更像哪类模式。

两阶段审阅：

1. Blind induction pass：只看盲审 packet，不看 target label。
2. Labeled alignment pass：再看 target label、target_match rate、active label counts 和 high_non_target 样例。

每个 latent 输出：

- one-sentence tentative interpretation。
- main recurring patterns。
- pattern types。
- relationship to target label。
- adjacent label risks。
- artifact risks。
- evidence quality。
- candidate name。
- alternative explanations。
- recommended follow-up checks。
- final concise conclusion。

强制措辞：

- appears associated with。
- may capture。
- candidate explanation。
- no stable interpretation can be assigned。

禁止措辞：

- this feature is QUO。
- this latent proves the model understands MI。
- causal mechanism。

输出：

- `latent_function_reviews.jsonl`
- `latent_function_reviews.csv`
- `latent_function_patterns.csv`
- label-level summary
- 中文总报告

### Phase 7：P3 held-out scoring

目标：

检验候选解释是否能泛化到未用于归纳的样例。

两个 scoring task：

1. Activation prediction：根据候选解释，判断某句是否应高激活该 latent。
2. Code discrimination：根据候选解释，判断某句是否符合目标 MISC 标签，而不是 sibling 或 surface-matched negative。

主要指标：

- accuracy。
- F1。
- balanced accuracy。
- precision。
- recall。

注意：

若 confidence 分布高度集中，AUROC 不应作为主指标。

解释：

- Activation prediction 更接近 feature behavior。
- Code discrimination 更接近 MISC label relation。
- 两者都高，说明解释既能描述 latent 激活，也与标签有较强关系。
- activation prediction 高但 code discrimination 低，说明 latent 可能捕捉的是非标签特异模式。
- code discrimination 高但 activation prediction 低，需要检查解释是否借用了标签先验。

### Phase 8：minimal pair validation

目标：

回答 RQ4：latent 是 surface-driven 还是 function-driven。

样本构造原则：

- 每个待验证 latent 选取代表性原始样例。
- 为该样例构造 `S+F+`, `S+F-`, `S-F+`, `S-F-` 四类变体。
- 每组至少包含多个 paraphrase，避免单句偶然性。
- 对 QUO/QUC、RES/REC、GI/SU/AF 等标签分别设计不同 minimal pair 模板。

示例设计思路：

QUO：

- `S+F+`：使用 what/how 问句并邀请展开。
- `S+F-`：保留 what/how 形式，但只要求一个事实性短答或封闭确认。
- `S-F+`：不用 what/how，但仍邀请展开。
- `S-F-`：不用开放式形式，也不邀请展开。

QUC：

- `S+F+`：助动词开头并要求 yes/no 或短答。
- `S+F-`：助动词开头但实际邀请展开。
- `S-F+`：不用典型 yes/no 句法但功能仍是封闭确认。
- `S-F-`：既不用封闭式形式，也不执行封闭功能。

RES/REC：

- 当前没有 client context 时，只能测试 reflective phrasing。
- 若加入 client context，应测试简单复述、复杂改写、情绪/意义推断。

GI：

- `S+F+`：解释、事实信息或建议。
- `S+F-`：保留 because/should/recommend 等词，但不提供实质信息。
- `S-F+`：不用典型信息词，但仍提供事实解释。
- `S-F-`：既无信息形式，也无信息功能。

判定指标：

- 四类样本的 latent activation 均值。
- `S-F+` 相对 `S+F+` 的保留比例。
- `S+F-` 相对 `S+F+` 的保留比例。
- `S-F-` 的负对照激活。
- 方差与样本稳定性。

建议分类：

- surface-driven。
- function-driven。
- surface-function hybrid。
- artifact-driven。
- mixed / unclear。

### Phase 9：triggering-token 检查

目标：

回答 RQ5：latent 是否由局部 token 或固定短语触发。

优先证据：

- token-level SAE activation。
- latent 对每个 token position 的 activation trace。

若当前只有 utterance-level pooled activation，则使用替代方法：

- 删除疑似触发短语。
- 替换疑似触发短语。
- 保留功能但改写表面形式。
- 保留表面形式但改变功能。
- 比较 activation delta。

需要记录：

- suspected trigger。
- trigger type：question word、reflection phrase、positive word、information phrase、punctuation、template、artifact。
- deletion effect。
- substitution effect。
- paraphrase stability。
- conclusion。

判定逻辑：

- 单个局部 token 删除后激活崩塌：更像 surface/token feature。
- 多个同义表达都高激活：更像 function feature。
- 标点或格式触发：artifact risk 高。
- 无稳定 trigger：可能是分布式功能，也可能是 mixed，需要结合 minimal pairs。

### Phase 10：人工 MI coder 审核

目标：

把自动解释转化为可发表前的人工审阅结论。

审核者需要判断：

- candidate name 是否过度概念化。
- pattern type 是否合理。
- 是否存在 artifact risk。
- 是否支持 target label。
- 是否更像 family-level pattern，而不是 label-specific pattern。
- 对 RES/REC 是否因缺少 client context 而无法判断。

建议 final status：

- robust_code_candidate。
- family_level_candidate。
- subskill_candidate。
- surface_artifact。
- mixed_unclear。
- reject。

当前自动结果只能作为 draft，不应替代人工审核。

### Phase 11：机制验证与输出侧实验

目标：

回答 RQ6：latent 是否影响模型行为或 probe 判断。

可选实验：

- latent ablation。
- SAE decoder direction stimulation。
- probe logit delta。
- sibling label logit delta。
- generated response behavior change。

最小证据：

- 对目标 latent 增强或抑制。
- 观察目标标签 probe logit 是否按预期变化。
- 同时观察 sibling labels，避免只是整体激活幅度变化。
- 使用多个 alpha 强度。
- 使用 held-out examples。

结论口径：

若 intervention 结果与 P3 解释一致，可写：

> This provides intervention evidence consistent with the candidate interpretation.

仍不应写：

> This proves the LLM understands MI concepts.

## 7. 综合判定规则

### 7.1 latent 类型判定

建议每个 latent 最终给出以下字段：

- target_label。
- latent_idx。
- dominant_pattern_type。
- secondary_pattern_type。
- surface_score。
- function_score。
- mi_principle_score。
- context_relation_score。
- artifact_score。
- evidence_quality。
- final_status。
- required_followup。

### 7.2 类型判定标准

Surface-form latent：

- top examples 共享明显词形或句式。
- triggering-token 检查显示局部词或标点支配激活。
- minimal pairs 中 `S+F-` 仍高激活。
- `S-F+` 激活明显下降。

Dialogue-function latent：

- top examples 在咨询功能上稳定。
- 多种表面形式仍高激活。
- minimal pairs 中 `S-F+` 保持激活。
- triggering-token 不集中于单一词形。

MI-principle latent：

- 高激活样例共享自主支持、合作、肯定、同理、非评判或对抗性等互动取向。
- 可能跨多个 MISC 标签。
- 需要人工 MI coder 判断。
- 需要排除积极词或固定礼貌短语 artifact。

Context-relation latent：

- 必须有 client context。
- 能显示 counselor utterance 与 client utterance 的复述、改写、意义推断关系。
- 当前无 client context 时不能强判。

Artifact latent：

- 激活由格式、来源、重复文本、ASR 错误、极端长度或模板驱动。
- high_non_target 中也大量出现。
- minimal pairs 或 paraphrase 后不稳定。

Mixed / unclear：

- 证据冲突。
- top examples 缺少共同点。
- surface/function/artifact 都可能解释。
- 需要后续验证。

### 7.3 标签级结论判定

每个标签最终不只报告“有多少 robust latents”，还应报告：

- 主要 pattern type 分布。
- surface-driven 比例。
- function-driven 比例。
- artifact-risk 比例。
- mixed/unclear 比例。
- 是否需要 client context。
- 与 sibling labels 的混淆模式。

预期标签级解释：

- QU/QUO/QUC：更容易出现 question-form 与 elaboration/closed-question function 的混合。
- RES/REC/RE：若无 client context，容易退化为 reflective phrasing，而非真实 reflection。
- GI：可能同时包含信息提供功能、医学主题词和建议模板。
- SU：可能同时包含支持功能、安慰短语和礼貌模板。
- AF：可能同时包含肯定功能和正向评价词。

## 8. 论文或报告中的主线叙述

建议主线：

1. 先证明 MISC 标签信息能从 LLM 表征中被解码。
2. 再说明 SAE 提供了稀疏 latent 级候选入口。
3. 然后用 evidence packets 和 P3 审阅，把 latent 解释分为 surface、function、MI principle、context relation、artifact。
4. 再用 minimal pairs 区分表面形式和咨询功能。
5. 再用 triggering-token 检查判断是否由局部词或模板驱动。
6. 最后把可干预实验作为更强机制证据的后续阶段。

可以使用的核心表述：

> This study treats SAE latents as auditable candidate evidence for MI/MISC-related representational patterns, and asks whether these patterns are primarily surface-form driven or more consistent with counseling-function and MI-principle interpretations.

中文表述：

> 本研究不直接声称 SAE latent 等同于 MISC 概念，而是将其作为可审计的候选证据，检验其高激活模式究竟更像表面语言形式，还是更接近咨询行为功能与 MI 原则。

## 9. 不能过度声称的内容

不能说：

- 某个 latent 就是 QUO、RES 或 GI。
- SAE 证明 LLM 理解了 MI。
- top activating examples 证明了模型的临床推理机制。
- Cohen's d 高就表示该 latent 是核心概念。
- P3 自动解释可以替代 MI coder 审核。
- 没有 client context 时可以确认 RES/REC 的上下文反射功能。

可以说：

- 某些标签可从该层表征中被线性解码。
- 某些 SAE latents 与 MISC 标签存在统计关联。
- 某些 latents 的高激活样例支持候选 surface-form 或 dialogue-function 解释。
- minimal pairs 和 triggering-token 检查可提高解释类型归因的可信度。
- 当前证据支持 structured audit 和 candidate interpretation，而非因果机制证明。

## 10. 最终交付物设计

建议最终形成以下材料：

### 数据与表征层

- 数据合同说明。
- 标签分布报告。
- 层选择报告。
- raw hidden / PCA / SAE probe 对比报告。

### latent 筛选层

- feature filtering manifest。
- retained latent list。
- latent-label association matrix。
- top20 latents by label。

### 解释证据层

- latent evidence packets。
- blind review prompts。
- labeled alignment prompts。
- latent function reviews。
- P3 feature cards。

### 验证层

- held-out scoring metrics。
- minimal pair activation report。
- triggering-token report。
- artificial surface/function contrast report。
- optional intervention report。

### 论文结果层

- label-level pattern summary。
- robust candidate latent list。
- artifact-risk latent list。
- mixed/unclear latent list。
- representative case studies。
- limitations and claim-boundary section。

## 11. 当前项目的下一步建议

建议按以下顺序推进。

第一步：固定当前 P3 自动结果。

- 保留 180 cards 的当前输出。
- 用 manifest 和 validate 记录当前状态。
- 不再让旧 PCA 或旧 top-n 结论污染新报告。

第二步：为 top robust candidates 做人工审核。

- 优先审查 `robust_code_candidate`。
- 对 `surface_artifact` 和 `mixed_unclear` 抽样复核。
- 重点审查 QUO、QU、QUC、GI、SU、AF。
- RES、REC、RE 等待 client context 或降低结论强度。

第三步：设计 minimal pairs。

- 先从每个标签选 3-5 个代表性 latent。
- 每个 latent 生成 `S+F+`, `S+F-`, `S-F+`, `S-F-`。
- 先做小规模验证，确认模板不会引入新 artifact。
- 再扩展到更多 robust candidates。

第四步：做 triggering-token 检查。

- 如果能获取 token-level SAE activation，优先使用 token trace。
- 如果当前只有 utterance-level activation，则先做 phrase deletion / substitution / paraphrase。
- 把结果回填到每张 feature card。

第五步：整合成论文方法与结果。

- 主结果报告 label-level pattern distribution。
- 案例分析展示代表性 latent。
- 明确说明哪些是 surface-driven，哪些更接近 function-driven。
- 把 intervention 留作后续机制验证，除非已经实际完成。

## 12. 一句话总结

本流程把“LLM 是否理解 MI 概念”拆解为一组可审计、可验证、可降级表述的技术问题：标签是否可解码，相关 SAE latents 是否有稳定高激活证据，这些证据更像表面形式还是咨询功能，以及 minimal pairs 和 triggering-token 检查是否支持更强的功能性解释。

