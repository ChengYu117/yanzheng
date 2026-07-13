# SAE（Sparse Autoencoder，稀疏自编码器）线性探针未检出与错误检出案例分析报告

## 一、结论先行

本实验分析 stable-core SAE 线性探针与当前 MISC（Motivational Interviewing Skill Code，动机性访谈技能编码）参考标签不一致的案例，重点包括正例但probe预测为负例（FN，false negative，假阴性/未检出）、负例但probe预测为正例（FP，false positive，假阳性/错误检出），以及模型分数远离0.5阈值的高模型分数不一致。

主要结论如下：

1. **不一致具有明显的标签特异性，不是均匀随机误差。** GI、SU 和 RES 的 FN 率分别达到 47.1%、34.7% 和 33.1%，说明这些标签存在较明显的未检出问题；QUO、QUC 和 AF 的FN率相对较低。
2. **高模型分数不一致主要来自错误检出。** 七个叶标签共有64个高分FN和841个高分FP。说明probe较少以极低分强烈否定正例，却更容易以极高分把相似表面形式或相邻行为判断为目标标签。
3. **错误检出具有可解释的latent模式。** AF常被`good/great/perfect`等一般积极词触发；REC/RES常被`so you`、`sounds like`和第二人称改述触发；QUO/QUC主要依赖问句结构；GI受医疗主题词、数值和风险表达影响；SU受`sorry/help/tough for you`等支持性表达影响。
4. **一部分不一致来自真实模型局限。** 典型情况包括：同一行为采用了不同于stable-core常见模板的表达；线性probe无法区分功能相近的兄弟标签；模型把主题相关性当作行为功能；缺少client前文导致反映行为无法判断。
5. **一部分案例可能存在标签歧义、混合行为或reference-label问题，但当前不能直接判定为人工标注错误。** 对28条代表案例的独立质检发现，AI（人工智能）辅助审查会过度使用“可能标注错误”判断，尤其容易把REC/RES边界和反映式问句误写成标签矛盾。

因此，本实验最稳健的结论不是“发现了大量标注错误”，而是：**SAE线性探针的不一致案例揭示了MISC标签边界、混合行为、表面模式依赖、上下文缺失和稀疏表示覆盖不足。**

## 二、实验设计

### 2.1 预测设置

- 样本数：6,194；
- 标签数：9；
- OOF（out-of-fold，折外预测）label-case 数：55,746；
- 特征：每个标签对应的 stable-core SAE latent；
- 分类器：`LogisticRegression(class_weight="balanced", C=1.0, solver="liblinear")`；
- 划分：按 `file_id` 的5折 stratified-group cross-validation；
- 判定阈值：0.5；
- 所有45个fold的训练/测试文件重合数为0。

### 2.2 不一致定义

- **FN**：reference label = 1，SAE probe probability < 0.5；
- **FP（错误检出）**：reference label = 0，SAE probe probability ≥ 0.5；
- **高模型分数FN**：reference = 1，probability（模型输出的正例概率）≤ 0.1；
- **高模型分数FP**：reference = 0，probability ≥ 0.9。

“高模型分数”不等于统计置信度。当前balanced logistic regression的probability没有进行独立校准。

### 2.3 案例审查

共识别8,194个不一致label-case，其中1,180个达到高模型分数标准。每个标签分别抽取：

- 每种不一致方向中10条最大margin案例；
- 每种不一致方向中5条阈值附近案例。

最终形成270条分层样本。这里的margin用于选择最确定和最接近边界的案例。对每条案例重新拟合其原始OOF fold模型，并计算：

\[
\text{contribution}_{ij}=\beta_j z_{ij}
\]

由此获得支持和反对目标标签的Top latent贡献，并关联已有feature-card解释。重建probability与原始OOF结果的误差小于`1e-6`。

## 三、逐标签不一致结果

RE和QU是父标签（上位标签），只用于层级一致性审查。主要分析以下七个叶标签（更具体的下位标签）。

| 标签 | Reference正例 | FN | FN率 | 高分FN | Reference负例 | FP | FP率 | 高分FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RES | 516 | 171 | 33.1% | 1 | 5,678 | 1,451 | 25.6% | 73 |
| REC | 842 | 176 | 20.9% | 10 | 5,352 | 851 | 15.9% | 145 |
| QUO | 1,206 | 159 | 13.2% | 26 | 4,988 | 530 | 10.6% | 133 |
| QUC | 768 | 141 | 18.4% | 24 | 5,426 | 653 | 12.0% | 178 |
| GI | 681 | 321 | 47.1% | 0 | 5,513 | 785 | 14.2% | 69 |
| SU | 222 | 77 | 34.7% | 1 | 5,972 | 739 | 12.4% | 111 |
| AF | 349 | 80 | 22.9% | 2 | 5,845 | 428 | 7.3% | 132 |

![各叶标签的FN与FP率](../outputs/misc_full_sae_eval/interpretability/sae_annotation_audit_stable_core/semantic_case_analysis/figures/disagreement_rates_by_label.png)

*图1：各叶标签的OOF FN率与FP率。GI、SU和RES的FN率最高，RES同时具有最高的FP率。*

## 四、未检出案例分析

### 4.1 GI：高FN率但没有高分FN

GI共有321个FN，FN率47.1%，但没有probability ≤ 0.1的高分FN。这意味着probe并非强烈排斥GI正例，而是许多正例没有形成足够强的正向latent组合。

可能原因包括：

- stable-core GI latent集中于药物、风险、测量和医疗建议；
- 一般性解释、反馈或非医疗领域的信息提供覆盖不足；
- 部分GI文本表达为立场、建议或简单事实，难以由线性边界统一识别。

代表案例`audit_0031`描述转诊原因，reference为GI，但probe probability仅为0.295。其文本确实执行信息提供功能，但不包含stable-core GI中最常见的药物、数值或风险模板，属于表示覆盖不足。

### 4.2 SU：支持行为表达多样

SU共有77个FN，FN率34.7%，但只有1个高分FN。说明支持行为可能通过共情、验证、帮助提议、许可、道歉等多种形式实现，stable-core子空间难以覆盖全部表达。

代表案例`audit_0255`仅包含“so it makes sense”。该表达可能是一种验证，但缺少client前文时无法确认其支持功能。此类案例同时体现上下文不足和模型覆盖局限。

### 4.3 RES/REC：依赖前文和反映复杂度

RES和REC的FN率分别为33.1%和20.9%。当前数据只有counselor utterance，没有前一句client文本，因此probe和审查者都无法确认当前表达是否重复、推断或重构了client的原意。

代表案例包括：

- `audit_0225`：“out of control”，RES probability=0.499。文本过短，必须依赖前文判断；
- `audit_0187`：“I'm not sure I understand what the problem is.”，REC probability=0.089。当前文本不像复杂反映，但仍不能脱离上下文直接判定reference错误；
- `audit_0195`：第一人称焦虑表达，REC probability=0.499。说话功能和引用关系不清晰，应优先归为上下文不足。

### 4.4 AF和问题标签：模板外表达

AF的FN案例常出现较长的优势确认或承诺评价，而不是简短的`good/great`表达。例如`audit_0001`明确赞扬来访者提出的具体策略，但probability仅为0.078，说明probe可能过度依赖高频积极词模板。

QUO/QUC的FN通常来自转录不规范、复杂句式或问句类型边界。例如`audit_0097`请求一个最大数量，属于QUC，但probability仅为0.074，说明数量功能没有被稳定识别。

## 五、错误检出案例分析

### 5.1 AF：一般积极词被误当作肯定行为

AF的FP率只有7.3%，但428个FP中有132个达到高模型分数。说明AF错误不是广泛随机发生，而是少数表面模式会强烈触发probe。

代表案例`audit_0016`包含`perfect`和`good fit`，但主要功能是确认计划并提问，不是在肯定来访者的优势或努力。probe probability达到1.000，显示积极词latent被错误泛化为AF功能。

### 5.2 REC/RES：反映模板和兄弟标签混淆

REC的高分FP为145个，RES为73个。典型触发包括：

- `sounds like`；
- `so you`；
- 第二人称状态描述；
- 以反映式短语开头、随后转为提问的混合表达。

`audit_0201`以“doesn't sound like”开头，但整体主要询问药物使用情况。REC probability为0.994，表明probe被反映式开头触发。

`audit_0227`的reference为RE|REC、RES=0。AI最初把它判断为漏标RES，但更合理的解释是：文本中的情绪推断可能属于REC，probe将复杂反映错误归入RES。这是兄弟标签边界，而不是标签逻辑矛盾。

### 5.3 QUO/QUC：问句骨架强，但子类边界有限

QUO和QUC分别有133和178个高分FP。probe能强烈识别what/how、助动词、数量和量表等问句形式，但可能无法稳定区分：

- 开放回答与限定回答；
- 反映式问句与普通问题；
- 语法上是问题、功能上是许可或建议的表达。

`audit_0150`是二选一问题，reference为QUC，但QUO probability略高于阈值，属于开放/封闭边界混淆。

### 5.4 GI和SU：主题与功能混淆

GI的FP常由医疗主题、风险词、指标和数值触发，即使文本主要功能是提问或反映。`audit_0060`只是“what is it precisely that you want”，但GI probability略高于阈值，属于上下文或主题过度泛化。

SU的FP常由`sorry`、`help`、`tough`等词触发。部分确实可能是漏标支持行为，部分只是礼貌表达或其他行为中的支持性语气。

## 六、高模型分数不一致

![各叶标签的高模型分数不一致](../outputs/misc_full_sae_eval/interpretability/sae_annotation_audit_stable_core/semantic_case_analysis/figures/high_score_disagreements_by_label.png)

*图2：高模型分数FN与FP数量。七个叶标签中高分FP为841个，高分FN仅64个。*

高分不一致明显偏向FP，说明probe更常见的问题是“看到相似模式后过度预测”，而不是“对真实正例作出强烈否定”。

主要高分FP来源：

- QUC：178个，常由yes/no、数量、量表或事实核验结构触发；
- REC：145个，常由`sounds like`等反映模板触发；
- QUO：133个，常由what/how和探索式结构触发；
- AF：132个，常由积极评价词触发；
- SU：111个，常由帮助、共情或道歉表达触发。

这些结果说明stable-core latent具有可解释的预测作用，但其选择性仍有限：它们能够识别行为相关证据，也会把局部证据错误推广为完整标签。

## 七、270条分层案例的诊断结果

AI辅助主诊断为：

| 主诊断 | FN | FP | 合计 |
|---|---:|---:|---:|
| 可能的reference-label问题 | 59 | 57 | 116 |
| 模型局限 | 26 | 53 | 79 |
| 上下文不足 | 30 | 12 | 42 |
| 阈值边界 | 14 | 5 | 19 |
| 标签歧义 | 5 | 5 | 10 |
| 表面artifact | 1 | 3 | 4 |

允许一条案例具有多个诊断后：

- 标签歧义候选：53条；
- 混合行为候选：6条；
- 可能的reference-label问题：123条；
- 模型局限：115条；
- 表面artifact：33条；
- 上下文不足：73条；
- 阈值边界：65条。

![分层样本的诊断标志](../outputs/misc_full_sae_eval/interpretability/sae_annotation_audit_stable_core/semantic_case_analysis/figures/sampled_diagnosis_flags.png)

*图3：270条分层审查案例的多标签诊断标志。一条案例可以同时具有多个原因，因此各项不能相加为270。*

这270条不是从8,194条不一致中简单随机抽样，而是按标签、方向和margin分层选择。因此上述诊断比例只描述审查样本，不能用于估计全部不一致的真实原因比例。

## 八、代表性混合与歧义案例

### 8.1 反映与问句共存

`audit_0226`：“So you want nicotine replacement but you don't want to quit smoking?”

- Reference：QU|QUC；
- RES probability：0.991；
- 解释：文本既重新表述来访者的矛盾，也以问题形式请求确认；
- 结论：更适合作为RES/QUC混合或标签歧义案例，而不是单纯的probe错误。

### 8.2 GI与反映共存

`audit_0048`总结吸烟历史，reference为RE|REC，但GI probability为0.998。该文本既可能是反映，也可能是基于历史信息的反馈，说明MISC行为单元可能包含多个功能。

### 8.3 SU与反映共存

`audit_0257`：“this is really tough for you the alcohol”

- Reference：RE|REC；
- SU probability：0.997；
- 解释：文本既反映困难，也表达同情和支持；
- 结论：可能是SU漏标，也可能是一个混合行为单元，需要人工coder确认。

## 九、独立质检结论

对28条叶标签代表案例进行独立抽查后，AI生成的latent解释通常能够合理说明probe为什么产生预测，但AI案例裁决存在以下问题：

1. 容易把REC/RES兄弟标签边界误写成reference-label错误；
2. 容易把语法上的问句直接判断为QUO/QUC漏标，而忽略反映式问句；
3. 容易把包含帮助或支持语气的QUO/REC判断为SU漏标；
4. 在缺少client前文时仍可能对RES/REC作出过度确定的判断。

因此，116条“可能的reference-label问题”只能作为人工复核队列规模，不能作为已确认标注错误数量。相较之下，表面词触发、模型覆盖不足、兄弟标签边界和上下文不足是当前更可靠的分析结论。

## 十、研究意义

本实验补充了probe性能指标无法回答的问题。AUC、F1等指标只能说明表征能否支持分类，而不一致案例能够进一步揭示：

- 哪些MI行为表达没有被stable-core子空间覆盖；
- 哪些表面或语义latent会导致错误泛化；
- 哪些标签边界本身具有歧义；
- 哪些行为单元包含多个功能；
- 哪些标签必须依赖对话上下文。

因此，该实验的贡献不是证明SAE预测比reference更正确，而是使用可追溯的latent证据解释模型与标签为什么发生分歧，从而把分类误差转化为可由人类检查的行为边界证据。

## 十一、结论边界

1. 当前`label_matrix.csv`的reference labels来自LLM分段后的MISC标注，并非独立人工gold。如果存在独立人工标签文件，应替换输入后重新运行。
2. FN和FP是相对于0.5阈值及当前reference的操作性名称，不等于真实漏报和误报。
3. balanced logistic regression probability未经校准，高模型分数不等于统计置信度。
4. RE和QU是父标签，其不一致不能与叶标签重复计数或独立解释。
5. RE/RES/REC缺少client前文，不能根据当前utterance直接确认反映类型。
6. feature-card解释和线性probe贡献是相关性、结构性和预测性证据，不构成因果机制证明。
7. 所有可能的reference-label问题必须经过独立MISC coder复核后才能进入论文结论。

## 十二、结果文件

- 完整语义辅助报告：`outputs/misc_full_sae_eval/interpretability/sae_annotation_audit_stable_core/semantic_case_analysis/semantic_disagreement_analysis_report.md`
- 各标签FN/FP率：`disagreement_rates_by_label.csv`
- 高模型分数案例：`high_score_disagreements.csv`
- 270条完整分析：`case_analysis_enriched.csv`
- 多标签诊断统计：`sampled_diagnosis_flags.csv`
- 混合行为候选：`mixed_behavior_candidates.csv`
- 28条代表案例：`representative_cases.csv`

## 十三、专业名词说明

| 术语 | 中文解释 | 本报告中的具体含义 |
|---|---|---|
| SAE | Sparse Autoencoder，稀疏自编码器 | 将LLM的高维内部激活分解为大量较稀疏latent的模型。 |
| MISC | Motivational Interviewing Skill Code，动机性访谈技能编码 | 本项目使用的咨询行为标签体系。 |
| MI | Motivational Interviewing，动机性访谈 | MISC标签所描述的心理咨询沟通框架。 |
| OOF Prediction | Out-of-Fold Prediction，折外预测 | 样本只使用未参与训练它的测试折模型进行预测，避免训练集泄漏。 |
| FN | False Negative，假阴性/未检出 | 参考标签为正，但SAE线性探针预测为负。 |
| FP | False Positive，假阳性/错误检出 | 参考标签为负，但SAE线性探针预测为正。 |
| High-Score Disagreement | 高模型分数不一致 | FN概率≤0.1或FP概率≥0.9的不一致案例；它表示模型分数极端，不等于统计置信度。 |
| MISC label codes | MISC标签代码 | RE/RES/REC表示反映类标签；QU/QUO/QUC表示提问类标签；GI、SU、AF分别表示信息提供、支持和肯定。 |

## 十四、阅读提示

阅读本报告时，可以先看“结论先行”和图1、图2，了解不同标签的不一致结构；再看第四、第五节中的FN/FP案例，理解具体的未检出和错误检出原因；最后结合第十三节术语表阅读latent贡献和feature-card解释。

本报告的核心证据链是：

`reference label → OOF probe prediction → FN/FP direction → latent contribution → feature-card interpretation → human-review candidate`

其中最后一步仍然需要独立人工MISC coder确认。
