# Stable Core Latents P3 流程分析报告

## 1. 报告对象与证据边界

本报告分析的是 stable-core latent set 上运行的 P3 feature-card 流程，核心输出目录为：

`outputs/misc_full_sae_eval/interpretability/p3_feature_cards_stable_core`

本轮 P3 覆盖 9 个核心 MISC 标签，共 303 个 stable-core latents。流程数量校验通过：303 个解释、8934 条 scoring predictions、606 行 latent x task metrics、303 行 manual review template、303 张 final feature cards 均已生成。

需要明确的是，`final_cards/manifest.json` 中 `review_status` 仍为 `manual_draft`，`mi_coder_judgment` 为空，`p3_concept_cluster_seed_table.csv` 为空，`n_seeds_extracted=0`。因此以下分析应被视为 AI 解释卡片的结构化汇总，而不是已经由人工 MISC coder 完成验证的最终结论。

另一个重要限制是：当前数据只包含 counselor current utterance，没有前一句 client utterance。因此对 `RE / RES / REC` 的解释只能说 latent 可能捕捉到反射式措辞或 counselor 表达模式，不能强声称它识别了真实的 client-content reflection。

## 2. Stable Core P3 的整体状态

| 项目 | 数值或状态 |
|---|---:|
| latent cards | 303 |
| labels | 9 |
| source | stable_core |
| fallback generated packets | 303 |
| Task A explanations | 303 / 303 |
| Task B predictions | 8934 / 8934 |
| final cards | 303 |
| manual review status | manual_draft |
| MI coder judgment | not filled |

各标签 stable-core latent 数量：

| 标签 | stable-core latent 数 |
|---|---:|
| RE | 54 |
| REC | 47 |
| QUC | 45 |
| QUO | 32 |
| SU | 31 |
| AF | 27 |
| GI | 26 |
| QU | 26 |
| RES | 15 |

## 3. Final Status 分布

| 标签 | total | robust_code_candidate | surface_artifact | mixed_unclear |
|---|---:|---:|---:|---:|
| AF | 27 | 18 | 3 | 6 |
| GI | 26 | 25 | 0 | 1 |
| QU | 26 | 21 | 0 | 5 |
| QUC | 45 | 8 | 3 | 34 |
| QUO | 32 | 24 | 1 | 7 |
| RE | 54 | 27 | 6 | 21 |
| REC | 47 | 26 | 8 | 13 |
| RES | 15 | 0 | 10 | 5 |
| SU | 31 | 1 | 11 | 19 |

最清晰的标签是 `GI`, `QU`, `QUO`, `AF`。其中 `GI` 有 25 / 26 个 latent 被判为 robust code candidate，说明信息提供类行为在 stable latents 中形成了较清晰的功能/语义簇。`QU` 与 `QUO` 也较清晰，但它们主要依赖 question form、wh-word opener、开放式邀请等形式和功能混合线索。

最弱的是 `RES` 与 `SU`。`RES` 没有 robust candidate，10 / 15 被判为 artifact / uninterpretable。`SU` 只有 1 / 31 被判为 robust，大多数是 mixed 或 surface_artifact。这说明 simple reflection 与 support 在当前 counselor-only 数据中很难被 stable SAE latents 清晰分离。

## 4. 标签级主要组成 Latents

下面的“组成”不是说这些 latent 已被证明等价于某个 MISC 概念，而是指 AI feature-card 给出的候选解释簇及其代表 latent。

| 标签 | 主要组成簇 | 数量 | 代表 latent |
|---|---|---:|---|
| AF | positive evaluative phrase detector, 如 great / good / wonderful | 12 | 23464(r1), 7143(r2), 18492(r3), 30870(r5), 22411(r11) |
| AF | general affirmation signal, broader positive counselor register | 10 | 17793(r4), 6676(r6), 28469(r8), 2434(r10), 24856(r13) |
| AF | weak AF / positive-word cross-label | 5 | 32764(r7), 28724(r9), 1871(r21), 24167(r25), 19654(r26) |
| GI | medical / clinical information delivery | 14 | 16345(r1), 8294(r2), 26879(r3), 27515(r5), 26236(r6) |
| GI | general information-giving signal | 8 | 13751(r4), 17827(r7), 16515(r8), 8166(r12), 17685(r18) |
| GI | explanatory / informational long utterance | 4 | 9893(r10), 5828(r20), 10264(r25), 19483(r27) |
| QU | general question detector | 14 | 13430(r1), 9959(r3), 10916(r6), 21935(r10), 21125(r11) |
| QU | open question initiator, wh-word opener | 8 | 26485(r5), 29590(r9), 18310(r12), 23183(r15), 24943(r17) |
| QU | conditional / scale / auxiliary question subtypes | 4 | 11660(r4), 27061(r7), 664(r2), 12340(r8) |
| QUO | mixed open / closed question pattern | 23 | 9959(r1), 13430(r3), 24744(r6), 11660(r7), 24761(r9) |
| QUO | open-invitation question, what / how / why opener | 8 | 23183(r4), 26485(r5), 18310(r8), 3459(r14), 14833(r19) |
| QUO | scale-based open question | 1 | 664(r2) |
| QUC | mixed utterance with low closed-question specificity | 31 | 21935(r1), 13430(r2), 20869(r4), 18646(r5), 4998(r7) |
| QUC | yes/no closed question, auxiliary inversion | 8 | 14014(r3), 8969(r10), 12340(r11), 20463(r15), 17507(r20) |
| QUC | general question detector overlap | 6 | 22358(r6), 14003(r13), 28816(r16), 24943(r19), 12887(r24) |
| RE | weak / mixed RE signal | 21 | 29759(r1), 14875(r5), 5663(r6), 1516(r7), 26319(r9) |
| RE | general reflection signal, mixed markers | 19 | 19435(r2), 30224(r4), 7054(r8), 10181(r11), 15068(r13) |
| RE | sounds like / seems like reflective phrase anchor | 10 | 3993(r10), 31133(r12), 21800(r24), 29874(r28), 29190(r37) |
| RE | So... reflective discourse opening | 3 | 31930(r3), 28269(r25), 31363(r27) |
| RE | ambivalence / double-sided reflection framer | 1 | 20436(r16) |
| REC | general complex reflection, distributed signal | 19 | 26800(r3), 30224(r4), 19435(r5), 15068(r8), 10181(r10) |
| REC | weak REC signal, mixed labels | 18 | 29759(r6), 3805(r15), 14875(r19), 5663(r20), 9993(r22) |
| REC | sounds like reflective anchor shared with RE | 9 | 31133(r1), 3993(r7), 16292(r9), 21800(r14), 23670(r16) |
| REC | ambivalence / double-sided complex reflection | 1 | 20436(r2) |
| RES | artifact / uninterpretable, very low purity | 10 | 20808(r1), 11435(r3), 29759(r4), 13966(r6), 23077(r7) |
| RES | uninterpretable / low-purity RES signal | 5 | 28269(r2), 1516(r5), 19435(r8), 32727(r11), 16320(r12) |
| SU | general support / suggestion signal, mixed SU | 16 | 29825(r3), 16736(r4), 4756(r5), 30223(r6), 9720(r7) |
| SU | uninterpretable for SU, cross-label noise | 14 | 11948(r2), 19359(r9), 8760(r17), 29856(r18), 9578(r20) |
| SU | I understand empathy template | 1 | 24760(r1) |

## 5. 标签级解释

### AF

AF 的 stable latents 主要由正向评价词和积极 counselor register 组成，例如 great / good / wonderful 这类高可见度 lexical cue。18 / 27 张卡被判为 robust code candidate，整体可解释性较强。但这也说明 AF 在当前 SAE 中更像是由表面情感评价词和肯定语气组织起来，而不是完整的 MI 理论意义上的“强化来访者优势、努力或自主性”。

### GI

GI 是最清晰的标签。26 个 stable latents 中 25 个为 robust code candidate，主要组成是 medical / clinical information delivery、general information-giving 和 explanatory long utterance。这里的组织方式更接近 dialogue-function pattern，即“提供信息/解释/说明”。但部分 latent 可能也受到医学词汇、较长说明句、专业内容词的影响，因此仍需要区分“信息提供功能”和“领域词汇内容”。

### QU

QU 的 stable latents 主要是 general question detector 和 wh-word opener。它显示 LLM 内部对“提问行为”的组织较稳定，但这种稳定性首先来自 speech-act surface form，包括问句结构、疑问词、辅助动词倒装、scale question 等。QU 更像上位 question-family 表征，而不是某个 leaf MISC code。

### QUO

QUO 包含 24 / 32 个 robust candidate，但最大簇仍是 mixed open / closed question pattern。真正更接近 QUO 的是 what / how / why opener 和 open-invitation question。这说明开放式问题在 stable latents 中部分可见，但经常与 general question form 混合。换言之，LLM 可能先组织“这是一个问题”，再在其中部分区分开放式邀请。

### QUC

QUC 是 question family 中最弱的 leaf。45 个 stable latents 中 34 个是 mixed_unclear，只有 8 个 yes/no auxiliary inversion 比较像闭合式问题。该结果说明 closed-question specificity 较低，许多 QUC stable latents 实际上仍是 general question detector 或 QU/QUO overlap。QUC 的内部组织更像“问句形式的一部分”，而不是清晰独立的闭合式咨询功能。

### RE

RE 的 stable-core 数量最多，共 54 个，但只有 27 个 robust candidate，另有 21 个 mixed。主要组成包括 general reflection signal、sounds like / seems like anchor、So... discourse opening 和 ambivalence framer。该结构说明 reflection family 的表征较分散，包含若干明确的 reflective phrase anchor，但由于缺少 client context，不能判断它们是否真的复述或改写了来访者内容。

### REC

REC 与 RE 高度共享组织方式：general complex reflection、weak mixed signal、sounds like anchor、ambivalence / double-sided reflection。REC 的 robust candidate 比例为 26 / 47，略强于 RE 的纯度，但仍然混合。它更像“复杂反射常见措辞和高阶 reflective register”的集合，而不是纯粹的 context-relation latent。

### RES

RES 是 P3 stable-core 中最失败的标签。15 个 stable latents 中没有 robust candidate，10 个被判为 surface_artifact，5 个 mixed / uninterpretable。结合 RES 的低 top-activation target match rate，当前证据不支持“simple reflection 在 counselor-only utterance 中被稳定 SAE latent 清晰表示”。这更可能说明 simple reflection 强依赖上一句 client context，当前数据缺失导致其在 SAE 层面表现为低纯度或 artifact-like 模式。

### SU

SU 也很弱。31 个 latents 中只有 1 个 robust candidate，其余多为 mixed 或 artifact。主要簇是 general support / suggestion signal，但它可能混合了建议、支持、共情模板、积极语气和跨标签噪声。`I understand` empathy template 是唯一较明确的单点模式，但不足以支撑 SU 作为稳定可解释概念。

## 6. Task A / Task B 验证结果

P3 scoring 包含两个任务：

- Task A: activation prediction，即解释是否能帮助判断某句是否会高激活该 latent。
- Task B: code discrimination，即解释是否能帮助判断句子是否属于目标 MISC label。

分开看后，二者差异明显。

| 任务 | n | majority baseline | actual accuracy | balanced accuracy |
|---|---:|---:|---:|---:|
| activation_prediction_task | 3636 | 0.500 | 0.608 | 0.608 |
| code_discrimination_task | 5298 | 0.657 | 0.536 | 0.474 |

Task A 有一定正向信号，说明 AI 解释对“该 latent 激活什么样的 utterance”有部分可预测性。Task B 则失败：整体 accuracy 低于多数类基线，balanced accuracy 低于 0.5。因此不能声称这些解释已经验证了 MISC code discrimination。它们更适合作为 latent-level evidence organization，而不是 predictive sufficiency 证明。

按标签看，Task B 尤其不能支持 `RES` 和 `SU`。`QU / QUO / QUC` 也因为 sibling question overlap，code discrimination 不稳定。`RE / REC` 的部分 balanced accuracy 接近或略高于 0.5，但仍不足以作为强验证。

## 7. RQ3 回答

RQ3: 这些被解释的稀疏特征整体上揭示了 LLM 内部心理咨询行为的何种组织方式？

基于 stable-core P3 结果，一个保守、可发表的回答是：

LLM 内部与 MISC counselor behavior 相关的 sparse features 并不是按 MISC 标签一一对应组织的。它们更像是按三类线索混合组织：

1. 表面言语行为形式：问句、疑问词、辅助动词倒装、positive adjective、sounds like / seems like 等固定短语。
2. 对话功能 register：信息提供、开放式邀请、反射式回应、肯定、支持/建议等较宽的 counselor function。
3. 标签族层级与跨标签共享：QU 与 QUO/QUC 共享大量 question-family latent；RE 与 REC/RES 共享 reflection-family latent；AF 与 SU 共享 positive/supportive register。

最稳定、最清楚的内部组织出现在有强表面锚点或清晰话语功能的标签上，例如 `GI`, `QU`, `QUO`, `AF`。这些标签的 stable latents 可以形成较可解释的候选功能簇。相反，`RES` 和 `SU` 难以形成稳定、纯净的解释簇，说明当前 counselor-only 表征不足以单独恢复这些上下文依赖或边界模糊的 MISC 行为。

因此，RQ3 的核心结论应写成：

stable SAE latents reveal a hybrid organization of MI counselor behavior: the model appears to encode salient surface forms and broad dialogue-functional registers, with partial alignment to MISC label families, but not a clean one-latent-one-code conceptual map.

中文表述：

stable SAE 稀疏特征揭示的是一种“表面形式 + 对话功能 + 标签族共享”的混合组织方式。LLM 对提问、信息提供、肯定等具有明显语言形式或话语功能的行为有较清晰的内部可解码结构；但对简单反射、支持等依赖上下文或边界模糊的行为，当前 SAE latents 更多表现为混合、低纯度或 artifact-like 模式。

## 8. 可用于论文的主张

可以主张：

- stable-core latents 中存在与 MISC 行为相关的可审计候选模式。
- 这些模式对 `GI / QU / QUO / AF` 更清晰，对 `RES / SU` 更弱。
- LLM 内部组织更接近 speech-act / dialogue-function / label-family 的混合结构，而非 MISC code 的离散一一映射。
- P3 的 Task A 支持“解释对 latent 激活偏好有一定描述力”。

不应主张：

- 单个 latent 就是某个 MISC label。
- P3 已证明 LLM 理解了 MI 核心概念。
- Task B 验证成功。
- RE / REC / RES latents 已经证明捕捉了 client-context relation。

## 9. 下一步建议

1. 对 `GI / QU / QUO / AF` 先做人审，因为这些标签的 robust candidate 比例高，最可能形成论文中的正例案例。
2. 对 `RES / SU` 单独作为反例或限制讨论，不建议强行解释为成功案例。
3. 对 `QUO / QUC` 做 minimal pairs：保持问句形式改变开放/封闭功能，保持开放/封闭功能改变表面句式。
4. 对 `RE / REC` 补 client context 后重跑 evidence packet，否则无法验证真正的 reflection relation。
5. 对代表 latent 做 triggering-token 检查，确认激活是否集中在 what/how、sounds like、great/good、medical terms 等局部 token。
6. 对 `GI` 区分“医学内容词”与“提供信息功能”，避免把 topic vocabulary 当成咨询行为机制。

## 10. 关键文件

- `outputs/misc_full_sae_eval/interpretability/p3_feature_cards_stable_core/final_cards/p3_final_feature_cards.csv`
- `outputs/misc_full_sae_eval/interpretability/p3_feature_cards_stable_core/final_cards/p3_label_summary.csv`
- `outputs/misc_full_sae_eval/interpretability/p3_feature_cards_stable_core/ai_reviews/p3_input_explanations.csv`
- `outputs/misc_full_sae_eval/interpretability/p3_feature_cards_stable_core/ai_scoring/p3_scoring_metrics_by_latent.csv`
- `outputs/misc_full_sae_eval/interpretability/p3_feature_cards_stable_core/p3_agent_validation.json`
- `outputs/cross_val/stable_topk_selection/stable_topk_latent_set.csv`
