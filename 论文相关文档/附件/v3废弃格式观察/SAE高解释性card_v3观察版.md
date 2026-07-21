# SAE 高解释性 Card：废弃 v3 单一路径格式观察版

> **Status: Deprecated / Observation only**  
> **用途：仅用于观察废弃 v3 的信息压缩效果，不是当前正式实验产物。**  
> **禁止用途：不得作为正式 Scorer 输入、论文定量结果或新模型输出引用。**

## 1. 文档范围与来源

本文件把 6 张 SAE-only 高解释性冻结卡放进此前已废弃的 v3 单一路径结构，便于直接观察“语言形式与行为功能强制二选一”后保留和丢失了什么。

- 原始 Explainer：`contrastive_latent_faithfulness_v2_gpt55_low_full218`
- 原始 reduced-context Scorer：`contrastive_latent_faithfulness_v2_reduced_context_scorer_full207`
- 原始冻结卡目录：`../latent_card_reduced_context_gpt55_20260718/example_packets/cards/`
- F058 / latent 664：保留历史 v3 单卡模型的原始解释字段。
- 其余 5 张：由本文件编制时依据冻结 v2 卡进行格式转换；没有重新调用模型，也没有重新评分。
- 四组样本 ID 划分用于展示 v3 结构如何工作，不覆盖冻结 v2 卡中的原始解释。

## 2. 废弃 v3 的精确 JSON 结构

```json
{
  "feature_id": "F001",
  "activation_condition": "One narrow, testable primary condition.",
  "evidence_type": "behavioral_function | linguistic_form | affective_content | topic | surface_artifact | unclear",
  "exclusion_condition": "A superficially similar but insufficient condition.",
  "strong_supporting_sample_ids": [],
  "strong_outlier_sample_ids": [],
  "weak_boundary_sample_ids": [],
  "weak_counterexample_sample_ids": [],
  "alternative_explanation": "Only the strongest competing account."
}
```

v3 的核心约束是：`evidence_type` 只能选择一个；如果语言形式和行为功能都合理，只保留区分两组更好的一个作为 `activation_condition`，另一条路径降为 `alternative_explanation`。

## 3. 六张观察卡总览

| Feature | Latent | Stable-core labels | v3 主类型 | 被降级的竞争路径 | 当前冻结卡 held-out Spearman |
|---|---:|---|---|---|---:|
| F060 | 23183 | QUO | `linguistic_form` | 邀请客户探索观点、选项或后果 | 0.961 |
| F113 | 27857 | QUC | `linguistic_form` | 引出量化的自我评估 | 0.847 |
| F058 | 664 | QUC, QUO | `linguistic_form` | 引出经历、判断、计划或偏好 | 0.776 |
| F150 | 1713 | GI | `behavioral_function` | 第二人称临床实体和指令句式 | 0.733 |
| F165 | 29825 | SU | `linguistic_form` | 验证感受或征求许可 | 0.890 |
| F176 | 23464 | AF | `behavioral_function` | thanks/sorry 等人际公式 | 0.871 |

表中指标来自当前冻结 reduced-context 卡，仅用于识别原卡；它们不是下列格式转换后的新评分。F058 的历史 v3 单卡另有独立评分，见对应小节。

---

## 4. F060 / latent 23183 / QUO

来源卡：[F060_latent_23183.md](../latent_card_reduced_context_gpt55_20260718/example_packets/cards/F060_latent_23183.md)

```json
{
  "feature_id": "F060",
  "activation_condition": "The sentence contains a main-clause direct interrogative headed by 'what' or 'what’s', often following a discourse marker such as 'okay', 'so', or 'and'.",
  "evidence_type": "linguistic_form",
  "exclusion_condition": "Merely containing 'what' inside an embedded clause, declarative statement, reflection, or malformed fragment is insufficient when 'what' does not head the main question.",
  "strong_supporting_sample_ids": ["A001", "A002", "A003", "A004", "A005", "A006", "A007", "A008", "A009", "A010"],
  "strong_outlier_sample_ids": [],
  "weak_boundary_sample_ids": ["B002", "B003", "B009"],
  "weak_counterexample_sample_ids": ["B001", "B004", "B005", "B006", "B007", "B008", "B010"],
  "alternative_explanation": "A competing behavioral account is that the sentence invites the interlocutor to explore their own perspective, meaning, options, or anticipated consequences."
}
```

**观察：** 这张卡适合单一路径，因为直接 `what` 主问句本身已经能形成窄且可检验的边界。但 v3 会把开放式探索功能降为备选，因而不能单独回答它为何与 QUO 的心理/交际功能相关。

## 5. F113 / latent 27857 / QUC

来源卡：[F113_latent_27857.md](../latent_card_reduced_context_gpt55_20260718/example_packets/cards/F113_latent_27857.md)

```json
{
  "feature_id": "F113",
  "activation_condition": "The sentence explicitly prompts the addressee to give a numeric rating on a stated scale, typically from zero or one to ten.",
  "evidence_type": "linguistic_form",
  "exclusion_condition": "Asking about importance, confidence, pain, or feelings without explicitly presenting a numeric rating scale is insufficient; merely mentioning a number or scale endpoint is also insufficient.",
  "strong_supporting_sample_ids": ["A001", "A002", "A003", "A004", "A005", "A006", "A007", "A008", "A009", "A010"],
  "strong_outlier_sample_ids": [],
  "weak_boundary_sample_ids": ["B007"],
  "weak_counterexample_sample_ids": ["B001", "B002", "B003", "B004", "B005", "B006", "B008", "B009", "B010"],
  "alternative_explanation": "A competing behavioral account is a counseling or clinical assessment move that elicits a quantified self-assessment of readiness, confidence, severity, likelihood, pain, or another internal state."
}
```

**观察：** v3 可以准确命名数字量表模板，却把“量化自我评估”降为竞争解释。若科研问题是 latent 在标签表征中的功能，单看主解释会把 QUC 读成模板线索，而不是测量式提问功能。

## 6. F058 / latent 664 / QUC + QUO

来源卡：[F058_latent_664.md](../latent_card_reduced_context_gpt55_20260718/example_packets/cards/F058_latent_664.md)

以下是历史 v3 单卡模型的原始解释内容，不是本文件重新编写：

```json
{
  "feature_id": "F001",
  "activation_condition": "The sentence is primarily an information- or preference-seeking question addressed to the interlocutor, expressed with an explicit interrogative frame such as what/how/how many or an embedded wh-question, including a direct permission check like ‘is that okay with you.’",
  "evidence_type": "linguistic_form",
  "exclusion_condition": "It is not enough for the sentence to mention the listener, contain a question mark, or include a brief yes/no or rhetorical question embedded in advice, challenge, greeting, or reflection.",
  "strong_supporting_sample_ids": ["A001", "A002", "A003", "A004", "A005", "A006", "A007", "A008", "A009", "A010"],
  "strong_outlier_sample_ids": [],
  "weak_boundary_sample_ids": ["B003", "B005", "B006", "B007", "B008", "B010"],
  "weak_counterexample_sample_ids": ["B001", "B002", "B004", "B009"],
  "alternative_explanation": "A competing behavioral account is that Group A consists of clinician turns eliciting the other person’s own experience, plans, or preferences, but the clearest separator in these samples is the interrogative wh/permission-question form rather than independently verifiable counseling function."
}
```

历史 v3 单卡评分：

| Spearman | Pearson(log activation) | Positive-vs-zero AUROC | High-vs-weak accuracy |
|---:|---:|---:|---:|
| 0.788 | 0.809 | 0.953 | 0.840 |

**观察：** 这是废弃 v3 局限最清楚的例子。`what/how` 形式与“邀请对方提供经历、判断、计划或偏好”的功能共同构成响应边界。v3 强制选择 `linguistic_form` 后，会倾向于高估具有 wh 形式但承担挑战、筛查或寒暄功能的句子。该 latent 同时关联 QUC 与 QUO，也不应被写成某个单一标签的等价特征。

## 7. F150 / latent 1713 / GI

来源卡：[F150_latent_1713.md](../latent_card_reduced_context_gpt55_20260718/example_packets/cards/F150_latent_1713.md)

```json
{
  "feature_id": "F150",
  "activation_condition": "The utterance performs patient-specific clinical information giving, correction, risk explanation, referral explanation, or treatment instruction about the addressee’s medical facts, medication, or care.",
  "evidence_type": "behavioral_function",
  "exclusion_condition": "Second-person wording, medical vocabulary, or a general suggestion alone is insufficient when the turn mainly asks for the patient’s view, reflects their concern, or collaboratively plans an action.",
  "strong_supporting_sample_ids": ["A001", "A002", "A003", "A005", "A006", "A008", "A009", "A010"],
  "strong_outlier_sample_ids": ["A004", "A007"],
  "weak_boundary_sample_ids": ["B001", "B002", "B005", "B006"],
  "weak_counterexample_sample_ids": ["B003", "B004", "B007", "B008", "B009", "B010"],
  "alternative_explanation": "A competing linguistic account is a declarative or directive second-person clinical frame containing patient-specific phrases such as 'your medication', 'your cholesterol', 'your doctor', 'you should', or 'you can/can’t'."
}
```

**观察：** 该卡更适合以行为功能为主，因为具体临床词汇和第二人称句式只是信息给予、纠正或指导的实现方式。不过 A004、A007 与多个弱响应样本表明边界并不纯，和其较低的 high-vs-weak 指标（0.840）一致。

## 8. F165 / latent 29825 / SU

来源卡：[F165_latent_29825.md](../latent_card_reduced_context_gpt55_20260718/example_packets/cards/F165_latent_29825.md)

```json
{
  "feature_id": "F165",
  "activation_condition": "The utterance contains an explicit lexical frame of acceptability, reasonableness, understanding, or validation, especially copular 'is it okay' permission frames and predicates such as 'understandable', 'understand', 'fair', 'sense', 'important', or 'reason'.",
  "evidence_type": "linguistic_form",
  "exclusion_condition": "A generic counseling move, incidental 'okay', or a weaker permission formula such as 'do you mind if' is insufficient without an explicit evaluative or validation frame.",
  "strong_supporting_sample_ids": ["A001", "A002", "A003", "A005", "A006", "A007", "A008", "A009", "A010"],
  "strong_outlier_sample_ids": ["A004"],
  "weak_boundary_sample_ids": ["B001", "B002", "B003", "B006", "B007", "B008"],
  "weak_counterexample_sample_ids": ["B004", "B005", "B009", "B010"],
  "alternative_explanation": "A competing behavioral account is a counseling move that validates the client’s feelings or asks permission before sharing information or continuing the conversation."
}
```

**观察：** 单一路径能清楚抓住 `okay/understandable/fair/sense` 等显式形式，但会遮蔽这些形式在 SU 中可能实现的验证与许可功能。它最多支持“SU 关联中的语言代理”解释，不能据此推出一般性的同理或自主支持机制。

## 9. F176 / latent 23464 / AF

来源卡：[F176_latent_23464.md](../latent_card_reduced_context_gpt55_20260718/example_packets/cards/F176_latent_23464.md)

```json
{
  "feature_id": "F176",
  "activation_condition": "The utterance explicitly acknowledges the interlocutor in an affiliative way through gratitude, condolence, sympathy, validation, or supportive interpersonal uptake.",
  "evidence_type": "behavioral_function",
  "exclusion_condition": "Merely reflecting, summarizing, evaluating, or using a positive word is insufficient when the turn does not directly provide interpersonal support or acknowledge the other person’s disclosure.",
  "strong_supporting_sample_ids": ["A003", "A004", "A006", "A007", "A008", "A009", "A010"],
  "strong_outlier_sample_ids": ["A001", "A002", "A005"],
  "weak_boundary_sample_ids": ["B001", "B004", "B008", "B009"],
  "weak_counterexample_sample_ids": ["B002", "B003", "B005", "B006", "B007", "B010"],
  "alternative_explanation": "A competing linguistic account is the presence of conventional interpersonal formulas and stance markers such as 'thank you', 'thanks', 'sorry', 'my pleasure', 'I hear what you’re saying', or 'that’s terrible'."
}
```

**观察：** 以行为功能为主比单列词汇更能覆盖感谢、道歉、同情和支持性承接的异质表面实现。但 A001、A002、A005 是明显噪声或弱覆盖项，因此这仍是候选功能解释，不是 AF 的确定机制。

## 10. 对废弃 v3 的总体判断

v3 的优点是短、可执行、容易交给 held-out Scorer：Scorer 原设计只接收 `activation_condition` 与 `exclusion_condition`，不会看到竞争解释、样本划分或 evidence type。

它的主要问题不是字段质量，而是**单一路径约束与本研究问题不完全匹配**：

1. QU 类 latent 常由疑问形式实现提问功能，形式与功能不是互斥类别。
2. 强制二选一会把另一层证据降为“竞争解释”，使读者误以为只能有一个成立。
3. 选择语言形式时，卡片容易退化为语言代理说明，难以回答 latent 在标签表征中的心理/交际角色。
4. 选择行为功能时，又可能低估真正驱动响应的固定词汇或句式。

因此，本文件只用于对照观察。当前正式流程应继续保留形式实现与行为功能之间的关系，并让 Scorer 只依据冻结的精简预测规则对 held-out 句子评分。

## 11. 结论边界

- 这些 latent 只提供与标签相关的结构性和 held-out 可预测性证据，不是因果证明。
- 不得写成“单个 latent 等价于一个 MISC 标签”或“模型已形成与人相同的心理学概念”。
- 这里没有恢复 v3 源码、Schema 文件或正式运行入口。
- 本文件没有修改任何原始冻结卡。

