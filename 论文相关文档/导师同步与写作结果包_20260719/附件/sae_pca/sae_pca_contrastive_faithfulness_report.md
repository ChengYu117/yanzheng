# SAE–PCA 抽样对比式解释与 Held-out 忠实度报告 / Sampled Contrastive Faithfulness Report

> 本报告比较匿名表示单元解释的 held-out 预测忠实度，不提供因果机制证明，也不证明单元等同于 MISC 标签。

## Family summary

| Family | N | Mean Spearman | Median Spearman | Mean AUROC | Mean high-weak accuracy |
|---|---:|---:|---:|---:|---:|
| PCA-100 | 16 | 0.3128 | 0.2893 | 0.6292 | 0.7163 |
| PCA-50 | 16 | 0.3625 | 0.3241 | 0.6896 | 0.6863 |
| SAE | 16 | 0.5996 | 0.6739 | 0.8042 | 0.8337 |

- Valid matched comparisons: 32/32

## Explanation index / 解释索引

| Feature | Family (private audit) | Short name | Type | Confidence |
|---|---|---|---|---:|
| F006 | PCA-100 | Explicit scalar or categorical framing | linguistic_structure | 3/5 |
| F007 | PCA-100 | direct interrogative prompt | linguistic_structure | 4/5 |
| F009 | PCA-100 | patient-focused elicitation/reflection | behavioral_function | 4/5 |
| F010 | PCA-100 | addressee-focused reasons or options | behavioral_function | 4/5 |
| F011 | PCA-100 | compact simple utterance | linguistic_structure | 4/5 |
| F013 | PCA-100 | extended multi-clause conditional or explanatory turn | linguistic_structure | 4/5 |
| F014 | PCA-100 | second-person cognitive stance framing | linguistic_structure | 3/5 |
| F016 | PCA-100 | garbled or fragmentary transcript syntax | linguistic_structure | 4/5 |
| F018 | PCA-100 | transactional clinical encounter move | behavioral_function | 4/5 |
| F023 | PCA-100 | affiliative second-person check-in | behavioral_function | 4/5 |
| F030 | PCA-100 | direct client-focused prompting or offering | behavioral_function | 3/5 |
| F032 | PCA-100 | patient-centered exploration/reflection | behavioral_function | 4/5 |
| F040 | PCA-100 | brief deictic assessment/formulation | linguistic_structure | 3/5 |
| F044 | PCA-100 | concise direct addressee-focused counseling move | behavioral_function | 4/5 |
| F045 | PCA-100 | speaker stance or softening frame | linguistic_structure | 3/5 |
| F047 | PCA-100 | overt what-interrogative | linguistic_structure | 5/5 |
| F001 | PCA-50 | direct addressee-focused elicitation or exhortation | behavioral_function | 4/5 |
| F002 | PCA-50 | epistemic softening with think/might/would | linguistic_structure | 4/5 |
| F005 | PCA-50 | prominent first-person stance clause | linguistic_structure | 4/5 |
| F015 | PCA-50 | simple direct you-focused utterance | linguistic_structure | 3/5 |
| F017 | PCA-50 | how/open-ended question wording | linguistic_structure | 3/5 |
| F019 | PCA-50 | Closed polar interviewer move | linguistic_structure | 3/5 |
| F020 | PCA-50 | consequence-or-effect framing | linguistic_structure | 3/5 |
| F021 | PCA-50 | long_multi_clause_you_focused_turns | linguistic_structure | 4/5 |
| F027 | PCA-50 | direct what-question | linguistic_structure | 5/5 |
| F028 | PCA-50 | quantitative or specific-information elicitation | linguistic_structure | 4/5 |
| F033 | PCA-50 | dense direct second-person focus | linguistic_structure | 3/5 |
| F035 | PCA-50 | direct addressee-oriented elicitation or negotiation | behavioral_function | 3/5 |
| F037 | PCA-50 | rapport or epistemic framing formula | linguistic_structure | 4/5 |
| F041 | PCA-50 | utterance-initial okay transition | linguistic_structure | 4/5 |
| F043 | PCA-50 | declarative reflection or summary | behavioral_function | 3/5 |
| F048 | PCA-50 | compact standalone utterance | linguistic_structure | 3/5 |
| F003 | SAE | do/does-addressed question | linguistic_structure | 4/5 |
| F004 | SAE | addressee-directed tell/let-me-know request | linguistic_structure | 5/5 |
| F008 | SAE | extended multi-unit wh/formulation turn | linguistic_structure | 4/5 |
| F012 | SAE | positive supportive assessment | affective_content | 3/5 |
| F022 | SAE | patient-directed yes/no assessment or planning check | behavioral_function | 4/5 |
| F024 | SAE | broad exploratory elaboration prompt | behavioral_function | 4/5 |
| F025 | SAE | overt scalar or emphasis modifier | linguistic_structure | 4/5 |
| F026 | SAE | Asserted favorable adequacy appraisal | linguistic_structure | 4/5 |
| F029 | SAE | early discourse-marker so | linguistic_structure | 5/5 |
| F031 | SAE | concise reflective assessment with demonstrative framing | linguistic_structure | 4/5 |
| F034 | SAE | utterance-linking so | linguistic_structure | 4/5 |
| F036 | SAE | standalone positive affiliative evaluation | linguistic_structure | 4/5 |
| F038 | SAE | declarative second-person reflection | linguistic_structure | 4/5 |
| F039 | SAE | what-is / what's questions | linguistic_structure | 5/5 |
| F042 | SAE | direct auxiliary-initial polar questions | linguistic_structure | 4/5 |
| F046 | SAE | past-form auxiliary tokens | linguistic_structure | 4/5 |

注意：解释文本由标签盲、表示类型盲的 Explainer 生成；family仅在生成完成后用于私有审计和配对比较。
