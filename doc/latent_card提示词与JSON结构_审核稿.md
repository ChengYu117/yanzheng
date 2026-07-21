# Latent Card 提示词与 JSON 结构审核稿

> Status: Review Draft（审核草案）
>
> 用途：供人工审核已经使用过的 latent（潜特征）自然语言解释提示词与结构化输出字段。
>
> 当前实验唯一事实来源仍为：`docs/current/experiment_workflow.md`。本文不是新的实验规范。

## 1. 先说明：项目中存在两代 latent card 流程

| 版本 | 模型看到的句子 | 主要模型 | 状态 |
|---|---|---|---|
| 当前对比式 Explainer（解释器） | 10 条强响应句 + 10 条弱正响应句 | GPT-5.5，低推理强度（`reasoning_effort=low`） | **当前论文流程，正式使用** |
| 早期单组归纳卡 | 50 条去重高激活句 | DeepSeek，后来也由 Codex 复跑 | **历史流程，不用于当前新实验** |

下文首先完整记录当前版本；第 7 节再记录旧版，以便识别旧产物，避免误用。

## 2. 当前版本的输入边界

每次请求只给模型以下内容：

- 一个匿名 feature ID（特征 ID），例如 `F001`；
- Group A（强响应组）的 10 条强响应句；
- Group B（弱正响应组）的 10 条弱正响应句；
- 两组的相对身份：`STRONG`（强响应）与 `WEAK`（弱响应）；
- 文本是口语或转录对话这一基本背景。

模型不会看到：

- SAE（稀疏自编码器）、PCA（主成分分析）或神经网络背景；
- latent（潜特征）原始编号；
- MISC（动机性访谈技能编码系统）标签及标签定义；
- 激活值、激活排名或百分位；
- token（词元）激活；
- 数据行号、来源文件、质量组别；
- held-out（留出验证集）句子及其真实响应；
- 预期解释或人工答案。

实际运行参数为 GPT-5.5、`reasoning_effort=low`（低推理强度），每个 feature（特征）独立请求，空目录、ephemeral（一次性隔离环境）、零工具调用。

## 3. 当前 Explainer 基础指令（原样）

来源：`config/contrastive_explainer_v2_base_instructions.txt`

```text
You are a careful text-pattern induction and linguistic analysis evaluator. Use only the two sentence groups, their relative group identities, and the stated spoken/transcribed-dialogue background. Infer the narrowest condition distinguishing the groups. Do not infer how the anonymous feature was produced or assume a predefined label. Return only JSON matching the requested schema.
```

## 4. 当前单个 latent 的任务提示词（原样模板）

来源：`src/nlp_re_base/contrastive_faithfulness_v2.py` 中的 `build_explainer_prompt`。

其中 `{feature_id}`、`{strong_samples}` 和 `{weak_samples}` 在运行时替换为匿名 ID 与句子。每条句子只包含匿名样本 ID 和完整文本。

```text
Analyze the following two groups of spoken or transcribed dialogue sentences. They correspond to one anonymous text feature.

Anonymous feature ID:
{feature_id}

GROUP A — STRONG GROUP SENTENCES

{strong_samples，格式如下：
- sample_id=A001
  text: ...
...
- sample_id=A010
  text: ...}

GROUP B — WEAK GROUP SENTENCES

{weak_samples，格式如下：
- sample_id=B001
  text: ...
...
- sample_id=B010
  text: ...}

Find the narrowest stable natural-language condition that is common in Group A and absent, weaker, or less consistent in Group B.

Requirements:
1. State what must be present and what superficially similar property is insufficient.
2. Separately propose a surface/linguistic hypothesis and a behavioral/discourse hypothesis.
3. Prefer the simpler surface or linguistic explanation when behavioral evidence is insufficient.
4. Partition every Group A ID into strong_supporting_sample_ids（强组支持样本 ID）or strong_outlier_sample_ids（强组离群样本 ID）.
5. Partition every Group B ID into weak_boundary_supporting_sample_ids（弱组边界支持样本 ID）or weak_counterexample_sample_ids（弱组反例样本 ID）.
6. Cite 4–8 contrastive evidence（对比证据）items, including at least one from each group.
7. Give 1–3 alternatives（备选解释）, confounds（混杂因素）, limitations（局限性）, confidence 1–5（置信度 1-5）, and a short rationale（简短理由）.

The sentences may contain disfluencies, repetition, omissions, incomplete grammar, or transcription errors. Do not infer how the feature was produced. Do not infer a predefined label. Return only JSON matching the supplied schema. Use English prose and preserve IDs exactly.
```

### 4.1 对验证失败任务追加的重试约束（原样）

这段文字只追加到第一次输出未通过 ID 严格校验的任务末尾，不改变句子或分析目标。

```text
STRICT OUTPUT CORRECTION:
- representative_evidence_ids（代表性证据样本 ID）must contain exactly 2 or 3 bare sample IDs only.
- Each item must exactly match one supplied ID such as A001 or B003.
- Never place explanations, punctuation, multiple IDs, or descriptive text inside an ID string.
- All other ID fields must likewise contain bare supplied IDs only.
- Check these constraints before returning the JSON object.
```

## 5. 当前 Explainer JSON Schema（原样 + 中文注释）

来源：`config/contrastive_explainer_v2_schema.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "feature_id": {"type": "string"}, // feature_id（特征/Latent 匿名编号）
    "short_name": {"type": "string"}, // short_name（检索短名称）
    "contrastive_explanation": {"type": "string"}, // contrastive_explanation（强弱组对比解释）
    "necessary_or_characteristic_condition": {"type": "string"}, // necessary_or_characteristic_condition（必要或典型条件）
    "insufficient_conditions": {"type": "array", "items": {"type": "string"}}, // insufficient_conditions（看似相关但不足的条件列表）
    "surface_or_linguistic_hypothesis": {"type": "string"}, // surface_or_linguistic_hypothesis（表面形式/语言学假设）
    "behavioral_or_discourse_hypothesis": {"type": "string"}, // behavioral_or_discourse_hypothesis（行为/话语功能假设）
    "primary_explanation": {"type": "string"}, // primary_explanation（模型最终采用的主解释）
    "explanation_type": { // explanation_type（解释类型枚举）
      "type": "string",
      "enum": [
        "behavioral_function", // behavioral_function（行为/交际功能）
        "linguistic_structure", // linguistic_structure（语言/句法结构）
        "affective_content", // affective_content（情感/态度内容）
        "topic", // topic（主题/领域）
        "surface_artifact", // surface_artifact（表面伪影/转录格式）
        "unclear_or_mixed" // unclear_or_mixed（不明确或混合类型）
      ]
    },
    "strong_supporting_sample_ids": {"type": "array", "items": {"type": "string"}}, // strong_supporting_sample_ids（强组支持样本 ID 列表）
    "strong_outlier_sample_ids": {"type": "array", "items": {"type": "string"}}, // strong_outlier_sample_ids（强组离群样本 ID 列表）
    "weak_boundary_supporting_sample_ids": {"type": "array", "items": {"type": "string"}}, // weak_boundary_supporting_sample_ids（弱组边界支持样本 ID 列表）
    "weak_counterexample_sample_ids": {"type": "array", "items": {"type": "string"}}, // weak_counterexample_sample_ids（弱组反例样本 ID 列表）
    "representative_evidence_ids": { // representative_evidence_ids（代表性证据样本 ID 列表，2-3个）
      "type": "array",
      "items": {"type": "string"},
      "minItems": 2,
      "maxItems": 3
    },
    "contrastive_evidence": { // contrastive_evidence（对比证据对象列表，4-8项）
      "type": "array",
      "minItems": 4,
      "maxItems": 8,
      "items": {
        "type": "object",
        "properties": {
          "sample_id": {"type": "string"}, // sample_id（样本 ID）
          "group": {"type": "string", "enum": ["strong", "weak"]}, // group（所属分组：strong 强组 / weak 弱组）
          "evidence": {"type": "string"}, // evidence（引文/证据文本）
          "evidence_level": { // evidence_level（证据级别/层级）
            "type": "string",
            "enum": [
              "lexical", // lexical（词汇级）
              "syntactic", // syntactic（句法级）
              "discourse_marker", // discourse_marker（话语标记级）
              "semantic", // semantic（语义级）
              "pragmatic_function", // pragmatic_function（语用功能级）
              "affective", // affective（情感级）
              "topical", // topical（主题级）
              "surface_form" // surface_form（表面形式级）
            ]
          },
          "evidence_role": { // evidence_role（证据作用/角色）
            "type": "string",
            "enum": [
              "support", // support（支持所提出区分）
              "limit", // limit（揭示边界/限制）
              "contradict" // contradict（反驳/挑战所提出解释）
            ]
          }
        },
        "required": ["sample_id", "group", "evidence", "evidence_level", "evidence_role"],
        "additionalProperties": false
      }
    },
    "alternative_explanations": { // alternative_explanations（备选竞争解释列表，1-3项）
      "type": "array",
      "items": {"type": "string"},
      "minItems": 1,
      "maxItems": 3
    },
    "possible_confounds": {"type": "array", "items": {"type": "string"}}, // possible_confounds（可能混杂因素列表）
    "limitations": {"type": "array", "items": {"type": "string"}}, // limitations（局限性列表）
    "confidence": {"type": "integer", "minimum": 1, "maximum": 5}, // confidence（模型自评置信度，1-5分）
    "confidence_rationale": {"type": "string"} // confidence_rationale（置信度评分理由）
  },
  "required": [
    "feature_id",
    "short_name",
    "contrastive_explanation",
    "necessary_or_characteristic_condition",
    "insufficient_conditions",
    "surface_or_linguistic_hypothesis",
    "behavioral_or_discourse_hypothesis",
    "primary_explanation",
    "explanation_type",
    "strong_supporting_sample_ids",
    "strong_outlier_sample_ids",
    "weak_boundary_supporting_sample_ids",
    "weak_counterexample_sample_ids",
    "representative_evidence_ids",
    "contrastive_evidence",
    "alternative_explanations",
    "possible_confounds",
    "limitations",
    "confidence",
    "confidence_rationale"
  ],
  "additionalProperties": false
}
```

## 6. 当前字段含义与审核重点

| 字段 | 作用 | 人工审核重点 |
|---|---|---|
| `short_name`（检索短名称） | 可检索的短名称 | 是否过度概括或直接套用 MISC 标签 |
| `contrastive_explanation`（强弱对比解释） | 强组相对弱组的区别 | 是否真正使用了弱正响应作为 hard negatives（硬负例） |
| `necessary_or_characteristic_condition`（必要/典型条件） | 必要或典型条件 | 条件是否足够窄、可在新句子上判定 |
| `insufficient_conditions`（不充分条件） | 看似相关但不足的条件 | 是否排除了话题、关键词或问号等伪解释 |
| `surface_or_linguistic_hypothesis`（表面/语言学假设） | 词汇、句法、模板、表面形式解释 | 是否有句子证据，是否比功能解释更简洁 |
| `behavioral_or_discourse_hypothesis`（行为/话语假设） | 交际功能或话语行为解释 | 是否由当前句本身支持；尤其谨慎对待 RES/REC（反射性倾听/总结） |
| `primary_explanation`（主解释） | 模型最终采用的主解释 | 是否在多个候选中做了合理、保守的选择 |
| `explanation_type`（解释类型） | 主解释类别 | 是否与主解释内容一致 |
| 四个样本分组字段（`strong_supporting` 强组支持 / `strong_outlier` 强组离群 / `weak_boundary_supporting` 弱组边界支持 / `weak_counterexample` 弱组反例） | 对 A/B 每条句子的覆盖与反例划分 | A001–A010、B001–B010 是否各出现且只出现一次 |
| `contrastive_evidence`（对比证据） | 4–8 条逐句证据 | 是否同时包含强组与弱组；引文与结论是否匹配 |
| `alternative_explanations`（备选解释） | 竞争解释 | 是否包含真实可竞争的简化解释，而非敷衍文本 |
| `possible_confounds`（混杂因素） | 混杂因素 | 是否识别重复模板、话题、长度、转录问题等 |
| `limitations`（局限性） | 当前解释的边界 | 是否承认缺少上下文、样本混合和反例 |
| `confidence`（置信度） | 1–5 自评 | 不能直接当作解释忠实度；最终可靠性由 held-out scorer（留出集评分器）检验 |

当前程序还会执行 Schema 之外的语义结构校验：

- 所有 Group A ID 必须恰好分入支持（`strong_supporting_sample_ids`）或离群（`strong_outlier_sample_ids`）之一；
- 所有 Group B ID 必须恰好分入边界支持（`weak_boundary_supporting_sample_ids`）或反例（`weak_counterexample_sample_ids`）之一；
- 同一 ID 不能同时进入两个相反集合；
- `representative_evidence_ids`（代表性证据样本 ID）必须是输入中真实存在的 2–3 个纯 ID；
- `contrastive_evidence`（对比证据）必须同时包含 strong（强组）与 weak（弱组）证据；
- Schema 合法只证明输出结构合格，不证明解释语义正确。

## 7. 早期 50 条高激活句 latent card（历史流程）

> Status: Historical / Superseded for current experiments（历史参考 / 已弃用）
>
> 当前新实验不得使用这一采样和提示词。保留本节仅用于审核已有旧卡片的 provenance（出处与溯源记录）。

### 7.1 早期基础指令（原样）

```text
You are a text-pattern induction and linguistic analysis evaluator. Analyze a cluster of sentences, infer its most stable shared pattern, and distinguish behavioral function, linguistic structure, affective content, topic, and surface artifacts. Return only the requested JSON.
```

### 7.2 早期任务提示词（原样模板）

来源：`src/nlp_re_base/deepseek_latent_cards.py` 中的 `build_latent_card_prompt`。

```text
Analyze the following sentences as one set and produce a candidate interpretation of this anonymous text feature.

Feature ID:
{latent_idx}

Sentences:

{50 条去重高激活句，格式为：
- sample_id=S001
  text: ...}

Complete the following tasks:

1. Infer one primary candidate interpretation describing a stable pattern shared by a majority of the sentences.
2. Separately describe any candidate behavioral or discourse function.
3. If the evidence for a behavioral function is insufficient, state this explicitly instead of forcing one.
4. Cite specific sample IDs as linguistic evidence.
5. Identify sentences not adequately covered by the primary interpretation.
6. Provide 1 to 3 reasonable alternative interpretations.
7. Describe possible confounding factors.
8. Assign a confidence score from 1 to 5.

Analyze the sentence set as a whole. Do not produce a separate interpretation for every sentence. If no pattern covers a majority of the sentences, use `unclear_or_mixed`.

`explanation_type`（解释类型）must be exactly one of:

- `behavioral_function`（行为功能）: a request, question, reflection, affirmation, advice, evaluation, or another communicative or discourse function.
- `linguistic_structure`（语言结构）: recurring lexical combinations, syntax, sentence forms, discourse markers, or expression templates.
- `affective_content`（情感内容）: emotion, attitude, evaluative direction, or affective intensity.
- `topic`（主题）: a recurring event, object, activity, experience, or domain.
- `surface_artifact`（表面伪影）: incidental cues, data formats, or annotation biases unrelated to a genuine behavioral mechanism, including punctuation, text length, transcription conventions, repeated templates, fixed wording, or processing artifacts.
- `unclear_or_mixed`（不明确或混合）: multiple patterns cannot be separated reliably, or no stable pattern covers a majority of the sentences.

Each `linguistic_evidence`（语言学证据）item must contain `sample_id`, `evidence`, `evidence_level`, and `evidence_role`.

`evidence_level`（证据级别）must be exactly one of: `lexical`（词汇）, `syntactic`（句法）, `discourse_marker`（话语标记）, `semantic`（语义）, `pragmatic_function`（语用功能）, `affective`（情感）, `topical`（主题）, `surface_form`（表面形式）.

`evidence_role`（证据作用）must be exactly one of: `support`（支持）, `limit`（限制）, `contradict`（反驳）.

`representative_evidence_ids`（代表性证据样本 ID）should contain 2 to 3 sample IDs that best represent the primary interpretation.

Confidence calibration（置信度校准）:

- `1`: No consistent pattern can be identified.
- `2`: Only a weak pattern exists, or several interpretations are equally plausible.
- `3`: The primary interpretation covers a majority of the sentences, but clear exceptions or confounds remain.
- `4`: The primary pattern is clear and consistent, while alternative interpretations are weaker.
- `5`: Coverage is very high, the pattern remains stable after sentence deduplication, and alternative interpretations and confounds are weak.

Return only one JSON object matching this structure:

{运行时插入的 JSON 空模板}

Field requirements:

- Use a short, specific, searchable `short_name`（检索短名称）.
- `primary_explanation`（主要解释）must describe the most stable shared pattern and its boundary.
- If no stable behavioral function is supported, set `candidate_behavioral_explanation`（候选行为解释）to: "The available sentences do not support a stable behavioral-function interpretation."
- Every sample ID must appear exactly once in either `supporting_sample_ids`（支持样本 ID）or `outlier_sample_ids`（离群样本 ID）.
- `representative_evidence_ids`（代表性证据样本 ID）must be selected from `supporting_sample_ids`.
- Keep each prose field within two sentences.
```

### 7.3 早期 JSON Schema（原样 + 中文注释）

来源：`config/latent_card_output_schema.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "latent_idx": {"type": "integer"}, // latent_idx（Latent 原始索引编号）
    "short_name": {"type": "string"}, // short_name（检索短名称）
    "primary_explanation": {"type": "string"}, // primary_explanation（主要共享模式解释）
    "candidate_behavioral_explanation": {"type": "string"}, // candidate_behavioral_explanation（候选行为功能解释）
    "explanation_type": { // explanation_type（解释类型枚举）
      "type": "string",
      "enum": [
        "behavioral_function", // behavioral_function（行为功能）
        "linguistic_structure", // linguistic_structure（语言结构）
        "affective_content", // affective_content（情感内容）
        "topic", // topic（主题）
        "surface_artifact", // surface_artifact（表面伪影）
        "unclear_or_mixed" // unclear_or_mixed（不明确或混合）
      ]
    },
    "supporting_sample_ids": {"type": "array", "items": {"type": "string"}}, // supporting_sample_ids（支持样本 ID 列表）
    "outlier_sample_ids": {"type": "array", "items": {"type": "string"}}, // outlier_sample_ids（离群样本 ID 列表）
    "representative_evidence_ids": { // representative_evidence_ids（代表性证据样本 ID，2-3个）
      "type": "array",
      "items": {"type": "string"},
      "minItems": 2,
      "maxItems": 3
    },
    "linguistic_evidence": { // linguistic_evidence（语言学证据对象列表）
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "properties": {
          "sample_id": {"type": "string"}, // sample_id（样本 ID）
          "evidence": {"type": "string"}, // evidence（引文/证据文本）
          "evidence_level": { // evidence_level（证据级别）
            "type": "string",
            "enum": [
              "lexical", // lexical（词汇）
              "syntactic", // syntactic（句法）
              "discourse_marker", // discourse_marker（话语标记）
              "semantic", // semantic（语义）
              "pragmatic_function", // pragmatic_function（语用功能）
              "affective", // affective（情感）
              "topical", // topical（主题）
              "surface_form" // surface_form（表面形式）
            ]
          },
          "evidence_role": {"type": "string", "enum": ["support", "limit", "contradict"]} // evidence_role（证据作用：support 支持 / limit 限制 / contradict 反驳）
        },
        "required": ["sample_id", "evidence", "evidence_level", "evidence_role"],
        "additionalProperties": false
      }
    },
    "alternative_explanations": { // alternative_explanations（备选解释列表）
      "type": "array",
      "items": {"type": "string"},
      "minItems": 1,
      "maxItems": 3
    },
    "possible_confounds": {"type": "array", "items": {"type": "string"}}, // possible_confounds（可能混杂因素列表）
    "limitations": {"type": "array", "items": {"type": "string"}}, // limitations（局限性列表）
    "confidence": {"type": "integer", "minimum": 1, "maximum": 5}, // confidence（置信度评分，1-5）
    "confidence_rationale": {"type": "string"} // confidence_rationale（置信度评分理由）
  },
  "required": [
    "latent_idx",
    "short_name",
    "primary_explanation",
    "candidate_behavioral_explanation",
    "explanation_type",
    "supporting_sample_ids",
    "outlier_sample_ids",
    "representative_evidence_ids",
    "linguistic_evidence",
    "alternative_explanations",
    "possible_confounds",
    "limitations",
    "confidence",
    "confidence_rationale"
  ],
  "additionalProperties": false
}
```

## 8. 两代版本的关键差异

| 审核维度 | 早期 50 句单组卡 | 当前 10 强 + 10 弱卡 |
|---|---|---|
| 是否有 hard negatives（硬负例） | 无 | 有，弱正响应作为边界对照 |
| 模型能否排除宽泛话题 | 较弱 | 更强，必须说明强弱区别 |
| feature 标识（特征编号） | 直接暴露 `latent_idx` | 仅暴露匿名 `Fxxx` |
| 证据结构 | 支持/离群 | 强支持/强离群 + 弱边界/弱反例 |
| 解释目标 | 找多数句共享模式 | 找区分两组的最窄条件 |
| 后续未见数据验证 | 原卡片本身没有 | 独立 held-out Scorer（留出集评分器） |
| 当前论文用途 | 历史参考 | 正式解释生成 |

## 9. 实际文件与产物位置

当前提示词与 Schema（架构声明）：

- `config/contrastive_explainer_v2_base_instructions.txt`
- `config/contrastive_explainer_v2_schema.json`
- `src/nlp_re_base/contrastive_faithfulness_v2.py`

当前 218 stable-core 全量任务及输出：

- `outputs/rerun_new_dataset_20260716/min5_words/interpretability/contrastive_latent_faithfulness_v2_gpt55_low_full218/explainer/tasks.jsonl`
- `outputs/rerun_new_dataset_20260716/min5_words/interpretability/contrastive_latent_faithfulness_v2_gpt55_low_full218/explainer/validated_explanations.jsonl`
- `outputs/rerun_new_dataset_20260716/min5_words/interpretability/contrastive_latent_faithfulness_v2_gpt55_low_full218/explainer/validation_manifest.json`

历史 50 句版本来源：

- `src/nlp_re_base/deepseek_latent_cards.py`
- `config/codex_latent_evaluator_base_instructions.txt`
- `config/latent_card_output_schema.json`

## 10. 与 held-out Scorer 的关系

本审核稿主体描述的是“解释生成”，即 Explainer（解释器）。解释生成后，系统会冻结其中与定义有关的字段，再交给另一个独立请求对 20 条未见句子预测 0–100 匹配分。Scorer（评分器）不会修改 latent card（特征卡片），也不会看到真实激活值；真实响应只在离线计算 Spearman（斯皮尔曼等级相关系数）、AUROC（受试者工作特征曲线下面积）与 high–weak（高-弱激活排序准确率）时使用。

Scorer 的正式提示词与结构位于：

- `config/contrastive_scorer_v2_base_instructions.txt`
- `config/contrastive_scorer_v2_schema.json`

因此应分别审核两个问题：

1. Explainer 是否给出了窄、可判定、有反例意识的候选解释；
2. 该冻结解释能否让独立 Scorer 预测未见句子的真实响应排序。
