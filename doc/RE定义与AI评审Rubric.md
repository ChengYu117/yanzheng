# RE定义与AI评审Rubric

## 方法定位
这套 rubric 用于本项目的 `AI 专家代理评审` 管线。它的定位是：

- 自动解释证据
- 专家代理评审证据
- 用于补充 `MaxAct / latent cards / G1-G5-G20` 的人工阅读成本

它不是：

- 单独的因果性证明
- 对真实心理咨询专家评分的替代
- 对来访者上下文充分可见时的正式 MI 编码

如果后续要写因果结论，仍然需要和 `ablation`、`steering`、对照实验一起报告。

## 定义来源
本项目把 RE 定义固定为文献导向版本，主要参考：

- Motivational Interviewing 中对 reflective listening 的经典定义
- MITI / MISC 一类人工评审框架中的反映性倾听标准
- 近年 SAE 自动解释工作里“用模型代理人审”的思路

本项目当前数据只有 `unit_text`，没有稳定的来访者前句上下文。因此 judge 只能做：

- 句内 RE 线索评审

不能做：

- 严格的“这句话是否准确回应前一句来访者表达”的上下文级判断

## 本项目中的 RE 定义

### 1. Simple Reflection
`Simple Reflection` 指的是：

- 对来访者已经明确表达的内容或感受做复述
- 使用近义改写或轻度镜像
- 不明显增加新意义

直觉上，它更像：

> “你最近真的很累。”

这类表达通常是在重复或镜像来访者已经说出的内容。

### 2. Complex Reflection
`Complex Reflection` 指的是：

- 在忠实于来访者原意的前提下
- 合理补出情绪、意义、冲突、重点或更深一层的理解
- 帮助对方继续探索，而不是替对方下结论或给建议

直觉上，它更像：

> “你一方面很想改变，另一方面又担心改变会带来新的压力。”

这类表达通常不只是复述，而是在“理解的基础上做深化”。

### 3. 非 RE
本项目把以下内容统一视为 `非 RE`：

- 纯提问
- 建议
- 解释或教育
- 命令式表达
- 劝说
- 泛泛安慰
- 明显模板化、缺少真实映照的套话

## 固定评审维度
AI judge 必须按以下 4 个维度打分，每项范围 `1-5`：

### 1. mirrors_client_meaning
看这句话是否真的在反映来访者已经表达的内容、感受或含义。

### 2. adds_valid_meaning_or_empathy
看这句话是否在忠实前提下增加了合理的同理、情绪命名或意义深化。

### 3. non_directive_non_question
看这句话是否避免了提问、命令、建议和明显的指导性语言。

### 4. natural_therapeutic_language
看这句话是否像真实、自然的咨询式表达，而不是生硬模板。

## 固定风险标签
judge 还必须从以下风险标签中选择：

- `question_like`
- `advice_like`
- `information_giving`
- `lexical_template_only`
- `context_needed`

这些标签的作用是提醒我们：

- 某个 latent 也许抓到了 RE 风格
- 但也可能只是抓到了固定句式、提问习惯或模板语言

## 单句输出结构
每条单句评审都固定输出以下字段：

- `has_clear_re_feature`: `yes | partial | no`
- `re_type`: `simple | complex | mixed | non_re | unclear`
- `clarity_score`: `1-5`
- `dimension_scores`
- `evidence_spans`
- `reason_zh`
- `risk_flags`

## 为什么要把 rubric 写死
这一步不是形式问题，而是研究可信度问题。原因有三点：

- 如果每次 prompt 都临时解释 RE，结果会漂移
- 如果不固定排除项，judge 容易把“温和建议”误当成“同理反映”
- 如果不固定维度，后续就无法比较不同 latent 和不同组的评审结果

所以本项目的做法是：

- RE 定义固定
- 维度固定
- 风险标签固定
- 输出 schema 固定

这样做的目的，是让 AI 评审结果更像“可复查的结构化证据”，而不是一次性的主观评论。
