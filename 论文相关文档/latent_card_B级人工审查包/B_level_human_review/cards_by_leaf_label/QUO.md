# QUO B级 Latent Card 人工审查

## QUO / latent 9959: open-ended questions about personal experiences

- B级状态：`pending_human_review`
- 标签内 stable rank：1
- inclusion frequency：1.000
- |Cohen's d|：1.762
- 自动解释类型：`behavioral_function`
- 自动置信度：4/5
- LLM 自报支持比例：94.0%
- 入选理由：标签内稳定排名第1；围绕个人经历和原因的开放式提问。

**主要候选解释**

The majority of sentences are open-ended questions that ask about personal experiences, habits, preferences, or reasons, often starting with 'what', 'how', 'why', or 'so'. They are typical of a conversational or clinical interview setting.

**候选行为解释**

The sentences function as probes to elicit information from a respondent, likely in a therapeutic or counseling context, aimed at exploring behaviors, feelings, and decision-making.

**代表证据**

- `s001`：What kind of dog do you have?
- `s010`：okay well let's just start a little bit by talking about your cravings the cravings themselves when you wake up in the morning you start your day when does a typical cravings start for you
- `s024`：so why are you here today

**其他 supporting 抽样**

- `s002`：So how many children do you have?
- `s003`：so what brings you here today
- `s004`：all right well you mentioned a couple times that you've made a few decisions to try and limit the a dangerous aspect of the drinking why did you make those decisions
- `s005`：okay all right well you mentioned a couple times that you've made a few decisions to try and limit the a dangerous aspect of the drinking why did you make those decisions
- `s006`：okay what kind of dog do you have?

**Outlier 抽样**

- `s008`：About what, when the timing is right for any decision that you make, one way or the other.
- `s038`：And what do I do with it
- `s043`：on a scale that yeah yeah okay so where are you at the moment if if they your success rate at the moment is?

**替代解释**

- The sentences are all questions, but some are closed-ended (e.g., 'how many') while most are open-ended.
- The sentences are from a medical or therapeutic intake interview, focusing on health behaviors like smoking, drinking, and exercise.
- The sentences exhibit a pattern of using discourse markers like 'so', 'okay', 'well' to initiate questions.

**混淆与限制**

- The set includes a few non-question sentences (s008, s038, s043) that may be transcription errors or different contexts.
- Some questions are very similar (e.g., s004 and s005) which might inflate the pattern's apparent consistency.
- The presence of first-person question (s038) and fragmented sentence (s043) suggests possible annotation noise.
- The analysis is based solely on text without prosodic or contextual cues.
- The set may be drawn from multiple speakers or sessions, introducing variability.

**人工审查填写**

- [ ] 主要模式得到支持
- [ ] 行为功能与表层结构已分开
- [ ] supporting/outlier 分区合理
- [ ] 候选名称需要修改
- reviewer 1：
- reviewer 2：
- 最终名称：
- 审查意见：

## QUO / latent 23183: what-initial questions

- B级状态：`pending_human_review`
- 标签内 stable rank：4
- inclusion frequency：1.000
- |Cohen's d|：1.339
- 自动解释类型：`linguistic_structure`
- 自动置信度：4/5
- LLM 自报支持比例：92.0%
- 入选理由：标签内稳定排名第4；what-initial 结构明确，作为表层形式代表。

**主要候选解释**

The majority of sentences are questions beginning with 'what', often with repetition or hesitation, and are used to elicit information, opinions, or reflections from an interlocutor.

**候选行为解释**

The sentences function as open-ended probes or requests for information, typical of counseling, interviewing, or coaching contexts.

**代表证据**

- `s001`：What, what are your thoughts right now about some of the options out there for you to do, if, if anything?
- `s005`：What are, what are you thinking about?
- `s006`：What do you think?

**其他 supporting 抽样**

- `s002`：What, what thoughts do you have about how you might go about balancing that and, and, and thinking about, thinking about that?
- `s003`：What do you mean yo-, yo-, ha-, if you, as an, you know, to be informed about
- `s004`：What what do you think
- `s007`：What have you thought about?
- `s008`：What would you say?

**Outlier 抽样**

- `s026`：but but what I know is that most people doing a lot more of that so
- `s027`：ok so what's nothing to me is that your doctor saying you have an issue a health issue at your cholesterol and nearly an allison issue of this present time cycle
- `s036`：but what I'm the point I'm trying to make with you is that it was difficult and you rose to the challenge and you got through it
- `s046`：I understand that you're on the go all the time but what's really important is no more that you protect your health now imagine running your business if you're held in the dark deteriorate

**替代解释**

- The sentences may reflect a therapeutic or motivational interviewing style, with open-ended questions to explore client perspectives.
- The pattern could be a transcription artifact where many 'what' questions are from a single speaker in a dialogue, but the context is not provided.

**混淆与限制**

- The dataset may be drawn from a specific domain (e.g., counseling, health coaching) that naturally uses many 'what' questions.
- Transcription errors or disfluencies (repetitions, false starts) may obscure the underlying pattern.
- The analysis is based solely on surface form; without context, behavioral function is inferred but not confirmed.
- A few sentences are statements or garbled, indicating the pattern is not universal.

**人工审查填写**

- [ ] 主要模式得到支持
- [ ] 行为功能与表层结构已分开
- [ ] supporting/outlier 分区合理
- [ ] 候选名称需要修改
- reviewer 1：
- reviewer 2：
- 最终名称：
- 审查意见：

## QUO / latent 26485: how_questions

- B级状态：`pending_human_review`
- 标签内 stable rank：5
- inclusion frequency：1.000
- |Cohen's d|：1.312
- 自动解释类型：`behavioral_function`
- 自动置信度：4/5
- LLM 自报支持比例：96.0%
- 入选理由：标签内稳定排名第5；how 引导的感受、方式和观点询问。

**主要候选解释**

The majority of sentences are questions beginning with 'how', 'what', or 'can', often used to elicit information, opinions, or feelings from the addressee.

**候选行为解释**

The sentences function as requests for information or reflection, typical of counseling or coaching contexts.

**代表证据**

- `s005`：how do you feel about that
- `s013`：how do you think you might use it?
- `s032`：How do those thoughts make you feel?

**其他 supporting 抽样**

- `s001`：How do I, how do I respond to this
- `s002`：how can I help you
- `s003`：how can I help you t well
- `s004`：how do you see that working in like an evening kind of thing around the evening?
- `s006`：how do you view this problem

**Outlier 抽样**

- `s023`：I would like to know how confident you are in achieving this goal so again on a scale from  to   King unconfident and  the most confident
- `s029`：what do I need

**替代解释**

- The sentences are all questions, but some are self-directed (e.g., s001, s029) while most are other-directed.
- The sentences are from a therapeutic or motivational interviewing setting, focusing on behavior change (e.g., alcohol, exercise).

**混淆与限制**

- The dataset may be from a specific domain (e.g., addiction counseling) which influences the question types.
- Some sentences are incomplete or contain disfluencies (e.g., s003, s028) which may be transcription artifacts.
- Two sentences (s023, s029) do not fit the primary pattern of second-person questions.
- The behavioral function is inferred from the question form but not explicitly confirmed.

**人工审查填写**

- [ ] 主要模式得到支持
- [ ] 行为功能与表层结构已分开
- [ ] supporting/outlier 分区合理
- [ ] 候选名称需要修改
- reviewer 1：
- reviewer 2：
- 最终名称：
- 审查意见：

## QUO / latent 11660: wh_questions_with_would

- B级状态：`pending_human_review`
- 标签内 stable rank：7
- inclusion frequency：1.000
- |Cohen's d|：1.264
- 自动解释类型：`linguistic_structure`
- 自动置信度：4/5
- LLM 自报支持比例：84.0%
- 入选理由：标签内稳定排名第7，且与 QUC 共享；wh + would 假设/偏好问句。

**主要候选解释**

The majority of sentences are wh-questions (what, how, where, when, why) that use the modal 'would' to ask about preferences, hypothetical actions, or opinions. This pattern is consistent across diverse topics and contexts.

**候选行为解释**

The sentences function as probing questions, likely from a counselor or interviewer, aimed at eliciting the respondent's thoughts, feelings, or plans regarding behavior change or personal goals.

**代表证据**

- `s002`：What would you say?
- `s017`：what would you like to talk about
- `s035`：okay so what would you like me to do for you today

**其他 supporting 抽样**

- `s001`：would what would I be right in saying that
- `s003`：and what would you start thinking of doing
- `s005`：you've thought about a letter but who would you send it to
- `s006`：so in what ways would you like to do that
- `s007`：and what if what if anything would you choose to do would be to cut down with a bit to quit just in case but

**Outlier 抽样**

- `s004`：so you have your pencil where did you cut
- `s015`：at this point what do we have to lose
- `s025`：what do I need
- `s030`：What have you thought about?
- `s033`：you know this is just another little thing about when you first start getting used to counting carbohydrates just grant information for you so um I know you've been on the Lantis and it's an injectable how have you been doing the injection

**替代解释**

- The sentences are all questions, but some use 'would' while others use other modals or tenses, suggesting a broader pattern of interrogative forms.
- The sentences may reflect a therapeutic or coaching discourse where the speaker asks open-ended questions to explore the client's perspective.

**混淆与限制**

- The dataset may be drawn from a specific domain (e.g., counseling, health coaching) that naturally uses many 'would' questions.
- Transcription artifacts or incomplete sentences may obscure the pattern.
- Outliers include statements, non-'would' questions, and fragments that do not fit the primary pattern.
- The pattern is structural and does not capture the full functional diversity of the sentences.

**人工审查填写**

- [ ] 主要模式得到支持
- [ ] 行为功能与表层结构已分开
- [ ] supporting/outlier 分区合理
- [ ] 候选名称需要修改
- reviewer 1：
- reviewer 2：
- 最终名称：
- 审查意见：

## QUO / latent 18310: open-ended questions about change and barriers

- B级状态：`pending_human_review`
- 标签内 stable rank：8
- inclusion frequency：1.000
- |Cohen's d|：1.251
- 自动解释类型：`behavioral_function`
- 自动置信度：4/5
- LLM 自报支持比例：100.0%
- 入选理由：标签内稳定排名第8；围绕改变、障碍和计划的开放式功能候选。

**主要候选解释**

The majority of sentences are open-ended questions, often starting with 'what' or 'which', that ask about experiences, barriers, plans, or hypothetical scenarios related to behavior change, particularly in health or substance use contexts.

**候选行为解释**

The sentences function as therapeutic or motivational interviewing probes, designed to elicit client reflection, explore ambivalence, and identify barriers to change.

**代表证据**

- `s001`：in what ways might your life be better if you succeed in making the changes you mentioned
- `s008`：what was it that you know made you start back smoking again
- `s025`：what do you see as some of the barriers that might get in your way

**其他 supporting 抽样**

- `s002`：okay and I have a quick quiz for you what does fifteen grams of carbohydrates equal
- `s003`：3 what were you thinking right afterwards
- `s004`：what was it like
- `s005`：what was different about this place
- `s006`：what was that

**Outlier 抽样**

- 模型没有提供 outlier；人工审查必须检查是否存在被错误纳入 supporting 的反例。

**替代解释**

- The sentences are all from a single therapeutic or coaching session transcript, with the therapist asking questions and occasionally making reflective statements.
- The sentences are predominantly interrogative in form, with a subset being declarative statements that serve as reflections or summaries.

**混淆与限制**

- The dataset may be drawn from a specific domain (e.g., addiction counseling) that biases the topic and function.
- Transcription artifacts (e.g., repeated words, incomplete sentences) may obscure the intended pattern.
- A few sentences are declarative statements rather than questions, but they still serve a therapeutic function.
- The pattern is robust across the majority of sentences.

**人工审查填写**

- [ ] 主要模式得到支持
- [ ] 行为功能与表层结构已分开
- [ ] supporting/outlier 分区合理
- [ ] 候选名称需要修改
- reviewer 1：
- reviewer 2：
- 最终名称：
- 审查意见：

