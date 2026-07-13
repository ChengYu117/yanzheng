# QUC B级 Latent Card 人工审查

## QUC / latent 21935: health-behavior questions

- B级状态：`pending_human_review`
- 标签内 stable rank：1
- inclusion frequency：1.000
- |Cohen's d|：1.461
- 自动解释类型：`behavioral_function`
- 自动置信度：4/5
- LLM 自报支持比例：90.0%
- 入选理由：标签内稳定排名第1；健康行为相关的 do/have/can 问句代表。

**主要候选解释**

The majority of sentences are questions about health-related behaviors (sleep, weight, alcohol, drugs, diet, exercise) and attitudes, often using 'do you' or 'have you' structures.

**候选行为解释**

The sentences function as clinical or counseling interview questions, assessing patient behaviors and readiness to change.

**代表证据**

- `s001`：and you mentioned that you weren't sleeping too well do you have any weight changes during that time
- `s005`：okay can I ask you do you see yourself as overweight
- `s009`：um do you believe that changing these behaviors will be in your best interests

**其他 supporting 抽样**

- `s002`：yeah absolutely um do you know what type
- `s003`：mm-hmm and have you had any thoughts yourself about what you think might be causing it
- `s006`：can I ask if you were to if there was to be a situation where you decided I must quit what what would that situation be
- `s007`：um do you think that there's any changes you would want to make with your drinking or you content with like just drinking you know four days a week every weekend?
- `s008`：um do you think you've used to lived at home when your parents thinking about this like it's something with their degree precautions like you'd be kicked out or baby Madi you?

**Outlier 抽样**

- `s004`：there was one part where we actually asked her it was cut out we said you know do you want to go to treatment she says
- `s012`：no no no no no no we got hope heaps of time I decided that you're really meeting with the clients that I know so did you did you just tell them what the story was they're just gotta stop doing what they're doing
- `s031`：yeah I can definitely check with her and see if she can see me again okay yeah
- `s032`：Bush won last year why yes sure are you do you have a family are you married
- `s048`：that sounds quite interesting was that a fashion shoot

**替代解释**

- The sentences are all questions, but the topics vary; the pattern may be purely interrogative syntax.
- The sentences could be from a motivational interviewing session, focusing on change talk.
- Some sentences are about substance use (alcohol, marijuana, smoking) specifically, suggesting a substance abuse counseling context.

**混淆与限制**

- The dataset may be from multiple different interviews or contexts, mixing health topics.
- Transcription artifacts like 'um' and 'mm-hmm' may obscure the pattern.
- Some sentences are incomplete or contain disfluencies, making categorization difficult.
- Outliers include narrative statements and non-health questions, indicating the pattern is not universal.
- The behavioral function interpretation relies on pragmatic inference; explicit labeling is absent.

**人工审查填写**

- [ ] 主要模式得到支持
- [ ] 行为功能与表层结构已分开
- [ ] supporting/outlier 分区合理
- [ ] 候选名称需要修改
- reviewer 1：
- reviewer 2：
- 最终名称：
- 审查意见：

## QUC / latent 14014: medical_history_questions

- B级状态：`pending_human_review`
- 标签内 stable rank：3
- inclusion frequency：1.000
- |Cohen's d|：1.307
- 自动解释类型：`behavioral_function`
- 自动置信度：4/5
- LLM 自报支持比例：80.0%
- 入选理由：标签内稳定排名第3；did/have 引导的病史与行为事实询问。

**主要候选解释**

The majority of sentences are questions about the patient's medical history, symptoms, medication use, and lifestyle habits, typically starting with 'Did you' or 'Have you'.

**候选行为解释**

The sentences function as diagnostic or screening questions in a clinical interview, aimed at gathering patient information.

**代表证据**

- `s003`：Did you come here for a prescription for anti-depressants?
- `s007`：Have you had any major stressful events recently or any change in your work situation?
- `s008`：Have you ever used any prescription medications they weren't prescribed for you or in a way that wasn't prescribed?

**其他 supporting 抽样**

- `s001`：Did you like those programs?
- `s002`：Did you know that?
- `s004`：Have you ever experienced withdrawals when you've stopped drinking we experienced severe shakes use weightier trembling hearts beating really fast
- `s005`：Have you ever used them in the past
- `s006`：Have you at least been taking your diabetic medications?

**Outlier 抽样**

- `s012`：did you stand out that party is being more intoxicated or was everybody intoxicated
- `s013`：did you
- `s016`：did you talk to you though
- `s022`：okay so I think that's a pretty common misconception because they're fruit flavored people think they're okay those are really bad for you did you know that
- `s023`：so I think that's a pretty common misconception because they're fruit flavored people think they're okay those are really bad for you did you know that I'm

**替代解释**

- The sentences could be part of a substance abuse screening interview, focusing on alcohol and drug use.
- The sentences might be from a general health questionnaire covering various medical conditions.
- Some sentences appear to be from a conversation about nutrition or lifestyle (e.g., orange juice, candy jar).

**混淆与限制**

- Mixed topics: some sentences address non-medical subjects like party intoxication or nutrition.
- Incomplete or ungrammatical sentences may be transcription errors or fragments.
- The presence of statements before questions in some samples may indicate a different discourse context.
- The analysis is based solely on text without context of the conversation.
- Some sentences are ambiguous or incomplete, making classification difficult.

**人工审查填写**

- [ ] 主要模式得到支持
- [ ] 行为功能与表层结构已分开
- [ ] supporting/outlier 分区合理
- [ ] 候选名称需要修改
- reviewer 1：
- reviewer 2：
- 最终名称：
- 审查意见：

## QUC / latent 12555: do_you_questions

- B级状态：`pending_human_review`
- 标签内 stable rank：8
- inclusion frequency：1.000
- |Cohen's d|：0.846
- 自动解释类型：`linguistic_structure`
- 自动置信度：4/5
- LLM 自报支持比例：92.0%
- 入选理由：标签内稳定排名第8；do you 是可直接核查的封闭问句结构。

**主要候选解释**

The majority of sentences are questions beginning with 'do you' (or 'do' followed by a pronoun), often preceded by discourse markers like 'so', 'okay', 'and', or 'then'. These are typically yes/no questions about personal experiences, opinions, or actions, common in counseling or medical interviews.

**候选行为解释**

The sentences function as probing questions to gather information about the addressee's behaviors, beliefs, or experiences, typical of a therapeutic or medical assessment context.

**代表证据**

- `s005`：And if Oscar's that important to you then do you put Oscar higher than your baby?
- `s032`：so do you drink alcohol
- `s035`：so do you know all the bad stuff about alcohol?

**其他 supporting 抽样**

- `s001`：Oh, Okay, well, do what you can
- `s002`：but before we get there do you mind if I get to know you in your situation a little bit better
- `s003`：anyway yeah do you want me to make an appointment for you next week
- `s004`：okay well do any other questions for me today?
- `s006`：okay so do you think that your dad doesn't understand that you're tired

**Outlier 抽样**

- `s018`：I don't know that doesn't seem very serious to me really if you know it's like so yeah do it tomorrow nice do dresses what bullets
- `s045`：and just keeps you more alert when you're dehydrated do typically more sluggish and you're less more likely to make poor nutrition decisions
- `s049`：It measures the amount of sugar that's been on your hemoglobin in the last three months do determine how well you've been controlling your blood sugar.
- `s050`：you're clumsy a little bit do to get stitches on you

**替代解释**

- The pattern could be a discourse function of information gathering in a clinical interview, with 'do you' questions serving as probes.
- Alternatively, the pattern might reflect a specific annotation bias where transcribers consistently marked questions with 'do you' regardless of context.

**混淆与限制**

- Some sentences (e.g., s018, s045, s049, s050) are not questions or are garbled, possibly due to transcription errors or mixed data sources.
- The presence of discourse markers like 'so', 'okay', 'and' may be artifacts of conversational style rather than part of the stable pattern.
- The analysis is based solely on surface form; deeper pragmatic functions may vary across sentences.
- Outliers may indicate that the pattern is not perfectly stable across the entire set.

**人工审查填写**

- [ ] 主要模式得到支持
- [ ] 行为功能与表层结构已分开
- [ ] supporting/outlier 分区合理
- [ ] 候选名称需要修改
- reviewer 1：
- reviewer 2：
- 最终名称：
- 审查意见：

## QUC / latent 20463: are_you_questions

- B级状态：`pending_human_review`
- 标签内 stable rank：15
- inclusion frequency：1.000
- |Cohen's d|：0.768
- 自动解释类型：`linguistic_structure`
- 自动置信度：4/5
- LLM 自报支持比例：100.0%
- 入选理由：标签内稳定排名第15；are you 结构补充系词/进行体问句类型。

**主要候选解释**

The majority of sentences are questions beginning with 'are you' or 'are you' variants (e.g., 'are you', 'are we', 'aren't you'), often used to inquire about personal states, actions, or preferences. The pattern is a recurring interrogative structure with second-person subject.

**候选行为解释**

The sentences function as direct questions seeking information about the addressee's current state, past behavior, or preferences, typical of intake interviews or counseling contexts.

**代表证据**

- `s001`：are you picking up for yourself today or for somebody else
- `s010`：are you on drugs again
- `s041`：What are you thinking?

**其他 supporting 抽样**

- `s002`：are you surprised what that might be true
- `s003`：are you ready for that
- `s004`：are you in contact with him often
- `s005`：are you retired
- `s006`：are you were you a virgin before you would chase them

**Outlier 抽样**

- 模型没有提供 outlier；人工审查必须检查是否存在被错误纳入 supporting 的反例。

**替代解释**

- The sentences may be part of a structured interview or questionnaire about health behaviors (e.g., diet, smoking, medication adherence).
- The sentences could reflect a therapeutic or counseling discourse where the speaker probes the client's feelings and actions.

**混淆与限制**

- Some sentences include discourse markers like 'okay', 'so', 'yeah' that may indicate conversational context rather than a fixed template.
- A few sentences (e.g., s029, s036) are incomplete or fragmented, possibly due to transcription errors.
- The pattern is primarily structural; behavioral function is inferred but not directly evidenced.
- Some sentences (e.g., s024, s038) contain multiple clauses that deviate from the simple 'are you' question form.

**人工审查填写**

- [ ] 主要模式得到支持
- [ ] 行为功能与表层结构已分开
- [ ] supporting/outlier 分区合理
- [ ] 候选名称需要修改
- reviewer 1：
- reviewer 2：
- 最终名称：
- 审查意见：

## QUC / latent 2504: yes/no questions with auxiliary fronting

- B级状态：`pending_human_review`
- 标签内 stable rank：21
- inclusion frequency：0.960
- |Cohen's d|：0.674
- 自动解释类型：`linguistic_structure`
- 自动置信度：4/5
- LLM 自报支持比例：92.0%
- 入选理由：标签内稳定排名第21；辅助动词前置是最明确的 yes/no 句法候选，因多样性纳入。

**主要候选解释**

The majority of sentences are yes/no questions formed by fronting an auxiliary verb (Are, Is, Do, Does, Did, Can, Could, Were) before the subject, often in a healthcare or counseling context.

**候选行为解释**

The sentences function as diagnostic or information-gathering questions, typical of medical or therapeutic interviews.

**代表证据**

- `s001`：Are you quoting the Smiths?
- `s016`：Do you have any medical problems at all or diabetes high blood pressure
- `s038`：Can't you understand that you are putting your body at great harm by continuing to smoke

**其他 supporting 抽样**

- `s002`：Are you sure you like Star Wars?
- `s003`：Are you smoking?
- `s004`：Are you having some problems with it?
- `s005`：Are you having problems paying for it
- `s006`：Are you interested in reordering your prescription?

**Outlier 抽样**

- `s024`：Could be contributing to this, because I don't know what else might be causing it
- `s025`：there are also some suggestions that I could make about how to keep your blood Sugar's under control best
- `s031`：does look swollen
- `s033`：does it it doesn't look like you went and got any stitches

**替代解释**

- The sentences are all questions, but some are wh-questions (How, Did) and some are yes/no questions.
- The sentences are from a medical or counseling dialogue, focusing on patient habits and symptoms.
- The sentences exhibit a mix of auxiliary verbs (are, is, do, does, did, can, could, were) with no single dominant auxiliary.

**混淆与限制**

- The dataset may be extracted from a specific domain (e.g., healthcare) which biases the question types.
- Transcription artifacts (e.g., fragments, repetitions) may obscure the underlying pattern.
- Some sentences are incomplete or have unusual word order (e.g., s031, s033).
- Four sentences (s024, s025, s031, s033) do not fit the yes/no question pattern.
- The pattern is broad (any auxiliary-fronted yes/no question) and may not be distinctive.
- The behavioral function is inferred from context but not directly observable from the sentences alone.

**人工审查填写**

- [ ] 主要模式得到支持
- [ ] 行为功能与表层结构已分开
- [ ] supporting/outlier 分区合理
- [ ] 候选名称需要修改
- reviewer 1：
- reviewer 2：
- 最终名称：
- 审查意见：

