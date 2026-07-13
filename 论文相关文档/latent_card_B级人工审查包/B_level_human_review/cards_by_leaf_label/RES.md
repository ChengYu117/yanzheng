# RES B级 Latent Card 人工审查

## RES / latent 20808: second-person address

- B级状态：`pending_human_review`
- 标签内 stable rank：1
- inclusion frequency：1.000
- |Cohen's d|：0.701
- 自动解释类型：`linguistic_structure`
- 自动置信度：4/5
- LLM 自报支持比例：100.0%
- 入选理由：标签内稳定排名第1；明确的第二人称表层模式，作为避免行为过度解释的结构基线。

**主要候选解释**

All sentences begin with or contain the second-person pronoun 'you', often followed by a verb phrase, forming direct address or statements about the addressee. This pattern is consistent across the majority of sentences, though some include discourse markers like 'you know' or contractions like 'you're'.

**候选行为解释**

The sentences appear to function as conversational turns in a dialogue, likely from a counseling or medical interview context, where the speaker addresses the listener directly, making observations, asking questions, or giving advice.

**代表证据**

- `s002`：You've gotten pretty isolated.
- `s011`：You're putting yourself at risk for all these other diseases
- `s023`：You did great.

**其他 supporting 抽样**

- `s001`：You mentioned that parents of
- `s003`：You have this relationship that's been kinda off and on and but he's been very supportive and helpful in terms of giving you, meeting your basic needs and the baby's basic needs and wants to take care of you
- `s004`：You know, I find it so surprising that people can be depressed nowadays when there's new Star Wars movies coming out, like this guy.
- `s005`：You know, handsome guys lurking around clubs, selling pills to drunken -somethings who don't know any better.
- `s006`：You know there is quite alot of staining

**Outlier 抽样**

- 模型没有提供 outlier；人工审查必须检查是否存在被错误纳入 supporting 的反例。

**替代解释**

- The sentences may reflect a therapeutic or counseling discourse where the speaker reflects the client's feelings or behaviors.
- The sentences could be from a health advice or motivational interviewing context, focusing on the addressee's habits or health.

**混淆与限制**

- The dataset may be sampled from a specific genre (e.g., therapy transcripts) where second-person address is common.
- Some sentences are fragments or contain disfluencies (e.g., s026, s038) that may reflect spoken language artifacts.
- The pattern is purely structural; behavioral function is inferred but not strongly evidenced.
- Some sentences are incomplete or ambiguous (e.g., s016, s026), making interpretation less certain.

**人工审查填写**

- [ ] 主要模式得到支持
- [ ] 行为功能与表层结构已分开
- [ ] supporting/outlier 分区合理
- [ ] 候选名称需要修改
- reviewer 1：
- reviewer 2：
- 最终名称：
- 审查意见：

## RES / latent 28269: therapist_paraphrase_so_okay

- B级状态：`pending_human_review`
- 标签内 stable rank：2
- inclusion frequency：1.000
- |Cohen's d|：0.426
- 自动解释类型：`behavioral_function`
- 自动置信度：4/5
- LLM 自报支持比例：96.0%
- 入选理由：标签内稳定排名第2；so/okay 引导的改述模式，代表候选反映功能。

**主要候选解释**

The sentences predominantly consist of therapist paraphrases or reflections, often beginning with 'So' or 'Okay so', followed by a restatement of the client's prior utterance, frequently including second-person 'you' and hedges like 'kind of' or 'sort of'.

**候选行为解释**

The primary behavioral function is therapeutic reflection or summarizing, where the therapist checks understanding or encourages elaboration by restating the client's statements.

**代表证据**

- `s001`：So, you're gonna quit then?
- `s004`：So you are pretty confident about you wanting to change
- `s010`：So you want nicotine replacement but you don't want to quit smoking?

**其他 supporting 抽样**

- `s002`：So, you're realizing too, how much it really takes to focus attention on the baby and to be able to do it and it's like I, I, I it's hard to do both.
- `s003`：okay see you were standing at your locker and you heard somebody say something about Tim that you didn't want to hear
- `s005`：So you had been kinda thinking about maybe going back to work and doing, doing some of that on your, on your own.
- `s006`：So you kind of find yourself when you're in that bind trying to find ways of, of getting, getting what you need met, and, yeah.
- `s007`：So you might have to explore some other options for the child care and, and other ways of taking care of the, the baby so that you would be still okay about going back to work.

**Outlier 抽样**

- `s022`：A super handsome guy walks into a therapist's office, tricks her with his rustic charms.
- `s023`：uh no because you don't need them and now a friend where I work

**替代解释**

- The sentences could be interpreted as a collection of motivational interviewing reflections, where the therapist uses 'So' to summarize and affirm client statements.
- Alternatively, they might represent a linguistic template for summarizing client speech in clinical settings, with 'okay so' as a discourse marker.

**混淆与限制**

- The dataset may be drawn from a single therapeutic context, limiting generalizability.
- Transcription artifacts like repetitions ('I, I, I') and hesitations may obscure the pattern.
- Two sentences (s022, s023) do not fit the primary pattern, indicating some heterogeneity.
- The behavioral function is inferred from linguistic form; actual context is unknown.

**人工审查填写**

- [ ] 主要模式得到支持
- [ ] 行为功能与表层结构已分开
- [ ] supporting/outlier 分区合理
- [ ] 候选名称需要修改
- reviewer 1：
- reviewer 2：
- 最终名称：
- 审查意见：

## RES / latent 29759: therapeutic reflection and questioning

- B级状态：`pending_human_review`
- 标签内 stable rank：4
- inclusion frequency：1.000
- |Cohen's d|：0.421
- 自动解释类型：`behavioral_function`
- 自动置信度：4/5
- LLM 自报支持比例：100.0%
- 入选理由：标签内稳定排名第4，且与 REC 共享；用于审查反映与提问的功能混合。

**主要候选解释**

The sentences predominantly function as reflective statements or questions used in a therapeutic or counseling context, often summarizing, challenging, or exploring the speaker's experiences, feelings, or behaviors.

**候选行为解释**

The sentences serve to elicit self-reflection, confirm understanding, or gently challenge the interlocutor's perspective, typical of motivational interviewing or therapy.

**代表证据**

- `s007`：So that means you want to quit now right
- `s023`：So, you're gonna quit then?
- `s032`：this is really tough for you the alcohol

**其他 supporting 抽样**

- `s001`：Well, they add up quick, you know.
- `s002`：Yeah, because that leaves you in a real bind, when you're, you're there and you don't have what you need.
- `s003`：well that's not a very proactive then
- `s004`：it wouldn't happen to be about the eviction notice
- `s005`：that kinda rings Bell does it

**Outlier 抽样**

- 模型没有提供 outlier；人工审查必须检查是否存在被错误纳入 supporting 的反例。

**替代解释**

- The sentences could be from a structured interview or questionnaire about health behaviors, with the speaker summarizing or clarifying responses.
- The pattern might reflect a conversational style where the speaker frequently uses 'you' to engage the listener, but the function varies across sentences.

**混淆与限制**

- The sentences are extracted from a larger dialogue, so the context of preceding and following turns is missing, which could alter the perceived function.
- Transcription artifacts like false starts and repetitions may obscure the intended function.
- Some sentences (e.g., s006, s031) are more ambiguous and could be interpreted as simple statements rather than therapeutic reflections.
- The analysis relies on pragmatic inference without access to prosody or non-verbal cues.

**人工审查填写**

- [ ] 主要模式得到支持
- [ ] 行为功能与表层结构已分开
- [ ] supporting/outlier 分区合理
- [ ] 候选名称需要修改
- reviewer 1：
- reviewer 2：
- 最终名称：
- 审查意见：

## RES / latent 13966: you're statements about health and emotions

- B级状态：`pending_human_review`
- 标签内 stable rank：6
- inclusion frequency：0.990
- |Cohen's d|：0.412
- 自动解释类型：`linguistic_structure`
- 自动置信度：4/5
- LLM 自报支持比例：100.0%
- 入选理由：标签内稳定排名第6；you're + 状态描述结构，区别于纯 so-you 模板。

**主要候选解释**

The sentences predominantly use the second-person pronoun 'you' followed by a form of 'be' (are/'re) to make statements about the addressee's emotional state, health behaviors, or personal circumstances, often in a counseling or advice context.

**候选行为解释**

The sentences likely function as reflections, affirmations, or advice in a therapeutic or coaching dialogue, where the speaker mirrors or evaluates the listener's feelings or actions.

**代表证据**

- `s003`：You're putting yourself at risk for all these other diseases
- `s016`：you're not feeling like a normal person should be feeling
- `s036`：you're really frustrated because there's so many things going on in your life that you feel like you don't even have control

**其他 supporting 抽样**

- `s001`：you are frustrated with all this
- `s002`：you are disturbing I office
- `s004`：You're putting yourself at risk for lung cancer, for Emphysema, for oral cancers, for heart disease.
- `s005`：You're not on the...
- `s006`：You're not on the pill?

**Outlier 抽样**

- 模型没有提供 outlier；人工审查必须检查是否存在被错误纳入 supporting 的反例。

**替代解释**

- The sentences could be from a medical or counseling intake interview, focusing on patient health behaviors and emotional states.
- The pattern might reflect a specific annotation template where all sentences begin with 'you' followed by a verb, possibly from a corpus of motivational interviewing transcripts.

**混淆与限制**

- Some sentences are fragments or contain transcription errors (e.g., s002, s028), which may obscure the pattern.
- The set includes a few questions (s006, s032, s043) that deviate from the declarative pattern but still use 'you'.
- The behavioral function is inferred from content but not explicitly labeled; the exact discourse context is unknown.
- Some sentences (e.g., s010 'you're welcome') are formulaic and may not fit the health/emotion topic.

**人工审查填写**

- [ ] 主要模式得到支持
- [ ] 行为功能与表层结构已分开
- [ ] supporting/outlier 分区合理
- [ ] 候选名称需要修改
- reviewer 1：
- reviewer 2：
- 最终名称：
- 审查意见：

## RES / latent 19435: so_you_statements

- B级状态：`pending_human_review`
- 标签内 stable rank：8
- inclusion frequency：0.910
- |Cohen's d|：0.367
- 自动解释类型：`linguistic_structure`
- 自动置信度：4/5
- LLM 自报支持比例：100.0%
- 入选理由：标签内稳定排名第8，且与 REC 共享；so you 表层结构清晰，行为功能需人工收窄。

**主要候选解释**

The sentences predominantly begin with 'so you' followed by a verb phrase, forming a pattern of reflective or summarizing statements that appear to be therapist or counselor utterances in a clinical or motivational interviewing context.

**候选行为解释**

The sentences likely function as reflective listening or summarizing statements used in therapeutic settings to validate, clarify, or prompt further discussion from the client.

**代表证据**

- `s004`：soyou're concerned about having a stroke
- `s012`：so you're testing a lot
- `s024`：so you need to make some change

**其他 supporting 抽样**

- `s001`：So it sounds that you have been dealing with this for a while?
- `s002`：well so you're doing your best but but pretty pretty disappointed in the way things are for you these nights not seeing a whole lot of its kind of kind it is kind of your life isn't it I mean you guys kids what it is
- `s003`：so then you could tell me about how that's going
- `s005`：and so you're not too happy to be here this is the last thing that you want to be doing this afternoon
- `s006`：and so you're thinking that you're not as flexible as the other people

**Outlier 抽样**

- 模型没有提供 outlier；人工审查必须检查是否存在被错误纳入 supporting 的反例。

**替代解释**

- The sentences could be part of a structured interview or questionnaire where the interviewer paraphrases the respondent's answers.
- The pattern might reflect a specific therapeutic technique (e.g., motivational interviewing) where the therapist uses 'so you' to reflect the client's statements.

**混淆与限制**

- The sentences are extracted from a larger dialogue, so the pattern may be an artifact of the transcription or annotation process focusing on therapist turns.
- Some sentences (e.g., s002) are longer and more complex, but still follow the 'so you' pattern.
- The analysis is based solely on the provided sentences without context of the full conversation.
- The pattern is primarily structural; the behavioral function is inferred but not directly confirmed.

**人工审查填写**

- [ ] 主要模式得到支持
- [ ] 行为功能与表层结构已分开
- [ ] supporting/outlier 分区合理
- [ ] 候选名称需要修改
- reviewer 1：
- reviewer 2：
- 最终名称：
- 审查意见：

