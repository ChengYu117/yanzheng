# GI B级 Latent Card 人工审查

## GI / latent 16345: medication_health_discourse

- B级状态：`pending_human_review`
- 标签内 stable rank：1
- inclusion frequency：0.990
- |Cohen's d|：0.617
- 自动解释类型：`topic`
- 自动置信度：4/5
- LLM 自报支持比例：100.0%
- 入选理由：标签内稳定排名第1；药物和健康信息主题稳定，且未强行指定单一行为功能。

**主要候选解释**

The sentences predominantly discuss medications, health conditions, and lifestyle factors (e.g., alcohol, smoking, exercise) in a conversational or explanatory tone, often involving a healthcare provider or patient interaction.

**候选行为解释**

The available sentences do not support a stable behavioral-function interpretation.

**代表证据**

- `s001`：he's right this medicine is designed to help lower your cholesterol and that means that it will help lower your risk of having a heart attack or a stroke
- `s002`：yeah I'm the pharmacist on duty citalopram okay it's an antidepressant and it comes in   and  milligram strength it's excreted in the urine and it has a half-life of about  hours for side effects there are sexual side effects heart side effects and stomachs side effects
- `s003`：so there are lots of medications out there David that actually take away a lot of the cravings there they make quitting subtle things

**其他 supporting 抽样**

- `s004`：so there are lots of medications out there David that actually take away a lot of the cravings
- `s005`：so umm there are lots of medications out there, David that actually take away alot of the cravings there they make things soo much easier
- `s006`：okay so basically what you're saying is that we're drinking alcohol you like that it helps you sleep and it calms you in the way you did talk about alcohol that it's very expensive except what you're saying
- `s007`：so carbohydrates actually do make us feel better they actually raise something called serotonin
- `s008`：and caffeine scanners constrict your blood vessels in your brain and also can relieve pain from from the migraine

**Outlier 抽样**

- 模型没有提供 outlier；人工审查必须检查是否存在被错误纳入 supporting 的反例。

**替代解释**

- The sentences may represent a corpus of healthcare consultations, with a mix of provider explanations and patient responses.
- The pattern could be a collection of advice or information-giving utterances about health behaviors and medications.

**混淆与限制**

- The sentences vary in speaker role (provider vs. patient) and discourse function (question, statement, advice), making a single behavioral function hard to isolate.
- Some sentences are incomplete or truncated, potentially obscuring the intended function.
- The analysis is based solely on text without speaker labels or context, limiting functional interpretation.
- The set includes diverse topics (medications, alcohol, smoking, exercise) under a broad health umbrella, but no single sub-topic dominates.

**人工审查填写**

- [ ] 主要模式得到支持
- [ ] 行为功能与表层结构已分开
- [ ] supporting/outlier 分区合理
- [ ] 候选名称需要修改
- reviewer 1：
- reviewer 2：
- 最终名称：
- 审查意见：

## GI / latent 26879: health_risk_advice

- B级状态：`pending_human_review`
- 标签内 stable rank：3
- inclusion frequency：0.980
- |Cohen's d|：0.539
- 自动解释类型：`topic`
- 自动置信度：3/5
- LLM 自报支持比例：86.0%
- 入选理由：标签内稳定排名第3；健康风险、益处和统计证据说明。

**主要候选解释**

The majority of sentences discuss health risks, benefits, or advice related to behaviors such as smoking, exercise, diet, and medication, often citing research or statistics.

**候选行为解释**

The sentences function as informational or advisory statements, likely from a healthcare context, aimed at educating or persuading about health behaviors.

**代表证据**

- `s003`：because smoking is a major risk factor for DVT even if you even if we treat you now if you don't quit smoking I'm afraid the blood clots will return again
- `s021`：Okay well smoking really is really bad for you, you know it causes things like lung cancer, emphysema, heart disease.
- `s049`：well smoking really is really bad for you you know it causes things like lung cancer emphysema heart disease

**其他 supporting 抽样**

- `s001`：exercises is again a number of studies has been shown to be as effective as prescription antidepressants
- `s002`：Research shows that people take on average about  tries before they quit smoking
- `s004`：research shows that people take on average about 7 tries before they quit smoking
- `s006`：yeah well walk walking briskly does lead to very substantial health benefits yeah and
- `s007`：such as hypertension diabetes heart diseases

**Outlier 抽样**

- `s005`：with cigarette smoking I see what else
- `s010`：that's what most of the research shows
- `s012`：okay the CIA's using your cell phone to track your activities
- `s014`：below risk levels
- `s015`：so with anxiety reduction your chances of stopping marijuana increase and then you could take the child that you want yeah yeah I've been notably the best best outcome is that the outcome you would most want

**替代解释**

- The sentences may be from a corpus of doctor-patient conversations, with many sentences providing medical advice or information.
- The pattern could be a mix of health-related statements and conversational fillers, with no single dominant topic.

**混淆与限制**

- Some sentences are incomplete or contain transcription errors (e.g., s001, s009, s019).
- A few sentences are unrelated to health (e.g., s012, s050).
- The outlier sentences (s005, s010, s012, s014, s015, s047, s050) do not fit the health topic pattern.
- The pattern is broad and may include multiple sub-topics within health.

**人工审查填写**

- [ ] 主要模式得到支持
- [ ] 行为功能与表层结构已分开
- [ ] supporting/outlier 分区合理
- [ ] 候选名称需要修改
- reviewer 1：
- reviewer 2：
- 最终名称：
- 审查意见：

## GI / latent 27515: medical_advice_and_history

- B级状态：`pending_human_review`
- 标签内 stable rank：5
- inclusion frequency：0.980
- |Cohen's d|：0.519
- 自动解释类型：`behavioral_function`
- 自动置信度：4/5
- LLM 自报支持比例：100.0%
- 入选理由：标签内稳定排名第5；治疗方案、药物调整和生活方式建议。

**主要候选解释**

The sentences predominantly involve a healthcare provider discussing treatment plans, medication changes, lifestyle modifications, and patient history, often using directive or advisory language.

**候选行为解释**

The primary behavioral function is giving medical advice or recommendations, as seen in sentences like s001 (transitioning off oxy) and s005 (cut out drinking).

**代表证据**

- `s001`：at this point is to think about transitioning you off of the oxy onto one of these over-the-counter medications and using some of these other techniques to help you manage the pain
- `s005`：okay well for a man your age in this high risk category it is recommended that you actually cut out drinking altogether to reduce that risk
- `s010`：you can see in order to lose weight and improve on your symptoms diet modifications and some sort of exercise need to be implemented into your daily routine

**其他 supporting 抽样**

- `s002`：because smoking is a major risk factor for DVT even if you even if we treat you now if you don't quit smoking I'm afraid the blood clots will return again
- `s003`：sowe've talked about changing your medication stopping the medication you're on changing you to this ACE inhibitor and then perhaps looking at the weight and going getting back to the gym and starting an exercise program is that about right
- `s004`：so we've talked about changing your medication stopping the medication you're on changing you to this ACE inhibitor and then perhaps looking at the weight and going getting back to the gym and starting an exercise program is that about right
- `s006`：yeah so uh I think probably it sounds to me like probably changing medications at this point as a start we can always add another medication later if that becomes necessary
- `s007`：well mr. I'd you know we've talked about it length at this visit we've talked about other visits as well

**Outlier 抽样**

- 模型没有提供 outlier；人工审查必须检查是否存在被错误纳入 supporting 的反例。

**替代解释**

- The sentences could be interpreted as a mix of information gathering (history taking) and information giving (advice), but advice-giving dominates.
- Another interpretation is that the sentences are part of a medical consultation transcript, with a focus on patient education and shared decision-making.

**混淆与限制**

- Some sentences (e.g., s008, s017) are more conversational or include patient quotes, but still fit the medical context.
- The presence of transcription artifacts (e.g., s047 'ation') does not affect the functional interpretation.
- The analysis is based solely on text without speaker labels, so the exact role of each utterance is inferred.
- Some sentences are incomplete or contain errors, but the overall pattern remains clear.

**人工审查填写**

- [ ] 主要模式得到支持
- [ ] 行为功能与表层结构已分开
- [ ] supporting/outlier 分区合理
- [ ] 候选名称需要修改
- reviewer 1：
- reviewer 2：
- 最终名称：
- 审查意见：

## GI / latent 26236: cholesterol_and_blood_pressure_discourse

- B级状态：`pending_human_review`
- 标签内 stable rank：6
- inclusion frequency：0.930
- |Cohen's d|：0.511
- 自动解释类型：`topic`
- 自动置信度：4/5
- LLM 自报支持比例：100.0%
- 入选理由：标签内稳定排名第6；胆固醇和血压管理的具体信息主题。

**主要候选解释**

The sentences predominantly discuss cholesterol and blood pressure management, often in a clinical or counseling context, using discourse markers like 'okay' and 'so' to structure explanations or instructions.

**候选行为解释**

The available sentences do not support a stable behavioral-function interpretation.

**代表证据**

- `s001`：okay okay so this medication is for your cholesterol
- `s002`：he's right this medicine is designed to help lower your cholesterol and that means that it will help lower your risk of having a heart attack or a stroke
- `s003`：all right so this medication is for your cholesterol you're gonna take it once a day you can take it with a lot of water and you want to make sure you don't miss any doses okay

**其他 supporting 抽样**

- `s004`：we see this a lot with lowering cholesterol especially triglycerides which are a type of cholesterol
- `s005`：so what could you do this week to start now once you've in this office today to have lower your cholesterol
- `s006`：okay so what could you do this week to start now once you've in this office today to have lower your cholesterol
- `s007`：so keep doing that healthy diets all that exercise to help lower your cholesterol as well
- `s008`：right exactly so that's why he wrote you this prescription so it's greater cholesterol you're just gonna take it once a day it's really easy C okay

**Outlier 抽样**

- 模型没有提供 outlier；人工审查必须检查是否存在被错误纳入 supporting 的反例。

**替代解释**

- The sentences may reflect a clinical counseling or patient education discourse, with frequent use of 'okay' and 'so' as discourse markers.
- The sentences could be part of a medical interview or health coaching session, focusing on medication adherence and lifestyle changes.

**混淆与限制**

- The dataset may be drawn from multiple conversations or sources, leading to varied topics like cholesterol, blood pressure, and general health risks.
- Some sentences appear to be incomplete or contain transcription errors, which may obscure the intended pattern.
- The primary interpretation is broad (topic-based) and does not capture subtle differences in discourse function or linguistic structure.
- There is no clear behavioral function that applies to all sentences; some are questions, some are statements, some are instructions.

**人工审查填写**

- [ ] 主要模式得到支持
- [ ] 行为功能与表层结构已分开
- [ ] supporting/outlier 分区合理
- [ ] 候选名称需要修改
- reviewer 1：
- reviewer 2：
- 最终名称：
- 审查意见：

## GI / latent 16515: healthcare_discourse_quantities

- B级状态：`pending_human_review`
- 标签内 stable rank：8
- inclusion frequency：0.980
- |Cohen's d|：0.510
- 自动解释类型：`topic`
- 自动置信度：4/5
- LLM 自报支持比例：100.0%
- 入选理由：标签内稳定排名第8；剂量、指标、数量和量表等数值信息。

**主要候选解释**

The sentences predominantly occur in healthcare or counseling contexts, where a speaker provides numerical information (e.g., calories, tries, ounces, blood pressure, A1C, insulin units) or uses scales (0-10) to assess readiness or importance. Many sentences include phrases like 'about X', 'around X', or 'on a scale from 0 to 10'.

**候选行为解释**

The sentences serve to inform, educate, or assess patients/clients about health-related metrics, risks, or behaviors, often using numbers to quantify advice or progress.

**代表证据**

- `s001`：did you know that a glass of orange juice has almost as many calories as a can of cola
- `s010`：It measures the amount of sugar that's been on your hemoglobin in the last three months do determine how well you've been controlling your blood sugar.
- `s024`：it well let's see today it was eight point seven and three months ago it was . so normal is less than six your way out of control

**其他 supporting 抽样**

- `s002`：research hows that people take on average about seven tries before they quit smoking
- `s003`：Research shows that people take on average about  tries before they quit smoking
- `s004`：research shows that people take on average about 7 tries before they quit smoking
- `s005`：so each can of soda  actually has about  teaspoons of sugar  in it
- `s006`：your sanction will involve being placed on university disciplinary probation and attending an alcohol education seminar that will cost $150 RS

**Outlier 抽样**

- 模型没有提供 outlier；人工审查必须检查是否存在被错误纳入 supporting 的反例。

**替代解释**

- The sentences might be from a corpus of transcribed medical consultations, explaining the prevalence of numbers and health topics.
- The pattern could be a mix of educational (e.g., nutrition facts) and motivational interviewing (e.g., readiness scales) discourse.

**混淆与限制**

- Some sentences (e.g., s043, s044) are less clearly health-related but still fit a counseling context.
- The presence of numbers might be an artifact of the transcription process rather than a genuine topic pattern.
- The analysis is based solely on text content without speaker or context metadata.
- Some sentences are incomplete or contain placeholders (e.g., s003, s020), which may affect interpretation.

**人工审查填写**

- [ ] 主要模式得到支持
- [ ] 行为功能与表层结构已分开
- [ ] supporting/outlier 分区合理
- [ ] 候选名称需要修改
- reviewer 1：
- reviewer 2：
- 最终名称：
- 审查意见：

