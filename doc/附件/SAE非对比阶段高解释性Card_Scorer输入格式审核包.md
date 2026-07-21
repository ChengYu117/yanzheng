# SAE 非对比阶段高解释性 Card：去锚定 Scorer 实际输入审核包

> Status: Current Human Review Attachment（当前人工审核附件）
>
> 来源：full-218 SAE 去锚定 scorer 正式复跑；不包含 PCA–SAE 对比实验。
> 唯一方法规范仍为 `docs/current/experiment_workflow.md`。

## 修复状态

本审核包已改用 `contrastive_latent_faithfulness_v2_gpt55_low_full218_scorer_deanchored_20260718` 的实际任务和结果。Scorer 只看到以下五个冻结字段：

- `short_name`；
- `surface_or_linguistic_hypothesis`；
- `behavioral_or_discourse_hypothesis`；
- `primary_explanation`；
- `explanation_type`。

必要/典型条件、不足条件、对比解释、样本划分、对比证据、备选解释、混杂因素、局限和 Explainer 自评置信度均未发送给 Scorer。207/207 个正式单元通过最终验证，工具调用为 0。

## 审核方式

以下每个代码块均为 `scorer/tasks.jsonl` 中实际发送的 user prompt 原文，包含冻结解释与 20 条未见句子；不显示目标标签、真实分层或真实响应。建议先盲审代码块，最后再查看文末 reviewer-only 指标。

## Scorer 基础指令

```text
You are an independent evaluator of an anonymous text-feature explanation. Use only the frozen explanation and unseen spoken/transcribed-dialogue sentences. Predict how strongly each complete sentence matches the feature. Do not revise the explanation or infer hidden groups. Return only JSON matching the requested schema.
```

## Card 1: F058

### 实际发送给 Scorer 的用户提示词

```text
Use the frozen explanation below to predict how strongly each held-out spoken/transcribed-dialogue sentence matches the anonymous text feature.

Feature ID: F058

Frozen explanation:
{
  "short_name": "client-centered eliciting question",
  "surface_or_linguistic_hypothesis": "The feature is associated with direct or indirect open-ended question forms, especially wh- or how-phrases such as \"what,\" \"how many,\" \"how often,\" and \"what sort of,\" often addressed with \"you.\"",
  "behavioral_or_discourse_hypothesis": "The feature marks a collaborative elicitation move in an interview or counseling dialogue: the speaker invites the client/patient to supply information, assess seriousness, describe prior attempts, or choose a next step, rather than giving advice or making a statement.",
  "primary_explanation": "A stable distinction is that Group A turns are primarily questions soliciting the addressee's own account, assessment, or plan. Group B turns are less consistently eliciting: many are advice, rapport, reflections, affirmations, or closed/challenging questions.",
  "explanation_type": "behavioral_function"
}

Held-out sentences:
- sample_id=H001
  text: how many hours a day on average would you say that you work out?
- sample_id=H002
  text: okay bad idea on in that same scale with one being not confident at all to 10 being super super confident where would you say your are in terms of your confidence to actually lose the way
- sample_id=H003
  text: so on a scale of one to ten which number best reflects how important it is to you to drink the low-risk lifts
- sample_id=H004
  text: and then one final question on a scale of  to   being not ready at all and  being extremely ready how ready would you say you are to commit to taking some kind of brief break and that let's say the next couple of weeks
- sample_id=H005
  text: okay and so what made you say five or six rather than two or three
- sample_id=H006
  text: well so Linda what I'm hearing you say is that it's actually very important for you to exercise but on the other hand you are not that confidence to actually start a success
- sample_id=H007
  text: and and you've spoken to this already a little bit i believe but I'll ask you again why and eight and not on a one or two
- sample_id=H008
  text: so can you tell me about your day from when you wake up to
- sample_id=H009
  text: okay so you still had the pencil you got under the bleachers you say you were feeling angry still are you still on the scale of one to ten where were you when you got under the bleachers
- sample_id=H010
  text: what kind of activities do you like doing?
- sample_id=H011
  text: and whether you're getting me to do you know how long you have
- sample_id=H012
  text: do you think if you put up with a little bit of embarrassment
- sample_id=H013
  text: and if lo and your drinking would have that effect that something be willing to do right now
- sample_id=H014
  text: I'm hailing the dietitian how are you
- sample_id=H015
  text: Emily why why did you get your lip pierced why?
- sample_id=H016
  text: it so it sounds like you know you really you really have a reason that you really want to be here you care about your wife can you care about your job and you really like what you do is what I'm hearing right now you're good at it look
- sample_id=H017
  text: thank you very much for your time live
- sample_id=H018
  text: Well have you been taking your Rubiff reguarly?
- sample_id=H019
  text: well so you're doing your best but but pretty pretty disappointed in the way things are for you these nights not seeing a whole lot of its kind of kind it is kind of your life isn't it I mean you guys kids what it is
- sample_id=H020
  text: but you did perfect opposite action right there which is you are feeling really uncomfortable and I can tell that but you still did it

For each complete sentence output an integer predicted_feature_score from 0 to 100: 0–10 absent/contradicted; 11–30 weak/incidental; 31–60 partial/ambiguous; 61–80 clear; 81–100 strong and specific. Do not revise the explanation, infer hidden groups, or rely on broad domain membership. matching_evidence_span must be one exact contiguous substring copied from the sentence or empty. Preserve the sentence's original case, punctuation, and whitespace exactly; never normalize spacing, paraphrase, join disjoint spans, or insert ellipses. Return all 20 IDs exactly once and in input order. Return only JSON matching the supplied schema.
```

### 人工审核栏

- 匿名短名：`client-centered eliciting question`
- 主解释类型：`behavioral_function`
- [ ] 解释在 held-out 句子上具有清楚、可重复的匹配边界
- [ ] 解释主要反映心理/话语功能，而非关键词、句法或模板代理
- [ ] 仅凭冻结解释即可稳定区分应高分与应低分句子
- [ ] 没有明显反例足以推翻主解释
- 人工结论：保留 / 收窄后保留 / 仅作语言代理 / 拒绝
- 批注：

## Card 2: F176

### 实际发送给 Scorer 的用户提示词

```text
Use the frozen explanation below to predict how strongly each held-out spoken/transcribed-dialogue sentence matches the anonymous text feature.

Feature ID: F176

Frozen explanation:
{
  "short_name": "explicit affiliative uptake",
  "surface_or_linguistic_hypothesis": "The feature is triggered by conventional interpersonal formulas and stance markers, including 'thank you/thanks,' 'sorry,' 'my pleasure,' 'I hear what you're saying,' 'thanks for sharing,' 'oh wow okay,' and evaluative sympathy like 'that's terrible.'",
  "behavioral_or_discourse_hypothesis": "The feature marks rapport-building turns in dialogue, where the speaker responds to the interlocutor's disclosure with social support, gratitude, condolence, or validation rather than only asking, summarizing, or advising.",
  "primary_explanation": "The strongest commonality in Group A is explicit affiliative acknowledgment of the other speaker. Group B includes some counseling-style reflections and occasional empathy, but these are less consistently framed as direct interpersonal support or politeness formulas.",
  "explanation_type": "behavioral_function"
}

Held-out sentences:
- sample_id=H001
  text: okay that sounds really interesting
- sample_id=H002
  text: yeah that's great that's a really good goal
- sample_id=H003
  text: oh gosh I'm sorry it's getting that point pretty pretty pretty helpless
- sample_id=H004
  text: and I do get where you're coming from
- sample_id=H005
  text: well thanks for sharing a bit of information within this patient this kind of gives us a better idea of what you throughout the day
- sample_id=H006
  text: you know I appreciate you talking to me about this
- sample_id=H007
  text: I'd be happy to hear your ideas maybe even share some suggestions with you about a possible plan that big daddy
- sample_id=H008
  text: yeah and I can totally understand how you get to there if you're thinking I'm feeling fat I'm going to get fat you know it's an easy place to get to okay
- sample_id=H009
  text: Wow okay so it sounds like you're feeling pretty confident after our discussion
- sample_id=H010
  text: okay so that's pretty frightening very yeah
- sample_id=H011
  text: good so how is it they going since our last appointment?
- sample_id=H012
  text: so that was the face pretty good experience
- sample_id=H013
  text: Well according to your chart your blood pressure has gone up since July and you said that you've been drinking more recently
- sample_id=H014
  text: yeah so can we make a follow up appointment where we can review that journal after to see what's working for you what's not working for you yeah sure
- sample_id=H015
  text: good so to open up your social atmosphere baby
- sample_id=H016
  text: with a full glass of water and if you feel upset stomach you should take it with meals
- sample_id=H017
  text: yeah actually I did for a couple of days
- sample_id=H018
  text: mm-hmm so like once or twice a month you're having alcohol
- sample_id=H019
  text: okay so why don't you give a range of times so why don't we say four to seven times a week in the mornings for we can say until our next session so we can go you can go back and recap how the journaling is going for you
- sample_id=H020
  text: Did you like those programs?

For each complete sentence output an integer predicted_feature_score from 0 to 100: 0–10 absent/contradicted; 11–30 weak/incidental; 31–60 partial/ambiguous; 61–80 clear; 81–100 strong and specific. Do not revise the explanation, infer hidden groups, or rely on broad domain membership. matching_evidence_span must be one exact contiguous substring copied from the sentence or empty. Preserve the sentence's original case, punctuation, and whitespace exactly; never normalize spacing, paraphrase, join disjoint spans, or insert ellipses. Return all 20 IDs exactly once and in input order. Return only JSON matching the supplied schema.
```

### 人工审核栏

- 匿名短名：`explicit affiliative uptake`
- 主解释类型：`behavioral_function`
- [ ] 解释在 held-out 句子上具有清楚、可重复的匹配边界
- [ ] 解释主要反映心理/话语功能，而非关键词、句法或模板代理
- [ ] 仅凭冻结解释即可稳定区分应高分与应低分句子
- [ ] 没有明显反例足以推翻主解释
- 人工结论：保留 / 收窄后保留 / 仅作语言代理 / 拒绝
- 批注：

## Card 3: F047

### 实际发送给 Scorer 的用户提示词

```text
Use the frozen explanation below to predict how strongly each held-out spoken/transcribed-dialogue sentence matches the anonymous text feature.

Feature ID: F047

Frozen explanation:
{
  "short_name": "Second-person negative self/thought reflection",
  "surface_or_linguistic_hypothesis": "The feature tracks clauses with second-person pronouns combined with negative polarity or negative evaluative predicates, especially forms like 'you're not', 'you don't', 'you feel like', 'you think', or 'telling you that'.",
  "behavioral_or_discourse_hypothesis": "The feature marks therapist-style reflective statements that mirror a client's distressing cognition, self-criticism, catastrophic fear, or perceived social/emotional deficit back to them.",
  "primary_explanation": "Group A is best characterized by second-person reflections of negative internal content: the addressee is described as not good enough, not having support, fearing going off a bridge, being hard on themselves, or believing they must steal. Group B has some near misses, but its sentences often concern neutral questioning, confidence, behavioral advice, or the speaker's own difficulty rather than a reflected negative self-belief of the addressee.",
  "explanation_type": "behavioral_function"
}

Held-out sentences:
- sample_id=H001
  text: and so you're thinking that you're not as flexible as the other people
- sample_id=H002
  text: so your emotions your sense of maybe sadness and in grief and disappointment around this detract from your sense of readiness and in some way sort of giving up on a dream
- sample_id=H003
  text: what I'm hearing is it sounds like you might feel like maybe who's managing money to a degree the last yeah
- sample_id=H004
  text: okay so you're I hear that you're struggling to find time to fit exercise into your schedule and you're starting to feel pretty frustrated and discouraged with the situation
- sample_id=H005
  text: tell me a little bit more about not feeling like yourself
- sample_id=H006
  text: scale of one to ten one being you're not really sure at all ending returning to do with how how come are you in  different as you don't go to you mate
- sample_id=H007
  text: there's really no such thing as not drinking too much
- sample_id=H008
  text: it so it sounds like you know you really you really have a reason that you really want to be here you care about your wife can you care about your job and you really like what you do is what I'm hearing right now you're good at it look
- sample_id=H009
  text: you know you know that you want to make the changes and you feel ready to make you feel ready to make these changes
- sample_id=H010
  text: and given that experience your tells like your partner maybe not as inclined to think about calling the police again
- sample_id=H011
  text: Like I said, it's really putting yourself and your child in danger.
- sample_id=H012
  text: how confident you are and I may be again in an imaginary scale of  to  where  is the most confident one is the least confident
- sample_id=H013
  text: mm-hmm okay well on a scale of 1 to 10 I'm 10 mean like for sure and 1 being not really sure about it at all what would you say where you are on the scale with 20 an 80-acre know that you
- sample_id=H014
  text: so you've been looking for other alternatives so you still feel like you're doing one but they're doing I'm looking at the carbs a little or seeing where it would be with your sugars
- sample_id=H015
  text: well on this scale of one to ten you know like we use that pain scale with people as nurses how confident are you that you'll continue on with what you've been doing as far as the portions and though bike riding
- sample_id=H016
  text: well cast our walls I'm really glad that you were able to find some time to actually get get together
- sample_id=H017
  text: Okay, the number I have is.
- sample_id=H018
  text: you think it would be easier to do it after work
- sample_id=H019
  text: generally how you're feeling physically maybe the food reflect on the food choices that you had made the day before maybe set some you said you like setting little objectives maybe little mini goals for yourself for the day things like that
- sample_id=H020
  text: and whether you're getting me to do you know how long you have

For each complete sentence output an integer predicted_feature_score from 0 to 100: 0–10 absent/contradicted; 11–30 weak/incidental; 31–60 partial/ambiguous; 61–80 clear; 81–100 strong and specific. Do not revise the explanation, infer hidden groups, or rely on broad domain membership. matching_evidence_span must be one exact contiguous substring copied from the sentence or empty. Preserve the sentence's original case, punctuation, and whitespace exactly; never normalize spacing, paraphrase, join disjoint spans, or insert ellipses. Return all 20 IDs exactly once and in input order. Return only JSON matching the supplied schema.
```

### 人工审核栏

- 匿名短名：`Second-person negative self/thought reflection`
- 主解释类型：`behavioral_function`
- [ ] 解释在 held-out 句子上具有清楚、可重复的匹配边界
- [ ] 解释主要反映心理/话语功能，而非关键词、句法或模板代理
- [ ] 仅凭冻结解释即可稳定区分应高分与应低分句子
- [ ] 没有明显反例足以推翻主解释
- 人工结论：保留 / 收窄后保留 / 仅作语言代理 / 拒绝
- 批注：

## Card 4: F018

### 实际发送给 Scorer 的用户提示词

```text
Use the frozen explanation below to predict how strongly each held-out spoken/transcribed-dialogue sentence matches the anonymous text feature.

Feature ID: F018

Frozen explanation:
{
  "short_name": "elaborated reflective summary of the listener's perspective",
  "surface_or_linguistic_hypothesis": "The feature is associated with clauses of the form \"it sounds like / I'm hearing you say / you seem\" followed by second-person descriptions of feelings, motivations, concerns, or reasons, often with extended coordination or causal elaboration.",
  "behavioral_or_discourse_hypothesis": "The feature marks empathic reflective listening: the speaker is summarizing or validating what the other person has expressed, especially ambivalence, distress, motivation, or concern, rather than advising, questioning, informing, or evaluating.",
  "primary_explanation": "Group A is best distinguished by elaborated reflective formulations of the addressee's perspective. The strongest cases explicitly infer feelings or motivations: being afraid, concerned, motivated, frustrated, or seeing reasons on both sides. Group B includes a few partial reflections, but many are informational, directive, logistical, speculative, or judgmental, and the reflective items are usually brief or immediately subordinated to a question or recommendation.",
  "explanation_type": "behavioral_function"
}

Held-out sentences:
- sample_id=H001
  text: yeah I imagine that there's some you know a period of readjustment and healing and sadness or as in it sounds like also just trying to get things done
- sample_id=H002
  text: yeah you don't look terribly overweight but according to the the charts there it's in the overweight category
- sample_id=H003
  text: well you clearly have a lot of weight to lose cuz that feeling your main goal to lose weight
- sample_id=H004
  text: well you say cheerful funny so it's a lot of fun yes
- sample_id=H005
  text: and when you say it was scary I hear you saying on some level was scary for you and your your safety but it also sounds like it was scary because all these other variables or elements were put in motion that were disruptive
- sample_id=H006
  text: so given what you're attempting to do when would it be good for you to come back and talk about this
- sample_id=H007
  text: well I wanted to talk to you about something we talked about earlier session which was a time it sound like a few months ago
- sample_id=H008
  text: it's an eighth Wow so it's not a two or three it's an eighth
- sample_id=H009
  text: Well, you know, a lot of time just like, what's his name Georg Burns, I think he smoke or something all his life right, and, but even though she smoked you didn't want to smoke you just only want to chew?
- sample_id=H010
  text: so it sounds like you're really confused about why your symptoms even began
- sample_id=H011
  text: But you only use the leaf tobacco; you're not smoking or anything.
- sample_id=H012
  text: so you'd be willing to stop altogether that's what was indicated by the physician
- sample_id=H013
  text: and there has been some consequences in terms of your mom curtailing your independence based on your cutting
- sample_id=H014
  text: tell me more about what you've thought about and when you think about making a change in terms of leaving what what might that look like for you
- sample_id=H015
  text: could you talk to your boss about it?
- sample_id=H016
  text: so how many times you guys have sex that night on a scale of one to five five being the lowest how many times did you guys have sex that night
- sample_id=H017
  text: That's true and that's true.
- sample_id=H018
  text: yes here how's your injury?
- sample_id=H019
  text: Well, I don't know last time I called you and then the phone is not ringing and I'm not sure.
- sample_id=H020
  text: I'm wondering if that's what yoga is about civility

For each complete sentence output an integer predicted_feature_score from 0 to 100: 0–10 absent/contradicted; 11–30 weak/incidental; 31–60 partial/ambiguous; 61–80 clear; 81–100 strong and specific. Do not revise the explanation, infer hidden groups, or rely on broad domain membership. matching_evidence_span must be one exact contiguous substring copied from the sentence or empty. Preserve the sentence's original case, punctuation, and whitespace exactly; never normalize spacing, paraphrase, join disjoint spans, or insert ellipses. Return all 20 IDs exactly once and in input order. Return only JSON matching the supplied schema.
```

### 人工审核栏

- 匿名短名：`elaborated reflective summary of the listener's perspective`
- 主解释类型：`behavioral_function`
- [ ] 解释在 held-out 句子上具有清楚、可重复的匹配边界
- [ ] 解释主要反映心理/话语功能，而非关键词、句法或模板代理
- [ ] 仅凭冻结解释即可稳定区分应高分与应低分句子
- [ ] 没有明显反例足以推翻主解释
- 人工结论：保留 / 收窄后保留 / 仅作语言代理 / 拒绝
- 批注：

## Card 5: F150

### 实际发送给 Scorer 的用户提示词

```text
Use the frozen explanation below to predict how strongly each held-out spoken/transcribed-dialogue sentence matches the anonymous text feature.

Feature ID: F150

Frozen explanation:
{
  "short_name": "patient-specific clinical explanation or directive",
  "surface_or_linguistic_hypothesis": "The feature may track declarative second-person clinical statements with possessive patient-specific noun phrases such as 'your medication,' 'your cholesterol,' 'your doctor,' 'your GP,' or clauses like 'you should,' 'you can/can't,' and 'you've been.'",
  "behavioral_or_discourse_hypothesis": "The feature may mark provider information-giving, counseling, correcting misconceptions, or giving treatment instructions, as opposed to motivational interviewing prompts, reflections, and collaborative planning questions.",
  "primary_explanation": "The strongest distinction is discourse-functional: Group A is dominated by clinician assertions or instructions about the patient's medical facts, medications, risks, or referral, while Group B is mostly patient-centered elicitation/reflection or lifestyle planning. The surface correlate is dense second-person reference attached to concrete clinical entities or directives.",
  "explanation_type": "behavioral_function"
}

Held-out sentences:
- sample_id=H001
  text: And your doctor knows about it that you chew tobacco?
- sample_id=H002
  text: yeah so he he actually see me because he went to him for increased stress and he's concerned that your alcohol consumption may be a part of that increase and prior to prescribing you anything you want to make sure that you had someone to talk to about that
- sample_id=H003
  text: well thanks for sharing a bit of information within this patient this kind of gives us a better idea of what you throughout the day
- sample_id=H004
  text: will recheck your blood pressure and see how you're doing with all this.
- sample_id=H005
  text: all right so this medication because of your your headache is very bad I would recommend you to take two terabytes
- sample_id=H006
  text: we don't really recommend that the ideas you would use the nicotine replacement to actually quit so then you wouldn't be smoking at work but you also wouldn't be smoking the rest of the time either
- sample_id=H007
  text: okay that's perfect because cholesterol stenton works best if it's taken at night though you can take it anytime of the day
- sample_id=H008
  text: okay well we actually considered  cigarettes a day to be a significant amount and in fact I would then recommend to you that you would start on the highest dose of nicotine replacement
- sample_id=H009
  text: okay so before I recommend that medications to be and before I will I want you to ask you about your past medical history
- sample_id=H010
  text: and I was wondering how you are managing this blood pressure or are you aware of your hypertension
- sample_id=H011
  text: And you have not been checked with ultrasound or anything like that?
- sample_id=H012
  text: how did your doctor tell you to take this medicine?
- sample_id=H013
  text: does she get enough aerobic exercise?
- sample_id=H014
  text: okay so it's kind of linked to your drinking the fact that that you can if you can take care of that drinking you can take care of the birth control issue and this importance comes down as your reason why you're not down here
- sample_id=H015
  text: so what are you saying correct me if I'm wrong is you think that logging your sugar intake is going to help you see on top of the way you feel after you eat candy how much can you you're actually eating
- sample_id=H016
  text: I can see where that would be bothersome to you
- sample_id=H017
  text: it's maybe a big one for you right that you're waking up your don't have any scars you got to listen to your music you got to you're smelling good right you're feeling good maybe
- sample_id=H018
  text: so you would change the time you go to drinks or the lines which one you want to go
- sample_id=H019
  text: that sounds great I mean you know they have some flavor tea center delicious set up their format balls
- sample_id=H020
  text: it looks like it's gonna protect you

For each complete sentence output an integer predicted_feature_score from 0 to 100: 0–10 absent/contradicted; 11–30 weak/incidental; 31–60 partial/ambiguous; 61–80 clear; 81–100 strong and specific. Do not revise the explanation, infer hidden groups, or rely on broad domain membership. matching_evidence_span must be one exact contiguous substring copied from the sentence or empty. Preserve the sentence's original case, punctuation, and whitespace exactly; never normalize spacing, paraphrase, join disjoint spans, or insert ellipses. Return all 20 IDs exactly once and in input order. Return only JSON matching the supplied schema.
```

### 人工审核栏

- 匿名短名：`patient-specific clinical explanation or directive`
- 主解释类型：`behavioral_function`
- [ ] 解释在 held-out 句子上具有清楚、可重复的匹配边界
- [ ] 解释主要反映心理/话语功能，而非关键词、句法或模板代理
- [ ] 仅凭冻结解释即可稳定区分应高分与应低分句子
- [ ] 没有明显反例足以推翻主解释
- 人工结论：保留 / 收窄后保留 / 仅作语言代理 / 拒绝
- 批注：

## Card 6: F072

### 实际发送给 Scorer 的用户提示词

```text
Use the frozen explanation below to predict how strongly each held-out spoken/transcribed-dialogue sentence matches the anonymous text feature.

Feature ID: F072

Frozen explanation:
{
  "short_name": "open-ended exploration of personal patterns/preferences",
  "surface_or_linguistic_hypothesis": "Group A tends to use open-ended wh-forms with personal-state or pattern nouns and predicates, such as 'things', 'plan', 'like', 'typical amount', 'wish things would be like', and 'times that you've used'.",
  "behavioral_or_discourse_hypothesis": "The feature may track counseling-style elicitation of client self-knowledge: exploring motivations, preferences, coping techniques, personal history, or desired change, rather than collecting medical status or setting the appointment agenda.",
  "primary_explanation": "The strongest commonality is not merely interrogative form, but open-ended elicitation of personally meaningful patterns, preferences, plans, or coping resources. This is frequent and central in Group A, while Group B has only partial overlaps and many narrower medical or agenda-management questions.",
  "explanation_type": "behavioral_function"
}

Held-out sentences:
- sample_id=H001
  text: what are your plans for after high school
- sample_id=H002
  text: what did you use to cut
- sample_id=H003
  text: so what do you see is the key goal that you're trying to focus on right now or maybe a group of gold what are what are your key things that you are wanting to have happen?
- sample_id=H004
  text: okay and tell me a little bit more about your plan for exercise at this point
- sample_id=H005
  text: can I ask you about your exercise habits some things like any physical activity you engage in or anything like that
- sample_id=H006
  text: and also what kind of beverages and things that you drink what are some of the things that you drink throughout
- sample_id=H007
  text: okay and what are some alternatives
- sample_id=H008
  text: and tell me about the exercise that you're doing
- sample_id=H009
  text: any not-so-good things you mentioned one thing that's
- sample_id=H010
  text: okay so we're working on  point scale of one to ten one being not important can be very important one being not confident one being very confident
- sample_id=H011
  text: do you take any medications or do you have any medical conditions
- sample_id=H012
  text: well if there was anything about your health you're still thinking about now in terms of you know what's your next goal or what your next step or something you're thinking about
- sample_id=H013
  text: so you've said that a few times now that things got so out of control and crazy tell me tell me a little bit more about how what that was
- sample_id=H014
  text: can you think of any ways to maybe cut back on that to where you wouldn't be getting as much alcohol in your system on a regular basis
- sample_id=H015
  text: so so what is this yeah that's not really my thing I mean you know I'm gonna do this voluntarily rare that you my boss is making me yeah not too happy about it um okay so you need advice
- sample_id=H016
  text: no I wouldn't imagine it would
- sample_id=H017
  text: so 10 pounds sounds like something that might be negotiable
- sample_id=H018
  text: how are your kids doing?
- sample_id=H019
  text: but if you'd like to use anything else that we've used previously just let me know and I'll get them to you for you
- sample_id=H020
  text: can we explore this issue a little further

For each complete sentence output an integer predicted_feature_score from 0 to 100: 0–10 absent/contradicted; 11–30 weak/incidental; 31–60 partial/ambiguous; 61–80 clear; 81–100 strong and specific. Do not revise the explanation, infer hidden groups, or rely on broad domain membership. matching_evidence_span must be one exact contiguous substring copied from the sentence or empty. Preserve the sentence's original case, punctuation, and whitespace exactly; never normalize spacing, paraphrase, join disjoint spans, or insert ellipses. Return all 20 IDs exactly once and in input order. Return only JSON matching the supplied schema.
```

### 人工审核栏

- 匿名短名：`open-ended exploration of personal patterns/preferences`
- 主解释类型：`behavioral_function`
- [ ] 解释在 held-out 句子上具有清楚、可重复的匹配边界
- [ ] 解释主要反映心理/话语功能，而非关键词、句法或模板代理
- [ ] 仅凭冻结解释即可稳定区分应高分与应低分句子
- [ ] 没有明显反例足以推翻主解释
- 人工结论：保留 / 收窄后保留 / 仅作语言代理 / 拒绝
- 批注：

## Card 7: F040

### 实际发送给 Scorer 的用户提示词

```text
Use the frozen explanation below to predict how strongly each held-out spoken/transcribed-dialogue sentence matches the anonymous text feature.

Feature ID: F040

Frozen explanation:
{
  "short_name": "negative-feeling reflection",
  "surface_or_linguistic_hypothesis": "The feature tracks sentences with explicit affective-state vocabulary, especially feel/feeling/felt plus negative adjectives or complements such as depressed, overwhelmed, discouraged, frustrated, uncomfortable, fatigued, tired, or not so great.",
  "behavioral_or_discourse_hypothesis": "The feature tracks empathic reflections in counseling-style dialogue where the speaker acknowledges the client's distress or emotional burden, rather than giving advice, information, agenda-setting, or positive reinforcement.",
  "primary_explanation": "The strongest stable distinction is negative subjective affect being named or reflected. Group A repeatedly presents emotional distress or difficult feelings, while Group B contains only occasional boundary cases and mostly non-affective counseling moves.",
  "explanation_type": "affective_content"
}

Held-out sentences:
- sample_id=H001
  text: so you feel torn stuck between decisions about
- sample_id=H002
  text: a depressions and I know you feel like depressed now
- sample_id=H003
  text: so why don't we start by you telling me a bit about your day and the struggles you're having to include exercise where that those barriers are coming up and
- sample_id=H004
  text: are you still feeling depressed
- sample_id=H005
  text: Wow okay so it sounds like you're feeling pretty confident after our discussion
- sample_id=H006
  text: what you might do in a serious moment where you're scared?
- sample_id=H007
  text: so I'm hearing that you're not that confident
- sample_id=H008
  text: and I know exactly what you feel
- sample_id=H009
  text: you just not through that time
- sample_id=H010
  text: so so you were scared sounds like you were scared it was terrifying
- sample_id=H011
  text: okay so you noticing that it's kind of a little too long and so you want to find an alternative means of coping or trying to relax
- sample_id=H012
  text: I want I want to simply ask you how you feel about it about what what losing weight
- sample_id=H013
  text: all right so this medication because of your your headache is very bad I would recommend you to take two terabytes
- sample_id=H014
  text: yeah so he he actually see me because he went to him for increased stress and he's concerned that your alcohol consumption may be a part of that increase and prior to prescribing you anything you want to make sure that you had someone to talk to about that
- sample_id=H015
  text: and you told her heart she's she's hard work isn't she that's what I heard her years ago girly
- sample_id=H016
  text: okay well it's not a zero so you have some confidence in yours
- sample_id=H017
  text: and then you just got down you I don't understand why
- sample_id=H018
  text: Well, your wife make you, how did she make you?
- sample_id=H019
  text: what do we have here?
- sample_id=H020
  text: good how you doing today

For each complete sentence output an integer predicted_feature_score from 0 to 100: 0–10 absent/contradicted; 11–30 weak/incidental; 31–60 partial/ambiguous; 61–80 clear; 81–100 strong and specific. Do not revise the explanation, infer hidden groups, or rely on broad domain membership. matching_evidence_span must be one exact contiguous substring copied from the sentence or empty. Preserve the sentence's original case, punctuation, and whitespace exactly; never normalize spacing, paraphrase, join disjoint spans, or insert ellipses. Return all 20 IDs exactly once and in input order. Return only JSON matching the supplied schema.
```

### 人工审核栏

- 匿名短名：`negative-feeling reflection`
- 主解释类型：`affective_content`
- [ ] 解释在 held-out 句子上具有清楚、可重复的匹配边界
- [ ] 解释主要反映心理/话语功能，而非关键词、句法或模板代理
- [ ] 仅凭冻结解释即可稳定区分应高分与应低分句子
- [ ] 没有明显反例足以推翻主解释
- 人工结论：保留 / 收窄后保留 / 仅作语言代理 / 拒绝
- 批注：

## Card 8: F169

### 实际发送给 Scorer 的用户提示词

```text
Use the frozen explanation below to predict how strongly each held-out spoken/transcribed-dialogue sentence matches the anonymous text feature.

Feature ID: F169

Frozen explanation:
{
  "short_name": "explicit validation of the addressee's perspective",
  "surface_or_linguistic_hypothesis": "The feature is associated with explicit stance/validation constructions, especially first-person cognition or appreciation verbs and evaluative predicates directed at the addressee: 'I understand,' 'I appreciate,' 'I know,' 'I hear what you're saying,' 'you're right,' 'it's understandable,' and 'important to you.'",
  "behavioral_or_discourse_hypothesis": "The feature marks empathic validation in counseling-style dialogue: the speaker aligns with the listener's concerns or values before moving on, emphasizing that the listener's feelings, priorities, or difficulties make sense.",
  "primary_explanation": "The strongest stable distinction is explicit empathic validation of the addressee's perspective rather than simple positivity, reflection, or advice.",
  "explanation_type": "behavioral_function"
}

Held-out sentences:
- sample_id=H001
  text: That's true and that's true.
- sample_id=H002
  text: I understand you're you're having some difficulties health symptoms can you tell me about those
- sample_id=H003
  text: I know it's, it's going to be hard to do that
- sample_id=H004
  text: yeah yeah look it sounds like you did a terrific job
- sample_id=H005
  text: so the one hand it lowers it on the other hand it's increasing kind of a seesaw teeter-totter effect
- sample_id=H006
  text: so pretty confident that things will continue on it you can maintain what you've been doing and that's great you're not a zero
- sample_id=H007
  text: so so what is this yeah that's not really my thing I mean you know I'm gonna do this voluntarily rare that you my boss is making me yeah not too happy about it um okay so you need advice
- sample_id=H008
  text: yeah I understand exactly what you're saying
- sample_id=H009
  text: okay and I'm curious you mentioned you know your friends are important to you and your family's important to you sort of fitting in as important to you
- sample_id=H010
  text: I understand that well I have an important notice to let you know about
- sample_id=H011
  text: I mean anything that's drinking when you're underage is drinking too much
- sample_id=H012
  text: I know these things can be hard to talk about
- sample_id=H013
  text: that sounds great I mean you know they have some flavor tea center delicious set up their format balls
- sample_id=H014
  text: yeah all of these are like really understand aborning it's just of course you want something I mean you need to be close to use you don't have to commute so far and
- sample_id=H015
  text: then what do you make of that give them that on one hand it lowers and the other had it increases and now it's increasing even more
- sample_id=H016
  text: so you have a support group that gives you the ability to have fun and still drink
- sample_id=H017
  text: what else what else is going on with
- sample_id=H018
  text: Okay, what the doctor say?
- sample_id=H019
  text: how confident are you that you could do something about your drinking
- sample_id=H020
  text: oh gosh I'm sorry it's getting that point pretty pretty pretty helpless

For each complete sentence output an integer predicted_feature_score from 0 to 100: 0–10 absent/contradicted; 11–30 weak/incidental; 31–60 partial/ambiguous; 61–80 clear; 81–100 strong and specific. Do not revise the explanation, infer hidden groups, or rely on broad domain membership. matching_evidence_span must be one exact contiguous substring copied from the sentence or empty. Preserve the sentence's original case, punctuation, and whitespace exactly; never normalize spacing, paraphrase, join disjoint spans, or insert ellipses. Return all 20 IDs exactly once and in input order. Return only JSON matching the supplied schema.
```

### 人工审核栏

- 匿名短名：`explicit validation of the addressee's perspective`
- 主解释类型：`behavioral_function`
- [ ] 解释在 held-out 句子上具有清楚、可重复的匹配边界
- [ ] 解释主要反映心理/话语功能，而非关键词、句法或模板代理
- [ ] 仅凭冻结解释即可稳定区分应高分与应低分句子
- [ ] 没有明显反例足以推翻主解释
- 人工结论：保留 / 收窄后保留 / 仅作语言代理 / 拒绝
- 批注：

## Card 9: F006

### 实际发送给 Scorer 的用户提示词

```text
Use the frozen explanation below to predict how strongly each held-out spoken/transcribed-dialogue sentence matches the anonymous text feature.

Feature ID: F006

Frozen explanation:
{
  "short_name": "Utterance-initial so plus you-address",
  "surface_or_linguistic_hypothesis": "The feature tracks a surface opening formula: sentence-initial “so” directly followed by a second-person pronoun or contraction.",
  "behavioral_or_discourse_hypothesis": "The feature may reflect therapist/counselor summary or reformulation turns that present an inference about the client’s state, behavior, or concern, but the available contrast is more consistently explained by the opening wording than by discourse function alone.",
  "primary_explanation": "A narrow linguistic pattern distinguishes the groups: Group A sentences are framed as “so you…” statements, while Group B lacks that exact initial second-person formulation despite sometimes using related counseling language.",
  "explanation_type": "linguistic_structure"
}

Held-out sentences:
- sample_id=H001
  text: and so you're thinking that you're not as flexible as the other people
- sample_id=H002
  text: so you feel torn stuck between decisions about
- sample_id=H003
  text: so you have some friends that are already involved in an exercise program yeah
- sample_id=H004
  text: so you sleep a lot but even though you're sleeping you don't wake up rested
- sample_id=H005
  text: so you'd be willing to stop altogether that's what was indicated by the physician
- sample_id=H006
  text: okay so it's pretty hot up there again
- sample_id=H007
  text: so from your family perspective the drinking is not something that fits very well
- sample_id=H008
  text: so for my understanding drinking it helps you socialize his friends it relaxes you it's you know here we have just you know it breaks that ice for you
- sample_id=H009
  text: so on a scale of zero to 100 how important do you think it is for you to make this change
- sample_id=H010
  text: okay so I'm gonna write down then that you've decided to quit
- sample_id=H011
  text: last time we talked um you and Kelly had just gotten engaged and you were rather excited about plans that you had for your life
- sample_id=H012
  text: mmm-hmm and so because of it was cold and Wendy you decided to kind of cut the walk short
- sample_id=H013
  text: okay to continue so one other thing that I wanted to to mention to you um was you said that you'd walk your dog in nature
- sample_id=H014
  text: well it sounds like you are incredibly busy as a college student John sleeping
- sample_id=H015
  text: okay all right and let me ask you another question using a scale again so from scale of zero to ten zero being I'm not confident whatsoever and 10 being I'm super confident how confident are you that you will be able to make this change
- sample_id=H016
  text: why do you choose a  or a  or
- sample_id=H017
  text: what do you think you'll study
- sample_id=H018
  text: you know like think about it
- sample_id=H019
  text: to the same pencil as usual
- sample_id=H020
  text: I might not know my name, you know.

For each complete sentence output an integer predicted_feature_score from 0 to 100: 0–10 absent/contradicted; 11–30 weak/incidental; 31–60 partial/ambiguous; 61–80 clear; 81–100 strong and specific. Do not revise the explanation, infer hidden groups, or rely on broad domain membership. matching_evidence_span must be one exact contiguous substring copied from the sentence or empty. Preserve the sentence's original case, punctuation, and whitespace exactly; never normalize spacing, paraphrase, join disjoint spans, or insert ellipses. Return all 20 IDs exactly once and in input order. Return only JSON matching the supplied schema.
```

### 人工审核栏

- 匿名短名：`Utterance-initial so plus you-address`
- 主解释类型：`linguistic_structure`
- [ ] 解释在 held-out 句子上具有清楚、可重复的匹配边界
- [ ] 解释主要反映心理/话语功能，而非关键词、句法或模板代理
- [ ] 仅凭冻结解释即可稳定区分应高分与应低分句子
- [ ] 没有明显反例足以推翻主解释
- 人工结论：保留 / 收窄后保留 / 仅作语言代理 / 拒绝
- 批注：

## 附录：映射与去锚定 Held-out 指标（Reviewer-only）

以下字段没有发送给 Scorer。高分表示解释能预测未见响应，不自动证明标签特异性、心理学功能或因果机制。

| Feature | Latent | 关联叶标签 | 解释类型 | Spearman ρ | Pearson(log activation) | Positive-vs-zero AUROC | High-vs-weak accuracy |
|---|---:|---|---|---:|---:|---:|---:|
| F058 | 664 | QUO/QUC | `behavioral_function` | 0.791 | 0.791 | 0.973 | 0.960 |
| F176 | 23464 | AF | `behavioral_function` | 0.850 | 0.840 | 0.933 | 1.000 |
| F047 | 8583 | REC | `behavioral_function` | 0.719 | 0.794 | 0.860 | 0.960 |
| F018 | 3993 | REC | `behavioral_function` | 0.655 | 0.534 | 0.967 | 0.660 |
| F150 | 1713 | GI | `behavioral_function` | 0.689 | 0.693 | 0.933 | 0.800 |
| F072 | 32508 | QUO | `behavioral_function` | 0.580 | 0.552 | 0.840 | 0.700 |
| F040 | 26968 | REC | `affective_content` | 0.567 | 0.566 | 0.780 | 0.760 |
| F169 | 4756 | SU | `behavioral_function` | 0.589 | 0.549 | 0.893 | 0.700 |
| F006 | 19435 | RES/REC | `linguistic_structure` | 0.939 | 0.790 | 1.000 | 1.000 |

## 总体人工判定

- 可作为论文主例的卡：
- 可作为语言代理例的卡：
- 需要收窄解释的卡：
- 应拒绝的卡：
- 其他意见：
