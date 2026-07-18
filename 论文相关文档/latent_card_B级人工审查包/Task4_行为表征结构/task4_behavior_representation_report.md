# Task 4：标签的完整表征结构

## 1. 分析范围与方法

本分析直接提取冻结包中 225 张 feature card 已有的逐卡独立编码，并映射到 303 条 stable-core 标签–latent 关联。本报告不读取或依赖新的 API 二次编码结果，也没有修改卡片的 `explanation_type`。

六类既有编码全部进入分析：行为功能、语言结构、情感内容、主题、表层伪影以及不清晰/混合。归组由当前 AI 根据卡片解释、代表性证据、替代解释、混淆因素和限制完成。每个重复成分至少包含两个 stable-core 特征；不能形成多特征模式的卡片保留为孤立或未解析证据。

当前归组尚未经过人工审查。卡片的 support_fraction 是原生成模型的自报覆盖率，不是独立准确率。RE/RES/REC 缺少 client 前文，因此涉及反映的成分只能写成候选结构。

## 2. 总体编码构成

- 行为功能：174
- 语言结构：84
- 情感内容：11
- 主题：17
- 表层伪影：2
- 不清晰/混合：15

129 条非行为关联中，115 条进入 31 个多特征重复模式，14 条保留为孤立或未解析模式。多 latent 重复出现说明该模式值得作为系统性表征候选，但不等于已经排除了数据模板、转录偏差或共同语料主题。

## 3. 标签概览

| 标签 | 卡片 | 行为 | 语言 | 情感 | 主题 | 伪影 | 不清晰 | 行为成分 | 其他成分 | 孤立非行为 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RE | 54 | 29 | 21 | 0 | 0 | 1 | 3 | 5 | 5 | 2 |
| RES | 15 | 7 | 4 | 0 | 0 | 0 | 4 | 2 | 2 | 2 |
| REC | 47 | 29 | 14 | 0 | 0 | 1 | 3 | 4 | 3 | 3 |
| QU | 26 | 19 | 7 | 0 | 0 | 0 | 0 | 3 | 3 | 0 |
| QUO | 32 | 22 | 10 | 0 | 0 | 0 | 0 | 3 | 2 | 1 |
| QUC | 45 | 30 | 13 | 0 | 0 | 0 | 2 | 4 | 4 | 1 |
| GI | 26 | 6 | 3 | 0 | 17 | 0 | 0 | 2 | 5 | 0 |
| SU | 31 | 18 | 5 | 5 | 0 | 0 | 3 | 3 | 4 | 4 |
| AF | 27 | 14 | 7 | 6 | 0 | 0 | 0 | 3 | 3 | 1 |

## 4. 标签内行为成分

### RE

| 行为成分 | 支持特征 | 代表性证据 | 置信度 |
|---|---|---|---|
| 反映式重构与总结 | `F30224|F3993|F31133|F26800|F1211|F16292|F28269|F29874|F3416|F2167|F2089|F4625` | F30224/s005: it sounds like you really feel like you let yourself and your children down and that's hard to accept given how much you love your kids // F3993/s001: and so it's already it's pretty obvious and it also sounds like that pain is really taken up a pretty big role in your in your life and it feels like a pretty big barrier right there // F31133/s003: It does sound like something pretty significant happened that sort of  Freaked everybody out. | 高 |
| 矛盾与双面反映 | `F10181|F20436|F12852` | F10181/s001: so some of the negative things would be it's a impact on your health right the the smell clothes and so forth and then there's a lot of kind of social pushback // F20436/s001: okay so on one hand you feel like it would be a relief to be less stressed and overwhelmed about your schoolwork and then on the other hand you feel like you just don't have time to go to counseling because you have so much work to do // F12852/s002: okay so on one hand you feel like it would be a relief to be less stressed and overwhelmed about your schoolwork and then on the other hand you feel like you just don't have time to go to counseling because you have so much work to do | 高 |
| 情绪与压力反映 | `F22558|F26968` | F22558/s001: Wow okay so it sounds like you're feeling pretty confident after our discussion // F26968/s002: okay so right now you have so much going on that you're just feeling totally overwhelmed you don't even know where to start | 中 |
| 直接建议与风险反馈 | `F1516|F7054|F11405|F29845|F20564` | F1516/s003: Okay, well, um, you're going to have to cut the fast food down to no more than once a week // F7054/s001: yeah absolutely if you don't have that first if you can stop yourself having that first drink you're more inclined not to keep drinking of course // F11405/s002: Well, you know, you're doing good but don't forget what we talked about today, okay, about your chew tobacco and your mouth | 中 |
| 确认与探索性追问 | `F29759|F15504` | F29759/s007: So that means you want to quit now right // F15504/s005: So, you're gonna quit then? | 中 |

### RES

| 行为成分 | 支持特征 | 代表性证据 | 置信度 |
|---|---|---|---|
| 反映与理解核对 | `F28269|F29759|F8468` | F28269/s001: So, you're gonna quit then? // F29759/s007: So that means you want to quit now right // F8468/s002: and so because you were worried about them thinking that you were trying to avoid them you ended up using marijuana is that correct | 中 |
| 建议与风险提示 | `F1516|F32727|F27859` | F1516/s003: Okay, well, um, you're going to have to cut the fast food down to no more than once a week // F32727/s003: you're likely to face costly legal citations for grades health problems getting kicked out of school and disappointing your parents // F27859/s001: well those are some great alternatives instead of going to gym you save a lot of time | 中 |

### REC

| 行为成分 | 支持特征 | 代表性证据 | 置信度 |
|---|---|---|---|
| 反映式重构与总结 | `F31133|F26800|F30224|F3993|F16292|F29874|F1211|F2089|F1455|F31915|F3416` | F31133/s003: It does sound like something pretty significant happened that sort of  Freaked everybody out. // F26800/s002: you obviously know yourself pretty well // F30224/s005: it sounds like you really feel like you let yourself and your children down and that's hard to accept given how much you love your kids | 高 |
| 矛盾与双面反映 | `F20436|F10181|F12852` | F20436/s001: okay so on one hand you feel like it would be a relief to be less stressed and overwhelmed about your schoolwork and then on the other hand you feel like you just don't have time to go to counseling because you have so much work to do // F10181/s001: so some of the negative things would be it's a impact on your health right the the smell clothes and so forth and then there's a lot of kind of social pushback // F12852/s002: okay so on one hand you feel like it would be a relief to be less stressed and overwhelmed about your schoolwork and then on the other hand you feel like you just don't have time to go to counseling because you have so much work to do | 高 |
| 情绪与经历反映 | `F22558|F16256|F26968|F32596` | F22558/s001: Wow okay so it sounds like you're feeling pretty confident after our discussion // F16256/s003: and when you say it was scary I hear you saying on some level was scary for you and your your safety but it also sounds like it was scary because all these other variables or elements were put in motion that were disruptive // F26968/s002: okay so right now you have so much going on that you're just feeling totally overwhelmed you don't even know where to start | 中 |
| 建议、计划与风险反馈 | `F11405|F9993|F7054|F31867|F8226` | F11405/s002: Well, you know, you're doing good but don't forget what we talked about today, okay, about your chew tobacco and your mouth // F9993/s001: it might be your decision if you want children but as a doctor I'm not prepared to make that decision and to verify that decision for you // F7054/s001: yeah absolutely if you don't have that first if you can stop yourself having that first drink you're more inclined not to keep drinking of course | 中 |

### QU

| 行为成分 | 支持特征 | 代表性证据 | 置信度 |
|---|---|---|---|
| 开放式经历与观点探索 | `F9959|F26485|F21125|F24744|F26144|F24761|F18730|F32508` | F9959/s001: What kind of dog do you have? // F26485/s005: how do you feel about that // F21125/s001: okay so you said that you were a big runner what role did running play in your life | 高 |
| 改变动机与障碍唤起 | `F27061|F18310|F21203` | F27061/s001: do you know of any thing that will help you quit smoking? // F18310/s001: in what ways might your life be better if you succeed in making the changes you mentioned // F21203/s001: you also mentioned that smoking is a way to manage your weight and concern that if you are we're going to quit smoking what that would do in terms of gaining weight and that smoking after meals is pleasurable that it's something that would be really hard to give up | 高 |
| 聚焦式事实与行为评估 | `F13430|F29590|F21935|F24943|F22358|F21859` | F13430/s001: Did you come here for a prescription for anti-depressants? // F29590/s001: how long has it been since you kind of drink // F21935/s001: and you mentioned that you weren't sleeping too well do you have any weight changes during that time | 高 |

### QUO

| 行为成分 | 支持特征 | 代表性证据 | 置信度 |
|---|---|---|---|
| 开放式经历与观点探索 | `F9959|F26485|F24744|F24761|F18730|F21125|F19840|F32508|F26144|F19038|F3887|F19470|F3156|F14247` | F9959/s001: What kind of dog do you have? // F26485/s005: how do you feel about that // F24744/s001: um what do you think that there's more of more positive or negative | 高 |
| 改变动机与差距探索 | `F18310|F27061|F21203|F18990` | F18310/s001: in what ways might your life be better if you succeed in making the changes you mentioned // F27061/s001: do you know of any thing that will help you quit smoking? // F21203/s001: you also mentioned that smoking is a way to manage your weight and concern that if you are we're going to quit smoking what that would do in terms of gaining weight and that smoking after meals is pleasurable that it's something that would be really hard to give up | 高 |
| 聚焦式数量与病史询问 | `F13430|F29590` | F13430/s001: Did you come here for a prescription for anti-depressants? // F29590/s001: how long has it been since you kind of drink | 中 |

### QUC

| 行为成分 | 支持特征 | 代表性证据 | 置信度 |
|---|---|---|---|
| 结构化自我评分 | `F27857|F7037|F664|F12902|F5178|F6639|F21296|F2550|F29247|F29931` | F27857/s001: okay so how confident are you that you can make these changes on a scale of one to ten one being not confident at all and ten being the most confident you've ever felt // F7037/s001: great I'm in Romania to get a break from the sugar I sort of on a scale of 1 to 10 with 10 being the most how motivated do you feel you are to make this change // F664/s004: so on a scale of zero to 100 how important do you think it is for you to make this change | 高 |
| 聚焦式健康与病史评估 | `F21935|F13430|F14014|F20869|F8969|F14003|F28816|F24943|F17507|F29590|F12887|F20549|F31422` | F21935/s001: and you mentioned that you weren't sleeping too well do you have any weight changes during that time // F13430/s001: Did you come here for a prescription for anti-depressants? // F14014/s003: Did you come here for a prescription for anti-depressants? | 高 |
| 许可与选项协商 | `F18646|F4998|F8089|F32273` | F18646/s001: it's kind of the same old routine um I wonder if it would be okay if we could talk about some of the options that are out there for folks that have some mild depression // F4998/s001: so is this an inconvenience that you can move past or is this a major life event that does change things // F8089/s005: so does either of those sound like they might work for you | 中 |
| 一般性澄清追问 | `F22358|F10916|F26485` | F22358/s001: do you understand how important this is? // F10916/s001: okay so what other things can you tell me about that might affect your blood pressure // F26485/s005: how do you feel about that | 中 |

### GI

| 行为成分 | 支持特征 | 代表性证据 | 置信度 |
|---|---|---|---|
| 用药信息与执行说明 | `F8166|F16131|F19483` | F8166/s008: okay so I wrote a prescription for an antibiotic for Aidan that should help with the ear infection // F16131/s001: yeah I'm the pharmacist on duty citalopram okay it's an antidepressant and it comes in   and  milligram strength it's excreted in the urine and it has a half-life of about  hours for side effects there are sexual side effects heart side effects and stomachs side effects // F19483/s004: Okay well smoking really is really bad for you, you know it causes things like lung cancer, emphysema, heart disease. | 高 |
| 健康建议与风险指导 | `F27515|F10264` | F27515/s001: at this point is to think about transitioning you off of the oxy onto one of these over-the-counter medications and using some of these other techniques to help you manage the pain // F10264/s004: okay well for a man your age in this high risk category it is recommended that you actually cut out drinking altogether to reduce that risk | 中 |

### SU

| 行为成分 | 支持特征 | 代表性证据 | 置信度 |
|---|---|---|---|
| 共情理解与困难确认 | `F24760|F29825|F4756|F7389` | F24760/s001: I completely understand your you have a busy lifestyle // F29825/s001: sure it's quite understandable to be nervous coming in to talk to it like this // F4756/s004: and I do get where you're coming from | 高 |
| 提供帮助与支持可用性 | `F9720|F4512|F16190|F19935|F28603` | F9720/s005: listen I really want to help you // F4512/s005: yeah I really want to help you // F16190/s001: I'm here to help | 高 |
| 直接建议与风险评价 | `F9578|F26642|F20743` | F9578/s002: You've gotten pretty isolated. // F26642/s016: You're putting yourself at risk for all these other diseases // F20743/s016: It looks like you got a letter about your AC and you haven't been taking very good care of your diabetes. | 中 |

### AF

| 行为成分 | 支持特征 | 代表性证据 | 置信度 |
|---|---|---|---|
| 行动、目标与个人优势肯定 | `F30870|F6676|F28469|F22411|F24167` | F30870/s004: that's a great goal to have for yourself // F6676/s003: I'm glad you called. // F28469/s002: I am so proud of you that is awesome Taylor think I am just that couldn't be happier and wait till we tell the group | 高 |
| 肯定性反映 | `F7143|F25532` | F7143/s001: sounds like you're really motivated and you're doing some research and kind of staying on top of it that's you know trying // F25532/s001: that sounds like a great idea | 中 |
| 感谢与关系维持 | `F9869|F24206|F11511|F31149|F19654` | F9869/s003: Thank you so much for calling me back, I sure appreciate it // F24206/s001: so thank you so thank you for being here and congratulations again for taking that first step // F11511/s002: and this has been very useful I appreciate you | 高 |

## 5. 标签内语言、情感、主题及不清晰成分

### RE

| 证据层 | 表征成分 | 支持特征 | 代表性证据 | 置信度 |
|---|---|---|---|---|
| `linguistic_structure` | 反映式表层模板 | `F21800|F8034|F29190|F4018|F23670|F28688` | F21800/s004: okay so it sounds like you're willing to make a change and you're getting kind of excited about possibly having more energy // F8034/s004: Wow okay so it sounds like you're feeling pretty confident after our discussion // F29190/s001: yeah that sounds like that's a source of frustration | 高 |
| `linguistic_structure` | so 引导的话语启动结构 | `F19435|F31930|F14875|F5663|F9537|F17315` | F19435/s004: soyou're concerned about having a stroke // F31930/s003: So Greg, just focus on your drinking, can you tell me a bit about when you first started drinking when // F14875/s012: and so let's look at those two here | 高 |
| `linguistic_structure` | 第二人称评价与经历陈述 | `F17861|F20808|F4662|F13966` | F17861/s001: you keep throwing wine into the conversation // F20808/s002: You've gotten pretty isolated. // F4662/s001: yeah Chris come on you're getting peer pressure | 中 |
| `linguistic_structure` | it 引导的评价结构 | `F15068|F111|F30798|F15396` | F15068/s003: yeah and it got the feeling that it's something that you want to think about that this is a really big decision and it's not one that you're gonna make in a hurry // F111/s005: So it sounds that you have been dealing with this for a while? // F30798/s001: It does sound like something pretty significant happened that sort of  Freaked everybody out. | 中 |
| `unclear_or_mixed` | 混合型咨询话语 | `F2995|F3805|F13312` | F2995/s002: this is a big stuff for you I know I'm it's hard to make a step like this // F3805/s001: and I obviously haven't been this way like when I took this job I guess that's when my health started spiraling downwards and I don't know I guess that's when I kind of noticed things were kind of off as far as my health goes so // F13312/s001: but in looking through this chart I mean it seems like he's had six or seven of these just in the past year or so that's really a big problem | 低 |

### RES

| 证据层 | 表征成分 | 支持特征 | 代表性证据 | 置信度 |
|---|---|---|---|---|
| `linguistic_structure` | 第二人称指向结构 | `F20808|F13966|F19435` | F20808/s002: You've gotten pretty isolated. // F13966/s003: You're putting yourself at risk for all these other diseases // F19435/s004: soyou're concerned about having a stroke | 中 |
| `unclear_or_mixed` | 混合型医疗会谈 | `F11435|F23077|F32696` | F11435/s001: hey Monica how are you doing today // F23077/s027: five times a week yeah okay you know I wanted to show you on this chart here that using marijuana thought times a week the way you are and from what you said on our questionnaire put you in this category of being at risk // F32696/s001: Having diabetes affects you more than you realize and some people feel fine when they have a big ulcer on their foot. | 低 |

### REC

| 证据层 | 表征成分 | 支持特征 | 代表性证据 | 置信度 |
|---|---|---|---|---|
| `linguistic_structure` | 反映式表层模板 | `F15068|F21800|F23670|F29190|F111|F28688|F8034|F4018` | F15068/s003: yeah and it got the feeling that it's something that you want to think about that this is a really big decision and it's not one that you're gonna make in a hurry // F21800/s004: okay so it sounds like you're willing to make a change and you're getting kind of excited about possibly having more energy // F23670/s010: sounds like you're noticing quite a few not so good things about alcohol though mom got mad at you you're having decreased focus you're kind of not doing so well in school maybe | 高 |
| `linguistic_structure` | so 引导的话语启动结构 | `F19435|F14875|F5663|F31930` | F19435/s004: soyou're concerned about having a stroke // F14875/s012: and so let's look at those two here // F5663/s001: so you can't imagine that so is look at the other side can you imagine having further instance with law enforcement we're charges in the future | 高 |
| `unclear_or_mixed` | 混合型咨询话语 | `F3805|F31585|F2995` | F3805/s001: and I obviously haven't been this way like when I took this job I guess that's when my health started spiraling downwards and I don't know I guess that's when I kind of noticed things were kind of off as far as my health goes so // F31585/s001: so it sounds like you really want to go to Black Friday with your friends you're feeling a little anxious about it // F2995/s002: this is a big stuff for you I know I'm it's hard to make a step like this | 低 |

### QU

| 证据层 | 表征成分 | 支持特征 | 代表性证据 | 置信度 |
|---|---|---|---|---|
| `linguistic_structure` | WH 问句形式 | `F11660|F23183|F8658` | F11660/s002: What would you say? // F23183/s001: What, what are your thoughts right now about some of the options out there for you to do, if, if anything? // F8658/s001: so just drinking water so why don't we you said you've done it in the past right how do you feel about drinking five bottles | 高 |
| `linguistic_structure` | do you 助动词问句 | `F12340|F16969` | F12340/s001: do you think that this will be helpful I'm black Franny // F16969/s030: do you drink alcohol | 高 |
| `linguistic_structure` | 问句重复与不流利 | `F12544|F3459` | F12544/s004: what what's what kind of drinks do you have what do you drink // F3459/s001: What what do you think | 中 |

### QUO

| 证据层 | 表征成分 | 支持特征 | 代表性证据 | 置信度 |
|---|---|---|---|---|
| `linguistic_structure` | what 问句模板 | `F23183|F14833|F8191|F6129|F12337|F25210` | F23183/s001: What, what are your thoughts right now about some of the options out there for you to do, if, if anything? // F14833/s003: What do you think? // F8191/s003: what kind of dogs | 高 |
| `linguistic_structure` | WH 问句与不流利结构 | `F11660|F3459|F12544` | F11660/s002: What would you say? // F3459/s001: What what do you think // F12544/s004: what what's what kind of drinks do you have what do you drink | 中 |

### QUC

| 证据层 | 表征成分 | 支持特征 | 代表性证据 | 置信度 |
|---|---|---|---|---|
| `linguistic_structure` | 助动词前置的是非问句 | `F12555|F12340|F20463|F16969|F2504|F12676|F19364` | F12555/s005: And if Oscar's that important to you then do you put Oscar higher than your baby? // F12340/s001: do you think that this will be helpful I'm black Franny // F20463/s001: are you picking up for yourself today or for somebody else | 高 |
| `linguistic_structure` | 时长与数量问句 | `F29947|F8861` | F29947/s004: okay and how long have you been smoking? // F8861/s001: and I'm curious as to how much time did you spend walking your dog in nature | 高 |
| `linguistic_structure` | 量表问句模板 | `F18714|F32565|F2416` | F18714/s002: could you say on a scale of zero to ten ten being really ready zero being not ready at all how ready would you be to do some counselling // F32565/s004: so let me ask you on a scale of zero to ten zero being not confident at all and ten being 100% confident how confident are you that you will be able to stay consistent with this new exercise regime // F2416/s002: and on a scale from zero to 100 100 how confident are you in making this change | 高 |
| `unclear_or_mixed` | 混合问句格式 | `F9827|F736` | F9827/s001: okay so is there anything else that you like about it // F736/s013: did you have a sense when you came in | 低 |

### GI

| 证据层 | 表征成分 | 支持特征 | 代表性证据 | 置信度 |
|---|---|---|---|---|
| `linguistic_structure` | 会谈过渡与确认标记 | `F2055|F23723|F5828` | F2055/s001: um now I was taking a look at your medication profile earlier and it looks like you're on two different strengths of lipitor so that obviously it can't be right // F23723/s003: Actually, if you look up here... the roof of your mouth, I'm noticing a lot of irritation that's really consistent with tobacco use // F5828/s001: no but you know I was recording the show that I watch and it was on there and so I've just gone back and watched it you know again and again my wife I've heard it a couple other times on the news hey everybody yeah | 中 |
| `topic` | 药物与治疗主题 | `F16345|F8294|F21634|F24876|F1713|F17556` | F16345/s001: he's right this medicine is designed to help lower your cholesterol and that means that it will help lower your risk of having a heart attack or a stroke // F8294/s001: yeah I'm the pharmacist on duty citalopram okay it's an antidepressant and it comes in   and  milligram strength it's excreted in the urine and it has a half-life of about  hours for side effects there are sexual side effects heart side effects and stomachs side effects // F21634/s003: okay well you're going to take one tablet of cholesteryl sent in twenty one time a day by mouth and you can take that with or without food | 高 |
| `topic` | 健康风险与指标主题 | `F26879|F26236|F16515|F7148|F20329` | F26879/s003: because smoking is a major risk factor for DVT even if you even if we treat you now if you don't quit smoking I'm afraid the blood clots will return again // F26236/s001: okay okay so this medication is for your cholesterol // F16515/s001: did you know that a glass of orange juice has almost as many calories as a can of cola | 高 |
| `topic` | 饮食、体重与活动主题 | `F9893|F18490|F30517|F6651` | F9893/s004: there are there's a couple out there one called protein power and but a lot of my patients have had a lot of luck with South Beach diet // F18490/s001: overall, obesity can lead to many chronic conditions // F30517/s001: John is very important that you recognize that some fruits but especially fruit juice can contain a huge amount of calories | 高 |
| `topic` | 一般健康行为主题 | `F17827|F17685` | F17827/s007: your ac it measures the amount of sugar that's been on your hemoglobin in the last three months to determine how well you've been controlling your blood sugar // F17685/s001: most people who smoke do it because they as does something good for them | 中 |

### SU

| 证据层 | 表征成分 | 支持特征 | 代表性证据 | 置信度 |
|---|---|---|---|---|
| `linguistic_structure` | 第一人称支持性表达 | `F19359|F8760|F16045` | F19359/s001: I'm guessing you've learned a lot of lessons // F8760/s003: I understand you've come here today to the agency because you have a recent arrest // F16045/s001: I have a lot of thoughts about it a lot of memories about it I think I've just kind of got to work through them you know we had this really sad but I can't let it roll my life you said it | 中 |
| `linguistic_structure` | 指示词评价结构 | `F22730|F24856` | F22730/s001: This is the first time that you've had something that really happened, this incident that happened where he pushed you really, prompted you to, to take some steps // F24856/s010: and that's great | 中 |
| `affective_content` | 困难与负向体验内容 | `F16736|F11872` | F16736/s001: I mean realistically not everyone is a success story and we see its relapse it's a part of recovery but but for most cases though that difficult cases that do a good job and make that transformation when does that usually occur // F11872/s001: so you've tried some diets before and they haven't worked out and since you're a single mom with kids is it's been hard for you to cook meals or do grocery shopping? | 高 |
| `unclear_or_mixed` | 混合型支持话语 | `F5366|F2995` | F5366/s001: that's particularly hard when those anniversaries come around for the passings of our parents that's a that's a very difficult time I // F2995/s002: this is a big stuff for you I know I'm it's hard to make a step like this | 低 |

### AF

| 证据层 | 表征成分 | 支持特征 | 代表性证据 | 置信度 |
|---|---|---|---|---|
| `linguistic_structure` | 感谢公式结构 | `F17793|F28724|F2434` | F17793/s003: thanks so much for coming in today // F28724/s001: hi Heather thanks for coming in // F2434/s001: Thanks for comming today | 高 |
| `linguistic_structure` | 评价性句式（以正向为主） | `F32764|F24856|F16965` | F32764/s006: Wow yeah wow that is awesome // F24856/s010: and that's great // F16965/s001: does that capture for you some of the good things or things that are keeping you smoking | 高 |
| `affective_content` | 正向评价与赞扬内容 | `F23464|F22475|F29908|F453|F1871|F32320` | F23464/s002: that's good to hear // F22475/s010: it's good to see you again // F29908/s001: Wow, nice. | 高 |

## 6. 孤立、未归组和潜在伪影

- `RE`：未归组行为卡片 `F31363|F23242|F19005|F13169|F26681`；孤立/未解析非行为卡片 `F26319(linguistic_structure)|F26869(surface_artifact)`。
- `RES`：未归组行为卡片 `F16320`；孤立/未解析非行为卡片 `F29701(unclear_or_mixed)|F26319(linguistic_structure)`。
- `REC`：未归组行为卡片 `F29759|F19005|F30091|F23242|F13169|F3673`；孤立/未解析非行为卡片 `F26319(linguistic_structure)|F17861(linguistic_structure)|F26869(surface_artifact)`。
- `QU`：未归组行为卡片 `F664|F10916`；孤立/未解析非行为卡片 `无`。
- `QUO`：未归组行为卡片 `F664|F10916`；孤立/未解析非行为卡片 `F12340(linguistic_structure)`。
- `QUC`：未归组行为卡片 `无`；孤立/未解析非行为卡片 `F11660(linguistic_structure)`。
- `GI`：未归组行为卡片 `F13751`；孤立/未解析非行为卡片 `无`。
- `SU`：未归组行为卡片 `F11948|F9109|F29856|F20778|F5732|F26263`；孤立/未解析非行为卡片 `F30223(unclear_or_mixed)|F28795(affective_content)|F7382(affective_content)|F23464(affective_content)`。
- `AF`：未归组行为卡片 `F18492|F5320`；孤立/未解析非行为卡片 `F26938(linguistic_structure)`。

`F26869` 在 RE 和 REC 中均为 stable-core 关联，但卡片编码是 `surface_artifact`（所有格缩写/转录错误）。这说明它不是一次随机运行中临时出现的索引，却仍更可能反映数据或转录规律，而不是咨询行为理解。

## 7. 跨标签共享表征

| 共享成分 | 证据层 | 标签 | 范围 | 共享证据 | 标签差异 |
|---|---|---|---|---|---|
| 反映式重构与总结 | `behavioral_function` | `RE|RES|REC` | `parent_child` | 均以重述、概括或核对对方表述为核心。 | RES 更偏简短核对；REC/RE 包含更丰富的总结候选。 |
| 矛盾与双面反映 | `behavioral_function` | `RE|REC` | `parent_child` | 均组织冲突目标、利弊或一方面/另一方面结构。 | 现有数据无 client 前文，不能比较反映准确性与复杂程度。 |
| 直接建议与风险指导 | `behavioral_function` | `RE|RES|REC|GI|SU` | `cross_family` | 均直接提供建议、行动方向或风险反馈。 | GI 更偏专业健康建议；SU、RE 家族中的此类证据更可能代表标签内异质或共享 latent。 |
| 开放式探索 | `behavioral_function` | `QU|QUO|QUC` | `parent_child` | 均邀请说明经历、观点、感受或原因。 | QUC 组更偏澄清，QUO 组的展开性证据更集中。 |
| 改变动机与障碍唤起 | `behavioral_function` | `QU|QUO` | `parent_child` | 均探索改变理由、障碍、差距或两面性。 | QU 是父标签汇总，QUO 是叶级候选。 |
| 聚焦式事实评估 | `behavioral_function` | `QU|QUO|QUC` | `parent_child` | 均收集病史、数量、频率或明确行为状态。 | 该功能在 QUO 中属于标签内异质证据，在 QUC 中支持最集中。 |
| 反映式表层模板 | `linguistic_structure` | `RE|REC` | `parent_child` | 两个标签均集中出现 sounds/seems/looks like 与 okay so 等模板。 | 该共享首先是语言实现证据，不证明两标签具有同一行为机制。 |
| so 引导的话语启动 | `linguistic_structure` | `RE|REC` | `parent_child` | 多个 latent 重复编码 so/okay so 引导的问题、总结或回应。 | so 是通用会话标记，标签特异性弱。 |
| 第二人称指向结构 | `linguistic_structure` | `RE|RES` | `parent_child` | 均以 you/you're 指向对方经历、状态或行为。 | RE 组包含更多评价结构，RES 组规模较小。 |
| 未解析的混合咨询话语 | `unclear_or_mixed` | `RE|RES|REC` | `parent_child` | 三个标签均有多个 stable-core 卡片无法从咨询/医疗会谈语体中分离单一模式。 | 这是共享的不确定性证据，不是共享行为成分。 |
| 疑问句表层实现 | `linguistic_structure` | `QU|QUO|QUC` | `parent_child` | 三个标签均由明确疑问句模板构成重要表征成分。 | QUO 更偏 what/WH 模板，QUC 更偏助动词前置的是非问法。 |
| 问句重复与不流利 | `linguistic_structure` | `QU|QUO` | `parent_child` | 两个标签均出现疑问词重复、重启和犹豫。 | 该模式可能来自口语或转录过程，而非标签功能。 |

## 8. 各标签的综合表征方式

- `RE`：以反映、总结和矛盾呈现为主要行为候选，同时强烈依赖 sounds/seems like、so 和第二人称评价等语言模板；因此其结构兼有功能层与表层实现层。
- `RES`：行为候选较少，主要是理解核对及少量建议/风险提示；语言层以第二人称指向为主，同时存在较多无法解析的医疗会谈混合卡片。
- `REC`：反映式重构、矛盾反映和情绪经历反映构成主要行为候选，且与反映模板和 so 话语启动高度并行；缺少 client 前文使复杂反映解释仍受限。
- `QU`：同时包含开放探索、改变唤起和事实评估，并由 WH、do-you 与口语不流利等问句形式实现，表现为功能与句法共同组织。
- `QUO`：开放经历探索和改变动机探索最集中，表层上主要由 what/WH 模板实现；少量聚焦询问说明该标签内部并非纯粹开放问句。
- `QUC`：结构化评分和聚焦健康评估最突出，语言层对应助动词前置、数量问句和量表模板，功能与表面形式之间具有较强一致性。
- `GI`：除用药说明和健康建议行为外，17/26 的关联由药物、风险指标、饮食体重等主题卡片构成，说明该标签在当前模型中很大程度按领域内容组织。
- `SU`：共情确认和帮助提供是主要行为候选，同时编码第一人称支持句式与困难/负向体验；正负情感评价单例方向不一致，因此保留为孤立模式而不合并。
- `AF`：行动/优势肯定、肯定性反映和感谢行为与正向评价情感、评价句式及感谢公式共同出现，说明 AF 兼具行为功能与高度可预测的正向词汇模板。

## 9. 如何判断不是简单偶然误差

本分析使用三层证据区分系统性候选与偶然线索：第一，输入 latent 已属于跨重采样筛选得到的 stable core；第二，同一标签内至少两个独立 latent 共享模式才形成重复成分；第三，跨标签的同一 latent 复用与同类成分复现分别单列。

这些条件能够降低单一异常卡片造成的误判，但不能排除共同词汇、模板化语料、标签层级共现或转录偏差。因此，语言结构、主题和情感模式应写成模型表征标签的方式，而不是自动升级为行为机制；表层伪影和不清晰模式则作为结构性风险证据保留。

精确共享 latent 见 `task4_exact_shared_latents.csv`；全部 60 个多特征成分见 `task4_all_representation_components.csv`；只在一个标签形成的模式见 `task4_label_specific_components.csv`。人工双审查、kappa 和人工质量审核本轮暂不执行。
