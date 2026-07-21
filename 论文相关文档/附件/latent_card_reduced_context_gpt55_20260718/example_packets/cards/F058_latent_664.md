# F058 / latent 664：以客户为中心的信息引导问题

> Selection tier: `paper_ready_primary`  
> Stable-core labels: `QUC, QUO`  
> Explanation type: `behavioral_function`

## 中文审核结论

- 入选理由：同时保留 what/how 等表面线索与邀请对方提供经历、判断或计划的行为功能，且 held-out 区分稳定。
- 解释边界：QUO 与 QUC 共享，适合说明跨标签共享证据，而非标签特异 latent。
- 证据等级：held-out 相关性/排序忠实度证据；不是因果或标签等价证明。

## 冻结解释（英文原文）

- **Short name:** client-centered eliciting question
- **Surface/linguistic hypothesis:** The feature is associated with direct or indirect open-ended question forms, especially wh- or how-phrases such as "what," "how many," "how often," and "what sort of," often addressed with "you."
- **Behavioral/discourse hypothesis:** The feature marks a collaborative elicitation move in an interview or counseling dialogue: the speaker invites the client/patient to supply information, assess seriousness, describe prior attempts, or choose a next step, rather than giving advice or making a statement.
- **Primary explanation:** A stable distinction is that Group A turns are primarily questions soliciting the addressee's own account, assessment, or plan. Group B turns are less consistently eliciting: many are advice, rapport, reflections, affirmations, or closed/challenging questions.
- **Explanation type:** `behavioral_function`

## Held-out 忠实度

| Spearman | Pearson(log activation) | Positive-vs-zero AUROC | High-vs-weak accuracy |
|---:|---:|---:|---:|
| 0.776 | 0.767 | 0.920 | 0.940 |

## Discovery：强响应句

| ID | Sentence |
|---|---|
| A001 | Alright well what are the kinds of things do you do for fun on weekends? |
| A002 | I'm just wondering what's going on |
| A003 | so what are some of those that you came up with |
| A004 | um well on a one to ten scale  being the most serious how serious have a problem do you think this is and you're like right now |
| A005 | and what what sort of treatments have you tried for this before |
| A006 | what other lessons have you learned from quitting |
| A007 | so how many cigarettes would you say that you have you smoked each day? |
| A008 | is that okay with you if we have this conversation at some point in the future |
| A009 | so what would you like to do next you |
| A010 | so how how often do you think that you'll be able to exercise |

## Discovery：弱正响应句（hard negatives）

| ID | Sentence |
|---|---|
| B001 | yeah so uh I think probably it sounds to me like probably changing medications at this point as a start we can always add another medication later if that becomes necessary |
| B002 | hi-oh Vina how are you doing today it's good to see you |
| B003 | Is your, is your head not attached to your body or something? |
| B004 | so let me make sure that I understand all of the good things about smoking the things that you like about it |
| B005 | okay now with insulin there's some things that we'd like to that sometimes can be a problem too like hypoglycemia have you experience animal |
| B006 | excuse excuse me if not you've not had the time to take the whole course of tablets so how do you know they're not working |
| B007 | have you ever had any legal or work or social problems because you're drinking |
| B008 | so do you drink alcohol? |
| B009 | because you're you're the kind of parent that wants to make sure your kids are doing well |
| B010 | you did yeah I want to hear about |

## Held-out 预测

| ID | Stratum | True activation | Predicted score | Matching evidence | Sentence |
|---|---|---:|---:|---|---|
| H001 | high | 2.26562 | 98 | how many hours a day on average would you say that you work out? | how many hours a day on average would you say that you work out? |
| H002 | high | 2.3125 | 88 | where would you say your are in terms of your confidence | okay bad idea on in that same scale with one being not confident at all to 10 being super super confident where would you say your are in terms of your confidence to actually lose the way |
| H003 | high | 2.3125 | 90 | which number best reflects how important it is to you | so on a scale of one to ten which number best reflects how important it is to you to drink the low-risk lifts |
| H004 | high | 2.89062 | 92 | how ready would you say you are to commit | and then one final question on a scale of  to   being not ready at all and  being extremely ready how ready would you say you are to commit to taking some kind of brief break and that let's say the next couple of weeks |
| H005 | high | 2 | 96 | what made you say five or six rather than two or three | okay and so what made you say five or six rather than two or three |
| H006 | mid | 0.710938 | 8 |  | well so Linda what I'm hearing you say is that it's actually very important for you to exercise but on the other hand you are not that confidence to actually start a success |
| H007 | mid | 0.9375 | 94 | why and eight and not on a one or two | and and you've spoken to this already a little bit i believe but I'll ask you again why and eight and not on a one or two |
| H008 | mid | 0.671875 | 86 | can you tell me about your day from when you wake up | so can you tell me about your day from when you wake up to |
| H009 | mid | 0.734375 | 78 | where were you when you got under the bleachers | okay so you still had the pencil you got under the bleachers you say you were feeling angry still are you still on the scale of one to ten where were you when you got under the bleachers |
| H010 | mid | 0.9375 | 98 | what kind of activities do you like doing? | what kind of activities do you like doing? |
| H011 | weak | 0.429688 | 35 | how long you have | and whether you're getting me to do you know how long you have |
| H012 | weak | 0.386719 | 45 | do you think if you put up with a little bit of embarrassment | do you think if you put up with a little bit of embarrassment |
| H013 | weak | 0.376953 | 48 | something be willing to do right now | and if lo and your drinking would have that effect that something be willing to do right now |
| H014 | weak | 0.421875 | 40 | how are you | I'm hailing the dietitian how are you |
| H015 | weak | 0.421875 | 90 | why did you get your lip pierced why? | Emily why why did you get your lip pierced why? |
| H016 | zero | -0 | 5 |  | it so it sounds like you know you really you really have a reason that you really want to be here you care about your wife can you care about your job and you really like what you do is what I'm hearing right now you're good at it look |
| H017 | zero | -0 | 2 |  | thank you very much for your time live |
| H018 | zero | -0 | 55 | have you been taking your Rubiff reguarly? | Well have you been taking your Rubiff reguarly? |
| H019 | zero | -0 | 18 | isn't it | well so you're doing your best but but pretty pretty disappointed in the way things are for you these nights not seeing a whole lot of its kind of kind it is kind of your life isn't it I mean you guys kids what it is |
| H020 | zero | -0 | 3 |  | but you did perfect opposite action right there which is you are feeling really uncomfortable and I can tell that but you still did it |

## 方法提醒

该卡只说明冻结解释能够在未见句子上预测相对响应。stable-core 标签来自相关性筛选，不能写成该 latent 等价于该标签。
