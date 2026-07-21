# 代表性latent card

> 每标签选择一个当前v3中Spearman较高且尽量不重复的可评分stable-core latent。解释和句子保留原始英文，未重新调用模型或人工润色。

## RES — F006 / latent 19435

- Short name: `Utterance-initial so plus you-address`
- Type: `linguistic_structure`
- Spearman: `0.9374`
- AUROC: `0.9867`
- High–weak accuracy: `1.0000`

**Primary explanation**

A narrow linguistic pattern distinguishes the groups: Group A sentences are framed as “so you…” statements, while Group B lacks that exact initial second-person formulation despite sometimes using related counseling language.

**Surface/linguistic hypothesis**

The feature tracks a surface opening formula: sentence-initial “so” directly followed by a second-person pronoun or contraction.

**Behavioral/discourse hypothesis**

The feature may reflect therapist/counselor summary or reformulation turns that present an inference about the client’s state, behavior, or concern, but the available contrast is more consistently explained by the opening wording than by discourse function alone.

**Strong examples**

- `A001` so you don't know about the miracle you wake up believing that you would still have the problem
- `A002` so you've identified one of the four that might you might be willing to to try for a while to let go of
- `A003` so you're a little out of your comfort running zone so to say you don't feel as comfortable running here as you did back at home with your friends

**Weak-positive contrasts**

- `B001` all right well you mentioned a couple times that you've made a few decisions to try and limit the a dangerous aspect of the drinking why did you make those decisions
- `B002` and then you can tell them to stop - you know then that'd be really good good news

## REC — F022 / latent 15068

- Short name: `Declarative abstract-it framing`
- Type: `linguistic_structure`
- Spearman: `0.9282`
- AUROC: `0.9667`
- High–weak accuracy: `1.0000`

**Primary explanation**

The most stable distinction is linguistic: Group A repeatedly makes declarative predications with abstract subject 'it,' while Group B's 'it' tokens are usually interrogative, object-like, instructional, or discourse-management uses.

**Surface/linguistic hypothesis**

The feature tracks declarative clauses where abstract 'it' functions as grammatical subject, often with copular or evaluative predicates: 'it is/it's,' 'it sounds like,' 'it began,' 'it gets.'

**Behavioral/discourse hypothesis**

The feature may reflect counselor-style reflective summarizing or reframing, where the speaker packages the client's concern as an abstract situation or decision and comments on it rather than directly asking or instructing.

**Strong examples**

- `A001` well I appreciate you thinking about I hope you'll think about it really hard you know this is something that I'm going to continue to ask you about because it's something that really needs to change
- `A002` yeah and it got the feeling that it's something that you want to think about that this is a really big decision and it's not one that you're gonna make in a hurry
- `A003` so you think it began about four years ago after this holiday

**Weak-positive contrasts**

- `B001` how important is it to you to make an adjustment in your drinking or even to quit a lot
- `B002` so Center what's good about smoking what do you like about it

## QUO — F080 / latent 6129

- Short name: `Formulaic “what do you think” question`
- Type: `linguistic_structure`
- Spearman: `0.9446`
- AUROC: `0.9333`
- High–weak accuracy: `1.0000`

**Primary explanation**

Group A is unified by the formulaic interrogative “what do you think,” while Group B contains either different question forms, non-question uses of “think,” or challenges like “what’s there to think about.”

**Surface/linguistic hypothesis**

The feature tracks the lexical-syntactic string “what do you think” / close disfluent or mistranscribed variants of that string.

**Behavioral/discourse hypothesis**

The feature may mark open-ended counselor questions that solicit the client’s opinion, evaluation, or next-step planning, but the evidence is strongest for the repeated surface question formula rather than a broader counseling function.

**Strong examples**

- `A001` what do you think you'll do next
- `A002` so what do you think
- `A003` what do you think that the pros and cons of making this change our

**Weak-positive contrasts**

- `B001` how about when you were younger was there time when you didn't feel like they were bothering in
- `B002` what's there to think about you said it's not that something you do all that often I'm there's all these negative side effects there's all these risks that you're putting yourself in

## QUC — F124 / latent 2550

- Short name: `one-to-ten scale framing`
- Type: `linguistic_structure`
- Spearman: `0.9075`
- AUROC: `0.9667`
- High–weak accuracy: `0.9600`

**Primary explanation**

The narrowest stable distinction is explicit one-to-ten scale framing. Almost every Group A item asks a rating question using one and ten as endpoints, while many Group B items either use zero-to-ten scales, are not scale questions, or contain only partial/ambiguous one-to-ten references.

**Surface/linguistic hypothesis**

The feature tracks surface scale wording, especially phrases like “on a scale of one to ten,” “from one to ten,” or “one being ... and ten being ...”.

**Behavioral/discourse hypothesis**

The feature may reflect counselor-style elicitation of a self-rating, especially motivational-interviewing ruler questions about confidence, importance, or readiness; however, the strongest evidence is the explicit one-to-ten linguistic frame rather than the counseling function alone.

**Strong examples**

- `A001` on a scale of one to ten where one is not at all antennas completely what number
- `A002` mm-hmm and so in terms of a readiness to actually go ahead and make a quit attempt where do you think you fall on the scale of one to ten where one is I'm not considering it at all and ten is I'm definitely ready I want to quit
- `A003` so on  a scale of one to ten how confident are  you that you can make this change Inez?

**Weak-positive contrasts**

- `B001` seven pounds a week or
- `B002` could you say on a scale of zero to ten ten being really ready zero being not ready at all how ready would you be to do some counselling

## GI — F142 / latent 26236

- Short name: `cardiometabolic risk marker talk`
- Type: `linguistic_structure`
- Spearman: `0.8410`
- AUROC: `1.0000`
- High–weak accuracy: `0.8000`

**Primary explanation**

The simplest stable distinction is topical-lexical: Group A sentences contain explicit cardiometabolic risk-factor terminology or treatments, whereas Group B contains broader health counseling language with only occasional weak overlap.

**Surface/linguistic hypothesis**

The feature tracks lexical content: explicit cardiometabolic medical vocabulary such as cholesterol, triglycerides, blood pressure, hypertension, diabetes, heart disease, kidney disease, simvastatin, lisinopril, or hydrochlorothiazide.

**Behavioral/discourse hypothesis**

The feature may reflect counseling moments where the speaker explains or plans management of chronic cardiometabolic risk, including medication adherence, lowering cholesterol, controlling blood pressure, and preventing complications.

**Strong examples**

- `A001` right exactly so that's why he wrote you this prescription so it's greater cholesterol you're just gonna take it once a day it's really easy C okay
- `A002` ok so what's nothing to me is that your doctor saying you have an issue a health issue at your cholesterol and nearly an allison issue of this present time cycle
- `A003` such as hypertension diabetes heart diseases

**Weak-positive contrasts**

- `B001` well it sounds like you realize that when you lost the weight you feel better and that it's going to affect your overall health
- `B002` yeah exactly yeah in two in two of them had diabetes as well so sometimes we were just Chad you know see you people were you know just taking care of themselves get into the clinic doing whatever they needed to do so

## SU — F166 / latent 19359

- Short name: `utterance-initial first-person singular`
- Type: `linguistic_structure`
- Spearman: `0.9749`
- AUROC: `0.9667`
- High–weak accuracy: `1.0000`

**Primary explanation**

The stable distinction is initial position: all Group A examples start with "I" or "I'm," whereas Group B examples either lack first-person singular starts or place them after prefaces such as "so," "okay," "um," or other material.

**Surface/linguistic hypothesis**

The feature marks utterances whose surface form begins immediately with a first-person singular pronoun/contraction, especially "I" or "I'm."

**Behavioral/discourse hypothesis**

The feature may reflect speaker-centered turns where the speaker foregrounds their own state, intention, belief, or action at the very start of the turn, but the evidence is primarily surface-linguistic rather than behavioral.

**Strong examples**

- `A001` I'm just wondering what's going on
- `A002` I'm going to need about ten minutes
- `A003` I really think that making exercise important part of your wife is crucial for you

**Weak-positive contrasts**

- `B001` so before we start I want to know if it was okay if I take notes
- `B002` so if those sound like things that you might be willing to do I'm I'm all for it

## AF — F182 / latent 2434

- Short name: `thanks_for_construction`
- Type: `linguistic_structure`
- Spearman: `0.9136`
- AUROC: `0.9267`
- High–weak accuracy: `0.9800`

**Primary explanation**

The simplest stable distinction is linguistic: all strong examples contain explicit “thank you/thanks for ...” wording, while weak examples mostly contain greetings, appreciation, or discussion of decisions without that exact thanking-for construction.

**Surface/linguistic hypothesis**

The feature is triggered by the lexical-syntactic pattern “thank you/thanks” plus a prepositional complement headed by “for.”

**Behavioral/discourse hypothesis**

The feature marks a speaker performing a closing or rapport-building act of gratitude toward the interlocutor or audience, often thanking them for attending, sharing, trusting, watching, or giving time.

**Strong examples**

- `A001` excellent well thank you so much for coming in today
- `A002` thank you for trusting me
- `A003` thanks for discussing this with me

**Weak-positive contrasts**

- `B001` hi Hannah nice to meet you hi so I understand that you were referred to me by your GP because you've been experiencing some anxiety difficulties
- `B002` okay knows me a lot of work and I respect your decision that you are not ready to give the food
