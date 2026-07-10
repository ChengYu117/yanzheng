# Explainer Quality Sample Review
Total validated explanations: 602
Sampled: 18 (2 per label: highest + lowest confidence)

## AF - highest confidence (0.95)
- packet_id: ctli_0286
- latent_idx: 2434
- short_name: explicit_gratitude_thanks
- feature_type: surface_form
- main_hypothesis: This latent activates on counselor utterances containing explicit gratitude expressions -- 'thank you', 'thanks for', 'thank you for'. The trigger is the presence of the word 'thanks' or 'thank you' as a direct expression of appreciation, especially in session-opening or session-closing contexts.
- positive_triggers: ["Direct 'thank you' or 'thanks' expressions ('Thanks for coming today', 'thank you for sharing')", "Gratitude with specific object ('thank you for trusting me', 'thanks for completing the questionnaire')", "Appreciation statements ('I appreciate you', 'I sure appreciate it')", "Session-opening gratitude ('thanks so much for coming in today')"]
- explicit_exclusions: ['Reflective listening without gratitude', 'Questions or prompts', "Empathic statements that don't include gratitude", 'Positive evaluations that lack explicit thanks']
- possible_surface_confounds: ["Near-miss s012 ('I appreciate how difficult it must be') contains 'appreciate' but not 'thank/thanks', suggesting the trigger is specifically the 'thank' token family", 'Sample size limits ability to separate surface from functional triggers']
- **Classification: surface_form**
- key_evidence: ['s004', 's002', 's012', 's015', 's007', 's011']
- failure_modes: ['Low-activation samples (s011, s010) show weak triggering, suggesting the pattern threshold is not sharp', 'Near-miss samples with nonzero activation (s012) suggest partial pattern overlap that the hypothesis does not fully explain', 'Cannot determine from text alone whether the latent tracks surface tokens or deeper semantic properties', 'Sample selection may not cover the full activation landscape of this latent']

## AF - lowest confidence (0.3)
- packet_id: ctli_0281
- latent_idx: 30870
- short_name: mixed_activation_no_clear_pattern_r02
- feature_type: unclear
- main_hypothesis: The activation pattern for this latent does not cleanly separate active from non-active samples on any single linguistic feature. The active samples span diverse topics, syntactic forms, and pragmatic functions with no obvious shared property absent from near-miss samples.
- positive_triggers: ['No consistent trigger pattern identified across active samples', 'Active samples vary in length, syntax, topic, and pragmatic function', 'Possible dependence on hidden-state properties not visible in surface text']
- explicit_exclusions: ['No reliable exclusion criteria can be stated given sample heterogeneity', 'Near-miss samples share some features with active samples without triggering']
- possible_surface_confounds: ['The lack of a clear pattern suggests either insufficient sample diversity or a non-linguistic trigger', 'Sample size limits ability to separate surface from functional triggers']
- **Classification: unclear**
- key_evidence: ['s003', 's001', 's018', 's012', 's008', 's011']
- failure_modes: ['Low-activation samples (s011, s010) show weak triggering, suggesting the pattern threshold is not sharp', 'Near-miss samples with nonzero activation (s012, s013, s014, s015) suggest partial pattern overlap that the hypothesis does not fully explain', 'Cannot determine from text alone whether the latent tracks surface tokens or deeper semantic properties', 'Sample selection may not cover the full activation landscape of this latent']

## GI - highest confidence (0.95)
- packet_id: ctli_0238
- latent_idx: 23723
- short_name: adverb 'actually' in emphasis
- feature_type: surface_form
- main_hypothesis: The latent fires strongly on the word 'actually' used for emphasis, correction, or contrastive focus in counselor utterances. The trigger is primarily the presence of the token 'actually' itself, functioning as a discourse marker that signals unexpectedness, correction, or emphasis.
- positive_triggers: ["Emphatic 'actually': 'it's actually going to be time times of actually the soul'", "Corrective 'actually': 'actually getting out with people actually makes you even more anxious'", "Self-correcting 'actually': 'it was actually I feel pretty good'", "Informational 'actually': 'I can also give you a handout that shows you actually plate'", "Consequential 'actually': 'so that's actually kind of a lot of drinking then'"]
- explicit_exclusions: ["Utterances without 'actually' (e.g., 'does it hurt anymore', 'can you tell me more', 'well you're here today for the refill')", "Any fluent counseling utterance that does not contain the word 'actually'"]
- possible_surface_confounds: ["The token 'actually' in any context (counseling or otherwise) would likely trigger", "Synonymous adverbs ('really', 'in fact', 'truly') would not trigger because the latent is surface-form specific"]
- **Classification: surface_form**
- key_evidence: ['s002', 's004', 's005', 's015', 's013']
- failure_modes: ['The latent is highly surface-form specific and would not generalize to other emphasis markers', "Rare orthographic variants ('actualy') might not trigger"]

## GI - lowest confidence (0.2)
- packet_id: ctli_0243
- latent_idx: 10264
- short_name: latent_10264_mixed
- feature_type: unclear
- main_hypothesis: This latent does not cleanly map to a single interpretable counseling function. Active samples include directive (3/11), information (2/11), short_filler (2/11), but none strongly distinguishes them from inactive samples.
- positive_triggers: ['no single dominant trigger']
- explicit_exclusions: []
- possible_surface_confounds: ["Near-miss 'well you're here today for the refill...' shares features but falls below threshold -- possible activation boundary effect", "Near-miss 'so we have some materials on the table...' shares features but falls below threshold -- possible activation boundary effect", "Near-miss 'I'm wondering what you could have done to get in the mood...' shares features but falls below threshold -- possible activation boundary effect"]
- **Classification: unclear**
- key_evidence: ["Highest activation sample: 'it might not but it'll bring it down a little bit maybe each thing you try brings it down a little bit more'", "Near-miss boundary: 'well you're here today for the refill'"]
- failure_modes: ['Analysis based on limited sample set; pattern may not generalize across all counseling contexts', "Surface text features may not capture the true activation mechanism in the model's internal representation", 'Near-miss boundary may shift with different sample selection']

## QU - highest confidence (0.65)
- packet_id: ctli_0117
- latent_idx: 13430
- short_name: phrase-did-you
- feature_type: lexical
- main_hypothesis: This latent activates when the counselor's utterance contains the phrase pattern 'did you' or close variants. This pattern appears in 3/11 active samples but only 0/7 near-miss samples. Near-miss samples that look similar but lack this specific phrasing do not trigger the latent.
- positive_triggers: ["Presence of the phrase pattern 'did you' (found in 3/11 active samples, 0/7 near-miss samples)", 'Also involves: Closed yes/no question to client (active 2/11)']
- explicit_exclusions: ['Utterances enriched in near-miss-specific tokens: kind, name -- these words mark the non-triggering samples', "Utterances that lack the phrase 'did you' despite similar topic or tone"]
- possible_surface_confounds: ['Active samples are longer on average (12 vs 8 words), which may partly drive activation', 'Some near-miss activations (1.23) overlap with active range (0.57-2.81), suggesting the boundary is soft']
- **Classification: unclear**
- key_evidence: ['s001', 's002', 's011', 's012']
- failure_modes: ['Small sample size (20 utterances per task) limits the reliability of any pattern-based hypothesis', 'Activation overlap between active (min=0.57) and near-miss (max=1.23) makes the decision boundary uncertain', 'Length confound: active samples are substantially longer, making it hard to isolate semantic vs. structural triggers']

## QU - lowest confidence (0.3)
- packet_id: ctli_0140
- latent_idx: 21859
- short_name: counselor_inquiry_right_start
- feature_type: lexical-semantic
- main_hypothesis: This latent activates when the counselor's utterance contains language associated with open client inquiry — particularly words like 'right', 'start', 'stake' in a context where the counselor is soliciting the client's perspective. The trigger appears to be semantic rather than purely syntactic, requiring both a question-like pragmatic function and client-directed content.
- positive_triggers: ["Presence of inquiry-related vocabulary: 'right', 'start', 'stake'", 'Counselor utterances that solicit client self-disclosure or behavioral report', 'Questions or prompts that put the client in the role of information provider']
- explicit_exclusions: ['Statements, reflections, or summaries that use similar vocabulary without interrogative intent', 'Questions about logistics, scheduling, or administrative topics', 'Affirmations or backchannel responses']
- possible_surface_confounds: ['Lexical overlap between active and near-miss samples suggests the trigger is not purely lexical', 'Frequency of common counseling words may inflate activation regardless of speech act']
- **Classification: unclear**
- key_evidence: ['s001', 's002', 's003', 's007', 's006', 's018']
- failure_modes: ['Near-miss samples with overlapping vocabulary but different pragmatic function may confound', 'The explanation may be too broad; narrower trigger may require examining token-level activations']

## QUC - highest confidence (0.92)
- packet_id: ctli_0196
- latent_idx: 27857
- short_name: numeric_rating_scale_setup
- feature_type: counseling_function
- main_hypothesis: This latent activates when the counselor constructs an explicit numeric rating scale for the client, typically specifying a range (e.g., 'one to ten' or 'zero to ten') and anchoring both endpoints with descriptive labels (e.g., 'one being not confident at all and ten being the most confident'). The trigger is the act of setting up a bounded numeric assessment framework, not merely mentioning numbers or asking about confidence.
- positive_triggers: ["Explicit numeric range specification: 'on a scale of one to ten', 'on a scale of zero to ten', 'scale from to'", "Anchored endpoint descriptions: 'one being not confident at all and ten being the most confident', 'zero being not important and ten being super uber important'", 'Combined structure: numeric range + endpoint anchors + a question asking the client to place themselves on that scale', "Counselor asking the client to 'rate' something on a defined numeric continuum"]
- explicit_exclusions: ["Questions about frequency that mention numbers but do not define a scale (e.g., 'how many days a week?', 'how often do you drink?')", "References to numeric ratings in a client's response rather than the counselor's question setup (e.g., 'why a four and not a lower number')", "Questions about confidence or importance that do not specify a numeric scale (e.g., 'how confident would you say you are about making this change right now')", 'Non-numeric questions about feelings, activities, or preferences']
- possible_surface_confounds: ["The phrase 'on a scale' is a strong surface marker that co-occurs with activation but is part of the target pattern, not a confound", 'Longer sentences tend to score higher because scale-setup questions are inherently verbose -- sentence length is a correlate, not a cause', "The word 'confident' appears in many high-activation samples but also in some near-misses, so it is not the trigger by itself"]
- **Classification: counseling_function**
- key_evidence: ['s001', 's002', 's003', 's004', 's005', 's006', 's007', 's008', 's009', 's012', 's014']
- failure_modes: ["s010 'what kinds of physical activity do you enjoy' (ACTIVE_LOW 0.45) has no numeric component -- its low activation may be noise", "s011 'your friends are important to you' (ACTIVE_LOW 0.45) similarly lacks any scale element", "s007 'how confident would you say you are about making this change right now' is ACTIVE_MID (2.69) but has no explicit scale, suggesting partial activation from the word 'confident' alone", "s013 'why a four and not a lower number' references a rating but is NONACTIVE, confirming the latent tracks scale setup, not rating discussion"]

## QUC - lowest confidence (0.3)
- packet_id: ctli_0190
- latent_idx: 28816
- short_name: counselor_inquiry_like_know
- feature_type: lexical-semantic
- main_hypothesis: This latent activates when the counselor's utterance contains language associated with open client inquiry — particularly words like 'like', 'know', 'take' in a context where the counselor is soliciting the client's perspective. The trigger appears to be semantic rather than purely syntactic, requiring both a question-like pragmatic function and client-directed content.
- positive_triggers: ["Presence of inquiry-related vocabulary: 'like', 'know', 'take'", 'Counselor utterances that solicit client self-disclosure or behavioral report', 'Questions or prompts that put the client in the role of information provider']
- explicit_exclusions: ['Statements, reflections, or summaries that use similar vocabulary without interrogative intent', 'Questions about logistics, scheduling, or administrative topics', 'Affirmations or backchannel responses']
- possible_surface_confounds: ['Lexical overlap between active and near-miss samples suggests the trigger is not purely lexical', 'Frequency of common counseling words may inflate activation regardless of speech act']
- **Classification: unclear**
- key_evidence: ['s001', 's002', 's003', 's006', 's007', 's017']
- failure_modes: ['Near-miss samples with overlapping vocabulary but different pragmatic function may confound', 'The explanation may be too broad; narrower trigger may require examining token-level activations']

## QUO - highest confidence (0.5)
- packet_id: ctli_0143
- latent_idx: 9959
- short_name: counselor_modal_directive_question
- feature_type: syntactic-pragmatic
- main_hypothesis: This latent activates when the counselor uses a modal auxiliary (can/could/would/do) in a question directed at the client, prompting the client to consider or report on their behavior, thoughts, or intentions. The modal frames the question as an invitation rather than a direct command.
- positive_triggers: ["Questions beginning with or containing 'can you', 'could you', 'do you', 'would you'", 'Modal-framed inquiries about client behavior or plans', 'Counselor using polite modal questions to elicit client self-disclosure']
- explicit_exclusions: ['Imperative commands without modal framing', "Declarative statements about the client's situation", 'Social pleasantries or administrative questions']
- possible_surface_confounds: ['Modal verbs appear in many contexts; the trigger may require both modal AND client-directed content', "Polite social formulas ('could I speak to...') share modal structure but lack the reflective intent"]
- **Classification: unclear**
- key_evidence: ['s001', 's002', 's003', 's007', 's006', 's012']
- failure_modes: ["Non-modal open questions ('what do you eat?') may also activate, suggesting modals are not necessary", "Administrative modal questions ('could I speak to Faith?') are near-misses, indicating imperfect selectivity"]

## QUO - lowest confidence (0.3)
- packet_id: ctli_0173
- latent_idx: 3156
- short_name: counselor_inquiry_see_key
- feature_type: lexical-semantic
- main_hypothesis: This latent activates when the counselor's utterance contains language associated with open client inquiry — particularly words like 'see', 'key', 'right' in a context where the counselor is soliciting the client's perspective. The trigger appears to be semantic rather than purely syntactic, requiring both a question-like pragmatic function and client-directed content.
- positive_triggers: ["Presence of inquiry-related vocabulary: 'see', 'key', 'right'", 'Counselor utterances that solicit client self-disclosure or behavioral report', 'Questions or prompts that put the client in the role of information provider']
- explicit_exclusions: ['Statements, reflections, or summaries that use similar vocabulary without interrogative intent', 'Questions about logistics, scheduling, or administrative topics', 'Affirmations or backchannel responses']
- possible_surface_confounds: ['Lexical overlap between active and near-miss samples suggests the trigger is not purely lexical', 'Frequency of common counseling words may inflate activation regardless of speech act']
- **Classification: unclear**
- key_evidence: ['s001', 's002', 's003', 's008', 's007', 's013']
- failure_modes: ['Near-miss samples with overlapping vocabulary but different pragmatic function may confound', 'The explanation may be too broad; narrower trigger may require examining token-level activations']

## RE - highest confidence (0.75)
- packet_id: ctli_0009
- latent_idx: 26319
- short_name: meta_inquiry_framing
- feature_type: counseling_function
- main_hypothesis: This latent activates on counselor utterances that contain meta-communicative framing of inquiry -- phrases like 'so that I can better help/understand', 'let me step back and ask you a question', or 'I wanted to get your reaction'. The counselor is explicitly signaling their intent to understand or help before proceeding with a question or reflection.
- positive_triggers: ["Purpose clause: 'so that I can better help/understand'", "Meta-commentary on the counseling process: 'let me just step back and ask you a question'", "Explicit request for client reaction or perspective: 'I wanted to get your reaction to this news'"]
- explicit_exclusions: ['Direct questions without meta-communicative framing', 'Factual statements about medical or administrative matters', 'Short acknowledgments or greetings']
- possible_surface_confounds: ["The phrase 'so that I can' is a strong surface marker but the latent may be tracking the broader pragmatic function of meta-communicative framing", 'Multiple samples share nearly identical wording which may indicate template sensitivity']
- **Classification: counseling_function**
- key_evidence: ['s001', 's002', 's003', 's004', 's005', 's006', 's007', 's008', 's009']
- failure_modes: ["Near-miss s012 ('okay that's important so she knows that you're listening') uses 'so [someone] knows' but the purpose is about a third party, not the counselor's own understanding", "Near-miss s015 ('The case just got transferred to me so I'm not very familiar with your case') mentions counselor state but without the inquiry-framing function"]

## RE - lowest confidence (0.35)
- packet_id: ctli_0049
- latent_idx: 15396
- short_name: client_address_work
- feature_type: context_relation
- main_hypothesis: This latent activates on counselor utterances characterized by second person address. The pattern is present in 54% of active samples. The content tends to involve employment and work topics.
- positive_triggers: ['Utterances with second person address (present in 54% of active samples)', 'Presence of discourse markers: sounds like, so']
- explicit_exclusions: ['Utterances that lack the primary trigger pattern']
- possible_surface_confounds: ['Specific words enriched in active samples (type, youre, would) may be surface triggers']
- **Classification: context_relation**
- key_evidence: ['s001', 's002', 's004', 's017', 's015']
- failure_modes: ['Some near-miss samples show weak activation (4/7), suggesting the boundary is not sharp', 'The explanation is based on a small sample and may not generalize', 'The discriminative pattern is weak; the latent may be multi-modal or noisy']

## REC - highest confidence (0.85)
- packet_id: ctli_0097
- latent_idx: 26968
- short_name: phrase-that-you
- feature_type: lexical
- main_hypothesis: This latent activates when the counselor's utterance contains the phrase pattern 'that you' or close variants. This pattern appears in 4/11 active samples but only 0/7 near-miss samples. Near-miss samples that look similar but lack this specific phrasing do not trigger the latent.
- positive_triggers: ["Presence of the phrase pattern 'that you' (found in 4/11 active samples, 0/7 near-miss samples)", "Related pattern 'to start' (active 3/11, near-miss 0/7)", 'Also involves: Closed yes/no question to client (active 3/11)']
- explicit_exclusions: ['Utterances enriched in near-miss-specific tokens: good, control -- these words mark the non-triggering samples', "Utterances that lack the phrase 'that you' despite similar topic or tone"]
- possible_surface_confounds: ['Active samples are longer on average (25 vs 10 words), which may partly drive activation', 'Content words like feeling, start, now may co-occur with the true trigger rather than being the trigger itself']
- **Classification: unclear**
- key_evidence: ['s001', 's002', 's011', 's012']
- failure_modes: ['Small sample size (20 utterances per task) limits the reliability of any pattern-based hypothesis', 'Length confound: active samples are substantially longer, making it hard to isolate semantic vs. structural triggers']

## REC - lowest confidence (0.515)
- packet_id: ctli_0077
- latent_idx: 15068
- short_name: behavioral_directive
- feature_type: pragmatic-illocutionary
- main_hypothesis: This latent activates on counselor utterances that provide direction, suggestions, or guidance about actions the client should take. Active samples contain action verbs, imperative-like structures, or explicit suggestions for behavior change. The pattern involves the counselor moving from exploration to recommending specific actions. Nonactive samples may discuss similar topics but in a more exploratory or reflective mode rather than directive.
- positive_triggers: ['action-oriented language: try, make, stop, start, cut, reduce, stay', 'suggestions or recommendations for specific behavior', "guidance phrases: 'you could', 'make sure', 'you might want'", 'directing client toward concrete next steps']
- explicit_exclusions: ['open-ended exploratory questions', 'reflective summaries of client statements', 'purely empathic or validating statements', 'neutral information exchange']
- possible_surface_confounds: ['imperative vs suggestive mood distinction', 'action verb presence as surface feature']
- **Classification: unclear**
- key_evidence: ['s005', 's001', 's009', 's012', 's015']
- failure_modes: ['Some directives are embedded in questions, blurring the boundary', 'Near-miss samples may contain indirect guidance']

## RES - highest confidence (0.784)
- packet_id: ctli_0067
- latent_idx: 29701
- short_name: composite_hypothetical_scenario_probe
- feature_type: syntactic-semantic
- main_hypothesis: This latent activates on counselor utterances that construct hypothetical scenarios or conditional situations for the client to consider. Active samples use 'if', 'would', 'could', 'what if' to frame imagined situations or explore potential outcomes. The pattern involves the counselor creating a thought experiment about alternative choices, potential consequences, or imagined responses. Nonactive samples may discuss real situations but lack the hypothetical framing.
- positive_triggers: ["conditional constructions: 'if you...', 'would you...', 'could you...'", 'hypothetical scenario framing or thought experiments', 'invitations to imagine alternatives or consequences', 'exploratory what-if questions about behavior change', '[secondary] action-oriented directives (try, make sure, you should)']
- explicit_exclusions: ['statements about actual past events', 'unconditional directives or suggestions', 'simple factual questions', 'empathic statements about current reality']
- possible_surface_confounds: ['modal verb presence (would, could) as surface feature', 'sentence complexity correlated with hypothetical structure']
- **Classification: unclear**
- key_evidence: ['s002', 's005', 's007', 's016', 's015']
- failure_modes: ['Some conditionals may be polite hedging rather than genuine exploration', 'Active samples may cluster by topic (substance use scenarios)']

## RES - lowest confidence (0.35)
- packet_id: ctli_0061
- latent_idx: 23077
- short_name: client_address_work
- feature_type: context_relation
- main_hypothesis: This latent activates on counselor utterances characterized by second person address. The pattern is present in 45% of active samples. The content tends to involve emotional content.
- positive_triggers: ['Utterances with second person address (present in 45% of active samples)']
- explicit_exclusions: ['Utterances that lack the primary trigger pattern']
- possible_surface_confounds: ['Specific words enriched in active samples (your, right, its) may be surface triggers']
- **Classification: context_relation**
- key_evidence: ['s001', 's002', 's003', 's013', 's016']
- failure_modes: ['Some near-miss samples show weak activation (4/7), suggesting the boundary is not sharp', 'The explanation is based on a small sample and may not generalize', 'The discriminative pattern is weak; the latent may be multi-modal or noisy']

## SU - highest confidence (0.8)
- packet_id: ctli_0274
- latent_idx: 23464
- short_name: positive_affirmation_responses
- feature_type: counseling_function
- main_hypothesis: This latent activates on counselor utterances that serve as positive affirmation responses to client statements -- 'that's good to hear', 'that sounds great', 'that's wonderful', 'that's a great question'. The trigger is the response-frame structure: demonstrative + positive evaluative predicate.
- positive_triggers: ["'That's + positive adjective' response frames ('that's good to hear', 'that's wonderful', 'that's a great question')", "'That sounds + positive' evaluations ('that sounds great')", "Positive response to client disclosure ('that's good to hear that your friends are looking after you')", "Session-opening positive greeting ('hey Monica how are you doing today')"]
- explicit_exclusions: ['Questions or prompts without affirmation', 'Reflective listening without positive evaluation', 'Factual statements without evaluative framing', 'Negative or neutral responses']
- possible_surface_confounds: ["Near-miss s012 ('it's wonderful you can't have the strategies') uses 'wonderful' but in a different syntactic frame, showing different activation", 'Sample size limits ability to separate surface from functional triggers']
- **Classification: counseling_function**
- key_evidence: ['s003', 's001', 's015', 's014', 's008', 's010']
- failure_modes: ['Low-activation samples (s010, s011) show weak triggering, suggesting the pattern threshold is not sharp', 'Near-miss samples with nonzero activation (s015, s014, s012, s013) suggest partial pattern overlap that the hypothesis does not fully explain', 'Cannot determine from text alone whether the latent tracks surface tokens or deeper semantic properties', 'Sample selection may not cover the full activation landscape of this latent']

## SU - lowest confidence (0.2)
- packet_id: ctli_0272
- latent_idx: 16045
- short_name: latent_16045_mixed
- feature_type: unclear
- main_hypothesis: This latent does not cleanly map to a single interpretable counseling function. Active samples include various behaviors, but none strongly distinguishes them from inactive samples.
- positive_triggers: ['no single dominant trigger']
- explicit_exclusions: []
- possible_surface_confounds: ["Near-miss 'okay have you know participant I can exercise over the floor...' shares features but falls below threshold -- possible activation boundary effect", "Near-miss 'is there anything else that I can help you with today?...' shares features but falls below threshold -- possible activation boundary effect", "Near-miss 'that's terrible John asbestos in your first day coming back ...' shares features but falls below threshold -- possible activation boundary effect"]
- **Classification: unclear**
- key_evidence: ["Highest activation sample: 'I have a little handout here so we can talk about some like portion sizes'", "Near-miss boundary: 'okay have you know participant I can exercise over the floor'"]
- failure_modes: ['Analysis based on limited sample set; pattern may not generalize across all counseling contexts', "Surface text features may not capture the true activation mechanism in the model's internal representation", 'Near-miss boundary may shift with different sample selection']

