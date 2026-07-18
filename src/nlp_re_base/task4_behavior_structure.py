"""Task 4 behavior structure built from the existing independent feature-card codes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .contrastive_evidence_pack import read_jsonl, write_json


LABEL_ORDER = ("RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF")
LEAF_LABELS = {"RES", "REC", "QUO", "QUC", "GI", "SU", "AF"}
EVIDENCE_CLASSES = {
    "behavioral_function",
    "linguistic_structure",
    "affective_content",
    "topic",
    "surface_artifact",
    "unclear_or_mixed",
}


def _component(
    key: str,
    name_zh: str,
    name_en: str,
    definition: str,
    members: list[int],
    confidence: str,
    boundary: str,
) -> dict[str, Any]:
    return {
        "key": key,
        "name_zh": name_zh,
        "name_en": name_en,
        "definition": definition,
        "members": members,
        "confidence": confidence,
        "boundary": boundary,
    }


# This is a Task 4 grouping layer, not a rewrite of the independent card codes.
# Every behavioral_function association is either in one component or UNASSIGNED.
COMPONENTS: dict[str, list[dict[str, Any]]] = {
    "RE": [
        _component("reflective_reformulation", "反映式重构与总结", "reflective reformulation and summary", "重述、概括或核对对方已经表达的经历和立场。", [30224, 3993, 31133, 26800, 1211, 16292, 28269, 29874, 3416, 2167, 2089, 4625], "高", "缺少前一句 client utterance，反映关系只能作为候选。"),
        _component("ambivalence_reflection", "矛盾与双面反映", "ambivalence and two-sided reflection", "同时呈现冲突立场、利弊或一方面/另一方面的张力。", [10181, 20436, 12852], "高", "部分卡片同时含建议成分，不能视为纯粹反映。"),
        _component("emotion_focused_reflection", "情绪与压力反映", "emotion-focused reflection", "重述压力、挫折、羞愧或不堪重负等主观体验。", [22558, 26968], "中", "当前话语可显示情绪概括，但无法确认是否准确回应前文。"),
        _component("directive_risk_guidance", "直接建议与风险反馈", "directive advice and risk feedback", "直接提出行动建议、警示后果或强调健康风险。", [1516, 7054, 11405, 29845, 20564], "中", "该成分与 RE 标签并不等价，可能反映共享或混合特征。"),
        _component("confirmatory_exploration", "确认与探索性追问", "confirmatory and exploratory prompting", "通过确认式或开放式追问促使对方说明计划、原因或立场。", [29759, 15504], "中", "卡片内部混合确认、提问和反映，不宜进一步细分。"),
    ],
    "RES": [
        _component("reflection_clarification", "反映与理解核对", "reflection and understanding check", "重述对方表述并通过确认问题核对理解。", [28269, 29759, 8468], "中", "缺少 client 前文，无法确认其是否构成简单反映。"),
        _component("directive_risk_guidance", "建议与风险提示", "advice and risk warning", "提出行为建议、健康警示或直接评价可能后果。", [1516, 32727, 27859], "中", "与 RES 的关联可能来自混合卡片，而非反映功能本身。"),
    ],
    "REC": [
        _component("reflective_reformulation", "反映式重构与总结", "reflective reformulation and summary", "对经历、陈述或会谈内容进行重述、概括和理解核对。", [31133, 26800, 30224, 3993, 16292, 29874, 1211, 2089, 1455, 31915, 3416], "高", "没有前文时不能据此确认重构增加了多少语义。"),
        _component("ambivalence_reflection", "矛盾与双面反映", "ambivalence and two-sided reflection", "组织并呈现冲突目标、利弊或两面性。", [20436, 10181, 12852], "高", "部分证据同时包含引导改变的表达。"),
        _component("emotion_experience_reflection", "情绪与经历反映", "emotion and experience reflection", "概括情绪状态、困境或重要经历并表达理解。", [22558, 26968, 32596, 16256], "中", "缺少前文，准确性和复杂程度均待人工核对。"),
        _component("directive_plan_feedback", "建议、计划与风险反馈", "advice, planning, and risk feedback", "提供建议、归纳行动计划或强调行为风险。", [11405, 9993, 7054, 31867, 8226], "中", "该组是 REC 关联中的非典型行为功能，不应写成 REC 定义。"),
    ],
    "QU": [
        _component("open_exploration", "开放式经历与观点探索", "open exploration of experience and perspective", "邀请对方展开经历、感受、观点或个人情境。", [9959, 26485, 21125, 24744, 26144, 24761, 18730, 32508], "高", "部分卡片也包含事实收集，边界以是否鼓励展开作区分。"),
        _component("change_evocation", "改变动机与障碍唤起", "change-motivation and barrier evocation", "询问改变的理由、利弊、障碍和可能行动，以促发自我反思。", [27061, 18310, 21203], "高", "当前证据支持提问功能，不证明产生了实际改变言语。"),
        _component("focused_assessment", "聚焦式事实与行为评估", "focused factual and behavioral assessment", "收集病史、频率、数量、行为状态或其他相对明确的信息。", [13430, 29590, 21935, 24943, 22358, 21859], "高", "该成分混合开放和封闭问法，不能据此判断 QUO/QUC。"),
    ],
    "QUO": [
        _component("open_exploration", "开放式经历与观点探索", "open exploration of experience and perspective", "使用 what/how/tell me 等方式邀请对方展开经历、观点、目标和困难。", [9959, 26485, 24744, 24761, 18730, 21125, 19840, 32508, 26144, 19038, 3887, 19470, 3156, 14247], "高", "开放形式并不自动等于治疗性功能。"),
        _component("change_evocation", "改变动机与差距探索", "change-motivation and discrepancy exploration", "探索改变的理由、障碍、两面性以及当前状态与目标的差距。", [18310, 27061, 21203, 18990], "高", "部分样例是一般信息收集，成分边界仍需人工确认。"),
        _component("focused_assessment", "聚焦式数量与病史询问", "focused quantity and history assessment", "询问数量、频率、时长、病史或具体行为状态。", [13430, 29590], "中", "该组与 QUO 标签口径不完全一致，作为标签内异质证据保留。"),
    ],
    "QUC": [
        _component("structured_self_rating", "结构化自我评分", "structured self-rating", "要求在数值量表上报告重要性、信心、准备度或其他主观状态。", [27857, 7037, 664, 12902, 5178, 6639, 21296, 2550, 29247, 29931], "高", "量表形式高度稳定，但仍属于问句实现与评估功能的结合。"),
        _component("focused_health_assessment", "聚焦式健康与病史评估", "focused health and history assessment", "通过较明确的问题收集健康、病史、物质使用、频率或当前行为信息。", [21935, 13430, 14014, 20869, 8969, 14003, 28816, 24943, 17507, 29590, 12887, 20549, 31422], "高", "部分问题可以产生长回答，不能只凭形式断言均为封闭问句。"),
        _component("collaborative_option_question", "许可与选项协商", "permission and option negotiation", "征求讨论许可、询问方案接受度或让对方在选项间表达立场。", [18646, 4998, 8089, 32273], "中", "卡片内部也包含一般探索问题。"),
        _component("general_probing", "一般性澄清追问", "general clarification probing", "围绕理解、时长、感受或想法进行进一步澄清。", [22358, 10916, 26485], "中", "该组功能较宽，保留为候选而非精细行为机制。"),
    ],
    "GI": [
        _component("medication_information_instruction", "用药信息与执行说明", "medication information and administration instruction", "解释药物用途、剂量、频率、副作用或服用要求。", [8166, 16131, 19483], "高", "部分卡片同时包含建议和风险信息。"),
        _component("health_recommendation", "健康建议与风险指导", "health recommendation and risk guidance", "根据健康风险提出行为、治疗或生活方式建议。", [27515, 10264], "中", "F10264 同时包含提问和目标设定，功能并非纯信息提供。"),
    ],
    "SU": [
        _component("empathic_validation", "共情理解与困难确认", "empathic understanding and validation", "表达理解、承认困难或确认对方处境。", [24760, 29825, 4756, 7389], "高", "I understand 等模板可能只体现话语形式，真实共情效果仍需语境。"),
        _component("help_offering", "提供帮助与支持可用性", "offering help and support availability", "明确表达愿意帮助或询问对方需要何种帮助。", [9720, 4512, 16190, 19935, 28603], "高", "help 词汇高度重复，需把行为功能与词汇模板同时报告。"),
        _component("directive_risk_guidance", "直接建议与风险评价", "directive advice and risk evaluation", "直接要求改变、评价当前行为或指出健康风险。", [9578, 26642, 20743], "中", "该成分并非支持行为的唯一形式，且卡片内容较混合。"),
    ],
    "AF": [
        _component("positive_evaluation", "行动、目标与个人优势肯定", "affirmation of actions, goals, and strengths", "正向评价对方的行动、计划、努力或个人优势。", [30870, 6676, 28469, 22411, 24167], "高", "正向词汇可能是表层线索，不能仅凭形容词认定完整行为功能。"),
        _component("affirmative_reflection", "肯定性反映", "affirmative reflection", "在重述对方计划或状态时加入鼓励和正向强化。", [7143, 25532], "中", "部分证据更接近一般反映或固定 sounds like 模板。"),
        _component("gratitude_rapport", "感谢与关系维持", "gratitude and rapport maintenance", "通过感谢、赞赏参与或礼貌收尾维持互动关系。", [9869, 24206, 11511, 31149, 19654], "高", "感谢是社会互动功能，不应与对个人优势的肯定合并。"),
    ],
}

UNASSIGNED: dict[str, list[int]] = {
    "RE": [31363, 23242, 19005, 13169, 26681],
    "RES": [16320],
    "REC": [29759, 19005, 23242, 13169, 3673, 30091],
    "QU": [664, 10916],
    "QUO": [664, 10916],
    "QUC": [],
    "GI": [13751],
    "SU": [11948, 9109, 29856, 20778, 5732, 26263],
    "AF": [18492, 5320],
}


def _mode(
    evidence_class: str,
    key: str,
    name_zh: str,
    name_en: str,
    definition: str,
    members: list[int],
    confidence: str,
    boundary: str,
) -> dict[str, Any]:
    row = _component(key, name_zh, name_en, definition, members, confidence, boundary)
    row["evidence_class"] = evidence_class
    return row


# Bottom-up organization of the non-behavior card classes already present in
# validated_cards.jsonl. These modes are part of the label representation, not
# automatically discarded as noise.
NONBEHAVIOR_COMPONENTS: dict[str, list[dict[str, Any]]] = {
    "RE": [
        _mode("linguistic_structure", "reflection_surface_template", "反映式表层模板", "reflection surface template", "以 it/sounds/seems/looks like 等固定框架引出对对方状态的表述。", [21800, 8034, 29190, 4018, 23670, 28688], "高", "模板重复不等于已经识别反映行为，可能主要编码词法和句法。"),
        _mode("linguistic_structure", "so_discourse_launch", "so 引导的话语启动结构", "so-prefaced discourse launch", "使用 so、okay so 或相近标记启动问题、总结或回应。", [19435, 31930, 5663, 14875, 9537, 17315], "高", "同一标记可承载提问、过渡和反映等不同功能。"),
        _mode("linguistic_structure", "second_person_evaluation", "第二人称评价与经历陈述", "second-person evaluation and experience statement", "以 you/you're 指向对方的经历、健康状态或评价。", [17861, 20808, 4662, 13966], "中", "第二人称形式过于常见，必须与行为功能分开解释。"),
        _mode("linguistic_structure", "it_evaluative_construction", "it 引导的评价结构", "it-prefaced evaluative construction", "以 it/it's/that is 等结构组织评价、判断或状态描述。", [15068, 111, 30798, 15396], "中", "该结构覆盖面广，可能是通用英语句法而非标签特异机制。"),
        _mode("unclear_or_mixed", "mixed_counseling_discourse", "混合型咨询话语", "mixed counseling discourse", "卡片同时包含问题、反映、建议或一般会谈标记，无法形成单一解释。", [2995, 3805, 13312], "低", "保留为稳定核心中的未解析结构，不据此命名行为概念。"),
    ],
    "RES": [
        _mode("linguistic_structure", "second_person_address", "第二人称指向结构", "second-person address structure", "以 you/you're 或 so you 指向对方经历、状态或行为。", [20808, 13966, 19435], "中", "形式重复不能确认简单反映，且缺少 client 前文。"),
        _mode("unclear_or_mixed", "mixed_healthcare_discourse", "混合型医疗会谈", "mixed healthcare discourse", "医疗或咨询会谈中的多种话语功能混合出现。", [11435, 23077, 32696], "低", "无法确定共享成分是行为、主题还是语体。"),
    ],
    "REC": [
        _mode("linguistic_structure", "reflection_surface_template", "反映式表层模板", "reflection surface template", "使用 it/that sounds like、seems like、okay so 等模板引出解释或重述。", [15068, 21800, 23670, 29190, 111, 28688, 8034, 4018], "高", "该结果首先说明语言实现形式集中，不单独证明复杂反映。"),
        _mode("linguistic_structure", "so_discourse_launch", "so 引导的话语启动结构", "so-prefaced discourse launch", "使用 so 或相近话语标记启动问题、总结和回应。", [19435, 14875, 5663, 31930], "高", "so 是高频会话标记，标签特异性有限。"),
        _mode("unclear_or_mixed", "mixed_counseling_discourse", "混合型咨询话语", "mixed counseling discourse", "包含反映、建议、问题和医疗会谈等多种模式，无法稳定分离。", [3805, 31585, 2995], "低", "保留为未解析稳定特征，不强行归入反映层级。"),
    ],
    "QU": [
        _mode("linguistic_structure", "wh_question_form", "WH 问句形式", "WH-question form", "以 what/how 等疑问词构成信息请求。", [11660, 23183, 8658], "高", "疑问形式不能自动区分开放和封闭功能。"),
        _mode("linguistic_structure", "do_you_question_form", "do you 助动词问句", "do-you auxiliary question", "使用 do you 及相近助动词前置结构发问。", [12340, 16969], "高", "主要是句法形式证据。"),
        _mode("linguistic_structure", "question_disfluency", "问句重复与不流利", "question repetition and disfluency", "疑问词重复、重启或犹豫标记共同出现。", [12544, 3459], "中", "可能反映口语转录或说话风格，而非咨询功能。"),
    ],
    "QUO": [
        _mode("linguistic_structure", "what_question_template", "what 问句模板", "what-question template", "以 what、what do you、what do you think 等模板邀请回答。", [23183, 14833, 8191, 6129, 12337, 25210], "高", "模板本身不足以证明回答空间一定开放。"),
        _mode("linguistic_structure", "wh_question_disfluency", "WH 问句与不流利结构", "WH-question and disfluency structure", "WH 问句中出现 would、重复、重启或犹豫。", [11660, 3459, 12544], "中", "其中一部分可能是转录风格。"),
    ],
    "QUC": [
        _mode("linguistic_structure", "auxiliary_yes_no_form", "助动词前置的是非问句", "auxiliary-fronted yes-no question", "以 do/are/have 等助动词前置形成较明确的是非或筛查问题。", [12555, 12340, 20463, 16969, 2504, 12676, 19364], "高", "句法形式与实际回答长度并非一一对应。"),
        _mode("linguistic_structure", "quantitative_how_form", "时长与数量问句", "duration and quantity question form", "使用 how long/how much 等形式询问时长或数量。", [29947, 8861], "高", "编码的是量化问句实现，不是独立行为机制。"),
        _mode("linguistic_structure", "scale_question_template", "量表问句模板", "scale-question template", "重复出现量表、数值端点和评分提示结构。", [18714, 32565, 2416], "高", "与行为层的结构化自评相呼应，但此处证据层是表层模板。"),
        _mode("unclear_or_mixed", "mixed_question_formats", "混合问句格式", "mixed question formats", "多种问句与陈述混合，无法稳定归为单一形式。", [9827, 736], "低", "不据此判断 QUC 功能边界。"),
    ],
    "GI": [
        _mode("linguistic_structure", "healthcare_discourse_markers", "会谈过渡与确认标记", "dialogue transition and confirmation markers", "重复出现 actually、yeah、it is 等过渡和确认标记。", [2055, 23723, 5828], "中", "F5828 含非医疗样例，说明该模式更可能是通用话语结构。"),
        _mode("topic", "medication_treatment_topic", "药物与治疗主题", "medication and treatment topic", "围绕药物名称、处方、剂量、治疗和咨询场景。", [16345, 8294, 21634, 24876, 1713, 17556], "高", "主题集中不等于模型表征了信息给予功能。"),
        _mode("topic", "health_risk_vitals_topic", "健康风险与指标主题", "health risk and clinical metrics topic", "围绕疾病风险、血压、胆固醇、数量指标和健康后果。", [26879, 26236, 16515, 7148, 20329], "高", "可能同时包含建议、事实和量化语言。"),
        _mode("topic", "diet_weight_activity_topic", "饮食、体重与活动主题", "diet, weight, and activity topic", "围绕饮食、体重、锻炼和日常健康行为。", [9893, 18490, 30517, 6651], "高", "主题特征不能直接解释互动功能。"),
        _mode("topic", "general_health_behavior_topic", "一般健康行为主题", "general health-behavior topic", "多种健康行为和医疗内容混合出现。", [17827, 17685], "中", "主题范围较宽，作为剩余主题成分保留。"),
    ],
    "SU": [
        _mode("linguistic_structure", "first_person_supportive_form", "第一人称支持性表达", "first-person supportive construction", "使用 I、I've、I can、I would 等第一人称结构表达立场或支持。", [19359, 8760, 16045], "中", "第一人称结构可承载多种功能。"),
        _mode("linguistic_structure", "evaluative_demonstrative_form", "指示词评价结构", "demonstrative evaluative construction", "使用 this/that + 系词或形容词形成评价性陈述。", [22730, 24856], "中", "结构证据不能单独确认支持行为。"),
        _mode("affective_content", "difficulty_negative_affect", "困难与负向体验内容", "difficulty and negative-experience content", "表达困难、压力、艰难或不愉快体验。", [16736, 11872], "高", "情感内容可能支持共情，但本身不是共情行为。"),
        _mode("unclear_or_mixed", "mixed_supportive_discourse", "混合型支持话语", "mixed supportive discourse", "多种咨询、支持或一般会谈功能混合，无法稳定分离。", [5366, 2995], "低", "保留为不清晰成分，不推断统一支持机制。"),
    ],
    "AF": [
        _mode("linguistic_structure", "gratitude_formula", "感谢公式结构", "gratitude formula", "重复出现 thank you、thanks for coming 和 thank you for + 动名词等结构。", [17793, 28724, 2434], "高", "可能是固定礼貌模板，需与行为层感谢功能并列报告。"),
        _mode("linguistic_structure", "positive_evaluative_construction", "评价性句式（以正向为主）", "evaluative construction, predominantly positive", "使用 that's/that is/good/great 等系词和形容词表达正向或负向评价，其中正向用法占主导。", [32764, 24856, 16965], "高", "F32764/F24856 也覆盖 upsetting、hard 等负向评价，不能视为纯正向行为。"),
        _mode("affective_content", "positive_evaluation_affect", "正向评价与赞扬内容", "positive evaluation and praise content", "稳定出现 good、great、awesome 等正向评价和赞扬内容。", [23464, 22475, 29908, 453, 1871, 32320], "高", "高一致性可能来自词汇方向，仍需与行为肯定区分。"),
    ],
}

NONBEHAVIOR_UNASSIGNED: dict[str, list[int]] = {
    "RE": [26319, 26869],
    "RES": [26319, 29701],
    "REC": [17861, 26319, 26869],
    "QU": [],
    "QUO": [12340],
    "QUC": [11660],
    "GI": [],
    "SU": [28795, 30223, 7382, 23464],
    "AF": [26938],
}

CROSS_FAMILIES = [
    {"name": "反映式重构与总结", "refs": ["RE:reflective_reformulation", "RES:reflection_clarification", "REC:reflective_reformulation"], "basis": "均以重述、概括或核对对方表述为核心。", "differences": "RES 更偏简短核对；REC/RE 包含更丰富的总结候选。"},
    {"name": "矛盾与双面反映", "refs": ["RE:ambivalence_reflection", "REC:ambivalence_reflection"], "basis": "均组织冲突目标、利弊或一方面/另一方面结构。", "differences": "现有数据无 client 前文，不能比较反映准确性与复杂程度。"},
    {"name": "直接建议与风险指导", "refs": ["RE:directive_risk_guidance", "RES:directive_risk_guidance", "REC:directive_plan_feedback", "GI:health_recommendation", "SU:directive_risk_guidance"], "basis": "均直接提供建议、行动方向或风险反馈。", "differences": "GI 更偏专业健康建议；SU、RE 家族中的此类证据更可能代表标签内异质或共享 latent。"},
    {"name": "开放式探索", "refs": ["QU:open_exploration", "QUO:open_exploration", "QUC:general_probing"], "basis": "均邀请说明经历、观点、感受或原因。", "differences": "QUC 组更偏澄清，QUO 组的展开性证据更集中。"},
    {"name": "改变动机与障碍唤起", "refs": ["QU:change_evocation", "QUO:change_evocation"], "basis": "均探索改变理由、障碍、差距或两面性。", "differences": "QU 是父标签汇总，QUO 是叶级候选。"},
    {"name": "聚焦式事实评估", "refs": ["QU:focused_assessment", "QUO:focused_assessment", "QUC:focused_health_assessment"], "basis": "均收集病史、数量、频率或明确行为状态。", "differences": "该功能在 QUO 中属于标签内异质证据，在 QUC 中支持最集中。"},
]

NONBEHAVIOR_CROSS_FAMILIES = [
    {"name": "反映式表层模板", "evidence_class": "linguistic_structure", "refs": ["RE:reflection_surface_template", "REC:reflection_surface_template"], "basis": "两个标签均集中出现 sounds/seems/looks like 与 okay so 等模板。", "differences": "该共享首先是语言实现证据，不证明两标签具有同一行为机制。"},
    {"name": "so 引导的话语启动", "evidence_class": "linguistic_structure", "refs": ["RE:so_discourse_launch", "REC:so_discourse_launch"], "basis": "多个 latent 重复编码 so/okay so 引导的问题、总结或回应。", "differences": "so 是通用会话标记，标签特异性弱。"},
    {"name": "第二人称指向结构", "evidence_class": "linguistic_structure", "refs": ["RE:second_person_evaluation", "RES:second_person_address"], "basis": "均以 you/you're 指向对方经历、状态或行为。", "differences": "RE 组包含更多评价结构，RES 组规模较小。"},
    {"name": "未解析的混合咨询话语", "evidence_class": "unclear_or_mixed", "refs": ["RE:mixed_counseling_discourse", "RES:mixed_healthcare_discourse", "REC:mixed_counseling_discourse"], "basis": "三个标签均有多个 stable-core 卡片无法从咨询/医疗会谈语体中分离单一模式。", "differences": "这是共享的不确定性证据，不是共享行为成分。"},
    {"name": "疑问句表层实现", "evidence_class": "linguistic_structure", "refs": ["QU:wh_question_form", "QUO:what_question_template", "QUC:auxiliary_yes_no_form"], "basis": "三个标签均由明确疑问句模板构成重要表征成分。", "differences": "QUO 更偏 what/WH 模板，QUC 更偏助动词前置的是非问法。"},
    {"name": "问句重复与不流利", "evidence_class": "linguistic_structure", "refs": ["QU:question_disfluency", "QUO:wh_question_disfluency"], "basis": "两个标签均出现疑问词重复、重启和犹豫。", "differences": "该模式可能来自口语或转录过程，而非标签功能。"},
]

LABEL_INTERPRETATIONS = {
    "RE": "以反映、总结和矛盾呈现为主要行为候选，同时强烈依赖 sounds/seems like、so 和第二人称评价等语言模板；因此其结构兼有功能层与表层实现层。",
    "RES": "行为候选较少，主要是理解核对及少量建议/风险提示；语言层以第二人称指向为主，同时存在较多无法解析的医疗会谈混合卡片。",
    "REC": "反映式重构、矛盾反映和情绪经历反映构成主要行为候选，且与反映模板和 so 话语启动高度并行；缺少 client 前文使复杂反映解释仍受限。",
    "QU": "同时包含开放探索、改变唤起和事实评估，并由 WH、do-you 与口语不流利等问句形式实现，表现为功能与句法共同组织。",
    "QUO": "开放经历探索和改变动机探索最集中，表层上主要由 what/WH 模板实现；少量聚焦询问说明该标签内部并非纯粹开放问句。",
    "QUC": "结构化评分和聚焦健康评估最突出，语言层对应助动词前置、数量问句和量表模板，功能与表面形式之间具有较强一致性。",
    "GI": "除用药说明和健康建议行为外，17/26 的关联由药物、风险指标、饮食体重等主题卡片构成，说明该标签在当前模型中很大程度按领域内容组织。",
    "SU": "共情确认和帮助提供是主要行为候选，同时编码第一人称支持句式与困难/负向体验；正负情感评价单例方向不一致，因此保留为孤立模式而不合并。",
    "AF": "行动/优势肯定、肯定性反映和感谢行为与正向评价情感、评价句式及感谢公式共同出现，说明 AF 兼具行为功能与高度可预测的正向词汇模板。",
}


def _scope(labels: set[str]) -> str:
    if labels.issubset({"RE", "RES", "REC"}) or labels.issubset({"QU", "QUO", "QUC"}):
        return "sibling_leaf" if labels.issubset(LEAF_LABELS) else "parent_child"
    return "cross_family"


def _lookup(pack: dict[str, Any]) -> dict[str, str]:
    return {str(row["id"]): str(row["text"]).strip() for row in pack["samples_for_model"]}


def _representative(card: dict[str, Any], pack: dict[str, Any], limit: int = 2) -> str:
    lookup = _lookup(pack)
    return " || ".join(
        f"{sample_id}: {lookup.get(str(sample_id), '')}"
        for sample_id in card.get("representative_evidence_ids", [])[:limit]
    )


def validate_grouping_definition(*, cards: dict[int, dict[str, Any]], stable: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    expected_total = 0
    expected_nonbehavior_total = 0
    for label in LABEL_ORDER:
        expected = set(
            stable.loc[
                stable["label"].eq(label)
                & stable["latent_idx"].map(lambda value: cards[int(value)]["explanation_type"] == "behavioral_function"),
                "latent_idx",
            ].astype(int)
        )
        assigned: list[int] = []
        for component in COMPONENTS[label]:
            members = [int(value) for value in component["members"]]
            if len(members) < 2:
                errors.append(f"{label}:{component['key']} has fewer than two members")
            assigned.extend(members)
        assigned.extend(UNASSIGNED[label])
        if len(assigned) != len(set(assigned)):
            errors.append(f"{label} contains duplicate assignments")
        if set(assigned) != expected:
            errors.append(f"{label} coverage mismatch missing={sorted(expected-set(assigned))} extra={sorted(set(assigned)-expected)}")
        expected_total += len(expected)
        expected_nonbehavior = set(
            stable.loc[
                stable["label"].eq(label)
                & stable["latent_idx"].map(lambda value: cards[int(value)]["explanation_type"] != "behavioral_function"),
                "latent_idx",
            ].astype(int)
        )
        nonbehavior_assigned: list[int] = []
        for component in NONBEHAVIOR_COMPONENTS[label]:
            members = [int(value) for value in component["members"]]
            if component["evidence_class"] not in EVIDENCE_CLASSES - {"behavioral_function"}:
                errors.append(f"{label}:{component['key']} has invalid evidence class")
            if len(members) < 2:
                errors.append(f"{label}:{component['key']} has fewer than two members")
            for latent_idx in members:
                if cards[latent_idx]["explanation_type"] != component["evidence_class"]:
                    errors.append(f"{label}:{component['key']} class mismatch for F{latent_idx}")
            nonbehavior_assigned.extend(members)
        nonbehavior_assigned.extend(NONBEHAVIOR_UNASSIGNED[label])
        if len(nonbehavior_assigned) != len(set(nonbehavior_assigned)):
            errors.append(f"{label} contains duplicate nonbehavior assignments")
        if set(nonbehavior_assigned) != expected_nonbehavior:
            errors.append(
                f"{label} nonbehavior coverage mismatch "
                f"missing={sorted(expected_nonbehavior-set(nonbehavior_assigned))} "
                f"extra={sorted(set(nonbehavior_assigned)-expected_nonbehavior)}"
            )
        expected_nonbehavior_total += len(expected_nonbehavior)
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "labels": len(LABEL_ORDER),
        "behavior_associations": expected_total,
        "nonbehavior_associations": expected_nonbehavior_total,
        "status": "pass",
    }


def build_task4(*, package_dir: Path, output_dir: Path) -> dict[str, Any]:
    frozen = package_dir / "frozen_all_cards"
    cards = {int(row["latent_idx"]): row for row in read_jsonl(frozen / "validated_cards.jsonl")}
    packs = {int(row["latent_idx"]): row for row in read_jsonl(frozen / "sentence_packs.jsonl")}
    stable = pd.read_csv(frozen / "stable_topk_latent_set.csv")
    stable = stable[stable["stable_set_role"].astype(str).eq("stable_core")].copy()
    stable["label"] = stable["label"].astype(str).str.upper()
    stable["latent_idx"] = stable["latent_idx"].astype(int)
    stable["label_order"] = stable["label"].map({label: index for index, label in enumerate(LABEL_ORDER)})
    stable = stable.sort_values(["label_order", "rank_within_label", "latent_idx"])
    validation = validate_grouping_definition(cards=cards, stable=stable)

    assignment: dict[tuple[str, int], tuple[str, str]] = {}
    for label, components in COMPONENTS.items():
        for component in components:
            for latent_idx in component["members"]:
                assignment[(label, latent_idx)] = (component["key"], component["name_zh"])
    nonbehavior_assignment: dict[tuple[str, int], tuple[str, str]] = {}
    for label, components in NONBEHAVIOR_COMPONENTS.items():
        for component in components:
            for latent_idx in component["members"]:
                nonbehavior_assignment[(label, latent_idx)] = (component["key"], component["name_zh"])

    feature_rows: list[dict[str, Any]] = []
    for row in stable.itertuples(index=False):
        latent_idx = int(row.latent_idx)
        card = cards[latent_idx]
        component_key, component_name = assignment.get((row.label, latent_idx), ("", ""))
        representation_key, representation_name = nonbehavior_assignment.get(
            (row.label, latent_idx), (component_key, component_name)
        )
        if component_key:
            grouping_status = "assigned_behavior_component"
        elif representation_key:
            grouping_status = "assigned_nonbehavior_component"
        elif card["explanation_type"] == "behavioral_function":
            grouping_status = "behavior_unassigned"
        else:
            grouping_status = "nonbehavior_isolated_or_unresolved"
        feature_rows.append(
            {
                "item_id": f"{row.label}_{latent_idx}",
                "label": row.label,
                "is_leaf_label": row.label in LEAF_LABELS,
                "latent_idx": latent_idx,
                "rank_within_label": int(row.rank_within_label),
                "inclusion_frequency": float(row.inclusion_frequency),
                "abs_cohens_d": float(row.abs_cohens_d),
                "evidence_class": card["explanation_type"],
                "behavior_component_id": component_key,
                "behavior_component": component_name,
                "representation_component_id": representation_key,
                "representation_component": representation_name,
                "grouping_status": grouping_status,
                "short_name": card["short_name"],
                "primary_explanation": card["primary_explanation"],
                "candidate_behavioral_explanation": card["candidate_behavioral_explanation"],
                "representative_evidence": _representative(card, packs[latent_idx]),
                "alternative_explanations": " | ".join(card["alternative_explanations"]),
                "possible_confounds": " | ".join(card["possible_confounds"]),
                "limitations": " | ".join(card["limitations"]),
                "card_confidence": int(card["confidence"]),
                "model_reported_support_fraction": float(card["support_fraction"]),
                "coding_provenance": "extracted_unchanged_from_validated_feature_card",
                "grouping_provenance": "current_ai_evidence_review",
            }
        )
    feature_df = pd.DataFrame(feature_rows)

    component_rows: list[dict[str, Any]] = []
    for label in LABEL_ORDER:
        ranks = {int(row.latent_idx): int(row.rank_within_label) for row in stable[stable["label"].eq(label)].itertuples()}
        for component in COMPONENTS[label]:
            members = sorted(component["members"], key=lambda value: ranks[value])
            evidence = " || ".join(
                f"F{latent_idx}/{_representative(cards[latent_idx], packs[latent_idx], limit=1)}"
                for latent_idx in members[:3]
            )
            component_rows.append(
                {
                    "label": label,
                    "is_leaf_label": label in LEAF_LABELS,
                    "component_ref": f"{label}:{component['key']}",
                    "behavior_component": component["name_zh"],
                    "behavior_component_en": component["name_en"],
                    "definition": component["definition"],
                    "supporting_feature_count": len(members),
                    "supporting_sae_features": "|".join(f"F{value}" for value in members),
                    "representative_evidence": evidence,
                    "confidence": component["confidence"],
                    "boundary_note": component["boundary"],
                    "status": "current_ai_grouping_pending_human_review",
                }
            )
    components_df = pd.DataFrame(component_rows)
    singleton_df = feature_df[feature_df["grouping_status"].eq("behavior_unassigned")].copy()
    nonbehavior_df = feature_df[~feature_df["evidence_class"].eq("behavioral_function")].copy()

    nonbehavior_component_rows: list[dict[str, Any]] = []
    for label in LABEL_ORDER:
        ranks = {
            int(row.latent_idx): int(row.rank_within_label)
            for row in stable[stable["label"].eq(label)].itertuples()
        }
        for component in NONBEHAVIOR_COMPONENTS[label]:
            members = sorted(component["members"], key=lambda value: ranks[value])
            evidence = " || ".join(
                f"F{latent_idx}/{_representative(cards[latent_idx], packs[latent_idx], limit=1)}"
                for latent_idx in members[:3]
            )
            recurrence = (
                "recurrent_multi_latent_strong_candidate"
                if len(members) >= 3
                else "recurrent_multi_latent_candidate"
            )
            nonbehavior_component_rows.append(
                {
                    "label": label,
                    "is_leaf_label": label in LEAF_LABELS,
                    "evidence_class": component["evidence_class"],
                    "component_ref": f"{label}:{component['key']}",
                    "representation_component": component["name_zh"],
                    "representation_component_en": component["name_en"],
                    "definition": component["definition"],
                    "supporting_feature_count": len(members),
                    "supporting_sae_features": "|".join(f"F{value}" for value in members),
                    "representative_evidence": evidence,
                    "confidence": component["confidence"],
                    "boundary_note": component["boundary"],
                    "structural_recurrence": recurrence,
                    "status": "current_ai_grouping_pending_human_review",
                }
            )
    nonbehavior_components_df = pd.DataFrame(nonbehavior_component_rows)
    isolated_nonbehavior_df = feature_df[
        feature_df["grouping_status"].eq("nonbehavior_isolated_or_unresolved")
    ].copy()

    behavior_all_df = components_df.rename(
        columns={
            "behavior_component": "representation_component",
            "behavior_component_en": "representation_component_en",
        }
    ).copy()
    behavior_all_df["evidence_class"] = "behavioral_function"
    behavior_all_df["structural_recurrence"] = behavior_all_df["supporting_feature_count"].map(
        lambda value: "recurrent_multi_latent_strong_candidate"
        if int(value) >= 3
        else "recurrent_multi_latent_candidate"
    )
    all_components_df = pd.concat(
        [behavior_all_df, nonbehavior_components_df], ignore_index=True, sort=False
    )

    exact_rows: list[dict[str, Any]] = []
    for latent_idx, group in feature_df.groupby("latent_idx", sort=True):
        labels = set(group["label"].astype(str))
        if len(labels) < 2:
            continue
        leaf = labels & LEAF_LABELS
        exact_rows.append(
            {
                "latent_idx": int(latent_idx),
                "all_labels": "|".join(label for label in LABEL_ORDER if label in labels),
                "leaf_labels": "|".join(label for label in LABEL_ORDER if label in leaf),
                "sharing_scope_all_labels": _scope(labels),
                "sharing_scope_leaf_labels": _scope(leaf) if len(leaf) >= 2 else "no_multi_leaf_overlap",
                "evidence_class": group.iloc[0]["evidence_class"],
                "short_name": group.iloc[0]["short_name"],
                "component_assignments": "|".join(
                    f"{row.label}:{row.representation_component or 'isolated/unassigned'}"
                    for row in group.itertuples(index=False)
                ),
            }
        )
    exact_df = pd.DataFrame(exact_rows)

    component_refs = set(components_df["component_ref"])
    shared_refs = {ref for family in CROSS_FAMILIES for ref in family["refs"]}
    if not shared_refs.issubset(component_refs):
        raise ValueError(f"Unknown cross-label component refs: {sorted(shared_refs-component_refs)}")
    cross_rows = []
    for family in CROSS_FAMILIES:
        labels = {ref.split(":", 1)[0] for ref in family["refs"]}
        cross_rows.append({"shared_component": family["name"], "component_refs": "|".join(family["refs"]), "labels": "|".join(label for label in LABEL_ORDER if label in labels), "sharing_scope": _scope(labels), "shared_evidence_basis": family["basis"], "label_specific_differences": family["differences"], "status": "current_ai_cross_label_candidate"})
    cross_df = pd.DataFrame(cross_rows)
    all_component_refs = set(all_components_df["component_ref"])
    all_cross_rows = [
        {
            "shared_component": row["name"],
            "evidence_class": "behavioral_function",
            "component_refs": "|".join(row["refs"]),
            "labels": "|".join(
                label
                for label in LABEL_ORDER
                if label in {ref.split(":", 1)[0] for ref in row["refs"]}
            ),
            "sharing_scope": _scope({ref.split(":", 1)[0] for ref in row["refs"]}),
            "shared_evidence_basis": row["basis"],
            "label_specific_differences": row["differences"],
            "status": "current_ai_cross_label_candidate",
        }
        for row in CROSS_FAMILIES
    ]
    for family in NONBEHAVIOR_CROSS_FAMILIES:
        unknown = set(family["refs"]) - all_component_refs
        if unknown:
            raise ValueError(f"Unknown nonbehavior cross-label component refs: {sorted(unknown)}")
        labels = {ref.split(":", 1)[0] for ref in family["refs"]}
        all_cross_rows.append(
            {
                "shared_component": family["name"],
                "evidence_class": family["evidence_class"],
                "component_refs": "|".join(family["refs"]),
                "labels": "|".join(label for label in LABEL_ORDER if label in labels),
                "sharing_scope": _scope(labels),
                "shared_evidence_basis": family["basis"],
                "label_specific_differences": family["differences"],
                "status": "current_ai_cross_label_candidate",
            }
        )
    all_cross_df = pd.DataFrame(all_cross_rows)
    all_shared_refs = {
        ref
        for family in CROSS_FAMILIES + NONBEHAVIOR_CROSS_FAMILIES
        for ref in family["refs"]
    }
    specific_df = all_components_df[~all_components_df["component_ref"].isin(all_shared_refs)].copy()

    summary_rows = []
    for label in LABEL_ORDER:
        subset = feature_df[feature_df["label"].eq(label)]
        row = {
            "label": label,
            "n_feature_cards": len(subset),
            "n_behavior_components": int(components_df["label"].eq(label).sum()),
            "n_nonbehavior_components": int(nonbehavior_components_df["label"].eq(label).sum()),
            "n_all_representation_components": int(all_components_df["label"].eq(label).sum()),
            "n_behavior_assigned": int(subset["grouping_status"].eq("assigned_behavior_component").sum()),
            "n_behavior_unassigned": int(subset["grouping_status"].eq("behavior_unassigned").sum()),
            "n_nonbehavior_assigned": int(subset["grouping_status"].eq("assigned_nonbehavior_component").sum()),
            "n_nonbehavior_isolated": int(subset["grouping_status"].eq("nonbehavior_isolated_or_unresolved").sum()),
        }
        row.update({f"n_{kind}": int(subset["evidence_class"].eq(kind).sum()) for kind in sorted(EVIDENCE_CLASSES)})
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "feature_codings": output_dir / "task4_feature_codings.csv",
        "behavior_components": output_dir / "task4_behavior_components.csv",
        "all_representation_components": output_dir / "task4_all_representation_components.csv",
        "nonbehavior_components": output_dir / "task4_nonbehavior_components.csv",
        "behavior_unassigned": output_dir / "task4_behavior_singletons_unresolved.csv",
        "nonbehavior_features": output_dir / "task4_nonbehavior_features.csv",
        "isolated_nonbehavior_features": output_dir / "task4_isolated_nonbehavior_features.csv",
        "cross_label_components": output_dir / "task4_cross_label_components.csv",
        "cross_label_all_modes": output_dir / "task4_cross_label_all_modes.csv",
        "label_specific_components": output_dir / "task4_label_specific_components.csv",
        "exact_shared_latents": output_dir / "task4_exact_shared_latents.csv",
        "label_summary": output_dir / "task4_label_summary.csv",
    }
    for frame, key in (
        (feature_df, "feature_codings"),
        (components_df, "behavior_components"),
        (all_components_df, "all_representation_components"),
        (nonbehavior_components_df, "nonbehavior_components"),
        (singleton_df, "behavior_unassigned"),
        (nonbehavior_df, "nonbehavior_features"),
        (isolated_nonbehavior_df, "isolated_nonbehavior_features"),
        (cross_df, "cross_label_components"),
        (all_cross_df, "cross_label_all_modes"),
        (specific_df, "label_specific_components"),
        (exact_df, "exact_shared_latents"),
        (summary_df, "label_summary"),
    ):
        frame.to_csv(outputs[key], index=False, encoding="utf-8-sig")

    class_counts = feature_df["evidence_class"].value_counts().to_dict()
    lines = [
        "# Task 4：标签的完整表征结构",
        "",
        "## 1. 分析范围与方法",
        "",
        f"本分析直接提取冻结包中 {len(cards)} 张 feature card 已有的逐卡独立编码，并映射到 {len(feature_df)} 条 stable-core 标签–latent 关联。本报告不读取或依赖新的 API 二次编码结果，也没有修改卡片的 `explanation_type`。",
        "",
        "六类既有编码全部进入分析：行为功能、语言结构、情感内容、主题、表层伪影以及不清晰/混合。归组由当前 AI 根据卡片解释、代表性证据、替代解释、混淆因素和限制完成。每个重复成分至少包含两个 stable-core 特征；不能形成多特征模式的卡片保留为孤立或未解析证据。",
        "",
        "当前归组尚未经过人工审查。卡片的 support_fraction 是原生成模型的自报覆盖率，不是独立准确率。RE/RES/REC 缺少 client 前文，因此涉及反映的成分只能写成候选结构。",
        "",
        "## 2. 总体编码构成",
        "",
        f"- 行为功能：{class_counts.get('behavioral_function', 0)}",
        f"- 语言结构：{class_counts.get('linguistic_structure', 0)}",
        f"- 情感内容：{class_counts.get('affective_content', 0)}",
        f"- 主题：{class_counts.get('topic', 0)}",
        f"- 表层伪影：{class_counts.get('surface_artifact', 0)}",
        f"- 不清晰/混合：{class_counts.get('unclear_or_mixed', 0)}",
        "",
        f"129 条非行为关联中，{int(nonbehavior_components_df['supporting_feature_count'].sum())} 条进入 {len(nonbehavior_components_df)} 个多特征重复模式，{len(isolated_nonbehavior_df)} 条保留为孤立或未解析模式。多 latent 重复出现说明该模式值得作为系统性表征候选，但不等于已经排除了数据模板、转录偏差或共同语料主题。",
        "",
        "## 3. 标签概览",
        "",
        "| 标签 | 卡片 | 行为 | 语言 | 情感 | 主题 | 伪影 | 不清晰 | 行为成分 | 其他成分 | 孤立非行为 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(f"| {row.label} | {row.n_feature_cards} | {row.n_behavioral_function} | {row.n_linguistic_structure} | {row.n_affective_content} | {row.n_topic} | {row.n_surface_artifact} | {row.n_unclear_or_mixed} | {row.n_behavior_components} | {row.n_nonbehavior_components} | {row.n_nonbehavior_isolated} |")
    lines.extend(["", "## 4. 标签内行为成分", ""])
    for label in LABEL_ORDER:
        lines.extend([f"### {label}", "", "| 行为成分 | 支持特征 | 代表性证据 | 置信度 |", "|---|---|---|---|"])
        for row in components_df[components_df["label"].eq(label)].itertuples(index=False):
            lines.append(f"| {row.behavior_component} | `{row.supporting_sae_features}` | {str(row.representative_evidence).replace('|', '/')} | {row.confidence} |")
        lines.append("")
    lines.extend(["## 5. 标签内语言、情感、主题及不清晰成分", ""])
    for label in LABEL_ORDER:
        lines.extend([f"### {label}", "", "| 证据层 | 表征成分 | 支持特征 | 代表性证据 | 置信度 |", "|---|---|---|---|---|"])
        subset = nonbehavior_components_df[nonbehavior_components_df["label"].eq(label)]
        if subset.empty:
            lines.append("| - | 无多特征非行为成分 | - | - | - |")
        for row in subset.itertuples(index=False):
            lines.append(f"| `{row.evidence_class}` | {row.representation_component} | `{row.supporting_sae_features}` | {str(row.representative_evidence).replace('|', '/')} | {row.confidence} |")
        lines.append("")
    lines.extend(["## 6. 孤立、未归组和潜在伪影", ""])
    for label in LABEL_ORDER:
        unresolved = singleton_df[singleton_df["label"].eq(label)]
        isolated = isolated_nonbehavior_df[isolated_nonbehavior_df["label"].eq(label)]
        behavior_ids = "|".join(f"F{value}" for value in unresolved["latent_idx"].astype(int)) or "无"
        isolated_ids = "|".join(
            f"F{row.latent_idx}({row.evidence_class})" for row in isolated.itertuples(index=False)
        ) or "无"
        lines.append(f"- `{label}`：未归组行为卡片 `{behavior_ids}`；孤立/未解析非行为卡片 `{isolated_ids}`。")
    lines.extend([
        "",
        "`F26869` 在 RE 和 REC 中均为 stable-core 关联，但卡片编码是 `surface_artifact`（所有格缩写/转录错误）。这说明它不是一次随机运行中临时出现的索引，却仍更可能反映数据或转录规律，而不是咨询行为理解。",
        "",
        "## 7. 跨标签共享表征",
        "",
        "| 共享成分 | 证据层 | 标签 | 范围 | 共享证据 | 标签差异 |",
        "|---|---|---|---|---|---|",
    ])
    for row in all_cross_df.itertuples(index=False):
        lines.append(f"| {row.shared_component} | `{row.evidence_class}` | `{row.labels}` | `{row.sharing_scope}` | {row.shared_evidence_basis} | {row.label_specific_differences} |")
    lines.extend(["", "## 8. 各标签的综合表征方式", ""])
    for label in LABEL_ORDER:
        lines.append(f"- `{label}`：{LABEL_INTERPRETATIONS[label]}")
    lines.extend([
        "",
        "## 9. 如何判断不是简单偶然误差",
        "",
        "本分析使用三层证据区分系统性候选与偶然线索：第一，输入 latent 已属于跨重采样筛选得到的 stable core；第二，同一标签内至少两个独立 latent 共享模式才形成重复成分；第三，跨标签的同一 latent 复用与同类成分复现分别单列。",
        "",
        "这些条件能够降低单一异常卡片造成的误判，但不能排除共同词汇、模板化语料、标签层级共现或转录偏差。因此，语言结构、主题和情感模式应写成模型表征标签的方式，而不是自动升级为行为机制；表层伪影和不清晰模式则作为结构性风险证据保留。",
        "",
        f"精确共享 latent 见 `task4_exact_shared_latents.csv`；全部 {len(all_components_df)} 个多特征成分见 `task4_all_representation_components.csv`；只在一个标签形成的模式见 `task4_label_specific_components.csv`。人工双审查、kappa 和人工质量审核本轮暂不执行。",
        "",
    ])
    report_path = output_dir / "task4_behavior_representation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "analysis": "task4_behavior_representation_structure",
        "status": "current_ai_grouping_human_review_deferred",
        "inputs": {"cards": str(frozen / "validated_cards.jsonl"), "sentence_packs": str(frozen / "sentence_packs.jsonl"), "stable_latents": str(frozen / "stable_topk_latent_set.csv")},
        "outputs": {**{key: str(value) for key, value in outputs.items()}, "report": str(report_path)},
        "counts": {
            "unique_cards": len(cards),
            "label_latent_associations": len(feature_df),
            "behavior_associations": validation["behavior_associations"],
            "nonbehavior_associations": validation["nonbehavior_associations"],
            "behavior_components": len(components_df),
            "nonbehavior_components": len(nonbehavior_components_df),
            "all_representation_components": len(all_components_df),
            "behavior_unassigned": len(singleton_df),
            "nonbehavior_isolated_or_unresolved": len(isolated_nonbehavior_df),
            "cross_label_behavior_families": len(cross_df),
            "cross_label_all_mode_families": len(all_cross_df),
            "label_specific_components": len(specific_df),
            "exact_shared_latents": len(exact_df),
            "evidence_class_associations": class_counts,
        },
        "provenance": {"card_coding": "existing validated feature cards, extracted unchanged", "component_grouping": "current AI evidence review, no external API", "human_review": "deferred by user"},
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


__all__ = [
    "COMPONENTS",
    "EVIDENCE_CLASSES",
    "LABEL_ORDER",
    "NONBEHAVIOR_COMPONENTS",
    "NONBEHAVIOR_UNASSIGNED",
    "UNASSIGNED",
    "build_task4",
    "validate_grouping_definition",
]
