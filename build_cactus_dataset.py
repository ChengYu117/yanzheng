"""Build a cleaner CACTUS dataset for SAE-RE experiments.

The script downloads the CACTUS dataset from Hugging Face, extracts
turn-level `(client_prev, therapist_curr)` pairs, assigns one of three labels,
and exports a balanced 1500-example JSONL plus stats and a preview CSV.

Label set:
- RE: reflective / mirroring therapist responses
- NonRE_CBT: explicit CBT technique or intervention turns
- NonTech_Process: opening, support, session management, or generic exploration
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "cactus"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "LangAGI-Lab/cactus"
TEMPLATE = "<client>\n{client_prev}\n</client>\n<therapist>\n{therapist_curr}\n</therapist>"
MIN_WORDS = 6
MAX_PLAN_CHARS = 320

UNICODE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
        "\uff1f": "?",
    }
)

SPEAKER_RE = re.compile(
    r"^(client|patient|therapist|counselor|counsellor|user|assistant)\s*[:\-]\s*",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[a-z']+")

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "than",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "this",
    "that",
    "these",
    "those",
    "you",
    "your",
    "yours",
    "i",
    "me",
    "my",
    "mine",
    "we",
    "our",
    "ours",
    "they",
    "their",
    "theirs",
    "he",
    "she",
    "them",
    "his",
    "her",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "can",
    "could",
    "will",
    "would",
    "should",
    "may",
    "might",
    "must",
    "so",
    "just",
    "not",
    "no",
    "yes",
    "very",
    "really",
    "more",
    "most",
    "much",
    "many",
    "about",
    "into",
    "over",
    "under",
    "after",
    "before",
    "up",
    "down",
    "out",
    "off",
    "all",
    "some",
    "any",
    "there",
    "here",
    "when",
    "where",
    "why",
    "how",
    "what",
    "which",
    "who",
    "whom",
    "because",
    "while",
    "though",
    "although",
    "also",
    "too",
    "once",
}


RE_STRONG_PREFIXES = (
    "it sounds like",
    "that sounds like",
    "it seems like",
    "it sounds as though",
    "it seems as though",
    "i can see how",
    "i can understand",
    "it must be",
    "that must be",
    "those thoughts must be",
    "it makes sense that",
    "it's understandable",
    "that's understandable",
    "it's clear that",
    "what i'm hearing",
    "i hear that",
)
RE_EXTRA_CUES = (
    "that sounds",
    "these thoughts seem",
    "those thoughts seem",
    "these feelings seem",
    "this seems to be",
    "you're being hard on yourself",
    "you're carrying",
    "you're putting a lot of pressure",
    "you're dealing with",
    "this fear is affecting",
    "this issue is impacting",
    "this belief is having a significant impact",
    "these feelings are affecting",
    "taking a toll",
    "weighing heavily",
)
RE_IMPACT_CUES = (
    "affecting multiple areas",
    "affecting several areas",
    "impacting several areas",
    "impacting your",
    "affecting your",
    "spill over into",
    "taken a toll",
    "having a significant impact",
)
RE_AFFECT_WORDS = (
    "difficult",
    "hard",
    "challenging",
    "painful",
    "frustrating",
    "overwhelming",
    "discouraging",
    "isolating",
    "lonely",
    "exhausting",
    "upsetting",
    "distressing",
    "burden",
    "self-blame",
    "doubt",
)
RE_HARD_REJECT = (
    "let's",
    "let us",
    "we can",
    "we will",
    "we'll",
    "would you",
    "could you",
    "can you",
    "what evidence",
    "how about",
    "next session",
    "next week",
    "this week",
    "coming week",
    "write down",
    "keep track",
    "keep a journal",
    "keep a record",
    "keep a log",
    "keep a diary",
    "track your",
    "record your",
    "journal",
    "homework",
    "experiment",
    "reframe",
    "challenge",
    "alternative perspective",
    "alternative explanation",
    "alternative thought",
    "thought record",
    "behavioral",
    "goal",
    "action step",
    "plan",
    "practice",
    "review",
    "monitor",
    "set a goal",
    "set some",
    "small step",
    "start by",
    "identify",
    "gather",
    "test ",
    "would it help to",
    "might it help to",
    "one step could be",
    "cope with",
    "support system",
    "i'm curious",
    "i wonder if",
    "maybe ",
    "perhaps ",
    "consider ",
    "focus on",
    "observe ",
    "note ",
    "explore ",
    "work together",
    "try to ",
    "try this",
    "try that",
)
RE_SOFT_PENALTIES = (
    "good start",
    "great start",
    "positive step",
    "promising",
    "constructive approach",
    "solid approach",
    "wonderful idea",
    "remember ",
    "important to",
    "helpful to",
    "might help",
    "could help",
    "worth exploring",
    "i'm here to support",
    "you're not alone",
)


CBT_EVIDENCE_MARKERS = (
    "what evidence",
    "evidence for",
    "evidence against",
    "supports this thought",
    "supports this belief",
    "contradicts this thought",
    "contradicts this belief",
    "reality test",
    "test this belief",
)
CBT_REFRAME_MARKERS = (
    "alternative perspective",
    "alternative explanation",
    "alternative thought",
    "another way to look",
    "more balanced",
    "different perspective",
    "different way to view",
    "reframe",
    "other interpretation",
    "other explanation",
)
CBT_HOMEWORK_MARKERS = (
    "next session",
    "next week",
    "this week",
    "coming week",
    "keep track",
    "keep a journal",
    "keep a record",
    "keep a log",
    "keep a diary",
    "write down",
    "record your",
    "journal",
    "homework",
    "experiment",
    "practice",
    "goal",
    "plan",
    "monitor",
    "note the",
    "track your",
    "for the coming week",
)
CBT_GUIDED_DISCOVERY_MARKERS = (
    "let's explore",
    "let us explore",
    "let's look at",
    "let us look at",
    "how about we",
    "take a step back",
    "what patterns",
    "what assumptions",
    "what strategies",
    "what might",
    "what is another way",
    "what do you think might",
    "could there be another",
)
CBT_PSYCHOED_MARKERS = (
    "sometimes our minds",
    "it's common to",
    "automatic thought",
    "cognitive distortion",
    "catastrophizing",
    "black-and-white",
    "all-or-nothing",
    "mind reading",
    "fortune telling",
)
CBT_GENERIC_ACTION_MARKERS = (
    "would you be willing",
    "could you try",
    "can you practice",
    "can you try",
    "one step could be",
    "start by",
    "identify ",
    "gather more information",
    "test those thoughts",
    "challenge this",
    "challenge that",
    "challenge these",
    "challenge those",
)
CBT_MIXED_REJECT_PREFIXES = (
    "thank you for sharing",
    "thanks for sharing",
    "i'm sorry to hear",
    "you're very welcome",
    "take care",
    "see you next",
)

SUBTYPE_TARGETS: dict[str, dict[str, int]] = {
    "NonRE_CBT": {
        "homework_planning": 200,
        "reframe": 100,
        "evidence_check": 100,
        "guided_discovery": 50,
        "psychoeducation": 50,
    },
    "NonTech_Process": {
        "general_support": 150,
        "closing_transition": 150,
        "information_gathering": 150,
        "opening": 50,
    },
}


PROCESS_OPENING_PREFIXES = (
    "hi",
    "hello",
    "good morning",
    "good afternoon",
    "good evening",
    "nice to meet",
    "before we begin",
    "what brings you",
    "how are you feeling",
    "how are you doing",
)
PROCESS_INFO_MARKERS = (
    "can you tell me more",
    "can you tell me a bit more",
    "when did",
    "how long",
    "how often",
    "what happened",
    "what's been on your mind",
    "how has this affected",
    "what do you hope to achieve",
    "what would improvement look like",
    "tell me more about",
)
PROCESS_SUPPORT_MARKERS = (
    "thank you for sharing",
    "thanks for sharing",
    "i'm glad you reached out",
    "i'm glad you came",
    "i'm sorry to hear",
    "i appreciate your honesty",
    "i appreciate your openness",
    "i appreciate your willingness",
    "it's okay to feel",
    "it's completely normal",
    "that's understandable",
    "you're very welcome",
    "i hear you",
    "it makes sense that",
)
PROCESS_CLOSING_MARKERS = (
    "take care",
    "see you next",
    "we'll meet again",
    "we will meet again",
    "i'm looking forward to hearing",
    "we'll continue",
    "we will continue",
    "for today",
    "next time",
)


def normalize_text(text: str) -> str:
    text = text.translate(UNICODE_TRANSLATION)
    return re.sub(r"\s+", " ", text.strip())


def normalize_for_match(text: str) -> str:
    return normalize_text(text).lower()


def normalize_multiline_text(text: str) -> str:
    text = text.translate(UNICODE_TRANSLATION)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(re.sub(r"\s+", " ", line.strip()) for line in text.splitlines())


def contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def startswith_any(text: str, prefixes: tuple[str, ...]) -> bool:
    return any(text.startswith(prefix) for prefix in prefixes)


def content_tokens(text: str) -> set[str]:
    tokens = WORD_RE.findall(normalize_for_match(text))
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def shared_content_count(left: str, right: str) -> int:
    return len(content_tokens(left) & content_tokens(right))


def looks_damaged(text: str) -> bool:
    if not text:
        return True
    if "\ufffd" in text:
        return True
    alpha_words = WORD_RE.findall(normalize_for_match(text))
    if len(alpha_words) < 4:
        return True
    return False


def fingerprint(text: str) -> str:
    normalised = normalize_for_match(text)
    return hashlib.md5(normalised.encode("utf-8")).hexdigest()


def parse_dialogue_item(item: Any) -> tuple[str, str] | None:
    if isinstance(item, dict):
        role = str(item.get("role") or item.get("speaker") or "").lower()
        text = normalize_text(str(item.get("content") or item.get("text") or ""))
        if not text:
            return None
        if role in {"client", "patient", "user"}:
            return ("client", text)
        if role in {"therapist", "counselor", "counsellor", "assistant"}:
            return ("therapist", text)
        return None
    if isinstance(item, str):
        matches = parse_dialogue(item)
        if len(matches) == 1:
            turn = matches[0]
            return (turn["speaker"], turn["text"])
    return None


def parse_dialogue(dialogue_raw: Any) -> list[dict[str, str]]:
    if isinstance(dialogue_raw, list):
        turns: list[dict[str, str]] = []
        for item in dialogue_raw:
            parsed = parse_dialogue_item(item)
            if parsed is None:
                continue
            turns.append({"speaker": parsed[0], "text": parsed[1]})
        return turns

    if not isinstance(dialogue_raw, str):
        return []

    turns: list[dict[str, str]] = []
    for raw_line in normalize_multiline_text(dialogue_raw).split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        match = SPEAKER_RE.match(line)
        if match:
            role = match.group(1).lower()
            text = normalize_text(line[match.end() :])
            if not text:
                continue
            speaker = "client" if role in {"client", "patient", "user"} else "therapist"
            turns.append({"speaker": speaker, "text": text})
            continue
        if turns:
            turns[-1]["text"] = normalize_text(f"{turns[-1]['text']} {line}")
    return turns


def build_turn_pairs(turns: list[dict[str, str]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    last_client: str | None = None
    for turn_index, turn in enumerate(turns):
        speaker = turn["speaker"]
        text = normalize_text(turn["text"])
        if looks_damaged(text):
            continue
        if speaker == "client":
            last_client = text
            continue
        if speaker != "therapist" or last_client is None:
            continue
        if len(text.split()) < MIN_WORDS:
            continue
        pairs.append(
            {
                "turn_index": turn_index,
                "client_prev": last_client,
                "therapist_curr": text,
            }
        )
    return pairs


def classify_re_subtype(text_lc: str) -> str:
    if contains_any(text_lc, ("self-blame", "doubt", "hard on yourself", "pressure")):
        return "self_view_reflection"
    if contains_any(text_lc, RE_IMPACT_CUES):
        return "impact_reflection"
    if contains_any(text_lc, ("burden", "lonely", "isolating", "frustrating", "overwhelming", "painful")):
        return "feeling_reflection"
    return "summary_reflection"


def score_re(client_prev: str, therapist_curr: str) -> tuple[int, str] | None:
    client_lc = normalize_for_match(client_prev)
    text_lc = normalize_for_match(therapist_curr)

    if "?" in text_lc:
        return None
    if contains_any(text_lc, RE_HARD_REJECT):
        return None
    if startswith_any(text_lc, PROCESS_OPENING_PREFIXES):
        return None
    if contains_any(text_lc, PROCESS_CLOSING_MARKERS):
        return None
    if contains_any(text_lc, ("thank you for sharing", "you're very welcome", "take care")):
        return None

    score = 0
    if startswith_any(text_lc, RE_STRONG_PREFIXES):
        score += 4
    if contains_any(text_lc, RE_EXTRA_CUES):
        score += 2
    if contains_any(text_lc, RE_IMPACT_CUES):
        score += 1
    if contains_any(text_lc, RE_AFFECT_WORDS):
        score += 1
    if text_lc.startswith(("so ", "from what you've shared", "from what you're describing", "i see. ")):
        score += 1

    overlap = shared_content_count(client_lc, text_lc)
    score += min(4, overlap)

    token_count = len(content_tokens(text_lc))
    if 8 <= token_count <= 30:
        score += 1

    if contains_any(text_lc, RE_SOFT_PENALTIES):
        score -= 3

    if score < 3:
        return None

    return score, classify_re_subtype(text_lc)


def classify_cbt_subtype(text_lc: str) -> str:
    if contains_any(text_lc, CBT_EVIDENCE_MARKERS):
        return "evidence_check"
    if contains_any(text_lc, CBT_REFRAME_MARKERS):
        return "reframe"
    if contains_any(text_lc, CBT_HOMEWORK_MARKERS):
        return "homework_planning"
    if contains_any(text_lc, CBT_PSYCHOED_MARKERS):
        return "psychoeducation"
    return "guided_discovery"


def score_cbt(therapist_curr: str) -> tuple[int, str] | None:
    text_lc = normalize_for_match(therapist_curr)

    if startswith_any(text_lc, CBT_MIXED_REJECT_PREFIXES):
        return None
    if startswith_any(text_lc, PROCESS_OPENING_PREFIXES) and not contains_any(
        text_lc, CBT_EVIDENCE_MARKERS + CBT_REFRAME_MARKERS + CBT_HOMEWORK_MARKERS
    ):
        return None

    score = 0
    if contains_any(text_lc, CBT_EVIDENCE_MARKERS):
        score += 4
    if contains_any(text_lc, CBT_REFRAME_MARKERS):
        score += 4
    if contains_any(text_lc, CBT_HOMEWORK_MARKERS):
        score += 5
    if contains_any(text_lc, CBT_GUIDED_DISCOVERY_MARKERS):
        score += 3
    if contains_any(text_lc, CBT_PSYCHOED_MARKERS):
        score += 3
    if contains_any(text_lc, CBT_GENERIC_ACTION_MARKERS):
        score += 2
    if "?" in text_lc:
        score += 1

    if contains_any(text_lc, PROCESS_CLOSING_MARKERS):
        score -= 2

    if score < 4:
        return None

    return score, classify_cbt_subtype(text_lc)


def classify_process_subtype(text_lc: str) -> str:
    if contains_any(text_lc, PROCESS_CLOSING_MARKERS):
        return "closing_transition"
    if contains_any(text_lc, PROCESS_SUPPORT_MARKERS):
        return "general_support"
    if startswith_any(text_lc, PROCESS_OPENING_PREFIXES):
        return "opening"
    return "information_gathering"


def score_process(client_prev: str, therapist_curr: str) -> tuple[int, str] | None:
    client_lc = normalize_for_match(client_prev)
    text_lc = normalize_for_match(therapist_curr)

    if contains_any(
        text_lc,
        CBT_EVIDENCE_MARKERS
        + CBT_REFRAME_MARKERS
        + CBT_HOMEWORK_MARKERS
        + CBT_GUIDED_DISCOVERY_MARKERS
        + CBT_PSYCHOED_MARKERS
        + CBT_GENERIC_ACTION_MARKERS,
    ):
        return None

    score = 0
    if startswith_any(text_lc, PROCESS_OPENING_PREFIXES):
        score += 3
    if contains_any(text_lc, PROCESS_INFO_MARKERS):
        score += 3
    if contains_any(text_lc, PROCESS_SUPPORT_MARKERS):
        score += 2
    if contains_any(text_lc, PROCESS_CLOSING_MARKERS):
        score += 3
    if "?" in text_lc:
        score += 1
    if shared_content_count(client_lc, text_lc) >= 1:
        score += 1

    if score < 3:
        return None

    return score, classify_process_subtype(text_lc)


def classify_pair(client_prev: str, therapist_curr: str) -> tuple[str, str, int] | None:
    re_result = score_re(client_prev, therapist_curr)
    cbt_result = score_cbt(therapist_curr)
    process_result = score_process(client_prev, therapist_curr)

    if re_result is not None and cbt_result is None and process_result is None:
        score, subtype = re_result
        return ("RE", subtype, score)
    if cbt_result is not None and re_result is None:
        score, subtype = cbt_result
        return ("NonRE_CBT", subtype, score)
    if process_result is not None and re_result is None and cbt_result is None:
        score, subtype = process_result
        return ("NonTech_Process", subtype, score)
    return None


def format_sample(client_prev: str, therapist_curr: str) -> dict[str, Any]:
    formatted_text = TEMPLATE.format(client_prev=client_prev, therapist_curr=therapist_curr)
    marker = "<therapist>\n"
    char_start = formatted_text.find(marker) + len(marker)
    char_end = char_start + len(therapist_curr)
    if formatted_text[char_start:char_end] != therapist_curr:
        raise ValueError("Therapist span mismatch after formatting.")
    return {
        "formatted_text": formatted_text,
        "therapist_char_start": char_start,
        "therapist_char_end": char_end,
    }


def deduplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_pair: set[str] = set()
    seen_therapist: set[str] = set()
    deduped: list[dict[str, Any]] = []

    for sample in candidates:
        pair_fp = fingerprint(f"{sample['client_prev']} || {sample['therapist_curr']}")
        therapist_fp = fingerprint(sample["therapist_curr"])
        if pair_fp in seen_pair or therapist_fp in seen_therapist:
            continue
        seen_pair.add(pair_fp)
        seen_therapist.add(therapist_fp)
        deduped.append(sample)

    return deduped


def round_robin_select(
    samples: list[dict[str, Any]],
    target: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[sample["subtype"]].append(sample)

    for subtype_samples in grouped.values():
        rng.shuffle(subtype_samples)
        subtype_samples.sort(
            key=lambda item: (
                -item["_score"],
                -item["_shared"],
                item["sample_id"],
            )
        )

    queues = {subtype: deque(items) for subtype, items in grouped.items() if items}
    subtype_order = sorted(queues)
    chosen: list[dict[str, Any]] = []
    dialogue_counts: Counter[str] = Counter()
    thought_counts: Counter[str] = Counter()

    while queues and len(chosen) < target:
        progressed = False
        for subtype in list(subtype_order):
            queue = queues.get(subtype)
            if not queue:
                continue
            while queue:
                candidate = queue[0]
                if dialogue_counts[candidate["dialogue_id"]] >= 1:
                    queue.popleft()
                    continue
                if thought_counts[candidate["_thought_fp"]] >= 1:
                    queue.popleft()
                    continue
                chosen.append(queue.popleft())
                dialogue_counts[candidate["dialogue_id"]] += 1
                thought_counts[candidate["_thought_fp"]] += 1
                progressed = True
                break
            if not queue:
                queues.pop(subtype, None)
                subtype_order = [key for key in subtype_order if key in queues]
            if len(chosen) >= target:
                break
        if not progressed:
            break

    if len(chosen) < target:
        remaining = sorted(
            (
                sample
                for sample in samples
                if sample not in chosen
            ),
            key=lambda item: (-item["_score"], -item["_shared"], item["sample_id"]),
        )
        for sample in remaining:
            if len(chosen) >= target:
                break
            if dialogue_counts[sample["dialogue_id"]] >= 2:
                continue
            chosen.append(sample)
            dialogue_counts[sample["dialogue_id"]] += 1

    return chosen[:target]


def select_with_subtype_targets(
    samples: list[dict[str, Any]],
    targets: dict[str, int],
    seed: int,
    prefer_non_questions: bool = True,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[sample["subtype"]].append(sample)

    def sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
        return (
            item["_is_question"] if prefer_non_questions else 0,
            -item["_score"],
            -item["_shared"],
            item["sample_id"],
        )

    for subtype_samples in grouped.values():
        rng.shuffle(subtype_samples)
        subtype_samples.sort(key=sort_key)

    chosen: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()
    dialogue_counts: Counter[str] = Counter()
    thought_counts: Counter[str] = Counter()

    def try_take(sample: dict[str, Any]) -> bool:
        if sample["sample_id"] in chosen_ids:
            return False
        if dialogue_counts[sample["dialogue_id"]] >= 1:
            return False
        if thought_counts[sample["_thought_fp"]] >= 1:
            return False
        chosen.append(sample)
        chosen_ids.add(sample["sample_id"])
        dialogue_counts[sample["dialogue_id"]] += 1
        thought_counts[sample["_thought_fp"]] += 1
        return True

    for subtype, target in targets.items():
        queue = deque(grouped.get(subtype, []))
        while queue and sum(1 for item in chosen if item["subtype"] == subtype) < target:
            sample = queue.popleft()
            try_take(sample)

    target_total = sum(targets.values())
    if len(chosen) < target_total:
        remaining = sorted(
            (sample for sample in samples if sample["sample_id"] not in chosen_ids),
            key=sort_key,
        )
        for sample in remaining:
            if len(chosen) >= target_total:
                break
            if try_take(sample):
                continue

    if len(chosen) < target_total:
        remaining = sorted(
            (sample for sample in samples if sample["sample_id"] not in chosen_ids),
            key=lambda item: (-item["_score"], -item["_shared"], item["sample_id"]),
        )
        for sample in remaining:
            if len(chosen) >= target_total:
                break
            if dialogue_counts[sample["dialogue_id"]] >= 2:
                continue
            chosen.append(sample)
            chosen_ids.add(sample["sample_id"])
            dialogue_counts[sample["dialogue_id"]] += 1

    return chosen[:target_total]


def compute_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    question_counts: Counter[str] = Counter()
    lengths: defaultdict[str, list[int]] = defaultdict(list)
    subtype_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    technique_counts: Counter[str] = Counter()

    for sample in samples:
        label = sample["label"]
        label_counts[label] += 1
        lengths[label].append(len(sample["therapist_curr"].split()))
        if "?" in sample["therapist_curr"]:
            question_counts[label] += 1
        subtype_counts[label][sample.get("subtype", "") or "unknown"] += 1
        technique = sample.get("source_cbt_technique") or "unknown"
        technique_counts[technique] += 1

    stats: dict[str, Any] = {
        "total": len(samples),
        "per_label": {},
        "subtype_distribution": {},
        "technique_distribution": dict(technique_counts.most_common(20)),
    }

    for label, count in label_counts.items():
        stats["per_label"][label] = {
            "count": count,
            "mean_words": round(sum(lengths[label]) / len(lengths[label]), 1) if lengths[label] else 0.0,
            "question_ratio": round(question_counts[label] / count, 3) if count else 0.0,
        }
        stats["subtype_distribution"][label] = dict(subtype_counts[label].most_common())

    return stats


def load_cactus_records() -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: install the 'datasets' package first: pip install datasets")
        sys.exit(1)

    dataset = load_dataset(DATASET_NAME, split="train")
    return list(dataset)


def build_dataset(
    target_per_class: int = 500,
    seed: int = 42,
    output_dir: Path = DATA_DIR,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading {DATASET_NAME} ...")
    records = load_cactus_records()
    print(f"  loaded {len(records)} source dialogues")

    print("[2/6] Parsing dialogues and building turn pairs ...")
    candidates: dict[str, list[dict[str, Any]]] = {
        "RE": [],
        "NonRE_CBT": [],
        "NonTech_Process": [],
    }
    total_pairs = 0

    for idx, record in enumerate(records):
        dialogue = record.get("dialogue") or record.get("conversation") or record.get("messages")
        turns = parse_dialogue(dialogue)
        pairs = build_turn_pairs(turns)
        total_pairs += len(pairs)

        source_thought = normalize_text(str(record.get("thought") or ""))
        thought_fp = fingerprint(source_thought or f"thought-{idx}")
        cbt_technique = normalize_text(str(record.get("cbt_technique") or ""))
        cbt_plan = normalize_text(str(record.get("cbt_plan") or ""))

        for pair in pairs:
            shared = shared_content_count(pair["client_prev"], pair["therapist_curr"])
            span_info = format_sample(pair["client_prev"], pair["therapist_curr"])
            base_sample = {
                "sample_id": f"cactus_{idx:06d}_t{pair['turn_index']:03d}",
                "dialogue_id": f"cactus_{idx:06d}",
                "turn_index": pair["turn_index"],
                "client_prev": pair["client_prev"],
                "therapist_curr": pair["therapist_curr"],
                "formatted_text": span_info["formatted_text"],
                "therapist_char_start": span_info["therapist_char_start"],
                "therapist_char_end": span_info["therapist_char_end"],
                "source_cbt_technique": cbt_technique,
                "source_cbt_plan": cbt_plan[:MAX_PLAN_CHARS],
                "_shared": shared,
                "_thought_fp": thought_fp,
                "_is_question": "?" in pair["therapist_curr"],
            }

            re_result = score_re(pair["client_prev"], pair["therapist_curr"])
            if re_result is not None:
                score, subtype = re_result
                candidates["RE"].append(
                    {
                        **base_sample,
                        "label": "RE",
                        "subtype": subtype,
                        "_score": score,
                    }
                )
                continue

            cbt_result = score_cbt(pair["therapist_curr"])
            if cbt_result is not None:
                score, subtype = cbt_result
                candidates["NonRE_CBT"].append(
                    {
                        **base_sample,
                        "label": "NonRE_CBT",
                        "subtype": subtype,
                        "_score": score,
                    }
                )
                continue

            process_result = score_process(pair["client_prev"], pair["therapist_curr"])
            if process_result is not None:
                score, subtype = process_result
                candidates["NonTech_Process"].append(
                    {
                        **base_sample,
                        "label": "NonTech_Process",
                        "subtype": subtype,
                        "_score": score,
                    }
                )

    print(f"  built {total_pairs} turn pairs")
    for label, items in candidates.items():
        print(f"  raw candidates {label}: {len(items)}")

    print("[3/6] Deduplicating candidates ...")
    for label in candidates:
        before = len(candidates[label])
        candidates[label] = deduplicate_candidates(candidates[label])
        print(f"  {label}: {before} -> {len(candidates[label])}")

    print("[4/6] Selecting balanced samples ...")
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for offset, label in enumerate(("RE", "NonRE_CBT", "NonTech_Process")):
        pool = [sample for sample in candidates[label] if sample["sample_id"] not in used_ids]
        if label in SUBTYPE_TARGETS:
            chosen = select_with_subtype_targets(
                pool,
                SUBTYPE_TARGETS[label],
                seed + offset,
                prefer_non_questions=True,
            )
        else:
            chosen = round_robin_select(pool, target_per_class, seed + offset)
        if len(chosen) < target_per_class:
            raise RuntimeError(
                f"Not enough clean {label} samples after selection: "
                f"{len(chosen)} < {target_per_class}"
            )
        selected.extend(chosen)
        used_ids.update(sample["sample_id"] for sample in chosen)
        subtype_counter = Counter(sample["subtype"] for sample in chosen)
        print(f"  {label}: {len(chosen)} selected | subtypes={dict(subtype_counter)}")

    rng = random.Random(seed)
    rng.shuffle(selected)

    for sample in selected:
        sample.pop("_score", None)
        sample.pop("_shared", None)
        sample.pop("_thought_fp", None)
        sample.pop("_is_question", None)

    print("[5/6] Writing outputs ...")
    jsonl_path = output_dir / "cactus_re_small_1500.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for sample in selected:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"  wrote {jsonl_path} ({len(selected)} rows)")

    stats = compute_stats(selected)
    stats_path = output_dir / "cactus_re_small_1500_stats.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    print(f"  wrote {stats_path}")

    csv_path = output_dir / "cactus_re_small_1500_preview.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "label",
                "subtype",
                "client_prev",
                "therapist_curr",
                "source_cbt_technique",
            ],
        )
        writer.writeheader()
        for sample in selected:
            writer.writerow({field: sample.get(field, "") for field in writer.fieldnames})
    print(f"  wrote {csv_path}")

    print("[6/6] Done.")
    print(json.dumps(stats["per_label"], indent=2, ensure_ascii=False))
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cleaned CACTUS dataset.")
    parser.add_argument("--target-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(DATA_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build_dataset(
        target_per_class=arguments.target_per_class,
        seed=arguments.seed,
        output_dir=Path(arguments.output_dir),
    )
