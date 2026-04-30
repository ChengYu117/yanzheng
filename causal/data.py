"""causal/data.py — Data loading with therapist-span awareness for CACTUS dataset.

CACTUS samples contain `formatted_text` with <client>...<therapist>... template
and explicit `therapist_char_start` / `therapist_char_end` for span tracking.

The counselor (therapist) span covers only the therapist section tokens,
not the client context tokens.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.data import load_experiment_dataset


@dataclass
class CausalBatch:
    """One batch of examples ready for model forward pass."""

    input_ids: torch.Tensor          # [B, T]
    attention_mask: torch.Tensor     # [B, T]
    counselor_span_mask: torch.Tensor  # [B, T]  1 = therapist token
    labels: torch.Tensor             # [B]  1=RE, 0=NonRE
    texts: list[str]                 # raw formatted_text strings
    indices: list[int]               # global indices into the dataset


def build_dataset(
    data_dir: str | Path,
) -> tuple[list[str], list[int], list[dict]]:
    """Return (texts, labels, records) for the full dataset.

    Uses the unified experiment dataset loader. For the current MISC dataset,
    texts are MISC `unit_text` behavior units and labels are binary RE/NonRE.
    Legacy MI-RE and CACTUS remain supported through format auto-detection.
    """
    dataset = load_experiment_dataset(data_dir, data_format="auto")
    return dataset.texts, dataset.binary_labels, dataset.records


def tokenize_batch(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int = 128,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Tokenize and pad a list of strings; return dict on device."""
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    if device is not None:
        enc = {k: v.to(device) for k, v in enc.items()}
    return enc


def _get_therapist_token_mask(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    record: dict,
    max_seq_len: int,
) -> list[bool]:
    """Compute per-token therapist mask using char offsets from record.

    If the record has therapist_char_start/end (CACTUS format), uses precise
    offset mapping. Falls back to masking all non-padding tokens (legacy).
    """
    char_start = record.get("therapist_char_start")
    char_end   = record.get("therapist_char_end")

    if char_start is not None and char_end is not None:
        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_seq_len,
        )
        offsets = enc["offset_mapping"]
        mask = []
        for cs, ce in offsets:
            if cs == 0 and ce == 0:
                mask.append(False)  # special token
            elif ce <= char_start or cs >= char_end:
                mask.append(False)  # outside therapist span
            else:
                mask.append(True)   # overlaps with therapist span
        return mask
    else:
        # Legacy: all real tokens are therapist
        enc = tokenizer(text, truncation=True, max_length=max_seq_len)
        return [True] * len(enc["input_ids"])


def make_counselor_span_mask_batch(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    records: list[dict],
    attention_mask: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Build a [B, T] bool mask where 1 = therapist token.

    Combines per-sample therapist char-span detection with attention_mask.
    """
    B, T = attention_mask.shape
    device = attention_mask.device
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)

    for i, (text, record) in enumerate(zip(texts, records)):
        tok_mask = _get_therapist_token_mask(tokenizer, text, record, max_seq_len)
        length = min(len(tok_mask), T)
        for j in range(length):
            if tok_mask[j] and attention_mask[i, j]:
                mask[i, j] = True

    return mask


def make_counselor_span_mask(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Legacy: for datasets where each utterance IS the counselor sentence.
    Counselor span = all real (non-padding) tokens.
    Returns a bool tensor of the same shape as attention_mask.
    """
    return attention_mask.bool()


def iter_batches(
    texts: list[str],
    labels: list[int],
    records: list[dict] | None = None,
    *,
    tokenizer: PreTrainedTokenizerBase | None = None,
    batch_size: int = 8,
    max_seq_len: int = 128,
    device: torch.device | None = None,
) -> list[CausalBatch]:
    """Build CausalBatch objects over the full dataset.

    If records are provided (CACTUS format), uses therapist char-span
    for precise counselor span masking. Otherwise falls back to masking
    all non-padding tokens (legacy behavior).

    Supports both positional and keyword calling conventions for tokenizer:
        iter_batches(texts, labels, tokenizer, ...)       # legacy positional
        iter_batches(texts, labels, records, tokenizer=tokenizer, ...)  # new
    """
    # Handle backward-compatible positional calling:
    # iter_batches(texts, labels, tokenizer, ...) where 3rd arg is a tokenizer
    actual_tokenizer = tokenizer
    actual_records = records

    if records is not None and not isinstance(records, list):
        # 3rd positional arg is actually the tokenizer (legacy call pattern)
        actual_tokenizer = records  # type: ignore
        actual_records = None
    elif records is not None and len(records) > 0 and isinstance(records[0], dict):
        # 3rd arg is actually records, tokenizer must be keyword
        actual_records = records
    elif records is not None and hasattr(records, '__call__'):
        # 3rd arg is tokenizer (PreTrainedTokenizerBase)
        actual_tokenizer = records  # type: ignore
        actual_records = None

    if actual_tokenizer is None:
        raise ValueError("tokenizer must be provided")

    batches: list[CausalBatch] = []
    n = len(texts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = texts[start:end]
        batch_labels = labels[start:end]

        enc = tokenize_batch(batch_texts, actual_tokenizer, max_seq_len, device)

        if actual_records is not None:
            batch_records = actual_records[start:end]
            span_mask = make_counselor_span_mask_batch(
                actual_tokenizer, batch_texts, batch_records,
                enc["attention_mask"], max_seq_len
            )
        else:
            span_mask = make_counselor_span_mask(enc["attention_mask"])

        batches.append(CausalBatch(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            counselor_span_mask=span_mask,
            labels=torch.tensor(batch_labels, dtype=torch.long,
                                device=enc["input_ids"].device),
            texts=batch_texts,
            indices=list(range(start, end)),
        ))
    return batches
