"""causal/data.py — Data loading with counselor-span awareness.

Each record in our dataset (unit_text) IS the counselor utterance, so the
counselor span covers all non-padding tokens. We also track an optional
preceding client utterance when available in the JSONL (field: 'client_text').
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class CausalBatch:
    """One batch of examples ready for model forward pass."""

    input_ids: torch.Tensor          # [B, T]
    attention_mask: torch.Tensor     # [B, T]
    counselor_span_mask: torch.Tensor  # [B, T]  1 = counselor token
    labels: torch.Tensor             # [B]  1=RE, 0=NonRE
    texts: list[str]                 # raw counselor strings
    indices: list[int]               # global indices into the dataset


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dataset(
    re_path: str | Path,
    nonre_path: str | Path,
) -> tuple[list[str], list[int], list[dict]]:
    """Return (texts, labels, records) for the full dataset."""
    re_records = load_jsonl(re_path)
    nonre_records = load_jsonl(nonre_path)

    texts: list[str] = []
    labels: list[int] = []
    records: list[dict] = []

    for r in re_records:
        texts.append(r["unit_text"])
        labels.append(1)
        records.append(r)

    for r in nonre_records:
        texts.append(r["unit_text"])
        labels.append(0)
        records.append(r)

    return texts, labels, records


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


def make_counselor_span_mask(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """For our dataset each utterance IS the counselor sentence.
    Counselor span = all real (non-padding) tokens.
    Returns a bool tensor of the same shape as attention_mask.
    """
    return attention_mask.bool()


def iter_batches(
    texts: list[str],
    labels: list[int],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 8,
    max_seq_len: int = 128,
    device: torch.device | None = None,
) -> list[CausalBatch]:
    """Yield CausalBatch objects over the full dataset."""
    batches: list[CausalBatch] = []
    n = len(texts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = texts[start:end]
        batch_labels = labels[start:end]

        enc = tokenize_batch(batch_texts, tokenizer, max_seq_len, device)
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
