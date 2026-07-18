"""Archive short utterances and build an aligned minimum-word statistical view."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def whitespace_word_count(text: str) -> int:
    return len(str(text).split())


def _filter_feature_payload(
    payload: Any,
    *,
    index_tensor: torch.Tensor,
    expected_rows: int,
    name: str,
) -> Any:
    if isinstance(payload, torch.Tensor):
        if int(payload.shape[0]) != expected_rows:
            raise ValueError(f"{name} row count {payload.shape[0]} != {expected_rows}")
        return payload.index_select(0, index_tensor)
    if isinstance(payload, dict):
        filtered = dict(payload)
        matched = 0
        for key, value in payload.items():
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and int(value.shape[0]) == expected_rows:
                filtered[key] = value.index_select(0, index_tensor)
                matched += 1
        if matched == 0:
            raise ValueError(f"{name} contains no row-aligned tensor")
        return filtered
    raise TypeError(f"Unsupported feature payload in {name}: {type(payload).__name__}")


def build_min_word_statistical_scope(
    *,
    source_root: str | Path,
    output_root: str | Path,
    minimum_words: int = 5,
    filter_feature_tensors: bool = True,
) -> dict[str, Any]:
    """Build a row-aligned view containing utterances with at least minimum_words."""

    if minimum_words < 1:
        raise ValueError("minimum_words must be positive")
    source = Path(source_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(source / "label_matrix.csv")
    records = _read_jsonl(source / "records.jsonl")
    if len(labels) != len(records):
        raise ValueError(f"label/record length mismatch: {len(labels)} != {len(records)}")
    if "row_idx" not in labels.columns or labels["row_idx"].tolist() != list(range(len(labels))):
        raise ValueError("source label_matrix row_idx is not contiguous")
    record_texts = [str(row.get("unit_text", "")) for row in records]
    if labels["unit_text"].astype(str).tolist() != record_texts:
        raise ValueError("label_matrix and records text order mismatch")

    word_counts = labels["unit_text"].astype(str).map(whitespace_word_count)
    keep_mask = word_counts.ge(minimum_words)
    keep_indices = labels.index[keep_mask].to_list()
    archive_indices = labels.index[~keep_mask].to_list()

    index_frame = pd.DataFrame(
        {
            "original_row_idx": labels.index.astype(int),
            "word_count": word_counts.astype(int),
            "scope": keep_mask.map({True: "statistical_keep", False: "archived_lt_min_words"}),
        }
    )
    index_frame.to_csv(output / "row_scope_index.csv", index=False, encoding="utf-8-sig")

    archive_labels = labels.loc[archive_indices].copy()
    archive_labels.insert(0, "original_row_idx", archive_labels["row_idx"].astype(int))
    archive_labels.insert(1, "word_count", word_counts.loc[archive_indices].astype(int).to_numpy())
    archive_dir = output / f"archive_lt{minimum_words}_words"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_labels.to_csv(archive_dir / "label_matrix_lt5_words.csv", index=False, encoding="utf-8-sig")
    archived_records = []
    for idx in archive_indices:
        row = dict(records[idx])
        row["original_row_idx"] = int(idx)
        row["word_count"] = int(word_counts.iloc[idx])
        archived_records.append(row)
    _write_jsonl(archive_dir / "records_lt5_words.jsonl", archived_records)

    kept_labels = labels.loc[keep_indices].copy()
    kept_labels.insert(0, "original_row_idx", kept_labels["row_idx"].astype(int))
    kept_labels.insert(1, "word_count", word_counts.loc[keep_indices].astype(int).to_numpy())
    kept_labels["row_idx"] = range(len(kept_labels))
    kept_labels.to_csv(output / "label_matrix.csv", index=False, encoding="utf-8-sig")
    kept_records = []
    for idx in keep_indices:
        row = dict(records[idx])
        row["original_row_idx"] = int(idx)
        row["word_count"] = int(word_counts.iloc[idx])
        kept_records.append(row)
    _write_jsonl(output / "records.jsonl", kept_records)

    tensor_outputs: dict[str, str] = {}
    if filter_feature_tensors:
        feature_output = output / "feature_store"
        feature_output.mkdir(parents=True, exist_ok=True)
        index_tensor = torch.tensor(keep_indices, dtype=torch.long)
        for name in ("utterance_features.pt", "utterance_activations.pt"):
            source_path = source / "feature_store" / name
            if not source_path.exists():
                continue
            payload = torch.load(source_path, map_location="cpu", weights_only=True)
            filtered_payload = _filter_feature_payload(
                payload,
                index_tensor=index_tensor,
                expected_rows=len(labels),
                name=name,
            )
            destination = feature_output / name
            torch.save(filtered_payload, destination)
            tensor_outputs[name] = str(destination)
            del payload, filtered_payload

    label_counts_before = {label: int(labels[label].sum()) for label in labels.columns if label in {"RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF", "OTHER"}}
    label_counts_after = {label: int(kept_labels[label].sum()) for label in label_counts_before}
    manifest = {
        "status": "FILTERED_STATISTICAL_SCOPE_READY",
        "source_root": str(source),
        "output_root": str(output),
        "word_count_method": "whitespace_split",
        "minimum_words_inclusive": int(minimum_words),
        "archive_rule": f"word_count < {minimum_words}",
        "source_rows": len(labels),
        "archived_rows": len(archive_indices),
        "retained_rows": len(keep_indices),
        "archived_fraction": len(archive_indices) / len(labels),
        "label_counts_before": label_counts_before,
        "label_counts_after": label_counts_after,
        "feature_tensors": tensor_outputs,
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(
        source / "current_statistical_scope.json",
        {
            "status": "USE_FILTERED_SCOPE_FOR_NEW_STATISTICS",
            "statistical_scope_root": str(output),
            "archive_rule": f"whitespace word_count < {minimum_words}",
            "historical_outputs_unchanged": True,
        },
    )
    report = [
        "# Minimum-word Statistical Scope",
        "",
        f"- Rule: archive utterances with fewer than {minimum_words} whitespace-separated words.",
        f"- Source rows: {len(labels)}",
        f"- Archived rows: {len(archive_indices)} ({len(archive_indices) / len(labels):.2%})",
        f"- Retained rows: {len(keep_indices)} ({len(keep_indices) / len(labels):.2%})",
        "- Historical outputs remain unchanged.",
        "- New statistical analyses should use this directory as their data root.",
    ]
    (output / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return manifest


__all__ = ["build_min_word_statistical_scope", "whitespace_word_count"]
