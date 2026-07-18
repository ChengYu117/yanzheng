"""Build a row-aligned statistical dataset with normalized exact duplicates removed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .cross_val_framework import normalize_text


CORE_LABELS = ("RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF", "OTHER")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _label_signature(row: pd.Series) -> str:
    code = str(row.get("predicted_code", ""))
    subcode = row.get("predicted_subcode", "")
    if pd.isna(subcode) or not str(subcode).strip():
        return code
    return f"{code}/{subcode}"


def _filter_feature_payload(
    payload: Any,
    *,
    index_tensor: torch.Tensor,
    expected_rows: int,
    name: str,
) -> Any:
    if isinstance(payload, torch.Tensor):
        if payload.ndim < 1 or int(payload.shape[0]) != expected_rows:
            raise ValueError(f"{name} row count does not match {expected_rows}")
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


def build_deduplicated_statistical_scope(
    *,
    source_root: str | Path,
    output_root: str | Path,
    filter_feature_tensors: bool = True,
) -> dict[str, Any]:
    """Keep one highest-confidence row for each normalized exact utterance."""

    source = Path(source_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(source / "label_matrix.csv")
    records = _read_jsonl(source / "records.jsonl")
    if len(labels) != len(records):
        raise ValueError(f"label/record length mismatch: {len(labels)} != {len(records)}")
    if "row_idx" not in labels.columns or labels["row_idx"].astype(int).tolist() != list(range(len(labels))):
        raise ValueError("source label_matrix row_idx is not contiguous")
    record_texts = [str(row.get("unit_text", "")) for row in records]
    if labels["unit_text"].astype(str).tolist() != record_texts:
        raise ValueError("label_matrix and records text order mismatch")

    audit = pd.DataFrame(
        {
            "pre_dedup_row_idx": labels["row_idx"].astype(int),
            "original_row_idx": labels.get("original_row_idx", labels["row_idx"]).astype(int),
            "record_id": labels["record_id"].astype(str),
            "source_split": labels["source_split"].astype(str),
            "source_file": labels["source_file"].astype(str),
            "unit_text": labels["unit_text"].astype(str),
            "normalized_text": labels["unit_text"].map(normalize_text),
            "confidence": pd.to_numeric(labels["confidence"], errors="coerce").fillna(-1.0),
        }
    )
    audit["label_signature"] = labels.apply(_label_signature, axis=1)
    audit["dedup_group_id"] = audit.groupby("normalized_text", sort=True).ngroup().astype(int)
    audit["duplicate_count"] = audit.groupby("dedup_group_id")["dedup_group_id"].transform("size").astype(int)
    audit["label_signature_count"] = audit.groupby("dedup_group_id")["label_signature"].transform("nunique").astype(int)
    ranked = audit.sort_values(
        ["dedup_group_id", "confidence", "pre_dedup_row_idx"],
        ascending=[True, False, True],
        kind="stable",
    )
    representative_by_group = ranked.drop_duplicates("dedup_group_id", keep="first").set_index("dedup_group_id")
    audit["representative_pre_dedup_row_idx"] = audit["dedup_group_id"].map(
        representative_by_group["pre_dedup_row_idx"]
    ).astype(int)
    audit["representative_record_id"] = audit["dedup_group_id"].map(
        representative_by_group["record_id"]
    ).astype(str)
    audit["keep"] = audit["pre_dedup_row_idx"].eq(audit["representative_pre_dedup_row_idx"])
    audit = audit.sort_values("pre_dedup_row_idx", kind="stable").reset_index(drop=True)

    keep_indices = audit.loc[audit["keep"], "pre_dedup_row_idx"].astype(int).sort_values().tolist()
    removed = audit.loc[~audit["keep"]].copy()
    conflict_groups = audit.loc[
        audit["duplicate_count"].gt(1) & audit["label_signature_count"].gt(1)
    ].copy()
    audit.to_csv(output / "dedup_mapping.csv", index=False, encoding="utf-8-sig")
    removed.to_csv(output / "removed_duplicates.csv", index=False, encoding="utf-8-sig")
    conflict_groups.to_csv(output / "conflicting_duplicate_groups.csv", index=False, encoding="utf-8-sig")

    kept_labels = labels.iloc[keep_indices].copy()
    kept_labels.insert(0, "pre_dedup_row_idx", kept_labels["row_idx"].astype(int))
    kept_labels["row_idx"] = range(len(kept_labels))
    kept_labels.to_csv(output / "label_matrix.csv", index=False, encoding="utf-8-sig")
    kept_records: list[dict[str, Any]] = []
    for new_row_idx, old_row_idx in enumerate(keep_indices):
        record = dict(records[old_row_idx])
        record["pre_dedup_row_idx"] = int(old_row_idx)
        record["row_idx"] = int(new_row_idx)
        kept_records.append(record)
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

    label_counts_before = {label: int(labels[label].sum()) for label in CORE_LABELS if label in labels}
    label_counts_after = {label: int(kept_labels[label].sum()) for label in label_counts_before}
    manifest = {
        "status": "DEDUPLICATED_STATISTICAL_SCOPE_READY",
        "source_root": str(source),
        "output_root": str(output),
        "normalization": "strip + lowercase + collapse_whitespace; punctuation preserved",
        "representative_policy": "highest confidence, then earliest pre-dedup row",
        "source_rows": int(len(labels)),
        "retained_rows": int(len(kept_labels)),
        "removed_duplicate_rows": int(len(removed)),
        "duplicate_groups": int(audit.loc[audit["duplicate_count"].gt(1), "dedup_group_id"].nunique()),
        "conflicting_duplicate_groups": int(conflict_groups["dedup_group_id"].nunique()),
        "conflicting_duplicate_rows": int(len(conflict_groups)),
        "label_counts_before": label_counts_before,
        "label_counts_after": label_counts_after,
        "feature_tensors": tensor_outputs,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Deduplicated Minimum-word Statistical Scope",
        "",
        f"- Source: `{source}`",
        f"- Source rows: {len(labels)}",
        f"- Retained rows: {len(kept_labels)}",
        f"- Removed duplicate rows: {len(removed)}",
        f"- Duplicate groups: {manifest['duplicate_groups']}",
        f"- Conflicting-label duplicate groups: {manifest['conflicting_duplicate_groups']}",
        "- Normalization: lowercase, trim, and collapse whitespace; punctuation is preserved.",
        "- Representative: highest confidence, then earliest source row.",
    ]
    (output / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return manifest


__all__ = ["build_deduplicated_statistical_scope"]
