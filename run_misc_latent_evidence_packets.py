"""Build first-stage SAE latent evidence packets for MISC label review.

The packets are intentionally descriptive: they collect counselor-current
utterance examples around selected SAE latents, but do not interpret the latent
or claim causal mechanism evidence.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


DEFAULT_LABELS = ("RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF")
CONTEXT_NOTE = "Only counselor current utterance is available in phase 1."

FORBIDDEN_BLIND_KEYS = {
    "target_label",
    "label",
    "latent_idx",
    "rank_within_label",
    "cohens_d",
    "directional_auc",
    "precision_at_50",
    "target_match",
    "active_labels",
    "record_id",
    "file_id",
    "source_line",
    "source_split",
    "quality_label",
    "predicted_code",
    "predicted_subcode",
    "confidence",
    "rationale",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _load_feature_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        tensor = obj
    elif isinstance(obj, dict) and isinstance(obj.get("utterance_features"), torch.Tensor):
        tensor = obj["utterance_features"]
    else:
        raise TypeError(f"Could not find utterance feature tensor in {path}")
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D feature tensor, got shape {tuple(tensor.shape)}")
    return tensor.float().cpu()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _normalise_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text


def _escape_md(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _fmt_float(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _record_value(record: dict[str, Any], label_row: pd.Series, *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
        if key in label_row and pd.notna(label_row[key]):
            return label_row[key]
    return default


def _numeric_positive(value: Any) -> bool:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0]
    return float(number) > 0


def _active_labels(label_row: pd.Series, labels: tuple[str, ...]) -> str:
    active: list[str] = []
    for label in labels:
        if label in label_row and _numeric_positive(label_row[label]):
            active.append(label)
    return ",".join(active)


def _ordered_by_activation(activations: np.ndarray, candidates: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or len(candidates) == 0:
        return np.asarray([], dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)
    order = np.lexsort((candidates, -activations[candidates]))
    return candidates[order[: min(limit, len(order))]]


def _packet_seed(random_state: int, label_position: int, latent_idx: int) -> int:
    return int((int(random_state) + 9176 * int(label_position) + 1009 * int(latent_idx)) % (2**32 - 1))


def _sample_random_target(
    target_indices: np.ndarray,
    *,
    exclude_indices: set[int],
    requested: int,
    seed: int,
) -> np.ndarray:
    if requested <= 0:
        return np.asarray([], dtype=np.int64)
    candidates = np.asarray([idx for idx in target_indices.tolist() if int(idx) not in exclude_indices], dtype=np.int64)
    if len(candidates) == 0:
        return candidates
    rng = np.random.default_rng(seed)
    size = min(int(requested), len(candidates))
    sampled = rng.choice(candidates, size=size, replace=False)
    return np.asarray(sampled, dtype=np.int64)


def _normalise_latents(latents: pd.DataFrame, labels: tuple[str, ...]) -> pd.DataFrame:
    label_col = "label" if "label" in latents.columns else "target_label"
    required = {label_col, "latent_idx", "rank_within_label"}
    missing = sorted(required.difference(latents.columns))
    if missing:
        raise ValueError(f"Latents table is missing required columns: {missing}")

    out = latents.copy()
    out["target_label"] = out[label_col].astype(str).str.upper()
    out["latent_idx"] = pd.to_numeric(out["latent_idx"], errors="coerce").fillna(-1).astype(int)
    out["rank_within_label"] = pd.to_numeric(out["rank_within_label"], errors="coerce").fillna(-1).astype(int)
    for col in ("cohens_d", "directional_auc", "precision_at_50"):
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    allowed = set(labels)
    out = out[out["target_label"].isin(allowed)].copy()
    label_order = {label: i for i, label in enumerate(labels)}
    out["_label_order"] = out["target_label"].map(label_order).fillna(len(label_order)).astype(int)
    out = out.sort_values(["_label_order", "rank_within_label", "latent_idx"], kind="mergesort")
    return out.drop(columns=["_label_order"]).reset_index(drop=True)


def _validate_inputs(
    *,
    latents: pd.DataFrame,
    features: torch.Tensor,
    label_matrix: pd.DataFrame,
    records: list[dict[str, Any]],
    labels: tuple[str, ...],
) -> None:
    if len(label_matrix) != features.shape[0]:
        raise ValueError(f"Label matrix rows {len(label_matrix)} do not match feature rows {features.shape[0]}.")
    if len(records) != features.shape[0]:
        raise ValueError(f"Records rows {len(records)} do not match feature rows {features.shape[0]}.")
    missing_labels = [label for label in labels if label not in label_matrix.columns]
    if missing_labels:
        raise ValueError(f"Label matrix is missing labels: {missing_labels}")
    bad_latents = latents[(latents["latent_idx"] < 0) | (latents["latent_idx"] >= features.shape[1])]
    if not bad_latents.empty:
        bad = bad_latents[["target_label", "latent_idx"]].head(5).to_dict(orient="records")
        raise ValueError(f"Found latent_idx outside feature dimension {features.shape[1]}: {bad}")


def _make_labeled_example(
    *,
    packet_id: str,
    target_label: str,
    latent_idx: int,
    rank_within_label: int,
    example_group: str,
    rank_within_group: int,
    row_idx: int,
    activation: float,
    label_matrix: pd.DataFrame,
    records: list[dict[str, Any]],
    labels: tuple[str, ...],
) -> dict[str, Any]:
    label_row = label_matrix.iloc[row_idx]
    record = records[row_idx]
    target_match = int(_numeric_positive(label_row[target_label]))
    unit_text = _record_value(record, label_row, "unit_text", "text")
    normalized_text = _normalise_text(unit_text)
    return {
        "packet_id": packet_id,
        "target_label": target_label,
        "latent_idx": int(latent_idx),
        "rank_within_label": int(rank_within_label),
        "example_group": example_group,
        "rank_within_group": int(rank_within_group),
        "row_idx": int(row_idx),
        "activation": float(activation),
        "target_match": int(target_match),
        "active_labels": _active_labels(label_row, labels),
        "record_id": _record_value(record, label_row, "record_id"),
        "file_id": _record_value(record, label_row, "file_id"),
        "source_line": _record_value(record, label_row, "source_line", default=""),
        "source_split": _record_value(record, label_row, "source_split", default=""),
        "quality_label": _record_value(record, label_row, "quality_label", default=""),
        "predicted_code": _record_value(record, label_row, "predicted_code"),
        "predicted_subcode": _record_value(record, label_row, "predicted_subcode", default=""),
        "confidence": _record_value(record, label_row, "confidence", default=""),
        "unit_text": unit_text,
        "normalized_text": normalized_text,
        "duplicate_text_within_packet": False,
    }


def _add_duplicate_flags(examples: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for example in examples:
        key = str(example.get("normalized_text", ""))
        if key:
            counts[key] = counts.get(key, 0) + 1
    for example in examples:
        key = str(example.get("normalized_text", ""))
        example["duplicate_text_within_packet"] = bool(key and counts.get(key, 0) > 1)


def _count_active_labels(examples: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for example in examples:
        active = str(example.get("active_labels", ""))
        for label in active.split(","):
            label = label.strip()
            if label:
                counts[label] = counts.get(label, 0) + 1
    return ",".join(f"{label}:{counts[label]}" for label in sorted(counts))


def _activation_stats(examples: list[dict[str, Any]]) -> dict[str, float | None]:
    values = np.asarray([float(example["activation"]) for example in examples], dtype=np.float32)
    if len(values) == 0:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def _packet_summary(
    *,
    target_label: str,
    latent_idx: int,
    rank_within_label: int,
    cohens_d: float,
    directional_auc: float,
    precision_at_50: float,
    examples: list[dict[str, Any]],
    requested_counts: dict[str, int],
) -> dict[str, Any]:
    by_group = {
        group: [example for example in examples if example["example_group"] == group]
        for group in ("top_activating", "high_non_target", "random_target")
    }
    top_examples = by_group["top_activating"]
    top_match_rate = (
        float(np.mean([example["target_match"] for example in top_examples])) if top_examples else np.nan
    )
    all_match_rate = float(np.mean([example["target_match"] for example in examples])) if examples else np.nan
    duplicate_rows = sum(1 for example in examples if example["duplicate_text_within_packet"])
    duplicate_keys = {
        example["normalized_text"]
        for example in examples
        if example.get("duplicate_text_within_packet") and example.get("normalized_text")
    }
    files = {str(example.get("file_id", "")) for example in examples if example.get("file_id", "")}
    scarcity_flags = {
        group: len(by_group[group]) < int(requested_counts[group])
        for group in ("top_activating", "high_non_target", "random_target")
    }
    top_stats = _activation_stats(top_examples)
    all_stats = _activation_stats(examples)
    return {
        "target_label": target_label,
        "latent_idx": int(latent_idx),
        "rank_within_label": int(rank_within_label),
        "cohens_d": float(cohens_d) if pd.notna(cohens_d) else np.nan,
        "directional_auc": float(directional_auc) if pd.notna(directional_auc) else np.nan,
        "precision_at_50": float(precision_at_50) if pd.notna(precision_at_50) else np.nan,
        "n_top_activating": len(by_group["top_activating"]),
        "n_high_non_target": len(by_group["high_non_target"]),
        "n_random_target": len(by_group["random_target"]),
        "top_activating_target_match_rate": top_match_rate,
        "all_examples_target_match_rate": all_match_rate,
        "active_label_counts_top_activating": _count_active_labels(top_examples),
        "active_label_counts_all_examples": _count_active_labels(examples),
        "top_activation_min": top_stats["min"],
        "top_activation_max": top_stats["max"],
        "top_activation_mean": top_stats["mean"],
        "all_activation_min": all_stats["min"],
        "all_activation_max": all_stats["max"],
        "all_activation_mean": all_stats["mean"],
        "duplicate_text_row_count": int(duplicate_rows),
        "duplicate_text_unique_count": int(len(duplicate_keys)),
        "unique_file_count": int(len(files)),
        "scarce_top_activating": bool(scarcity_flags["top_activating"]),
        "scarce_high_non_target": bool(scarcity_flags["high_non_target"]),
        "scarce_random_target": bool(scarcity_flags["random_target"]),
    }


def _blind_example(example: dict[str, Any], *, latent_alias: str) -> dict[str, Any]:
    return {
        "packet_id": example["packet_id"],
        "latent_alias": latent_alias,
        "example_group": example["example_group"],
        "rank_within_group": int(example["rank_within_group"]),
        "row_idx": int(example["row_idx"]),
        "activation": float(example["activation"]),
        "unit_text": example["unit_text"],
        "normalized_text": example["normalized_text"],
        "duplicate_text_within_packet": bool(example["duplicate_text_within_packet"]),
    }


def _rationale_row(example: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    record = records[int(example["row_idx"])]
    return {
        "packet_id": example["packet_id"],
        "target_label": example["target_label"],
        "latent_idx": int(example["latent_idx"]),
        "example_group": example["example_group"],
        "rank_within_group": int(example["rank_within_group"]),
        "row_idx": int(example["row_idx"]),
        "record_id": example.get("record_id", ""),
        "rationale": record.get("rationale", ""),
    }


def build_latent_evidence_packets(
    *,
    latents: pd.DataFrame,
    features: torch.Tensor,
    label_matrix: pd.DataFrame,
    records: list[dict[str, Any]],
    labels: tuple[str, ...] = DEFAULT_LABELS,
    top_activating: int = 50,
    high_non_target: int = 20,
    random_target: int = 20,
    random_state: int = 42,
) -> dict[str, Any]:
    label_tuple = tuple(label.upper() for label in labels)
    selected = _normalise_latents(latents, label_tuple)
    _validate_inputs(latents=selected, features=features, label_matrix=label_matrix, records=records, labels=label_tuple)

    n_samples = features.shape[0]
    all_indices = np.arange(n_samples, dtype=np.int64)
    label_positions = {label: i for i, label in enumerate(label_tuple)}
    requested_counts = {
        "top_activating": int(top_activating),
        "high_non_target": int(high_non_target),
        "random_target": int(random_target),
    }

    labeled_packets: list[dict[str, Any]] = []
    blind_packets: list[dict[str, Any]] = []
    labeled_examples: list[dict[str, Any]] = []
    blind_examples: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    rationale_rows: list[dict[str, Any]] = []

    for packet_number, (_, latent) in enumerate(selected.iterrows(), start=1):
        target_label = str(latent["target_label"]).upper()
        latent_idx = int(latent["latent_idx"])
        rank_within_label = int(latent["rank_within_label"])
        packet_id = f"packet_{packet_number:04d}"
        latent_alias = f"latent_{packet_number:04d}"

        activations = features[:, latent_idx].numpy()
        target_mask = label_matrix[target_label].map(_numeric_positive).to_numpy(dtype=bool)
        non_target_indices = all_indices[~target_mask]
        target_indices = all_indices[target_mask]

        top_indices = _ordered_by_activation(activations, all_indices, int(top_activating))
        non_target_top_indices = _ordered_by_activation(activations, non_target_indices, int(high_non_target))
        random_target_indices = _sample_random_target(
            target_indices,
            exclude_indices={int(idx) for idx in top_indices.tolist()},
            requested=int(random_target),
            seed=_packet_seed(int(random_state), label_positions[target_label], latent_idx),
        )

        group_indices = [
            ("top_activating", top_indices),
            ("high_non_target", non_target_top_indices),
            ("random_target", random_target_indices),
        ]

        packet_examples: list[dict[str, Any]] = []
        for group_name, indices in group_indices:
            for rank, row_idx in enumerate(indices.tolist(), start=1):
                packet_examples.append(
                    _make_labeled_example(
                        packet_id=packet_id,
                        target_label=target_label,
                        latent_idx=latent_idx,
                        rank_within_label=rank_within_label,
                        example_group=group_name,
                        rank_within_group=rank,
                        row_idx=int(row_idx),
                        activation=float(activations[int(row_idx)]),
                        label_matrix=label_matrix,
                        records=records,
                        labels=label_tuple,
                    )
                )

        _add_duplicate_flags(packet_examples)
        summary = _packet_summary(
            target_label=target_label,
            latent_idx=latent_idx,
            rank_within_label=rank_within_label,
            cohens_d=float(latent["cohens_d"]),
            directional_auc=float(latent["directional_auc"]),
            precision_at_50=float(latent["precision_at_50"]),
            examples=packet_examples,
            requested_counts=requested_counts,
        )
        summary["packet_id"] = packet_id
        summary["latent_alias"] = latent_alias
        summary_rows.append(summary)

        packet = {
            "packet_id": packet_id,
            "target_label": target_label,
            "latent_idx": latent_idx,
            "rank_within_label": rank_within_label,
            "cohens_d": summary["cohens_d"],
            "directional_auc": summary["directional_auc"],
            "precision_at_50": summary["precision_at_50"],
            "client_context_available": False,
            "context_note": CONTEXT_NOTE,
            "summary": summary,
            "examples": packet_examples,
        }
        labeled_packets.append(packet)
        labeled_examples.extend(packet_examples)
        rationale_rows.extend(_rationale_row(example, records) for example in packet_examples)

        blind_packet_examples = [_blind_example(example, latent_alias=latent_alias) for example in packet_examples]
        blind_packet = {
            "packet_id": packet_id,
            "latent_alias": latent_alias,
            "client_context_available": False,
            "context_note": CONTEXT_NOTE,
            "summary": {
                "n_top_activating": summary["n_top_activating"],
                "n_high_non_target": summary["n_high_non_target"],
                "n_random_target": summary["n_random_target"],
                "duplicate_text_row_count": summary["duplicate_text_row_count"],
                "duplicate_text_unique_count": summary["duplicate_text_unique_count"],
                "scarce_top_activating": summary["scarce_top_activating"],
                "scarce_high_non_target": summary["scarce_high_non_target"],
                "scarce_random_target": summary["scarce_random_target"],
            },
            "examples": blind_packet_examples,
        }
        blind_packets.append(blind_packet)
        blind_examples.extend(blind_packet_examples)

    return {
        "labeled_packets": labeled_packets,
        "blind_packets": blind_packets,
        "labeled_examples": pd.DataFrame(labeled_examples),
        "blind_examples": pd.DataFrame(blind_examples),
        "summary": pd.DataFrame(summary_rows),
        "rationale_appendix": pd.DataFrame(rationale_rows),
    }


def _assert_blind_clean(blind_packets: list[dict[str, Any]], blind_examples: pd.DataFrame) -> None:
    def check_mapping(mapping: dict[str, Any], path: str) -> None:
        overlap = FORBIDDEN_BLIND_KEYS.intersection(mapping)
        if overlap:
            raise AssertionError(f"Blind packet leak at {path}: {sorted(overlap)}")
        for key, value in mapping.items():
            if isinstance(value, dict):
                check_mapping(value, f"{path}.{key}")
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    if isinstance(item, dict):
                        check_mapping(item, f"{path}.{key}[{idx}]")

    for i, packet in enumerate(blind_packets):
        check_mapping(packet, f"blind_packets[{i}]")
    overlap = FORBIDDEN_BLIND_KEYS.intersection(blind_examples.columns)
    if overlap:
        raise AssertionError(f"Blind examples contain forbidden columns: {sorted(overlap)}")


def write_markdown_report(
    packets: list[dict[str, Any]],
    output_path: Path,
    *,
    blind: bool,
    examples_per_group: int = 15,
) -> None:
    title = "SAE latent evidence packets - blind review" if blind else "SAE latent evidence packets - labeled review"
    lines = [
        f"# {title}",
        "",
        "这些 packet 只包含咨询师当前 utterance；第一阶段没有前一句 client utterance。",
        "因此，RES/REC 等上下文依赖标签不能仅凭本材料判断是否真正反映了 client 内容。",
        "本材料用于表征关联审阅，不是因果机制证明。",
        "",
    ]
    if blind:
        lines.append("盲审版不显示 target label、active labels、预测标签、rationale 或来源文件信息。")
        lines.append("")

    for packet in packets:
        summary = packet["summary"]
        if blind:
            heading = f"## {packet['packet_id']} / {packet['latent_alias']}"
        else:
            heading = (
                f"## {packet['packet_id']} - {packet['target_label']} latent {packet['latent_idx']} "
                f"(rank {packet['rank_within_label']})"
            )
        lines.extend([heading, ""])
        if blind:
            lines.extend(
                [
                    f"- top activating examples: {summary['n_top_activating']}",
                    f"- high non-target/contrast examples: {summary['n_high_non_target']}",
                    f"- random target/comparison examples: {summary['n_random_target']}",
                    f"- duplicate text rows: {summary['duplicate_text_row_count']}",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"- Cohen's d: {_fmt_float(packet['cohens_d'])}",
                    f"- directional AUC: {_fmt_float(packet['directional_auc'])}",
                    f"- precision@50: {_fmt_float(packet['precision_at_50'])}",
                    f"- top activating target match rate: {_fmt_float(summary['top_activating_target_match_rate'], 3)}",
                    f"- active label counts in top activating: {_escape_md(summary['active_label_counts_top_activating'])}",
                    f"- duplicate text rows: {summary['duplicate_text_row_count']}",
                    "",
                ]
            )

        for group_name in ("top_activating", "high_non_target", "random_target"):
            group_examples = [ex for ex in packet["examples"] if ex["example_group"] == group_name]
            lines.extend([f"### {group_name}", ""])
            if blind:
                lines.extend(
                    [
                        "| rank | activation | duplicate | text |",
                        "|---:|---:|---|---|",
                    ]
                )
                for example in group_examples[:examples_per_group]:
                    lines.append(
                        "| {rank} | {activation} | {dup} | {text} |".format(
                            rank=int(example["rank_within_group"]),
                            activation=_fmt_float(example["activation"]),
                            dup=str(bool(example["duplicate_text_within_packet"])),
                            text=_escape_md(example["unit_text"]),
                        )
                    )
            else:
                lines.extend(
                    [
                        "| rank | activation | target_match | active_labels | duplicate | text |",
                        "|---:|---:|---:|---|---|---|",
                    ]
                )
                for example in group_examples[:examples_per_group]:
                    lines.append(
                        "| {rank} | {activation} | {target} | {labels} | {dup} | {text} |".format(
                            rank=int(example["rank_within_group"]),
                            activation=_fmt_float(example["activation"]),
                            target=int(example["target_match"]),
                            labels=_escape_md(example["active_labels"]),
                            dup=str(bool(example["duplicate_text_within_packet"])),
                            text=_escape_md(example["unit_text"]),
                        )
                    )
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_latent_evidence_packet_export(
    *,
    latents_path: str | Path,
    feature_store_path: str | Path,
    label_matrix_path: str | Path,
    records_path: str | Path,
    output_dir: str | Path,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    top_activating: int = 50,
    high_non_target: int = 20,
    random_target: int = 20,
    random_state: int = 42,
    markdown_examples_per_group: int = 15,
) -> dict[str, Any]:
    latents_path = Path(latents_path)
    feature_store_path = Path(feature_store_path)
    label_matrix_path = Path(label_matrix_path)
    records_path = Path(records_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    latents = pd.read_csv(latents_path)
    features = _load_feature_tensor(feature_store_path)
    label_matrix = pd.read_csv(label_matrix_path)
    records = _read_jsonl(records_path)
    label_tuple = tuple(label.upper() for label in labels)

    built = build_latent_evidence_packets(
        latents=latents,
        features=features,
        label_matrix=label_matrix,
        records=records,
        labels=label_tuple,
        top_activating=top_activating,
        high_non_target=high_non_target,
        random_target=random_target,
        random_state=random_state,
    )
    _assert_blind_clean(built["blind_packets"], built["blind_examples"])

    paths = {
        "labeled_packets_jsonl": output_path / "latent_evidence_packets_labeled.jsonl",
        "blind_packets_jsonl": output_path / "latent_evidence_packets_blind.jsonl",
        "labeled_examples_csv": output_path / "latent_evidence_examples_labeled.csv",
        "blind_examples_csv": output_path / "latent_evidence_examples_blind.csv",
        "labeled_markdown": output_path / "latent_evidence_packets_labeled.md",
        "blind_markdown": output_path / "latent_evidence_packets_blind.md",
        "rationale_appendix_csv": output_path / "latent_evidence_rationale_appendix.csv",
        "summary_csv": output_path / "latent_evidence_packet_summary.csv",
        "manifest_json": output_path / "manifest.json",
    }

    _write_jsonl(paths["labeled_packets_jsonl"], built["labeled_packets"])
    _write_jsonl(paths["blind_packets_jsonl"], built["blind_packets"])
    built["labeled_examples"].to_csv(paths["labeled_examples_csv"], index=False)
    built["blind_examples"].to_csv(paths["blind_examples_csv"], index=False)
    built["summary"].to_csv(paths["summary_csv"], index=False)
    built["rationale_appendix"].to_csv(paths["rationale_appendix_csv"], index=False)
    write_markdown_report(
        built["labeled_packets"],
        paths["labeled_markdown"],
        blind=False,
        examples_per_group=markdown_examples_per_group,
    )
    write_markdown_report(
        built["blind_packets"],
        paths["blind_markdown"],
        blind=True,
        examples_per_group=markdown_examples_per_group,
    )

    manifest = {
        "analysis": "misc_sae_latent_evidence_packets_phase1",
        "labels": list(label_tuple),
        "client_context_available": False,
        "context_note": CONTEXT_NOTE,
        "selection_policy": {
            "source": "top20_positive_cohens_d_latents",
            "top_activating": int(top_activating),
            "high_non_target": int(high_non_target),
            "random_target": int(random_target),
            "random_target_policy": "same target label positives, excluding top_activating rows",
            "random_state": int(random_state),
            "duplicate_policy": "keep examples and flag normalized exact duplicate text within packet",
            "active_labels_policy": "core labels only; OTHER is excluded",
            "rationale_policy": "appendix only",
        },
        "inputs": {
            "latents": str(latents_path),
            "feature_store": str(feature_store_path),
            "label_matrix": str(label_matrix_path),
            "records": str(records_path),
        },
        "outputs": {key: str(value) for key, value in paths.items()},
        "n_packets": int(len(built["labeled_packets"])),
        "n_labeled_example_rows": int(len(built["labeled_examples"])),
        "n_blind_example_rows": int(len(built["blind_examples"])),
        "scarcity": {
            "top_activating": int(built["summary"]["scarce_top_activating"].sum()),
            "high_non_target": int(built["summary"]["scarce_high_non_target"].sum()),
            "random_target": int(built["summary"]["scarce_random_target"].sum()),
        },
    }
    paths["manifest_json"].write_text(json.dumps(_jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate first-stage MISC SAE latent evidence packets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--latents",
        default=(
            "outputs/misc_full_sae_eval/interpretability/top20_cohensd_latent_utterances/"
            "top20_cohensd_latents_by_label.csv"
        ),
    )
    parser.add_argument("--feature-store", default="outputs/misc_full_sae_eval/feature_store/utterance_features.pt")
    parser.add_argument("--label-matrix", default="outputs/misc_full_sae_eval/label_matrix.csv")
    parser.add_argument("--records", default="outputs/misc_full_sae_eval/records.jsonl")
    parser.add_argument(
        "--output-dir",
        default="outputs/misc_full_sae_eval/interpretability/top20_cohensd_latent_utterances/latent_evidence_packets",
    )
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--top-activating", type=int, default=50)
    parser.add_argument("--high-non-target", type=int, default=20)
    parser.add_argument("--random-target", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--markdown-examples-per-group", type=int, default=15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_latent_evidence_packet_export(
        latents_path=args.latents,
        feature_store_path=args.feature_store,
        label_matrix_path=args.label_matrix,
        records_path=args.records,
        output_dir=args.output_dir,
        labels=tuple(label.upper() for label in args.labels),
        top_activating=args.top_activating,
        high_non_target=args.high_non_target,
        random_target=args.random_target,
        random_state=args.random_state,
        markdown_examples_per_group=args.markdown_examples_per_group,
    )
    print("Completed latent evidence packet export.")
    print(f"Output dir: {args.output_dir}")
    print(f"Packets: {manifest['n_packets']}")
    print(f"Labeled example rows: {manifest['n_labeled_example_rows']}")
    print(f"Blind example rows: {manifest['n_blind_example_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
