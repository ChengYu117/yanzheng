"""Export top-3 strongest SAE latents per MISC label and their top utterances.

This script is intentionally focused on the Llama main result by default. It
does not recompute model activations; it reads the existing latent association
tables and feature store, then prepares a human-review friendly table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


DEFAULT_LABELS = ("RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF")
PARENT_LABELS = {"RE", "QU"}

SORT_COLUMNS = [
    "latent_label_weight",
    "precision_lift_absolute_at_50",
    "directional_auc",
    "abs_cohens_d",
    "association_rank",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


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


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _normalise_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "label" not in out.columns or "latent_idx" not in out.columns:
        raise ValueError("Candidate tables must contain `label` and `latent_idx` columns.")
    out["label"] = out["label"].astype(str).str.upper()
    out["latent_idx"] = pd.to_numeric(out["latent_idx"], errors="coerce").fillna(-1).astype(int)
    for col in SORT_COLUMNS + [
        "precision_at_50",
        "precision_at_100",
        "cohens_d",
        "mean_diff",
        "precision_lift_at_50",
        "formal_edge_weight",
    ]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["stable_edge", "positive_support", "negative_boundary", "significant_fdr"]:
        if col not in out.columns:
            out[col] = False
        out[col] = out[col].map(_truthy)
    if "edge_type" not in out.columns:
        out["edge_type"] = "unknown"
    return out


def _sort_strongest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    for col in SORT_COLUMNS[:-1]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(-np.inf)
    out["association_rank"] = pd.to_numeric(out["association_rank"], errors="coerce").fillna(np.inf)
    return out.sort_values(
        by=SORT_COLUMNS,
        ascending=[False, False, False, False, True],
        kind="mergesort",
    )


def _label_role(label: str) -> str:
    return "parent_consistency_only" if label in PARENT_LABELS else "leaf_or_atomic"


def select_top3_latents(
    association: pd.DataFrame,
    thresholded: pd.DataFrame,
    *,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    top_n: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    assoc = _normalise_candidates(association)
    thresh = _normalise_candidates(thresholded)
    selected_rows: list[dict[str, Any]] = []
    audit_labels: dict[str, Any] = {}

    for label in labels:
        label = label.upper()
        stable = thresh[
            (thresh["label"] == label)
            & (thresh["latent_idx"] >= 0)
            & (thresh["stable_edge"])
            & (thresh["positive_support"])
        ]
        stable = _sort_strongest(stable).drop_duplicates("latent_idx", keep="first")
        stable_count = int(len(stable))
        chosen = stable.head(top_n).copy()
        chosen["selection_source"] = "stable_positive"

        needed = max(top_n - len(chosen), 0)
        fallback_count = 0
        if needed:
            already = set(chosen["latent_idx"].astype(int).tolist())
            fallback = assoc[
                (assoc["label"] == label)
                & (assoc["latent_idx"] >= 0)
                & (~assoc["negative_boundary"])
                & (~assoc["latent_idx"].isin(already))
            ].copy()
            preferred = fallback[
                (pd.to_numeric(fallback["cohens_d"], errors="coerce").fillna(0.0) > 0)
                | (pd.to_numeric(fallback["mean_diff"], errors="coerce").fillna(0.0) > 0)
            ]
            nonpreferred = fallback.drop(preferred.index, errors="ignore")
            fallback_sorted = pd.concat(
                [_sort_strongest(preferred), _sort_strongest(nonpreferred)],
                ignore_index=True,
            ).drop_duplicates("latent_idx", keep="first")
            fill = fallback_sorted.head(needed).copy()
            fallback_count = int(len(fill))
            fill["selection_source"] = "fallback_top_association"
            chosen = pd.concat([chosen, fill], ignore_index=True)

        if len(chosen) < top_n:
            raise ValueError(f"Could not select {top_n} latents for label {label}; got {len(chosen)}.")

        note = (
            f"stable_positive_count={stable_count}; filled_by_top_association"
            if stable_count < top_n
            else f"stable_positive_count={stable_count}; selected_from_stable_positive"
        )
        audit_labels[label] = {
            "label_role": _label_role(label),
            "stable_positive_count": stable_count,
            "fallback_count": fallback_count,
            "selected_latents": [int(x) for x in chosen["latent_idx"].head(top_n).tolist()],
        }

        for rank, (_, row) in enumerate(chosen.head(top_n).iterrows(), start=1):
            selected_rows.append(
                {
                    "label": label,
                    "label_role": _label_role(label),
                    "rank_within_label": rank,
                    "latent_idx": int(row["latent_idx"]),
                    "selection_source": str(row["selection_source"]),
                    "selection_note": note,
                    "edge_type": str(row.get("edge_type", "unknown")),
                    "stable_edge": bool(_truthy(row.get("stable_edge", False))),
                    "positive_support": bool(_truthy(row.get("positive_support", False))),
                    "association_rank": float(row.get("association_rank", np.nan)),
                    "latent_label_weight": float(row.get("latent_label_weight", np.nan)),
                    "directional_auc": float(row.get("directional_auc", np.nan)),
                    "abs_cohens_d": float(row.get("abs_cohens_d", np.nan)),
                    "precision_at_50": float(row.get("precision_at_50", np.nan)),
                    "precision_lift_absolute_at_50": float(
                        row.get("precision_lift_absolute_at_50", row.get("precision_lift_at_50", np.nan))
                    ),
                    "precision_at_100": float(row.get("precision_at_100", np.nan)),
                }
            )
    return pd.DataFrame(selected_rows), audit_labels


def _active_labels(label_row: pd.Series, labels: tuple[str, ...]) -> str:
    active = []
    for label in labels:
        if label in label_row and pd.to_numeric(pd.Series([label_row[label]]), errors="coerce").fillna(0).iloc[0] > 0:
            active.append(label)
    return ",".join(active)


def _record_value(record: dict[str, Any], row: pd.Series, *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
        if key in row and pd.notna(row[key]):
            return row[key]
    return default


def build_top_utterances(
    top_latents: pd.DataFrame,
    features: torch.Tensor,
    label_matrix: pd.DataFrame,
    records: list[dict[str, Any]],
    *,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    top_k: int = 20,
) -> pd.DataFrame:
    if len(label_matrix) != features.shape[0]:
        raise ValueError(f"Label matrix rows {len(label_matrix)} do not match feature rows {features.shape[0]}.")
    if len(records) != features.shape[0]:
        raise ValueError(f"Records rows {len(records)} do not match feature rows {features.shape[0]}.")

    rows: list[dict[str, Any]] = []
    for _, latent in top_latents.iterrows():
        label = str(latent["label"]).upper()
        latent_idx = int(latent["latent_idx"])
        if latent_idx < 0 or latent_idx >= features.shape[1]:
            raise ValueError(f"latent_idx {latent_idx} out of feature dimension {features.shape[1]}.")
        acts = features[:, latent_idx]
        k = min(top_k, int(acts.numel()))
        values, indices = torch.topk(acts, k=k, largest=True, sorted=True)
        for utterance_rank, (value, idx_tensor) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            idx = int(idx_tensor)
            label_row = label_matrix.iloc[idx]
            record = records[idx]
            target_value = label_row[label] if label in label_row else 0
            target_match = int(pd.to_numeric(pd.Series([target_value]), errors="coerce").fillna(0).iloc[0] > 0)
            rows.append(
                {
                    "label": label,
                    "latent_idx": latent_idx,
                    "latent_rank_within_label": int(latent["rank_within_label"]),
                    "utterance_rank": utterance_rank,
                    "row_idx": idx,
                    "activation": float(value),
                    "target_match": target_match,
                    "active_labels": _active_labels(label_row, labels),
                    "record_id": _record_value(record, label_row, "record_id"),
                    "file_id": _record_value(record, label_row, "file_id"),
                    "source_line": _record_value(record, label_row, "source_line", default=""),
                    "quality_label": _record_value(record, label_row, "quality_label", "source_split"),
                    "text": _record_value(record, label_row, "text", "unit_text"),
                }
            )
    return pd.DataFrame(rows)


def _fmt_float(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _escape_md(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def write_report(
    top_latents: pd.DataFrame,
    top_utterances: pd.DataFrame,
    audit: dict[str, Any],
    output_path: Path,
) -> None:
    lines = [
        "# Llama 主结果：每标签 Top3 最强 SAE latents 与 Top20 激活语句",
        "",
        "本报告用于人工审批和语义一致性验证。它只导出 Llama 主结果，不覆盖 Gemma。",
        "",
        "## 最强 latent 判定口径",
        "",
        "本报告采用 Strict-first 综合口径：",
        "",
        "1. 每个标签优先选择 `stable_edge=True` 且 `positive_support=True` 的 stable positive latents。",
        "2. stable positive 候选按 `latent_label_weight`、`precision_lift_absolute_at_50`、`directional_auc`、`abs_cohens_d` 降序，再按 `association_rank` 升序排序。",
        "3. 如果某标签 stable positive 不足 3 个，则从 `latent_label_association_v2.csv` 的 Top association 候选中补齐，并标记为 `fallback_top_association`。",
        "4. 本报告中的 top20 utterances 只展示当前 utterance 本句，不附带局部上下文。",
        "",
        "## fallback 审计",
        "",
        "| 标签 | 角色 | stable positive 数 | fallback 数 | selected latents |",
        "|---|---|---:|---:|---|",
    ]
    for label, info in audit["labels"].items():
        lines.append(
            "| {label} | {role} | {stable} | {fallback} | {latents} |".format(
                label=label,
                role=info["label_role"],
                stable=info["stable_positive_count"],
                fallback=info["fallback_count"],
                latents=",".join(str(x) for x in info["selected_latents"]),
            )
        )

    lines.extend(["", "## Top3 latents by label", ""])
    for label, group in top_latents.groupby("label", sort=False):
        lines.extend(
            [
                f"### {label}",
                "",
                "| rank | latent | source | edge type | stable | weight | AUC | abs d | P@50 | note |",
                "|---:|---:|---|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for _, row in group.iterrows():
            lines.append(
                "| {rank} | {latent} | {source} | {edge} | {stable} | {weight} | {auc} | {d} | {p50} | {note} |".format(
                    rank=int(row["rank_within_label"]),
                    latent=int(row["latent_idx"]),
                    source=_escape_md(row["selection_source"]),
                    edge=_escape_md(row["edge_type"]),
                    stable=str(bool(row["stable_edge"])),
                    weight=_fmt_float(row["latent_label_weight"]),
                    auc=_fmt_float(row["directional_auc"]),
                    d=_fmt_float(row["abs_cohens_d"]),
                    p50=_fmt_float(row["precision_at_50"]),
                    note=_escape_md(row["selection_note"]),
                )
            )
        lines.append("")
        for _, row in group.iterrows():
            latent_idx = int(row["latent_idx"])
            examples = top_utterances[
                (top_utterances["label"] == label) & (top_utterances["latent_idx"] == latent_idx)
            ]
            lines.extend(
                [
                    f"#### {label} latent {latent_idx} top20 激活语句",
                    "",
                    "| rank | activation | target | active labels | record | text |",
                    "|---:|---:|---:|---|---|---|",
                ]
            )
            for _, ex in examples.iterrows():
                lines.append(
                    "| {rank} | {activation} | {target} | {labels} | {record} | {text} |".format(
                        rank=int(ex["utterance_rank"]),
                        activation=_fmt_float(ex["activation"], 4),
                        target=int(ex["target_match"]),
                        labels=_escape_md(ex["active_labels"]),
                        record=_escape_md(ex["record_id"]),
                        text=_escape_md(ex["text"]),
                    )
                )
            lines.append("")

    lines.extend(
        [
            "## 解释边界",
            "",
            "- `stable_positive` 可以作为强语义候选优先审查。",
            "- `fallback_top_association` 只是为了保证每个标签都有 3 个候选，不应写成强 stable latent。",
            "- 本报告展示的是激活样例和统计关联，不构成因果机制证明。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_top3_latent_utterance_export(
    *,
    association_path: str | Path,
    thresholded_path: str | Path,
    feature_store_path: str | Path,
    label_matrix_path: str | Path,
    records_path: str | Path,
    output_dir: str | Path,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    top_latents: int = 3,
    top_utterances: int = 20,
) -> dict[str, Any]:
    association_path = Path(association_path)
    thresholded_path = Path(thresholded_path)
    feature_store_path = Path(feature_store_path)
    label_matrix_path = Path(label_matrix_path)
    records_path = Path(records_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    association = pd.read_csv(association_path)
    thresholded = pd.read_csv(thresholded_path)
    features = _load_feature_tensor(feature_store_path)
    label_matrix = pd.read_csv(label_matrix_path)
    records = _read_jsonl(records_path)

    selected, label_audit = select_top3_latents(
        association,
        thresholded,
        labels=tuple(label.upper() for label in labels),
        top_n=top_latents,
    )
    utterances = build_top_utterances(
        selected,
        features,
        label_matrix,
        records,
        labels=tuple(label.upper() for label in labels),
        top_k=top_utterances,
    )

    selected_path = output_path / "top3_latents_by_label.csv"
    utterances_path = output_path / "top20_utterances_by_top3_latents.csv"
    report_path = output_path / "top3_latent_utterance_report.md"
    audit_path = output_path / "strength_metric_audit.json"

    selected.to_csv(selected_path, index=False)
    utterances.to_csv(utterances_path, index=False)
    audit = {
        "analysis": "misc_top3_latent_utterances",
        "model_scope": "llama_main_result",
        "labels": label_audit,
        "ranking_rule": SORT_COLUMNS,
        "selection_policy": {
            "top_latents_per_label": int(top_latents),
            "top_utterances_per_latent": int(top_utterances),
            "primary_filter": "stable_edge == True and positive_support == True and latent_idx >= 0",
            "fallback_filter": "negative_boundary != True and latent_idx >= 0; prefer cohens_d > 0 or mean_diff > 0",
        },
        "inputs": {
            "association": str(association_path),
            "thresholded": str(thresholded_path),
            "feature_store": str(feature_store_path),
            "label_matrix": str(label_matrix_path),
            "records": str(records_path),
        },
        "outputs": {
            "top3_latents_by_label": str(selected_path),
            "top20_utterances_by_top3_latents": str(utterances_path),
            "report": str(report_path),
            "audit": str(audit_path),
        },
        "n_selected_latents": int(len(selected)),
        "n_top_utterance_rows": int(len(utterances)),
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(selected, utterances, audit, report_path)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Llama MISC top-3 strongest SAE latents and top-20 activating utterances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--association",
        default="outputs/misc_full_sae_eval/interpretability/latent_space_search_v2/latent_label_association_v2.csv",
    )
    parser.add_argument(
        "--thresholded",
        default="outputs/misc_full_sae_eval/interpretability/latent_space_search_v2/thresholded_latent_sets_v2.csv",
    )
    parser.add_argument("--feature-store", default="outputs/misc_full_sae_eval/feature_store/utterance_features.pt")
    parser.add_argument("--label-matrix", default="outputs/misc_full_sae_eval/label_matrix.csv")
    parser.add_argument("--records", default="outputs/misc_full_sae_eval/records.jsonl")
    parser.add_argument(
        "--output-dir",
        default="outputs/misc_full_sae_eval/interpretability/top3_latent_utterances",
    )
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--top-latents", type=int, default=3)
    parser.add_argument("--top-utterances", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = run_top3_latent_utterance_export(
        association_path=args.association,
        thresholded_path=args.thresholded,
        feature_store_path=args.feature_store,
        label_matrix_path=args.label_matrix,
        records_path=args.records,
        output_dir=args.output_dir,
        labels=tuple(label.upper() for label in args.labels),
        top_latents=args.top_latents,
        top_utterances=args.top_utterances,
    )
    print("Completed Top3 latent + Top20 utterance export.")
    print(f"Output dir: {args.output_dir}")
    print(f"Selected latents: {audit['n_selected_latents']}")
    print(f"Top utterance rows: {audit['n_top_utterance_rows']}")
    print(f"Report: {audit['outputs']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
