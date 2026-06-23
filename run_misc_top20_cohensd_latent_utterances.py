"""Export top positive-Cohen's-d SAE latents and their strongest utterances.

This script is a human-review export for the saved MISC SAE evaluation.  It does
not recompute activations; it reads the existing latent-label association table
and utterance-level SAE feature store, then writes compact CSV artifacts plus a
Markdown review document.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


DEFAULT_LABELS = ("RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF")

NUMERIC_ASSOCIATION_COLUMNS = (
    "n_positive",
    "n_negative",
    "prevalence",
    "pos_mean",
    "neg_mean",
    "mean_diff",
    "cohens_d",
    "abs_cohens_d",
    "auc",
    "directional_auc",
    "auc_effect",
    "p_value",
    "precision_at_10",
    "precision_lift_at_10",
    "precision_at_50",
    "precision_lift_at_50",
)


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


def _normalise_association_table(df: pd.DataFrame) -> pd.DataFrame:
    required = {"label", "latent_idx", "cohens_d"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Association table is missing required columns: {missing}")

    out = df.copy()
    out["label"] = out["label"].astype(str).str.upper()
    out["latent_idx"] = pd.to_numeric(out["latent_idx"], errors="coerce").fillna(-1).astype(int)
    for col in NUMERIC_ASSOCIATION_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "significant_fdr" not in out.columns:
        out["significant_fdr"] = False
    out["significant_fdr"] = out["significant_fdr"].map(_truthy)
    return out


def select_top_positive_cohensd_latents(
    association: pd.DataFrame,
    *,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    top_features: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if top_features <= 0:
        raise ValueError("top_features must be positive.")

    assoc = _normalise_association_table(association)
    selected_rows: list[dict[str, Any]] = []
    audit_labels: dict[str, Any] = {}

    for label in labels:
        label = label.upper()
        label_rows = assoc[
            (assoc["label"] == label)
            & (assoc["latent_idx"] >= 0)
            & (assoc["cohens_d"] > 0)
        ].copy()
        label_rows = label_rows.drop_duplicates("latent_idx", keep="first")
        label_rows = label_rows.sort_values(
            by=["cohens_d", "directional_auc", "precision_at_50", "latent_idx"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )

        if len(label_rows) < top_features:
            raise ValueError(
                f"Label {label} has only {len(label_rows)} positive-Cohen's-d latents; "
                f"expected at least {top_features}."
            )

        chosen = label_rows.head(top_features).copy()
        audit_labels[label] = {
            "positive_cohens_d_candidate_count": int(len(label_rows)),
            "selected_latents": [int(x) for x in chosen["latent_idx"].tolist()],
            "cohens_d_min": float(chosen["cohens_d"].min()),
            "cohens_d_max": float(chosen["cohens_d"].max()),
            "directional_auc_mean": float(chosen["directional_auc"].mean()),
            "precision_at_50_mean": float(chosen["precision_at_50"].mean()),
        }

        for rank, (_, row) in enumerate(chosen.iterrows(), start=1):
            item = {
                "label": label,
                "rank_within_label": rank,
                "latent_idx": int(row["latent_idx"]),
                "selection_source": "positive_cohens_d",
            }
            for col in NUMERIC_ASSOCIATION_COLUMNS:
                value = row.get(col, np.nan)
                item[col] = float(value) if pd.notna(value) else np.nan
            item["significant_fdr"] = bool(_truthy(row.get("significant_fdr", False)))
            selected_rows.append(item)

    return pd.DataFrame(selected_rows), audit_labels


def _active_labels(label_row: pd.Series, labels: tuple[str, ...]) -> str:
    active: list[str] = []
    for label in labels:
        if label in label_row:
            value = pd.to_numeric(pd.Series([label_row[label]]), errors="coerce").fillna(0).iloc[0]
            if value > 0:
                active.append(label)
    return ",".join(active)


def _record_value(record: dict[str, Any], row: pd.Series, *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
        if key in row and pd.notna(row[key]):
            return row[key]
    return default


def build_top_activating_utterances(
    top_latents: pd.DataFrame,
    features: torch.Tensor,
    label_matrix: pd.DataFrame,
    records: list[dict[str, Any]],
    *,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    top_utterances: int = 50,
) -> pd.DataFrame:
    if top_utterances <= 0:
        raise ValueError("top_utterances must be positive.")
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
        k = min(top_utterances, int(acts.numel()))
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
                    "predicted_code": _record_value(record, label_row, "predicted_code"),
                    "predicted_subcode": _record_value(record, label_row, "predicted_subcode"),
                    "confidence": _record_value(record, label_row, "confidence", default=""),
                    "unit_text": _record_value(record, label_row, "unit_text", "text"),
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
        "# 每标签 Top20 正向 Cohen's d SAE features 与 Top50 高激活语句",
        "",
        "本报告用于人工审阅每个 MISC 标签对应的正向 SAE feature 候选及其最高激活语句。",
        "",
        "## 判定口径",
        "",
        "- Feature 排名按正向 `cohens_d` 降序；不使用 `abs_cohens_d`，因此反向边界 feature 不会被选为目标标签正向 feature。",
        "- 高激活句子来自全数据集，不只限于目标标签正例。",
        "- `target_match=1` 表示该句带有当前目标标签；`target_match=0` 表示 feature 在非目标标签句子上也强激活。",
        "- 本报告展示的是表征关联和人工审阅材料，不是因果机制证明。",
        "",
        "## 标签概览",
        "",
        "| label | selected features | positive candidates | Cohen's d range | mean directional AUC | mean P@50 | target_match rate in top utterances |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for label, info in audit["labels"].items():
        rows = top_utterances[top_utterances["label"] == label]
        match_rate = float(rows["target_match"].mean()) if not rows.empty else np.nan
        lines.append(
            "| {label} | {n_selected} | {n_candidates} | {d_min}-{d_max} | {auc} | {p50} | {match_rate} |".format(
                label=label,
                n_selected=len(info["selected_latents"]),
                n_candidates=info["positive_cohens_d_candidate_count"],
                d_min=_fmt_float(info["cohens_d_min"]),
                d_max=_fmt_float(info["cohens_d_max"]),
                auc=_fmt_float(info["directional_auc_mean"]),
                p50=_fmt_float(info["precision_at_50_mean"]),
                match_rate=_fmt_float(match_rate),
            )
        )

    for label, group in top_latents.groupby("label", sort=False):
        lines.extend(
            [
                "",
                f"## {label}",
                "",
                "| rank | latent_idx | Cohen's d | AUC | directional AUC | P@10 | P@50 | p value | FDR significant |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for _, row in group.iterrows():
            lines.append(
                "| {rank} | {latent} | {d} | {auc} | {dauc} | {p10} | {p50} | {pvalue} | {sig} |".format(
                    rank=int(row["rank_within_label"]),
                    latent=int(row["latent_idx"]),
                    d=_fmt_float(row["cohens_d"]),
                    auc=_fmt_float(row["auc"]),
                    dauc=_fmt_float(row["directional_auc"]),
                    p10=_fmt_float(row["precision_at_10"]),
                    p50=_fmt_float(row["precision_at_50"]),
                    pvalue=_fmt_float(row["p_value"], digits=3),
                    sig=str(bool(row["significant_fdr"])),
                )
            )

        for _, row in group.iterrows():
            latent_idx = int(row["latent_idx"])
            examples = top_utterances[
                (top_utterances["label"] == label) & (top_utterances["latent_idx"] == latent_idx)
            ]
            lines.extend(
                [
                    "",
                    f"### {label} latent {latent_idx} top50 高激活语句",
                    "",
                    f"- rank within label: {int(row['rank_within_label'])}",
                    f"- Cohen's d: {_fmt_float(row['cohens_d'], 4)}",
                    f"- directional AUC: {_fmt_float(row['directional_auc'], 4)}",
                    "",
                    "| rank | activation | target_match | active_labels | record_id | text |",
                    "|---:|---:|---:|---|---|---|",
                ]
            )
            for _, ex in examples.iterrows():
                lines.append(
                    "| {rank} | {activation} | {target} | {labels} | {record} | {text} |".format(
                        rank=int(ex["utterance_rank"]),
                        activation=_fmt_float(ex["activation"], digits=4),
                        target=int(ex["target_match"]),
                        labels=_escape_md(ex["active_labels"]),
                        record=_escape_md(ex["record_id"]),
                        text=_escape_md(ex["unit_text"]),
                    )
                )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_top20_cohensd_latent_utterance_export(
    *,
    association_path: str | Path,
    feature_store_path: str | Path,
    label_matrix_path: str | Path,
    records_path: str | Path,
    output_dir: str | Path,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    top_features: int = 20,
    top_utterances: int = 50,
) -> dict[str, Any]:
    association_path = Path(association_path)
    feature_store_path = Path(feature_store_path)
    label_matrix_path = Path(label_matrix_path)
    records_path = Path(records_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    association = pd.read_csv(association_path)
    features = _load_feature_tensor(feature_store_path)
    label_matrix = pd.read_csv(label_matrix_path)
    records = _read_jsonl(records_path)
    label_tuple = tuple(label.upper() for label in labels)

    selected, label_audit = select_top_positive_cohensd_latents(
        association,
        labels=label_tuple,
        top_features=top_features,
    )
    utterances = build_top_activating_utterances(
        selected,
        features,
        label_matrix,
        records,
        labels=label_tuple,
        top_utterances=top_utterances,
    )

    selected_path = output_path / "top20_cohensd_latents_by_label.csv"
    utterances_path = output_path / "top50_utterances_by_top20_cohensd_latents.csv"
    report_path = output_path / "top20_cohensd_feature_activation_report.md"
    audit_path = output_path / "top20_cohensd_feature_activation_audit.json"

    selected.to_csv(selected_path, index=False)
    utterances.to_csv(utterances_path, index=False)
    audit = {
        "analysis": "misc_top20_positive_cohensd_latent_utterances",
        "model_scope": "llama_main_result",
        "labels": label_audit,
        "selection_policy": {
            "top_features_per_label": int(top_features),
            "top_utterances_per_feature": int(top_utterances),
            "feature_filter": "cohens_d > 0 and latent_idx >= 0",
            "feature_sort": ["cohens_d desc", "directional_auc desc", "precision_at_50 desc", "latent_idx asc"],
            "utterance_scope": "full_dataset",
        },
        "inputs": {
            "association": str(association_path),
            "feature_store": str(feature_store_path),
            "label_matrix": str(label_matrix_path),
            "records": str(records_path),
        },
        "outputs": {
            "top20_cohensd_latents_by_label": str(selected_path),
            "top50_utterances_by_top20_cohensd_latents": str(utterances_path),
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
        description="Export top positive-Cohen's-d MISC SAE latents and top activating utterances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--association",
        default="outputs/misc_full_sae_eval/functional/misc_label_mapping/latent_label_matrix.csv",
    )
    parser.add_argument("--feature-store", default="outputs/misc_full_sae_eval/feature_store/utterance_features.pt")
    parser.add_argument("--label-matrix", default="outputs/misc_full_sae_eval/label_matrix.csv")
    parser.add_argument("--records", default="outputs/misc_full_sae_eval/records.jsonl")
    parser.add_argument(
        "--output-dir",
        default="outputs/misc_full_sae_eval/interpretability/top20_cohensd_latent_utterances",
    )
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--top-features", type=int, default=20)
    parser.add_argument("--top-utterances", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = run_top20_cohensd_latent_utterance_export(
        association_path=args.association,
        feature_store_path=args.feature_store,
        label_matrix_path=args.label_matrix,
        records_path=args.records,
        output_dir=args.output_dir,
        labels=tuple(label.upper() for label in args.labels),
        top_features=args.top_features,
        top_utterances=args.top_utterances,
    )
    print("Completed Top20 positive-Cohen's-d latent + Top50 utterance export.")
    print(f"Output dir: {args.output_dir}")
    print(f"Selected latents: {audit['n_selected_latents']}")
    print(f"Top utterance rows: {audit['n_top_utterance_rows']}")
    print(f"Report: {audit['outputs']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
