"""Cross-quality split validation for filtered MISC SAE latent candidates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .cross_val_framework import CrossValInputs, compute_subset_associations, top_latents_for_label


def run_cross_quality_validation(
    inputs: CrossValInputs,
    *,
    output_dir: str | Path,
    reference_matrix_path: str | Path | None = None,
    ranking_metric: str = "cohens_d",
    top_k: int = 100,
    stable_drop_threshold: float = 0.05,
    min_positive: int = 10,
    min_negative: int = 10,
    chunk_size: int = 512,
) -> dict:
    """Run E3 high -> low and low -> high latent association validation."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if "source_split" not in inputs.label_matrix.columns:
        raise ValueError("label_matrix must contain source_split for cross-quality validation")

    split_values = inputs.label_matrix["source_split"].astype(str).str.lower()
    high_rows = np.flatnonzero(split_values == "high")
    low_rows = np.flatnonzero(split_values == "low")
    if high_rows.size == 0 or low_rows.size == 0:
        raise ValueError("both high and low source_split rows are required")

    high_matrix = compute_subset_associations(
        inputs,
        high_rows,
        min_positive=min_positive,
        min_negative=min_negative,
        chunk_size=chunk_size,
    )
    low_matrix = compute_subset_associations(
        inputs,
        low_rows,
        min_positive=min_positive,
        min_negative=min_negative,
        chunk_size=chunk_size,
    )
    high_matrix.to_csv(output_dir / "high_latent_label_matrix.csv", index=False, encoding="utf-8-sig")
    low_matrix.to_csv(output_dir / "low_latent_label_matrix.csv", index=False, encoding="utf-8-sig")

    reference_matrix = pd.read_csv(reference_matrix_path) if reference_matrix_path is not None and Path(reference_matrix_path).exists() else None

    rows: list[dict] = []
    for label in inputs.labels:
        directions = [
            ("high_to_low", "high", "low", high_matrix, low_matrix),
            ("low_to_high", "low", "high", low_matrix, high_matrix),
        ]
        if reference_matrix is not None:
            directions.extend(
                [
                    ("full_to_high", "full", "high", reference_matrix, high_matrix),
                    ("full_to_low", "full", "low", reference_matrix, low_matrix),
                ]
            )
        for direction, in_name, out_name, in_matrix, out_matrix in directions:
            selected = top_latents_for_label(
                in_matrix,
                label,
                metric=ranking_metric,
                top_k=top_k,
                positive_only=ranking_metric == "cohens_d",
            )
            out_by_latent = out_matrix[out_matrix["label"] == label].set_index("latent_idx")
            for rank, (_, in_row) in enumerate(selected.iterrows(), start=1):
                latent_idx = int(in_row["latent_idx"])
                if latent_idx not in out_by_latent.index:
                    continue
                out_row = out_by_latent.loc[latent_idx]
                in_auc = float(in_row.get("directional_auc", np.nan))
                out_auc = float(out_row.get("directional_auc", np.nan))
                cross_drop = in_auc - out_auc
                rows.append(
                    {
                        "label": label,
                        "direction": direction,
                        "in_split": in_name,
                        "out_split": out_name,
                        "rank_within_in_split": int(rank),
                        "latent_idx": latent_idx,
                        "ranking_metric": ranking_metric,
                        "in_directional_auc": in_auc,
                        "out_directional_auc": out_auc,
                        "auc_cross_drop": float(cross_drop),
                        "in_cohens_d": float(in_row.get("cohens_d", np.nan)),
                        "out_cohens_d": float(out_row.get("cohens_d", np.nan)),
                        "in_n_positive": int(in_row.get("n_positive", 0)),
                        "out_n_positive": int(out_row.get("n_positive", 0)),
                        "stable_cross_quality": bool(cross_drop < stable_drop_threshold),
                    }
                )

    comparison = pd.DataFrame(rows)
    comparison_path = output_dir / "cross_quality_auc_comparison.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    summary_rows: list[dict] = []
    for label in inputs.labels:
        high_label = high_matrix[high_matrix["label"] == label]
        low_label = low_matrix[low_matrix["label"] == label]
        merged = high_label.merge(
            low_label,
            on=["label", "latent_idx"],
            suffixes=("_high", "_low"),
            how="inner",
        )
        if len(merged) >= 3:
            rho, p_value = stats.spearmanr(
                merged[f"{ranking_metric}_high"],
                merged[f"{ranking_metric}_low"],
            )
        else:
            rho, p_value = np.nan, np.nan
        label_comp = comparison[comparison["label"] == label]
        summary_rows.append(
            {
                "label": label,
                "ranking_metric": ranking_metric,
                "top_k": int(top_k),
                "n_comparison_rows": int(len(label_comp)),
                "n_stable_cross_quality": int(label_comp["stable_cross_quality"].sum())
                if not label_comp.empty
                else 0,
                "stable_fraction": float(label_comp["stable_cross_quality"].mean())
                if not label_comp.empty
                else np.nan,
                "mean_auc_cross_drop": float(label_comp["auc_cross_drop"].mean())
                if not label_comp.empty
                else np.nan,
                "max_auc_cross_drop": float(label_comp["auc_cross_drop"].max())
                if not label_comp.empty
                else np.nan,
                "high_low_rank_spearman": float(rho) if np.isfinite(rho) else np.nan,
                "high_low_rank_spearman_p": float(p_value) if np.isfinite(p_value) else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "cross_quality_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    manifest = {
        "analysis": "cross_quality_validation",
        "reference_matrix": str(reference_matrix_path) if reference_matrix_path is not None else None,
        "ranking_metric": ranking_metric,
        "top_k": int(top_k),
        "stable_drop_threshold": float(stable_drop_threshold),
        "n_high_rows": int(len(high_rows)),
        "n_low_rows": int(len(low_rows)),
        "labels": list(inputs.labels),
        "outputs": {
            "comparison": str(comparison_path),
            "summary": str(summary_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary_md(output_dir / "cross_quality_summary.md", summary)
    return manifest


def _write_summary_md(path: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Cross-quality Validation Summary",
        "",
        "| label | stable fraction | mean AUC drop | max AUC drop | high-low Spearman |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in summary.sort_values("label").iterrows():
        lines.append(
            f"| {row['label']} | {row['stable_fraction']:.3f} | {row['mean_auc_cross_drop']:.3f} | "
            f"{row['max_auc_cross_drop']:.3f} | {row['high_low_rank_spearman']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
