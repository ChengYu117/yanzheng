"""Plot latent-by-label heatmaps from the MISC latent-label matrix.

The association metrics are assumed to have already been computed by the
MISC label mapping pipeline. This script only reshapes and visualizes them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_LABELS = ["RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF"]
SORT_COLUMNS = ["abs_cohens_d", "directional_auc", "latent_idx"]


def _read_matrix(path: Path, labels: list[str]) -> pd.DataFrame:
    matrix = pd.read_csv(path)
    matrix["label"] = matrix["label"].astype(str).str.upper()
    matrix = matrix[matrix["label"].isin(labels)].copy()
    numeric_cols = [
        "latent_idx",
        "cohens_d",
        "abs_cohens_d",
        "auc",
        "directional_auc",
        "precision_at_50",
        "precision_lift_at_50",
        "p_value",
        "prevalence",
    ]
    for col in numeric_cols:
        if col in matrix.columns:
            matrix[col] = pd.to_numeric(matrix[col], errors="coerce")
    if "significant_fdr" in matrix.columns:
        matrix["significant_fdr"] = (
            matrix["significant_fdr"].astype(str).str.lower().isin(["true", "1", "yes"])
        )
    return matrix


def _ranked_by_label(matrix: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for label, group in matrix.groupby("label", sort=False):
        ranked = group.sort_values(SORT_COLUMNS, ascending=[False, False, True]).copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        parts.append(ranked)
    return pd.concat(parts, ignore_index=True)


def _top_union(ranked: pd.DataFrame, k: int) -> pd.Index:
    return pd.Index(ranked.loc[ranked["rank"] <= k, "latent_idx"].astype(int).unique())


def _adaptive_pool(adaptive_summary: Path, ranked: pd.DataFrame) -> pd.Index:
    if adaptive_summary.exists():
        rows = pd.read_csv(adaptive_summary)
        latents: list[int] = []
        for value in rows.get("adaptive_latents_cap30", pd.Series(dtype=str)).fillna(""):
            for item in str(value).split(","):
                item = item.strip()
                if item:
                    latents.append(int(item))
        if latents:
            return pd.Index(latents).unique()

    # Fallback: reproduce a conservative adaptive pool if the audit file is absent.
    gated = ranked[
        (ranked["rank"] <= 30)
        & (ranked["significant_fdr"])
        & (ranked["abs_cohens_d"] >= 0.5)
        & (ranked["directional_auc"] >= 0.60)
        & (
            (ranked["cohens_d"] < 0)
            | (ranked.get("precision_lift_at_50", pd.Series(0.0, index=ranked.index)) >= 0.10)
        )
    ]
    return pd.Index(gated["latent_idx"].astype(int).unique())


def _ordered_latents(matrix: pd.DataFrame, latent_ids: pd.Index, label_order: list[str]) -> list[int]:
    subset = matrix[matrix["latent_idx"].isin(latent_ids)].copy()
    if subset.empty:
        return []
    label_rank = {label: idx for idx, label in enumerate(label_order)}
    dom_rows = []
    for latent_idx, group in subset.groupby("latent_idx"):
        best = group.sort_values(SORT_COLUMNS, ascending=[False, False, True]).iloc[0]
        dom_rows.append(
            {
                "latent_idx": int(latent_idx),
                "dominant_label": str(best["label"]),
                "dominant_label_rank": label_rank.get(str(best["label"]), 999),
                "max_abs_d": float(group["abs_cohens_d"].max()),
                "max_auc": float(group["directional_auc"].max()),
            }
        )
    order = pd.DataFrame(dom_rows).sort_values(
        ["dominant_label_rank", "max_abs_d", "max_auc", "latent_idx"],
        ascending=[True, False, False, True],
    )
    return order["latent_idx"].astype(int).tolist()


def _pivot(matrix: pd.DataFrame, metric: str, latent_order: list[int], label_order: list[str]) -> pd.DataFrame:
    values = matrix[matrix["latent_idx"].isin(latent_order)].pivot(
        index="latent_idx",
        columns="label",
        values=metric,
    )
    values = values.reindex(index=latent_order, columns=label_order)
    return values.fillna(0.0)


def _plot_heatmap(
    values: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    metric: str,
    show_y_labels: bool,
) -> None:
    n_rows, n_cols = values.shape
    fig_height = min(24.0, max(5.0, n_rows * 0.09 if show_y_labels else 10.0))
    fig_width = max(7.0, n_cols * 0.75)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    arr = values.to_numpy(dtype=float)
    if metric in {"cohens_d", "precision_lift_at_50"}:
        vmax = float(np.nanpercentile(np.abs(arr), 99.0)) if arr.size else 1.0
        vmax = max(vmax, 1e-6)
        im = ax.imshow(arr, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    elif metric == "directional_auc":
        im = ax.imshow(arr, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
    else:
        vmax = float(np.nanpercentile(arr, 99.0)) if arr.size else 1.0
        vmax = max(vmax, 1e-6)
        im = ax.imshow(arr, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)

    ax.set_title(title)
    ax.set_xlabel("MISC label")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(values.columns.tolist())
    ax.set_ylabel("SAE latent")
    if show_y_labels:
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(values.index.astype(int).astype(str).tolist(), fontsize=6)
    else:
        ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.024, pad=0.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_bundle(
    matrix: pd.DataFrame,
    *,
    latent_ids: pd.Index,
    subset_name: str,
    output_dir: Path,
    label_order: list[str],
) -> list[dict[str, str]]:
    latent_order = _ordered_latents(matrix, latent_ids, label_order)
    if not latent_order:
        return []

    show_y_labels = len(latent_order) <= 220
    records: list[dict[str, str]] = []
    metrics = [
        ("cohens_d", "Signed Cohen's d"),
        ("directional_auc", "Directional AUC"),
        ("precision_at_50", "Precision@50"),
        ("precision_lift_at_50", "Precision@50 lift"),
    ]
    for metric, label in metrics:
        if metric not in matrix.columns:
            continue
        values = _pivot(matrix, metric, latent_order, label_order)
        csv_path = output_dir / f"{subset_name}_{metric}_matrix.csv"
        png_path = output_dir / f"{subset_name}_{metric}_heatmap.png"
        values.to_csv(csv_path)
        _plot_heatmap(
            values,
            title=f"{subset_name}: latent x label {label}",
            output_path=png_path,
            metric=metric,
            show_y_labels=show_y_labels,
        )
        records.append(
            {
                "subset": subset_name,
                "metric": metric,
                "n_latents": str(len(latent_order)),
                "csv": str(csv_path),
                "png": str(png_path),
            }
        )
    return records


def _write_report(output_dir: Path, records: list[dict[str, str]]) -> None:
    lines = [
        "# MISC latent x label heatmaps",
        "",
        "These figures visualize the already-computed latent-label association metrics.",
        "",
        "Generated heatmap bundles:",
        "",
        "| Subset | Metric | N latents | PNG | CSV |",
        "|---|---|---:|---|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['subset']} | {record['metric']} | {record['n_latents']} | "
            f"`{record['png']}` | `{record['csv']}` |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `top20_union` is the union of each label's top 20 latents.",
            "- `top30_union` includes near-boundary candidates beyond the current Top20 window.",
            "- `adaptive_pool` uses the cutoff audit's quality-gated candidate list when available.",
            "- `full_all_latents` contains all SAE latents, sorted by their strongest label association; it is a compressed completeness view, not a readable per-latent figure.",
            "- `precision_lift_at_50` is usually easier to compare across labels than raw `precision_at_50`, because labels have different base rates.",
        ]
    )
    (output_dir / "latent_label_heatmap_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        default="outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered/latent_label_matrix.csv",
        help="Path to latent_label_matrix.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/misc_full_sae_eval/interpretability/latent_label_heatmaps",
        help="Directory for heatmap PNGs and reshaped CSV matrices.",
    )
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    parser.add_argument(
        "--adaptive-summary",
        default="outputs/misc_full_sae_eval/interpretability/topk_cutoff_audit/adaptive_filter_summary.csv",
    )
    args = parser.parse_args()

    labels = [label.upper() for label in args.labels]
    matrix = _read_matrix(Path(args.matrix), labels)
    ranked = _ranked_by_label(matrix)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_specs = {
        "top20_union": _top_union(ranked, 20),
        "top30_union": _top_union(ranked, 30),
        "adaptive_pool": _adaptive_pool(Path(args.adaptive_summary), ranked),
        "full_all_latents": pd.Index(sorted(matrix["latent_idx"].astype(int).unique())),
    }

    records: list[dict[str, str]] = []
    for subset_name, latent_ids in subset_specs.items():
        records.extend(
            _plot_bundle(
                matrix,
                latent_ids=latent_ids,
                subset_name=subset_name,
                output_dir=output_dir,
                label_order=labels,
            )
        )
    pd.DataFrame(records).to_csv(output_dir / "latent_label_heatmap_manifest.csv", index=False)
    _write_report(output_dir, records)
    print(f"Wrote {len(records)} heatmap files to {output_dir}")


if __name__ == "__main__":
    main()
