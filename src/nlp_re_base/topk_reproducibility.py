"""Split-half reproducibility for filtered MISC SAE label-latent rankings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .cross_val_framework import (
    CrossValInputs,
    compute_subset_associations,
    fisher_ci_for_spearman,
    jaccard,
    null_jaccard_distribution,
    split_rows_by_group,
    top_latents_for_label,
)


def run_topk_reproducibility(
    inputs: CrossValInputs,
    *,
    output_dir: str | Path,
    reference_matrix_path: str | Path | None = None,
    group_column: str = "source_file",
    ranking_metrics: Iterable[str] = ("cohens_d", "directional_auc"),
    top_k: int = 20,
    top_k_grid: Iterable[int] | None = None,
    n_repeats: int = 50,
    null_iter: int = 10000,
    random_state: int = 42,
    min_positive: int = 10,
    min_negative: int = 10,
    chunk_size: int = 512,
) -> dict:
    """Run repeated E1 split-half rank and TopK set reproducibility."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if group_column not in inputs.label_matrix.columns:
        raise ValueError(f"group column {group_column!r} not found in label matrix")
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")

    top_k_values = _normalize_top_k_grid(top_k=top_k, top_k_grid=top_k_grid, pool_size=len(inputs.latent_indices))
    max_top_k = max(top_k_values)
    if int(top_k) not in top_k_values:
        top_k_values = tuple(sorted((*top_k_values, int(top_k))))

    rank_rows: list[dict] = []
    jaccard_rows: list[dict] = []
    inclusion_counts: dict[tuple[str, str, int, int], int] = {}
    pool_size = int(len(inputs.latent_indices))
    null_cache: dict[tuple[str, int], np.ndarray] = {}
    reference_tops = _load_reference_tops(
        reference_matrix_path,
        labels=inputs.labels,
        ranking_metrics=ranking_metrics,
        top_k_values=top_k_values,
    )

    first_matrices: list[pd.DataFrame] | None = None
    split_random_states = [int(random_state + repeat_idx) for repeat_idx in range(n_repeats)]

    for repeat_idx, split_seed in enumerate(split_random_states):
        row_splits = split_rows_by_group(
            inputs.label_matrix[group_column].astype(str).to_numpy(),
            n_splits=2,
            random_state=split_seed,
        )
        if len(row_splits) != 2:
            raise ValueError("split-half reproducibility requires exactly two splits")

        matrices = [
            compute_subset_associations(
                inputs,
                rows,
                min_positive=min_positive,
                min_negative=min_negative,
                chunk_size=chunk_size,
            )
            for rows in row_splits
        ]
        if repeat_idx == 0:
            first_matrices = matrices

        for label in inputs.labels:
            a = matrices[0][matrices[0]["label"] == label]
            b = matrices[1][matrices[1]["label"] == label]
            merged = a.merge(
                b,
                on=["label", "latent_idx"],
                suffixes=("_a", "_b"),
                how="inner",
            )
            split_a_pos = int(inputs.label_matrix.iloc[row_splits[0]][label].astype(int).sum())
            split_b_pos = int(inputs.label_matrix.iloc[row_splits[1]][label].astype(int).sum())
            low_n_warning = bool(min(split_a_pos, split_b_pos) < 200)

            for metric in ranking_metrics:
                metric_a = f"{metric}_a"
                metric_b = f"{metric}_b"
                if metric_a not in merged.columns or metric_b not in merged.columns:
                    continue
                x = pd.to_numeric(merged[metric_a], errors="coerce")
                y = pd.to_numeric(merged[metric_b], errors="coerce")
                valid = x.notna() & y.notna()
                if int(valid.sum()) >= 3:
                    rho, p_value = stats.spearmanr(x[valid], y[valid])
                else:
                    rho, p_value = np.nan, np.nan
                ci_lo, ci_hi = fisher_ci_for_spearman(float(rho), int(valid.sum()))
                rank_rows.append(
                    {
                        "repeat_index": int(repeat_idx),
                        "split_random_state": int(split_seed),
                        "label": label,
                        "ranking_metric": metric,
                        "n_latents": int(valid.sum()),
                        "split_a_n": int(len(row_splits[0])),
                        "split_b_n": int(len(row_splits[1])),
                        "split_a_positive": split_a_pos,
                        "split_b_positive": split_b_pos,
                        "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                        "spearman_ci_lo": ci_lo,
                        "spearman_ci_hi": ci_hi,
                        "spearman_p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                        "low_n_warning": low_n_warning,
                    }
                )

                positive_only = metric == "cohens_d"
                top_a_full = top_latents_for_label(
                    matrices[0], label, metric=metric, top_k=max_top_k, positive_only=positive_only
                )["latent_idx"].astype(int).tolist()
                top_b_full = top_latents_for_label(
                    matrices[1], label, metric=metric, top_k=max_top_k, positive_only=positive_only
                )["latent_idx"].astype(int).tolist()
                for grid_k in top_k_values:
                    top_a = top_a_full[:grid_k]
                    top_b = top_b_full[:grid_k]
                    for latent_idx in top_a:
                        key = (label, metric, int(grid_k), int(latent_idx))
                        inclusion_counts[key] = inclusion_counts.get(key, 0) + 1
                    for latent_idx in top_b:
                        key = (label, metric, int(grid_k), int(latent_idx))
                        inclusion_counts[key] = inclusion_counts.get(key, 0) + 1
                    observed = jaccard(top_a, top_b)
                    cache_key = (metric, int(grid_k))
                    if cache_key not in null_cache:
                        null_cache[cache_key] = null_jaccard_distribution(
                            pool_size=pool_size,
                            top_k=int(grid_k),
                            n_iter=null_iter,
                            random_state=random_state + len(null_cache) * 997,
                        )
                    null = null_cache[cache_key]
                    null_mean = float(null.mean())
                    null_std = float(null.std(ddof=1)) if null.size > 1 else 0.0
                    p_value = float((np.count_nonzero(null >= observed) + 1) / (len(null) + 1))
                    reference_top = reference_tops.get((label, metric, int(grid_k)))
                    row = {
                        "repeat_index": int(repeat_idx),
                        "split_random_state": int(split_seed),
                        "label": label,
                        "ranking_metric": metric,
                        "top_k": int(grid_k),
                        "split_a_topk": ",".join(map(str, top_a)),
                        "split_b_topk": ",".join(map(str, top_b)),
                        "observed_jaccard": observed,
                        "null_mean": null_mean,
                        "null_std": null_std,
                        "null_p_value": p_value,
                        "passes_null_2sd": bool(observed >= null_mean + 2.0 * null_std),
                        "low_n_warning": low_n_warning,
                    }
                    if reference_top is not None:
                        top_a_set = set(int(x) for x in top_a)
                        top_b_set = set(int(x) for x in top_b)
                        reference_set = set(int(x) for x in reference_top)
                        cv_union = top_a_set | top_b_set
                        all_three = reference_set & top_a_set & top_b_set
                        row.update(
                            {
                                "reference_topk": ",".join(map(str, reference_top)),
                                "reference_vs_split_a_overlap_n": int(len(reference_set & top_a_set)),
                                "reference_vs_split_b_overlap_n": int(len(reference_set & top_b_set)),
                                "reference_vs_cv_union_overlap_n": int(len(reference_set & cv_union)),
                                "reference_vs_cv_union_overlap_fraction": float(
                                    len(reference_set & cv_union) / max(len(reference_set), 1)
                                ),
                                "reference_in_all_three_n": int(len(all_three)),
                            }
                        )
                    jaccard_rows.append(row)

    if first_matrices is None:
        raise RuntimeError("no split matrices were computed")
    first_matrices[0].to_csv(output_dir / "split_a_latent_label_matrix.csv", index=False, encoding="utf-8-sig")
    first_matrices[1].to_csv(output_dir / "split_b_latent_label_matrix.csv", index=False, encoding="utf-8-sig")

    rank_df = pd.DataFrame(rank_rows)
    jaccard_df = pd.DataFrame(jaccard_rows)
    rank_single = _representative_rows(rank_df)
    primary_jaccard_df = jaccard_df[jaccard_df["top_k"].astype(int) == int(top_k)].copy()
    jaccard_single = _representative_rows(primary_jaccard_df)
    repeated_summary = _summarize_repeated_splits(rank_df, primary_jaccard_df)
    grid_summary = _summarize_repeated_splits(rank_df, jaccard_df)
    inclusion_df = _build_inclusion_frequency(
        inclusion_counts,
        n_half_runs=2 * n_repeats,
        reference_matrix_path=reference_matrix_path,
        labels=inputs.labels,
        ranking_metrics=ranking_metrics,
    )

    rank_path = output_dir / "split_half_rank_correlation.csv"
    jaccard_path = output_dir / "split_half_top20_jaccard.csv"
    repeated_rank_path = output_dir / "repeated_split_rank_correlation.csv"
    repeated_jaccard_path = output_dir / "repeated_split_top20_jaccard.csv"
    repeated_summary_path = output_dir / "repeated_split_summary.csv"
    grid_jaccard_path = output_dir / "repeated_split_topk_grid.csv"
    grid_summary_path = output_dir / "repeated_split_topk_grid_summary.csv"
    inclusion_path = output_dir / "topk_inclusion_frequency.csv"
    rank_single.to_csv(rank_path, index=False, encoding="utf-8-sig")
    jaccard_single.to_csv(jaccard_path, index=False, encoding="utf-8-sig")
    rank_df.to_csv(repeated_rank_path, index=False, encoding="utf-8-sig")
    primary_jaccard_df.to_csv(repeated_jaccard_path, index=False, encoding="utf-8-sig")
    repeated_summary.to_csv(repeated_summary_path, index=False, encoding="utf-8-sig")
    jaccard_df.to_csv(grid_jaccard_path, index=False, encoding="utf-8-sig")
    grid_summary.to_csv(grid_summary_path, index=False, encoding="utf-8-sig")
    inclusion_df.to_csv(inclusion_path, index=False, encoding="utf-8-sig")
    _write_grid_figures(output_dir, grid_summary, inclusion_df)

    summary_path = output_dir / "split_half_summary.md"
    _write_summary(
        summary_path,
        rank_single,
        jaccard_single,
        repeated_summary,
        group_column=group_column,
        n_repeats=n_repeats,
    )
    manifest = {
        "analysis": "topk_reproducibility",
        "group_column": group_column,
        "labels": list(inputs.labels),
        "ranking_metrics": list(ranking_metrics),
        "top_k": int(top_k),
        "top_k_grid": list(map(int, top_k_values)),
        "n_repeats": int(n_repeats),
        "null_iter": int(null_iter),
        "random_state": int(random_state),
        "split_random_states": split_random_states,
        "reference_matrix": str(reference_matrix_path) if reference_matrix_path is not None else None,
        "n_rows": int(len(inputs.label_matrix)),
        "n_filtered_latents": int(len(inputs.latent_indices)),
        "outputs": {
            "rank_correlation": str(rank_path),
            "topk_jaccard": str(jaccard_path),
            "repeated_rank_correlation": str(repeated_rank_path),
            "repeated_topk_jaccard": str(repeated_jaccard_path),
            "repeated_summary": str(repeated_summary_path),
            "repeated_topk_grid": str(grid_jaccard_path),
            "repeated_topk_grid_summary": str(grid_summary_path),
            "topk_inclusion_frequency": str(inclusion_path),
            "topk_jaccard_vs_k": str(output_dir / "figures" / "topk_jaccard_vs_k.png"),
            "topk_stable_count_vs_k": str(output_dir / "figures" / "topk_stable_count_vs_k.png"),
            "summary": str(summary_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _normalize_top_k_grid(
    *,
    top_k: int,
    top_k_grid: Iterable[int] | None,
    pool_size: int,
) -> tuple[int, ...]:
    values = [int(top_k)] if top_k_grid is None else [int(value) for value in top_k_grid]
    out = sorted({max(1, min(int(pool_size), value)) for value in values if value > 0})
    if not out:
        out = [max(1, min(int(pool_size), int(top_k)))]
    return tuple(out)


def _load_reference_tops(
    reference_matrix_path: str | Path | None,
    *,
    labels: Iterable[str],
    ranking_metrics: Iterable[str],
    top_k_values: Iterable[int],
) -> dict[tuple[str, str, int], list[int]]:
    if reference_matrix_path is None:
        return {}
    path = Path(reference_matrix_path)
    if not path.exists():
        return {}
    matrix = pd.read_csv(path)
    top_k_values = tuple(sorted({int(value) for value in top_k_values if int(value) > 0}))
    max_top_k = max(top_k_values) if top_k_values else 1
    out: dict[tuple[str, str, int], list[int]] = {}
    for label in labels:
        for metric in ranking_metrics:
            positive_only = metric == "cohens_d"
            top = top_latents_for_label(
                matrix,
                str(label),
                metric=str(metric),
                top_k=max_top_k,
                positive_only=positive_only,
            )["latent_idx"].astype(int)
            values = top.tolist()
            for top_k in top_k_values:
                out[(str(label), str(metric), int(top_k))] = values[: int(top_k)]
    return out


def _load_reference_ranks(
    reference_matrix_path: str | Path | None,
    *,
    labels: Iterable[str],
    ranking_metrics: Iterable[str],
) -> dict[tuple[str, str, int], int]:
    if reference_matrix_path is None:
        return {}
    path = Path(reference_matrix_path)
    if not path.exists():
        return {}
    matrix = pd.read_csv(path)
    out: dict[tuple[str, str, int], int] = {}
    for label in labels:
        label = str(label)
        for metric in ranking_metrics:
            metric = str(metric)
            positive_only = metric == "cohens_d"
            ranked = top_latents_for_label(
                matrix,
                label,
                metric=metric,
                top_k=len(matrix),
                positive_only=positive_only,
            )
            for rank, latent_idx in enumerate(ranked["latent_idx"].astype(int).tolist(), start=1):
                out[(label, metric, int(latent_idx))] = int(rank)
    return out


def _build_inclusion_frequency(
    inclusion_counts: dict[tuple[str, str, int, int], int],
    *,
    n_half_runs: int,
    reference_matrix_path: str | Path | None,
    labels: Iterable[str],
    ranking_metrics: Iterable[str],
) -> pd.DataFrame:
    reference_ranks = _load_reference_ranks(
        reference_matrix_path,
        labels=labels,
        ranking_metrics=ranking_metrics,
    )
    rows: list[dict] = []
    denom = max(int(n_half_runs), 1)
    for (label, metric, top_k, latent_idx), count in sorted(inclusion_counts.items()):
        rank = reference_ranks.get((label, metric, int(latent_idx)))
        rows.append(
            {
                "label": label,
                "ranking_metric": metric,
                "top_k": int(top_k),
                "latent_idx": int(latent_idx),
                "inclusion_count": int(count),
                "n_half_runs": int(denom),
                "inclusion_frequency": float(count / denom),
                "full_data_rank": rank if rank is not None else np.nan,
                "in_full_data_topk": bool(rank is not None and rank <= int(top_k)),
            }
        )
    return pd.DataFrame(rows)


def _representative_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["repeat_index"] == 0].copy()
    return out.drop(columns=["repeat_index", "split_random_state"], errors="ignore")


def _series_stats(series: pd.Series, prefix: str) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_p05": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p95": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        f"{prefix}_min": float(values.min()),
        f"{prefix}_p05": float(values.quantile(0.05)),
        f"{prefix}_median": float(values.median()),
        f"{prefix}_p95": float(values.quantile(0.95)),
        f"{prefix}_max": float(values.max()),
    }


def _summarize_repeated_splits(rank_df: pd.DataFrame, jaccard_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (label, metric, top_k), j_group in jaccard_df.groupby(["label", "ranking_metric", "top_k"], sort=True):
        r_group = rank_df[(rank_df["label"] == label) & (rank_df["ranking_metric"] == metric)]
        row: dict[str, object] = {
            "label": label,
            "ranking_metric": metric,
            "n_repeats": int(j_group["repeat_index"].nunique()),
            "top_k": int(top_k),
            "passes_null_2sd_rate": float(j_group["passes_null_2sd"].astype(bool).mean()),
            "low_n_warning_rate": float(r_group["low_n_warning"].astype(bool).mean())
            if not r_group.empty
            else np.nan,
            "rho_ge_0_60_rate": float((pd.to_numeric(r_group["spearman_rho"], errors="coerce") >= 0.60).mean())
            if not r_group.empty
            else np.nan,
        }
        row.update(_series_stats(j_group["observed_jaccard"], "observed_jaccard"))
        row.update(_series_stats(r_group["spearman_rho"], "spearman_rho"))
        if "reference_vs_cv_union_overlap_fraction" in j_group.columns:
            row.update(
                _series_stats(
                    j_group["reference_vs_cv_union_overlap_fraction"],
                    "reference_vs_cv_union_overlap_fraction",
                )
            )
            row.update(_series_stats(j_group["reference_in_all_three_n"], "reference_in_all_three_n"))
        rows.append(row)
    return pd.DataFrame(rows)


def _write_grid_figures(output_dir: Path, grid_summary: pd.DataFrame, inclusion_df: pd.DataFrame) -> None:
    if grid_summary.empty and inclusion_df.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not grid_summary.empty:
        metric = "cohens_d" if "cohens_d" in set(grid_summary["ranking_metric"]) else str(grid_summary["ranking_metric"].iloc[0])
        view = grid_summary[grid_summary["ranking_metric"] == metric].copy()
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, group in view.groupby("label", sort=True):
            group = group.sort_values("top_k")
            ax.plot(group["top_k"], group["observed_jaccard_mean"], linewidth=1.3, label=label)
        ax.axhline(0.40, color="#777777", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Top-K")
        ax.set_ylabel("Mean split-half Jaccard")
        ax.set_title(f"Top-K set stability vs K ({metric})")
        ax.set_ylim(0.0, min(1.0, max(0.45, float(view["observed_jaccard_mean"].max()) + 0.08)))
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "topk_jaccard_vs_k.png", dpi=180)
        plt.close(fig)

    if not inclusion_df.empty:
        metric = "cohens_d" if "cohens_d" in set(inclusion_df["ranking_metric"]) else str(inclusion_df["ranking_metric"].iloc[0])
        view = inclusion_df[inclusion_df["ranking_metric"] == metric].copy()
        counts = (
            view[view["inclusion_frequency"] >= 0.70]
            .groupby(["label", "top_k"], as_index=False)
            .size()
            .rename(columns={"size": "stable_count"})
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, group in counts.groupby("label", sort=True):
            group = group.sort_values("top_k")
            ax.plot(group["top_k"], group["stable_count"], linewidth=1.3, label=label)
        ax.set_xlabel("Top-K")
        ax.set_ylabel("Latents with inclusion frequency >= 0.70")
        ax.set_title(f"Stable latent count vs K ({metric})")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "topk_stable_count_vs_k.png", dpi=180)
        plt.close(fig)


def _write_summary(
    path: Path,
    rank_df: pd.DataFrame,
    jaccard_df: pd.DataFrame,
    repeated_summary: pd.DataFrame,
    *,
    group_column: str,
    n_repeats: int,
) -> None:
    lines = [
        "# Split-half TopK Reproducibility",
        "",
        f"- Group column: `{group_column}`",
        f"- Repeated grouped split-half runs: `{n_repeats}`.",
        "- Spearman threshold for stable ranking: `rho >= 0.60`.",
        "- TopK set stability gate: observed Jaccard >= null mean + 2 * null std.",
        "",
        "## Repeated Split Summary",
        "",
        "| label | metric | Jaccard mean | Jaccard p05..p95 | pass rate | Spearman mean | Spearman p05..p95 | rho>=0.60 rate | reference-union overlap mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in repeated_summary.sort_values(["label", "ranking_metric"]).iterrows():
        reference_overlap = row.get("reference_vs_cv_union_overlap_fraction_mean", np.nan)
        reference_text = "" if pd.isna(reference_overlap) else f"{float(reference_overlap):.3f}"
        lines.append(
            f"| {row['label']} | {row['ranking_metric']} | "
            f"{row['observed_jaccard_mean']:.3f} | "
            f"{row['observed_jaccard_p05']:.3f}..{row['observed_jaccard_p95']:.3f} | "
            f"{row['passes_null_2sd_rate']:.3f} | "
            f"{row['spearman_rho_mean']:.3f} | "
            f"{row['spearman_rho_p05']:.3f}..{row['spearman_rho_p95']:.3f} | "
            f"{row['rho_ge_0_60_rate']:.3f} | {reference_text} |"
        )
    lines.extend(
        [
            "",
            "## Representative Split Ranking Stability",
            "",
            f"The following table keeps the first split (`random_state` seed) for backward-compatible comparison files.",
            "",
            "| label | metric | rho | 95% CI | low-n |",
            "|---|---|---:|---:|---|",
        ]
    )
    for _, row in rank_df.sort_values(["label", "ranking_metric"]).iterrows():
        ci = f"{row['spearman_ci_lo']:.3f}..{row['spearman_ci_hi']:.3f}"
        lines.append(
            f"| {row['label']} | {row['ranking_metric']} | {row['spearman_rho']:.3f} | {ci} | {bool(row['low_n_warning'])} |"
        )
    lines.extend(
        [
            "",
            "## Representative Split TopK Set Stability",
            "",
            "| label | metric | observed Jaccard | null mean | null std | pass |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for _, row in jaccard_df.sort_values(["label", "ranking_metric"]).iterrows():
        lines.append(
            f"| {row['label']} | {row['ranking_metric']} | {row['observed_jaccard']:.3f} | "
            f"{row['null_mean']:.4f} | {row['null_std']:.4f} | {bool(row['passes_null_2sd'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
