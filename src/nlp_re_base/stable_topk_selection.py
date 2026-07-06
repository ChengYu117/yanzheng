"""Select stable per-label Top-K SAE latent sets from CV evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .cross_val_framework import DEFAULT_LABELS, top_latents_for_label


@dataclass(frozen=True)
class StableTopKSelectionConfig:
    labels: tuple[str, ...] = DEFAULT_LABELS
    ranking_metric: str = "cohens_d"
    max_k: int = 100
    auc_gap_floor: float = 0.01
    pass_rate_threshold: float = 0.95
    jaccard_p05_threshold: float = 0.40
    inclusion_stable_threshold: float = 0.70
    inclusion_boundary_threshold: float = 0.40
    cross_quality_drop_threshold: float = 0.05
    require_cross_quality_gate: bool = False
    require_label_stability_gate: bool = False


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


def _select_k_auc(label_curve: pd.DataFrame, config: StableTopKSelectionConfig) -> dict[str, Any]:
    curve = label_curve[
        (label_curve["subspace_ranking"].astype(str) == config.ranking_metric)
        & (pd.to_numeric(label_curve["top_k"], errors="coerce") <= config.max_k)
    ].copy()
    if curve.empty:
        return {
            "k_auc": 0,
            "best_auc": np.nan,
            "best_k": 0,
            "target_auc": np.nan,
            "best_auc_se": np.nan,
            "auc_status": "missing_auc_curve",
        }
    curve["top_k"] = pd.to_numeric(curve["top_k"], errors="coerce").astype(int)
    curve["auc_mean"] = pd.to_numeric(curve["auc_mean"], errors="coerce")
    curve["auc_std"] = pd.to_numeric(curve.get("auc_std", np.nan), errors="coerce")
    curve["n_folds"] = pd.to_numeric(curve.get("n_folds", 0), errors="coerce").fillna(0).astype(int)
    best = curve.sort_values(["auc_mean", "top_k"], ascending=[False, True]).iloc[0]
    n_folds = max(int(best.get("n_folds", 0)), 1)
    auc_std = float(best.get("auc_std", 0.0))
    best_auc_se = auc_std / np.sqrt(n_folds) if np.isfinite(auc_std) else 0.0
    target_auc = float(best["auc_mean"]) - max(config.auc_gap_floor, best_auc_se)
    candidates = curve[curve["auc_mean"] >= target_auc].sort_values("top_k")
    k_auc = int(candidates["top_k"].iloc[0]) if not candidates.empty else int(best["top_k"])
    return {
        "k_auc": k_auc,
        "best_auc": float(best["auc_mean"]),
        "best_k": int(best["top_k"]),
        "target_auc": target_auc,
        "best_auc_se": float(best_auc_se),
        "auc_status": "ok",
    }


def _select_k_stab(
    *,
    label: str,
    k_auc: int,
    topk_grid_summary: pd.DataFrame,
    config: StableTopKSelectionConfig,
) -> dict[str, Any]:
    grid = topk_grid_summary[
        (topk_grid_summary["label"].astype(str) == str(label))
        & (topk_grid_summary["ranking_metric"].astype(str) == config.ranking_metric)
    ].copy()
    if grid.empty:
        return {"k_stab": np.nan, "k_star": int(k_auc), "selection_status": "performance_only_unstable"}
    grid["top_k"] = pd.to_numeric(grid["top_k"], errors="coerce").astype(int)
    for col in ("passes_null_2sd_rate", "observed_jaccard_p05"):
        grid[col] = pd.to_numeric(grid[col], errors="coerce")
    eligible = grid[
        (grid["top_k"] >= max(int(k_auc), 1))
        & (grid["top_k"] <= config.max_k)
        & (grid["passes_null_2sd_rate"] >= config.pass_rate_threshold)
        & (grid["observed_jaccard_p05"] >= config.jaccard_p05_threshold)
    ].sort_values("top_k")
    if eligible.empty:
        return {"k_stab": np.nan, "k_star": int(k_auc), "selection_status": "performance_only_unstable"}
    k_stab = int(eligible["top_k"].iloc[0])
    return {"k_stab": k_stab, "k_star": k_stab, "selection_status": "stable_topk_found"}


def _full_data_ranked_latents(
    association: pd.DataFrame,
    *,
    label: str,
    metric: str,
    max_k: int,
) -> pd.DataFrame:
    ranked = top_latents_for_label(
        association,
        label,
        metric=metric,
        top_k=max_k,
        positive_only=metric == "cohens_d",
    ).copy()
    ranked["full_data_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def _build_cross_quality_lookup(cross_quality: pd.DataFrame) -> pd.DataFrame:
    if cross_quality.empty:
        return pd.DataFrame(columns=["label", "latent_idx", "cross_quality_max_auc_drop", "cross_quality_tested"])
    out = cross_quality.copy()
    out["latent_idx"] = pd.to_numeric(out["latent_idx"], errors="coerce").fillna(-1).astype(int)
    out["auc_cross_drop"] = pd.to_numeric(out["auc_cross_drop"], errors="coerce")
    grouped = out.groupby(["label", "latent_idx"], as_index=False)["auc_cross_drop"].max()
    grouped = grouped.rename(columns={"auc_cross_drop": "cross_quality_max_auc_drop"})
    grouped["cross_quality_tested"] = True
    return grouped


def _build_stable_latent_rows(
    *,
    label: str,
    k_star: int,
    selection_status: str,
    association: pd.DataFrame,
    inclusion: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
    cross_quality_lookup: pd.DataFrame,
    config: StableTopKSelectionConfig,
) -> pd.DataFrame:
    if k_star <= 0:
        return pd.DataFrame()
    ranked = _full_data_ranked_latents(
        association,
        label=label,
        metric=config.ranking_metric,
        max_k=max(k_star, 1),
    )
    if ranked.empty:
        return ranked
    inc = inclusion[
        (inclusion["label"].astype(str) == str(label))
        & (inclusion["ranking_metric"].astype(str) == config.ranking_metric)
        & (pd.to_numeric(inclusion["top_k"], errors="coerce").astype(int) == int(k_star))
    ].copy()
    inc = inc[["latent_idx", "inclusion_frequency", "inclusion_count", "n_half_runs"]]
    ci = bootstrap_ci[bootstrap_ci["label"].astype(str) == str(label)].copy()
    ci_cols = [
        "latent_idx",
        "cohens_d_ci_lo",
        "cohens_d_ci_hi",
        "ci_excludes_zero",
        "rank_within_label",
    ]
    ci = ci[[col for col in ci_cols if col in ci.columns]].copy()
    q = cross_quality_lookup[cross_quality_lookup["label"].astype(str) == str(label)].copy()
    out = ranked.merge(inc, on="latent_idx", how="left")
    out = out.merge(ci, on="latent_idx", how="left")
    out = out.merge(q, on=["label", "latent_idx"], how="left")
    out["top_k_star"] = int(k_star)
    out["label_selection_status"] = str(selection_status)
    out["stable_claim_allowed"] = str(selection_status) == "stable_topk_found"
    out["inclusion_frequency"] = pd.to_numeric(out["inclusion_frequency"], errors="coerce").fillna(0.0)
    out["inclusion_count"] = pd.to_numeric(out["inclusion_count"], errors="coerce").fillna(0).astype(int)
    out["n_half_runs"] = pd.to_numeric(out["n_half_runs"], errors="coerce").fillna(0).astype(int)
    out["cohens_d_ci_lo"] = pd.to_numeric(out.get("cohens_d_ci_lo", np.nan), errors="coerce")
    out["ci_supported"] = out["cohens_d_ci_lo"] > 0
    out["cross_quality_max_auc_drop"] = pd.to_numeric(out.get("cross_quality_max_auc_drop", np.nan), errors="coerce")
    out["cross_quality_tested"] = out.get("cross_quality_tested", False).fillna(False).astype(bool)
    out["cross_quality_stable"] = (
        out["cross_quality_tested"]
        & (out["cross_quality_max_auc_drop"] < config.cross_quality_drop_threshold)
    )
    out["cross_quality_risk_or_missing"] = ~out["cross_quality_stable"]
    label_gate = (
        out["stable_claim_allowed"]
        if config.require_label_stability_gate
        else pd.Series(True, index=out.index)
    )
    quality_gate = (
        out["cross_quality_stable"]
        if config.require_cross_quality_gate
        else pd.Series(True, index=out.index)
    )
    stable = (
        (out["full_data_rank"] <= int(k_star))
        & (out["inclusion_frequency"] >= config.inclusion_stable_threshold)
        & out["ci_supported"]
        & label_gate
        & quality_gate
    )
    boundary = (
        (out["full_data_rank"] <= int(k_star))
        & (out["inclusion_frequency"] >= config.inclusion_boundary_threshold)
        & (out["inclusion_frequency"] < config.inclusion_stable_threshold)
        & out["ci_supported"]
        & label_gate
        & quality_gate
    )
    conditions = [stable, boundary, ~out["ci_supported"]]
    choices = ["stable_core", "boundary_candidate", "ci_not_supported"]
    if config.require_label_stability_gate:
        conditions.append(~out["stable_claim_allowed"])
        choices.append("performance_only_unstable_candidate")
    if config.require_cross_quality_gate:
        conditions.append(~out["cross_quality_stable"])
        choices.append("cross_quality_risk_or_missing")
    conditions.append(out["inclusion_frequency"] < config.inclusion_boundary_threshold)
    choices.append("unstable_inclusion")
    out["stable_set_role"] = np.select(
        conditions,
        choices,
        default="uncertain_candidate",
    )
    return out


def _write_report(path: Path, summary: pd.DataFrame, latent_set: pd.DataFrame, config: StableTopKSelectionConfig) -> None:
    lines = [
        "# Stable Top-K latent set selection",
        "",
        f"- Ranking metric: `{config.ranking_metric}`",
        f"- Max K: `{config.max_k}`",
        f"- Stable inclusion threshold: `{config.inclusion_stable_threshold:.2f}`",
        f"- Boundary inclusion threshold: `{config.inclusion_boundary_threshold:.2f}`",
        f"- Cross-quality gate enabled: `{config.require_cross_quality_gate}`",
        f"- Label-level stability gate enabled: `{config.require_label_stability_gate}`",
        "",
        "## K selection by label",
        "",
        "| label | K_auc | K_stab | K* | status | best AUC | stable core | boundary |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for _, row in summary.sort_values("label").iterrows():
        k_stab = "-" if pd.isna(row["k_stab"]) else str(int(row["k_stab"]))
        lines.append(
            f"| {row['label']} | {int(row['k_auc'])} | {k_stab} | {int(row['k_star'])} | "
            f"{row['selection_status']} | {float(row['best_auc']):.3f} | "
            f"{int(row['stable_core_count'])} | {int(row['boundary_candidate_count'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "- `stable_core`: full-data TopK* member with high repeated split inclusion and positive bootstrap CI.",
            "- `boundary_candidate`: full-data TopK* member with moderate repeated split inclusion and positive bootstrap CI.",
            "- Cross-quality and label-level stability status are retained as audit fields, but they are not hard gates unless the corresponding config flags are enabled.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stable_topk_selection(
    *,
    auc_curve_path: str | Path,
    topk_grid_summary_path: str | Path,
    inclusion_frequency_path: str | Path,
    association_matrix_path: str | Path,
    bootstrap_ci_path: str | Path,
    cross_quality_path: str | Path,
    output_dir: str | Path,
    config: StableTopKSelectionConfig | None = None,
) -> dict[str, Any]:
    config = config or StableTopKSelectionConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    auc_curve = pd.read_csv(auc_curve_path)
    topk_grid_summary = pd.read_csv(topk_grid_summary_path)
    inclusion = pd.read_csv(inclusion_frequency_path)
    association = pd.read_csv(association_matrix_path)
    bootstrap_ci = pd.read_csv(bootstrap_ci_path)
    cross_quality = pd.read_csv(cross_quality_path)
    cross_quality_lookup = _build_cross_quality_lookup(cross_quality)

    summary_rows: list[dict[str, Any]] = []
    latent_frames: list[pd.DataFrame] = []
    for label in config.labels:
        label = str(label).upper()
        label_curve = auc_curve[auc_curve["label"].astype(str) == label].copy()
        k_auc_info = _select_k_auc(label_curve, config)
        k_stab_info = _select_k_stab(
            label=label,
            k_auc=int(k_auc_info["k_auc"]),
            topk_grid_summary=topk_grid_summary,
            config=config,
        )
        k_star = int(k_stab_info["k_star"])
        label_latents = _build_stable_latent_rows(
            label=label,
            k_star=k_star,
            selection_status=str(k_stab_info["selection_status"]),
            association=association,
            inclusion=inclusion,
            bootstrap_ci=bootstrap_ci,
            cross_quality_lookup=cross_quality_lookup,
            config=config,
        )
        if not label_latents.empty:
            latent_frames.append(label_latents)
        stable_count = int((label_latents.get("stable_set_role", pd.Series(dtype=str)) == "stable_core").sum())
        boundary_count = int((label_latents.get("stable_set_role", pd.Series(dtype=str)) == "boundary_candidate").sum())
        summary_rows.append(
            {
                "label": label,
                **k_auc_info,
                **k_stab_info,
                "stable_core_count": stable_count,
                "boundary_candidate_count": boundary_count,
                "candidate_rows": int(len(label_latents)),
            }
        )

    summary = pd.DataFrame(summary_rows)
    latent_set = pd.concat(latent_frames, ignore_index=True, sort=False) if latent_frames else pd.DataFrame()
    union = (
        latent_set[latent_set["stable_set_role"].isin(["stable_core", "boundary_candidate"])]
        .groupby("latent_idx", as_index=False)
        .agg(
            labels=("label", lambda values: ",".join(sorted(set(map(str, values))))),
            max_inclusion_frequency=("inclusion_frequency", "max"),
            min_full_data_rank=("full_data_rank", "min"),
            roles=("stable_set_role", lambda values: ",".join(sorted(set(map(str, values))))),
        )
        if not latent_set.empty
        else pd.DataFrame()
    )

    summary_path = output_path / "stable_k_by_label.csv"
    latent_set_path = output_path / "stable_topk_latent_set.csv"
    union_path = output_path / "stable_topk_global_union.csv"
    report_path = output_path / "stable_topk_selection_report.md"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    latent_set.to_csv(latent_set_path, index=False, encoding="utf-8-sig")
    union.to_csv(union_path, index=False, encoding="utf-8-sig")
    _write_report(report_path, summary, latent_set, config)

    manifest = {
        "analysis": "stable_topk_selection",
        "config": asdict(config),
        "inputs": {
            "auc_curve": str(auc_curve_path),
            "topk_grid_summary": str(topk_grid_summary_path),
            "inclusion_frequency": str(inclusion_frequency_path),
            "association_matrix": str(association_matrix_path),
            "bootstrap_ci": str(bootstrap_ci_path),
            "cross_quality": str(cross_quality_path),
        },
        "outputs": {
            "stable_k_by_label": str(summary_path),
            "stable_topk_latent_set": str(latent_set_path),
            "stable_topk_global_union": str(union_path),
            "stable_topk_selection_report": str(report_path),
        },
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return {"summary": summary, "latent_set": latent_set, "union": union, "manifest": manifest}


__all__ = [
    "StableTopKSelectionConfig",
    "run_stable_topk_selection",
]
