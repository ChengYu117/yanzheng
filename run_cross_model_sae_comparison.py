"""Compare Llama SAE and GemmaScope SAE MISC interpretability outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


DEFAULT_LABELS = ("RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _best_assoc(matrix: pd.DataFrame, labels: tuple[str, ...]) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame(columns=["label", "best_directional_auc", "best_abs_cohens_d"])
    rows = []
    for label in labels:
        group = matrix[matrix["label"].astype(str).str.upper() == label]
        if group.empty:
            rows.append({"label": label, "best_directional_auc": np.nan, "best_abs_cohens_d": np.nan})
            continue
        sort_cols = [col for col in ["directional_auc", "abs_cohens_d"] if col in group.columns]
        group = group.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        best = group.iloc[0]
        rows.append(
            {
                "label": label,
                "best_directional_auc": float(best.get("directional_auc", np.nan)),
                "best_abs_cohens_d": float(best.get("abs_cohens_d", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _collect_model(root: Path, name: str, labels: tuple[str, ...]) -> pd.DataFrame:
    assoc = _read_csv(root / "functional" / "misc_label_mapping" / "latent_label_matrix.csv")
    frag = _read_csv(root / "interpretability" / "latent_space_search_v2" / "fragmentation_v2.csv")
    minimal = _read_csv(
        root / "interpretability" / "minimal_sufficient_subspace_v2" / "minimal_sufficient_summary_v2.csv"
    )
    best = _best_assoc(assoc, labels)
    out = pd.DataFrame({"label": list(labels)})
    out = out.merge(best, on="label", how="left")
    if not frag.empty:
        frag_cols = [
            col
            for col in [
                "label",
                "thresholded_latent_count",
                "effective_fragmentation",
                "fragmentation_class",
                "selection_status",
            ]
            if col in frag.columns
        ]
        out = out.merge(frag[frag_cols], on="label", how="left")
    if not minimal.empty:
        minimal_cols = [
            col
            for col in [
                "label",
                "formal_status",
                "full_auc_mean",
                "minimal_k_median",
                "mean_pairwise_jaccard_between_folds",
                "predictive_redundancy_ratio",
            ]
            if col in minimal.columns
        ]
        out = out.merge(minimal[minimal_cols], on="label", how="left")
    out.insert(0, "model", name)
    return out


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _write_report(
    *,
    output_dir: Path,
    comparison: pd.DataFrame,
    llama_root: Path,
    gemma_root: Path,
) -> Path:
    report_path = output_dir / "model_specificity_report.md"
    pivot_auc = comparison.pivot(index="label", columns="model", values="best_directional_auc")
    pivot_k = (
        comparison.pivot(index="label", columns="model", values="minimal_k_median")
        if "minimal_k_median" in comparison.columns
        else pd.DataFrame()
    )
    lines = [
        "# Cross-Model SAE Specificity Comparison",
        "",
        "## Inputs",
        "",
        f"- Llama root: `{llama_root}`",
        f"- Gemma root: `{gemma_root}`",
        "",
        "## Label-Level Comparison",
        "",
        "| Label | Llama best AUC | Gemma best AUC | AUC delta | Llama minimal K | Gemma minimal K |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in DEFAULT_LABELS:
        llama_auc = _safe_float(pivot_auc.loc[label, "llama"] if label in pivot_auc.index and "llama" in pivot_auc.columns else np.nan)
        gemma_auc = _safe_float(pivot_auc.loc[label, "gemma"] if label in pivot_auc.index and "gemma" in pivot_auc.columns else np.nan)
        llama_k = _safe_float(pivot_k.loc[label, "llama"] if not pivot_k.empty and label in pivot_k.index and "llama" in pivot_k.columns else np.nan)
        gemma_k = _safe_float(pivot_k.loc[label, "gemma"] if not pivot_k.empty and label in pivot_k.index and "gemma" in pivot_k.columns else np.nan)
        delta = None if llama_auc is None or gemma_auc is None else gemma_auc - llama_auc
        lines.append(
            "| {label} | {la} | {ga} | {delta} | {lk} | {gk} |".format(
                label=label,
                la=f"{llama_auc:.3f}" if llama_auc is not None else "NA",
                ga=f"{gemma_auc:.3f}" if gemma_auc is not None else "NA",
                delta=f"{delta:+.3f}" if delta is not None else "NA",
                lk=f"{llama_k:.0f}" if llama_k is not None else "NA",
                gk=f"{gemma_k:.0f}" if gemma_k is not None else "NA",
            )
        )
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- This comparison does not align latent IDs across models.",
            "- Stable structures across models support model-robust MISC representation claims.",
            "- Gemma layer 18 is RE-selected; QU-specific claims should mention that QU's Gemma probe optimum was layer 10.",
            "- Differences here are representational and correlational, not causal intervention results.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-model SAE comparison between Llama and GemmaScope outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--llama-root", default="outputs/misc_full_sae_eval")
    parser.add_argument("--gemma-root", default="outputs/gemma3_l18_gemmascope_sae_eval")
    parser.add_argument(
        "--output-dir",
        default="outputs/gemma3_l18_gemmascope_sae_eval/interpretability/model_specificity_comparison",
    )
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = tuple(label.upper() for label in args.labels)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    llama_root = Path(args.llama_root)
    gemma_root = Path(args.gemma_root)

    comparison = pd.concat(
        [
            _collect_model(llama_root, "llama", labels),
            _collect_model(gemma_root, "gemma", labels),
        ],
        ignore_index=True,
    )
    comparison.to_csv(output_dir / "label_level_model_comparison.csv", index=False)
    wide = comparison.pivot(index="label", columns="model")
    wide.to_csv(output_dir / "label_level_model_comparison_wide.csv")
    report_path = _write_report(
        output_dir=output_dir,
        comparison=comparison,
        llama_root=llama_root,
        gemma_root=gemma_root,
    )
    summary = {
        "llama_root": str(llama_root),
        "gemma_root": str(gemma_root),
        "labels": list(labels),
        "comparison_rows": int(len(comparison)),
        "report": str(report_path),
    }
    (output_dir / "model_specificity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Completed cross-model SAE comparison.")
    print(f"Output dir: {output_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

