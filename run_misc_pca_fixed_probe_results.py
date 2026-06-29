"""Build PCA-fixed probe result tables from existing ranked SAE outputs.

The ranked SAE top-n probes are unaffected by the PCA baseline definition. This
script reuses those rows and replaces the old post-standardized PCA baseline
with a fair full-rank PCA baseline equivalent to standardized raw hidden.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("outputs/misc_full_sae_eval/interpretability/ranked_sae_subspace_probe")
DEFAULT_OUTPUT_DIR = Path("outputs/misc_full_sae_eval/interpretability/ranked_sae_subspace_probe_pca_fixed")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _replace_pca_fold_rows(fold_rows: pd.DataFrame) -> pd.DataFrame:
    keep = fold_rows[fold_rows["representation"] != "pca_raw_hidden"].copy()
    raw = fold_rows[fold_rows["representation"] == "raw_hidden"].copy()
    pca = raw.copy()
    pca["representation"] = "pca_raw_hidden"
    pca["pca_n_components"] = pca["n_features"].astype(int)
    pca["pca_max_components"] = pca["n_features"].astype(int)
    pca["pca_explained_variance_ratio_sum"] = 1.0
    pca["pca_svd_solver"] = "full_rank_equivalence"
    pca["pca_pre_standardized"] = True
    pca["pca_post_standardized"] = False
    pca["pca_equivalent_to_standardized_raw"] = True
    return pd.concat([keep, pca], ignore_index=True)


def _replace_pca_by_label_rows(by_label: pd.DataFrame) -> pd.DataFrame:
    keep = by_label[by_label["representation"] != "pca_raw_hidden"].copy()
    raw = by_label[by_label["representation"] == "raw_hidden"].copy()
    pca = raw.copy()
    pca["representation"] = "pca_raw_hidden"
    return pd.concat([keep, pca], ignore_index=True).sort_values(["representation", "label"]).reset_index(drop=True)


def _replace_pca_summary_rows(summary: pd.DataFrame) -> pd.DataFrame:
    keep = summary[summary["representation"] != "pca_raw_hidden"].copy()
    raw = summary[summary["representation"] == "raw_hidden"].copy()
    pca = raw.copy()
    pca["representation"] = "pca_raw_hidden"
    return pd.concat([keep, pca], ignore_index=True)


def _refresh_convergence(convergence: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    baseline_auc = {
        str(row["representation"]): float(row["macro_auc"])
        for _, row in summary[summary["subspace_ranking"].isna()].iterrows()
        if pd.notna(row.get("macro_auc"))
    }
    refreshed = convergence.copy()
    for baseline in ("full_sae_latents", "raw_hidden", "pca_raw_hidden"):
        if baseline in baseline_auc:
            refreshed[f"delta_macro_auc_vs_{baseline}"] = refreshed["macro_auc"].astype(float) - baseline_auc[baseline]
    return refreshed


def _write_pca_fixed_report(output_dir: Path, summary: pd.DataFrame, convergence: pd.DataFrame) -> None:
    base = summary[summary["subspace_ranking"].isna()].set_index("representation")
    raw_auc = float(base.loc["raw_hidden", "macro_auc"])
    pca_auc = float(base.loc["pca_raw_hidden", "macro_auc"])
    sae_auc = float(base.loc["full_sae_latents", "macro_auc"])
    best = convergence.sort_values("macro_auc", ascending=False).iloc[0]
    lines = [
        "# PCA 修正后的 representation probe 结果说明",
        "",
        "本目录复用既有 SAE top-n probe 结果，并将旧的 `pca_raw_hidden` baseline 替换为修正后的 full-rank PCA 等价基线。",
        "",
        "## 修正内容",
        "",
        "- 旧口径：raw hidden 直接标准化；PCA 先在未标准化 raw 上拟合，再对 PCA 主成分二次标准化。",
        "- 新口径：训练折内先标准化 raw hidden；full-rank PCA 作为该标准化空间的正交旋转等价基线；PCA 后不再二次标准化。",
        "- 因为 full-rank PCA + L2 线性 probe 与标准化 raw hidden 在几何上等价，本脚本直接复用 raw hidden fold 指标作为修正后的 `pca_raw_hidden`。",
        "",
        "## 关键结果",
        "",
        f"- Raw hidden macro AUC: {raw_auc:.3f}",
        f"- Fixed full PCA macro AUC: {pca_auc:.3f}",
        f"- Full SAE macro AUC: {sae_auc:.3f}",
        f"- Best SAE top-n: `{best['subspace_ranking']}` n={int(best['top_n'])}, macro AUC={float(best['macro_auc']):.3f}",
        "",
        "## 解释边界",
        "",
        "- 修正后不能再把 PCA 写成明显弱基线；它现在是 raw hidden 的公平 full-rank PCA 对照。",
        "- SAE 的优势应主要表述为稀疏、可分解、便于 latent-level evidence 审计，而不是 full-rank AUC 明显强于 PCA。",
        "- SAE top-n 超过 raw/PCA 仍应解释为训练折内监督特征选择带来的 probe-space 去噪/正则化效果，不是因果机制证明。",
        "",
    ]
    (output_dir / "pca_fixed_probe_report_zh.md").write_text("\n".join(lines), encoding="utf-8")


def build_pca_fixed_results(input_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fold_rows = pd.read_csv(input_path / "full_probe_by_label.csv")
    by_label = pd.read_csv(input_path / "full_probe_by_label_summary.csv")
    summary = pd.read_csv(input_path / "full_probe_summary.csv")
    convergence = pd.read_csv(input_path / "ranked_sae_subspace_convergence.csv")

    fixed_fold_rows = _replace_pca_fold_rows(fold_rows)
    fixed_by_label = _replace_pca_by_label_rows(by_label)
    fixed_summary = _replace_pca_summary_rows(summary)
    fixed_convergence = _refresh_convergence(convergence, fixed_summary)

    fixed_fold_rows.to_csv(output_path / "full_probe_by_label.csv", index=False)
    fixed_by_label.to_csv(output_path / "full_probe_by_label_summary.csv", index=False)
    fixed_summary.to_csv(output_path / "full_probe_summary.csv", index=False)
    fixed_convergence.to_csv(output_path / "ranked_sae_subspace_convergence.csv", index=False)

    selected = input_path / "ranked_sae_subspace_selected_latents.csv"
    if selected.exists():
        shutil.copy2(selected, output_path / selected.name)

    payload = {
        "analysis": "misc_ranked_probe_pca_fixed_results",
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "pca_fix": {
            "old_issue": "post-PCA standardization changed the L2 probe regularization geometry",
            "new_policy": "full-rank PCA is treated as equivalent to standardized raw hidden; no post-PCA standardization",
            "pca_equivalent_to_standardized_raw": True,
        },
        "baseline_macro_auc": {
            str(row["representation"]): float(row["macro_auc"])
            for _, row in fixed_summary[fixed_summary["subspace_ranking"].isna()].iterrows()
        },
        "files": {
            "full_probe_by_label": str(output_path / "full_probe_by_label.csv"),
            "full_probe_by_label_summary": str(output_path / "full_probe_by_label_summary.csv"),
            "full_probe_summary": str(output_path / "full_probe_summary.csv"),
            "ranked_sae_subspace_convergence": str(output_path / "ranked_sae_subspace_convergence.csv"),
            "ranked_sae_subspace_selected_latents": str(output_path / "ranked_sae_subspace_selected_latents.csv"),
            "pca_fixed_probe_report": str(output_path / "pca_fixed_probe_report_zh.md"),
            "manifest": str(output_path / "manifest.json"),
        },
    }
    _write_pca_fixed_report(output_path, fixed_summary, fixed_convergence)
    (output_path / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PCA-fixed full representation probe tables by reusing existing ranked SAE top-n results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_pca_fixed_results(args.input_dir, args.output_dir)
    print("Completed PCA-fixed probe result rebuild.")
    print(f"Output dir: {args.output_dir}")
    print(f"Baseline macro AUC: {payload['baseline_macro_auc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
