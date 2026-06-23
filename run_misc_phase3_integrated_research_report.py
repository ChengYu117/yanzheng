"""Phase-3 integrated research report for MISC SAE analyses.

This script synthesizes full-representation probes, ranked SAE top-n probe
results, latent evidence packets, and phase-2 review status into a cautious
Chinese report. It answers the research question at the evidence level that is
currently supported: predictive decodability and correlational interpretability,
not causal mechanism proof.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LABELS = ["RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF"]
DEFAULT_PROBE_DIR = Path("outputs/misc_full_sae_eval/interpretability/ranked_sae_subspace_probe")
DEFAULT_TOP20_DIR = Path("outputs/misc_full_sae_eval/interpretability/top20_cohensd_latent_utterances")
DEFAULT_EVIDENCE_DIR = DEFAULT_TOP20_DIR / "latent_evidence_packets"
DEFAULT_FUNCTION_DIR = DEFAULT_TOP20_DIR / "latent_function_induction"
DEFAULT_OUTPUT_DIR = Path("outputs/misc_full_sae_eval/interpretability/phase3_integrated_research_report")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


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
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_int(value: Any) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return str(int(value))
    except (TypeError, ValueError):
        return "NA"


def _is_integer_display_column(col: str) -> bool:
    return (
        col in {"mean_n_features", "best_n", "n_latents", "latents_ge_0.75_match", "latents_lt_0.33_match"}
        or col.startswith("first_n_")
        or col.endswith("_best_n")
        or col.endswith("_n_within_0.05_full_sae")
        or col.endswith("_n_at_least_full_sae")
    )


def _format_cell(col: str, value: Any) -> str:
    if _is_integer_display_column(col):
        return _fmt_int(value)
    if isinstance(value, float):
        return _fmt(value)
    if isinstance(value, (np.floating,)):
        return _fmt(float(value))
    return str(value)


def _markdown_table(df: pd.DataFrame, columns: list[str], headers: list[str] | None = None, max_rows: int | None = None) -> str:
    if df.empty:
        return "（无数据）"
    view = df.loc[:, [col for col in columns if col in df.columns]].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    headers = headers or list(view.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        cells: list[str] = []
        for col in view.columns:
            cells.append(_format_cell(col, row[col]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _first_n_within(group: pd.DataFrame, metric: str, target: float, tolerance: float = 0.05) -> int | None:
    if group.empty or pd.isna(target):
        return None
    near = group[(group[metric] - float(target)).abs() <= tolerance].sort_values("top_n")
    return None if near.empty else int(near.iloc[0]["top_n"])


def _first_n_at_least(group: pd.DataFrame, metric: str, target: float) -> int | None:
    if group.empty or pd.isna(target):
        return None
    reached = group[group[metric] >= float(target)].sort_values("top_n")
    return None if reached.empty else int(reached.iloc[0]["top_n"])


def _parse_counts(text: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not isinstance(text, str) or not text.strip():
        return counts
    for part in text.split(","):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        try:
            counts[key] += int(float(value.strip()))
        except ValueError:
            continue
    return counts


def _top_counter_items(counter: Counter[str], exclude: str, n: int = 3) -> str:
    for key in list(counter):
        if key == exclude:
            del counter[key]
    if not counter:
        return ""
    return ", ".join(f"{key}:{value}" for key, value in counter.most_common(n))


def build_baseline_table(summary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    baseline = summary[summary["subspace_ranking"].isna()].copy()
    order = {"raw_hidden": 0, "full_sae_latents": 1, "pca_raw_hidden": 2}
    names = {
        "raw_hidden": "Raw hidden",
        "full_sae_latents": "Full SAE latents",
        "pca_raw_hidden": "Full PCA(raw hidden)",
    }
    baseline["display_name"] = baseline["representation"].map(names).fillna(baseline["representation"])
    baseline["sort_key"] = baseline["representation"].map(order).fillna(99)
    baseline = baseline.sort_values("sort_key")
    aucs = {str(row["representation"]): float(row["macro_auc"]) for _, row in baseline.iterrows()}
    return baseline, aucs


def build_overall_topn_table(convergence: pd.DataFrame, baseline_auc: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ranking, group in convergence.groupby("subspace_ranking", sort=False):
        group = group.sort_values("top_n")
        best = group.loc[group["macro_auc"].idxmax()]
        last = group.iloc[-1]
        rows.append(
            {
                "ranking": ranking,
                "best_n": int(best["top_n"]),
                "best_macro_auc": float(best["macro_auc"]),
                "n200_macro_auc": float(last["macro_auc"]),
                "first_n_within_0.05_full_sae": _first_n_within(group, "macro_auc", baseline_auc.get("full_sae_latents", np.nan)),
                "first_n_at_least_full_sae": _first_n_at_least(group, "macro_auc", baseline_auc.get("full_sae_latents", np.nan)),
                "first_n_within_0.05_raw_hidden": _first_n_within(group, "macro_auc", baseline_auc.get("raw_hidden", np.nan)),
                "first_n_at_least_raw_hidden": _first_n_at_least(group, "macro_auc", baseline_auc.get("raw_hidden", np.nan)),
                "delta_best_vs_full_sae": float(best["macro_auc"]) - baseline_auc.get("full_sae_latents", np.nan),
                "delta_best_vs_raw_hidden": float(best["macro_auc"]) - baseline_auc.get("raw_hidden", np.nan),
                "delta_best_vs_pca": float(best["macro_auc"]) - baseline_auc.get("pca_raw_hidden", np.nan),
                "last_4_step_mean_auc_gain": float(last.get("last_4_step_mean_auc_gain", np.nan)),
                "platformed_by_last_steps": bool(last.get("platformed_by_last_steps", False)),
            }
        )
    return pd.DataFrame(rows)


def build_label_topn_table(by_label: pd.DataFrame) -> pd.DataFrame:
    baseline = by_label[by_label["subspace_ranking"].isna()].copy()
    base_auc = baseline.pivot(index="label", columns="representation", values="probe_auc_mean")
    subspaces = by_label[by_label["subspace_ranking"].notna()].copy()
    rows: list[dict[str, Any]] = []
    for (label, ranking), group in subspaces.groupby(["label", "subspace_ranking"], sort=False):
        group = group.sort_values("top_n")
        best = group.loc[group["probe_auc_mean"].idxmax()]
        diffs = group["probe_auc_mean"].diff().dropna()
        full = float(base_auc.loc[label, "full_sae_latents"]) if "full_sae_latents" in base_auc.columns else np.nan
        raw = float(base_auc.loc[label, "raw_hidden"]) if "raw_hidden" in base_auc.columns else np.nan
        pca = float(base_auc.loc[label, "pca_raw_hidden"]) if "pca_raw_hidden" in base_auc.columns else np.nan
        rows.append(
            {
                "label": label,
                "ranking": ranking,
                "full_sae_auc": full,
                "raw_hidden_auc": raw,
                "pca_auc": pca,
                "best_n": int(best["top_n"]),
                "best_topn_auc": float(best["probe_auc_mean"]),
                "n200_auc": float(group.iloc[-1]["probe_auc_mean"]),
                "first_n_within_0.05_full_sae": _first_n_within(group, "probe_auc_mean", full),
                "first_n_at_least_full_sae": _first_n_at_least(group, "probe_auc_mean", full),
                "first_n_within_0.05_raw_hidden": _first_n_within(group, "probe_auc_mean", raw),
                "first_n_at_least_raw_hidden": _first_n_at_least(group, "probe_auc_mean", raw),
                "delta_best_vs_full_sae": float(best["probe_auc_mean"]) - full,
                "delta_best_vs_raw_hidden": float(best["probe_auc_mean"]) - raw,
                "delta_best_vs_pca": float(best["probe_auc_mean"]) - pca,
                "last_4_step_mean_auc_gain": float(diffs.tail(4).mean()) if not diffs.empty else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["label", "ranking"]).reset_index(drop=True)


def build_evidence_label_table(evidence_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, group in evidence_summary.groupby("target_label", sort=False):
        duplicate_den = group["n_top_activating"] + group["n_high_non_target"] + group["n_random_target"]
        co_counts: Counter[str] = Counter()
        for value in group.get("active_label_counts_top_activating", pd.Series(dtype=str)):
            co_counts.update(_parse_counts(value))
        target_mean = float(group["top_activating_target_match_rate"].mean())
        if target_mean >= 0.75:
            grade = "strong"
        elif target_mean >= 0.50:
            grade = "moderate"
        elif target_mean >= 0.33:
            grade = "weak"
        else:
            grade = "very_weak"
        rows.append(
            {
                "label": label,
                "n_latents": int(len(group)),
                "mean_target_match": target_mean,
                "min_target_match": float(group["top_activating_target_match_rate"].min()),
                "max_target_match": float(group["top_activating_target_match_rate"].max()),
                "latents_ge_0.75_match": int((group["top_activating_target_match_rate"] >= 0.75).sum()),
                "latents_lt_0.33_match": int((group["top_activating_target_match_rate"] < 0.33).sum()),
                "mean_precision_at_50": float(group["precision_at_50"].mean()),
                "mean_directional_auc": float(group["directional_auc"].mean()),
                "mean_cohens_d": float(group["cohens_d"].mean()),
                "mean_duplicate_rate": float((group["duplicate_text_row_count"] / duplicate_den).mean()),
                "mean_unique_file_count": float(group["unique_file_count"].mean()),
                "top_coactive_labels": _top_counter_items(co_counts, exclude=label, n=3),
                "evidence_grade": grade,
            }
        )
    return pd.DataFrame(rows)


def build_label_integrated_table(
    by_label: pd.DataFrame,
    label_topn: pd.DataFrame,
    evidence_label: pd.DataFrame,
) -> pd.DataFrame:
    baseline = by_label[by_label["subspace_ranking"].isna()].copy()
    base_auc = baseline.pivot(index="label", columns="representation", values="probe_auc_mean").reset_index()
    base_auc = base_auc.rename(
        columns={
            "full_sae_latents": "full_sae_auc",
            "raw_hidden": "raw_hidden_auc",
            "pca_raw_hidden": "pca_auc",
        }
    )
    directional = label_topn[label_topn["ranking"] == "directional_auc"].copy()
    directional = directional.rename(
        columns={
            "best_n": "directional_best_n",
            "best_topn_auc": "directional_best_auc",
            "first_n_within_0.05_full_sae": "directional_n_within_0.05_full_sae",
            "first_n_at_least_full_sae": "directional_n_at_least_full_sae",
        }
    )
    keep = [
        "label",
        "directional_best_n",
        "directional_best_auc",
        "directional_n_within_0.05_full_sae",
        "directional_n_at_least_full_sae",
        "delta_best_vs_raw_hidden",
    ]
    merged = base_auc.merge(directional[[col for col in keep if col in directional.columns]], on="label", how="left")
    merged = merged.merge(evidence_label, on="label", how="left")
    return merged.sort_values("label").reset_index(drop=True)


def _phase2_status(function_dir: Path) -> dict[str, Any]:
    manifest = _read_json(function_dir / "manifest.json", default={}) or {}
    status_counts = manifest.get("status_counts", {})
    return {
        "dry_run_prompts": bool(manifest.get("dry_run_prompts", False)),
        "n_reviews": int(manifest.get("n_reviews", 0) or 0),
        "n_success": int(manifest.get("n_success", 0) or 0),
        "n_failed": int(manifest.get("n_failed", 0) or 0),
        "n_pending_dry_run": int(manifest.get("n_pending_dry_run", status_counts.get("pending_dry_run", 0)) or 0),
        "manifest": manifest,
    }


def _claim_strength(evidence_label: pd.DataFrame, phase2: dict[str, Any]) -> list[dict[str, str]]:
    function_status = "未完成" if phase2["n_success"] == 0 else "部分完成" if phase2["n_success"] < phase2["n_reviews"] else "完成"
    return [
        {
            "claim": "MISC 标签信息可从 SAE features 中线性读出",
            "support": "强",
            "reason": "Full SAE macro AUC 高，且 top-n SAE 子空间可接近或超过 full SAE。",
        },
        {
            "claim": "SAE 比 PCA 更适合当前标签识别",
            "support": "强",
            "reason": "Full SAE 与 top-n SAE macro AUC 明显高于 full PCA(raw hidden)。",
        },
        {
            "claim": "SAE 明显优于 raw hidden 的完整表征",
            "support": "不支持",
            "reason": "Full SAE macro AUC 略低于 raw hidden；top-n 超过 raw hidden属于监督筛选后的 probe-space 现象。",
        },
        {
            "claim": "单个或少数 top latents 就等价于 MISC 行为机制",
            "support": "不支持",
            "reason": "Top20 target-match 在标签间差异很大，RES/SU 较弱；且没有因果干预。",
        },
        {
            "claim": "可以给每个 latent 下稳定功能名称",
            "support": function_status,
            "reason": "第二阶段结构化 LLM 功能归纳当前状态为 dry-run 或未全量成功；只能使用统计证据和已有审计作为候选线索。",
        },
        {
            "claim": "RES/REC 的 SAE latent 反映了来访者内容关系",
            "support": "弱",
            "reason": "第一阶段没有前一句 client utterance，无法强判反映是否复述或改写来访者内容。",
        },
    ]


def write_report(
    *,
    output_path: Path,
    baseline: pd.DataFrame,
    overall_topn: pd.DataFrame,
    label_topn: pd.DataFrame,
    evidence_label: pd.DataFrame,
    integrated: pd.DataFrame,
    phase2: dict[str, Any],
    claim_strength: list[dict[str, str]],
    inputs: dict[str, str],
) -> None:
    raw_auc = float(baseline.loc[baseline["representation"] == "raw_hidden", "macro_auc"].iloc[0])
    sae_auc = float(baseline.loc[baseline["representation"] == "full_sae_latents", "macro_auc"].iloc[0])
    pca_auc = float(baseline.loc[baseline["representation"] == "pca_raw_hidden", "macro_auc"].iloc[0])
    best_topn = overall_topn.loc[overall_topn["best_macro_auc"].idxmax()]
    directional = overall_topn[overall_topn["ranking"] == "directional_auc"]
    directional_text = ""
    if not directional.empty:
        row = directional.iloc[0]
        directional_text = (
            f"`directional_auc` ranking 的最佳点为 n={int(row['best_n'])}, "
            f"macro AUC={_fmt(row['best_macro_auc'])}。"
        )

    dry_run_note = (
        "第二阶段结构化 LLM 功能归纳目前是 dry-run，占位结果不能当作真实语义审阅。"
        if phase2["n_success"] == 0
        else f"第二阶段已有 {phase2['n_success']}/{phase2['n_reviews']} 个 latent 成功完成结构化审阅。"
    )

    claim_df = pd.DataFrame(claim_strength)
    label_display = integrated.copy()
    for col in [
        "directional_n_within_0.05_full_sae",
        "directional_n_at_least_full_sae",
        "directional_best_n",
    ]:
        if col in label_display.columns:
            label_display[col] = label_display[col].map(_fmt_int)

    directional_label = label_topn[label_topn["ranking"] == "directional_auc"].copy()
    abs_label = label_topn[label_topn["ranking"] == "abs_cohens_d"].copy()
    weakest = evidence_label.sort_values("mean_target_match").head(3)
    strongest = evidence_label.sort_values("mean_target_match", ascending=False).head(3)

    lines = [
        "# 第三阶段综合分析报告：MISC 标签、SAE features 与 LLM 内部表征",
        "",
        "## 0. 结论先行",
        "",
        (
            f"可以较有把握地回答：人工 MISC 标签**能够从 LLM 内部 SAE features 中被线性识别出来**。"
            f"Full SAE latents 的 macro AUC={_fmt(sae_auc)}，接近 raw hidden 的 {_fmt(raw_auc)}，"
            f"并明显高于 full PCA(raw hidden) 的 {_fmt(pca_auc)}。"
        ),
        "",
        (
            f"但更强的问题，即“LLM 是否以 MISC codebook 意义上的咨询行为机制理解这些标签”，"
            f"当前证据**还不能证明**。现有结果主要支持 predictive decodability 和 label-latent association；"
            f"缺少 client 前文、token-level attribution、ablation/steering 等因果证据。{dry_run_note}"
        ),
        "",
        (
            f"一个值得保留但要谨慎表述的发现是：监督筛选出的 SAE top-n 子空间可以超过完整 raw hidden。"
            f"{directional_text}这不是必然错误，因为 top-n 是按训练折标签相关性做特征选择，"
            f"可能起到去噪和正则化作用；但它不是“SAE 无监督表征天然强于 raw hidden”的结论。"
        ),
        "",
        "## 1. 本阶段整合了哪些证据",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {
                        "证据": "Full representation probe",
                        "文件": Path(inputs["full_probe_summary"]).name,
                        "回答": "标签信息是否可从 full SAE/raw/PCA 中线性读出",
                    },
                    {
                        "证据": "Ranked SAE top-n probe",
                        "文件": Path(inputs["ranked_convergence"]).name,
                        "回答": "少量 top latents 是否足够接近 full SAE 或 raw hidden",
                    },
                    {
                        "证据": "Latent evidence packets",
                        "文件": Path(inputs["evidence_summary"]).name,
                        "回答": "Top20 latents 的高激活语句是否真的落在目标标签上",
                    },
                    {
                        "证据": "Phase-2 function induction status",
                        "文件": Path(inputs["function_manifest"]).name,
                        "回答": "结构化语义审阅是否已经真实完成",
                    },
                ]
            ),
            ["证据", "文件", "回答"],
        ),
        "",
        "## 2. 研究问题与子问题回答",
        "",
        "### Q1：人工 MISC 标签能否从 SAE features 中被识别出来？",
        "",
        (
            f"能，但要限定为“线性探针可识别”。Full SAE macro AUC={_fmt(sae_auc)}，"
            f"说明 SAE feature 向量中保留了大量标签相关信息。这个结论比单个 feature 的 AUC 更强，"
            f"因为它考察的是整个 SAE 表征或由多个 latents 组成的子空间。"
        ),
        "",
        "### Q2：SAE 相比 raw hidden 和 PCA 如何？",
        "",
        (
            f"Raw hidden 仍是最强 full representation 基线之一，macro AUC={_fmt(raw_auc)}；"
            f"Full SAE 略低 {_fmt(raw_auc - sae_auc)}，但非常接近。Full PCA(raw hidden)={_fmt(pca_auc)} 明显较弱，"
            f"说明当前 MISC 标签判别方向不等同于最大方差方向。"
        ),
        "",
        "### Q3：需要多少 SAE latents 才能接近 full SAE？",
        "",
        "接近定义为 AUC 与 full SAE 相差不超过 0.05。整体 macro 层面的结果如下：",
        "",
        _markdown_table(
            overall_topn,
            [
                "ranking",
                "best_n",
                "best_macro_auc",
                "first_n_within_0.05_full_sae",
                "first_n_at_least_full_sae",
                "first_n_within_0.05_raw_hidden",
                "first_n_at_least_raw_hidden",
                "platformed_by_last_steps",
            ],
            [
                "ranking",
                "best n",
                "best AUC",
                "n within .05 full SAE",
                "n >= full SAE",
                "n within .05 raw",
                "n >= raw",
                "末端平台",
            ],
        ),
        "",
        "逐标签的 `directional_auc` top-n 结果如下：",
        "",
        _markdown_table(
            directional_label,
            [
                "label",
                "full_sae_auc",
                "raw_hidden_auc",
                "best_n",
                "best_topn_auc",
                "first_n_within_0.05_full_sae",
                "first_n_at_least_full_sae",
                "first_n_within_0.05_raw_hidden",
                "first_n_at_least_raw_hidden",
            ],
            [
                "label",
                "full SAE",
                "raw",
                "best n",
                "best top-n",
                "n within .05 full SAE",
                "n >= full SAE",
                "n within .05 raw",
                "n >= raw",
            ],
        ),
        "",
        "### Q4：Top20 Cohen's d latents 的高激活语句能否支持标签解释？",
        "",
        "支持程度明显分标签。以下表格聚合每个标签 20 个 latents 的 top50 高激活语句 target-match：",
        "",
        _markdown_table(
            evidence_label,
            [
                "label",
                "mean_target_match",
                "min_target_match",
                "max_target_match",
                "latents_ge_0.75_match",
                "latents_lt_0.33_match",
                "mean_duplicate_rate",
                "top_coactive_labels",
                "evidence_grade",
            ],
            [
                "label",
                "mean match",
                "min",
                "max",
                "latents >=.75",
                "latents <.33",
                "duplicate rate",
                "top co-labels",
                "grade",
            ],
        ),
        "",
        (
            "最清晰的标签是 "
            + ", ".join(f"{row['label']}({_fmt(row['mean_target_match'])})" for _, row in strongest.iterrows())
            + "；最弱的是 "
            + ", ".join(f"{row['label']}({_fmt(row['mean_target_match'])})" for _, row in weakest.iterrows())
            + "。这意味着不能把所有 Top20 latents 都当成同等质量的语义机制证据。"
        ),
        "",
        "### Q5：这些结果是否已经说明 LLM 的 MISC 理解机制？",
        "",
        (
            "还不能。当前可以说的是：MISC 标签相关信息在 SAE 空间中可读出，并且部分标签的高激活 latents "
            "呈现与标签一致的语言模式。不能说的是：这些 latents 已经构成模型实际分类或生成咨询行为的因果机制。"
            "尤其是 RES/REC 这类依赖前文 client utterance 的标签，仅凭 counselor 当前句子无法可靠判断反映是否复述、改写或深化了来访者内容。"
        ),
        "",
        "## 3. Full representation 对比",
        "",
        _markdown_table(
            baseline,
            [
                "display_name",
                "mean_n_features",
                "macro_auc",
                "macro_average_precision",
                "macro_f1",
                "macro_balanced_accuracy",
                "macro_accuracy",
            ],
            ["representation", "dim", "macro AUC", "AP", "macro F1", "balanced acc", "accuracy"],
        ),
        "",
        "解释：AUC 看排序能力，F1 看固定阈值下的离散分类质量。Top-n 子空间 AUC 高，不代表阈值后的 F1 或真实机制也一定更强。",
        "",
        "## 4. 逐标签综合视图",
        "",
        _markdown_table(
            label_display,
            [
                "label",
                "full_sae_auc",
                "raw_hidden_auc",
                "pca_auc",
                "directional_best_n",
                "directional_best_auc",
                "directional_n_within_0.05_full_sae",
                "directional_n_at_least_full_sae",
                "mean_target_match",
                "evidence_grade",
            ],
            [
                "label",
                "full SAE AUC",
                "raw AUC",
                "PCA AUC",
                "best n",
                "best top-n AUC",
                "n within .05 full SAE",
                "n >= full SAE",
                "Top20 mean match",
                "evidence",
            ],
        ),
        "",
        "## 5. 为什么 top185 子空间可能比 full SAE 和 raw hidden 更高？",
        "",
        "这件事看起来反直觉，但不必首先解释为代码错误。更合理的解释有四点：",
        "",
        "1. **监督特征选择的正则化效应**：top-n 不是无监督压缩，而是在每个训练折内按标签相关性筛选 SAE features。它会移除大量与当前标签无关或噪声较高的 latents。",
        "2. **SAE 重构损失不等于标签信息损失**：SAE 相对 raw hidden 有重构误差，但标签判别所需信息可能集中在少数保留下来的 sparse features 中。",
        "3. **Full SAE 维度过高会带来 probe 噪声**：32768 维完整 SAE 向量包含很多无关 feature；在有限样本下，线性探针可能不如筛选后的子空间稳定。",
        "4. **raw hidden 是密集纠缠表征**：raw hidden 信息完整，但标签方向可能分散在纠缠维度中；SAE top-n 通过标签监督把相关方向挑了出来。",
        "",
        "我核对过当前实现的关键路径：feature ranking 在每个 fold 内用训练折 SAE features 和训练折标签计算，测试折只用于评估。因此从代码结构看，top-n 高于 raw hidden 并不直接构成测试集泄漏证据。不过它仍然是 probe-space supervised selection，不能写成 SAE 无监督机制天然优于 raw hidden。",
        "",
        "## 6. 证据强度与不能声称的内容",
        "",
        _markdown_table(pd.DataFrame(claim_strength), ["claim", "support", "reason"], ["主张", "证据强度", "理由"]),
        "",
        "特别需要避免的表述：",
        "",
        "- 不要说“这个 SAE feature 就是 QUO/RES/GI”。应说“该 latent 的高激活语句似乎与某类表面形式或咨询功能相关”。",
        "- 不要说“LLM 已经按 MISC 指南理解咨询行为”。当前只能说标签信息可解码，并存在可审阅的候选关联。",
        "- 不要把 top-n probe 的 predictive sufficiency 写成 causal sufficiency。",
        "- 不要对 RES/REC 做强 context-relation 解释，因为第一阶段没有前一句 client context。",
        "",
        "## 7. 后续最值得做的验证",
        "",
        "1. 运行非 dry-run 的第二阶段结构化审阅，或进行人工盲审，给每个 latent 形成可审计的 candidate functional interpretation。",
        "2. 给 evidence packet 加入前一句 client utterance，重点重审 RE/RES/REC。",
        "3. 对高质量标签先做 token-level attribution，检查 latent 是否由问号、what/how、good/great、固定模板等表面线索触发。",
        "4. 做 latent/group ablation 或 steering，并加入 random、bottom-k、orthogonal controls，区分预测相关与因果作用。",
        "5. 对 top-n probe 做 C-sweep、不同随机种子、不同 split 策略复核，确认 top185 超过 raw hidden 不是正则或切分偶然性。",
        "",
        "## 8. 产物位置",
        "",
        "- `phase3_integrated_research_report_zh.md`：本报告。",
        "- `phase3_label_integrated_summary.csv`：逐标签综合表。",
        "- `phase3_label_topn_convergence.csv`：逐标签 top-n 收敛与接近 full SAE 所需 n。",
        "- `phase3_evidence_label_summary.csv`：Top20 latent evidence packet 的标签级质量摘要。",
        "- `phase3_claim_strength.json`：主张-证据强度清单。",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase3_report(
    *,
    full_probe_summary_path: str | Path,
    full_probe_by_label_summary_path: str | Path,
    ranked_convergence_path: str | Path,
    evidence_summary_path: str | Path,
    function_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    full_probe_summary_path = Path(full_probe_summary_path)
    full_probe_by_label_summary_path = Path(full_probe_by_label_summary_path)
    ranked_convergence_path = Path(ranked_convergence_path)
    evidence_summary_path = Path(evidence_summary_path)
    function_manifest_path = Path(function_manifest_path)

    summary = _read_csv(full_probe_summary_path)
    by_label = _read_csv(full_probe_by_label_summary_path)
    convergence = _read_csv(ranked_convergence_path)
    evidence_summary = _read_csv(evidence_summary_path)

    baseline, baseline_auc = build_baseline_table(summary)
    overall_topn = build_overall_topn_table(convergence, baseline_auc)
    label_topn = build_label_topn_table(by_label)
    evidence_label = build_evidence_label_table(evidence_summary)
    integrated = build_label_integrated_table(by_label, label_topn, evidence_label)
    phase2 = _phase2_status(function_manifest_path.parent)
    claims = _claim_strength(evidence_label, phase2)

    files = {
        "label_integrated_summary": output_path / "phase3_label_integrated_summary.csv",
        "label_topn_convergence": output_path / "phase3_label_topn_convergence.csv",
        "overall_topn_summary": output_path / "phase3_overall_topn_summary.csv",
        "evidence_label_summary": output_path / "phase3_evidence_label_summary.csv",
        "claim_strength": output_path / "phase3_claim_strength.json",
        "report": output_path / "phase3_integrated_research_report_zh.md",
        "manifest": output_path / "manifest.json",
    }
    integrated.to_csv(files["label_integrated_summary"], index=False)
    label_topn.to_csv(files["label_topn_convergence"], index=False)
    overall_topn.to_csv(files["overall_topn_summary"], index=False)
    evidence_label.to_csv(files["evidence_label_summary"], index=False)
    _write_json(files["claim_strength"], claims)

    inputs = {
        "full_probe_summary": str(full_probe_summary_path),
        "full_probe_by_label_summary": str(full_probe_by_label_summary_path),
        "ranked_convergence": str(ranked_convergence_path),
        "evidence_summary": str(evidence_summary_path),
        "function_manifest": str(function_manifest_path),
    }
    write_report(
        output_path=files["report"],
        baseline=baseline,
        overall_topn=overall_topn,
        label_topn=label_topn,
        evidence_label=evidence_label,
        integrated=integrated,
        phase2=phase2,
        claim_strength=claims,
        inputs=inputs,
    )

    status_counts = dict(Counter(evidence_label["evidence_grade"]))
    manifest = {
        "analysis": "misc_sae_phase3_integrated_research_report",
        "inputs": inputs,
        "outputs": {key: str(value) for key, value in files.items()},
        "n_labels": int(len(integrated)),
        "baseline_macro_auc": baseline_auc,
        "best_overall_topn": overall_topn.sort_values("best_macro_auc", ascending=False).iloc[0].to_dict()
        if not overall_topn.empty
        else {},
        "evidence_grade_counts": status_counts,
        "phase2_review_status": {key: value for key, value in phase2.items() if key != "manifest"},
    }
    _write_json(files["manifest"], manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the phase-3 integrated MISC SAE research report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--probe-dir", default=str(DEFAULT_PROBE_DIR))
    parser.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR))
    parser.add_argument("--function-dir", default=str(DEFAULT_FUNCTION_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--full-probe-summary", default=None)
    parser.add_argument("--full-probe-by-label-summary", default=None)
    parser.add_argument("--ranked-convergence", default=None)
    parser.add_argument("--evidence-summary", default=None)
    parser.add_argument("--function-manifest", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    probe_dir = Path(args.probe_dir)
    evidence_dir = Path(args.evidence_dir)
    function_dir = Path(args.function_dir)
    manifest = run_phase3_report(
        full_probe_summary_path=args.full_probe_summary or probe_dir / "full_probe_summary.csv",
        full_probe_by_label_summary_path=args.full_probe_by_label_summary or probe_dir / "full_probe_by_label_summary.csv",
        ranked_convergence_path=args.ranked_convergence or probe_dir / "ranked_sae_subspace_convergence.csv",
        evidence_summary_path=args.evidence_summary or evidence_dir / "latent_evidence_packet_summary.csv",
        function_manifest_path=args.function_manifest or function_dir / "manifest.json",
        output_dir=args.output_dir,
    )
    print("Completed phase-3 integrated research report.")
    print(f"Output dir: {args.output_dir}")
    print(f"Labels: {manifest['n_labels']}")
    print(f"Best top-n: {manifest['best_overall_topn']}")
    print(f"Phase-2 status: {manifest['phase2_review_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
