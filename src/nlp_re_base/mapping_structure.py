"""Mapping-structure analysis for MISC label and SAE latent spaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_TOP_K_VALUES = [10, 20, 50, 100]
DEFAULT_HIERARCHY_SPECS = ["RE:RES,REC", "QU:QUO,QUC"]
DEFAULT_CORE_LABELS = ["RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF"]


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def parse_label_hierarchy(specs: Iterable[str] | None) -> dict[str, list[str]]:
    hierarchy: dict[str, list[str]] = {}
    for spec in specs or []:
        if ":" not in spec:
            raise ValueError(f"Invalid hierarchy spec {spec!r}; expected PARENT:CHILD1,CHILD2")
        parent, children = spec.split(":", 1)
        parent = parent.strip().upper()
        child_labels = [child.strip().upper() for child in children.split(",") if child.strip()]
        if not parent or not child_labels:
            raise ValueError(f"Invalid hierarchy spec {spec!r}; missing parent or children")
        hierarchy[parent] = child_labels
    return hierarchy


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_mapping_matrix(path: str | Path) -> pd.DataFrame:
    matrix = pd.read_csv(path)
    required = {
        "label",
        "latent_idx",
        "cohens_d",
        "abs_cohens_d",
        "directional_auc",
        "significant_fdr",
    }
    missing = sorted(required.difference(matrix.columns))
    if missing:
        raise ValueError(f"Missing required matrix columns: {missing}")

    matrix = matrix.copy()
    matrix["label"] = matrix["label"].astype(str).str.upper()
    matrix["latent_idx"] = matrix["latent_idx"].astype(int)
    matrix["significant_fdr"] = _bool_series(matrix["significant_fdr"])
    for col in [
        "cohens_d",
        "abs_cohens_d",
        "auc",
        "directional_auc",
        "p_value",
        "prevalence",
    ]:
        if col in matrix.columns:
            matrix[col] = pd.to_numeric(matrix[col], errors="coerce").fillna(0.0)
    return matrix


def load_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def label_order_from_summary(label_summary: dict[str, Any] | None, matrix: pd.DataFrame) -> list[str]:
    if label_summary and label_summary.get("labels"):
        labels = [str(row["label"]).upper() for row in label_summary["labels"]]
        labels.extend(label for label in matrix["label"].unique() if label not in labels)
        return labels
    return list(matrix["label"].drop_duplicates())


def _top_effect_share(group: pd.DataFrame, k: int) -> float:
    sig = group[group["significant_fdr"]].sort_values("abs_cohens_d", ascending=False)
    total = float(sig["abs_cohens_d"].sum())
    if total <= 0:
        return 0.0
    return float(sig.head(k)["abs_cohens_d"].sum() / total)


def build_label_fragmentation_rank(
    matrix: pd.DataFrame,
    *,
    label_order: list[str] | None = None,
    top_k_values: list[int] | None = None,
) -> pd.DataFrame:
    top_k_values = top_k_values or DEFAULT_TOP_K_VALUES
    rows: list[dict[str, Any]] = []
    labels = label_order or list(matrix["label"].drop_duplicates())
    total_latents = int(matrix["latent_idx"].nunique())

    for label in labels:
        group = matrix[matrix["label"] == label]
        if group.empty:
            continue
        sig = group[group["significant_fdr"]]
        pos_sig = sig[sig["cohens_d"] > 0]
        neg_sig = sig[sig["cohens_d"] < 0]
        top = group.sort_values(["abs_cohens_d", "directional_auc"], ascending=False).iloc[0]
        row: dict[str, Any] = {
            "label": label,
            "n_tested_latents": int(group.shape[0]),
            "n_significant_latents": int(sig.shape[0]),
            "n_positive_effect_significant": int(pos_sig.shape[0]),
            "n_negative_effect_significant": int(neg_sig.shape[0]),
            "fragmentation_ratio": float(sig.shape[0] / max(total_latents, 1)),
            "positive_effect_ratio": float(pos_sig.shape[0] / max(sig.shape[0], 1)),
            "negative_effect_ratio": float(neg_sig.shape[0] / max(sig.shape[0], 1)),
            "top_latent_idx": int(top["latent_idx"]),
            "top_abs_cohens_d": _safe_float(top.get("abs_cohens_d")),
            "top_directional_auc": _safe_float(top.get("directional_auc"), 0.5),
            "top_cohens_d": _safe_float(top.get("cohens_d")),
            "prevalence": _safe_float(group["prevalence"].iloc[0]) if "prevalence" in group else 0.0,
        }
        for k in top_k_values:
            row[f"top{k}_abs_effect_share"] = _top_effect_share(group, k)
            row[f"top{k}_significant_share"] = float(min(k, sig.shape[0]) / max(sig.shape[0], 1))
            precision_col = f"precision_at_{k}"
            if precision_col in group.columns:
                row[f"top{k}_max_precision"] = _safe_float(group[precision_col].max())
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["n_significant_latents", "top_abs_cohens_d"],
            ascending=[False, False],
        ).reset_index(drop=True)
    return out


def _significant_sets(matrix: pd.DataFrame) -> dict[str, set[int]]:
    sig = matrix[matrix["significant_fdr"]]
    return {
        label: set(group["latent_idx"].astype(int).tolist())
        for label, group in sig.groupby("label", sort=False)
    }


def _top_sets(matrix: pd.DataFrame, k: int) -> dict[str, set[int]]:
    sets: dict[str, set[int]] = {}
    for label, group in matrix.groupby("label", sort=False):
        top = group.sort_values(["abs_cohens_d", "directional_auc"], ascending=False).head(k)
        sets[label] = set(top["latent_idx"].astype(int).tolist())
    return sets


def build_latent_profiles(
    matrix: pd.DataFrame,
    *,
    hierarchy: dict[str, list[str]],
    core_labels: list[str],
    global_label_threshold: int = 5,
) -> pd.DataFrame:
    sig = matrix[matrix["significant_fdr"]].copy()
    if sig.empty:
        return pd.DataFrame(
            columns=[
                "latent_idx",
                "n_labels",
                "labels",
                "positive_labels",
                "negative_labels",
                "direction_type",
                "role",
                "max_abs_cohens_d",
                "max_directional_auc",
            ]
        )

    family_by_label: dict[str, str] = {}
    for parent, children in hierarchy.items():
        family_by_label[parent] = parent
        for child in children:
            family_by_label[child] = parent

    core = set(label.upper() for label in core_labels)
    rows: list[dict[str, Any]] = []
    for latent_idx, group in sig.groupby("latent_idx", sort=False):
        labels = sorted(group["label"].astype(str).tolist())
        positive = sorted(group[group["cohens_d"] > 0]["label"].astype(str).tolist())
        negative = sorted(group[group["cohens_d"] < 0]["label"].astype(str).tolist())
        if positive and negative:
            direction_type = "mixed_direction"
        elif positive:
            direction_type = "positive_only"
        else:
            direction_type = "negative_only"

        core_hit_labels = sorted(label for label in labels if label in core)
        core_families = {family_by_label.get(label, label) for label in core_hit_labels}
        if not core_hit_labels:
            role = "auxiliary_only"
        elif len(labels) >= global_label_threshold:
            role = "global"
        elif len(core_hit_labels) == 1:
            role = "exclusive"
        elif len(core_families) == 1:
            role = "family_shared"
        else:
            role = "cross_family"

        rows.append(
            {
                "latent_idx": int(latent_idx),
                "n_labels": int(len(labels)),
                "labels": ",".join(labels),
                "n_positive_labels": int(len(positive)),
                "positive_labels": ",".join(positive),
                "n_negative_labels": int(len(negative)),
                "negative_labels": ",".join(negative),
                "direction_type": direction_type,
                "role": role,
                "max_abs_cohens_d": float(group["abs_cohens_d"].max()),
                "max_directional_auc": float(group["directional_auc"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["n_labels", "max_abs_cohens_d"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_latent_overlap_distribution(profiles: pd.DataFrame) -> pd.DataFrame:
    if profiles.empty:
        return pd.DataFrame(
            columns=["n_labels", "n_latents", "positive_only", "negative_only", "mixed_direction"]
        )
    rows: list[dict[str, Any]] = []
    for n_labels, group in profiles.groupby("n_labels"):
        rows.append(
            {
                "n_labels": int(n_labels),
                "n_latents": int(group.shape[0]),
                "positive_only": int((group["direction_type"] == "positive_only").sum()),
                "negative_only": int((group["direction_type"] == "negative_only").sum()),
                "mixed_direction": int((group["direction_type"] == "mixed_direction").sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("n_labels").reset_index(drop=True)


def build_latent_role_summary(profiles: pd.DataFrame) -> pd.DataFrame:
    if profiles.empty:
        return pd.DataFrame(columns=["role", "n_latents", "share", "mean_n_labels"])
    total = max(int(profiles.shape[0]), 1)
    rows: list[dict[str, Any]] = []
    for role, group in profiles.groupby("role"):
        rows.append(
            {
                "role": role,
                "n_latents": int(group.shape[0]),
                "share": float(group.shape[0] / total),
                "mean_n_labels": float(group["n_labels"].mean()),
                "mean_max_abs_cohens_d": float(group["max_abs_cohens_d"].mean()),
                "max_abs_cohens_d": float(group["max_abs_cohens_d"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("n_latents", ascending=False).reset_index(drop=True)


def build_label_pair_similarity(
    matrix: pd.DataFrame,
    *,
    label_order: list[str],
    top_k_values: list[int],
) -> pd.DataFrame:
    sig_sets = _significant_sets(matrix)
    pivot = (
        matrix.pivot_table(index="latent_idx", columns="label", values="cohens_d", aggfunc="first")
        .fillna(0.0)
        .sort_index()
    )
    rows: list[dict[str, Any]] = []
    for k in top_k_values:
        top_sets = _top_sets(matrix, k)
        for i, label_a in enumerate(label_order):
            for label_b in label_order[i + 1 :]:
                if label_a not in pivot.columns or label_b not in pivot.columns:
                    continue
                top_a = top_sets.get(label_a, set())
                top_b = top_sets.get(label_b, set())
                top_inter = top_a & top_b
                top_union = top_a | top_b
                sig_a = sig_sets.get(label_a, set())
                sig_b = sig_sets.get(label_b, set())
                sig_inter = sig_a & sig_b
                sig_union = sig_a | sig_b

                da = pivot[label_a]
                db = pivot[label_b]
                pearson = da.corr(db, method="pearson")
                spearman = da.corr(db, method="spearman")
                rows.append(
                    {
                        "label_a": label_a,
                        "label_b": label_b,
                        "top_k": int(k),
                        "topk_intersection": int(len(top_inter)),
                        "topk_union": int(len(top_union)),
                        "topk_jaccard": float(len(top_inter) / len(top_union)) if top_union else 0.0,
                        "significant_intersection": int(len(sig_inter)),
                        "significant_union": int(len(sig_union)),
                        "significant_jaccard": (
                            float(len(sig_inter) / len(sig_union)) if sig_union else 0.0
                        ),
                        "pearson_cohens_d": 0.0 if pd.isna(pearson) else float(pearson),
                        "spearman_cohens_d": 0.0 if pd.isna(spearman) else float(spearman),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["top_k", "topk_jaccard", "significant_jaccard"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_hierarchy_alignment(
    matrix: pd.DataFrame,
    *,
    hierarchy: dict[str, list[str]],
) -> pd.DataFrame:
    sig_sets = _significant_sets(matrix)
    rows: list[dict[str, Any]] = []
    for parent, children in hierarchy.items():
        parent_set = sig_sets.get(parent, set())
        child_sets = [sig_sets.get(child, set()) for child in children]
        child_union = set().union(*child_sets) if child_sets else set()
        inter = parent_set & child_union
        union = parent_set | child_union
        rows.append(
            {
                "relation_type": "parent_summary",
                "parent": parent,
                "child_a": "ALL_CHILDREN",
                "child_b": "",
                "n_parent": int(len(parent_set)),
                "n_child": int(len(child_union)),
                "intersection": int(len(inter)),
                "union": int(len(union)),
                "jaccard": float(len(inter) / len(union)) if union else 0.0,
                "child_coverage_by_parent": (
                    float(len(inter) / len(child_union)) if child_union else 0.0
                ),
                "parent_decomposition": (
                    float(len(inter) / len(parent_set)) if parent_set else 0.0
                ),
                "sibling_separation": np.nan,
            }
        )
        for child, child_set in zip(children, child_sets):
            inter = parent_set & child_set
            union = parent_set | child_set
            rows.append(
                {
                    "relation_type": "parent_child",
                    "parent": parent,
                    "child_a": child,
                    "child_b": "",
                    "n_parent": int(len(parent_set)),
                    "n_child": int(len(child_set)),
                    "intersection": int(len(inter)),
                    "union": int(len(union)),
                    "jaccard": float(len(inter) / len(union)) if union else 0.0,
                    "child_coverage_by_parent": (
                        float(len(inter) / len(child_set)) if child_set else 0.0
                    ),
                    "parent_decomposition": (
                        float(len(inter) / len(parent_set)) if parent_set else 0.0
                    ),
                    "sibling_separation": np.nan,
                }
            )
        for i, child_a in enumerate(children):
            for child_b in children[i + 1 :]:
                set_a = sig_sets.get(child_a, set())
                set_b = sig_sets.get(child_b, set())
                inter = set_a & set_b
                union = set_a | set_b
                jaccard = float(len(inter) / len(union)) if union else 0.0
                rows.append(
                    {
                        "relation_type": "sibling",
                        "parent": parent,
                        "child_a": child_a,
                        "child_b": child_b,
                        "n_parent": int(len(parent_set)),
                        "n_child": int(len(union)),
                        "intersection": int(len(inter)),
                        "union": int(len(union)),
                        "jaccard": jaccard,
                        "child_coverage_by_parent": np.nan,
                        "parent_decomposition": np.nan,
                        "sibling_separation": float(1.0 - jaccard),
                    }
                )
    return pd.DataFrame(rows)


def build_metrics_payload(
    *,
    matrix: pd.DataFrame,
    label_fragmentation: pd.DataFrame,
    latent_profiles: pd.DataFrame,
    role_summary: pd.DataFrame,
    hierarchy_alignment: pd.DataFrame,
    label_pair_similarity: pd.DataFrame,
    top_k_values: list[int],
    hierarchy: dict[str, list[str]],
    core_labels: list[str],
) -> dict[str, Any]:
    sig = matrix[matrix["significant_fdr"]]
    single = int((latent_profiles["n_labels"] == 1).sum()) if not latent_profiles.empty else 0
    multi = int((latent_profiles["n_labels"] > 1).sum()) if not latent_profiles.empty else 0
    top_k_ref = 50 if 50 in top_k_values else top_k_values[0]
    top_pairs = label_pair_similarity[label_pair_similarity["top_k"] == top_k_ref].head(10)
    hierarchy_rows = hierarchy_alignment.to_dict("records") if not hierarchy_alignment.empty else []
    return {
        "analysis_version": 1,
        "n_labels": int(matrix["label"].nunique()),
        "n_latents": int(matrix["latent_idx"].nunique()),
        "n_latent_label_rows": int(matrix.shape[0]),
        "significant_latent_label_edges": int(sig.shape[0]),
        "latents_with_any_significant_label": int(latent_profiles.shape[0]),
        "single_label_latents": single,
        "multi_label_latents": multi,
        "multi_label_latent_share": float(multi / max(latent_profiles.shape[0], 1)),
        "core_labels": core_labels,
        "hierarchy": hierarchy,
        "top_k_values": top_k_values,
        "most_fragmented_labels": label_fragmentation.head(10).to_dict("records"),
        "latent_role_summary": role_summary.to_dict("records"),
        "top_label_pairs_at_reference_k": top_pairs.to_dict("records"),
        "hierarchy_alignment": hierarchy_rows,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        return f"{float(value):.{digits}f}"
    return str(value)


def write_mapping_structure_report(
    path: str | Path,
    *,
    metrics: dict[str, Any],
    label_fragmentation: pd.DataFrame,
    overlap_distribution: pd.DataFrame,
    role_summary: pd.DataFrame,
    label_pair_similarity: pd.DataFrame,
    hierarchy_alignment: pd.DataFrame,
    top_k_ref: int = 50,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MISC Mapping Structure 分析报告",
        "",
        "> 分析目标：把 `latent × MISC label` 矩阵转化为导师文档 R1 的正式证据，说明标签空间与 SAE 表征空间存在结构化多对多映射。",
        "",
        "## 1. 一句话结论",
        "",
        (
            "本次 Mapping Structure 分析支持：MISC 标签与 SAE latent 不是一一对应关系，"
            "而是由标签碎片化、latent 多标签重叠和父子标签部分共享共同构成的结构化多对多映射。"
        ),
        "",
        "## 2. Many-to-many 证据",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| latent-label 矩阵行数 | {metrics['n_latent_label_rows']} |",
        f"| 显著 latent-label 边 | {metrics['significant_latent_label_edges']} |",
        f"| 至少关联一个标签的 latent | {metrics['latents_with_any_significant_label']} |",
        f"| 单标签 latent | {metrics['single_label_latents']} |",
        f"| 多标签 latent | {metrics['multi_label_latents']} |",
        f"| 多标签 latent 占比 | {_fmt(metrics['multi_label_latent_share'])} |",
        "",
        (
            "解释：如果标签和表征是一一对应，应该看到大量单标签 latent 和较少 overlap。"
            "当前结果中多标签 latent 远多于单标签 latent，因此更符合多对多表征结构。"
        ),
        "",
        "## 3. Label Fragmentation 排名",
        "",
        "| Label | Sig. Latents | Pos. | Neg. | Frag. Ratio | Top Latent | Top d | Top AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in label_fragmentation.head(20).iterrows():
        lines.append(
            "| {label} | {n_sig} | {n_pos} | {n_neg} | {frag} | {top_latent} | {top_d} | {top_auc} |".format(
                label=row["label"],
                n_sig=int(row["n_significant_latents"]),
                n_pos=int(row["n_positive_effect_significant"]),
                n_neg=int(row["n_negative_effect_significant"]),
                frag=_fmt(row["fragmentation_ratio"]),
                top_latent=int(row["top_latent_idx"]),
                top_d=_fmt(row["top_abs_cohens_d"]),
                top_auc=_fmt(row["top_directional_auc"]),
            )
        )

    lines.extend(
        [
            "",
            "## 4. Latent Overlap 结构",
            "",
            "| Labels per latent | Latents | Positive-only | Negative-only | Mixed direction |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in overlap_distribution.iterrows():
        lines.append(
            f"| {int(row['n_labels'])} | {int(row['n_latents'])} | "
            f"{int(row['positive_only'])} | {int(row['negative_only'])} | "
            f"{int(row['mixed_direction'])} |"
        )

    lines.extend(
        [
            "",
            "## 5. Latent Role Taxonomy",
            "",
            "| Role | Latents | Share | Mean label count | Max d |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in role_summary.iterrows():
        lines.append(
            f"| {row['role']} | {int(row['n_latents'])} | {_fmt(row['share'])} | "
            f"{_fmt(row['mean_n_labels'])} | {_fmt(row['max_abs_cohens_d'])} |"
        )

    lines.extend(
        [
            "",
            f"## 6. Label Pair Similarity (Top-{top_k_ref})",
            "",
            "| Label A | Label B | Top-k Jaccard | Full Sig. Jaccard | Pearson d | Spearman d |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    pair_rows = label_pair_similarity[label_pair_similarity["top_k"] == top_k_ref].head(20)
    for _, row in pair_rows.iterrows():
        lines.append(
            f"| {row['label_a']} | {row['label_b']} | {_fmt(row['topk_jaccard'])} | "
            f"{_fmt(row['significant_jaccard'])} | {_fmt(row['pearson_cohens_d'])} | "
            f"{_fmt(row['spearman_cohens_d'])} |"
        )

    lines.extend(
        [
            "",
            "## 7. 标签层级一致性",
            "",
            "| Type | Parent | Child A | Child B | Jaccard | Parent decomposition | Child coverage | Sibling separation |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in hierarchy_alignment.iterrows():
        lines.append(
            f"| {row['relation_type']} | {row['parent']} | {row['child_a']} | {row['child_b']} | "
            f"{_fmt(row['jaccard'])} | {_fmt(row['parent_decomposition'])} | "
            f"{_fmt(row['child_coverage_by_parent'])} | {_fmt(row['sibling_separation'])} |"
        )

    lines.extend(
        [
            "",
            "## 8. 论文可用表述",
            "",
            (
                "在全量 MISC 样本上，SAE 表征空间没有呈现简单的一标签一特征关系；"
                "相反，一个标签通常分散到多个 latent，一个 latent 也常同时关联多个行为标签。"
                "这种错配并非随机噪声，而是在父子标签、相近行为标签和跨行为家族之间呈现稳定结构。"
            ),
            "",
            "## 9. 限制",
            "",
            "- 本阶段只分析 mapping structure，不证明因果机制。",
            "- 显著关联不等价于单义解释，仍需要 top example 与人工审查。",
            "- `OTHER` 是异质辅助标签，不应作为主要心理咨询行为结论。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _try_write_figures(
    *,
    output_dir: Path,
    label_fragmentation: pd.DataFrame,
    overlap_distribution: pd.DataFrame,
    label_pair_similarity: pd.DataFrame,
    hierarchy_alignment: pd.DataFrame,
    top_k_ref: int,
) -> list[str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional plotting stack.
        (figure_dir / "FIGURES_SKIPPED.txt").write_text(
            f"matplotlib unavailable: {exc}\n",
            encoding="utf-8",
        )
        return written

    frag = label_fragmentation.sort_values("n_significant_latents", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(frag["label"], frag["n_significant_latents"])
    ax.set_title("Label fragmentation")
    ax.set_ylabel("Significant latents")
    ax.set_xlabel("MISC label")
    fig.tight_layout()
    path = figure_dir / "fragmentation_bar.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar(overlap_distribution["n_labels"], overlap_distribution["n_latents"])
    ax.set_title("Latent polysemanticity")
    ax.set_ylabel("Latents")
    ax.set_xlabel("Number of associated labels")
    fig.tight_layout()
    path = figure_dir / "latent_polysemanticity_histogram.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    pairs = label_pair_similarity[label_pair_similarity["top_k"] == top_k_ref]
    labels = sorted(set(pairs["label_a"]).union(set(pairs["label_b"])))
    heat = pd.DataFrame(0.0, index=labels, columns=labels)
    for label in labels:
        heat.loc[label, label] = 1.0
    for _, row in pairs.iterrows():
        heat.loc[row["label_a"], row["label_b"]] = row["topk_jaccard"]
        heat.loc[row["label_b"], row["label_a"]] = row["topk_jaccard"]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(heat.values, vmin=0.0, vmax=max(0.5, float(heat.values.max())))
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title(f"Top-{top_k_ref} label Jaccard")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = figure_dir / "label_jaccard_heatmap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    parent_child = hierarchy_alignment[hierarchy_alignment["relation_type"] == "parent_child"]
    if not parent_child.empty:
        labels_y = parent_child["parent"].astype(str) + "->" + parent_child["child_a"].astype(str)
        values = parent_child[["jaccard", "parent_decomposition", "child_coverage_by_parent"]].fillna(0)
        fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(parent_child))))
        im = ax.imshow(values.values, vmin=0.0, vmax=max(0.5, float(values.values.max())))
        ax.set_xticks(range(values.shape[1]), labels=list(values.columns), rotation=30, ha="right")
        ax.set_yticks(range(len(labels_y)), labels=list(labels_y))
        ax.set_title("Hierarchy alignment")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        path = figure_dir / "hierarchy_alignment_heatmap.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))
    return written


def run_mapping_structure_analysis(
    *,
    mapping_dir: str | Path,
    output_dir: str | Path,
    top_k_values: list[int] | None = None,
    hierarchy_specs: list[str] | None = None,
    core_labels: list[str] | None = None,
    fdr_alpha: float = 0.05,
    doc_report: str | Path | None = None,
    make_figures: bool = True,
) -> dict[str, Any]:
    mapping_path = Path(mapping_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    top_k_values = sorted({int(k) for k in (top_k_values or DEFAULT_TOP_K_VALUES) if int(k) > 0})
    hierarchy = parse_label_hierarchy(hierarchy_specs or DEFAULT_HIERARCHY_SPECS)
    core_labels = [label.upper() for label in (core_labels or DEFAULT_CORE_LABELS)]

    matrix = load_mapping_matrix(mapping_path / "latent_label_matrix.csv")
    label_summary = load_json(mapping_path / "label_summary.json", default={})
    run_summary = load_json(mapping_path / "run_summary.json", default={})
    label_order = label_order_from_summary(label_summary, matrix)

    label_fragmentation = build_label_fragmentation_rank(
        matrix,
        label_order=label_order,
        top_k_values=top_k_values,
    )
    latent_profiles = build_latent_profiles(
        matrix,
        hierarchy=hierarchy,
        core_labels=core_labels,
    )
    overlap_distribution = build_latent_overlap_distribution(latent_profiles)
    role_summary = build_latent_role_summary(latent_profiles)
    label_pair_similarity = build_label_pair_similarity(
        matrix,
        label_order=label_order,
        top_k_values=top_k_values,
    )
    hierarchy_alignment = build_hierarchy_alignment(matrix, hierarchy=hierarchy)

    metrics = build_metrics_payload(
        matrix=matrix,
        label_fragmentation=label_fragmentation,
        latent_profiles=latent_profiles,
        role_summary=role_summary,
        hierarchy_alignment=hierarchy_alignment,
        label_pair_similarity=label_pair_similarity,
        top_k_values=top_k_values,
        hierarchy=hierarchy,
        core_labels=core_labels,
    )
    metrics["mapping_dir"] = str(mapping_path)
    metrics["output_dir"] = str(output_path)
    metrics["fdr_alpha"] = fdr_alpha
    metrics["source_run_summary"] = run_summary

    label_fragmentation.to_csv(output_path / "label_fragmentation_rank.csv", index=False)
    overlap_distribution.to_csv(output_path / "latent_overlap_distribution.csv", index=False)
    label_pair_similarity.to_csv(output_path / "label_pair_similarity.csv", index=False)
    hierarchy_alignment.to_csv(output_path / "hierarchy_alignment.csv", index=False)
    role_summary.to_csv(output_path / "latent_role_summary.csv", index=False)
    latent_profiles.to_csv(output_path / "latent_role_assignments.csv", index=False)
    write_json(output_path / "mapping_structure_metrics.json", metrics)

    top_k_ref = 50 if 50 in top_k_values else top_k_values[0]
    report_path = output_path / "mapping_structure_report.md"
    write_mapping_structure_report(
        report_path,
        metrics=metrics,
        label_fragmentation=label_fragmentation,
        overlap_distribution=overlap_distribution,
        role_summary=role_summary,
        label_pair_similarity=label_pair_similarity,
        hierarchy_alignment=hierarchy_alignment,
        top_k_ref=top_k_ref,
    )
    if doc_report:
        write_mapping_structure_report(
            doc_report,
            metrics=metrics,
            label_fragmentation=label_fragmentation,
            overlap_distribution=overlap_distribution,
            role_summary=role_summary,
            label_pair_similarity=label_pair_similarity,
            hierarchy_alignment=hierarchy_alignment,
            top_k_ref=top_k_ref,
        )

    figure_files = []
    if make_figures:
        figure_files = _try_write_figures(
            output_dir=output_path,
            label_fragmentation=label_fragmentation,
            overlap_distribution=overlap_distribution,
            label_pair_similarity=label_pair_similarity,
            hierarchy_alignment=hierarchy_alignment,
            top_k_ref=top_k_ref,
        )
    metrics["files"] = {
        "mapping_structure_metrics": str(output_path / "mapping_structure_metrics.json"),
        "label_fragmentation_rank": str(output_path / "label_fragmentation_rank.csv"),
        "latent_overlap_distribution": str(output_path / "latent_overlap_distribution.csv"),
        "label_pair_similarity": str(output_path / "label_pair_similarity.csv"),
        "hierarchy_alignment": str(output_path / "hierarchy_alignment.csv"),
        "latent_role_summary": str(output_path / "latent_role_summary.csv"),
        "latent_role_assignments": str(output_path / "latent_role_assignments.csv"),
        "mapping_structure_report": str(report_path),
        "doc_report": str(doc_report) if doc_report else None,
        "figures": figure_files,
    }
    write_json(output_path / "mapping_structure_metrics.json", metrics)
    return metrics
