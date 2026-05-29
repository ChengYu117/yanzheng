"""Hierarchy-aware MISC structural relation analysis.

This script consumes the already-computed MISC latent-label association outputs
and produces a v2 structural analysis that treats RE/QU parent labels as
consistency checks rather than independent overlap evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PARENT_CHILDREN = {"RE": ["RES", "REC"], "QU": ["QUO", "QUC"]}
PARENT_LABELS = ["RE", "QU"]
LEAF_LABELS = ["RES", "REC", "QUO", "QUC", "GI", "SU", "AF"]
FAMILY_MEMBERS = {
    "RE_family": ["RES", "REC"],
    "QU_family": ["QUO", "QUC"],
    "GI": ["GI"],
    "SU": ["SU"],
    "AF": ["AF"],
}
SUPPLEMENTAL_BLOCKS = {
    "advice_support_info_block": ["GI", "SU", "AF"],
}
ALL_LABELS = PARENT_LABELS + LEAF_LABELS


def _family_for_leaf(label: str) -> str:
    for family, members in FAMILY_MEMBERS.items():
        if label in members:
            return family
    return label


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _classify_overlap(value: float) -> str:
    if value >= 0.25:
        return "high"
    if value >= 0.10:
        return "moderate"
    return "low"


def _effective_n(abs_values: Iterable[float]) -> float:
    values = np.asarray([float(v) for v in abs_values if float(v) > 0], dtype=float)
    if values.size == 0:
        return 0.0
    weights = values / values.sum()
    entropy = -float(np.sum(weights * np.log(weights)))
    return float(math.exp(entropy))


def _classify_fragmentation(n_support: int, effective_n: float) -> str:
    if n_support >= 13 or effective_n > 8:
        return "distributed"
    if n_support <= 5 or effective_n <= 4:
        return "compact"
    return "moderate"


def _classify_role(labels: list[str], families: set[str]) -> str:
    n_labels = len(labels)
    n_families = len(families)
    if n_labels == 1:
        return "label_specific"
    if n_families == 1:
        return "sibling_shared"
    if n_labels >= 4 or n_families >= 3:
        return "generalized"
    return "cross_family"


def _read_topk(path: Path, top_k: int) -> pd.DataFrame:
    matrix = pd.read_csv(path)
    matrix["label"] = matrix["label"].astype(str).str.upper()
    numeric_cols = [
        "topk_rank",
        "latent_idx",
        "cohens_d",
        "abs_cohens_d",
        "directional_auc",
        "precision_lift_at_50",
        "precision_at_50",
    ]
    for col in numeric_cols:
        if col in matrix.columns:
            matrix[col] = pd.to_numeric(matrix[col], errors="coerce").fillna(0.0)
    if "significant_fdr" in matrix.columns:
        matrix["significant_fdr"] = _bool_series(matrix["significant_fdr"])
    else:
        matrix["significant_fdr"] = False
    matrix = matrix[matrix["label"].isin(ALL_LABELS)].copy()
    matrix = matrix[matrix["topk_rank"] <= top_k].copy()
    matrix["latent_idx"] = matrix["latent_idx"].astype(int)
    matrix["support_edge"] = (
        matrix["significant_fdr"]
        & (matrix["abs_cohens_d"] >= 0.5)
        & (matrix["directional_auc"] >= 0.60)
        & (
            (matrix["cohens_d"] < 0)
            | (matrix["precision_lift_at_50"] >= 0.10)
        )
    )
    matrix["edge_type"] = np.where(
        matrix["support_edge"] & (matrix["cohens_d"] < 0),
        "negative_boundary",
        np.where(matrix["support_edge"], "positive_support", "weak_top20"),
    )
    return matrix


def _sets_by_label(matrix: pd.DataFrame, *, support_only: bool) -> dict[str, set[int]]:
    rows = matrix[matrix["support_edge"]] if support_only else matrix
    return {
        label: set(group["latent_idx"].astype(int).tolist())
        for label, group in rows.groupby("label", sort=False)
    }


def _union_for_members(label_sets: dict[str, set[int]], members: list[str]) -> set[int]:
    out: set[int] = set()
    for member in members:
        out |= label_sets.get(member, set())
    return out


def build_fragmentation(matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label in LEAF_LABELS:
        group = matrix[matrix["label"] == label].copy()
        support = group[group["support_edge"]].copy()
        support_abs = support["abs_cohens_d"].astype(float).tolist()
        top20_abs_sum = float(group["abs_cohens_d"].sum())
        support_abs_sum = float(support["abs_cohens_d"].sum())
        effective = _effective_n(support_abs)
        rows.append(
            {
                "label": label,
                "family": _family_for_leaf(label),
                "topk_n": int(group.shape[0]),
                "n_support_latents": int(support.shape[0]),
                "n_positive_support": int((support["cohens_d"] > 0).sum()),
                "n_negative_boundary_support": int((support["cohens_d"] < 0).sum()),
                "n_weak_top20_latents": int((~group["support_edge"]).sum()),
                "effective_n": effective,
                "top1_latent_idx": int(group.sort_values("abs_cohens_d", ascending=False).iloc[0]["latent_idx"]),
                "top1_effect_share": (
                    float(group["abs_cohens_d"].max() / top20_abs_sum)
                    if top20_abs_sum > 0
                    else 0.0
                ),
                "support_effect_share": (
                    float(support_abs_sum / top20_abs_sum) if top20_abs_sum > 0 else 0.0
                ),
                "support_latents": ",".join(str(int(v)) for v in support["latent_idx"].tolist()),
                "fragmentation_class": _classify_fragmentation(int(support.shape[0]), effective),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["fragmentation_class", "effective_n", "n_support_latents"],
        ascending=[True, False, False],
    )


def build_leaf_pair_overlap(matrix: pd.DataFrame) -> pd.DataFrame:
    raw_sets = _sets_by_label(matrix[matrix["label"].isin(LEAF_LABELS)], support_only=False)
    support_sets = _sets_by_label(matrix[matrix["label"].isin(LEAF_LABELS)], support_only=True)
    rows: list[dict[str, object]] = []
    for left, right in combinations(LEAF_LABELS, 2):
        left_family = _family_for_leaf(left)
        right_family = _family_for_leaf(right)
        raw_left = raw_sets.get(left, set())
        raw_right = raw_sets.get(right, set())
        support_left = support_sets.get(left, set())
        support_right = support_sets.get(right, set())
        support_j = _jaccard(support_left, support_right)
        relation_type = (
            "same_parent_sibling"
            if left_family == right_family and left_family in {"RE_family", "QU_family"}
            else "cross_family"
        )
        rows.append(
            {
                "label_a": left,
                "label_b": right,
                "family_a": left_family,
                "family_b": right_family,
                "relation_type": relation_type,
                "raw_intersection": int(len(raw_left & raw_right)),
                "raw_union": int(len(raw_left | raw_right)),
                "raw_jaccard": _jaccard(raw_left, raw_right),
                "support_intersection": int(len(support_left & support_right)),
                "support_union": int(len(support_left | support_right)),
                "support_jaccard": support_j,
                "overlap_class": _classify_overlap(support_j),
                "shared_support_latents": ",".join(str(v) for v in sorted(support_left & support_right)),
                "shared_raw_latents": ",".join(str(v) for v in sorted(raw_left & raw_right)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["support_jaccard", "raw_jaccard", "support_intersection"],
        ascending=[False, False, False],
    )


def build_family_overlap(matrix: pd.DataFrame) -> pd.DataFrame:
    raw_sets = _sets_by_label(matrix[matrix["label"].isin(LEAF_LABELS)], support_only=False)
    support_sets = _sets_by_label(matrix[matrix["label"].isin(LEAF_LABELS)], support_only=True)
    family_defs = {**FAMILY_MEMBERS, **SUPPLEMENTAL_BLOCKS}
    rows: list[dict[str, object]] = []
    main_names = list(FAMILY_MEMBERS)
    for left, right in combinations(main_names, 2):
        rows.append(_family_row(left, right, FAMILY_MEMBERS, raw_sets, support_sets, "main_family_union"))
    for block in SUPPLEMENTAL_BLOCKS:
        for target in ["RE_family", "QU_family"]:
            rows.append(_family_row(block, target, family_defs, raw_sets, support_sets, "supplemental_block"))
    return pd.DataFrame(rows).sort_values(
        ["comparison_scope", "support_jaccard", "raw_jaccard"],
        ascending=[True, False, False],
    )


def _family_row(
    left: str,
    right: str,
    family_defs: dict[str, list[str]],
    raw_sets: dict[str, set[int]],
    support_sets: dict[str, set[int]],
    scope: str,
) -> dict[str, object]:
    raw_left = _union_for_members(raw_sets, family_defs[left])
    raw_right = _union_for_members(raw_sets, family_defs[right])
    support_left = _union_for_members(support_sets, family_defs[left])
    support_right = _union_for_members(support_sets, family_defs[right])
    support_j = _jaccard(support_left, support_right)
    return {
        "family_a": left,
        "family_b": right,
        "comparison_scope": scope,
        "members_a": ",".join(family_defs[left]),
        "members_b": ",".join(family_defs[right]),
        "raw_intersection": int(len(raw_left & raw_right)),
        "raw_union": int(len(raw_left | raw_right)),
        "raw_jaccard": _jaccard(raw_left, raw_right),
        "support_intersection": int(len(support_left & support_right)),
        "support_union": int(len(support_left | support_right)),
        "support_jaccard": support_j,
        "overlap_class": _classify_overlap(support_j),
        "shared_support_latents": ",".join(str(v) for v in sorted(support_left & support_right)),
        "shared_raw_latents": ",".join(str(v) for v in sorted(raw_left & raw_right)),
    }


def build_parent_child_consistency(matrix: pd.DataFrame) -> pd.DataFrame:
    raw_sets = _sets_by_label(matrix, support_only=False)
    support_sets = _sets_by_label(matrix, support_only=True)
    rows: list[dict[str, object]] = []
    for parent, children in PARENT_CHILDREN.items():
        rows.append(
            _parent_row(parent, children, "parent_child_union", raw_sets, support_sets)
        )
        for child in children:
            rows.append(_parent_row(parent, [child], "parent_child", raw_sets, support_sets))
    return pd.DataFrame(rows)


def _parent_row(
    parent: str,
    children: list[str],
    comparison_type: str,
    raw_sets: dict[str, set[int]],
    support_sets: dict[str, set[int]],
) -> dict[str, object]:
    parent_raw = raw_sets.get(parent, set())
    child_raw = _union_for_members(raw_sets, children)
    parent_support = support_sets.get(parent, set())
    child_support = _union_for_members(support_sets, children)
    raw_inter = parent_raw & child_raw
    support_inter = parent_support & child_support
    return {
        "parent_label": parent,
        "child_labels": ",".join(children),
        "comparison_type": comparison_type,
        "parent_raw_n": int(len(parent_raw)),
        "child_raw_n": int(len(child_raw)),
        "raw_intersection": int(len(raw_inter)),
        "raw_jaccard": _jaccard(parent_raw, child_raw),
        "parent_covered_by_children_raw": (
            len(raw_inter) / len(parent_raw) if parent_raw else 0.0
        ),
        "children_covered_by_parent_raw": (
            len(raw_inter) / len(child_raw) if child_raw else 0.0
        ),
        "parent_support_n": int(len(parent_support)),
        "child_support_n": int(len(child_support)),
        "support_intersection": int(len(support_inter)),
        "support_jaccard": _jaccard(parent_support, child_support),
        "parent_covered_by_children_support": (
            len(support_inter) / len(parent_support) if parent_support else 0.0
        ),
        "children_covered_by_parent_support": (
            len(support_inter) / len(child_support) if child_support else 0.0
        ),
        "shared_raw_latents": ",".join(str(v) for v in sorted(raw_inter)),
        "shared_support_latents": ",".join(str(v) for v in sorted(support_inter)),
        "interpretation_scope": "consistency_check_only",
    }


def build_polysemanticity(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    leaf = matrix[matrix["label"].isin(LEAF_LABELS)].copy()
    support = leaf[leaf["support_edge"]].copy()
    rows: list[dict[str, object]] = []
    for latent_idx, group in support.groupby("latent_idx"):
        labels = sorted(group["label"].astype(str).unique().tolist())
        families = {_family_for_leaf(label) for label in labels}
        role = _classify_role(labels, families)
        rows.append(
            {
                "latent_idx": int(latent_idx),
                "n_leaf_labels_supported": int(len(labels)),
                "leaf_labels_supported": ",".join(labels),
                "n_families_supported": int(len(families)),
                "families_supported": ",".join(sorted(families)),
                "role": role,
                "max_abs_cohens_d": float(group["abs_cohens_d"].max()),
                "max_directional_auc": float(group["directional_auc"].max()),
                "direction_types": ",".join(sorted(group["edge_type"].unique().tolist())),
            }
        )
    assignments = pd.DataFrame(rows)
    if assignments.empty:
        assignments = pd.DataFrame(
            columns=[
                "latent_idx",
                "n_leaf_labels_supported",
                "leaf_labels_supported",
                "n_families_supported",
                "families_supported",
                "role",
                "max_abs_cohens_d",
                "max_directional_auc",
                "direction_types",
            ]
        )
    else:
        assignments = assignments.sort_values(
            ["n_leaf_labels_supported", "n_families_supported", "max_abs_cohens_d"],
            ascending=[False, False, False],
        )
    distribution = (
        assignments.groupby(["n_leaf_labels_supported", "role"], dropna=False)
        .size()
        .reset_index(name="n_latents")
        .sort_values(["n_leaf_labels_supported", "role"])
    )
    return assignments, distribution


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _fmt(value: object, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def build_summary(
    fragmentation: pd.DataFrame,
    leaf_overlap: pd.DataFrame,
    family_overlap: pd.DataFrame,
    parent_child: pd.DataFrame,
    assignments: pd.DataFrame,
) -> dict[str, object]:
    role_counts = assignments["role"].value_counts().to_dict() if not assignments.empty else {}
    return {
        "analysis_version": "structural_relations_v2",
        "top_k": 20,
        "support_edge_rule": {
            "significant_fdr": True,
            "min_abs_cohens_d": 0.5,
            "min_directional_auc": 0.60,
            "positive_min_precision_lift_at_50": 0.10,
            "negative_latents": "kept as boundary latents without high-activation precision gate",
        },
        "leaf_labels": LEAF_LABELS,
        "parent_labels": PARENT_LABELS,
        "fragmentation_counts": fragmentation["fragmentation_class"].value_counts().to_dict(),
        "highest_fragmentation": fragmentation.sort_values(
            ["effective_n", "n_support_latents"], ascending=[False, False]
        ).head(3).to_dict("records"),
        "highest_leaf_overlap": leaf_overlap.head(5).to_dict("records"),
        "highest_family_overlap": family_overlap.head(5).to_dict("records"),
        "parent_child_consistency": parent_child.to_dict("records"),
        "polysemanticity_role_counts": role_counts,
    }


def write_report(
    path: Path,
    summary: dict[str, object],
    fragmentation: pd.DataFrame,
    leaf_overlap: pd.DataFrame,
    family_overlap: pd.DataFrame,
    parent_child: pd.DataFrame,
    poly_distribution: pd.DataFrame,
) -> None:
    lines: list[str] = [
        "# MISC structural relations v2",
        "",
        "This report uses hierarchy-aware operational thresholds. Parent labels RE and QU are consistency checks only; the main overlap conclusions use leaf labels and family unions.",
        "",
        "## Support edge rule",
        "",
        "- FDR significant",
        "- `abs_cohens_d >= 0.5`",
        "- `directional_auc >= 0.60`",
        "- positive edges require `precision_lift_at_50 >= 0.10`",
        "- negative edges are retained as boundary latents without high-activation precision gating",
        "",
        "## Fragmentation",
        "",
        "| Label | Class | Support latents | Effective n | Top1 effect share | Weak Top20 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    frag_order = fragmentation.sort_values(["effective_n", "n_support_latents"], ascending=[False, False])
    for _, row in frag_order.iterrows():
        lines.append(
            f"| {row['label']} | {row['fragmentation_class']} | {int(row['n_support_latents'])} | "
            f"{_fmt(row['effective_n'])} | {_fmt(row['top1_effect_share'])} | {int(row['n_weak_top20_latents'])} |"
        )

    lines.extend(
        [
            "",
            "## Leaf-label overlap",
            "",
            "| Label A | Label B | Relation | Support Jaccard | Raw Jaccard | Class | Shared support latents |",
            "|---|---|---|---:|---:|---|---|",
        ]
    )
    for _, row in leaf_overlap.head(12).iterrows():
        lines.append(
            f"| {row['label_a']} | {row['label_b']} | {row['relation_type']} | "
            f"{_fmt(row['support_jaccard'])} | {_fmt(row['raw_jaccard'])} | {row['overlap_class']} | "
            f"{row['shared_support_latents'] or '-'} |"
        )

    lines.extend(
        [
            "",
            "## Family-union overlap",
            "",
            "| Family A | Family B | Scope | Support Jaccard | Raw Jaccard | Class | Shared support latents |",
            "|---|---|---|---:|---:|---|---|",
        ]
    )
    for _, row in family_overlap.iterrows():
        lines.append(
            f"| {row['family_a']} | {row['family_b']} | {row['comparison_scope']} | "
            f"{_fmt(row['support_jaccard'])} | {_fmt(row['raw_jaccard'])} | {row['overlap_class']} | "
            f"{row['shared_support_latents'] or '-'} |"
        )

    lines.extend(
        [
            "",
            "## Parent-child consistency",
            "",
            "| Parent | Child labels | Type | Parent covered raw | Parent covered support | Raw Jaccard | Support Jaccard |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in parent_child.iterrows():
        lines.append(
            f"| {row['parent_label']} | {row['child_labels']} | {row['comparison_type']} | "
            f"{_fmt(row['parent_covered_by_children_raw'])} | {_fmt(row['parent_covered_by_children_support'])} | "
            f"{_fmt(row['raw_jaccard'])} | {_fmt(row['support_jaccard'])} |"
        )

    lines.extend(
        [
            "",
            "## Polysemanticity",
            "",
            "| N leaf labels | Role | N latents |",
            "|---:|---|---:|",
        ]
    )
    for _, row in poly_distribution.iterrows():
        lines.append(
            f"| {int(row['n_leaf_labels_supported'])} | {row['role']} | {int(row['n_latents'])} |"
        )

    lines.extend(
        [
            "",
            "## Corrected conclusion",
            "",
            "- Parent-child overlaps are not independent findings; they are hierarchy consistency checks.",
            "- The main overlap evidence should be read from leaf-label and family-union tables.",
            "- Labels with large `effective_n` and many support latents are more distributed; labels with few support latents are more compact or weak under this operational rule.",
            "- Polysemantic latents are defined by support edges across leaf labels and families, not by raw parent-child reuse.",
            "",
            "## Files",
            "",
            "- `label_fragmentation_v2.csv`",
            "- `leaf_pair_overlap_v2.csv`",
            "- `family_union_overlap_v2.csv`",
            "- `parent_child_consistency_v2.csv`",
            "- `latent_polysemanticity_v2.csv`",
            "- `latent_role_assignments_v2.csv`",
            "- `structural_relation_summary.json`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_outputs(
    output_dir: Path,
    fragmentation: pd.DataFrame,
    leaf_overlap: pd.DataFrame,
    family_overlap: pd.DataFrame,
    poly_distribution: pd.DataFrame,
    assignments: pd.DataFrame,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    frag = fragmentation.set_index("label").loc[LEAF_LABELS].reset_index()
    x = np.arange(len(frag))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - 0.2, frag["effective_n"], width=0.4, label="effective_n")
    ax.bar(x + 0.2, frag["n_support_latents"], width=0.4, label="n_support_latents")
    ax.set_xticks(x)
    ax.set_xticklabels(frag["label"])
    ax.set_ylabel("latent count")
    ax.set_title("Fragmentation v2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "fragmentation_v2_bar.png", dpi=220)
    plt.close(fig)

    _pair_heatmap(
        leaf_overlap,
        LEAF_LABELS,
        "label_a",
        "label_b",
        "support_jaccard",
        "Leaf-label support Jaccard",
        figure_dir / "leaf_label_jaccard_heatmap_v2.png",
    )

    main_family = family_overlap[family_overlap["comparison_scope"] == "main_family_union"]
    family_labels = list(FAMILY_MEMBERS)
    _pair_heatmap(
        main_family,
        family_labels,
        "family_a",
        "family_b",
        "support_jaccard",
        "Family-union support Jaccard",
        figure_dir / "family_union_jaccard_heatmap_v2.png",
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not poly_distribution.empty:
        by_n = poly_distribution.groupby("n_leaf_labels_supported")["n_latents"].sum()
        ax.bar(by_n.index.astype(str), by_n.values)
    ax.set_xlabel("leaf labels supported")
    ax.set_ylabel("n latents")
    ax.set_title("Latent polysemanticity v2")
    fig.tight_layout()
    fig.savefig(figure_dir / "polysemanticity_histogram_v2.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not assignments.empty:
        counts = assignments["role"].value_counts().reindex(
            ["label_specific", "sibling_shared", "cross_family", "generalized"],
            fill_value=0,
        )
        ax.bar(counts.index, counts.values)
        ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel("n latents")
    ax.set_title("Latent role taxonomy v2")
    fig.tight_layout()
    fig.savefig(figure_dir / "latent_role_taxonomy_v2.png", dpi=220)
    plt.close(fig)


def _pair_heatmap(
    rows: pd.DataFrame,
    labels: list[str],
    col_a: str,
    col_b: str,
    value_col: str,
    title: str,
    output_path: Path,
) -> None:
    heat = pd.DataFrame(0.0, index=labels, columns=labels)
    for _, row in rows.iterrows():
        a = row[col_a]
        b = row[col_b]
        if a in heat.index and b in heat.columns:
            heat.loc[a, b] = float(row[value_col])
            heat.loc[b, a] = float(row[value_col])
    for label in labels:
        heat.loc[label, label] = 1.0
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(heat.values, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def validate_outputs(
    matrix: pd.DataFrame,
    fragmentation: pd.DataFrame,
    leaf_overlap: pd.DataFrame,
    family_overlap: pd.DataFrame,
    parent_child: pd.DataFrame,
    output_dir: Path,
) -> None:
    for label in LEAF_LABELS:
        n_rows = matrix[matrix["label"] == label].shape[0]
        if n_rows != 20:
            raise AssertionError(f"{label} expected 20 TopK rows, found {n_rows}")
    pair_labels = set(leaf_overlap["label_a"]) | set(leaf_overlap["label_b"])
    if any(parent in pair_labels for parent in PARENT_LABELS):
        raise AssertionError("Parent labels leaked into main leaf overlap table")
    for frame, cols in [
        (leaf_overlap, ["raw_jaccard", "support_jaccard"]),
        (family_overlap, ["raw_jaccard", "support_jaccard"]),
        (parent_child, ["raw_jaccard", "support_jaccard"]),
    ]:
        for col in cols:
            if not frame[col].between(0.0, 1.0).all():
                raise AssertionError(f"{col} has values outside [0, 1]")
    if not (fragmentation["effective_n"] <= fragmentation["n_support_latents"] + 1e-9).all():
        raise AssertionError("effective_n must be <= n_support_latents")
    if not fragmentation["n_support_latents"].between(0, 20).all():
        raise AssertionError("n_support_latents must be within [0, 20]")
    sibling_pairs = {
        tuple(sorted(v))
        for v in leaf_overlap[leaf_overlap["relation_type"] == "same_parent_sibling"][
            ["label_a", "label_b"]
        ].values.tolist()
    }
    if tuple(sorted(["QUO", "QUC"])) not in sibling_pairs:
        raise AssertionError("QUO-QUC must be a sibling comparison")
    if tuple(sorted(["RES", "REC"])) not in sibling_pairs:
        raise AssertionError("RES-REC must be a sibling comparison")
    parent_pairs = set(zip(parent_child["parent_label"], parent_child["child_labels"]))
    if ("QU", "QUO") not in parent_pairs or ("RE", "REC") not in parent_pairs:
        raise AssertionError("Required parent-child consistency rows are missing")
    report_text = (output_dir / "structural_relation_report.md").read_text(encoding="utf-8")
    forbidden = "QU-QUO and RE-REC show the strongest"
    if forbidden in report_text:
        raise AssertionError("Report contains forbidden parent-child independent-overlap wording")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topk-matrix",
        default="outputs/misc_full_sae_eval/interpretability/mapping_structure/topk_candidate_matrix.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/misc_full_sae_eval/interpretability/structural_relations_v2",
    )
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = _read_topk(Path(args.topk_matrix), top_k=args.top_k)
    fragmentation = build_fragmentation(matrix)
    leaf_overlap = build_leaf_pair_overlap(matrix)
    family_overlap = build_family_overlap(matrix)
    parent_child = build_parent_child_consistency(matrix)
    assignments, poly_distribution = build_polysemanticity(matrix)

    fragmentation.to_csv(output_dir / "label_fragmentation_v2.csv", index=False)
    leaf_overlap.to_csv(output_dir / "leaf_pair_overlap_v2.csv", index=False)
    family_overlap.to_csv(output_dir / "family_union_overlap_v2.csv", index=False)
    parent_child.to_csv(output_dir / "parent_child_consistency_v2.csv", index=False)
    poly_distribution.to_csv(output_dir / "latent_polysemanticity_v2.csv", index=False)
    assignments.to_csv(output_dir / "latent_role_assignments_v2.csv", index=False)

    summary = build_summary(fragmentation, leaf_overlap, family_overlap, parent_child, assignments)
    _write_json(output_dir / "structural_relation_summary.json", summary)
    write_report(
        output_dir / "structural_relation_report.md",
        summary,
        fragmentation,
        leaf_overlap,
        family_overlap,
        parent_child,
        poly_distribution,
    )
    plot_outputs(output_dir, fragmentation, leaf_overlap, family_overlap, poly_distribution, assignments)
    validate_outputs(matrix, fragmentation, leaf_overlap, family_overlap, parent_child, output_dir)
    print(f"Wrote structural relations v2 outputs to {output_dir}")


if __name__ == "__main__":
    main()
