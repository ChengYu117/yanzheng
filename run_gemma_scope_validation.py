"""Validate Gemma3-4B + GemmaScope SAE experiment outputs.

The validator is intentionally lightweight: it does not rerun model inference,
but checks the artifacts produced by the full GemmaScope pipeline and the
downstream interpretability analyses.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch


DEFAULT_LABELS = ("RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF")
DEFAULT_LEAF_LABELS = ("RES", "REC", "QUO", "QUC", "GI", "SU", "AF")


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


class Validator:
    def __init__(self) -> None:
        self.checks: list[CheckResult] = []

    def pass_(self, name: str, detail: str) -> None:
        self.checks.append(CheckResult(name, "PASS", detail))

    def warn(self, name: str, detail: str) -> None:
        self.checks.append(CheckResult(name, "WARN", detail))

    def fail(self, name: str, detail: str) -> None:
        self.checks.append(CheckResult(name, "FAIL", detail))

    @property
    def failed(self) -> bool:
        return any(check.status == "FAIL" for check in self.checks)

    @property
    def warned(self) -> bool:
        return any(check.status == "WARN" for check in self.checks)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _check_required_files(v: Validator, root: Path, output_dir: Path) -> dict[str, Path]:
    files = {
        "records": root / "records.jsonl",
        "label_matrix": root / "label_matrix.csv",
        "features": root / "feature_store" / "utterance_features.pt",
        "hidden": root / "feature_store" / "utterance_activations.pt",
        "feature_metadata": root / "feature_store" / "feature_metadata.json",
        "structural_metrics": root / "metrics_structural.json",
        "run_summary": root / "run_summary.json",
        "mapping_matrix": root / "functional" / "misc_label_mapping" / "latent_label_matrix.csv",
        "latent_space_summary": root
        / "interpretability"
        / "latent_space_search_v2"
        / "latent_space_search_summary.json",
        "latent_space_report": root
        / "interpretability"
        / "latent_space_search_v2"
        / "latent_space_search_report.md",
        "fragmentation": root / "interpretability" / "latent_space_search_v2" / "fragmentation_v2.csv",
        "thresholded_sets": root
        / "interpretability"
        / "latent_space_search_v2"
        / "thresholded_latent_sets_v2.csv",
        "overlap_thresholded": root
        / "interpretability"
        / "latent_space_search_v2"
        / "overlap_thresholded_v2.csv",
        "overlap_weighted": root / "interpretability" / "latent_space_search_v2" / "overlap_weighted_v2.csv",
        "polysemanticity": root / "interpretability" / "latent_space_search_v2" / "polysemanticity_v2.csv",
        "minimal_summary": root
        / "interpretability"
        / "minimal_sufficient_subspace_v2"
        / "minimal_sufficient_summary_v2.csv",
        "minimal_report": root
        / "interpretability"
        / "minimal_sufficient_subspace_v2"
        / "minimal_sufficient_subspace_report.md",
        "baseline_table": root
        / "interpretability"
        / "baseline_comparison_step6"
        / "table4_baseline_comparison.csv",
        "baseline_report": root
        / "interpretability"
        / "baseline_comparison_step6"
        / "baseline_comparison_report.md",
        "cross_model_table": root
        / "interpretability"
        / "model_specificity_comparison"
        / "label_level_model_comparison.csv",
        "cross_model_report": root
        / "interpretability"
        / "model_specificity_comparison"
        / "model_specificity_report.md",
    }
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        v.fail("required_files", f"Missing required artifacts: {', '.join(missing)}")
    else:
        v.pass_("required_files", f"All {len(files)} required artifacts exist.")
    output_dir.mkdir(parents=True, exist_ok=True)
    return files


def _extract_tensor(obj: Any, key: str) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict) and key in obj and isinstance(obj[key], torch.Tensor):
        return obj[key]
    raise TypeError(f"Could not find tensor key {key!r}.")


def _check_feature_store(v: Validator, files: dict[str, Path], labels: tuple[str, ...]) -> tuple[int, int, int]:
    metadata = _load_json(files["feature_metadata"])
    expected_records = int(metadata.get("n_records", -1))
    expected_feature_shape = tuple(metadata.get("feature_shape", []))
    expected_activation_shape = tuple(metadata.get("activation_shape", []))

    features_obj = torch.load(files["features"], map_location="cpu")
    hidden_obj = torch.load(files["hidden"], map_location="cpu")
    features = _extract_tensor(features_obj, "utterance_features")
    hidden = _extract_tensor(hidden_obj, "utterance_activations")
    feature_shape = tuple(features.shape)
    hidden_shape = tuple(hidden.shape)

    if feature_shape == expected_feature_shape == (expected_records, 16384):
        v.pass_("feature_shape", f"GemmaScope feature shape is {feature_shape}.")
    else:
        v.fail("feature_shape", f"Unexpected feature shape {feature_shape}; metadata={expected_feature_shape}.")

    if hidden_shape == expected_activation_shape and hidden_shape[0] == expected_records and hidden_shape[1] == 2560:
        v.pass_("hidden_shape", f"Gemma raw hidden shape is {hidden_shape}.")
    else:
        v.fail("hidden_shape", f"Unexpected hidden shape {hidden_shape}; metadata={expected_activation_shape}.")

    feature_finite = bool(torch.isfinite(features).all())
    hidden_finite = bool(torch.isfinite(hidden).all())
    if feature_finite and hidden_finite:
        v.pass_("feature_finiteness", "Feature and hidden tensors contain only finite values.")
    else:
        v.fail("feature_finiteness", f"finite(features)={feature_finite}, finite(hidden)={hidden_finite}.")

    label_matrix = pd.read_csv(files["label_matrix"])
    missing_labels = [label for label in labels if label not in label_matrix.columns]
    extra_other = "OTHER" in label_matrix.columns
    if len(label_matrix) == expected_records and not missing_labels and not extra_other:
        v.pass_("label_matrix", f"Label matrix has {len(label_matrix)} rows and the 9 core labels only.")
    else:
        v.fail(
            "label_matrix",
            f"rows={len(label_matrix)}, missing={missing_labels}, has_OTHER={extra_other}.",
        )
    return expected_records, feature_shape[1], hidden_shape[1]


def _check_structural_metrics(v: Validator, files: dict[str, Path]) -> None:
    metrics = _load_json(files["structural_metrics"])
    required = [
        "n_valid_tokens",
        "n_invalid_tokens_skipped",
        "mse",
        "mae",
        "explained_variance",
        "cosine_similarity",
        "l0_mean",
        "l0_std",
    ]
    missing = [key for key in required if key not in metrics]
    nonfinite = [key for key in required if key in metrics and not _finite(metrics[key])]
    if missing or nonfinite:
        v.fail("structural_metrics", f"missing={missing}, nonfinite={nonfinite}.")
        return
    ev = float(metrics["explained_variance"])
    cosine = float(metrics["cosine_similarity"])
    invalid = int(metrics["n_invalid_tokens_skipped"])
    l0 = float(metrics["l0_mean"])
    if invalid != 0:
        v.fail("structural_metrics", f"n_invalid_tokens_skipped={invalid}; expected 0.")
    elif not (-1.0 <= ev <= 1.01 and -1.0 <= cosine <= 1.01 and l0 > 0):
        v.fail("structural_metrics", f"Out-of-range EV={ev}, cosine={cosine}, l0_mean={l0}.")
    else:
        v.pass_("structural_metrics", f"finite metrics: EV={ev:.6f}, cosine={cosine:.6f}, L0={l0:.3f}.")


def _check_mapping(v: Validator, files: dict[str, Path], labels: tuple[str, ...], n_features: int) -> None:
    matrix = pd.read_csv(files["mapping_matrix"])
    found_labels = set(matrix["label"].astype(str).str.upper())
    missing = [label for label in labels if label not in found_labels]
    if missing or "OTHER" in found_labels:
        v.fail("mapping_labels", f"missing={missing}, has_OTHER={'OTHER' in found_labels}.")
    else:
        v.pass_("mapping_labels", f"Mapping matrix covers {len(labels)} core labels and excludes OTHER.")

    expected_rows = n_features * len(labels)
    if len(matrix) == expected_rows:
        v.pass_("mapping_shape", f"latent-label matrix has expected {expected_rows} rows.")
    else:
        v.fail("mapping_shape", f"rows={len(matrix)}, expected={expected_rows}.")

    bounded_cols = [
        col
        for col in ["auc", "directional_auc", "precision_at_50", "precision_at_100", "prevalence"]
        if col in matrix.columns
    ]
    bad_cols = []
    for col in bounded_cols:
        values = pd.to_numeric(matrix[col], errors="coerce")
        if values.isna().any() or ((values < 0) | (values > 1)).any():
            bad_cols.append(col)
    if bad_cols:
        v.fail("mapping_metric_ranges", f"Out-of-range or NaN columns: {bad_cols}.")
    else:
        v.pass_("mapping_metric_ranges", f"Bounded mapping metrics valid: {bounded_cols}.")


def _check_latent_space(v: Validator, files: dict[str, Path], labels: tuple[str, ...]) -> None:
    frag = pd.read_csv(files["fragmentation"])
    thresholded = pd.read_csv(files["thresholded_sets"])
    overlap_thresholded = pd.read_csv(files["overlap_thresholded"])
    overlap_weighted = pd.read_csv(files["overlap_weighted"])
    poly = pd.read_csv(files["polysemanticity"])

    frag_labels = set(frag["label"].astype(str).str.upper())
    missing = [label for label in labels if label not in frag_labels]
    if len(frag) == len(labels) and not missing:
        v.pass_("fragmentation_labels", "fragmentation_v2 contains one row per core label.")
    else:
        v.fail("fragmentation_labels", f"rows={len(frag)}, missing={missing}.")

    if "thresholded_latent_count" in frag and "effective_fragmentation" in frag:
        counts = pd.to_numeric(frag["thresholded_latent_count"], errors="coerce")
        effective = pd.to_numeric(frag["effective_fragmentation"], errors="coerce")
        if counts.isna().any() or effective.isna().any() or (effective - counts > 1e-8).any():
            v.fail("fragmentation_consistency", "effective_fragmentation exceeds thresholded_latent_count.")
        else:
            v.pass_("fragmentation_consistency", "effective_fragmentation <= thresholded_latent_count for all labels.")

    expected_top20 = 20 * len(labels)
    top20 = pd.read_csv(files["latent_space_summary"].parent / "top20_candidate_set_v2.csv")
    if len(top20) == expected_top20:
        v.pass_("top20_candidates", f"Top20 candidate set has expected {expected_top20} rows.")
    else:
        v.fail("top20_candidates", f"Top20 rows={len(top20)}, expected={expected_top20}.")

    stable_count = (
        int(thresholded["stable_edge"].fillna(False).astype(bool).sum())
        if "stable_edge" in thresholded.columns
        else len(thresholded)
    )
    thresholded_labels = set(thresholded.loc[thresholded.get("latent_idx", 0) != -1, "label"].astype(str).str.upper())
    if stable_count > 0 and thresholded_labels.issubset(set(labels)):
        v.pass_("thresholded_sets", f"thresholded_latent_sets_v2 has {stable_count} stable edges.")
    else:
        v.warn("thresholded_sets", "Thresholded set is empty or has unexpected labels.")

    for name, df in [("overlap_thresholded", overlap_thresholded), ("overlap_weighted", overlap_weighted)]:
        jaccard_cols = [col for col in df.columns if "jaccard" in col.lower()]
        bad = []
        for col in jaccard_cols:
            values = pd.to_numeric(df[col], errors="coerce")
            finite_values = values.dropna()
            # Some top-K rows are not applicable for family-union scopes and
            # are intentionally stored as blank/NaN. Validate the values that
            # are defined instead of treating N/A as a failed statistic.
            if ((finite_values < -1e-12) | (finite_values > 1 + 1e-12)).any():
                bad.append(col)
        if bad:
            v.fail(name, f"Out-of-range Jaccard columns: {bad}.")
        else:
            v.pass_(name, f"All Jaccard columns are in [0, 1]: {jaccard_cols}.")

    leaf_col = "n_leaf_labels_supported" if "n_leaf_labels_supported" in poly.columns else "n_thresholded_labels"
    family_col = "n_families_supported" if "n_families_supported" in poly.columns else "n_families"
    if leaf_col in poly.columns and family_col in poly.columns:
        leaf = pd.to_numeric(poly[leaf_col], errors="coerce")
        fam = pd.to_numeric(poly[family_col], errors="coerce")
        if leaf.isna().any() or fam.isna().any() or (leaf < 0).any() or (fam < 0).any():
            v.fail("polysemanticity", "Invalid polysemanticity counts.")
        else:
            v.pass_("polysemanticity", f"Polysemanticity table valid with {len(poly)} rows.")
    else:
        v.fail("polysemanticity", "Missing polysemanticity count columns.")


def _check_minimal(v: Validator, files: dict[str, Path], labels: tuple[str, ...], leaf_labels: tuple[str, ...]) -> None:
    summary = pd.read_csv(files["minimal_summary"])
    found = set(summary["label"].astype(str).str.upper())
    missing = [label for label in labels if label not in found]
    if len(summary) == len(labels) and not missing:
        v.pass_("minimal_labels", "Minimal sufficient summary contains one row per core label.")
    else:
        v.fail("minimal_labels", f"rows={len(summary)}, missing={missing}.")

    bounded_cols = [col for col in ["full_auc_mean", "full_average_precision_mean", "full_f1_mean"] if col in summary.columns]
    bad_cols = []
    for col in bounded_cols:
        values = pd.to_numeric(summary[col], errors="coerce")
        if values.isna().any() or ((values < 0) | (values > 1)).any():
            bad_cols.append(col)
    if bad_cols:
        v.fail("minimal_metric_ranges", f"Out-of-range minimal metrics: {bad_cols}.")
    else:
        v.pass_("minimal_metric_ranges", f"Minimal metrics valid: {bounded_cols}.")

    if {"minimal_k_median", "candidate_pool_size"}.issubset(summary.columns):
        k = pd.to_numeric(summary["minimal_k_median"], errors="coerce")
        pool = pd.to_numeric(summary["candidate_pool_size"], errors="coerce")
        recoverable = summary["formal_status"].astype(str).str.contains("minimal_sufficient", na=False)
        invalid = recoverable & (k.isna() | pool.isna() | (k > pool))
        if invalid.any():
            labels_bad = summary.loc[invalid, "label"].astype(str).tolist()
            v.fail("minimal_k_consistency", f"minimal_k_median exceeds candidate pool for {labels_bad}.")
        else:
            v.pass_("minimal_k_consistency", "minimal K is within candidate pool for all recoverable labels.")

    leaf = summary[summary["label"].astype(str).str.upper().isin(leaf_labels)]
    if not leaf.empty and set(leaf["fragmentation_class"].astype(str)).issubset({"compact", "moderate", "distributed", "not_recoverable"}):
        v.pass_("minimal_leaf_classes", "Leaf labels have valid compact/moderate/distributed status.")
    else:
        v.fail("minimal_leaf_classes", "Leaf labels missing valid fragmentation_class values.")


def _check_baselines_and_cross_model(v: Validator, files: dict[str, Path], labels: tuple[str, ...]) -> None:
    baseline = pd.read_csv(files["baseline_table"])
    expected_reps = {"sae_latents", "pca_components", "raw_hidden_dims"}
    found_reps = set(baseline["representation"].astype(str))
    if expected_reps.issubset(found_reps):
        v.pass_("baseline_representations", "Step6 table contains SAE, PCA and raw hidden baselines.")
    else:
        v.fail("baseline_representations", f"found={sorted(found_reps)}.")

    for col in ["mean_label_auc", "macro_probe_auc", "macro_probe_f1", "pair_recovery_at_k"]:
        values = pd.to_numeric(baseline[col], errors="coerce")
        if values.isna().any() or ((values < 0) | (values > 1)).any():
            v.fail("baseline_metric_ranges", f"Column {col} has invalid values.")
            break
    else:
        v.pass_("baseline_metric_ranges", "Step6 bounded metrics are valid.")

    cross = pd.read_csv(files["cross_model_table"])
    models = set(cross["model"].astype(str))
    cross_labels = set(cross["label"].astype(str).str.upper())
    if {"llama", "gemma"}.issubset(models) and set(labels).issubset(cross_labels):
        v.pass_("cross_model_coverage", "Cross-model table covers Llama/Gemma and all core labels.")
    else:
        v.fail("cross_model_coverage", f"models={models}, missing_labels={set(labels) - cross_labels}.")


def _write_outputs(v: Validator, output_dir: Path, root: Path) -> tuple[Path, Path]:
    summary_path = output_dir / "gemma_scope_validation_summary.json"
    report_path = output_dir / "gemma_scope_validation_report.md"
    status = "FAILED" if v.failed else ("PASSED_WITH_WARNINGS" if v.warned else "PASSED")
    summary = {
        "status": status,
        "root": str(root),
        "n_checks": len(v.checks),
        "n_pass": sum(check.status == "PASS" for check in v.checks),
        "n_warn": sum(check.status == "WARN" for check in v.checks),
        "n_fail": sum(check.status == "FAIL" for check in v.checks),
        "checks": [check.__dict__ for check in v.checks],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# GemmaScope Layer-18 Validation Report",
        "",
        f"- Root: `{root}`",
        f"- Verification status: **{status}**",
        f"- Checks: {summary['n_pass']} pass, {summary['n_warn']} warning, {summary['n_fail']} fail",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for check in v.checks:
        detail = check.detail.replace("|", "\\|")
        lines.append(f"| `{check.name}` | {check.status} | {detail} |")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- This validation checks artifact integrity, finite metrics, label coverage, shape consistency, and bounded metric ranges.",
            "- It does not re-run expensive Gemma model inference unless upstream artifacts are regenerated separately.",
            "- Passing validation supports reproducible use of the current artifacts, not causal mechanism claims.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Gemma3-4B + GemmaScope SAE experiment outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", default="outputs/gemma3_l18_gemmascope_sae_eval")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "validation"
    labels = tuple(label.upper() for label in args.labels)
    leaf_labels = tuple(label for label in DEFAULT_LEAF_LABELS if label in labels)
    v = Validator()

    if not root.exists():
        v.fail("root_exists", f"Root does not exist: {root}")
        files = {}
    else:
        v.pass_("root_exists", f"Root exists: {root}")
        files = _check_required_files(v, root, output_dir)

    if files and not v.failed:
        n_records, n_features, _ = _check_feature_store(v, files, labels)
        _check_structural_metrics(v, files)
        _check_mapping(v, files, labels, n_features)
        _check_latent_space(v, files, labels)
        _check_minimal(v, files, labels, leaf_labels)
        _check_baselines_and_cross_model(v, files, labels)

    summary_path, report_path = _write_outputs(v, output_dir, root)
    status = "FAILED" if v.failed else ("PASSED_WITH_WARNINGS" if v.warned else "PASSED")
    print(f"GemmaScope validation {status}.")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    return 1 if v.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
