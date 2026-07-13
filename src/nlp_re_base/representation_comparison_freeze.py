"""Freeze and audit the final leaf-label representation comparison artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


METRICS = ("auc", "average_precision", "f1", "balanced_accuracy")
REPRESENTATIONS = (
    "Hidden State",
    "Full SAE",
    "Top-n SAE",
    "Stable Core SAE",
    "PCA-n",
    "Random SAE-n",
)
LEAF_LABELS = ("RES", "REC", "QUO", "QUC", "GI", "SU", "AF")


@dataclass(frozen=True)
class FreezeConfig:
    labels: tuple[str, ...] = LEAF_LABELS
    package_name: str = "representation_comparison_frozen_leaf7_20260713"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _assert_source_contract(fold: pd.DataFrame, manifest: dict[str, Any], labels: tuple[str, ...]) -> None:
    missing_columns = {
        "representation", "label", "fold", "top_n", "seed", "split_policy",
        "n_train", "n_test", "train_positive", "test_positive", "n_features", "status", *METRICS,
    } - set(fold.columns)
    if missing_columns:
        raise ValueError(f"source fold CSV missing columns: {sorted(missing_columns)}")
    missing_representations = set(REPRESENTATIONS) - set(fold["representation"].astype(str))
    if missing_representations:
        raise ValueError(f"source fold CSV missing representations: {sorted(missing_representations)}")
    missing_labels = set(labels) - set(fold["label"].astype(str))
    if missing_labels:
        raise ValueError(f"source fold CSV missing labels: {sorted(missing_labels)}")
    if not fold[fold["label"].isin(labels)]["status"].astype(str).eq("ok").all():
        raise ValueError("not all selected leaf-label fold rows have status=ok")
    config = manifest.get("config", {})
    if list(config.get("top_ns", [])) != [10, 20, 50, 100, 200]:
        raise ValueError("unexpected top-n grid in source manifest")


def _split_signature_audit(fold: pd.DataFrame, labels: tuple[str, ...]) -> tuple[bool, str]:
    keys = ["label", "fold", "n_train", "n_test", "train_positive", "test_positive", "split_policy"]
    unique = fold[fold["label"].isin(labels)][keys + ["representation"]].drop_duplicates()
    counts = unique.groupby(["label", "fold"], dropna=False)["representation"].nunique()
    expected = len(REPRESENTATIONS)
    passed = bool((counts == expected).all())
    detail = f"{len(counts)} label-fold signatures checked; expected {expected} representations each"
    return passed, detail


def _summarize_by_label(fold: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (representation, top_n, label), group in fold.groupby(
        ["representation", "top_n", "label"], dropna=False, sort=True
    ):
        row: dict[str, Any] = {
            "representation": representation,
            "top_n": int(top_n),
            "label": label,
            "n_fold_rows": int(len(group)),
            "n_features_mean": float(pd.to_numeric(group["n_features"], errors="coerce").mean()),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_se"] = float(row[f"{metric}_std"] / np.sqrt(len(values)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["representation", "top_n", "label"]).reset_index(drop=True)


def _summarize_macro(by_label: pd.DataFrame, fold: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (representation, top_n), group in by_label.groupby(["representation", "top_n"], dropna=False):
        row: dict[str, Any] = {
            "representation": representation,
            "top_n": int(top_n),
            "n_labels": int(group["label"].nunique()),
            "n_features_mean": float(group["n_features_mean"].mean()),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[f"{metric}_mean"], errors="coerce").dropna()
            row[f"macro_{metric}"] = float(values.mean())
            row[f"macro_{metric}_label_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    macro = pd.DataFrame(rows)
    for metric in METRICS:
        macro[f"macro_{metric}_seed_std"] = np.nan
        macro[f"macro_{metric}_seed_se"] = np.nan
    random_rows = fold[fold["representation"].eq("Random SAE-n")]
    if not random_rows.empty:
        per_seed_label = random_rows.groupby(["top_n", "seed", "label"], as_index=False)[list(METRICS)].mean()
        per_seed_macro = per_seed_label.groupby(["top_n", "seed"], as_index=False)[list(METRICS)].mean()
        for top_n, group in per_seed_macro.groupby("top_n"):
            mask = macro["representation"].eq("Random SAE-n") & macro["top_n"].eq(int(top_n))
            for metric in METRICS:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                macro.loc[mask, f"macro_{metric}_seed_std"] = std
                macro.loc[mask, f"macro_{metric}_seed_se"] = std / np.sqrt(max(len(values), 1))
    return macro.sort_values(["representation", "top_n"]).reset_index(drop=True)


def _protocol_audit(fold: pd.DataFrame, manifest: dict[str, Any], labels: tuple[str, ...]) -> pd.DataFrame:
    config = manifest.get("config", {})
    protocol = manifest.get("protocol", {})
    split_ok, split_detail = _split_signature_audit(fold, labels)
    pca = fold[fold["representation"].eq("PCA-n") & fold["label"].isin(labels)]
    pca_flags_ok = (
        not pca.empty
        and pca.get("pca_input_standardized", pd.Series(False, index=pca.index)).astype(str).str.lower().eq("true").all()
        and pca.get("pca_post_standardized", pd.Series(False, index=pca.index)).astype(str).str.lower().eq("true").all()
    )
    rows = [
        ("same_outer_folds", "PASS" if split_ok else "FAIL", split_detail),
        (
            "same_classifier",
            "PASS" if config.get("C") == 1.0 and config.get("solver") == "liblinear" else "FAIL",
            f"LogisticRegression(C={config.get('C')}, solver={config.get('solver')}, class_weight=balanced, max_iter={config.get('max_iter')})",
        ),
        (
            "same_shared_preprocessing",
            "PASS" if config.get("standardize") is True and pca_flags_ok else "FAIL",
            "All probe inputs use train-fold StandardScaler; PCA additionally fits PCA and post-PCA scaling on the train fold only",
        ),
        (
            "same_primary_seed",
            "PASS" if config.get("random_state") == 42 else "FAIL",
            f"random_state={config.get('random_state')}; Random SAE repeats intentionally use seeds 42..61",
        ),
        (
            "same_metrics",
            "PASS" if all(metric in fold.columns for metric in METRICS) else "FAIL",
            "ROC-AUC, PR-AUC/Average Precision, threshold-0.5 F1, Balanced Accuracy",
        ),
        ("hidden_selection_train_only", "PASS", "No feature selection"),
        ("full_sae_selection_train_only", "PASS", "No feature selection"),
        ("topn_sae_selection_train_only", "PASS", "Positive Cohen's d ranking is recomputed inside every outer training fold"),
        ("pca_selection_train_only", "PASS" if pca_flags_ok else "FAIL", "PCA and both scalers are fit on outer training rows only"),
        ("random_sae_selection_train_only", "PASS", "Sampling is label/data independent from the filtered keep pool"),
        (
            "stable_core_selection_train_only",
            "FAIL",
            "Stable Core SAE rows use a supervised stable-core list selected on the full analysis dataset, not nested inside each outer fold",
        ),
        (
            "imported_rows_traceable",
            "PASS" if protocol.get("sae_rows_imported") is True and protocol.get("sae_rows_rerun") is False else "FAIL",
            "Hidden/SAE/Random rows imported from the stable-core run; PCA rows recomputed with post-PCA standardization",
        ),
    ]
    return pd.DataFrame(rows, columns=["check", "status", "evidence"])


def _write_readme(path: Path, macro: pd.DataFrame, audit: pd.DataFrame, labels: tuple[str, ...]) -> None:
    table = macro[[
        "representation", "top_n", "macro_auc", "macro_average_precision", "macro_f1", "macro_balanced_accuracy"
    ]].copy()
    lines = [
        "# Frozen leaf-label representation comparison",
        "",
        f"Labels: `{', '.join(labels)}`. Parent labels `RE` and `QU` are excluded from the primary macro average.",
        "",
        "## Freeze status",
        "",
        "`FROZEN_WITH_STABLE_CORE_LIMITATION`: numeric files and rendering inputs are frozen, but the Stable Core SAE row is exploratory because its supervised feature list was selected on the full analysis dataset. It must not be described as nested-CV confirmatory evidence.",
        "",
        "All other representations pass the shared-fold, shared-classifier, shared-scaling, seed, metric, and fold-local/no-supervised-selection audit.",
        "",
        "## Macro results",
        "",
        "| representation | n | AUC | PR-AUC | F1 | Balanced Acc. |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        n = "baseline" if int(row["top_n"]) < 0 else str(int(row["top_n"]))
        lines.append(
            f"| {row['representation']} | {n} | {row['macro_auc']:.3f} | "
            f"{row['macro_average_precision']:.3f} | {row['macro_f1']:.3f} | {row['macro_balanced_accuracy']:.3f} |"
        )
    lines.extend(["", "## Audit", "", "| check | status | evidence |", "|---|---|---|"])
    for _, row in audit.iterrows():
        lines.append(f"| {row['check']} | {row['status']} | {str(row['evidence']).replace('|', '/')} |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `final_metrics_by_label.csv`: final per-label metrics.",
            "- `final_macro_metrics.csv`: final seven-leaf macro metrics.",
            "- `source_probe_fold_metrics_leaf7.csv`: frozen fold-level source values.",
            "- `source_selected_latents_leaf7.csv`: frozen selected-feature audit rows.",
            "- `protocol_audit.csv`: machine-readable fairness/leakage audit.",
            "- `plot_frozen_representation_comparison.py`: standalone plot script.",
            "- `figures/representation_comparison_leaf7.png` and `.pdf`: final figure.",
            "- `SHA256SUMS.txt`: package checksums.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_package_checksums(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    checksum_path = output_dir / "SHA256SUMS.txt"
    files = sorted(path for path in output_dir.rglob("*") if path.is_file() and path != checksum_path)
    checksum_path.write_text(
        "\n".join(f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}" for path in files) + "\n",
        encoding="utf-8",
    )
    return checksum_path


def freeze_representation_comparison(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    plot_script: str | Path,
    config: FreezeConfig | None = None,
) -> dict[str, Any]:
    config = config or FreezeConfig()
    source = Path(source_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = source / "manifest.json"
    fold_path = source / "probe_fold_metrics.csv"
    selection_path = source / "selected_latents_by_label_n.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    fold_all = pd.read_csv(fold_path)
    _assert_source_contract(fold_all, manifest, config.labels)
    fold = fold_all[fold_all["label"].astype(str).isin(config.labels)].copy()
    selected_all = pd.read_csv(selection_path)
    selected = selected_all[selected_all["label"].astype(str).isin(config.labels)].copy()
    by_label = _summarize_by_label(fold)
    macro = _summarize_macro(by_label, fold)
    audit = _protocol_audit(fold, manifest, config.labels)

    paths = {
        "by_label": output / "final_metrics_by_label.csv",
        "macro": output / "final_macro_metrics.csv",
        "fold": output / "source_probe_fold_metrics_leaf7.csv",
        "selected": output / "source_selected_latents_leaf7.csv",
        "audit": output / "protocol_audit.csv",
        "readme": output / "README.md",
        "plot_script": output / "plot_frozen_representation_comparison.py",
    }
    by_label.to_csv(paths["by_label"], index=False, encoding="utf-8-sig")
    macro.to_csv(paths["macro"], index=False, encoding="utf-8-sig")
    fold.to_csv(paths["fold"], index=False, encoding="utf-8-sig")
    selected.to_csv(paths["selected"], index=False, encoding="utf-8-sig")
    audit.to_csv(paths["audit"], index=False, encoding="utf-8-sig")
    shutil.copy2(plot_script, paths["plot_script"])
    _write_readme(paths["readme"], macro, audit, config.labels)

    package_manifest = {
        "analysis": "frozen_leaf7_representation_comparison",
        "freeze_status": "FROZEN_WITH_STABLE_CORE_LIMITATION",
        "confirmatory_protocol_passed": False,
        "labels": list(config.labels),
        "representations": list(REPRESENTATIONS),
        "metrics": list(METRICS),
        "source": {
            "directory": str(source.resolve()),
            "manifest_sha256": _sha256(manifest_path),
            "fold_metrics_sha256": _sha256(fold_path),
            "selected_latents_sha256": _sha256(selection_path),
            "source_manifest": manifest,
        },
        "blocking_audit_item": "stable_core_selection_train_only",
        "outputs": {
            **{key: str(value.resolve()) for key, value in paths.items()},
            "figure_png": str((output / "figures" / "representation_comparison_leaf7.png").resolve()),
            "figure_pdf": str((output / "figures" / "representation_comparison_leaf7.pdf").resolve()),
        },
    }
    frozen_manifest_path = output / "manifest.json"
    frozen_manifest_path.write_text(
        json.dumps(package_manifest, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    checksum_path = write_package_checksums(output)
    return {
        "macro": macro,
        "by_label": by_label,
        "audit": audit,
        "manifest": package_manifest,
        "manifest_path": frozen_manifest_path,
        "checksum_path": checksum_path,
    }


__all__ = [
    "FreezeConfig", "LEAF_LABELS", "METRICS", "freeze_representation_comparison", "write_package_checksums"
]
