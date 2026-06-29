"""Fix a Llama-first layer selection strategy for the MISC SAE project.

This script is intentionally read-only with respect to existing experiment
artifacts. It records the current Llama/OpenMOSS SAE layer constraint and uses
Gemma layer-wise probe metrics only as a control profile.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LABELS = ("RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF")
DEFAULT_NEAR_DELTA = 0.005
NOT_AVAILABLE = "not_available"


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def parse_hook_layer(hook_point: str) -> int:
    """Parse hook points such as blocks.19.hook_resid_post."""

    match = re.fullmatch(r"blocks\.(\d+)\.hook_resid_post", str(hook_point))
    if match is None:
        raise ValueError(
            f"Expected hook point like 'blocks.<N>.hook_resid_post', got: {hook_point!r}"
        )
    return int(match.group(1))


def _normalise_labels(labels: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalised = tuple(str(label).upper() for label in labels)
    if not normalised:
        raise ValueError("At least one label is required.")
    if len(set(normalised)) != len(normalised):
        raise ValueError(f"Duplicate labels are not allowed: {normalised}")
    return normalised


def _extract_llama_layer_info(
    *,
    sae_config_path: Path,
    feature_metadata_path: Path,
) -> dict[str, Any]:
    sae_config = _load_json(sae_config_path)
    feature_metadata = _load_json(feature_metadata_path) if feature_metadata_path.exists() else {}

    config_hook = sae_config.get("hook_point")
    metadata_hook = feature_metadata.get("hook_point")
    if config_hook and metadata_hook and str(config_hook) != str(metadata_hook):
        raise ValueError(
            "Llama SAE config hook point does not match feature metadata hook point: "
            f"{config_hook!r} vs {metadata_hook!r}"
        )

    hook_point = config_hook or metadata_hook
    if not hook_point:
        raise ValueError(
            "Could not find hook_point in either Llama SAE config or feature metadata."
        )

    canonical_layer = parse_hook_layer(str(hook_point))
    return {
        "model_scope": "llama_main",
        "global_canonical_layer": canonical_layer,
        "hook_point": str(hook_point),
        "sae_repo_id": sae_config.get("sae_repo_id"),
        "sae_subfolder": sae_config.get("sae_subfolder"),
        "canonical_reason": "sae_checkpoint_hook_point",
        "feature_metadata_available": bool(feature_metadata),
        "feature_shape": feature_metadata.get("feature_shape"),
        "activation_shape": feature_metadata.get("activation_shape"),
        "n_records": feature_metadata.get("n_records"),
    }


def build_llama_selection_rows(
    *,
    labels: tuple[str, ...],
    llama_info: dict[str, Any],
    layer_metrics: pd.DataFrame | None = None,
    near_delta: float = DEFAULT_NEAR_DELTA,
) -> list[dict[str, Any]]:
    if layer_metrics is not None and not layer_metrics.empty:
        return build_llama_selection_rows_from_metrics(
            labels=labels,
            llama_info=llama_info,
            metrics=layer_metrics,
            near_delta=near_delta,
        )

    rows: list[dict[str, Any]] = []
    for label in labels:
        rows.append(
            {
                "model_scope": "llama_main",
                "target_label": label,
                "global_canonical_layer": int(llama_info["global_canonical_layer"]),
                "canonical_layer": int(llama_info["global_canonical_layer"]),
                "hook_point": llama_info["hook_point"],
                "sae_repo_id": llama_info.get("sae_repo_id"),
                "sae_subfolder": llama_info.get("sae_subfolder"),
                "canonical_reason": "sae_checkpoint_hook_point",
                "label_specific_best_layer": NOT_AVAILABLE,
                "label_specific_best_pooling": NOT_AVAILABLE,
                "label_specific_best_auc": NOT_AVAILABLE,
                "label_specific_best_auc_std": NOT_AVAILABLE,
                "label_specific_best_f1_mean": NOT_AVAILABLE,
                "label_specific_best_balanced_accuracy_mean": NOT_AVAILABLE,
                "early_stable_layer": NOT_AVAILABLE,
                "early_stable_pooling": NOT_AVAILABLE,
                "early_stable_auc": NOT_AVAILABLE,
                "early_stable_delta_from_best": NOT_AVAILABLE,
                "n_near_optimal_layers": 0,
                "near_delta": float(near_delta),
                "near_optimal_layers": "",
                "availability_note": "requires_llama_cross_layer_probe_metrics",
            }
        )
    return rows


def _require_columns(df: pd.DataFrame, columns: set[str], path_label: str) -> None:
    missing = sorted(columns.difference(df.columns))
    if missing:
        raise ValueError(f"{path_label} is missing required columns: {missing}")


def _collapse_pooling_by_label_layer(metrics: pd.DataFrame, labels: tuple[str, ...]) -> pd.DataFrame:
    _require_columns(metrics, {"label", "layer_idx", "pooling", "auc_mean"}, "Layer metrics")
    out = metrics.copy()
    out["label"] = out["label"].astype(str).str.upper()
    out = out[out["label"].isin(set(labels))].copy()
    if out.empty:
        return out

    out["layer_idx"] = pd.to_numeric(out["layer_idx"], errors="coerce")
    out["auc_mean"] = pd.to_numeric(out["auc_mean"], errors="coerce")
    out = out.dropna(subset=["layer_idx", "auc_mean"]).copy()
    out["layer_idx"] = out["layer_idx"].astype(int)

    for column in (
        "auc_std",
        "accuracy_mean",
        "f1_mean",
        "balanced_accuracy_mean",
        "n_positive",
        "n_negative",
        "n_folds",
    ):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        else:
            out[column] = pd.NA
    if "recognized" not in out.columns:
        out["recognized"] = pd.NA

    pool_rank = {"mean": 0, "last": 1}
    out["_pool_rank"] = out["pooling"].map(pool_rank).fillna(99).astype(int)
    out = out.sort_values(
        ["label", "layer_idx", "auc_mean", "_pool_rank"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    out = out.drop_duplicates(["label", "layer_idx"], keep="first").copy()
    return out.drop(columns=["_pool_rank"]).reset_index(drop=True)


def _build_metric_selection_rows(
    *,
    metrics: pd.DataFrame,
    labels: tuple[str, ...],
    near_delta: float,
    model_scope: str,
    missing_note: str,
    available_note: str,
    field_prefix: str,
) -> list[dict[str, Any]]:
    if near_delta < 0:
        raise ValueError("--near-delta must be non-negative.")

    collapsed = _collapse_pooling_by_label_layer(metrics, labels)
    rows: list[dict[str, Any]] = []
    tolerance = 1e-12

    for label in labels:
        label_rows = collapsed[collapsed["label"] == label].copy()
        if label_rows.empty:
            rows.append(
                {
                    "model_scope": model_scope,
                    "target_label": label,
                    f"{field_prefix}best_layer": NOT_AVAILABLE,
                    f"{field_prefix}best_pooling": NOT_AVAILABLE,
                    f"{field_prefix}best_auc": NOT_AVAILABLE,
                    f"{field_prefix}best_auc_std": NOT_AVAILABLE,
                    f"{field_prefix}best_f1_mean": NOT_AVAILABLE,
                    f"{field_prefix}best_balanced_accuracy_mean": NOT_AVAILABLE,
                    f"{field_prefix}best_recognized": NOT_AVAILABLE,
                    f"{field_prefix}earliest_near_optimal_layer": NOT_AVAILABLE,
                    f"{field_prefix}earliest_near_optimal_pooling": NOT_AVAILABLE,
                    f"{field_prefix}earliest_near_optimal_auc": NOT_AVAILABLE,
                    f"{field_prefix}earliest_delta_from_best": NOT_AVAILABLE,
                    "n_near_optimal_layers": 0,
                    "near_delta": float(near_delta),
                    "near_optimal_layers": "",
                    "availability_note": missing_note,
                }
            )
            continue

        ranked = label_rows.sort_values(
            ["auc_mean", "layer_idx"],
            ascending=[False, True],
            kind="mergesort",
        ).copy()
        best = ranked.iloc[0]
        best_auc = float(best["auc_mean"])
        ranked["delta_from_best"] = best_auc - ranked["auc_mean"].astype(float)
        near = ranked[ranked["delta_from_best"] <= float(near_delta) + tolerance].copy()
        near = near.sort_values(["layer_idx", "delta_from_best"], kind="mergesort")
        earliest = near.iloc[0]

        rows.append(
            {
                "model_scope": model_scope,
                "target_label": label,
                f"{field_prefix}best_layer": int(best["layer_idx"]),
                f"{field_prefix}best_pooling": str(best["pooling"]),
                f"{field_prefix}best_auc": best_auc,
                f"{field_prefix}best_auc_std": best["auc_std"],
                f"{field_prefix}best_f1_mean": best["f1_mean"],
                f"{field_prefix}best_balanced_accuracy_mean": best["balanced_accuracy_mean"],
                f"{field_prefix}best_recognized": best["recognized"],
                f"{field_prefix}earliest_near_optimal_layer": int(earliest["layer_idx"]),
                f"{field_prefix}earliest_near_optimal_pooling": str(earliest["pooling"]),
                f"{field_prefix}earliest_near_optimal_auc": float(earliest["auc_mean"]),
                f"{field_prefix}earliest_delta_from_best": float(earliest["delta_from_best"]),
                "n_near_optimal_layers": int(len(near)),
                "near_delta": float(near_delta),
                "near_optimal_layers": ",".join(str(int(layer)) for layer in near["layer_idx"].tolist()),
                "availability_note": available_note,
            }
        )
    return rows


def build_llama_selection_rows_from_metrics(
    *,
    labels: tuple[str, ...],
    llama_info: dict[str, Any],
    metrics: pd.DataFrame,
    near_delta: float,
) -> list[dict[str, Any]]:
    metric_rows = _build_metric_selection_rows(
        metrics=metrics,
        labels=labels,
        near_delta=near_delta,
        model_scope="llama_main",
        missing_note="missing_llama_layer_metrics_for_label",
        available_note="llama_cross_layer_probe_metrics_available",
        field_prefix="label_specific_",
    )
    rows: list[dict[str, Any]] = []
    for row in metric_rows:
        rows.append(
            {
                "model_scope": "llama_main",
                "target_label": row["target_label"],
                "global_canonical_layer": int(llama_info["global_canonical_layer"]),
                "canonical_layer": int(llama_info["global_canonical_layer"]),
                "hook_point": llama_info["hook_point"],
                "sae_repo_id": llama_info.get("sae_repo_id"),
                "sae_subfolder": llama_info.get("sae_subfolder"),
                "canonical_reason": "sae_checkpoint_hook_point",
                "label_specific_best_layer": row["label_specific_best_layer"],
                "label_specific_best_pooling": row["label_specific_best_pooling"],
                "label_specific_best_auc": row["label_specific_best_auc"],
                "label_specific_best_auc_std": row["label_specific_best_auc_std"],
                "label_specific_best_f1_mean": row["label_specific_best_f1_mean"],
                "label_specific_best_balanced_accuracy_mean": row[
                    "label_specific_best_balanced_accuracy_mean"
                ],
                "label_specific_best_recognized": row["label_specific_best_recognized"],
                "early_stable_layer": row["label_specific_earliest_near_optimal_layer"],
                "early_stable_pooling": row["label_specific_earliest_near_optimal_pooling"],
                "early_stable_auc": row["label_specific_earliest_near_optimal_auc"],
                "early_stable_delta_from_best": row[
                    "label_specific_earliest_delta_from_best"
                ],
                "n_near_optimal_layers": row["n_near_optimal_layers"],
                "near_delta": row["near_delta"],
                "near_optimal_layers": row["near_optimal_layers"],
                "availability_note": row["availability_note"],
            }
        )
    return rows


def build_gemma_control_rows(
    *,
    metrics: pd.DataFrame,
    labels: tuple[str, ...],
    near_delta: float,
) -> list[dict[str, Any]]:
    return _build_metric_selection_rows(
        metrics=metrics,
        labels=labels,
        near_delta=near_delta,
        model_scope="gemma_control",
        missing_note="missing_gemma_layer_metrics_for_label",
        available_note="gemma_control_only_not_llama_selection",
        field_prefix="",
    )


def _fmt_float(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        escaped = [str(item).replace("\n", " ").replace("|", "\\|") for item in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return lines


def write_layer_selection_report(
    *,
    path: Path,
    labels: tuple[str, ...],
    near_delta: float,
    llama_info: dict[str, Any],
    llama_rows: list[dict[str, Any]],
    gemma_rows: list[dict[str, Any]],
    llama_layer_metrics_path: Path | None,
    llama_layer_metrics_available: bool,
) -> None:
    lines: list[str] = []
    lines.append("# Llama-First MISC 层选择策略")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append(
        f"- Llama / OpenMOSS 主线 canonical layer 固定为 layer {llama_info['global_canonical_layer']}。"
    )
    lines.append(
        f"- 固定依据是当前 SAE checkpoint 的 hook point：`{llama_info['hook_point']}`。"
    )
    lines.append(
        "- 这不是 Llama 全层搜索后的最优层结论，而是当前 SAE artifact 的层绑定约束。"
    )
    lines.append("- Gemma 仅为对照，不能决定 Llama 主层。")
    if llama_layer_metrics_available:
        lines.append(
            "- 已接入 Llama cross-layer probe metrics；逐标签 best layer 和 earliest near-optimal layer 作为辅助定位结果报告。"
        )
    else:
        lines.append(
            "- Llama 的 label-specific best layer 和 early stable layer 当前标记为 `not_available`，需要新增 Llama cross-layer probe 后才能计算。"
        )
    if llama_layer_metrics_path is not None:
        lines.append(f"- Llama layer metrics path: `{llama_layer_metrics_path}`。")
    lines.append("")
    lines.append("## Llama 主线固定结果")
    lines.append("")
    lines.extend(
        _markdown_table(
            [
                "Label",
                "Canonical layer",
                "Hook point",
                "Label-specific best",
                "Best AUC",
                "Early stable",
                "Early AUC",
                "Note",
            ],
            [
                [
                    row["target_label"],
                    row["canonical_layer"],
                    f"`{row['hook_point']}`",
                    row["label_specific_best_layer"],
                    _fmt_float(row["label_specific_best_auc"]),
                    row["early_stable_layer"],
                    _fmt_float(row["early_stable_auc"]),
                    row["availability_note"],
                ]
                for row in llama_rows
            ],
        )
    )
    lines.append("")
    lines.append("## Gemma 对照层分布")
    lines.append("")
    lines.append(
        f"Near-optimal 判定：同一标签内 `best_auc - auc <= {near_delta}`。同一 `(label, layer)` 如有多个 pooling，先保留 AUC 更高者。"
    )
    lines.append("")
    lines.extend(
        _markdown_table(
            [
                "Label",
                "Best layer",
                "Best pooling",
                "Best AUC",
                "Earliest near-optimal layer",
                "Earliest AUC",
                "Delta",
                "N near",
                "Near layers",
            ],
            [
                [
                    row["target_label"],
                    row["best_layer"],
                    row["best_pooling"],
                    _fmt_float(row["best_auc"]),
                    row["earliest_near_optimal_layer"],
                    _fmt_float(row["earliest_near_optimal_auc"]),
                    _fmt_float(row["earliest_delta_from_best"]),
                    row["n_near_optimal_layers"],
                    row["near_optimal_layers"],
                ]
                for row in gemma_rows
            ],
        )
    )
    lines.append("")
    lines.append("## 使用边界")
    lines.append("")
    lines.append("- 当前脚本只固定层选择策略，不修改 SAE、probe、top-latent 或报告生成链路。")
    lines.append("- 即使接入 Llama cross-layer probe，主实验 canonical layer 仍固定为当前 SAE checkpoint 所在的 layer 19。")
    lines.append("- Gemma 的 best / early-stable 结果只能说明 Gemma 控制模型的层间可解码性分布。")
    lines.append("- 若 Llama metrics 尚未生成，需要先运行 Llama cross-layer hidden-state probe；若已生成，本报告只把它作为辅助层定位，不覆盖 SAE 主线层。")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_layer_selection_strategy(
    *,
    llama_sae_config_path: str | Path,
    llama_feature_metadata_path: str | Path,
    gemma_layer_metrics_path: str | Path,
    output_dir: str | Path,
    llama_layer_metrics_path: str | Path | None = None,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    near_delta: float = DEFAULT_NEAR_DELTA,
) -> dict[str, Any]:
    label_tuple = _normalise_labels(labels)
    sae_config_path = _resolve_path(llama_sae_config_path)
    feature_metadata_path = _resolve_path(llama_feature_metadata_path)
    gemma_metrics_path = _resolve_path(gemma_layer_metrics_path)
    llama_metrics_path = (
        _resolve_path(llama_layer_metrics_path)
        if llama_layer_metrics_path is not None
        else None
    )
    output_path = _resolve_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    llama_info = _extract_llama_layer_info(
        sae_config_path=sae_config_path,
        feature_metadata_path=feature_metadata_path,
    )
    llama_metrics_available = bool(llama_metrics_path and llama_metrics_path.exists())
    llama_metrics = pd.read_csv(llama_metrics_path) if llama_metrics_available else None
    llama_rows = build_llama_selection_rows(
        labels=label_tuple,
        llama_info=llama_info,
        layer_metrics=llama_metrics,
        near_delta=float(near_delta),
    )

    gemma_metrics = pd.read_csv(gemma_metrics_path)
    gemma_rows = build_gemma_control_rows(
        metrics=gemma_metrics,
        labels=label_tuple,
        near_delta=float(near_delta),
    )

    paths = {
        "strategy_json": output_path / "layer_selection_strategy.json",
        "llama_csv": output_path / "llama_layer_selection.csv",
        "gemma_csv": output_path / "gemma_control_layer_selection.csv",
        "report_md": output_path / "layer_selection_strategy_report.md",
    }

    pd.DataFrame(llama_rows).to_csv(paths["llama_csv"], index=False)
    pd.DataFrame(gemma_rows).to_csv(paths["gemma_csv"], index=False)
    write_layer_selection_report(
        path=paths["report_md"],
        labels=label_tuple,
        near_delta=float(near_delta),
        llama_info=llama_info,
        llama_rows=llama_rows,
        gemma_rows=gemma_rows,
        llama_layer_metrics_path=llama_metrics_path,
        llama_layer_metrics_available=llama_metrics_available,
    )

    strategy = {
        "analysis": "llama_first_misc_layer_selection_strategy",
        "labels": list(label_tuple),
        "near_delta": float(near_delta),
        "llama_main": {
            **llama_info,
            "layer_metrics_path": str(llama_metrics_path) if llama_metrics_path is not None else None,
            "layer_metrics_available": llama_metrics_available,
            "selection_rows": llama_rows,
            "label_specific_best_status": (
                "computed_from_llama_cross_layer_probe_metrics"
                if llama_metrics_available
                else "not_computed_requires_llama_cross_layer_probe_metrics"
            ),
            "early_stable_status": (
                "computed_from_llama_cross_layer_probe_metrics"
                if llama_metrics_available
                else "not_computed_requires_llama_cross_layer_probe_metrics"
            ),
        },
        "gemma_control": {
            "metrics_path": str(gemma_metrics_path),
            "selection_rows": gemma_rows,
            "interpretation_boundary": "control_only_not_used_to_choose_llama_layer",
        },
        "inputs": {
            "llama_sae_config": str(sae_config_path),
            "llama_feature_metadata": str(feature_metadata_path),
            "llama_layer_metrics": str(llama_metrics_path) if llama_metrics_path is not None else None,
            "gemma_layer_metrics": str(gemma_metrics_path),
        },
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["strategy_json"].write_text(
        json.dumps(_jsonable(strategy), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix the Llama-first layer selection strategy and summarize Gemma control layers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--llama-sae-config", default="config/sae_config.json")
    parser.add_argument(
        "--llama-feature-metadata",
        default="outputs/misc_full_sae_eval/feature_store/feature_metadata.json",
    )
    parser.add_argument(
        "--gemma-layer-metrics",
        default="outputs/gemma3_misc_layer_probe/layer_probe_metrics.csv",
    )
    parser.add_argument(
        "--llama-layer-metrics",
        default="outputs/llama_misc_layer_probe/layer_probe_metrics.csv",
        help=(
            "Optional Llama cross-layer metrics. If the file exists, label-specific "
            "best and earliest near-optimal layers are computed from it; otherwise "
            "they remain not_available."
        ),
    )
    parser.add_argument("--output-dir", default="outputs/layer_selection_strategy")
    parser.add_argument("--near-delta", type=float, default=DEFAULT_NEAR_DELTA)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strategy = run_layer_selection_strategy(
        llama_sae_config_path=args.llama_sae_config,
        llama_feature_metadata_path=args.llama_feature_metadata,
        gemma_layer_metrics_path=args.gemma_layer_metrics,
        output_dir=args.output_dir,
        llama_layer_metrics_path=args.llama_layer_metrics,
        labels=tuple(args.labels),
        near_delta=args.near_delta,
    )
    print("Completed Llama-first layer selection strategy export.")
    print(f"Output dir: {args.output_dir}")
    print(f"Llama canonical layer: {strategy['llama_main']['global_canonical_layer']}")
    print(f"Gemma control labels: {len(strategy['gemma_control']['selection_rows'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
