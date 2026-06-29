"""Run Llama/OpenMOSS cross-layer hidden-state probes on the project dataset.

The output schema intentionally matches the Gemma layer probe artifacts:
``layer_probe_metrics.csv`` contains one-vs-rest probe metrics for each
label/layer/pooling combination, and ``best_layers_by_label.csv`` stores the
best layer per label. These metrics can then be consumed by
``run_layer_selection_strategy.py``.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_LABELS = ["RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF"]
DEFAULT_OUTPUT_DIR = "outputs/llama_misc_layer_probe"


def _texts_from_records(records: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for idx, record in enumerate(records):
        text = str(record.get("text") or record.get("unit_text") or "").strip()
        if not text:
            sample_id = record.get("sample_id") or record.get("record_id") or idx
            raise ValueError(f"Record {sample_id!r} does not contain text/unit_text.")
        texts.append(text)
    return texts


def _labels_or_default(
    records: list[dict[str, Any]],
    *,
    labels: list[str] | None,
    include_other: bool,
) -> tuple[list[str], np.ndarray]:
    from nlp_re_base.misc_label_mapping import select_labels

    selected = labels
    if selected is None:
        selected = list(DEFAULT_LABELS)
        if include_other:
            selected.append("OTHER")
    elif not include_other:
        selected = [label for label in selected if label.strip().upper() != "OTHER"]
    ordered, indicators = select_labels(records, labels=selected)
    return ordered, indicators


def _load_project_records(
    *,
    data_dir: str | Path,
    data_format: str,
    confidence_threshold: float | None,
    limit_records: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from nlp_re_base.data import load_experiment_dataset

    dataset = load_experiment_dataset(
        data_dir,
        data_format=data_format,
        confidence_threshold=confidence_threshold,
        limit=limit_records,
    )
    return dataset.records, dataset.summary


def _is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _extract_with_oom_retry(
    *,
    model: Any,
    tokenizer: Any,
    texts: list[str],
    output_dir: str | Path,
    pooling: list[str],
    max_seq_len: int,
    batch_size: int,
    device: str | None,
    feature_dtype: str,
    overwrite_cache: bool,
):
    from nlp_re_base.layer_probe import extract_layer_features

    import torch

    current_batch_size = max(1, int(batch_size))
    last_exc: BaseException | None = None
    while current_batch_size >= 1:
        try:
            print(f"Extracting Llama layer features with batch_size={current_batch_size}...")
            return extract_layer_features(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                output_dir=output_dir,
                pooling=pooling,
                max_seq_len=max_seq_len,
                batch_size=current_batch_size,
                device=device,
                feature_dtype=feature_dtype,
                reuse_cache=not overwrite_cache,
            )
        except RuntimeError as exc:
            last_exc = exc
            if not _is_oom_error(exc) or current_batch_size == 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            print(f"OOM during Llama extraction; retrying with batch_size={current_batch_size}.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    assert last_exc is not None
    raise last_exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def run_llama_layer_probe_for_records(
    *,
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    output_dir: str | Path,
    model_source: str,
    labels: list[str] | None = None,
    include_other: bool = False,
    pooling: list[str] | None = None,
    cv_folds: int = 5,
    recognition_auc_threshold: float = 0.70,
    max_seq_len: int = 128,
    batch_size: int = 1,
    device: str | None = None,
    feature_dtype: str = "float16",
    overwrite_cache: bool = False,
    dataset_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from nlp_re_base.layer_probe import (
        evaluate_layer_probes,
        export_best_probe_artifacts,
        make_dataset_summary,
        write_probe_report,
    )

    if not records:
        raise RuntimeError("No records provided for Llama layer probing.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pool_values = list(pooling or ["mean", "last"])

    texts = _texts_from_records(records)
    label_names, label_indicators = _labels_or_default(
        records,
        labels=labels,
        include_other=include_other,
    )
    print(f"  records: {len(records)}")
    print(f"  labels: {', '.join(label_names)}")

    feature_store = _extract_with_oom_retry(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        output_dir=output_path,
        pooling=pool_values,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        device=device,
        feature_dtype=feature_dtype,
        overwrite_cache=overwrite_cache,
    )

    print("Running Llama one-vs-rest layer probes...")
    metrics_df, best_df = evaluate_layer_probes(
        feature_store=feature_store,
        label_indicators=label_indicators,
        labels=label_names,
        output_dir=output_path,
        cv_folds=cv_folds,
        recognition_auc_threshold=recognition_auc_threshold,
    )

    print("Exporting Llama best-layer probe artifacts...")
    artifacts = export_best_probe_artifacts(
        feature_store=feature_store,
        label_indicators=label_indicators,
        best_df=best_df,
        labels=label_names,
        output_dir=output_path,
    )

    summary = make_dataset_summary(
        records=records,
        labels=label_names,
        label_indicators=label_indicators,
        model_source=model_source,
        feature_store=feature_store,
        cv_folds=cv_folds,
        recognition_auc_threshold=recognition_auc_threshold,
    )
    if dataset_summary is not None:
        summary["dataset"] = dataset_summary
    summary.update(
        {
            "analysis": "llama_cross_layer_misc_probe",
            "metrics_path": str(output_path / "layer_probe_metrics.csv"),
            "best_layers_path": str(output_path / "best_layers_by_label.csv"),
            "report_path": str(output_path / "layer_probe_report.md"),
            "n_metric_rows": int(len(metrics_df)),
            "n_artifacts": int(len(artifacts)),
        }
    )
    _write_json(output_path / "dataset_summary.json", summary)
    report_path = write_probe_report(
        output_dir=output_path,
        best_df=best_df,
        model_source=model_source,
        recognition_auc_threshold=recognition_auc_threshold,
        report_title="Llama/OpenMOSS MISC Cross-Layer Probe Report",
    )
    summary["report_path"] = str(report_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Llama hidden activations for the project dataset and run cross-layer MISC probes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-config", default="config/model_config.json")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Optional local model directory. Overrides model_config.json and MODEL_DIR.",
    )
    parser.add_argument("--data-dir", default="data/mi_quality_counseling_misc")
    parser.add_argument(
        "--data-format",
        choices=["auto", "misc_full", "legacy_re_nonre", "cactus"],
        default="auto",
    )
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--include-other", action="store_true")
    parser.add_argument("--pooling", nargs="+", choices=["mean", "last"], default=["mean", "last"])
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--recognition-auc-threshold", type=float, default=0.70)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--feature-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    import torch

    from nlp_re_base.model import load_local_model_and_tokenizer

    args = parse_args()
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Llama/OpenMOSS MISC Cross-Layer Probe")
    print(f"  model_config: {args.model_config}")
    print(f"  model_dir: {args.model_dir or '<from config/MODEL_DIR>'}")
    print(f"  data_dir: {args.data_dir}")
    print(f"  output_dir: {output_dir}")

    records, dataset_summary = _load_project_records(
        data_dir=args.data_dir,
        data_format=args.data_format,
        confidence_threshold=args.confidence_threshold,
        limit_records=args.limit_records,
    )

    model, tokenizer, model_config = load_local_model_and_tokenizer(
        args.model_config,
        model_dir=args.model_dir,
        device=args.device,
    )
    model_source = str(model_config.get("model_path") or model_config.get("model_name") or "llama")

    summary = run_llama_layer_probe_for_records(
        model=model,
        tokenizer=tokenizer,
        records=records,
        output_dir=output_dir,
        model_source=model_source,
        labels=args.labels,
        include_other=args.include_other,
        pooling=list(args.pooling),
        cv_folds=args.cv_folds,
        recognition_auc_threshold=args.recognition_auc_threshold,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=args.device if str(model_config.get("device_map", "")).lower() in {"none", "null"} else None,
        feature_dtype=args.feature_dtype,
        overwrite_cache=args.overwrite_cache,
        dataset_summary=dataset_summary,
    )
    summary["runtime_seconds"] = round(time.time() - started, 3)
    _write_json(output_dir / "dataset_summary.json", summary)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nCompleted Llama cross-layer probe.")
    print(f"  metrics: {output_dir / 'layer_probe_metrics.csv'}")
    print(f"  best layers: {output_dir / 'best_layers_by_label.csv'}")
    print(f"  report: {output_dir / 'layer_probe_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
