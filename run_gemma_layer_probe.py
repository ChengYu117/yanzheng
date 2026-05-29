"""Run Gemma3-4B layer-wise probes for MISC or RE/NonRE layer selection.

The default model source is the Hugging Face Gemma 3 4B base model. A local
model directory can still be supplied with --model-dir for offline reruns.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.layer_probe import (  # noqa: E402
    DEFAULT_CORE_LABELS,
    DEFAULT_POOLING,
    evaluate_layer_probes,
    export_best_probe_artifacts,
    extract_layer_features,
    make_dataset_summary,
    write_json,
    write_probe_report,
)
from nlp_re_base.data import load_experiment_dataset  # noqa: E402
from nlp_re_base.misc_label_mapping import (  # noqa: E402
    load_misc_annotation_records,
    select_labels,
)


DEFAULT_HF_MODEL_ID = "google/gemma-3-4b-pt"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_EVALUATION_LABELS = [
    label for label in DEFAULT_CORE_LABELS if label != "OTHER"
]
DEFAULT_MISC_OUTPUT_DIR = "outputs/gemma3_misc_layer_probe"
DEFAULT_RE_NONRE_OUTPUT_DIR = "outputs/gemma3_re_nonre_layer_probe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemma3-4B MISC concept layer probe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_HF_MODEL_ID,
        help="Hugging Face model id used when --model-dir is not supplied.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Optional local model directory. Overrides --model-id.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/mi_quality_counseling_misc",
        help=(
            "Dataset root. For --probe-dataset=misc this is the MISC root or "
            "misc_annotations directory; for re_nonre this is the RE/NonRE root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for feature cache, metrics, artifacts, and report.",
    )
    parser.add_argument(
        "--probe-dataset",
        choices=["misc", "re_nonre"],
        default="misc",
        help=(
            "Dataset used only for best-layer screening. Downstream MISC "
            "analysis remains controlled by its own scripts and data directories."
        ),
    )
    parser.add_argument(
        "--no-balance-re-nonre",
        action="store_true",
        help="Do not downsample RE/NonRE records to equal class counts.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help=(
            "Labels to probe. Defaults to the 9 core MISC labels, excluding OTHER."
        ),
    )
    parser.add_argument(
        "--include-other",
        action="store_true",
        help="Include OTHER in evaluation. By default OTHER is excluded.",
    )
    parser.add_argument(
        "--pooling",
        nargs="+",
        choices=DEFAULT_POOLING,
        default=DEFAULT_POOLING,
        help="Utterance pooling methods over token hidden states.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--recognition-auc-threshold", type=float, default=0.70)
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument(
        "--torch-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model loading dtype.",
    )
    parser.add_argument(
        "--feature-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Cached feature dtype.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device used when --device-map=none. Defaults to cuda if available.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map. Use 'none' to load on a single device.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help=(
            "Environment variable containing a Hugging Face token for gated "
            "models. HUGGING_FACE_HUB_TOKEN is also checked as a fallback."
        ),
    )
    parser.add_argument(
        "--ignore-hf-token-env",
        action="store_true",
        help="Ignore HF_TOKEN/HUGGING_FACE_HUB_TOKEN and use cached hf auth login token.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=DEFAULT_HF_ENDPOINT,
        help=(
            "Hugging Face endpoint. Defaults to hf-mirror.com for faster "
            "downloads in China. Use an empty string to keep the environment default."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Passed through to Transformers model/tokenizer loaders.",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Re-extract hidden-state features even if the cache looks reusable.",
    )
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _model_source(args: argparse.Namespace) -> str:
    return args.model_dir or args.model_id


def _configure_hf_endpoint(args: argparse.Namespace) -> None:
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        os.environ["HF_HUB_ENDPOINT"] = args.hf_endpoint


def _get_cached_hf_token(*, ignore_env: bool) -> str | None:
    removed: dict[str, str] = {}
    env_names = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    if ignore_env:
        for env_name in env_names:
            if env_name in os.environ:
                removed[env_name] = os.environ.pop(env_name)
    try:
        from huggingface_hub import HfFolder

        token = HfFolder.get_token()
    except Exception:
        token = None
    finally:
        os.environ.update(removed)
    return token


def _hf_auth_kwargs(args: argparse.Namespace) -> dict[str, str]:
    if not args.ignore_hf_token_env:
        for env_name in (args.hf_token_env, "HUGGING_FACE_HUB_TOKEN"):
            token = os.environ.get(env_name)
            if token:
                return {"token": token}

    token = _get_cached_hf_token(ignore_env=args.ignore_hf_token_env)
    if token:
        return {"token": token}
    return {}


def _preflight_hf_access(source: str, args: argparse.Namespace) -> None:
    if args.model_dir:
        return
    try:
        from huggingface_hub import HfApi
        from requests import HTTPError
    except Exception:
        return

    auth_kwargs = _hf_auth_kwargs(args)
    if "token" in auth_kwargs:
        try:
            HfApi(endpoint="https://huggingface.co").whoami(token=auth_kwargs["token"])
        except HTTPError as exc:
            raise RuntimeError(
                "The Hugging Face token available to this Python environment is "
                "invalid. Set a fresh token in "
                f"{args.hf_token_env}, or run `hf auth login` in the active env."
            ) from exc
    try:
        api = HfApi(endpoint=args.hf_endpoint or None)
        api.model_info(source, **auth_kwargs)
    except HTTPError as exc:
        raise RuntimeError(
            "Hugging Face access check failed. If this is a gated Gemma model, "
            "accept the model license on Hugging Face and set a valid token in "
            f"{args.hf_token_env}, or run `hf auth login` in the active env."
        ) from exc
    except Exception:
        return


def _load_tokenizer(source: str, args: argparse.Namespace) -> Any:
    from transformers import AutoProcessor, AutoTokenizer

    load_kwargs: dict[str, Any] = {
        "cache_dir": args.cache_dir,
        "trust_remote_code": args.trust_remote_code,
    }
    load_kwargs.update(_hf_auth_kwargs(args))
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, **load_kwargs)
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(source, use_fast=False, **load_kwargs)
        except Exception:
            processor = AutoProcessor.from_pretrained(source, **load_kwargs)
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError(
                    "Could not load a tokenizer from AutoTokenizer or AutoProcessor."
                )

    if getattr(tokenizer, "pad_token", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        else:
            tokenizer.pad_token = eos_token
    return tokenizer


def _load_model(source: str, args: argparse.Namespace) -> torch.nn.Module:
    import transformers

    dtype = _dtype_from_name(args.torch_dtype)
    base_kwargs: dict[str, Any] = {
        "cache_dir": args.cache_dir,
        "trust_remote_code": args.trust_remote_code,
    }
    base_kwargs.update(_hf_auth_kwargs(args))
    if args.device_map.lower() != "none":
        base_kwargs["device_map"] = args.device_map

    model_classes: list[type[Any]] = []
    for class_name in (
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ):
        cls = getattr(transformers, class_name, None)
        if cls is not None:
            model_classes.append(cls)

    errors: list[str] = []
    for model_cls in model_classes:
        for dtype_key in ("dtype", "torch_dtype"):
            load_kwargs = dict(base_kwargs)
            load_kwargs[dtype_key] = dtype
            try:
                model = model_cls.from_pretrained(source, **load_kwargs)
                break
            except TypeError as exc:
                errors.append(f"{model_cls.__name__}({dtype_key}): {exc}")
                continue
            except Exception as exc:
                errors.append(f"{model_cls.__name__}({dtype_key}): {exc}")
                model = None
                break
        else:
            model = None
        if model is not None:
            break
    else:
        joined = "\n".join(errors[-8:])
        raise RuntimeError(f"Could not load model from {source!r}.\n{joined}")

    if args.device_map.lower() == "none":
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    model.eval()
    return model


def _texts_from_records(records: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for record in records:
        text = str(record.get("text") or record.get("unit_text") or "").strip()
        if not text:
            sample_id = record.get("sample_id") or record.get("record_id") or "<unknown>"
            raise ValueError(f"Record {sample_id} does not contain text/unit_text.")
        texts.append(text)
    return texts


def _balanced_re_nonre_records(
    records: list[dict[str, Any]],
    *,
    limit: int | None,
) -> list[dict[str, Any]]:
    re_records = [record for record in records if int(record.get("label_re", 0)) == 1]
    nonre_records = [record for record in records if int(record.get("label_re", 0)) == 0]
    if not re_records or not nonre_records:
        raise RuntimeError(
            f"RE/NonRE screening requires both classes; got "
            f"RE={len(re_records)}, NonRE={len(nonre_records)}."
        )

    per_class = min(len(re_records), len(nonre_records))
    if limit is not None:
        per_class = min(per_class, max(1, int(limit) // 2))

    balanced: list[dict[str, Any]] = []
    for idx in range(per_class):
        balanced.append(re_records[idx])
        balanced.append(nonre_records[idx])
    return balanced


def _load_probe_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    if args.probe_dataset == "misc":
        records = load_misc_annotation_records(
            args.data_dir,
            limit=args.limit_records,
        )
        if not records:
            raise RuntimeError("No MISC annotation records loaded.")
        return records, "misc"

    dataset = load_experiment_dataset(
        args.data_dir,
        data_format="auto",
        limit=None if not args.no_balance_re_nonre else args.limit_records,
    )
    records = dataset.records
    if not args.no_balance_re_nonre:
        records = _balanced_re_nonre_records(records, limit=args.limit_records)
    if not records:
        raise RuntimeError("No RE/NonRE records loaded.")
    return records, dataset.summary["data_format"]


def _labels_or_default(records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[str], Any]:
    if args.probe_dataset == "re_nonre":
        labels = ["RE"]
        indicators = np.array(
            [[bool(int(record.get("label_re", 0)))] for record in records],
            dtype=bool,
        )
        return labels, indicators

    labels_arg = args.labels
    if labels_arg is None:
        labels_arg = DEFAULT_CORE_LABELS if args.include_other else DEFAULT_EVALUATION_LABELS
    elif not args.include_other:
        labels_arg = [label for label in labels_arg if label.strip().upper() != "OTHER"]
    labels, indicators = select_labels(records, labels=labels_arg)
    return labels, indicators


def _is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _extract_with_oom_retry(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    output_dir: str | Path,
    args: argparse.Namespace,
):
    batch_size = max(1, int(args.batch_size))
    last_exc: BaseException | None = None
    while batch_size >= 1:
        try:
            print(f"Extracting features with batch_size={batch_size}...")
            return extract_layer_features(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                output_dir=output_dir,
                pooling=list(args.pooling),
                max_seq_len=args.max_seq_len,
                batch_size=batch_size,
                device=args.device if args.device_map.lower() == "none" else None,
                feature_dtype=args.feature_dtype,
                reuse_cache=not args.overwrite_cache,
            )
        except RuntimeError as exc:
            last_exc = exc
            if not _is_oom_error(exc) or batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(f"OOM during extraction; retrying with batch_size={batch_size}.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    assert last_exc is not None
    raise last_exc


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.output_dir is None:
        args.output_dir = (
            DEFAULT_RE_NONRE_OUTPUT_DIR
            if args.probe_dataset == "re_nonre"
            else DEFAULT_MISC_OUTPUT_DIR
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_source = _model_source(args)
    _configure_hf_endpoint(args)
    print("Gemma3 Layer Probe")
    print(f"  model_source: {model_source}")
    if args.hf_endpoint:
        print(f"  hf_endpoint: {args.hf_endpoint}")
    print(f"  probe_dataset: {args.probe_dataset}")
    print(f"  data_dir: {args.data_dir}")
    print(f"  output_dir: {output_dir}")

    records, resolved_data_format = _load_probe_records(args)
    texts = _texts_from_records(records)
    labels, label_indicators = _labels_or_default(records, args)

    print(f"  records: {len(records)}")
    print(f"  labels: {', '.join(labels)}")

    _preflight_hf_access(model_source, args)

    print("Loading tokenizer...")
    tokenizer = _load_tokenizer(model_source, args)
    print("Loading Gemma model...")
    model = _load_model(model_source, args)

    feature_store = _extract_with_oom_retry(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        output_dir=output_dir,
        args=args,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Running one-vs-rest layer probes...")
    metrics_df, best_df = evaluate_layer_probes(
        feature_store=feature_store,
        label_indicators=label_indicators,
        labels=labels,
        output_dir=output_dir,
        cv_folds=args.cv_folds,
        recognition_auc_threshold=args.recognition_auc_threshold,
    )

    print("Exporting best-layer probe artifacts...")
    artifacts = export_best_probe_artifacts(
        feature_store=feature_store,
        label_indicators=label_indicators,
        best_df=best_df,
        labels=labels,
        output_dir=output_dir,
    )

    summary = make_dataset_summary(
        records=records,
        labels=labels,
        label_indicators=label_indicators,
        model_source=model_source,
        feature_store=feature_store,
        cv_folds=args.cv_folds,
        recognition_auc_threshold=args.recognition_auc_threshold,
    )
    summary.update(
        {
            "runtime_seconds": round(time.time() - started, 3),
            "probe_dataset": args.probe_dataset,
            "resolved_data_format": resolved_data_format,
            "balanced_re_nonre": bool(
                args.probe_dataset == "re_nonre" and not args.no_balance_re_nonre
            ),
            "metrics_path": str(output_dir / "layer_probe_metrics.csv"),
            "best_layers_path": str(output_dir / "best_layers_by_label.csv"),
            "n_metric_rows": int(len(metrics_df)),
            "n_artifacts": int(len(artifacts)),
        }
    )
    write_json(output_dir / "dataset_summary.json", summary)
    report_path = write_probe_report(
        output_dir=output_dir,
        best_df=best_df,
        model_source=model_source,
        recognition_auc_threshold=args.recognition_auc_threshold,
        report_title=(
            "Gemma3 RE/NonRE Layer Probe Report"
            if args.probe_dataset == "re_nonre"
            else "Gemma3 MISC Layer Probe Report"
        ),
    )

    recognized = int(best_df["recognized"].sum()) if not best_df.empty else 0
    print("\nCompleted Gemma3 layer probe.")
    print(f"  recognized labels: {recognized}/{len(best_df)}")
    print(f"  metrics: {output_dir / 'layer_probe_metrics.csv'}")
    print(f"  best layers: {output_dir / 'best_layers_by_label.csv'}")
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
