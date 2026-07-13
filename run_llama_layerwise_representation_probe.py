"""Compare PCA-100 and SAE Top-100 across matched Llama/OpenMOSS layers.

The default run evaluates Llama layers 15 and 24.  It uses the same MISC label
matrix and grouped five-fold linear-probe protocol as the filtered-pool layer-19
comparison, but each layer receives its own matching OpenMOSS SAE checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.activations import extract_and_process_streaming
from nlp_re_base.layerwise_representation_probe import (
    DEFAULT_LABELS,
    LayerMatrices,
    LayerwiseRepresentationProbeConfig,
    run_layerwise_representation_probe,
)
from nlp_re_base.misc_label_mapping import FeatureFilterConfig, build_feature_filter, summarize_feature_filter
from nlp_re_base.model import load_local_model_and_tokenizer
from nlp_re_base.sae import load_sae_from_hub


DEFAULT_OUTPUT_DIR = "outputs/misc_full_sae_eval/interpretability/llama_layer15_24_pca100_sae_top100"
DEFAULT_SAE_REPO = "OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x"


def _parse_layers(values: list[str]) -> tuple[int, ...]:
    layers = tuple(sorted({int(value) for value in values}))
    if not layers or min(layers) < 0:
        raise ValueError("--layers must contain non-negative layer indices")
    return layers


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_no} is not a JSON object")
            records.append(value)
    if not records:
        raise ValueError(f"No records loaded from {path}")
    return records


def _texts_and_digest(records: list[dict[str, Any]]) -> tuple[list[str], str]:
    digest = hashlib.sha256()
    texts: list[str] = []
    for idx, record in enumerate(records):
        text = str(record.get("unit_text") or record.get("text") or "").strip()
        if not text:
            raise ValueError(f"record {idx} has no unit_text/text")
        record_id = str(record.get("record_id") or idx)
        digest.update(record_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")
        texts.append(text)
    return texts, digest.hexdigest()


def _validate_alignment(records: list[dict[str, Any]], label_df: pd.DataFrame) -> None:
    if len(records) != len(label_df):
        raise ValueError(f"records/label rows differ: {len(records)} vs {len(label_df)}")
    if "record_id" in label_df.columns:
        record_ids = [str(record.get("record_id") or idx) for idx, record in enumerate(records)]
        label_ids = label_df["record_id"].fillna("").astype(str).tolist()
        if record_ids != label_ids:
            raise ValueError("records.jsonl and label_matrix.csv record_id order differs")
    if "unit_text" in label_df.columns:
        record_texts = [str(record.get("unit_text") or record.get("text") or "").strip() for record in records]
        label_texts = label_df["unit_text"].fillna("").astype(str).str.strip().tolist()
        if record_texts != label_texts:
            raise ValueError("records.jsonl and label_matrix.csv unit_text order differs")


def _feature_paths(layer_dir: Path) -> dict[str, Path]:
    feature_dir = layer_dir / "feature_store"
    return {
        "features": feature_dir / "utterance_features.pt",
        "activations": feature_dir / "utterance_activations.pt",
        "metadata": feature_dir / "feature_metadata.json",
    }


def _load_cached_layer(
    *,
    layer_dir: Path,
    layer_idx: int,
    n_records: int,
    records_digest: str,
    max_seq_len: int,
    sae_subfolder: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    paths = _feature_paths(layer_dir)
    if not all(path.exists() for path in paths.values()):
        return None
    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if (
            int(metadata.get("layer_idx", -1)) != int(layer_idx)
            or int(metadata.get("n_records", -1)) != int(n_records)
            or str(metadata.get("records_digest", "")) != records_digest
            or str(metadata.get("aggregation", "")) != "max"
            or int(metadata.get("max_seq_len", -1)) != int(max_seq_len)
            or str(metadata.get("sae_subfolder", "")) != sae_subfolder
        ):
            return None
        feature_payload = torch.load(paths["features"], map_location="cpu")
        activation_payload = torch.load(paths["activations"], map_location="cpu")
        features = feature_payload["utterance_features"]
        activations = activation_payload["utterance_activations"]
        features_np = np.asarray(features.detach().cpu().float().numpy(), dtype=np.float32)
        activations_np = np.asarray(activations.detach().cpu().float().numpy(), dtype=np.float32)
        if features_np.shape[0] != n_records or activations_np.shape[0] != n_records:
            return None
        return features_np, activations_np
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _extract_layer_feature_store(
    *,
    model: Any,
    tokenizer: Any,
    sae: Any,
    texts: list[str],
    layer_idx: int,
    layer_dir: Path,
    records_digest: str,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
    sae_repo_id: str,
    sae_subfolder: str,
) -> tuple[np.ndarray, np.ndarray]:
    result = extract_and_process_streaming(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        texts=texts,
        hook_point=f"blocks.{int(layer_idx)}.hook_resid_post",
        max_seq_len=int(max_seq_len),
        batch_size=int(batch_size),
        aggregation="max",
        device=device,
        collect_structural_samples=0,
    )
    features = result["utterance_features"].detach().cpu().float()
    activations = result["utterance_activations"].detach().cpu().float()
    paths = _feature_paths(layer_dir)
    paths["features"].parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "utterance_features": features,
            "aggregation": "max",
            "hook_point": f"blocks.{int(layer_idx)}.hook_resid_post",
        },
        paths["features"],
    )
    torch.save(
        {
            "utterance_activations": activations,
            "aggregation": "max",
            "hook_point": f"blocks.{int(layer_idx)}.hook_resid_post",
        },
        paths["activations"],
    )
    metadata = {
        "layer_idx": int(layer_idx),
        "hook_point": f"blocks.{int(layer_idx)}.hook_resid_post",
        "aggregation": "max",
        "n_records": int(len(texts)),
        "records_digest": records_digest,
        "feature_shape": list(features.shape),
        "activation_shape": list(activations.shape),
        "max_seq_len": int(max_seq_len),
        "batch_size": int(batch_size),
        "sae_repo_id": sae_repo_id,
        "sae_subfolder": sae_subfolder,
    }
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return np.asarray(features.numpy(), dtype=np.float32), np.asarray(activations.numpy(), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--layers", nargs="+", default=["15", "24"])
    parser.add_argument("--records", default="outputs/misc_full_sae_eval/records.jsonl")
    parser.add_argument("--label-matrix", default="outputs/misc_full_sae_eval/label_matrix.csv")
    parser.add_argument("--model-config", default="config/model_config.json")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--sae-repo-id", default=DEFAULT_SAE_REPO)
    parser.add_argument("--sae-subfolder-template", default="Llama3_1-8B-Base-L{layer}R-8x")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--pca-components", type=int, default=100)
    parser.add_argument("--sae-top-n", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-policy", choices=["stratified-group-kfold", "stratified-kfold"], default="stratified-group-kfold")
    parser.add_argument("--group-column", default="file_id")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--solver", default="liblinear")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--association-chunk-size", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--include-full-sae",
        action="store_true",
        help="Optional expanded baseline; omitted by default for the layer-15/24 reduced comparison.",
    )
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    layers = _parse_layers(args.layers)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = Path(args.records)
    label_path = Path(args.label_matrix)
    records = _read_jsonl(records_path)
    label_df = pd.read_csv(label_path)
    _validate_alignment(records, label_df)
    texts, records_digest = _texts_and_digest(records)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Llama layer-wise representation probe")
    print(f"  layers: {list(layers)}")
    print(f"  records: {len(records)}")
    print(f"  device: {device}")
    print("Loading base model once for all requested layers...")
    model, tokenizer, _ = load_local_model_and_tokenizer(
        args.model_config,
        model_dir=args.model_dir,
        device=device,
    )

    layer_matrices: dict[int, LayerMatrices] = {}
    filter_config = FeatureFilterConfig()
    for layer_idx in layers:
        layer_dir = output_dir / f"layer_{layer_idx:02d}"
        sae_subfolder = str(args.sae_subfolder_template).format(layer=layer_idx)
        cached = None if args.force_extract else _load_cached_layer(
            layer_dir=layer_dir,
            layer_idx=layer_idx,
            n_records=len(records),
            records_digest=records_digest,
            max_seq_len=args.max_seq_len,
            sae_subfolder=sae_subfolder,
        )
        if cached is None:
            print(f"Loading matching SAE for layer {layer_idx}: {sae_subfolder}")
            sae = load_sae_from_hub(
                repo_id=args.sae_repo_id,
                subfolder=sae_subfolder,
                device=device,
                dtype=torch.bfloat16,
            )
            try:
                features, activations = _extract_layer_feature_store(
                    model=model,
                    tokenizer=tokenizer,
                    sae=sae,
                    texts=texts,
                    layer_idx=layer_idx,
                    layer_dir=layer_dir,
                    records_digest=records_digest,
                    max_seq_len=args.max_seq_len,
                    batch_size=args.batch_size,
                    device=device,
                    sae_repo_id=args.sae_repo_id,
                    sae_subfolder=sae_subfolder,
                )
            finally:
                del sae
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            features, activations = cached
            print(f"Reusing validated layer {layer_idx} max-pooled feature store.")

        audit = build_feature_filter(features, filter_config)
        filter_summary = summarize_feature_filter(audit, filter_config, n_samples=len(records))
        layer_dir.mkdir(parents=True, exist_ok=True)
        audit.to_csv(layer_dir / "feature_filter_audit_global.csv", index=False, encoding="utf-8-sig")
        (layer_dir / "feature_filter_global_summary.json").write_text(
            json.dumps(filter_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        layer_matrices[layer_idx] = LayerMatrices(
            sae_features=features,
            raw_hidden=activations,
            feature_filter_audit=audit,
        )

    config = LayerwiseRepresentationProbeConfig(
        labels=tuple(str(label).upper() for label in args.labels),
        pca_components=int(args.pca_components),
        sae_top_n=int(args.sae_top_n),
        include_full_sae=bool(args.include_full_sae),
        folds=int(args.folds),
        split_policy=args.split_policy,
        group_column=args.group_column,
        C=float(args.C),
        solver=args.solver,
        max_iter=int(args.max_iter),
        association_chunk_size=int(args.association_chunk_size),
        quiet=bool(args.quiet),
    )
    result = run_layerwise_representation_probe(
        layer_matrices=layer_matrices,
        label_df=label_df,
        output_dir=output_dir,
        config=config,
    )
    manifest_path = output_dir / "run_manifest.json"
    manifest = {
        "analysis": "llama_layerwise_pca100_sae_top100",
        "elapsed_seconds": round(time.time() - started, 3),
        "layers": list(layers),
        "records": str(records_path),
        "label_matrix": str(label_path),
        "records_digest": records_digest,
        "aggregation": "max",
        "sae_repo_id": args.sae_repo_id,
        "sae_subfolder_template": args.sae_subfolder_template,
        "probe_config": result["manifest"]["config"],
        "outputs": result["manifest"]["outputs"],
        "filter_summaries": result["filter_summaries"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Completed layer-wise comparison.")
    print(result["macro_summary"].to_string(index=False))
    print(f"Report: {output_dir / 'layerwise_representation_probe_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
