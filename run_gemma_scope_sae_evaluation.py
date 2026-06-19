"""Run Gemma3-4B layer-18 GemmaScope SAE evaluation on full MISC data."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.gemma_scope_pipeline import (  # noqa: E402
    DEFAULT_GEMMA_MODEL_ID,
    DEFAULT_LABELS,
    DEFAULT_LOCAL_GEMMA_DIR,
    DEFAULT_OUTPUT_DIR,
    GemmaScopeEvaluationConfig,
    resolve_gemma_model_source,
    run_gemma_scope_sae_evaluation,
)
from nlp_re_base.gemma_scope_sae import (  # noqa: E402
    DEFAULT_GEMMA_SCOPE_LAYER_IDX,
    DEFAULT_GEMMA_SCOPE_REPO_ID,
    gemma_scope_subfolder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemma3-4B + GemmaScope layer-18 SAE MISC evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-source", default=None)
    parser.add_argument("--local-model-dir", default=DEFAULT_LOCAL_GEMMA_DIR)
    parser.add_argument("--hf-model-id", default=DEFAULT_GEMMA_MODEL_ID)
    parser.add_argument("--data-dir", default="data/mi_quality_counseling_misc")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layer-idx", type=int, default=DEFAULT_GEMMA_SCOPE_LAYER_IDX)
    parser.add_argument("--sae-repo-id", default=DEFAULT_GEMMA_SCOPE_REPO_ID)
    parser.add_argument("--sae-subfolder", default=None)
    parser.add_argument("--sae-width", default="16k")
    parser.add_argument("--sae-l0", default="small")
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--aggregation", choices=["max", "mean"], default="max")
    parser.add_argument("--model-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--sae-dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--ignore-hf-token-env", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--association-chunk-size", type=int, default=512)
    parser.add_argument("--min-positive", type=int, default=10)
    parser.add_argument("--min-negative", type=int, default=10)
    parser.add_argument("--skip-label-mapping", action="store_true")
    return parser.parse_args()


def _resolve_source(args: argparse.Namespace) -> str:
    if args.model_source:
        return args.model_source
    if args.local_model_dir and Path(args.local_model_dir).exists():
        return str(Path(args.local_model_dir))
    return resolve_gemma_model_source(args.hf_model_id)


def _hf_token(args: argparse.Namespace) -> str | None:
    if args.ignore_hf_token_env:
        return None
    return os.environ.get(args.hf_token_env) or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def main() -> int:
    args = parse_args()
    subfolder = args.sae_subfolder or gemma_scope_subfolder(
        layer_idx=args.layer_idx,
        width=args.sae_width,
        l0=args.sae_l0,
    )
    config = GemmaScopeEvaluationConfig(
        model_source=_resolve_source(args),
        sae_repo_id=args.sae_repo_id,
        sae_subfolder=subfolder,
        layer_idx=args.layer_idx,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        labels=tuple(label.upper() for label in args.labels),
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        aggregation=args.aggregation,
        model_dtype=args.model_dtype,
        sae_dtype=args.sae_dtype,
        device=args.device,
        device_map=args.device_map,
        cache_dir=args.cache_dir,
        limit_records=args.limit_records,
        fdr_alpha=args.fdr_alpha,
        association_chunk_size=args.association_chunk_size,
        min_positive=args.min_positive,
        min_negative=args.min_negative,
        skip_label_mapping=args.skip_label_mapping,
    )
    print("Gemma3 + GemmaScope SAE evaluation")
    print(f"  model_source: {config.model_source}")
    print(f"  sae: {config.sae_repo_id}/{config.sae_subfolder}")
    print(f"  layer_idx: {config.layer_idx}")
    print(f"  output_dir: {config.output_dir}")
    summary = run_gemma_scope_sae_evaluation(
        config=config,
        hf_endpoint=args.hf_endpoint,
        trust_remote_code=args.trust_remote_code,
        hf_token=_hf_token(args),
    )
    print("Completed GemmaScope SAE evaluation.")
    print(f"  records: {summary['n_records']}")
    print(f"  feature_shape: {summary['feature_shape']}")
    print(f"  structural_metrics: {summary['files']['metrics_structural']}")
    print(f"  mapping_matrix: {summary['files']['mapping_matrix']}")
    print(f"  report: {summary['files']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
