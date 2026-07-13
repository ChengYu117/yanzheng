"""Generate an annotation-review table from stable-core SAE OOF disagreements."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nlp_re_base.sae_annotation_audit import SAEAnnotationAuditConfig, run_sae_annotation_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-store",
        default="outputs/misc_full_sae_eval/feature_store/utterance_features.pt",
    )
    parser.add_argument(
        "--label-matrix", default="outputs/misc_full_sae_eval/label_matrix.csv"
    )
    parser.add_argument(
        "--stable-core",
        default="outputs/cross_val/stable_topk_selection/stable_topk_latent_set.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/misc_full_sae_eval/interpretability/sae_annotation_audit_stable_core",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--high-per-type", type=int, default=10)
    parser.add_argument("--boundary-per-type", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SAEAnnotationAuditConfig(
        folds=args.folds,
        random_state=args.random_state,
        high_per_type=args.high_per_type,
        boundary_per_type=args.boundary_per_type,
    )
    result = run_sae_annotation_audit(
        feature_store_path=args.feature_store,
        label_matrix_path=args.label_matrix,
        stable_core_path=args.stable_core,
        output_dir=args.output_dir,
        config=config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
