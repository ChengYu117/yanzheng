"""Export MISC label latent groups for downstream causal validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.causal_candidates import (  # noqa: E402
    DEFAULT_CAUSAL_CANDIDATE_OUTPUT,
    DEFAULT_GROUP_SIZES,
    export_misc_causal_candidates,
)
from nlp_re_base.mapping_structure import DEFAULT_CORE_LABELS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export label-specific SAE latent groups for MISC causal validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eval-dir", default="outputs/misc_full_sae_eval")
    parser.add_argument("--mapping-dir", default=None)
    parser.add_argument("--mapping-structure-dir", default=None)
    parser.add_argument("--followup-dir", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_CAUSAL_CANDIDATE_OUTPUT)
    parser.add_argument("--labels", nargs="+", default=DEFAULT_CORE_LABELS)
    parser.add_argument("--group-sizes", nargs="+", type=int, default=list(DEFAULT_GROUP_SIZES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--doc-report",
        default="doc/MISC因果验证候选组说明.md",
        help="Optional doc report path. Pass an empty string to skip.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = export_misc_causal_candidates(
        eval_dir=args.eval_dir,
        mapping_dir=args.mapping_dir,
        mapping_structure_dir=args.mapping_structure_dir,
        followup_dir=args.followup_dir,
        output_dir=args.output_dir,
        labels=args.labels,
        group_sizes=args.group_sizes,
        seed=args.seed,
        doc_report=args.doc_report or None,
    )
    print("Completed MISC causal candidate export.")
    print(f"Output dir: {payload['output_dir']}")
    print(f"Groups: {payload['files']['causal_candidate_groups']}")
    print(f"Summary: {payload['files']['candidate_group_summary']}")
    if payload["files"].get("doc_report"):
        print(f"Doc report: {payload['files']['doc_report']}")


if __name__ == "__main__":
    main()
