"""Build the minimum-word statistical dataset and archive excluded utterances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.nlp_re_base.statistical_scope_filter import build_min_word_statistical_scope


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("outputs/misc_full_sae_eval"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/misc_full_sae_eval_min5_words"))
    parser.add_argument("--minimum-words", type=int, default=5)
    parser.add_argument("--skip-feature-tensors", action="store_true")
    args = parser.parse_args()
    result = build_min_word_statistical_scope(
        source_root=args.source_root,
        output_root=args.output_root,
        minimum_words=args.minimum_words,
        filter_feature_tensors=not args.skip_feature_tensors,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
