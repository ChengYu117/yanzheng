"""Generate Task 4 tables from existing feature-card codes and reviewed groupings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.nlp_re_base.task4_behavior_structure import build_task4


DEFAULT_PACKAGE = Path("论文相关文档/latent_card_B级人工审查包")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.package_dir / "Task4_行为表征结构"
    print(json.dumps(build_task4(package_dir=args.package_dir, output_dir=output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
