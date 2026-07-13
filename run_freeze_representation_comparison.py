"""Freeze the final seven-leaf representation comparison result package."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.representation_comparison_freeze import (
    FreezeConfig,
    freeze_representation_comparison,
    write_package_checksums,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default="outputs/misc_full_sae_eval/interpretability/representation_probe_comparison_pca_post_standardized",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/misc_full_sae_eval/interpretability/representation_comparison_frozen_leaf7_20260713",
    )
    parser.add_argument(
        "--plot-script",
        default="scripts/plot_frozen_representation_comparison.py",
    )
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = freeze_representation_comparison(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        plot_script=args.plot_script,
        config=FreezeConfig(),
    )
    if not args.skip_plot:
        subprocess.run(
            [sys.executable, str(Path(args.output_dir) / "plot_frozen_representation_comparison.py"), "--package-dir", args.output_dir],
            check=True,
        )
        write_package_checksums(args.output_dir)
    print(result["macro"].to_string(index=False))
    print(f"[done] frozen package: {Path(args.output_dir).resolve()}")
    print("[audit] Stable Core SAE remains exploratory: its supervised feature list is not outer-fold nested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
