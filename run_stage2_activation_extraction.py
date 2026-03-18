"""Standalone CLI for Stage 2 activation extraction and per-layer probes."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.stage2_activation_extraction import main


if __name__ == "__main__":
    main()
