from __future__ import annotations

from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nlp_re_base.data import dataset_summary


def main() -> None:
    summary = dataset_summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
