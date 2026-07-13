"""Smoke checks for Task 4 task construction and strict validators."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.nlp_re_base.task4_behavior_structure import EVIDENCE_CLASSES, _scope


def main() -> None:
    assert "behavioral_function" in EVIDENCE_CLASSES
    assert _scope({"QU", "QUO"}) == "parent_child"
    assert _scope({"QUO", "QUC"}) == "sibling_leaf"
    assert _scope({"AF", "SU"}) == "cross_family"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "utf8.json"
        path.write_text(json.dumps({"status": "自动候选"}, ensure_ascii=False), encoding="utf-8")
        assert json.loads(path.read_text(encoding="utf-8"))["status"] == "自动候选"
    print("Task 4 behavior-structure smoke test passed")


if __name__ == "__main__":
    main()
