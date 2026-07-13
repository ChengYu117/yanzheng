"""Compute raw agreement and Cohen's kappa for completed Task 4 reviewer sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.nlp_re_base.contrastive_evidence_pack import write_json


def _agreement(left: pd.Series, right: pd.Series) -> dict[str, float | int | None]:
    a = left.fillna("").astype(str).str.strip()
    b = right.fillna("").astype(str).str.strip()
    mask = a.ne("") & b.ne("")
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return {"n_double_coded": 0, "raw_agreement": None, "cohens_kappa": None}
    raw = float((a == b).mean())
    categories = sorted(set(a) | set(b))
    expected = sum(float((a == category).mean() * (b == category).mean()) for category in categories)
    kappa = None if np.isclose(1 - expected, 0) else float((raw - expected) / (1 - expected))
    return {"n_double_coded": len(a), "raw_agreement": raw, "cohens_kappa": kappa}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    a = pd.read_csv(args.reviewer_a, dtype=str).set_index("item_id")
    b = pd.read_csv(args.reviewer_b, dtype=str).set_index("item_id")
    common = sorted(set(a.index) & set(b.index))
    if not common:
        raise ValueError("Reviewer sheets have no common item_id values")
    joined = a.loc[common].add_suffix("_A").join(b.loc[common].add_suffix("_B"))
    evidence = _agreement(joined["reviewer_evidence_class_A"], joined["reviewer_evidence_class_B"])
    component = _agreement(joined["reviewer_behavior_component_A"], joined["reviewer_behavior_component_B"])
    disagreements = joined[
        (
            joined["reviewer_evidence_class_A"].fillna("").str.strip().ne("")
            & joined["reviewer_evidence_class_B"].fillna("").str.strip().ne("")
            & joined["reviewer_evidence_class_A"].ne(joined["reviewer_evidence_class_B"])
        )
        | (
            joined["reviewer_behavior_component_A"].fillna("").str.strip().ne("")
            & joined["reviewer_behavior_component_B"].fillna("").str.strip().ne("")
            & joined["reviewer_behavior_component_A"].ne(joined["reviewer_behavior_component_B"])
        )
    ].reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    disagreement_path = args.output_dir / "task4_reviewer_disagreements.csv"
    disagreements.to_csv(disagreement_path, index=False, encoding="utf-8-sig")
    result = {
        "analysis": "task4_two_reviewer_agreement",
        "n_common_items": len(common),
        "evidence_class": evidence,
        "behavior_component": component,
        "n_disagreement_rows": len(disagreements),
        "disagreements": str(disagreement_path),
    }
    write_json(args.output_dir / "task4_reviewer_agreement.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
