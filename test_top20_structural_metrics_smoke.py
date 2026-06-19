"""Smoke test for Top20-only structural metric recomputation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from run_misc_top20_structural_metrics import run_top20_structural_metrics


def _make_topk_fixture(path: Path) -> None:
    latent_sets = {
        "RE": [1, 3, 4],
        "QU": [10, 11, 13],
        "RES": [1, 2, 3],
        "REC": [3, 4, 5],
        "QUO": [10, 11, 12],
        "QUC": [11, 12, 13],
        "GI": [20, 30, 40],
        "SU": [20, 31, 41],
        "AF": [50, 60, 70],
    }
    rows: list[dict[str, object]] = []
    for label, latents in latent_sets.items():
        for rank, latent_idx in enumerate(latents, start=1):
            rows.append(
                {
                    "label": label,
                    "latent_idx": latent_idx,
                    "association_rank": rank,
                    "abs_cohens_d": 1.0 / rank,
                    "directional_auc": 0.9 - rank * 0.01,
                    "formal_edge_weight": 1.0 / rank,
                    "significant_fdr": True,
                    "stable_edge": True,
                    "positive_support": True,
                    "negative_boundary": False,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_top20_structural_metrics_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        topk_path = root / "top20.csv"
        output_dir = root / "out"
        _make_topk_fixture(topk_path)

        summary = run_top20_structural_metrics(
            top20_candidates=topk_path,
            output_dir=output_dir,
            top_k=3,
            make_figures=False,
        )

        overlap = pd.read_csv(output_dir / "top20_leaf_pair_overlap.csv")
        pair_labels = set(overlap["label_a"]) | set(overlap["label_b"])
        assert "RE" not in pair_labels
        assert "QU" not in pair_labels

        relations = {
            tuple(sorted((row["label_a"], row["label_b"]))): row["relation_type"]
            for _, row in overlap.iterrows()
        }
        assert relations[("QUC", "QUO")] == "same_family"
        assert relations[("REC", "RES")] == "same_family"
        assert relations[("GI", "SU")] == "same_supplemental_block"

        poly = pd.read_csv(output_dir / "top20_polysemanticity.csv")
        roles = poly.set_index("latent_idx")["role"].to_dict()
        assert roles[3] == "same_family_shared"
        assert roles[11] == "same_family_shared"
        assert roles[20] == "same_supplemental_block"

        frag = pd.read_csv(output_dir / "top20_fragmentation.csv")
        assert (frag["effective_fragmentation_by_weight"] <= 3.0 + 1e-9).all()

        assert summary["top_k"] == 3
        assert (output_dir / "top20_structural_report.md").exists()


if __name__ == "__main__":
    test_top20_structural_metrics_smoke()
    print("top20 structural metrics smoke passed")
