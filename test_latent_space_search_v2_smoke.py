"""Smoke tests for latent-space search v2."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.latent_space_search_v2 import (  # noqa: E402
    DEFAULT_LABELS,
    LatentSpaceSearchV2Config,
    run_latent_space_search_v2,
)


def _make_label_df(n: int, labels: list[str]) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row = {
            "row_idx": i,
            "unit_text": f"synthetic counseling utterance {i}?",
            "source_split": "high" if i % 2 == 0 else "low",
        }
        for label in labels:
            row[label] = 0
        if i < 12:
            row["RES"] = 1
            row["RE"] = 1
        if 8 <= i < 20:
            row["REC"] = 1
            row["RE"] = 1
        if 10 <= i < 26:
            row["QUO"] = 1
            row["QU"] = 1
        if 22 <= i < 34:
            row["QUC"] = 1
            row["QU"] = 1
        if 30 <= i < 38:
            row["GI"] = 1
        rows.append(row)
    return pd.DataFrame(rows)


def _make_matrix(labels: list[str], d_sae: int) -> pd.DataFrame:
    stable = {
        "RES": {0, 1, 2},
        "REC": {2, 3},
        "RE": {0, 2, 3},
        "QUO": {10, 11},
        "QUC": {11, 12},
        "QU": {10, 11, 12},
        "GI": {20},
        "SU": set(),
        "AF": set(),
    }
    weak_shared = {
        "RES": {50},
        "GI": {50},
    }
    rows = []
    for label in labels:
        prevalence = 0.25
        for idx in range(d_sae):
            is_stable = idx in stable.get(label, set())
            is_weak = idx in weak_shared.get(label, set())
            abs_d = 0.12 + 0.001 * ((idx * 7) % 30)
            auc = 0.53 + 0.0005 * ((idx * 11) % 30)
            precision = prevalence + 0.01
            p_value = 0.7
            significant = False
            if is_stable:
                abs_d = 1.2 - 0.01 * idx
                auc = 0.86 - 0.002 * idx
                precision = prevalence + 0.35
                p_value = 0.001
                significant = True
            elif is_weak:
                abs_d = 0.55
                auc = 0.61
                precision = prevalence + 0.02
                p_value = 0.02
                significant = True
            rows.append(
                {
                    "label": label,
                    "latent_idx": idx,
                    "n_positive": int(prevalence * 40),
                    "n_negative": int((1.0 - prevalence) * 40),
                    "prevalence": prevalence,
                    "cohens_d": abs_d,
                    "abs_cohens_d": abs(abs_d),
                    "auc": auc,
                    "directional_auc": auc,
                    "p_value": p_value,
                    "significant_fdr": significant,
                    "precision_at_50": precision,
                    "precision_lift_at_50": precision - prevalence,
                }
            )
    return pd.DataFrame(rows)


def test_latent_space_search_v2_smoke() -> None:
    labels = list(DEFAULT_LABELS)
    n_samples = 40
    d_sae = 130
    rng = np.random.default_rng(0)
    features = rng.normal(size=(n_samples, d_sae)).astype(np.float32)
    for idx in [0, 1, 2, 3, 10, 11, 12, 20]:
        features[:20, idx] += 5.0

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        matrix_path = root / "latent_label_matrix.csv"
        label_path = root / "label_matrix.csv"
        feature_path = root / "features.npy"
        records_path = root / "records.jsonl"
        output_dir = root / "out"

        _make_matrix(labels, d_sae).to_csv(matrix_path, index=False)
        _make_label_df(n_samples, labels).to_csv(label_path, index=False)
        np.save(feature_path, features)
        records_path.write_text(
            "\n".join(
                json.dumps({"unit_text": f"record {i} synthetic text?"}) for i in range(n_samples)
            )
            + "\n",
            encoding="utf-8",
        )

        result = run_latent_space_search_v2(
            matrix_path=matrix_path,
            feature_store=feature_path,
            label_matrix=label_path,
            records_path=records_path,
            output_dir=output_dir,
            step6_table=None,
            config=LatentSpaceSearchV2Config(random_baseline_repeats=3),
            make_figures=False,
        )

        top20 = result["top20"]
        assert set(top20.groupby("label").size().tolist()) == {20}

        frag = result["fragmentation"].set_index("label")
        assert frag.loc["RES", "thresholded_latent_count"] == 3
        assert frag.loc["GI", "thresholded_latent_count"] == 1
        assert frag.loc["SU", "selection_status"] == "no_stable_latents"
        assert (frag["effective_fragmentation"] <= frag["thresholded_latent_count"] + 1e-9).all()

        leaf_pairs = result["overlap_thresholded"]
        leaf_pairs = leaf_pairs[leaf_pairs["comparison_scope"] == "leaf_pair"]
        assert "RE" not in set(leaf_pairs["label_a"]) | set(leaf_pairs["label_b"])
        assert "QU" not in set(leaf_pairs["label_a"]) | set(leaf_pairs["label_b"])
        assert {"specific_jaccard", "generalized_shared_latents"}.issubset(leaf_pairs.columns)

        weighted = result["overlap_weighted"]
        wp = weighted[weighted["comparison_scope"] == "leaf_pair"].set_index(["label_a", "label_b"])
        assert wp.loc[("RES", "REC"), "weighted_jaccard"] > wp.loc[("RES", "GI"), "weighted_jaccard"]

        roles = result["polysemanticity"].set_index("latent_idx")
        assert roles.loc[2, "role"] == "sibling_shared"

        sensitivity = result["k_sensitivity"]
        assert set(sensitivity["k"].unique()) == {5, 10, 20, 50, 100}
        assert {"RES-REC", "QUO-QUC", "RE_family-QU_family"}.issubset(
            set(sensitivity["comparison"])
        )

        required = [
            "latent_label_association_v2.csv",
            "top20_candidate_set_v2.csv",
            "thresholded_latent_sets_v2.csv",
            "weighted_latent_label_matrix_v2.csv",
            "fragmentation_v2.csv",
            "overlap_thresholded_v2.csv",
            "overlap_weighted_v2.csv",
            "polysemanticity_v2.csv",
            "latent_role_assignments_v2.csv",
            "hierarchy_recovery_v2.csv",
            "k_sensitivity_summary_v2.csv",
            "semantic_review_candidates_v2.csv",
            "latent_space_search_summary.json",
            "latent_space_search_report.md",
        ]
        for name in required:
            assert (output_dir / name).exists(), name


if __name__ == "__main__":
    test_latent_space_search_v2_smoke()
    print("latent_space_search_v2 smoke passed")
