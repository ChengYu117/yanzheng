"""Smoke tests for minimal sufficient subspace v2."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _make_synthetic() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    n = 400
    d = 64
    features = rng.normal(0, 0.15, size=(n, d)).astype(np.float32)

    one = np.zeros(n, dtype=int)
    one[:100] = 1
    features[one == 1, 0] += 3.0

    two = np.zeros(n, dtype=int)
    two[100:220] = 1
    # Two complementary latents, each covers half of the positive class.
    features[100:160, 10] += 3.0
    features[160:220, 11] += 3.0
    features[two == 0, 10] -= 0.3
    features[two == 0, 11] -= 0.3

    dist = np.zeros(n, dtype=int)
    dist[220:320] = 1
    for block, lid in enumerate([20, 21, 22, 23, 24]):
        start = 220 + block * 20
        end = start + 20
        features[start:end, lid] += 3.2

    noise = rng.binomial(1, 0.5, size=n)
    parent = np.maximum(one, two)

    labels = pd.DataFrame(
        {
            "RE": parent,
            "RES": one,
            "REC": two,
            "QUO": dist,
            "GI": noise,
        }
    )

    stable = {
        "RE": [0, 10, 11],
        "RES": [0, 1],
        "REC": [10, 11, 12],
        "QUO": [20, 21, 22, 23, 24],
        "GI": [40, 41, 42],
    }
    rows = []
    for label, ids in stable.items():
        for rank, lid in enumerate(ids + [x for x in range(d) if x not in ids][:12], start=1):
            is_stable = lid in ids and label != "GI"
            rows.append(
                {
                    "label": label,
                    "latent_idx": lid,
                    "association_rank": rank,
                    "directional_auc": 0.9 - 0.01 * rank if is_stable else 0.56,
                    "abs_cohens_d": 1.2 if is_stable else 0.12,
                    "cohens_d": 1.2 if is_stable else 0.12,
                    "prevalence": float(labels[label].mean()),
                    "precision_at_50": float(labels[label].mean()) + (0.3 if is_stable else 0.0),
                    "precision_lift_absolute_at_50": 0.3 if is_stable else 0.0,
                    "stable_edge": is_stable,
                    "positive_support": is_stable,
                    "negative_boundary": False,
                    "edge_type": "positive_support" if is_stable else "weak_or_noise",
                    "formal_edge_weight": 1.0 if is_stable else 0.0,
                }
            )
    return features, labels, pd.DataFrame(rows)


def test_minimal_sufficient_subspace_v2_smoke() -> None:
    from nlp_re_base.minimal_sufficient_subspace_v2 import (
        MinimalSufficientSubspaceConfig,
        run_minimal_sufficient_subspace_v2,
    )

    features, label_df, association = _make_synthetic()
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        feature_path = root / "features.npy"
        label_path = root / "labels.csv"
        association_path = root / "association.csv"
        threshold_path = root / "thresholded.csv"
        out = root / "out"
        np.save(feature_path, features)
        label_df.to_csv(label_path, index=False)
        association.to_csv(association_path, index=False)
        association[association["stable_edge"]].to_csv(threshold_path, index=False)

        result = run_minimal_sufficient_subspace_v2(
            association_matrix=association_path,
            thresholded_sets=threshold_path,
            feature_store=feature_path,
            label_matrix=label_path,
            output_dir=out,
            config=MinimalSufficientSubspaceConfig(
                labels=("RE", "RES", "REC", "QUO", "GI"),
                leaf_labels=("RES", "REC", "QUO", "GI"),
                candidate_top_k=12,
                cv_folds=3,
                min_auc=0.70,
                max_search_k=8,
                random_state=3,
            ),
            make_figures=False,
        )

        summary = result["summary"].set_index("label")
        assert int(summary.loc["RES", "minimal_k_median"]) == 1
        assert int(summary.loc["REC", "minimal_k_median"]) == 2
        assert int(summary.loc["QUO", "minimal_k_median"]) > 3
        assert summary.loc["GI", "formal_status"] == "not_recoverable_under_candidate_pool"
        assert summary.loc["RE", "label_role"] == "parent_consistency_only"

        redundancy = result["redundancy"]
        assert "redundancy_role" in redundancy.columns
        assert (out / "minimal_sufficient_summary_v2.csv").exists()
        assert (out / "minimal_sufficient_selected_latents_v2.csv").exists()
        assert (out / "minimal_sufficient_fold_results_v2.csv").exists()
        assert (out / "minimal_sufficient_selection_steps_v2.csv").exists()
        assert (out / "minimal_sufficient_redundancy_audit_v2.csv").exists()
        assert (out / "minimal_sufficient_curves_v2.csv").exists()
        assert (out / "minimal_sufficient_summary.json").exists()
        assert (out / "minimal_sufficient_subspace_report.md").exists()


def test_filtered_topk_policy_without_legacy_seed() -> None:
    from nlp_re_base.minimal_sufficient_subspace_v2 import (
        MinimalSufficientSubspaceConfig,
        run_minimal_sufficient_subspace_v2,
    )

    features, label_df, association = _make_synthetic()
    filtered_association = association.drop(
        columns=[
            "association_rank",
            "stable_edge",
            "positive_support",
            "negative_boundary",
            "edge_type",
            "formal_edge_weight",
        ],
        errors="ignore",
    )
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        feature_path = root / "features.npy"
        label_path = root / "labels.csv"
        association_path = root / "filtered_association.csv"
        audit_path = root / "feature_filter_audit.csv"
        out = root / "out"
        np.save(feature_path, features)
        label_df.to_csv(label_path, index=False)
        filtered_association.to_csv(association_path, index=False)
        pd.DataFrame(
            {
                "latent_idx": list(range(features.shape[1])),
                "keep": [True] * features.shape[1],
            }
        ).to_csv(audit_path, index=False)

        result = run_minimal_sufficient_subspace_v2(
            association_matrix=association_path,
            thresholded_sets=None,
            feature_store=feature_path,
            label_matrix=label_path,
            output_dir=out,
            feature_filter_audit=audit_path,
            config=MinimalSufficientSubspaceConfig(
                labels=("RES", "REC"),
                leaf_labels=("RES", "REC"),
                candidate_policy="filtered_topk_only",
                candidate_top_k=8,
                cv_folds=3,
                min_auc=0.70,
                max_search_k=5,
                random_state=3,
            ),
            make_figures=False,
        )

        candidates = result["candidates"]
        assert not candidates.empty
        assert set(candidates["candidate_source"]) == {"filtered_top8"}
        assert not candidates["stable_edge"].astype(bool).any()
        assert "association_rank" in candidates.columns
        assert int(candidates.groupby("label").size().max()) <= 8
        assert result["json_summary"]["candidate_policy"] == "filtered_topk_only"
        assert result["json_summary"]["keep_pool_audit"]["keep_true_latents"] == features.shape[1]
        assert result["json_summary"]["candidate_checks"]["candidate_pool_within_limit"] is True


def test_filtered_topk_rejects_dropped_latent() -> None:
    from nlp_re_base.minimal_sufficient_subspace_v2 import (
        MinimalSufficientSubspaceConfig,
        run_minimal_sufficient_subspace_v2,
    )

    features, label_df, association = _make_synthetic()
    filtered_association = association[
        (association["label"] == "RES") & (association["latent_idx"].isin([0, 1]))
    ].drop(columns=["association_rank", "stable_edge"], errors="ignore")
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        feature_path = root / "features.npy"
        label_path = root / "labels.csv"
        association_path = root / "filtered_association.csv"
        audit_path = root / "feature_filter_audit.csv"
        out = root / "out"
        np.save(feature_path, features)
        label_df.to_csv(label_path, index=False)
        filtered_association.to_csv(association_path, index=False)
        pd.DataFrame(
            {
                "latent_idx": [0, 1],
                "keep": [True, False],
            }
        ).to_csv(audit_path, index=False)

        try:
            run_minimal_sufficient_subspace_v2(
                association_matrix=association_path,
                thresholded_sets=None,
                feature_store=feature_path,
                label_matrix=label_path,
                output_dir=out,
                feature_filter_audit=audit_path,
                config=MinimalSufficientSubspaceConfig(
                    labels=("RES",),
                    leaf_labels=("RES",),
                    candidate_policy="filtered_topk_only",
                    candidate_top_k=2,
                    cv_folds=3,
                    max_search_k=2,
                ),
                make_figures=False,
            )
        except ValueError as exc:
            assert "keep=True" in str(exc)
        else:
            raise AssertionError("Expected dropped latent in filtered association to be rejected")


def main() -> int:
    test_minimal_sufficient_subspace_v2_smoke()
    test_filtered_topk_policy_without_legacy_seed()
    test_filtered_topk_rejects_dropped_latent()
    print("minimal_sufficient_subspace_v2 smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
