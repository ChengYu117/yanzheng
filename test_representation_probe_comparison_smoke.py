"""Smoke tests for run_misc_representation_probe_comparison.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from run_misc_representation_probe_comparison import (
    METRICS,
    RepresentationProbeComparisonConfig,
    _topn_fold_order,
    run_representation_probe_comparison,
)


def _make_synthetic_inputs(root: Path) -> dict[str, Path]:
    rng = np.random.default_rng(123)
    n = 72
    sae_dim = 10
    hidden_dim = 8
    groups = np.repeat([f"file_{i:02d}" for i in range(12)], 6)

    y_re = np.zeros(n, dtype=int)
    y_qu = np.zeros(n, dtype=int)
    for i, group in enumerate(np.unique(groups)):
        rows = np.flatnonzero(groups == group)
        if i % 2 == 0:
            y_re[rows[:3]] = 1
        else:
            y_qu[rows[:3]] = 1

    sae = rng.normal(0.0, 0.2, size=(n, sae_dim)).astype(np.float32)
    sae[:, 0] += y_re * 3.0
    sae[:, 1] += y_re * 2.0
    sae[:, 2] += y_qu * 3.0
    sae[:, 3] += y_qu * 2.0
    sae[:, 4] -= y_re * 2.0  # negative Cohen's d for RE; must not be selected by Top-n.
    sae[:, 8] += y_re * 0.4
    sae[:, 9] += y_qu * 0.4

    hidden = rng.normal(0.0, 0.3, size=(n, hidden_dim)).astype(np.float32)
    hidden[:, 0] += y_re * 1.5
    hidden[:, 1] += y_qu * 1.5

    label_df = pd.DataFrame(
        {
            "row_idx": np.arange(n),
            "record_id": [f"r{i}" for i in range(n)],
            "file_id": groups,
            "source_file": [f"{g}.jsonl" for g in groups],
            "source_split": ["high" if i % 2 == 0 else "low" for i in range(n)],
            "unit_text": [f"utterance {i}" for i in range(n)],
            "RE": y_re,
            "RES": 0,
            "REC": 0,
            "QU": y_qu,
            "QUO": 0,
            "QUC": 0,
            "GI": 0,
            "SU": 0,
            "AF": 0,
            "OTHER": 1 - np.maximum(y_re, y_qu),
        }
    )

    audit = pd.DataFrame({"latent_idx": np.arange(sae_dim), "keep": True})
    audit.loc[audit["latent_idx"] == 6, "keep"] = False

    association_rows = []
    for label, scores in {
        "RE": {0: 2.0, 1: 1.5, 8: 0.4, 4: -2.0, 7: 0.1},
        "QU": {2: 2.0, 3: 1.5, 9: 0.4, 5: 0.1},
    }.items():
        for latent_idx in range(sae_dim):
            d = scores.get(latent_idx, 0.0)
            association_rows.append(
                {
                    "label": label,
                    "latent_idx": latent_idx,
                    "cohens_d": d,
                    "abs_cohens_d": abs(d),
                    "auc": 0.5 + min(abs(d), 2.0) / 10.0,
                    "directional_auc": 0.5 + min(abs(d), 2.0) / 10.0,
                }
            )
    association = pd.DataFrame(association_rows)

    stable = pd.DataFrame(
        [
            {"label": "RE", "latent_idx": 7, "stable_set_role": "stable_core", "full_data_rank": 5},
            {"label": "RE", "latent_idx": 8, "stable_set_role": "boundary_candidate", "full_data_rank": 3},
            {"label": "QU", "latent_idx": 9, "stable_set_role": "stable_core", "full_data_rank": 3},
        ]
    )

    paths = {
        "association": root / "association.csv",
        "audit": root / "feature_filter_audit.csv",
        "stable": root / "stable_core.csv",
    }
    association.to_csv(paths["association"], index=False)
    audit.to_csv(paths["audit"], index=False)
    stable.to_csv(paths["stable"], index=False)
    return {
        **paths,
        "sae": sae,
        "hidden": hidden,
        "labels": label_df,
    }


def test_topn_fold_order_uses_positive_cohens_d() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        data = _make_synthetic_inputs(Path(tmp))
        y = data["labels"]["RE"].to_numpy(dtype=int)
        keep = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9], dtype=np.int32)
        selected, d = _topn_fold_order(
            data["sae"],
            np.arange(len(y)),
            y,
            keep,
            max_n=4,
            chunk_size=16,
        )
        assert 4 not in selected.tolist(), "negative Cohen's d latent should not be selected"
        assert selected[0] == 0
        assert selected[1] == 1
        assert np.all(d > 0)


def test_representation_probe_comparison_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data = _make_synthetic_inputs(root)
        out = root / "out"
        result = run_representation_probe_comparison(
            sae_features=data["sae"],
            raw_hidden=data["hidden"],
            label_df=data["labels"],
            filtered_association_path=data["association"],
            feature_filter_audit_path=data["audit"],
            stable_core_path=data["stable"],
            output_dir=out,
            config=RepresentationProbeComparisonConfig(
                labels=("RE", "QU"),
                top_ns=(2, 4),
                random_repeats=3,
                folds=3,
                random_state=42,
                quiet=True,
            ),
        )

        expected = [
            "probe_fold_metrics.csv",
            "probe_summary_by_label.csv",
            "probe_macro_summary.csv",
            "selected_latents_by_label_n.csv",
            "representation_probe_comparison_report.md",
            "manifest.json",
            "figures/performance_curves_macro.png",
        ]
        for rel in expected:
            assert (out / rel).exists(), rel

        fold = result["fold_metrics"]
        for metric in METRICS:
            assert metric in fold.columns

        stable_selected = result["selected_latents"]
        re_stable = stable_selected[
            (stable_selected["representation"] == "Stable Core SAE")
            & (stable_selected["label"] == "RE")
        ]["latent_idx"].astype(int).tolist()
        assert re_stable == [7], "stable core should use only stable_set_role == stable_core"

        topn_rows = stable_selected[stable_selected["representation"] == "Top-n SAE"]
        assert not topn_rows.empty
        assert "fold_positive_cohens_d" in set(topn_rows["selection_type"])
        re_topn_rows = topn_rows[topn_rows["label"] == "RE"]
        assert 4 not in set(re_topn_rows["latent_idx"].astype(int)), "Top-n must not use negative Cohen's d"

        random_rows = stable_selected[stable_selected["representation"] == "Random SAE-n"]
        assert not random_rows.empty
        assert 6 not in set(random_rows["latent_idx"].astype(int)), "Random SAE-n must use keep=True pool only"
        assert set(random_rows["seed"].astype(int)) == {42, 43, 44}

        # Every representation should reuse the same fold sizes for a label/fold.
        re_fold = fold[fold["label"] == "RE"]
        grouped = re_fold.groupby("fold")[["n_train", "n_test"]].nunique()
        assert (grouped["n_train"] == 1).all()
        assert (grouped["n_test"] == 1).all()

        macro = result["macro_summary"]
        assert {"Hidden State", "Full SAE", "Stable Core SAE", "Top-n SAE", "PCA-n", "Random SAE-n"}.issubset(
            set(macro["representation"])
        )


if __name__ == "__main__":
    test_topn_fold_order_uses_positive_cohens_d()
    test_representation_probe_comparison_smoke()
    print("test_representation_probe_comparison_smoke passed")
