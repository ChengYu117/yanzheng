"""Grouped bootstrap confidence intervals for MISC SAE latent associations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cross_val_framework import (
    CrossValInputs,
    compute_auc_and_cohens_d,
    grouped_bootstrap_row_indices,
    top_latents_for_label,
)


def run_bootstrap_ci(
    inputs: CrossValInputs,
    *,
    association_matrix_path: str | Path,
    output_dir: str | Path,
    group_column: str = "source_file",
    ranking_metric: str = "cohens_d",
    top_k: int = 100,
    n_bootstrap: int = 2000,
    random_state: int = 42,
    chunk_size: int = 512,
) -> dict:
    """Run E2 grouped bootstrap CIs for top label-latent pairs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if group_column not in inputs.label_matrix.columns:
        raise ValueError(f"group column {group_column!r} not found in label matrix")

    association = pd.read_csv(association_matrix_path)
    latent_to_col = {int(latent): i for i, latent in enumerate(inputs.latent_indices.tolist())}
    groups = inputs.label_matrix[group_column].astype(str).to_numpy()
    group_to_indices = {
        str(group): np.flatnonzero(groups == str(group)).astype(np.int64)
        for group in sorted(pd.Series(groups).unique())
    }
    rng = np.random.default_rng(random_state)
    rows: list[dict] = []

    for label in inputs.labels:
        top = top_latents_for_label(
            association,
            label,
            metric=ranking_metric,
            top_k=top_k,
            positive_only=ranking_metric == "cohens_d",
        ).copy()
        if top.empty:
            continue
        latent_indices = top["latent_idx"].astype(int).to_numpy()
        feature_cols = np.array([latent_to_col[int(latent)] for latent in latent_indices], dtype=np.int64)
        features = inputs.features[:, feature_cols]
        y_full = inputs.label_matrix[label].astype(int).to_numpy(dtype=bool)
        point_auc, point_d = compute_auc_and_cohens_d(features, y_full, chunk_size=chunk_size)
        boot_auc = np.empty((int(n_bootstrap), len(latent_indices)), dtype=np.float32)
        boot_d = np.empty_like(boot_auc)

        for boot_idx in range(int(n_bootstrap)):
            row_idx = grouped_bootstrap_row_indices(group_to_indices, rng=rng)
            auc, d = compute_auc_and_cohens_d(
                features[row_idx, :],
                y_full[row_idx],
                chunk_size=chunk_size,
            )
            boot_auc[boot_idx, :] = auc
            boot_d[boot_idx, :] = d

        auc_lo, auc_hi = np.percentile(boot_auc, [2.5, 97.5], axis=0)
        d_lo, d_hi = np.percentile(boot_d, [2.5, 97.5], axis=0)
        top_by_latent = top.set_index("latent_idx")
        for i, latent_idx in enumerate(latent_indices):
            assoc_row = top_by_latent.loc[int(latent_idx)]
            rows.append(
                {
                    "label": label,
                    "latent_idx": int(latent_idx),
                    "ranking_metric": ranking_metric,
                    "rank_within_label": int(i + 1),
                    "point_auc": float(point_auc[i]),
                    "auc_ci_lo": float(auc_lo[i]),
                    "auc_ci_hi": float(auc_hi[i]),
                    "point_cohens_d": float(point_d[i]),
                    "cohens_d_ci_lo": float(d_lo[i]),
                    "cohens_d_ci_hi": float(d_hi[i]),
                    "ci_excludes_zero": bool(d_lo[i] > 0 or d_hi[i] < 0),
                    "source_auc": float(assoc_row.get("auc", np.nan)),
                    "source_directional_auc": float(assoc_row.get("directional_auc", np.nan)),
                    "source_cohens_d": float(assoc_row.get("cohens_d", np.nan)),
                    "n_positive": int(y_full.sum()),
                    "n_negative": int(len(y_full) - y_full.sum()),
                    "n_bootstrap": int(n_bootstrap),
                    "bootstrap_group_column": group_column,
                    "n_bootstrap_groups": int(len(group_to_indices)),
                }
            )

    out = pd.DataFrame(rows)
    csv_path = output_dir / "bootstrap_ci_by_label_latent.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    manifest = {
        "analysis": "bootstrap_ci",
        "association_matrix": str(association_matrix_path),
        "group_column": group_column,
        "ranking_metric": ranking_metric,
        "top_k": int(top_k),
        "n_bootstrap": int(n_bootstrap),
        "random_state": int(random_state),
        "labels": list(inputs.labels),
        "n_rows": int(len(inputs.label_matrix)),
        "n_filtered_latents": int(len(inputs.latent_indices)),
        "outputs": {"bootstrap_ci": str(csv_path)},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest

