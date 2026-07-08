"""Evaluate whether Cohen's d AUC@K upper bounds change beyond K=100."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.full_representation_probe import (  # noqa: E402
    DEFAULT_LABELS,
    FeatureFilterConfig,
    FullRepresentationProbeConfig,
    _build_auc_by_k_curve,
    _chunked_sae_train_associations,
    _make_splits,
    _prepare_fold_features,
    _rank_sae_features,
    _score_sae_top_n_subspace,
    _select_available_labels,
    _summarize_by_label,
    build_feature_filter,
    load_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae-features", default="outputs/misc_full_sae_eval/feature_store/utterance_features.pt")
    parser.add_argument("--label-matrix", default="outputs/misc_full_sae_eval/label_matrix.csv")
    parser.add_argument(
        "--base-auc-curve",
        default="outputs/misc_full_sae_eval/interpretability/ranked_sae_subspace_probe_k001_100/auc_by_k_curve_0_100.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/misc_full_sae_eval/interpretability/cohensd_auc_k_sensitivity_100_150_200",
    )
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-policy", default="stratified-group-kfold")
    parser.add_argument("--group-column", default="file_id")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--solver", default="liblinear")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--association-chunk-size", type=int, default=512)
    parser.add_argument("--start-k", type=int, default=101)
    parser.add_argument("--max-k", type=int, default=200)
    parser.add_argument("--compare-k", nargs="+", type=int, default=[100, 150, 200])
    parser.add_argument("--subspace-n-jobs", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _series_stats(values: pd.Series) -> dict[str, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return {"auc_mean": np.nan, "auc_std": np.nan, "n_folds": 0}
    return {
        "auc_mean": float(vals.mean()),
        "auc_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "n_folds": int(len(vals)),
    }


def _summarize_auc_by_k(fold_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (label, top_k), group in fold_rows.groupby(["label", "top_n"], sort=True):
        stats = _series_stats(group["probe_auc"])
        rows.append(
            {
                "label": str(label),
                "subspace_ranking": "cohens_d",
                "top_k": int(top_k),
                **stats,
                "row_type": "label",
            }
        )
    return pd.DataFrame(rows).sort_values(["label", "top_k"]).reset_index(drop=True)


def _best_within(curve: pd.DataFrame, *, max_k: int) -> pd.DataFrame:
    label_curve = curve[
        (curve["row_type"].astype(str) == "label")
        & (curve["subspace_ranking"].astype(str) == "cohens_d")
        & (pd.to_numeric(curve["top_k"], errors="coerce") <= int(max_k))
    ].copy()
    label_curve["auc_mean"] = pd.to_numeric(label_curve["auc_mean"], errors="coerce")
    label_curve["top_k"] = pd.to_numeric(label_curve["top_k"], errors="coerce").astype(int)
    idx = label_curve.sort_values(["label", "auc_mean", "top_k"], ascending=[True, False, True]).groupby("label").head(1).index
    out = label_curve.loc[idx, ["label", "top_k", "auc_mean", "auc_std", "n_folds"]].copy()
    out = out.rename(
        columns={
            "top_k": f"best_k_le_{max_k}",
            "auc_mean": f"best_auc_le_{max_k}",
            "auc_std": f"best_auc_std_le_{max_k}",
            "n_folds": f"n_folds_le_{max_k}",
        }
    )
    return out.sort_values("label").reset_index(drop=True)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = FullRepresentationProbeConfig(
        labels=tuple(label.upper() for label in args.labels),
        folds=args.folds,
        split_policy=args.split_policy,
        group_column=args.group_column,
        C=args.C,
        solver=args.solver,
        max_iter=args.max_iter,
        random_state=args.random_state,
        standardize=True,
        verbose=not args.quiet,
        include_sae_ranked_subspaces=True,
        sae_subspace_top_ns=tuple(range(int(args.start_k), int(args.max_k) + 1)),
        sae_subspace_rankings=("cohens_d",),
        association_chunk_size=args.association_chunk_size,
        filter_sae_subspace_candidates=True,
        sae_subspace_n_jobs=args.subspace_n_jobs,
    )

    print(f"[load] SAE features: {args.sae_features}", flush=True)
    sae_features = load_matrix(args.sae_features)
    label_df = pd.read_csv(args.label_matrix)
    labels = _select_available_labels(label_df, args.labels)
    top_ns = tuple(range(int(args.start_k), int(args.max_k) + 1))
    fold_records: list[dict] = []
    warnings: list[str] = []

    for label in labels:
        y = label_df[label].astype(int).to_numpy()
        splits, split_policy_used, split_warnings = _make_splits(y, label_df, config)
        warnings.extend(f"{label}: {warning}" for warning in split_warnings)
        print(f"[label] {label} folds={len(splits)}", flush=True)
        for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
            sae_train, sae_test = _prepare_fold_features(
                sae_features,
                train_idx,
                test_idx,
                standardize=config.standardize,
            )
            filter_audit = build_feature_filter(sae_features[train_idx], FeatureFilterConfig())
            candidate_indices = filter_audit.loc[filter_audit["keep"].astype(bool), "latent_idx"].to_numpy(dtype=np.int32)
            if candidate_indices.size == 0:
                warnings.append(f"{label}: fold {fold_id} has no candidates after filtering")
                continue
            candidate_train = np.ascontiguousarray(sae_train[:, candidate_indices], dtype=np.float32)
            metrics = _chunked_sae_train_associations(
                candidate_train,
                y[train_idx],
                chunk_size=config.association_chunk_size,
            )
            order = _rank_sae_features(metrics, "cohens_d")
            selected = order[: int(args.max_k)]
            selected_original = candidate_indices[selected]
            sub_train = np.ascontiguousarray(candidate_train[:, selected], dtype=np.float32)
            sub_test = np.ascontiguousarray(sae_test[:, selected_original], dtype=np.float32)
            if config.sae_subspace_n_jobs and int(config.sae_subspace_n_jobs) != 1:
                from joblib import Parallel, delayed

                rows = Parallel(n_jobs=int(config.sae_subspace_n_jobs), prefer="threads")(
                    delayed(_score_sae_top_n_subspace)(
                        top_n=top_n,
                        ranking="cohens_d",
                        label=label,
                        fold=fold_id,
                        sub_train=sub_train,
                        sub_test=sub_test,
                        y_train=y[train_idx],
                        y_test=y[test_idx],
                        split_policy_used=split_policy_used,
                        candidate_pool_size=int(candidate_indices.size),
                        candidate_filter_enabled=True,
                        config=config,
                    )
                    for top_n in top_ns
                )
                fold_records.extend(rows)
            else:
                for top_n in top_ns:
                    fold_records.append(
                        _score_sae_top_n_subspace(
                            top_n=top_n,
                            ranking="cohens_d",
                            label=label,
                            fold=fold_id,
                            sub_train=sub_train,
                            sub_test=sub_test,
                            y_train=y[train_idx],
                            y_test=y[test_idx],
                            split_policy_used=split_policy_used,
                            candidate_pool_size=int(candidate_indices.size),
                            candidate_filter_enabled=True,
                            config=config,
                        )
                    )

    incremental = pd.DataFrame(fold_records)
    incremental_by_k = _summarize_auc_by_k(incremental)
    base_curve = pd.read_csv(args.base_auc_curve)
    base_curve = base_curve[
        (base_curve["subspace_ranking"].astype(str) == "cohens_d")
        & (pd.to_numeric(base_curve["top_k"], errors="coerce") < int(args.start_k))
    ].copy()
    combined = pd.concat([base_curve, incremental_by_k], ignore_index=True, sort=False)
    combined = combined.sort_values(["row_type", "subspace_ranking", "label", "top_k"]).reset_index(drop=True)

    compare_frames = [_best_within(combined, max_k=k) for k in args.compare_k]
    comparison = compare_frames[0]
    for frame in compare_frames[1:]:
        comparison = comparison.merge(frame, on="label", how="outer")
    base_k = int(args.compare_k[0])
    for k in args.compare_k[1:]:
        comparison[f"delta_best_auc_{k}_vs_{base_k}"] = (
            pd.to_numeric(comparison[f"best_auc_le_{k}"], errors="coerce")
            - pd.to_numeric(comparison[f"best_auc_le_{base_k}"], errors="coerce")
        )
        comparison[f"delta_best_k_{k}_vs_{base_k}"] = (
            pd.to_numeric(comparison[f"best_k_le_{k}"], errors="coerce")
            - pd.to_numeric(comparison[f"best_k_le_{base_k}"], errors="coerce")
        )

    incremental.to_csv(output_dir / "fold_auc_k_101_200.csv", index=False, encoding="utf-8-sig")
    incremental_by_k.to_csv(output_dir / "auc_by_k_101_200.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(output_dir / "auc_by_k_0_200_combined.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output_dir / "auc_upper_bound_100_150_200.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "analysis": "cohensd_auc_k_sensitivity",
        "question": "whether K<=100 truncates the Cohen's d AUC@K upper bound",
        "inputs": {
            "sae_features": args.sae_features,
            "label_matrix": args.label_matrix,
            "base_auc_curve": args.base_auc_curve,
        },
        "config": {
            "candidate_pool": "filtered train-fold SAE latent pool",
            "ranking": "cohens_d",
            "computed_k_range": [int(args.start_k), int(args.max_k)],
            "compare_k": list(map(int, args.compare_k)),
            "folds": int(args.folds),
            "split_policy": args.split_policy,
            "group_column": args.group_column,
            "random_state": int(args.random_state),
        },
        "warnings": warnings,
        "outputs": {
            "fold_auc_k_101_200": str(output_dir / "fold_auc_k_101_200.csv"),
            "auc_by_k_101_200": str(output_dir / "auc_by_k_101_200.csv"),
            "auc_by_k_0_200_combined": str(output_dir / "auc_by_k_0_200_combined.csv"),
            "auc_upper_bound_100_150_200": str(output_dir / "auc_upper_bound_100_150_200.csv"),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[done] AUC upper-bound comparison", flush=True)
    print(comparison.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
