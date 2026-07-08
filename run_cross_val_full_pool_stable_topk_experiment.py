"""Run stable Top-K selection on the unfiltered full MISC SAE latent pool."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.cross_val_framework import (  # noqa: E402
    DEFAULT_LABELS,
    load_full_inputs,
    save_dedup_group_mapping,
)
from nlp_re_base.cross_quality_val import run_cross_quality_validation  # noqa: E402
from nlp_re_base.effect_size_ci import run_bootstrap_ci  # noqa: E402
from nlp_re_base.stable_topk_selection import (  # noqa: E402
    StableTopKSelectionConfig,
    run_stable_topk_selection,
)
from nlp_re_base.topk_reproducibility import run_topk_reproducibility  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-store", default="outputs/misc_full_sae_eval/feature_store/utterance_features.pt")
    parser.add_argument("--raw-hidden", default="outputs/misc_full_sae_eval/feature_store/utterance_activations.pt")
    parser.add_argument("--label-matrix", default="outputs/misc_full_sae_eval/label_matrix.csv")
    parser.add_argument(
        "--association-matrix",
        default="outputs/misc_full_sae_eval/functional/misc_label_mapping/latent_label_matrix.csv",
        help="Full, unfiltered label-latent association matrix.",
    )
    parser.add_argument("--output-root", default="outputs/cross_val_full_pool")
    parser.add_argument(
        "--auc-output-dir",
        default="outputs/misc_full_sae_eval/interpretability/ranked_sae_subspace_probe_k001_100_full_pool",
    )
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--group-column", default="source_file")
    parser.add_argument("--probe-group-column", default="file_id")
    parser.add_argument("--ranking-metrics", nargs="+", default=["cohens_d"])
    parser.add_argument("--stable-ranking-metric", default="cohens_d")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-k", type=int, default=100)
    parser.add_argument("--n-repeats", type=int, default=50)
    parser.add_argument("--null-iter", type=int, default=10000)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-positive", type=int, default=10)
    parser.add_argument("--min-negative", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--subspace-n-jobs", type=int, default=8)
    parser.add_argument(
        "--no-fast-topk-associations",
        action="store_true",
        help="Keep p-value/FDR and precision@k columns in E1 split-half matrices.",
    )
    parser.add_argument(
        "--skip-auc-probe",
        action="store_true",
        help="Reuse an existing auc_by_k_curve_0_100.csv in --auc-output-dir.",
    )
    return parser.parse_args()


def _run_auc_probe(args: argparse.Namespace) -> Path:
    output_dir = Path(args.auc_output_dir)
    auc_curve = output_dir / "auc_by_k_curve_0_100.csv"
    if args.skip_auc_probe:
        if not auc_curve.exists():
            raise FileNotFoundError(f"--skip-auc-probe requested but missing: {auc_curve}")
        print(f"[E0] reusing full-pool AUC curve: {auc_curve}")
        return auc_curve

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_misc_full_representation_probe.py"),
        "--sae-features",
        args.feature_store,
        "--raw-hidden",
        args.raw_hidden,
        "--label-matrix",
        args.label_matrix,
        "--output-dir",
        args.auc_output_dir,
        "--labels",
        *args.labels,
        "--group-column",
        args.probe_group_column,
        "--include-sae-ranked-subspaces",
        "--no-filter-sae-subspace-candidates",
        "--subspace-max-n",
        str(args.max_k),
        "--subspace-step",
        "1",
        "--subspace-rankings",
        *args.ranking_metrics,
        "--subspace-n-jobs",
        str(args.subspace_n_jobs),
        "--quiet",
    ]
    print("[E0] running full-pool AUC@K probe")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    if not auc_curve.exists():
        raise FileNotFoundError(f"full-pool AUC probe did not create expected file: {auc_curve}")
    return auc_curve


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    labels = tuple(label.upper() for label in args.labels)
    auc_curve = _run_auc_probe(args)

    dedup_path = output_root / "dedup_group_mapping.csv"
    print(f"[E0] writing dedup mapping: {dedup_path}")
    save_dedup_group_mapping(args.label_matrix, dedup_path)

    print("[load] full, unfiltered SAE latent pool")
    inputs = load_full_inputs(
        feature_store_path=args.feature_store,
        label_matrix_path=args.label_matrix,
        labels=labels,
    )
    print(f"[load] rows={len(inputs.label_matrix)} full_latents={len(inputs.latent_indices)}")

    top_k_grid = list(range(1, int(args.max_k) + 1))
    topk_dir = output_root / "topk_reproducibility"
    print(f"[E1] split-half TopK reproducibility: {topk_dir}")
    topk_manifest = run_topk_reproducibility(
        inputs,
        output_dir=topk_dir,
        reference_matrix_path=args.association_matrix,
        group_column=args.group_column,
        ranking_metrics=args.ranking_metrics,
        top_k=args.top_k,
        top_k_grid=top_k_grid,
        n_repeats=args.n_repeats,
        null_iter=args.null_iter,
        random_state=args.random_state,
        min_positive=args.min_positive,
        min_negative=args.min_negative,
        chunk_size=args.chunk_size,
        fast_associations=not args.no_fast_topk_associations,
    )

    bootstrap_dir = output_root / "bootstrap_ci"
    print(f"[E2] grouped bootstrap CI: {bootstrap_dir}")
    bootstrap_manifest = run_bootstrap_ci(
        inputs,
        association_matrix_path=args.association_matrix,
        output_dir=bootstrap_dir,
        group_column=args.group_column,
        ranking_metric=args.stable_ranking_metric,
        top_k=args.max_k,
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
        chunk_size=args.chunk_size,
    )

    quality_dir = output_root / "cross_quality_validation"
    print(f"[E3] cross-quality validation: {quality_dir}")
    quality_manifest = run_cross_quality_validation(
        inputs,
        output_dir=quality_dir,
        reference_matrix_path=args.association_matrix,
        ranking_metric=args.stable_ranking_metric,
        top_k=args.max_k,
        min_positive=args.min_positive,
        min_negative=args.min_negative,
        chunk_size=args.chunk_size,
    )

    stable_dir = output_root / "stable_topk_selection"
    print(f"[E4] stable Top-K selection: {stable_dir}")
    stable_result = run_stable_topk_selection(
        auc_curve_path=auc_curve,
        topk_grid_summary_path=topk_dir / "repeated_split_topk_grid_summary.csv",
        inclusion_frequency_path=topk_dir / "topk_inclusion_frequency.csv",
        association_matrix_path=args.association_matrix,
        bootstrap_ci_path=bootstrap_dir / "bootstrap_ci_by_label_latent.csv",
        cross_quality_path=quality_dir / "cross_quality_auc_comparison.csv",
        output_dir=stable_dir,
        config=StableTopKSelectionConfig(
            labels=labels,
            ranking_metric=args.stable_ranking_metric,
            max_k=args.max_k,
        ),
    )

    manifest = {
        "analysis": "full_pool_stable_topk_experiment",
        "candidate_pool": "full_unfiltered",
        "labels": list(labels),
        "n_rows": int(len(inputs.label_matrix)),
        "n_latents": int(len(inputs.latent_indices)),
        "inputs": {
            "feature_store": str(args.feature_store),
            "raw_hidden": str(args.raw_hidden),
            "label_matrix": str(args.label_matrix),
            "association_matrix": str(args.association_matrix),
            "auc_curve": str(auc_curve),
        },
        "outputs": {
            "output_root": str(output_root),
            "dedup_group_mapping": str(dedup_path),
            "auc_output_dir": str(args.auc_output_dir),
            "topk_reproducibility": str(topk_dir),
            "bootstrap_ci": str(bootstrap_dir),
            "cross_quality_validation": str(quality_dir),
            "stable_topk_selection": str(stable_dir),
        },
        "component_manifests": {
            "topk_reproducibility": topk_manifest,
            "bootstrap_ci": bootstrap_manifest,
            "cross_quality_validation": quality_manifest,
            "stable_topk_selection": stable_result["manifest"],
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done] Full-pool stable Top-K selection")
    print(stable_result["summary"].to_string(index=False))
    print(f"[done] manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
