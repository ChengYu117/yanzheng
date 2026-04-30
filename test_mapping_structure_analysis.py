"""Smoke tests for Mapping Structure analysis."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _synthetic_matrix() -> pd.DataFrame:
    labels = ["RE", "RES", "REC", "QU", "QUO", "OTHER"]
    latents = list(range(6))
    significant = {
        0: {"RE": 0.8},
        1: {"RE": 0.7, "RES": 0.6},
        2: {"RE": 0.5, "QU": 0.4},
        3: {"RE": 0.9, "RES": 0.8, "REC": 0.75, "QU": 0.7, "QUO": 0.65},
        4: {"OTHER": 0.55},
        5: {"RES": -0.5, "REC": -0.45},
    }
    rows = []
    for label in labels:
        for latent_idx in latents:
            d = significant.get(latent_idx, {}).get(label, 0.01)
            is_sig = label in significant.get(latent_idx, {})
            rows.append(
                {
                    "label": label,
                    "latent_idx": latent_idx,
                    "n_positive": 10,
                    "n_negative": 10,
                    "prevalence": 0.5,
                    "pos_mean": max(d, 0.0),
                    "neg_mean": 0.0,
                    "mean_diff": d,
                    "cohens_d": d,
                    "abs_cohens_d": abs(d),
                    "auc": 0.7 if d >= 0 else 0.3,
                    "directional_auc": 0.7,
                    "auc_effect": 0.4,
                    "p_value": 0.001 if is_sig else 0.9,
                    "significant_fdr": is_sig,
                    "precision_at_10": 0.8 if is_sig else 0.2,
                    "precision_lift_at_10": 0.3 if is_sig else -0.3,
                    "precision_at_50": 0.6 if is_sig else 0.2,
                    "precision_lift_at_50": 0.1 if is_sig else -0.3,
                }
            )
    return pd.DataFrame(rows)


def _write_mapping_dir(root: Path) -> Path:
    mapping_dir = root / "mapping"
    mapping_dir.mkdir(parents=True)
    matrix = _synthetic_matrix()
    matrix.to_csv(mapping_dir / "latent_label_matrix.csv", index=False)
    labels = []
    for label in ["RE", "RES", "REC", "QU", "QUO", "OTHER"]:
        labels.append(
            {
                "label": label,
                "n_positive": 10,
                "n_negative": 10,
                "prevalence": 0.5,
            }
        )
    (mapping_dir / "label_summary.json").write_text(
        json.dumps({"n_records": 20, "n_labels": len(labels), "labels": labels}),
        encoding="utf-8",
    )
    return mapping_dir


def test_mapping_structure_module():
    from nlp_re_base.mapping_structure import run_mapping_structure_analysis

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mapping_dir = _write_mapping_dir(root)
        out_dir = root / "out"
        metrics = run_mapping_structure_analysis(
            mapping_dir=mapping_dir,
            output_dir=out_dir,
            top_k_values=[2, 3],
            hierarchy_specs=["RE:RES,REC", "QU:QUO"],
            core_labels=["RE", "RES", "REC", "QU", "QUO"],
            doc_report=root / "report.md",
            make_figures=False,
        )

        assert metrics["significant_latent_label_edges"] == 13
        assert metrics["latents_with_any_significant_label"] == 6
        assert metrics["single_label_latents"] == 2
        assert metrics["multi_label_latents"] == 4
        assert (out_dir / "label_fragmentation_rank.csv").exists()
        assert (out_dir / "latent_overlap_distribution.csv").exists()
        assert (out_dir / "label_pair_similarity.csv").exists()
        assert (out_dir / "hierarchy_alignment.csv").exists()
        assert (out_dir / "latent_role_summary.csv").exists()
        assert (out_dir / "mapping_structure_report.md").exists()
        assert (root / "report.md").exists()

        frag = pd.read_csv(out_dir / "label_fragmentation_rank.csv")
        re_row = frag[frag["label"] == "RE"].iloc[0]
        assert int(re_row["n_significant_latents"]) == 4

        roles = pd.read_csv(out_dir / "latent_role_summary.csv")
        role_counts = dict(zip(roles["role"], roles["n_latents"]))
        assert role_counts["exclusive"] == 1
        assert role_counts["family_shared"] == 2
        assert role_counts["cross_family"] == 1
        assert role_counts["global"] == 1
        assert role_counts["auxiliary_only"] == 1

        hierarchy = pd.read_csv(out_dir / "hierarchy_alignment.csv")
        re_summary = hierarchy[
            (hierarchy["relation_type"] == "parent_summary") & (hierarchy["parent"] == "RE")
        ].iloc[0]
        assert abs(float(re_summary["parent_decomposition"]) - 0.5) < 1e-9


def test_mapping_structure_cli():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mapping_dir = _write_mapping_dir(root)
        out_dir = root / "cli_out"
        doc_report = root / "cli_report.md"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "run_misc_mapping_structure_analysis.py"),
            "--mapping-dir",
            str(mapping_dir),
            "--output-dir",
            str(out_dir),
            "--top-k",
            "2",
            "3",
            "--label-hierarchy",
            "RE:RES,REC",
            "QU:QUO",
            "--core-labels",
            "RE",
            "RES",
            "REC",
            "QU",
            "QUO",
            "--doc-report",
            str(doc_report),
            "--no-figures",
        ]
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        metrics = json.loads((out_dir / "mapping_structure_metrics.json").read_text())
        assert metrics["significant_latent_label_edges"] == 13
        assert doc_report.exists()


def main() -> int:
    tests = [test_mapping_structure_module, test_mapping_structure_cli]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {test.__name__}")
            raise
    print(f"Results: {len(tests) - failures} passed, {failures} failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
