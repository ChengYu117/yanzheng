"""Tests for MISC causal candidate export."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _write_synthetic_inputs(root: Path) -> tuple[Path, Path, Path]:
    eval_dir = root / "eval"
    mapping_dir = eval_dir / "functional" / "misc_label_mapping"
    structure_dir = eval_dir / "interpretability" / "mapping_structure"
    followup_dir = eval_dir / "interpretability" / "followup_analysis"
    mapping_dir.mkdir(parents=True)
    structure_dir.mkdir(parents=True)
    (followup_dir / "latent_cases").mkdir(parents=True)
    (followup_dir / "behavior_asymmetry").mkdir(parents=True)

    rows = []
    for label in ["RE", "QU"]:
        for idx in range(30):
            sign = 1 if idx < 24 else -1
            rows.append(
                {
                    "label": label,
                    "latent_idx": idx + (100 if label == "QU" else 0),
                    "n_positive": 10,
                    "n_negative": 10,
                    "prevalence": 0.5,
                    "pos_mean": 1.0,
                    "neg_mean": 0.1,
                    "mean_diff": sign * 0.5,
                    "cohens_d": sign * (2.0 - idx * 0.02),
                    "abs_cohens_d": 2.0 - idx * 0.02,
                    "auc": 0.9 - idx * 0.005,
                    "directional_auc": 0.9 - idx * 0.005,
                    "auc_effect": 0.8,
                    "p_value": 0.001,
                    "significant_fdr": True,
                    "precision_at_10": 0.8,
                    "precision_lift_at_10": 0.3,
                    "precision_at_50": 0.7,
                    "precision_lift_at_50": 0.2,
                }
            )
    pd.DataFrame(rows).to_csv(mapping_dir / "latent_label_matrix.csv", index=False)

    case_rows = [
        {
            "label": "RE",
            "latent_idx": 0,
            "top_example_target_purity": 0.9,
            "dominant_labels_top_examples": "RE:9",
            "quality_distribution_top_examples": "high:8,low:2",
            "interpretation_status": "high_purity_candidate",
            "card_path": "cards/re_0.md",
        },
        {
            "label": "QU",
            "latent_idx": 100,
            "top_example_target_purity": 0.8,
            "dominant_labels_top_examples": "QU:8",
            "quality_distribution_top_examples": "high:5,low:5",
            "interpretation_status": "high_purity_candidate",
            "card_path": "cards/qu_100.md",
        },
    ]
    pd.DataFrame(case_rows).to_csv(
        followup_dir / "latent_cases" / "latent_case_summary.csv",
        index=False,
    )

    behavior_rows = [
        {
            "label": "RE",
            "pattern": "shared_distributed",
            "top_positive_latents": ",".join(str(i) for i in range(20)),
        },
        {
            "label": "QU",
            "pattern": "compact_strong",
            "top_positive_latents": ",".join(str(100 + i) for i in range(20)),
        },
    ]
    pd.DataFrame(behavior_rows).to_csv(
        followup_dir / "behavior_asymmetry" / "behavior_asymmetry_summary.csv",
        index=False,
    )

    role_rows = [
        {
            "latent_idx": i,
            "n_labels": 1,
            "labels": "RE",
            "positive_labels": "RE",
            "negative_labels": "",
            "direction_type": "positive_only",
            "role": "exclusive",
        }
        for i in range(30)
    ]
    role_rows += [
        {
            "latent_idx": 100 + i,
            "n_labels": 1,
            "labels": "QU",
            "positive_labels": "QU",
            "negative_labels": "",
            "direction_type": "positive_only",
            "role": "exclusive",
        }
        for i in range(30)
    ]
    pd.DataFrame(role_rows).to_csv(structure_dir / "latent_role_assignments.csv", index=False)
    return eval_dir, mapping_dir, structure_dir


def test_causal_candidate_export_module() -> None:
    from nlp_re_base.causal_candidates import export_misc_causal_candidates

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        eval_dir, _, _ = _write_synthetic_inputs(root)
        output_dir = root / "out"
        payload = export_misc_causal_candidates(
            eval_dir=eval_dir,
            output_dir=output_dir,
            labels=["RE", "QU"],
            doc_report=root / "doc.md",
        )
        assert (output_dir / "causal_candidate_groups.json").exists()
        assert (output_dir / "candidate_group_summary.csv").exists()
        assert payload["groups"]["RE"]["candidate_groups"]["G20"][0] == 0
        assert len(payload["groups"]["RE"]["controls"]["random"]) > 0
        assert not set(payload["groups"]["RE"]["controls"]["random"]) & set(
            payload["groups"]["RE"]["candidate_groups"]["G20"]
        )
        summary = pd.read_csv(output_dir / "candidate_group_summary.csv")
        assert set(summary["label"]) == {"RE", "QU"}


def test_causal_candidate_export_cli() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        eval_dir, _, _ = _write_synthetic_inputs(root)
        output_dir = root / "cli_out"
        cmd = [
            sys.executable,
            "run_misc_causal_candidate_export.py",
            "--eval-dir",
            str(eval_dir),
            "--output-dir",
            str(output_dir),
            "--labels",
            "RE",
            "QU",
            "--doc-report",
            str(root / "doc.md"),
        ]
        result = subprocess.run(cmd, cwd=Path(__file__).parent, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        payload = json.loads((output_dir / "causal_candidate_groups.json").read_text(encoding="utf-8"))
        assert payload["groups"]["QU"]["candidate_groups"]["G5"][0] == 100


def main() -> int:
    tests = [
        test_causal_candidate_export_module,
        test_causal_candidate_export_cli,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # pragma: no cover
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    print(f"Results: {len(tests) - failed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
