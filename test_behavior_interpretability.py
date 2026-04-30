"""Smoke tests for follow-up MISC interpretability analyses."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _make_synthetic_eval(root: Path) -> Path:
    eval_dir = root / "eval"
    mapping_dir = eval_dir / "functional" / "misc_label_mapping"
    structure_dir = eval_dir / "interpretability" / "mapping_structure"
    feature_dir = eval_dir / "feature_store"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    records = []
    labels = ["RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF"]
    specs = [
        ("RE", ["RE", "REC"], "high", "it sounds like you feel stuck"),
        ("RE", ["RE", "RES"], "low", "you are tired"),
        ("QU", ["QU", "QUO"], "high", "what would help you today"),
        ("QU", ["QU", "QUC"], "low", "do you want to stop"),
        ("AF", ["AF"], "high", "that is a great step"),
        ("SU", ["SU"], "low", "you should call tomorrow"),
        ("GI", ["GI"], "high", "the program starts monday"),
        ("OTHER", ["OTHER"], "low", "okay"),
    ]
    for i, (code, rec_labels, quality, text) in enumerate(specs):
        records.append(
            {
                "record_id": f"r{i}",
                "sample_id": f"r{i}",
                "file_id": f"f{i // 2}",
                "text": text,
                "unit_text": text,
                "predicted_code": code,
                "predicted_subcode": rec_labels[-1] if len(rec_labels) > 1 else "",
                "quality_label": quality,
                "labels": rec_labels,
            }
        )
    _write_jsonl(eval_dir / "records.jsonl", records)

    features = np.zeros((len(records), 12), dtype=np.float32)
    features[0:2, 0] = [3.0, 2.7]  # RE
    features[0:1, 1] = [2.5]       # REC
    features[1:2, 2] = [2.4]       # RES
    features[2:4, 3] = [3.1, 2.9]  # QU
    features[2:3, 4] = [2.8]       # QUO
    features[3:4, 5] = [2.6]       # QUC
    features[4:5, 6] = [3.2]       # AF
    features[5:6, 7] = [3.3]       # SU
    features[6:7, 8] = [3.4]       # GI
    rng = np.random.default_rng(4)
    features += rng.normal(0.0, 0.05, size=features.shape).astype(np.float32)
    torch.save({"utterance_features": torch.from_numpy(features)}, feature_dir / "utterance_features.pt")

    matrix_rows = []
    effects = {
        "RE": {0: 1.0, 1: 0.45, 2: 0.35, 3: 0.2},
        "REC": {1: 0.9, 0: 0.4},
        "RES": {2: 0.85, 0: 0.3},
        "QU": {3: 1.1, 4: 0.45, 5: 0.45, 0: 0.2},
        "QUO": {4: 0.95, 3: 0.4},
        "QUC": {5: 0.9, 3: 0.4},
        "AF": {6: 1.2},
        "SU": {7: 1.1},
        "GI": {8: 1.0},
        "OTHER": {9: 0.8},
    }
    for label in ["RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF", "OTHER"]:
        for latent in range(12):
            d = effects.get(label, {}).get(latent, 0.01)
            sig = latent in effects.get(label, {})
            matrix_rows.append(
                {
                    "label": label,
                    "latent_idx": latent,
                    "n_positive": 2,
                    "n_negative": 6,
                    "prevalence": 0.25,
                    "pos_mean": d,
                    "neg_mean": 0.0,
                    "mean_diff": d,
                    "cohens_d": d,
                    "abs_cohens_d": abs(d),
                    "auc": 0.75 if d > 0 else 0.25,
                    "directional_auc": 0.75,
                    "auc_effect": 0.5,
                    "p_value": 0.001 if sig else 0.9,
                    "significant_fdr": sig,
                    "precision_at_10": 0.8 if sig else 0.2,
                    "precision_lift_at_10": 0.55 if sig else -0.05,
                    "precision_at_50": 0.5 if sig else 0.2,
                    "precision_lift_at_50": 0.25 if sig else -0.05,
                }
            )
    pd.DataFrame(matrix_rows).to_csv(mapping_dir / "latent_label_matrix.csv", index=False)
    _write_json(
        mapping_dir / "label_summary.json",
        {
            "labels": [
                {"label": label, "n_positive": 2, "n_negative": 6, "prevalence": 0.25}
                for label in ["RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF", "OTHER"]
            ]
        },
    )

    pd.DataFrame(
        [
            {
                "label": label,
                "n_significant_latents": len(effects.get(label, {})),
                "n_positive_effect_significant": len(effects.get(label, {})),
                "n_negative_effect_significant": 0,
                "fragmentation_ratio": len(effects.get(label, {})) / 12,
                "positive_effect_ratio": 1.0,
                "negative_effect_ratio": 0.0,
                "top_latent_idx": max(effects[label], key=effects[label].get),
                "top_abs_cohens_d": max(effects[label].values()),
                "top_directional_auc": 0.75,
            }
            for label in effects
        ]
    ).to_csv(structure_dir / "label_fragmentation_rank.csv", index=False)
    pd.DataFrame(
        [
            {
                "label_a": "RE",
                "label_b": "REC",
                "top_k": 50,
                "topk_jaccard": 0.4,
                "significant_jaccard": 0.3,
                "pearson_cohens_d": 0.7,
                "spearman_cohens_d": 0.6,
            },
            {
                "label_a": "QU",
                "label_b": "QUO",
                "top_k": 50,
                "topk_jaccard": 0.4,
                "significant_jaccard": 0.3,
                "pearson_cohens_d": 0.7,
                "spearman_cohens_d": 0.6,
            },
        ]
    ).to_csv(structure_dir / "label_pair_similarity.csv", index=False)
    pd.DataFrame(
        [
            {
                "latent_idx": i,
                "n_labels": 1,
                "labels": label,
                "role": "exclusive",
                "direction_type": "positive_only",
                "max_abs_cohens_d": 1.0,
                "max_directional_auc": 0.75,
            }
            for i, label in enumerate(labels)
        ]
    ).to_csv(structure_dir / "latent_role_assignments.csv", index=False)
    return eval_dir


def test_followup_interpretability_module():
    from nlp_re_base.behavior_interpretability import run_followup_interpretability_analysis

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        eval_dir = _make_synthetic_eval(root)
        out_dir = root / "out"
        doc_report = root / "doc_report.md"
        metrics = run_followup_interpretability_analysis(
            eval_dir=eval_dir,
            output_dir=out_dir,
            labels=["RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF"],
            top_latents_per_label=2,
            top_examples_per_latent=4,
            doc_report=doc_report,
        )
        assert metrics["case_metrics"]["n_case_cards"] >= 9
        assert (out_dir / "behavior_asymmetry" / "behavior_asymmetry_summary.csv").exists()
        assert (out_dir / "latent_cases" / "latent_case_summary.csv").exists()
        assert (out_dir / "followup_interpretability_report.md").exists()
        assert doc_report.exists()


def test_followup_interpretability_cli():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        eval_dir = _make_synthetic_eval(root)
        out_dir = root / "cli_out"
        doc_report = root / "cli_doc.md"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "run_misc_interpretability_analysis.py"),
            "--eval-dir",
            str(eval_dir),
            "--output-dir",
            str(out_dir),
            "--labels",
            "RE",
            "RES",
            "REC",
            "QU",
            "QUO",
            "QUC",
            "GI",
            "SU",
            "AF",
            "--top-latents-per-label",
            "2",
            "--top-examples-per-latent",
            "4",
            "--doc-report",
            str(doc_report),
        ]
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        assert (out_dir / "followup_interpretability_metrics.json").exists()
        assert doc_report.exists()


def main() -> int:
    tests = [test_followup_interpretability_module, test_followup_interpretability_cli]
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
