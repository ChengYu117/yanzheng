"""Smoke tests for the MISC Latent x Label mapping path."""

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


def _write_synthetic_misc_dataset(root: Path) -> list[dict]:
    ann_dir = root / "misc_annotations" / "high"
    ann_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    specs = []
    specs.extend([("RE", "RES")] * 6)
    specs.extend([("RE", "REC")] * 6)
    specs.extend([("QU", "QUO")] * 6)
    specs.extend([("QU", "QUC")] * 6)
    specs.extend([("GI", "")] * 4)
    specs.extend([("SU", "")] * 4)
    specs.extend([("AF", "")] * 4)

    for i, (code, subcode) in enumerate(specs):
        rec = {
            "file_id": "high_001",
            "unit_text": f"synthetic counselor utterance {i} for {code} {subcode}".strip(),
            "predicted_code": code,
            "predicted_subcode": subcode,
            "rationale": "synthetic",
            "confidence": 0.9,
        }
        records.append(rec)

    with (ann_dir / "high_001.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return records


def _synthetic_features(n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    features = rng.normal(0.0, 0.15, size=(n, 16)).astype(np.float32)
    features[:, :] = np.maximum(features, 0.0)

    # Match the synthetic label order from _write_synthetic_misc_dataset.
    features[:12, 0] += 3.0      # RE
    features[12:24, 1] += 3.0    # QU
    features[:6, 2] += 2.5       # RES
    features[6:12, 3] += 2.5     # REC
    return features


def test_misc_label_mapping_module():
    from nlp_re_base.misc_label_mapping import (
        load_misc_annotation_records,
        run_misc_label_mapping,
        select_labels,
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw_records = _write_synthetic_misc_dataset(root)
        features = _synthetic_features(len(raw_records))

        records = load_misc_annotation_records(root)
        labels, indicators = select_labels(records, labels=["RE", "QU", "RES", "REC"])

        assert len(records) == len(raw_records)
        assert labels == ["RE", "QU", "RES", "REC"]
        assert indicators.shape == (len(raw_records), 4)

        out_dir = root / "out"
        summary = run_misc_label_mapping(
            records=records,
            features=features,
            output_dir=out_dir,
            labels=labels,
            min_positive=3,
            min_negative=3,
            precision_k_values=[3, 6],
            chunk_size=4,
            top_k_per_label=5,
            top_example_latents=2,
            top_examples_per_latent=3,
        )

        matrix = pd.read_csv(out_dir / "latent_label_matrix.csv")
        assert (out_dir / "label_summary.json").exists()
        assert (out_dir / "label_fragmentation.json").exists()
        assert (out_dir / "latent_overlap.json").exists()
        assert (out_dir / "behavior_asymmetry.md").exists()
        assert (out_dir / "top_latents_by_label" / "RE.csv").exists()
        assert summary["feature_shape"] == [len(raw_records), 16]

        top_re = int(matrix[matrix["label"] == "RE"].iloc[0]["latent_idx"])
        top_qu = int(matrix[matrix["label"] == "QU"].iloc[0]["latent_idx"])
        top_res = int(matrix[matrix["label"] == "RES"].iloc[0]["latent_idx"])
        top_rec = int(matrix[matrix["label"] == "REC"].iloc[0]["latent_idx"])

        assert top_re == 0, top_re
        assert top_qu == 1, top_qu
        assert top_res == 2, top_res
        assert top_rec == 3, top_rec


def test_misc_label_mapping_cli_with_precomputed_features():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw_records = _write_synthetic_misc_dataset(root)
        features = _synthetic_features(len(raw_records))
        features_path = root / "features.pt"
        torch.save({"utterance_features": torch.from_numpy(features)}, features_path)

        out_dir = root / "cli_out"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "run_misc_label_mapping.py"),
            "--data-dir",
            str(root),
            "--features-path",
            str(features_path),
            "--output-dir",
            str(out_dir),
            "--labels",
            "RE",
            "QU",
            "RES",
            "REC",
            "--min-positive",
            "3",
            "--min-negative",
            "3",
            "--precision-k",
            "3",
            "6",
            "--chunk-size",
            "4",
            "--top-k-per-label",
            "5",
            "--top-example-latents",
            "1",
            "--top-examples-per-latent",
            "2",
        ]
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        assert (out_dir / "latent_label_matrix.csv").exists()
        assert (out_dir / "run_config.json").exists()
        assert (out_dir / "run_summary.json").exists()


def main() -> int:
    tests = [
        test_misc_label_mapping_module,
        test_misc_label_mapping_cli_with_precomputed_features,
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
            raise
    print(f"Results: {len(tests) - failures} passed, {failures} failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
