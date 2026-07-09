from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.nlp_re_base.contrastive_evidence_pack import (
    ContrastiveEvidenceConfig,
    FORBIDDEN_EXPLAINER_KEYS,
    build_contrastive_evidence_packs,
    pack_summary_rows,
    run_build_contrastive_evidence_packs,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _safe_smoke_root() -> Path:
    root = (Path.cwd() / "outputs" / "_smoke_contrastive_evidence_pack").resolve()
    cwd = Path.cwd().resolve()
    if cwd not in root.parents:
        raise RuntimeError(f"Refusing smoke output outside repo: {root}")
    return root


def _toy_inputs() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, list[dict]]:
    texts = []
    rows = []
    records = []
    for idx in range(40):
        if idx < 10:
            text = f"What would help you explore option {idx}?"
            labels = {"QU": 1, "QUO": 1, "QUC": 0}
        elif idx < 18:
            text = f"Do you want option {idx}?"
            labels = {"QU": 1, "QUO": 0, "QUC": 1}
        elif idx < 28:
            text = f"How does that question compare {idx}?"
            labels = {"QU": 1, "QUO": 0, "QUC": 1}
        elif idx < 34:
            text = f"What else matters for low target {idx}?"
            labels = {"QU": 1, "QUO": 1, "QUC": 0}
        else:
            text = f"plain statement {idx}"
            labels = {"QU": 0, "QUO": 0, "QUC": 0}
        row = {
            "row_idx": idx,
            "record_id": f"r{idx}",
            "file_id": f"f{idx // 4}",
            "source_split": "toy",
            "unit_text": text,
            **labels,
        }
        texts.append(text)
        rows.append(row)
        records.append(dict(row))

    latent0 = np.asarray(
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1.5, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        + [0.45, 0.42, 0.4, 0.38, 0.36, 0.34, 0.32, 0.30, 0.28, 0.26, 0.24, 0.22]
        + [0.20, 0.18, 0.16, 0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    latent1 = latent0[::-1].copy()
    features = np.stack([latent0, latent1], axis=1).astype(np.float32)
    latents = pd.DataFrame(
        [
            {
                "label": "QUO",
                "latent_idx": 0,
                "rank_within_label": 1,
                "cohens_d": 1.1,
                "auc": 0.8,
                "directional_auc": 0.8,
                "precision_at_50": 0.5,
                "inclusion_frequency": 1.0,
                "stable_set_role": "stable_core",
            },
            {
                "label": "QUO",
                "latent_idx": 1,
                "rank_within_label": 2,
                "cohens_d": 0.7,
                "auc": 0.7,
                "directional_auc": 0.7,
                "precision_at_50": 0.4,
                "inclusion_frequency": 0.5,
                "stable_set_role": "boundary_candidate",
            },
            {
                "label": "QUC",
                "latent_idx": 1,
                "rank_within_label": 1,
                "cohens_d": 1.0,
                "auc": 0.78,
                "directional_auc": 0.78,
                "precision_at_50": 0.5,
                "inclusion_frequency": 0.9,
                "stable_set_role": "stable_core",
            },
        ]
    )
    return latents, features, pd.DataFrame(rows), records


def test_contrastive_evidence_pack() -> None:
    root = _safe_smoke_root()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    try:
        latents, features, label_matrix, records = _toy_inputs()
        config = ContrastiveEvidenceConfig(
            labels=("QU", "QUO", "QUC"),
            expected_stable_core_count=None,
            active_high=2,
            active_mid=2,
            active_low=1,
            near_miss_surface=2,
            nonactive_label_match=1,
            nonactive_random=1,
            heldout_active_high=2,
            heldout_active_mid=2,
            heldout_near_miss=2,
            heldout_label_match=1,
        )
        packs = build_contrastive_evidence_packs(
            latents=latents,
            features=features,
            label_matrix=label_matrix,
            records=records,
            config=config,
        )
        assert len(packs) == 2
        assert {pack["stable_set_role"] for pack in packs} == {"stable_core"}
        assert {pack["target_label"] for pack in packs} == {"QUO", "QUC"}

        for pack in packs:
            visible = pack["samples_for_explainer"]
            assert visible
            for sample in visible:
                assert set(sample) == {"id", "tag", "activation", "text"}
                assert set(sample).isdisjoint(FORBIDDEN_EXPLAINER_KEYS)
                assert sample["tag"] != "NONACTIVE_LABEL_MATCH"
            evidence_rows = {int(sample["row_idx"]) for sample in pack["samples_internal"]}
            heldout_rows = {
                int(sample["row_idx"])
                for samples in pack["heldout_internal_by_tag"].values()
                for sample in samples
            }
            assert evidence_rows.isdisjoint(heldout_rows)
            assert {"ACTIVE_HIGH", "ACTIVE_MID", "ACTIVE_LOW", "NONACTIVE_NEAR_MISS", "NONACTIVE_RANDOM"}.issubset(
                {sample["tag"] for sample in pack["samples_internal"]}
            )
            assert "NONACTIVE_LABEL_MATCH" in pack["heldout_internal_by_tag"]

        summary = pd.DataFrame(pack_summary_rows(packs))
        assert summary["n_evidence_rows"].min() > 0
        assert summary["n_heldout_rows"].min() > 0

        latents_path = root / "latents.csv"
        features_path = root / "features.npy"
        labels_path = root / "labels.csv"
        records_path = root / "records.jsonl"
        out_dir = root / "out"
        latents.to_csv(latents_path, index=False)
        np.save(features_path, features)
        label_matrix.to_csv(labels_path, index=False)
        _write_jsonl(records_path, records)
        manifest = run_build_contrastive_evidence_packs(
            latents_path=latents_path,
            feature_store_path=features_path,
            label_matrix_path=labels_path,
            records_path=records_path,
            output_dir=out_dir,
            config=config,
        )
        assert manifest["n_packs"] == 2
        assert (out_dir / "evidence_packs" / "contrastive_evidence_packs.jsonl").exists()
        assert (out_dir / "evidence_packs" / "contrastive_evidence_pack_summary.csv").exists()
        assert (out_dir / "manifest.json").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_contrastive_evidence_pack()
    print("test_contrastive_evidence_pack_smoke passed")
