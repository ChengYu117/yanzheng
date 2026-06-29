from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from run_misc_p3_feature_cards import run_p3_feature_card_export


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _safe_smoke_root() -> Path:
    root = (Path.cwd() / "outputs" / "_smoke_misc_p3_feature_cards").resolve()
    cwd = Path.cwd().resolve()
    if cwd not in root.parents:
        raise RuntimeError(f"Refusing to use smoke output outside repo: {root}")
    return root


def test_misc_p3_feature_cards() -> None:
    root = _safe_smoke_root()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    try:
        labels = ("QUO", "QUC", "QU")
        texts = [
            "What would make this change easier?",
            "How do you feel about that?",
            "Do you want to quit now?",
            "Are you ready?",
            "Can you do that?",
            "Did you take it?",
            "Tell me more about what matters.",
            "What else would help?",
            "plain low activation statement",
            "Is that correct?",
            "Why is that important?",
            "short ordinary statement",
        ]
        label_rows = []
        label_values = [
            {"QUO": 1, "QUC": 0, "QU": 1},
            {"QUO": 1, "QUC": 0, "QU": 1},
            {"QUO": 0, "QUC": 1, "QU": 1},
            {"QUO": 0, "QUC": 1, "QU": 1},
            {"QUO": 0, "QUC": 1, "QU": 1},
            {"QUO": 0, "QUC": 1, "QU": 1},
            {"QUO": 1, "QUC": 0, "QU": 1},
            {"QUO": 1, "QUC": 0, "QU": 1},
            {"QUO": 0, "QUC": 0, "QU": 0},
            {"QUO": 0, "QUC": 1, "QU": 1},
            {"QUO": 0, "QUC": 0, "QU": 0},
            {"QUO": 0, "QUC": 0, "QU": 0},
        ]
        records = []
        for idx, text in enumerate(texts):
            row = {
                "row_idx": idx,
                "record_id": f"r{idx}",
                "file_id": "toy",
                "source_line": idx + 1,
                "source_split": "toy",
                "quality_label": "toy",
                "predicted_code": "QU" if label_values[idx].get("QU") else "",
                "predicted_subcode": "QUO" if label_values[idx].get("QUO") else ("QUC" if label_values[idx].get("QUC") else ""),
                "confidence": 0.9,
                "unit_text": text,
                **label_values[idx],
            }
            label_rows.append(row)
            records.append(dict(row))

        features = np.asarray(
            [
                [10.0],
                [9.0],
                [8.0],
                [7.0],
                [6.0],
                [5.0],
                [0.2],
                [0.1],
                [0.0],
                [4.0],
                [3.0],
                [0.0],
            ],
            dtype=np.float32,
        )
        latents = pd.DataFrame(
            [
                {
                    "label": "QUO",
                    "rank_within_label": 1,
                    "latent_idx": 0,
                    "cohens_d": 1.25,
                    "directional_auc": 0.86,
                    "precision_at_50": 0.72,
                },
                {
                    "label": "QUC",
                    "rank_within_label": 1,
                    "latent_idx": 0,
                    "cohens_d": 0.95,
                    "directional_auc": 0.74,
                    "precision_at_50": 0.44,
                },
            ]
        )
        packet = {
            "packet_id": "packet_0001",
            "target_label": "QUO",
            "latent_idx": 0,
            "rank_within_label": 1,
            "cohens_d": 1.25,
            "directional_auc": 0.86,
            "precision_at_50": 0.72,
            "summary": {
                "top_activating_target_match_rate": 1.0,
                "duplicate_text_row_count": 0,
            },
            "examples": [
                {
                    "packet_id": "packet_0001",
                    "target_label": "QUO",
                    "latent_idx": 0,
                    "rank_within_label": 1,
                    "example_group": "top_activating",
                    "rank_within_group": 1,
                    "row_idx": 0,
                    "activation": 10.0,
                    "target_match": 1,
                    "active_labels": "QUO,QU",
                    "record_id": "r0",
                    "unit_text": texts[0],
                    "normalized_text": texts[0].lower(),
                },
                {
                    "packet_id": "packet_0001",
                    "target_label": "QUO",
                    "latent_idx": 0,
                    "rank_within_label": 1,
                    "example_group": "top_activating",
                    "rank_within_group": 2,
                    "row_idx": 1,
                    "activation": 9.0,
                    "target_match": 1,
                    "active_labels": "QUO,QU",
                    "record_id": "r1",
                    "unit_text": texts[1],
                    "normalized_text": texts[1].lower(),
                },
                {
                    "packet_id": "packet_0001",
                    "target_label": "QUO",
                    "latent_idx": 0,
                    "rank_within_label": 1,
                    "example_group": "high_non_target",
                    "rank_within_group": 1,
                    "row_idx": 2,
                    "activation": 8.0,
                    "target_match": 0,
                    "active_labels": "QUC,QU",
                    "record_id": "r2",
                    "unit_text": texts[2],
                    "normalized_text": texts[2].lower(),
                },
                {
                    "packet_id": "packet_0001",
                    "target_label": "QUO",
                    "latent_idx": 0,
                    "rank_within_label": 1,
                    "example_group": "random_target",
                    "rank_within_group": 1,
                    "row_idx": 6,
                    "activation": 0.2,
                    "target_match": 1,
                    "active_labels": "QUO,QU",
                    "record_id": "r6",
                    "unit_text": texts[6],
                    "normalized_text": texts[6].lower(),
                },
                {
                    "packet_id": "packet_0001",
                    "target_label": "QUO",
                    "latent_idx": 0,
                    "rank_within_label": 1,
                    "example_group": "random_target",
                    "rank_within_group": 2,
                    "row_idx": 7,
                    "activation": 0.1,
                    "target_match": 1,
                    "active_labels": "QUO,QU",
                    "record_id": "r7",
                    "unit_text": texts[7],
                    "normalized_text": texts[7].lower(),
                },
            ],
        }
        layer_selection = pd.DataFrame(
            [
                {
                    "target_label": "QUO",
                    "canonical_layer": 19,
                    "hook_point": "blocks.19.hook_resid_post",
                    "label_specific_best_layer": 13,
                    "label_specific_best_auc": 0.95,
                    "early_stable_layer": 2,
                    "early_stable_auc": 0.948,
                    "near_delta": 0.005,
                    "availability_note": "llama_cross_layer_probe_metrics_available",
                }
            ]
        )

        latents_path = root / "latents.csv"
        packets_path = root / "packets.jsonl"
        features_path = root / "features.npy"
        labels_path = root / "labels.csv"
        records_path = root / "records.jsonl"
        layer_path = root / "layer_selection.csv"
        out_dir = root / "out"

        latents.to_csv(latents_path, index=False)
        _write_jsonl(packets_path, [packet])
        np.save(features_path, features)
        pd.DataFrame(label_rows).to_csv(labels_path, index=False)
        _write_jsonl(records_path, records)
        layer_selection.to_csv(layer_path, index=False)

        manifest = run_p3_feature_card_export(
            latents_path=latents_path,
            packets_path=packets_path,
            feature_store_path=features_path,
            label_matrix_path=labels_path,
            records_path=records_path,
            layer_selection_path=layer_path,
            output_dir=out_dir,
            labels=labels,
            top_features=1,
            top_activating=2,
            high_non_target=1,
            random_target=2,
            sibling_contrast=2,
            surface_contrast=2,
            low_activation=2,
            prompt_examples_per_group=2,
            scoring_heldout_start=1,
            scoring_examples_per_class=2,
        )

        expected_files = [
            "p3_feature_card_packets.jsonl",
            "p3_feature_card_examples.csv",
            "p3_scoring_tasks.jsonl",
            "p3_feature_cards_dryrun.md",
            "p3_feature_card_summary.csv",
            "manifest.json",
        ]
        for name in expected_files:
            assert (out_dir / name).exists(), name
        assert (out_dir / "p3_input_explanation_prompts").is_dir()

        assert manifest["n_cards"] == 2
        assert manifest["n_expected_cards"] == 2
        assert manifest["n_fallback_packets"] == 1
        assert manifest["n_missing_packets"] == 0
        assert manifest["n_prompts"] == 2
        assert manifest["n_scoring_tasks"] == 4

        with (out_dir / "p3_feature_card_packets.jsonl").open(encoding="utf-8") as handle:
            cards = [json.loads(line) for line in handle if line.strip()]
        card = cards[0]
        fallback_card = cards[1]
        assert card["layer_metadata"]["sae_canonical_layer"] == 19
        assert card["layer_metadata"]["llama_label_specific_best_layer"] == 13
        assert "prior client context is unavailable" in card["context_limitation"].lower()
        assert fallback_card["source_packet_status"] == "fallback_generated_from_feature_store"
        assert len([ex for ex in fallback_card["examples"] if ex["example_group"] == "top_activating"]) == 2

        examples = pd.read_csv(out_dir / "p3_feature_card_examples.csv")
        groups = set(examples["example_group"])
        assert {"top_activating", "sibling_code_contrast", "surface_matched_contrast", "low_activation_scoring"}.issubset(groups)

        sibling = examples[examples["example_group"] == "sibling_code_contrast"]
        assert not sibling.empty
        assert sibling["target_match"].tolist() == [0] * len(sibling)
        assert any("QUC" in str(value) for value in sibling["active_labels"].tolist())

        surface = examples[examples["example_group"] == "surface_matched_contrast"]
        assert not surface.empty
        assert surface["target_match"].tolist() == [0] * len(surface)
        assert surface["surface_match_score"].min() > 0

        with (out_dir / "p3_scoring_tasks.jsonl").open(encoding="utf-8") as handle:
            tasks = [json.loads(line) for line in handle if line.strip()]
        assert {task["task_type"] for task in tasks} == {
            "activation_prediction_task",
            "code_discrimination_task",
        }
        assert {task["status"] for task in tasks} == {"pending_score"}

        prompt_files = list((out_dir / "p3_input_explanation_prompts").glob("*.json"))
        assert len(prompt_files) == 2
        prompt_text = prompt_files[0].read_text(encoding="utf-8")
        assert "Do not say" in prompt_text
        assert "this feature is QUO" not in prompt_text
        assert "target_misc_label_for_alignment" in prompt_text
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_misc_p3_feature_cards()
    print("test_misc_p3_feature_cards_smoke passed")
