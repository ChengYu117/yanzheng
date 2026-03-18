"""Smoke tests for the Stage 2 activation extraction module."""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class FakeTokenizer:
    def __call__(
        self,
        texts,
        *,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=8,
        return_attention_mask=True,
    ):
        batch = len(texts)
        input_ids = torch.zeros((batch, max_length), dtype=torch.long)
        attention_mask = torch.zeros((batch, max_length), dtype=torch.long)
        for row_idx, text in enumerate(texts):
            tokens = list(range(1, min(len(text.split()), max_length) + 1))
            input_ids[row_idx, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[row_idx, : len(tokens)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class FakeModel(torch.nn.Module):
    def __init__(self, d_model: int = 6, n_layers: int = 2):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, d_model)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(d_model, d_model) for _ in range(n_layers)]
        )

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        hidden = self.embedding(input_ids)
        hidden_states = [hidden]
        for layer in self.layers:
            hidden = layer(hidden)
            hidden_states.append(hidden)

        class _Outputs:
            pass

        outputs = _Outputs()
        outputs.hidden_states = tuple(hidden_states)
        return outputs


def test_load_stage2_dataset():
    from nlp_re_base.stage2_activation_extraction import load_stage2_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        re_rows = [
            {
                "file_id": "a",
                "unit_text": "reflective response one",
                "predicted_code": "RE",
                "predicted_subcode": "RES",
            }
        ]
        nonre_rows = [
            {
                "file_id": "b",
                "unit_text": "non reflective response",
                "predicted_code": "QUC",
                "predicted_subcode": "QUC",
            }
        ]
        (data_dir / "re_dataset.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in re_rows),
            encoding="utf-8",
        )
        (data_dir / "nonre_dataset.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in nonre_rows),
            encoding="utf-8",
        )

        dataset = load_stage2_dataset(data_dir)
        assert len(dataset) == 2
        assert dataset[0]["label_re"] == 1
        assert dataset[1]["label_re"] == 0

    print("  PASS test_load_stage2_dataset")


def test_extract_hidden_activations_by_layer():
    from nlp_re_base.stage2_activation_extraction import extract_hidden_activations_by_layer

    dataset = [
        {"unit_text": "one two three", "label_re": 1},
        {"unit_text": "one two", "label_re": 0},
        {"unit_text": "one two three four", "label_re": 1},
    ]
    model = FakeModel(d_model=6, n_layers=2)
    tokenizer = FakeTokenizer()

    X_by_layer, y = extract_hidden_activations_by_layer(
        model,
        tokenizer,
        dataset,
        batch_size=2,
        max_seq_len=8,
    )

    assert sorted(X_by_layer.keys()) == [0, 1, 2]
    assert X_by_layer[0].shape == (3, 6)
    assert X_by_layer[2].shape == (3, 6)
    assert y.tolist() == [1, 0, 1]

    print("  PASS test_extract_hidden_activations_by_layer")


def test_activation_bundle_roundtrip():
    from nlp_re_base.stage2_activation_extraction import (
        load_activations_bundle,
        save_activations_bundle,
    )

    X_by_layer = {
        0: np.random.randn(4, 5).astype(np.float32),
        1: np.random.randn(4, 5).astype(np.float32),
    }
    y = np.array([1, 0, 1, 0], dtype=np.int64)
    dataset = [{"unit_id": f"id_{i}"} for i in range(4)]

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path, meta_path = save_activations_bundle(
            X_by_layer,
            y,
            dataset,
            output_dir=tmpdir,
            model_name="fake-model",
        )
        loaded_X, loaded_y, loaded_path = load_activations_bundle(output_dir=tmpdir)
        assert loaded_path == npz_path
        assert meta_path.exists()
        assert np.array_equal(loaded_y, y)
        assert np.allclose(loaded_X[0], X_by_layer[0])

    print("  PASS test_activation_bundle_roundtrip")


def test_train_linear_probes():
    from nlp_re_base.stage2_activation_extraction import (
        summarize_probe_results,
        train_linear_probes,
    )

    rng = np.random.default_rng(42)
    y = np.array([0] * 40 + [1] * 40, dtype=np.int64)
    weak = rng.normal(size=(80, 8)).astype(np.float32)
    strong = rng.normal(size=(80, 8)).astype(np.float32)
    strong[y == 1, 0] += 3.0

    X_by_layer = {
        0: weak,
        1: strong,
    }
    results = train_linear_probes(
        X_by_layer,
        y,
        test_size=0.25,
        random_state=7,
    )
    summary = summarize_probe_results(results)

    assert len(results) == 2
    assert summary["best_layer_auc"]["layer"] == 1
    assert summary["best_layer_accuracy"]["accuracy"] >= 0.7

    print("  PASS test_train_linear_probes")


def main():
    print("=" * 72)
    print("Running Stage 2 smoke tests")
    print("=" * 72)

    tests = [
        test_load_stage2_dataset,
        test_extract_hidden_activations_by_layer,
        test_activation_bundle_roundtrip,
        test_train_linear_probes,
    ]

    failures = 0
    for test_fn in tests:
        try:
            test_fn()
        except Exception:
            failures += 1
            print(f"  FAIL {test_fn.__name__}")
            traceback.print_exc()

    print("=" * 72)
    if failures == 0:
        print(f"All Stage 2 smoke tests passed ({len(tests)}/{len(tests)}).")
    else:
        print(f"Stage 2 smoke tests failed: {failures} / {len(tests)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
