"""Smoke tests for Gemma-style MISC layer probing."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class ToyTokenizer:
    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del padding, truncation, return_tensors
        encoded: list[list[int]] = []
        for text in texts:
            ids = [(ord(ch) % 19) + 1 for ch in text][:max_length]
            encoded.append(ids or [1])
        width = max(len(ids) for ids in encoded)
        input_ids = torch.zeros((len(encoded), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class ToyLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, layer_idx: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size)
        self.layer_idx = layer_idx

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(hidden)) + float(self.layer_idx)


class ToyBackbone(torch.nn.Module):
    def __init__(self, n_layers: int = 3, hidden_size: int = 8) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(32, hidden_size)
        self.layers = torch.nn.ModuleList(
            [ToyLayer(hidden_size, layer_idx) for layer_idx in range(n_layers)]
        )


class ToyGemma(torch.nn.Module):
    def __init__(self, n_layers: int = 3, hidden_size: int = 8) -> None:
        super().__init__()
        self.model = ToyBackbone(n_layers=n_layers, hidden_size=hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        hidden = self.model.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


def test_layer_discovery_and_feature_extraction() -> None:
    from nlp_re_base.layer_probe import discover_decoder_layers, extract_layer_features

    torch.manual_seed(42)
    model = ToyGemma(n_layers=3, hidden_size=8)
    layers, path = discover_decoder_layers(model)
    assert path == "model.layers"
    assert len(layers) == 3

    with tempfile.TemporaryDirectory() as tmp:
        store = extract_layer_features(
            model=model,
            tokenizer=ToyTokenizer(),
            texts=["reflection", "question", "information"],
            output_dir=tmp,
            pooling=["mean", "last"],
            max_seq_len=16,
            batch_size=2,
            feature_dtype="float32",
            reuse_cache=False,
        )

        assert store.n_layers == 3
        assert store.n_records == 3
        assert store.hidden_size == 8
        mean = np.load(store.feature_path(1, "mean"))
        last = np.load(store.feature_path(1, "last"))
        assert mean.shape == (3, 8)
        assert last.shape == (3, 8)
        assert np.isfinite(mean).all()
        assert np.isfinite(last).all()


def test_one_vs_rest_probe_selects_signal_layer() -> None:
    from nlp_re_base.layer_probe import LayerFeatureStore, evaluate_layer_probes

    rng = np.random.default_rng(42)
    n_records = 72
    hidden_size = 6
    n_layers = 3
    labels = ["RE", "GI"]
    y_re = np.array([0, 1] * (n_records // 2), dtype=np.int64)
    y_gi = 1 - y_re
    label_indicators = np.stack([y_re, y_gi], axis=1).astype(bool)

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        feature_dir = output_dir / "feature_store"
        feature_dir.mkdir(parents=True)

        for layer_idx in range(n_layers):
            X = rng.normal(0.0, 0.7, size=(n_records, hidden_size)).astype(np.float32)
            if layer_idx == 1:
                X[:, 0] = y_re * 5.0 + rng.normal(0.0, 0.1, size=n_records)
            np.save(feature_dir / f"layer_{layer_idx:02d}_mean.npy", X)

        store = LayerFeatureStore(
            output_dir=output_dir,
            feature_dir=feature_dir,
            layer_path="model.layers",
            n_layers=n_layers,
            n_records=n_records,
            hidden_size=hidden_size,
            pooling=["mean"],
            feature_dtype="float32",
            max_seq_len=16,
        )

        _metrics_df, best_df = evaluate_layer_probes(
            feature_store=store,
            label_indicators=label_indicators,
            labels=labels,
            output_dir=output_dir,
            cv_folds=5,
            recognition_auc_threshold=0.70,
        )

        best_re = best_df[best_df["label"] == "RE"].iloc[0]
        best_gi = best_df[best_df["label"] == "GI"].iloc[0]
        assert int(best_re["best_layer_idx"]) == 1
        assert bool(best_re["recognized"])
        assert float(best_re["best_auc_mean"]) >= 0.95
        assert int(best_gi["best_layer_idx"]) == 1
        assert bool(best_gi["recognized"])


def test_auc_threshold_marks_unrecognized() -> None:
    from nlp_re_base.layer_probe import summarize_best_layers

    import pandas as pd

    metrics = pd.DataFrame(
        [
            {
                "label": "QU",
                "layer_idx": 0,
                "pooling": "mean",
                "n_positive": 10,
                "n_negative": 10,
                "n_folds": 5,
                "auc_mean": 0.69,
                "auc_std": 0.01,
                "accuracy_mean": 0.6,
                "f1_mean": 0.55,
                "balanced_accuracy_mean": 0.6,
                "recognized": False,
            },
            {
                "label": "QU",
                "layer_idx": 1,
                "pooling": "last",
                "n_positive": 10,
                "n_negative": 10,
                "n_folds": 5,
                "auc_mean": 0.68,
                "auc_std": 0.02,
                "accuracy_mean": 0.55,
                "f1_mean": 0.5,
                "balanced_accuracy_mean": 0.55,
                "recognized": False,
            },
        ]
    )

    best = summarize_best_layers(metrics, recognition_auc_threshold=0.70)
    row = best.iloc[0]
    assert row["label"] == "QU"
    assert int(row["best_layer_idx"]) == 0
    assert not bool(row["recognized"])
    assert not bool(row["recommended_for_ablation"])


def test_gemma_cli_excludes_other_by_default() -> None:
    from run_gemma_layer_probe import _labels_or_default, parse_args

    old_argv = sys.argv
    try:
        sys.argv = ["run_gemma_layer_probe.py"]
        args = parse_args()
    finally:
        sys.argv = old_argv

    records = [
        {"labels": ["RE"], "text": "reflection"},
        {"labels": ["OTHER"], "text": "fallback"},
    ]
    labels, indicators = _labels_or_default(records, args)
    assert "OTHER" not in labels
    assert labels == ["RE", "RES", "REC", "QU", "QUO", "QUC", "GI", "SU", "AF"]
    assert indicators.shape == (2, 9)


def test_gemma_cli_re_nonre_uses_binary_re_label() -> None:
    from run_gemma_layer_probe import (
        DEFAULT_RE_NONRE_OUTPUT_DIR,
        _labels_or_default,
        parse_args,
    )

    old_argv = sys.argv
    try:
        sys.argv = ["run_gemma_layer_probe.py", "--probe-dataset", "re_nonre"]
        args = parse_args()
    finally:
        sys.argv = old_argv

    records = [
        {"label_re": 1, "text": "reflection"},
        {"label_re": 0, "text": "question"},
        {"label_re": 1, "text": "complex reflection"},
    ]
    labels, indicators = _labels_or_default(records, args)
    assert labels == ["RE"]
    assert indicators.shape == (3, 1)
    assert indicators[:, 0].tolist() == [True, False, True]
    assert DEFAULT_RE_NONRE_OUTPUT_DIR.endswith("gemma3_re_nonre_layer_probe")


def main() -> int:
    tests = [
        test_layer_discovery_and_feature_extraction,
        test_one_vs_rest_probe_selects_signal_layer,
        test_auc_threshold_marks_unrecognized,
        test_gemma_cli_excludes_other_by_default,
        test_gemma_cli_re_nonre_uses_binary_re_label,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"Results: {len(tests)} passed, 0 failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
