from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - allows lightweight env smoke runs.
    torch = None
    TORCH_AVAILABLE = False

if not TORCH_AVAILABLE:  # pragma: no cover
    class _TorchPlaceholder:
        class nn:
            class Module:
                pass

    torch = _TorchPlaceholder()

from run_llama_layer_probe import run_llama_layer_probe_for_records


class ToyTokenizer:
    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, "torch.Tensor"]:
        del padding, truncation, return_tensors
        encoded: list[list[int]] = []
        for text in texts:
            ids = [(ord(ch) % 29) + 1 for ch in text][:max_length]
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

    def forward(self, hidden: "torch.Tensor") -> "torch.Tensor":
        return torch.tanh(self.linear(hidden)) + float(self.layer_idx)


class ToyBackbone(torch.nn.Module):
    def __init__(self, n_layers: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(64, hidden_size)
        self.layers = torch.nn.ModuleList(
            [ToyLayer(hidden_size, idx) for idx in range(n_layers)]
        )


class ToyLlama(torch.nn.Module):
    def __init__(self, n_layers: int = 3, hidden_size: int = 6) -> None:
        super().__init__()
        self.model = ToyBackbone(n_layers=n_layers, hidden_size=hidden_size)

    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
        use_cache: bool | None = None,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


def _safe_smoke_root() -> Path:
    root = (Path.cwd() / "outputs" / "_smoke_llama_layer_probe").resolve()
    cwd = Path.cwd().resolve()
    if cwd not in root.parents:
        raise RuntimeError(f"Refusing to use smoke output outside repo: {root}")
    return root


def test_llama_layer_probe_for_records() -> None:
    if not TORCH_AVAILABLE:
        print("test_llama_layer_probe_smoke skipped: torch is unavailable")
        return

    root = _safe_smoke_root()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    try:
        records = []
        for idx in range(36):
            label = "A" if idx % 2 == 0 else "B"
            records.append(
                {
                    "record_id": f"r{idx}",
                    "unit_text": f"{label.lower()} sample text {idx}",
                    "labels": [label],
                }
            )

        summary = run_llama_layer_probe_for_records(
            model=ToyLlama(n_layers=3, hidden_size=6),
            tokenizer=ToyTokenizer(),
            records=records,
            output_dir=root / "out",
            model_source="toy-llama",
            labels=["A", "B"],
            pooling=["mean", "last"],
            cv_folds=3,
            recognition_auc_threshold=0.50,
            max_seq_len=16,
            batch_size=6,
            feature_dtype="float32",
        )

        output_dir = root / "out"
        expected_files = [
            "layer_probe_metrics.csv",
            "best_layers_by_label.csv",
            "layer_probe_report.md",
            "dataset_summary.json",
            "feature_store/layer_feature_metadata.json",
        ]
        for name in expected_files:
            assert (output_dir / name).exists(), name

        metrics = pd.read_csv(output_dir / "layer_probe_metrics.csv")
        best = pd.read_csv(output_dir / "best_layers_by_label.csv")
        assert len(metrics) == 2 * 2 * 3
        assert set(metrics["label"]) == {"A", "B"}
        assert set(metrics["pooling"]) == {"mean", "last"}
        assert set(best["label"]) == {"A", "B"}
        assert summary["analysis"] == "llama_cross_layer_misc_probe"
        assert summary["n_metric_rows"] == 12
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_llama_layer_probe_for_records()
    print("test_llama_layer_probe_smoke passed")
