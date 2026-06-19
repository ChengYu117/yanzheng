"""Smoke tests for the GemmaScope extraction pipeline."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

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
        ids = [[(ord(ch) % 17) + 1 for ch in text][:max_length] or [1] for text in texts]
        width = max(len(row) for row in ids)
        input_ids = torch.zeros((len(ids), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row_idx, row in enumerate(ids):
            input_ids[row_idx, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[row_idx, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class ToyLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, layer_idx: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, hidden_size)
        self.layer_idx = layer_idx

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(hidden)) + self.layer_idx


class ToyModelPath(torch.nn.Module):
    def __init__(self, *, path: str, n_layers: int = 3, hidden_size: int = 8) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(32, hidden_size)
        layers = torch.nn.ModuleList([ToyLayer(hidden_size, i) for i in range(n_layers)])
        if path == "model.layers":
            self.model = SimpleNamespace(layers=layers)
        elif path == "language_model.layers":
            self.language_model = SimpleNamespace(layers=layers)
        else:
            raise ValueError(path)
        self._layers = layers

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, use_cache: bool | None = None):
        del attention_mask, use_cache
        hidden = self.embed(input_ids)
        for layer in self._layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


def _toy_sae(hidden_size: int = 8, d_sae: int = 12):
    from nlp_re_base.gemma_scope_sae import GemmaScopeJumpReLUSAE, GemmaScopeSaeConfig

    cfg = GemmaScopeSaeConfig(
        repo_id="dummy/repo",
        subfolder="resid_post_all/layer_1_width_12_l0_small",
        layer_idx=1,
        width=d_sae,
        l0=3,
        model_name="toy",
        architecture="jump_relu",
        hf_hook_point_in="model.layers.1.output",
        hf_hook_point_out="model.layers.1.output",
        affine_connection=False,
        config_path="config.json",
        params_path="params.safetensors",
    )
    return GemmaScopeJumpReLUSAE(
        config=cfg,
        w_enc=torch.randn(hidden_size, d_sae),
        b_enc=torch.zeros(d_sae),
        threshold=torch.full((d_sae,), -0.2),
        w_dec=torch.randn(d_sae, hidden_size),
        b_dec=torch.zeros(hidden_size),
    )


def _run_path(path: str) -> None:
    from nlp_re_base.gemma_scope_pipeline import extract_gemma_scope_feature_store

    torch.manual_seed(1)
    model = ToyModelPath(path=path)
    sae = _toy_sae()
    with tempfile.TemporaryDirectory() as tmp:
        result = extract_gemma_scope_feature_store(
            model=model,
            tokenizer=ToyTokenizer(),
            sae=sae,
            texts=["reflection", "question", "affirm"],
            layer_idx=1,
            output_dir=tmp,
            max_seq_len=16,
            batch_size=2,
            aggregation="max",
        )
        assert result["utterance_features"].shape == (3, 12)
        assert result["utterance_activations"].shape == (3, 8)
        assert result["structural_metrics"]["n_valid_tokens"] > 0
        assert (Path(tmp) / "feature_store" / "utterance_features.pt").exists()
        assert (Path(tmp) / "feature_store" / "utterance_activations.pt").exists()


def test_model_layers_path() -> None:
    _run_path("model.layers")


def test_language_model_layers_path() -> None:
    _run_path("language_model.layers")


if __name__ == "__main__":
    test_model_layers_path()
    test_language_model_layers_path()
    print("gemma_scope_pipeline smoke passed")

