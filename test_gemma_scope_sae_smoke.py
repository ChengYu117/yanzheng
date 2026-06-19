"""Smoke tests for GemmaScope SAE loading primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_jumprelu_encode_decode_shapes() -> None:
    from nlp_re_base.gemma_scope_sae import GemmaScopeJumpReLUSAE, GemmaScopeSaeConfig

    torch.manual_seed(0)
    d_model = 5
    d_sae = 7
    cfg = GemmaScopeSaeConfig(
        repo_id="dummy/repo",
        subfolder="resid_post_all/layer_18_width_16k_l0_small",
        layer_idx=18,
        width=d_sae,
        l0=2,
        model_name="google/gemma-3-4b-pt",
        architecture="jump_relu",
        hf_hook_point_in="model.layers.18.output",
        hf_hook_point_out="model.layers.18.output",
        affine_connection=False,
        config_path="config.json",
        params_path="params.safetensors",
    )
    sae = GemmaScopeJumpReLUSAE(
        config=cfg,
        w_enc=torch.randn(d_model, d_sae),
        b_enc=torch.zeros(d_sae),
        threshold=torch.full((d_sae,), 0.1),
        w_dec=torch.randn(d_sae, d_model),
        b_dec=torch.zeros(d_model),
    )
    x = torch.randn(3, 4, d_model)
    details = sae.forward_with_details(x)
    assert details["latents"].shape == (3, 4, d_sae)
    assert details["reconstructed_raw"].shape == x.shape
    assert torch.all(details["latents"] >= 0)


def test_layer18_config_is_available() -> None:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="google/gemma-scope-2-4b-pt",
        filename="resid_post_all/layer_18_width_16k_l0_small/config.json",
    )
    text = Path(path).read_text(encoding="utf-8")
    assert '"width": 16384' in text
    assert '"model_name": "google/gemma-3-4b-pt"' in text
    assert '"hf_hook_point_out": "model.layers.18.output"' in text


if __name__ == "__main__":
    test_jumprelu_encode_decode_shapes()
    test_layer18_config_is_available()
    print("gemma_scope_sae smoke passed")

