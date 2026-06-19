"""GemmaScope SAE loading and forward utilities.

This module implements the lightweight JumpReLU SAE format used by
``google/gemma-scope-2-4b-pt``.  It intentionally avoids depending on
``sae_lens`` so the project can run with the current qwen-env-py311
environment.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


DEFAULT_GEMMA_SCOPE_REPO_ID = "google/gemma-scope-2-4b-pt"
DEFAULT_GEMMA_SCOPE_LAYER_IDX = 18
DEFAULT_GEMMA_SCOPE_WIDTH = "16k"
DEFAULT_GEMMA_SCOPE_L0 = "small"


@dataclass(frozen=True)
class GemmaScopeSaeConfig:
    repo_id: str
    subfolder: str
    layer_idx: int
    width: int
    l0: int
    model_name: str
    architecture: str
    hf_hook_point_in: str
    hf_hook_point_out: str
    affine_connection: bool
    config_path: str
    params_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def gemma_scope_subfolder(
    *,
    layer_idx: int = DEFAULT_GEMMA_SCOPE_LAYER_IDX,
    width: str = DEFAULT_GEMMA_SCOPE_WIDTH,
    l0: str = DEFAULT_GEMMA_SCOPE_L0,
    stream: str = "resid_post_all",
) -> str:
    """Return the GemmaScope subfolder for a layer/width/L0 variant."""

    return f"{stream}/layer_{int(layer_idx)}_width_{width}_l0_{l0}"


def _parse_layer_from_hook(hook_point: str) -> int | None:
    match = re.search(r"layers\.(\d+)\.output", str(hook_point))
    return int(match.group(1)) if match else None


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


class GemmaScopeJumpReLUSAE(nn.Module):
    """JumpReLU SAE used by GemmaScope residual stream checkpoints."""

    def __init__(
        self,
        *,
        config: GemmaScopeSaeConfig,
        w_enc: torch.Tensor,
        b_enc: torch.Tensor,
        threshold: torch.Tensor,
        w_dec: torch.Tensor,
        b_dec: torch.Tensor,
    ) -> None:
        super().__init__()
        if w_enc.ndim != 2 or w_dec.ndim != 2:
            raise ValueError("w_enc and w_dec must be rank-2 tensors.")
        if w_enc.shape[1] != w_dec.shape[0]:
            raise ValueError(
                f"SAE width mismatch: w_enc={tuple(w_enc.shape)}, "
                f"w_dec={tuple(w_dec.shape)}"
            )
        if b_enc.shape[0] != w_enc.shape[1] or threshold.shape[0] != w_enc.shape[1]:
            raise ValueError("b_enc/threshold length must match SAE width.")
        if b_dec.shape[0] != w_dec.shape[1]:
            raise ValueError("b_dec length must match decoded activation size.")

        self.config = config
        self.d_model = int(w_enc.shape[0])
        self.d_sae = int(w_enc.shape[1])
        self.register_buffer("w_enc", w_enc.contiguous())
        self.register_buffer("b_enc", b_enc.contiguous())
        self.register_buffer("threshold", threshold.contiguous())
        self.register_buffer("w_dec", w_dec.contiguous())
        self.register_buffer("b_dec", b_dec.contiguous())

    @property
    def device(self) -> torch.device:
        return self.w_enc.device

    @property
    def sae_dtype(self) -> torch.dtype:
        return self.w_enc.dtype

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.sae_dtype)
        pre = x @ self.w_enc + self.b_enc
        return torch.where(pre > self.threshold, pre, torch.zeros_like(pre))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(device=self.device, dtype=self.sae_dtype)
        return latents @ self.w_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(x)
        reconstructed = self.decode(latents)
        return reconstructed, latents

    def forward_with_details(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        latents = self.encode(flat)
        reconstructed = self.decode(latents)
        return {
            "latents": latents.reshape(*original_shape[:-1], self.d_sae),
            "reconstructed_raw": reconstructed.reshape(*original_shape),
            "input_raw": x,
        }


def load_gemma_scope_sae(
    *,
    repo_id: str = DEFAULT_GEMMA_SCOPE_REPO_ID,
    subfolder: str | None = None,
    layer_idx: int = DEFAULT_GEMMA_SCOPE_LAYER_IDX,
    width: str = DEFAULT_GEMMA_SCOPE_WIDTH,
    l0: str = DEFAULT_GEMMA_SCOPE_L0,
    cache_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float16,
) -> GemmaScopeJumpReLUSAE:
    """Download/load a GemmaScope SAE checkpoint."""

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    subfolder = subfolder or gemma_scope_subfolder(
        layer_idx=layer_idx,
        width=width,
        l0=l0,
    )
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{subfolder}/config.json",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    params_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{subfolder}/params.safetensors",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    raw_config = _load_json(config_path)
    hook_in = str(raw_config.get("hf_hook_point_in", ""))
    hook_out = str(raw_config.get("hf_hook_point_out", ""))
    hook_layer = _parse_layer_from_hook(hook_out or hook_in)
    if hook_layer is not None and int(hook_layer) != int(layer_idx):
        raise ValueError(
            f"GemmaScope checkpoint layer mismatch: requested layer_idx={layer_idx}, "
            f"config hook layer={hook_layer} ({hook_out or hook_in})."
        )
    expected_hook = f"model.layers.{int(layer_idx)}.output"
    if hook_out and hook_out != expected_hook:
        raise ValueError(
            f"Unexpected GemmaScope hook: {hook_out!r}; expected {expected_hook!r}."
        )

    state = load_file(params_path, device="cpu")
    required = {"w_enc", "b_enc", "threshold", "w_dec", "b_dec"}
    missing = sorted(required.difference(state))
    if missing:
        raise KeyError(f"GemmaScope params missing keys: {missing}")

    cfg = GemmaScopeSaeConfig(
        repo_id=repo_id,
        subfolder=subfolder,
        layer_idx=int(layer_idx),
        width=int(raw_config.get("width", state["w_enc"].shape[1])),
        l0=int(raw_config.get("l0", 0)),
        model_name=str(raw_config.get("model_name", "")),
        architecture=str(raw_config.get("architecture", "")),
        hf_hook_point_in=hook_in,
        hf_hook_point_out=hook_out,
        affine_connection=bool(raw_config.get("affine_connection", False)),
        config_path=str(config_path),
        params_path=str(params_path),
    )

    sae = GemmaScopeJumpReLUSAE(
        config=cfg,
        w_enc=state["w_enc"],
        b_enc=state["b_enc"],
        threshold=state["threshold"],
        w_dec=state["w_dec"],
        b_dec=state["b_dec"],
    )
    if device is not None:
        sae = sae.to(device=torch.device(device), dtype=dtype)
    else:
        sae = sae.to(dtype=dtype)
    sae.eval()
    return sae

