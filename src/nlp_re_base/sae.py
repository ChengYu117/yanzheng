"""SAE loading utilities with an optional official lm-saes backend.

This module keeps the legacy local SAE implementation for compatibility, while
preferring OpenMOSS's official ``lm-saes`` model class when that package is
available in the environment.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors


class JumpReLU(nn.Module):
    """Legacy fixed-threshold JumpReLU used by the original local implementation."""

    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (x > self.threshold).to(x.dtype)


class SparseAutoencoder(nn.Module):
    """Legacy local SAE implementation retained as a fallback path."""

    def __init__(
        self,
        d_model: int = 4096,
        d_sae: int = 32768,
        jump_relu_threshold: float = 0.52734375,
        use_decoder_bias: bool = True,
        norm_scale: float | None = None,
        top_k: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.norm_scale = norm_scale
        self.top_k = top_k

        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_pre = nn.Parameter(torch.zeros(d_model))

        self.W_dec = nn.Parameter(torch.empty(d_model, d_sae))
        self.use_decoder_bias = use_decoder_bias
        if use_decoder_bias:
            self.b_dec = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_buffer("b_dec", torch.zeros(d_model))

        self.activation = JumpReLU(threshold=jump_relu_threshold)

        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)

    def _apply_sparse_activation(self, pre_activation: torch.Tensor) -> torch.Tensor:
        latents = self.activation(pre_activation)
        if self.top_k is None or self.top_k <= 0 or self.top_k >= latents.shape[-1]:
            return latents

        topk_indices = latents.topk(self.top_k, dim=-1).indices
        keep_mask = torch.zeros_like(latents, dtype=torch.bool)
        keep_mask.scatter_(-1, topk_indices, True)
        return latents * keep_mask.to(latents.dtype)

    def normalize_with_stats(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.norm_scale is None:
            return x, None
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=1e-8)
        x_normalized = x * (self.norm_scale / x_norm)
        return x_normalized, x_norm

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x_normalized, _ = self.normalize_with_stats(x)
        return x_normalized

    def denormalize_reconstruction(
        self,
        x_hat_normalized: torch.Tensor,
        input_norm: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.norm_scale is None or input_norm is None:
            return x_hat_normalized
        return x_hat_normalized * (input_norm / self.norm_scale)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_normed, _ = self.normalize_with_stats(x)
        pre_activation = (x_normed - self.b_pre) @ self.W_enc.T + self.b_enc
        return self._apply_sparse_activation(pre_activation)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents @ self.W_dec.T + self.b_dec

    def forward_with_details(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x_normalized, input_norm = self.normalize_with_stats(x)
        input_scale_factor = None
        if self.norm_scale is not None and input_norm is not None:
            input_scale_factor = self.norm_scale / input_norm
        pre_activation = (x_normalized - self.b_pre) @ self.W_enc.T + self.b_enc
        latents = self._apply_sparse_activation(pre_activation)
        reconstructed_normalized = self.decode(latents)
        reconstructed_raw = self.denormalize_reconstruction(
            reconstructed_normalized,
            input_norm,
        )
        return {
            "input_normalized": x_normalized,
            "reconstructed_normalized": reconstructed_normalized,
            "reconstructed_raw": reconstructed_raw,
            "latents": latents,
            "input_scale_factor": input_scale_factor,
        }

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        details = self.forward_with_details(x)
        return details["reconstructed_raw"], details["latents"]


class OpenMossLmSaesAdapter(nn.Module):
    """Compatibility adapter around the official lm-saes SparseAutoEncoder."""

    def __init__(
        self,
        backend_sae: nn.Module,
        *,
        hook_point: str,
        sae_dtype: torch.dtype,
        checkpoint_topk_semantics: str,
    ) -> None:
        super().__init__()
        self.backend_sae = backend_sae
        self.hook_point = hook_point
        self.sae_dtype = sae_dtype
        self.checkpoint_topk_semantics = checkpoint_topk_semantics
        self.d_model = int(getattr(backend_sae.cfg, "d_model"))
        self.d_sae = int(getattr(backend_sae.cfg, "d_sae"))
        self.top_k = int(getattr(backend_sae.cfg, "top_k", 0))
        self.use_decoder_bias = bool(getattr(backend_sae.cfg, "use_decoder_bias", True))
        dataset_norm = getattr(backend_sae, "dataset_average_activation_norm", None)
        self.norm_scale = None if not dataset_norm else dataset_norm.get(hook_point)

    @property
    def W_dec(self) -> torch.Tensor:
        """Compatibility view matching the legacy local SAE shape [d_model, d_sae]."""
        return self.backend_sae.W_D.T

    @property
    def W_enc(self) -> torch.Tensor:
        """Compatibility view matching the legacy local SAE shape [d_sae, d_model]."""
        return self.backend_sae.W_E.T

    @property
    def b_dec(self) -> torch.Tensor:
        return getattr(self.backend_sae, "b_D")

    @property
    def b_enc(self) -> torch.Tensor:
        return getattr(self.backend_sae, "b_E")

    def _apply_topk_if_needed(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply optional checkpoint-semantic hard top-k after the official backend."""
        if self.checkpoint_topk_semantics != "hard":
            return latents
        if self.top_k is None or self.top_k <= 0 or self.top_k >= latents.shape[-1]:
            return latents

        topk_indices = latents.topk(self.top_k, dim=-1).indices
        keep_mask = torch.zeros_like(latents, dtype=torch.bool)
        keep_mask.scatter_(-1, topk_indices, True)
        return latents * keep_mask.to(latents.dtype)

    def normalize_with_stats(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch = {self.hook_point: x}
        normalized_batch, scale_factors = self.backend_sae.normalize_activations(
            batch,
            return_scale_factor=True,
        )
        return normalized_batch[self.hook_point], scale_factors[self.hook_point]

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        normalized, _ = self.normalize_with_stats(x)
        return normalized

    def denormalize_reconstruction(
        self,
        x_hat_normalized: torch.Tensor,
        input_scale_factor: torch.Tensor | None,
    ) -> torch.Tensor:
        if input_scale_factor is None:
            return x_hat_normalized
        return self.backend_sae.denormalize_activations(
            {self.hook_point: x_hat_normalized},
            {self.hook_point: input_scale_factor},
        )[self.hook_point]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        normalized, _ = self.normalize_with_stats(x)
        latents = self.backend_sae.encode(normalized)
        return self._apply_topk_if_needed(latents)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.backend_sae.decode(latents)

    def forward_with_details(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(device=next(self.backend_sae.parameters()).device, dtype=self.sae_dtype)
        x_normalized, scale_factor = self.normalize_with_stats(x)
        latents = self._apply_topk_if_needed(self.backend_sae.encode(x_normalized))
        reconstructed_normalized = self.backend_sae.decode(latents)
        reconstructed_raw = self.denormalize_reconstruction(
            reconstructed_normalized,
            scale_factor,
        )
        return {
            "input_normalized": x_normalized,
            "reconstructed_normalized": reconstructed_normalized,
            "reconstructed_raw": reconstructed_raw,
            "latents": latents,
            "input_scale_factor": scale_factor,
        }

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        details = self.forward_with_details(x)
        return details["reconstructed_raw"], details["latents"]


def _has_lm_saes() -> bool:
    return importlib.util.find_spec("lm_saes") is not None


def _device_to_str(device: str | torch.device) -> str:
    resolved = torch.device(device)
    return resolved.type if resolved.index is None else f"{resolved.type}:{resolved.index}"


def _load_hyperparams(
    repo_id: str,
    subfolder: str,
) -> dict[str, Any]:
    try:
        hp_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{subfolder}/hyperparams.json",
            local_files_only=True,
        )
    except Exception:
        hp_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{subfolder}/hyperparams.json",
        )
    with open(hp_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_checkpoint_state_dict(
    repo_id: str,
    subfolder: str,
) -> dict[str, torch.Tensor]:
    ckpt_dir = _download_checkpoint_dir(repo_id, subfolder)
    state_dict: dict[str, torch.Tensor] = {}
    safetensor_files = sorted(Path(ckpt_dir).glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(
            f"No .safetensors files found in checkpoint dir: {ckpt_dir}"
        )
    print(f"  Found {len(safetensor_files)} safetensors file(s)")
    for sf_path in safetensor_files:
        part = load_safetensors(str(sf_path))
        state_dict.update(part)
    print(f"  Raw checkpoint keys: {list(state_dict.keys())}")
    return state_dict


def _build_lm_saes_state_dict(
    raw_state_dict: dict[str, torch.Tensor],
    *,
    d_sae: int,
    jump_relu_threshold: float,
    use_decoder_bias: bool,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    if "encoder.weight" not in raw_state_dict or "encoder.bias" not in raw_state_dict:
        raise RuntimeError("OpenMOSS checkpoint is missing encoder weights needed for lm-saes.")
    if "decoder.weight" not in raw_state_dict:
        raise RuntimeError("OpenMOSS checkpoint is missing decoder weights needed for lm-saes.")

    state_dict: dict[str, torch.Tensor] = {
        "W_E": raw_state_dict["encoder.weight"].T.contiguous(),
        "b_E": raw_state_dict["encoder.bias"].contiguous(),
        "W_D": raw_state_dict["decoder.weight"].T.contiguous(),
        "activation_function.log_jumprelu_threshold": torch.full(
            (d_sae,),
            math.log(max(jump_relu_threshold, 1e-8)),
            dtype=dtype,
        ),
    }
    if use_decoder_bias:
        if "decoder.bias" not in raw_state_dict:
            raise RuntimeError("OpenMOSS checkpoint is missing decoder.bias.")
        state_dict["b_D"] = raw_state_dict["decoder.bias"].contiguous()
    return state_dict


def _infer_hook_point(subfolder: str, hyperparams: dict[str, Any]) -> str:
    configured = str(hyperparams.get("hook_point", "")).strip()
    if configured and "." in configured:
        return configured

    marker = subfolder.split("-L", 1)[-1]
    layer = marker.split("R", 1)[0]
    return f"blocks.{layer}.hook_resid_post"


def _try_load_with_lm_saes(
    *,
    repo_id: str,
    subfolder: str,
    device: str | torch.device,
    dtype: torch.dtype,
    hyperparams: dict[str, Any],
    raw_state_dict: dict[str, torch.Tensor],
    checkpoint_topk_semantics: str,
) -> OpenMossLmSaesAdapter | None:
    if not _has_lm_saes():
        raise RuntimeError(
            "Official OpenMOSS SAE backend is required, but `lm_saes` is not installed."
        )

    try:
        from lm_saes import SAEConfig
        from lm_saes.models.sae import SparseAutoEncoder as OfficialSparseAutoEncoder
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(f"Failed to import official `lm_saes` backend: {exc}") from exc

    try:
        d_model = int(hyperparams["d_model"])
        d_sae = int(hyperparams["d_sae"])
        hook_point = _infer_hook_point(subfolder, hyperparams)
        cfg = SAEConfig(
            device=_device_to_str(device),
            dtype=dtype,
            d_model=d_model,
            expansion_factor=float(d_sae / d_model),
            use_decoder_bias=bool(hyperparams.get("use_decoder_bias", True)),
            act_fn=str(hyperparams.get("act_fn", "jumprelu")).lower(),
            norm_activation=str(hyperparams.get("norm_activation", "dataset-wise")).lower(),
            sparsity_include_decoder_norm=bool(
                hyperparams.get("sparsity_include_decoder_norm", True)
            ),
            top_k=int(hyperparams.get("top_k", 50)),
            hook_point_in=hook_point,
            hook_point_out=hook_point,
            use_glu_encoder=bool(hyperparams.get("use_glu_encoder", False)),
        )

        official = OfficialSparseAutoEncoder(cfg)
        norm_info = hyperparams.get("dataset_average_activation_norm", {})
        if isinstance(norm_info, dict) and norm_info:
            dataset_norm = float(norm_info.get("in", norm_info.get(hook_point, 0.0)))
            official.set_dataset_average_activation_norm(
                {
                    hook_point: dataset_norm,
                }
            )

        official_state_dict = _build_lm_saes_state_dict(
            raw_state_dict,
            d_sae=d_sae,
            jump_relu_threshold=float(hyperparams.get("jump_relu_threshold", 1e-8)),
            use_decoder_bias=bool(hyperparams.get("use_decoder_bias", True)),
            dtype=dtype,
        )
        official.load_full_state_dict(official_state_dict, strict=True)
        if (
            hasattr(official, "standardize_parameters_of_dataset_norm")
            and getattr(official.cfg, "norm_activation", None) == "dataset-wise"
            and getattr(official, "dataset_average_activation_norm", None) is not None
        ):
            official.standardize_parameters_of_dataset_norm()
        official = official.to(device=device, dtype=dtype)
        official.eval()
        print("  [SAE] Loaded with official lm-saes backend.")
        return OpenMossLmSaesAdapter(
            official,
            hook_point=hook_point,
            sae_dtype=dtype,
            checkpoint_topk_semantics=checkpoint_topk_semantics,
        )
    except Exception as exc:  # pragma: no cover - live fallback path
        raise RuntimeError(
            "Official lm-saes SAE loader failed. "
            "Legacy fallback has been disabled to avoid silent metric drift. "
            f"Original error: {exc}"
        ) from exc


def load_sae_from_hub(
    repo_id: str = "OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x",
    subfolder: str = "Llama3_1-8B-Base-L19R-8x",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    prefer_official: bool = True,
    checkpoint_topk_semantics: str = "disabled",
) -> nn.Module:
    """Download and load a pre-trained SAE from HuggingFace Hub.

    When ``prefer_official=True`` (default), this requires the official OpenMOSS
    ``lm-saes`` backend and raises immediately if official loading fails.
    The legacy local implementation is only available via ``prefer_official=False``.
    """
    print(f"Downloading SAE hyperparams from {repo_id}/{subfolder}...")
    hyperparams = _load_hyperparams(repo_id, subfolder)

    d_model = hyperparams["d_model"]
    d_sae = hyperparams["d_sae"]
    threshold = hyperparams.get("jump_relu_threshold", 0.0)
    use_decoder_bias = hyperparams.get("use_decoder_bias", True)
    norm_info = hyperparams.get("dataset_average_activation_norm", {})
    norm_scale = norm_info.get("in", None)
    top_k = hyperparams.get("top_k")

    print(
        f"SAE config: d_model={d_model}, d_sae={d_sae}, "
        f"threshold={threshold}, norm_scale={norm_scale}, top_k={top_k}"
    )

    print("Downloading SAE checkpoint weights...")
    raw_state_dict = _load_checkpoint_state_dict(repo_id, subfolder)

    if prefer_official:
        official_adapter = _try_load_with_lm_saes(
            repo_id=repo_id,
            subfolder=subfolder,
            device=device,
            dtype=dtype,
            hyperparams=hyperparams,
            raw_state_dict=raw_state_dict,
            checkpoint_topk_semantics=checkpoint_topk_semantics,
        )
        total_params = sum(p.numel() for p in official_adapter.parameters())
        print(f"SAE loaded: {total_params:,} parameters on {device} ({dtype})")
        return official_adapter

    sae = SparseAutoencoder(
        d_model=d_model,
        d_sae=d_sae,
        jump_relu_threshold=threshold,
        use_decoder_bias=use_decoder_bias,
        norm_scale=norm_scale,
        top_k=top_k,
    )

    mapped_state_dict = _map_state_dict(raw_state_dict, sae)
    sae.load_state_dict(mapped_state_dict, strict=True)

    sae = sae.to(device=device, dtype=dtype)
    sae.sae_dtype = dtype
    sae.eval()

    total_params = sum(p.numel() for p in sae.parameters())
    print(f"SAE loaded: {total_params:,} parameters on {device} ({dtype})")
    return sae


def _download_checkpoint_dir(repo_id: str, subfolder: str) -> str:
    """Download the checkpoint directory from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=f"{subfolder}/checkpoints/*",
            local_files_only=True,
        )
    except Exception:
        local_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=f"{subfolder}/checkpoints/*",
        )
    ckpt_path = Path(local_dir) / subfolder / "checkpoints"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found after download: {ckpt_path}"
        )
    return str(ckpt_path)


def _map_state_dict(
    raw_state_dict: dict[str, torch.Tensor],
    sae: SparseAutoencoder,
) -> dict[str, torch.Tensor]:
    """Map checkpoint keys to the legacy local SparseAutoencoder parameter names."""
    mapped: dict[str, torch.Tensor] = {}
    key_mapping = {
        "encoder.weight": "W_enc",
        "encoder.bias": "b_enc",
        "decoder.weight": "W_dec",
        "decoder.bias": "b_dec",
        "pre_bias": "b_pre",
        "W_enc": "W_enc",
        "b_enc": "b_enc",
        "W_dec": "W_dec",
        "b_dec": "b_dec",
        "b_pre": "b_pre",
        "encoder.W": "W_enc",
        "encoder.b": "b_enc",
        "decoder.W": "W_dec",
        "decoder.b": "b_dec",
        "encoder_bias": "b_enc",
        "decoder_bias": "b_dec",
        "pre_encoder_bias": "b_pre",
        "b_dec_out": "b_dec",
    }
    prefixes = ("sae.", "model.", "sparse_autoencoder.", "ae.", "autoencoder.")

    for raw_key, tensor in raw_state_dict.items():
        clean_key = raw_key
        for prefix in prefixes:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
                break

        if clean_key in key_mapping:
            param_name = key_mapping[clean_key]
            mapped[param_name] = tensor
            print(f"  [SAE] Mapped: {raw_key} -> {param_name} (shape={tensor.shape})")
        else:
            print(f"  [SAE] Unmapped checkpoint key: {raw_key} (shape={tensor.shape})")

    model_params = dict(sae.named_parameters())
    for name in ("W_enc", "W_dec"):
        if name not in mapped:
            continue
        expected_shape = model_params[name].shape
        actual_shape = mapped[name].shape
        if actual_shape != expected_shape:
            transposed_shape = (expected_shape[1], expected_shape[0])
            if len(actual_shape) == 2 and actual_shape == transposed_shape:
                print(f"  [SAE] Transposing {name}: {actual_shape} -> {expected_shape}")
                mapped[name] = mapped[name].T
            else:
                raise RuntimeError(
                    f"Shape mismatch for {name}: "
                    f"expected {expected_shape}, got {actual_shape}. "
                    f"Cannot auto-resolve."
                )

    critical_keys = {"W_enc", "W_dec", "b_enc"}
    missing_critical = critical_keys - set(mapped.keys())
    if missing_critical:
        raise RuntimeError(
            f"CRITICAL: Missing SAE weights from checkpoint: {missing_critical}. "
            f"Available mapped keys: {set(mapped.keys())}. "
            f"Raw checkpoint keys: {list(raw_state_dict.keys())}. "
            f"Cannot proceed with partially initialized SAE."
        )

    all_expected = set(model_params.keys())
    missing_other = all_expected - set(mapped.keys()) - critical_keys
    if missing_other:
        print(f"  [SAE] Non-critical keys using zero-init: {missing_other}")
        for key in missing_other:
            mapped[key] = torch.zeros_like(model_params[key])

    return mapped
