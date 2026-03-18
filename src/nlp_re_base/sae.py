"""Sparse Autoencoder (SAE) model definition and HuggingFace loading utilities.

Implements the JumpReLU SAE architecture used by OpenMOSS LXR series.
Reference model: Llama3_1-8B-Base-L19R-8x
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors


class JumpReLU(nn.Module):
    """JumpReLU activation: f(x) = x * (x > threshold)."""

    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (x > self.threshold).to(x.dtype)


class SparseAutoencoder(nn.Module):
    """JumpReLU Sparse Autoencoder.

    Architecture:
        encode:  h = JumpReLU(W_enc @ (x_norm - b_pre) + b_enc)
        decode:  x_hat = W_dec @ h + b_dec

    Where x_norm = x * (norm_scale / ||x||) for dataset-wise normalization.
    """

    def __init__(
        self,
        d_model: int = 4096,
        d_sae: int = 32768,
        jump_relu_threshold: float = 0.52734375,
        use_decoder_bias: bool = True,
        norm_scale: float | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.norm_scale = norm_scale

        # Encoder: d_model -> d_sae
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_pre = nn.Parameter(torch.zeros(d_model))  # pre-encoder bias

        # Decoder: d_sae -> d_model
        self.W_dec = nn.Parameter(torch.empty(d_model, d_sae))
        self.use_decoder_bias = use_decoder_bias
        if use_decoder_bias:
            self.b_dec = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_buffer("b_dec", torch.zeros(d_model))

        # Activation function
        self.activation = JumpReLU(threshold=jump_relu_threshold)

        # Initialize weights
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)

    def normalize_with_stats(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Normalize x and keep the original token norms for inverse scaling."""
        if self.norm_scale is None:
            return x, None
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=1e-8)
        x_normalized = x * (self.norm_scale / x_norm)
        return x_normalized, x_norm

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dataset-wise normalization: rescale x to have L2 norm = norm_scale."""
        x_normalized, _ = self.normalize_with_stats(x)
        return x_normalized

    def denormalize_reconstruction(
        self,
        x_hat_normalized: torch.Tensor,
        input_norm: torch.Tensor | None,
    ) -> torch.Tensor:
        """Map a normalized-space reconstruction back to the raw activation space."""
        if self.norm_scale is None or input_norm is None:
            return x_hat_normalized
        return x_hat_normalized * (input_norm / self.norm_scale)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse latents.

        Args:
            x: Input activations [..., d_model]

        Returns:
            latents: Sparse feature activations [..., d_sae]
        """
        x_normed, _ = self.normalize_with_stats(x)
        pre_activation = (x_normed - self.b_pre) @ self.W_enc.T + self.b_enc
        return self.activation(pre_activation)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode sparse latents back to activation space.

        Args:
            latents: Sparse feature activations [..., d_sae]

        Returns:
            x_hat: Reconstructed activations [..., d_model]
        """
        return latents @ self.W_dec.T + self.b_dec

    def forward_with_details(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full SAE forward pass with both normalized- and raw-space reconstructions."""
        x_normalized, input_norm = self.normalize_with_stats(x)
        pre_activation = (x_normalized - self.b_pre) @ self.W_enc.T + self.b_enc
        latents = self.activation(pre_activation)
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
        }

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full SAE forward pass.

        Returns the raw-space reconstruction so downstream metrics and
        activation replacement operate in the model's original space.
        """
        details = self.forward_with_details(x)
        return details["reconstructed_raw"], details["latents"]


def load_sae_from_hub(
    repo_id: str = "OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x",
    subfolder: str = "Llama3_1-8B-Base-L19R-8x",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> SparseAutoencoder:
    """Download and load a pre-trained SAE from HuggingFace Hub.

    The checkpoint directory is expected to contain safetensors files
    and the parent folder should have hyperparams.json.

    Raises RuntimeError if critical weights (W_enc, W_dec, b_enc) are
    missing from the checkpoint — never silently falls back to random init.
    """
    print(f"Downloading SAE hyperparams from {repo_id}/{subfolder}...")
    hp_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{subfolder}/hyperparams.json",
    )
    with open(hp_path, "r", encoding="utf-8") as f:
        hyperparams = json.load(f)

    d_model = hyperparams["d_model"]
    d_sae = hyperparams["d_sae"]
    threshold = hyperparams.get("jump_relu_threshold", 0.0)
    use_decoder_bias = hyperparams.get("use_decoder_bias", True)
    norm_info = hyperparams.get("dataset_average_activation_norm", {})
    norm_scale = norm_info.get("in", None)

    print(f"SAE config: d_model={d_model}, d_sae={d_sae}, "
          f"threshold={threshold}, norm_scale={norm_scale}")

    sae = SparseAutoencoder(
        d_model=d_model,
        d_sae=d_sae,
        jump_relu_threshold=threshold,
        use_decoder_bias=use_decoder_bias,
        norm_scale=norm_scale,
    )

    # Download checkpoint files
    print("Downloading SAE checkpoint weights...")
    ckpt_dir = _download_checkpoint_dir(repo_id, subfolder)

    # Load safetensors weights
    state_dict = {}
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

    # Map checkpoint keys to our model's parameter names — fails hard on missing
    mapped_state_dict = _map_state_dict(state_dict, sae)
    sae.load_state_dict(mapped_state_dict, strict=True)

    sae = sae.to(device=device, dtype=dtype)
    sae.sae_dtype = dtype  # store for downstream dtype alignment
    sae.eval()

    total_params = sum(p.numel() for p in sae.parameters())
    print(f"SAE loaded: {total_params:,} parameters on {device} ({dtype})")
    return sae


def _download_checkpoint_dir(repo_id: str, subfolder: str) -> str:
    """Download the checkpoint directory from HuggingFace Hub.

    Returns the local directory path containing the safetensors files.
    """
    from huggingface_hub import snapshot_download

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
    """Map checkpoint keys to SparseAutoencoder parameter names.

    Exhaustive key mapping covering known LXR checkpoint conventions.
    Raises RuntimeError if any critical weight is missing.
    """
    mapped = {}
    # Exhaustive mapping: checkpoint key -> our param name
    key_mapping = {
        # Standard naming
        "encoder.weight": "W_enc",
        "encoder.bias": "b_enc",
        "decoder.weight": "W_dec",
        "decoder.bias": "b_dec",
        "pre_bias": "b_pre",
        # Direct naming
        "W_enc": "W_enc",
        "b_enc": "b_enc",
        "W_dec": "W_dec",
        "b_dec": "b_dec",
        "b_pre": "b_pre",
        # LXR / LM-SAE naming variants
        "encoder.W": "W_enc",
        "encoder.b": "b_enc",
        "decoder.W": "W_dec",
        "decoder.b": "b_dec",
        "encoder_bias": "b_enc",
        "decoder_bias": "b_dec",
        "pre_encoder_bias": "b_pre",
        "b_dec_out": "b_dec",
    }

    # Common prefixes to strip
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

    # ── Validate shapes and transpose if needed ──
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

    # ── Hard-fail on missing critical keys ──
    critical_keys = {"W_enc", "W_dec", "b_enc"}
    missing_critical = critical_keys - set(mapped.keys())
    if missing_critical:
        raise RuntimeError(
            f"CRITICAL: Missing SAE weights from checkpoint: {missing_critical}. "
            f"Available mapped keys: {set(mapped.keys())}. "
            f"Raw checkpoint keys: {list(raw_state_dict.keys())}. "
            f"Cannot proceed with partially initialized SAE."
        )

    # Warn (but don't fail) for non-critical missing keys
    all_expected = set(model_params.keys())
    missing_other = all_expected - set(mapped.keys()) - critical_keys
    if missing_other:
        print(f"  [SAE] Non-critical keys using zero-init: {missing_other}")
        # Provide zero tensors for non-critical missing keys
        for key in missing_other:
            mapped[key] = torch.zeros_like(model_params[key])

    return mapped
