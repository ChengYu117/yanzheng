from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import load_model_config, resolve_repo_path


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _has_accelerate() -> bool:
    return importlib.util.find_spec("accelerate") is not None


def _has_transformer_lens() -> bool:
    return importlib.util.find_spec("transformer_lens") is not None


def is_transformer_lens_model(model: object) -> bool:
    """Return True when `model` is a TransformerLens HookedTransformer."""
    return hasattr(model, "run_with_cache") and hasattr(model, "run_with_hooks")


def _resolve_target_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _resolve_transformer_lens_model_name(config: dict, model_path: Path) -> str:
    configured = config.get("model_name")
    if configured:
        aliases = {
            "Meta-Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
            "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
        }
        return aliases.get(configured, configured)
    return model_path.name


def _load_hf_model_and_tokenizer(
    *,
    model_path: Path,
    torch_dtype: torch.dtype,
    device_map: str | None,
    device: str | torch.device | None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"dtype": torch_dtype}
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    if device_map is None:
        model = model.to(_resolve_target_device(device))
    model.eval()
    return model, tokenizer


def _load_transformer_lens_model_and_tokenizer(
    *,
    model_name: str,
    model_path: Path,
    torch_dtype: torch.dtype,
    device: str | torch.device | None,
):
    from transformer_lens import HookedTransformer

    target_device = _resolve_target_device(device)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tl_model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        hf_model=hf_model,
        hf_config=hf_model.config,
        tokenizer=tokenizer,
        dtype=torch_dtype,
        device=target_device,
    )
    tl_model.eval()
    return tl_model, tokenizer


def load_local_model_and_tokenizer(
    config_path: str | Path | None = None,
    *,
    model_dir: str | Path | None = None,
    device: str | torch.device | None = None,
):
    config = load_model_config(config_path, model_dir=model_dir)
    model_path = Path(config["model_path"])
    if not model_path.is_absolute():
        model_path = resolve_repo_path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            "Local model path not found: "
            f"{model_path}. Set --model-dir or MODEL_DIR to the downloaded model directory."
        )

    dtype_name = config.get("torch_dtype", "float16")
    torch_dtype = DTYPE_MAP.get(dtype_name, torch.float16)
    device_map = config.get("device_map", "auto")
    backend = config.get("backend", "huggingface")

    if device_map is not None and not _has_accelerate():
        print(
            "accelerate is not installed; falling back from device_map="
            f"{device_map!r} to single-device loading."
        )
        device_map = None

    print(f"Loading local model from: {model_path}")
    print(f"torch_dtype={dtype_name}, device_map={device_map}, backend={backend}")

    if backend == "transformer_lens" and _has_transformer_lens():
        if device_map not in (None, "auto"):
            print(
                "TransformerLens backend ignores custom device_map values; "
                "loading on a single target device."
            )
        if device_map == "auto":
            print("TransformerLens backend selected; ignoring device_map='auto'.")
        model_name = _resolve_transformer_lens_model_name(config, model_path)
        model, tokenizer = _load_transformer_lens_model_and_tokenizer(
            model_name=model_name,
            model_path=model_path,
            torch_dtype=torch_dtype,
            device=device,
        )
        config["resolved_backend"] = "transformer_lens"
        return model, tokenizer, config

    model, tokenizer = _load_hf_model_and_tokenizer(
        model_path=model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        device=device,
    )
    config["resolved_backend"] = "huggingface"
    return model, tokenizer, config
