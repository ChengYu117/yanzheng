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

    if device_map is not None and not _has_accelerate():
        print(
            "accelerate is not installed; falling back from device_map="
            f"{device_map!r} to single-device loading."
        )
        device_map = None

    print(f"Loading local model from: {model_path}")
    print(f"torch_dtype={dtype_name}, device_map={device_map}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"dtype": torch_dtype}
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    if device_map is None:
        target_device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model = model.to(target_device)
    model.eval()
    return model, tokenizer, config
