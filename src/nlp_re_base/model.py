from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import load_model_config


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def load_local_model_and_tokenizer(config_path: str | Path | None = None):
    config = load_model_config(config_path)
    model_path = Path(config["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Local model path not found: {model_path}")

    dtype_name = config.get("torch_dtype", "float16")
    torch_dtype = DTYPE_MAP.get(dtype_name, torch.float16)
    device_map = config.get("device_map", "auto")

    print(f"Loading local model from: {model_path}")
    print(f"torch_dtype={dtype_name}, device_map={device_map}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()
    return model, tokenizer, config
