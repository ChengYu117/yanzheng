from __future__ import annotations

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.json"


def resolve_repo_path(path: str | Path | None, *, default: Path | None = None) -> Path:
    if path is None:
        if default is None:
            raise ValueError("Either path or default must be provided.")
        return default
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def resolve_output_dir(
    output_dir: str | Path | None,
    *,
    default_subdir: str,
) -> Path:
    if output_dir:
        return Path(output_dir)

    output_root = os.getenv("OUTPUT_ROOT")
    if output_root:
        return Path(output_root) / default_subdir

    return Path("outputs") / default_subdir


def load_model_config(
    config_path: str | Path | None = None,
    *,
    model_dir: str | Path | None = None,
) -> dict:
    path = resolve_repo_path(config_path, default=DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if model_dir:
        config["model_path"] = str(model_dir)
    elif os.getenv("MODEL_DIR"):
        config["model_path"] = os.environ["MODEL_DIR"]

    return config
