from __future__ import annotations

import argparse

import torch

from .model import load_local_model_and_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local inference with the configured Llama model.")
    parser.add_argument("--prompt", required=True, help="Prompt text for generation.")
    parser.add_argument("--config", default=None, help="Optional path to model_config.json.")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Override the local model directory. Takes precedence over MODEL_DIR and model_config.json.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--do-sample", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model, tokenizer, config = load_local_model_and_tokenizer(
        args.config,
        model_dir=args.model_dir,
    )
    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Model: {config['model_name']}")
    print(generated)


if __name__ == "__main__":
    main()
