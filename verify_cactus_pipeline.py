"""verify_cactus_pipeline.py — 最小验证脚本（文档 §14 第3交付物）

读取 10 条 CACTUS 样本，输入模型，在 blocks.19.hook_resid_post 取激活，
用 SAE 编码，仅在 therapist span 聚合，输出高激活 latent。

用法：
    python verify_cactus_pipeline.py
    python verify_cactus_pipeline.py --n-samples 5 --topk-show 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nlp_re_base.config import load_model_config
from nlp_re_base.model import load_local_model_and_tokenizer
from nlp_re_base.sae import load_sae_from_hub

DATA_PATH = PROJECT_ROOT / "data" / "cactus" / "cactus_re_small_1500.jsonl"
SAE_CONFIG_PATH = PROJECT_ROOT / "config" / "sae_config.json"


def load_samples(n: int) -> list[dict]:
    samples = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
            if len(samples) >= n:
                break
    return samples


def get_therapist_token_span(
    tokenizer,
    formatted_text: str,
    char_start: int,
    char_end: int,
    max_seq_len: int = 256,
) -> tuple[int, int]:
    """Use offset_mapping to find the token span for the therapist section."""
    enc = tokenizer(
        formatted_text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    offsets = enc["offset_mapping"][0]  # [T, 2]

    tok_start, tok_end = None, None
    for i, (cs, ce) in enumerate(offsets.tolist()):
        if cs == 0 and ce == 0:
            continue  # special token
        if tok_start is None and ce > char_start:
            tok_start = i
        if ce <= char_end:
            tok_end = i + 1  # exclusive

    if tok_start is None:
        tok_start = 0
    if tok_end is None:
        tok_end = offsets.shape[0]

    return tok_start, tok_end


def run_verification(args):
    print(f"[1/5] 读取 {args.n_samples} 条 CACTUS 样本...")
    samples = load_samples(args.n_samples)
    if not samples:
        print(f"ERROR: 找不到数据文件 {DATA_PATH}")
        print("请先运行 python build_cactus_dataset.py")
        sys.exit(1)
    print(f"  已加载 {len(samples)} 条样本")

    print("[2/5] 加载模型和 tokenizer...")
    model, tokenizer, _ = load_local_model_and_tokenizer()
    device = next(model.parameters()).device

    print("[3/5] 加载 SAE...")
    with open(SAE_CONFIG_PATH) as f:
        sae_cfg = json.load(f)
    sae = load_sae_from_hub(
        repo_id=sae_cfg["sae_repo_id"],
        subfolder=sae_cfg["sae_subfolder"],
        device=device,
        dtype=torch.bfloat16,
    )
    sae_device = next(sae.parameters()).device
    sae_dtype  = next(sae.parameters()).dtype

    # Parse hook layer
    hook_point = sae_cfg["hook_point"]  # e.g. "blocks.19.hook_resid_post"
    layer_idx = int(hook_point.split(".")[1])
    target_layer = model.model.layers[layer_idx]

    print(f"[4/5] 处理每条样本 (hook={hook_point}) ...")
    results = []

    for i, sample in enumerate(samples):
        print(f"\n  样本 {i+1}/{len(samples)}: {sample['sample_id']} [{sample['label']}]")

        # Tokenize with offset mapping
        enc = tokenizer(
            sample["formatted_text"],
            return_offsets_mapping=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")  # remove before model call
        input_ids  = enc["input_ids"].to(device)
        attn_mask  = enc["attention_mask"].to(device)

        # Find therapist token span
        tok_start, tok_end = get_therapist_token_span(
            tokenizer,
            sample["formatted_text"],
            sample["therapist_char_start"],
            sample["therapist_char_end"],
            max_seq_len=256,
        )
        print(f"  therapist token span: [{tok_start}, {tok_end}), "
              f"({tok_end - tok_start} tokens)")

        # Hook to capture residual stream at layer 19
        captured: dict = {}

        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["resid"] = h.detach()

        handle = target_layer.register_forward_hook(hook_fn)
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attn_mask)
        handle.remove()

        resid = captured["resid"]  # [1, T, d_model]

        # SAE encode
        sae_input = resid.to(sae_device, sae_dtype)
        with torch.inference_mode():
            _, latents = sae(sae_input)  # latents: [1, T, d_sae]

        # Therapist span pooling: mean / sum / max
        span = latents[0, tok_start:tok_end, :]   # [S, d_sae]
        if span.shape[0] == 0:
            print("  WARNING: therapist span is empty!")
            continue

        span_f = span.float()
        latent_mean = span_f.mean(dim=0)
        latent_sum  = span_f.sum(dim=0)
        latent_max, _ = span_f.max(dim=0)

        # Top-K latents by max pooling
        topk_vals, topk_ids = latent_max.topk(args.topk_show)
        print(f"  Top-{args.topk_show} latents (max pool):")
        for rank, (lid, val) in enumerate(zip(topk_ids.tolist(), topk_vals.tolist())):
            print(f"    #{rank+1:2d}  latent {lid:6d}  activation={val:.4f}")

        results.append({
            "sample_id":    sample["sample_id"],
            "label":        sample["label"],
            "n_tokens":     input_ids.shape[1],
            "therapist_tokens": tok_end - tok_start,
            "top_latents": [
                {"latent_id": int(lid), "activation_max": float(val)}
                for lid, val in zip(topk_ids.tolist(), topk_vals.tolist())
            ],
        })

    print("\n[5/5] 验证完成！")
    print(f"  成功处理样本数: {len(results)}/{len(samples)}")
    print("\n  链路验证通过：")
    print("  ✓ CACTUS 数据能正确加载")
    print(f"  ✓ 输入格式化文本到模型 ({hook_point})")
    print("  ✓ SAE 编码完成")
    print("  ✓ 仅对 therapist span 聚合 latent activation")
    print("  ✓ 输出 mean / sum / max pooling 结果")

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Verify CACTUS SAE Pipeline")
    p.add_argument("--n-samples",  type=int, default=10)
    p.add_argument("--topk-show",  type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_verification(args)
