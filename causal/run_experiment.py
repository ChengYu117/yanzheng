"""causal/run_experiment.py — End-to-end causal validation orchestrator.

Runs necessity (ablation) and sufficiency (steering) experiments for
G1/G5/G20 latent groups, with random/Bottom-K/orthogonal controls.
Also runs group-structure analysis (cumulative top-K, LOO, synergy).

Usage:
    python causal/run_experiment.py \\
        --candidate-csv outputs/sae_eval/candidate_latents.csv \\
        --re-features   outputs/sae_eval/re_features.pt \\
        --nonre-features outputs/sae_eval/nonre_features.pt \\
        --output-dir    outputs/causal_validation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from nlp_re_base.activations import extract_and_process_streaming
from nlp_re_base.config import resolve_output_dir, resolve_repo_path
from nlp_re_base.data import load_cactus_dataset, load_jsonl
from nlp_re_base.model import load_local_model_and_tokenizer
from nlp_re_base.sae import load_sae_from_hub

try:
    from .data import iter_batches, build_dataset
    from .evaluation import REProbeScorer, score_delta, eval_text_quality
    from .intervention import (
        zero_ablate, mean_ablate, cond_token_ablate, decode_delta,
        constant_steer, cond_input_steer, cond_token_steer,
        make_steering_direction, make_orthogonal_direction, make_random_direction,
        steer_with_direction,
    )
    from .selection import rank_latents, bootstrap_stability, make_bottom_k, make_random_control
except ImportError:
    from causal.data import iter_batches, build_dataset
    from causal.evaluation import REProbeScorer, score_delta, eval_text_quality
    from causal.intervention import (
        zero_ablate, mean_ablate, cond_token_ablate, decode_delta,
        constant_steer, cond_input_steer, cond_token_steer,
        make_steering_direction, make_orthogonal_direction, make_random_direction,
        steer_with_direction,
    )
    from causal.selection import rank_latents, bootstrap_stability, make_bottom_k, make_random_control


# ─────────────────────────────────────────────────────────────────────────────
# Hook-based intervention runner
# ─────────────────────────────────────────────────────────────────────────────

class CausalRunner:
    """Wraps the model + SAE and applies hook-based interventions.

    The intervention is injected into the layer-19 residual stream
    via a forward hook that:
    1) Grabs the raw activations h
    2) Runs the SAE to get latents z and reconstruction z_hat
    3) Applies the specified ablation or steering operation
    4) Patches the modified residual back into the forward pass
    """

    def __init__(
        self,
        model,
        tokenizer,
        sae,
        hook_point: str,
        device: torch.device,
        sae_config: dict,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae
        self.hook_point = hook_point
        self.device = device
        self.sae_config = sae_config

        # Cache SAE dtype
        self.sae_dtype = getattr(sae, "sae_dtype", next(sae.parameters()).dtype)
        self.sae_device = next(sae.parameters()).device

        # Parse layer index from hook_point
        # e.g. "blocks.19.hook_resid_post" → 19
        parts = hook_point.split(".")
        self.layer_idx = int(parts[1])

        # Pre-compute ref mean (population mean latent activations)
        self._ref_mean: torch.Tensor | None = None

    def set_ref_mean(self, ref_mean: torch.Tensor) -> None:
        """Set the reference mean for mean-ablation (§4.2)."""
        self._ref_mean = ref_mean.to(device="cpu")

    def _default_ref_mean(self) -> torch.Tensor:
        d_sae = self.sae.W_dec.shape[1] if self.sae.W_dec.shape[0] < self.sae.W_dec.shape[1] else self.sae.W_dec.shape[0]
        return torch.zeros(d_sae, dtype=torch.float32)

    def _run_forward_with_hook(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_mask: torch.Tensor,
        intervention_fn,  # callable(h, z) -> h_new
    ) -> torch.Tensor:
        """Run model forward pass with hook injecting intervention_fn.

        Returns the modified hidden states at the hook layer [B, T, d_model].
        """
        captured: dict[str, Any] = {}
        target_layer = self.model.model.layers[self.layer_idx]

        def _aligned_span_mask(h_tensor: torch.Tensor) -> torch.Tensor:
            current_mask = span_mask.to(h_tensor.device)
            current_len = h_tensor.shape[1]
            mask_len = current_mask.shape[1]
            if mask_len == current_len:
                return current_mask
            if mask_len < current_len:
                pad = torch.zeros(
                    current_mask.shape[0],
                    current_len - mask_len,
                    dtype=current_mask.dtype,
                    device=current_mask.device,
                )
                return torch.cat([current_mask, pad], dim=1)
            return current_mask[:, :current_len]

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output

            # Run SAE
            sae_input = h.to(device=self.sae_device, dtype=self.sae_dtype)
            with torch.no_grad():
                z_hat, z = self.sae(sae_input)

            # Apply intervention
            h_new = intervention_fn(
                h=h,
                z=z,
                z_hat=z_hat,
                span_mask=_aligned_span_mask(h),
            )

            captured["h_new"] = h_new.detach()
            captured["z_before"] = z.detach().cpu().float()

            sae_input_new = h_new.to(device=self.sae_device, dtype=self.sae_dtype)
            with torch.no_grad():
                z_hat_new, z_new = self.sae(sae_input_new)
            captured["z_hat"] = z_hat_new.detach().cpu().float()
            captured["z"] = z_new.detach().cpu().float()

            # Return modified output
            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new

        handle = target_layer.register_forward_hook(hook_fn)
        try:
            with torch.inference_mode():
                self.model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                )
        finally:
            handle.remove()

        return captured

    def _generate_with_hook(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_mask: torch.Tensor,
        intervention_fn,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """Generate continuations while applying the intervention hook."""
        target_layer = self.model.model.layers[self.layer_idx]

        def _aligned_span_mask(h_tensor: torch.Tensor) -> torch.Tensor:
            current_mask = span_mask.to(h_tensor.device)
            current_len = h_tensor.shape[1]
            mask_len = current_mask.shape[1]
            if mask_len == current_len:
                return current_mask
            if mask_len < current_len:
                pad = torch.zeros(
                    current_mask.shape[0],
                    current_len - mask_len,
                    dtype=current_mask.dtype,
                    device=current_mask.device,
                )
                return torch.cat([current_mask, pad], dim=1)
            return current_mask[:, :current_len]

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output

            sae_input = h.to(device=self.sae_device, dtype=self.sae_dtype)
            with torch.no_grad():
                z_hat, z = self.sae(sae_input)

            h_new = intervention_fn(
                h=h,
                z=z,
                z_hat=z_hat,
                span_mask=_aligned_span_mask(h),
            )
            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new

        handle = target_layer.register_forward_hook(hook_fn)
        try:
            with torch.inference_mode():
                generated = self.model.generate(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            handle.remove()

        return generated

    def run_baseline(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run forward pass with NO intervention; return z and h."""
        def identity_fn(h, z, z_hat, span_mask):
            return h

        return self._run_forward_with_hook(
            input_ids, attention_mask, span_mask, identity_fn
        )

    def run_zero_ablation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_mask: torch.Tensor,
        latent_ids: list[int],
    ) -> dict[str, torch.Tensor]:
        def ablate_fn(h, z, z_hat, span_mask):
            z_new = zero_ablate(z.cpu().float(), span_mask.cpu(), latent_ids)
            delta_z = (z_new - z.cpu().float()).to(h.dtype).to(h.device)
            h_new = h + decode_delta(delta_z.to(h.device), self.sae.W_dec.to(h.device, h.dtype))
            return h_new

        return self._run_forward_with_hook(
            input_ids, attention_mask, span_mask, ablate_fn
        )

    def run_mean_ablation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_mask: torch.Tensor,
        latent_ids: list[int],
    ) -> dict[str, torch.Tensor]:
        ref_mean = self._ref_mean if self._ref_mean is not None else self._default_ref_mean()

        def ablate_fn(h, z, z_hat, span_mask):
            z_new = mean_ablate(z.cpu().float(), span_mask.cpu(), latent_ids, ref_mean)
            delta_z = (z_new - z.cpu().float()).to(h.dtype).to(h.device)
            h_new = h + decode_delta(delta_z.to(h.device), self.sae.W_dec.to(h.device, h.dtype))
            return h_new

        return self._run_forward_with_hook(
            input_ids, attention_mask, span_mask, ablate_fn
        )

    def run_cond_token_ablation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_mask: torch.Tensor,
        latent_ids: list[int],
        tau: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        def ablate_fn(h, z, z_hat, span_mask):
            z_new = cond_token_ablate(z.cpu().float(), span_mask.cpu(), latent_ids, tau=tau)
            delta_z = (z_new - z.cpu().float()).to(h.dtype).to(h.device)
            h_new = h + decode_delta(delta_z.to(h.device), self.sae.W_dec.to(h.device, h.dtype))
            return h_new

        return self._run_forward_with_hook(
            input_ids, attention_mask, span_mask, ablate_fn
        )

    def run_constant_steer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_mask: torch.Tensor,
        latent_ids: list[int],
        weights: list[float],
        strength: float,
    ) -> dict[str, torch.Tensor]:
        W_dec = self.sae.W_dec.detach().cpu()

        def steer_fn(h, z, z_hat, span_mask):
            h_new = constant_steer(
                resid=h.cpu().float(),
                span_mask=span_mask.cpu(),
                latent_ids=latent_ids,
                weights=weights,
                W_dec=W_dec,
                strength=strength,
            )
            return h_new.to(h.dtype).to(h.device)

        return self._run_forward_with_hook(
            input_ids, attention_mask, span_mask, steer_fn
        )

    def run_cond_input_steer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_mask: torch.Tensor,
        latent_ids: list[int],
        weights: list[float],
        strength: float,
        tau: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        W_dec = self.sae.W_dec.detach().cpu()

        def steer_fn(h, z, z_hat, span_mask):
            h_new = cond_input_steer(
                z=z.cpu().float(),
                resid=h.cpu().float(),
                span_mask=span_mask.cpu(),
                latent_ids=latent_ids,
                weights=weights,
                W_dec=W_dec,
                strength=strength,
                tau=tau,
            )
            return h_new.to(h.dtype).to(h.device)

        return self._run_forward_with_hook(
            input_ids, attention_mask, span_mask, steer_fn
        )

    def run_cond_token_steer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_mask: torch.Tensor,
        latent_ids: list[int],
        weights: list[float],
        strength: float,
        tau: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        W_dec = self.sae.W_dec.detach().cpu()

        def steer_fn(h, z, z_hat, span_mask):
            h_new = cond_token_steer(
                z=z.cpu().float(),
                resid=h.cpu().float(),
                span_mask=span_mask.cpu(),
                latent_ids=latent_ids,
                weights=weights,
                W_dec=W_dec,
                strength=strength,
                tau=tau,
            )
            return h_new.to(h.dtype).to(h.device)

        return self._run_forward_with_hook(
            input_ids, attention_mask, span_mask, steer_fn
        )

    def run_direction_steer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_mask: torch.Tensor,
        direction: torch.Tensor,   # [d_model]
        strength: float,
    ) -> dict[str, torch.Tensor]:
        """Apply a pre-computed direction (for controls)."""
        def steer_fn(h, z, z_hat, span_mask):
            dir_h = direction.to(h.dtype).to(h.device)
            h_new = steer_with_direction(h, span_mask, dir_h, strength)
            return h_new

        return self._run_forward_with_hook(
            input_ids, attention_mask, span_mask, steer_fn
        )

    def generate_baseline(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_mask: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        def identity_fn(h, z, z_hat, span_mask):
            return h

        return self._generate_with_hook(
            input_ids, attention_mask, span_mask, identity_fn, max_new_tokens
        )

    def generate_cond_token_steer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_mask: torch.Tensor,
        latent_ids: list[int],
        weights: list[float],
        strength: float,
        max_new_tokens: int,
        tau: float = 0.0,
    ) -> torch.Tensor:
        W_dec = self.sae.W_dec.detach().cpu()

        def steer_fn(h, z, z_hat, span_mask):
            h_new = cond_token_steer(
                z=z.cpu().float(),
                resid=h.cpu().float(),
                span_mask=span_mask.cpu(),
                latent_ids=latent_ids,
                weights=weights,
                W_dec=W_dec,
                strength=strength,
                tau=tau,
            )
            return h_new.to(h.dtype).to(h.device)

        return self._generate_with_hook(
            input_ids, attention_mask, span_mask, steer_fn, max_new_tokens
        )

    def generate_direction_steer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_mask: torch.Tensor,
        direction: torch.Tensor,
        strength: float,
        max_new_tokens: int,
    ) -> torch.Tensor:
        def steer_fn(h, z, z_hat, span_mask):
            dir_h = direction.to(h.dtype).to(h.device)
            return steer_with_direction(h, span_mask, dir_h, strength)

        return self._generate_with_hook(
            input_ids, attention_mask, span_mask, steer_fn, max_new_tokens
        )


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helper (reuse streaming pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def extract_utterance_features(
    model, tokenizer, sae, texts, hook_point, device,
    batch_size=4, max_seq_len=128,
    aggregation: str = "max",
    binarized_threshold: float = 0.0,
) -> np.ndarray:
    """Extract utterance-level SAE features [N, d_sae] for scoring."""
    result = extract_and_process_streaming(
        model=model, tokenizer=tokenizer, sae=sae, texts=texts,
        hook_point=hook_point, max_seq_len=max_seq_len,
        batch_size=batch_size,
        aggregation=aggregation,
        binarized_threshold=binarized_threshold,
        device=device, collect_structural_samples=0,
    )
    return result["utterance_features"].numpy()


def compute_reference_latent_mean(
    runner: CausalRunner,
    batches,
) -> torch.Tensor:
    """Compute token-level latent reference mean over the counselor span."""
    running_sum: torch.Tensor | None = None
    token_count = 0

    for batch in batches:
        cap = runner.run_baseline(
            batch.input_ids,
            batch.attention_mask,
            batch.counselor_span_mask,
        )
        z_tokens = cap["z"]  # post-hook SAE latents [B, T, d_sae]
        span_mask = batch.counselor_span_mask.cpu()
        masked = z_tokens[span_mask]
        if masked.numel() == 0:
            continue
        if running_sum is None:
            running_sum = masked.sum(dim=0)
        else:
            running_sum += masked.sum(dim=0)
        token_count += masked.shape[0]

    if running_sum is None or token_count == 0:
        raise ValueError("Failed to compute token-level reference latent mean.")

    return running_sum / float(token_count)


def _decode_continuations(
    tokenizer,
    output_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> list[str]:
    """Decode only the newly generated continuation tokens."""
    decoded: list[str] = []
    input_lengths = attention_mask.sum(dim=1).tolist()
    for row, input_len in zip(output_ids, input_lengths):
        continuation = row[int(input_len):]
        decoded.append(tokenizer.decode(continuation, skip_special_tokens=True).strip())
    return decoded


def _select_side_effect_subset(
    texts: list[str],
    labels: list[int],
    max_samples: int,
) -> tuple[list[str], list[int]]:
    """Pick a balanced subset of RE and NonRE prompts for generation checks."""
    if max_samples <= 0 or max_samples >= len(texts):
        return texts, labels

    re_idx = [i for i, y in enumerate(labels) if y == 1]
    nonre_idx = [i for i, y in enumerate(labels) if y == 0]
    half = max(1, max_samples // 2)
    picked = re_idx[:half] + nonre_idx[:half]
    if len(picked) < max_samples:
        remaining = [i for i in range(len(texts)) if i not in picked]
        picked.extend(remaining[: max_samples - len(picked)])
    picked = picked[:max_samples]
    return [texts[i] for i in picked], [labels[i] for i in picked]


# ─────────────────────────────────────────────────────────────────────────────
# Core experiment loops
# ─────────────────────────────────────────────────────────────────────────────

def run_necessity_experiment(
    runner: CausalRunner,
    batches,
    probe: REProbeScorer,
    groups: dict[str, list[int]],
    controls: dict[str, list[int]],
    true_labels: np.ndarray,
    model, tokenizer, sae, hook_point, device,
    batch_size: int = 4,
    max_seq_len: int = 128,
    aggregation: str = "max",
    binarized_threshold: float = 0.0,
) -> dict[str, Any]:
    """Run zero/mean/cond_token ablation for all G groups and controls.

    Returns a structured dict for Table 1 (necessity).
    """
    results: dict[str, Any] = {}

    all_groups = {**{f"G{k}": v for k, v in [
        ("1", groups["G1"]), ("5", groups["G5"]), ("20", groups["G20"])
    ]}, **controls}

    # First get baseline scores (no intervention)
    print("  [Necessity] Computing baseline features...")
    all_texts = [t for b in batches for t in b.texts]
    baseline_feats = extract_utterance_features(
        model, tokenizer, sae, all_texts, hook_point, device,
        batch_size=batch_size, max_seq_len=max_seq_len,
        aggregation=aggregation,
        binarized_threshold=binarized_threshold,
    )
    baseline_logits = probe.score_features(baseline_feats)

    for group_name, latent_ids in all_groups.items():
        results[group_name] = {}
        for mode in ["zero", "mean", "cond_token"]:
            print(f"  [Necessity] {group_name} / {mode} ablation ({len(latent_ids)} latents)...")
            intervened_feats_list = []

            for batch in batches:
                if mode == "zero":
                    cap = runner.run_zero_ablation(
                        batch.input_ids, batch.attention_mask,
                        batch.counselor_span_mask, latent_ids,
                    )
                elif mode == "mean":
                    cap = runner.run_mean_ablation(
                        batch.input_ids, batch.attention_mask,
                        batch.counselor_span_mask, latent_ids,
                    )
                else:
                    cap = runner.run_cond_token_ablation(
                        batch.input_ids, batch.attention_mask,
                        batch.counselor_span_mask, latent_ids,
                    )

                # Re-run SAE on modified z to get intervened features
                z_intervened = cap.get("z", None)
                if z_intervened is not None:
                    feat = _pool_features(
                        z_intervened,
                        batch.counselor_span_mask.bool(),
                        method=aggregation,
                        threshold=binarized_threshold,
                    )
                    intervened_feats_list.append(feat)

            if intervened_feats_list:
                intervened_feats = np.concatenate(intervened_feats_list, axis=0)
                intervened_logits = probe.score_features(intervened_feats)
                delta = score_delta(baseline_logits, intervened_logits, true_labels)
                results[group_name][mode] = delta
            else:
                results[group_name][mode] = {"error": "no features collected"}

    return results


def run_sufficiency_experiment(
    runner: CausalRunner,
    batches,
    probe: REProbeScorer,
    groups: dict[str, list[int]],
    probe_weights: dict[str, list[float]],
    controls_directions: dict[str, torch.Tensor],
    true_labels: np.ndarray,
    model, tokenizer, sae, hook_point, device,
    lambdas: list[float] | None = None,
    batch_size: int = 4,
    max_seq_len: int = 128,
    aggregation: str = "max",
    binarized_threshold: float = 0.0,
) -> dict[str, Any]:
    """Run constant / cond_input / cond_token steering for G groups and controls."""
    if lambdas is None:
        lambdas = [0.5, 1.0, 1.5, 2.0]

    results: dict[str, Any] = {}

    all_texts = [t for b in batches for t in b.texts]
    print("  [Sufficiency] Computing baseline features...")
    baseline_feats = extract_utterance_features(
        model, tokenizer, sae, all_texts, hook_point, device,
        batch_size=batch_size, max_seq_len=max_seq_len,
        aggregation=aggregation,
        binarized_threshold=binarized_threshold,
    )
    baseline_logits = probe.score_features(baseline_feats)

    for gname in ["G1", "G5", "G20"]:
        latent_ids = groups[gname]
        weights = probe_weights.get(gname, [1.0 / len(latent_ids)] * len(latent_ids))
        results[gname] = {
            "constant": {},
            "cond_input": {},
            "cond_token": {},
        }

        for mode in ["constant", "cond_input", "cond_token"]:
            for lam in lambdas:
                print(f"  [Sufficiency] {gname} {mode} steer λ={lam}...")
                feats_list = []
                for batch in batches:
                    if mode == "constant":
                        cap = runner.run_constant_steer(
                            batch.input_ids, batch.attention_mask,
                            batch.counselor_span_mask, latent_ids, weights, lam,
                        )
                    elif mode == "cond_input":
                        cap = runner.run_cond_input_steer(
                            batch.input_ids, batch.attention_mask,
                            batch.counselor_span_mask, latent_ids, weights, lam,
                        )
                    else:
                        cap = runner.run_cond_token_steer(
                            batch.input_ids, batch.attention_mask,
                            batch.counselor_span_mask, latent_ids, weights, lam,
                        )

                    z_out = cap.get("z", None)
                    if z_out is not None:
                        feats_list.append(_pool_features(
                            z_out,
                            batch.counselor_span_mask.bool(),
                            method=aggregation,
                            threshold=binarized_threshold,
                        ))

                if feats_list:
                    feats = np.concatenate(feats_list, axis=0)
                    logits = probe.score_features(feats)
                    results[gname][mode][f"lam_{lam}"] = score_delta(
                        baseline_logits, logits, true_labels
                    )

    for ctrl_name, direction in controls_directions.items():
        results[ctrl_name] = {"direction": {}}
        for lam in lambdas:
            print(f"  [Sufficiency] {ctrl_name} control direction λ={lam}...")
            feats_list = []
            for batch in batches:
                cap = runner.run_direction_steer(
                    batch.input_ids, batch.attention_mask,
                    batch.counselor_span_mask, direction, lam,
                )
                z_out = cap.get("z", None)
                if z_out is not None:
                    feats_list.append(_pool_features(
                        z_out,
                        batch.counselor_span_mask.bool(),
                        method=aggregation,
                        threshold=binarized_threshold,
                    ))

            if feats_list:
                feats = np.concatenate(feats_list, axis=0)
                logits = probe.score_features(feats)
                results[ctrl_name]["direction"][f"lam_{lam}"] = score_delta(
                    baseline_logits, logits, true_labels
                )

    return results


def run_side_effect_evaluation(
    runner: CausalRunner,
    probe: REProbeScorer,
    groups: dict[str, list[int]],
    probe_weights: dict[str, list[float]],
    controls_directions: dict[str, torch.Tensor],
    texts: list[str],
    labels: list[int],
    tokenizer,
    sae,
    hook_point: str,
    device: torch.device,
    batch_size: int = 4,
    max_seq_len: int = 128,
    max_samples: int = 16,
    max_new_tokens: int = 24,
    lambda_value: float = 1.0,
    aggregation: str = "max",
    binarized_threshold: float = 0.0,
) -> dict[str, Any]:
    """Generate continuations under steering and compute lightweight side-effect proxies."""
    subset_texts, subset_labels = _select_side_effect_subset(texts, labels, max_samples=max_samples)
    if not subset_texts:
        return {}

    batches = iter_batches(
        subset_texts,
        subset_labels,
        tokenizer,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        device=device,
    )

    baseline_outputs: list[str] = []
    for batch in batches:
        generated = runner.generate_baseline(
            batch.input_ids,
            batch.attention_mask,
            batch.counselor_span_mask,
            max_new_tokens=max_new_tokens,
        )
        baseline_outputs.extend(_decode_continuations(tokenizer, generated.cpu(), batch.attention_mask.cpu()))

    baseline_features = extract_utterance_features(
        runner.model, tokenizer, sae, baseline_outputs, hook_point, device,
        batch_size=batch_size, max_seq_len=max_seq_len,
        aggregation=aggregation,
        binarized_threshold=binarized_threshold,
    )
    baseline_logits = probe.score_features(baseline_features)

    results: dict[str, Any] = {
        "config": {
            "max_samples": len(subset_texts),
            "max_new_tokens": max_new_tokens,
            "lambda": lambda_value,
            "modes": ["cond_token", "direction"],
            "aggregation": aggregation,
            "binarized_threshold": binarized_threshold,
        },
        "baseline_generation_quality": eval_text_quality(baseline_outputs),
        "groups": {},
        "controls": {},
    }

    for gname in ["G1", "G5", "G20"]:
        latent_ids = groups[gname]
        weights = probe_weights.get(gname, [1.0 / len(latent_ids)] * len(latent_ids))
        generated_texts: list[str] = []
        for batch in batches:
            generated = runner.generate_cond_token_steer(
                batch.input_ids,
                batch.attention_mask,
                batch.counselor_span_mask,
                latent_ids,
                weights,
                lambda_value,
                max_new_tokens=max_new_tokens,
            )
            generated_texts.extend(_decode_continuations(tokenizer, generated.cpu(), batch.attention_mask.cpu()))

        feats = extract_utterance_features(
            runner.model, tokenizer, sae, generated_texts, hook_point, device,
            batch_size=batch_size, max_seq_len=max_seq_len,
            aggregation=aggregation,
            binarized_threshold=binarized_threshold,
        )
        logits = probe.score_features(feats)
        results["groups"][gname] = {
            "mode": "cond_token",
            "quality": eval_text_quality(baseline_outputs, generated_texts),
            "mean_generated_re_logit": float(np.mean(logits)),
            "mean_generated_re_logit_delta": float(np.mean(logits - baseline_logits)),
            "sample_outputs": generated_texts[:5],
        }

    for ctrl_name, direction in controls_directions.items():
        generated_texts = []
        for batch in batches:
            generated = runner.generate_direction_steer(
                batch.input_ids,
                batch.attention_mask,
                batch.counselor_span_mask,
                direction,
                lambda_value,
                max_new_tokens=max_new_tokens,
            )
            generated_texts.extend(_decode_continuations(tokenizer, generated.cpu(), batch.attention_mask.cpu()))

        feats = extract_utterance_features(
            runner.model, tokenizer, sae, generated_texts, hook_point, device,
            batch_size=batch_size, max_seq_len=max_seq_len,
            aggregation=aggregation,
            binarized_threshold=binarized_threshold,
        )
        logits = probe.score_features(feats)
        results["controls"][ctrl_name] = {
            "mode": "direction",
            "quality": eval_text_quality(baseline_outputs, generated_texts),
            "mean_generated_re_logit": float(np.mean(logits)),
            "mean_generated_re_logit_delta": float(np.mean(logits - baseline_logits)),
            "sample_outputs": generated_texts[:5],
        }

    return results


def run_group_structure_experiment(
    runner: CausalRunner,
    batches,
    probe: REProbeScorer,
    ranked_latents: list[int],  # full top-20 in ranked order
    probe_weights_full: list[float],
    true_labels: np.ndarray,
    model, tokenizer, sae, hook_point, device,
    strength: float = 1.0,
    batch_size: int = 4,
    max_seq_len: int = 128,
    aggregation: str = "max",
    binarized_threshold: float = 0.0,
) -> dict[str, Any]:
    """Cumulative top-K curve, Leave-One-Out, Add-One-In, Synergy. (§7)"""
    results: dict[str, Any] = {
        "cumulative_topk": {},
        "leave_one_out": {},
        "add_one_in": {},
        "synergy": {},
    }

    all_texts = [t for b in batches for t in b.texts]
    baseline_feats = extract_utterance_features(
        model, tokenizer, sae, all_texts, hook_point, device,
        batch_size=batch_size, max_seq_len=max_seq_len,
        aggregation=aggregation,
        binarized_threshold=binarized_threshold,
    )
    baseline_logits = probe.score_features(baseline_feats)

    def _steer_and_score(latent_ids, weights, lam=strength):
        feats_list = []
        for batch in batches:
            cap = runner.run_cond_token_steer(
                batch.input_ids, batch.attention_mask,
                batch.counselor_span_mask, latent_ids, weights, lam,
            )
            z_out = cap.get("z", None)
            if z_out is not None:
                feats_list.append(_pool_features(
                    z_out,
                    batch.counselor_span_mask.bool(),
                    method=aggregation,
                    threshold=binarized_threshold,
                ))
        if not feats_list:
            return float("nan")
        feats = np.concatenate(feats_list, axis=0)
        logits = probe.score_features(feats)
        delta = score_delta(baseline_logits, logits, true_labels)
        return delta["mean_delta_re"]

    max_k = min(len(ranked_latents), 20)

    # ── Cumulative top-K ──
    print("  [Group Structure] Cumulative top-K curve...")
    cum_effects = []
    individual_effects = []
    for k in range(1, max_k + 1):
        lids = ranked_latents[:k]
        w = probe_weights_full[:k]
        w_norm = [x / (sum(w) + 1e-12) for x in w]
        effect = _steer_and_score(lids, w_norm)
        cum_effects.append({"k": k, "latent_idx": lids[-1], "mean_delta_re": effect})

        # Individual effect for each latent (for synergy)
        ind = _steer_and_score([lids[-1]], [1.0])
        individual_effects.append(ind)

    results["cumulative_topk"] = cum_effects

    # ── Leave-One-Out (on top-10 for speed) ──
    print("  [Group Structure] Leave-one-out...")
    loo_k = min(10, max_k)
    full_lids = ranked_latents[:loo_k]
    full_w = probe_weights_full[:loo_k]
    full_w_norm = [x / (sum(full_w) + 1e-12) for x in full_w]
    full_effect = _steer_and_score(full_lids, full_w_norm)

    loo_results = []
    for i, lid in enumerate(full_lids):
        remaining_ids = [x for j, x in enumerate(full_lids) if j != i]
        remaining_w   = [x for j, x in enumerate(full_w_norm) if j != i]
        if not remaining_ids:
            loo_results.append({"latent_idx": lid, "loo_effect": 0.0, "delta_loo": full_effect})
            continue
        remaining_w_norm = [x / (sum(remaining_w) + 1e-12) for x in remaining_w]
        loo_effect = _steer_and_score(remaining_ids, remaining_w_norm)
        loo_results.append({
            "latent_idx": lid,
            "full_effect": full_effect,
            "loo_effect":  loo_effect,
            "delta_loo":   full_effect - loo_effect,  # contribution of latent i
        })
    results["leave_one_out"] = loo_results

    print("  [Group Structure] Add-one-in...")
    add_one_in = []
    prev_effect = None
    for k in range(1, loo_k + 1):
        lids = full_lids[:k]
        w = full_w[:k]
        w_norm = [x / (sum(w) + 1e-12) for x in w]
        effect = _steer_and_score(lids, w_norm)
        add_one_in.append({
            "k": k,
            "latent_idx": lids[-1],
            "effect": effect,
            "delta_add": effect if prev_effect is None else effect - prev_effect,
        })
        prev_effect = effect
    results["add_one_in"] = add_one_in

    # ── Synergy (§7.4) ──
    sum_individual = sum(e for e in individual_effects[:loo_k] if not np.isnan(e))
    synergy = full_effect - sum_individual
    results["synergy"] = {
        "full_effect": full_effect,
        "sum_individual_effects": sum_individual,
        "synergy_score": synergy,
        "interpretation": (
            "positive (super-additive)" if synergy > 0.1 else
            "near-zero (additive)"      if abs(synergy) <= 0.1 else
            "negative (redundant)"
        ),
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _max_pool_features(z: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Max-pool z [B, T, d_sae] → [B, d_sae] using mask."""
    z = z.float()
    mask3d = mask.unsqueeze(-1).float().to(z.device)
    # Mask out padding with -inf before max
    z_masked = z + (1 - mask3d) * (-1e9)
    pooled = z_masked.max(dim=1).values  # [B, d_sae]
    return pooled.cpu().numpy()


def _pool_features(
    z: torch.Tensor,
    mask: torch.Tensor,
    method: str = "max",
    threshold: float = 0.0,
) -> np.ndarray:
    """Pool token-level latents [B, T, d_sae] to [B, d_sae]."""
    z = z.float()
    mask_bool = mask.bool().to(z.device)
    mask3d = mask_bool.unsqueeze(-1).float()

    if method == "max":
        z_masked = z + (1 - mask3d) * (-1e9)
        pooled = z_masked.max(dim=1).values
        pooled[pooled < -1e8] = 0.0
        return pooled.cpu().numpy()
    if method == "sum":
        return (z * mask3d).sum(dim=1).cpu().numpy()
    if method == "binarized_sum":
        summed = (z * mask3d).sum(dim=1)
        return (summed > threshold).to(z.dtype).cpu().numpy()
    raise ValueError(f"Unknown pooling method: {method}")


def _normalise_probe_weights(
    probe: REProbeScorer,
    latent_ids: list[int],
) -> list[float]:
    """Extract and normalise probe weights for the given latent ids."""
    model_weights = probe.probe_state["model"].weight.data.squeeze().detach().cpu().numpy()
    # model_weights is over candidate_indices
    candidate_to_pos = {lid: i for i, lid in enumerate(probe.candidate_indices)}
    raw_weights = [
        float(model_weights[candidate_to_pos[lid]])
        for lid in latent_ids
        if lid in candidate_to_pos
    ]
    if not raw_weights:
        return [1.0 / len(latent_ids)] * len(latent_ids)
    total = sum(abs(w) for w in raw_weights) + 1e-12
    return [w / total for w in raw_weights]


def _stabilize_group(
    ranked_latents: list[int],
    stable_latents: list[int],
    k: int,
) -> list[int]:
    """Prefer stability-filtered latents, then fill from ranked order."""
    selected: list[int] = []
    for lid in stable_latents:
        if lid not in selected:
            selected.append(int(lid))
        if len(selected) >= k:
            return selected[:k]
    for lid in ranked_latents:
        if lid not in selected:
            selected.append(int(lid))
        if len(selected) >= k:
            break
    return selected[:k]


def _render_table(header: list[str], rows: list[list]) -> str:
    """Render a simple markdown table."""
    sep = "| " + " | ".join(["---"] * len(header)) + " |"
    lines = ["| " + " | ".join(str(h) for h in header) + " |", sep]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Report generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_summary_tables(
    necessity: dict,
    sufficiency: dict,
    group_structure: dict,
    side_effects: dict,
    output_path: Path,
) -> None:
    lines = ["# Causal Validation — Summary Tables\n"]

    # Table 1: Necessity
    lines.append("## Table 1: Necessity (Ablation)\n")
    lines.append("Mean Δ-logit for RE samples after ablation "
                 "(negative = RE signal reduced).\n")
    header = ["Group", "Mode", "Δ-logit RE", "Δ-logit NonRE", "Fraction↑"]
    rows = []
    for gname, modes in necessity.items():
        for mode, delta in modes.items():
            if isinstance(delta, dict) and "mean_delta_re" in delta:
                rows.append([
                    gname, mode,
                    f"{delta['mean_delta_re']:+.3f}",
                    f"{delta['mean_delta_nonre']:+.3f}",
                    f"{delta['fraction_improved']:.2f}",
                ])
    lines.append(_render_table(header, rows))

    # Table 2: Sufficiency
    lines.append("\n\n## Table 2: Sufficiency (Steering)\n")
    lines.append("Mean Δ-logit for RE samples after steering "
                 "(positive = RE signal increased).\n")
    header = ["Group", "Mode", "λ", "Δ-logit RE", "Δ-logit NonRE", "Fraction↑"]
    rows = []
    for gname, mode_payload in sufficiency.items():
        for mode, lambdas in mode_payload.items():
            if not isinstance(lambdas, dict):
                continue
            for lname, delta in lambdas.items():
                if isinstance(delta, dict) and "mean_delta_re" in delta:
                    lam_val = lname.replace("lam_", "")
                    rows.append([
                        gname, mode, lam_val,
                        f"{delta['mean_delta_re']:+.3f}",
                        f"{delta['mean_delta_nonre']:+.3f}",
                        f"{delta['fraction_improved']:.2f}",
                    ])
    lines.append(_render_table(header, rows))

    lines.append("\n\n## Table 3: Selectivity / Side Effects\n")
    lines.append("Generation-time lexical proxy metrics on a small intervention subset.\n")
    header = ["Name", "Mode", "Delta RE logit", "Retention", "Delta TTR", "Delta Repeat"]
    rows = []
    for name, payload in side_effects.get("groups", {}).items():
        quality = payload.get("quality", {})
        rows.append([
            name,
            payload.get("mode", ""),
            f"{payload.get('mean_generated_re_logit_delta', 0.0):+.3f}",
            f"{quality.get('mean_content_retention', 0.0):.3f}",
            f"{quality.get('delta_ttr', 0.0):+.3f}",
            f"{quality.get('delta_bigram_repetition', 0.0):+.3f}",
        ])
    for name, payload in side_effects.get("controls", {}).items():
        quality = payload.get("quality", {})
        rows.append([
            name,
            payload.get("mode", ""),
            f"{payload.get('mean_generated_re_logit_delta', 0.0):+.3f}",
            f"{quality.get('mean_content_retention', 0.0):.3f}",
            f"{quality.get('delta_ttr', 0.0):+.3f}",
            f"{quality.get('delta_bigram_repetition', 0.0):+.3f}",
        ])
    lines.append(_render_table(header, rows))

    # Table 4: Group Structure
    lines.append("\n\n## Table 4: Group Structure\n")

    lines.append("### Cumulative Top-K\n")
    ck = group_structure.get("cumulative_topk", [])
    header = ["K", "Latent added", "Cumulative Δ-logit RE"]
    rows = [[r["k"], r["latent_idx"], f"{r['mean_delta_re']:+.3f}"]
            for r in ck if "mean_delta_re" in r]
    lines.append(_render_table(header, rows))

    lines.append("\n### Leave-One-Out\n")
    loo = group_structure.get("leave_one_out", [])
    header = ["Latent removed", "Full effect", "LOO effect", "Individual contribution"]
    rows = [[r["latent_idx"],
             f"{r.get('full_effect', float('nan')):+.3f}",
             f"{r['loo_effect']:+.3f}",
             f"{r['delta_loo']:+.3f}"]
            for r in loo]
    lines.append(_render_table(header, rows))

    lines.append("\n### Add-One-In\n")
    add_one = group_structure.get("add_one_in", [])
    header = ["K", "Latent added", "Effect", "Marginal gain"]
    rows = [[r["k"], r["latent_idx"], f"{r['effect']:+.3f}", f"{r['delta_add']:+.3f}"]
            for r in add_one]
    lines.append(_render_table(header, rows))

    lines.append("\n### Synergy\n")
    syn = group_structure.get("synergy", {})
    lines.append(f"- Full group effect: **{syn.get('full_effect', 0):+.3f}**\n")
    lines.append(f"- Sum of individual effects: **{syn.get('sum_individual_effects', 0):+.3f}**\n")
    lines.append(f"- Synergy score: **{syn.get('synergy_score', 0):+.3f}** — _{syn.get('interpretation', '')}_\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SAE-RE Causal Validation")
    p.add_argument("--candidate-csv", default="outputs/sae_eval/candidate_latents.csv")
    p.add_argument("--data-dir", default="data/cactus")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory for causal outputs. Falls back to OUTPUT_ROOT/causal_validation if set.",
    )
    p.add_argument("--sae-config", default="config/sae_config.json")
    p.add_argument("--model-config", default=None)
    p.add_argument(
        "--model-dir",
        default=None,
        help="Override the local model directory. Takes precedence over MODEL_DIR and model_config.json.",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--device", default=None)
    p.add_argument("--lambdas", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    p.add_argument("--skip-group-structure", action="store_true")
    p.add_argument("--skip-side-effects", action="store_true")
    p.add_argument("--side-effect-max-samples", type=int, default=16)
    p.add_argument("--side-effect-max-new-tokens", type=int, default=24)
    p.add_argument("--side-effect-lambda", type=float, default=1.0)
    p.add_argument("--n-bootstrap", type=int, default=10,
                   help="Bootstrap seeds for latent stability (0 to skip).")
    return p.parse_args()


def main():
    args = parse_args()
    start = time.time()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ──
    with open(Path(args.sae_config), "r") as f:
        sae_config = json.load(f)
    hook_point = sae_config["hook_point"]

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # ── Load model + SAE ──
    print("[1/8] Loading model...")
    model, tokenizer, _ = load_local_model_and_tokenizer(args.model_config)

    print("[2/8] Loading SAE...")
    sae = load_sae_from_hub(
        repo_id=sae_config["sae_repo_id"],
        subfolder=sae_config["sae_subfolder"],
        device=device, dtype=torch.bfloat16,
    )

    # ── Load dataset ──
    print("[3/8] Loading dataset...")
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    texts, labels, records = build_dataset(data_dir)
    true_labels = np.array(labels)
    n_re = sum(labels)

    # ── Extract utterance features ──
    print("[4/8] Extracting utterance features for probe training...")
    all_feats = extract_utterance_features(
        model, tokenizer, sae, texts, hook_point, device,
        batch_size=args.batch_size, max_seq_len=args.max_seq_len,
    )
    re_feats    = all_feats[:n_re]
    nonre_feats = all_feats[n_re:]

    # ── Latent selection ──
    print("[5/8] Ranking latents (G1/G5/G20)...")
    candidate_df = pd.read_csv(Path(args.candidate_csv))
    ranking = rank_latents(candidate_df, re_feats, nonre_feats, top_k=20)
    G1, G5, G20 = ranking["G1"], ranking["G5"], ranking["G20"]
    ranked_df   = ranking["ranked_df"]
    print(f"  G1={G1}  G5={G5}")
    print(f"  G20={G20}")

    # Bootstrap stability (optional)
    stability = {}
    if args.n_bootstrap > 0:
        print(f"  Bootstrap stability ({args.n_bootstrap} seeds)...")
        stability = bootstrap_stability(
            re_feats, nonre_feats, candidate_df, n_seeds=args.n_bootstrap
        )
        print(f"  Stable G5:  {stability['stable_G5']}")
        print(f"  Stable G20: {stability['stable_G20']}")

    groups = {"G1": G1, "G5": G5, "G20": G20}
    if stability:
        ranked_latents_full = ranked_df["latent_idx"].astype(int).tolist()
        if stability.get("stable_G5"):
            groups["G5"] = _stabilize_group(ranked_latents_full, stability["stable_G5"], 5)
        if stability.get("stable_G20"):
            groups["G20"] = _stabilize_group(ranked_latents_full, stability["stable_G20"], 20)
        if groups["G1"] and groups["G1"][0] not in groups["G5"]:
            groups["G5"] = _stabilize_group(ranked_latents_full, groups["G1"] + groups["G5"], 5)
        if groups["G1"] and groups["G1"][0] not in groups["G20"]:
            groups["G20"] = _stabilize_group(ranked_latents_full, groups["G1"] + groups["G20"], 20)
        G1, G5, G20 = groups["G1"], groups["G5"], groups["G20"]
        print(f"  Using stabilized groups -> G5={G5}")
        print(f"  Using stabilized groups -> G20={G20}")

    # ── Train RE probe ──
    print("[6/8] Training RE probe...")
    probe = REProbeScorer.fit(re_feats, nonre_feats, candidate_indices=G20)
    baseline_eval = probe.evaluate(all_feats, true_labels)
    print(f"  Probe baseline: acc={baseline_eval['accuracy']:.3f}, "
          f"auc={baseline_eval['auc']:.3f}")

    # ── Build controls ──
    W_dec = sae.W_dec.detach().cpu()
    g20_weights = _normalise_probe_weights(probe, G20)
    steering_dir = make_steering_direction(W_dec, G20, g20_weights)

    orth_dir   = make_orthogonal_direction(steering_dir)
    random_dir = make_random_direction(steering_dir.shape[0], steering_dir.dtype, steering_dir.device)
    # Scale controls to same magnitude as steering direction
    orth_dir   = orth_dir / (orth_dir.norm() + 1e-8) * steering_dir.norm()
    random_dir = random_dir / (random_dir.norm() + 1e-8) * steering_dir.norm()

    bottom_k_ids = make_bottom_k(ranked_df, k=20)
    random_ids   = make_random_control(
        candidate_df,
        k=20,
        reference_latents=G20,
        all_features=all_feats,
        seed=42,
    )

    necessity_controls = {
        "Bottom20": bottom_k_ids,
        "Random20": random_ids,
    }
    sufficiency_controls = {
        "Orthogonal": orth_dir,
        "Random_dir": random_dir,
    }

    # ── Build batches for intervention ──
    batches = iter_batches(
        texts, labels, records, tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        device=device,
    )

    # ── Compute reference mean for mean-ablation ──
    runner = CausalRunner(model, tokenizer, sae, hook_point, device, sae_config)
    ref_mean = compute_reference_latent_mean(runner, batches)
    runner.set_ref_mean(ref_mean)

    # ── Necessity ──
    print("[7/8] Running necessity experiments...")
    necessity_results = run_necessity_experiment(
        runner, batches, probe, groups, necessity_controls, true_labels,
        model, tokenizer, sae, hook_point, device,
        batch_size=args.batch_size, max_seq_len=args.max_seq_len,
    )

    # ── Sufficiency ──
    print("[7b/8] Running sufficiency experiments...")
    probe_weights = {
        "G1":  _normalise_probe_weights(probe, G1),
        "G5":  _normalise_probe_weights(probe, G5),
        "G20": g20_weights,
    }
    sufficiency_results = run_sufficiency_experiment(
        runner, batches, probe, groups, probe_weights, sufficiency_controls,
        true_labels, model, tokenizer, sae, hook_point, device,
        lambdas=args.lambdas,
        batch_size=args.batch_size, max_seq_len=args.max_seq_len,
    )

    side_effect_results = {}
    if not args.skip_side_effects:
        print("[7c/8] Running selectivity / side-effect evaluation...")
        side_effect_results = run_side_effect_evaluation(
            runner, probe, groups, probe_weights, sufficiency_controls,
            texts, labels, tokenizer, sae, hook_point, device,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            max_samples=args.side_effect_max_samples,
            max_new_tokens=args.side_effect_max_new_tokens,
            lambda_value=args.side_effect_lambda,
        )

    # ── Group structure ──
    group_results = {}
    if not args.skip_group_structure:
        print("[7d/8] Running group structure analysis...")
        ranked_latents = ranked_df["latent_idx"].tolist()
        all_weights = _normalise_probe_weights(probe, ranked_latents[:20])
        group_results = run_group_structure_experiment(
            runner, batches, probe, ranked_latents[:20], all_weights,
            true_labels, model, tokenizer, sae, hook_point, device,
            strength=1.0,
            batch_size=args.batch_size, max_seq_len=args.max_seq_len,
        )

    # ── Save results ──
    print("[8/8] Saving results...")
    def _save(obj, name):
        p = output_dir / name
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        print(f"  Saved {p}")

    _save({
        "G1": G1, "G5": G5, "G20": G20,
        "stability": stability,
        "ranked_latents": ranked_df["latent_idx"].tolist(),
        "probe_baseline": baseline_eval,
    }, "selected_groups.json")

    _save(necessity_results,  "results_necessity.json")
    _save(sufficiency_results, "results_sufficiency.json")
    _save(side_effect_results, "results_selectivity.json")
    _save(group_results,       "results_group.json")

    generate_summary_tables(
        necessity_results, sufficiency_results, group_results, side_effect_results,
        output_dir / "summary_tables.md",
    )

    elapsed = time.time() - start
    print(f"\n  DONE — {elapsed:.1f}s  |  Output: {output_dir}")


POOLING_COMPARE_METHODS = ["max", "sum", "binarized_sum"]


def _save_json(path: Path, obj: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved {path}")


def _nearest_lambda_key(mode_payload: dict[str, Any], target_lambda: float = 1.0) -> str | None:
    candidates: list[tuple[float, str]] = []
    for key, value in mode_payload.items():
        if not key.startswith("lam_") or not isinstance(value, dict):
            continue
        try:
            lam = float(key.replace("lam_", "", 1))
        except ValueError:
            continue
        candidates.append((abs(lam - target_lambda), key))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def build_pooling_comparison_summary(
    run_payloads: dict[str, dict[str, Any]],
    target_lambda: float = 1.0,
) -> dict[str, Any]:
    summary_runs: dict[str, Any] = {}

    for pooling_method, payload in run_payloads.items():
        selected = payload["selected_groups"]
        necessity = payload["necessity"]
        sufficiency = payload["sufficiency"]
        side_effects = payload["side_effects"]
        group_results = payload["group"]

        lambda_key = None
        for group_name in ["G1", "G5", "G20"]:
            lambda_key = _nearest_lambda_key(
                sufficiency.get(group_name, {}).get("cond_token", {}),
                target_lambda=target_lambda,
            )
            if lambda_key is not None:
                break

        necessity_groups = {}
        for group_name in ["G1", "G5", "G20"]:
            group_payload = necessity.get(group_name, {})
            necessity_groups[group_name] = {
                "zero": group_payload.get("zero", {}),
                "cond_token": group_payload.get("cond_token", {}),
            }

        sufficiency_groups = {}
        for group_name in ["G1", "G5", "G20"]:
            group_payload = sufficiency.get(group_name, {}).get("cond_token", {})
            sufficiency_groups[group_name] = {
                "lambda_key": lambda_key,
                "delta": group_payload.get(lambda_key, {}) if lambda_key else {},
            }

        selectivity_groups = {}
        for group_name in ["G1", "G5", "G20"]:
            group_payload = side_effects.get("groups", {}).get(group_name, {})
            selectivity_groups[group_name] = {
                "mean_generated_re_logit_delta": group_payload.get("mean_generated_re_logit_delta"),
                "quality": {
                    "mean_content_retention": group_payload.get("quality", {}).get("mean_content_retention"),
                    "delta_bigram_repetition": group_payload.get("quality", {}).get("delta_bigram_repetition"),
                },
            }

        summary_runs[pooling_method] = {
            "pooling_method": pooling_method,
            "binarized_threshold": payload["binarized_threshold"],
            "probe_baseline": {
                "accuracy": payload["probe_baseline"].get("accuracy"),
                "auc": payload["probe_baseline"].get("auc"),
            },
            "selected_groups": {
                "G1": selected.get("G1", []),
                "G5": selected.get("G5", []),
                "G20": selected.get("G20", []),
            },
            "necessity": necessity_groups,
            "sufficiency": sufficiency_groups,
            "selectivity": {"groups": selectivity_groups},
            "group": {
                "synergy": {
                    "synergy_score": group_results.get("synergy", {}).get("synergy_score"),
                }
            },
        }

    def _necessity_strength(item: dict[str, Any]) -> float:
        vals = []
        for group_name in ["G1", "G5", "G20"]:
            val = item["necessity"][group_name]["cond_token"].get("mean_delta_re")
            if isinstance(val, (int, float)):
                vals.append(abs(float(val)))
        return float(np.mean(vals)) if vals else float("-inf")

    def _sufficiency_strength(item: dict[str, Any]) -> float:
        vals = []
        for group_name in ["G1", "G5", "G20"]:
            val = item["sufficiency"][group_name]["delta"].get("mean_delta_re")
            if isinstance(val, (int, float)):
                vals.append(float(val))
        return float(np.mean(vals)) if vals else float("-inf")

    def _side_effect_score(item: dict[str, Any]) -> tuple[float, float]:
        retentions = []
        repeat_shifts = []
        for group_name in ["G1", "G5", "G20"]:
            quality = item["selectivity"]["groups"][group_name]["quality"]
            retention = quality.get("mean_content_retention")
            repeat_shift = quality.get("delta_bigram_repetition")
            if isinstance(retention, (int, float)):
                retentions.append(float(retention))
            if isinstance(repeat_shift, (int, float)):
                repeat_shifts.append(abs(float(repeat_shift)))
        retention_score = float(np.mean(retentions)) if retentions else float("-inf")
        repeat_penalty = float(np.mean(repeat_shifts)) if repeat_shifts else float("inf")
        return retention_score, repeat_penalty

    best_probe = None
    best_necessity = None
    best_sufficiency = None
    best_side_effect = None

    if summary_runs:
        best_probe = max(
            summary_runs,
            key=lambda name: summary_runs[name]["probe_baseline"].get("auc", float("-inf")),
        )
        best_necessity = max(summary_runs, key=lambda name: _necessity_strength(summary_runs[name]))
        best_sufficiency = max(summary_runs, key=lambda name: _sufficiency_strength(summary_runs[name]))

        side_candidates = [
            name for name, item in summary_runs.items()
            if _side_effect_score(item)[0] != float("-inf")
        ]
        if side_candidates:
            best_side_effect = max(
                side_candidates,
                key=lambda name: (_side_effect_score(summary_runs[name])[0], -_side_effect_score(summary_runs[name])[1]),
            )

    return {
        "pooling_order": list(run_payloads.keys()),
        "target_lambda": target_lambda,
        "runs": summary_runs,
        "best_probe_pooling": best_probe,
        "best_necessity_pooling": best_necessity,
        "best_sufficiency_pooling": best_sufficiency,
        "lightest_side_effect_pooling": best_side_effect,
    }


def generate_pooling_comparison_report(
    comparison: dict[str, Any],
    output_path: Path,
) -> None:
    lines = ["# Pooling Comparison", ""]
    lines.append("## Overview")
    lines.append("")
    lines.append("| Pooling | Probe AUC | Necessity (G20 cond-token) | Sufficiency (G20 cond-token) | Retention | Synergy |")
    lines.append("| --- | --- | --- | --- | --- | --- |")

    for pooling_method in comparison.get("pooling_order", []):
        run = comparison["runs"][pooling_method]
        necessity_g20 = run["necessity"]["G20"]["cond_token"].get("mean_delta_re")
        sufficiency_g20 = run["sufficiency"]["G20"]["delta"].get("mean_delta_re")
        retention_g20 = run["selectivity"]["groups"]["G20"]["quality"].get("mean_content_retention")
        synergy = run["group"]["synergy"].get("synergy_score")
        lines.append(
            "| "
            + " | ".join([
                pooling_method,
                f"{run['probe_baseline'].get('auc', float('nan')):.3f}",
                f"{float(necessity_g20):+.3f}" if isinstance(necessity_g20, (int, float)) else "n/a",
                f"{float(sufficiency_g20):+.3f}" if isinstance(sufficiency_g20, (int, float)) else "n/a",
                f"{float(retention_g20):.3f}" if isinstance(retention_g20, (int, float)) else "n/a",
                f"{float(synergy):+.3f}" if isinstance(synergy, (int, float)) else "n/a",
            ])
            + " |"
        )

    lines.extend([
        "",
        "## Key Answers",
        "",
        f"- 哪种 pooling 的 probe 判别最好：`{comparison.get('best_probe_pooling') or 'n/a'}`。",
        f"- 哪种 pooling 的 necessity / sufficiency 信号最强：`{comparison.get('best_necessity_pooling') or 'n/a'}` / `{comparison.get('best_sufficiency_pooling') or 'n/a'}`。",
        f"- 哪种 pooling 的 side-effect 最轻：`{comparison.get('lightest_side_effect_pooling') or 'n/a (side effects skipped)'}`。",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved {output_path}")


def _run_single_pooling_experiment(
    model,
    tokenizer,
    sae,
    candidate_df: pd.DataFrame,
    texts: list[str],
    labels: list[int],
    records: list[dict[str, Any]],
    hook_point: str,
    device: torch.device,
    sae_config: dict[str, Any],
    args,
    output_dir: Path,
    pooling_method: str,
    binarized_threshold: float,
) -> dict[str, Any]:
    true_labels = np.array(labels)
    n_re = sum(labels)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Pooling: {pooling_method} ===")
    print("[4/8] Extracting utterance features for probe training...")
    all_feats = extract_utterance_features(
        model, tokenizer, sae, texts, hook_point, device,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        aggregation=pooling_method,
        binarized_threshold=binarized_threshold,
    )
    re_feats = all_feats[:n_re]
    nonre_feats = all_feats[n_re:]

    print("[5/8] Ranking latents (G1/G5/G20)...")
    ranking = rank_latents(candidate_df, re_feats, nonre_feats, top_k=20)
    G1, G5, G20 = ranking["G1"], ranking["G5"], ranking["G20"]
    ranked_df = ranking["ranked_df"]
    print(f"  G1={G1}  G5={G5}")
    print(f"  G20={G20}")

    stability = {}
    if args.n_bootstrap > 0:
        print(f"  Bootstrap stability ({args.n_bootstrap} seeds)...")
        stability = bootstrap_stability(
            re_feats, nonre_feats, candidate_df, n_seeds=args.n_bootstrap
        )
        print(f"  Stable G5:  {stability['stable_G5']}")
        print(f"  Stable G20: {stability['stable_G20']}")

    groups = {"G1": G1, "G5": G5, "G20": G20}
    if stability:
        ranked_latents_full = ranked_df["latent_idx"].astype(int).tolist()
        if stability.get("stable_G5"):
            groups["G5"] = _stabilize_group(ranked_latents_full, stability["stable_G5"], 5)
        if stability.get("stable_G20"):
            groups["G20"] = _stabilize_group(ranked_latents_full, stability["stable_G20"], 20)
        if groups["G1"] and groups["G1"][0] not in groups["G5"]:
            groups["G5"] = _stabilize_group(ranked_latents_full, groups["G1"] + groups["G5"], 5)
        if groups["G1"] and groups["G1"][0] not in groups["G20"]:
            groups["G20"] = _stabilize_group(ranked_latents_full, groups["G1"] + groups["G20"], 20)
        G1, G5, G20 = groups["G1"], groups["G5"], groups["G20"]
        print(f"  Using stabilized groups -> G5={G5}")
        print(f"  Using stabilized groups -> G20={G20}")

    print("[6/8] Training RE probe...")
    probe = REProbeScorer.fit(re_feats, nonre_feats, candidate_indices=G20)
    baseline_eval = probe.evaluate(all_feats, true_labels)
    print(f"  Probe baseline: acc={baseline_eval['accuracy']:.3f}, auc={baseline_eval['auc']:.3f}")

    W_dec = sae.W_dec.detach().cpu()
    g20_weights = _normalise_probe_weights(probe, G20)
    steering_dir = make_steering_direction(W_dec, G20, g20_weights)
    orth_dir = make_orthogonal_direction(steering_dir)
    random_dir = make_random_direction(steering_dir.shape[0], steering_dir.dtype, steering_dir.device)
    orth_dir = orth_dir / (orth_dir.norm() + 1e-8) * steering_dir.norm()
    random_dir = random_dir / (random_dir.norm() + 1e-8) * steering_dir.norm()

    bottom_k_ids = make_bottom_k(ranked_df, k=20)
    random_ids = make_random_control(
        candidate_df,
        k=20,
        reference_latents=G20,
        all_features=all_feats,
        seed=42,
    )

    necessity_controls = {
        "Bottom20": bottom_k_ids,
        "Random20": random_ids,
    }
    sufficiency_controls = {
        "Orthogonal": orth_dir,
        "Random_dir": random_dir,
    }

    batches = iter_batches(
        texts, labels, records, tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        device=device,
    )

    runner = CausalRunner(model, tokenizer, sae, hook_point, device, sae_config)
    ref_mean = compute_reference_latent_mean(runner, batches)
    runner.set_ref_mean(ref_mean)

    print("[7/8] Running necessity experiments...")
    necessity_results = run_necessity_experiment(
        runner, batches, probe, groups, necessity_controls, true_labels,
        model, tokenizer, sae, hook_point, device,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        aggregation=pooling_method,
        binarized_threshold=binarized_threshold,
    )

    print("[7b/8] Running sufficiency experiments...")
    probe_weights = {
        "G1": _normalise_probe_weights(probe, G1),
        "G5": _normalise_probe_weights(probe, G5),
        "G20": g20_weights,
    }
    sufficiency_results = run_sufficiency_experiment(
        runner, batches, probe, groups, probe_weights, sufficiency_controls,
        true_labels, model, tokenizer, sae, hook_point, device,
        lambdas=args.lambdas,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        aggregation=pooling_method,
        binarized_threshold=binarized_threshold,
    )

    side_effect_results = {}
    if not args.skip_side_effects:
        print("[7c/8] Running selectivity / side-effect evaluation...")
        side_effect_results = run_side_effect_evaluation(
            runner, probe, groups, probe_weights, sufficiency_controls,
            texts, labels, tokenizer, sae, hook_point, device,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            max_samples=args.side_effect_max_samples,
            max_new_tokens=args.side_effect_max_new_tokens,
            lambda_value=args.side_effect_lambda,
            aggregation=pooling_method,
            binarized_threshold=binarized_threshold,
        )

    group_results = {}
    if not args.skip_group_structure:
        print("[7d/8] Running group structure analysis...")
        ranked_latents = ranked_df["latent_idx"].tolist()
        all_weights = _normalise_probe_weights(probe, ranked_latents[:20])
        group_results = run_group_structure_experiment(
            runner, batches, probe, ranked_latents[:20], all_weights,
            true_labels, model, tokenizer, sae, hook_point, device,
            strength=1.0,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            aggregation=pooling_method,
            binarized_threshold=binarized_threshold,
        )

    print("[8/8] Saving results...")
    selected_payload = {
        "pooling_method": pooling_method,
        "binarized_threshold": binarized_threshold,
        "G1": G1,
        "G5": G5,
        "G20": G20,
        "stability": stability,
        "ranked_latents": ranked_df["latent_idx"].tolist(),
        "probe_baseline": baseline_eval,
    }
    _save_json(output_dir / "selected_groups.json", selected_payload)
    _save_json(output_dir / "results_necessity.json", {
        "pooling_method": pooling_method,
        "binarized_threshold": binarized_threshold,
        **necessity_results,
    })
    _save_json(output_dir / "results_sufficiency.json", {
        "pooling_method": pooling_method,
        "binarized_threshold": binarized_threshold,
        **sufficiency_results,
    })
    _save_json(output_dir / "results_selectivity.json", {
        "pooling_method": pooling_method,
        "binarized_threshold": binarized_threshold,
        **side_effect_results,
    })
    _save_json(output_dir / "results_group.json", {
        "pooling_method": pooling_method,
        "binarized_threshold": binarized_threshold,
        **group_results,
    })

    generate_summary_tables(
        necessity_results,
        sufficiency_results,
        group_results,
        side_effect_results,
        output_dir / "summary_tables.md",
    )

    return {
        "pooling_method": pooling_method,
        "binarized_threshold": binarized_threshold,
        "selected_groups": selected_payload,
        "probe_baseline": baseline_eval,
        "necessity": necessity_results,
        "sufficiency": sufficiency_results,
        "side_effects": side_effect_results,
        "group": group_results,
    }


def parse_args():
    p = argparse.ArgumentParser(description="SAE-RE Causal Validation")
    p.add_argument("--candidate-csv", default="outputs/sae_eval/candidate_latents.csv")
    p.add_argument("--data-dir", default="data/cactus")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory for causal outputs. Falls back to OUTPUT_ROOT/causal_validation if set.",
    )
    p.add_argument("--sae-config", default="config/sae_config.json")
    p.add_argument("--model-config", default=None)
    p.add_argument(
        "--model-dir",
        default=None,
        help="Override the local model directory. Takes precedence over MODEL_DIR and model_config.json.",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--device", default=None)
    p.add_argument("--lambdas", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    p.add_argument("--sentence-pooling", choices=POOLING_COMPARE_METHODS, default="max")
    p.add_argument("--compare-pooling", action="store_true")
    p.add_argument("--binarized-threshold", type=float, default=0.0)
    p.add_argument("--skip-group-structure", action="store_true")
    p.add_argument("--skip-side-effects", action="store_true")
    p.add_argument("--side-effect-max-samples", type=int, default=16)
    p.add_argument("--side-effect-max-new-tokens", type=int, default=24)
    p.add_argument("--side-effect-lambda", type=float, default=1.0)
    p.add_argument("--n-bootstrap", type=int, default=10,
                   help="Bootstrap seeds for latent stability (0 to skip).")
    return p.parse_args()


def main():
    args = parse_args()
    start = time.time()

    output_dir = resolve_output_dir(args.output_dir, default_subdir="causal_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    sae_config_path = resolve_repo_path(args.sae_config)
    with open(sae_config_path, "r", encoding="utf-8") as f:
        sae_config = json.load(f)
    hook_point = sae_config["hook_point"]

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    print("[1/8] Loading model...")
    model, tokenizer, _ = load_local_model_and_tokenizer(
        args.model_config,
        model_dir=args.model_dir,
        device=device,
    )

    print("[2/8] Loading SAE...")
    sae = load_sae_from_hub(
        repo_id=sae_config["sae_repo_id"],
        subfolder=sae_config["sae_subfolder"],
        device=device, dtype=torch.bfloat16,
    )

    print("[3/8] Loading dataset...")
    data_dir = resolve_repo_path(args.data_dir)
    texts, labels, records = build_dataset(data_dir)
    candidate_df = pd.read_csv(resolve_repo_path(args.candidate_csv))

    if args.compare_pooling:
        run_payloads: dict[str, dict[str, Any]] = {}
        for pooling_method in POOLING_COMPARE_METHODS:
            child_output_dir = output_dir / f"pooling_{pooling_method}"
            run_payloads[pooling_method] = _run_single_pooling_experiment(
                model,
                tokenizer,
                sae,
                candidate_df,
                texts,
                labels,
                records,
                hook_point,
                device,
                sae_config,
                args,
                child_output_dir,
                pooling_method,
                args.binarized_threshold,
            )

        comparison = build_pooling_comparison_summary(
            run_payloads,
            target_lambda=1.0,
        )
        _save_json(output_dir / "pooling_comparison.json", comparison)
        generate_pooling_comparison_report(
            comparison,
            output_dir / "pooling_comparison.md",
        )
    else:
        _run_single_pooling_experiment(
            model,
            tokenizer,
            sae,
            candidate_df,
            texts,
            labels,
            records,
            hook_point,
            device,
            sae_config,
            args,
            output_dir,
            args.sentence_pooling,
            args.binarized_threshold,
        )

    elapsed = time.time() - start
    print(f"\n  DONE - {elapsed:.1f}s  |  Output: {output_dir}")


if __name__ == "__main__":
    main()
