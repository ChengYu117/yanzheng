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

from nlp_re_base.activations import extract_and_process_streaming
from nlp_re_base.config import load_model_config
from nlp_re_base.data import load_jsonl
from nlp_re_base.model import load_local_model_and_tokenizer
from nlp_re_base.sae import load_sae_from_hub

from .data import iter_batches, build_dataset
from .evaluation import REProbeScorer, score_delta, eval_text_quality
from .intervention import (
    zero_ablate, mean_ablate, cond_token_ablate, decode_delta,
    constant_steer, cond_token_steer,
    make_steering_direction, make_orthogonal_direction, make_random_direction,
    steer_with_direction,
)
from .selection import rank_latents, bootstrap_stability, make_bottom_k, make_random_control


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
                span_mask=span_mask.to(h.device),
            )

            captured["h_new"] = h_new.detach()
            captured["z"] = z.detach().cpu().float()

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
        ref_mean = self._ref_mean if self._ref_mean is not None else torch.zeros(self.sae.W_dec.shape[0])

        def ablate_fn(h, z, z_hat, span_mask):
            z_new = mean_ablate(z.cpu().float(), span_mask.cpu(), latent_ids, ref_mean)
            delta_z = (z_new - z.cpu().float()).to(h.dtype).to(h.device)
            h_new = h + decode_delta(delta_z.to(h.device), self.sae.W_dec.to(h.device, h.dtype))
            return h_new

        return self._run_forward_with_hook(
            input_ids, attention_mask, span_mask, ablate_fn
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


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helper (reuse streaming pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def extract_utterance_features(
    model, tokenizer, sae, texts, hook_point, device,
    batch_size=4, max_seq_len=128,
) -> np.ndarray:
    """Extract utterance-level SAE features [N, d_sae] for scoring."""
    result = extract_and_process_streaming(
        model=model, tokenizer=tokenizer, sae=sae, texts=texts,
        hook_point=hook_point, max_seq_len=max_seq_len,
        batch_size=batch_size, aggregation="max",
        device=device, collect_structural_samples=0,
    )
    return result["utterance_features"].numpy()


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
) -> dict[str, Any]:
    """Run zero + cond_token ablation for all G groups and controls.

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
    )
    baseline_logits = probe.score_features(baseline_feats)

    for group_name, latent_ids in all_groups.items():
        results[group_name] = {}
        for mode in ["zero", "cond_token"]:
            print(f"  [Necessity] {group_name} / {mode} ablation ({len(latent_ids)} latents)...")
            intervened_feats_list = []

            for batch in batches:
                if mode == "zero":
                    cap = runner.run_zero_ablation(
                        batch.input_ids, batch.attention_mask,
                        batch.counselor_span_mask, latent_ids,
                    )
                else:
                    cap = runner.run_mean_ablation(
                        batch.input_ids, batch.attention_mask,
                        batch.counselor_span_mask, latent_ids,
                    )

                # Re-run SAE on modified z to get intervened features
                z_intervened = cap.get("z", None)
                if z_intervened is not None:
                    # Max-pool over tokens using attention_mask
                    mask = batch.attention_mask.bool()
                    feat = _max_pool_features(z_intervened, mask)
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
) -> dict[str, Any]:
    """Run conditional per-token steering for G groups and controls.

    Returns a structured dict for Table 2 (sufficiency).
    """
    if lambdas is None:
        lambdas = [0.5, 1.0, 1.5, 2.0]

    results: dict[str, Any] = {}

    all_texts = [t for b in batches for t in b.texts]
    print("  [Sufficiency] Computing baseline features...")
    baseline_feats = extract_utterance_features(
        model, tokenizer, sae, all_texts, hook_point, device,
        batch_size=batch_size, max_seq_len=max_seq_len,
    )
    baseline_logits = probe.score_features(baseline_feats)

    # Target groups
    for gname in ["G1", "G5", "G20"]:
        latent_ids = groups[gname]
        weights = probe_weights.get(gname, [1.0 / len(latent_ids)] * len(latent_ids))
        results[gname] = {}

        for lam in lambdas:
            print(f"  [Sufficiency] {gname} cond_token steer λ={lam}...")
            feats_list = []
            for batch in batches:
                cap = runner.run_cond_token_steer(
                    batch.input_ids, batch.attention_mask,
                    batch.counselor_span_mask, latent_ids, weights, lam,
                )
                z_out = cap.get("z", None)
                if z_out is not None:
                    mask = batch.attention_mask.bool()
                    feats_list.append(_max_pool_features(z_out, mask))

            if feats_list:
                feats = np.concatenate(feats_list, axis=0)
                logits = probe.score_features(feats)
                results[gname][f"lam_{lam}"] = score_delta(
                    baseline_logits, logits, true_labels
                )

    # Control directions
    for ctrl_name, direction in controls_directions.items():
        results[ctrl_name] = {}
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
                    mask = batch.attention_mask.bool()
                    feats_list.append(_max_pool_features(z_out, mask))

            if feats_list:
                feats = np.concatenate(feats_list, axis=0)
                logits = probe.score_features(feats)
                results[ctrl_name][f"lam_{lam}"] = score_delta(
                    baseline_logits, logits, true_labels
                )

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
) -> dict[str, Any]:
    """Cumulative top-K curve, Leave-One-Out, Add-One-In, Synergy. (§7)"""
    results: dict[str, Any] = {
        "cumulative_topk": {},
        "leave_one_out": {},
        "synergy": {},
    }

    all_texts = [t for b in batches for t in b.texts]
    baseline_feats = extract_utterance_features(
        model, tokenizer, sae, all_texts, hook_point, device,
        batch_size=batch_size, max_seq_len=max_seq_len,
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
                mask = batch.attention_mask.bool()
                feats_list.append(_max_pool_features(z_out, mask))
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
    header = ["Group", "λ", "Δ-logit RE", "Δ-logit NonRE", "Fraction↑"]
    rows = []
    for gname, lambdas in sufficiency.items():
        for lname, delta in lambdas.items():
            if isinstance(delta, dict) and "mean_delta_re" in delta:
                lam_val = lname.replace("lam_", "")
                rows.append([
                    gname, lam_val,
                    f"{delta['mean_delta_re']:+.3f}",
                    f"{delta['mean_delta_nonre']:+.3f}",
                    f"{delta['fraction_improved']:.2f}",
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

    lines.append("\n### Synergy\n")
    syn = group_structure.get("synergy", {})
    lines.append(f"- Full group effect: **{syn.get('full_effect', 0):+.3f}**\n")
    lines.append(f"- Sum of individual effects: **{syn.get('sum_individual_effects', 0):+.3f}**\n")
    lines.append(f"- Synergy score: **{syn.get('synergy_score', 0):+.3f}** — _{syn.get('interpretation', '')}_\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓ Saved {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SAE-RE Causal Validation")
    p.add_argument("--candidate-csv", default="outputs/sae_eval/candidate_latents.csv")
    p.add_argument("--data-dir", default="data/mi_re")
    p.add_argument("--output-dir", default="outputs/causal_validation")
    p.add_argument("--sae-config", default="config/sae_config.json")
    p.add_argument("--model-config", default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--device", default=None)
    p.add_argument("--lambdas", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    p.add_argument("--skip-group-structure", action="store_true")
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
    texts, labels, _ = build_dataset(
        data_dir / "re_dataset.jsonl",
        data_dir / "nonre_dataset.jsonl",
    )
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
    random_ids   = make_random_control(candidate_df, k=20, seed=42)

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
        texts, labels, tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        device=device,
    )

    # ── Compute reference mean for mean-ablation ──
    ref_mean = torch.from_numpy(all_feats.mean(axis=0))

    runner = CausalRunner(model, tokenizer, sae, hook_point, device, sae_config)
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

    # ── Group structure ──
    group_results = {}
    if not args.skip_group_structure:
        print("[7c/8] Running group structure analysis...")
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
        print(f"  ✓ {p}")

    _save({
        "G1": G1, "G5": G5, "G20": G20,
        "stability": stability,
        "ranked_latents": ranked_df["latent_idx"].tolist(),
        "probe_baseline": baseline_eval,
    }, "selected_groups.json")

    _save(necessity_results,  "results_necessity.json")
    _save(sufficiency_results, "results_sufficiency.json")
    _save(group_results,       "results_group.json")

    generate_summary_tables(
        necessity_results, sufficiency_results, group_results,
        output_dir / "summary_tables.md",
    )

    elapsed = time.time() - start
    print(f"\n  DONE — {elapsed:.1f}s  |  Output: {output_dir}")


if __name__ == "__main__":
    main()
