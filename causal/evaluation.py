"""causal/evaluation.py — RE scoring and side-effect evaluation.

Scoring is based on a logistic probe trained on SAE utterance-level features.
Side effects are measured via lightweight lexical metrics (no external LLM needed).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ─────────────────────────────────────────────────────────────────────────────
# RE Probe Scorer
# ─────────────────────────────────────────────────────────────────────────────

class REProbeScorer:
    """Logistic probe over SAE utterance-level features.

    Trained once on the full RE/NonRE feature set, then applied to
    interventional outputs by re-running the SAE on modified residuals.
    """

    def __init__(
        self,
        probe_state: dict[str, Any],   # from _fit_torch_probe
        candidate_indices: list[int],  # probe was trained on these latent columns
    ) -> None:
        self.probe_state = probe_state
        self.candidate_indices = candidate_indices

    @classmethod
    def fit(
        cls,
        re_features: np.ndarray,       # [N_re, d_sae]
        nonre_features: np.ndarray,    # [N_nonre, d_sae]
        candidate_indices: list[int],  # top-k latent indices to use as features
        max_steps: int = 300,
        lr: float = 0.05,
    ) -> "REProbeScorer":
        """Train the probe and return a REProbeScorer instance."""
        # Import from eval_functional to avoid code duplication
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from nlp_re_base.eval_functional import _fit_torch_probe

        labels = np.concatenate([
            np.ones(len(re_features)),
            np.zeros(len(nonre_features)),
        ]).astype(np.float32)
        X = np.concatenate([re_features, nonre_features], axis=0)
        X_sub = np.ascontiguousarray(X[:, candidate_indices], dtype=np.float32)

        probe_state = _fit_torch_probe(X_sub, labels, max_steps=max_steps, learning_rate=lr)
        return cls(probe_state, candidate_indices)

    def score_features(self, features: np.ndarray) -> np.ndarray:
        """Return RE logit scores for utterance-level features.

        Args:
            features: [N, d_sae]
        Returns:
            logits: [N]  (positive = more RE-like)
        """
        from nlp_re_base.eval_functional import _predict_torch_probe

        X_sub = np.ascontiguousarray(
            features[:, self.candidate_indices], dtype=np.float32
        )
        _preds, probs = _predict_torch_probe(self.probe_state, X_sub)
        # Convert probs to logit for a more sensitive measure
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(probs / (1 - probs))
        return logits

    def score_labels(self, features: np.ndarray) -> np.ndarray:
        """Return binary predictions: 1=RE, 0=NonRE."""
        from nlp_re_base.eval_functional import _predict_torch_probe

        X_sub = np.ascontiguousarray(
            features[:, self.candidate_indices], dtype=np.float32
        )
        preds, _ = _predict_torch_probe(self.probe_state, X_sub)
        return preds

    def evaluate(
        self,
        features: np.ndarray,
        true_labels: np.ndarray,
    ) -> dict[str, float]:
        """Full evaluation metrics."""
        from nlp_re_base.eval_functional import _predict_torch_probe

        X_sub = np.ascontiguousarray(
            features[:, self.candidate_indices], dtype=np.float32
        )
        preds, probs = _predict_torch_probe(self.probe_state, X_sub)
        logits = np.log(np.clip(probs, 1e-7, 1 - 1e-7) / np.clip(1 - probs, 1e-7, 1 - 1e-7))
        auc = roc_auc_score(true_labels, probs) if len(np.unique(true_labels)) > 1 else 0.5
        return {
            "accuracy": float(accuracy_score(true_labels, preds)),
            "f1":       float(f1_score(true_labels, preds, zero_division=0)),
            "auc":      float(auc),
            "mean_logit_re":    float(logits[true_labels == 1].mean()) if (true_labels == 1).any() else 0.0,
            "mean_logit_nonre": float(logits[true_labels == 0].mean()) if (true_labels == 0).any() else 0.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Delta scoring: intervention effect measurement
# ─────────────────────────────────────────────────────────────────────────────

def score_delta(
    baseline_logits: np.ndarray,     # [N]
    intervened_logits: np.ndarray,   # [N]
    true_labels: np.ndarray,         # [N]  1=RE, 0=NonRE
) -> dict[str, float]:
    """Compute Δ-logit for RE and NonRE subsets separately."""
    delta = intervened_logits - baseline_logits
    re_mask    = true_labels == 1
    nonre_mask = true_labels == 0

    return {
        "mean_delta":          float(delta.mean()),
        "mean_delta_re":       float(delta[re_mask].mean())    if re_mask.any()    else float("nan"),
        "mean_delta_nonre":    float(delta[nonre_mask].mean()) if nonre_mask.any() else float("nan"),
        "std_delta":           float(delta.std()),
        "fraction_improved":   float((delta > 0).mean()),   # fraction of samples that moved up
    }


# ─────────────────────────────────────────────────────────────────────────────
# Side-effect / quality metrics  §6.2  (lightweight, no LLM judge)
# ─────────────────────────────────────────────────────────────────────────────

def _type_token_ratio(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _bigram_repetition_rate(text: str) -> float:
    words = text.lower().split()
    if len(words) < 2:
        return 0.0
    bigrams = list(zip(words[:-1], words[1:]))
    return float(len(bigrams) - len(set(bigrams))) / float(len(bigrams))


def _content_retention_ratio(source_text: str, compared_text: str) -> float:
    source_tokens = set(source_text.lower().split())
    compared_tokens = set(compared_text.lower().split())
    if not source_tokens:
        return 0.0
    return float(len(source_tokens & compared_tokens)) / float(len(source_tokens))


def eval_text_quality(
    original_texts: list[str],
    intervened_texts: list[str] | None = None,
) -> dict[str, float]:
    """Lightweight quality metrics on original or intervened texts.

    If intervened_texts is None, returns stats for original only.
    """
    ttrs = [_type_token_ratio(t) for t in original_texts]
    lengths = [len(t.split()) for t in original_texts]
    repetitions = [_bigram_repetition_rate(t) for t in original_texts]

    result: dict[str, float] = {
        "mean_ttr":     float(np.mean(ttrs)),
        "mean_length":  float(np.mean(lengths)),
        "mean_bigram_repetition": float(np.mean(repetitions)),
    }

    if intervened_texts is not None:
        ttrs_i   = [_type_token_ratio(t) for t in intervened_texts]
        lengths_i = [len(t.split()) for t in intervened_texts]
        repetitions_i = [_bigram_repetition_rate(t) for t in intervened_texts]
        retention = [
            _content_retention_ratio(src, dst)
            for src, dst in zip(original_texts, intervened_texts)
        ]
        result.update({
            "mean_ttr_intervened":    float(np.mean(ttrs_i)),
            "mean_length_intervened": float(np.mean(lengths_i)),
            "mean_bigram_repetition_intervened": float(np.mean(repetitions_i)),
            "delta_ttr":              float(np.mean(ttrs_i) - np.mean(ttrs)),
            "delta_length":           float(np.mean(lengths_i) - np.mean(lengths)),
            "delta_bigram_repetition": float(np.mean(repetitions_i) - np.mean(repetitions)),
            "mean_content_retention": float(np.mean(retention)),
        })

    return result
