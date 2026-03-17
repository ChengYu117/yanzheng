# NLP RE Dataset + Model Base

A SAE-RE evaluation pipeline for identifying Reflective Listening (RE) features in Llama-3.1-8B using Sparse Autoencoders.

## What This Does

1. Loads a pre-trained SAE (`Llama3_1-8B-Base-L19R-8x`, JumpReLU, 32768 latents) from HuggingFace
2. Loads a local Llama-3.1-8B base model
3. Feeds RE and NonRE datasets through the model → extracts layer-19 residual activations → runs SAE forward pass
4. Computes **structural metrics** (MSE, Cosine Sim, L₀, CE Loss, KL Divergence, Dead Features)
5. Computes **functional metrics** (Cohen's d + BH-FDR, Sparse Probing, DiffMean, TPP, Feature Absorption, Feature Geometry, MaxAct cards)
6. Identifies candidate RE-associated latents

## Project Layout

```text
NLP_re_dataset_model_base/
  config/
    model_config.json         # Local Llama-3.1-8B path
    sae_config.json           # SAE hyperparams & eval settings
  data/
    mi_re/
      re_dataset.jsonl        # 800 RE utterances
      nonre_dataset.jsonl     # 798 NonRE utterances
  src/
    nlp_re_base/
      config.py               # Config loader
      data.py                 # JSONL loader
      model.py                # Llama model loader
      sae.py                  # SAE model + HuggingFace loading
      activations.py          # Streaming activation extraction + SAE forward
      eval_structural.py      # Structural metrics
      eval_functional.py      # Functional metrics (probing, TPP, absorption, geometry)
      infer.py                # Basic text generation
  run_sae_evaluation.py       # End-to-end pipeline runner
  test_pipeline_smoke.py      # CPU smoke tests
  doc/
    SAE评估指标说明.md        # Evaluation framework reference
```

## Setup

```powershell
conda activate qwen-env
pip install -e .
```

## Run Evaluation

```powershell
# Full evaluation (GPU recommended)
python run_sae_evaluation.py --output-dir outputs/sae_eval --batch-size 4

# Skip slow CE/KL computation
python run_sae_evaluation.py --skip-ce-kl --output-dir outputs/sae_eval

# CPU smoke tests (no model needed)
python test_pipeline_smoke.py
```

## Output Files

| File | Contents |
|------|----------|
| `metrics_structural.json` | MSE, cosine sim, L₀, CE/KL, dead features |
| `metrics_functional.json` | Probe results, univariate stats, absorption, geometry, TPP |
| `candidate_latents.csv` | Ranked latents with Cohen's d, AUC, p-value, FDR |
| `latent_cards/` | Markdown reports for top candidate latents |

## Local Model

Configured in `config/model_config.json`:
- default relative path: `models/Llama-3.1-8B`
- can be overridden by:
  - `--model-dir`
  - `MODEL_DIR`

## Key Design Decisions

- **Streaming pipeline**: Activations are processed batch-by-batch to avoid OOM (no full `[N, T, 32768]` tensor in memory)
- **Strict SAE loading**: Hard-fails if critical weights (W_enc, W_dec, b_enc) are missing from checkpoint
- **dtype alignment**: Activations are cast to SAE dtype (bfloat16) before forward pass

## GCE Deployment

This repo now includes a GCE deployment bundle:

- `python package_project.py`
- `deploy/gce/bootstrap.sh`
- `deploy/gce/download_model.sh`
- `deploy/gce/run_full_eval.sh`
- `deploy/gce/run_causal.sh`

See [GCE云GPU部署说明](doc/GCE云GPU部署说明.md) for the full Google Compute Engine workflow.
