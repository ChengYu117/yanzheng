# Causal Validation — Summary Tables

## Table 1: Necessity (Ablation)

Mean Δ-logit for RE samples after ablation (negative = RE signal reduced).

| Group | Mode | Δ-logit RE | Δ-logit NonRE | Fraction↑ |
| --- | --- | --- | --- | --- |
| G1 | zero | -0.073 | -0.009 | 0.00 |
| G1 | mean | -0.073 | -0.008 | 0.11 |
| G1 | cond_token | -0.073 | -0.009 | 0.00 |
| G5 | zero | -0.716 | -0.289 | 0.07 |
| G5 | mean | -0.454 | -0.061 | 0.14 |
| G5 | cond_token | -0.716 | -0.289 | 0.07 |
| G20 | zero | -1.814 | +1.123 | 0.35 |
| G20 | mean | -1.469 | +1.352 | 0.41 |
| G20 | cond_token | -1.814 | +1.123 | 0.35 |
| Bottom20 | zero | +0.012 | +0.013 | 0.89 |
| Bottom20 | mean | -0.019 | +0.009 | 0.43 |
| Bottom20 | cond_token | +0.012 | +0.013 | 0.89 |
| Random20 | zero | -0.028 | -0.017 | 0.28 |
| Random20 | mean | -0.030 | -0.010 | 0.45 |
| Random20 | cond_token | -0.028 | -0.017 | 0.28 |


## Table 2: Sufficiency (Steering)

Mean Δ-logit for RE samples after steering (positive = RE signal increased).

| Group | Mode | λ | Δ-logit RE | Δ-logit NonRE | Fraction↑ |
| --- | --- | --- | --- | --- | --- |
| G1 | constant | 0.5 | -0.048 | +0.007 | 0.29 |
| G1 | constant | 1.0 | -0.078 | +0.031 | 0.35 |
| G1 | constant | 1.5 | -0.013 | +0.150 | 0.62 |
| G1 | constant | 2.0 | +0.093 | +0.366 | 0.79 |
| G1 | cond_input | 0.5 | +0.003 | +0.001 | 0.03 |
| G1 | cond_input | 1.0 | +0.005 | +0.002 | 0.03 |
| G1 | cond_input | 1.5 | +0.007 | +0.002 | 0.03 |
| G1 | cond_input | 2.0 | +0.006 | +0.003 | 0.03 |
| G1 | cond_token | 0.5 | +0.007 | +0.001 | 0.04 |
| G1 | cond_token | 1.0 | +0.012 | +0.002 | 0.04 |
| G1 | cond_token | 1.5 | +0.018 | +0.003 | 0.04 |
| G1 | cond_token | 2.0 | +0.023 | +0.004 | 0.04 |
| G5 | constant | 0.5 | +0.029 | +0.075 | 0.85 |
| G5 | constant | 1.0 | +0.057 | +0.150 | 0.86 |
| G5 | constant | 1.5 | +0.083 | +0.242 | 0.85 |
| G5 | constant | 2.0 | +0.105 | +0.307 | 0.86 |
| G5 | cond_input | 0.5 | +0.026 | +0.069 | 0.82 |
| G5 | cond_input | 1.0 | +0.051 | +0.140 | 0.83 |
| G5 | cond_input | 1.5 | +0.076 | +0.230 | 0.82 |
| G5 | cond_input | 2.0 | +0.096 | +0.291 | 0.82 |
| G5 | cond_token | 0.5 | +0.039 | +0.062 | 0.94 |
| G5 | cond_token | 1.0 | +0.071 | +0.122 | 0.94 |
| G5 | cond_token | 1.5 | +0.104 | +0.194 | 0.94 |
| G5 | cond_token | 2.0 | +0.138 | +0.251 | 0.94 |
| G20 | constant | 0.5 | +0.104 | +0.132 | 0.99 |
| G20 | constant | 1.0 | +0.193 | +0.234 | 0.99 |
| G20 | constant | 1.5 | +0.299 | +0.345 | 0.99 |
| G20 | constant | 2.0 | +0.384 | +0.453 | 1.00 |
| G20 | cond_input | 0.5 | +0.104 | +0.132 | 0.99 |
| G20 | cond_input | 1.0 | +0.193 | +0.234 | 0.99 |
| G20 | cond_input | 1.5 | +0.299 | +0.345 | 0.99 |
| G20 | cond_input | 2.0 | +0.384 | +0.453 | 1.00 |
| G20 | cond_token | 0.5 | +0.089 | +0.103 | 0.99 |
| G20 | cond_token | 1.0 | +0.176 | +0.197 | 0.99 |
| G20 | cond_token | 1.5 | +0.269 | +0.293 | 0.99 |
| G20 | cond_token | 2.0 | +0.349 | +0.395 | 1.00 |
| Orthogonal | direction | 0.5 | -0.010 | -0.006 | 0.16 |
| Orthogonal | direction | 1.0 | -0.015 | -0.012 | 0.16 |
| Orthogonal | direction | 1.5 | -0.023 | -0.014 | 0.18 |
| Orthogonal | direction | 2.0 | -0.032 | -0.021 | 0.18 |
| Random_dir | direction | 0.5 | +0.011 | +0.010 | 0.71 |
| Random_dir | direction | 1.0 | +0.021 | +0.027 | 0.86 |
| Random_dir | direction | 1.5 | +0.031 | +0.040 | 0.90 |
| Random_dir | direction | 2.0 | +0.045 | +0.047 | 0.89 |


## Table 3: Selectivity / Side Effects

Generation-time lexical proxy metrics on a small intervention subset.

| Name | Mode | Delta RE logit | Retention | Delta TTR | Delta Repeat |
| --- | --- | --- | --- | --- | --- |
| G1 | cond_token | +0.000 | 1.000 | +0.000 | +0.000 |
| G5 | cond_token | +0.032 | 0.948 | +0.007 | -0.002 |
| G20 | cond_token | +0.000 | 1.000 | +0.000 | +0.000 |
| Orthogonal | direction | -0.071 | 0.958 | -0.011 | +0.011 |
| Random_dir | direction | -0.138 | 0.929 | -0.014 | +0.010 |


## Table 4: Group Structure

### Cumulative Top-K

| K | Latent added | Cumulative Δ-logit RE |
| --- | --- | --- |
| 1 | 6966 | +0.012 |
| 2 | 5551 | +0.015 |
| 3 | 26276 | +0.015 |
| 4 | 15539 | +0.121 |
| 5 | 16969 | +0.849 |
| 6 | 24490 | +0.427 |
| 7 | 22698 | +0.246 |
| 8 | 8104 | +0.695 |
| 9 | 11658 | -0.972 |
| 10 | 20936 | +0.879 |
| 11 | 31133 | +0.442 |
| 12 | 23464 | +2.408 |
| 13 | 12340 | +1.372 |
| 14 | 7462 | +3.233 |
| 15 | 22358 | -1.034 |
| 16 | 3416 | -3.394 |
| 17 | 3222 | +4.720 |
| 18 | 13430 | -1.094 |
| 19 | 27541 | -1.137 |
| 20 | 16320 | -2.501 |

### Leave-One-Out

| Latent removed | Full effect | LOO effect | Individual contribution |
| --- | --- | --- | --- |
| 6966 | +0.879 | +2.014 | -1.135 |
| 5551 | +0.879 | +1.039 | -0.160 |
| 26276 | +0.879 | +1.336 | -0.457 |
| 15539 | +0.879 | +1.333 | -0.454 |
| 16969 | +0.879 | +0.391 | +0.488 |
| 24490 | +0.879 | +1.458 | -0.579 |
| 22698 | +0.879 | +3.274 | -2.395 |
| 8104 | +0.879 | +0.435 | +0.444 |
| 11658 | +0.879 | +0.404 | +0.476 |
| 20936 | +0.879 | -0.972 | +1.851 |

### Add-One-In

| K | Latent added | Effect | Marginal gain |
| --- | --- | --- | --- |
| 1 | 6966 | +0.012 | +0.012 |
| 2 | 5551 | +0.015 | +0.003 |
| 3 | 26276 | +0.015 | -0.001 |
| 4 | 15539 | +0.121 | +0.106 |
| 5 | 16969 | +0.849 | +0.728 |
| 6 | 24490 | +0.427 | -0.421 |
| 7 | 22698 | +0.246 | -0.181 |
| 8 | 8104 | +0.695 | +0.449 |
| 9 | 11658 | -0.972 | -1.667 |
| 10 | 20936 | +0.879 | +1.851 |

### Synergy

- Full group effect: **+0.879**

- Sum of individual effects: **+0.762**

- Synergy score: **+0.117** — _positive (super-additive)_
