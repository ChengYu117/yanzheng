# Split-half TopK Reproducibility

- Group column: `source_file`
- Repeated grouped split-half runs: `50`.
- Spearman threshold for stable ranking: `rho >= 0.60`.
- TopK set stability gate: observed Jaccard >= null mean + 2 * null std.

## Repeated Split Summary

| label | metric | Jaccard mean | Jaccard p05..p95 | pass rate | Spearman mean | Spearman p05..p95 | rho>=0.60 rate | reference-union overlap mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AF | cohens_d | 0.573 | 0.379..0.667 | 1.000 | 0.503 | 0.467..0.530 | 0.000 | 0.967 |
| AF | directional_auc | 0.523 | 0.429..0.600 | 1.000 | 0.518 | 0.505..0.536 | 0.000 | 0.998 |
| GI | cohens_d | 0.192 | 0.095..0.290 | 1.000 | 0.510 | 0.486..0.532 | 0.000 | 0.921 |
| GI | directional_auc | 0.418 | 0.310..0.538 | 1.000 | 0.444 | 0.422..0.465 | 0.000 | 0.973 |
| QU | cohens_d | 0.800 | 0.739..0.818 | 1.000 | 0.657 | 0.641..0.672 | 1.000 | 0.995 |
| QU | directional_auc | 0.799 | 0.739..0.905 | 1.000 | 0.567 | 0.555..0.583 | 0.000 | 1.000 |
| QUC | cohens_d | 0.578 | 0.401..0.739 | 1.000 | 0.485 | 0.461..0.506 | 0.000 | 0.990 |
| QUC | directional_auc | 0.637 | 0.481..0.739 | 1.000 | 0.450 | 0.435..0.462 | 0.000 | 0.996 |
| QUO | cohens_d | 0.831 | 0.739..0.957 | 1.000 | 0.607 | 0.578..0.631 | 0.640 | 0.996 |
| QUO | directional_auc | 0.852 | 0.739..0.905 | 1.000 | 0.531 | 0.514..0.549 | 0.000 | 0.998 |
| RE | cohens_d | 0.490 | 0.333..0.637 | 1.000 | 0.669 | 0.646..0.689 | 1.000 | 0.976 |
| RE | directional_auc | 0.766 | 0.600..0.905 | 1.000 | 0.593 | 0.573..0.611 | 0.320 | 0.998 |
| REC | cohens_d | 0.424 | 0.310..0.538 | 1.000 | 0.689 | 0.664..0.713 | 1.000 | 0.989 |
| REC | directional_auc | 0.536 | 0.429..0.667 | 1.000 | 0.641 | 0.619..0.658 | 1.000 | 1.000 |
| RES | cohens_d | 0.223 | 0.143..0.314 | 1.000 | 0.423 | 0.389..0.455 | 0.000 | 0.911 |
| RES | directional_auc | 0.548 | 0.452..0.600 | 1.000 | 0.448 | 0.430..0.469 | 0.000 | 0.942 |
| SU | cohens_d | 0.184 | 0.026..0.379 | 1.000 | 0.512 | 0.476..0.552 | 0.000 | 0.956 |
| SU | directional_auc | 0.342 | 0.193..0.429 | 1.000 | 0.557 | 0.542..0.575 | 0.000 | 0.988 |

## Representative Split Ranking Stability

The following table keeps the first split (`random_state` seed) for backward-compatible comparison files.

| label | metric | rho | 95% CI | low-n |
|---|---|---:|---:|---|
| AF | cohens_d | 0.508 | 0.495..0.521 | True |
| AF | directional_auc | 0.506 | 0.493..0.518 | True |
| GI | cohens_d | 0.500 | 0.487..0.513 | False |
| GI | directional_auc | 0.439 | 0.425..0.453 | False |
| QU | cohens_d | 0.657 | 0.647..0.667 | False |
| QU | directional_auc | 0.560 | 0.548..0.572 | False |
| QUC | cohens_d | 0.470 | 0.456..0.483 | False |
| QUC | directional_auc | 0.450 | 0.436..0.463 | False |
| QUO | cohens_d | 0.621 | 0.611..0.632 | False |
| QUO | directional_auc | 0.530 | 0.517..0.542 | False |
| RE | cohens_d | 0.661 | 0.651..0.671 | False |
| RE | directional_auc | 0.591 | 0.579..0.602 | False |
| REC | cohens_d | 0.682 | 0.672..0.691 | False |
| REC | directional_auc | 0.638 | 0.627..0.648 | False |
| RES | cohens_d | 0.428 | 0.414..0.442 | False |
| RES | directional_auc | 0.448 | 0.434..0.462 | False |
| SU | cohens_d | 0.512 | 0.499..0.524 | True |
| SU | directional_auc | 0.556 | 0.544..0.568 | True |

## Representative Split TopK Set Stability

| label | metric | observed Jaccard | null mean | null std | pass |
|---|---|---:|---:|---:|---|
| AF | cohens_d | 0.600 | 0.0008 | 0.0045 | True |
| AF | directional_auc | 0.600 | 0.0007 | 0.0044 | True |
| GI | cohens_d | 0.176 | 0.0008 | 0.0045 | True |
| GI | directional_auc | 0.379 | 0.0007 | 0.0044 | True |
| QU | cohens_d | 0.739 | 0.0008 | 0.0045 | True |
| QU | directional_auc | 0.739 | 0.0007 | 0.0044 | True |
| QUC | cohens_d | 0.481 | 0.0008 | 0.0045 | True |
| QUC | directional_auc | 0.538 | 0.0007 | 0.0044 | True |
| QUO | cohens_d | 0.667 | 0.0008 | 0.0045 | True |
| QUO | directional_auc | 0.818 | 0.0007 | 0.0044 | True |
| RE | cohens_d | 0.481 | 0.0008 | 0.0045 | True |
| RE | directional_auc | 0.818 | 0.0007 | 0.0044 | True |
| REC | cohens_d | 0.333 | 0.0008 | 0.0045 | True |
| REC | directional_auc | 0.481 | 0.0007 | 0.0044 | True |
| RES | cohens_d | 0.250 | 0.0008 | 0.0045 | True |
| RES | directional_auc | 0.538 | 0.0007 | 0.0044 | True |
| SU | cohens_d | 0.176 | 0.0008 | 0.0045 | True |
| SU | directional_auc | 0.429 | 0.0007 | 0.0044 | True |
