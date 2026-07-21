# Stable Top-K latent set selection

- Ranking metric: `cohens_d`
- Max K: `200`
- Stable inclusion threshold: `0.70`
- Boundary inclusion threshold: `0.40`
- Cross-quality gate enabled: `False`
- Label-level stability gate enabled: `False`

## K selection by label

| label | K_auc | K_stab | K* | status | best AUC | stable core | boundary |
|---|---:|---:|---:|---|---:|---:|---:|
| AF | 58 | 58 | 58 | stable_topk_found | 0.922 | 43 | 14 |
| GI | 85 | - | 85 | performance_only_unstable | 0.764 | 26 | 46 |
| QUC | 28 | 50 | 50 | stable_topk_found | 0.914 | 40 | 10 |
| QUO | 45 | 45 | 45 | stable_topk_found | 0.955 | 42 | 3 |
| REC | 58 | 58 | 58 | stable_topk_found | 0.888 | 46 | 12 |
| RES | 55 | - | 55 | performance_only_unstable | 0.786 | 14 | 20 |
| SU | 30 | - | 30 | performance_only_unstable | 0.864 | 17 | 12 |

## Interpretation rule

- `stable_core`: full-data TopK* member with high repeated split inclusion and positive bootstrap CI.
- `boundary_candidate`: full-data TopK* member with moderate repeated split inclusion and positive bootstrap CI.
- Cross-quality and label-level stability status are retained as audit fields, but they are not hard gates unless the corresponding config flags are enabled.
