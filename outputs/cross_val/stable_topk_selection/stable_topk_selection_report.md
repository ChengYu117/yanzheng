# Stable Top-K latent set selection

- Ranking metric: `cohens_d`
- Max K: `100`
- Stable inclusion threshold: `0.70`
- Boundary inclusion threshold: `0.40`
- Cross-quality gate enabled: `False`
- Label-level stability gate enabled: `False`

## K selection by label

| label | K_auc | K_stab | K* | status | best AUC | stable core | boundary |
|---|---:|---:|---:|---|---:|---:|---:|
| AF | 33 | 33 | 33 | stable_topk_found | 0.925 | 27 | 6 |
| GI | 72 | - | 72 | performance_only_unstable | 0.766 | 26 | 37 |
| QU | 28 | 28 | 28 | stable_topk_found | 0.976 | 26 | 2 |
| QUC | 26 | 59 | 59 | stable_topk_found | 0.926 | 45 | 14 |
| QUO | 36 | 36 | 36 | stable_topk_found | 0.959 | 32 | 4 |
| RE | 66 | 66 | 66 | stable_topk_found | 0.858 | 54 | 12 |
| REC | 56 | 56 | 56 | stable_topk_found | 0.911 | 47 | 9 |
| RES | 45 | - | 45 | performance_only_unstable | 0.794 | 15 | 16 |
| SU | 60 | - | 60 | performance_only_unstable | 0.860 | 31 | 28 |

## Interpretation rule

- `stable_core`: full-data TopK* member with high repeated split inclusion and positive bootstrap CI.
- `boundary_candidate`: full-data TopK* member with moderate repeated split inclusion and positive bootstrap CI.
- Cross-quality and label-level stability status are retained as audit fields, but they are not hard gates unless the corresponding config flags are enabled.
