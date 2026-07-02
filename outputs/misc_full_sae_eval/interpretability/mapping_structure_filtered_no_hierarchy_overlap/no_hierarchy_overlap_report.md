# Filtered Top20 Overlap Analysis Excluding Hierarchical Parent-Child Pairs

## Scope

- Input: `outputs\misc_full_sae_eval\interpretability\mapping_structure_filtered\topk_candidate_matrix.csv`
- Analysis TopK: 20
- Excluded parent-child pairs: `RE-RES`, `RE-REC`, `QU-QUO`, `QU-QUC`.
- `cross_family_only` additionally excludes same-family sibling pairs: `RES-REC`, `QUO-QUC`.

## Summary

| Metric | Value |
|---|---:|
| all label pairs | 36 |
| parent-child pairs excluded | 4 |
| no-parent-child pairs | 32 |
| no-parent-child pairs with overlap | 10 |
| no-parent-child overlap latent rows | 35 |
| cross-family pairs | 30 |
| cross-family pairs with overlap | 8 |
| cross-family overlap latent rows | 31 |

## Top no-parent-child overlap pairs

| label_a | label_b | intersection | jaccard | overlap_latents |
|---|---|---:|---:|---|
| RES | QU | 6 | 0.176 | `664;9959;10916;11660;13430;27061` |
| RES | QUO | 6 | 0.176 | `664;9959;10916;11660;13430;27061` |
| QU | GI | 5 | 0.143 | `664;9959;11660;13430;26485` |
| QUO | GI | 5 | 0.143 | `664;9959;11660;13430;26485` |
| RES | GI | 4 | 0.111 | `664;9959;11660;13430` |
| QUC | GI | 2 | 0.053 | `11660;13430` |
| QUO | QUC | 2 | 0.053 | `11660;13430` |
| RES | QUC | 2 | 0.053 | `11660;13430` |
| RES | REC | 2 | 0.053 | `19435;29759` |
| RE | SU | 1 | 0.026 | `2995` |

## Cross-family overlap pairs

| label_a | label_b | intersection | jaccard | overlap_latents |
|---|---|---:|---:|---|
| RES | QU | 6 | 0.176 | `664;9959;10916;11660;13430;27061` |
| RES | QUO | 6 | 0.176 | `664;9959;10916;11660;13430;27061` |
| QU | GI | 5 | 0.143 | `664;9959;11660;13430;26485` |
| QUO | GI | 5 | 0.143 | `664;9959;11660;13430;26485` |
| RES | GI | 4 | 0.111 | `664;9959;11660;13430` |
| QUC | GI | 2 | 0.053 | `11660;13430` |
| RES | QUC | 2 | 0.053 | `11660;13430` |
| RE | SU | 1 | 0.026 | `2995` |

## Output files

- `outputs\misc_full_sae_eval\interpretability\mapping_structure_filtered_no_hierarchy_overlap\label_pair_similarity_with_hierarchy_flags.csv`
- `outputs\misc_full_sae_eval\interpretability\mapping_structure_filtered_no_hierarchy_overlap\label_pair_similarity_no_parent_child.csv`
- `outputs\misc_full_sae_eval\interpretability\mapping_structure_filtered_no_hierarchy_overlap\label_pair_similarity_cross_family_only.csv`
- `outputs\misc_full_sae_eval\interpretability\mapping_structure_filtered_no_hierarchy_overlap\overlap_latents_no_parent_child.csv`
- `outputs\misc_full_sae_eval\interpretability\mapping_structure_filtered_no_hierarchy_overlap\overlap_latents_cross_family_only.csv`
- `outputs\misc_full_sae_eval\interpretability\mapping_structure_filtered_no_hierarchy_overlap\no_hierarchy_overlap_summary.json`
