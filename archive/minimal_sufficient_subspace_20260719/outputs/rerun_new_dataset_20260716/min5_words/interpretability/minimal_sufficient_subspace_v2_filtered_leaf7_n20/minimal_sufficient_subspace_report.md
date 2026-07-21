# MISC minimal sufficient SAE latent subspace v2

This report estimates the smallest predictive SAE latent subset for each MISC label.
It is a probe-space sufficiency analysis and should not be written as causal sufficiency.

## Criteria

- Candidate pool: filtered association Top200 only; legacy stable-edge seeds are not used.
- Full-candidate recoverable if mean CV AUC >= 0.70.
- Minimal sufficient K must be within 0.02 AUC, 0.03 AUPRC, and 0.05 P@50 lift of the full-candidate probe.
- Parent labels `RE` and `QU` are consistency-only rows.

## Label summary

| Label | Role | Status | Class | Candidate pool | Filtered TopK candidates | Full AUC | Minimal K median | Stability Jaccard | Predictive redundancy | Selected latents |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| RES | leaf_or_atomic | minimal_sufficient_found | distributed | 200 | 200 | 0.876 | 16.0 | 0.145 | 0.312 | 11660,2462,6743,28269,6623,1137,13430,29759,31585,9959,8104,2342,20028,2068,2013,13303 |
| REC | leaf_or_atomic | minimal_sufficient_found | distributed | 200 | 200 | 0.912 | 15.0 | 0.104 | 0.333 | 11660,20436,9993,29759,26800,19005,11405,9959,9403,664,10181,26319,29590,30091,5569 |
| QUO | leaf_or_atomic | minimal_sufficient_found | distributed | 200 | 200 | 0.952 | 11.0 | 0.296 | 0.273 | 664,21190,18310,13430,14247,8890,15679,19840,22553,14833,6129 |
| QUC | leaf_or_atomic | minimal_sufficient_found | distributed | 200 | 200 | 0.911 | 13.0 | 0.213 | 0.154 | 13430,21935,22358,29947,5413,9959,20402,8226,2386,26485,6100,1622,4998 |
| GI | leaf_or_atomic | minimal_sufficient_found | distributed | 200 | 200 | 0.867 | 13.0 | 0.118 | 0.077 | 13430,664,9893,17827,2055,13751,16515,31844,18490,8166,26879,9728,16131 |
| SU | leaf_or_atomic | minimal_sufficient_found | moderate | 200 | 200 | 0.858 | 4.0 | 0.033 | 0.000 | 2995,24760,2699,21178 |
| AF | leaf_or_atomic | minimal_sufficient_found | moderate | 200 | 200 | 0.888 | 6.0 | 0.179 | 0.000 | 23464,13430,7143,21144,24167,28724 |

## Interpretation

- Compact labels: none.
- Moderate labels: SU, AF.
- Distributed labels: RES, REC, QUO, QUC, GI.
- Not recoverable under the candidate pool: none.
- Selected subspaces can be used as prioritized groups for later ablation or steering, but this report itself is predictive rather than causal.
