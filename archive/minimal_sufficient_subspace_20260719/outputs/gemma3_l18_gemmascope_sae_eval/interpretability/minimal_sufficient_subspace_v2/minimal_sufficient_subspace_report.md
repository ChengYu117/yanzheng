# MISC minimal sufficient SAE latent subspace v2

This report estimates the smallest predictive SAE latent subset for each MISC label.
It is a probe-space sufficiency analysis and should not be written as causal sufficiency.

## Criteria

- Candidate pool: stable latents plus association-rank Top100 backup.
- Full-candidate recoverable if mean CV AUC >= 0.70.
- Minimal sufficient K must be within 0.02 AUC, 0.03 AUPRC, and 0.05 P@50 lift of the full-candidate probe.
- Parent labels `RE` and `QU` are consistency-only rows.

## Label summary

| Label | Role | Status | Class | Candidate pool | Stable candidates | Full AUC | Minimal K median | Stability Jaccard | Predictive redundancy | Selected latents |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| RE | parent_consistency_only | minimal_sufficient_found | parent_consistency_only | 100 | 1 | 0.906 | 18.0 | 0.309 | 0.444 | 6688,459,9812,691,5184,3511,4015,2929,1455,98,3414,285,602,6218,14796,9734,10426,1573 |
| RES | leaf_or_atomic | minimal_sufficient_found | distributed | 100 | 0 | 0.870 | 18.0 | 0.197 | 0.278 | 459,6688,715,550,1221,98,5184,2919,14295,480,1750,10426,3309,8070,883,2929,226,14 |
| REC | leaf_or_atomic | minimal_sufficient_found | distributed | 100 | 5 | 0.918 | 15.0 | 0.205 | 0.267 | 2929,691,148,3414,55,3363,7170,1517,897,6688,113,9812,244,991,14796 |
| QU | parent_consistency_only | minimal_sufficient_found | parent_consistency_only | 100 | 4 | 0.974 | 9.0 | 0.447 | 0.333 | 98,459,2929,14295,807,7100,5690,289,49 |
| QUO | leaf_or_atomic | minimal_sufficient_found | distributed | 100 | 5 | 0.961 | 14.0 | 0.356 | 0.429 | 2929,459,7677,7100,5184,9503,2520,6607,12571,12721,2646,9347,5624,8000 |
| QUC | leaf_or_atomic | minimal_sufficient_found | distributed | 100 | 1 | 0.919 | 10.0 | 0.358 | 0.100 | 98,14295,12628,12178,459,14624,883,13214,590,4285 |
| GI | leaf_or_atomic | minimal_sufficient_found | distributed | 100 | 0 | 0.857 | 20.0 | 0.197 | 0.350 | 98,1548,883,5184,10860,13146,3932,296,823,9,652,2978,1815,536,2929,298,4963,1745,4213,459 |
| SU | leaf_or_atomic | minimal_sufficient_found | distributed | 100 | 1 | 0.921 | 17.0 | 0.178 | 0.059 | 394,98,4213,5237,10067,10149,12883,2929,3477,8070,2199,15875,1689,6139,10038,916,257 |
| AF | leaf_or_atomic | minimal_sufficient_found | moderate | 100 | 2 | 0.944 | 7.0 | 0.188 | 0.286 | 235,1396,98,2913,3098,459,12546 |

## Interpretation

- Compact labels: none.
- Moderate labels: AF.
- Distributed labels: RES, REC, QUO, QUC, GI, SU.
- Not recoverable under the candidate pool: none.
- Selected subspaces can be used as prioritized groups for later ablation or steering, but this report itself is predictive rather than causal.
