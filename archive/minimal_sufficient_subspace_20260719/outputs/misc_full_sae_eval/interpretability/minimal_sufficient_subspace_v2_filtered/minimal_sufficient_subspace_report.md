# MISC minimal sufficient SAE latent subspace v2

This report estimates the smallest predictive SAE latent subset for each MISC label.
It is a probe-space sufficiency analysis and should not be written as causal sufficiency.

## Criteria

- Candidate pool: filtered association Top100 only; legacy stable-edge seeds are not used.
- Full-candidate recoverable if mean CV AUC >= 0.70.
- Minimal sufficient K must be within 0.02 AUC, 0.03 AUPRC, and 0.05 P@50 lift of the full-candidate probe.
- Parent labels `RE` and `QU` are consistency-only rows.

## Label summary

| Label | Role | Status | Class | Candidate pool | Filtered TopK candidates | Full AUC | Minimal K median | Stability Jaccard | Predictive redundancy | Selected latents |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| RE | parent_consistency_only | minimal_sufficient_found | parent_consistency_only | 100 | 100 | 0.898 | 19.0 | 0.204 | 0.526 | 29759,19435,9959,11660,17861,20808,20436,15396,10181,27061,26800,13430,31363,31133,14875,12852,30091,4018,8468 |
| RES | leaf_or_atomic | minimal_sufficient_found | distributed | 100 | 100 | 0.862 | 21.0 | 0.166 | 0.429 | 11660,20808,2068,29759,2462,28269,664,6623,31701,24943,28670,21859,13430,23077,8468,1976,2342,12560,20028,31585,5860 |
| REC | leaf_or_atomic | minimal_sufficient_found | distributed | 100 | 100 | 0.907 | 14.0 | 0.189 | 0.143 | 29759,31133,20436,31363,26800,30091,29845,26869,5569,12852,14875,9993,2995,31930 |
| QU | parent_consistency_only | minimal_sufficient_found | parent_consistency_only | 100 | 100 | 0.975 | 6.0 | 0.474 | 0.167 | 13430,664,22358,10916,18310,14247 |
| QUO | leaf_or_atomic | minimal_sufficient_found | distributed | 100 | 100 | 0.961 | 10.0 | 0.470 | 0.100 | 664,13430,18310,15679,14247,19840,21190,6129,8890,11071 |
| QUC | leaf_or_atomic | minimal_sufficient_found | distributed | 100 | 100 | 0.927 | 15.0 | 0.352 | 0.133 | 13430,22358,5413,664,21935,14014,29947,18646,8861,7037,2386,21859,6639,9216,20869 |
| GI | leaf_or_atomic | minimal_sufficient_found | distributed | 100 | 100 | 0.851 | 17.0 | 0.282 | 0.059 | 13430,664,26879,16345,13751,2055,1061,9893,16131,11873,29249,23723,17685,8166,4209,30153,19780 |
| SU | leaf_or_atomic | minimal_sufficient_found | distributed | 100 | 100 | 0.895 | 24.0 | 0.145 | 0.208 | 31311,29244,10539,16736,1681,23973,664,7741,6250,12852,25724,13430,23464,2699,5663,19663,31401,7389,19935,17861,27435,28603,9166,30223 |
| AF | leaf_or_atomic | minimal_sufficient_found | distributed | 100 | 100 | 0.930 | 12.0 | 0.245 | 0.083 | 23464,18492,7143,13430,9869,28724,664,30236,23242,23726,10116,15160 |

## Interpretation

- Compact labels: none.
- Moderate labels: none.
- Distributed labels: RES, REC, QUO, QUC, GI, SU, AF.
- Not recoverable under the candidate pool: none.
- Selected subspaces can be used as prioritized groups for later ablation or steering, but this report itself is predictive rather than causal.
