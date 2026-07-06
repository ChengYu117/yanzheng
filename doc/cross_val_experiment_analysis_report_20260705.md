# AUC / Cohen's d cross-validation experiment analysis report

Date: 2026-07-05

## 1. Executive conclusion

This experiment largely supports the reliability of the filtered-pool label-latent association results as **candidate-discovery evidence**. The original full-data AUC and Cohen's d values are not contradicted by the cross-validation analyses: split-half recomputation produces very small AUC differences, and the merged cross-validation matrix preserves the original association values exactly.

The stronger conclusion is not that individual latents are fixed mechanisms. The evidence supports this narrower claim:

> In the filtered SAE latent pool, many MISC label-latent associations are numerically stable enough to serve as reproducible statistical candidates, but label-specific Top20 lists and latent rankings still vary across splits, especially for SU, GI, RES, and QUC. These results should be used as an auditable candidate layer before human semantic review, contrastive validation, and causal/intervention tests.

The most robust labels under the current checks are AF, QUO, GI, REC, RE, and RES in the sense that their cross-quality stable fractions are high. The weakest label is SU: it has the smallest positive sample size, the lowest cross-quality stable fraction, and the largest cross-quality risk count.

## 2. Research purpose

The original label-latent association analysis computed AUC, Cohen's d, directional AUC, FDR, and Precision@K on the full filtered dataset as a single snapshot. That produced useful candidate rankings, but it left four major uncertainties:

1. **Data leakage risk**: related utterances from the same `source_file` could influence both training/selection and evaluation-like calculations.
2. **Repeated text risk**: repeated or near-repeated `unit_text` entries could make association scores look more stable than they are.
3. **Winner's curse risk**: with 12,758 filtered latents and 9 labels, the highest-ranking Top20 latents may include noise peaks.
4. **No uncertainty interval**: AUC and Cohen's d were point estimates, so the variance of each effect estimate was unknown, especially for low-frequency labels such as SU and RES.

The experiment therefore aimed to upgrade the association matrix from a full-data point estimate into an auditable evidence object with reproducibility checks, grouped resampling, and candidate status labels.

## 3. Experimental design

The experiment used the filtered latent pool rather than the full 32,768-dimensional SAE space. The candidate pool was:

- `outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered/latent_label_matrix.csv`
- `outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered/feature_filter_audit.csv`
- `outputs/misc_full_sae_eval/feature_store/utterance_features.pt`
- `outputs/misc_full_sae_eval/label_matrix.csv`

The retained filtered pool contains 12,758 latents and produces 114,822 label-latent rows, equal to 12,758 latents times 9 MISC labels.

The implemented experiment has four parts.

### E0: Grouping and deduplication basis

E0 creates an audit basis for grouped validation. The core purpose is to avoid treating highly related utterances as independent observations. The generated mapping is:

- `outputs/cross_val/dedup_group_mapping.csv`

### E1: Repeated split-half TopK reproducibility

E1 repeatedly splits the data by `source_file`, recomputes the filtered label-latent association matrix on each half, and compares the two halves. The current run uses 50 random grouped bisections with seeds 42 through 91. The original `split_a_*`, `split_b_*`, and `split_half_*` files are retained as the first representative split for backward-compatible inspection, while the repeated-run evidence is stored separately.

The two main questions are:

- Do the full ranked lists correlate across independent grouped splits?
- Do the Top20 candidates overlap more than random Top20 draws from the same filtered latent pool?

The outputs are:

- `outputs/cross_val/topk_reproducibility/split_a_latent_label_matrix.csv`
- `outputs/cross_val/topk_reproducibility/split_b_latent_label_matrix.csv`
- `outputs/cross_val/topk_reproducibility/split_half_rank_correlation.csv`
- `outputs/cross_val/topk_reproducibility/split_half_top20_jaccard.csv`
- `outputs/cross_val/topk_reproducibility/repeated_split_rank_correlation.csv`
- `outputs/cross_val/topk_reproducibility/repeated_split_top20_jaccard.csv`
- `outputs/cross_val/topk_reproducibility/repeated_split_summary.csv`
- `outputs/cross_val/topk_reproducibility/split_half_summary.md`

### E2: Grouped bootstrap confidence intervals

E2 estimates uncertainty for Top100 positive Cohen's d latents per label using grouped bootstrap over `source_file`. It reports confidence intervals for AUC and Cohen's d.

The output is:

- `outputs/cross_val/bootstrap_ci/bootstrap_ci_by_label_latent.csv`

The important interpretation rule is that a positive Cohen's d candidate should not be promoted if its CI includes zero.

### E3: Cross-quality validation

E3 tests whether associations selected in high-quality subsets transfer to low-quality subsets, and vice versa. This checks whether a latent is primarily tracking label behavior or quality/source confounds.

The outputs are:

- `outputs/cross_val/cross_quality_validation/high_latent_label_matrix.csv`
- `outputs/cross_val/cross_quality_validation/low_latent_label_matrix.csv`
- `outputs/cross_val/cross_quality_validation/cross_quality_auc_comparison.csv`
- `outputs/cross_val/cross_quality_validation/cross_quality_summary.csv`
- `outputs/cross_val/cross_quality_validation/cross_quality_summary.md`

The final merged evidence object is:

- `outputs/cross_val/filtered_pool_association_matrix_with_cv.csv`

## 4. Result summary

### 4.1 Original scores versus split-half recomputation

The merged CV matrix has the same row count as the original filtered association matrix:

| artifact | rows |
|---|---:|
| original filtered association matrix | 114,822 |
| split A matrix | 114,822 |
| split B matrix | 114,822 |
| merged CV matrix | 114,822 |

The original AUC, directional AUC, and Cohen's d columns in `filtered_pool_association_matrix_with_cv.csv` are exactly identical to the original `latent_label_matrix.csv` values. The maximum absolute difference is 0 for all three metrics.

When recomputed on the two source-file split halves, the average absolute split-vs-original differences over all rows are small:

| metric | mean abs diff | p95 abs diff | max abs diff |
|---|---:|---:|---:|
| Cohen's d | 0.0298 | 0.0857 | 0.4962 |
| AUC | 0.0020 | 0.0066 | 0.0459 |
| directional AUC | 0.0020 | 0.0066 | 0.0675 |

For the original per-label Top20 positive Cohen's d candidates, the split-vs-original differences are still modest:

| metric | mean abs diff | p95 abs diff | max abs diff |
|---|---:|---:|---:|
| Cohen's d | 0.0716 | 0.1858 | 0.3335 |
| AUC | 0.0088 | 0.0208 | 0.0329 |
| directional AUC | 0.0088 | 0.0208 | 0.0329 |

If the split A and split B estimates are averaged before comparing with the original full-data estimate, the Top20 differences are tiny:

| metric | mean abs diff | p95 abs diff | max abs diff |
|---|---:|---:|---:|
| Cohen's d | 0.0074 | 0.0282 | 0.1859 |
| AUC | 0.0005 | 0.0017 | 0.0031 |
| directional AUC | 0.0005 | 0.0017 | 0.0031 |

Interpretation: the original numerical effect estimates are not meaningfully contradicted by grouped split recomputation. The full-data point estimates look like stable central estimates, not arbitrary one-off values.

### 4.2 E1 repeated split-half reproducibility

All labels pass the Top20 Jaccard null test in all 50 repeated grouped split-half runs for both Cohen's d and directional AUC. This means every tested label/metric pair repeatedly produced Top20 overlap far above random Top20 draws from the 12,758-latent filtered pool.

However, the magnitude of the overlap varies, and the repeated run changes the strength of the interpretation. The evidence is strong for Top20 set stability in QU and QUO, moderate for AF, QUC, RE, and REC depending on metric, and weak for GI, RES, and SU under positive Cohen's d. Full ranked-list stability is more limited: only QU, RE, and REC have Cohen's d Spearman consistently above 0.60; QUO does so in 64% of repeated splits. Directional AUC gives stronger Top20 set overlap for several labels, but its full-ranking Spearman often remains below 0.60.

| label | positive n | Cohen's d Jaccard mean | Cohen's d p05..p95 | Cohen's d Spearman mean | directional AUC Jaccard mean | directional AUC p05..p95 | directional AUC Spearman mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| AF | 349 | 0.573 | 0.379..0.667 | 0.503 | 0.523 | 0.429..0.600 | 0.518 |
| GI | 681 | 0.192 | 0.095..0.290 | 0.510 | 0.418 | 0.310..0.538 | 0.444 |
| QU | 1974 | 0.800 | 0.739..0.818 | 0.657 | 0.799 | 0.739..0.905 | 0.567 |
| QUC | 768 | 0.578 | 0.401..0.739 | 0.485 | 0.637 | 0.481..0.739 | 0.450 |
| QUO | 1206 | 0.831 | 0.739..0.957 | 0.607 | 0.852 | 0.739..0.905 | 0.531 |
| RE | 1358 | 0.490 | 0.333..0.637 | 0.669 | 0.766 | 0.600..0.905 | 0.593 |
| REC | 842 | 0.424 | 0.310..0.538 | 0.689 | 0.536 | 0.429..0.667 | 0.641 |
| RES | 516 | 0.223 | 0.143..0.314 | 0.423 | 0.548 | 0.452..0.600 | 0.448 |
| SU | 222 | 0.184 | 0.026..0.379 | 0.512 | 0.342 | 0.193..0.429 | 0.557 |

The repeated run also compares each split pair to the full-data Top20. The union of split A and split B usually covers most full-data Top20 latents: mean reference-union coverage is at least 0.911 for all Cohen's d labels and at least 0.942 for all directional AUC labels. This means the original full-data Top20 candidates usually reappear somewhere across the two halves, even when the two halves disagree about their exact Top20 membership.

Interpretation: E1 now gives stronger evidence than the original one-shot split that the candidate pool is not random. It still does not prove exact ranking stability. The safe reporting language is: "repeated grouped split-half validates a non-random, mostly recoverable candidate set; exact Top20 membership and rank order remain label-dependent and unstable for GI, RES, and SU under positive Cohen's d."

### 4.3 E2 bootstrap CI and candidate status

The merged matrix assigns one of four CV candidate statuses:

| status | count |
|---|---:|
| `ci_not_available_or_includes_zero` | 113,927 |
| `ci_supported_not_cross_quality_tested` | 625 |
| `reproducible_candidate` | 199 |
| `cross_quality_risk` | 71 |

The large number of `ci_not_available_or_includes_zero` rows is expected because bootstrap CI was only run on Top100 positive Cohen's d latents per label, not on every label-latent row. The meaningful candidate layer is therefore the Top100-per-label subset.

Per-label Top100 status distribution:

| label | reproducible | cross-quality risk | CI-supported only | CI unavailable or includes zero |
|---|---:|---:|---:|---:|
| AF | 25 | 1 | 74 | 12,658 |
| GI | 35 | 1 | 64 | 12,658 |
| QU | 18 | 8 | 74 | 12,658 |
| QUC | 17 | 14 | 69 | 12,658 |
| QUO | 21 | 2 | 77 | 12,658 |
| RE | 22 | 7 | 71 | 12,658 |
| REC | 24 | 5 | 71 | 12,658 |
| RES | 24 | 6 | 65 | 12,663 |
| SU | 13 | 27 | 60 | 12,658 |

Interpretation: CI support is useful for screening out unstable point estimates. A `reproducible_candidate` row is stronger than the old full-data association row, but it is still a statistical candidate status. It is not a semantic label and not a causal mechanism.

### 4.4 E3 cross-quality stability

E3 shows that most labels have limited average cross-quality AUC drop, but the risk is label-specific.

| label | stable fraction | mean AUC drop | max AUC drop | high-low rank Spearman |
|---|---:|---:|---:|---:|
| AF | 0.975 | 0.007 | 0.078 | 0.413 |
| GI | 0.925 | 0.018 | 0.089 | 0.409 |
| QU | 0.775 | 0.028 | 0.212 | 0.456 |
| QUC | 0.650 | 0.045 | 0.230 | 0.352 |
| QUO | 0.950 | 0.004 | 0.077 | 0.415 |
| RE | 0.825 | 0.013 | 0.070 | 0.442 |
| REC | 0.875 | 0.023 | 0.088 | 0.509 |
| RES | 0.850 | 0.015 | 0.117 | 0.278 |
| SU | 0.325 | 0.065 | 0.151 | 0.368 |

Interpretation: most labels transfer reasonably across high/low quality splits. SU is the clear exception. QUC also shows elevated risk, with lower stable fraction and the highest max AUC drop.

## 5. Label-level interpretation

### Stronger labels

**AF** has strong cross-quality stability: stable fraction 0.975 and mean AUC drop 0.007. Its Top20 Jaccard is 0.600 for both Cohen's d and directional AUC. Although the sample size is modest, AF candidates look comparatively reliable under this audit.

**QUO** is also strong: directional AUC Top20 Jaccard is 0.818, cross-quality stable fraction is 0.950, and mean AUC drop is only 0.004. This label is suitable for higher-confidence candidate reporting.

**GI** has weak Cohen's d Top20 set overlap, but cross-quality stability is high: stable fraction 0.925 and mean AUC drop 0.018. This suggests that the broad association signal is robust, while the exact Top20 Cohen's d list is not as stable.

### Moderate labels

**RE** has strong directional AUC Top20 overlap at 0.818 and acceptable cross-quality stability at 0.825. Since RE is a parent label containing RES/REC structure, claims should be hierarchy-aware and should not treat RE overlap as an independent mechanistic discovery.

**REC** has high ranking Spearman for Cohen's d and directional AUC, but only moderate Top20 Jaccard. This suggests the overall ranking structure is more stable than the exact Top20 membership.

**RES** remains cautionary. It has only 516 positives, Cohen's d Top20 Jaccard is 0.250, and rank correlation is low. However, cross-quality stable fraction is 0.850 and mean AUC drop is 0.015, so it should not be discarded. The right interpretation is: RES has a usable association signal, but exact latent selection is unstable and should be validated manually.

### Higher-risk labels

**QUC** has moderate Top20 overlap but weaker cross-quality stability: stable fraction 0.650, mean AUC drop 0.045, and max drop 0.230. It should be reported as partially stable, with source-quality confounding risk.

**SU** is the weakest label in this audit. It has only 222 positives, Cohen's d Top20 Jaccard is 0.176, stable fraction is 0.325, mean AUC drop is 0.065, and it has 27 cross-quality risk rows among the Top100 bootstrapped candidates. SU should be downgraded to an exploratory candidate label unless later evidence improves its stability.

## 6. What the experiment validates

The experiment validates four useful claims:

1. The filtered-pool association matrix is not a purely accidental full-data snapshot.
2. Many Top20 candidate sets overlap across grouped splits far more than random latent selections.
3. Bootstrap CI can separate candidate rows with stable positive effect estimates from rows whose effect is uncertain or unavailable.
4. Cross-quality validation identifies label-specific source-confounding risk, especially for SU and QUC.

These results make the filtered-pool association matrix more suitable for downstream evidence packets and human review than the previous point-estimate-only matrix.

## 7. What the experiment does not validate

The experiment does not prove that a latent is a clinical concept, a human-interpretable mechanism, or a causal representation of a MISC behavior.

The following claims remain too strong:

- "This latent is the SU mechanism."
- "The Top20 list is a fixed representation of the label."
- "High Cohen's d means the latent encodes clinical understanding."
- "Cross-validation proves causal sufficiency."

The safer claims are:

- "This latent is a reproducible statistical candidate for label association."
- "This label has a distributed candidate subspace in the filtered SAE pool."
- "The candidate set survives grouped split and/or cross-quality checks."
- "The result is suitable for semantic review and contrastive validation."

## 8. Decision recommendation

The cross-validation experiment should be treated as a successful audit layer for candidate discovery.

Recommended stage decision:

| stage gate | decision |
|---|---|
| Use filtered-pool association matrix for candidate discovery | pass |
| Use `reproducible_candidate` as stronger candidate flag | pass |
| Promote exact Top20 lists to mechanism claims | fail |
| Use SU as a main positive example | caution / defer |
| Use QUO, AF, RE, REC, GI as stronger examples | pass with label-specific caveats |
| Move to human semantic review and contrastive validation | ready |

## 9. Recommended next steps

1. Update downstream feature cards to include `cv_candidate_status`, bootstrap CI columns, and cross-quality risk flags.
2. In reports, rank candidates by a combined evidence rule: positive Cohen's d, CI excludes zero, Top20 split-half overlap, and cross-quality stable status.
3. For SU, avoid using a single strongest latent as a headline example. Require manual review plus contrastive prompts before interpretation.
4. For RES, separate positive reflective-listening candidates from boundary or negative-discriminative latents.
5. For RE/RES/REC and QU/QUO/QUC, preserve label hierarchy in interpretation; parent-child overlap is a hierarchy consistency check, not standalone evidence of independent shared mechanisms.
6. Use the cross-validation report as a gate before any latent-function induction or human annotation packet is promoted to the main paper narrative.

## 10. Source artifacts

Experiment plan:

- `doc/交叉验证实验计划.md`

Execution log:

- `doc/日志.md`

Summary outputs:

- `outputs/cross_val/cross_val_summary.md`
- `outputs/cross_val/topk_reproducibility/split_half_summary.md`
- `outputs/cross_val/cross_quality_validation/cross_quality_summary.md`

Primary data outputs:

- `outputs/cross_val/filtered_pool_association_matrix_with_cv.csv`
- `outputs/cross_val/bootstrap_ci/bootstrap_ci_by_label_latent.csv`
- `outputs/cross_val/topk_reproducibility/split_half_top20_jaccard.csv`
- `outputs/cross_val/topk_reproducibility/split_half_rank_correlation.csv`
- `outputs/cross_val/cross_quality_validation/cross_quality_auc_comparison.csv`
