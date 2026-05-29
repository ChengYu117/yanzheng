# Step 6 Baseline Comparison Design

## 0. Purpose of This Step

This step is designed to compare whether **SAE latents** reveal a clearer and more interpretable label–representation structure than alternative feature bases such as **PCA components**, **raw hidden-state dimensions**, and optionally **linear-probe weights**.

The purpose is **not** to show only that SAE gives better classification performance. Instead, the core question is:

> Compared with PCA components and raw hidden-state dimensions, does SAE expose a clearer, sparser, and more interpretable structure between human-defined MISC behavior labels and LLM internal representations?

In the project, this step supports the broader claim that human-defined counseling interaction labels may align or misalign with LLM internal representations, and that the structure of this alignment can be systematically analyzed.

---

## 1. Main Research Questions

### RQ1. Predictive Association

Do SAE latents, PCA components, and raw hidden-state dimensions show measurable associations with MISC behavior labels?

### RQ2. Sparsity and Compactness

Does SAE concentrate each label's association into fewer features than PCA or raw hidden states?

### RQ3. Label Overlap

Does SAE reveal meaningful overlap between semantically related MISC labels?

### RQ4. Hierarchy Recovery

Does SAE better recover the expected MISC family structure, such as:

- Reflection: RE, REC
- Question: QU, QUO
- Information / Advice: GI, ADP
- Support / Affirmation: AF, SU
- Confrontation: CO

### RQ5. Interpretability

Are SAE features more semantically interpretable than PCA components or raw hidden-state dimensions?

---

## 2. Representations to Compare

Use the same utterance dataset, same label set, same model layer, and same pooling strategy for all representations.

| Representation | Feature Unit | Role in Comparison |
|---|---|---|
| SAE latents | Sparse autoencoder latent activations | Main interpretable representation |
| PCA components | PCA-transformed hidden-state dimensions | Linear dense baseline |
| Raw hidden states | Original hidden-state dimensions | Native model representation baseline |
| Linear-probe weights / contributions | Supervised discriminative directions | Optional predictive baseline |

The required minimum comparison should include:

1. SAE latents
2. PCA components
3. Raw hidden-state dimensions

Linear-probe weights can be included as an additional baseline, but they should be described as a supervised discriminative baseline rather than an inherently interpretable representation.

---

## 3. Data and Experimental Setup

### 3.1 Dataset

Use the same MISC utterance dataset as in the main analysis.

Each sample should include:

```text
utterance_id
utterance text
MISC labels
speaker or local context, if available
train / dev / test split
```

### 3.2 Label Set

Use the same representative MISC labels selected in earlier steps, for example:

```text
RE, REC, QU, QUO, GI, ADP, AF, SU, CO
```

If the project is time-constrained, a smaller but representative subset can be used, such as:

```text
RE, REC, QU, ADP, AF, CO
```

However, the label subset should still cover several behavior families.

### 3.3 Model Layer

All representation methods should be extracted from the same model layer.

Recommended setup:

```text
Primary layer: one middle-to-late layer, such as layer 12 or layer 16
Robustness check: one early layer, one middle layer, one late layer
```

For Gemma 3 4B with Gemma Scope 2, if the SAE is trained on `resid_post`, then PCA and raw hidden-state baselines should also use the corresponding residual stream hidden states from the same layer.

### 3.4 Token Pooling

Convert token-level representations into utterance-level representations using a fixed pooling strategy.

Recommended primary strategy:

```text
Mean pooling over utterance tokens
```

Possible alternatives:

| Pooling Strategy | Notes |
|---|---|
| Last-token pooling | Simple but may overemphasize sentence-final information |
| Mean pooling | Recommended primary strategy |
| Max pooling | Useful for SAE activations but may not be directly comparable to PCA/raw hidden states |
| Target-span pooling | Useful if utterance span boundaries are precise |

For fair baseline comparison, the main Table 4 should use the same pooling strategy across representations.

---

## 4. Constructing Each Representation

### 4.1 SAE Latents

For each utterance, extract hidden states from the selected model layer and pass them through the SAE encoder.

```text
hidden state h
↓
SAE encoder
↓
SAE latent activation z
```

Each utterance receives a vector:

```text
z_i ∈ R^M
```

where `M` is the number of SAE latents.

Each SAE latent is treated as one candidate interpretable feature.

---

### 4.2 PCA Components

Fit PCA only on the training split to avoid leakage.

Procedure:

```text
1. Collect hidden states H_train from training utterances.
2. Fit PCA on H_train.
3. Transform train/dev/test hidden states using the fitted PCA.
4. Use PCA component scores as features.
```

Each utterance receives:

```text
p_i ∈ R^K
```

Recommended choices for `K`:

| Choice | Description |
|---|---|
| K = d_model | Keeps dimensionality equal to raw hidden states; useful for fair comparison |
| K explaining 95% variance | More compact but may complicate fragmentation comparison |

For the main comparison, `K = d_model` is often cleaner because PCA becomes a rotated basis of the original hidden space.

---

### 4.3 Raw Hidden-State Dimensions

Use the original hidden state vector from the selected model layer:

```text
h_i ∈ R^d
```

Each hidden dimension is treated as one feature.

This baseline answers:

> Are the label structures already visible in the native hidden-state coordinates, or does SAE reveal a clearer basis?

---

### 4.4 Linear-Probe Weights or Contributions

This baseline is optional.

Train one-vs-rest linear classifiers for each label:

```text
label L ~ hidden state h
```

There are two possible uses:

#### A. Classification Baseline

Use the linear probe to report predictive performance:

```text
AUC
F1
Macro-F1
```

#### B. Structure Baseline

Use the probe weight vector for label `L`:

```text
w_L
```

or compute per-feature contribution:

```text
contribution_{i,j,L} = h_{i,j} * w_{j,L}
```

Important caveat:

> Linear-probe weights are supervised discriminative directions. They are useful as a predictive baseline but should not be treated as naturally interpretable units in the same way as SAE latents.

---

## 5. Unified Analysis Pipeline

For each representation method, run the same analysis function:

```text
analyze_representation(X, labels)
```

where `X` can be:

```text
SAE latent activations
PCA component scores
Raw hidden-state dimensions
Linear-probe contributions
```

The function should produce:

```text
feature-label association matrix
selected feature sets per label
fragmentation / compactness metrics
overlap metrics
polysemanticity metrics
hierarchy recovery metrics
classification performance
interpretability summary
```

This ensures the comparison is structural rather than merely predictive.

---

## 6. Feature–Label Association Analysis

For every feature `f_j` and every label `L`, compute association scores.

### 6.1 AUC

Use the feature value as a score for predicting the binary label.

```text
score = X[:, j]
target = y_L
```

Compute:

```text
AUC(f_j, L)
```

Interpretation:

- AUC close to 0.5: little association
- AUC much greater than 0.5: positive association
- AUC much less than 0.5: negative association or inverse signal

A useful nonnegative association score is:

```text
association_score = max(AUC - 0.5, 0)
```

This is convenient for later concentration, overlap, and hierarchy analysis.

---

### 6.2 Cohen's d

Compute effect size between positive and negative examples:

```text
d(f_j, L) = (mean_pos - mean_neg) / pooled_std
```

This captures how strongly the feature separates positive and negative examples.

---

### 6.3 Precision@k and Lift@k

For each feature, sort utterances by feature activation or score and compute:

```text
Precision@k = positive examples in top-k / k
```

Because labels may be imbalanced, also compute:

```text
Lift@k = Precision@k / base_rate(label)
```

Lift is useful because it accounts for different label frequencies.

---

### 6.4 Association Matrix

For each method, build a matrix:

```text
feature × label
```

Examples:

```text
SAE latent × MISC label
PCA component × MISC label
Raw hidden dimension × MISC label
```

Each cell can contain:

```text
AUC
AUC - 0.5
Cohen's d
Precision@k
Lift@k
Normalized association score
```

Recommended main score:

```text
max(AUC - 0.5, 0)
```

---

## 7. Selecting Label-Associated Feature Sets

For each label `L` and method `m`, define a selected feature set:

```text
S_L^m
```

These feature sets are used for fragmentation, overlap, polysemanticity, and hierarchy recovery.

### 7.1 Top-k Feature Sets

For each label, select the top-k features by association score:

```text
Top-20 features per label
```

Use this for:

- feature ranking tables
- qualitative examples
- heatmaps
- semantic inspection

However, top-k sets should not be the only basis for fragmentation, because if every label has exactly 20 features, fragmentation cannot vary.

---

### 7.2 Thresholded Feature Sets

Use thresholded sets for fragmentation and overlap.

Possible criteria:

```text
AUC > 0.65 or 0.70
Cohen's d > 0.4 or 0.5
Lift@50 > 1.5
FDR-corrected p < 0.05
```

The exact values can be adjusted based on data size and score distribution.

### 7.3 Null-Distribution Thresholds

A more robust approach is to generate thresholds from label permutation:

```text
1. Shuffle labels 100–500 times.
2. Recompute feature-label associations.
3. Estimate null distribution of maximum association scores.
4. Select features above the 95th percentile null threshold.
```

This is more computationally expensive but provides a fairer comparison across representations.

### 7.4 Threshold-Free Alternatives

Because thresholds can be unstable, also report threshold-free metrics such as:

- effective number of associated features
- association concentration
- entropy
- Gini coefficient
- top-k association mass

These are especially useful when comparing SAE, PCA, and raw hidden-state spaces with different dimensionalities.

---

## 8. Fragmentation and Compactness Metrics

The goal is to measure whether a label is represented compactly or distributed across many features.

### 8.1 Thresholded Fragmentation

For each label:

```text
Fragmentation(L) = |S_L|
```

To compare across representation spaces with different dimensionality:

```text
NormalizedFragmentation(L) = |S_L| / D
```

where `D` is the number of features in that representation.

Caution:

- SAE spaces may have many more features than hidden-state dimensions.
- Therefore, absolute fragmentation and normalized fragmentation should both be reported.
- Threshold-free metrics are recommended as the main comparison.

---

### 8.2 Effective Number of Associated Features

Let the nonnegative association score be:

```text
a_{j,L} = max(AUC(j,L) - 0.5, 0)
```

Normalize across features:

```text
p_{j,L} = a_{j,L} / sum_j a_{j,L}
```

Then compute:

```text
N_eff(L) = 1 / sum_j p_{j,L}^2
```

Interpretation:

- Smaller `N_eff`: label association is concentrated in fewer features.
- Larger `N_eff`: label association is distributed across many features.

This is a strong metric for comparing compactness across SAE, PCA, and raw hidden states.

---

### 8.3 Top-k Association Concentration

Compute how much association mass is captured by the top-k features:

```text
Concentration@k(L) = sum_{j in TopK(L)} p_{j,L}
```

Recommended values:

```text
Concentration@10
Concentration@20
Concentration@50
```

Interpretation:

- Higher concentration means the label structure is more compact.
- If SAE has higher Concentration@20 than PCA/raw hidden states, this supports the claim that SAE reveals a clearer sparse structure.

---

## 9. Label Overlap Metrics

The goal is to measure whether two labels share representation features.

### 9.1 Jaccard Overlap

For label pair `(L_i, L_j)`:

```text
Overlap(L_i, L_j) = |S_i ∩ S_j| / |S_i ∪ S_j|
```

Compute a label × label overlap matrix for each representation method.

Important comparisons:

```text
RE vs REC
QU vs QUO
GI vs ADP
AF vs SU
RE/REC vs QU/QUO
Information/advice vs reflection
Support/affirmation vs reflection
```

---

### 9.2 Weighted Overlap / Profile Similarity

Because thresholded sets can be unstable, also compare full association profiles.

For each label, define an association vector:

```text
a_L = [a_{1,L}, a_{2,L}, ..., a_{D,L}]
```

Then compute label-label similarity:

```text
cosine_similarity(a_Li, a_Lj)
Spearman rank correlation
rank-biased overlap
```

Recommended main weighted metric:

```text
WeightedOverlap(L_i, L_j) = cosine(a_Li, a_Lj)
```

This captures whether two labels are supported by similar representation profiles.

---

## 10. Polysemanticity and Feature Reuse

Although the minimum requirement emphasizes fragmentation and overlap, polysemanticity is useful for interpreting feature reuse.

For each feature `f_j`:

```text
Polysemanticity(f_j) = number of labels for which f_j ∈ S_L
```

Report:

| Metric | Meaning |
|---|---|
| % label-specific features | Features selected for only one label |
| % family-shared features | Features shared by labels within the same MISC family |
| % cross-family features | Features shared across different MISC families |
| Mean polysemanticity | Average number of labels supported per feature |
| Max polysemanticity | Whether highly generic features exist |

Expected pattern:

| Representation | Expected Structure |
|---|---|
| SAE | More label-specific or family-shared features |
| PCA | More dense and mixed components |
| Raw hidden states | More entangled dimensions |
| Linear probe | Supervised discriminative but not naturally interpretable |

---

## 11. Hierarchy Recovery

Hierarchy recovery tests whether the representation method recovers the expected structure of MISC behavior families.

### 11.1 Define MISC Family Structure

Example:

```text
Reflection: RE, REC
Question: QU, QUO
Information / Advice: GI, ADP
Support / Affirmation: AF, SU
Confrontation: CO
```

### 11.2 Label Profile Clustering

For each label, use its association profile:

```text
a_L = [a_{1,L}, ..., a_{D,L}]
```

Compute label-label similarity:

```text
cosine similarity
Spearman correlation
```

Then perform hierarchical clustering over labels.

Expected result:

- RE clusters with REC.
- QU clusters with QUO.
- GI clusters with ADP.
- AF clusters with SU.
- CO may remain relatively isolated.

---

### 11.3 Family Contrast Score

Compute:

```text
FamilyContrast =
mean(Sim(L_i, L_j) | same family)
-
mean(Sim(L_i, L_j) | different family)
```

Interpretation:

- Higher FamilyContrast means the representation better separates MISC families.
- If SAE has higher FamilyContrast than PCA/raw hidden states, it better recovers the human-defined label hierarchy.

---

### 11.4 Expected Pair Recovery

Define expected close label pairs:

```text
(RE, REC)
(QU, QUO)
(GI, ADP)
(AF, SU)
```

Rank all label pairs by similarity.

Compute:

```text
PairRecovery@4 =
number of expected family pairs appearing in top-4 most similar label pairs / 4
```

This is a simple and interpretable hierarchy recovery metric.

---

### 11.5 Optional: Silhouette by Family

Treat each label's association profile as a point and MISC family as its group.

Compute a silhouette score.

Caution:

- The number of labels may be small.
- Use this only as an auxiliary metric, not the main result.

---

## 12. Classification Performance as a Secondary Metric

Classification performance can be reported, but it should not be the main comparison.

For each representation, train the same classifier:

```text
one-vs-rest logistic regression
same train/dev/test split
same class balancing
same hyperparameter search
```

Input features:

```text
SAE activations
PCA components
Raw hidden states
```

Report:

```text
AUC
F1
Macro-F1
Per-label F1
Calibration, if available
```

Important framing:

> Classification performance is reported to contextualize predictive utility, but the primary comparison concerns the structure and interpretability of label–representation associations.

This prevents the baseline comparison from becoming a simple performance leaderboard.

---

## 13. Main Deliverable: Table 4

### Table 4: SAE vs PCA vs Raw Hidden States

| Representation | Mean AUC ↑ | Macro F1 ↑ | Mean N_eff ↓ | Concentration@20 ↑ | Mean Label Overlap | Family Contrast ↑ | Pair Recovery@4 ↑ | Mean Polysemanticity ↓ | Interpretability Summary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SAE latents |  |  |  |  |  |  |  |  |  |
| PCA components |  |  |  |  |  |  |  |  |  |
| Raw hidden dims |  |  |  |  |  |  |  |  |  |
| Linear probe contributions |  |  |  |  |  |  |  |  | Optional |

### Metric Interpretation

| Metric | Preferred Direction | Meaning |
|---|---|---|
| Mean AUC | Higher | Average feature-label association strength |
| Macro F1 | Higher | Predictive performance, secondary |
| Mean N_eff | Lower | Label association is concentrated in fewer features |
| Concentration@20 | Higher | Top features capture more label association mass |
| Mean Label Overlap | Context-dependent | Should be high mainly for semantically related labels |
| Family Contrast | Higher | Same-family labels are more similar than different-family labels |
| Pair Recovery@4 | Higher | Expected MISC family pairs are recovered |
| Mean Polysemanticity | Lower or moderate | Too high suggests overly generic features |
| Interpretability Summary | Qualitative | Whether top features/components are semantically coherent |

---

## 14. Recommended Figures

### Figure 4A: Label–Label Similarity Heatmaps

Produce one heatmap per representation:

```text
SAE label similarity heatmap
PCA label similarity heatmap
Raw hidden-state label similarity heatmap
```

Purpose:

- Show whether same-family labels are closer in each representation.
- Visually compare hierarchy recovery.

---

### Figure 4B: Association Concentration Curves

For each method, plot:

```text
x-axis: top-k features
y-axis: cumulative association mass
```

Purpose:

- Show whether SAE concentrates label association into fewer features.
- A steeper curve indicates a more compact representation.

---

### Figure 4C: Hierarchical Clustering Dendrogram

Cluster labels using association-profile similarity.

Purpose:

- Show whether expected MISC family pairs cluster together.
- Compare SAE vs PCA vs raw hidden states.

---

### Figure 4D: Per-Label Effective Number

Bar plot:

```text
x-axis: labels
y-axis: N_eff
grouped by representation method
```

Purpose:

- Identify which labels are compact or fragmented in each representation.

---

## 15. Minimal Implementation Plan

If time is limited, implement the following minimal version.

### Required Representations

```text
SAE latents
PCA components
Raw hidden states
```

### Required Metrics

```text
1. Feature-label AUC
2. Effective number of associated features
3. Concentration@20
4. Label-label cosine similarity
5. Family contrast score
6. Expected pair recovery
```

### Required Outputs

```text
Table 4: SAE vs PCA vs raw hidden states
Three label-label similarity heatmaps
One concentration curve plot
One short interpretation paragraph
```

### Minimal Table 4

| Representation | Mean Label AUC ↑ | Mean N_eff ↓ | Concentration@20 ↑ | Family Contrast ↑ | Pair Recovery@4 ↑ | Main Interpretation |
|---|---:|---:|---:|---:|---:|---|
| SAE latents |  |  |  |  |  |  |
| PCA components |  |  |  |  |  |  |
| Raw hidden states |  |  |  |  |  |  |

This minimal version is enough to answer whether SAE reveals a more sparse and hierarchy-consistent label structure.

---

## 16. Full Implementation Plan

If time allows, implement the full version.

### Representations

```text
SAE latents
PCA components
Raw hidden states
Linear-probe contributions
```

### Layer Robustness

```text
Early layer
Middle layer
Late layer
```

### Feature Selection Rules

```text
Top-k
Thresholded association
Null-distribution threshold
```

### Structure Metrics

```text
Fragmentation
Normalized fragmentation
Effective number
Concentration@k
Jaccard overlap
Weighted overlap
Polysemanticity
Family contrast
Expected pair recovery
Hierarchy clustering
```

### Performance Metrics

```text
AUC
F1
Macro-F1
Per-label F1
Calibration, if useful
```

### Interpretability Validation

For each method, sample top features/components and inspect high-scoring utterances.

Compare whether:

- SAE latents show coherent counseling behavior patterns.
- PCA components mix multiple unrelated behaviors.
- Raw hidden-state dimensions are difficult to interpret.
- Probe directions are predictive but not easily grounded in examples.

---

## 17. Important Controls and Caveats

### 17.1 PCA Must Be Fit on Train Split Only

Do not fit PCA on the full dataset.

Correct procedure:

```text
fit PCA on train hidden states
transform dev/test hidden states
```

### 17.2 Do Not Use Only Fixed Top-k for Fragmentation

If every label receives exactly top-20 features, fragmentation is artificially fixed.

Use:

```text
thresholded sets
effective number
concentration curves
```

### 17.3 SAE Has More Features Than Raw Hidden States

SAE feature count may be much larger than `d_model`.

Therefore, avoid relying only on absolute feature counts.

Report:

```text
absolute fragmentation
normalized fragmentation
effective number
concentration@k
```

### 17.4 Linear Probe Is Not an Interpretable Unit Basis

Linear probes are useful but supervised.

Suggested wording:

> We include linear probes as a discriminative baseline, but not as an inherently interpretable feature basis.

### 17.5 Classification Performance Is Not Interpretability

Raw hidden states may classify labels well but still be difficult to interpret.

Suggested wording:

> Raw hidden states may provide strong predictive performance, but SAE latents may expose more localized, sparse, and human-interpretable label structures.

---

## 18. Suggested Method Section Text

You can adapt the following text for the paper:

> To evaluate whether SAE features reveal a more interpretable label–representation structure than alternative feature bases, we compare SAE latents against PCA components and raw hidden-state dimensions extracted from the same model layer and utterance set. For each representation, we compute feature–label association scores using AUC, effect size, and top-k precision. We then derive label-specific feature sets and evaluate their structural properties, including fragmentation, association concentration, label overlap, feature polysemanticity, and recovery of the MISC label hierarchy. Classification performance is reported as a secondary measure, while our primary comparison focuses on whether each representation exposes sparse, label-specific, and hierarchy-consistent structure.

---

## 19. Expected Interpretation of Results

The ideal result is not necessarily that SAE achieves the highest classification performance.

The stronger and more relevant claim is:

> SAE latents achieve comparable predictive association to PCA and raw hidden states, but reveal a more compact and hierarchy-consistent structure: label associations are concentrated in fewer features, within-family label pairs show stronger overlap, and top activating examples are more semantically coherent.

Possible outcomes and interpretations:

| Outcome | Interpretation |
|---|---|
| SAE has higher Concentration@20 | SAE provides more compact label representations |
| SAE has higher Family Contrast | SAE better recovers MISC hierarchy |
| SAE has lower N_eff | Fewer features carry most label association |
| SAE has clearer top examples | SAE is more human-interpretable |
| Raw hidden states have higher F1 | Raw states may be predictive but less interpretable |
| PCA has similar classification but weaker hierarchy recovery | PCA captures variance but not necessarily behavior-label structure |

---

## 20. Final Recommendation

Use Step 6 as a structural interpretability comparison, not a performance leaderboard.

The recommended primary comparison is:

```text
SAE latents vs PCA components vs raw hidden states
```

The recommended main metrics are:

```text
Mean feature-label AUC
Effective number of associated features
Concentration@20
Label overlap matrix
Family contrast score
Expected family pair recovery
```

The recommended supporting metrics are:

```text
Macro-F1
Polysemanticity
Semantic coherence examples
```

The main conclusion should answer:

> Does SAE expose a clearer, sparser, and more hierarchy-consistent label–representation structure than PCA or raw hidden states?

If yes, this supports the broader project claim that SAE can make human-defined counseling behavior labels more interpretable at the level of LLM internal representations.
