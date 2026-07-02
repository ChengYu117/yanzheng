# MISC 95% Full SAE 最小 latent set 分析报告

## 结论先行

本次新增并运行 `minimal_full_sae_ratio_subspace_95` 分析，用 filtered Top100 SAE latents 为候选池，为每个 MISC 标签寻找能够达到同标签 Full SAE probe AUC 95% 的最小 latent set。

最终 9 个标签均找到达标 set。这里的达标口径是 **mean CV AUC 达到 `0.95 * Full SAE mean CV AUC`**，不是要求每个 fold 都单独超过 target；因此它是 predictive probe sufficiency，不是 causal sufficiency。

## 实验口径

- 候选池：`outputs/misc_full_sae_eval/functional/misc_label_mapping_filtered/latent_label_matrix.csv` 中每标签 filtered Top100。
- Full SAE baseline：复用 PCA 修正版 fold rows 中的 `full_sae_latents` AUC。
- split：`stratified-group-kfold`，`file_id` group，5 folds。
- 搜索：先在 fold 内做逐步候选探针搜索，再对聚合候选前缀做全局 CV 复核；最终 `final_selected_latents` 必须自身达到 mean CV AUC 95% target。
- 输出目录：`outputs/misc_full_sae_eval/interpretability/minimal_full_sae_ratio_subspace_95`。

## 结果总表

| Label | Role | Full SAE AUC | 95% target | Final K | Final CV AUC | Mean margin | Final selected latents |
|---|---|---:|---:|---:|---:|---:|---|
| RE | parent | 0.888 | 0.843 | 6 | 0.848 | +0.005 | 29759,11660,9959,31363,31133,19435 |
| RES | leaf | 0.794 | 0.754 | 4 | 0.756 | +0.002 | 11660,23077,31585,29759 |
| REC | leaf | 0.902 | 0.857 | 6 | 0.872 | +0.015 | 29759,26800,9993,31133,14875,31363 |
| QU | parent | 0.970 | 0.922 | 1 | 0.926 | +0.004 | 13430 |
| QUO | leaf | 0.946 | 0.898 | 2 | 0.907 | +0.009 | 664,13430 |
| QUC | leaf | 0.889 | 0.844 | 2 | 0.862 | +0.018 | 13430,22358 |
| GI | leaf | 0.823 | 0.782 | 5 | 0.791 | +0.009 | 13430,9893,31401,664,16345 |
| SU | leaf | 0.871 | 0.828 | 6 | 0.839 | +0.011 | 16736,17861,26739,24760,13430,13827 |
| AF | leaf | 0.927 | 0.881 | 3 | 0.881 | +0.000 | 23464,7143,13430 |

## 结果解读

- 问题类标签最紧：`QU` 作为父标签只需 1 个 latent，`QUO` 需要 2 个，`QUC` 需要 2 个。
- `RES/REC/GI/SU/AF` 在 95% Full SAE 口径下不需要 v2 full-candidate 充分性中那么大的 set，但仍不是单 latent 映射。
- `13430` 出现在多个最终 set 中，尤其覆盖 `QU/QUO/QUC/GI/SU/AF`。这应被看作共享判别 latent 或高复用候选，不应直接解释为某一个 MISC 标签的专属语义机制。
- `RE` 和 `QU` 是父标签一致性行，主结论仍应以 leaf/atomic 标签为主。

## 审计结论

- 45 个 fold-level discovered set 全部满足 `selected_auc >= target_auc`。
- 9 个 global final set 全部满足 `global_selected_auc_mean >= global_target_auc_mean`。
- 候选池最大 `candidate_order=100`，未越出 filtered Top100。
- filtered matrix 的所有候选 latent 都在 `feature_filter_audit keep=True` 池内。
- 第一次实现曾只用 fold 共识前缀作为 `final_selected_latents`，审计发现部分最终 set 自身 CV 复训后不达标；已修正为全局前缀 CV 复核后再确定最终 set。

## 解释边界

这些结果说明：在 filtered SAE latent 候选池内，可以找到很小的 latent 组合，使线性 probe 的 mean CV AUC 接近 Full SAE。它不能证明这些 latents 是模型执行 MISC 行为识别或生成的因果机制；后续仍需要 ablation、steering 或 activation patching。
