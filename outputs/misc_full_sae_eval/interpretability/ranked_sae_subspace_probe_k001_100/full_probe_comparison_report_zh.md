# SAE top-n 子空间探针对比报告

## 问题

本实验检验人工 MISC 标签能否从 LLM 内部 SAE features 中被线性识别，并比较三类基线：原始 hidden activations、full SAE latents、full PCA(raw hidden)。新增部分使用每个标签在训练折内按 `abs_cohens_d` 或 `directional_auc` 排名最高的 top-n SAE features 组成子空间训练探针。

## 关键设置

- CV: `stratified-group-kfold`, folds=5, group column=`file_id`。
- PCA: `full`，每个训练折单独拟合。
- SAE 子空间排序: `cohens_d, directional_auc`。
- top-n 网格: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100`。
- top-n probe 并行数: `8`。
- 重要防泄漏设置: top-n feature 排名只使用当前训练折标签和训练折 SAE features，测试折只用于最终评估。

## Macro AUC 总览

| representation | mean_n_features | macro_auc | macro_average_precision | macro_f1 | macro_balanced_accuracy |
| --- | --- | --- | --- | --- | --- |
| full_sae_latents | 32768.000 | 0.890 | 0.600 | 0.572 | 0.815 |
| pca_raw_hidden | 4096.000 | 0.900 | 0.679 | 0.642 | 0.811 |
| raw_hidden | 4096.000 | 0.900 | 0.679 | 0.642 | 0.811 |

## SAE top-n 子空间最佳点

| subspace_ranking | top_n | macro_auc | macro_average_precision | macro_f1 | delta_macro_auc_vs_full_sae_latents | delta_macro_auc_vs_raw_hidden | delta_macro_auc_vs_pca_raw_hidden |
| --- | --- | --- | --- | --- | --- | --- | --- |
| directional_auc | 98.000 | 0.908 | 0.622 | 0.567 | 0.018 | 0.008 | 0.008 |
| cohens_d | 95.000 | 0.884 | 0.604 | 0.557 | -0.006 | -0.016 | -0.016 |

## 收敛性判断

- `cohens_d`: 最佳 macro AUC=0.884 at n=95; n=100 时 macro AUC=0.883; 首次进入最终点 0.01 范围的 n=48; 末 4 个步进平均 AUC 增益=-0.0002; 末端平台化=是。
- `directional_auc`: 最佳 macro AUC=0.908 at n=98; n=100 时 macro AUC=0.908; 首次进入最终点 0.01 范围的 n=57; 末 4 个步进平均 AUC 增益=0.0001; 末端平台化=是。

## 解释

- 最强 SAE top-n 子空间为 `directional_auc` n=98，macro AUC=0.908。
- 相比 raw hidden macro AUC=0.900，最佳 SAE top-n 子空间差值为 0.008。
- 相比 full SAE macro AUC=0.890，最佳 SAE top-n 子空间差值为 0.018。
- 相比 full PCA macro AUC=0.900，最佳 SAE top-n 子空间差值为 0.008。
- 审稿口径上，这支持 `MISC 标签信息可从 SAE feature 组合中被线性解码`；但仍不能单独证明这些 SAE features 是因果机制，或等同于人类标注机制。

## 标签层面提示

| label | representation | probe_auc_mean | probe_f1_mean | n_features_mean |
| --- | --- | --- | --- | --- |
| AF | sae_top_directional_auc_n098 | 0.936 | 0.509 | 98.000 |
| GI | sae_top_directional_auc_n100 | 0.846 | 0.424 | 100.000 |
| QU | sae_top_directional_auc_n097 | 0.977 | 0.909 | 97.000 |
| QUC | sae_top_directional_auc_n098 | 0.930 | 0.631 | 98.000 |
| QUO | sae_top_cohens_d_n100 | 0.959 | 0.781 | 100.000 |
| RE | sae_top_directional_auc_n100 | 0.903 | 0.672 | 100.000 |
| REC | sae_top_cohens_d_n096 | 0.911 | 0.583 | 96.000 |
| RES | sae_top_directional_auc_n100 | 0.840 | 0.346 | 100.000 |
| SU | sae_top_directional_auc_n091 | 0.880 | 0.266 | 91.000 |

完整 label 级结果见 `full_probe_by_label_summary.csv`。

## 输出文件

- `full_probe_summary.csv`: baseline 与所有 top-n 子空间 macro 指标。
- `full_probe_by_label_summary.csv`: label 级均值/方差。
- `full_probe_by_label.csv`: fold 级原始结果。
- `ranked_sae_subspace_convergence.csv`: top-n 收敛与相对 baseline 差值。
- `ranked_sae_subspace_selected_latents.csv`: 每个 label/fold/ranking 的 top feature 排名。
