# MISC SAE Interpretability Pipeline

本项目面向心理咨询 MISC 行为单元，使用 Llama-3.1-8B 的中间层 hidden states 和 Sparse Autoencoder（SAE）特征，分析人工行为标签与模型内部 latent 表征之间的结构关系。

当前主线问题不是训练分类器，而是回答：

- MISC 行为标签是否对应模型内部的单个 SAE latent？
- 不同行为标签是否由多个 latent 共同表达？
- 同一个 latent 是否会同时参与多个行为标签？
- `RE/RES/REC`、`QU/QUO/QUC` 等父子标签结构是否能在 SAE 空间中被部分恢复？

## 当前主数据

默认数据目录：

```text
data/mi_quality_counseling_misc/
  misc_annotations/*/*.jsonl
  metadata/labels.csv
```

主流程以每条 `unit_text` 作为统一样本单位，并标准化为：

- `sample_id`
- `file_id`
- `quality_label`
- `text`
- `predicted_code`
- `predicted_subcode`
- `label_re`
- `label_family`
- `confidence`
- `rationale`

标签口径：

- 二分类：`RE` vs `NonRE`
- MISC 多标签：`RE/RES/REC/QU/QUO/QUC/GI/SU/AF/OTHER`

`data/mi_re` 仍作为 legacy 兼容数据保留，需通过参数显式指定。

## 主要入口

### 1. 全量 SAE 评估与矩阵生成

```powershell
python run_sae_evaluation.py `
  --model-dir D:\project\NLP_v3\NLP_data\Llama-3.1-8B `
  --device cuda `
  --data-dir data/mi_quality_counseling_misc `
  --data-format misc_full `
  --label-mode misc_multilabel `
  --batch-size 4 `
  --max-seq-len 128 `
  --full-structural `
  --checkpoint-topk-semantics hard `
  --output-dir outputs/misc_full_sae_eval
```

关键输出：

```text
outputs/misc_full_sae_eval/
  dataset_summary.json
  records.jsonl
  label_matrix.csv
  feature_store/
    utterance_features.pt
    utterance_activations.pt
  functional/
    misc_label_mapping/
    re_binary/
  metrics_structural.json
  metrics_functional.json
```

### 2. Mapping Structure 分析

```powershell
python run_misc_mapping_structure_analysis.py `
  --mapping-dir outputs/misc_full_sae_eval/functional/misc_label_mapping `
  --output-dir outputs/misc_full_sae_eval/interpretability/mapping_structure
```

关键输出：

```text
outputs/misc_full_sae_eval/interpretability/mapping_structure/
  mapping_structure_metrics.json
  label_fragmentation_rank.csv
  latent_overlap_distribution.csv
  label_pair_similarity.csv
  hierarchy_alignment.csv
  latent_role_summary.csv
  mapping_structure_report.md
  figures/
```

### 3. 后续可解释性分析

```powershell
python run_misc_interpretability_analysis.py `
  --eval-dir outputs/misc_full_sae_eval `
  --output-dir outputs/misc_full_sae_eval/interpretability/followup
```

### 4. 因果验证候选 latent 导出

```powershell
python run_misc_causal_candidate_export.py `
  --eval-dir outputs/misc_full_sae_eval `
  --output-dir outputs/misc_full_sae_eval/causal_candidates
```

### 5. Latent 空间搜索 v2

```powershell
python run_misc_latent_space_search_v2.py
```

该入口复用已经生成的 `outputs/misc_full_sae_eval` 特征和标签矩阵，不重新抽取模型 hidden states。它把旧 Top20 口径降级为人工可审查候选窗口，正式结论改用 thresholded set、weighted metrics、K-sensitivity、semantic review candidates 和 baseline comparison 支撑。

默认输入：

```text
outputs/misc_full_sae_eval/functional/misc_label_mapping/latent_label_matrix.csv
outputs/misc_full_sae_eval/feature_store/utterance_features.pt
outputs/misc_full_sae_eval/label_matrix.csv
outputs/misc_full_sae_eval/records.jsonl
```

关键输出：

```text
outputs/misc_full_sae_eval/interpretability/latent_space_search_v2/
  latent_label_association_v2.csv
  top20_candidate_set_v2.csv
  thresholded_latent_sets_v2.csv
  weighted_latent_label_matrix_v2.csv
  fragmentation_v2.csv
  overlap_thresholded_v2.csv
  overlap_weighted_v2.csv
  polysemanticity_v2.csv
  hierarchy_recovery_v2.csv
  k_sensitivity_summary_v2.csv
  semantic_review_candidates_v2.csv
  latent_space_search_report.md
  figures/
```

正式阈值口径为 `directional_auc >= 0.70`、`abs_cohens_d >= 0.50`、`significant_fdr=True`，正向 support latent 额外要求 `precision_at_50 >= prevalence + 0.10`。负向 latent 只作为 `negative_boundary`，不写成正向语义证据。

### 6. 最小充分 latent 子空间 v2

```powershell
python run_misc_minimal_sufficient_subspace_v2.py
```

该入口回答“每个 MISC 标签至少需要多少个 SAE latents 才能接近完整候选池的预测表现”。它不是因果充分性实验，而是 probe-space 的预测充分性实验；输出的 `S*` 用作后续 ablation / steering 的优先候选组。

默认输入：

```text
outputs/misc_full_sae_eval/interpretability/latent_space_search_v2/latent_label_association_v2.csv
outputs/misc_full_sae_eval/interpretability/latent_space_search_v2/thresholded_latent_sets_v2.csv
outputs/misc_full_sae_eval/feature_store/utterance_features.pt
outputs/misc_full_sae_eval/label_matrix.csv
```

方法口径：

- 候选池优先使用 `stable_edge=True` 的 latents，再补充 `association_rank <= 100` 的备份候选。
- 每个标签使用 `5-fold Stratified CV`。
- 每折训练 full-candidate logistic probe，再用 full probe 的线性贡献做 additive greedy selection。
- 最小充分 K 需同时满足 `AUC >= 0.70`、`AUC >= full_candidate_auc - 0.02`、`AUPRC >= full_candidate_auprc - 0.03`、`Precision@50 lift >= full_candidate_precision_lift_at_50 - 0.05`。
- `RE/QU` 只作 parent-child consistency，不进入 leaf-label 主结论。

关键输出：

```text
outputs/misc_full_sae_eval/interpretability/minimal_sufficient_subspace_v2/
  minimal_sufficient_summary_v2.csv
  minimal_sufficient_selected_latents_v2.csv
  minimal_sufficient_fold_results_v2.csv
  minimal_sufficient_selection_steps_v2.csv
  minimal_sufficient_redundancy_audit_v2.csv
  minimal_sufficient_curves_v2.csv
  candidate_pool_v2.csv
  minimal_sufficient_summary.json
  minimal_sufficient_subspace_report.md
  figures/
```

## 关键代码结构

```text
src/nlp_re_base/
  data.py                         # MISC/legacy 数据读取与标准化
  model.py                        # 本地 Llama 模型与 tokenizer 加载
  sae.py                          # SAE 加载与前向计算封装
  activations.py                  # hidden states 抽取、SAE feature 聚合、特征保存
  eval_structural.py              # 结构指标
  eval_functional.py              # 二分类功能指标
  misc_label_mapping.py           # latent × MISC label 矩阵
  mapping_structure.py            # Mapping Structure 结构分析
  behavior_interpretability.py    # 后续解释性统计
  causal_candidates.py            # 因果验证候选 latent 导出
  latent_space_search_v2.py       # thresholded / weighted latent 空间搜索 v2
  minimal_sufficient_subspace_v2.py # 最小充分 latent 子空间搜索 v2
```

## 测试

轻量测试：

```powershell
python test_dataset_loader_smoke.py
python test_misc_label_mapping_smoke.py
python test_mapping_structure_analysis.py
python test_behavior_interpretability.py
python test_causal_candidate_export.py
python test_latent_space_search_v2_smoke.py
python test_minimal_sufficient_subspace_v2_smoke.py
python test_deploy_smoke.py
```

完整模型推理需要本地或云端 GPU 与 Llama-3.1-8B 权重。

## 云端运行

GCE/云服务器脚本位于：

```text
deploy/gce/
  bootstrap.sh
  download_model.sh
  run_full_pipeline.sh
  run_full_eval.sh
  run_causal.sh
```

推荐先参考：

```text
doc/云服务器部署运行教程.md
doc/本地实验运行说明.md
docs/PROJECT_DOSSIER.md
```

## 当前清理状态

早期 cactus 数据构建、基础文本生成 demo、Stage2 activation extraction 旧阶段入口已经从主代码中移除。当前仓库主线以 MISC 全量数据、SAE 特征矩阵、Mapping Structure 和后续可解释性分析为准。
