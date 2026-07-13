# PROJECT DOSSIER

## 一句话项目目标

本项目用 Llama-3.1-8B 的中间层激活和 OpenMOSS / Llama Scope Sparse Autoencoder（SAE）latent，分析心理咨询 MISC 行为标签在模型内部表征空间中的结构化映射关系，并为后续因果干预与人工语义审查提供候选 latent。

主要依据：

- 数据默认入口：`src/nlp_re_base/data.py` 中的 `DEFAULT_DATA_DIR = data/mi_quality_counseling_misc`
- 主实验入口：`run_sae_evaluation.py`
- MISC 映射分析入口：`run_misc_mapping_structure_analysis.py`
- 后续可解释性入口：`run_misc_interpretability_analysis.py`
- 因果候选导出入口：`run_misc_causal_candidate_export.py`
- 云端全流程入口：`deploy/gce/run_full_pipeline.sh`

## 研究问题 / hypothesis

1. MISC 人工标签与 SAE latent 不是一一对应关系，而是多对多的结构化映射。
   - 对应实现：`src/nlp_re_base/misc_label_mapping.py` 的 `compute_latent_label_associations()`、`build_label_fragmentation()`、`build_latent_overlap()`
   - 对应结构分析：`src/nlp_re_base/mapping_structure.py` 的 `run_mapping_structure_analysis()`

2. 不同行为标签在 SAE 空间中的表征结构不同，例如 `QU/QUO/QUC` 与 `RE/RES/REC` 可能呈现不同的集中度、重叠度和碎片化。
   - 核心标签定义：`src/nlp_re_base/mapping_structure.py` 的 `DEFAULT_CORE_LABELS`
   - 层级定义：`src/nlp_re_base/mapping_structure.py` 的 `DEFAULT_HIERARCHY_SPECS = ["RE:RES,REC", "QU:QUO,QUC"]`

3. 当前阶段主要提供相关性与结构性证据，不直接宣称单个 latent 等价于某个 MISC 标签，也不直接宣称因果机制成立。
   - 因果候选只做导出：`run_misc_causal_candidate_export.py`
   - 真正因果干预入口：`causal/run_experiment.py`

## 数据来源与数据格式

当前正式数据源是全量 MISC 数据集：

代码默认读取：

- `data/mi_quality_counseling_misc/misc_annotations/*/*.jsonl`
- `data/mi_quality_counseling_misc/metadata/labels.csv`

对应函数：

- `src/nlp_re_base/data.py::load_misc_full_records()`
- `src/nlp_re_base/data.py::load_experiment_dataset()`
- `src/nlp_re_base/data.py::infer_data_format()`

标准化后的样本字段包括：

- `sample_id`
- `record_id`
- `file_id`
- `quality_label`
- `text` / `unit_text`
- `predicted_code`
- `predicted_subcode`
- `label_re`
- `label_family`
- `labels`
- `confidence`
- `rationale`

当前支持的数据格式：

- `misc_full`：当前正式 MISC 全量数据
- `auto`：由 `src/nlp_re_base/data.py::infer_data_format()` 自动判断

## 数据预处理流程

数据预处理由 `src/nlp_re_base/data.py` 统一完成。

1. 解析数据目录。
   - 默认目录来自 `DEFAULT_DATA_DIR`
   - CLI 参数来自 `run_sae_evaluation.py --data-dir`

2. 判断数据格式。
   - `infer_data_format()` 检查 `misc_annotations`、`re_dataset.jsonl` / `nonre_dataset.jsonl`、`cactus_re_small_1500.jsonl`

3. 将 MISC JSONL 行标准化为统一 record。
   - 主函数：`load_misc_full_records()`
   - 文本单位：`unit_text`

4. 构造二分类标签。
   - `label_re=True` 的规则在 `src/nlp_re_base/data.py` 中实现
   - 规则：`predicted_code == "RE"` 或 `predicted_subcode in {"RES", "REC"}`

5. 构造 MISC 多标签矩阵。
   - 核心标签：`RE, RES, REC, QU, QUO, QUC, GI, SU, AF`
   - 非核心标签进入 `OTHER`
   - 输出文件：`label_matrix.csv`

6. 保存统一记录。
   - 输出文件：`records.jsonl`

## 模型 / 算法 / 实验流程

### 模型配置

基础模型配置来自 `config/model_config.json`：

- `model_name`: `Meta-Llama-3.1-8B`
- `model_path`: `models/Llama-3.1-8B`
- `torch_dtype`: `float16`
- `device_map`: `auto`

SAE 配置来自 `config/sae_config.json`：

- SAE repo：`OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x`
- SAE subfolder：`Llama3_1-8B-Base-L19R-8x`
- hook point：`blocks.19.hook_resid_post`
- `d_model`: 4096
- `d_sae`: 32768
- activation：`jumprelu`
- 默认聚合：`max`
- 默认最大长度：`max_seq_len = 128`

### 主实验流程

主入口：

```bash
python run_sae_evaluation.py
```

核心流程：

1. `run_sae_evaluation.py` 解析 CLI 参数。
2. `load_experiment_dataset()` 读取 MISC 或 legacy 数据。
3. `load_local_model_and_tokenizer()` 加载本地 Llama 模型。
4. `load_sae_from_hub()` 加载 OpenMOSS SAE checkpoint。
5. `extract_and_process_streaming()` 抽取 Llama hidden states 并送入 SAE。
6. `_save_feature_store()` 保存样本级 SAE features 和模型 activation。
7. `run_structural_evaluation()` 计算结构指标。
8. `run_functional_evaluation()` 计算 RE/NonRE 功能指标。
9. `run_misc_label_mapping()` / `compute_latent_label_associations()` 计算 MISC latent-label 矩阵。

### Mapping Structure 分析

入口：

```bash
python run_misc_mapping_structure_analysis.py
```

默认读取：

```text
outputs/misc_full_sae_eval/functional/misc_label_mapping
```

默认输出：

```text
outputs/misc_full_sae_eval/interpretability/mapping_structure
```

核心函数：

- `src/nlp_re_base/mapping_structure.py::run_mapping_structure_analysis()`
- `load_mapping_matrix()`
- `compute_label_fragmentation()`
- `compute_label_pair_similarity()`
- `compute_hierarchy_alignment()`
- `write_top20_mapping_structure_report()`

当前正式解释口径：

- `DEFAULT_INTERPRETABILITY_TOP_K = 20`
- 每个核心 MISC 标签取 Top20 latent 作为候选解释空间

### 后续可解释性分析

入口：

```bash
python run_misc_interpretability_analysis.py
```

核心函数：

- `src/nlp_re_base/behavior_interpretability.py::run_followup_interpretability_analysis()`

默认输出：

```text
outputs/misc_full_sae_eval/interpretability/followup_analysis
```

### 因果候选导出

入口：

```bash
python run_misc_causal_candidate_export.py
```

核心函数：

- `src/nlp_re_base/causal_candidates.py::export_misc_causal_candidates()`

默认输出：

```text
outputs/misc_full_sae_eval/interpretability/causal_candidates
```

候选组默认大小：

- `G1`
- `G5`
- `G10`
- `G20`

### 因果验证实验

入口：

```bash
python causal/run_experiment.py
```

云端包装脚本：

```bash
bash deploy/gce/run_causal.sh
```

当前云端脚本默认拒绝误用旧数据：

- `deploy/gce/run_causal.sh` 会拒绝 `data/mi_re`、`data/cactus`、`derived/re_nonre`
- 如需旧数据复跑，需要显式设置 `ALLOW_LEGACY_CAUSAL_DATA=1`

## 评估指标

### 结构指标

主要实现：

- `src/nlp_re_base/eval_structural.py::run_structural_evaluation()`
- `src/nlp_re_base/diagnostics.py::apply_full_structural_metrics()`

主要输出：

- `metrics_structural.json`
- `metrics_ce_kl.json`

核心指标：

- `mse`：重构误差
- `cosine_similarity`：重构向量与原始 hidden state 的余弦相似度
- `ev_openmoss_legacy`：当前优先使用的 OpenMOSS legacy explained variance 字段
- `ev_openmoss_aligned`
- `ev_llamascope_paper`
- `ev_centered_legacy`
- `l0_mean` / `l0_std`：SAE latent 稀疏度
- `dead_ratio`：未激活 latent 比例
- `ce_loss_delta`：用 SAE 重构激活替换原激活后的交叉熵损失变化
- `kl_divergence`：原模型输出分布与 SAE 替换输出分布的 KL 差异

### RE/NonRE 功能指标

主要实现：

- `src/nlp_re_base/eval_functional.py::run_functional_evaluation()`
- `univariate_analysis()`
- `sparse_probe_cv()`
- `dense_probe_cv()`
- `diff_mean_probe()`

主要输出：

- `metrics_functional.json`
- `candidate_latents.csv`

核心指标：

- `cohens_d`：RE 与 NonRE 上 latent 激活差异的效应量
- `auc`：单个 latent 或 probe 对二分类的区分能力
- `p_value`
- `significant_fdr`：Benjamini-Hochberg FDR 校正后的显著性
- sparse probe / dense probe 的 accuracy、F1、AUC

### MISC 多标签映射指标

主要实现：

- `src/nlp_re_base/misc_label_mapping.py::compute_latent_label_associations()`
- `build_label_fragmentation()`
- `build_latent_overlap()`
- `build_topk_jaccard()`

主要输出：

- `functional/misc_label_mapping/latent_label_matrix.csv`
- `functional/misc_label_mapping/label_summary.json`
- `functional/misc_label_mapping/label_fragmentation.json`
- `functional/misc_label_mapping/latent_overlap.json`
- `functional/misc_label_mapping/label_topk_jaccard.json`

核心指标：

- `cohens_d`
- `abs_cohens_d`
- `auc`
- `directional_auc`
- `auc_effect`
- `precision_at_10`
- `precision_at_50`
- `precision_lift_at_10`
- `precision_lift_at_50`
- `significant_fdr`

### Mapping Structure 指标

主要实现：

- `src/nlp_re_base/mapping_structure.py`

主要输出：

- `mapping_structure_metrics.json`
- `label_fragmentation_rank.csv`
- `latent_overlap_distribution.csv`
- `label_pair_similarity.csv`
- `hierarchy_alignment.csv`
- `latent_role_summary.csv`
- `topk_candidate_matrix.csv`
- `mapping_structure_report.md`

核心指标：

- label fragmentation：每个标签关联多少候选 latent
- latent overlap：每个 latent 关联多少标签
- top-k Jaccard：两个标签 TopK latent 集合的重叠比例
- Pearson / Spearman similarity：标签 effect-size 向量相似度
- hierarchy alignment：`RE -> RES/REC`、`QU -> QUO/QUC` 的父子标签结构恢复程度
- latent role taxonomy：exclusive、family-shared、cross-family、global latent

### 因果验证指标

主要实现：

- `causal/run_experiment.py`
- `causal/intervention.py`
- `causal/evaluation.py`
- `causal/selection.py`

主要实验类型：

- necessity：ablation 后目标行为信号是否下降
- sufficiency：steering 后目标行为信号是否上升
- selectivity / side effects：干预是否影响非目标行为或生成质量
- group structure：G1/G5/G10/G20 组合是否优于单个 latent

## 结果产物

主 SAE 输出目录通常为：

```text
outputs/misc_full_sae_eval
```

关键结果文件：

- `dataset_summary.json`
- `records.jsonl`
- `label_matrix.csv`
- `feature_store/utterance_features.pt`
- `feature_store/utterance_activations.pt`
- `feature_store/feature_metadata.json`
- `metrics_structural.json`
- `metrics_ce_kl.json`
- `metrics_functional.json`
- `candidate_latents.csv`
- `functional/misc_label_mapping/latent_label_matrix.csv`
- `interpretability/mapping_structure/mapping_structure_report.md`
- `interpretability/followup_analysis/followup_interpretability_report.md`
- `interpretability/causal_candidates/causal_candidate_report.md`

### 论文表征比较冻结包（2026-07-13）

当前 7 个 leaf 标签（`RES, REC, QUO, QUC, GI, SU, AF`）的权威表征比较入口为：

- `outputs/misc_full_sae_eval/interpretability/representation_comparison_frozen_leaf7_20260713/README.md`
- `outputs/misc_full_sae_eval/interpretability/representation_comparison_frozen_leaf7_20260713/final_metrics_by_label.csv`
- `outputs/misc_full_sae_eval/interpretability/representation_comparison_frozen_leaf7_20260713/final_macro_metrics.csv`
- `outputs/misc_full_sae_eval/interpretability/representation_comparison_frozen_leaf7_20260713/protocol_audit.csv`
- `outputs/misc_full_sae_eval/interpretability/representation_comparison_frozen_leaf7_20260713/figures/representation_comparison_leaf7.png`
- `outputs/misc_full_sae_eval/interpretability/representation_comparison_frozen_leaf7_20260713/SHA256SUMS.txt`

冻结状态为 `FROZEN_WITH_STABLE_CORE_LIMITATION`。Hidden、Full SAE、Top-n SAE、PCA-n 和 Random SAE-n 通过统一折分、分类器、训练折标准化、随机种子与指标审计；Stable Core SAE 使用全分析数据生成的监督式候选清单，没有 outer-fold nested selection，因此只可作为 exploratory 行，不能写成完全无泄漏的 confirmatory 性能结果。旧 9-label macro 只保留作父标签敏感性/附录，不再作为论文主表。

云端 pipeline 结果：

- `deploy/gce/run_full_pipeline.sh` 默认写入 `${PIPELINE_OUTPUT_DIR}`
- `deploy/gce/common.sh` 默认 `PIPELINE_OUTPUT_DIR=${OUTPUT_ROOT}/misc_full_pipeline`
- pipeline 状态文件：`pipeline_status.json`
- pipeline 事件日志：`pipeline_events.jsonl`
- pipeline 主日志：`pipeline.log`

## 运行命令

### 本地正式 MISC SAE 主流程

```powershell
conda activate qwen-env-py311
$env:MODEL_DIR="D:\project\NLP_v3\NLP_data\Llama-3.1-8B"
python run_sae_evaluation.py `
  --model-dir "$env:MODEL_DIR" `
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

### 只跑 Mapping Structure

```powershell
conda activate qwen-env-py311
python run_misc_mapping_structure_analysis.py `
  --mapping-dir outputs/misc_full_sae_eval/functional/misc_label_mapping `
  --output-dir outputs/misc_full_sae_eval/interpretability/mapping_structure `
  --analysis-top-k 20 `
  --top-k 20
```

### 只跑后续可解释性分析

```powershell
conda activate qwen-env-py311
python run_misc_interpretability_analysis.py `
  --eval-dir outputs/misc_full_sae_eval `
  --top-latents-per-label 20
```

### 只导出因果候选

```powershell
conda activate qwen-env-py311
python run_misc_causal_candidate_export.py `
  --eval-dir outputs/misc_full_sae_eval `
  --candidate-top-k 20
```

### 云端全流程

```bash
cp deploy/gce/env.example deploy/gce/.env
nano deploy/gce/.env
bash deploy/gce/bootstrap.sh
bash deploy/gce/download_model.sh
bash deploy/gce/start_full_pipeline_tmux.sh
```

直接前台运行：

```bash
bash deploy/gce/run_full_pipeline.sh
```

只跑解释阶段：

```bash
bash deploy/gce/run_interpretability.sh
```

只跑因果阶段：

```bash
bash deploy/gce/run_causal.sh
```

### 测试命令

```powershell
conda activate qwen-env-py311
python -m unittest test_dataset_loader_smoke.py
python -m unittest test_misc_label_mapping_smoke.py
python -m unittest test_mapping_structure_analysis.py
python -m unittest test_behavior_interpretability.py
python -m unittest test_causal_candidate_export.py
python -m unittest test_causal_smoke.py
python -m unittest test_deploy_smoke.py
python -m unittest test_pipeline_smoke.py
```

## 关键文件索引

### 数据层

- `src/nlp_re_base/data.py`
- `data/mi_quality_counseling_misc/README.md`
- `data/mi_quality_counseling_misc/MANIFEST.json`
- `data/mi_quality_counseling_misc/misc_annotations/`
- `data/mi_quality_counseling_misc/metadata/labels.csv`

### 模型与 SAE

- `config/model_config.json`
- `config/sae_config.json`
- `src/nlp_re_base/model.py`
- `src/nlp_re_base/sae.py`
- `src/nlp_re_base/activations.py`

### 主流程

- `run_sae_evaluation.py`
- `src/nlp_re_base/eval_structural.py`
- `src/nlp_re_base/eval_functional.py`
- `src/nlp_re_base/diagnostics.py`

### MISC 可解释性

- `run_misc_mapping_structure_analysis.py`
- `run_misc_interpretability_analysis.py`
- `run_misc_causal_candidate_export.py`
- `src/nlp_re_base/misc_label_mapping.py`
- `src/nlp_re_base/mapping_structure.py`
- `src/nlp_re_base/behavior_interpretability.py`
- `src/nlp_re_base/causal_candidates.py`

### 因果验证

- `causal/run_experiment.py`
- `causal/data.py`
- `causal/intervention.py`
- `causal/evaluation.py`
- `causal/selection.py`

### AI judge

- `run_ai_re_judge.py`
- `src/nlp_re_base/ai_re_judge.py`
- `src/nlp_re_base/re_judge_rubric.py`

### 部署

- `deploy/gce/env.example`
- `deploy/gce/common.sh`
- `deploy/gce/bootstrap.sh`
- `deploy/gce/download_model.sh`
- `deploy/gce/run_full_eval.sh`
- `deploy/gce/run_interpretability.sh`
- `deploy/gce/run_causal.sh`
- `deploy/gce/run_full_pipeline.sh`
### 当前主线文档

- `doc/云服务器部署运行教程.md`
- `doc/本地实验运行说明.md`
- `doc/MISC研究目标流程与指标设计说明.md`
- `doc/MISC数据集代码对接说明.md`
- `doc/MISC矩阵生成结果评测报告.md`
- `doc/MISC可解释性工作简明汇报.md`
- `doc/MISC可解释性分析效果评估报告.md`
- `doc/MISC Top20可解释性流程复跑科研可接受性评估报告.md`
- `doc/日志.md`

### 历史归档

- `doc/old/`

## 已知问题

1. README 仍偏早期 SAE-RE / RE vs NonRE 项目说明，和当前 MISC 全量主线不完全同步。
   - 文件：`README.md`

2. 文档体系仍有新旧口径混杂。
   - 当前主线在 `doc/MISC*.md`、`doc/云服务器部署运行教程.md`
   - 旧文档已归档到 `doc/old/`

3. 当前正式解释口径是 Top20 candidate space，不应把全量 `latent_label_matrix.csv` 直接写成完整机制证明。
   - 默认值：`src/nlp_re_base/mapping_structure.py::DEFAULT_INTERPRETABILITY_TOP_K`
   - 入口参数：`run_misc_mapping_structure_analysis.py --analysis-top-k`

4. 当前 MISC 映射和 Mapping Structure 属于相关性 / 结构性证据，不是因果证明。
   - 因果候选导出：`run_misc_causal_candidate_export.py`
   - 因果验证另由 `causal/run_experiment.py` 执行

5. `causal/run_experiment.py` 体量很大，并且看起来有历史兼容逻辑与新版逻辑并存的痕迹；重构前需要单独审查。

6. 项目根目录存在明显本地产物或临时文件。
   - 例：`.venv/`、`.uv-cache/`、`.tmp_lm_saes/`、`outputs/`、`dist/`、`0.34`、根目录大型 `torch-2.11.0+cu126-*.whl`

7. `ev_openmoss_legacy` 在 MISC 数据上的数值低于用户预期的官方同分布结果时，需要同时考虑数据域差异、hook 对齐、归一化、token 聚合和官方评估口径。
   - 相关实现：`src/nlp_re_base/diagnostics.py::apply_full_structural_metrics()`
   - 结构指标输出：`outputs/misc_full_sae_eval/metrics_structural.json`

## 不确定问题

1. 当前论文最终主结论是否只采用 Top20 candidate space，还是同时报告全量 FDR 显著 latent-label 边？代码两者都支持。

2. `OTHER` 标签是否应该进入论文主表，还是只作为异质辅助标签保留？当前 `DEFAULT_CORE_LABELS` 不包含 `OTHER`。

3. 因果验证第一版是否只做 `RE`，还是也要对 `QU/QUO` 等高信号标签做对照实验？

4. MISC 输入是否长期只使用单条 `unit_text`，还是后续需要拼接前一轮来访者话语与当前咨询师话语？当前数据层和主流程以 `unit_text` 为单位。

5. 是否还需要恢复一个独立的离线 mapping CLI？当前正式入口已收敛到 `run_sae_evaluation.py` 的集成输出，核心实现仍保留在 `src/nlp_re_base/misc_label_mapping.py`。

6. PAI-EAS 部署已从当前主线清理，后续是否需要恢复为单独分支或外部部署模板仍待决定。

7. 数据目录 `data/mi_quality_counseling_misc` 是否允许长期随仓库分发，尤其是 `raw_transcripts` 可能涉及隐私边界。

## 下一步建议

1. 建立正式文档索引。
   - 建议新增或更新：`doc/文档索引.md`
   - 明确当前主线文档、历史归档文档和结果报告的优先级。

2. 更新根目录 `README.md`。
   - 将项目目标从旧 SAE-RE / RE vs NonRE 改为 MISC 全量 SAE 可解释性主线。
   - 引用 `docs/PROJECT_DOSSIER.md` 和 `doc/MISC可解释性工作简明汇报.md`。

3. 固化正式运行命令。
   - 本地以 `run_sae_evaluation.py --data-dir data/mi_quality_counseling_misc --data-format misc_full --label-mode misc_multilabel` 为准。
   - 云端以 `deploy/gce/run_full_pipeline.sh` 为准。

4. 为论文结果固定一个统计口径。
   - 明确 Top20 candidate space 与 full FDR matrix 的关系。
   - 在 `mapping_structure_report.md` 和老师版汇报中统一表述。

5. 对因果验证做小范围正式复跑。
   - 首选 `RE` 的 `G1/G5/G10/G20`
   - 建议添加 `QU/QUO` 作为高信号对照标签
   - 输出应进入 `outputs/misc_full_sae_eval/interpretability/causal_candidates` 和正式 causal output 目录。

6. 清理或归档根目录临时产物。
   - 对 `.venv/`、`.uv-cache/`、`.tmp_lm_saes/`、`dist/`、根目录 wheel、PDF、`0.34` 做保留/删除决策。
   - 清理前先确认是否有未备份实验产物。

7. 单独审查 `causal/run_experiment.py`。
   - 目标是拆出配置、监控、干预、评估和报告生成逻辑。
   - 优先保持现有测试 `test_causal_smoke.py` 通过。
