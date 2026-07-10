# P3 stable-core 对比式 latent 解释：Claude Code 执行工作流

状态：`canonical`

更新时间：2026-07-10

适用仓库：`D:/project/NLP_re_dataset_model_base`

## 1. 文档用途

本文档是 P3 stable-core 对比式 SAE latent 解释的唯一执行规范。Claude Code 中的模型应按本文档修复、实现、测试和运行本地流程，不得自行从旧 P3 文档拼接步骤。

本轮目标不是尽快生成一批解释，而是建立一条可审计的证据链：

```text
stable_core latent
-> 无文本重复泄漏的对比证据包
-> label-blind LLM 候选解释
-> 不暴露答案的 held-out activation Scorer
-> 受控 minimal-pair SAE 激活测试
-> 仅基于通过验证的 distinct latent 做子概念聚合
-> probe-space ablation
-> 阶段状态真实、结论边界明确的最终报告
```

## 2. 角色边界

### 2.1 Claude Code 模型

Claude Code 是工程实施者，负责：

- 阅读本文件和 `CLAUDE.md`；
- 审计并修复 `src/nlp_re_base/contrastive_*.py`；
- 编写和运行 smoke tests；
- 生成 LLM task queue；
- 校验 Claude Code 模型生成的 LLM raw outputs；
- 计算指标、minimal-pair 激活和 ablation；
- 生成 manifest、质量审计表和最终报告。

Claude Code 不得：

- 用规则脚本代替 LLM 解释或评分；
- 为了让 schema 通过而合成 raw outputs；
- 把一个任务的模型输出伪装成另一个任务或执行者的输出；
- 在模型 raw output 尚未到位时伪造完成状态；
- 绕过任务 JSONL 和 raw JSON 审计边界直接生成最终指标。

### 2.2 LLM 评估者

Claude Code 中的模型直接承担 Explainer、Scorer、label baseline、minimal-pair Designer 和子概念聚类。任务 JSONL 是模型输入合同，`expected_output_path` 是原始输出合同。

执行时必须：

1. 在 manifest 中记录真实 model display name、Claude Code 版本和执行时间；
2. 每个任务只读取该行 prompt 与必要的系统规范，不读取 answer key；
3. Explainer、Scorer 与 label baseline 使用相互隔离的上下文；
4. 直接写 raw JSON，不用正则、固定词表、硬编码概率或 fallback 代替模型推理。

## 3. 固定研究范围

### 3.1 Latent 输入

唯一候选来源：

`outputs/cross_val/stable_topk_selection/stable_topk_latent_set.csv`

固定筛选条件：

`stable_set_role == "stable_core"`

预期规模：

- 303 条 label-latent 关系；
- 225 个 unique latent id；
- AF 27、GI 26、QU 26、QUC 45、QUO 32、RE 54、REC 47、RES 15、SU 31。

禁止改用：

- Top20 positive Cohen's d；
- minimal sufficient subspace；
- full 32768 latent pool；
- `p3_feature_cards` 或 `p3_feature_cards_stable_core` 的旧卡片列表。

### 3.2 数据和特征

```text
records:
  outputs/misc_full_sae_eval/records.jsonl
label matrix:
  outputs/misc_full_sae_eval/label_matrix.csv
SAE utterance features:
  outputs/misc_full_sae_eval/feature_store/utterance_features.pt
raw hidden activations:
  outputs/misc_full_sae_eval/feature_store/utterance_activations.pt
```

固定模型口径：

- Llama-3.1-8B；
- hook：`blocks.19.hook_resid_post`；
- SAE 维度：32768；
- utterance 聚合：max pooling；
- 当前没有前一句 client utterance。

RES、REC、RE 的上下文功能解释必须降级为候选解释。

### 3.3 环境

所有命令必须使用：

```powershell
conda run -n qwen-env-py311 python <script>
```

测试是独立脚本，不使用 pytest。

## 4. 输出合同

本轮重新生成的唯一输出根目录：

```text
outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp/
```

预期结构：

```text
contrastive_latent_interp/
|-- evidence_packs/
|   |-- contrastive_evidence_packs.jsonl
|   `-- contrastive_evidence_pack_summary.csv
|-- llm_tasks/
|   |-- explainer_tasks.jsonl
|   |-- scorer_tasks.jsonl
|   |-- baseline_tasks.jsonl
|   |-- minimal_pair_designer_tasks.jsonl
|   `-- subconcept_cluster_tasks.jsonl
|-- explainer_outputs/
|   |-- raw/
|   |-- validated_explanations.jsonl
|   |-- consistency_by_latent.csv
|   |-- validation_errors.csv
|   `-- retry_tasks.jsonl
|-- scorer_outputs/
|   |-- raw_explanation_scorer/
|   |-- raw_label_baseline/
|   |-- heldout_answer_key.csv
|   |-- scorer_metrics.csv
|   |-- baseline_metrics.csv
|   |-- latent_level_status.csv
|   `-- validation_errors.csv
|-- minimal_pairs/
|   |-- raw_designer_outputs/
|   |-- validated_minimal_pairs.jsonl
|   |-- minimal_pair_quality_audit.csv
|   `-- minimal_pair_results.csv
|-- subconcepts/
|   |-- raw_cluster_outputs/
|   |-- subconcept_cluster_input.csv
|   `-- subconcept_table.csv
|-- ablation/
|   |-- set_ablation_fold_metrics.csv
|   `-- set_ablation.csv
|-- llm_execution_manifest.jsonl
|-- manifest.json
`-- contrastive_latent_interp_report.md
```

raw output 只能来自任务中声明的 LLM 执行者。任何本地 fallback 必须写到独立的 `diagnostics/`，不得进入 `raw/` 或最终指标。

## 5. 已修复缺陷与防回归约束

以下 8 类缺陷已于 2026-07-10 修复。后续不得删除对应 smoke 断言；任何一项回归都必须在重跑真实数据前停止：

1. `contrastive_evidence_pack.py`：不同 held-out tag 不能重复生成相同 `sample_id`。
2. `contrastive_evidence_pack.py`：除 row index 不重叠外，还要保证规范化文本不重复、不跨 evidence/held-out 泄漏。
3. `contrastive_scorer.py`：Scorer 和 baseline prompt 不能暴露 tag、activation、internal source、target label、ground truth 或 row index。
4. `contrastive_llm_io.py`：预测校验必须使用 `(task_id, unique_sample_id)` 一一匹配；禁止 `set` 掩盖重复 ID，禁止用同名 ID 的第一行代替其余样本。
5. `contrastive_scorer.py`：单类别或样本数不足的任务必须标为 invalid，不能把 AUROC 强制写成 0.5。
6. `contrastive_minimal_pairs.py`：计划集合固定为主层级 6 个标签各 3 个代表 latent，共 18 个；不能因 Scorer 只接受一个 latent 就把任务数缩成 1。
7. `contrastive_subconcepts.py`：输入必须是一行一个 distinct latent，不能把两次重复 explanation 都送入聚类；主结果只能使用通过 latent-level 门禁的解释。
8. `run_misc_contrastive_latent_interp.py`：报告必须根据实际文件和 manifest 标注 `not_started / blocked / partial / complete`，缺少 Step 4 时不能声称提供 minimal-pair 证据。

## 6. Phase 0：预检和污染防护

Claude Code 首先运行只读检查：

```powershell
git status --short
Test-Path outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp
```

必须确认：

- 无效结果目录已删除；
- 仓库根目录没有 `run_contrastive_batch_explainer.py`、`run_contrastive_batch_scorer.py` 或 `run_batch_*_scorer.py`；
- `scratch/` 中没有本轮硬编码 Scorer 输出脚本；
- stable-core 输入、feature store、records 和 label matrix 存在；
- 旧 `p3_feature_cards*` 不作为输入。

若发现任何脚本会直接写入 `explainer_outputs/raw`、`scorer_outputs/raw_*` 或 `subconcepts/raw_cluster_outputs`，必须停止并报告。

## 7. Phase 1：修复实现并建立防回归测试

至少新增或扩展以下测试：

### T1：唯一 held-out ID

- 每个 task 的 sample ID 全部唯一；
- 20 个样本应有 20 个不同 ID；
- ID 使用中性序号或随机 token，不包含 `active`、`nonactive`、`high`、`mid`、`label_match` 等答案信息。

### T2：文本去重和隔离

规范化规则至少包括：lowercase、去首尾空白、合并空白、去非语义标点。

对每个 latent 验证：

- evidence 内无重复规范化文本；
- held-out 内无重复规范化文本；
- evidence 与 held-out 的规范化文本交集为空；
- 同一文本即使来自不同 row，也不能跨集合出现。

### T3：Scorer prompt 无答案泄漏

Scorer 和 baseline prompt 只能包含：

```text
sample_id
text
explanation（仅 Scorer）或 label definition（仅 baseline）
输出 schema
```

测试必须断言 prompt 不含：

```text
ACTIVE_HIGH
ACTIVE_MID
ACTIVE_LOW
NONACTIVE
activation
ground_truth
target_label
internal_tag
```

### T4：严格预测映射

- 重复 sample ID 必须报错；
- 缺失或多余 ID 必须报错；
- 每个 task 必须恰好一条预测对应一个 expected ID；
- 不能通过 `.iloc[0]` 消解重复答案键。

### T5：指标可计算性

- 合格 Scorer task 必须同时包含正负类；
- 标准目标为 10 正例、10 反例；
- 样本不足或单类别时写入 exclusion audit，不生成 AUROC；
- metrics 中不得用 0.5 伪装 invalid task。

### T6：下游输入门禁

- minimal-pair 计划表固定包含 18 个代表 latent；
- 子概念输入按 `(target_label, latent_idx)` 去重；
- 未通过 latent-level 验证的解释不能进入主子概念表；
- 报告不得为缺失阶段生成完成式结论。

验证命令：

```powershell
conda run -n qwen-env-py311 python -m py_compile run_misc_contrastive_latent_interp.py src/nlp_re_base/contrastive_evidence_pack.py src/nlp_re_base/contrastive_explainer.py src/nlp_re_base/contrastive_llm_io.py src/nlp_re_base/contrastive_scorer.py src/nlp_re_base/contrastive_minimal_pairs.py src/nlp_re_base/contrastive_subconcepts.py src/nlp_re_base/contrastive_ablation.py
conda run -n qwen-env-py311 python test_contrastive_evidence_pack_smoke.py
conda run -n qwen-env-py311 python test_contrastive_llm_io_smoke.py
conda run -n qwen-env-py311 python test_contrastive_workflow_integrity_smoke.py
```

阶段门禁 G1：所有测试通过后才能构建真实 evidence packs。

## 8. Step 1：构建对比证据包

命令：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step build-packs --output-dir outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp
```

目标采样结构：

### Explainer 可见证据

- ACTIVE_HIGH：5；
- ACTIVE_MID：4；
- ACTIVE_LOW：2；
- NONACTIVE_NEAR_MISS：7；
- NONACTIVE_RANDOM：2。

### Scorer held-out

- 内部正例：5 high + 5 mid；
- 内部反例：5 near miss + 5 target-label-positive-but-nonactive；
- 对 Scorer 序列化时全部删除 tag 和 activation；
- 20 条文本随机打乱，仅保留中性 sample ID 和 text。

不要为了凑数复制文本。去重后不足的 pack 应记录：

```text
interpretability_eligible
scorer_eligible
scarcity_reason
actual_evidence_count
actual_heldout_positive_count
actual_heldout_negative_count
```

阶段门禁 G2：

- 正好 303 个 stable-core pack；
- stable role 全部为 `stable_core`；
- 所有 pack 的 row 集不相交且规范化文本不相交；
- summary 报告各标签 eligible/excluded 数量；
- 任一重复或泄漏计数必须为 0。

## 9. Step 2：生成并执行 Explainer 任务

生成任务：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step make-explainer-tasks --output-dir outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp
```

每个 eligible latent 生成两条独立任务。两条任务可以使用不同样本顺序和轻微不同的提示措辞，但不得互相读取输出。

Explainer 可见：

- anonymous latent id；
- ACTIVE/NEAR-MISS tag；
- activation；
- utterance text。

Explainer 不可见：

- target MISC label；
- target_match；
- active_labels；
- predicted_code / predicted_subcode；
- `NONACTIVE_LABEL_MATCH` 内部来源；
- association metrics 和 label definition。

输出必须包含：

```text
latent_idx
short_name
main_hypothesis
positive_triggers
explicit_exclusions
possible_surface_confounds
feature_type
confidence
alternative_hypotheses
key_evidence
failure_modes
```

Claude Code LLM 执行要求：

1. 在独立 Claude Code 模型上下文中打开 task batch；
2. 逐任务输出 JSON，不生成 Python 脚本；
3. 原样保存到 task 的 `expected_output_path`；
4. 在 `llm_execution_manifest.jsonl` 记录 task id、实际 model display name、执行时间、prompt SHA256、raw output SHA256；
5. 解析失败时保留原始文件并进入 retry，不人工改写内容。

校验：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step validate-explainer --output-dir outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp
```

阶段门禁 G3：

- 每个 eligible latent 至少一条 schema-valid 解释；
- 无 MISC label metadata 泄漏；
- confidence 在 0 至 1；
- key evidence 均来自任务样本；
- 报告两次解释的语义一致性，而不是只报告 schema；
- 若 95% 以上 latent 的两次所有语义字段完全相同，停止并审查是否复制、缓存或规则生成。

在进入 Step 3 前，Claude Code 必须做一次分层抽样审计：每个标签选最高 confidence 和最低 confidence 各 1 个，共 18 个 latent，输出 `explainer_outputs/quality_sample_review.md`。审计必须区分 surface form、counseling function、context relation、artifact 和 unclear。

## 10. Step 3：Held-out activation Scorer 和 label baseline

生成任务：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step make-scorer-tasks --output-dir outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp
```

Scorer 使用与 Explainer 隔离的新上下文。Scorer 只能看到 explanation JSON、20 个中性 sample ID 和文本。baseline 只能看到 label definition、同一组中性 sample ID 和文本。

任何 task 中出现激活 tag 或 activation 数值都属于阻断错误。

执行后分别保存到：

```text
scorer_outputs/raw_explanation_scorer/{task_id}.json
scorer_outputs/raw_label_baseline/{task_id}.json
```

校验和计算：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step validate-scorer --output-dir outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp
```

task-level 状态：

- `accepted`：AUROC >= 0.70 且 latent_gap > 0；
- `ambiguous`：0.60 <= AUROC < 0.70 且 latent_gap > 0；
- `no_latent_contribution`：latent_gap <= 0；
- `rejected`：AUROC < 0.60 且 latent_gap > 0；
- `invalid`：样本、ID、schema 或类别分布不满足计算条件。

latent-level 状态使用两次解释合并：

- `accepted_stable`：两次解释均 accepted；
- `accepted_single`：只有一次 accepted；
- `ambiguous`：没有 accepted，但至少一次 ambiguous；
- `rejected`：其余可计算情况；
- `invalid`：任一必要任务无法形成可信指标。

阶段门禁 G4：

- 所有参与指标的 task 都是 20 个 unique ID、10 正例、10 反例；
- validation error 和 retry 数必须在继续前清零，或将对应 latent 明确排除；
- 报告 task 数和 distinct latent 数，不能把同一 latent 的两次解释写成两个 accepted latent；
- 输出 `latent_level_status.csv`。

## 11. Step 4：Minimal-pair 生成、质量审计和 SAE 激活

固定计划集合：主层级 `QU QUO QUC RE REC AF`，每标签按 inclusion frequency 取前 3 个，共 18 个 label-latent 对。

生成任务：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step make-minimal-pair-tasks --output-dir outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp
```

每个 latent 生成 5 对，共计划 90 对。每对包含：

```text
positive_text
negative_text
changed_factor
held_constant
expected_direction = positive_greater_than_negative
```

质量要求：

- 只改变一个触发因素；
- 主题、说话者角色、咨询场景、语气和近似长度保持一致；
- 不得同时把开放问题改成封闭问题并更换主题或情绪；
- 不复制 evidence pack 原句；
- positive 与 negative 都必须是自然、完整的 counselor utterance；
- 若解释只是标点、代词或固定词，应明确标为 surface minimal pair，不能包装成功能验证。

校验：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step validate-minimal-pairs --output-dir outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp
```

Claude Code 必须额外抽样审查每个 latent 至少 2 对，共至少 36 对，写入 `minimal_pair_quality_audit.csv`，字段包括：

```text
latent_idx
pair_id
single_factor_pass
topic_held_pass
style_held_pass
length_reasonable_pass
naturalness_pass
surface_or_function
review_note
```

任何 latent 少于 3 对合格 pair 时，该 latent 不运行激活测试。

GPU 激活测试：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step run-minimal-pairs --output-dir outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp --device cuda --batch-size 4
```

阶段门禁 G5：

- 计划表包含 18 个代表 latent；
- 每个进入测试的 latent 至少 3 对质量合格 pair；
- 结果包含 positive activation、negative activation、gap、mean gap 和 pass rate；
- `minimal_pair_confirmed` 至少要求 mean gap > 0 且 pass rate >= 0.60；
- 该结果仅支持该解释所描述触发因素的激活敏感性，不证明 MISC 概念或因果机制。

## 12. Step 5：子概念聚合

主聚类输入只允许：

- 一行一个 distinct `(target_label, latent_idx)`；
- latent-level status 为 `accepted_stable`；
- 使用两次解释的共识版本；
- 附带 Scorer 和 minimal-pair 状态，但不重复计权。

若某标签少于 3 个 `accepted_stable` latent，不强行聚类，标记 `insufficient_validated_latents`。

生成任务：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step make-subconcept-tasks --output-dir outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp
```

聚类约束：

- 每标签最多 3 至 6 个候选子概念；
- 每个 latent 只能归入一个主子概念，可另列 secondary relation；
- singleton 只能标为 `tentative_singleton`，不能伪装成稳定概念簇；
- SU、GI、RES 的聚类统一标 `tentative`；
- RE、REC、RES 必须重复声明缺少 client context。

构建表：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step build-subconcept-table --output-dir outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp
```

阶段门禁 G6：

- 输入无重复 latent；
- 所有 representative latent 都来自输入表；
- 每标签聚类数满足约束，或明确标记证据不足；
- 表中同时显示 Scorer 和 minimal-pair 支持状态；
- 不得把 `no_latent_contribution` 的 latent 写入主子概念结论。

## 13. Step 6：Probe-space ablation

运行：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step ablation --output-dir outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp
```

必须记录：

- 5-fold stratified-group-kfold；
- group 为 `file_id`；
- 每标签 stable-core 数；
- target drop、random baseline drop、drop vs random、non-target preservation；
- union latent 数与 label-latent 数的区别。

唯一允许的结论是：stable-core 集合对 probe-space 标签预测存在选择性依赖。不得写成 latent 概念解释得到验证，也不得写成因果机制证明。

## 14. 最终报告

运行：

```powershell
conda run -n qwen-env-py311 python run_misc_contrastive_latent_interp.py --step report --output-dir outputs/misc_full_sae_eval/interpretability/contrastive_latent_interp
```

报告必须首先给出阶段状态表：

| 阶段 | planned | completed | valid | excluded/errors | status |
|---|---:|---:|---:|---:|---|
| Evidence pack | | | | | |
| Explainer | | | | | |
| Scorer | | | | | |
| Minimal pair | | | | | |
| Subconcept | | | | | |
| Ablation | | | | | |

报告必须回答：

1. 哪些 distinct latent 得到 `accepted_stable` 解释；
2. 哪些解释主要是 surface cue、artifact 或 label prior；
3. 哪些解释得到 minimal-pair 激活支持；
4. 哪些标签证据不足，不能形成子概念；
5. stable-core set 的 probe-space ablation 结果；
6. RES/REC/RE 的上下文限制；
7. 实际 LLM provider/model 和执行方式；
8. 所有排除、失败和重试任务。

若某阶段未完成，报告只能写 `not run`、`partial` 或 `blocked`，不得用完成式摘要替代缺失结果。

## 15. 停止规则

出现以下任一情况立即停止当前阶段：

- 发现规则脚本或硬编码概率写入 raw output；
- Scorer prompt 暴露 tag、activation 或 ground truth；
- sample ID 重复；
- evidence 与 held-out 存在规范化文本重叠；
- AUROC 输入只有一个类别；
- validation error 未清零却继续生成下游任务；
- minimal-pair 任务数不是计划的 18，且没有明确 exclusion ledger；
- 子概念输入包含重复解释或未验证 latent；
- 最终报告声称存在实际缺失的证据。

停止后输出：问题、受影响文件、受影响任务数、修复建议和是否需要重跑上游。不要静默 fallback。

## 16. Claude Code 最终交付清单

Claude Code 只有在以下项目全部满足后才能报告完成：

- [ ] 已修复第 5 节列出的代码缺陷；
- [ ] 语法检查和所有 smoke tests 通过；
- [ ] evidence pack 无 ID、row 或文本泄漏；
- [ ] Claude Code LLM provenance 完整；
- [ ] Explainer 分层质量抽样已生成；
- [ ] Scorer 无 tag 泄漏且指标输入为双类别；
- [ ] task-level 和 distinct latent-level 状态分开；
- [ ] minimal-pair 计划、质量审计和 SAE 激活结果齐全；
- [ ] 子概念只使用通过门禁的 distinct latent；
- [ ] ablation 结论保持在 probe-space dependency；
- [ ] 最终报告准确反映 partial/complete 状态；
- [ ] 未主动 commit，除非用户明确要求。

## 17. 可直接交给 Claude Code 的起始指令

```text
请阅读 CLAUDE.md 和 doc/P3_stable_core对比式latent解释_Claude_Code执行工作流.md。
把后者视为本任务唯一执行规范，不要执行其他 P3 文档中的旧流程。

先完成 Phase 0 和 Phase 1：审计现有 contrastive_* 实现，修复文档列出的 ID 重复、文本泄漏、Scorer tag 泄漏、错误答案映射、单类别 AUROC、minimal-pair 任务缩减、子概念重复输入和报告状态错误，并新增防回归 smoke test。

不要用规则、正则、固定词表、硬编码概率或 fallback 生成任何 LLM raw output。Claude Code 同时负责本地代码和 LLM 评估：到达模型阶段时，逐条读取任务 JSONL，直接将模型原始 JSON 写到 `expected_output_path`，再运行校验；失败项只按 retry task 重做。

每通过一个阶段门禁再进入下一阶段。任何门禁失败都停止并报告，不得伪造完成状态。所有 run_*.py 和 test_*.py 使用 qwen-env-py311。不要 commit。
```
