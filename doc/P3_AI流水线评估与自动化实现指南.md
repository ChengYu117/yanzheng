# P3 AI 流水线评估与自动化实现指南

本文档用于把当前项目中已经生成的 P3 SAE feature card 材料交接给另一个 AI 或自动化代理继续实现。重点不是重新解释研究动机，而是说明：现有什么、prompt 在哪里、如何调用、还缺哪些流水线、每一步应该读什么文件、写什么文件、如何验收。

## 1. 当前项目上下文

仓库路径：

```text
D:/project/NLP_re_dataset_model_base
```

研究问题：

```text
LLM 内部 SAE features 是否捕捉到了 MISC 咨询师行为标签相关的可解释输入模式？
这些模式是咨询行为功能、MI 原则、上下文关系，还是问号、固定句式、长度、数据模板等表面 artifact？
```

当前 P3 的位置：

```text
P2: top-latent / representation probe 已筛出候选 SAE latents
P3: 自动解释 + 评估 + 人工审核 feature cards
P4: feature clustering
P5: causal ablation / steering
```

当前已经存在三类 P3 材料：

```text
1. dry-run feature card 材料：
   Top20 positive Cohen's d SAE latents
   → MaxAct / contrast / low-activation examples
   → input-centric explanation prompts
   → scoring task sets
   → feature card packets

2. IDE AI agent 下游草稿产物：
   input-centric explanations
   → scoring predictions / metrics
   → MI coder manual review template
   → draft final feature cards

3. 分批执行脚本：
   run_misc_p3_agent_orchestrator.py
   用于把 Task A/B 拆成小批次，让当前 AI 持续读取、生成、合并和校验。
```

当前仍未闭环或需要后续人工/独立脚本完成：

```text
1. output-centric feature stimulation 与输出侧解释
2. MI coder 真实人工审核，而不仅是 review template / draft card
3. concept cluster / label-level 总结报告的最终可发表版本
4. 如果需要重跑或审计 Task A-D，应通过 orchestrator 分批执行，而不是一次性让 AI 读取全部 prompt / scoring examples
```

重要边界：

```text
第一阶段只有 counselor current utterance，没有前一句 client utterance。
RES / REC / RE 等依赖上下文的解释不能强称为“复述/改写了来访者内容”。
top activating utterances 只能作为候选解释证据，不是语义证明，也不是因果机制证明。
```

## 2. 当前已生成的 P3 产物

主输出目录：

```text
outputs/misc_full_sae_eval/interpretability/p3_feature_cards
```

核心文件：

| 文件 | 作用 |
|---|---|
| `manifest.json` | 本次 P3 dry-run 的总清单和数量 |
| `p3_feature_card_packets.jsonl` | 每个 latent 一条完整材料包 |
| `p3_feature_card_examples.csv` | 所有样例展开成 CSV，便于筛选和人工查看 |
| `p3_feature_card_summary.csv` | 每个 latent 一行摘要 |
| `p3_input_explanation_prompts/` | 180 个 input-centric explanation prompt |
| `p3_scoring_tasks.jsonl` | 360 个待评分任务 |
| `p3_feature_cards_dryrun.md` | dry-run 总览报告 |

分批执行与校验脚本：

| 文件 | 作用 |
|---|---|
| `run_misc_p3_agent_orchestrator.py` | 当前 AI agent 的批次管理入口，负责准备小批次、合并输出、计算指标、生成 Task C/D 草稿、写入进度快照 |
| `agent_batches/` | 每次 `prepare` 或 `next-batch` 生成的批次输入与说明文件 |
| `p3_agent_batch_progress.json` | 当前 Task A-D 的进度快照，由 orchestrator 自动更新 |
| `p3_agent_validation.json` | 全量产物校验结果，由 `validate` 生成 |

已生成的 IDE AI agent 草稿产物：

| 目录/文件 | 当前作用 |
|---|---|
| `ai_reviews/p3_input_explanations.jsonl` | Task A 的 input-centric explanation 汇总 |
| `ai_scoring/p3_scoring_predictions.jsonl` | Task B 的逐 example prediction |
| `ai_scoring/p3_scoring_metrics_by_latent.csv` | Task B 的 latent x task_type 级指标 |
| `ai_scoring/p3_scoring_metrics_by_label.csv` | Task B 的标签级指标 |
| `manual_review/p3_mi_coder_review_template.csv` | Task C 的人工审核模板 |
| `final_cards/p3_final_feature_cards.jsonl` | Task D 的 draft final cards |
| `final_cards/p3_final_feature_cards.md` | Task D 的 Markdown 草稿报告 |

当前数量：

```text
feature cards: 180
labels: RE, RES, REC, QU, QUO, QUC, GI, SU, AF
cards per label: 20
example rows: 12400
input prompts: 180
scoring tasks: 360
missing packets: 0
fallback packets: 17
```

当前 Task A-D 草稿产物的校验状态：

```text
Task A explanations: 180 / 180, parse_status=ok 180
Task B scoring predictions: 5280 / 5280
Task B metrics_by_latent: 360 rows
Task B metrics_by_label: 9 rows
Task C manual review template: 180 rows
Task D final cards: 180 rows, Markdown exists
Full validation: PASS, checked by run_misc_p3_agent_orchestrator.py validate on 2026-06-29
```

17 个 fallback packet 全部来自 `RES rank 4-20`。原因是第一阶段 evidence packet 与当前 top20 Cohen's d latent 表不同步。当前 P3 输出以最新 top20 latent 表为准，缺失的 RES packet 已从 `utterance_features.pt + label_matrix.csv + records.jsonl` 现场重建。

## 3. 如何重新生成当前 dry-run 材料

脚本：

```text
run_misc_p3_feature_cards.py
```

默认输入：

```text
outputs/misc_full_sae_eval/interpretability/top20_cohensd_latent_utterances/top20_cohensd_latents_by_label.csv
outputs/misc_full_sae_eval/interpretability/top20_cohensd_latent_utterances/latent_evidence_packets/latent_evidence_packets_labeled.jsonl
outputs/misc_full_sae_eval/feature_store/utterance_features.pt
outputs/misc_full_sae_eval/label_matrix.csv
outputs/misc_full_sae_eval/records.jsonl
outputs/layer_selection_strategy/llama_layer_selection.csv
```

默认输出：

```text
outputs/misc_full_sae_eval/interpretability/p3_feature_cards
```

运行命令：

```powershell
python run_misc_p3_feature_cards.py
```

测试命令：

```powershell
python -m py_compile run_misc_p3_feature_cards.py test_misc_p3_feature_cards_smoke.py
python test_misc_p3_feature_cards_smoke.py
```

当前脚本不会调用 LLM，不读取 API key，不执行 output intervention。它只生成 prompt 和待评分任务。

启动或继续 P3 的统一入口：

```powershell
python run_misc_p3_agent_orchestrator.py 启动p3 --batch-size 3
```

该入口不会调用外部模型。它只准备下一批 agent batch、合并当前 agent 已写回的 output、计算指标和执行后续纯 Python 步骤。解释与评分由当前对话模型读取 batch input 后完成。

## 4. Feature card packet 的结构

读取：

```text
outputs/misc_full_sae_eval/interpretability/p3_feature_cards/p3_feature_card_packets.jsonl
```

每行一个 latent card。关键字段：

```json
{
  "p3_version": "misc_p3_feature_cards_v1",
  "packet_id": "packet_0001",
  "target_label": "RE",
  "latent_idx": 29759,
  "rank_within_label": 1,
  "dry_run": true,
  "source_packet_status": "phase1_packet",
  "context_limitation": "Only counselor current utterance is available...",
  "association_metrics": {
    "cohens_d": 0.847,
    "directional_auc": 0.694,
    "precision_at_50": 0.54
  },
  "layer_metadata": {
    "sae_canonical_layer": 19,
    "sae_hook_point": "blocks.19.hook_resid_post",
    "llama_label_specific_best_layer": 6,
    "llama_early_stable_layer": 5
  },
  "dry_run_explanation_slots": {
    "input_centric_explanation": "pending",
    "artifact_risk": "pending",
    "mi_coder_judgment": "pending",
    "final_status": "pending"
  },
  "scoring_slots": {
    "activation_prediction_score": "pending_score",
    "code_discrimination_score": "pending_score"
  },
  "examples": []
}
```

`examples` 中的 `example_group` 包括：

| group | 含义 |
|---|---|
| `top_activating` | 该 latent 激活最高的样例 |
| `high_non_target` | 非目标标签中激活较高的样例 |
| `random_target` | 目标标签正例中的随机样例 |
| `sibling_code_contrast` | 相邻/易混标签负例，例如 QUO vs QUC, RES vs REC |
| `surface_matched_contrast` | 表面形式相似但目标标签为负的样例 |
| `low_activation_scoring` | 低激活样例，用于 activation prediction scoring |

## 5. Prompt 在哪里，怎么使用

prompt 目录：

```text
outputs/misc_full_sae_eval/interpretability/p3_feature_cards/p3_input_explanation_prompts
```

每个 prompt 文件命名格式：

```text
{LABEL}_rank{RANK}_latent{LATENT_ID}_input_prompt.json
```

例如：

```text
AF_rank01_latent23464_input_prompt.json
RE_rank01_latent29759_input_prompt.json
RES_rank04_latent29759_input_prompt.json
```

每个 prompt 文件结构：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "..."
    },
    {
      "role": "user",
      "content": "{...JSON string...}"
    }
  ]
}
```

使用方式（IDE AI agent 模式）：

1. IDE AI agent 读取 prompt JSON 文件。
2. 提取 `messages` 中的 `system` 和 `user` 内容，作为自身的分析任务。
3. 按照 prompt 要求生成结构化 JSON 响应。
4. 将结果写入对应的 review 文件。

执行方式：不再需要独立 Python 脚本调用外部 API。由用户在 IDE 中向 AI agent 发出评估指令，agent 直接读取 prompt 文件、分析内容、生成解释、写入结果。

输入路径：

```text
prompts: outputs/misc_full_sae_eval/interpretability/p3_feature_cards/p3_input_explanation_prompts/*.json
cards:   outputs/misc_full_sae_eval/interpretability/p3_feature_cards/p3_feature_card_packets.jsonl
```

输出路径：

```text
outputs/misc_full_sae_eval/interpretability/p3_feature_cards/ai_reviews
```

输出文件：

```text
p3_input_explanations.jsonl
p3_input_explanations.csv
raw_responses/
failed_responses.jsonl
manifest.json
```

每条 `p3_input_explanations.jsonl` 建议结构：

```json
{
  "packet_id": "packet_0001",
  "target_label": "RE",
  "latent_idx": 29759,
  "rank_within_label": 1,
  "model": "MODEL_NAME",
  "prompt_path": "...",
  "one_sentence_tentative_interpretation": "...",
  "main_patterns": [
    {
      "pattern_name": "...",
      "pattern_type": "surface_form",
      "evidence": "..."
    }
  ],
  "relationship_to_target_label": "...",
  "artifact_risk": "low|medium|high|unclear",
  "evidence_quality": "high|medium|low|uninterpretable",
  "candidate_feature_name": "...",
  "alternative_explanations": ["...", "..."],
  "recommended_followup_checks": ["..."],
  "final_concise_conclusion": "...",
  "parse_status": "ok"
}
```

必须保留原始响应：

```text
raw_responses/{LABEL}_rank{RANK}_latent{LATENT_ID}.json
```

失败时不要中断全量流程。将失败记录到：

```text
failed_responses.jsonl
```

## 6. Input-centric explanation 的输出要求

LLM 不允许直接写：

```text
this feature is QUO
this latent proves the model understands REC
this is a causal mechanism
```

推荐措辞：

```text
This latent appears associated with ...
It may capture ...
A candidate explanation is ...
The evidence is mostly surface-form / dialogue-functional / artifact-like ...
No stable interpretation can be assigned ...
```

解释维度必须区分：

| 类型 | 说明 |
|---|---|
| `surface_form` | 问号、what/how、固定短语、长度、标点、句法模板 |
| `dialogue_function` | 邀请展开、提供信息、反射、肯定、建议 |
| `context_relation` | 是否复述或改写前一句 client 内容；当前阶段因为缺少 client context，必须谨慎 |
| `mi_principle` | 自主支持、合作、非评判、指导性、对抗性 |
| `artifact` | 转录格式、模板、数据来源、ASR 错误、重复话术 |
| `mixed_unclear` | 多模式混合或证据不足 |

对 `RES`, `REC`, `RE` 的特殊限制：

```text
当前 packet 没有前一句 client utterance。
不能仅凭 counselor utterance 判断它是否真正反射了 client 内容。
只能说“utterance 内部形式类似 reflective phrasing”或“可能与反射性句式相关”。
```

## 7. Scoring task 在哪里，怎么使用

待评分任务：

```text
outputs/misc_full_sae_eval/interpretability/p3_feature_cards/p3_scoring_tasks.jsonl
```

当前数量：

```text
360 tasks = 180 latents × 2 task types
```

任务类型：

| task_type | 目标 |
|---|---|
| `activation_prediction_task` | 给定 explanation，判断 held-out 句子是否应该高激活 |
| `code_discrimination_task` | 给定 explanation，判断句子是否符合目标 MISC label，而不是 sibling/surface negatives |

每条 task 结构：

```json
{
  "packet_id": "packet_0001",
  "target_label": "RE",
  "latent_idx": 29759,
  "task_type": "activation_prediction_task",
  "status": "pending_score",
  "instructions": "...",
  "examples": [
    {
      "row_idx": 3397,
      "group": "top_activating",
      "activation": 3.53125,
      "active_labels": "RE,REC",
      "text": "...",
      "expected_high_activation": 1
    }
  ]
}
```

## 8. 如何实现 input-centric scoring

执行方式（IDE AI agent 模式）：由 IDE AI agent 直接读取 explanations 和 scoring tasks，对每个 task 中的 examples 逐条判断，写入预测结果并计算汇总指标。不需要独立脚本或 API 调用。

输入路径：

```text
explanations: outputs/misc_full_sae_eval/interpretability/p3_feature_cards/ai_reviews/p3_input_explanations.jsonl
tasks:        outputs/misc_full_sae_eval/interpretability/p3_feature_cards/p3_scoring_tasks.jsonl
```

输出路径：

```text
outputs/misc_full_sae_eval/interpretability/p3_feature_cards/ai_scoring
```

输出文件：

```text
p3_scoring_predictions.jsonl
p3_scoring_metrics_by_latent.csv
p3_scoring_metrics_by_label.csv
p3_scoring_metrics_overall.json
raw_responses/
failed_scoring_responses.jsonl
manifest.json
```

### 8.1 Activation prediction scoring

输入：

```text
candidate explanation
activation_prediction_task examples
```

让 scoring LLM 对每个 example 输出：

```json
{
  "row_idx": 3397,
  "predicted_high_activation": 1,
  "confidence": 0.82,
  "rationale_short": "Matches reflective phrasing and inferred meaning."
}
```

与 `expected_high_activation` 比较。

建议指标：

| 指标 | 说明 |
|---|---|
| AUROC | 用 `confidence` 作为连续分数，评估 high vs low activation |
| accuracy | 二分类准确率 |
| F1 | high-activation 类别的 F1 |
| balanced accuracy | 类别不均衡时更稳 |

如果模型只输出 0/1，没有连续 `confidence`，AUROC 不稳定，应报告 accuracy/F1，并把 AUROC 标为 `not_available_binary_only`。

### 8.2 Code discrimination scoring

输入：

```text
candidate explanation
target_label
code_discrimination_task examples
```

让 scoring LLM 对每个 example 输出：

```json
{
  "row_idx": 1204,
  "predicted_target_match": 0,
  "confidence": 0.74,
  "rationale_short": "Question-like surface form but not aligned with the candidate function."
}
```

与 `expected_target_match` 比较。

建议指标：

| 指标 | 说明 |
|---|---|
| hard-negative accuracy | 是否能排除 sibling/surface negatives |
| target recall | 是否能识别 random target positives |
| target precision | 预测为目标时有多少是真的 |
| F1 | target-match 二分类综合指标 |
| sibling false positive rate | sibling negatives 被误判为目标的比例 |
| surface false positive rate | surface-matched negatives 被误判为目标的比例 |

## 9. Output-centric explanation 与 stimulation 怎么补

当前 P3 尚未实现 output-centric 部分。原因是当前材料主要是 utterance-level SAE activation 和 feature card，不包含完整的可干预 forward pass pipeline。

注意：output stimulation 需要加载模型权重和 SAE checkpoint 执行 forward pass，无法纯由 IDE AI agent 完成。此部分仍需编写独立脚本。

推荐新增脚本（仅此任务需要独立脚本）：

```text
run_misc_p3_output_stimulation.py
```

前置条件：

```text
1. 能加载 Llama/OpenMOSS 主模型
2. 能加载 layer 19 SAE checkpoint
3. 能拿到 SAE decoder direction W_dec[latent_idx]
4. 能在 blocks.19.hook_resid_post 注入 h <- h + alpha * W_dec[j]
5. 有一个可计算 MISC label logits 或分类概率的 probe/classifier
```

最小可做版本：

```text
对每个 latent:
  取若干 held-out counselor utterances
  baseline: 计算各 MISC label probe logits
  stimulation: 在 layer 19 residual 注入 alpha * decoder_direction
  记录 Δlogit(target_label), Δlogit(sibling_labels), Δprob
```

建议 alpha：

```text
alpha ∈ {0.5, 1.0, 2.0, 4.0}
```

建议输出：

```text
p3_output_stimulation_effects.jsonl
p3_output_stimulation_metrics_by_latent.csv
p3_output_stimulation_metrics_by_label.csv
p3_output_centric_prompts/
p3_output_centric_explanations.jsonl
manifest.json
```

输出侧解释 prompt 应该包含：

```text
input-centric explanation
target label
latent id
top input examples summary
Δlogit target and sibling labels
top increased/decreased labels or generated phrases
```

输出侧结论限制：

```text
只有 stimulation 与 input explanation 一致，才可以说“支持该 candidate functional interpretation”。
即便一致，也只能说 causal intervention evidence for probe/logit behavior，不等于证明模型具有临床理解。
```

## 10. MI coder 人工审核怎么补

推荐新增人工审核表：

```text
outputs/misc_full_sae_eval/interpretability/p3_feature_cards/manual_review/p3_mi_coder_review_template.csv
```

每行一个 latent。建议字段：

```text
packet_id
target_label
latent_idx
rank_within_label
candidate_feature_name
one_sentence_tentative_interpretation
dominant_pattern_type
artifact_risk_ai
evidence_quality_ai
activation_prediction_auroc
code_discrimination_accuracy
output_delta_target_logit
mi_coder_label
mi_coder_confidence
mi_coder_artifact_risk
final_status
reviewer_notes
```

`final_status` 建议枚举：

| final_status | 含义 |
|---|---|
| `robust_code_candidate` | 较稳定地支持目标 code 候选功能 |
| `family_level_candidate` | 更像标签族，例如 question-family / reflection-family |
| `subskill_candidate` | 捕捉某种子技能或局部策略 |
| `surface_artifact` | 主要是问号、长度、模板、ASR 等 |
| `mixed_unclear` | 多模式混合或证据不足 |
| `reject` | 明显不支持当前解释 |

人工审核需要看：

```text
p3_feature_card_packets.jsonl
p3_feature_card_examples.csv
p3_input_explanations.jsonl
p3_scoring_metrics_by_latent.csv
p3_output_stimulation_effects.jsonl
```

## 11. 最终 feature card 汇总怎么生成

推荐新增脚本：

```text
run_misc_p3_final_feature_cards.py
```

默认输入：

```text
--cards outputs/misc_full_sae_eval/interpretability/p3_feature_cards/p3_feature_card_packets.jsonl
--input-explanations outputs/misc_full_sae_eval/interpretability/p3_feature_cards/ai_reviews/p3_input_explanations.jsonl
--input-scoring outputs/misc_full_sae_eval/interpretability/p3_feature_cards/ai_scoring/p3_scoring_metrics_by_latent.csv
--output-stimulation outputs/misc_full_sae_eval/interpretability/p3_feature_cards/output_stimulation/p3_output_stimulation_metrics_by_latent.csv
--manual-review outputs/misc_full_sae_eval/interpretability/p3_feature_cards/manual_review/p3_mi_coder_review_completed.csv
```

默认输出：

```text
outputs/misc_full_sae_eval/interpretability/p3_feature_cards/final_cards
```

建议输出：

```text
p3_final_feature_cards.jsonl
p3_final_feature_cards.csv
p3_final_feature_cards.md
p3_label_summary.csv
p3_concept_cluster_seed_table.csv
manifest.json
```

每张最终 card 至少包含：

```text
Feature ID:
Layer:
Target code from P2:
Candidate type:
Input-centric explanation:
Input-centric score:
Output-centric explanation:
Output effect:
Top activating examples:
Contrastive examples:
MI coder judgment:
Artifact risk:
Final decision:
Limitations:
```

## 12. 推荐实现顺序

不要一开始做 output stimulation。先把可审计的 input-centric pipeline 跑通。

建议顺序（IDE AI agent 模式）：

```text
1. [IDE agent] 读取 prompt -> 生成 input-centric explanations -> 写入 ai_reviews/
2. [IDE agent] 读取 explanations + scoring tasks -> 生成 scoring predictions -> 计算 metrics -> 写入 ai_scoring/
3. [IDE agent] 合并 cards + explanations + scoring -> 生成人工审核模板 -> 写入 manual_review/
4. [IDE agent] 合并所有结果 -> 生成最终 feature cards -> 写入 final_cards/
5. [独立脚本] run_misc_p3_output_stimulation.py（需要加载模型，不能由 IDE agent 完成）
6. [IDE agent] 将 output-centric 结果合并回 final cards
```

第一轮由 IDE AI agent 完成 1-4：

```text
自动解释（agent 读取 prompt 并分析）
自动评分（agent 读取 explanation 并对 held-out examples 评分）
人工审核模板（agent 合并数据生成 CSV）
最终 feature card 草稿（agent 汇总生成）
```

第二轮需要独立脚本完成 5，再由 agent 完成 6：

```text
SAE decoder intervention（需加载模型 -> 独立脚本）
target/sibling logit change
output-centric explanation（agent 可辅助解释）
```

## 13. IDE AI agent 评估任务清单

以下 Task A-D 均由 IDE AI agent（当前对话 AI）直接执行，用户在 IDE 中发出评估指令即可。Task E 需要独立脚本。

实际执行时不要让 AI 一次性读取全部 180 个 prompt 或 5280 条 scoring examples。推荐使用第 14 节的 `run_misc_p3_agent_orchestrator.py` 生成小批次输入，AI 每次只读一个 `*_instructions.md` 和对应的 `*_input.json`，完成后写回同批次 `*_output.json`，再由脚本合并。

### Task A: Input explanation（IDE agent 执行）

执行方式：IDE AI agent 直接处理

流程：

```text
1. agent 读取 p3_input_explanation_prompts/*.json 中的 prompt
2. 提取 system + user 消息中的分析任务
3. 按要求生成结构化 JSON 解释
4. 将结果追加到 p3_input_explanations.jsonl
5. 同时保存 raw response 到 raw_responses/ 目录
6. 失败时记录到 failed_responses.jsonl，不中断后续处理
7. 完成后更新 manifest.json
```

验收：

```text
p3_input_explanations.jsonl 行数应为 180
每条都有 packet_id, target_label, latent_idx
parse_status=ok 的比例需要报告
不得丢弃失败样本
```

### Task B: Input scoring（IDE agent 执行）

执行方式：IDE AI agent 直接处理

流程：

```text
1. agent 读取 p3_input_explanations.jsonl 获取每个 latent 的候选解释
2. agent 读取 p3_scoring_tasks.jsonl 获取评分任务
3. 对每个 task 的每个 example，基于候选解释判断是否应高激活 / 是否匹配目标标签
4. 输出逐条 prediction 到 p3_scoring_predictions.jsonl
5. 计算 by-latent / by-label / overall metrics
6. 完成后更新 manifest.json
```

验收：

```text
prediction rows 数量 = 所有 scoring task examples 数量
metrics_by_latent.csv 行数 = 180
metrics_by_label.csv 行数 = 9
每个 latent 至少有 activation_prediction 和 code_discrimination 两类结果
```

### Task C: Manual review template（IDE agent 执行）

执行方式：IDE AI agent 直接处理

流程：

```text
1. agent 读取 cards + explanations + scoring metrics
2. 合并为 180 行的人工审核 CSV
3. 保留空列给 MI coder 填写
4. 写入 manual_review/ 目录
```

验收：

```text
p3_mi_coder_review_template.csv 行数 = 180
字段包含 final_status, mi_coder_label, reviewer_notes
```

### Task D: Final card builder（IDE agent 执行）

执行方式：IDE AI agent 直接处理

流程：

```text
1. 如果 manual review 文件不存在，生成 draft final cards
2. 如果 manual review 文件存在，合并人工结果
3. 输出 JSONL/CSV/Markdown 到 final_cards/ 目录
```

验收：

```text
p3_final_feature_cards.jsonl 行数 = 180
p3_final_feature_cards.md 包含每个 label 的 section
p3_label_summary.csv 包含每个 label 的 robust / artifact / unclear 数量
```

### Task E: Output stimulation runner（需独立脚本）

执行方式：此任务需要加载模型权重执行 forward pass，必须使用独立 Python 脚本

实现：

```text
run_misc_p3_output_stimulation.py
```

要求：

```text
先做 dry-run / config validation 模式
确认模型、SAE checkpoint、decoder direction、probe classifier 都能加载
再执行少量 latents smoke run
最后全量运行
```

验收：

```text
能对至少一个 latent 输出 baseline logits, stimulated logits, delta logits
全量输出包含 target label 和 sibling label 的 delta
报告 alpha sweep 结果
```

## 14. IDE AI agent 分批评估模式说明

当前推荐模式：由 `run_misc_p3_agent_orchestrator.py` 作为持久批次管理器，IDE 内置 AI agent 只处理当前批次。

这样做的目的：

```text
1. 避免当前 AI 一次性读取 180 个 prompt 或 5280 条 scoring examples 后丢失上下文
2. 每个批次都有独立 input / instructions / output 文件，便于暂停、恢复、审计和重跑
3. 合并、指标计算、进度快照、全量校验由 Python 脚本完成，减少手工拼接错误
4. 保留 IDE agent 直接分析文本的优势，同时把大任务拆成可控小任务
```

### 14.1 批次文件在哪里

批次目录：

```text
outputs/misc_full_sae_eval/interpretability/p3_feature_cards/agent_batches
```

每个批次会生成两类输入文件，并要求 AI 写回一个输出文件：

| 文件 | 作用 |
|---|---|
| `task_a_batch_{NNN}_input.json` | Task A 当前批次的 prompt messages |
| `task_a_batch_{NNN}_instructions.md` | Task A 当前批次的人类可读执行说明 |
| `task_a_batch_{NNN}_output.json` | AI 需要写回的 Task A 解释结果 |
| `task_b_batch_{NNN}_input.json` | Task B 当前批次的 scoring tasks |
| `task_b_batch_{NNN}_instructions.md` | Task B 当前批次的人类可读执行说明 |
| `task_b_batch_{NNN}_output.json` | AI 需要写回的 Task B prediction 结果 |

AI 每次只需要读取同一个批次的 `*_instructions.md` 和 `*_input.json`。不要让 AI 直接遍历整个 `p3_input_explanation_prompts/` 或完整 `p3_scoring_tasks.jsonl`。

### 14.2 先查看当前进度

```powershell
python run_misc_p3_agent_orchestrator.py status
```

它会报告：

```text
Task A prompts / explanations / batch in-out
Task B scoring tasks / expected predictions / current predictions / metrics
Task C manual review template rows
Task D final card rows and Markdown 是否存在
Pipeline complete: True / False
```

进度快照会写入：

```text
outputs/misc_full_sae_eval/interpretability/p3_feature_cards/p3_agent_batch_progress.json
```

全量校验命令：

```powershell
python run_misc_p3_agent_orchestrator.py validate
```

校验结果会写入：

```text
outputs/misc_full_sae_eval/interpretability/p3_feature_cards/p3_agent_validation.json
```

### 14.3 自动生成下一批

如果只是继续未完成任务，优先使用：

```powershell
python run_misc_p3_agent_orchestrator.py next-batch --batch-size 3
```

行为：

```text
1. 如果 Task A 未完成，生成下一个 Task A 批次
2. 如果 Task A 已完成但 Task B 未完成，生成下一个 Task B 批次
3. 如果 Task A/B 都已完成，提示下一步应运行 Task C 或 Task D
4. 如果 Task A-D 都已完成，提示运行 validate
```

生成批次后，AI 的操作是：

```text
1. 打开最新的 *_instructions.md
2. 按 instructions 指向的路径打开同批次 *_input.json
3. 逐 item 生成结构化结果
4. 写入 instructions 中指定的 *_output.json
5. 运行对应 merge 命令
6. 再运行 status 查看下一步
```

Task A 的基本循环：

```powershell
python run_misc_p3_agent_orchestrator.py status
python run_misc_p3_agent_orchestrator.py next-batch --batch-size 3
# AI 读取 task_a_batch_{NNN}_instructions.md 和 task_a_batch_{NNN}_input.json
# AI 写入 task_a_batch_{NNN}_output.json
python run_misc_p3_agent_orchestrator.py task-a merge
python run_misc_p3_agent_orchestrator.py task-a validate
python run_misc_p3_agent_orchestrator.py status
```

Task B 的基本循环：

```powershell
python run_misc_p3_agent_orchestrator.py next-batch --batch-size 3
# AI 读取 task_b_batch_{NNN}_instructions.md 和 task_b_batch_{NNN}_input.json
# AI 写入 task_b_batch_{NNN}_output.json
python run_misc_p3_agent_orchestrator.py task-b merge
python run_misc_p3_agent_orchestrator.py task-b metrics
python run_misc_p3_agent_orchestrator.py status
```

Task C/D 不需要 AI 大规模逐条判断，直接由脚本合并已有结果：

```powershell
python run_misc_p3_agent_orchestrator.py task-c run
python run_misc_p3_agent_orchestrator.py task-d run
python run_misc_p3_agent_orchestrator.py validate
```

### 14.4 手动指定小批次

按标签和 rank 切小批次：

```powershell
python run_misc_p3_agent_orchestrator.py task-a prepare --label RE --rank-min 1 --rank-max 5 --limit 2
```

限制 Task A 每组展示样例数，适合当前 AI 上下文不足时做抽样审计：

```powershell
python run_misc_p3_agent_orchestrator.py task-a prepare --label RE --rank-min 1 --rank-max 5 --limit 2 --max-examples-per-group 2
```

按标签切 Task B 小批次：

```powershell
python run_misc_p3_agent_orchestrator.py task-b prepare --label QUO --limit 2
```

限制 Task B 每个 scoring task 的 examples 数，适合抽样检查 scoring rubric 是否可用：

```powershell
python run_misc_p3_agent_orchestrator.py task-b prepare --label QUO --limit 2 --max-examples-per-task 6
```

重要限制：

```text
1. 正式全量评估时，不建议使用 --max-examples-per-group 或 --max-examples-per-task 作为最终结果口径。
2. 这两个参数会减少当前批次输入量，适合 smoke run、人工抽查、prompt 质量审计。
3. 若用截断批次生成最终 outputs，validate 可能无法代表全量任务完成情况。
4. 要完整覆盖 Task B，必须让所有 scoring task examples 都被预测并 merge。
```

### 14.5 重跑、审计和覆盖已有结果

默认情况下，`prepare` 和 `next-batch` 会跳过已经完成的 item。

如果只是继续工作，不要加 `--include-completed`。

只有在审计、抽样复核或有意重跑时才使用：

```powershell
python run_misc_p3_agent_orchestrator.py task-a prepare --include-completed --label RE --rank-min 1 --rank-max 1 --limit 1
python run_misc_p3_agent_orchestrator.py task-b prepare --include-completed --label RE --rank-min 1 --rank-max 1 --limit 1
```

如果指定了已经存在的 `--batch-id`，脚本会拒绝覆盖。只有明确要覆盖测试批次时才使用：

```powershell
python run_misc_p3_agent_orchestrator.py task-a prepare --batch-id 901 --include-completed --label RE --rank-min 1 --rank-max 1 --limit 1 --force
```

### 14.6 AI 写回输出时必须保持的字段

Task A output 每个 result 至少保留：

```text
item_index
packet_id
target_label
latent_idx
rank_within_label
one_sentence_tentative_interpretation
main_patterns
relationship_to_target_label
artifact_risk
evidence_quality
candidate_feature_name
alternative_explanations
recommended_followup_checks
final_concise_conclusion
parse_status
```

Task B output 每个 result 至少保留：

```text
item_index
packet_id
target_label
latent_idx
task_type
predictions[row_idx, predicted, confidence, rationale_short]
```

`packet_id`, `target_label`, `latent_idx`, `rank_within_label`, `task_type`, `row_idx` 必须从 input 原样复制。否则 merge 后无法和原始 prompt / scoring task 对齐。

### 14.7 当前仓库中的已完成状态

截至 2026-06-29，当前工作区通过了：

```powershell
python run_misc_p3_agent_orchestrator.py status
python run_misc_p3_agent_orchestrator.py validate
```

当前状态：

```text
Task A complete: 180 / 180 explanations
Task B complete: 5280 / 5280 predictions
Task C generated: 180 manual review template rows
Task D generated: 180 final cards, Markdown exists
Pipeline complete: True
Validation: PASS
```

因此，后续若不是从零重跑，应把 orchestrator 主要用于：

```text
1. 按标签 / rank 抽样审计某些 latent 的 explanation
2. 重跑某些低质量或疑似错误的 batch
3. 用小批次验证新 prompt / 新 rubric
4. 重新生成 Task C/D 草稿并运行 validate
```

## 15. 论文与报告中的表述边界

可以说：

```text
We generated candidate input-centric interpretations for SAE latents using top-activating and contrastive MISC utterances.
The explanations were evaluated by whether they predicted held-out feature activation and discriminated target MISC labels from sibling and surface-matched negatives.
```

谨慎说：

```text
This latent appears associated with open-ended elaboration invitations.
This feature may capture reflective phrasing rather than context-grounded reflection.
The evidence supports a candidate functional interpretation, not a causal mechanism.
```

不能说：

```text
This feature is QUO.
The SAE proves the model understands MISC.
Top activating examples demonstrate the model's clinical reasoning mechanism.
```

如果未完成 output stimulation，不能写：

```text
feature stimulation increases target-code logits
ablation changes model behavior
causal role of the feature
```

## 16. 快速检查命令

优先检查当前 AI agent 批处理流水线状态：

```powershell
python run_misc_p3_agent_orchestrator.py status
python run_misc_p3_agent_orchestrator.py validate
```

检查当前 P3 dry-run 产物：

```powershell
python -c "import json,pathlib,pandas as pd; d=pathlib.Path('outputs/misc_full_sae_eval/interpretability/p3_feature_cards'); m=json.loads((d/'manifest.json').read_text(encoding='utf-8')); s=pd.read_csv(d/'p3_feature_card_summary.csv'); print(m['n_cards'], m['n_scoring_tasks'], m['n_prompts'], m['n_missing_packets']); print(s.groupby('target_label').size().to_dict()); print(s['source_packet_status'].value_counts().to_dict())"
```

预期：

```text
180 360 180 0
每个 label 20
phase1_packet 163
fallback_generated_from_feature_store 17
```

检查 prompt 数量：

```powershell
(Get-ChildItem outputs\misc_full_sae_eval\interpretability\p3_feature_cards\p3_input_explanation_prompts\*.json).Count
```

预期：

```text
180
```

检查 scoring task 数量：

```powershell
(Get-Content outputs\misc_full_sae_eval\interpretability\p3_feature_cards\p3_scoring_tasks.jsonl).Count
```

预期：

```text
360
```

## 17. 最小可交付版本

如果另一个 AI 只做第一轮自动化，最低交付标准是：

```text
1. p3_input_explanations.jsonl: 180 条
2. p3_scoring_predictions.jsonl: 所有 task example 的逐条预测
3. p3_scoring_metrics_by_latent.csv: 180 行
4. p3_scoring_metrics_by_label.csv: 9 行
5. p3_mi_coder_review_template.csv: 180 行
6. p3_final_feature_cards_draft.md: 可人工阅读
7. manifest.json: 记录输入、输出、模型、成功/失败数量
```

此时仍不能声称完成 P3 全部因果机制验证，只能声称：

```text
已完成 input-centric 自动解释与 held-out/scaffolded scoring 的第一版。
output-centric stimulation 和 MI coder final audit 尚待完成。
```
