# CLAUDE.md

本项目：MISC 心理咨询行为标签的 SAE 可解释性研究（Llama-3.1-8B / Gemma3-4B hidden states + SAE latent）。
运行命令与产物清单见 `README.md`；本文件只记录易踩坑、反复用得上的约定。

## 环境（硬规则）

- 所有 `run_*.py` 和 `test_*.py` 必须用 conda 环境 **`qwen-env-py311`**（Python ≥3.11，`lm-saes` / `transformer-lens` 依赖要求）。
- 调用方式：`conda run -n qwen-env-py311 python <script>`，或全路径
  `C:\Users\chengyu\AppData\Local\miniconda3\envs\qwen-env-py311\python.exe <script>`。
- 不要用裸 `qwen-env` 环境，也不要用默认 `python`。

## 测试与验证

- 测试是独立可执行脚本（`python test_xxx.py`），不是 pytest 套件；没有 Makefile。
- 改动代码后的标准动作：先 `python -m py_compile <file>` 语法检查，再跑相关 `test_*_smoke.py`。
- smoke 测试用合成数据，CPU 即可；全量推理（`run_sae_evaluation.py`、`run_gemma_scope_sae_evaluation.py`）需要 GPU + 本地模型权重。

## 数据口径

- 主数据：`data/mi_quality_counseling_misc`（6194 条，每条 `unit_text` 是一个样本单位）。
- `data/mi_re` 是 legacy 兼容数据，必须显式 `--data-dir` 指定才用，不作默认。
- 核心标签 9 个：`RE RES REC QU QUO QUC GI SU AF`；`OTHER` 不写入主结论。
- `RE`、`QU` 是父标签，只作 MISC 层级一致性检查，不进 leaf 标签主结论。

### 数据格式与分布模式（给后续 AI 的工作上下文）

- 原始可用数据根目录是 `data/mi_quality_counseling_misc`，其中 `misc_annotations/high/*.jsonl` 和 `misc_annotations/low/*.jsonl` 是 LLM 分段后的 MISC 2.1 counselor utterance 标注；`counselor_utterances/*/*.jsonl` 是更早的 counselor-only 抽取输入。
- 当前大多数分析不要重新扫原始 JSONL，优先使用已合并产物：`outputs/misc_full_sae_eval/records.jsonl` 与 `outputs/misc_full_sae_eval/label_matrix.csv`。二者行顺序与 `outputs/misc_full_sae_eval/feature_store/utterance_features.pt` 的第 0 维严格对齐。
- `records.jsonl` 每行包含 `file_id`, `unit_text`, `predicted_code`, `predicted_subcode`, `rationale`, `confidence`, `record_id`, `quality_label`, `labels`, `source_split`, `source_file`, `source_line` 等字段。这里没有前一句 client utterance，不要为 RES/REC 推断上下文关系。
- `label_matrix.csv` 形状为 `6194 × 19`，列为 `row_idx, record_id, file_id, source_split, source_file, predicted_code, predicted_subcode, confidence, unit_text, RE, RES, REC, QU, QUO, QUC, GI, SU, AF, OTHER`。
- `label_matrix.csv` 的核心标签是多标签二值列，但层级导致多数“多标签”只是父子共现：`RE == RES OR REC`，`QU == QUO OR QUC`。每行核心标签激活数分布是 0 个: 1610，1 个: 1252，2 个: 3332。
- 核心标签全量计数/占比：`QU 1974 / 31.9%`, `RE 1358 / 21.9%`, `QUO 1206 / 19.5%`, `REC 842 / 13.6%`, `QUC 768 / 12.4%`, `GI 681 / 11.0%`, `RES 516 / 8.3%`, `AF 349 / 5.6%`, `SU 222 / 3.6%`, `OTHER 1610 / 26.0%`。
- `predicted_code` 中非核心标签会并入 `OTHER` 口径：`FA, ST, AD, FI, DI, NR, RC, CO, WA, EC, RF`。不要把这些列当作 9 个核心 MISC 标签直接建模。
- `source_split` 是原会话质量来源，不是 train/test split：`high=4169` 条、`low=2025` 条；文件数为 `high=153`, `low=99`。高低质量会话标签分布不同，例如 high 中 `RE/REC/QUO/AF` 占比更高，low 中 `GI/QUC/OTHER` 相对更多。
- 文本是 counselor 当前行为单元，`unit_text` 无空值；词数均值约 13.1，中位数 10，最大 50。规范化 exact duplicate 较多：6194 行中约 1214 行属于重复文本，唯一文本约 5392 条；top-activation 分析必须把重复视为潜在模板/artifact 证据。
- 当前 Llama 主 SAE 特征文件：`outputs/misc_full_sae_eval/feature_store/utterance_features.pt`，形状 `[6194, 32768]`；raw hidden 聚合激活：`utterance_activations.pt`，形状 `[6194, 4096]`；hook 点为 `blocks.19.hook_resid_post`，utterance 聚合方式为 `max`，没有保存全 token-level latent。
- 解释或报告时的安全表述：top activating examples 只用于候选解释生成；不能仅凭这些样例给 latent 命名，也不能声称因果机制。对 `RES/REC/RE` 尤其要说明缺少前文 client context。

## 方法论口径（关键，避免误写结论）

- `minimal_sufficient_subspace_v2` = probe 空间预测充分性，回答"最少多少 latent 能接近完整候选池表现"。
- Step6 对照用于比较 SAE / PCA / raw hidden 的结构可解释性，不用分类 F1 单独判定谁更可解释。
- **结论边界**：当前所有结果是相关性 / 结构性 / 预测充分性证据，不是因果证明。禁止写成"单个 latent 等价于某 MISC 标签"或"已完成因果证明"。

## 输出与文档

- 所有运行产物统一写到 `outputs/<run-name>/...`，不要污染仓库根目录。
- 结论性文档放 `doc/`；项目总览在 `docs/PROJECT_DOSSIER.md`。
- 建议（非强制）：有意义的复跑或代码改动，在 `doc/日志.md` 追加一条带日期的条目，记录命令和验证结果——这是本项目现有习惯。

## 代码结构

- 核心逻辑在 `src/nlp_re_base/`；根目录 `run_*.py` 是薄 CLI 入口。
- 新增分析遵循三件套：`src/nlp_re_base/<module>.py` + 根目录 `run_<name>.py` + 配套 `test_<name>_smoke.py`，并在 `src/nlp_re_base/__init__.py` 按需导出。

## P3 stable-core 对比式解释（唯一执行入口）

- 实施或复跑 P3 stable-core 对比式 latent 解释时，只以 `doc/P3_stable_core对比式latent解释_Claude_Code执行工作流.md` 为执行规范。
- `doc/P3流程详解.md`、`doc/P3_AI流水线评估与自动化实现指南.md`、`doc/P3研究设计与方法规则.md` 和 `doc/P3对比式latent解释流程设计.md` 只作历史背景，不得从中复制命令、状态或验收口径。
- 禁止用关键词规则、正则、固定词表、硬编码概率或本地 fallback 伪造 Explainer、Scorer、baseline、minimal-pair Designer 或子概念聚类的 LLM raw outputs。
- Claude Code 同时负责工程实现与 LLM 评估。模型必须逐条读取 `llm_tasks/*.jsonl`，并把原始 JSON 回答写入任务声明的 `expected_output_path`，不得绕过任务文件直接合成最终指标。
- 任一阶段门禁未通过时必须停止，不得继续生成下游结果或在最终报告中写成已完成。

## Git

- 不要主动 commit，除非用户明确要求。
- 注意 `outputs/` 下有大量大体积产物，提交时只 stage 明确相关的文件，别误提交大文件。
