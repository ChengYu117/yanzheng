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

## Git

- 不要主动 commit，除非用户明确要求。
- 注意 `outputs/` 下有大量大体积产物，提交时只 stage 明确相关的文件，别误提交大文件。
