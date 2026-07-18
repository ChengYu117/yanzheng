# AGENTS.md

本项目研究 MISC 心理咨询行为标签与 Llama SAE latent 的关系。

## 单一事实来源（最高优先级）

当前实验唯一有效的需求、数据口径、stable-core 规则、运行命令、解释协议和完成门禁是：

`docs/current/experiment_workflow.md`

执行任何实验、修改科研说明或回答当前口径问题前，必须先读取该文件。其他 README、`doc/` 报告、日志、代码默认参数和历史产物即使仍可检索，也不得覆盖它；发生冲突时以该文件为准。不要从多个旧文档拼接流程。

旧文档若保留，顶部必须包含：

```text
Status: Superseded
Replaced by: docs/current/experiment_workflow.md
Do not use this document for implementation or experiment decisions.
```

## 环境（硬规则）

- 所有 `run_*.py` 和 `test_*.py` 必须使用 conda 环境 `qwen-env-py311`。
- 调用方式：`conda run -n qwen-env-py311 python <script>`，或使用该环境的 Python 全路径。
- 不要使用默认 `python`、裸 `qwen-env` 或其他环境。

## 测试与验证

- 测试是独立脚本，不是 pytest 套件。
- 代码改动后先执行 `python -m py_compile <file>`，再运行相关 `test_*_smoke.py`。
- smoke 测试可用 CPU；全量 Llama 特征抽取需要 GPU 和本地模型权重。
- 当前冻结实验不启动 Gemma 或其他 Llama 层；具体模型、层和输入见单一事实来源。

## 数据与结论边界

- `data/mi_re` 仅作 legacy 兼容，不是当前默认输入。
- `records.jsonl` 不含前一句 client utterance，不得据此声称直接识别 RES/REC 的上下文新增关系。
- 当前结果属于相关性、结构性、预测充分性和解释忠实度证据，不是因果证明。
- 禁止写成“单个 latent 等价于 MISC 标签”“模型已经理解 MI”或“已完成因果证明”。

## 输出与文档

- 运行产物写入单一事实来源指定的 `outputs/...` 子目录，不污染仓库根目录。
- 新的执行规范不得散落到其他文档；只更新 `docs/current/experiment_workflow.md`。
- 其他文档只能引用单一事实来源，或作为明确标记的 superseded 历史记录。
- 历史事件可追加到 `doc/日志.md`，但日志不是执行规范。

## 代码结构

- 核心逻辑位于 `src/nlp_re_base/`；根目录 `run_*.py` 是 CLI 入口。
- 新增分析遵循：核心模块 + `run_<name>.py` + `test_<name>_smoke.py`，并按需从 `src/nlp_re_base/__init__.py` 导出。

## Git

- 不主动 commit，除非用户明确要求。
- `outputs/` 含大量产物，只 stage 明确相关文件，避免误提交大文件。
