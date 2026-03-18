# 百炼 Qwen3.5-Plus 本地专家代理评审说明

本文档说明如何在本地直接调用阿里云百炼平台的 `qwen3.5-plus`，运行本项目的 AI 专家代理评审管线。

## 1. 官方兼容方式

当前实现复用了仓库已有的 OpenAI 兼容 judge 客户端。根据百炼官方文档：

- OpenAI 兼容端点可使用：
  - `https://dashscope.aliyuncs.com/compatible-mode/v1`
- `qwen3.5-plus` 支持兼容模式调用
- 在 JSON 模式下，需要显式关闭思考模式：
  - `enable_thinking=false`

本仓库已经将这一点接入为：

```text
OPENAI_EXTRA_BODY_JSON={"enable_thinking":false}
```

## 2. 需要准备什么

- 本地 Python 环境
- 已经跑出 `judge_bundle/` 的 SAE 主流程输出目录
- 一个可用的百炼 API Key

## 3. 推荐用法：本地 env 文件 + PowerShell

先复制模板：

```powershell
copy deploy\local\bailian_qwen35plus.env.example deploy\local\bailian_qwen35plus.env
```

编辑：

```powershell
notepad deploy\local\bailian_qwen35plus.env
```

至少填写：

```text
OPENAI_API_KEY=你的百炼APIKey
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen3.5-plus
OPENAI_EXTRA_BODY_JSON={"enable_thinking":false}
JUDGE_INPUT_DIR=outputs/sae_eval_bailian_qwen35plus
JUDGE_OUTPUT_DIR=outputs/sae_eval_bailian_qwen35plus/ai_judge_qwen35plus
```

然后运行：

```powershell
powershell -ExecutionPolicy Bypass -File deploy\local\run_bailian_ai_judge.ps1
```

## 4. 直接命令行运行

如果你不想写 env 文件，也可以直接传参数：

```powershell
powershell -ExecutionPolicy Bypass -File deploy\local\run_bailian_ai_judge.ps1 `
  -InputDir outputs\sae_eval_bailian_qwen35plus `
  -OutputDir outputs\sae_eval_bailian_qwen35plus\ai_judge_qwen35plus `
  -PythonExe C:\Users\chengyu\AppData\Local\miniconda3\envs\qwen-env\python.exe `
  -ApiKey <你的APIKey>
```

脚本内部会自动设置：

- `OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`
- `OPENAI_MODEL=qwen3.5-plus`
- `OPENAI_EXTRA_BODY_JSON={"enable_thinking":false}`

## 5. 先做 dry-run

建议第一次先做 prompt 演练，不真正调用 API：

```powershell
powershell -ExecutionPolicy Bypass -File deploy\local\run_bailian_ai_judge.ps1 -DryRunPrompts
```

这样会生成 prompts 和占位产物，但不会消耗 API。

## 6. 产物位置

跑完后主要看这些文件：

- `utterance_reviews.jsonl`
- `latent_reviews.json`
- `group_reviews.json`
- `calibration.json`
- `report.md`
- `logs/`
- `prompts/`

## 7. 常见问题

### 7.1 JSON 模式报错

如果报错类似 “JSON mode is not supported when enable_thinking is true”，说明没有正确传：

```text
OPENAI_EXTRA_BODY_JSON={"enable_thinking":false}
```

### 7.2 模型名无效

请确认使用的是：

```text
qwen3.5-plus
```

### 7.3 输入目录不对

`--input-dir` 可以传：

- SAE 主输出目录
- 或者直接传 `judge_bundle/` 目录

## 8. 安全建议

- 不要把 API Key 写入仓库
- `deploy/local/*.env` 已加入 `.gitignore`
- 如果 API Key 已经公开暴露，建议尽快在百炼控制台轮换

## 9. 参考

- 阿里云百炼 OpenAI 兼容文档：
  - https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope
- 百炼大模型列表：
  - https://help.aliyun.com/zh/model-studio/getting-started/models
- Qwen Code / OpenAI 兼容示例：
  - https://help.aliyun.com/zh/model-studio/qwen-code
