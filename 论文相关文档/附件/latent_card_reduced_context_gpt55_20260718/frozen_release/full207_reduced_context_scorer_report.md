# v2 Reduced-context Scorer 全量运行报告

> Status: Complete / Pending human review
>
> Model: GPT-5.5, reasoning effort low, concurrency 20

## 1. 评估范围

- 复用 v2 全量 Explainer 已冻结的解释，不重新生成解释；
- 复用原有 held-out 句子与真实激活，不重新采样；
- 214 张结构有效解释中，207 张满足 Scorer 资格并进入本次评估；
- 每张卡评估 20 条 held-out 句子，共 4,140 条预测。

Scorer 只看到：

```text
short_name
surface_or_linguistic_hypothesis
behavioral_or_discourse_hypothesis
primary_explanation
explanation_type
```

Scorer 看不到 `contrastive_explanation`、必要条件、不足条件、混杂因素、局限、备选解释、置信度、discovery 分组、真实响应、激活层级、MISC 标签、SAE 背景或 latent 编号。

## 2. 执行与验证

- 主批次：207/207 请求成功；
- 模型：GPT-5.5；低推理强度；20 路并发；
- 每张卡使用独立 ephemeral 请求，空工作目录，插件关闭，工具调用为 0；
- 最终结构验证：207 valid，0 failed；
- 最终预测数：4,140；
- 非空 `matching_evidence_span`：2,620；
- 非逐字证据片段：0。

初次运行发现 8 个证据片段存在省略号、空格归一化或轻微改写。它们不影响数值评分，但违反逐字证据约束。第一次重试修复 5 张；随后增强提示词与验证器，只重跑剩余 3 张。最初 8 个输出保存在 `scorer/raw_nonverbatim_initial/`，两轮重试的完整执行清单分别保存在相邻的 `evidence_retry8` 与 `evidence_retry3_strict` 目录。

## 3. 原 v2 与 Reduced-context 全量结果

| 指标 | 原 v2 均值 | Reduced-context 均值 | 均值差 | 原中位数 | 新中位数 |
|---|---:|---:|---:|---:|---:|
| Spearman | 0.6197 | 0.6212 | +0.0015 | 0.6753 | 0.6927 |
| Pearson（log activation） | 0.5908 | 0.5987 | +0.0078 | 0.6588 | 0.6717 |
| Positive-vs-zero AUROC | 0.8488 | 0.8507 | +0.0019 | 0.8867 | 0.8867 |
| High-vs-weak pair accuracy | 0.7894 | 0.7918 | +0.0024 | 0.8400 | 0.8400 |

阈值计数：

- Spearman > 0.3：183/207；
- Spearman > 0.5：153/207；
- AUROC > 0.7：179/207；
- AUROC < 0.5：8/207。

## 4. 初步判断

删除必要条件、不足条件、混杂因素和局限后，四项全量均值均未下降，变化接近于零。这支持使用 reduced-context Scorer：冻结解释的核心内容本身足以进行 held-out 匹配，评分不需要依赖 Explainer 预先提供的排除规则。

这仍然是自然语言解释对未见句子响应排序的忠实度证据，不是 latent 与 MISC 标签等价、模型具有特定心理咨询机制或因果作用的证明。207 张卡中仍存在 AUROC 低于 0.5 的失败解释，应单独进入人工失败案例审核。

## 5. 正式产物

- `scorer/faithfulness_metrics.csv`：207 张卡的最终指标；
- `scorer/validated_predictions_private.jsonl`：4,140 条带真实响应的私有验证记录；
- `scorer/validation_failures.jsonl`：最终为空；
- `scorer/tasks.jsonl` 与 `scorer/task_manifest.json`：任务和字段暴露审计；
- `scorer/llm_execution_manifest.jsonl`：主批次执行审计；
- `scorer/raw_nonverbatim_initial/`：被替换的 8 个初始原始输出。
