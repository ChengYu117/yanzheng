# 阶段5：Latent Card生成与Held-out解释评分

> 状态：FROZEN  
> 导航：[返回实验设置总览](../02_实验设置冻结摘要.md)  
> 执行规范：以[实验SSOT](../../../../docs/current/experiment_workflow.md)为准。

## 1. 研究子目标

回答两个问题：一个stable-core latent在什么样的咨询师语句上响应；根据这些句子归纳出的解释，能否预测没有参与解释生成的新句子上的真实响应强弱。

## 2. 为什么分为Explainer和Scorer两阶段

只展示Top激活句并生成流畅解释容易过拟合。为此，Explainer用强响应与弱正响应做对比，弱正响应是“相似但不满足全部条件”的hard negatives；随后Scorer只在独立held-out句子上检验冻结解释。解释生成与解释评估因此使用不同文本。

## 3. Explainer阶段如何实现

### 3.1 输入给AI的内容

- 10条强响应句；
- 10条弱但非零响应句；
- 文本来源的基本背景：口语/转录对话；
- 任务：找出区分强组与弱组的最窄自然语言条件。

AI只看到句子及强/弱组别，不看到token激活、数值激活、SAE背景、MISC标签、latent编号或项目代码。这样做是为了减少标签定义和机制术语对解释的锚定。

### 3.2 输出内容

冻结解释包括简短名称、语言/表面假设、行为/话语功能假设、一句话主解释和综合解释类型。表面线索与功能线索允许同时存在；主解释负责说明该latent最可能依赖哪种组合，而不是强迫二者互斥。

## 4. Held-out Scorer如何实现

每个可评分latent在解释前就冻结20条独立句子：5 high、5 mid、5 weak-positive和5 control。随后按确定性种子随机排列，排列后才赋`H001...H020`公开ID。

Scorer只能看到：

- 冻结解释中的`short_name`、两类hypothesis、`primary_explanation`和`explanation_type`；
- 每条held-out样本的`sample_id`和句子文本。

Scorer看不到真实响应层级、激活值、标签、latent编号、发现集样本，也看不到必要条件、不足条件、混杂因素、局限或Explainer自评置信度。它根据解释预测每条句子的相对响应，最后通过`sample_id`与私有truth连接计分，而不是按数组位置对齐。

## 5. AI执行隔离

- 模型：GPT-5.5，`reasoning_effort=low`。
- 每个latent单独一次`codex exec --ephemeral`请求。
- 使用空目录，忽略个人配置和rules。
- 只读sandbox，禁用Shell、Web、apps/plugins、memory、multi-agent和plan tool。
- 并发上限20；成功请求工具调用总数为0。

这些设置保证第200个任务不会继承前面任务的latent解释上下文。

## 6. 评分指标怎样理解

| 指标 | 它检验什么 |
|---|---|
| Spearman | 预测排序与真实响应排序是否一致 |
| Pearson(log activation) | 预测强度与对数真实激活幅度是否线性一致 |
| Positive-vs-zero AUROC | 能否区分有响应句与零响应句 |
| High-vs-weak accuracy | 能否在强响应和弱正响应之间选对更强者 |

## 7. 冻结完成状态与结果

- Stable-core候选：218个。
- 可构建Explainer packet并通过验证：214个。
- 可构建完整held-out分层并通过Scorer：207个。
- 4个`explainer_ineligible`和7个`scorer_ineligible`均显式记录，没有复制句子或放宽采样。
- packet完整性、迁移与最终完成审计全部pass。

| 指标 | 207个单元均值 |
|---|---:|
| Spearman | 0.6170 |
| Pearson(log activation) | 0.6101 |
| Positive-vs-zero AUROC | 0.8485 |
| High-vs-weak accuracy | 0.7832 |

## 8. 主要产物

- `private/master_packets.jsonl`与`private/heldout_truth.jsonl`：私有采样和计分truth。
- `public_packets/explainer_packets.jsonl`与`scorer_packets.jsonl`：AI可见输入。
- `explainer/validated_explanations.jsonl`：214张冻结解释。
- `scorer/faithfulness_metrics.csv`：207个单元的held-out指标。
- `packet_manifest.json`、`packet_integrity_audit.json`和`final_completion_audit.json`：协议审计。

## 9. 结果如何回答研究问题

平均Spearman 0.6170和AUROC 0.8485说明，多数解释包含能够迁移到未见句子的真实响应规律，而不只是复述发现集句子。不同latent的分数仍有明显差异，因此论文应把解释可靠性视为连续量，并优先展示高忠实度card。

## 10. 限制

held-out忠实度证明“这段自然语言描述能预测latent响应”，不证明描述抓住了模型内部真实因果计算，也不证明模型已经理解MI。本轮没有shuffled/empty explanation基线，也没有为解释指标计算bootstrap CI。
