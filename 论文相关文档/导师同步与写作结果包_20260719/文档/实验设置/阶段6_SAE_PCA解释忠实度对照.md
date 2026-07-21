# 阶段6：SAE–PCA解释忠实度对照

> 状态：FROZEN  
> 导航：[返回实验设置总览](../02_实验设置冻结摘要.md)  
> 执行规范：以[实验SSOT](../../../../docs/current/experiment_workflow.md)为准。

## 1. 研究子目标

在相同句子采样、相同AI提示词和相同held-out评分协议下，比较SAE latent与PCA component哪一种更容易形成能够预测未见响应的自然语言解释。

该问题与阶段4不同：阶段4比较“标签信息能否被线性probe读出”，本阶段比较“单个表示方向能否被人类语言忠实描述”。

## 2. 冻结抽样范围

- 标签：REC、QUO、QUC、AF。
- 每标签选择4个去重SAE单元，共16个SAE。
- 对每个SAE匹配一个PCA-50方向和一个PCA-100方向。
- 总计16个SAE、16个PCA-50、16个PCA-100，即48个匿名单元。
- 形成16个SAE–PCA50和16个SAE–PCA100配对，共32个配对。

## 3. 实验实现方法

1. PCA只在训练数据上拟合，并根据其与目标标签的训练折关联确定方向，使正响应方向可比较。
2. 对SAE、PCA-50和PCA-100都使用阶段5相同的10强＋10弱Explainer与独立20条held-out Scorer。
3. 所有held-out在解释前冻结，公开样本只含随机化后的`sample_id + text`，计分按ID join。
4. Explainer和Scorer不知道当前单元来自SAE还是PCA，也不知道目标标签。
5. 三类表示使用同一GPT-5.5低推理配置、Schema、并发与零工具隔离要求。
6. 先比较表示族均值，再在同一标签和匹配单元内计算`SAE Spearman - PCA Spearman`，减少不同抽样对象造成的混淆。

PCA的control样本不强制真实响应等于零，因为稠密PCA方向通常不会产生SAE式的大量精确零值；其余流程保持一致。

## 4. Family结果

| 表示族 | N | Spearman均值 | Spearman中位数 | AUROC | High–weak |
|---|---|---|---|---|---|
| PCA-100 | 16 | 0.3128 | 0.2893 | 0.6292 | 0.7163 |
| PCA-50 | 16 | 0.3625 | 0.3241 | 0.6896 | 0.6863 |
| SAE | 16 | 0.5996 | 0.6739 | 0.8042 | 0.8337 |

## 5. 配对结果

| PCA维数 | 配对数 | SAE-PCA Spearman | SAE胜出 | SAE high–weak | PCA high–weak |
|---|---|---|---|---|---|
| 50 | 16 | 0.2371 | 11/16 | 0.8337 | 0.6863 |
| 100 | 16 | 0.2868 | 13/16 | 0.8337 | 0.7163 |

## 6. 完成门禁与产物

- 48/48 Explainer与48/48 Scorer通过严格验证。
- 32/32配对完整。
- 所有公开packet字段、5/5/5/5分层、随机顺序、hash和discovery/held-out不重叠审计通过。
- 主要产物：`faithfulness_metrics.csv`、`family_summary.csv`、`paired_comparison.csv`、正式报告和`pipeline_status.json`。

## 7. 结果如何回答研究问题

冻结样本内，SAE平均Spearman为0.5996，高于PCA-50的0.3625和PCA-100的0.3128。SAE相对PCA-50平均高0.2371并在11/16对胜出；相对PCA-100平均高0.2868并在13/16对胜出。

因此，当前结果支持“在这组冻结匹配单元中，SAE比PCA更容易获得能预测响应排序的自然语言解释”。它与PCA在阶段4拥有更高probe AUC并不矛盾：PCA可以保留更多线性预测信息，但单个PCA方向仍可能较难用简洁概念描述。

## 8. 限制

该实验不是全量PCA评估，只覆盖4个叶标签和32个配对。不能推广为“所有SAE latent都比所有PCA component更可解释”，也不能据此声称SAE在预测任务上优于PCA。
