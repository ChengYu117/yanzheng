# 阶段1：数据范围与Llama Layer-19特征

> 状态：FROZEN  
> 导航：[返回实验设置总览](../02_实验设置冻结摘要.md)  
> 执行规范：以[实验SSOT](../../../../docs/current/experiment_workflow.md)为准；本文用于解释实验设计，不另立运行口径。

## 1. 研究子目标

建立一个行级严格对齐的分析数据集，使每一条咨询师语句同时拥有：原始文本、`file_id`、7个MISC叶标签、Llama第19层Raw Hidden表示和SAE稀疏特征。后续所有统计比较都必须基于同一批语句、同一行顺序。

## 2. 为什么需要这一阶段

如果文本、标签、hidden或SAE特征错位，后续即使得到很高的AUC，也可能只是把某条语句的表征错误地配给了另一条语句。因此，本阶段首先固定“研究对象是谁”，再进入latent筛选。

## 3. 输入与冻结范围

- 数据来源：`data/mi_quality_counseling_misc`。
- 分析单位：counselor的`unit_text`，不是完整会话。
- 初始提取：6194条counselor utterance。
- 统计范围：按空格分词后至少5词，最终保留5018条，归档1176条。
- 分组变量：252个`file_id`；后续划分以此为组，避免同一文件跨训练/测试。
- 标签：`RES, REC, QUO, QUC, GI, SU, AF`。
- 模型：Llama-3.1-8B，只读取`blocks.19.hook_resid_post`。
- SAE：OpenMOSS `Llama3_1-8B-Base-L19R-8x`，JumpReLU，`d_model=4096`，`d_sae=32768`。

## 4. 实验实现方法

1. 将每条咨询师语句输入Llama-3.1-8B，取得第19层每个token的residual-stream向量。
2. 在token维度做max pooling，把变长语句压缩为一个4096维Raw Hidden向量。
3. 用固定SAE对同一层表示编码，再在token维度做max pooling，得到一个32768维utterance-level SAE向量。
4. 根据“至少5词”规则生成保留掩码，并用同一掩码同步过滤文本、标签、Raw Hidden和SAE特征。
5. 保存过滤后的四类对象；任何下游脚本只读取该冻结结果根，不再自行改变样本范围。

直观地说：最终矩阵中的第`i`行都指向同一条咨询师语句；区别只是Raw Hidden用4096个连续维度描述它，SAE用32768个稀疏latent响应描述它。

## 5. 主要产物

| 产物 | 规模 | 含义 |
|---|---:|---|
| `records.jsonl` | 5018行 | 文本、`file_id`、质量来源和标签provenance |
| `label_matrix.csv` | 5018行 | 7叶标签目标与分组变量 |
| `feature_store/utterance_activations.pt` | 5018 × 4096 | 第19层Raw Hidden语句表示 |
| `feature_store/utterance_features.pt` | 5018 × 32768 | 第19层SAE语句特征 |

结果根：`outputs/rerun_new_dataset_20260716/min5_words`。

## 6. 完成门禁

- 四类产物第0维均为5018。
- 行顺序完全一致。
- 模型、层、hook和pooling方式与冻结配置一致。
- `source_split=high/low`只表示数据质量来源，不能被误当作训练/测试划分。

## 7. 本阶段得到什么

本阶段得到的是后续实验的统一测量基座，而不是关于latent语义的结论。它保证后续Stable Core、probe和card实验讨论的是同一批5018条语句。

## 8. 限制

`records.jsonl`不包含前一句client utterance，因此不能据此直接声称模型识别了RES/REC相对来访者上文“新增了什么”。当前证据以咨询师语句自身可见内容为边界。

## 9. 标签规模

| 标签 | 阳性数 | 占比 |
|---|---|---|
| RES | 416 | 0.0829 |
| REC | 829 | 0.1652 |
| QUO | 1134 | 0.2260 |
| QUC | 687 | 0.1369 |
| GI | 604 | 0.1204 |
| SU | 147 | 0.0293 |
| AF | 265 | 0.0528 |
