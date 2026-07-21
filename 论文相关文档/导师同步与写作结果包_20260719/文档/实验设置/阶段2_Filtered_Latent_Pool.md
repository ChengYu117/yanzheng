# 阶段2：Filtered Latent Pool

> 状态：FROZEN  
> 导航：[返回实验设置总览](../02_实验设置冻结摘要.md)  
> 执行规范：以[实验SSOT](../../../../docs/current/experiment_workflow.md)为准。

## 1. 研究子目标

从32768个SAE维度中排除几乎从不激活、几乎总在激活、缺少方差或由极少数异常值支配的latent，建立后续候选筛选的统一质量池。

## 2. 为什么不能直接使用全部32768维

SAE字典中包含大量在当前咨询数据上不工作的维度。一个只在极少数句子中出现的latent可能产生很大的偶然效应值；一个几乎总激活的latent也难以区分标签。先做与标签无关的激活质量过滤，可以减少这类不可靠候选，同时避免根据目标标签预先挑选特征。

## 3. 输入

- 5018 × 32768的`utterance_features.pt`。
- 与其逐行对齐的`records.jsonl`和`label_matrix.csv`。
- 本阶段过滤只使用latent激活分布；不会根据某个MISC标签决定latent是否进入候选池。

## 4. 实验实现方法

对每个latent统计激活次数、激活率、均值、标准差、非零激活分位数、异常值比例以及Top-1/Top-5激活占比。当前主要门槛为：

- 激活判定阈值：`activation > 1e-8`；
- 至少激活10条语句，且激活率至少0.001；
- 激活率不得超过0.995；
- 标准差必须大于`1e-8`；
- 对激活数不少于20的latent，检查IQR/Z-score异常值比例以及激活质量是否被极少数样本支配；
- Top-1激活占比不得超过0.80，Top-5不得超过0.95，异常值比例不得超过0.50。

只要命中任一排除原因，`keep=False`。同一latent可能同时命中多个原因，因此各drop reason数量不能简单相加。

过滤后，再针对7个叶标签在保留池内计算关联统计表，为下一阶段的Cohen's d排序提供统一输入。

## 5. 主要产物

- `functional/misc_label_mapping_filtered/feature_filter_audit.csv`：逐latent保留状态和诊断统计。
- `functional/misc_label_mapping_filtered/feature_filter_summary.json`：过滤配置与汇总数量。
- `functional/misc_label_mapping_filtered/latent_label_matrix.csv`：保留latent与各标签的关联统计。

## 6. 冻结结果

| 项目 | 数量 |
|---|---:|
| 原始SAE latent | 32768 |
| 保留latent | 12589 |
| 排除latent | 20179 |
| 保留率 | 38.42% |

排除原因中，20127个latent属于`rarely_active`，52个属于`almost_always_active`，10776个属于`zero_or_near_zero_variance`；原因存在重叠。

## 7. 完成门禁

- 后续候选只能来自`feature_filter_audit.csv`中`keep=True`的12589个latent。
- 过滤配置、样本数和输入文件hash必须落盘。
- 不允许后续阶段静默恢复被排除维度。

## 8. 本阶段结果如何理解

12589并不是“与MI有关的latent数量”，而只是“在当前数据上有足够激活质量、值得继续比较的latent数量”。标签相关性和可复现性要到下一阶段才判断。
