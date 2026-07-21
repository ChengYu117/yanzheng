# 阶段3：AUC@K与宽松Stable Core筛选

> 状态：FROZEN  
> 导航：[返回实验设置总览](../02_实验设置冻结摘要.md)  
> 执行规范：以[实验SSOT](../../../../docs/current/experiment_workflow.md)为准。

## 1. 研究子目标

在训练数据划分发生变化时，仍能找回与某一MISC标签正相关的候选latent，从而区分“可重复的标签关联”与“一次划分中的偶然高分”。

## 2. 输入

- 阶段2保留的12589个filtered latent。
- 5018条语句的7叶标签矩阵。
- `file_id`分组变量。

## 3. 实验实现方法

### 3.1 训练折内关联排序

对每个标签，把该标签阳性语句与阴性语句的latent激活进行比较，并计算正向Cohen's d。d越大，表示该latent在标签阳性语句中的激活相对更高。排序必须只在训练折内完成，不能看到测试折。

### 3.2 用held-out AUC选择集合规模

按训练折排序依次加入Top-K latent，计算`K=0..200`时的held-out分类AUC。`K_auc`定义为首个达到`best_auc - max(0.01, SE)`的K；它回答“需要多少个高排名latent，预测性能已接近最佳”。

### 3.3 20次grouped split-half重复抽样

按`file_id`把数据重复划成两半，共20次，随机种子42。每次重新计算排名，记录Top-K集合重叠和单latent inclusion frequency。频率0.80表示该latent在20次重采样中约有16次进入目标集合。

### 3.4 2000次grouped bootstrap

按`file_id`为单位执行2000次bootstrap，得到每条候选label–latent关联的Cohen's d置信区间。`cohens_d_ci_lo > 0`要求区间下界仍为正，用于排除方向不确定的候选。

### 3.5 宽松成员口径

每个标签先确定`K*`：若存在集合稳定平台则取`K_stab`，否则取`K_auc`。随后只在full-data Top-K*中保留同时满足以下条件的成员：

```text
inclusion_frequency >= 0.70
cohens_d_ci_lo > 0
```

跨high/low质量来源的结果和标签级稳定平台状态保留为审计字段，但不作为成员硬剔除条件。这就是当前唯一有效的“宽松stable-core”口径。

## 4. 冻结结果

| 标签 | K* | Stable-core边 | 标签选择状态 |
|---|---|---|---|
| RES | 55 | 14 | performance_only_unstable |
| REC | 58 | 46 | stable_topk_found |
| QUO | 45 | 42 | stable_topk_found |
| QUC | 50 | 40 | stable_topk_found |
| GI | 85 | 26 | performance_only_unstable |
| SU | 30 | 17 | performance_only_unstable |
| AF | 58 | 43 | stable_topk_found |

- 总计228条label–latent边。
- 去重后为218个latent。
- 208个latent只关联一个叶标签，10个关联两个标签。
- 共享关系：RES–REC 4个、QUO–QUC 5个、REC–SU 1个。

## 5. 主要产物

- 正式成员表：`cross_val/stable_topk_selection_n20_relaxed_leaf7/stable_topk_latent_set.csv`。
- 选择报告：`stable_topk_selection_report.md`。
- 支撑产物：AUC@K、20次split-half、bootstrap CI与跨质量审计目录。

## 6. 完成门禁

- 只分析7个叶标签，不混入RE/QU父标签。
- split-half必须为20次、group=`file_id`、seed=42。
- 成员必须同时满足Top-K*、频率阈值和正向CI下界。
- 最终必须复现228条边和218个去重latent。

## 7. 结果如何回答研究问题

该结果支持：与MISC标签相关的信息由多个可重复候选latent共同承载，并以标签相对特异的成员为主、少量同家族共享为辅。它不支持“一latent等于一标签”。

## 8. 限制

RES、GI和SU的标签级状态为`performance_only_unstable`。这表示其内部一部分成员满足宽松可复现条件，但整个Top-K集合没有形成稳定平台；论文中不能把这三个标签描述为“标签整体稳定”。
