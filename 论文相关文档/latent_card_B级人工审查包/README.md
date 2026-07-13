# Latent Card B级人工审查包

冻结日期：2026-07-13

## B级定义

B级表示：来自 stable-core、通过自动结构校验，并经过候选内容筛选，适合进入论文人工审查；它不表示已经完成人工确认，也不表示 latent 已被证明具有唯一行为功能。

## 选择范围

- 叶级标签：`RES, REC, QUO, QUC, GI, SU, AF`。
- 每个叶级标签固定 5 个候选，共 35 条 label-latent 关联。
- 共享候选 8 个：RES-REC 2 个、QUO-QUC 4 个、AF-SU 2 个。
- 父标签 `RE` 和 `QU` 不进入叶级候选主表。

## 选择原则

1. 优先标签内 stable rank 靠前的 latent。
2. 排除 `unclear_or_mixed`、明显低覆盖和内容抽样中不可靠的 card。
3. 同一标签内保留行为功能、句法结构、情感内容和主题等不同模式，避免五张卡片表达同一个模板。
4. 对 RES/REC 和 QUO/QUC，优先保留可观察的表层模式，并将治疗/MI 功能写成待审候选。
5. `support_fraction_model_reported` 是 LLM 自报分区，不是独立准确率。

## 人工审查顺序

1. 打开 `B_level_human_review/cards_by_leaf_label/<LABEL>.md`，逐卡核对代表证据、其他 supporting 和 outlier。
2. 在 `leaf_label_B_candidates.csv` 填写最终名称、审查者和意见。
3. 审查 `shared_features.md`，先判断共享是否仅来自问句、第二人称或评价模板。
4. 两名审查者达成一致后，才能把 `review_status` 更新为人工确认状态。

## 冻结说明

`frozen_all_cards` 保存全部 225 张 card、全部句子包、质量统计和内容抽样报告。`package_checksums.sha256` 用于确认论文审查期间输入没有变化。
