# DeepSeek Latent Card 样本质量与标签统计报告

## 1. 统计范围

本报告覆盖 225 个 unique stable-core latent、303 条 label-latent 关联和 11,250 条 latent 内句子记录。一个 latent 可以关联多个标签，因此标签表按关联口径统计；同一 latent 的 50 条句子会分别计入其关联的每个标签。

DeepSeek 在生成时只看到 latent ID 与 50 条去重文本，没有看到 MISC 标签、激活值、排名或对照分组。所有 225 张 card 已通过 JSON schema、ID 完整分区、代表证据归属和多数门槛校验。

## 2. 质量指标定义

- `support_fraction`：模型归入主要解释支持集合的句子比例。低于 0.75 记为低覆盖，仅表示解释边界较弱，不自动等于解释错误。
- `very_short_sample_rate`：不超过 3 个词的句子比例。
- `fragment_like_sample_rate`：以连接词/限定词结尾、包含破碎标点或明显未完结的句子比例。该指标是启发式审查线索。
- `transcription_noise_sample_rate`：连续重复词、破碎标点、替换字符等转录噪声比例。口语中的自然重复也可能被计入。
- `near_duplicate_sample_rate`：同一 latent 的 50 条唯一文本中，字符相似度至少 0.92 的近重复句子所占比例。
- `high_confidence_low_support_rate`：置信度至少 4，但支持比例低于 0.75 的 card 比例，用于发现可能的置信度校准问题。

这些规则用于定位需要人工复核的样本，不是人工 gold 质量标签，也不证明 latent 的真实机制。

## 3. 总体结果

- 明确类型 card 的平均支持比例：92.3%
- 低覆盖明确类型 card：23/213 (10.8%)
- 高置信度但低覆盖：11/213 (5.2%)
- 极短句：2.6%
- 疑似残句：7.3%
- 疑似转录噪声：0.9%
- 近重复句：4.1%

解释类型（unique latent 口径）：

- `behavioral_function`：128 (56.9%)
- `linguistic_structure`：57 (25.3%)
- `topic`：17 (7.6%)
- `unclear_or_mixed`：12 (5.3%)
- `affective_content`：10 (4.4%)
- `surface_artifact`：1 (0.4%)

置信度分布（unique latent 口径）：

- `2` 分：10 (4.4%)
- `3` 分：22 (9.8%)
- `4` 分：186 (82.7%)
- `5` 分：7 (3.1%)

## 4. 各标签统计

| 标签 | 关联数 | 平均置信度 | 明确类型平均支持 | 低覆盖 | 行为功能 | 语言结构 | 不清晰/混合 | 极短句 | 疑似残句 | 转录噪声 | 近重复 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AF | 27 | 3.89 | 89.3% | 14.8% | 51.9% | 25.9% | 0.0% | 6.8% | 6.2% | 1.0% | 4.4% |
| GI | 26 | 3.65 | 91.0% | 19.2% | 23.1% | 11.5% | 0.0% | 0.5% | 4.7% | 0.8% | 7.8% |
| QU | 26 | 4.04 | 94.2% | 3.8% | 73.1% | 26.9% | 0.0% | 2.8% | 10.2% | 0.9% | 3.5% |
| QUC | 45 | 4.00 | 93.6% | 7.0% | 66.7% | 28.9% | 4.4% | 2.7% | 7.1% | 0.5% | 2.8% |
| QUO | 32 | 4.00 | 93.6% | 6.2% | 68.8% | 31.2% | 0.0% | 1.6% | 10.1% | 0.9% | 4.1% |
| RE | 54 | 3.91 | 92.5% | 13.7% | 53.7% | 38.9% | 5.6% | 1.4% | 6.0% | 1.0% | 3.9% |
| REC | 47 | 3.89 | 92.2% | 13.6% | 61.7% | 29.8% | 6.4% | 0.6% | 5.9% | 1.0% | 3.7% |
| RES | 15 | 3.40 | 95.3% | 0.0% | 46.7% | 26.7% | 26.7% | 5.5% | 7.2% | 1.2% | 2.0% |
| SU | 31 | 3.71 | 91.9% | 10.7% | 58.1% | 16.1% | 9.7% | 3.5% | 10.3% | 1.5% | 3.6% |

## 5. 标签层面观察

- `QU/QUC/QUO` 的明确解释支持率均在 93.6% 以上，且行为功能与语言结构合计接近或达到 100%。这说明问题类 latent 的句子簇较一致，但也提示问句结构可能与行为功能共同驱动解释。
- `GI` 的低覆盖率最高（19.2%），并且 65.4% 的 card 被归为主题类型。部分解释停留在宽泛的 healthcare/health behavior 主题，需要优先检查是否足够具体。
- `AF` 的明确解释平均支持率最低（89.3%），极短句率也最高（6.8%）。赞许、感谢和一般正向评价容易被合并，需要检查行为功能与情感内容是否混淆。
- `RE` 与 `REC` 的构成接近：明确解释支持率分别为 92.5% 和 92.2%，但低覆盖率均超过 13%。低覆盖项多使用 reflection/advice 等宽泛名称，不能仅凭名称视为已识别反映功能。
- `RES` 的平均置信度最低（3.40），`unclear_or_mixed` 比例最高（26.7%）。其明确类型 card 虽有较高支持率，但总体可解释性受不清晰子集限制。
- `SU` 的疑似残句率（10.3%）和转录噪声率（1.5%）在标签中最高，建议人工审查时同时查看原句，不只阅读 card 摘要。

总体判断：句子簇和生成 card 足以支持后续分层人工复核，但不能直接作为最终人工确认解释。主要风险集中在宽泛行为命名、问题句式与功能混淆、近重复模板，以及少量残句和转录噪声。

## 6. 各标签优先复核对象

### AF

- latent `11511`，`appreciation_expression`：类型 `behavioral_function`，置信度 3，支持 50.0%，疑似残句 12.0%，转录噪声 0.0%，近重复 4.0%。
- latent `5320`，`gratitude_and_greetings`：类型 `behavioral_function`，置信度 3，支持 52.0%，疑似残句 4.0%，转录噪声 2.0%，近重复 8.0%。
- latent `28469`，`affirmative_evaluative_responses`：类型 `behavioral_function`，置信度 3，支持 64.0%，疑似残句 4.0%，转录噪声 0.0%，近重复 4.0%。

### GI

- latent `6651`，`healthcare_discourse`：类型 `topic`，置信度 3，支持 60.0%，疑似残句 6.0%，转录噪声 2.0%，近重复 8.0%。
- latent `19483`，`healthcare_advice_dialogue`：类型 `behavioral_function`，置信度 3，支持 62.0%，疑似残句 2.0%，转录噪声 0.0%，近重复 4.0%。
- latent `17685`，`health_behavior_discourse`：类型 `topic`，置信度 3，支持 70.0%，疑似残句 8.0%，转录噪声 2.0%，近重复 14.0%。

### QU

- latent `664`，`motivational_interviewing_scale_questions`：类型 `behavioral_function`，置信度 3，支持 58.0%，疑似残句 6.0%，转录噪声 0.0%，近重复 0.0%。
- latent `11660`，`wh_questions_with_would`：类型 `linguistic_structure`，置信度 4，支持 84.0%，疑似残句 16.0%，转录噪声 0.0%，近重复 0.0%。
- latent `21859`，`wh-questions about personal history and habits`：类型 `behavioral_function`，置信度 4，支持 84.0%，疑似残句 10.0%，转录噪声 0.0%，近重复 8.0%。

### QUC

- latent `9827`，`question_formats_varied`：类型 `unclear_or_mixed`，置信度 2，支持 100.0%，疑似残句 14.0%，转录噪声 0.0%，近重复 4.0%。
- latent `736`，`mixed_questions_and_statements`：类型 `unclear_or_mixed`，置信度 2，支持 100.0%，疑似残句 8.0%，转录噪声 0.0%，近重复 0.0%。
- latent `664`，`motivational_interviewing_scale_questions`：类型 `behavioral_function`，置信度 3，支持 58.0%，疑似残句 6.0%，转录噪声 0.0%，近重复 0.0%。

### QUO

- latent `664`，`motivational_interviewing_scale_questions`：类型 `behavioral_function`，置信度 3，支持 58.0%，疑似残句 6.0%，转录噪声 0.0%，近重复 0.0%。
- latent `19840`，`open-ended_questions_about_experience`：类型 `behavioral_function`，置信度 4，支持 74.0%，疑似残句 4.0%，转录噪声 0.0%，近重复 4.0%。
- latent `3156`，`open-ended_questions_about_knowledge_goals`：类型 `behavioral_function`，置信度 4，支持 78.0%，疑似残句 8.0%，转录噪声 2.0%，近重复 0.0%。

### RE

- latent `13312`，`counseling_dialogue_mixed`：类型 `unclear_or_mixed`，置信度 2，支持 100.0%，疑似残句 2.0%，转录噪声 0.0%，近重复 12.0%。
- latent `2995`，`counseling_dialogue_mixed`：类型 `unclear_or_mixed`，置信度 3，支持 100.0%，疑似残句 6.0%，转录噪声 2.0%，近重复 4.0%。
- latent `3805`，`therapeutic_discourse_mixed`：类型 `unclear_or_mixed`，置信度 3，支持 100.0%，疑似残句 2.0%，转录噪声 0.0%，近重复 4.0%。

### REC

- latent `31585`，`healthcare_dialogue_mixed`：类型 `unclear_or_mixed`，置信度 2，支持 100.0%，疑似残句 6.0%，转录噪声 0.0%，近重复 0.0%。
- latent `2995`，`counseling_dialogue_mixed`：类型 `unclear_or_mixed`，置信度 3，支持 100.0%，疑似残句 6.0%，转录噪声 2.0%，近重复 4.0%。
- latent `3805`，`therapeutic_discourse_mixed`：类型 `unclear_or_mixed`，置信度 3，支持 100.0%，疑似残句 2.0%，转录噪声 0.0%，近重复 4.0%。

### RES

- latent `11435`，`healthcare_dialogue_mixed`：类型 `unclear_or_mixed`，置信度 2，支持 100.0%，疑似残句 10.0%，转录噪声 0.0%，近重复 0.0%。
- latent `29701`，`you_know_what_questions`：类型 `unclear_or_mixed`，置信度 2，支持 28.0%，疑似残句 8.0%，转录噪声 0.0%，近重复 0.0%。
- latent `32696`，`healthcare_discourse_mixed`：类型 `unclear_or_mixed`，置信度 2，支持 100.0%，疑似残句 6.0%，转录噪声 0.0%，近重复 0.0%。

### SU

- latent `5366`，`therapeutic_discourse_mixed`：类型 `unclear_or_mixed`，置信度 2，支持 100.0%，疑似残句 8.0%，转录噪声 0.0%，近重复 0.0%。
- latent `30223`，`apology_and_permission_request`：类型 `unclear_or_mixed`，置信度 3，支持 78.0%，疑似残句 10.0%，转录噪声 2.0%，近重复 0.0%。
- latent `2995`，`counseling_dialogue_mixed`：类型 `unclear_or_mixed`，置信度 3，支持 100.0%，疑似残句 6.0%，转录噪声 2.0%，近重复 4.0%。

## 7. 结论与使用边界

1. 当前 card 在结构层面完整，但结构通过不等于概念解释已经人工确认。优先审查低支持、高置信度低支持以及不清晰/混合 card。
2. `unclear_or_mixed` card 的 supporting/outlier 分区不具有统一语义，部分结果将全部句子列为 supporting；因此主表的平均支持率只在明确解释类型中计算。
3. 句子簇已经消除规范化完全重复，但仍存在近重复、转录残句和口语重复；这些现象可能让模型更容易归纳表层结构。
4. 标签统计描述的是 stable-core latent 的 card 构成，不表示标签本身由某一种解释类型定义。共享 latent 会出现在多个标签中。
5. 分层抽样表每个标签列出 3 个风险最高的关联，适合作为第一轮人工复核入口。

## 8. 输出文件

- 标签汇总：`outputs\misc_full_sae_eval\interpretability\contrastive_latent_interp\deepseek_v4_flash_latent_cards\quality_report\latent_card_quality_by_label.csv`
- latent 明细：`outputs\misc_full_sae_eval\interpretability\contrastive_latent_interp\deepseek_v4_flash_latent_cards\quality_report\latent_card_quality_by_latent.csv`
- 标签-latent 关联：`outputs\misc_full_sae_eval\interpretability\contrastive_latent_interp\deepseek_v4_flash_latent_cards\quality_report\label_latent_card_associations.csv`
- 句子级质量明细：`outputs\misc_full_sae_eval\interpretability\contrastive_latent_interp\deepseek_v4_flash_latent_cards\quality_report\sentence_quality_details.csv`
- 分层抽样复核：`outputs\misc_full_sae_eval\interpretability\contrastive_latent_interp\deepseek_v4_flash_latent_cards\quality_report\stratified_quality_review_sample.csv`
