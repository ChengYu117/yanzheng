# Pooling策略比较说明

本项目现在将 pooling 比较作为 SAE-RE 主流程和因果流程中的一等能力，用来回答：

> token 级 SAE latent 如何稳定映射到 utterance 级 RE 信号？

## 当前纳入比较的 pooling

v1 固定支持以下 6 种：

- `max`
- `mean`
- `sum`
- `binarized_sum`
- `last_token`
- `weighted_mean`

其中：

- `weighted_mean` 采用 SGPT 风格的位置递增线性权重
- `last_token` 定义为有效 token 范围内最后一个 token
- `binarized_sum` 定义为按 token 求和后做阈值化

## 为什么不纳入 `cls`

当前基座模型是 decoder-only Llama，不存在 encoder-style 的天然 `CLS` token，因此不将 `cls` 作为主比较集合。

## 为什么不纳入 attention pooling

attention pooling / generalized pooling 会引入额外可训练参数或额外学习过程。当前阶段的目标是比较**无参数聚合口径**，而不是训练新的句级读出器，因此先不纳入。

## therapist span 是默认主口径

对于 `cactus` 数据，RE 判断关注的是 therapist 当前话语本身，而不是 `client + therapist` 混合整句。

因此：

- 如果样本里存在 `therapist_char_start / therapist_char_end`
- 默认 pooling scope 使用 `therapist_span`
- 否则回退到 `full_sequence`

这条规则同时适用于：

- 主 SAE 评估流程
- 因果验证流程

## 输出文件

当启用 compare 模式时：

- 主流程输出 `pooling_comparison.json` / `pooling_comparison.md`
- 因果流程输出 `pooling_comparison.json` / `pooling_comparison.md`

每种 pooling 会写到各自子目录中，便于横向比较。
