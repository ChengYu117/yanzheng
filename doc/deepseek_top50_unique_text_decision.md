# DeepSeek Top-50 唯一文本选择决策

- 日期：2026-07-11
- 决策：将 DeepSeek P3 第 3 步的 Top-50 选择单位从原始 utterance 行改为 `unique_normalized_text`。
- 背景：同一句文本可在原始记录中出现多次；按行取 Top-50 会让同一模板重复占位，扩大其作为独立模式证据的权重。
- 新规则：先按规范化文本分组，每组保留激活值最高的行；若激活值并列，保留较小 `row_idx`；再按激活值降序取 50 条唯一文本。
- 产物隔离：原始行版本保留在 `deepseek_v4_flash_top50_induction`；唯一文本版本使用新的 `deepseek_v4_flash_top50_unique_text_induction` 目录，两者的 LLM 解释不可混用。
- 可逆条件：如果未来研究问题转为估计原始数据出现频率，而非归纳 latent 的文本模式，可另行建立带频率权重的分析；该分析不替代唯一文本的解释证据。

## 2026-07-11 提示词与输出 schema 校准

- 决策：`representative_evidence_ids` 收紧为 2--3 条；新增必填字段 `candidate_explanation`；删除禁止命名 MISC 标签、禁止表述候选功能的两条限制。
- 置信度：采用五档校准标准（0.00--0.20 无一致模式；0.21--0.40 弱或多解；0.41--0.60 刚过多数且混淆明显；0.61--0.80 两个多数条件明显满足；0.81--1.00 高覆盖、去重稳定且替代解释弱）。
- 后果：已有旧 schema 的 LLM 输出不应再用新校验器重新验证；新的唯一文本任务包必须重建后才可发送给 API。
- 校验器同步：不再因候选解释中出现 MISC 代码而拒绝输出；当前任务仍保持标签盲输入，是否命名标签取决于人工后续是否向模型提供标签定义。
- 运行修复：resume 预筛选现在同时比对 prompt 与 raw-output 的 SHA-256；修改重试 prompt 后，同一 task id 不会再被旧成功记录静默跳过。

## 2026-07-11 双轨模式归纳

- 决策：废弃互斥的 `feature_type=surface|semantic|mixed|no_stable_pattern` 输出；每个 latent 必须同时给出 surface 与 semantic 两条轨道。
- 理由：单一 `mixed` 字段掩盖了表层形式与语义/话语功能各自的证据覆盖，且不能表达两者支持不同样本子集的情形。
- 新 schema：每条轨道各有模式、完整 partition、2--3 条代表证据和独立 confidence；校验器分别计算多数门槛并输出 `both_majority`、`surface_only`、`semantic_only` 或 `neither_majority`。
- 产物隔离：新运行使用 `deepseek_v4_flash_top50_dual_track_induction`，不覆盖已经完成的唯一文本单轨结果。
- 提示词微调：删除“不在 surface、semantic、mixed、no_stable_pattern 中单选”及“最狭窄”措辞；保留双轨强制输出和两条轨道各自的多数支持约束。
- 双轨代表证据规则：有支持样本的轨道保留 2--3 条代表证据；无支持样本的轨道允许空代表列表，避免与“该轨道无稳定模式”的诚实输出冲突。
