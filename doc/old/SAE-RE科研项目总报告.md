# SAE-RE科研项目总报告

## 1. 适用范围与定位

本文档面向答辩、导师汇报和阶段性归档，目的是从科研项目视角系统说明当前 SAE-RE 项目：

- 想解决什么问题
- 目前采用了什么方法
- 已经得到哪些证据
- 当前最关键的不足是什么
- 下一步研究应该怎么推进

这是一份总览文档，不替代现有专题文档。具体实现细节以以下文档和代码为准：

- [项目技术流程说明](./项目技术流程说明.md)
- [因果性分析结果达标性报告](./因果性分析结果达标性报告.md)
- [RE定义与AI评审Rubric](./RE定义与AI评审Rubric.md)
- [AI专家代理评审管线说明](./AI专家代理评审管线说明.md)
- [sae_causal_validation_guide](./sae_causal_validation_guide.md)

本文中的结论均按“当前阶段性证据”表述，不采用超出当前结果支持范围的强结论。

---

## 2. 项目背景与研究动机

### 2.1 研究对象：RE 是什么

本项目研究的核心概念是 RE，即 Reflective Listening / 反映式倾听。按照当前仓库固定的定义，RE 不是泛泛的“温和咨询语气”，也不是一般支持性表达，而是：

- 对来访者已经表达的内容、情绪或意义进行复述、镜映或深化
- 尽量避免提问、建议、教育和明显指导性语言
- 关注“是否真的在反映来访者表达”，而不是只看一句话听上去像不像咨询腔

这一点的定义边界由 [RE定义与AI评审Rubric](./RE定义与AI评审Rubric.md) 固定。

### 2.2 为什么使用 SAE

项目不是单纯做一个 RE / NonRE 分类器，而是希望回答一个更强的问题：

> 在 Llama-3.1-8B 第 19 层 residual stream 中，是否存在一组可以被 SAE 拆出的、与 RE 概念相关的 latent 特征或小子空间？

使用 SAE 的动机是：

- 它提供一种比原始 dense activation 更稀疏的表示
- 这些稀疏 latent 更适合做候选概念发现
- 后续可以在 latent 层面做消融、注入和对照实验，探索因果相关性

### 2.3 当前项目的实际边界

当前项目研究的不是“完整心理咨询系统中的 RE 机制”，而是更窄的目标：

- 在单句或 therapist span 级别识别 RE 相关信息
- 发现与 RE 区分相关的 latent
- 测试这些 latent 是否对当前 RE 判别方向具有干预作用

因此它更像是一个“概念发现与因果验证管线”，而不是一个完整的对话建模项目。

---

## 3. 研究目标与分阶段目标

项目当前可拆成三个逐层增强的科研目标。

### 3.1 目标一：证明 SAE 空间存在 RE 相关候选特征

目标一的意思不是“已经找到典型 RE 概念”，而是先证明：

- SAE latent 空间里确实有和 RE / NonRE 差异显著相关的候选特征
- 这些特征不是完全随机噪声
- 少量 top latent 组合起来，已经能较强地区分 RE 和 NonRE

### 3.2 目标二：证明一组 latent 子空间与 RE 判别方向具有因果相关性

目标二要求更强。它要求：

- 选出一组 latent
- 在消融时显著打掉 RE 方向
- 在注入时显著抬高 RE 方向
- 而且这种效应不是随机组、bottom 组或正交方向都能轻易复制的

### 3.3 目标三：证明该组是稳定、清晰、选择性强、可解释的 RE 概念表示

这是目前最难、也最接近论文级结论的目标。它不只要求“有效”，还要求：

- 这组 latent 在不同采样或不同运行下稳定出现
- 高激活样本看起来真的清晰像 RE
- 干预后主要影响 RE，而不是 RE / NonRE 一起推高
- 组内结构不要高度冲突、抵消或依赖偶然组合

### 3.4 当前阶段性判断

结合现有文档与结果，当前最稳妥的阶段性判断是：

- 目标一：基本达成
- 目标二：部分达成
- 目标三：尚未达成

---

## 4. 数据与任务定义

### 4.1 当前代码支持两类数据来源

当前代码的数据层兼容两种输入：

1. `data/cactus/cactus_re_small_1500.jsonl`
   - 统一 JSONL
   - 记录 `formatted_text`、`therapist_char_start/end`
   - 支持 therapist span 限定聚合
2. `data/mi_re/re_dataset.jsonl` 与 `data/mi_re/nonre_dataset.jsonl`
   - 历史 split 格式
   - 以 `unit_text` 为主
   - 没有 therapist span 边界

### 4.2 必须区分“当前默认数据”与“当前已有结果快照所用数据”

这点非常重要。

- 当前代码默认数据目录已经偏向 `data/cactus`
- 但仓库里现成的 `sae_eval_full/` 和 `causal_validation_full/` 结果快照，是基于 **799 条 RE + 799 条 NonRE** 的历史 split 数据跑出来的
- 这一点可以从 `sae_eval_full/run.log` 直接看到：
  - `RE samples: 799`
  - `NonRE samples: 799`
  - `Total: 1598`

因此：

- 当前仓库同时处于“代码已兼容 CACTUS”和“现成 full 结果仍来自 legacy RE/NonRE split”的过渡状态
- 解释现有实验结果时，必须以 `1598` 条历史 split 为准
- 解释当前数据处理能力时，必须说明代码已支持 CACTUS therapist-span 方案

### 4.3 当前分析单位

当前项目的分析单位不是完整咨询会话，而是：

- 历史 split 数据中的单条 `unit_text`
- 或 CACTUS 中 `(client_prev, therapist_curr)` 形成的 therapist span 样本

这意味着当前研究回答的是：

> 某条句子 / 某段 therapist span 是否呈现 RE 信息，以及这种信息是否能在 SAE latent 空间里被发现和干预。

它不能直接回答：

> 模型是否学会了完整多轮对话里的 RE 机制。

### 4.4 RE / NonRE 的定义边界

本项目中的 RE 定义来自固定 rubric：

- RE：simple reflection、complex reflection、meaning reflection、feeling reflection
- NonRE：问题、建议、教育、命令、劝说、泛泛安慰、模板性套话等

这意味着“咨询味”“温和语气”“支持性表达”不自动等于 RE。

### 4.5 CACTUS 与 MI-RE 的不同角色

两类数据在项目中的角色不同：

- MI-RE：
  - 更接近原始二分类任务标签
  - 当前 full 结果快照主要基于它
- CACTUS：
  - 用于构建 therapist-span、最小上下文、therapist-only 聚合的新实验链路
  - 其回合级 RE / NonRE 标签是后处理构造，不是原始字段直接给定

---

## 5. 整体方法链路

当前科研方法可以概括为 7 段连续链路。

### 5.1 基底模型与目标层

项目加载本地 `Llama-3.1-8B` 基底模型，并固定关注：

- `blocks.19.hook_resid_post`

也就是第 19 层 residual stream。

### 5.2 SAE 编码

在该层 residual activation 上运行：

- `OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x`
- `Llama3_1-8B-Base-L19R-8x`

对应 SAE 配置：

- `d_model = 4096`
- `d_sae = 32768`

这一步把 dense activation 转成稀疏 latent 表示。

### 5.3 utterance-level / therapist-span 聚合

token 级 latent 不直接进入后续统计，而是先聚合成 utterance 级表示。

当前主要使用：

- `max` pooling

在 CACTUS therapist-span 链路里，聚合仅发生在 therapist token 范围；在历史 split 数据里，整句都视为 counselor text。

### 5.4 结构性评估

结构性评估回答：

- SAE 重构是否保真
- latent 是否足够稀疏
- 是否存在大量 dead features
- 用 SAE 重构替换后，模型输出分布偏移有多大

对应指标包括：

- `MSE`
- `Cosine Similarity`
- `Explained Variance`
- `FVU`
- `L0 sparsity`
- `dead_ratio`
- `CE loss delta`
- `KL divergence`

### 5.5 功能性评估

功能性评估回答：

- 哪些 latent 和 RE / NonRE 显著相关
- 少量 top latent 是否足够支持分类
- MaxAct 样本是否高纯度
- latent 间是否存在吸收、几何重叠和局部因果贡献

对应指标包括：

- univariate analysis + BH-FDR
- sparse probe
- dense probe
- diffmean
- MaxAct purity
- feature absorption
- feature geometry
- TPP

### 5.6 因果验证

因果验证围绕候选组 `G1 / G5 / G20` 展开，包含：

- necessity：
  - zero ablation
  - mean ablation
  - cond_token ablation
- sufficiency：
  - constant steer
  - cond_input steer
  - cond_token steer
- selectivity / side effects：
  - 生成后再回打分
  - retention / lexical quality
- group structure：
  - cumulative top-k
  - leave-one-out
  - add-one-in
  - synergy

### 5.7 AI 专家代理评审

主流程还会导出 `judge_bundle`，然后用固定 rubric 做：

- latent 级高激活句评审
- group 级高分样本评审
- top_examples 与 control_examples 对照

这一部分的作用是：

- 给统计指标补上“文本是否真的像 RE”的结构化证据
- 避免把“分类有用”直接误当成“概念清晰”

---

## 6. 主流程结果告诉了我们什么

以下解释基于：

- `sae_eval_full/metrics_structural.json`
- `sae_eval_full/metrics_functional.json`
- `sae_eval_full/candidate_latents.csv`
- `sae_eval_full/judge_bundle/`

### 6.1 SAE 空间中确实存在 RE 相关候选特征

从 `metrics_functional.json` 可见：

- `total_latents = 32768`
- `significant_fdr = 1530`
- `sparse_probe_k20 AUC = 0.9129`
- `dense_probe AUC = 0.9667`
- `diffmean AUC = 0.9011`

这说明：

- SAE latent 空间里确实包含大量与 RE / NonRE 差异显著相关的候选特征
- 少量 top latent 已经能携带较强判别信息
- 但 SAE 小子空间依然弱于 dense activation，说明信息没有被一个很小、很干净的子空间完全吸收

### 6.2 当前最强候选 latent 是什么

同一文件中，按 `|Cohen's d|` 排序的 top latent 包括：

- `19435`
- `13430`
- `31930`
- `5663`
- `29759`

这说明项目已经有一批清晰的候选 latent 排名，但“候选”不等于“已经证明是 RE 概念”。

### 6.3 MaxAct 纯度仍然偏低

`maxact_summary.avg_re_purity = 0.456`

这意味着：

- top latent 的高激活样本里，平均只有约 45.6% 是 RE
- 它们更像“对 RE 有帮助的混合候选特征”
- 还不够支撑“这是高纯度、单义、拿出来就像 RE 的典型特征”

### 6.4 SAE 保真度仍然偏弱

从 `metrics_structural.json` 可见：

- `cosine_similarity = 0.8114`
- `explained_variance = 0.0717`
- `fvu = 0.9283`
- `ce_loss_delta = 2.3635`
- `kl_divergence = 2.9417`
- `dead_ratio = 0.5343`

这些数字说明：

- SAE 能保留一定方向信息
- 但重构仍然较弱，尤其 CE/KL 偏移明显
- 当前因果实验是在一个保真度有限的 SAE 表示上完成的，因此结论必须保守

更准确地说，当前结果支持：

> 在这套 SAE 表示中存在 RE 相关方向

但还不支持：

> 我们已经在原模型里无失真地锁定了 RE 机制本身

---

## 7. 因果验证结果说明了什么

以下解释基于：

- `causal_validation_full/results_necessity.json`
- `causal_validation_full/results_sufficiency.json`
- `causal_validation_full/results_selectivity.json`
- `causal_validation_full/results_group.json`
- `causal_validation_full/selected_groups.json`

### 7.1 必要性：G20 消融会显著削弱 RE 信号

在 `results_necessity.json` 中：

- `G20 zero mean_delta_re = -1.8139`
- `G20 cond_token mean_delta_re = -1.8139`
- `G5 zero mean_delta_re = -0.7164`
- `G1 zero mean_delta_re = -0.0727`

对照组：

- `Bottom20 zero mean_delta_re = +0.0117`
- `Random20 zero mean_delta_re = -0.0281`

这说明：

- 拿掉 `G20` 这组 latent，会明显削弱当前 probe 下的 RE 方向
- 这个效应远强于随机组和 bottom 组
- 因此这组 latent 对当前 RE 判别方向具有明显必要性

### 7.2 充分性：G20 注入会整体推高 RE 方向

在 `results_sufficiency.json` 中，`cond_token` 模式下：

- `G20 lam_1.0 mean_delta_re = +0.1761`
- `G20 lam_2.0 mean_delta_re = +0.3494`
- `G5 lam_1.0 mean_delta_re = +0.0712`
- `G1 lam_1.0 mean_delta_re = +0.0123`

这说明：

- 单个 latent 的推动作用很弱
- 小组 latent 已有一定推动作用
- 更大的组更强

这支持一个关键判断：

> 当前 RE 信号更像分布在一组 latent 上，而不是集中在某一个单独 latent 上

### 7.3 选择性：这是当前最严重的问题之一

`results_sufficiency.json` 同时显示：

- `G20 cond_token mean_delta_re = +0.1761`
- `G20 cond_token mean_delta_nonre = +0.1965`

以及：

- `G5 cond_token mean_delta_re = +0.0712`
- `G5 cond_token mean_delta_nonre = +0.1222`

这意味着：

- 干预不是只把 RE 样本推向更 RE
- NonRE 样本也被几乎同样强地推向 RE 判别方向

所以当前更稳妥的解释不是：

> 找到了 RE 专属概念开关

而是：

> 找到了一个会整体推高当前 RE probe 分数的 latent 子空间

这会削弱“概念特异性”结论，因为它更像任务方向控制，而不是 RE 专属机制。

### 7.4 生成层选择性仍然偏弱

在 `results_selectivity.json` 中：

- `G1 mean_generated_re_logit_delta = 0.000`
- `G5 mean_generated_re_logit_delta = +0.0317`
- `G20 mean_generated_re_logit_delta = 0.000`

同时：

- `G5 mean_content_retention = 0.9479`
- `G1` 和 `G20` 的生成指标几乎不变

这说明：

- hidden-state 层面的 steering 效果没有稳定转化成生成层面的 RE 风格提升
- 尤其 `G20` 在内部 logit 层面最强，但在生成层面没有稳定收益

因此当前生成层证据不足以支撑强因果机制结论。

### 7.5 组结构表现出明显拮抗与不稳定

在 `results_group.json` 中：

- `full_effect = +0.8793`
- `sum_individual_effects = +0.7619`
- `synergy_score = +0.1174`

但更关键的是：

- cumulative top-k 不单调
- `add_one_in` 里存在明显负边际增益
- `leave_one_out` 里存在“去掉某个 latent 后效果反而更强”的情况

这说明当前组内 latent：

- 不是干净同向贡献
- 存在明显拮抗、抵消、条件依赖和非线性组合

所以即便组有效，也还不够“清晰”和“可解释”。

---

## 8. 当前最重要的不足，以及它们为什么削弱结论

### 8.1 选择性不足

问题：

- RE 和 NonRE 都被一起推高

为什么这会削弱结论：

- 这说明 latent 组可能抓到的是更一般的 probe 偏好方向
- 而不是 RE 概念本身
- 因果效应存在，但概念特异性不成立

### 8.2 稳定性不足

问题：

- `selected_groups.json` 中 `stable_G5 = []`
- `stable_G20` 只有 8 个 latent

为什么这会削弱结论：

- 如果一个 supposedly causal 组在 bootstrap 重采样下不能稳定复现
- 它更像数据切分敏感的任务子空间
- 而不像真实稳定的 RE 概念表示

### 8.3 主流程 top latent 与因果组选重合度很低

问题：

- 主流程 top-20 与因果 `G20` 几乎不重合

为什么这会削弱结论：

- 这说明当前“最可分 RE 的 latent”和“最可干预的 latent”并未对齐
- 可能是方法发现了另一条子空间
- 也可能是当前选组逻辑本身仍不稳定

### 8.4 MaxAct 纯度不高

问题：

- `avg_re_purity = 0.456`

为什么这会削弱结论：

- 高激活样本本身不够清晰像 RE
- 因此 latent 可能是混合信号，而不是高纯度 RE 特征

### 8.5 生成层面的证据弱

问题：

- `G20` 在生成文本层面几乎没有稳定提升

为什么这会削弱结论：

- 这说明 hidden-state 内部效应还没有被稳健地转化为外显语言行为
- 也就不能轻易说“该组 latent 在行为层面驱动了 RE”

### 8.6 数据定义仍偏单句级

问题：

- 当前多数分析仍以单条 `unit_text` 或 therapist span 为单位

为什么这会削弱结论：

- RE 本质上是对来访者表达的回应
- 如果缺少完整上下文，很多判断只能是“句内 RE 线索”，而非对话级机制

---

## 9. 当前最稳妥的结论

基于现有全部证据，当前最稳妥、最不冒进的结论是：

1. SAE latent 空间中确实存在大量与 RE / NonRE 显著相关的候选特征。
2. 一个由 latent 组成的小子空间，在消融时会明显削弱 RE 方向，在注入时会抬高 RE 方向。
3. 这种效应显著强于随机组和低排名对照组，因此不是完全随机噪声。
4. 但该子空间目前仍存在明显问题：
   - 选择性不足
   - 稳定性不足
   - 组内结构不干净
   - 生成层证据弱
5. 因此当前还不能稳妥声称：
   - 已经找到 RE 的强因果机制
   - 已经锁定稳定、清晰、可解释的 RE 概念子空间

更准确的定位应该是：

> 一组有研究价值、具备初步因果证据的 RE 候选 latent 子空间。

---

## 10. 下一步研究路线

### 10.1 修正稳定组选策略

当前最优先的问题是因果组选稳定性。后续应优先：

- 用稳定性做硬约束，而不是仅做参考
- 区分 ranked group 与 stable core group
- 避免把不稳定 latent 补进最终论证组

### 10.2 强化选择性验证

后续要进一步证明“RE 特异性”，需要：

- 检查干预是否能主要提升 RE 而不显著提升 NonRE
- 增加更强的对照组与负控制
- 引入更高纯度的 NonRE 难例

### 10.3 提升数据纯度

后续数据层改进方向包括：

- 提高 RE 标注纯度
- 减少“咨询语气但非 RE”的混合样本
- 引入更高质量的外部评审或人工复核

### 10.4 强化生成层与人工评审

后续应继续推进：

- 更强的生成层干预评估
- 更高质量的文本样例集
- AI 评审与人工复核联合

### 10.5 从单句走向对话级

如果目标是论文级 RE 机制结论，最终仍需推进到：

- 带稳定前句上下文的 therapist 响应判断
- 更真实的对话级 RE 数据
- therapist-only span 与上下文条件的联合机制分析

---

## 11. 结语

当前项目已经不是一个“想法验证原型”，而是一条可复现、可扩展、能产出多层证据的 SAE-RE 研究管线。

它已经完成了第一阶段最关键的工作：

- 找到 RE 相关候选 latent
- 建立结构性与功能性评估框架
- 建立必要性、充分性、选择性与组结构分析框架
- 建立 AI 专家代理评审链路

但它也清楚地表明：

- 研究离“论文级强结论”还有距离
- 关键短板在于稳定性、选择性、语义纯度和生成层验证

因此，当前项目最合适的定位不是“已经解决 RE 概念机制问题”，而是：

> 已经搭建并跑通一条有研究价值的 SAE-RE 概念发现与因果验证管线，并明确识别出了下一阶段必须攻克的核心问题。
