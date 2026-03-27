# SAE-RE同类研究指标对比

> 用途：
> - 对照同类 SAE-LLM 下游任务论文，判断本项目当前结果处于什么位置。
> - 只做“精确可比”与“近似可比”的知识库式整理，不做不受控 leaderboard。
>
> 重要提醒：
> - 非同任务、非同模型、非同数据、非同 SAE 配置下的结果，原则上都**不能直接当排名**。
> - 本文档的作用是帮助判断“我们缺的是哪一类证据”，不是给出一个简单高低顺序。

---

## 1. 对比原则

本文把对比分成两类：

### 1.1 精确可比

满足以下条件中的大部分，才算精确可比：

- 指标名一致或近乎一致
- 公式/口径一致
- 评价对象一致
- 数值方向一致

即便如此，只要下面任一条件不一致，仍然不能把结果直接当 leaderboard：

- 模型不同
- 层位不同
- SAE 类型不同
- 数据任务不同
- token-level / utterance-level 口径不同

### 1.2 近似可比

指标不完全同名，但在功能上回答的是相似问题，例如：

- `TPP accuracy_drop` vs `ablation effect` / `steering effect`
- `MaxAct purity` vs `monosemanticity score` / `auto-interpretability`
- `Feature Absorption / Geometry` vs `RAVEL` / `monolinguality` / `feature splitting`

这类结果只能帮助判断：

> 我们在哪些证据轴上已经有结果，哪些轴还偏弱。

不能帮助判断：

> 我们的数值一定优于/劣于对方。

---

## 2. 本项目当前结果快照

### 2.1 结构指标

| 指标 | 当前值 | 备注 |
|---|---:|---|
| `MSE` | `4.5804` | 已在现有报告中注明：当前实现口径有问题，数值被放大，不能直接拿来做最终结论。 |
| `cosine_similarity` | `0.8088` | 方向相似度中等偏上。 |
| `explained_variance` | `0.0682` | 只解释约 6.8% 方差。 |
| `FVU` | `0.9318` | 大部分方差未被解释。 |
| `l0_mean` | `172.58` | 平均每个 token 激活约 172 个 latent。 |
| `dead_ratio` | `90.05%` | dead feature 比例较高。 |
| `ce_loss_orig` | `5.0569` | 来自全量 CE/KL 运行。 |
| `ce_loss_sae` | `7.4467` | 同上。 |
| `ce_loss_delta` | `2.3898` | SAE 替换后行为偏移较大。 |
| `kl_divergence` | `3.1969` | 替换后输出分布偏移较大。 |

### 2.2 功能指标

| 指标 | 当前值 | 备注 |
|---|---:|---|
| 显著 latent 数 | `1540 / 32768` | BH-FDR 后显著。 |
| Top-1 `Cohen's d` | `0.8991` | 单 latent 效应量。 |
| Top-1 `AUC` | `0.7003` | 单 latent 区分能力。 |
| `sparse_probe_k1 AUC` | `0.7009` | |
| `sparse_probe_k5 AUC` | `0.8534` | |
| `sparse_probe_k20 AUC` | `0.9131` | 已较强。 |
| `sparse_probe_k20 accuracy` | `0.8298` | |
| `dense_probe AUC` | `0.9677` | 仍强于 sparse probe。 |
| `dense_probe accuracy` | `0.9268` | |
| `diffmean AUC` | `0.9007` | simple baseline 已很强。 |
| `avg_re_purity` | `0.576` | MaxAct 纯度中等。 |
| `overall_mean_absorption` | `0.5464` | 冗余不低。 |
| `feature_geometry mean_cosine` | `0.0475` | 全局不拥挤。 |
| `feature_geometry max_cosine` | `0.5519` | 局部重叠明显。 |
| `TPP baseline_accuracy` | `0.8342` | in-sample probe-space。 |
| `TPP max accuracy_drop` | `0.040` | 有局部贡献，但仍属弱因果证据。 |

---

## 3. 精确可比指标表

这一节只放“同名或高度同口径”的指标。

### 3.1 结构指标：精确可比或高同源口径

| 指标 | 我们的值 | 论文与值 | 是否可直接比较 | 备注 |
|---|---:|---|---|---|
| `explained_variance` | `0.0682` | `Scaling Monosemanticity` 报告其 SAE 至少解释约 `65%` 方差；`Llama Scope` 也把 `EV` 作为核心主指标 | 否 | 指标同名，但模型、层位、SAE 规模完全不同；只能说明我们当前 fidelity 明显偏弱。 |
| `l0_mean` | `172.58` | `Scaling Monosemanticity` 报告每 token 平均激活 feature 少于 `300`；`Llama Scope` 讨论在不同 SAE 设计下把 `L0` 从约 `150` 降到约 `50` | 否 | 口径接近，但对方是不同模型/不同 tokenizer/不同 SAE 宽度。我们的稀疏性不算异常，但不能脱离 fidelity 单看。 |
| `dead_ratio` | `90.05%` | `Scaling Monosemanticity` 报告三种 SAE 的 dead feature 比例约 `2% / 35% / 65%` | 否 | 指标同名；我们的 dead ratio 相对偏高，提示字典利用率和概念可达性可能都偏弱。 |
| `MSE` | `4.5804` | `Llama Scope` 与 `SAEBench` 都使用重构误差类指标 | 否 | 我们自己的 `MSE` 当前实现已知有口径问题，因此不应做外部数值比较。 |
| `ce_loss_delta` / `loss gap` | `2.3898` | `Revisiting End-To-End...` 报告短时 `KL+MSE` finetune 可把 `CE loss gap` 降低约 `20-50%` | 部分 | 同属行为层 fidelity 指标，但对方报告的是“gap reduction”，不是原始 `delta` 标量。只能方向性说明：当前我们这条轴偏弱。 |
| `kl_divergence` | `3.1969` | `Revisiting End-To-End...` 直接讨论 `KL+MSE` 训练目标；`Llama Scope` / `SAEBench` 更常用 `Delta LM loss` 或 `loss recovered` | 部分 | 同属行为分布偏移口径，但未找到完全同格式、同任务、同模型的公开统一数值。 |
| `FVU` | `0.9318` | 同类论文更常公开 `explained_variance` 或 `loss recovered` | 否 | 未找到主 corpus 中公开、同一口径、可直接对照的 `FVU` 数值。 |

### 3.2 功能指标：精确可比或高同源口径

| 指标 | 我们的值 | 论文与值 | 是否可直接比较 | 备注 |
|---|---:|---|---|---|
| `accuracy` | `sparse_probe_k20 = 0.8298` | `SAEBench`、`AxBench` 都使用 concept detection / sparse probing accuracy 体系，但公开摘要页未给出统一单值 | 否 | 指标家族一致，但未找到同任务、同设定、同公开口径的单值可直接对照。 |
| `AUC` | `sparse_probe_k20 = 0.9131` | 主 corpus 中很少把 `AUC` 作为统一 headline metric；更多使用 accuracy、score、loss recovered、judge score | 否 | `AUC` 是你们项目里很有用的局部口径，但不是当前 SAE benchmark 的主流统一对比标尺。 |
| `f1` | 当前正式报告未给 headline 值 | 主 corpus 中公开摘要页普遍不以 `f1` 为 headline | 否 | 未找到直接同口径对比。 |
| `Feature Absorption` | `0.5464` | `SAEBench` 把 `feature absorption` 列为标准指标之一，但公开摘要页未给统一单值 | 部分 | 指标同名同义，但没有可直接搬来对照的 benchmark scalar。 |
| `TPP` / `accuracy_drop` | `max_drop = 0.040` | `SAEBench` 将 `TPP` 纳入 benchmark；其余任务论文更多报告 ablation/steering effect 而不是统一 `accuracy_drop` | 部分 | 同任务族，但同名同值对照仍缺。 |
| 单 latent `Cohen's d` | `0.8991` | 主 corpus 普遍不把 `Cohen's d` 作为 headline 指标 | 否 | 这是你们项目用于候选筛选的内部强项口径，不是当前 SAE 文献的统一对照语言。 |
| 单 latent `p_value` / `FDR` | `1540 / 32768` 显著 | 主 corpus 少有把多重比较显著性作为 headline | 否 | 很适合内部候选发现，但不适合作为跨论文主对比轴。 |

### 3.3 精确可比部分的直接结论

可以相对稳妥地说：

1. 在**结构 fidelity** 上，本项目当前结果明显弱于 `Scaling Monosemanticity` 一类大规模、高保真 SAE 报告的水位。
2. 在**行为层 fidelity** 上，本项目已经补上 `CE/KL`，这比很多只看 `MSE` 的原型工作更严谨；但数值本身说明偏移仍大。
3. 在**功能指标**上，你们用了很多当前 benchmark 认可的指标名，但其中相当一部分并不是跨论文的统一 headline scalar，因此更适合作为“研究闭环是否完整”的证据，而不是直接数值排名。

---

## 4. 近似可比指标表

这一节只讨论“功能相似，但不是同名同口径”的结果。

| 我们的指标 | 我们的值 | 近似可比论文指标 | 近似对比结论 | 备注 |
|---|---:|---|---|---|
| `avg_re_purity` | `0.576` | `monosemanticity score`（Llama Scope），`auto-interpretability`（SAEBench / Anthropic），feature description quality | 我们的 top latent 有 RE 倾向，但纯度不高；更接近“候选 feature”而非高单义 feature | `purity` 和 judge-based interpretability 都在问“高激活样本是否真的集中表达同一概念” |
| `overall_mean_absorption` | `0.5464` | `feature absorption`（SAEBench），`feature occlusion / over-splitting`（Principled Evaluations），`monolinguality` / synergy（ACL multilingual SAE） | 我们已经能量化冗余，但结果提示冗余较明显；与“概念被 cleanly 拆开”仍有距离 | 这条证据支持“有候选子空间”，不支持“概念完全解缠” |
| `feature_geometry max_cosine` | `0.5519` | geometry / feature neighbors（Llama Scope、Scaling） | 我们的全局几何不塌缩，但局部重叠明显；与 absorption 的“冗余偏高”是一致的 | geometry 更像结构提示，不是单独判决指标 |
| `TPP max accuracy_drop` | `0.040` | `single-feature ablation effect`（Unveiling Language-Specific...），`steering effect`（SAIF、SAEs Are Good for Steering、Scaling） | 我们已有局部贡献证据，但强度和证据等级仍低于真正的 model-space steering / held-out ablation 研究 | 当前 TPP 还是 in-sample probe-space |
| `dense_probe AUC` vs `sparse_probe AUC` | `0.9677` vs `0.9131` | `AxBench` 中 simple / representation baselines 强于 SAE 的现象；`DiffMean` 在 detection 上很强 | 我们的结果和 AxBench 的提醒一致：不能因为 SAE probe 强，就忽视更简单 baseline 也可能解释大部分信号 | 这一点反而增加了你们结论的稳健性，因为你们已经把 simple baseline 报出来了 |
| `diffmean AUC` | `0.9007` | `AxBench` concept detection 上 representation-based simple baselines 表现强 | 我们任务中 simple direction baseline 已很强，说明目标概念在当前数据上本来就有较强线性可分性 | 这会抬高“确认概念表示”的门槛 |
| `CE/KL` 明显偏移 | `delta=2.3898`, `KL=3.1969` | `Revisiting End-To-End...` 强调必须改善 behavior fidelity；`Llama Scope` 把 `Delta LM loss` 列为主指标 | 我们已经具备正确的 fidelity 评估方向，但结果还不能算高保真 SAE | 这条证据限制了所有更强解释结论 |

---

## 5. 按四条证据轴，对照同类研究看我们现在在哪

### 5.1 Fidelity / Approximation：偏弱

外部参照：

- `Scaling Monosemanticity`：强调较高 `EV` 与较低 dead rate
- `Llama Scope`：把 `EV + L0 + Delta LM loss` 作为基本盘
- `Revisiting End-To-End...`：强调 `CE loss gap` 仍然是关键瓶颈

本项目现状：

- 你们已经补上 `CE/KL`，这一步是对的；
- 但 `explained_variance = 0.0682`、`ce_loss_delta = 2.3898`、`kl_divergence = 3.1969` 都说明 fidelity 仍弱。

### 5.2 Task Alignment / Predictivity：较强

外部参照：

- `SAEBench`、`AxBench` 都把 sparse probing / concept detection 放在重要位置。

本项目现状：

- `sparse_probe_k20 AUC = 0.9131` 已经说明候选子空间很有信息量；
- 但 `dense_probe` 仍明显更强，`diffmean` 也非常强，因此这条轴只能支持“有强相关信号”，还不能支持“概念已被 cleanly isolated”。

### 5.3 Interpretability / Disentanglement：中等

外部参照：

- `Llama Scope` 用 monosemanticity 打分；
- `SAEBench` 用 auto-interpretability、absorption、RAVEL；
- multilingual SAE 用 monolinguality、synergy、ablation。

本项目现状：

- `MaxAct`、`absorption`、`geometry` 这套框架本身是完整的；
- 但 `avg_re_purity = 0.576`、`overall_mean_absorption = 0.5464` 表明解释性证据还不够“干净”。

### 5.4 Control / Causality / Robustness：偏弱

外部参照：

- `Scaling`、`SAIF`、`SAEs Are Good for Steering...` 都强调真正的行为操纵；
- `Evaluating Adversarial Robustness...` 强调改写/扰动稳健性；
- `AxBench` 强调不能只看单一 steering case。

本项目现状：

- 你们的 `TPP` 已经比纯相关性更进一步；
- 但它还是 probe-space、in-sample；
- 还缺 held-out 干预、输出质量副作用、改写稳健性与更强的 model-space steering/ablation。

---

## 6. 对比后的结论

### 6.1 可以明确支持的结论

对照同类研究，本项目已经可以比较稳妥地支持：

> SAE latent 空间里存在与 RE 概念相关的候选特征簇，而且一组 top latent 已经能够较强地区分 `RE / NonRE`。

这是因为你们已经具备：

- 候选筛选
- sparse probe 与 baseline 比较
- MaxAct
- absorption / geometry
- TPP
- CE/KL

这套证据链已经超出“只跑一个 probe”的原型研究。

### 6.2 还不能支持的结论

对照更强论文的标准，本项目还不足以支持：

> 我们已经确认了强效、典型、稳健、因果可信的 RE 概念表示。

主要缺口仍是：

1. fidelity 偏弱
2. MaxAct 纯度不高
3. 冗余偏高
4. TPP 仍是 in-sample probe-space
5. 缺少改写稳健性与输出层副作用报告

### 6.3 一句话判断

> 如果把同类 SAE-LLM 下游研究分成“候选发现”“概念候选”“强结论确认”三档，那么本项目当前最合理的位置是：已经稳稳进入“候选发现”，并开始具备“概念候选”的证据，但距离“强结论确认”还有一段明显距离。

---

## 7. 哪些指标最值得优先补

如果只看与同类论文的差距，下一步最值得补的不是再加一个新的相关性分数，而是补更强的这三类证据：

1. **held-out 干预证据**
   - 把 `TPP` 从 in-sample probe-space 推向 held-out 或 model-space。
2. **稳健性证据**
   - 同义改写、去关键词、换模板、跨数据切分。
3. **副作用/隔离性证据**
   - 如果 future work 做 steering，要同时报告质量损失和非目标能力污染。

这三条补上后，你们的研究位置才会从“候选发现”更稳地走向“概念确认”。
