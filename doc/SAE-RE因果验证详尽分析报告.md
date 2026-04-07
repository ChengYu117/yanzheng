# SAE-RE因果验证详尽分析报告

## 1. 执行概览

本报告分析的是 `E:\LLM_project\yanzheng\causal_validation_full` 中保存的完整因果验证结果。根据 [run.log](E:\LLM_project\yanzheng\causal_validation_full\run.log)，本次实验完成于 **2026-04-03**，总耗时约 **5436.5s**，使用本地 `Llama-3.1-8B` 与公开 SAE `OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x / Llama3_1-8B-Base-L19R-8x`，在 `max` pooling 下完成了：

- latent group 选择与 bootstrap 稳定化
- necessity / ablation
- sufficiency / steering
- selectivity / side-effect evaluation
- group structure analysis

本次分析不重跑实验、不改代码、不额外抽样。所有判断只基于现有结果文件和现有文档说明。

本次使用的结果文件如下：

- `results_necessity.json`
- `results_sufficiency.json`
- `results_selectivity.json`
- `results_group.json`
- `selected_groups.json`
- `summary_tables.md`
- `run.log`

需要先强调两点边界：

1. 本报告本身不做新的抽样判断，但实验结果中有两个模块天然不是“全量结论”：
   - `results_selectivity.json` 的生成侧评估内置 `max_samples=16`
   - `results_group.json` 里的 `leave-one-out / add-one-in / synergy` 实际只在 **top-10** 子组上计算
2. 因果实验这里的 RE 信号不是人工临床评分，而是 **probe-logit 层面的 RE 证据变化**。

因此，本报告的目标不是证明“已经完成 RE 的最终因果机制确认”，而是判断：**在现有 probe-space 因果验证框架下，这批 latent 已经达到了什么程度。**

---

## 2. 证据地图与实验口径

### 2.1 每个结果文件回答什么问题

| 文件 | 回答的问题 | 本报告中的作用 |
|---|---|---|
| `selected_groups.json` | `G1/G5/G20` 是哪些 latent、bootstrap 稳定性如何、probe baseline 多强 | 定义被验证对象和基线能力 |
| `results_necessity.json` | 去掉一组 latent 后，RE 证据是否下降 | 判断必要性 |
| `results_sufficiency.json` | 加上一组 latent 后，RE 证据是否上升 | 判断充分性 |
| `results_selectivity.json` | 干预后是否伴随生成副作用或保真问题 | 补充判断选择性与副作用 |
| `results_group.json` | 这组 latent 是协同、冗余还是互相抵消 | 判断群组结构与 clean 程度 |
| `summary_tables.md` | 已格式化汇总表 | 快速校对主结论 |
| `run.log` | 实验真实执行顺序、采用的组、稳定化过程 | 用于确认实验口径 |

### 2.2 四类实验分别检验什么

1. **Necessity = ablation**
   问题是：如果把某组 latent 去掉，RE 证据会不会下降。

2. **Sufficiency = steering**
   问题是：如果给某组 latent 加上干预，RE 证据会不会上升。

3. **Selectivity = side effects / retention**
   问题是：干预后是否伴随明显副作用，例如内容保留差、重复率变化、文本多样性异常。

4. **Group Structure = cumulative / LOO / synergy**
   问题是：这组 latent 是 clean additive group，还是内部有明显冗余、冲突和相互抵消。

### 2.3 本次实验对象到底是什么

根据 [selected_groups.json](E:\LLM_project\yanzheng\causal_validation_full\selected_groups.json) 与 [run.log](E:\LLM_project\yanzheng\causal_validation_full\run.log)，本次实际验证的不是主流程原始 top-20，而是**重新排序并做 bootstrap 稳定化之后的 group**：

- `G1 = [6966]`
- `G5 = [31133, 6966, 16969, 5551, 26276]`
- `G20 = [8104, 6966, 20936, 11658, 22358, 23464, 31133, 16969, 5551, 26276, 3416, 23670, 22698, 24490, 30498, 27541, 16292, 15539, 24943, 13243]`

稳定性结果为：

- `stable_G5 = [31133]`
- `stable_G20 = [8104, 6966, 20936, 11658, 22358, 23464, 31133]`
- bootstrap seeds = `10`

这意味着本次因果验证回答的是：

> “在重排并稳定化后的候选 latent group 上，是否存在 necessity / sufficiency / selectivity / group-structure 证据”

而不是：

> “原主流程 top-20 latent 在任何定义下都已经被最终确认”

### 2.4 probe baseline 口径

`selected_groups.json` 给出的 probe baseline 是：

- `accuracy = 0.747`
- `f1 = 0.758`
- `auc = 0.842`

这与主流程文档中的 `sparse_probe_k20 AUC = 0.9131` **不是同一条 baseline**。原因是本次因果实验重新做了 group ranking 和 bootstrap 稳定化，并用新的 `G20` 重新训练了 probe。因此，本报告只在因果实验内部使用 `0.842` 作为基线，不把它与主流程 `0.9131` 做横向优劣判断。

### 2.5 干预 span 的实际含义

根据 [causal/data.py](e:\LLM_project\yanzheng\causal\data.py)，当前 `counselor_span_mask` 实际上等于整条 `unit_text` 的全部非 padding token。因为当前数据集中每条记录本身就是 counselor utterance。

所以本次因果实验的干预对象是：

> **整条 counselor utterance 的全 token latent / residual 表示**

而不是更细粒度的局部 span-level intervention。

---

## 3. 必要性分析

### 3.1 结果总表

`results_necessity.json` 中，核心指标是：

- `mean_delta_re`：对 RE 样本，ablation 后 RE-logit 的平均变化
- `mean_delta_nonre`：对 NonRE 样本，ablation 后 RE-logit 的平均变化

在 necessity 语境下，**我们最关心的是 `mean_delta_re` 是否显著为负**。因为负值说明 RE 证据被削弱。

| Group | Mode | `mean_delta_re` | `mean_delta_nonre` | `fraction_improved` |
|---|---|---:|---:|---:|
| `G1` | `zero` | `-0.092` | `-0.012` | `0.000` |
| `G1` | `mean` | `-0.093` | `-0.012` | `0.101` |
| `G1` | `cond_token` | `-0.092` | `-0.012` | `0.000` |
| `G5` | `zero` | `-0.443` | `+0.184` | `0.078` |
| `G5` | `mean` | `-0.448` | `+0.192` | `0.188` |
| `G5` | `cond_token` | `-0.443` | `+0.184` | `0.078` |
| `G20` | `zero` | `-1.860` | `+1.061` | `0.291` |
| `G20` | `mean` | `-1.447` | `+1.401` | `0.357` |
| `G20` | `cond_token` | `-1.860` | `+1.061` | `0.291` |
| `Bottom20` | `zero` | `+0.004` | `-0.008` | `0.379` |
| `Bottom20` | `mean` | `-0.033` | `+0.074` | `0.263` |
| `Bottom20` | `cond_token` | `+0.004` | `-0.008` | `0.379` |
| `Random20` | `zero` | `+0.014` | `-0.013` | `0.295` |
| `Random20` | `mean` | `+0.006` | `+0.001` | `0.392` |
| `Random20` | `cond_token` | `+0.014` | `-0.013` | `0.295` |

补充说明：

- `fraction_improved` 在 `score_delta()` 中被定义为“全样本 delta > 0 的比例”。对于 necessity 来说，它不是主判据，因为 necessity 目标恰恰希望 RE 样本的 delta 为负。因此这一列只作补充，不作为 necessity 强弱的主要依据。

### 3.2 G1：单 latent 有信号，但明显不够强

`G1` 在三种 ablation 下都稳定出现约 `-0.092` 的 `mean_delta_re`，说明：

- 去掉这个单 latent，RE 证据确实会下降；
- 因此它不是无关 latent；
- 但幅度非常有限，远不足以支撑“单 latent 足以承载 RE 概念”的说法。

与控制组相比：

- `Bottom20` 约为 `+0.004`
- `Random20` 约为 `+0.014`

所以 `G1` 显著不同于随机和 Bottom-K，但它只能支持：

> 单 latent 可以带有 RE 相关信息

不能支持：

> 单 latent 已经构成强效 RE feature

### 3.3 G5：必要性明显增强，说明 RE 信息不是单点承载

`G5` 的 `mean_delta_re` 稳定在 `-0.443 ~ -0.448`，相比 `G1` 大约放大了 4.8 倍。

这表明：

- 从 `G1` 到 `G5`，ablation 的必要性效应显著增强；
- RE 证据不是被一个单 latent 独占，而是分布在一小组 latent 上；
- 因此“distributed group / small subspace”假设比“single monosemantic latent”更符合当前结果。

不过 `G5` 同时带来：

- `mean_delta_nonre ≈ +0.184 ~ +0.192`

这说明 ablation 后不仅 RE 样本的 RE-logit 降了，NonRE 样本的 RE-logit 也在上升方向变化。更准确地说，`G5` 像是一组**和标签边界相关的方向**，而不是“只对 RE 样本单向起作用”的 clean RE-only group。

### 3.4 G20：必要性非常强，但不是干净的单向组

`G20` 是 necessity 最强的一组：

- `zero / cond_token`: `mean_delta_re ≈ -1.860`
- `mean`: `mean_delta_re ≈ -1.447`

和控制组相比差距非常明显：

- `Bottom20 / cond_token ≈ +0.004`
- `Random20 / cond_token ≈ +0.014`

这组结果足以支持一个很强的 necessity 判断：

> 当前 `G20` 这组 latent 对 RE 证据的维持具有明显必要性，去掉它们会显著削弱 RE 样本上的 RE-logit。

但是 `G20` 同时也表现出明显的非 clean 特征：

- `mean_delta_nonre ≈ +1.061 ~ +1.401`

也就是说，`G20` 更像是一组**深度参与 RE/NonRE 判别边界的 group**，而不是“只影响 RE 样本、几乎不碰 NonRE”的高度专一 group。

### 3.5 necessity 的整体结论

最稳妥的 necessity 结论是：

1. `G20` 的必要性显著强于 `Bottom20` 和 `Random20`，这不是随机波动。
2. `G1` 明显弱于 `G20`，不支持“单 latent 足够”。
3. 当前结果支持：
   - **RE 更像一组 latent / 一个小子空间**
4. 当前结果不支持：
   - **存在单一、主导型 monosemantic RE latent**

因此，在 necessity 这条证据链上，本次结果已经明显强于“只有统计显著性和 sparse probe”的阶段。

---

## 4. 充分性分析

### 4.1 结果读取规则

在 sufficiency 里，我们最关心的是：

- 对 RE 样本，`mean_delta_re` 是否稳定为正
- 这个正向提升是否明显高于控制方向
- `mean_delta_nonre` 是否也被一起抬高

如果 `RE` 和 `NonRE` 都一起明显上升，就不能写成“选择性的 RE steering 成功”，最多只能写成：

> 存在 steering 效应，但选择性不足

### 4.2 G1：只有粗暴 constant steering 在高 λ 下才出现明显增益

`G1` 的结果可以分成两部分：

1. `constant` 模式：
   - `λ=0.5 / 1.0` 时 `mean_delta_re` 仍为负
   - 直到 `λ=1.5 / 2.0` 时才转正，最高约 `+0.269`
   - 但同时 `mean_delta_nonre` 更高，最高约 `+0.483`

2. `cond_input / cond_token` 模式：
   - 始终只有很小正增益，约 `+0.005 ~ +0.029`
   - 基本接近零效应

这说明：

- 单 latent 的充分性很弱；
- 只有在粗暴 constant steering 且强度较高时，才会看到表面上的增强；
- 但这种增强并不选择性，甚至对 NonRE 的抬升更大。

结论：

> `G1` 不支持“单 latent 已足以充分诱发 RE 证据”。

### 4.3 G5：有微弱 steering，但几乎始终对 NonRE 更敏感

`G5` 在三类 steering 下的模式非常一致：

- `mean_delta_re` 始终为正，但幅度很小，最高约 `+0.056`
- `mean_delta_nonre` 始终更大，最高约 `+0.220`

这说明 `G5` 虽然比 `G1` 稍强，但仍然不具备 clean 的 RE-selective steering。

更像是：

- 对 probe-logit 边界施加了一个正方向扰动；
- 但这个方向并不特异于 RE；
- 它更像“泛化的 counseling / style / response-like direction”之一部分，而不是窄义 RE-only feature group。

结论：

> `G5` 提供的是很弱的充分性信号，不足以支持“这组 latent 本身就足以选择性诱发 RE”。

### 4.4 G20：明确存在 steering 效应，但选择性不足

`G20` 是 sufficiency 最值得讨论的一组。它在三种 steering 方式下，对不同 `λ` 都呈现稳定正向响应：

#### `constant`

- `λ=0.5`: `RE +0.054`, `NonRE +0.140`
- `λ=1.0`: `RE +0.103`, `NonRE +0.249`
- `λ=1.5`: `RE +0.161`, `NonRE +0.388`
- `λ=2.0`: `RE +0.213`, `NonRE +0.508`

#### `cond_input`

- `λ=0.5`: `RE +0.052`, `NonRE +0.139`
- `λ=1.0`: `RE +0.101`, `NonRE +0.248`
- `λ=1.5`: `RE +0.159`, `NonRE +0.386`
- `λ=2.0`: `RE +0.210`, `NonRE +0.507`

#### `cond_token`

- `λ=0.5`: `RE +0.041`, `NonRE +0.126`
- `λ=1.0`: `RE +0.081`, `NonRE +0.227`
- `λ=1.5`: `RE +0.127`, `NonRE +0.358`
- `λ=2.0`: `RE +0.171`, `NonRE +0.464`

可以明确得出两条结论：

1. **G20 确实有 steering 效应。**
   这一点已经成立，因为对 RE 样本的 `mean_delta_re` 在所有模式和 λ 下都稳定为正，且随 λ 增长呈上升趋势。

2. **G20 没有表现出强选择性的 RE steering。**
   因为在所有模式和 λ 下，`mean_delta_nonre` 都比 `mean_delta_re` 更大，且增长同样显著。

也就是说，`G20` 更像是在推动一个更宽的“probe 认为更接近 RE 的方向”，而不是只把 RE 样本往 RE 推、同时让 NonRE 保持稳定。

### 4.5 控制方向：Orthogonal 基本无效，但 Random_dir 不能忽略

控制方向结果非常重要：

#### Orthogonal

- `mean_delta_re` 基本在 `-0.003 ~ -0.013`
- `mean_delta_nonre` 基本在 `+0.004 ~ +0.011`

这说明正交方向基本无效，是一个好的负控制。

#### Random_dir

- `λ=0.5`: `RE +0.011`
- `λ=1.0`: `RE +0.020`
- `λ=1.5`: `RE +0.031`
- `λ=2.0`: `RE +0.044`

虽然明显弱于 `G20`，但它并不是完全零。这意味着：

- 只要在 residual space 施加某些非正交方向，也可能带来一定 RE-logit 变化；
- 因而不能把 `G20` 的 sufficiency 结果直接写成“完全特异且排他”的 steering 证据。

### 4.6 sufficiency 的整体结论

本次 sufficiency 最稳妥的总结是：

1. `G20` 已经显示出**明确的 steering 效应**。
2. 但 `G20` 的 steering **不具选择性**，因为 `NonRE` 也被显著抬高。
3. `G1` 和 `G5` 不足以支持强充分性结论。
4. 控制方向中 `Orthogonal` 无效，但 `Random_dir` 仍有轻微正效应，因此当前 sufficiency 只能算：
   - **部分支持**
   - 而不是强充分性确认

因此，本次实验支持：

> 存在可操纵的 RE 相关 group direction

但不支持：

> 已经找到能选择性诱发 RE、且几乎不影响 NonRE 的强充分性概念方向

---

## 5. 选择性与副作用分析

### 5.1 这一节的证据等级要降级

`results_selectivity.json` 来自实验内置的生成侧子模块，配置里写明：

- `max_samples = 16`
- `max_new_tokens = 24`
- `lambda = 1.0`
- `modes = ["cond_token", "direction"]`

因此，这一节不是全量结果，而是：

> **基于小规模 generation subset 的补充性证据**

它可以提供方向性观察，但不能当作 strongest evidence。

### 5.2 当前观测到的事实

#### groups

| Group | `mean_generated_re_logit_delta` | `mean_content_retention` | `delta_ttr` | `delta_bigram_repetition` |
|---|---:|---:|---:|---:|
| `G1 / cond_token` | `+0.000` | `1.000` | `+0.000` | `+0.000` |
| `G5 / cond_token` | `+0.000` | `1.000` | `+0.000` | `+0.000` |
| `G20 / cond_token` | `+0.000` | `1.000` | `+0.000` | `+0.000` |

#### controls

| Control | `mean_generated_re_logit_delta` | `mean_content_retention` | `delta_ttr` | `delta_bigram_repetition` |
|---|---:|---:|---:|---:|
| `Orthogonal / direction` | `+0.000` | `1.000` | `+0.000` | `+0.000` |
| `Random_dir / direction` | `-0.004` | `0.944` | `+0.010` | `-0.014` |

### 5.3 如何解释这一节

当前 generation-side 结果最突出的特征是：

- `G1 / G5 / G20` 的 `cond_token` 生成侧 delta 几乎为零
- `Orthogonal` 也几乎为零
- 只有 `Random_dir` 出现了轻微变化

这至少说明：

1. 在本次 generation 子模块中，`cond_token` steering **没有显现出稳定的生成侧收益**。
2. 不能把当前 `G20` 写成“已经在生成层面稳定产生 RE 风格输出”。
3. 由于 groups 和 orthogonal 都接近零，当前这一节更像是在说：
   - “生成侧尚未显示出可复现收益”
   - 而不是“生成侧已经证明选择性很好”

同时，`retention ≈ 1.0`、`delta_ttr ≈ 0`、`delta_bigram_repetition ≈ 0` 也不能直接写成“没有副作用”。因为在这里，**更大的可能是干预本身在 generation 子模块中几乎没有有效改变输出**，所以这些指标自然不变。

### 5.4 选择性结论

这一节最稳妥的判断是：

- 生成侧补充证据 **偏弱**
- 当前没有显示出稳定的 generation-time RE 增益
- 当前也不能据此宣称“副作用很小所以选择性很好”

因此，本节只能支持：

> 当前 sampled generation 子模块未观察到稳定收益

不能支持：

> 已经完成选择性 / 鲁棒性确认

---

## 6. 组结构分析

### 6.1 cumulative top-k：不是单调提升，而是明显波动

`results_group.json` 的 cumulative top-k 曲线显示：

- `K=1`: `+0.016`
- `K=2`: `+0.038`
- `K=3`: `-0.392`
- `K=4`: `+0.270`
- `K=5`: `+0.139`
- `K=6`: `-0.242`
- `K=7`: `+0.710`
- `K=8`: `-0.438`
- `K=9`: `-0.749`
- `K=10`: `-0.321`

这条曲线不是“加 latent 越多越好”，而是：

- 有时新增 latent 会明显增强效果；
- 有时新增 latent 会显著把方向拉反；
- 说明组内并不是纯粹的同向协作结构。

因此，当前 `G20` 不能被解释为 clean additive circuit，更像是：

> 一组带有真实任务信号、但内部方向不完全一致的混合 group

### 6.2 leave-one-out：组内确实存在“有益 latent”和“有害 latent”

LOO 分析是在 **top-10** 子组上完成的，不是全 `G20`。这是解释时必须保留的边界。

关键现象：

- 去掉 `16969` 后，组效应从 `-0.321` 变成 `+0.783`
  - 说明 `16969` 在该 top-10 组合里是**明显破坏性 latent**
- 去掉 `22358` 后，组效应从 `-0.321` 变成 `-1.805`
  - 说明 `22358` 是**明显有益 latent**
- 去掉 `8104` 后，组效应从 `-0.321` 变成 `-0.767`
  - 说明它也在当前组合里带来正贡献

因此，LOO 不是在告诉我们“某一个 latent 完全主导一切”，而是在告诉我们：

- 这组 latent 内部有明确的功能异质性；
- 其中有些 latent 帮忙，有些 latent 拖后腿；
- 组结构尚未 clean 到可以直接叫做“稳定 RE circuit”。

### 6.3 add-one-in：前几步不是稳定增益，而是冲突和补偿交替出现

Add-one-in 结果也支持同样判断：

- 加入 `16969` 时，边际增益是 `-0.429`
- 加入 `5551` 时，又出现 `+0.662`
- 加入 `22358` 时，又是 `-0.382`
- 加入 `3416` 时，出现 `+0.952`
- 加入 `8104` 时，再次 `-1.147`

这说明组内不是“按排名顺次叠加越来越强”，而是：

- 某些 latent 补信息
- 某些 latent 抵消信息
- 整体存在较强的非单调结构

### 6.4 synergy：当前是负 synergy，不是 clean 协同

`results_group.json` 中给出的 synergy 为：

- `full_effect = -0.321`
- `sum_individual_effects = +0.156`
- `synergy_score = -0.477`
- interpretation = `negative (redundant)`

这意味着：

- 单个 latent 的效果加起来，本来是轻微正向；
- 但把它们合到一个组里之后，整体反而变成负向；
- 当前组合不是 super-additive，也不是近似 additive，而是**冗余并伴随相互抵消**。

从研究判级上看，这非常关键。因为这说明当前最强组不是“一个已经整理干净的小电路”，而是“一个仍需继续筛和修的候选混合组”。

### 6.5 bootstrap 稳定性：G20 有稳定核心，但 G5 很不稳定

根据 `selected_groups.json`：

- `stable_G5 = [31133]`
- `stable_G20 = [8104, 6966, 20936, 11658, 22358, 23464, 31133]`

这意味着：

1. `G5` 级别的稳定核心非常弱，只稳定出 1 个 latent。
2. `G20` 至少有 7 个 latent 在 bootstrap 下稳定出现，说明它不是完全随机拼出来的组。

因此，当前组结构最合理的判断是：

- **候选 RE 子空间存在**
- **但其边界还不够干净，内部仍存在明显冲突、冗余与方向混杂**

---

## 7. 综合判级：当前达到什么程度

本项目现有达标性报告把研究目标分成两层：

1. **第 1 层：发现 RE 相关候选特征**
2. **第 2 层：确认强效、典型、稳健、因果可信的 RE 概念表示**

结合本次因果验证结果，可以给出如下判级。

### 7.1 候选特征发现：已明显达成

这一级别不仅在主流程统计/probe 中已成立，在本次因果验证中还得到了进一步强化：

- `G20` ablation 会显著削弱 RE 证据
- 且这种必要性远强于 `Bottom20 / Random20`
- 说明候选 RE latent group 不是随机噪声

结论：

> **已明显达成。**

### 7.2 群组必要性：已有较强支持

这是本次因果验证最强的一条结果链。

理由：

- `G20 / cond_token mean_delta_re ≈ -1.860`
- 控制组接近 0
- `G1` 远弱于 `G20`

结论：

> **已有较强支持。**

### 7.3 群组充分性：仅部分支持

理由：

- `G20` 在三种 steering 下都能稳定提升 RE-logit
- 但 `NonRE` 也同步大幅上升
- `Random_dir` 也有一定正向变化

因此，当前只能说：

> 有 steering 效应，但没有足够强的 RE-selective steering 证据。

结论：

> **仅部分支持。**

### 7.4 选择性 / 鲁棒性：证据不足

理由：

- 生成侧结果本身只来自 `max_samples=16`
- `G1/G5/G20 cond_token` 的 generation-side delta 近乎为 0
- 这不能证明选择性好，只能说明目前未显示稳定生成收益

结论：

> **证据不足。**

### 7.5 单 latent monosemantic RE feature：不支持

理由：

- `G1` 必要性远弱于 `G20`
- `G1` 充分性只在高强度 constant steering 下表面成立，且非选择性
- group 结构中也没有显示出“单 latent 近乎完全主导”的 clean 模式

结论：

> **不支持。**

### 7.6 强因果概念确认：尚未达成

如果要写成“强效、典型、稳健、因果可信的 RE 概念表示已确认”，至少应同时具备：

- 强 necessity
- 至少部分 clean sufficiency
- 控制方向无效
- 非 sampled 的选择性 / 副作用不过度恶化
- 组结构不表现为严重冗余和内耗

当前结果并不满足这组条件。最主要的缺口在：

- sufficiency 选择性不足
- generation-side 证据弱
- group synergy 为负
- group clean 程度不足

结论：

> **尚未达成。**

### 7.7 最终结论话术

最稳妥、最适合复用于汇报或正式报告的一段话是：

> 当前因果验证结果已经明显强于“只有统计显著与 probe 可分”的阶段。它可以较有力地支持：SAE 空间中确实存在一组与 RE 相关、并且对 RE 证据具有明显必要性的 latent 子空间。  
> 但现阶段还不能说这组 latent 已经构成强效、典型、稳健、因果可信的 RE 概念表示。主要原因是：充分性只得到部分支持，选择性证据不足，且 group 内部存在明显冗余与方向冲突。

---

## 8. 边界与后续

### 8.1 本次结果的边界

本次结果在使用时必须保留以下边界：

1. 干预 span 是整条 counselor utterance 的全 token，不是更细粒度局部 span。
2. necessity / sufficiency 使用的是 probe-logit 口径，不是人工临床结果口径。
3. `selectivity` 是实验内置 sampled 子模块，不能当 strongest evidence。
4. `results_group.json` 里的 `leave-one-out / add-one-in / synergy` 实际是 top-10 子组分析，不是全 G20。

### 8.2 下一步最值得推进的方向

1. 扩展非 sampled 的 selectivity / generation evaluation  
   让生成侧结论脱离 `max_samples=16` 的限制。

2. 做 held-out 或 cross-pooling 复核  
   例如在不同 pooling、不同 split 下复核 `G20` 的 necessity / sufficiency 是否稳定。

3. 基于稳定子集而非原始 top-20 做更精简 group 验证  
   当前最值得优先验证的是 `stable_G20` 这 7 个 latent，而不是继续把明显冲突 latent 一起打包。

---

## 9. 一句话总结

这批因果验证结果已经足以把项目从“发现相关 latent”推进到“发现了一个具有明显必要性的候选 RE latent 子空间”；但它还没有把项目推进到“已确认强因果 RE concept circuit”的阶段。当前最准确的定位是：

> **候选子空间已找到，必要性较强，充分性部分成立，选择性与 clean group 结构仍需继续验证。**
