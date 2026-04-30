# `causal_validation_full` 结果分析报告

## 1. 报告对象与一句话结论

本报告分析的对象是 [causal_validation_full](C:/Users/chengyu/Desktop/causal_validation_full) 目录中的完整因果结果：

- `selected_groups.json`
- `results_necessity.json`
- `results_sufficiency.json`
- `results_selectivity.json`
- `results_group.json`
- `summary_tables.md`
- `run.log`

一句话结论：

> 当前结果已经提供了“存在一组 latent 会显著影响当前 RE probe 方向”的必要性与充分性证据，尤其是 `G20`；但这些干预缺乏选择性，会同时把 NonRE 一起推高，且组内存在明显拮抗与负协同，因此这份结果更支持“存在 RE 相关可操纵子空间”，还不支持“已经发现稳定、干净、协同的 RE 因果机制组”。

## 2. 运行概况与实验口径

从 [run.log](C:/Users/chengyu/Desktop/causal_validation_full/run.log) 可以确认：

- 数据集：`data/mi_re`
- pooling：`max`
- lambda：`0.5 / 1.0 / 1.5 / 2.0`
- SAE：`OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x / Llama3_1-8B-Base-L19R-8x`
- 候选 latent 来源：主流程 `candidate_latents.csv`
- 输出已完整生成：necessity / sufficiency / selectivity / group structure 全部存在

`selected_groups.json` 还显示：

- `pooling_method = "max"`
- `probe_baseline.auc = 0.842`
- `probe_baseline.accuracy = 0.747`

这里要特别说明：这个 `probe_baseline` 不是主流程里 dense probe 的基线，而是**在当前因果流程所选 latent 子空间内训练的 probe 基线**。因此它只能用来判断当前组选方向是否可读出 RE，不能和主流程 `dense_probe` 直接等价比较。

## 3. 组选与稳定性：G20 有信号，但“稳定组”并不强

从 [selected_groups.json](C:/Users/chengyu/Desktop/causal_validation_full/selected_groups.json) 看：

- `G1 = [6966]`
- `G5 = [31133, 6966, 16969, 5551, 26276]`
- `G20` 为一组 20 latent 的稳定化扩展列表

Bootstrap 稳定性不算强：

- `stable_G5 = [31133]`
- `stable_G20 = [8104, 6966, 20936, 11658, 22358, 23464, 31133]`

这意味着：

1. `G5` 中真正跨 seed 稳定保留下来的只有 `31133`
2. `G20` 虽然有若干重复出现的核心成员，但其余多数 latent 还是靠“原排序补齐”
3. 当前所谓 `G20` 更像“有一个核心稳定子集 + 一圈不那么稳定的外围成员”

因此，当前结果并不支持把 `G20` 直接说成“稳定 latent 组”。更准确的说法应是：

> 当前已经识别出一个带有稳定核心的候选因果组，但完整 `G20` 的边界和组成仍不够稳定。

## 4. 必要性：G20 的必要性证据很强

必要性结果见 [results_necessity.json](C:/Users/chengyu/Desktop/causal_validation_full/results_necessity.json)。

### 4.1 G1 / G5 / G20 的梯度关系

对 RE 样本的 `mean_delta_re`：

- `G1 zero = -0.092`
- `G5 zero = -0.443`
- `G20 zero = -1.860`

`cond_token` 与 `zero` 几乎一致：

- `G1 cond_token = -0.092`
- `G5 cond_token = -0.443`
- `G20 cond_token = -1.860`

这说明：

- 单个 latent 已有一定必要性，但很弱
- `G5` 开始产生中等规模影响
- `G20` 的必要性非常强，删掉后 RE 方向明显塌陷

就“必要性强度”而言，这是一组相当有力的正面证据。

### 4.2 与对照组比较

必要性最关键的不是绝对数值，而是与控制组对比：

- `Bottom20 zero mean_delta_re = +0.004`
- `Random20 zero mean_delta_re = +0.014`

而 `G20 zero mean_delta_re = -1.860`。

这个对比非常清楚地说明：

> 不是任意删掉 20 个 latent 都会破坏 RE 方向；真正有显著影响的是被选中的那组。

因此，“存在一组对 RE probe 方向具有强必要性的 latent”这个结论是成立的。

### 4.3 必要性里的一个风险点

虽然对 RE 的负向影响很强，但 `G20` 对 NonRE 的影响同时是：

- `mean_delta_nonre = +1.061`（`zero`/`cond_token`）
- `mean_delta_nonre = +1.401`（`mean`）

这意味着：

- 删除这组 latent 后，RE 样本的 RE logit 明显下降
- 同时 NonRE 样本的 RE logit 反而明显上升

这不是“坏结果”，但说明这组 latent 更像是：

> 共同维持当前 RE/NonRE 决策边界的一组方向

而不是只对 RE 样本本身发挥孤立作用的“纯 RE latent”。

## 5. 充分性：G20 能推高 RE，但选择性不足

充分性结果见 [results_sufficiency.json](C:/Users/chengyu/Desktop/causal_validation_full/results_sufficiency.json)。

### 5.1 G20 的充分性是存在的

以 `cond_token` 为例，`G20` 对 RE 样本的 `mean_delta_re` 随强度单调增加：

- `lam_0.5 = +0.041`
- `lam_1.0 = +0.081`
- `lam_1.5 = +0.127`
- `lam_2.0 = +0.171`

`constant` 和 `cond_input` 也有类似趋势，且幅度更强：

- `constant lam_1.0 = +0.103`
- `cond_input lam_1.0 = +0.101`

这说明：

- 当前 `G20` 不只是“删掉会坏”
- 把它加回去、推上去，也确实会把 RE 方向往上抬

因此，“存在一组 latent 对 RE probe 方向具有一定充分性”这个结论也成立。

### 5.2 但它同时会把 NonRE 一起推高

这是当前因果结果最重要的限制。

以 `G20 cond_token` 为例：

- `lam_1.0 mean_delta_re = +0.081`
- `lam_1.0 mean_delta_nonre = +0.227`

`NonRE` 的抬升幅度甚至更大。

同样地，在 `constant` 和 `cond_input` 模式下也存在这个问题：

- `constant lam_1.0`: RE `+0.103`，NonRE `+0.249`
- `cond_input lam_1.0`: RE `+0.101`，NonRE `+0.248`

这说明当前干预缺乏选择性。最稳妥的解释不是：

> “G20 是 RE 专属因果子空间”

而是：

> “G20 是一个会整体推高当前 RE 判别方向的子空间，但它并没有只在 RE 样本上起作用。”

### 5.3 控制方向提供了有用参照

对照方向结果：

- `Orthogonal lam_1.0 mean_delta_re = -0.006`
- `Random_dir lam_1.0 mean_delta_re = +0.020`

这说明：

- 正交方向基本不起作用
- 随机方向会有一点点正向漂移，但明显弱于 `G20`

因此充分性不是纯随机现象，但也还没强到“只有目标方向才有效、且效果完全专属”的程度。

## 6. 选择性与副作用：当前 generation-level 结果不能乐观解读

选择性结果见 [results_selectivity.json](C:/Users/chengyu/Desktop/causal_validation_full/results_selectivity.json)。

### 6.1 表面现象

对 `G1 / G5 / G20 / Orthogonal`：

- `mean_generated_re_logit_delta = 0.0`
- `mean_content_retention = 1.0`
- `delta_ttr = 0.0`
- `delta_bigram_repetition = 0.0`

表面看像是：

- 干预对生成完全没影响
- 也完全没有副作用

但这不能直接这样解读。

### 6.2 为什么这些 generation-level 指标几乎全零

结合 [run_experiment.py](D:/project/NLP_re_dataset_model_base/causal/run_experiment.py) 中 `run_side_effect_evaluation(...)` 的实现，可以看到：

- 只抽取 `max_samples = 16`
- 每条只生成 `max_new_tokens = 24`
- 用的是轻量 proxy：
  - `mean_generated_re_logit_delta`
  - `type-token ratio`
  - `bigram repetition`
  - `content_retention`

同时，结果文件里的 `sample_outputs` 明显显示：

- baseline 和多数组干预后的文本几乎完全相同
- 文本本身质量也比较差，出现重复、乱码式片段和模板化 continuation

因此更合理的解释是：

> 当前 generation-side 评估没有测出稳定差异，更多反映了评估设置过轻、continuation 过短、样本过少、生成本身噪声较大，而不能据此说“完全没有副作用”。

换句话说，这一块结果目前只能算“未观测到稳定差异”，不能算“证实零副作用”。

### 6.3 当前选择性结论

当前选择性最可靠的判断仍然应该建立在 **probe-space sufficiency** 上，而不是这组 generation proxy 上。

从 probe-space 看，结论很明确：

- `G20` 会推高 RE
- 但也会明显推高 NonRE

所以选择性不足是成立的。

## 7. 组结构：当前不是协同组，而是明显互相拮抗

组结构结果见 [results_group.json](C:/Users/chengyu/Desktop/causal_validation_full/results_group.json)。

### 7.1 Cumulative top-K 曲线强烈振荡

`cumulative_topk` 并不是平滑增强，而是剧烈上下波动：

- `k=2`: `+0.038`
- `k=3`: `-0.392`
- `k=4`: `+0.270`
- `k=7`: `+0.710`
- `k=8`: `-0.438`
- `k=14`: `-1.451`
- `k=20`: `-0.322`

这说明组内 latent 并不是一致朝同一个方向推动效果，而是在互相拉扯。

### 7.2 Leave-One-Out 直接暴露了“反向成员”

例如：

- 去掉 `16969` 后，整体效果从 `-0.321` 变成 `+0.783`
  - `delta_loo = -1.105`
- 去掉 `22358` 后，整体效果从 `-0.321` 变成 `-1.805`
  - `delta_loo = +1.484`

这代表：

- 有的 latent 在当前组里是明显的“拖后腿”项
- 有的 latent 才是真正支撑组效应的核心项

因此当前 `G20` 不是一个“人人同向贡献”的 clean group。

### 7.3 Add-One-In 也支持这一点

边际增益正负交替：

- `16969`: `-0.429`
- `5551`: `+0.662`
- `22358`: `-0.382`
- `3416`: `+0.952`
- `8104`: `-1.147`

这基本坐实了一个判断：

> 当前排序选出的组内存在方向不一致的 latent，组不是协同叠加，而是包含明显拮抗项。

### 7.4 Synergy 为负

最终：

- `full_effect = -0.321`
- `sum_individual_effects = +0.156`
- `synergy_score = -0.477`
- `interpretation = negative (redundant)`

这个结果不只是“没有超加性”，而是已经到**明显负协同/相互抵消**的程度。

因此，当前最不应该说的话就是：

> “我们发现了一个干净的、协同的 RE latent 组。”

相反，更准确的说法是：

> 当前 latent 组内部明显存在冗余和相互拮抗，组层级结果尚不干净。

## 8. 这份因果结果支持什么，不支持什么

### 8.1 已支持的结论

1. **存在一组 latent 对当前 RE probe 方向具有强必要性**
   - `G20` ablation 对 RE 样本打击很大
   - 控制组几乎无效

2. **存在一组 latent 对当前 RE probe 方向具有一定充分性**
   - `G20` steering 随 lambda 增大而稳定推高 RE logit

3. **这种效应不是纯随机方向造成的**
   - `Orthogonal` 基本无效
   - `Random_dir` 有弱漂移，但明显不及目标组

### 8.2 暂时不能支持的结论

1. **不支持“G20 是 RE 专属因果子空间”**
   - 因为 NonRE 也被明显推高

2. **不支持“当前组是稳定组”**
   - `stable_G5` 过小
   - `stable_G20` 只保留了部分核心成员

3. **不支持“当前组是干净协同组”**
   - `cumulative_topk` 强烈振荡
   - `leave_one_out` / `add_one_in` 显示强拮抗
   - `synergy_score` 为显著负值

4. **不支持“生成层面已经看到可靠副作用/选择性证据”**
   - 当前 generation-side proxy 几乎没有分辨力

## 9. 下一步建议

### 9.1 值得继续做的

1. **继续做专家评审**
   - 因为现在已经有一组有必要性/充分性信号的候选 latent，可让专家评审判断它们到底像不像 RE 概念

2. **重做组选逻辑**
   - 当前组内明显有方向不一致成员
   - 应优先按“单 latent 方向一致性”和“leave-one-out 贡献”重新筛组

3. **加强选择性评估**
   - 现在最缺的是“是否只推高 RE 而不推高 NonRE”的直接证据

4. **重做更有信息量的生成评估**
   - 增加样本数
   - 增长 continuation
   - 引入更强的语义/人工或 LLM judge 评估，而不是只靠 lexical proxy

### 9.2 当前最稳妥的科研表述

建议用这种表述：

> 当前因果结果支持：存在一组 latent 对当前 RE 判别方向具有明显的必要性与一定充分性；但这组 latent 并不具有足够选择性，且组内存在明显的负协同与不稳定性。因此，现阶段更适合将其解释为“RE 相关、可操纵的候选子空间”，而不是“稳定、干净、机制级的 RE 因果表征”。

## 10. 最终判断

如果问题是“这份因果结果够不够强”，我的判断是：

- **强的地方**：必要性很有说服力，`G20` 不是随机组
- **弱的地方**：选择性差，NonRE 一起上升；组结构很乱，存在明显拮抗

因此这份结果最适合扮演的角色是：

> 为“RE 相关 latent 子空间确实可操纵”提供第一轮因果证据

而不是：

> 直接宣称已经锁定稳定、专属、协同的 RE 因果机制组
