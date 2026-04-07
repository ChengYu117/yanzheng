# SAE 本地实现偏差审计与结构指标归因结论

## 1. 先给结论

对当前项目而言，答案不是“完全是数据集原因”，也不是“完全是模型坏了”。

更准确的最终归因是：

> 当前结构指标不理想，**主因更像本地实现和官方技术口径没有完全同步**；  
> **数据集分布差异是次因**，主要影响 dead ratio、活跃特征覆盖率和任务内 feature 使用范围；  
> 两者共同作用，才形成了现在“论文里很强，但本地 raw-space 指标偏差明显”的现象。

如果只问一句最核心的话：

> **不是纯数据集问题。**  
> 当前本地实现至少存在“归一化/反归一化闭环不完整”和“未同步 decoder-norm-aware 稀疏激活”这两类高风险偏差，它们足以系统性拉低 raw-space `EV/FVU` 和 `CE/KL`。

---

## 2. 本地实现链路摘要

当前项目的本地 SAE 调用逻辑是：

1. 从底模抓取 `blocks.19.hook_resid_post` 的原始激活
2. 把原始激活直接送给本地 `SparseAutoencoder.forward(...)`
3. 在 `forward` 内部先做输入归一化
4. 再 `encode -> decode`
5. 直接把 `decode` 输出：
   - 拿去和原始激活算 `MSE / EV / FVU`
   - 或直接替换回模型计算 `CE/KL`

这条链路的关键问题是：

- 它只显式做了“输入归一化”
- 没有显式做“输出反归一化”
- 也没有同步 OpenMOSS 官方的 `sparsity_include_decoder_norm`

因此，本地链路更像是：

> “半套官方 dataset-wise 推理链”

而不是：

> “完整官方 dataset-wise 链路”  
> 或  
> “完整官方 inference-mode 链路”

---

## 3. 官方实现 vs 本地实现对齐表

| 技术点 | 官方口径 | 本地实现 | 是否一致 | 证据来源 | 可能影响的指标 |
| --- | --- | --- | --- | --- | --- |
| hook 位置 | `hook_point_in/out = blocks.19.hook_resid_post` | `config/sae_config.json` 也是 `blocks.19.hook_resid_post` | 一致 | HF `hyperparams.json`；本地 `config/sae_config.json` | 不是当前主要问题 |
| SAE 宽度 | `d_model=4096`, `d_sae=32768` | 一致 | 一致 | HF `hyperparams.json`；本地配置 | 不是当前主要问题 |
| 激活函数 | `act_fn=jumprelu` | 本地也是 JumpReLU | 基本一致 | HF `hyperparams.json`；`src/nlp_re_base/sae.py` | 不是当前主要问题 |
| `top_k=50` 元数据 | HF 中存在 | 本地未使用 | 暂不判为偏差 | HF `hyperparams.json`；本地 `sae.py` | 当前证据不足以把它当主因 |
| 输入归一化模式 | 官方支持 `dataset-wise` | 本地 `encode()` 内部按固定范数归一化 | 部分一致 | 官方 `sparse_dictionary.py`；本地 `sae.py:72-90` | 会影响 latent 激活口径 |
| 输出反归一化 | 官方 `dataset-wise` 模式需要外层 `denormalize_activations(...)`，或切到 `inference` 模式 | 本地没有显式输出反归一化 | **不一致** | 官方 `sparse_dictionary.py:421-487`；本地 `sae.py:93-118` | **raw-space EV/FVU/MSE、CE/KL** |
| `fold_activation_scale` / inference mode | 官方可通过 `standardize_parameters_of_dataset_norm()` 折叠尺度 | 本地没有同类步骤 | **不一致** | 官方 `sparse_dictionary.py:215-255, 593-599, 602-617` | **raw-space fidelity、替换回模 fidelity** |
| `sparsity_include_decoder_norm=true` | 官方编码时按 decoder norm 重标定 `hidden_pre` 与 `feature_acts` | 本地完全未实现 | **不一致** | HF `hyperparams.json`；官方 `sae.py:333-341`；本地 `sae.py:89-91` | **L0、dead ratio、latent 排序、probe、TPP、组选择** |
| `apply_decoder_bias_to_pre_encoder=false` | 本 checkpoint 不应把 decoder bias 回灌到 encoder 前 | 本地有 `b_pre` 参数，但 checkpoint 缺失后零初始化 | 基本无害差异 | HF `hyperparams.json`；本地 `sae.py:55` 和缺失键处理 | 不是主因 |
| checkpoint 缺失 `b_pre` | 官方模型 schema不依赖这一项作为主路径 | 本地缺失后零初始化 | 可接受 | 本地加载日志与 `sae.py` | 不是主因 |
| 结构评估口径 | 论文评估基于后处理后的正式口径 | 本地 `run_structural_evaluation()` 先跑 sample，再 full 覆写 JSON | **口径不一致** | `run_sae_evaluation.py:210, 258-280` | 日志易误导，sample/full 混淆 |
| `metric_definition_version=2` 结果格式 | 结果文件已区分 `raw` 与 `normalized` | 当前工作区源码不生成这些字段 | **结果生成器版本与源码不一致** | `sae_eval_full/metrics_structural.json`；当前 `eval_structural.py` | 影响结果解释，但不是 fidelity 主因 |

---

## 4. 结构指标归因表

### 4.1 当前关键现象

`sae_eval_full/metrics_structural.json` 给出的全量结果是：

- `dead_ratio = 0.5642`
- `l0_mean = 25.45`
- `raw explained_variance = -0.9632`
- `raw fvu = 1.9632`
- `normalized explained_variance = 0.4836`
- `normalized fvu = 0.5164`
- `ce_loss_delta = 2.2507`
- `kl_divergence = 2.9095`

这组结果里最关键的异常不是死亡率，而是：

> raw-space 和 normalized-space 的 fidelity 明显分裂。

### 4.2 逐指标归因

| 指标 | 当前现象 | 可能由数据集造成 | 可能由实现偏差造成 | 当前证据强度 | 暂定结论 |
| --- | --- | --- | --- | --- | --- |
| `dead_ratio` | `56.42%` | 是。窄域咨询语料会降低 feature 覆盖面 | 是。忽略 `decoder_norm` 会改变 feature 激活分布 | 中等 | 共同作用，但数据集因素更大 |
| `l0_mean` | `25.45` | 是。窄域任务通常只激活少量 latent | 是。缺失 `sparsity_include_decoder_norm` 会改稀疏度 | 中等 | 共同作用 |
| raw `EV/FVU` | `EV<0`, `FVU>1` | 单靠数据集一般不足以解释到这么差 | **是，高度可疑**。若 decode 仍处于 normalized space，raw-space 比较会系统性变差 | **高** | **主要由实现/口径问题解释** |
| normalized `EV/FVU` | `EV≈0.48`, `FVU≈0.52` | 是。窄域任务可能不如论文大语料稳 | 是，但至少说明在归一化空间里不至于完全坏掉 | 高 | 说明 SAE 有一定重构能力，问题不只是数据集 |
| `MSE` | raw-space 较大 | 可能有影响 | 是。空间不一致会直接放大 | 高 | 主要看作口径问题 |
| `CE/KL` | `delta≈2.25`, `KL≈2.91` | 有影响，但一般是次因 | **是，高度可疑**。若替换回模型的是 normalized-space recon，会严重恶化输出分布 | **高** | **主要由实现/口径问题解释** |
| 功能侧 probe 很强 | `sparse_probe_k20 AUC≈0.912`, `dense_probe≈0.971` | 与窄域任务高度相关 | 说明 SAE latent 仍抓到了任务信号 | 高 | SAE 不是“完全坏了” |

---

## 5. 为什么“完全是数据集原因”站不住

### 5.1 数据集确实会影响什么

你的 `RE / NonRE` 数据集有这些特点：

- 规模小：`799 + 799 = 1598` 条
- 同一大领域：都是心理咨询访谈
- 语义范围窄：不是开放域自然语料

这会合理影响：

1. `dead_ratio`
   - 很多在大语料里有用的 latent，在你的窄域任务里不会被激活
2. `L0`
   - 每个 token 激活的 latent 数可能更少
3. 活跃 latent 覆盖范围
   - 任务只会用到大字典中的一小部分特征

因此，如果只看到“死亡率偏高”，完全可以先归因为窄域任务。

### 5.2 但数据集不能单独解释 raw/normalized 的强烈分裂

当前最难用“纯数据集差异”解释的是：

- normalized-space `EV/FVU` 有明显改善
- raw-space `EV/FVU` 却极差
- `CE/KL` 也明显恶化

如果只是数据集窄，常见情况更像：

- dead ratio 高
- 活跃特征少
- 但活跃子空间内的重构不一定会出现“normalized 尚可、raw 极差”这种强分裂

而你这里恰好出现了这种分裂，这更像：

> 本地实现把“归一化空间里的重构”直接拿去和“原始空间激活”比较或替换。

这类现象是实现/评估口径问题的典型信号，不是数据集差异的典型信号。

---

## 6. 为什么“完全是实现错误”也不准确

也不能把所有问题都归为实现错误，原因有三点：

1. 功能侧仍然很强
   - `1138` 个 FDR 显著 latent
   - `sparse_probe_k20 AUC ≈ 0.912`
   - 说明 SAE latent 不是随机噪声，而是在抓任务相关方向
2. full-dataset `dead_ratio = 56.42%`
   - 这个数不算“结构完全坏掉”
   - 比 sample 日志里的 `93.81%` 要健康得多
3. 你的任务确实是窄域
   - 这会自然降低一些结构统计的漂亮程度

因此，更准确的说法是：

> 本地实现不是“完全错误到不能用”，而是“在关键 fidelity 口径上没有和官方技术路线完全同步”，从而把 raw-space 结构指标系统性拉差了。

---

## 7. 结果文件与当前源码版本的额外问题

`sae_eval_full/metrics_structural.json` 中有这些字段：

- `metric_definition_version = 2`
- `structural_scope = "full_dataset"`
- `space_metrics.raw`
- `space_metrics.normalized`

但当前工作区里的 `src/nlp_re_base/eval_structural.py` 并不会生成这些字段。

这说明：

> 生成 `sae_eval_full` 的结构评估脚本版本，比当前工作区源码更新。

这不是导致指标差的根本原因，但会带来两个解释风险：

1. 你不能把 `sae_eval_full` 的 JSON 字段逐项等同于当前源码行为
2. 你也不能只看当前源码，就忽略结果文件其实已经在区分 raw-space 和 normalized-space

不过，这个版本差异本身反而支持一个判断：

> 生成结果的人已经意识到“raw-space 与 normalized-space 需要分开报告”，这和我们现在的归因方向是一致的。

---

## 8. 最终结论矩阵

| 归因类别 | 是否成立 | 结论 |
| --- | --- | --- |
| 主要由数据集差异解释 | 否 | 只能解释 dead ratio、活跃覆盖率偏低，不能单独解释 raw/normalized fidelity 分裂与较差的 CE/KL |
| 主要由实现未同步解释 | 是 | 归一化/反归一化闭环不完整，且缺失 `sparsity_include_decoder_norm`，这是最强主因 |
| 数据集 + 实现共同作用 | 是 | 当前最合理的总体归因 |
| 现有证据不足 | 否 | 现有证据已足以排除“纯数据集原因”这个说法 |

---

## 9. 最后的直接回答

### 9.1 为什么论文里该 SAE 很强，而本地结构指标却不理想？

因为你现在本地跑的，并不是“论文中那个完整官方评估口径的 SAE 推理链”。

更准确地说，你拿到的是同一份 checkpoint，但本地实现至少缺了两类关键同步：

1. **归一化/反归一化闭环没有完整对齐官方实现**
2. **`sparsity_include_decoder_norm=true` 没有在本地编码路径里实现**

再叠加你的任务数据本身是窄域小样本，所以最终表现成：

- 功能侧仍然有明显任务信号
- 但 raw-space 结构指标明显比论文口径难看

### 9.2 是否“完全是数据集原因”？

不是。

数据集差异是次因，不是主因。

它可以解释：

- 为什么 dead ratio 不会像大语料 benchmark 那么漂亮
- 为什么很多 latent 在你的任务里不活跃

但它不足以独立解释：

- raw-space `EV/FVU` 极差
- normalized-space `EV/FVU` 相对明显改善
- `CE/KL` 同步明显恶化

### 9.3 最终主因和次因怎么排？

最稳妥的排序是：

1. **主因：实现/评估口径未完全同步官方技术路线**
   - 输入输出尺度闭环不完整
   - 未同步 decoder-norm-aware 稀疏语义
2. **次因：数据集分布与论文评估分布差异很大**
   - 心理咨询窄域
   - 任务语义覆盖面小
3. **额外解释风险：结果文件与当前源码版本不完全一致**
   - 影响的是结果解释的清晰度，不是原始 fidelity 差的核心成因

---

## 10. 审计后的最保守判断

如果把这件事压缩成一句最严谨的话：

> 当前“论文上强、本地结构指标差”的现象，**不能被解释为纯数据集问题**。  
> 现有证据更支持：**本地实现没有完整同步 OpenMOSS 官方 SAE 的推理与评估口径，尤其是归一化闭环和 decoder-norm-aware 激活机制**；窄域数据集进一步放大了 dead ratio 与特征覆盖范围方面的差异。

因此，后续如果要真正接近论文口径，优先级应该是：

1. 先修实现与评估闭环
2. 再讨论数据集适配性

而不是反过来。

