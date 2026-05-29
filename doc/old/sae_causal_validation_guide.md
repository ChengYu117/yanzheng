# SAE 因果性验证技术说明书（面向已筛选出 1 / 5 / 20 个 latent 的情况）

## 适用场景

本文档面向以下研究流程：

1. 你已经基于 SAE 表示，从心理咨询 RE / 非RE 任务中筛选出一组候选 latent  
   - `G1`：top-1 latent  
   - `G5`：top-5 latents  
   - `G20`：top-20 latents
2. 你希望进一步回答：
   - 这些 latent 是否 **必要**（ablation 后 RE 信号下降）
   - 这些 latent 是否 **充分**（steering 后 RE 信号上升）
   - RE 更像是 **单个 latent** 还是 **一组 latent / 小子空间**
   - 这些 latent 是否真的和 RE 因果相关，而不是只对应某些表面词、句长、问号或模板句

---

## 一、文献导向下的总体原则

近年的 SAE 相关工作给出的共同经验不是“找到一个神奇 latent 然后强行点亮”，而是：

1. 先选出稳定的 latent 组
2. 做必要性验证（ablation）
3. 做充分性验证（steering）
4. 做随机组 / Bottom-K / 正交方向对照
5. 做副作用评估
6. 必要时升级到 feature circuit / indirect effect 级别的因果分析

与本说明书最相关的几条论文思路：

- **GradSAE (EMNLP 2025)**  
  用梯度 × 激活估计 latent 对输出的影响，比较 TopK vs BottomK masking，并做 local steering。
- **Breaking Bad Tokens (EMNLP 2025)**  
  系统比较单个 / 多个 SAE feature 的 ablation 和 steering，并强调 `conditional per-token` 比粗暴常量 steering 更稳。
- **SAE-SSV (EMNLP 2025)**  
  通过线性分类器识别一个小型任务相关子空间，再做 steering，并用 random / orthogonal controls 验证。
- **Sparse Feature Circuits (ICLR 2025)**  
  用 approximate indirect effect 构建 feature-level circuit，并用 faithfulness / completeness 做因果评估。

---

## 二、实验对象与表示层固定

在任何因果实验前，必须先固定以下内容，不要边实验边改：

### 2.1 模型与层
- 基础模型：例如 Llama
- SAE 所附着的层：例如某层 residual stream
- SAE checkpoint：固定版本

### 2.2 输入形式
推荐输入为：

- 前一轮来访者话语
- 当前咨询师话语

即：

```text
[client previous utterance] [SEP] [counselor current utterance]
```

### 2.3 标签对齐单位
标签仍然是 **咨询师当前句** 的 RE / 非RE。

### 2.4 句子级聚合
SAE 是 token-level 的，因此需要先把咨询师句子的 token activations 聚合成一句话级表示。

设：
- 咨询师当前句 token span 为 `S`
- 每个 token 的 SAE latent activation 为 `z_t ∈ R^d`

句子级表示：

```math
\bar z = Pool(\{ z_t \mid t \in S \})
```

推荐至少固定并比较两种：

### Sum pooling
```math
\bar z^{(sum)} = \sum_{t \in S} z_t
```

### Binarized sum pooling
```math
\bar z^{(bin)} = 1\left[\sum_{t \in S} z_t > \tau\right]
```

不建议一开始只用 `max` 作为唯一方案，因为 `max` 更容易被单个强触发词绑架。

---

## 三、候选 latent 组的稳定化

不要只根据激活强度选 latent。更稳妥的是结合：

- 句子级 probe 权重
- latent influence score
- bootstrap / 多 seed 稳定性

### 3.1 Probe 选组
在句子级 pooled SAE 表示上训练一个稀疏 probe，例如 L1 logistic regression：

- 输入：句子级 SAE 表示
- 输出：RE / 非RE

得到每个 latent 的 probe 权重 `w_i`。

### 3.2 Influence 选组
可参考 GradSAE 的思想，用梯度 × 激活近似 latent 对输出的影响。

对 latent `i` 的分数可定义为：

```math
g_i = \text{mean over tokens} \left( \frac{\partial \text{RE-logit}}{\partial z_{t,i}} \cdot z_{t,i} \right)
```

### 3.3 综合排序
定义综合分数：

```math
s_i = rank(|w_i|) + rank(g_i)
```

按 `s_i` 排序后取：

- `G1 = top-1`
- `G5 = top-5`
- `G20 = top-20`

### 3.4 稳定性筛选
重复以下过程：

- 不同 seed
- 不同 bootstrap split
- 不同 fold

只保留在较高比例实验中重复入选的 latent，例如：
- 进入 `G5` 的频率 ≥ 60%
- 进入 `G20` 的频率 ≥ 70%

这样可以减少“偶然被选中”的 latent。

---

## 四、必要性验证（Necessity）

目标问题：

> 如果把这组 latent 去掉，RE 信号会不会下降？

---

### 4.1 Zero Ablation

对 counselor sentence span 内的 latent 直接置零：

```math
z'_{t,i} =
\begin{cases}
0, & t \in S,\ i \in G_K \\
z_{t,i}, & \text{otherwise}
\end{cases}
```

其中 `K ∈ {1, 5, 20}`。

然后通过 SAE decoder 将改变量映射回残差流：

```math
\Delta z = z' - z
```

```math
h' = h + W_{dec}(\Delta z)
```

### 4.2 Mean Ablation

如果担心 zero ablation 太 out-of-distribution，可以改成 mean ablation：

```math
z'_{t,i} =
\begin{cases}
\mu_i, & t \in S,\ i \in G_K \\
z_{t,i}, & \text{otherwise}
\end{cases}
```

其中 `μ_i` 是参考集上的平均激活。

### 4.3 Conditional Token Ablation

只在 counselor 句子中、且该 latent 原本激活超过阈值时才消融：

```math
z'_{t,i} =
\begin{cases}
0, & t \in S,\ i \in G_K,\ z_{t,i} > \tau_i \\
z_{t,i}, & \text{otherwise}
\end{cases}
```

这种做法更细粒度，更不容易影响无关 token。

### 4.4 必须做的对照

#### Bottom-K 对照
选最低影响的一组 latent 进行同样的 ablation。

#### 随机组对照
随机抽取与 `G1/G5/G20` 相同数量、激活频率相近的 latent。

#### Matched lexical 对照
选一些“看起来像 RE，但标注不是 RE”的 hard negatives，检查 ablation 是否仍只影响真正的 RE，而不是表面词汇。

### 4.5 必要性指标

#### 分类型指标
- RE logit
- RE probability
- RE/非RE 分类准确率或 F1

#### 生成型指标
- 外部 RE judge 的分数
- 人工评分

定义效果：

```math
\Delta^{ablate}_{RE}(G_K) =
Score(x) - Score(x^{ablate(G_K)})
```

如果 `G5` 或 `G20` 的 ablation 显著打掉 RE，而随机组 / Bottom-K 不起作用，就说明这组 latent 具有必要性。

---

## 五、充分性验证（Sufficiency）

目标问题：

> 只把这些 latent 往上推，输出会不会更像 RE？

---

### 5.1 单 latent steering

对于 `G1 = {i}`，可直接沿 decoder vector 干预：

```math
h'_t = h_t + \lambda v_i
```

其中：
- `v_i = W_dec[i]`
- `\lambda` 是 steering 强度

### 5.2 组级 steering：不要默认等权

对于 `G5` 或 `G20`，不建议所有 latent 等强度同时点亮。  
推荐：

```math
h'_t = h_t + \lambda \sum_{i \in G_K} \alpha_i v_i
```

其中 `α_i` 可来自：

- probe 权重归一化
- influence score 归一化
- probe × influence 联合归一化

### 5.3 三种 steering 版本

#### A. Constant Steering
对 counselor span 全部 token 施加同样的 steering。

优点：
- 足够强
- 最适合检验“是否具有充分性”

缺点：
- 副作用最大
- 容易污染整个句子

#### B. Conditional Per-Input Steering
若这句话中任一 token 触发某个目标 latent，则对整句施加 steering。

#### C. Conditional Per-Token Steering
只在 counselor span 内、且该 token 上对应 latent 超过阈值时施加 steering。

这是最推荐优先实现的版本，因为它最细，也最不容易把整句搞坏。

### 5.4 强度扫描
建议扫描：

```text
0.25, 0.5, 1.0, 1.5, 2.0, 2.5
```

记录每个强度下：
- RE 提升
- 副作用变化

### 5.5 充分性指标

定义：

```math
\Delta^{steer}_{RE}(G_K) =
Score(x^{steer(G_K)}) - Score(x)
```

如果 `G5` 或 `G20` steering 后 RE 分数稳定上升，而随机组 / 正交方向不行，则说明这组 latent 具有充分性。

---

## 六、选择性与副作用验证（Selectivity）

只看 RE 提升不够，必须证明：

> 你改的是 RE，而不是把模型整体语言能力搞坏了。

### 6.1 正交方向 / 随机方向对照
如果组方向是：

```math
u = \sum_{i \in G_K} \alpha_i v_i
```

则构造：

- 与 `u` 正交的方向
- 随机方向

对它们施加相同强度 steering，作为对照。

### 6.2 输出质量指标

建议至少评估：

- Fluency
- Coherence
- Content retention（是否仍回应上一句来访者内容）
- Lexical diversity
- 是否出现重复、语义崩坏、无关转移

### 6.3 非目标能力控制
找一个非 RE 的对话任务或一般语言任务，检查 steering / ablation 后是否明显退化。

### 6.4 结果拆分
可把输出分成三类：

- **Success**：更 RE，且仍然自然
- **Retained**：原属性仍然很强，RE 提升有限
- **Disorder**：输出开始崩坏、不连贯、与上下文脱节

---

## 七、判断 RE 是单 latent 还是分布式 latent group

这部分非常重要。

---

### 7.1 Cumulative Top-K 曲线
按 latent 排名逐步加入：

- top-1
- top-2
- ...
- top-20

观察 RE 效果随 K 的变化。

如果表现为：

- top-1 有效
- top-5 明显更好
- top-20 继续提升但边际递减

那通常说明 RE 更像是一个小组 latent 承载的概念。

### 7.2 Leave-One-Out
对 `G20` 中每个 latent `i`：

```math
\Delta^{LOO}_i = Effect(G_{20}) - Effect(G_{20} \setminus \{i\})
```

如果没有任何一个 latent 单独决定成败，而是多个 latent 都有中小贡献，则更支持 distributed representation。

### 7.3 Add-One-In
从 `G1` 开始逐个加：

```math
\Delta^{add}_i = Effect(G_{\le i}) - Effect(G_{< i})
```

看新增 latent 是否不断补充 RE 信息。

### 7.4 Synergy / Redundancy
定义：

```math
Synergy(G_K) = Effect(G_K) - \sum_{i \in G_K} Effect(\{i\})
```

- 若 `Synergy > 0`：组合作用强于单独求和，说明组级机制成立
- 若接近 0：说明组内可能只是简单叠加
- 若小于 0：说明组内可能冗余或相互冲突

---

## 八、升级版：Feature Circuit 级因果验证

如果你发现 `G20` 中有些 latent 看起来只是相关，不确定谁真因果，可以进一步升级到 circuit 分析。

### 8.1 定义目标 metric
可选：

#### Probe-logit 版本
```math
m(x) = logit_{RE}(x) - logit_{nonRE}(x)
```

#### Judge-score 版本
```math
m(x) = REJudge(x)
```

### 8.2 计算 node-level indirect effect
对 counselor sentence span 内各 latent 节点计算 effect：

- attribution patching
- integrated gradients
- zero ablation 近似

### 8.3 计算 edge-level indirect effect
追踪哪些上游 latent 影响哪些下游 latent / logits。

### 8.4 构造 RE feature circuit
保留：

- 高 node effect 的 latent
- 高 edge effect 的边

最后得到 RE circuit，而不是仅仅一个相关的 latent set。

### 8.5 Faithfulness / Completeness

#### Faithfulness
只保留 circuit，模型还能保留多少 RE 行为？

#### Completeness
把 circuit 拿掉，RE 行为还剩多少？

如果：
- faithfulness 高
- completeness 低

则说明这个 circuit 确实解释了大部分 RE 机制。

---

## 九、推荐的代码实现顺序

### 阶段 1：最小可运行版本
先实现这三类：

1. `G1 / G5 / G20` 的 zero ablation
2. `G1 / G5 / G20` 的 conditional per-token steering
3. 随机组 / Bottom-K / orthogonal 对照

### 阶段 2：增强版本
再补：

4. mean ablation
5. leave-one-out / add-one-in / cumulative top-k
6. fluency / coherence / content-retention 评估

### 阶段 3：高阶版本
最后补：

7. node-level indirect effect
8. edge-level indirect effect
9. circuit faithfulness / completeness

---

## 十、推荐的工程模块划分

建议至少拆成这几个文件：

```text
data.py
selection.py
intervention.py
evaluation.py
run_experiment.py
```

### 10.1 `data.py`
负责：
- 加载对话数据
- 构造 `[client_prev] [SEP] [counselor_current]`
- 保存 counselor span mask
- 生成 RE / 非RE 标签

### 10.2 `selection.py`
负责：
- SAE 编码
- 句子级 pooling
- probe 训练
- influence score 计算
- 生成 `G1 / G5 / G20`

### 10.3 `intervention.py`
负责：
- zero / mean ablation
- single latent steering
- weighted group steering
- conditional per-input steering
- conditional per-token steering

### 10.4 `evaluation.py`
负责：
- RE classifier 分数
- 外部 RE judge
- fluency / coherence / retention
- random / Bottom-K / orthogonal controls

### 10.5 `run_experiment.py`
负责：
- 遍历 `K ∈ {1,5,20}`
- 遍历强度 `λ`
- 执行 ablation / steering
- 汇总结果表

---

## 十一、核心接口建议

```python
class SAEInterventionRunner:
    def encode(self, resid):
        pass

    def sentence_pool(self, z, span_mask, mode="sum"):
        pass

    def rank_latents(self, pooled_z, labels, method="probe+influence"):
        pass

    def ablate_latents(self, z, span_mask, latent_ids, mode="zero", ref=None):
        pass

    def steer_latents(
        self,
        resid,
        z,
        span_mask,
        latent_ids,
        weights,
        mode="cond_token",
        threshold=None,
        strength=1.0,
    ):
        pass

    def decode_delta(self, delta_z):
        pass

    def eval_re_classifier(self, inputs):
        pass

    def eval_re_judge(self, dialogues):
        pass

    def eval_quality(self, outputs):
        pass
```

---

## 十二、最小伪代码

### 12.1 Latent Ablation

```python
z = sae.encode(resid)
z_new = z.clone()
z_new[:, span_mask, latent_ids] = 0   # zero ablation

delta_z = z_new - z
delta_resid = sae.decode(delta_z)
resid_new = resid + delta_resid
```

### 12.2 Weighted Group Steering

```python
delta = 0
for i, w in zip(latent_ids, weights):
    delta = delta + w * sae.W_dec[i]

resid_new = resid.clone()
resid_new[:, span_mask, :] += strength * delta
```

### 12.3 Conditional Per-Token Steering

```python
active = (z[:, :, latent_ids] > threshold[latent_ids]) & span_mask[:, :, None]

delta_tok = 0
for j, i in enumerate(latent_ids):
    delta_tok += active[:, :, j].float().unsqueeze(-1) * weights[j] * sae.W_dec[i]

resid_new = resid + strength * delta_tok
```

---

## 十三、最终应产出的结果表

### 表 1：必要性
- K = 1 / 5 / 20
- zero / mean / cond-token ablation
- RE 分数下降
- Bottom-K / random 对照

### 表 2：充分性
- K = 1 / 5 / 20
- constant / cond-input / cond-token steering
- RE 分数提升
- orthogonal / random 对照

### 表 3：副作用
- fluency
- coherence
- content retention
- 非目标能力变化

### 表 4：组结构
- cumulative top-k
- leave-one-out
- add-one-in
- synergy

### 表 5：若做 circuit
- faithfulness
- completeness
- node IE
- edge IE

---

## 十四、推荐的结论判据

如果实验结果呈现以下模式，就可以比较有力地支持：

> RE 更像是一组 latent / 一个小子空间，而不是单个 latent。

### 支持 distributed RE representation 的典型模式
- `G1` 有效，但效果有限
- `G5` 显著强于 `G1`
- `G20` 继续提升有限，或提升伴随副作用
- leave-one-out 没有发现单个绝对支配的 latent
- add-one-in 显示前几个 latent 不断补充信息
- random / Bottom-K / orthogonal 对照都无效
- conditional per-token steering 明显优于粗暴 constant steering
- circuit faithfulness 高、completeness 低

---

## 十五、实施建议总结

如果你现在准备开始写代码，最推荐的顺序是：

1. 固定句子级 pooling
2. 用 probe + influence 稳定化选出 `G1/G5/G20`
3. 做 zero ablation
4. 做 conditional per-token steering
5. 做 random / Bottom-K / orthogonal controls
6. 做 cumulative top-k / leave-one-out / synergy
7. 若结果稳定，再上 indirect effect / circuit

这样可以最快判断：

- RE 到底是不是一个真正可干预的 SAE feature group
- 这个 group 是不是因果相关，而不是只在数据集上“看起来相关”

---

## 十六、一句话总括

不要只问：

> 哪个 latent 是 RE？

更应该问：

> 哪个 **稳定、稀疏、可干预的小 latent group / 子空间**，最能解释并控制 RE？

这才是当前 SAE 可解释性工作里更符合文献趋势的研究方式。
