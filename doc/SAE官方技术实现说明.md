# SAE 官方技术实现说明

## 1. 文档目的

本文档只回答一个问题：

> `OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x / Llama3_1-8B-Base-L19R-8x` 这份 SAE 在论文、Hugging Face 和 OpenMOSS 官方实现里到底是怎么工作的？

它不分析我们本地代码是否偏差，那部分放在另一份文档里。

---

## 2. 官方来源地图

本说明只使用以下一手或准一手来源：

1. 论文：`Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders`
   - PDF: https://lingjiechen2.github.io/data/llama_scope_EXTRACTING%20MILLIONS%20OF%20FEATURES%20FROM%20LLAMA-3.1-8B%20WITH%20SPARSE%20AUTOENCODERS.pdf
   - arXiv 摘要页: https://arxiv.org/abs/2410.20526
2. Hugging Face checkpoint：
   - 仓库: https://huggingface.co/OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x
   - 本 checkpoint 的 `hyperparams.json`:
     https://huggingface.co/OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x/resolve/main/Llama3_1-8B-Base-L19R-8x/hyperparams.json
3. OpenMOSS 官方实现：
   - 模型实现: https://raw.githubusercontent.com/OpenMOSS/Language-Model-SAEs/refs/heads/dev/src/lm_saes/models/sae.py
   - 抽象基类与加载逻辑:
     https://raw.githubusercontent.com/OpenMOSS/Language-Model-SAEs/refs/heads/dev/src/lm_saes/models/sparse_dictionary.py
   - 项目主页 / 官方说明: https://openmoss.ai/en/language-model-SAEs/

---

## 3. 这份 checkpoint 的官方身份

根据 Hugging Face 上的 `hyperparams.json`，`Llama3_1-8B-Base-L19R-8x` 的关键身份如下：

| 项 | 官方值 | 含义 |
| --- | --- | --- |
| `hook_point_in` | `blocks.19.hook_resid_post` | 输入来自 Llama-3.1-8B 第 19 层 block 后的 residual stream |
| `hook_point_out` | `blocks.19.hook_resid_post` | 输出也重构到同一位置，说明这是同层 SAE，不是 transcoder |
| `d_model` | `4096` | 底模该层 residual 维度 |
| `d_sae` | `32768` | SAE 字典宽度 |
| `expansion_factor` | `8` | `32768 / 4096 = 8x` |
| `act_fn` | `jumprelu` | 推理期激活函数口径是 JumpReLU |
| `jump_relu_threshold` | `0.52734375` | JumpReLU 门槛 |
| `norm_activation` | `dataset-wise` | 官方配置里保留了 dataset-wise activation norm 机制 |
| `dataset_average_activation_norm.in/out` | `17.125 / 17.125` | 该 checkpoint 对应的数据集平均激活范数 |
| `use_decoder_bias` | `true` | 启用 decoder bias |
| `apply_decoder_bias_to_pre_encoder` | `false` | 不把 decoder bias 回灌到 encoder 前 |
| `sparsity_include_decoder_norm` | `true` | 稀疏激活计算显式包含 decoder norm |
| `top_k` | `50` | 训练/后处理谱系中保留了 TopK 相关元数据 |

这里有两个容易误读的点：

1. 论文整体叙述里，Llama Scope 的主系列是以 TopK SAE 为主，并在评估前做 post-training processing。
2. 但这个具体 checkpoint 的推理激活函数字段是 `jumprelu`，同时保留了 `top_k=50` 这类谱系元数据。

最稳妥的理解是：

> 这份公开 checkpoint 不是“最朴素的标准 SAE”，而是 Llama Scope 体系下经过后处理、带有 dataset norm 与 decoder norm 语义的公开推理版本。

---

## 4. 论文里的官方技术口径

### 4.1 论文中的训练与评估对象

论文核心目标不是只训一个 SAE，而是：

- 在 Llama-3.1-8B 的多个层位大规模训练 SAE
- 在 held-out 自然语料上评估重构、稀疏性、解释性与下游可用性
- 最终发布一套可直接用于分析的 checkpoint 集合

论文报告的“强性能”主要建立在：

- 大规模自然语料分布
- 与训练分布相近的 held-out 评估
- 经过 post-training processing 后的 SAE

也就是说，论文里的“性能很强”并不等价于：

> 在任何窄域小数据集上，直接把任意本地实现接到模型里，raw-space fidelity 都应同样漂亮。

### 4.2 论文中的 post-training processing

论文第 3.3.4 节明确写到，作者在评估前会做 post-training processing。其目标包括：

- 让 decoder 列向量达到单位范数
- 对特征抑制问题做 pruning / fine-tuning 处理
- 使 SAE 在评估时“以原始模型激活为输入，并重构原始模型激活为输出”

官方 openmoss.ai 说明页也明确写了同样的口径：经过 post-training processing 后，模型应直接作用于原始激活空间，而不是让用户手动在评估阶段混用训练态与推理态尺度。

因此，论文里的高 fidelity 不是“裸 checkpoint + 任意自定义前向”天然保证的，而是建立在：

1. 后处理后的参数语义
2. 正确的推理期归一化/反归一化口径
3. 与论文一致的评估流程

---

## 5. OpenMOSS 官方实现中的关键机制

下面只看官方代码，不看论文文字。

### 5.1 官方抽象并不是“输入直接进 SAE，输出直接拿来比”

`SparseDictionary` 抽象基类明确区分了四种 `norm_activation` 模式：

- `token-wise`
- `batch-wise`
- `dataset-wise`
- `inference`

官方代码注释给出的含义是：

- `dataset-wise`：
  - 先按数据集平均激活范数对输入激活做缩放
  - 调用者在外层负责做 `normalize_activations(...)`
  - 解码后若要回到原始空间，需要再做 `denormalize_activations(...)`
- `inference`：
  - 不再对输入做归一化
  - 因为权重和 bias 已被 `standardize_parameters_of_dataset_norm()` 折叠过

这点在官方代码里写得非常明确：

- `compute_norm_factor(...)` 把 `dataset-wise` 和 `inference` 明确分开
- `normalize_activations(...)` 与 `denormalize_activations(...)` 是成对出现的
- `SparseDictionary.forward(...)` 的注释明确要求：输入在调用 `forward` 前应先完成归一化

这说明官方技术口径并不是：

> 任意 raw residual 直接丢给 SAE，再把 decode 输出直接当 raw residual 使用。

### 5.2 官方支持两条正确的推理路径

OpenMOSS 官方实现支持两种正确的推理路径：

#### 路径 A：保持 `dataset-wise` 模式

1. 先对输入激活做 `normalize_activations(...)`
2. 再执行 `encode / decode`
3. 最后对重构结果做 `denormalize_activations(...)`

这条路径下，模型参数本身不折叠 activation scale，但调用者必须负责输入输出尺度闭环。

#### 路径 B：切到 `inference` 模式

1. 调用 `standardize_parameters_of_dataset_norm()`
2. 把数据集平均激活范数折叠进权重与 bias
3. 此后输入可直接使用原始激活，输出也直接处于原始空间

官方 `SparseDictionary.from_pretrained(...)` 里提供了 `fold_activation_scale` 参数。设置为 `True` 时，就会在加载后调用 `standardize_parameters_of_dataset_norm()`。

因此，官方实现语义是：

> 你要么显式做“归一化 -> SAE -> 反归一化”，要么把 activation scale 折叠进参数后直接跑 raw-space。两者必须二选一，且输入输出口径要闭环。

### 5.3 官方 SAE 编码时还会把 decoder norm 纳入稀疏激活

这是本项目最容易忽视、但很关键的一点。

在 OpenMOSS 官方 `sae.py` 中，编码路径不是简单的：

```text
hidden_pre = x @ W_E + b_E
feature_acts = JumpReLU(hidden_pre)
```

而是：

1. 先算 `hidden_pre`
2. 如果 `sparsity_include_decoder_norm=true`
   - 先把 `hidden_pre` 乘上 `decoder_norm()`
   - 再过激活函数
   - 再把 `feature_acts` 和 `hidden_pre` 除回 `decoder_norm()`

本 checkpoint 的 `hyperparams.json` 明确写了：

```text
"sparsity_include_decoder_norm": true
```

这意味着该 SAE 的“哪些 feature 会被激活、激活有多大”并不是只由 `W_E / b_E / threshold` 决定，还显式依赖 decoder 范数。

如果一个本地实现忽略了这一步，那么：

- latent 稀疏模式会变
- L0 / dead ratio 会变
- 单 latent 的激活排序也可能变
- 后续 probe、MaxAct、TPP、因果组选择都会受影响

### 5.4 官方 decode 本身不负责反归一化

官方 `sae.py` 的 `decode(...)` 很朴素：

- `reconstructed = feature_acts @ W_D`
- 如果启用 decoder bias，再加 `b_D`

它不会在 decoder 内部自动做 dataset-wise 的输出反缩放。

这和前面第 5.2 节完全一致：

- 如果还在 `dataset-wise` 模式，外层必须负责 `denormalize_activations(...)`
- 如果已经 `fold_activation_scale=True` 切到了 `inference` 模式，那权重/偏置已被折叠，decoder 输出天然就是原始尺度

---

## 6. 论文中的评估口径到底强在什么地方

### 6.1 论文的结构指标不是在窄域小样本上定义的

论文报告强性能时，使用的是大规模 held-out 语料评估。例如：

- 50M held-out tokens
- reconstruction / sparsity / language modeling fidelity 等联合评估

所以论文里“这个 SAE 很强”更准确地指：

> 这份 SAE 在其训练和评估所依赖的大语料自然分布上，经官方后处理后，具有较强的 reconstruction-fidelity / sparsity / interpretability 折中表现。

### 6.2 论文里最核心的强项不是“所有窄域任务 raw-space EV 都会很好”

论文与官方页面反复强调的强项包括：

- 高覆盖率的 feature discovery
- 较好的 language modeling recovery
- 较强的 interpretability
- 对模型内部概念的广泛可视化和下游分析能力

这类“强”首先是：

- 同分布 held-out 强
- 官方处理链路强
- 分析与可解释性生态完整

它不自动推出：

> 在一个非常窄域、非常小的心理咨询对话数据集上，用一个不完全同步官方推理链的本地实现，raw-space EV/FVU/CE-KL 也必须同样优秀。

---

## 7. 官方实现关键点检查表

| 技术点 | 官方口径 | 来源 | 如果缺失/不同步会影响什么 |
| --- | --- | --- | --- |
| 输入/输出 hook | 同层 `blocks.19.hook_resid_post` | HF `hyperparams.json` | hook 错位会直接破坏结构与功能指标 |
| activation function | `jumprelu` | HF `hyperparams.json` | 激活函数不一致会改变稀疏模式 |
| dataset norm 元数据 | `in=17.125`, `out=17.125` | HF `hyperparams.json` | 输入输出尺度闭环会失真 |
| 推理路径 | `dataset-wise` 或 `inference` 二选一 | `sparse_dictionary.py` | 只做前半截归一化会让 raw-space 比较失真 |
| 参数折叠 | `fold_activation_scale=True` 时调用 `standardize_parameters_of_dataset_norm()` | `sparse_dictionary.py` | 如果需要 raw-space 直接替换但未折叠，会系统性拉低 fidelity |
| 外层反归一化 | `dataset-wise` 模式下需 `denormalize_activations(...)` | `sparse_dictionary.py` | 不做会让 decode 输出停留在归一化空间 |
| decoder norm 参与稀疏性 | `sparsity_include_decoder_norm=true` | HF `hyperparams.json` + 官方 `sae.py` | L0、dead ratio、latent 排序、组选择都会偏移 |
| decoder bias 回灌 encoder | 本 checkpoint 为 `false` | HF `hyperparams.json` | 不应额外引入 decoder-bias-to-pre-encoder 逻辑 |
| post-training processing | 论文明确评估前使用 | 论文第 3.3.4 节 + openmoss.ai | 若本地未对齐，会出现“论文强、本地弱”的结构断层 |

---

## 8. 本文档的直接结论

关于这份 SAE 的官方技术口径，可以压缩成三句话：

1. 这不是一个“裸 JumpReLU SAE”这么简单，而是带有 dataset-wise normalization 元数据、decoder-norm-aware 稀疏语义、且论文评估时经过 post-training processing 的公开 checkpoint。
2. 官方实现允许两条正确推理路径：  
   - `dataset-wise`：外层显式归一化与反归一化  
   - `inference`：把 activation scale 折叠进参数后直接处理 raw activation
3. 如果本地实现只对输入做了部分归一化，却没有把输出尺度、decoder norm 语义和论文后处理口径完整同步，那么出现“论文上强、本地结构指标不尽人意”是完全可能的。

因此，后续判断问题归因时，不能默认把锅全部甩给数据集。

更严格的说法应当是：

> 在讨论数据集分布差异之前，必须先确认本地实现是否完整同步了 OpenMOSS 官方 SAE 的推理与评估口径。

---

## 9. 参考链接

- Llama Scope 论文 PDF:
  https://lingjiechen2.github.io/data/llama_scope_EXTRACTING%20MILLIONS%20OF%20FEATURES%20FROM%20LLAMA-3.1-8B%20WITH%20SPARSE%20AUTOENCODERS.pdf
- arXiv 摘要页:
  https://arxiv.org/abs/2410.20526
- Hugging Face 仓库:
  https://huggingface.co/OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x
- 本 checkpoint 的 `hyperparams.json`:
  https://huggingface.co/OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x/resolve/main/Llama3_1-8B-Base-L19R-8x/hyperparams.json
- OpenMOSS 官方项目页:
  https://openmoss.ai/en/language-model-SAEs/
- OpenMOSS 官方模型实现:
  https://raw.githubusercontent.com/OpenMOSS/Language-Model-SAEs/refs/heads/dev/src/lm_saes/models/sae.py
- OpenMOSS 官方抽象与加载实现:
  https://raw.githubusercontent.com/OpenMOSS/Language-Model-SAEs/refs/heads/dev/src/lm_saes/models/sparse_dictionary.py
