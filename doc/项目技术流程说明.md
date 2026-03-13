# SAE-RE Pipeline 技术实现说明

## 1. 文档目的

本文档从工程实现角度解释当前仓库如何完成以下研究任务：

1. 加载本地 `Llama-3.1-8B` 基底模型。
2. 从 Hugging Face 下载并加载 `OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x/Llama3_1-8B-Base-L19R-8x` 的 SAE 参数。
3. 将 `data/mi_re/re_dataset.jsonl` 和 `data/mi_re/nonre_dataset.jsonl` 作为输入。
4. 在 layer 19 residual stream 上运行 SAE。
5. 计算结构性与功能性指标，筛选与 RE 概念相关的候选 latent。

本文档重点解释“代码实际上怎么做”，而不是只复述研究目标。

## 2. 项目要解决的问题

这个项目把 RE 识别问题转成一个“概念特征挖掘”流程：

1. 用基底语言模型生成某一层的残差流激活。
2. 用预训练 SAE 把高维激活分解成更稀疏的 latent 特征。
3. 在 utterance 级别比较 RE 与 NonRE 样本在 latent 空间中的差异。
4. 用统计、探针和定性分析，找出最像“RE 概念”的 latent。

当前仓库的研究单位是单条 `unit_text`，不是完整对话上下文。也就是说，项目当前识别的是“单句是否呈现 RE 风格/信息”，而不是更强的对话级机制。

## 3. 代码结构总览

### 3.1 配置与数据

- `config/model_config.json`
  - 指定本地 Llama 模型路径、`torch_dtype` 和 `device_map`。
- `config/sae_config.json`
  - 指定 SAE 仓库、子目录、hook 点和评估参数。
- `data/mi_re/re_dataset.jsonl`
  - RE 样本。
- `data/mi_re/nonre_dataset.jsonl`
  - NonRE 样本。

### 3.2 核心代码

- `run_sae_evaluation.py`
  - 端到端总入口。
- `src/nlp_re_base/config.py`
  - 配置读取。
- `src/nlp_re_base/data.py`
  - JSONL 数据读取。
- `src/nlp_re_base/model.py`
  - 本地 Llama 模型和 tokenizer 加载。
- `src/nlp_re_base/sae.py`
  - SAE 定义、checkpoint 下载与权重映射。
- `src/nlp_re_base/activations.py`
  - residual 激活提取、SAE 前向、流式聚合。
- `src/nlp_re_base/eval_structural.py`
  - 结构性指标计算。
- `src/nlp_re_base/eval_functional.py`
  - 功能性指标计算。

## 4. 端到端运行流程

## 4.1 入口脚本

运行入口是：

```powershell
python run_sae_evaluation.py --output-dir outputs/sae_eval --batch-size 4
```

主函数做的事情可以概括为 7 步：

1. 读取 SAE 配置和运行参数。
2. 读取 RE / NonRE 数据。
3. 加载本地 Llama-3.1-8B。
4. 从 Hugging Face 下载并加载 SAE。
5. 以流式方式提取激活、运行 SAE、聚合 utterance 特征。
6. 计算结构性指标。
7. 计算功能性指标并输出结果文件。

## 4.2 数据装载

`src/nlp_re_base/data.py` 中的 `load_jsonl()` 逐行读取 JSONL。当前主流程只取每条记录的：

- `unit_text`

也就是说，模型输入不包含 `rationale`、`predicted_subcode` 或更长上下文。

在 `run_sae_evaluation.py` 中：

1. `re_dataset.jsonl` 读成 `re_records`
2. `nonre_dataset.jsonl` 读成 `nonre_records`
3. 提取 `unit_text`
4. 拼接成 `all_texts`
5. 构造标签 `all_labels`

这样后续所有评估都共享同一套样本顺序。

## 4.3 本地基底模型加载

`src/nlp_re_base/model.py` 的 `load_local_model_and_tokenizer()` 负责：

1. 从 `model_config.json` 读取本地模型路径。
2. 根据 `torch_dtype` 选择 `float16` / `bfloat16` / `float32`。
3. 用 `AutoTokenizer.from_pretrained()` 载入 tokenizer。
4. 如果没有 `pad_token`，回退到 `eos_token`。
5. 用 `AutoModelForCausalLM.from_pretrained()` 加载本地权重。

这意味着仓库并不复制大模型文件，而是依赖外部已经存在的本地权重目录。

## 4.4 SAE 下载与构建

`src/nlp_re_base/sae.py` 做了三件事：

1. 定义 `JumpReLU` 和 `SparseAutoencoder`。
2. 从 Hugging Face 下载 `hyperparams.json` 和 checkpoint。
3. 把 checkpoint 键名映射到本地 SAE 参数名。

### 4.4.1 SAE 架构

当前实现的 SAE 是 JumpReLU 形式：

- 编码：
  - 先对输入激活做 dataset-wise norm。
  - 再做线性变换和偏置。
  - 最后经过 `JumpReLU`。
- 解码：
  - `latents @ W_dec.T + b_dec`

输入维度由 `d_model=4096` 给定，latent 维度由 `d_sae=32768` 给定。

### 4.4.2 权重加载策略

加载 SAE 时，代码会：

1. 下载 `hyperparams.json`
2. 下载 `checkpoints/*.safetensors`
3. 合并多个 safetensors 文件
4. 映射常见权重名
5. 对 `W_enc` 和 `W_dec` 做 shape 检查，必要时转置
6. 使用 `strict=True` 执行 `load_state_dict`

这比早期“缺权重也继续跑”的做法更严格，能避免在部分随机初始化的 SAE 上继续计算指标。

## 4.5 Hook 点与激活提取

`src/nlp_re_base/activations.py` 通过 forward hook 捕获指定层的 residual stream。

配置中的 hook 点是：

```json
"hook_point": "blocks.19.hook_resid_post"
```

它会被解析成：

- `model.model.layers[19]`

然后在该层注册 `register_forward_hook()`，从 layer 输出里取出 hidden states。

如果目标模块输出是 tuple，代码取 `output[0]`；否则直接取 `output`。

## 4.6 为什么采用流式处理

如果把全量样本的 token 级 latent 全部保存在内存里，张量规模会非常大。为避免出现完整的 `[N, T, d_sae]` 常驻内存，当前实现改成了流式架构：

1. 按 batch tokenize。
2. 跑基底模型，拿到当前 batch 的 residual activation。
3. 立刻把 activation cast 到 SAE 的 dtype。
4. 立刻跑 SAE forward。
5. 立刻把 token 级结果聚合成 utterance 级表示。
6. 只保留 `[B, d_sae]` 和 `[B, d_model]` 级别的聚合结果。

这样主存里保留的是：

- utterance-level SAE features
- utterance-level raw activations
- 少量 token-level sample

而不是全量 token-level latent。

## 4.7 dtype 对齐

这是当前实现里一个很关键的工程细节。

基底模型和 SAE 的 dtype 不一定相同。为避免 SAE 接收到错误 dtype 的激活，代码在 `extract_and_process_streaming()` 和 `compute_ce_kl_with_intervention()` 中都做了显式转换：

1. 从 SAE 对象读 `sae_dtype`
2. 把 residual activation cast 到 SAE dtype
3. SAE 输出后，再在需要时 cast 回原模型 dtype

这一步是保证整条管线不因 `float16` / `bfloat16` 不一致而报错的关键。

## 4.8 utterance 级聚合

token 级 latent 需要变成 utterance 级特征，当前支持两种方式：

- `max`
- `mean`

默认配置使用：

```json
"aggregation": "max"
```

也就是说，对每个 latent，句子级特征取该 latent 在该句所有有效 token 上的最大激活值。

这一步会分别生成：

- `utterance_features`
  - 维度 `[N, d_sae]`
- `utterance_activations`
  - 维度 `[N, d_model]`

后者主要用于 dense probe baseline。

## 5. 结构性评估实现

结构性评估位于 `src/nlp_re_base/eval_structural.py`。

当前实现的指标包括：

- `MSE`
- `Cosine Similarity`
- `Explained Variance`
- `FVU`
- `L0 sparsity`
- `firing frequency`
- `dead feature ratio`
- `CE loss delta`
- `KL divergence`

### 5.1 reconstruction fidelity

对原始激活 `z` 和 SAE 重建 `z_hat`，代码计算：

- 均方误差
- 余弦相似度
- 方差解释率

这些指标回答的问题是：

- SAE 是否保留了原激活的大部分信息？
- 重建后的方向是否与原激活接近？
- 丢失的信息有多少？

### 5.2 稀疏性相关指标

代码把 latent 非零个数视为稀疏度的近似：

- `L0 Mean`
- `L0 Std`
- 每个 latent 的 firing frequency
- dead feature 数量和比例

这些指标回答的问题是：

- SAE 是否足够稀疏？
- 是否有大量“从不触发”的 latent？

### 5.3 CE / KL intervention

`compute_ce_kl_with_intervention()` 会对每个 batch 做两次 forward：

1. 正常 forward，得到原始 logits 与 hook 激活。
2. 用 `SAE(reconstruction)` 替换 hook 点激活，再 forward 一次。

然后比较：

- `ce_loss_orig`
- `ce_loss_sae`
- `ce_loss_delta`
- `kl_divergence`

这里测的是“如果把该层 residual stream 替换成 SAE 重建后的版本，模型分布偏移有多大”。

### 5.4 结构指标当前的采样方式

为了控制内存，当前结构指标不是基于全量 token 级结果，而是只保留前若干个 batch 的 token 级 sample：

- `collect_structural_samples=5`

然后用这些 sample 来算结构指标。

这是一种工程折中。它能显著减小内存，但也意味着结构指标目前是 sample-based，不是全量 token-based。

## 6. 功能性评估实现

功能性评估位于 `src/nlp_re_base/eval_functional.py`。

它回答的是另一个问题：

- SAE latent 是否真的和 RE 概念对齐？

### 6.1 单 latent 统计检验

`univariate_analysis()` 对每个 latent 计算：

- Cohen's d
- AUC
- t-test p-value
- BH-FDR 显著性

这一步的作用是先做“逐 latent 排名”，找出最可疑的候选 latent。

输出文件：

- `candidate_latents.csv`

这是后续所有功能性分析的候选入口。

### 6.2 Sparse probing

`sparse_probing()` 取按 `|Cohen's d|` 排序后的 top-k latent，训练逻辑回归 probe，并使用 5 折分层交叉验证评估：

- `accuracy`
- `f1`
- `auc`

同时还给出两个 baseline：

- dense probe baseline
- diffmean baseline

它回答的问题是：

- 少量 latent 是否足以区分 RE / NonRE？
- SAE latent 是否比原始 dense activation 更紧凑？

### 6.3 MaxAct 分析

`maxact_analysis()` 对候选 latent 做定性分析：

1. 找到该 latent 激活值最高的若干 utterance
2. 输出文本、激活值、标签
3. 统计 top-N 中 RE 的纯度

输出目录：

- `latent_cards/`

这一步的目的不是做最终统计结论，而是帮助研究者人工判断：

- 这个 latent 看起来是不是像“反射性倾听”？
- 它到底在响应哪种语言模式？

### 6.4 Feature Absorption

`feature_absorption()` 会对候选 latent 做近邻相关性分析：

1. 对某个目标 latent，寻找最相关的其他 latent。
2. 当目标 latent 不激活时，检查这些近邻是否替代性激活。
3. 统计 mean absorption / full absorption。

这个指标想回答：

- “RE 概念”是集中在一个 latent 上，还是被多个相似 latent 分摊了？

### 6.5 Feature Geometry

`feature_geometry()` 读取 SAE 的 decoder 矩阵 `W_dec`，只取候选 latent 对应的 decoder 列向量，计算两两 cosine similarity。

它回答的问题是：

- 候选 latent 在 decoder 空间里是否高度重合？
- 如果高度重合，说明这些 latent 可能并不是彼此独立的概念单元。

### 6.6 TPP

`targeted_probe_perturbation()` 的实现流程是：

1. 取 top-k 候选 latent。
2. 在这些 latent 上训练一个逻辑回归 probe。
3. 逐个把某个 latent 的特征列置零。
4. 观察 probe accuracy 下降多少。

`accuracy_drop` 越大，说明该 latent 对 probe 更重要。

当前 TPP 是 probe-space 的扰动，不是模型层级的 activation patching。它更接近“latent 对分类器决策的贡献度分析”。

## 7. 输出文件说明

默认输出目录是 `outputs/sae_eval`。

主要产物包括：

- `metrics_structural.json`
  - 重建、稀疏性、CE/KL 等结构指标。
- `metrics_functional.json`
  - 功能性指标汇总。
- `candidate_latents.csv`
  - 候选 latent 排名表。
- `latent_cards/*.md`
  - 候选 latent 的 MaxAct 卡片。

如果从研究流程角度看，这些文件分别对应：

- 结构保真
- 概念对齐
- 候选筛选
- 定性解释

## 8. 这个项目当前最重要的工程设计

### 8.1 本地模型 + 远端 SAE

项目不复制大模型权重，而是：

- 本地加载 Llama-3.1-8B
- 在线下载 SAE checkpoint

这样仓库体积更小，但运行时依赖：

- 本地模型路径存在
- 网络能访问 Hugging Face

### 8.2 严格 SAE 加载

当前 SAE 加载器比常见原型代码更保守：

- 检查 safetensors 是否存在
- 检查关键权重是否缺失
- shape 不一致时尽量转置，否则直接报错
- `strict=True`

这保证研究结果不会建立在“半随机 SAE”上。

### 8.3 流式激活处理

这是当前仓库最关键的可运行性设计：

- 不保留全量 token 级 latent
- 只保留 utterance 级特征和少量 sample

它直接决定了这个项目能否在普通单机环境里完成评估。

## 9. 当前实现边界与注意事项

### 9.1 结构指标是 sample-based

当前结构指标不是对全量 token 级输出求值，而是对前几个 batch 的 sample 求值。

优点：

- 内存可控

代价：

- 指标代表性下降
- 如果样本顺序有偏，指标可能被数据排列影响

### 9.2 TPP 仍然是分类器层面的因果近似

当前 TPP 并没有直接干预基底模型的 latent-to-output 因果链，而是在 probe 输入层做列置零。

因此它更适合回答：

- “这个 latent 对 probe 是否重要？”

而不是最强版本的：

- “这个 latent 是否因果驱动了 LLM 的 RE 行为？”

### 9.3 Feature Absorption 计算代价较高

当前 `feature_absorption()` 逐候选 latent 扫描全体 latent 做相关性计算，复杂度较高。对 `d_sae=32768` 的 SAE 来说，这部分可能成为功能评估阶段的主要耗时项。

### 9.4 输入粒度是单句

当前样本是单条 utterance，不包含长程对话上下文，因此研究结论更适合解释为：

- “RE 风格句子的局部特征”

而不是：

- “完整治疗对话中的 RE 机制”

## 10. 如何阅读这个项目

如果你要从代码层面快速理解整个项目，建议按下面顺序读：

1. `run_sae_evaluation.py`
2. `src/nlp_re_base/activations.py`
3. `src/nlp_re_base/sae.py`
4. `src/nlp_re_base/eval_structural.py`
5. `src/nlp_re_base/eval_functional.py`
6. `config/model_config.json`
7. `config/sae_config.json`

这个顺序基本对应真实运行顺序。

## 11. 总结

从技术实现上看，这个项目已经具备一条完整的 SAE-RE 评估管线：

1. 本地模型加载
2. SAE 下载和严格装载
3. 指定层 residual 激活提取
4. 流式 SAE 编码
5. utterance 级特征聚合
6. 结构性指标计算
7. 功能性候选挖掘与定性解释

它的核心价值不在“重新训练 SAE”，而在“把现成 SAE 作为一个概念显微镜”，去观察 RE 与 NonRE 在 layer 19 residual stream 上的差异。

如果后续要把这个项目推进到更强的研究版本，优先级通常会是：

1. 让结构指标从 sample-based 变成更可靠的分层估计或全量估计。
2. 把 TPP 从 probe-space 扩展到更强的模型层级干预。
3. 优化 absorption 等高复杂度模块的计算代价。
4. 在单句分析之外加入更强的上下文条件。
