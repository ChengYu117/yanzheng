# SAE-RE软件工程实现总报告

## 1. 适用范围与定位

本文档面向答辩、导师汇报和后续接手项目的工程人员，从软件工程视角说明该仓库如何实现 SAE-RE 研究任务。

它关注的问题是：

- 代码如何组织
- 主流程如何运行
- 数据、模型、SAE、评估和因果验证如何连接
- 当前有哪些工程设计是刻意选择
- 当前有哪些实现不足和维护风险

这是一份总览型工程文档，不替代现有专题文档。进一步细节可参考：

- [项目技术流程说明](./项目技术流程说明.md)
- [因果性分析结果达标性报告](./因果性分析结果达标性报告.md)
- [AI专家代理评审管线说明](./AI专家代理评审管线说明.md)
- [RE定义与AI评审Rubric](./RE定义与AI评审Rubric.md)
- [sae_causal_validation_guide](./sae_causal_validation_guide.md)

---

## 2. 项目定位

这是一个典型的“科研管线型软件项目”，而不是通用产品服务。它的目标不是提供对外稳定 API，而是：

- 支撑 RE 概念发现研究
- 复现实验
- 保存阶段性结果
- 为后续因果验证、AI 评审和论文撰写提供证据链

因此它的代码组织呈现出明显的科研项目特点：

- 入口脚本多
- 中间结果文件多
- 文档分散但信息密度高
- 结果目录与代码逻辑耦合较强

---

## 3. 仓库整体架构

当前仓库可以分成 7 个主要区域。

### 3.1 配置层

- `config/model_config.json`
  - 本地基底模型路径、dtype、device_map
- `config/sae_config.json`
  - SAE 仓库、子目录、hook 点、评估参数
- 其他实验配置文件

### 3.2 数据层

- `data/mi_re/`
  - 历史 RE / NonRE split 数据
- `data/cactus/`
  - 新版 CACTUS therapist-span 数据与统计

### 3.3 主库代码

- `src/nlp_re_base/`
  - 数据、模型、SAE、激活提取、结构/功能评估、AI judge 等核心逻辑

### 3.4 因果验证子模块

- `causal/`
  - therapist span 感知的数据批处理
  - latent 组选取
  - ablation / steering
  - side-effect 评估
  - 因果实验总控

### 3.5 顶层运行脚本

- `run_sae_evaluation.py`
- `run_stage2_activation_extraction.py`
- `run_ai_re_judge.py`
- `causal/run_experiment.py`

### 3.6 结果目录

- `sae_eval_full/`
- `causal_validation_full/`

这些目录是当前仓库内最重要的现成结果快照。

### 3.7 文档层

- `doc/`
  - 研究分析
  - 工程流程
  - 因果验证说明
  - AI judge 说明
  - Rubric

---

## 4. 端到端主流程总览

从工程上看，项目主线可以分成 4 条互相关联的流水线。

### 4.1 主评估流水线

入口：

- [run_sae_evaluation.py](../run_sae_evaluation.py)

作用：

1. 加载数据
2. 加载基底模型
3. 加载 SAE
4. 提取 residual activation
5. 跑 SAE
6. 聚合为 utterance-level feature
7. 计算结构性指标与功能性指标
8. 产出候选 latent、judge bundle 和 latent cards

### 4.2 Stage 2 线性可分层定位流水线

入口：

- [run_stage2_activation_extraction.py](../run_stage2_activation_extraction.py)

作用：

1. 对所有 hidden layer 抽 utterance-level 表示
2. 对每层训练轻量线性 probe
3. 找出最线性可分的 RE / NonRE 层

### 4.3 因果验证流水线

入口：

- [run_experiment.py](../causal/run_experiment.py)

作用：

1. 从候选 latent 排名中构建 `G1 / G5 / G20`
2. 跑 necessity ablation
3. 跑 sufficiency steering
4. 跑 selectivity / side-effect
5. 跑 group structure 分析
6. 输出 `selected_groups.json` 与多份结果 JSON

### 4.4 AI 专家代理评审流水线

入口：

- [run_ai_re_judge.py](../run_ai_re_judge.py)

作用：

1. 读取主流程导出的 `judge_bundle`
2. 调用固定 rubric
3. 逐句评审 latent / group 的高激活样本
4. 生成结构化评审结果

---

## 5. 数据层实现

### 5.1 统一数据入口

核心文件：

- [data.py](../src/nlp_re_base/data.py)

它提供：

- `load_jsonl()`
- `load_cactus_dataset()`
- `_load_legacy_split()`
- `dataset_summary()`

### 5.2 当前数据层的兼容策略

当前数据层并不是“只有一种数据定义”，而是同时兼容：

1. CACTUS 统一 JSONL
   - 默认路径 `data/cactus/cactus_re_small_1500.jsonl`
   - `label`、`formatted_text`、`therapist_char_start/end`
2. legacy MI-RE split
   - `data/mi_re/re_dataset.jsonl`
   - `data/mi_re/nonre_dataset.jsonl`
   - 主要字段是 `unit_text`

`load_cactus_dataset()` 的策略是：

- 优先读 CACTUS unified JSONL
- 若不存在，则回退到 legacy split
- 对 CACTUS 样本补出：
  - `unit_text`
  - `label_re`
- 对 legacy split 样本补出：
  - `label_re`

### 5.3 当前数据层的一个关键现实

当前代码默认偏向 CACTUS，但现有全量结果快照不是基于 CACTUS 跑的。

从：

- `sae_eval_full/run.log`
- `causal_validation_full/run.log`

可以确认当前 checked-in 的 full 结果使用的是：

- `799 RE`
- `799 NonRE`
- `1598` 总样本

也就是历史 split 数据链路。

这意味着工程上存在一个必须向读者说明的现实：

- 当前代码能力已经演进到支持 CACTUS therapist-span
- 但当前最完整的一批结果产物仍然来自 legacy split

### 5.4 数据层风险

这一兼容策略让代码可继续运行，但也带来维护负担：

- 同一个函数名承载两种数据语义
- `unit_text` 与 `formatted_text` 并存
- therapist span 只在 CACTUS 模式下可用
- 文档和结果目录容易和“当前默认数据源”产生误解

---

## 6. 模型与 SAE 层实现

### 6.1 本地基底模型加载

核心文件：

- [model.py](../src/nlp_re_base/model.py)

核心函数：

- `load_local_model_and_tokenizer()`

职责：

1. 读取 `model_config.json`
2. 解析本地模型路径
3. 解析 dtype
4. 根据是否安装 `accelerate` 决定是否使用 `device_map`
5. 加载 tokenizer
6. 补 pad_token
7. 加载 `AutoModelForCausalLM`

### 6.2 SAE 定义与加载

核心文件：

- [sae.py](../src/nlp_re_base/sae.py)

核心对象与函数：

- `JumpReLU`
- `SparseAutoencoder`
- `load_sae_from_hub()`

当前实现的关键信息：

- `d_model = 4096`
- `d_sae = 32768`
- `W_enc` 形状为 `[d_sae, d_model]`
- `W_dec` 形状为 `[d_model, d_sae]`

### 6.3 SAE 加载策略为什么比较严格

当前实现不是“能跑就行”，而是：

- 下载 `hyperparams.json`
- 下载 safetensors checkpoint
- 映射 checkpoint 键名
- 对关键权重做 shape 对齐和转置检查
- `strict=True` 加载

这样设计的原因是：

- 科研结果不能建立在“部分随机初始化的 SAE”上
- 一旦权重缺失或 shape 错位，宁可立即失败，也不要产生伪结果

### 6.4 dtype 与 device 对齐

这是工程层面一个很关键的细节。

因为：

- 基底模型可以是 `float16`
- SAE 可以跑在 `bfloat16`

所以代码会显式读取 SAE 的 dtype，再把 residual activation cast 到 SAE dtype 后前向。这一步主要发生在 [activations.py](../src/nlp_re_base/activations.py) 和因果运行器中。

---

## 7. 激活提取与特征聚合

### 7.1 主激活提取入口

核心文件：

- [activations.py](../src/nlp_re_base/activations.py)

核心函数：

- `_parse_hook_point()`
- `_tokenize_batch()`
- `extract_and_process_streaming()`
- `_aggregate_batch()`

### 7.2 为什么采用 streaming 架构

如果直接保留全量 `[N, T, d_sae]` latent 张量，内存会非常大。当前设计改成：

1. batch tokenize
2. forward 到目标层
3. 立即跑 SAE
4. 立即聚合成 utterance-level 特征
5. 只保留 `[N, d_sae]` 和少量 sample 级 token 张量

这样主存里常驻的是：

- utterance-level latent features
- utterance-level raw activations
- 少量结构性评估 sample

而不是完整 token 级 latent。

### 7.3 hook 点解析

当前 hook 点格式固定为：

- `blocks.19.hook_resid_post`

`_parse_hook_point()` 会把它解析成：

- `model.model.layers[19]`

这就是主流程对目标层的统一定位方式。

### 7.4 utterance-level 聚合方式

当前主评估支持：

- `max`
- `mean`

因果流程里还扩展支持：

- `sum`
- `binarized_sum`

这意味着不同阶段的聚合空间并不完全一致，也是后续解释时必须注意的一点。

### 7.5 Stage 2 的差异化实现

核心文件：

- [stage2_activation_extraction.py](../src/nlp_re_base/stage2_activation_extraction.py)

Stage 2 的表征不是 SAE latent，而是：

- 对每个 hidden layer 取最后一个非 padding token 的 hidden state
- 对每层训练轻量 Logistic Regression probe

它的目标不是概念发现，而是：

- 找出最线性可分的层
- 为主流程提供层级直觉

---

## 8. 指标与候选筛选实现

### 8.1 结构性指标实现

核心文件：

- [eval_structural.py](../src/nlp_re_base/eval_structural.py)

主要负责：

- reconstruction fidelity
- sparsity
- firing frequency
- dead feature
- CE / KL intervention

这些指标主要回答：

- SAE 是否按预期工作
- SAE 替换后模型分布偏移有多大

### 8.2 功能性指标实现

核心文件：

- [eval_functional.py](../src/nlp_re_base/eval_functional.py)

主要负责：

- `cohens_d()`
- `benjamini_hochberg()`
- `univariate_analysis()`
- sparse probe
- dense probe / diffmean
- MaxAct cards
- feature absorption
- feature geometry
- TPP

### 8.3 `candidate_latents.csv` 是怎么来的

流程是：

1. 对所有 latent 做 univariate analysis
2. 计算 `cohens_d`、AUC、p-value
3. 做 BH-FDR 校正
4. 形成 DataFrame
5. 按 `|Cohen's d|` 排序输出

所以 `candidate_latents.csv` 本质上是：

- 一份候选 latent 排名表
- 后续因果验证和 AI judge 都会依赖它

### 8.4 当前 checked-in 的主结果快照

从 `sae_eval_full/metrics_functional.json` 和 `metrics_structural.json` 可读到：

- `1530` 个 FDR 显著 latent
- `sparse_probe_k20 AUC = 0.9129`
- `dense_probe AUC = 0.9667`
- `avg_re_purity = 0.456`
- `cosine_similarity = 0.8114`
- `explained_variance = 0.0717`
- `dead_ratio = 0.5343`
- `ce_loss_delta = 2.3635`
- `kl_divergence = 2.9417`

这些结果构成了后续科研解释的主证据输入。

---

## 9. 因果验证实现

### 9.1 因果数据层

核心文件：

- [causal/data.py](../causal/data.py)

这个模块的关键职责是把数据组织成：

- `input_ids`
- `attention_mask`
- `counselor_span_mask`
- `labels`
- `texts`
- `indices`

如果记录带有：

- `therapist_char_start`
- `therapist_char_end`

就会通过 offset mapping 构造 therapist token mask；否则回退到“整句都视为 counselor span”的 legacy 行为。

### 9.2 干预实现

核心文件：

- [intervention.py](../causal/intervention.py)

其中实现了：

- `zero_ablate`
- `mean_ablate`
- `cond_token_ablate`
- `constant_steer`
- `cond_input_steer`
- `cond_token_steer`
- `make_steering_direction`
- `make_orthogonal_direction`
- `make_random_direction`

本质上，所有干预都在 SAE latent 空间 `z` 上操作，再通过 `W_dec` 投回 residual space。

### 9.3 因果评分实现

核心文件：

- [evaluation.py](../causal/evaluation.py)

这里定义了：

- `REProbeScorer`
- `score_delta()`
- `eval_text_quality()`

也就是说，因果实验不是直接靠人工判断，而是靠一个训练好的 RE probe 作为读出器，比较干预前后：

- RE 样本 logit 的变化
- NonRE 样本 logit 的变化

### 9.4 组选与稳定性问题

核心文件：

- [selection.py](../causal/selection.py)

当前逻辑是：

1. 从 `candidate_latents.csv` 读取候选 latent
2. 用 `influence_abs` 和 `probe_weight_abs` 做组合排序
3. 生成 `G1 / G5 / G20`
4. 再做 bootstrap stability

当前最关键的工程问题在于：

- 稳定性不是硬约束，只是后置参考
- 最终 `G5 / G20` 仍可能被不稳定但本次排得靠前的 latent 补满

这就是为什么当前“组选不稳定”不是编号 bug，而是设计问题。

### 9.5 因果总控脚本

核心文件：

- [run_experiment.py](../causal/run_experiment.py)

它负责串起：

1. 数据加载
2. utterance feature 提取
3. latent 排名
4. bootstrap 稳定性
5. RE probe 训练
6. necessity 实验
7. sufficiency 实验
8. selectivity 实验
9. group structure 实验
10. 结果 JSON 和汇总表输出

### 9.6 当前因果实现的一个重要维护风险

`causal/run_experiment.py` 当前文件规模很大，而且包含：

- 主因果运行逻辑
- pooling comparison 分支
- 汇总报告逻辑

这使得它既是实验控制器，又是报告生成器，职责偏重。后续维护时很容易在一个文件里同时碰到：

- 选组逻辑
- 干预逻辑
- 结果 schema
- CLI 参数

这就是典型的科研代码膨胀风险。

---

## 10. AI 专家代理评审实现

### 10.1 入口与核心模块

入口：

- [run_ai_re_judge.py](../run_ai_re_judge.py)

核心模块：

- [ai_re_judge.py](../src/nlp_re_base/ai_re_judge.py)
- [re_judge_rubric.py](../src/nlp_re_base/re_judge_rubric.py)

### 10.2 judge bundle 的位置和作用

当前主评估会导出：

- `sae_eval_full/judge_bundle/`

其中包含：

- `latent_examples.jsonl`
- `group_examples.json`
- `manifest.json`
- `rubric_snapshot.json`

这相当于把主流程结果压缩成一个供 AI 评审消费的轻量包。

### 10.3 AI judge 做了什么

AI judge 的逻辑不是重新定义 RE，而是：

- 使用固定 rubric
- 对 latent / group 的高激活句子做结构化评审
- 输出：
  - 是否有清晰 RE 特征
  - 是 simple / complex / mixed / non_re / unclear
  - 风险标签
  - 组级比单 latent 是否更清晰

### 10.4 当前 AI judge 的边界

这一条管线非常重要，但适用边界也很明确：

- 它提供的是自动解释与专家代理证据
- 它不是因果证明
- 它不能替代真实人工标注专家
- 在缺少稳定上下文时，它更多是在做“句内 RE 线索评审”

因此它应被视为：

- 结果解释增强器
- 不是最终真值来源

---

## 11. 输出文件与可复现实验路径

### 11.1 主评估输出

目录：

- `sae_eval_full/`

关键文件：

- `metrics_structural.json`
- `metrics_functional.json`
- `candidate_latents.csv`
- `latent_cards/`
- `judge_bundle/`
- `run.log`

### 11.2 因果验证输出

目录：

- `causal_validation_full/`

关键文件：

- `selected_groups.json`
- `results_necessity.json`
- `results_sufficiency.json`
- `results_selectivity.json`
- `results_group.json`
- `summary_tables.md`
- `run.log`

### 11.3 典型运行命令

主评估：

```powershell
python run_sae_evaluation.py --output-dir outputs/sae_eval --batch-size 4
```

Stage 2：

```powershell
python run_stage2_activation_extraction.py --all --batch-size 4
```

因果验证：

```powershell
python causal/run_experiment.py --candidate-csv outputs/sae_eval/candidate_latents.csv --output-dir outputs/causal_validation
```

AI judge：

```powershell
python run_ai_re_judge.py --input-dir sae_eval_full --output-dir outputs/ai_judge
```

### 11.4 从零复现实验至少需要什么

至少需要：

- 本地可加载的 `Llama-3.1-8B`
- 可访问 Hugging Face 下载 SAE checkpoint
- Python 环境和依赖
- 足够的 GPU 资源
- 数据目录
- 输出目录写权限

如需部署到云端，仓库还提供：

- `package_project.py`
- `deploy/gce/` 脚本

---

## 12. 工程不足与维护风险

### 12.1 文档分散、入口多

当前文档内容丰富，但分散在：

- 技术流程
- 因果报告
- AI judge 说明
- Rubric
- 入门指南

优点是专题清晰，缺点是主线理解成本高。

### 12.2 数据兼容逻辑复杂

当前系统同时支持：

- CACTUS unified JSONL
- legacy MI-RE split

这保证了兼容性，但也意味着：

- 当前默认数据
- 当前 full 结果来源
- 文档口径

三者容易错位。

### 12.3 因果组选稳定性问题是设计问题，不是单纯 bug

当前稳定性问题来自：

- 排名与稳定性解耦
- 不稳定 latent 仍被补进最终组

所以这不是简单“修一个编号错误”就能解决的问题，而是组选策略要重新定义。

### 12.4 外部依赖重

项目依赖：

- 本地大模型目录
- Hugging Face SAE
- GPU
- 某些阶段还依赖 API 模型

这意味着：

- 复现门槛高
- 环境差异容易影响运行

### 12.5 结果目录与代码逻辑绑定紧

当前许多文档和分析默认引用：

- `sae_eval_full/`
- `causal_validation_full/`

如果后续结果目录切换而文档不更新，解释就容易失真。

### 12.6 部分脚本存在历史演化痕迹

例如：

- `causal/run_experiment.py` 文件较大且职责较重
- 主流程与 comparison 逻辑耦合
- 结果 schema 逐步演化

这在科研项目中常见，但长期会增加维护难度。

---

## 13. 总结

从软件工程角度看，这个项目已经形成了一条完整的科研管线：

1. 数据兼容加载
2. 本地模型加载
3. SAE 下载与严格装载
4. residual activation 抽取
5. SAE 编码与特征聚合
6. 结构性评估
7. 功能性评估
8. 因果验证
9. AI 专家代理评审
10. 结果目录落盘与报告输出

它的核心价值不在于“训练一个新模型”，而在于：

> 把现成 SAE 当作概念显微镜，围绕 RE 研究建立了一条可运行、可评估、可干预、可解释的实验软件链路。

但从工程成熟度上看，它仍然是一个典型的科研代码库，而不是产品级系统。当前最值得优先治理的工程问题包括：

- 数据语义兼容带来的复杂性
- 因果组选稳定性的设计缺陷
- 文档与结果目录的同步维护
- 大型总控脚本的职责膨胀

这些问题并不否定项目价值，但决定了后续如果要继续扩展、答辩或写论文，必须先把“主线定义、结果口径和组选策略”进一步收敛。
