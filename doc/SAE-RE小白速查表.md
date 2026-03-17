# SAE-RE 项目速查版

这份速查版适合你快速回看，不适合第一次深入学习。  
如果你第一次看这个项目，先看：

- [小白总文档](D:/project/NLP_re_dataset_model_base/doc/SAE-RE小白入门指南.md)

---

## 1. 项目一页纸摘要

| 问题 | 简短答案 |
|---|---|
| 这个项目研究什么？ | 研究 Llama-3.1-8B 第 19 层里，是否存在和 RE（反射性倾听）相关的 SAE latent。 |
| 基础模型是谁？ | 本地 `Meta-Llama-3.1-8B`。 |
| SAE 来自哪里？ | Hugging Face 上的 `OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x / L19R-8x`。 |
| 输入数据是什么？ | `re_dataset.jsonl` 和 `nonre_dataset.jsonl` 的 `unit_text`。 |
| 最终想得到什么？ | 一批和 RE / NonRE 显著相关的候选 latent。 |
| 当前已经拿到什么？ | `metrics_structural.json` 和 `candidate_latents.csv`。 |
| 当前还没稳定拿到什么？ | `metrics_functional.json` 和 `latent_cards/`。 |

---

## 2. 五步看懂整条管线

| 步骤 | 一句话解释 |
|---|---|
| 第 1 步 | 读 RE / NonRE 文本。 |
| 第 2 步 | 让本地 Llama-3.1-8B 处理这些文本。 |
| 第 3 步 | 在第 19 层把 residual activation 抓出来。 |
| 第 4 步 | 用 SAE 把这些激活拆成 32768 个稀疏 latent。 |
| 第 5 步 | 把 token 特征聚合成句子特征，再做结构评估和候选 latent 筛选。 |

---

## 3. 六个最重要文件的职责

| 文件 | 记忆锚点 |
|---|---|
| `run_sae_evaluation.py` | 总调度器：决定整条实验顺序。 |
| `src/nlp_re_base/data.py` | 读 JSONL，把 `unit_text` 读出来。 |
| `src/nlp_re_base/model.py` | 加载本地 Llama 模型和 tokenizer。 |
| `src/nlp_re_base/sae.py` | 下载并构建 SAE。 |
| `src/nlp_re_base/activations.py` | 抓第 19 层激活，跑 SAE，做 streaming 和聚合。 |
| `src/nlp_re_base/eval_structural.py` + `src/nlp_re_base/eval_functional.py` | 一个看 SAE 本身质量，一个看 latent 有没有研究价值。 |

---

## 4. 十个最关键参数

| 参数 | 当前值 | 记忆锚点 |
|---|---:|---|
| `hook_point` | `blocks.19.hook_resid_post` | 在第 19 层抓激活。 |
| `d_model` | `4096` | 原始激活维度。 |
| `d_sae` | `32768` | SAE latent 数量。 |
| `act_fn` | `jumprelu` | SAE 的激活函数。 |
| `jump_relu_threshold` | `0.52734375` | latent 超过这个门槛才明显“开”。 |
| `norm_activation` | `dataset-wise` | 激活会先做 dataset-wise 归一化。 |
| `max_seq_len` | `128` | 每条文本最多保留 128 token。 |
| `aggregation` | `max` | 句级特征取 token 最大激活。 |
| `fdr_alpha` | `0.05` | 候选 latent 的显著性阈值。 |
| `top_k_candidates` | `50` | 后续深分析最多保留 50 个候选。 |

补充提醒：

- `sae_config.json` 里写了 `batch_size=8`
- 但主程序默认运行时用的是命令行默认 `batch_size=4`

---

## 5. 当前最重要的实验结果

### 结构指标

| 指标 | 数值 | 快速理解 |
|---|---:|---|
| `MSE` | `4.5804` | 重建误差不低。 |
| `cosine_similarity` | `0.8088` | 方向相似度还可以。 |
| `explained_variance` | `0.0682` | 只解释了约 6.8% 的方差。 |
| `FVU` | `0.9318` | 大部分方差还没被解释掉。 |
| `l0_mean` | `172.58` | 平均每个 token 激活约 172 个 latent。 |
| `dead_ratio` | `90.05%` | 大量 latent 在 sample 里没激活。 |

### 候选 latent

| 结果 | 数值 |
|---|---:|
| 总 latent 数 | 32768 |
| FDR 显著 latent 数 | 1540 |
| Top-1 latent | `19435` |
| Top-1 Cohen's d | `0.8991` |
| Top-1 AUC | `0.7003` |

一句话解释：

> SAE 空间里已经能看到一批和 RE / NonRE 显著相关的候选特征，但完整功能评估还没形成正式结果文件。

---

## 6. 当前最重要的三个风险 / 未完成点

| 风险 | 为什么重要 |
|---|---|
| 功能评估还没稳定产出 `metrics_functional.json` | 说明完整研究闭环还没正式落地。 |
| 结构指标是 sample-based | 当前结构数字不是全量 token 的严格总体结果。 |
| 正式主实验跳过了 CE/KL | 现在还不能正式评价 SAE 重建对模型输出分布的影响。 |

---

## 记住这句话

> 这个项目不是在“训练一个 RE 分类器”，而是在“拆开大模型的中间表示，找出和 RE 概念有关的稀疏特征”。
