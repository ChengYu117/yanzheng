# AI 专家代理评审管线说明

## 1. 这条管线是做什么的

这条管线是主实验之后的一个独立后处理步骤。它读取主流程导出的 `judge_bundle/`，然后让一个 OpenAI 兼容 API 模型去评审：

- 单个 latent 的高激活句子
- `G1 / G5 / G20` 这几组 latent 子空间的高分句子

目标不是重新训练模型，而是回答一个更直观的问题：

> 某个 latent 或一组 latent 高激活时，对应的句子看起来是否真的具有清晰的 RE 特征？

## 2. 它在研究里扮演什么角色

这条管线的定位固定为：

- 自动解释证据
- 专家代理评审证据

它不是：

- 典型的因果性判定标准
- 单独成立的机制证明

如果后续要写“这一组 latent 事实上驱动了 RE 概念”，仍然需要把它和：

- `ablation`
- `steering`
- `TPP`
- 对照实验

一起报告。

## 3. 主流程如何给它准备输入

在 `run_functional_evaluation()` 结束时，系统会自动导出一个 `judge_bundle/`，放在 `metrics_functional.json` 同级目录。

其中固定包含：

- `manifest.json`
- `latent_examples.jsonl`
- `group_examples.json`
- `rubric_snapshot.json`

### 3.1 latent_examples.jsonl

这里每条记录对应一个候选 latent，至少包含：

- `latent_idx`
- `candidate_rank`
- `re_purity_top_n`
- `top_examples`
- `control_examples`

其中：

- `top_examples` 是该 latent 激活最高的句子
- `control_examples` 是该 latent 激活处在 `40%-60%` 分位区间、最接近中位数的一组句子

这样设计是为了比较：

- 高激活样本到底是不是更像 RE
- 还是 judge 对所有样本都一律打高分

### 3.2 group_examples.json

这里固定导出三组：

- `G1`
- `G5`
- `G20`

每组都包含：

- `latent_ids`
- `weights`
- `top_examples`
- `control_examples`

组分数的定义是：

1. 先取组内每个 latent 的句级激活
2. 对每一列做 z-score 标准化
3. 用组权重加权求和

也就是：

`group_score(u) = Σ alpha_i * zscore(a_{u,i})`

默认情况下：

- `alpha_i` 来自组 probe 的绝对值权重归一化
- 如果组 probe 权重不可用，就退化为等权重

## 4. AI judge 实际怎么评审

judge 分两层：

### 4.1 单句评审

对每个 latent 或 group 的 `top_examples` / `control_examples`，逐句调用 judge。

这一层输出：

- 是否清晰体现 RE
- 更像 simple 还是 complex reflection
- 清晰度分数
- 四个维度打分
- 中文理由
- 风险标签

### 4.2 latent / group 汇总评审

在单句结果基础上，再让 judge 汇总：

- 这组高激活样本有没有共同特征
- 这个特征是否像清晰的 RE feature
- 如果是 group，它是否比单 latent 更像一个分布式 RE 子空间

## 5. 如何运行

### 5.1 只导出 prompt，不实际请求 API

```powershell
python run_ai_re_judge.py `
  --input-dir outputs/sae_eval_bailian_qwen35plus `
  --output-dir outputs/sae_eval_bailian_qwen35plus/ai_judge_dry `
  --model your-model-name `
  --dry-run-prompts
```

### 5.2 实际运行 judge

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
$env:OPENAI_MODEL="gpt-4.1-mini"

python run_ai_re_judge.py `
  --input-dir outputs/sae_eval_bailian_qwen35plus `
  --output-dir outputs/sae_eval_bailian_qwen35plus/ai_judge
```

也可以显式指定：

- `--top-latents`
- `--top-n`
- `--control-n`
- `--groups G1,G5,G20`
- `--temperature`
- `--max-retries`

## 6. 输出文件怎么看

最终会生成：

- `utterance_reviews.jsonl`
- `latent_reviews.json`
- `group_reviews.json`
- `calibration.json`
- `report.md`
- `logs/`
- `prompts/`

### 6.1 utterance_reviews.jsonl

逐句结果。适合排查 judge 到底为什么认为某句话是 RE 或不是 RE。

### 6.2 latent_reviews.json

每个候选 latent 的汇总结论。核心字段包括：

- `judge_re_rate`
- `control_re_rate`
- `avg_clarity_score`
- `dominant_re_type`
- `shared_feature_name`
- `final_latent_judgement`

### 6.3 group_reviews.json

每个组的汇总结论。核心字段包括：

- `group_judge_re_rate`
- `group_control_re_rate`
- `group_avg_clarity`
- `final_group_judgement`
- `is_distributed_re_subspace`

### 6.4 calibration.json

这是最关键的约束文件，用来防止 “judge 看什么都说像 RE”。

它至少包含两类统计：

- 标签一致性
  - judge 对 RE / NonRE 的判断和本地标签是否一致
- top vs control 区分能力
  - 高激活样本是否明显比中位激活对照更像 RE

## 7. 为什么必须做校准

LLM judge 很容易出现一个问题：

> 说得温和一点、像咨询语气一点，就被打成“像 RE”。

所以这条管线不能只输出理由文本，必须额外输出：

- `accuracy`
- `macro_f1`
- `cohens_kappa`
- `confusion_matrix`
- `judge_re_rate_gap`
- `clarity_gap`

只有在 `top_examples` 明显优于 `control_examples` 时，某个 latent 才更有资格被称为“清晰 RE feature”。

## 8. 当前实现的边界

这条管线目前仍然有三个研究边界：

1. 它是句内评审，不是上下文级咨询轮次评审。
2. 它是专家代理，不是真实心理咨询专家的人审。
3. 它提供的是解释性证据，不是单独的因果证明。

所以最稳妥的结论表述应该是：

> 这条 AI 评审管线可以帮助我们更系统地判断某个 latent 或子空间是否呈现清晰的 RE 风格特征，但它需要和干预实验一起使用，才能支撑更强的机制解释结论。
