# DeepSeek Top-50 双轨归纳提示词审阅稿

本版本只执行 P3 第 3 步。每个 latent 输入 50 条规范化唯一文本；不运行 scorer、baseline、minimal pair 或干预任务。

执行版本说明：本轮已完成的 dual-track 批次使用的精确 user prompt 保存在对应任务 JSONL 的 `prompt` 字段。其后为解决“无支持轨道却被要求提供代表证据”的结构矛盾，校验器与下一次运行的模板允许该轨道输出空代表列表；这项修正不改变已返回的 surface/semantic 模式或 partition。

## English System Prompt

```text
You evaluate anonymous SAE latent activation patterns. Infer one aggregate pattern across the full sample set, distinguish semantic evidence from surface form, and return only the requested JSON.
```

## English User Prompt

```text
Analyze one anonymous SAE latent from its 50 highest-activation utterances as one set.

Latent metadata: latent_idx={latent_idx}
Evidence size: 50 items, {n_unique} unique normalized texts. Each item is the highest-activation representative of its normalized text group.

Core instruction:
- Produce BOTH required tracks.
- Surface track: identify a recurring lexical, phrase, syntactic, discourse-marker, formatting, or domain-terminology pattern that explains a majority of the samples, if one exists.
- Semantic track: identify a recurring semantic content or dialogue function that explains a majority of the samples, if one exists.
- The two tracks may describe different patterns and may have different supporting samples. Do not force them to agree. If a track has no majority-supported pattern, state that explicitly in its pattern field.
- Reason across all 50 utterances. Do not produce 50 sentence-by-sentence mini-analyses.
- For EACH track, partition every sample id exactly once into that track's supporting ids or outlier ids.
- candidate_explanation is a short synthesis for a human reader. It must state whether the most useful candidate is surface-form, semantic/dialogue-function, or unresolved between the two tracks.

Top-50 samples:
{top_50_samples}

Return only one JSON object with exactly these fields:
[
  "latent_idx",
  "short_name",
  "candidate_explanation",
  "surface_pattern",
  "surface_supporting_sample_ids",
  "surface_outlier_sample_ids",
  "surface_representative_evidence_ids",
  "surface_confidence",
  "semantic_pattern",
  "semantic_supporting_sample_ids",
  "semantic_outlier_sample_ids",
  "semantic_representative_evidence_ids",
  "semantic_confidence",
  "alternative_hypotheses",
  "failure_modes"
]

Requirements:
- When a track has supporting ids, its representative evidence ids must contain 2 to 3 ids from that same track's supporting ids. When a track has no supporting ids, its representative evidence ids must be an empty array.
- alternative_hypotheses and failure_modes must be JSON arrays of strings.
- Keep each prose field to at most 2 sentences and each prose array to at most 4 items.
- A majority claim requires at least 50% of the 50 unique normalized texts in that track's supporting ids.
- surface_confidence and semantic_confidence must each be numbers from 0 to 1 and use this calibration:
  0.00-0.20: no consistent pattern.
  0.21-0.40: only a weak pattern, or multiple equally plausible explanations.
  0.41-0.60: barely meets the majority condition, with clear confounds.
  0.61-0.80: the majority condition is clearly met and the track's main pattern is fairly consistent.
  0.81-1.00: coverage is very high and alternative explanations or confounds are weak.
```

## 中文等价审阅稿

```text
请将一个匿名 SAE latent 的 50 条最高激活语句作为一个整体进行分析。

latent 元信息：latent_idx={latent_idx}
证据规模：50 条语句，均为规范化后互不重复的文本。每条语句都是其规范化文本组中激活值最高的代表。

核心要求：
- 必须同时输出两条轨道。
- 表层轨道：识别能解释多数样本的重复词汇、短语、句法、话语标记、格式或领域术语模式；若不存在，明确说明。
- 语义轨道：识别能解释多数样本的重复语义内容或话语功能模式；若不存在，明确说明。
- 两条轨道可以描述不同模式，并拥有不同的支持样本。不要强迫两者一致。
- 必须综合分析全部 50 条语句，不要逐句生成 50 个独立分析。
- 对每一条轨道，必须将全部样本 ID 恰好分入该轨道的支持样本或离群样本。
- candidate_explanation 是面向人工阅读的简短综合解释，必须说明最有用的候选是表层形式、语义/话语功能，还是两条轨道之间尚不能判定。

Top-50 样本：
{top_50_samples}

只返回一个 JSON 对象，字段为：
[
  "latent_idx", "short_name", "candidate_explanation",
  "surface_pattern", "surface_supporting_sample_ids", "surface_outlier_sample_ids",
  "surface_representative_evidence_ids", "surface_confidence",
  "semantic_pattern", "semantic_supporting_sample_ids", "semantic_outlier_sample_ids",
  "semantic_representative_evidence_ids", "semantic_confidence",
  "alternative_hypotheses", "failure_modes"
]

字段约束：
- 某条轨道有 supporting ids 时，其 representative_evidence_ids 必须从对应 supporting ids 中选择 2--3 条；某条轨道没有 supporting ids 时，其 representative_evidence_ids 必须是空数组。
- alternative_hypotheses 与 failure_modes 必须是字符串数组。
- 每个说明文本最多 2 句话；每个说明数组最多 4 项。
- 任一轨道声称“多数模式”时，其支持样本必须至少覆盖 50 条唯一规范化文本中的一半。
- surface_confidence 与 semantic_confidence 都必须介于 0 和 1，且使用以下校准：
  0.00--0.20：没有一致模式。
  0.21--0.40：只有弱模式，或有多个同样合理的解释。
  0.41--0.60：刚好满足多数条件，但混淆因素明显。
  0.61--0.80：多数条件明确满足，且该轨道的主模式较一致。
  0.81--1.00：覆盖率很高，且替代解释或混淆因素很弱。
```
