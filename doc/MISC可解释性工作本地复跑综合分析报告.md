# MISC 可解释性工作本地复跑综合分析报告

> 生成日期：2026-04-30
> 复跑输入：`outputs/misc_full_sae_eval`
> 复跑输出：`outputs/misc_full_sae_eval/interpretability/local_rerun_20260430_202407`

## 1. 本次复跑内容

本次没有重新加载 Llama，也没有重新抽取 SAE features，而是基于已经完成的全量 MISC SAE 结果，复跑导师要求中“可解释性工作”的三个阶段：

1. Mapping Structure 分析
   脚本：`run_misc_mapping_structure_analysis.py`
   目标：分析 `latent × MISC label` 矩阵，回答标签空间和 SAE latent 空间是否是一一对应。

2. 后续可解释性分析
   脚本：`run_misc_interpretability_analysis.py`
   目标：分析不同行为标签的表征差异、质量分层差异，并生成 latent-level case cards。

3. 因果候选组导出
   脚本：`run_misc_causal_candidate_export.py`
   目标：为后续 causal intervention 准备每个标签的 G1/G5/G10/G20 latent 组。

本次复跑校验到的关键产物均已存在：

| 阶段 | 关键文件 |
|---|---|
| Mapping Structure | `mapping_structure_metrics.json`, `label_fragmentation_rank.csv`, `label_pair_similarity.csv`, `hierarchy_alignment.csv`, `latent_role_summary.csv` |
| Follow-up Interpretability | `behavior_asymmetry_summary.csv`, `quality_label_activation_shift.csv`, `latent_case_summary.csv`, `latent_case_metrics.json` |
| Causal Candidates | `causal_candidate_groups.json`, `candidate_group_summary.csv`, `label_candidates/*_candidate_latents.csv` |

对应测试也已通过：

| 测试 | 结果 |
|---|---|
| `test_mapping_structure_analysis.py` | 2 passed, 0 failed |
| `test_behavior_interpretability.py` | 2 passed, 0 failed |
| `test_causal_candidate_export.py` | 2 passed, 0 failed |
| `test_ai_re_judge_smoke.py` | 6 passed, 0 failed |

## 2. 总体判断

当前结果已经能够支持导师文档中的核心研究叙事：

> MISC 人工行为标签和 LLM 内部 SAE 表征不是一一对应，而是存在结构化、多对多、标签依赖的映射关系。

更具体地说：

- R1 Mapping Structure：证据强。多标签 latent 大量存在，标签明显碎片化，父子标签结构也被部分恢复。
- R2 Asymmetry Across Behaviors：证据较强。不同 MISC 行为呈现不同表征形态，有的紧凑，有的分散，有的更像边界/缺失信号。
- R3 Implications：已有初步证据。latent case cards 和候选组已经能支撑人工命名、AI 评审和后续因果验证，但现在还不能直接宣称 causal mechanism。

因此，这个 SAE 在当前 MISC 数据集上的定位应是：

> 可用于分析咨询行为标签与模型内部表征之间的结构关系，也可用于筛选候选因果 latent；但不应被简单写成“发现了某个标签对应的单一 latent”或“训练了一个新的分类器”。

## 3. R1：Mapping Structure 是否成立

### 3.1 多对多关系非常明显

| 指标 | 数值 |
|---|---:|
| MISC 标签数 | 10 |
| SAE latent 数 | 32768 |
| latent-label 矩阵行数 | 327680 |
| FDR 显著 latent-label 边 | 44308 |
| 至少关联一个标签的 latent | 13052 |
| 单标签 latent | 2147 |
| 多标签 latent | 10905 |
| 多标签 latent 占比 | 0.836 |

最关键的数字是：在所有至少关联一个标签的 latent 中，`83.6%` 是多标签 latent。这个比例很高，说明模型内部表征空间并不是按照 MISC 标签做离散分仓，而是用大量共享 latent 编码跨行为、跨标签家族的混合信号。

这直接支持 R1：MISC 标签和 SAE latent 是结构化多对多映射。

### 3.2 标签碎片化程度差异大

| Label | 显著 latent 数 | fragmentation ratio | 正向显著 | 负向显著 | top latent | top AUC |
|---|---:|---:|---:|---:|---:|---:|
| SU | 8691 | 0.265 | 234 | 8457 | 24760 | 0.571 |
| AF | 6891 | 0.210 | 342 | 6549 | 23464 | 0.800 |
| RES | 4470 | 0.136 | 99 | 4371 | 20808 | 0.626 |
| OTHER | 4467 | 0.136 | 513 | 3954 | 13430 | 0.714 |
| REC | 4192 | 0.128 | 2592 | 1600 | 31133 | 0.611 |
| QUO | 3872 | 0.118 | 972 | 2900 | 9959 | 0.800 |
| QU | 3522 | 0.107 | 1260 | 2262 | 13430 | 0.925 |
| RE | 3462 | 0.106 | 2012 | 1450 | 29759 | 0.694 |
| QUC | 2868 | 0.088 | 540 | 2328 | 21935 | 0.740 |
| GI | 1873 | 0.057 | 774 | 1099 | 13430 | 0.682 |

这里有两个值得挖掘的现象：

1. `SU/AF/RES` 的显著 latent 很多，但绝大多数是负向显著。它们更像是“边界型标签”：模型可能不是用一个清晰的正向语义特征识别它们，而是通过大量其他行为特征的缺失、抑制或反向模式来区分它们。
2. `QU/QUO` 的 top latent AUC 很高，尤其 `QU` top AUC 达到 `0.925`，说明提问类行为在 SAE 空间中有更紧凑、更强的局部表征，可能同时包含句法形式和咨询行为功能。

### 3.3 latent 角色结构支持多对多叙事

| latent role | latent 数 | 占比 | 平均关联标签数 |
|---|---:|---:|---:|
| cross_family | 6594 | 0.505 | 2.897 |
| global | 3328 | 0.255 | 6.230 |
| exclusive | 2416 | 0.185 | 1.179 |
| family_shared | 551 | 0.042 | 2.655 |
| auxiliary_only | 163 | 0.012 | 1.000 |

`exclusive` latent 只占 `18.5%`，而 `cross_family + global` 合计约 `76.0%`。这说明最主要的表征形态不是“一个 latent 对一个 MISC 标签”，而是跨标签共享和全局混合。

这对论文叙事很重要：人工标签压缩了连续的咨询行为空间，而 SAE 暴露了模型内部更细、更混合的行为表征结构。

## 4. R2：不同行为标签的表征是否不对称

后续可解释性分析把 9 个核心标签分成了三类模式：

| pattern | 标签数 | 典型标签 |
|---|---:|---|
| compact_strong | 2 | QU, QUO |
| shared_distributed | 4 | RE, REC, QUC, GI |
| negative_boundary | 3 | SU, AF, RES |

### 4.1 compact strong：QU / QUO 是最清晰的正例

`QU` 和 `QUO` 是当前最干净的解释性对象：

- `QU` top latent AUC = `0.925`，top Cohen's d = `2.584`
- `QUO` top latent AUC = `0.800`，top Cohen's d = `1.762`
- `QU-QUO` top-50 Jaccard = `0.408`
- `QU-QUO` 全显著集合 Jaccard = `0.532`
- `QU-QUO` Cohen's d Pearson = `0.877`

解释：开放式问题和总问题标签在 SAE 空间中高度接近，父标签 `QU` 的表征大部分可以由 `QUO/QUC` 子标签解释，其中 `QUO` 贡献更强。

研究价值：`QU/QUO` 适合当作“可解释性正例”，用于展示 SAE 能够捕捉清晰行为模式。

### 4.2 shared distributed：RE / REC 是强结构但非单 latent 结构

`RE` 和 `REC` 的关系最能支撑导师说的“结构化错配”：

- `RE-REC` top-50 Jaccard = `0.389`
- `RE-REC` 全显著集合 Jaccard = `0.645`
- `RE-REC` Cohen's d Pearson = `0.937`
- `RE` 的 parent decomposition = `0.958`
- `RE -> REC` child coverage = `0.716`
- `RE -> RES` child coverage = `0.359`

解释：`RE` 父标签在 SAE 空间中被明显恢复，但它更靠近 `REC`，而不是均匀覆盖 `RES` 与 `REC`。这可能有两个原因：

1. 数据分布中 `REC` 本身更多，模型对复杂反映更容易形成稳定表征。
2. `RES` 可能更像短促复述或边界行为，在模型表示中不如 `REC` 形成明确正向语义簇。

研究价值：`RE/REC/RES` 可以作为“父子标签被部分恢复，但恢复不对称”的核心案例。

### 4.3 negative boundary：SU / AF / RES 需要谨慎解释

`SU/AF/RES` 的共同特征是：显著 latent 多，但正向显著 latent 少。

| Label | 正向显著比例 | 负向显著比例 | pattern |
|---|---:|---:|---|
| SU | 0.027 | 0.973 | negative_boundary |
| AF | 0.050 | 0.950 | negative_boundary |
| RES | 0.022 | 0.978 | negative_boundary |

解释：这些标签可能不是靠大量“该行为本身”的正向激活来区分，而是靠其他行为信号的排除、缺失或负向边界来定义。尤其 `SU` 的 top AUC 只有 `0.571`，虽然碎片化很高，但单个 latent 的区分能力较弱。

研究价值：这类标签不适合直接写成“模型发现了支持/肯定的概念 latent”。更合理的写法是：SAE 显示这些人工标签在模型内部更像分散边界，而不是紧凑语义簇。

## 5. MISC 父子标签结构是否被 SAE 恢复

### 5.1 RE 家族

| 关系 | Jaccard | parent decomposition | child coverage | sibling separation |
|---|---:|---:|---:|---:|
| RE -> RES + REC | 0.463 | 0.958 | 0.473 | - |
| RE -> RES | 0.253 | 0.463 | 0.359 | - |
| RE -> REC | 0.645 | 0.867 | 0.716 | - |
| RES vs REC | 0.234 | - | - | 0.766 |

结论：`RE` 的父子层级被明显恢复，但恢复偏向 `REC`。`RES` 和 `REC` 又有较强 sibling separation，说明它们不是完全相同的表征子集。

### 5.2 QU 家族

| 关系 | Jaccard | parent decomposition | child coverage | sibling separation |
|---|---:|---:|---:|---:|
| QU -> QUO + QUC | 0.545 | 0.906 | 0.578 | - |
| QU -> QUO | 0.532 | 0.729 | 0.663 | - |
| QU -> QUC | 0.373 | 0.493 | 0.606 | - |
| QUO vs QUC | 0.222 | - | - | 0.778 |

结论：`QU` 家族也被部分恢复，且 `QUO` 比 `QUC` 更接近父标签 `QU`。这和咨询行为直觉一致：开放式问题常常是问题行为的典型形式，而封闭式问题可能混入更强的上下文控制或具体信息确认成分。

## 6. 质量分层结果：哪些行为和 high/low 会话更相关

在同一标签内部比较 high quality 与 low quality 会话时，得到：

| Label | high 样本数 | low 样本数 | high-low d |
|---|---:|---:|---:|
| RE | 1139 | 219 | 0.348 |
| REC | 744 | 98 | 0.292 |
| AF | 248 | 101 | 0.242 |
| QU | 1417 | 557 | -0.066 |
| GI | 329 | 352 | -0.105 |
| SU | 121 | 101 | -0.119 |
| QUO | 941 | 265 | -0.128 |
| RES | 395 | 121 | -0.248 |
| QUC | 476 | 292 | -0.496 |

可挖掘的结论：

1. `RE/REC` 在 high quality 会话中更强，符合咨询质量直觉，也支持把反映性倾听作为后续因果验证主线。
2. `QUC` 在 low quality 会话中更强，可能说明封闭式问题在当前数据中与较低质量会话相关；但这只是相关，不是因果。
3. `QU/QUO` 的高低质量差异很小或略偏 low，说明“提问”本身不是质量的充分条件；问题形式、上下文和后续回应方式可能更重要。
4. `RES` 负向质量差异值得人工检查。它可能表示简单复述在低质量会话中更常见，也可能是数据或标注口径导致的混合效应。

这部分可以作为论文讨论中的“行为标签和会话质量并非线性对应”的证据，但不应写成因果结论。

## 7. latent case cards 质量

本次生成了 45 个 latent case cards：

| 指标 | 数值 |
|---|---:|
| case cards | 45 |
| 平均 target purity | 0.604 |
| high_purity_candidate | 20 |
| mixed_but_label_relevant | 14 |
| low_purity_review_required | 11 |

判断：

- 约 44.4% 的 case cards 是高纯度候选，适合进入人工命名或 AI 评审。
- 约 31.1% 是“混合但相关”，这类 latent 反而很适合支撑多对多叙事。
- 约 24.4% 需要人工复查，不适合直接作为论文中的代表案例。

因此，latent case cards 当前已经足够作为解释性分析入口，但还需要挑选少量最稳的案例进入正式报告或论文。

优先人工检查的 latent：

| 标签 | 优先 latent |
|---|---|
| RE | 19435, 29759, 30224, 20436, 31930 |
| REC | 20436, 31133, 26800, 30224, 19435 |
| QU | 13430, 664, 9959, 11660, 26485 |
| QUO | 9959, 26485, 664, 13430, 24761 |
| QUC | 14014, 21935, 20869, 13430, 22358 |
| AF | 23464, 7143, 30870, 17793, 18492 |

## 8. 因果候选组导出结果

每个核心标签都已导出 G1/G5/G10/G20、bottom control 和 random control。

| Label | pattern | candidates | G1 | G20 高纯度数 |
|---|---|---:|---:|---:|
| RE | shared_distributed | 2012 | 19435 | 1 |
| RES | negative_boundary | 99 | 20808 | 0 |
| REC | shared_distributed | 2592 | 20436 | 1 |
| QU | compact_strong | 1260 | 13430 | 5 |
| QUO | compact_strong | 972 | 9959 | 1 |
| QUC | shared_distributed | 540 | 14014 | 2 |
| GI | shared_distributed | 774 | 16345 | 4 |
| SU | negative_boundary | 234 | 24760 | 2 |
| AF | negative_boundary | 342 | 23464 | 4 |

建议的因果验证顺序：

1. 先跑 `RE`：和现有 RE/NonRE 因果流程兼容，能延续之前实验主线。
2. 再跑 `QU` 或 `QUO`：它们是 compact strong 标签，最适合做清晰正例。
3. 再跑 `REC`：用于验证 RE 家族中更强的子标签结构。
4. 谨慎跑 `AF/SU/RES`：它们更像 negative-boundary 标签，优先测试 necessity 或 side effect，而不要过早期待强 sufficiency。

## 9. AI 专家代理评审接口检查

项目中还存在 `run_ai_re_judge.py`，用于对 `judge_bundle` 做 OpenAI-compatible API 的 RE 专家代理评审。该阶段属于外部模型调用，不是本次纯本地矩阵分析的一部分。

本次做了本地接口检查：

| 检查项 | 结果 |
|---|---|
| `python run_ai_re_judge.py --help` | 通过 |
| `python test_ai_re_judge_smoke.py` | 6 passed, 0 failed |

因此当前判断是：AI judge 管线本身可用，但本次没有正式调用外部 API，也没有把 AI judge 结果计入本报告的研究结论。后续如要把 latent case cards 升级为“自动专家代理评审证据”，需要配置 `OPENAI_API_KEY`、`OPENAI_MODEL` 和可选 `OPENAI_BASE_URL` 后单独运行。

## 10. 对当前结果“做得怎么样”的评价

### 10.1 已经做得比较好的部分

1. 多对多结构证据非常强
   `10905` 个多标签 latent 对比 `2147` 个单标签 latent，足以支撑 R1。

2. 父子标签恢复清晰
   `RE-REC` 和 `QU-QUO` 都表现出强相似性，说明 SAE 空间不是随机噪声，而是部分恢复了 MISC 层级结构。

3. 行为不对称结果有研究价值
   `QU/QUO` 紧凑，`RE/REC` 分布式共享，`SU/AF/RES` 更像负向边界。这比简单做分类器更符合导师要求。

4. 已经具备下一步因果验证入口
   每个标签都有 G1/G5/G10/G20 候选组和对照组，工程上已经能接云端或本地因果脚本。

### 10.2 还需要谨慎的部分

1. 显著相关不等于可解释概念
   特别是 global latent 和 cross-family latent，不能直接命名为单一 MISC 行为。

2. negative-boundary 标签不能按正向概念解释
   `SU/AF/RES` 大量负向显著，说明它们可能是边界信号，而不是清晰正向语义簇。

3. 质量分层只是相关分析
   `RE/REC` 在 high quality 中更强，`QUC/RES` 在 low quality 中更强，但不能直接写成“这些行为导致质量变化”。

4. case cards 需要人工或 AI 评审
   平均 purity 为 `0.604`，说明有不少候选可用，但正式论文例子仍需要人工筛选。

## 11. 可以写入论文/报告的主结论

建议主结论写成下面这种口径：

> 在 6194 条 MISC 行为单元上，SAE latent 与 MISC 标签之间呈现显著的结构化多对多映射。单个行为标签通常分散到大量 latent，而单个 latent 也常常同时关联多个标签。该结构并非随机重叠：RE/REC 与 QU/QUO 等父子或相近标签在 SAE 空间中表现出更高共享度，同时 RES、REC 与 QUO、QUC 等 sibling 标签仍保持一定分离。这说明人工 MISC 标签压缩了模型内部更连续、更混合的咨询行为表征。

可作为补充结论：

> 不同行为标签具有不同表征形态。提问类标签，尤其 QU/QUO，呈现更紧凑的强 latent 结构；反映类标签 RE/REC 更像分布式共享结构；支持、肯定和简单反映类标签则更像由负向边界或缺失信号定义。这种不对称性说明，用单一分类器指标评价咨询行为会掩盖内部表征结构差异。

## 12. 下一步建议

1. 挑选 6 到 10 个高质量 latent case cards 做人工命名
   优先选择 `QU/QUO` 的 compact strong latent 和 `RE/REC` 的 shared latent。

2. 跑第一轮因果验证
   先跑 `RE` 的 G1/G5/G10/G20，再跑 `QU` 或 `QUO` 作为清晰正例。

3. 做 overlap latent 的副作用分析
   检查干预 `RE` latent 是否同时改变 `REC/QU/OTHER` 等方向，避免把共享 latent 误解成单标签机制。

4. 对 negative-boundary 标签单独成段讨论
   `SU/AF/RES` 不应强行纳入“清晰概念 latent”叙事，而应作为人工标签和模型内部表征错配的典型证据。

5. 在正式文档中区分三类证据
   Mapping evidence 是统计结构证据，case cards 是语义解释证据，causal intervention 才是机制证据。

## 13. 复跑命令记录

```powershell
conda run --no-capture-output -n qwen-env-py311 python run_misc_mapping_structure_analysis.py `
  --mapping-dir outputs/misc_full_sae_eval/functional/misc_label_mapping `
  --output-dir outputs/misc_full_sae_eval/interpretability/local_rerun_20260430_202407/mapping_structure `
  --doc-report outputs/misc_full_sae_eval/interpretability/local_rerun_20260430_202407/mapping_structure/mapping_structure_doc_report.md

conda run --no-capture-output -n qwen-env-py311 python run_misc_interpretability_analysis.py `
  --eval-dir outputs/misc_full_sae_eval `
  --output-dir outputs/misc_full_sae_eval/interpretability/local_rerun_20260430_202407/followup_analysis `
  --doc-report outputs/misc_full_sae_eval/interpretability/local_rerun_20260430_202407/followup_analysis/followup_doc_report.md

conda run --no-capture-output -n qwen-env-py311 python run_misc_causal_candidate_export.py `
  --eval-dir outputs/misc_full_sae_eval `
  --mapping-dir outputs/misc_full_sae_eval/functional/misc_label_mapping `
  --mapping-structure-dir outputs/misc_full_sae_eval/interpretability/local_rerun_20260430_202407/mapping_structure `
  --followup-dir outputs/misc_full_sae_eval/interpretability/local_rerun_20260430_202407/followup_analysis `
  --output-dir outputs/misc_full_sae_eval/interpretability/local_rerun_20260430_202407/causal_candidates `
  --doc-report outputs/misc_full_sae_eval/interpretability/local_rerun_20260430_202407/causal_candidates/causal_candidate_doc_report.md

conda run --no-capture-output -n qwen-env-py311 python run_ai_re_judge.py --help

conda run --no-capture-output -n qwen-env-py311 python test_ai_re_judge_smoke.py

conda run --no-capture-output -n qwen-env-py311 python test_mapping_structure_analysis.py

conda run --no-capture-output -n qwen-env-py311 python test_behavior_interpretability.py

conda run --no-capture-output -n qwen-env-py311 python test_causal_candidate_export.py
```
