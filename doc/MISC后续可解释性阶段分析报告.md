# MISC 后续可解释性阶段总报告

> 本报告汇总 R2 行为差异分析与 latent-level 案例解释分析。

## 1. 本阶段完成内容

- 行为差异分析：比较各 MISC 标签在 SAE 空间中的碎片化、共享和质量分层差异。
- latent 案例解释：为核心标签 top latent 生成高激活样本卡片。
- 阶段评估：判断哪些 latent 可进入人工命名和后续因果验证。

## 2. 行为差异结论

- 分析标签数：9
- Pattern 分布：{'mixed_distributed': 7, 'compact_strong': 1, 'shared_distributed': 1}

最碎片化标签：

| Label | TopK | Shared Ratio | Top AUC |
|---|---:|---:|---:|
| QU | 20 | 0.950 | 0.925 |
| RE | 20 | 0.700 | 0.694 |
| QUO | 20 | 0.700 | 0.800 |
| REC | 20 | 0.650 | 0.611 |
| QUC | 20 | 0.350 | 0.740 |

## 3. Latent 案例结论

- 生成 case cards：180
- 平均 top-example target purity：0.546
- 自动解释状态分布：{'high_purity_candidate': 64, 'mixed_but_label_relevant': 63, 'low_purity_review_required': 53}

## 4. 总体判断

当前证据链已经从 R1 的结构映射推进到 R2 的行为差异，并补上了 latent-level 案例审查入口。结果支持：标签与表征之间的错配不仅是多对多的，而且不同行为标签具有不同的碎片化和共享模式。

## 5. 输出位置

- 阶段输出目录：`outputs\misc_full_sae_eval\interpretability\followup_analysis`
- 行为差异报告：`outputs\misc_full_sae_eval\interpretability\followup_analysis\behavior_asymmetry\behavior_asymmetry_report.md`
- latent 案例报告：`outputs\misc_full_sae_eval\interpretability\followup_analysis\latent_cases\latent_case_report.md`

## 6. Top20 cutoff 合理性补充审计

本阶段额外检查了 Top20 是否存在过硬截断。审计输出位于：

- `outputs\misc_full_sae_eval\interpretability\topk_cutoff_audit`

主要结论：

1. Top20 不是统计上的天然断点，而是为了跨标签可比性设置的固定解释窗口。
2. 所有标签的 rank20 与 rank21/22 都非常接近；rank21 的 `abs_cohens_d` 均达到 rank20 的 95% 以上。
3. 因此，如果目标是人工审查或因果候选，不应机械地只看 Top20，可以把 rank21-25 作为边界候选池。
4. 如果目标是论文主表和跨标签结构比较，仍建议保留 Top20，因为它能保持每个标签候选数量一致，便于比较 shared ratio、role taxonomy 和 hierarchy alignment。
5. RES、GI、SU、QUC、AF 的 Top20 尾部存在较多弱候选标记，解释时应更谨慎；QU、QUO、REC 的 rank21-25 中仍有若干值得复核的近边界候选。

建议采用“双口径”：

- 论文主分析：继续使用 Top20，保证可比性。
- 人工命名/因果验证：使用 adaptive pool，即 Top20 加 rank21-25 中接近 rank20 且通过基础质量门槛的候选；同时剔除明显低 AUC、低效应量或低纯度的尾部候选。
