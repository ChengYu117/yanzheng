# MISC 后续可解释性阶段总报告

> 本报告汇总 R2 行为差异分析与 latent-level 案例解释分析。

## 1. 本阶段完成内容

- 行为差异分析：比较各 MISC 标签在 SAE 空间中的碎片化、共享和质量分层差异。
- latent 案例解释：为核心标签 top latent 生成高激活样本卡片。
- 阶段评估：判断哪些 latent 可进入人工命名和后续因果验证。

## 2. 行为差异结论

- 分析标签数：9
- Pattern 分布：{'shared_distributed': 4, 'negative_boundary': 3, 'compact_strong': 2}

最碎片化标签：

| Label | Sig. Latents | Frag. Ratio | Top AUC |
|---|---:|---:|---:|
| SU | 8691 | 0.265 | 0.571 |
| AF | 6891 | 0.210 | 0.800 |
| RES | 4470 | 0.136 | 0.626 |
| REC | 4192 | 0.128 | 0.611 |
| QUO | 3872 | 0.118 | 0.800 |

## 3. Latent 案例结论

- 生成 case cards：45
- 平均 top-example target purity：0.604
- 自动解释状态分布：{'high_purity_candidate': 20, 'mixed_but_label_relevant': 14, 'low_purity_review_required': 11}

## 4. 总体判断

当前证据链已经从 R1 的结构映射推进到 R2 的行为差异，并补上了 latent-level 案例审查入口。结果支持：标签与表征之间的错配不仅是多对多的，而且不同行为标签具有不同的碎片化和共享模式。

## 5. 输出位置

- 阶段输出目录：`outputs\misc_full_sae_eval\interpretability\followup_analysis`
- 行为差异报告：`outputs\misc_full_sae_eval\interpretability\followup_analysis\behavior_asymmetry\behavior_asymmetry_report.md`
- latent 案例报告：`outputs\misc_full_sae_eval\interpretability\followup_analysis\latent_cases\latent_case_report.md`
