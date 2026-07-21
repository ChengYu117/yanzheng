# Latent card 论文示例包

> Status: Frozen attachment / Pending final author selection
>
> 解释与指标均逐字复制自 2026-07-18 的 GPT-5.5 reduced-context 正式产物；中文入选理由是人工审核注释，不属于模型输出。

## 推荐优先用于论文的 6 张主例

| Feature | Latent | Stable-core labels | 类型 | 中文概括 | Spearman | AUROC | High–weak |
|---|---:|---|---|---|---:|---:|---:|
| [F060](cards/F060_latent_23183.md) | 23183 | QUO | linguistic_structure | 直接 what 问句 | 0.961 | 1.000 | 1.000 |
| [F113](cards/F113_latent_27857.md) | 27857 | QUC | linguistic_structure | 数字量表评分引导 | 0.847 | 0.867 | 1.000 |
| [F058](cards/F058_latent_664.md) | 664 | QUC+QUO | behavioral_function | 以客户为中心的信息引导问题 | 0.776 | 0.920 | 0.940 |
| [F150](cards/F150_latent_1713.md) | 1713 | GI | behavioral_function | 患者特异的临床解释或指令 | 0.733 | 0.920 | 0.840 |
| [F165](cards/F165_latent_29825.md) | 29825 | SU | linguistic_structure | 明确认可或许可框架 | 0.890 | 0.880 | 1.000 |
| [F176](cards/F176_latent_23464.md) | 23464 | AF | behavioral_function | 明确的亲和性承接 | 0.871 | 0.973 | 1.000 |

## 3 张强但需谨慎解释的对照例

| Feature | Latent | Stable-core labels | 类型 | 中文概括 | Spearman | AUROC | 主要风险 |
|---|---:|---|---|---|---:|---:|---|
| [F015](cards/F015_latent_31133.md) | 31133 | REC | linguistic_structure | sounds/seems like 反映模板 | 0.862 | 1.000 | records 缺少前一轮 client 话语，因此只能可靠确认模板，不能确认反映内容是否准确。 |
| [F047](cards/F047_latent_8583.md) | 8583 | REC | behavioral_function | 对负面自我认知的第二人称反映 | 0.698 | 0.860 | 功能解释依赖对负面内部状态的语义归纳，且缺少前文，可靠性低于纯语言结构卡。 |
| [F014](cards/F014_latent_13966.md) | 13966 | RES | linguistic_structure | you are/you're 陈述式刻画 | 0.868 | 0.960 | 主要支持语言代理而不是反映功能；无 client 前文时不能据此认定为简单反映。 |

## 选择原则

1. Scorer 结构验证通过，且证据片段逐字有效；
2. held-out Spearman、AUROC 与 high–weak 区分整体较高；
3. 解释条件足够窄，能够指出句子中可审核的证据；
4. 主例覆盖语言结构与行为功能，并覆盖 QUO、QUC、GI、SU、AF；
5. REC/RES 因缺少前一轮 client 话语，单列为带边界的强例，不把形式代理升级为功能或因果结论。

## 使用建议

- 图或正文优先使用 F060、F113、F058、F150、F165、F176；
- F015、F047、F014 更适合方法讨论、误差边界或语言代理分析；
- 每张卡的完整 discovery 强/弱句与 held-out 预测见 `cards/`；机器可读版本见 `selected_example_packets.jsonl`。
