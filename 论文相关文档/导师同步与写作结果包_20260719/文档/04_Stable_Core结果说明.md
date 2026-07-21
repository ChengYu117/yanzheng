# Stable Core结果说明

## 冻结结果

| 标签 | K* | Stable-core边 | 标签选择状态 |
|---|---|---|---|
| RES | 55 | 14 | performance_only_unstable |
| REC | 58 | 46 | stable_topk_found |
| QUO | 45 | 42 | stable_topk_found |
| QUC | 50 | 40 | stable_topk_found |
| GI | 85 | 26 | performance_only_unstable |
| SU | 30 | 17 | performance_only_unstable |
| AF | 58 | 43 | stable_topk_found |

- 总label–latent边：228。
- 去重latent：218。
- 单标签latent：208。
- 两标签共享latent：10。
- RES–REC共享4个；QUO–QUC共享5个；REC–SU共享1个。

## 科学解释

结果更符合“标签由多个碎片化证据共同表示”，而不是“一标签一latent”。共享主要发生在同一行为家族的叶标签之间，但共享比例较低，说明家族共性与叶标签差异同时存在。

## 必须保留的限定

- RES、GI、SU的标签级状态是`performance_only_unstable`：其内部成员满足宽松可复现条件，但不能声称整个标签集合形成稳定平台。
- 跨质量差异属于稳健性审计，不是当前成员硬剔除条件。
- 主线没有RE/QU父标签stable-core，不能直接写父子层级分解。
