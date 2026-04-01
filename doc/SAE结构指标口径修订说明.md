# SAE结构指标口径修订说明

本说明用于解释 `2026-03-31` 之后结构指标结果的统计口径，避免将旧结果与新结果混用。

## 1. 结构指标口径修订

本项目现在固定采用以下定义：

- `mse`
  - 按元素平均的平方误差
  - 公式：`sum((z - z_hat)^2) / (n_tokens * d_model)`
- `fvu`
  - `SSE / SST_centered`
  - `SSE = sum((z - z_hat)^2)`
  - `SST_centered = sum((z - mean(z))^2)`
- `explained_variance`
  - 固定定义为 `1 - fvu`

这与旧实现不同。旧实现实际上更接近：

- `Var(z - z_hat) / Var(z)`

这种写法会漏掉“系统性偏移”误差。比如 `z_hat = z + 常数偏移` 时，旧口径会错误地给出很好的 `EV/FVU`，而新口径会把这种偏移正确算进误差。

## 2. 输出元信息

新的 `metrics_structural.json` 会包含：

- `metric_definition_version = 2`
- `structural_scope = "sample_batches"` 或 `"full_dataset"`

其中：

- 顶层结构指标始终代表 `raw-space`
- `space_metrics.raw`
- `space_metrics.normalized`

用于区分原始激活空间与归一化空间的结果。

## 3. SAE forward 对齐修订

本项目已将 Hugging Face checkpoint 中的 `top_k` 视为模型定义的一部分。

当前 SAE 激活路径为：

1. `JumpReLU`
2. 对每个 token 的 latent 向量执行 `top_k` 截断

这意味着：

- 如果 checkpoint 带有 `top_k=50`，则每个 token 最多保留 `50` 个非零 latent
- `encode()` 与 `forward_with_details()["latents"]` 现在使用同一套稀疏激活逻辑

## 4. 结果解释建议

今后在论文或报告里引用结构指标时，应优先使用：

- `--full-structural` 结果
- `metric_definition_version = 2` 的结果

旧目录中的结构指标，如果没有这些元信息，应视为旧口径结果，只能做历史参考，不能直接用于最终科研结论。
