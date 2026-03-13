# qwen-env 运行报告

## 1. 运行目标

本次运行目标是：

1. 切换到 `qwen-env` 环境。
2. 实际执行 SAE-RE 评估项目。
3. 记录运行结果。
4. 给出结果解释和工程问题定位。

## 2. 环境信息

- Python:
  - `C:\Users\chengyu\AppData\Local\miniconda3\envs\qwen-env\python.exe`
- Python 版本:
  - `3.10.19`
- PyTorch:
  - `2.6.0+cu124`
- CUDA:
  - `True`
- GPU 数量:
  - `1`

本次是在 `qwen-env` 解释器下直接运行，没有依赖 shell 级 `conda activate`。

## 3. 实际执行的内容

### 3.1 smoke test

执行：

```powershell
python test_pipeline_smoke.py
```

结果：

- 初次运行被 Windows 控制台 `gbk` 编码卡住，`✓` / `✗` 字符无法输出。
- 改用 `PYTHONIOENCODING=utf-8` 后，前半段测试可以正常执行。
- 但完整 smoke log 未稳定收敛到最终汇总行，因此这次更可靠的结论来自正式评估实跑。

### 3.2 正式评估

执行：

```powershell
python run_sae_evaluation.py --skip-ce-kl --output-dir outputs/sae_eval_20260310_skipcekl
```

另外又尝试了一次缩小候选数量的配置：

```powershell
python run_sae_evaluation.py --skip-ce-kl --sae-config config/sae_config_smallrun.json --output-dir outputs/sae_eval_20260310_smallrun
```

说明：

- 两次都跳过了 CE/KL，因此这次报告不包含分布干预指标。
- 第二次不是改代码，只是缩小 `top_k_candidates`，用于验证功能评估后半段是否能完整跑通。

## 4. 运行过程观察

### 4.1 基模加载

本地 `Llama-3.1-8B` 成功加载。

日志显示：

- checkpoint shards: `4/4` 正常载入
- `device_map=auto`
- 单卡 CUDA 可用

### 4.2 SAE 下载与载入

SAE 从 Hugging Face 成功下载。

运行时关键观察：

- 下载 `final.safetensors` 时出现过一次 SSL retry，但最终成功。
- checkpoint 中实际读到的键只有：
  - `decoder.bias`
  - `decoder.weight`
  - `encoder.bias`
  - `encoder.weight`
- 本地 loader 因此将 `b_pre` 视为缺失项，并用零初始化：
  - `Non-critical keys using zero-init: {'b_pre'}`

这说明当前这份 checkpoint 在运行时并没有显式提供 `pre_bias`。

### 4.3 streaming 主路径

主数据流成功执行到：

- 激活提取
- SAE 前向
- utterance 聚合
- 结构性评估
- 功能性评估中的 univariate 阶段

其中 streaming 阶段日志显示：

- `400` 个 batch
- 总耗时约 `4 分 55 秒`
- 最终得到：
  - `utterance_features: [1598, 32768]`
  - `utterance_activations: [1598, 4096]`

## 5. 成功产出的文件

### 5.1 默认运行

目录：

- `outputs/sae_eval_20260310_skipcekl/`

实际产出：

- `metrics_structural.json`
- `candidate_latents.csv`

### 5.2 缩小候选运行

目录：

- `outputs/sae_eval_20260310_smallrun/`

实际产出：

- `metrics_structural.json`
- `candidate_latents.csv`

### 5.3 未产出的文件

两次运行都没有产出：

- `metrics_functional.json`
- `latent_cards/`

这说明功能评估阶段没有完整走完。

## 6. 结构性结果解读

默认运行得到的结构指标如下：

- `MSE = 4.5804`
- `cosine_similarity = 0.8088`
- `explained_variance = 0.0682`
- `FVU = 0.9318`
- `L0 mean = 172.6`
- `L0 std = 506.1`
- `dead_count = 29507`
- `dead_ratio = 90.05%`
- `alive_count = 3261`

### 6.1 结构保真度

这些数字说明：

- 方向保留还可以：
  - cosine ~ `0.81`
- 但方差重建能力很弱：
  - explained variance 只有 `6.8%`
  - FVU 高达 `93.2%`

含义是：

- SAE 重建后的激活方向与原激活有一定一致性。
- 但它并没有重构掉大部分原始变化量。
- 如果你的研究目的偏向“概念特征发现”，这不一定是坏事。
- 如果你的研究目的偏向“高保真替代原层表示”，那这组 SAE 的重建质量偏弱。

### 6.2 稀疏性

`32768` 个 latent 中：

- 平均每个 token 激活约 `172.6` 个 latent
- 死特征占比 `90.05%`
- 只有 `3261` 个 latent 在 sample 中表现为 alive

这意味着：

- SAE 非常稀疏
- 但也说明在当前 sample 上，真正参与表示的 latent 子集很小

## 7. 候选 latent 结果解读

`candidate_latents.csv` 成功生成，统计如下：

- 总 latent 数：
  - `32768`
- BH-FDR 显著 latent 数：
  - `1540`

Top 10 latent 中：

- 正向 RE 相关：
  - `8` 个
- 反向 RE 相关：
  - `2` 个

前几名候选如下：

| rank | latent_idx | Cohen's d | AUC | 解释 |
|------|------------|-----------|-----|------|
| 1 | 19435 | 0.8991 | 0.7003 | 强正向 RE 候选 |
| 2 | 13430 | -0.8887 | 0.3279 | 强反向候选，更像 NonRE 特征 |
| 3 | 31930 | 0.8796 | 0.6966 | 强正向 RE 候选 |
| 4 | 5663 | 0.8470 | 0.6989 | 强正向 RE 候选 |
| 5 | 29759 | 0.8232 | 0.6722 | 中高强度正向候选 |
| 6 | 1516 | 0.7652 | 0.6802 | 稳定正向候选 |
| 7 | 1211 | 0.7526 | 0.6937 | 稳定正向候选 |
| 8 | 26681 | 0.7526 | 0.7500 | 很强的判别候选 |

### 7.1 如何理解这些数字

- `Cohen's d > 0`
  - latent 在 RE 样本中更强
- `Cohen's d < 0`
  - latent 在 NonRE 样本中更强
- `AUC > 0.65`
  - 基本具备单特征判别信息
- `AUC < 0.35`
  - 反向区分能力较强

因此，这次运行已经说明：

- 该 SAE 中确实存在一批与 RE / NonRE 显著相关的候选 latent
- 至少在单 latent 统计层面，研究假设是成立的

## 8. 本次运行中断点

两次运行都停在同一个位置：

- `Functional Evaluation`
- `univariate analysis` 完成
- 之后没有产出 `metrics_functional.json`

从产物和日志看，可以确认：

1. 功能评估已经进入 `run_functional_evaluation()`
2. `candidate_latents.csv` 已写出
3. 但后续阶段没有完成

这意味着问题更可能出现在：

- `sparse_probing()`
- `maxact_analysis()`
- `feature_absorption()`
- `feature_geometry()`
- `targeted_probe_perturbation()`

中的某一个或多个阶段。

## 9. 对中断原因的技术判断

结合当前代码，我认为最值得怀疑的点有三个：

### 9.1 `feature_absorption()` 复杂度过高

它会对每个候选 latent 扫描整个 `d_sae=32768` 空间并逐个做相关系数计算。  
这一部分在真实数据上代价很高，是最可疑的性能瓶颈。

### 9.2 post-univariate 阶段缺少稳定日志

`candidate_latents.csv` 已经写出，但日志没有留下后续阶段的明确完成标记。  
这说明：

- 要么后续阶段异常退出
- 要么输出缓冲没有及时刷新
- 要么某一步进入了非常慢但没有进度条的计算

### 9.3 checkpoint 缺失 `b_pre`

这次实跑确认 `b_pre` 并不在 checkpoint 中，而是被本地 loader 零初始化。  
这不会阻止运行，但说明当前本地 SAE 前向与 checkpoint 的“最小可用结构”一致，不代表百分之百还原原始训练时的全部参数化细节。

## 10. 本次运行结论

可以给出一个分层结论：

### 10.1 已确认成功的部分

- `qwen-env` 可用
- CUDA 可用
- 本地 Llama-3.1-8B 能加载
- SAE checkpoint 能下载并加载
- streaming 主路径能跑通
- 结构性指标能产出
- 单 latent 候选筛选能产出

### 10.2 尚未成功的部分

- 完整功能评估没有跑完
- `metrics_functional.json` 没有生成
- MaxAct / Absorption / Geometry / TPP 的最终结果本次没有拿到

### 10.3 研究层面的意义

即使本次功能评估未完整结束，已有结果已经能支持一个初步判断：

- 这组 SAE 内确实存在一批与 RE 有显著统计关联的 latent
- 但项目当前还没有把这些候选 latent 完整推进到“探针 + 定性卡片 + 吸收 + 几何 + TPP”的全链路结果

换句话说：

- 候选已经找到了
- 完整解释报告还没有完全跑出来

## 11. 建议的下一步

建议按这个顺序继续：

1. 单独定位 `run_functional_evaluation()` 在 univariate 之后的失败点。
2. 优先检查 `feature_absorption()` 的复杂度和日志输出。
3. 给功能评估每个阶段增加明确的开始/结束日志。
4. 如果需要严格架构一致性，再确认 `b_pre` 是否应视为必需权重。
5. 在功能阶段稳定后，再补跑一次包含 CE/KL 的完整评估。
