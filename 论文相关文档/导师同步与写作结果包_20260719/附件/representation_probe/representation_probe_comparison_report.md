# Stable Core vs Top-n/PCA/Random SAE 线性探针对比报告

## 实验设置

- Labels: `RES, REC, QUO, QUC, GI, SU, AF`
- n grid: `10, 20, 50, 100, 200`
- Random SAE repeats: `20`
- Split: `stratified-group-kfold` with group column `file_id`
- Classifier: `LogisticRegression(class_weight="balanced", C=1.0, solver="liblinear")`
- Preprocessing: train-fold standardization before PCA and again on PCA scores before the probe.
- Metrics: AUC, PR-AUC / Average Precision, F1, Balanced Accuracy

## Macro 结果

| representation | n | AUC | PR-AUC | F1 | Balanced Acc. | features |
|---|---:|---:|---:|---:|---:|---:|
| Hidden State | baseline | 0.887 | 0.642 | 0.606 | 0.792 | 4096.000 |
| Full SAE | baseline | 0.876 | 0.551 | 0.529 | 0.799 | 32768.000 |
| Stable Core SAE | baseline | 0.853 | 0.557 | 0.484 | 0.789 | 32.571 |
| Top-n SAE | 10 | 0.810 | 0.477 | 0.448 | 0.755 | 10.000 |
| Top-n SAE | 20 | 0.839 | 0.512 | 0.468 | 0.775 | 20.000 |
| Top-n SAE | 50 | 0.856 | 0.539 | 0.492 | 0.791 | 50.000 |
| Top-n SAE | 100 | 0.865 | 0.544 | 0.500 | 0.791 | 100.000 |
| Top-n SAE | 200 | 0.852 | 0.532 | 0.505 | 0.783 | 200.000 |
| PCA-n | 10 | 0.872 | 0.454 | 0.430 | 0.794 | 10.000 |
| PCA-n | 20 | 0.900 | 0.538 | 0.478 | 0.820 | 20.000 |
| PCA-n | 50 | 0.917 | 0.596 | 0.526 | 0.838 | 50.000 |
| PCA-n | 100 | 0.921 | 0.641 | 0.553 | 0.838 | 100.000 |
| PCA-n | 200 | 0.920 | 0.655 | 0.598 | 0.835 | 200.000 |
| Random SAE-n | 10 | 0.552 | 0.145 | 0.193 | 0.542 | 10.000 |
| Random SAE-n | 20 | 0.581 | 0.160 | 0.215 | 0.561 | 20.000 |
| Random SAE-n | 50 | 0.644 | 0.205 | 0.249 | 0.605 | 50.000 |
| Random SAE-n | 100 | 0.699 | 0.253 | 0.285 | 0.649 | 100.000 |
| Random SAE-n | 200 | 0.741 | 0.295 | 0.325 | 0.684 | 200.000 |

## Stable Core 位置

- Stable Core SAE macro AUC = `0.853`。
- 相比 `Hidden State`，Stable Core SAE macro AUC 差值为 `-0.034`。
- 相比 `Full SAE`，Stable Core SAE macro AUC 差值为 `-0.023`。
- 相比 `Top-n SAE` 最佳点 n=100 (macro AUC=0.865)，Stable Core 差值为 `-0.013`。
- 相比 `PCA-n` 最佳点 n=100 (macro AUC=0.921)，Stable Core 差值为 `-0.068`。
- 相比 `Random SAE-n` 最佳点 n=200 (macro AUC=0.741)，Stable Core 差值为 `+0.112`。

## 解释边界

- `Stable Core SAE` 是独立 baseline，不参与 Top-n 曲线。
- `Top-n SAE` 的 top 是训练折内正向 Cohen's d 排名，避免测试折信息泄漏。
- `PCA-n` 在每个训练折单独拟合 PCA，不在全量数据上预拟合；默认对 PCA score 再按训练折标准化后送入 probe。
- `Random SAE-n` 从 filtered keep pool 抽样，误差带反映随机 latent 选择方差。
- 本实验是线性可解码性比较，不证明 latent 具有因果机制或临床概念语义。
