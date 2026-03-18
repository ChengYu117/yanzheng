# GCE 云 GPU 部署说明

本文档对应仓库内的 `deploy/gce/` 脚本组，目标是把 SAE-RE 项目部署到 Google Compute Engine 的单机 GPU 虚拟机上运行。

## 1. 推荐机器规格

- 机器类型：`g2-standard-8`
- GPU：`1 x NVIDIA L4`
- 系统：`Ubuntu 22.04`
- 数据盘：`200GB pd-balanced`
- 镜像建议：选择 Google 提供的 GPU/Deep Learning VM 镜像，保证 NVIDIA 驱动和 CUDA 已预装

这套规格的目标是先把 `Llama 3.1 8B + SAE + 因果验证` 稳定跑起来，不追求最低成本。

## 2. 本地打包

在仓库根目录执行：

```powershell
python package_project.py
```

产物会写到：

```text
dist/nlp-re-gce-src-YYYYMMDD.tar.gz
```

这个发布包默认包含：

- `src/`
- `causal/`
- `config/`
- `data/mi_re/`
- `deploy/`
- `doc/`
- 主入口脚本和依赖声明

默认不包含：

- `outputs/`
- 本地模型权重
- Hugging Face 缓存
- 虚拟环境和日志

## 3. 创建 GCE GPU 虚拟机

推荐直接在 Google Cloud Console 中创建：

1. 新建 Compute Engine VM
2. 选择 `g2-standard-8`
3. 添加 `1 x NVIDIA L4`
4. 选择 Ubuntu 22.04 的 GPU/Deep Learning VM 镜像
5. 再挂载一个 `200GB` 的数据盘，挂载点建议后续统一为 `/mnt/disks/data`

如果你偏好命令行，也可以使用 `gcloud compute instances create`，但镜像家族名变化较快，建议在控制台选择预装驱动镜像，避免后续手动装驱动。

## 4. 上传并解压源码包

本地上传：

```powershell
gcloud compute scp dist/nlp-re-gce-src-YYYYMMDD.tar.gz <vm-name>:~/ --zone <zone>
```

登录远程机器：

```powershell
gcloud compute ssh <vm-name> --zone <zone>
```

远程解压：

```bash
mkdir -p ~/nlp-re
tar -xzf ~/nlp-re-gce-src-YYYYMMDD.tar.gz -C ~/nlp-re
cd ~/nlp-re
```

## 5. 配置环境变量

复制模板：

```bash
cp deploy/gce/env.example deploy/gce/.env
```

至少修改这些项：

- `HF_TOKEN`
- `MODEL_HF_REPO_ID`
- `MODEL_DIR`
- `HF_HOME`
- `OUTPUT_ROOT`

默认目录约定是：

- 模型：`/mnt/disks/data/models/Llama-3.1-8B`
- HF 缓存：`/mnt/disks/data/hf-cache`
- 输出目录：`/mnt/disks/data/outputs`

## 6. 初始化环境

在仓库根目录执行：

```bash
bash deploy/gce/bootstrap.sh
```

这个脚本会做这些事：

- 创建 Python 3.10 虚拟环境
- 安装 PyTorch 和项目依赖
- 安装 `tmux`
- 校验 GPU、Torch 和主入口脚本是否可用

## 7. 下载基础模型

```bash
bash deploy/gce/download_model.sh
```

这个脚本会把 Hugging Face 上的基础模型下载到 `MODEL_DIR`。  
如果你使用的是受限模型，`HF_TOKEN` 必须具有相应访问权限。

## 8. 运行 SAE 主流程

建议放到 `tmux` 里：

```bash
tmux new -s sae
bash deploy/gce/run_full_eval.sh
```

如果需要附加参数，可以直接追加：

```bash
bash deploy/gce/run_full_eval.sh --compare-mean
```

默认输出目录：

```text
/mnt/disks/data/outputs/sae_eval_full
```

## 9. 运行因果验证流程

确认 SAE 主流程已经产出 `candidate_latents.csv` 后，再执行：

```bash
tmux new -s causal
bash deploy/gce/run_causal.sh
```

如果想先做一版轻量 smoke：

```bash
bash deploy/gce/run_causal.sh --skip-side-effects
```

默认输出目录：

```text
/mnt/disks/data/outputs/causal_validation_full
```

## 10. 验收清单

部署完成后，至少确认以下几点：

- `nvidia-smi` 正常
- `python run_sae_evaluation.py --help` 正常
- `python causal/run_experiment.py --help` 正常
- `python run_ai_re_judge.py --help` 正常
- `MODEL_DIR` 下存在完整模型文件
- `OUTPUT_ROOT` 下开始生成实验目录和 `run.log`

## 11. 常用命令

打包：

```powershell
python package_project.py
```

初始化环境：

```bash
bash deploy/gce/bootstrap.sh
```

下载模型：

```bash
bash deploy/gce/download_model.sh
```

运行 SAE：

```bash
bash deploy/gce/run_full_eval.sh
```

运行因果：

```bash
bash deploy/gce/run_causal.sh
```
