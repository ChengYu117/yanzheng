# PAI / ACR / EAS 快速部署

## 1. 准备镜像参数

PowerShell:

```powershell
$env:ACR_REGISTRY="registry-vpc.cn-hangzhou.aliyuncs.com"
$env:ACR_NAMESPACE="your-namespace"
$env:ACR_REPOSITORY="sae-re-job"
$env:IMAGE_TAG="v1"
```

## 2. 构建并推送镜像

PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File deploy/pai/build_and_push.ps1
```

Bash:

```bash
export ACR_REGISTRY=registry-vpc.cn-hangzhou.aliyuncs.com
export ACR_NAMESPACE=your-namespace
export ACR_REPOSITORY=sae-re-job
export IMAGE_TAG=v1
bash deploy/pai/build_and_push.sh
```

## 3. 在 EAS 中使用模板

- 打开 `deploy/pai/eas_service.json`
- 替换：
  - `__REGION_ID__`
  - `__OSS_BUCKET__`
  - `__IMAGE_URI__`
- 在 PAI-EAS 控制台选择“自定义镜像部署”
- 选择 `ScalableJobService`
- 挂载 OSS 数据集后启动服务

## 4. 调用服务

说明：

- 当前 v1 版本已经适合在 EAS 上用“自定义镜像 + OSS 挂载”的方式部署。
- 任务执行器会在服务容器内以子进程调用现有 runner。
- `eas_service.json` 采用 `ScalableJobService` 形态，便于后续接入真正的 EAS 弹性作业模式；但当前代码还没有直接使用阿里云专有任务队列 SDK。

健康检查：

```bash
curl http://<eas-endpoint>/healthz
```

提交任务：

```bash
curl -X POST http://<eas-endpoint>/jobs/run \
  -H "Content-Type: application/json" \
  -d '{"job_type":"llamascope_sanity","output_subdir":"run_20260422","args":{"max_docs":8,"batch_size":4,"max_seq_len":128,"checkpoint_topk_semantics":"hard"}}'
```

查询状态：

```bash
curl http://<eas-endpoint>/jobs/<job_id>
```
