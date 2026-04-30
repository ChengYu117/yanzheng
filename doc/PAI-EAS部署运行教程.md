# PAI-EAS 部署运行教程

本项目已经补充了阿里云 PAI / ACR / EAS 的部署入口，核心文件位于：

- `deploy/pai/Dockerfile`
- `deploy/pai/eas_service.json`
- `deploy/pai/build_and_push.sh`
- `deploy/pai/build_and_push.ps1`
- `deploy/pai/README_pai.md`

推荐部署路径：

1. 本地构建 Docker 镜像
2. 推送到阿里云 ACR
3. 在 PAI-EAS 中使用自定义镜像部署
4. 通过 OSS 挂载：
   - `/mnt/pai/models/Llama-3.1-8B`
   - `/mnt/pai/outputs`
   - `/mnt/pai/cache/hf`
   - `/mnt/pai/data`
5. 使用 `POST /jobs/run` 提交长时评测任务

最短操作说明请直接看：

- [deploy/pai/README_pai.md](../deploy/pai/README_pai.md)
