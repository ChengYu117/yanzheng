# 云服务器 SSH 断线运行说明

如果你的云服务器 SSH 经常断开，最稳的做法是让实验在 `tmux` 后台会话中运行，而不是把任务绑在当前 SSH 终端上。

本仓库已经提供了现成脚本：

- `deploy/gce/start_full_eval_tmux.sh`
- `deploy/gce/start_causal_tmux.sh`
- `deploy/gce/tmux_status.sh`

## 1. 启动 SAE 主流程

```bash
cd ~/nlp-re
bash deploy/gce/start_full_eval_tmux.sh
```

如果要追加参数：

```bash
bash deploy/gce/start_full_eval_tmux.sh --compare-mean
```

默认会话名：

```text
sae_eval
```

## 2. 启动因果流程

```bash
cd ~/nlp-re
bash deploy/gce/start_causal_tmux.sh
```

如果先做轻量版本：

```bash
bash deploy/gce/start_causal_tmux.sh --skip-side-effects
```

默认会话名：

```text
causal_eval
```

## 3. 查看任务是否还在跑

```bash
tmux ls
```

## 4. 重新连接会话

连接 SAE：

```bash
tmux attach -t sae_eval
```

连接 causal：

```bash
tmux attach -t causal_eval
```

## 5. 只看日志，不进会话

查看 SAE 日志：

```bash
tail -f /home/unenergysdg/data/outputs/sae_eval_full/run.log
```

查看 causal 日志：

```bash
tail -f /home/unenergysdg/data/outputs/causal_validation_full/run.log
```

如果你的 `OUTPUT_ROOT` 不是这个路径，把它替换成你 `.env` 中的值。

也可以用仓库自带脚本快速列出日志：

```bash
bash deploy/gce/tmux_status.sh /home/unenergysdg/data/outputs
```

## 6. 常见问题

### 6.1 再次启动时报 session 已存在

说明任务已经在后台跑了，不要重复开。

先看：

```bash
tmux ls
```

然后：

```bash
tmux attach -t sae_eval
```

### 6.2 我断线后任务会不会停

如果任务是在 `tmux` 里启动的，通常不会因为 SSH 断线而停止。

### 6.3 我想换会话名

可以在启动前设置：

```bash
export TMUX_SESSION_NAME=my_sae_job
bash deploy/gce/start_full_eval_tmux.sh
```
