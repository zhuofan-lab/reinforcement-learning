#!/usr/bin/env bash
# 8 卡上依次训练 DPO / PPO / GRPO，再评测基座模型和三个适配器。结果都在仓库的 outputs/ 下。
#
#   bash run_8gpu.sh                                  # 模型用 common.py 里的 MODEL
#   MODEL=/path/to/model NGPU=4 bash run_8gpu.sh
set -euo pipefail
cd "$(dirname "$0")"
# 用 python -m 启动：保证用的是当前（虚拟）环境里的 Python；--standalone 自动挑空闲端口，不和机器上别的任务冲突；
# --log-dir 让 torchrun 的日志目录也留在仓库里（默认会在 /tmp 下建 torchelastic_* 目录）
LAUNCH=${LAUNCH:-"python -m torch.distributed.run --standalone --nproc_per_node ${NGPU:-8} --log-dir outputs/torchrun"}

for algo in dpo ppo grpo; do
  $LAUNCH train_$algo.py ${MODEL:+--model "$MODEL"} --out outputs/$algo
done

$LAUNCH infer.py ${MODEL:+--model "$MODEL"}  # 基座模型，做对照
for algo in dpo ppo grpo; do
  $LAUNCH infer.py ${MODEL:+--model "$MODEL"} --adapter outputs/$algo
done
