#!/usr/bin/env bash
# 8 卡上依次训练 DPO / PPO / GRPO，再评测基座模型和三个适配器。
#
#   bash run_8gpu.sh
#   MODEL=/path/to/local/model NGPU=4 bash run_8gpu.sh
set -euo pipefail
cd "$(dirname "$0")"
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
LAUNCH="accelerate launch --multi_gpu --num_processes ${NGPU:-8}"

for algo in dpo ppo grpo; do
  $LAUNCH train_$algo.py --model "$MODEL" --out outputs/$algo
done

$LAUNCH infer.py --model "$MODEL"  # 基座模型，做对照
for algo in dpo ppo grpo; do
  $LAUNCH infer.py --model "$MODEL" --adapter outputs/$algo
done
