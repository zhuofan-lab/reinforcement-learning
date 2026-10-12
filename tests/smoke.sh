#!/usr/bin/env bash
# 冒烟测试：造一个随机初始化的迷你模型，把三个训练脚本和推理脚本各跑几步。不联网，CPU 上约 1 分钟。
# 只检查流程能跑通（迷你模型什么都不会，reward 一直是 0 属于正常）。
#
#   bash tests/smoke.sh                                                            # 单进程（有 GPU 就用第一张）
#   LAUNCH="python -m torch.distributed.run --standalone --nproc_per_node 8 --log-dir outputs/torchrun" bash tests/smoke.sh   # 正式训练前先验一遍 8 卡流程
set -euo pipefail
cd "$(dirname "$0")/.."
LAUNCH=${LAUNCH:-python}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

python tests/test_math.py
python tests/make_tiny_model.py "$TMP/tiny"
for algo in dpo ppo grpo; do
  $LAUNCH train_$algo.py --model "$TMP/tiny" --out "$TMP/$algo" --steps 3 --batch_size 4 --log_every 1
  $LAUNCH infer.py --model "$TMP/tiny" --adapter "$TMP/$algo" --n 16
done
python infer.py --model "$TMP/tiny" --adapter "$TMP/grpo" --prompt "计算：37 * 6"
echo "smoke test passed"
