# reinforcement-learning

大模型后训练里三种常见算法的最小实现：**PPO**、**DPO**、**GRPO**。每个算法一个百来行的训练脚本，外加一个通用的推理 / 评测脚本。

三者共用同一个玩具任务（整数算术题，要求输出 `<answer>结果</answer>`）和同一个打分函数，方便放在一起比较。默认模型是本机的 `/home/xzf/data/model/Qwen2.5-0.5B-Instruct` + LoRA，按 8 卡 3090 配置。

> **验证状态**：目前只在 CPU 上用迷你模型验证过，还没有在 GPU 和真实的 Qwen 上跑过。默认超参是按经验给的起点，不是调出来的。详见[验证情况](#验证情况)。

## 文件

| 文件 | 作用 |
|---|---|
| `common.py` | 任务（出题、打分、DPO 偏好对）、模型和 LoRA 加载、采样、逐 token 对数概率 |
| `train_ppo.py` / `train_dpo.py` / `train_grpo.py` | 三个训练脚本，输出 LoRA 适配器 |
| `infer.py` | 加载基座 + 适配器，问一句或批量评测 |
| `run_8gpu.sh` | 8 卡上依次训练三种算法并评测 |
| `tests/` | 损失函数的单元测试，和不联网的冒烟测试 |

## 运行

```bash
cd reinforcement-learning   # 下面的命令都在仓库目录下执行

# 依赖装进仓库自己的虚拟环境，不改动现有环境；--system-site-packages 直接复用已经装好的 torch
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt

L="python -m torch.distributed.run --standalone --nproc_per_node 8 --log-dir outputs/torchrun"   # 8 卡启动命令
LAUNCH="$L" bash tests/smoke.sh   # 建议先跑一遍冒烟测试：随机迷你模型，不联网，约 1 分钟
bash run_8gpu.sh                  # 依次训练 DPO / PPO / GRPO，再评测基座和三个适配器
```

单独跑某一个：

```bash
$L train_grpo.py                                                 # 训练，适配器存到 outputs/grpo
$L infer.py --adapter outputs/grpo                               # 评测；不加 --adapter 就是基座
python infer.py --adapter outputs/grpo --prompt "计算：37 * 6"   # 问一句
python train_grpo.py                                             # 单卡
```

用 `python -m torch.distributed.run` 而不是 `accelerate launch` 启动，是为了保证用的是当前（虚拟）环境里的 Python；`--standalone` 会自动挑一个空闲端口，不会和机器上别的任务冲突；`--log-dir` 让它的日志目录也留在仓库里，而不是在 `/tmp` 下建 `torchelastic_*`。

模型默认读 `common.py` 里的 `MODEL`（`/home/xzf/data/model/Qwen2.5-0.5B-Instruct`），不联网下载。换模型就改这个常量，或者运行时加 `--model 路径`（`run_8gpu.sh` 用 `MODEL=路径 bash run_8gpu.sh`）。每个脚本的全部参数见 `--help`。

训练结果只写到当前目录下的 `outputs/<算法>/`（只有 LoRA 适配器，几 MB；`run_8gpu.sh` 会先进入仓库目录），重跑会覆盖同名目录。此外只有冒烟测试会在 `/tmp` 建临时目录，跑完自动删除。用本地模型时不联网，也不写 Hugging Face 缓存。

## 三种算法

| | PPO | DPO | GRPO |
|---|---|---|---|
| 数据 | 模型自己采样 | 现成的 (chosen, rejected) 偏好对 | 模型自己采样，每道题 G 条 |
| 学习信号 | `reward_fn` 的得分减去 KL 惩罚，用价值头 + GAE 算逐 token 优势 | 相对参考模型抬高 chosen、压低 rejected | `reward_fn` 的得分在同一道题的 G 条回答里做标准化 |
| 额外的网络 | 价值头（一个线性层） | 无 | 无 |
| 每卡每步（默认） | 32 条回答，更新 8 次 | 16 对 | 4 题 × 8 条回答，更新 1 次 |
| 日志里看什么 | `reward`、`acc` 上升 | `logp_chosen` 上升、`logp_rejected` 下降 | `reward`、`acc` 上升 |

三者的参考模型 π_ref 都是"关掉 LoRA 的基座"，所以显存里只有一份权重。

## 多卡

每张卡是一个独立进程：用各自的随机种子出题、采样、打分，只有梯度通过 DDP 求平均。所以 8 卡的效果是每步样本数 ×8，而不是把同样的样本算得更快。`--batch_size` 指的都是单卡。

基座权重在 GPU 上用 bf16，LoRA 权重是 fp32。默认 batch 是照 24GB 显存保守估算的；如果 OOM，调小 `--batch_size`（PPO 还有 `--minibatch_size`）。换更大的模型时也要相应调小。

## 换成自己的任务

- **PPO / GRPO**：改 `common.py` 里的 `make_problem`（出题）和 `reward_fn`（打分），需要的话改 `SYSTEM` 提示词和 `--max_new_tokens`。
- **DPO**：`--data pairs.jsonl`，每行 `{"prompt": ..., "chosen": ..., "rejected": ...}`，可选 `"system"`。
- **换模型**：改 `common.py` 里的 `MODEL`，或者运行时加 `--model`（本地目录或 Hugging Face 上的模型名都行）。

## 注意

- **冷启动**：如果基座模型从来不输出 `<answer>` 标签，`reward` 会一直是 0，PPO 和 GRPO 就没有学习信号。这时需要改提示词、放宽 `reward_fn`，或者换一个更强的基座。DPO 不依赖采样，没有这个问题。
- **DPO 只认偏好对里的对比**：它保证 chosen 相对 rejected 变得更可能，但不保证 chosen 的绝对概率上升。如果 `margin` 在涨而 `logp_chosen` 明显下降，说明模型只是在压低 rejected，应该调小 `--lr` / `--steps` 或调大 `--beta`。
- **采样分布**：训练时固定用温度 1、不加 top-k / top-p / 重复惩罚来采样（`common.generate` 里显式覆盖了模型自带的 `generation_config`），因为策略梯度要求采样分布和训练时算的 log π 是同一个分布。
- 只在训练结束时保存一次适配器。

## 验证情况

已经做过的（都在 CPU 上）：

- 单元测试（`tests/test_math.py`）：GAE 与定义逐项一致；三个损失在起点的梯度、裁剪规则、EOS 之后位置的屏蔽；偏好对与打分函数一致。
- 采样和训练对得上：`generate` 解码时的 log π 与训练前向算出的 log π 一致（误差 < 1e-3），模型自带的 top-k 等采样设置确实被覆盖了。
- 能学到东西：把一个迷你模型预训练成"一位数加法总是多答 1"（贪心准确率 0），三种算法都能把它纠正过来（贪心准确率：GRPO 1.00，PPO 0.99 以上，DPO 0.92）。
- 2 进程 DDP：两个进程抽到的题目不同，训练结束时参数完全一致。
- bf16 基座 + fp32 LoRA 这条路径（在 CPU 上强制走了一遍）。
- 两套依赖：torch 2.5 / transformers 4.56 / peft 0.13 / accelerate 1.0，以及 torch 2.14 / transformers 5.19 / peft 0.21 / accelerate 1.15。

没有做的：

- 没有在 GPU 上跑过，也没有用真实的 Qwen 跑过。所以默认的 `--lr`、`--steps`、batch 大小能不能在 200 步内看到明显提升、显存实际占多少，都需要在 8 卡上实测后再调。
