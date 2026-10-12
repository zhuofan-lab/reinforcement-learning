# reinforcement-learning

大模型后训练里几种常见算法的最小实现：**PPO**、**DPO**、**GRPO**，以及 GRPO 的三个改进版 **Dr. GRPO**、**GSPO**、**DAPO**。每个算法一个百来行的训练脚本，外加一个通用的推理 / 评测脚本。

所有算法共用同一个玩具任务（整数算术题，要求输出 `<answer>结果</answer>`）和同一个打分函数，方便放在一起比较。默认模型是本机的 `/home/xzf/data/model/Qwen2.5-0.5B-Instruct` + LoRA，按 8 卡 3090 配置。

> **验证状态**：PPO / DPO / GRPO 已经在 8 卡 3090 上跑过；Dr. GRPO / GSPO / DAPO 目前只在 CPU 上用迷你模型验证过，还没在 GPU 上跑。默认超参是按经验给的起点，不是调出来的。详见[验证情况](#验证情况)。

## 文件

| 文件 | 作用 |
|---|---|
| `common.py` | 任务（出题、打分、DPO 偏好对）、模型和 LoRA 加载、采样、逐 token 对数概率 |
| `train_ppo.py` / `train_dpo.py` / `train_grpo.py` | 三个基础算法的训练脚本，输出 LoRA 适配器 |
| `train_drgrpo.py` / `train_gspo.py` / `train_dapo.py` | GRPO 的三个变体，结构和 `train_grpo.py` 一样，`diff` 一下就能看到改了哪里 |
| `infer.py` | 加载基座 + 适配器，问一句或批量评测 |
| `run_8gpu.sh` | 8 卡上依次训练各个算法并评测 |
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
bash run_8gpu.sh                  # 依次训练全部 6 个算法，再评测基座和各个适配器
ALGOS="drgrpo gspo dapo" bash run_8gpu.sh   # 只跑其中几个
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

## PPO、DPO、GRPO

| | PPO | DPO | GRPO |
|---|---|---|---|
| 数据 | 模型自己采样 | 现成的 (chosen, rejected) 偏好对 | 模型自己采样，每道题 G 条 |
| 学习信号 | `reward_fn` 的得分减去 KL 惩罚，用价值头 + GAE 算逐 token 优势 | 相对参考模型抬高 chosen、压低 rejected | `reward_fn` 的得分在同一道题的 G 条回答里做标准化 |
| 额外的网络 | 价值头（一个线性层） | 无 | 无 |
| 每卡每步（默认） | 32 条回答，更新 8 次 | 16 对 | 4 题 × 8 条回答，更新 1 次 |
| 日志里看什么 | `reward`、`acc` 上升 | `logp_chosen` 上升、`logp_rejected` 下降 | `reward`、`acc` 上升 |

三者的参考模型 π_ref 都是"关掉 LoRA 的基座"，所以显存里只有一份权重。

## GRPO 的三个变体

| | GRPO | Dr. GRPO | GSPO | DAPO |
|---|---|---|---|---|
| 优势 | (r − 组内均值) / 组内标准差 | r − 组内均值，不除标准差 | 同 GRPO | 同 GRPO |
| 重要性比 | 逐 token | 逐 token | 整条回答一个：回答内 log 比值的平均再取 exp | 逐 token |
| 裁剪范围 | [1 − 0.2, 1 + 0.2] | 同 GRPO | [1 − 3e-4, 1 + 4e-4]，按整条回答裁剪 | [1 − 0.2, 1 + 0.28]，上界放宽 |
| 损失怎么平均 | 先在每条回答内平均，再对回答平均 | 除以常数：回答数 × 最大生成长度 | 对回答平均 | 所有卡、所有回答的 token 一起平均 |
| KL | β = 0.04 | 默认不加 | 不加 | 不加 |
| 采样 | 每道题 G 条 | 同 GRPO | 同 GRPO | 动态采样：丢掉奖励全一样的组，继续采到凑满 |
| 奖励 | `reward_fn` | 同 GRPO | 同 GRPO | `reward_fn` 加超长惩罚；被截断的回答不算损失 |
| 每卡每步（默认） | 4 题 × 8 条，更新 1 次 | 同 GRPO | 8 题 × 8 条，分 4 个小批量更新 | 8 题 × 8 条，分 4 个小批量更新 |

- Dr. GRPO 去掉的两个归一化都会带来偏差：除以标准差会放大太容易或太难的题；按回答长度平均会让答错的长回答每个 token 挨的罚更轻，模型倾向于把错的回答写长。日志里多一列 `len`（平均回答长度），就是为了看这个。
- GSPO 和 DAPO 的裁剪只有在一批数据更新多次时才起作用（第一次更新时比值恰好是 1），所以默认切成 4 个小批量。GSPO 的裁剪范围很小，日志里 `clip`（被裁剪的回答比例）比 GRPO 高得多是正常的，论文里也是这样。
- DAPO 日志里的 `valid` 是有学习信号的组的占比，`rounds` 是平均采了几轮。模型越强，整组全对的题越多，每步要多采几轮（最多 `--max_rounds` 轮），生成时间会变长。

## 多卡

每张卡是一个独立进程：用各自的随机种子出题、采样、打分，只有梯度通过 DDP 求平均。所以 8 卡的效果是每步样本数 ×8，而不是把同样的样本算得更快。`--batch_size` 指的都是单卡。

基座权重在 GPU 上用 bf16，LoRA 权重是 fp32。默认 batch 是照 24GB 显存保守估算的；如果 OOM，调小 `--batch_size`（PPO、GSPO、DAPO 还有 `--minibatch_size`）。换更大的模型时也要相应调小。

## 换成自己的任务

- **PPO / GRPO 及其变体**：改 `common.py` 里的 `make_problem`（出题）和 `reward_fn`（打分），需要的话改 `SYSTEM` 提示词和 `--max_new_tokens`（DAPO 的超长惩罚从 `max_new_tokens - overlong_cache` 开始扣分，要一起调）。
- **DPO**：`--data pairs.jsonl`，每行 `{"prompt": ..., "chosen": ..., "rejected": ...}`，可选 `"system"`。
- **换模型**：改 `common.py` 里的 `MODEL`，或者运行时加 `--model`（本地目录或 Hugging Face 上的模型名都行）。

## 注意

- **冷启动**：如果基座模型从来不输出 `<answer>` 标签，`reward` 会一直是 0，PPO 和 GRPO 这一族就没有学习信号。这时需要改提示词、放宽 `reward_fn`，或者换一个更强的基座。DPO 不依赖采样，没有这个问题。
- **DPO 只认偏好对里的对比**：它保证 chosen 相对 rejected 变得更可能，但不保证 chosen 的绝对概率上升。如果 `margin` 在涨而 `logp_chosen` 明显下降，说明模型只是在压低 rejected，应该调小 `--lr` / `--steps` 或调大 `--beta`。
- **采样分布**：训练时固定用温度 1、不加 top-k / top-p / 重复惩罚来采样（`common.generate` 里显式覆盖了模型自带的 `generation_config`），因为策略梯度要求采样分布和训练时算的 log π 是同一个分布。
- 只在训练结束时保存一次适配器。

## 验证情况

已经做过的（都在 CPU 上）：

- 单元测试（`tests/test_math.py`）：GAE 与定义逐项一致；六个损失在起点的梯度、裁剪规则（包括 GSPO 的整条回答裁剪和 DAPO 分开的上下界）、EOS 之后位置的屏蔽；回答等长时 Dr. GRPO 和 GRPO 的损失相等；DAPO 的超长惩罚和多轮采样结果的拼接；偏好对与打分函数一致。
- 采样和训练对得上：`generate` 解码时的 log π 与训练前向算出的 log π 一致（误差 < 1e-3），模型自带的 top-k 等采样设置确实被覆盖了。
- 能学到东西：把一个迷你模型预训练成"一位数加法总是多答 1"（贪心准确率 0），六种算法都能把它纠正过来（贪心准确率：GRPO 和 Dr. GRPO 1.00，PPO 0.99 以上，DAPO 0.98 以上，GSPO 0.96 以上，DPO 0.92）。
- GSPO 和 DAPO 每步第一个小批量的重要性比严格等于 1（之后的小批量才偏离），bf16 下也是。
- 2 进程 DDP：两个进程抽到的题目不同，训练结束时参数完全一致；DAPO 两个进程动态采样的轮数不一样时也是。
- bf16 基座 + fp32 LoRA 这条路径（在 CPU 上强制走了一遍）。
- 两套依赖：torch 2.5 / transformers 4.56 / peft 0.13 / accelerate 1.0，以及 torch 2.14 / transformers 5.19 / peft 0.21 / accelerate 1.15。

没有做的：

- Dr. GRPO、GSPO、DAPO 还没有在 GPU 上、用真实的 Qwen 跑过。默认的 `--lr`、`--steps`、batch 大小和 GSPO 的裁剪范围能不能在 200 步内看到明显提升、显存实际占多少，都需要实测后再调。
