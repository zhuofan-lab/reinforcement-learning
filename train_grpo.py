"""GRPO（Group Relative Policy Optimization）最小实现。

每道题采样 G 条回答 -> reward_fn 打分 -> 组内标准化得到优势 -> 带裁剪的策略梯度 + 对参考模型的 KL 惩罚。
不需要价值网络。参考模型就是"关掉 LoRA 的基座"，所以显存里只有一份权重。

单卡：python train_grpo.py
8 卡：python -m torch.distributed.run --standalone --nproc_per_node 8 --log-dir outputs/torchrun train_grpo.py
"""

import argparse
import random

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from common import MODEL, build_prompt, generate, load, make_problem, parse_answer, reward_fn, token_logprobs


def grpo_loss(logp, old_logp, ref_logp, adv, mask, clip, beta):
    """logp / old_logp / ref_logp / mask 是 [N, T]；adv 是 [N, 1]，整条回答共用一个优势。

    返回 loss 和每条回答的平均 KL（只用来看日志）。
    差值先乘 mask：EOS 之后的位置上数值没有意义，这样那里 ratio=1、kl=0，不会有怪数进 exp。
    """
    ratio = ((logp - old_logp) * mask).exp()
    surrogate = torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
    diff = (ref_logp - logp) * mask
    kl = diff.exp() - diff - 1  # KL(π‖π_ref) 的逐 token 估计，恒 >= 0
    per_token = -(surrogate - beta * kl) * mask
    loss = (per_token.sum(1) / mask.sum(1)).mean()  # 先按回答长度平均，再对 batch 平均
    return loss, kl.detach().sum(1) / mask.sum(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL, help="模型目录，默认见 common.py 的 MODEL")
    p.add_argument("--out", default="outputs/grpo")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4, help="每张卡每步的题目数")
    p.add_argument("--group_size", type=int, default=8, help="每道题采样的回答数 G（至少 2）")
    p.add_argument("--epochs", type=int, default=1, help="每批采样数据上更新几次")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=0.04, help="KL(π‖π_ref) 的系数")
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    acc = Accelerator(mixed_precision="no")  # 精度在 common.load 里自己管（bf16 基座 + fp32 LoRA）
    # 每个进程（每张卡）用不同的种子，否则 8 张卡会采到一模一样的题目和回答
    set_seed(args.seed + acc.process_index)
    rng = random.Random(args.seed + acc.process_index)

    tok, lm = load(args.model, lora_r=args.lora_r)
    opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=args.lr)
    # 多卡时 model 是 lm 的 DDP 包装，只有要回传梯度的前向走它；采样和无梯度前向直接用 lm。
    # 全程不调 .train()：保持 dropout 关闭，old_logp 才和第一次更新时的 logp 一致。
    model, opt = acc.prepare(lm, opt)

    for step in range(1, args.steps + 1):
        # 1) 采样：每道题 G 条回答
        problems = [make_problem(rng) for _ in range(args.batch_size)]
        prompts = [build_prompt(tok, q) for q, _ in problems]
        ids, mask, cmask, texts, start = generate(lm, tok, prompts, args.max_new_tokens, n=args.group_size)

        # 2) 打分，组内标准化得到优势
        answers = [a for _, a in problems for _ in range(args.group_size)]
        rewards = torch.tensor([reward_fn(t, a) for t, a in zip(texts, answers)], device=acc.device)
        r = rewards.view(-1, args.group_size)
        adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1, 1)

        # 3) 采样时的策略 π_old 和参考模型 π_ref 的 log 概率
        with torch.no_grad():
            old_logp = token_logprobs(lm(input_ids=ids, attention_mask=mask).logits, ids, start)
            with lm.disable_adapter():
                ref_logp = token_logprobs(lm(input_ids=ids, attention_mask=mask).logits, ids, start)

        # 4) 更新
        for _ in range(args.epochs):
            logp = token_logprobs(model(input_ids=ids, attention_mask=mask).logits, ids, start)
            loss, kl = grpo_loss(logp, old_logp, ref_logp, adv, cmask, args.clip, args.beta)
            opt.zero_grad()
            acc.backward(loss)
            acc.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if step % args.log_every == 0 or step == 1:
            correct = torch.tensor([parse_answer(t) == a for t, a in zip(texts, answers)], device=acc.device)
            stats = acc.gather(torch.stack([rewards, correct.float(), kl], 1)).mean(0)  # 汇总所有卡
            acc.print(f"step {step:4d} | reward {stats[0]:.3f} | acc {stats[1]:.3f} | kl {stats[2]:.4f}", flush=True)

    acc.wait_for_everyone()
    if acc.is_main_process:
        lm.save_pretrained(args.out)  # 只存 LoRA 适配器，几 MB
        print(f"saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
