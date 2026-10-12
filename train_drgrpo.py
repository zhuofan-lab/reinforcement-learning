"""Dr. GRPO（GRPO Done Right）最小实现，出自 "Understanding R1-Zero-Like Training: A Critical Perspective"。

和 train_grpo.py 只差两处，都是去掉 GRPO 里的一个归一化：
1. 优势不再除以组内标准差：A_i = r_i - mean(r)。除以 std 会让太容易 / 太难（std 很小）的题权重被放大。
2. 损失不再按每条回答的长度平均，而是除以一个常数（最大生成长度）。按长度平均会让答错的长回答
   每个 token 受的惩罚变小，模型就倾向于把错的回答写得更长。
另外默认不加 KL（β=0），这是 R1-Zero 式训练的常见做法；需要的话用 --beta 打开。

单卡：python train_drgrpo.py
8 卡：python -m torch.distributed.run --standalone --nproc_per_node 8 --log-dir outputs/torchrun train_drgrpo.py
"""

import argparse
import random

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from common import MODEL, build_prompt, generate, load, make_problem, parse_answer, reward_fn, token_logprobs


def drgrpo_loss(logp, old_logp, ref_logp, adv, mask, clip, beta, max_len):
    """logp / old_logp / ref_logp / mask 是 [N, T]；adv 是 [N, 1]。max_len 是最大生成长度（常数分母）。

    返回 loss 和每条回答的平均 KL（只用来看日志）。和 grpo_loss 唯一的区别在最后求平均的方式。
    """
    ratio = ((logp - old_logp) * mask).exp()
    surrogate = torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
    diff = (ref_logp - logp) * mask
    kl = diff.exp() - diff - 1  # KL(π‖π_ref) 的逐 token 估计，恒 >= 0
    per_token = -(surrogate - beta * kl) * mask
    loss = per_token.sum() / (len(per_token) * max_len)  # 不按回答长度平均，统一除以常数
    return loss, kl.detach().sum(1) / mask.sum(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL, help="模型目录，默认见 common.py 的 MODEL")
    p.add_argument("--out", default="outputs/drgrpo")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4, help="每张卡每步的题目数")
    p.add_argument("--group_size", type=int, default=8, help="每道题采样的回答数 G（至少 2）")
    p.add_argument("--epochs", type=int, default=1, help="每批采样数据上更新几次")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=0.0, help="KL(π‖π_ref) 的系数，默认不加")
    p.add_argument("--max_new_tokens", type=int, default=24, help="最大生成长度，也是损失的常数分母")
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

        # 2) 打分，优势 = 奖励 - 组内均值（不除以标准差）
        answers = [a for _, a in problems for _ in range(args.group_size)]
        rewards = torch.tensor([reward_fn(t, a) for t, a in zip(texts, answers)], device=acc.device)
        r = rewards.view(-1, args.group_size)
        adv = (r - r.mean(1, keepdim=True)).view(-1, 1)

        # 3) 采样时的策略 π_old 和参考模型 π_ref 的 log 概率
        with torch.no_grad():
            old_logp = token_logprobs(lm(input_ids=ids, attention_mask=mask).logits, ids, start)
            with lm.disable_adapter():
                ref_logp = token_logprobs(lm(input_ids=ids, attention_mask=mask).logits, ids, start)

        # 4) 更新
        for _ in range(args.epochs):
            logp = token_logprobs(model(input_ids=ids, attention_mask=mask).logits, ids, start)
            loss, kl = drgrpo_loss(logp, old_logp, ref_logp, adv, cmask, args.clip, args.beta, args.max_new_tokens)
            opt.zero_grad()
            acc.backward(loss)
            acc.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if step % args.log_every == 0 or step == 1:
            correct = torch.tensor([parse_answer(t) == a for t, a in zip(texts, answers)], device=acc.device)
            stats = acc.gather(torch.stack([rewards, correct.float(), cmask.sum(1), kl], 1)).mean(0)  # 汇总所有卡
            acc.print(
                f"step {step:4d} | reward {stats[0]:.3f} | acc {stats[1]:.3f} | len {stats[2]:.1f} | kl {stats[3]:.4f}",
                flush=True,
            )

    acc.wait_for_everyone()
    if acc.is_main_process:
        lm.save_pretrained(args.out)  # 只存 LoRA 适配器，几 MB
        print(f"saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
