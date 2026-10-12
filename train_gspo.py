"""GSPO（Group Sequence Policy Optimization）最小实现，出自 Qwen 团队的 "Group Sequence Policy Optimization"。

和 train_grpo.py 的区别：
1. 重要性比按整条回答算：s_i = (π_θ(y_i) / π_old(y_i))^(1/|y_i|)，即回答内逐 token log 比值的平均再取 exp。
   GRPO 每个 token 一个比值，单个 token 的噪声会被放大；GSPO 整条回答共用一个比值。
2. 裁剪也在序列级做，而且范围小得多（论文用 3e-4 / 4e-4），因为长度平均后的 s_i 本来就很接近 1。
3. 每批采样数据切成几个小批量依次更新（论文是 4 个）。第一个小批量完全 on-policy，s_i = 1、不会被裁剪；
   从第二个起策略已经变了，裁剪才起作用。
4. 不加 KL。
优势和 GRPO 一样：组内标准化。

单卡：python train_gspo.py
8 卡：python -m torch.distributed.run --standalone --nproc_per_node 8 --log-dir outputs/torchrun train_gspo.py
"""

import argparse
import random

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from common import MODEL, build_prompt, generate, load, make_problem, parse_answer, reward_fn, token_logprobs


def gspo_loss(logp, old_logp, adv, mask, clip_low, clip_high):
    """logp / old_logp / mask 是 [N, T]；adv 是 [N, 1]。返回 loss 和被裁剪的回答比例（只用来看日志）。"""
    seq_log_ratio = ((logp - old_logp) * mask).sum(1) / mask.sum(1)  # 回答内 log 比值的平均
    s = seq_log_ratio.exp()  # 序列级重要性比
    a = adv.squeeze(1)
    loss = -torch.min(s * a, s.clamp(1 - clip_low, 1 + clip_high) * a).mean()  # 对回答求平均
    clipped = ((s > 1 + clip_high) & (a > 0)) | ((s < 1 - clip_low) & (a < 0))  # 这些回答的梯度被裁掉了
    return loss, clipped.float().mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL, help="模型目录，默认见 common.py 的 MODEL")
    p.add_argument("--out", default="outputs/gspo")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8, help="每张卡每步的题目数")
    p.add_argument("--group_size", type=int, default=8, help="每道题采样的回答数 G（至少 2）")
    p.add_argument("--minibatch_size", type=int, default=16, help="每次更新用几条回答；默认每批数据更新 4 次")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--clip_low", type=float, default=3e-4, help="s_i 的下界是 1 - clip_low")
    p.add_argument("--clip_high", type=float, default=4e-4, help="s_i 的上界是 1 + clip_high")
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

        # 3) 先定好小批量的切法，再按同样的切法算 π_old 的 log 概率。GPU 上 batch 形状不同，算出的 log 概率会有
        #    细小的数值差异，而 GSPO 的裁剪范围只有 1e-4 量级；形状一致，第一个小批量的 s_i 才严格等于 1
        chunks = torch.randperm(len(ids), device=ids.device).split(args.minibatch_size)
        old_logp = torch.zeros_like(cmask)
        with torch.no_grad():
            for idx in chunks:
                old_logp[idx] = token_logprobs(lm(input_ids=ids[idx], attention_mask=mask[idx]).logits, ids[idx], start)

        # 4) 逐个小批量更新
        clip_frac = []
        for idx in chunks:
            logp = token_logprobs(model(input_ids=ids[idx], attention_mask=mask[idx]).logits, ids[idx], start)
            loss, clipped = gspo_loss(logp, old_logp[idx], adv[idx], cmask[idx], args.clip_low, args.clip_high)
            opt.zero_grad()
            acc.backward(loss)
            acc.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            clip_frac.append(clipped)

        if step % args.log_every == 0 or step == 1:
            correct = torch.tensor([parse_answer(t) == a for t, a in zip(texts, answers)], device=acc.device)
            stats = acc.gather(torch.stack([rewards, correct.float(), cmask.sum(1)], 1)).mean(0)  # 汇总所有卡
            clip = acc.gather(torch.stack(clip_frac).mean()).mean()
            acc.print(
                f"step {step:4d} | reward {stats[0]:.3f} | acc {stats[1]:.3f} | len {stats[2]:.1f} | clip {clip:.3f}",
                flush=True,
            )

    acc.wait_for_everyone()
    if acc.is_main_process:
        lm.save_pretrained(args.out)  # 只存 LoRA 适配器，几 MB
        print(f"saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
