"""DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）最小实现，
出自 "DAPO: An Open-Source LLM Reinforcement Learning System at Scale"。

在 train_grpo.py 的基础上改了四处：
1. Clip-Higher：裁剪上下界分开，下界 1 - 0.2，上界放宽到 1 + 0.28，让概率低的好 token 有更多上升空间，防止熵塌缩。
2. 动态采样：一组回答的奖励完全一样（全对或全错）时，优势全是 0，没有学习信号。把这样的组丢掉、继续采样，
   直到凑满 batch（最多采 --max_rounds 轮，还不够就拿丢掉的组补齐）。
3. token 级损失：所有回答的所有 token 一起平均（除以总 token 数），而不是先在每条回答内平均，
   这样长回答里的 token 和短回答里的一样重要。
4. 超长惩罚：回答长度超过 max_new_tokens - overlong_cache 后，奖励线性扣分，到最大长度扣满 1 分；
   被截断（没写完就到了最大长度）的回答不参与算损失。
不加 KL。优势和 GRPO 一样：组内标准化。每批采样数据切成几个小批量依次更新。

单卡：python train_dapo.py
8 卡：python -m torch.distributed.run --standalone --nproc_per_node 8 --log-dir outputs/torchrun train_dapo.py
"""

import argparse
import random

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed

from common import MODEL, Rollout, build_prompt, generate, load, make_problem, parse_answer, reward_fn, token_logprobs


def overlong_penalty(length, max_len, cache):
    """软超长惩罚：长度 <= max_len - cache 不扣分，之后线性扣分，到 max_len 时扣 1。"""
    return ((max_len - cache - length) / cache).clamp(-1, 0)


def dapo_loss(logp, old_logp, adv, mask, clip_low, clip_high, n_tokens):
    """logp / old_logp / mask 是 [N, T]；adv 是 [N, 1]；n_tokens 是做 token 级平均用的 token 数。

    返回 loss 和被裁剪的 token 比例（只用来看日志）。
    """
    ratio = ((logp - old_logp) * mask).exp()
    surrogate = torch.min(ratio * adv, ratio.clamp(1 - clip_low, 1 + clip_high) * adv)
    loss = -(surrogate * mask).sum() / n_tokens  # token 级平均：除以 token 总数
    clipped = (((ratio > 1 + clip_high) & (adv > 0)) | ((ratio < 1 - clip_low) & (adv < 0))) * mask
    return loss, clipped.sum() / mask.sum().clamp(min=1)


def pad_cat(rollouts, pad_id):
    """把几轮采样的结果拼成一个 batch：prompt 部分往左补齐，回答部分往右补齐。"""
    P = max(r.start for r in rollouts)
    T = max(r.comp_mask.size(1) for r in rollouts)
    ids = torch.cat([F.pad(r.ids, (P - r.start, T - r.comp_mask.size(1)), value=pad_id) for r in rollouts])
    mask = torch.cat([F.pad(r.mask, (P - r.start, T - r.comp_mask.size(1))) for r in rollouts])
    cmask = torch.cat([F.pad(r.comp_mask, (0, T - r.comp_mask.size(1))) for r in rollouts])
    return ids, mask, cmask, P


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL, help="模型目录，默认见 common.py 的 MODEL")
    p.add_argument("--out", default="outputs/dapo")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8, help="每张卡每步训练用的题目数（动态采样凑满这么多组）")
    p.add_argument("--group_size", type=int, default=8, help="每道题采样的回答数 G（至少 2）")
    p.add_argument("--minibatch_size", type=int, default=16, help="每次更新用几条回答；默认每批数据更新 4 次")
    p.add_argument("--max_rounds", type=int, default=4, help="动态采样每步最多采几轮")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--clip_low", type=float, default=0.2, help="ratio 的下界是 1 - clip_low")
    p.add_argument("--clip_high", type=float, default=0.28, help="ratio 的上界是 1 + clip_high")
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--overlong_cache", type=int, default=8, help="长度超过 max_new_tokens - 这个值 后开始扣分")
    p.add_argument("--overlong_filter", action=argparse.BooleanOptionalAction, default=True, help="截断的回答不算损失")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    G = args.group_size

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
        # 1) 动态采样：每轮出 batch_size 道题、每题 G 条回答；奖励全一样的组放进 spare
        groups, spare, seen, rounds = [], [], [], 0  # seen：这一步采到的所有回答的 (得分, 是否答对, 长度)
        while len(groups) < args.batch_size and rounds < args.max_rounds:
            rounds += 1
            problems = [make_problem(rng) for _ in range(args.batch_size)]
            ro = generate(lm, tok, [build_prompt(tok, q) for q, _ in problems], args.max_new_tokens, n=G)
            finished = (ro.ids[:, ro.start :] == tok.eos_token_id).any(1)  # 没有 EOS 就是被截断了
            length = ro.comp_mask.sum(1)
            for j, (_, answer) in enumerate(problems):
                rows = slice(j * G, (j + 1) * G)
                score = torch.tensor([reward_fn(t, answer) for t in ro.texts[rows]], device=acc.device)
                reward = score + overlong_penalty(length[rows], args.max_new_tokens, args.overlong_cache)
                seen += [
                    (s, parse_answer(t) == answer, n)
                    for s, t, n in zip(score.tolist(), ro.texts[rows], length[rows].tolist())
                ]
                group = Rollout(ro.ids[rows], ro.mask[rows], ro.comp_mask[rows], ro.texts[rows], ro.start)
                (groups if reward.std() > 0 else spare).append((group, reward, finished[rows]))
        n_valid = min(len(groups), args.batch_size)
        groups = (groups + spare)[: args.batch_size]  # 采满 max_rounds 轮还不够，就用没有学习信号的组补齐

        # 2) 拼成一个 batch，组内标准化得到优势；被截断的回答从损失里去掉
        ids, mask, cmask, start = pad_cat([g for g, _, _ in groups], tok.pad_token_id)
        rewards = torch.stack([r for _, r, _ in groups])
        adv = ((rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, keepdim=True) + 1e-4)).view(-1, 1)
        loss_mask = cmask * torch.cat([f for _, _, f in groups])[:, None] if args.overlong_filter else cmask

        # 3) 先定好小批量的切法，再按同样的切法算 π_old 的 log 概率（GPU 上 batch 形状不同会有细小的数值差异），
        #    这样第一个小批量的 ratio 严格等于 1
        chunks = torch.randperm(len(ids), device=ids.device).split(args.minibatch_size)
        old_logp = torch.zeros_like(cmask)
        with torch.no_grad():
            for idx in chunks:
                old_logp[idx] = token_logprobs(lm(input_ids=ids[idx], attention_mask=mask[idx]).logits, ids[idx], start)

        # 4) 逐个小批量更新
        clip_frac = []
        for idx in chunks:
            m = loss_mask[idx]
            # 所有卡这个小批量的 token 数之和 / 卡数。DDP 会把各卡梯度取平均，合起来正好是全局的 token 级平均
            n_tokens = (acc.reduce(m.sum(), reduction="sum") / acc.num_processes).clamp(min=1)
            logp = token_logprobs(model(input_ids=ids[idx], attention_mask=mask[idx]).logits, ids[idx], start)
            loss, clipped = dapo_loss(logp, old_logp[idx], adv[idx], m, args.clip_low, args.clip_high, n_tokens)
            opt.zero_grad()
            acc.backward(loss)
            acc.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            clip_frac.append(clipped)

        if step % args.log_every == 0 or step == 1:
            # 各卡采到的回答数不一样，所以先在每张卡上求和，再在所有卡之间求和
            local = torch.tensor(seen, dtype=torch.float).sum(0).tolist()
            local += [len(seen), n_valid, rounds, torch.stack(clip_frac).mean().item()]
            totals = acc.reduce(torch.tensor(local, device=acc.device), reduction="sum").tolist()
            r_sum, acc_sum, len_sum, n, valid_sum, rounds_sum, clip_sum = totals
            n_proc = acc.num_processes
            acc.print(
                f"step {step:4d} | reward {r_sum / n:.3f} | acc {acc_sum / n:.3f} | len {len_sum / n:.1f}"
                f" | valid {valid_sum / (n_proc * args.batch_size):.2f} | rounds {rounds_sum / n_proc:.1f}"
                f" | clip {clip_sum / n_proc:.3f}",
                flush=True,
            )

    acc.wait_for_everyone()
    if acc.is_main_process:
        lm.save_pretrained(args.out)  # 只存 LoRA 适配器，几 MB
        print(f"saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
