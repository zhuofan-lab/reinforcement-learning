"""PPO（RLHF 里用的那种）最小实现。

采样回答 -> reward_fn 的得分加在最后一个 token 上，每个 token 再减去偏离参考模型的 KL 惩罚
-> 价值头 + GAE 算逐 token 优势 -> 在同一批数据上做几轮带裁剪的策略梯度 + 价值回归。
参考模型就是"关掉 LoRA 的基座"，所以显存里只有一份权重。

单卡：python train_ppo.py
8 卡：accelerate launch --multi_gpu --num_processes 8 train_ppo.py
"""

import argparse
import random

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from common import build_prompt, generate, load, make_problem, parse_answer, reward_fn, token_logprobs


class ActorCritic(torch.nn.Module):
    """策略（带 LoRA 的语言模型）+ 价值头（接在最后一层隐状态上的一个线性层）。"""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.v_head = torch.nn.Linear(lm.config.hidden_size, 1)
        torch.nn.init.zeros_(self.v_head.weight)
        torch.nn.init.zeros_(self.v_head.bias)

    def forward(self, ids, mask, start):
        """返回回答部分每个 token 的 log π，和生成它之前的状态价值 V，形状都是 [N, T]。"""
        out = self.lm(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        hidden = out.hidden_states[-1][:, start - 1 : -1].float()
        return token_logprobs(out.logits, ids, start), self.v_head(hidden).squeeze(-1)


def gae(rewards, values, mask, gamma, lam):
    """广义优势估计，从后往前递推。mask=0 的位置（EOS 之后）不参与，回答结束后价值为 0。"""
    adv = torch.zeros_like(rewards)
    last_adv = next_value = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(rewards.size(1))):
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_adv = (delta + gamma * lam * last_adv) * mask[:, t]
        adv[:, t] = last_adv
        next_value = values[:, t] * mask[:, t]
    return adv, adv + values  # 优势，回报（价值头的回归目标）


def ppo_loss(logp, value, old_logp, adv, returns, mask, clip, vf_coef):
    """所有输入都是 [N, T]。返回总 loss 和价值损失（只用来看日志）。"""
    ratio = ((logp - old_logp) * mask).exp()  # 乘 mask：EOS 之后的位置 ratio=1，不让没意义的数进 exp
    pg_loss = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
    vf_loss = 0.5 * (value - returns) ** 2
    loss = ((pg_loss + vf_coef * vf_loss) * mask).sum() / mask.sum()
    return loss, (vf_loss.detach() * mask).sum() / mask.sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--out", default="outputs/ppo")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32, help="每张卡每步采样的回答数")
    p.add_argument("--minibatch_size", type=int, default=16)
    p.add_argument("--ppo_epochs", type=int, default=4, help="每批采样数据上更新几轮")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf_coef", type=float, default=0.1, help="价值损失的系数")
    p.add_argument("--kl_coef", type=float, default=0.05, help="偏离参考模型的 KL 惩罚系数")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.95)
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
    ac = ActorCritic(lm)
    opt = torch.optim.AdamW([p for p in ac.parameters() if p.requires_grad], lr=args.lr)
    # 多卡时 model 是 ac 的 DDP 包装，只有要回传梯度的前向走它；采样和无梯度前向直接用 ac / lm。
    # 全程不调 .train()：保持 dropout 关闭，old_logp 才和第一次更新时的 logp 一致。
    model, opt = acc.prepare(ac, opt)

    for step in range(1, args.steps + 1):
        # 1) 采样：每道题一条回答，打分
        problems = [make_problem(rng) for _ in range(args.batch_size)]
        prompts = [build_prompt(tok, q) for q, _ in problems]
        ids, mask, cmask, texts, start = generate(lm, tok, prompts, args.max_new_tokens)
        scores = torch.tensor([reward_fn(t, a) for t, (_, a) in zip(texts, problems)], device=acc.device)

        # 2) 逐 token 奖励 -> GAE
        with torch.no_grad():
            old_logp, old_value = ac(ids, mask, start)
            with lm.disable_adapter():
                ref_logp = token_logprobs(lm(input_ids=ids, attention_mask=mask).logits, ids, start)
            kl = (old_logp - ref_logp) * cmask  # KL(π‖π_ref) 的单样本估计
            rewards = -args.kl_coef * kl
            last = cmask.sum(1).long() - 1  # 每条回答最后一个有效 token 的位置，任务得分加在这里
            rewards[torch.arange(len(ids), device=ids.device), last] += scores
            adv, returns = gae(rewards, old_value, cmask, args.gamma, args.lam)
            valid = cmask.bool()
            adv = (adv - adv[valid].mean()) / (adv[valid].std() + 1e-8)

        # 3) 在这批数据上做几轮小批量更新
        for _ in range(args.ppo_epochs):
            for idx in torch.randperm(len(ids), device=ids.device).split(args.minibatch_size):
                logp, value = model(ids[idx], mask[idx], start)
                loss, vf_loss = ppo_loss(
                    logp, value, old_logp[idx], adv[idx], returns[idx], cmask[idx], args.clip, args.vf_coef
                )
                opt.zero_grad()
                acc.backward(loss)
                acc.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        if step % args.log_every == 0 or step == 1:
            correct = torch.tensor([parse_answer(t) == a for t, (_, a) in zip(texts, problems)], device=acc.device)
            # 汇总所有卡；vf_loss 是最后一个小批量上的
            stats = acc.gather(torch.stack([scores, correct.float(), kl.sum(1) / cmask.sum(1)], 1)).mean(0)
            acc.print(
                f"step {step:4d} | reward {stats[0]:.3f} | acc {stats[1]:.3f} | kl {stats[2]:.4f}"
                f" | vf_loss {acc.gather(vf_loss).mean():.4f}",
                flush=True,
            )

    acc.wait_for_everyone()
    if acc.is_main_process:
        lm.save_pretrained(args.out)  # 只存 LoRA 适配器，几 MB；价值头推理用不到，不存
        print(f"saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
