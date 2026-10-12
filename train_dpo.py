"""DPO（Direct Preference Optimization）最小实现。

离线算法：不采样、不需要奖励函数，只要 (prompt, chosen, rejected) 偏好对。
loss = -log σ( β·[ (log π(chosen) - log π_ref(chosen)) - (log π(rejected) - log π_ref(rejected)) ] )
参考模型就是"关掉 LoRA 的基座"，所以显存里只有一份权重。

默认用 common.make_pair 合成的算术偏好对；--data 换成自己的 jsonl，
每行 {"prompt": ..., "chosen": ..., "rejected": ...}，可选 "system"。

单卡：python train_dpo.py
8 卡：python -m torch.distributed.run --standalone --nproc_per_node 8 --log-dir outputs/torchrun train_dpo.py
"""

import argparse
import json
import random

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed

from common import MODEL, SYSTEM, build_prompt, load, make_pair


def encode(tok, prompts, responses, device):
    """把每条 prompt+回答+EOS 拼成右填充的 batch，返回 ids、attention mask、回答部分的 mask。"""
    rows = []
    for prompt, response in zip(prompts, responses):
        p = tok(prompt, add_special_tokens=False).input_ids
        r = tok(response, add_special_tokens=False).input_ids + [tok.eos_token_id]
        rows.append((p + r, len(p)))
    width = max(len(seq) for seq, _ in rows)
    ids = torch.full((len(rows), width), tok.pad_token_id)
    mask = torch.zeros(len(rows), width, dtype=torch.long)
    resp = torch.zeros(len(rows), width)
    for i, (seq, n_prompt) in enumerate(rows):
        ids[i, : len(seq)] = torch.tensor(seq)
        mask[i, : len(seq)] = 1
        resp[i, n_prompt : len(seq)] = 1
    return ids.to(device), mask.to(device), resp.to(device)


def response_logprob(logits, ids, resp):
    """每条序列里回答部分的 log π 之和。位置 t 的 logits 预测的是第 t+1 个 token。"""
    logp = logits[:, :-1].float().log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return (logp * resp[:, 1:]).sum(1)


def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta):
    """输入是策略 / 参考模型对 chosen / rejected 整条回答的 log 概率，形状 [B]。返回 loss 和 margin。"""
    margin = (pi_c - ref_c) - (pi_r - ref_r)
    return -F.logsigmoid(beta * margin).mean(), margin.detach()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL, help="模型目录，默认见 common.py 的 MODEL")
    p.add_argument("--out", default="outputs/dpo")
    p.add_argument("--data", default=None, help="偏好对 jsonl；不给就用合成的算术偏好对")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16, help="每张卡每步的偏好对数")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    acc = Accelerator(mixed_precision="no")  # 精度在 common.load 里自己管（bf16 基座 + fp32 LoRA）
    set_seed(args.seed + acc.process_index)  # 每张卡抽不同的数据
    rng = random.Random(args.seed + acc.process_index)

    if args.data:
        with open(args.data, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        data = [(r.get("system", ""), r["prompt"], r["chosen"], r["rejected"]) for r in rows]
        next_pair = lambda: rng.choice(data)  # noqa: E731
    else:
        next_pair = lambda: (SYSTEM, *make_pair(rng))  # noqa: E731

    tok, lm = load(args.model, lora_r=args.lora_r)
    opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=args.lr)
    model, opt = acc.prepare(lm, opt)  # 多卡时 model 是 lm 的 DDP 包装；无梯度的参考前向直接用 lm

    for step in range(1, args.steps + 1):
        batch = [next_pair() for _ in range(args.batch_size)]
        prompts = [build_prompt(tok, question, system) for system, question, _, _ in batch]
        # 前一半是 chosen，后一半是 rejected，一次前向算完
        responses = [chosen for _, _, chosen, _ in batch] + [rejected for _, _, _, rejected in batch]
        ids, mask, resp = encode(tok, prompts * 2, responses, acc.device)

        with torch.no_grad(), lm.disable_adapter():
            ref_c, ref_r = response_logprob(lm(input_ids=ids, attention_mask=mask).logits, ids, resp).chunk(2)
        pi_c, pi_r = response_logprob(model(input_ids=ids, attention_mask=mask).logits, ids, resp).chunk(2)

        loss, margin = dpo_loss(pi_c, pi_r, ref_c, ref_r, args.beta)
        opt.zero_grad()
        acc.backward(loss)
        acc.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            # 正常情况是 logp_chosen 上升、logp_rejected 下降。如果 margin 在涨但 logp_chosen 明显下降，
            # 说明模型只是在一味压低 rejected（DPO 的常见毛病），该调小 lr / steps 或调大 beta 了。
            m, c, r = acc.gather(torch.stack([margin, pi_c, pi_r], 1).detach()).unbind(1)  # 汇总所有卡
            acc.print(
                f"step {step:4d} | loss {F.softplus(-args.beta * m).mean():.4f} | margin {m.mean():.2f}"
                f" | logp_chosen {c.mean():.2f} | logp_rejected {r.mean():.2f}",
                flush=True,
            )

    acc.wait_for_everyone()
    if acc.is_main_process:
        lm.save_pretrained(args.out)  # 只存 LoRA 适配器，几 MB
        print(f"saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
