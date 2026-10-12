"""推理 / 评测。不加 --adapter 就是原始基座模型，用来做对照。

问一句：       python infer.py --adapter outputs/grpo --prompt "计算：37 * 6"
评测（单卡）： python infer.py --adapter outputs/grpo
评测（8 卡）：
    python -m torch.distributed.run --standalone --nproc_per_node 8 --log-dir outputs/torchrun \
        infer.py --adapter outputs/grpo
"""

import argparse
import random

from accelerate import Accelerator
from accelerate.utils import gather_object

from common import MODEL, SYSTEM, build_prompt, generate, load, make_problem, parse_answer, reward_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL, help="模型目录，默认见 common.py 的 MODEL")
    p.add_argument("--adapter", default=None, help="训练脚本的输出目录（LoRA 适配器）")
    p.add_argument("--prompt", default=None, help="给了就只回答这一句；不给就在算术题上评测")
    p.add_argument("--system", default=SYSTEM, help='系统提示词，传 "" 表示不用')
    p.add_argument("--n", type=int, default=1000, help="评测题目数")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.0, help="0 表示贪心解码")
    p.add_argument("--seed", type=int, default=1234, help="评测题目的随机种子")
    args = p.parse_args()

    acc = Accelerator()
    tok, lm = load(args.model, adapter=args.adapter)
    lm.to(acc.device)

    def answer(questions):
        prompts = [build_prompt(tok, q, args.system) for q in questions]
        return generate(lm, tok, prompts, args.max_new_tokens, temperature=args.temperature).texts

    if args.prompt is not None:
        if acc.is_main_process:
            print(answer([args.prompt])[0])
        return

    rng = random.Random(args.seed)  # 每个进程生成同一份题目，再各取自己那一片
    problems = [make_problem(rng) for _ in range(args.n)]
    results = []
    with acc.split_between_processes(problems) as shard:
        for i in range(0, len(shard), args.batch_size):
            batch = shard[i : i + args.batch_size]
            results += [(q, a, t) for (q, a), t in zip(batch, answer([q for q, _ in batch]))]
    results = gather_object(results)  # 汇总所有卡

    if acc.is_main_process:
        for q, a, t in results[:5]:
            print(f"{q}  ->  {t!r}  (正确答案 {a})")
        accuracy = sum(parse_answer(t) == a for _, a, t in results) / len(results)
        reward = sum(reward_fn(t, a) for _, a, t in results) / len(results)
        print(f"{args.adapter or args.model}: accuracy {accuracy:.3f} | reward {reward:.3f} | n {len(results)}")


if __name__ == "__main__":
    main()
