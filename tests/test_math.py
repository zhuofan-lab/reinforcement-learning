"""任务、奖励和三个损失函数的单元测试，不需要模型。

运行：python tests/test_math.py（或 pytest tests/）
"""

import math
import random
import sys

import torch

sys.path.insert(0, ".")
from common import make_pair, make_problem, parse_answer, reward_fn  # noqa: E402
from train_dpo import dpo_loss  # noqa: E402
from train_grpo import grpo_loss  # noqa: E402
from train_ppo import gae, ppo_loss  # noqa: E402


def random_mask(n, t):
    """每行前若干位为 1（有效 token），其余为 0（EOS 之后）。"""
    lengths = torch.randint(1, t + 1, (n,))
    lengths[0] = t
    return (torch.arange(t)[None] < lengths[:, None]).float()


def test_task_and_reward():
    rng = random.Random(0)
    for _ in range(2000):
        question, answer = make_problem(rng)
        assert answer == eval(question.removeprefix("计算："))
        question, chosen, rejected = make_pair(rng)
        answer = eval(question.removeprefix("计算："))
        # 偏好对和奖励函数必须说的是同一件事
        assert reward_fn(chosen, answer) == 1.0 and reward_fn(rejected, answer) < 1.0

    assert reward_fn("<answer>42</answer>", 42) == 1.0
    assert reward_fn(" <answer> 42 </answer>\n", 42) == 1.0
    assert math.isclose(reward_fn("答案是 <answer>42</answer>", 42), 0.8)  # 答对但有多余文字
    assert math.isclose(reward_fn("<answer>41</answer>", 42), 0.4)  # 格式对但答错
    assert math.isclose(reward_fn("答案是 <answer>41</answer>", 42), 0.2)
    assert reward_fn("42", 42) == 0.0 and reward_fn("", 42) == 0.0
    assert parse_answer("<answer>-7</answer>") == -7 and parse_answer("-7") is None


def test_gae_matches_definition():
    torch.manual_seed(0)
    n, t, gamma, lam = 5, 7, 0.9, 0.8
    rewards, values, mask = torch.randn(n, t), torch.randn(n, t), random_mask(n, t)
    adv, returns = gae(rewards, values, mask, gamma, lam)
    for i in range(n):
        length = int(mask[i].sum())
        for s in range(length):
            # 定义：A_s = Σ_k (γλ)^(k-s) δ_k，δ_k = r_k + γ V_{k+1} - V_k，回答结束后 V = 0
            want = 0.0
            for k in range(s, length):
                next_value = values[i, k + 1] if k + 1 < length else 0.0
                want += (gamma * lam) ** (k - s) * (rewards[i, k] + gamma * next_value - values[i, k])
            assert torch.isclose(adv[i, s], torch.as_tensor(want), atol=1e-5)
            assert torch.isclose(returns[i, s], want + values[i, s], atol=1e-5)
        assert (adv[i, length:] == 0).all()

    # γ = λ = 1 时，优势 = 之后的奖励总和 - V
    adv, _ = gae(rewards, values, mask, 1.0, 1.0)
    reward_to_go = (rewards * mask).flip(1).cumsum(1).flip(1)
    assert torch.allclose(adv, (reward_to_go - values) * mask, atol=1e-5)


def test_grpo_loss():
    torch.manual_seed(0)
    n, t = 6, 5
    mask, adv, old = random_mask(n, t), torch.randn(n, 1), -torch.rand(n, t)

    # 第一次更新时 π = π_old = π_ref：梯度就是普通的策略梯度 -A_i / (回答长度 · N)，KL 为 0
    logp = old.clone().requires_grad_()
    loss, kl = grpo_loss(logp, old, old, adv, mask, clip=0.2, beta=0.04)
    loss.backward()
    assert torch.allclose(logp.grad, -adv * mask / mask.sum(1, keepdim=True) / n, atol=1e-6)
    assert torch.allclose(kl, torch.zeros(n))

    # 裁剪：ratio 已经朝有利方向走出 [1-ε, 1+ε] 就不再给梯度；朝不利方向走出去则照常给
    one = torch.ones(1, 1)
    for shift, advantage, clipped in [(0.5, 1.0, True), (0.5, -1.0, False), (-0.5, 1.0, False), (-0.5, -1.0, True)]:
        logp = torch.full((1, 1), shift, requires_grad=True)
        loss, _ = grpo_loss(logp, 0 * one, logp.detach(), advantage * one, one, clip=0.2, beta=0.04)
        loss.backward()
        want = 0.0 if clipped else -advantage * math.exp(shift)
        assert math.isclose(logp.grad.item(), want, abs_tol=1e-6), (shift, advantage)

    # KL 项：非负，并且把 logp 往 ref_logp 拉
    for shift in (0.7, -0.7):
        logp = torch.full((1, 1), shift, requires_grad=True)
        loss, kl = grpo_loss(logp, logp.detach(), 0 * one, 0 * one, one, clip=0.2, beta=1.0)
        loss.backward()
        assert kl.item() > 0 and math.isclose(kl.item(), math.exp(-shift) + shift - 1, rel_tol=1e-5)
        assert logp.grad.item() * shift > 0  # 高于参考就往下推，低于参考就往上推

    # EOS 之后的位置无论是什么数都不影响结果
    logp, ref = old + 0.1 * torch.randn(n, t), old + 0.1 * torch.randn(n, t)
    junk = (1 - mask) * 1e4
    a = grpo_loss(logp, old, ref, adv, mask, 0.2, 0.04)
    b = grpo_loss(logp - junk, old + junk, ref + junk, adv, mask, 0.2, 0.04)
    assert torch.isfinite(b[0]) and torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])


def test_ppo_loss():
    torch.manual_seed(0)
    n, t, vf_coef = 6, 5, 0.1
    mask, adv, returns, old = random_mask(n, t), torch.randn(n, t), torch.randn(n, t), -torch.rand(n, t)

    # ratio = 1 时：策略梯度是 -A / 有效 token 数，价值梯度是 vf_coef·(V - R) / 有效 token 数
    logp, value = old.clone().requires_grad_(), torch.randn(n, t, requires_grad=True)
    loss, vf_loss = ppo_loss(logp, value, old, adv, returns, mask, clip=0.2, vf_coef=vf_coef)
    loss.backward()
    assert torch.allclose(logp.grad, -adv * mask / mask.sum(), atol=1e-6)
    assert torch.allclose(value.grad, vf_coef * (value.detach() - returns) * mask / mask.sum(), atol=1e-6)
    assert torch.isclose(vf_loss, (0.5 * (value.detach() - returns) ** 2 * mask).sum() / mask.sum())

    # 裁剪规则和 GRPO 相同
    one = torch.ones(1, 1)
    for shift, advantage, clipped in [(0.5, 1.0, True), (0.5, -1.0, False), (-0.5, 1.0, False), (-0.5, -1.0, True)]:
        logp = torch.full((1, 1), shift, requires_grad=True)
        loss, _ = ppo_loss(logp, 0 * one, 0 * one, advantage * one, 0 * one, one, clip=0.2, vf_coef=vf_coef)
        loss.backward()
        want = 0.0 if clipped else -advantage * math.exp(shift)
        assert math.isclose(logp.grad.item(), want, abs_tol=1e-6), (shift, advantage)

    # EOS 之后的位置无论是什么数都不影响结果
    logp, value = old + 0.1 * torch.randn(n, t), torch.randn(n, t)
    junk = (1 - mask) * 1e4
    a = ppo_loss(logp, value, old, adv, returns, mask, 0.2, vf_coef)
    b = ppo_loss(logp + junk, value + junk, old - junk, adv + junk, returns - junk, mask, 0.2, vf_coef)
    assert torch.isfinite(b[0]) and torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])


def test_dpo_loss():
    zero = torch.zeros(4)
    loss, margin = dpo_loss(zero, zero, zero, zero, beta=0.1)
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6) and (margin == 0).all()  # 起点：π = π_ref

    pi_c = torch.tensor([-1.0, -2.0], requires_grad=True)
    pi_r = torch.tensor([-3.0, -1.0], requires_grad=True)
    ref_c, ref_r = torch.tensor([-2.0, -2.5]), torch.tensor([-2.0, -0.5])
    loss, margin = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.5)
    assert torch.allclose(margin, torch.tensor([2.0, 1.0]))
    assert math.isclose(loss.item(), (math.log1p(math.exp(-1.0)) + math.log1p(math.exp(-0.5))) / 2, rel_tol=1e-6)
    loss.backward()
    assert (pi_c.grad < 0).all() and (pi_r.grad > 0).all()  # 梯度下降会抬高 chosen、压低 rejected


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
