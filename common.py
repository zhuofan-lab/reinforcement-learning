"""训练和推理脚本共用的部分：玩具任务、奖励函数、模型加载、采样、对数概率。

任务是整数算术题，三种算法共用同一个打分函数。
想换成自己的任务，改 make_problem（出题）、reward_fn（打分）、make_pair（DPO 的偏好对）即可。
"""

import operator
import re
from collections import namedtuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "你是一个计算器。只输出最终结果，不要输出任何别的内容。输出格式示例：<answer>42</answer>"
OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul}
ANSWER = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")
# generate() 的返回值：ids / mask 是 prompt+回答 [N, P+T]；comp_mask 是回答部分的有效位 [N, T]
# （到第一个 EOS 为止，含 EOS）；texts 是解码后的回答；start 是 prompt 长度 P
Rollout = namedtuple("Rollout", "ids mask comp_mask texts start")


# ---------------------------------------------------------------- 任务 ----
def make_problem(rng):
    """随机出一道题，返回 (题目, 答案)。"""
    op = rng.choice("+-*")
    if op == "*":
        a, b = rng.randint(2, 99), rng.randint(2, 9)
    else:
        a, b = rng.randint(10, 999), rng.randint(10, 999)
    return f"计算：{a} {op} {b}", OPS[op](a, b)


def parse_answer(text):
    """取出 <answer> 里的整数，没有就返回 None。"""
    m = ANSWER.search(text)
    return int(m.group(1)) if m else None


def reward_fn(text, answer):
    """打分（0~1）：有 <answer> 标签得 0.2，没有多余文字再加 0.2，答对再加 0.6。"""
    m = ANSWER.search(text)
    if m is None:
        return 0.0
    return 0.2 + 0.2 * (m.group(0) == text.strip()) + 0.6 * (int(m.group(1)) == answer)


def make_pair(rng):
    """合成一条 DPO 偏好数据 (题目, chosen, rejected)：rejected 要么答错，要么格式不对。"""
    question, answer = make_problem(rng)
    chosen = f"<answer>{answer}</answer>"
    if rng.random() < 0.5:
        rejected = f"<answer>{answer + rng.choice([-10, -1, 1, 10])}</answer>"
    else:
        rejected = f"结果是 {answer}。"
    return question, chosen, rejected


def build_prompt(tok, question, system=SYSTEM):
    """套聊天模板，得到喂给模型的完整 prompt 字符串。"""
    msgs = [{"role": "system", "content": system}] if system else []
    msgs.append({"role": "user", "content": question})
    # enable_thinking=False：Qwen3 这类模板默认先输出 <think>，这里关掉；其他模型会忽略这个参数
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


# ---------------------------------------------------------------- 模型 ----
def load(model_name, lora_r=0, adapter=None):
    """加载分词器和模型。lora_r>0：挂一个新 LoRA 用于训练；adapter：加载训练好的 LoRA 用于推理。"""
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"  # 批量生成要左填充
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # 基座权重冻结，GPU 上用 bf16 省一半显存；LoRA 权重由 peft 保持 fp32
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    elif lora_r:
        cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, target_modules="all-linear", task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
    return tok, model


@torch.no_grad()
def generate(lm, tok, prompts, max_new_tokens, n=1, temperature=1.0):
    """每个 prompt 生成 n 条回答（temperature=0 为贪心），同一个 prompt 的 n 条在结果里是连续的。"""
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(lm.device)
    # 显式关掉 top-k / top-p / 重复惩罚：很多模型的 generation_config 默认开着它们，
    # 开着的话采样分布就不是 softmax(logits)，和训练时算的 log π 对不上。
    if temperature > 0:
        how = dict(do_sample=True, temperature=temperature, top_k=0, top_p=1.0)
    else:
        how = dict(do_sample=False, temperature=None, top_k=None, top_p=None)
    ids = lm.generate(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        **how,
        repetition_penalty=1.0,
        max_new_tokens=max_new_tokens,
        num_return_sequences=n,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    start = enc.input_ids.size(1)
    comp = ids[:, start:]
    is_eos = comp == tok.eos_token_id
    comp_mask = (is_eos.cumsum(1) - is_eos.long() == 0).float()
    mask = torch.cat([enc.attention_mask.repeat_interleave(n, 0), comp_mask.long()], 1)
    return Rollout(ids, mask, comp_mask, tok.batch_decode(comp, skip_special_tokens=True), start)


def token_logprobs(logits, ids, start):
    """回答部分每个 token 的 log π，形状 [N, T]。位置 t 的 logits 预测的是第 t+1 个 token。"""
    logp = logits[:, start - 1 : -1].float().log_softmax(-1)
    return logp.gather(-1, ids[:, start:, None]).squeeze(-1)
