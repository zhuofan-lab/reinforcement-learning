"""造一个随机初始化的迷你 Qwen2（约 10 万参数）和配套分词器，不用联网下载。

只用来在 CPU 上几十秒跑通全流程（见 tests/smoke.sh），模型本身什么都不会。
用法：python tests/make_tiny_model.py /tmp/tiny-qwen
"""

import random
import sys

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, ".")
from common import SYSTEM, make_pair, make_problem  # noqa: E402

CHATML = (
    "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def main(out):
    rng = random.Random(0)
    corpus = [SYSTEM, "system user assistant"]
    for _ in range(2000):
        corpus += list(make_pair(rng)) + [make_problem(rng)[0]]

    # 字节级 BPE，数字逐位切开（和 Qwen 一样）
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Digits(individual_digits=True), pre_tokenizers.ByteLevel(add_prefix_space=False)]
    )
    tk.decoder = decoders.ByteLevel()
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    trainer = trainers.BpeTrainer(
        vocab_size=320, special_tokens=special, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tk.train_from_iterator(corpus, trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.chat_template = CHATML
    tok.save_pretrained(out)

    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    model = Qwen2ForCausalLM(cfg)
    # 故意带上和真 Qwen 一样的默认生成参数，确认训练脚本里把它们覆盖掉了
    model.generation_config.update(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.1)
    model.save_pretrained(out)
    print(f"saved to {out}: vocab={len(tok)}, params={sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main(sys.argv[1])
