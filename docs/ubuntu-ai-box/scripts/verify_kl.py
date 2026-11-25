"""复核：换 3 个种子 × 190 道均衡题，看 KL 组的 14.2% 是否稳定"""
import os
import re
import sys

import torch

ROOT = os.path.expanduser("~/apps/minimind")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SYS = ("你是一个计算器。只输出结果，格式：<answer>数字</answer>\n\n"
       "示例：\n3+4=<answer>7</answer>\n8+5=<answer>13</answer>\n6+2=<answer>8</answer>")
NUM = re.compile(r"-?\d+")
tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "model"))


def build(a, b):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{a}+{b}=<|im_end|>\n<|im_start|>assistant\n<answer>")


def load(p):
    cfg = MiniMindConfig(hidden_size=512, num_hidden_layers=8, use_moe=False, vocab_size=6400)
    m = MiniMindForCausalLM(cfg)
    m.load_state_dict(torch.load(os.path.join(ROOT, p), map_location="cpu"), strict=False)
    return m.cuda().eval()


import random  # noqa: E402


def balanced(seed):
    rng = random.Random(seed)
    out = []
    for s in range(19):
        lo, hi = max(0, s - 9), min(9, s)
        for _ in range(10):
            a = rng.randint(lo, hi)
            out.append((a, s - a))
    return out


@torch.no_grad()
def acc(model, pairs):
    ok = 0
    for a, b in pairs:
        ids = tok(build(a, b), return_tensors="pt").input_ids.cuda()
        g = model.generate(ids, max_new_tokens=8, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        t = tok.decode(g[0, ids.size(1):], skip_special_tokens=True)
        m = NUM.search(t)
        ok += bool(m and int(m.group()) == a + b)
    return ok / len(pairs)


models = {"DPO 基线": load("out_dpo/dpo_512.pth"),
          "GRPO+KL": load("out_grpo/grpo_kl.pth")}
print(f"{'种子':>6s} " + " ".join(f"{k:>10s}" for k in models))
tot = {k: [] for k in models}
for seed in (12345, 777, 2024):
    pairs = balanced(seed)
    line = f"{seed:6d} "
    for k, m in models.items():
        a = acc(m, pairs)
        tot[k].append(a)
        line += f"{a*100:9.1f}% "
    print(line, flush=True)
print("\n平均: " + " ".join(f"{k} {sum(v)/len(v)*100:.1f}%" for k, v in tot.items()))
print("随机猜测基线 5.3%")
