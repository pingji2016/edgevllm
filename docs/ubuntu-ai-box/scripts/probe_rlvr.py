"""RLVR 基线探测：30M minimind-DPO 做加法，能答对多少？

同时测两种难度：一位数 / 两位数加法
"""
import os
import re
import sys

import torch

ROOT = os.path.expanduser("~/apps/minimind")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SYS = "你是一个计算器。只输出结果，格式：<answer>数字</answer>"
CKPT = os.path.join(ROOT, "out_dpo", "dpo_512.pth")
TOK = os.path.join(ROOT, "model")

cfg = MiniMindConfig(hidden_size=512, num_hidden_layers=8, use_moe=False, vocab_size=6400)
model = MiniMindForCausalLM(cfg)
sd = torch.load(CKPT, map_location="cpu")
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"  missing={len(missing)} unexpected={len(unexpected)}")
model = model.cuda().eval()
print(f"  参数 {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
tok = AutoTokenizer.from_pretrained(TOK)

ANS = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")


def prompt(a, b):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{a}+{b}=<|im_end|>\n<|im_start|>assistant\n")


def run(pairs):
    out = []
    for a, b in pairs:
        ids = tok(prompt(a, b), return_tensors="pt").input_ids.cuda()
        with torch.inference_mode():
            gen = model.generate(ids, max_new_tokens=24, do_sample=False,
                                 pad_token_id=tok.pad_token_id or 0,
                                 eos_token_id=tok.eos_token_id)
        txt = tok.decode(gen[0, ids.size(1):], skip_special_tokens=True)
        m = ANS.search(txt)
        out.append((a, b, txt.strip()[:60], int(m.group(1)) if m else None))
    return out


for name, lo, hi in (("一位数", 0, 9), ("两位数", 10, 99)):
    pairs = [((i * 7) % (hi - lo + 1) + lo, (i * 13 + 5) % (hi - lo + 1) + lo)
             for i in range(40)]
    res = run(pairs)
    right = sum(1 for a, b, _, p in res if p == a + b)
    fmt = sum(1 for *_, t, p in [(x[0], x[1], x[2], x[3]) for x in res] if p is not None)
    print(f"\n=== {name}加法（40 题，贪心）===")
    print(f"  答对 {right}/40 ({right/40*100:.0f}%) | 格式正确 {fmt}/40")
    for a, b, t, p in res[:6]:
        print(f"    {a}+{b}={a+b:<4d} 模型: {t[:44]!r} parsed={p}")
