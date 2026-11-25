"""继续加压：找 Qwen3-1.7B 的 30-70% 难度带"""
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
SYS = ("你是一个计算器。只输出结果，格式：<answer>数字</answer>\n\n"
       "示例：\n3+4=<answer>7</answer>\n8+5=<answer>13</answer>\n6+2=<answer>8</answer>")
NUM = re.compile(r"-?\d+")


def build(expr):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{expr}=<|im_end|>\n"
            f"<|im_start|>assistant\n<answer>")


tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16,
                                             device_map={"": 0}).eval()
rng = random.Random(11)

CASES = {
    "三位数加法": [(rng.randint(100, 999), "+", rng.randint(100, 999)) for _ in range(40)],
    "四位数加法": [(rng.randint(1000, 9999), "+", rng.randint(1000, 9999)) for _ in range(40)],
    "三位×一位": [(rng.randint(100, 999), "*", rng.randint(2, 9)) for _ in range(40)],
    "两位×两位": [(rng.randint(10, 99), "*", rng.randint(10, 99)) for _ in range(40)],
    "(a+b)×c": [(rng.randint(10, 99), "*", rng.randint(2, 9)) for _ in range(40)],
    "四位数×一位": [(rng.randint(1000, 9999), "*", rng.randint(2, 9)) for _ in range(40)],
}

for name, items in CASES.items():
    right = 0
    bad = []
    for a, op, b in items:
        if name == "(a+b)×c":
            base = a
            expr = f"({base}+{rng.randint(10,99)})*{b}"
            gold = (base + int(expr.split("+")[1].split(")")[0])) * b
        else:
            expr = f"{a}{op}{b}"
            gold = a + b if op == "+" else a * b
        ids = tok(build(expr), return_tensors="pt").input_ids.cuda()
        with torch.inference_mode():
            g = model.generate(ids, max_new_tokens=12, do_sample=False,
                               pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        t = tok.decode(g[0, ids.size(1):], skip_special_tokens=True)
        m = NUM.search(t)
        ok = bool(m and int(m.group()) == gold)
        right += ok
        if not ok and len(bad) < 2:
            bad.append(f"{expr}={gold} -> {t[:14]!r}")
    print(f"  {name:12s} {right:2d}/40 = {right/40*100:5.1f}%   {bad}", flush=True)
