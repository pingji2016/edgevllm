"""锁定 ~50% 难度：三位×两位 / 两位×两位（带进位）"""
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
rng = random.Random(23)

CASES = {
    "三位×两位": [(rng.randint(100, 999), rng.randint(10, 99)) for _ in range(40)],
    "两位×两位(进位多)": [(rng.randint(50, 99), rng.randint(50, 99)) for _ in range(40)],
    "四位+四位(带进位)": [(rng.randint(5000, 9999), rng.randint(5000, 9999)) for _ in range(40)],
}

for name, items in CASES.items():
    right = 0
    bad = []
    for a, b in items:
        expr = f"{a}*{b}" if "×" in name else f"{a}+{b}"
        gold = a * b if "×" in name else a + b
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
    print(f"  {name:16s} {right:2d}/40 = {right/40*100:5.1f}%   {bad}", flush=True)
