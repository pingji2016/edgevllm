"""探测：Qwen3-1.7B 在「强制 <answer> 前缀」下会不会直接给数字"""
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
SYS = ("你是一个计算器。只输出结果，格式：<answer>数字</answer>\n\n"
       "示例：\n3+4=<answer>7</answer>\n8+5=<answer>13</answer>\n6+2=<answer>8</answer>")
NUM = re.compile(r"-?\d+")


def build(a, b):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{a}+{b}=<|im_end|>\n"
            f"<|im_start|>assistant\n<answer>")


tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16,
                                             device_map={"": 0}).eval()
print(f"  参数 {sum(p.numel() for p in model.parameters())/1e9:.2f} B | "
      f"词表 {len(tok)}", flush=True)

right = 0
for i, (a, b) in enumerate([(7, 8), (23, 45), (3, 4), (9, 9), (0, 5), (6, 2),
                            (13, 27), (48, 36), (5, 5), (2, 9)]):
    ids = tok(build(a, b), return_tensors="pt").input_ids.cuda()
    with torch.inference_mode():
        g = model.generate(ids, max_new_tokens=12, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    t = tok.decode(g[0, ids.size(1):], skip_special_tokens=True)
    m = NUM.search(t)
    ok = bool(m and int(m.group()) == a + b)
    right += ok
    print(f"  {a}+{b}={a+b:<3d} {'✅' if ok else '❌'}  输出 {t[:40]!r}")
print(f"\n  10 题答对 {right}/10")
