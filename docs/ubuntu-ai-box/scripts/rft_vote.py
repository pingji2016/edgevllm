"""RFT 模型 + 自一致性投票（N=16），对照基座"""
import json
import os
import random
import re
from collections import Counter

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
RFT = "/home/pingji/apps/minimind/out_rft_qwen"
NUM = re.compile(r"-?\d+")
SYS = ("你是一个计算器。只输出结果，格式：<answer>数字</answer>\n\n"
       "示例：\n3+4=<answer>7</answer>\n8+5=<answer>13</answer>\n6+2=<answer>8</answer>")
N = 16


def prompt(a, b):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{a}*{b}=<|im_end|>\n<|im_start|>assistant\n<answer>")


tok = AutoTokenizer.from_pretrained(MD)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16, device_map={"": 0}).eval()
model = PeftModel.from_pretrained(base, RFT).eval()

rng = random.Random(999)
pairs = [(rng.randint(100, 999), rng.randint(10, 99)) for _ in range(200)]


@torch.no_grad()
def run(m):
    acc_vote = acc1 = oracle = 0
    for i in range(0, len(pairs), 16):
        chunk = pairs[i:i + 16]
        enc = tok([prompt(a, b) for a, b in chunk], return_tensors="pt",
                  padding=True, add_special_tokens=False).to("cuda")
        g = m.generate(**enc, max_new_tokens=12, do_sample=True, temperature=0.8,
                       top_p=0.95, num_return_sequences=N,
                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        pl = enc.input_ids.size(1)
        for j, (a, b) in enumerate(chunk):
            ps = []
            for k in range(N):
                t = tok.decode(g[j * N + k][pl:], skip_special_tokens=True)
                mm = NUM.search(t)
                ps.append(int(mm.group()) if mm else None)
            c = Counter(p for p in ps if p is not None)
            acc_vote += (c.most_common(1)[0][0] == a * b) if c else 0
            acc1 += ps[0] == a * b
            oracle += any(p == a * b for p in ps)
        del g
    n = len(pairs)
    return dict(n1=acc1 / n, vote=acc_vote / n, oracle=oracle / n)


for name, m in (("基座", base), ("RFT", model)):
    r = run(m)
    print(f"  {name:4s} N=1 {r['n1']*100:5.1f}% | N=16 投票 {r['vote']*100:5.1f}% | "
          f"oracle@16 {r['oracle']*100:5.1f}%", flush=True)

json.dump({}, open(os.path.expanduser("~/apps/minimind/out_rft_qwen/vote.json"), "w"))
