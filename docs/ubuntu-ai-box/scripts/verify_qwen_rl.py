"""配对验证：同一组全新 200 题，对比「基座」vs「基座+LoRA」"""
import random
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
LORA = "/home/pingji/apps/minimind/out_grpo_qwen"
SYS = ("你是一个计算器。只输出结果，格式：<answer>数字</answer>\n\n"
       "示例：\n3+4=<answer>7</answer>\n8+5=<answer>13</answer>\n6+2=<answer>8</answer>")
NUM = re.compile(r"-?\d+")


def build(a, b):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{a}*{b}=<|im_end|>\n<|im_start|>assistant\n<answer>")


@torch.no_grad()
def run(model, tok, pairs):
    right = near = 0
    errs = []
    for a, b in pairs:
        ids = tok(build(a, b), return_tensors="pt").input_ids.cuda()
        g = model.generate(ids, max_new_tokens=12, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        t = tok.decode(g[0, ids.size(1):], skip_special_tokens=True)
        m = NUM.search(t)
        p = int(m.group()) if m else None
        ok = p == a * b
        right += ok
        near += p is not None and abs(p - a * b) / (a * b) < 1e-2
        if not ok:
            errs.append(abs(p - a * b) / (a * b) if p is not None else 1.0)
    errs.sort()
    return dict(acc=right / len(pairs), near=near / len(pairs),
                med_err=errs[len(errs) // 2] if errs else 0.0)


tok = AutoTokenizer.from_pretrained(MD)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16, device_map={"": 0}).eval()
model = PeftModel.from_pretrained(base, LORA).eval()

rng = random.Random(999)
pairs = [(rng.randint(100, 999), rng.randint(10, 99)) for _ in range(200)]

a = run(model, tok, pairs)
print(f"  LoRA  精确 {a['acc']*100:5.1f}% | <1% {a['near']*100:5.1f}% | "
      f"错题中位相对误差 {a['med_err']*100:.2f}%", flush=True)
with model.disable_adapter():
    b = run(model, tok, pairs)
print(f"  基座  精确 {b['acc']*100:5.1f}% | <1% {b['near']*100:5.1f}% | "
      f"错题中位相对误差 {b['med_err']*100:.2f}%", flush=True)
print(f"\n  配对差：+{(a['acc']-b['acc'])*100:.1f} pp（精确）/ "
      f"+{(a['near']-b['near'])*100:.1f} pp（<1%）")
