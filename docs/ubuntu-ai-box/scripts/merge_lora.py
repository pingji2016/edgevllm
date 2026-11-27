"""模型合并：GRPO LoRA + RFT LoRA 线性插值

alpha=1.0 纯 GRPO（60%），alpha=0 纯 RFT（~46.5%）
测 0.75 / 0.5 / 0.25 三个中间点，同一 200 题贪心评测
"""
import json
import os
import random
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
A_DIR = "/home/pingji/apps/minimind/out_grpo_qwen"   # GRPO
B_DIR = "/home/pingji/apps/minimind/out_rft_qwen"    # RFT
NUM = re.compile(r"-?\d+")
SYS = ("你是一个计算器。只输出结果，格式：<answer>数字</answer>\n\n"
       "示例：\n3+4=<answer>7</answer>\n8+5=<answer>13</answer>\n6+2=<answer>8</answer>")


def prompt(a, b):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{a}*{b}=<|im_end|>\n<|im_start|>assistant\n<answer>")


@torch.no_grad()
def evaluate(model, tok, pairs):
    model.eval()
    right = near = 0
    for i in range(0, len(pairs), 16):
        chunk = pairs[i:i + 16]
        enc = tok([prompt(a, b) for a, b in chunk], return_tensors="pt",
                  padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=12, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        pl = enc.input_ids.size(1)
        for j, (a, b) in enumerate(chunk):
            t = tok.decode(g[j, pl:], skip_special_tokens=True)
            m = NUM.search(t)
            p = int(m.group()) if m else None
            right += p == a * b
            near += p is not None and abs(p - a * b) / (a * b) < 1e-2
        del g
    return right / len(pairs), near / len(pairs)


def main():
    tok = AutoTokenizer.from_pretrained(MD)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    rng = random.Random(999)
    pairs = [(rng.randint(100, 999), rng.randint(10, 99)) for _ in range(200)]

    base = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16, device_map={"": 0})
    pa = PeftModel.from_pretrained(base, A_DIR)
    sd_a = {k: v.clone() for k, v in pa.state_dict().items() if "lora_" in k}
    pb = PeftModel.from_pretrained(base, B_DIR)
    sd_b = {k: v.clone() for k, v in pb.state_dict().items() if "lora_" in k}
    keys = [k for k in sd_a if k in sd_b]
    print(f"  可合并参数 {len(keys)} 个", flush=True)

    res = {}
    for alpha in (1.0, 0.75, 0.5, 0.25, 0.0):
        m = PeftModel.from_pretrained(base, A_DIR)
        sd = m.state_dict()
        for k in keys:
            sd[k] = (alpha * sd_a[k].float() + (1 - alpha) * sd_b[k].float()).to(sd[k].dtype)
        m.load_state_dict(sd, strict=False)
        acc, near = evaluate(m, tok, pairs)
        res[alpha] = dict(acc=acc, near=near)
        print(f"  alpha={alpha:.2f}: 精确 {acc*100:5.1f}% | <1% {near*100:5.1f}%", flush=True)
        del m
        torch.cuda.empty_cache()

    json.dump({str(k): v for k, v in res.items()},
              open("/home/pingji/apps/minimind/merge_result.json", "w"), indent=2)
    print("已存 merge_result.json")


if __name__ == "__main__":
    main()
