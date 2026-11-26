"""测试时扩展（self-consistency）vs 训练时扩展（GRPO）

任务：三位数 × 两位数（与昨天 RL 实验同题）
对每个问题采样 N=32 个答案，评估前缀 N=1/2/4/8/16/32 的多数投票准确率

两个模型：
  基座        Qwen3-1.7B（昨天基线 47%）
  基座 + LoRA 昨天 GRPO 训了 300 步的适配器（60%）
"""
import json
import os
import random
import re
import statistics
import time
from collections import Counter

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
LORA = "/home/pingji/apps/minimind/out_grpo_qwen"
OUT = os.path.expanduser("~/apps/minimind/out_tta.json")
SYS = ("你是一个计算器。只输出结果，格式：<answer>数字</answer>\n\n"
       "示例：\n3+4=<answer>7</answer>\n8+5=<answer>13</answer>\n6+2=<answer>8</answer>")
NUM = re.compile(r"-?\d+")
NS = [1, 2, 4, 8, 16, 32]
MAXN = 32
BATCH = 16


def build(a, b):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{a}*{b}=<|im_end|>\n<|im_start|>assistant\n<answer>")


def make_pairs(n=200, seed=20250101):
    rng = random.Random(seed)
    return [(rng.randint(100, 999), rng.randint(10, 99)) for _ in range(n)]


def vote(preds, n):
    c = Counter(p for p in preds[:n] if p is not None)
    return c.most_common(1)[0][0] if c else None


@torch.no_grad()
def sample_all(model, tok, pairs):
    """返回 [每个问题的 MAXN 个预测] + 总耗时"""
    out = []
    t0 = time.time()
    for i in range(0, len(pairs), BATCH):
        chunk = pairs[i:i + BATCH]
        prompts = [build(a, b) for a, b in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        gen = model.generate(enc.input_ids, attention_mask=enc.attention_mask,
                             max_new_tokens=12, do_sample=True, temperature=0.8,
                             top_p=0.95, num_return_sequences=MAXN,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        plen = enc.input_ids.size(1)
        for j in range(len(chunk)):
            preds = []
            for k in range(MAXN):
                txt = tok.decode(gen[j * MAXN + k][plen:], skip_special_tokens=True)
                m = NUM.search(txt)
                preds.append(int(m.group()) if m else None)
            out.append(preds)
        del gen
    return out, time.time() - t0


def report(name, pairs, samples, elapsed):
    rows = {}
    for n in NS:
        acc = sum(vote(s, n) == a * b for (a, b), s in zip(pairs, samples)) / len(pairs)
        oracle = sum(any(p == a * b for p in s[:n])
                     for (a, b), s in zip(pairs, samples)) / len(pairs)
        rows[n] = dict(acc=round(acc, 3), oracle=round(oracle, 3))
    per_q = elapsed / len(pairs) * 1000
    print(f"\n=== {name} ===")
    print(f"  {'N':>3s} {'多数投票':>9s} {'oracle@N':>10s}")
    for n in NS:
        print(f"  {n:3d} {rows[n]['acc']*100:8.1f}% {rows[n]['oracle']*100:9.1f}%")
    print(f"  每个问题耗时 {per_q:.0f} ms（{MAXN} 采样）| 单答案 {per_q/MAXN:.1f} ms")
    return dict(rows=rows, per_q_ms=round(per_q, 1), per_ans_ms=round(per_q / MAXN, 1))


def main():
    tok = AutoTokenizer.from_pretrained(MD)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pairs = make_pairs()

    base = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16,
                                                device_map={"": 0}).eval()
    s, dt = sample_all(base, tok, pairs)
    r_base = report("基座 Qwen3-1.7B", pairs, s, dt)

    lora = PeftModel.from_pretrained(base, LORA).eval()
    s2, dt2 = sample_all(lora, tok, pairs)
    r_lora = report("基座 + GRPO LoRA", pairs, s2, dt2)

    # 两者投票并集（32 采样合并）——只是看信息量
    json.dump(dict(base=r_base, lora=r_lora, n=len(pairs), seed=20250101),
              open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"\n已存 {OUT}")


if __name__ == "__main__":
    main()
