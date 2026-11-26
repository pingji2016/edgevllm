"""机制分析：为什么投票没用？
1) 每个问题的答案集中度（top-1 占比）
2) 错题里是否出现过正确答案（oracle）
3) 提高温度到 1.2 能不能增加多样性
"""
import random
import re
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
SYS = ("你是一个计算器。只输出结果，格式：<answer>数字</answer>\n\n"
       "示例：\n3+4=<answer>7</answer>\n8+5=<answer>13</answer>\n6+2=<answer>8</answer>")
NUM = re.compile(r"-?\d+")


def build(a, b):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{a}*{b}=<|im_end|>\n<|im_start|>assistant\n<answer>")


tok = AutoTokenizer.from_pretrained(MD)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16,
                                             device_map={"": 0}).eval()

rng = random.Random(31337)
pairs = [(rng.randint(100, 999), rng.randint(10, 99)) for _ in range(100)]
N = 16


@torch.no_grad()
def sample(temp, top_p):
    out = []
    for i in range(0, len(pairs), 16):
        chunk = pairs[i:i + 16]
        enc = tok([build(a, b) for a, b in chunk], return_tensors="pt",
                  padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(enc.input_ids, attention_mask=enc.attention_mask,
                           max_new_tokens=12, do_sample=True, temperature=temp,
                           top_p=top_p, num_return_sequences=N,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        plen = enc.input_ids.size(1)
        for j in range(len(chunk)):
            ps = []
            for k in range(N):
                t = tok.decode(g[j * N + k][plen:], skip_special_tokens=True)
                m = NUM.search(t)
                ps.append(int(m.group()) if m else None)
            out.append(ps)
        del g
    return out


for temp, tp in ((0.8, 0.95), (1.2, 1.0)):
    s = sample(temp, tp)
    acc = 0
    distinct = []
    modal = []
    oracle = 0
    wrong_never = 0
    for (a, b), ps in zip(pairs, s):
        gold = a * b
        c = Counter(p for p in ps if p is not None)
        top = c.most_common(1)[0][0] if c else None
        acc += top == gold
        distinct.append(len(c))
        modal.append(max(c.values()) / N if c else 0)
        has = any(p == gold for p in ps)
        oracle += has
        if not has:
            wrong_never += 1
    print(f"\n=== T={temp} top_p={tp}（{len(pairs)} 题 × {N} 采样）===")
    print(f"  多数投票准确率 {acc/len(pairs)*100:.1f}% | oracle@{N} {oracle/len(pairs)*100:.1f}%")
    print(f"  每题不同答案数 均值 {sum(distinct)/len(distinct):.2f} | "
          f"top-1 占比 均值 {sum(modal)/len(modal)*100:.0f}%")
    print(f"  {N} 次采样里一次都没答对的题: {wrong_never}/{len(pairs)} "
          f"({wrong_never/len(pairs)*100:.0f}%)")
