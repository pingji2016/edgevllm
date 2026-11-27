"""三个臭皮匠 vs 一个诸葛亮

任务：三位数 × 两位数，200 题（seed 999，与之前一致）
对照基线：1×1.7B 贪心 47%（基座）/ 60%（GRPO）

五组：
  1. 1x1.7B 贪心
  2. 3x0.6B 投票（T=0.2/0.7/0.9，各答一次，多数票）
  3. 3x0.6B 辩论 1 轮（看到彼此答案后重答，再投票）
  4. 3x0.6B 辩论 2 轮
  5. MoA：3 个 0.6B 出方案 + 1.7B 汇总

同时统计生成 token 数（成本）与逐题对错（可做 McNemar）
"""
import json
import math
import random
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M06 = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master"
M17 = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
NUM = re.compile(r"-?\d+")
SYS = ("你是一个计算器。只输出结果，格式：<answer>数字</answer>\n\n"
       "示例：\n3+4=<answer>7</answer>\n8+5=<answer>13</answer>\n6+2=<answer>8</answer>")
BS = 16
TEMPS = [0.2, 0.7, 0.9]


def p_math(a, b):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{a}*{b}=<|im_end|>\n<|im_start|>assistant\n<answer>")


def p_debate(a, b, others):
    o = "、".join(str(x) for x in others if x is not None) or "无"
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{a}*{b}=\n其他 agent 的答案：{o}。"
            f"请独立重新计算，确认或修正，只输出 <answer>数字</answer>。<|im_end|>\n"
            f"<|im_start|>assistant\n<answer>")


def p_moa(a, b, props):
    o = "\n".join(f"方案{i+1}: {x}" for i, x in enumerate(props))
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{a}*{b}=\n下面是三个 agent 的候选答案：\n{o}\n"
            f"请判断哪个正确，不确定就自己算，只输出 <answer>数字</answer>。<|im_end|>\n"
            f"<|im_start|>assistant\n<answer>")


@torch.no_grad()
def ask(model, tok, prompts, temp, seed, max_new=12):
    outs = []
    for i in range(0, len(prompts), BS):
        chunk = prompts[i:i + BS]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        torch.manual_seed(seed + i)
        g = model.generate(**enc, max_new_tokens=max_new,
                           do_sample=temp > 0, temperature=max(temp, 1e-5),
                           top_p=0.95, pad_token_id=tok.pad_token_id,
                           eos_token_id=tok.eos_token_id)
        for j in range(len(chunk)):
            outs.append(tok.decode(g[j, enc.input_ids.size(1):],
                                   skip_special_tokens=True))
        del g
    return outs


def parse(t):
    m = NUM.search(t)
    return int(m.group()) if m else None


def vote(vals):
    from collections import Counter
    c = Counter(v for v in vals if v is not None)
    return c.most_common(1)[0][0] if c else None


def mcnemar(a, b):
    x = sum(1 for i, j in zip(a, b) if i and not j)
    y = sum(1 for i, j in zip(a, b) if j and not i)
    return x, y, (x - y) / math.sqrt(x + y) if x + y else 0.0


def main():
    tok = AutoTokenizer.from_pretrained(M06)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print("  加载模型…", flush=True)
    m06 = AutoModelForCausalLM.from_pretrained(M06, dtype=torch.bfloat16,
                                               device_map={"": 0}).eval()
    m17 = AutoModelForCausalLM.from_pretrained(M17, dtype=torch.bfloat16,
                                               device_map={"": 0}).eval()
    rng = random.Random(999)
    pairs = [(rng.randint(100, 999), rng.randint(10, 99)) for _ in range(200)]
    gold = [a * b for a, b in pairs]
    prompts = [p_math(a, b) for a, b in pairs]
    res, cost, per = {}, {}, {}
    t0 = time.time()

    # 1. 1.7B 贪心
    out = ask(m17, tok, prompts, 0.0, 0)
    pred = [parse(t) for t in out]
    res["1x1.7B"] = sum(p == g for p, g in zip(pred, gold)) / 200
    per["1x1.7B"] = [p == g for p, g in zip(pred, gold)]
    cost["1x1.7B"] = sum(len(t) for t in out)
    print(f"  1x1.7B 贪心        {res['1x1.7B']*100:5.1f}%", flush=True)

    # 2-4. 三个 0.6B
    rounds = []
    for r in range(3):                      # 0=独立, 1=辩论1, 2=辩论2
        cur = []
        for k, T in enumerate(TEMPS):
            if r == 0:
                ps = prompts
            else:
                prev = [rounds[r - 1][j] for j in range(3) if j != k]
                ps = [p_debate(a, b, [prev[0][i], prev[1][i]])
                      for i, (a, b) in enumerate(pairs)]
            cur.append([parse(t) for t in ask(m06, tok, ps, T, seed=100 * (k + 1) + r)])
        rounds.append(cur)
        votes = [vote([cur[k][i] for k in range(3)]) for i in range(200)]
        name = ["3x0.6B 投票", "3x0.6B 辩论1轮", "3x0.6B 辩论2轮"][r]
        res[name] = sum(p == g for p, g in zip(votes, gold)) / 200
        per[name] = [p == g for p, g in zip(votes, gold)]
        cost[name] = r * 3
        print(f"  {name:16s} {res[name]*100:5.1f}%", flush=True)

    # 5. MoA
    props = [[rounds[0][k][i] for k in range(3)] for i in range(200)]
    moa = ask(m17, tok, [p_moa(a, b, props[i]) for i, (a, b) in
                         enumerate(pairs)], 0.0, 0)
    pred = [parse(t) for t in moa]
    res["MoA(3x0.6B+1.7B)"] = sum(p == g for p, g in zip(pred, gold)) / 200
    per["MoA(3x0.6B+1.7B)"] = [p == g for p, g in zip(pred, gold)]
    print(f"  MoA(3x0.6B+1.7B)   {res['MoA(3x0.6B+1.7B)']*100:5.1f}%", flush=True)

    print(f"\n  用时 {time.time()-t0:.0f}s")
    base = per["1x1.7B"]
    print(f"{'配置':20s} {'准确率':>7s} {'vs 1.7B':>18s}")
    for k, v in res.items():
        x, y, z = mcnemar(per[k], base)
        print(f"  {k:18s} {v*100:6.1f}%   只有它对 {x:2d} / 只有1.7B对 {y:2d}  z={z:+.2f}")

    json.dump(dict(res=res, per=per), open("/home/pingji/apps/minimind/multi_agent.json", "w"),
              ensure_ascii=False, indent=2)
    print("已存 multi_agent.json")


if __name__ == "__main__":
    main()
