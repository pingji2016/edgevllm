"""评测 GRPO 结果：区分"学会算术"和"学到答案分布的众数"

三套题：
  nat     200 道均匀随机（自然分布，众数 9 占 10%）
  bal     190 道均衡（每个和 0..18 各 10 道 → 只会猜众数只有 5.3%）
  nat9    自然分布里排除 sum==9 的题（剔除"猜 9"的收益）

同时报告预测分布（不同取值个数 / 熵），看是否发生了模式塌缩。
"""
import glob
import json
import math
import os
import random
import re
import sys

import torch

ROOT = os.path.expanduser("~/apps/minimind")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SYS = ("你是一个计算器。只输出结果，格式：<answer>数字</answer>\n\n"
       "示例：\n3+4=<answer>7</answer>\n8+5=<answer>13</answer>\n6+2=<answer>8</answer>")
NUM = re.compile(r"-?\d+")
tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "model"))


def build(a, b):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{a}+{b}=<|im_end|>\n"
            f"<|im_start|>assistant\n<answer>")


def make_sets(seed=12345):
    rng = random.Random(seed)
    nat = [(rng.randint(0, 9), rng.randint(0, 9)) for _ in range(200)]
    bal = []
    for s in range(19):
        lo, hi = max(0, s - 9), min(9, s)
        for _ in range(10):
            a = rng.randint(lo, hi)
            bal.append((a, s - a))
    return nat, bal


def load(path):
    cfg = MiniMindConfig(hidden_size=512, num_hidden_layers=8, use_moe=False, vocab_size=6400)
    m = MiniMindForCausalLM(cfg)
    m.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    return m.cuda().eval()


@torch.no_grad()
def run(model, pairs):
    preds = []
    for a, b in pairs:
        ids = tok(build(a, b), return_tensors="pt").input_ids.cuda()
        g = model.generate(ids, max_new_tokens=8, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        t = tok.decode(g[0, ids.size(1):], skip_special_tokens=True)
        m = NUM.search(t)
        preds.append(int(m.group()) if m else None)
    return preds


def metrics(pairs, preds):
    ok = sum(p == a + b for (a, b), p in zip(pairs, preds))
    n = len(pairs)
    hist = {}
    for p in preds:
        hist[p] = hist.get(p, 0) + 1
    ent = -sum((c / n) * math.log2(c / n) for c in hist.values())
    return dict(acc=ok / n, n=n, distinct=len(hist),
                entropy=round(ent, 2), hist=dict(sorted(hist.items(), key=lambda x: -x[1])[:6]))


if __name__ == "__main__":
    nat, bal = make_sets()
    nat9 = [(a, b) for a, b in nat if a + b != 9]
    base = sum(1 for a, b in nat if a + b == 9) / len(nat)
    print(f"题集：nat={len(nat)} bal={len(bal)} nat9={len(nat9)} | "
          f"「总答 9」得分 nat={base*100:.1f}% bal=5.3%", flush=True)

    rows = {}
    for name, path in (("DPO 基线", "out_dpo/dpo_512.pth"),
                       ("GRPO 实验组", "out_grpo/grpo_main.pth"),
                       ("GRPO 对照组", "out_grpo/grpo_control.pth"),
                       ("GRPO 均衡组", "out_grpo/grpo_balanced.pth")):
        if not os.path.exists(os.path.join(ROOT, path)):
            continue
        m = load(os.path.join(ROOT, path))
        pn = run(m, nat)
        pb = run(m, bal)
        p9 = run(m, nat9)
        r = dict(nat=metrics(nat, pn), bal=metrics(bal, pb), nat9=metrics(nat9, p9))
        rows[name] = r
        print(f"\n=== {name} ===")
        print(f"  nat   {r['nat']['acc']*100:5.1f}%   bal {r['bal']['acc']*100:5.1f}%   "
              f"nat9 {r['nat9']['acc']*100:5.1f}%")
        print(f"  预测分布 不同取值 {r['nat']['distinct']} 个 | 熵 {r['nat']['entropy']} bit | "
              f"最常见 {r['nat']['hist']}")
        del m
        torch.cuda.empty_cache()

    json.dump(dict(sets=dict(n=len(nat), n_bal=len(bal), n_nat9=len(nat9)), rows=rows),
              open(os.path.join(ROOT, "out_grpo", "eval_final.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\n已存 out_grpo/eval_final.json")
