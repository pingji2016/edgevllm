"""真实草稿模型投机：Qwen3-0.6B 当 draft，猜 Qwen3-1.7B

对比：
  无投机      1.7B
  草稿投机    1.7B + 0.6B draft（5 个草稿 token）
并发 1 / 8 / 32，贪心，固定输出 128 token

同时采集接受率（accepted / drafted）
"""
import json
import os
import statistics
import time

import torch
from vllm import LLM, SamplingParams

TARGET = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
DRAFT = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master"
OUT = os.path.expanduser("~/apps/infer-engines/draft_spec.json")
PROMPT = "第 {i} 问：用两句话说明量化对显存和吞吐的影响。"
CONCS = [1, 8, 32]


def build(spec):
    t0 = time.time()
    kw = dict(speculative_config={"method": "draft_model", "model": DRAFT,
                                  "num_speculative_tokens": 5}) if spec else {}
    llm = LLM(model=TARGET, dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.78, disable_log_stats=True, **kw)
    print(f"  引擎就绪 {time.time()-t0:.0f}s", flush=True)
    return llm


def spec_counters(llm):
    acc = draft = 0
    try:
        for m in llm.get_metrics():
            if m.name.endswith("spec_decode_num_accepted_tokens_total"):
                acc = m.value
            elif m.name.endswith("spec_decode_num_draft_tokens_total"):
                draft = m.value
    except Exception:
        pass
    return acc, draft


def run(llm, n, max_tokens=128):
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
    prompts = [PROMPT.format(i=i) for i in range(n)]
    llm.generate(prompts[:1], sp)  # 预热
    a0, d0 = spec_counters(llm)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    a1, d1 = spec_counters(llm)
    total = sum(len(o.outputs[0].token_ids) for o in outs)
    return dict(conc=n, tps=round(total / dt, 1), wall_s=round(dt, 3),
                accepted=a1 - a0, drafted=d1 - d0,
                accept_rate=round((a1 - a0) / (d1 - d0), 3) if d1 - d0 else None)


res = {}
for tag, spec in (("no_spec", False), ("draft_spec", True)):
    print(f"\n=== {tag} ===", flush=True)
    llm = build(spec)
    rows = []
    for c in CONCS:
        r = run(llm, c)
        rows.append(r)
        print(f"  {c:2d} 并发: {r['tps']:8.1f} tok/s | 接受率 {r['accept_rate']} "
              f"({r['accepted']}/{r['drafted']})", flush=True)
    res[tag] = rows
    del llm
    torch.cuda.empty_cache()

json.dump(res, open(OUT, "w"), indent=2)
print("\n=== 汇总 ===")
print(f"{'并发':>5s} {'无投机':>10s} {'草稿投机':>10s} {'加速':>8s} {'接受率':>8s}")
a = {r["conc"]: r for r in res["no_spec"]}
b = {r["conc"]: r for r in res["draft_spec"]}
for c in CONCS:
    if c in a and c in b:
        print(f"{c:5d} {a[c]['tps']:10.1f} {b[c]['tps']:10.1f} "
              f"{b[c]['tps']/a[c]['tps']:7.2f}x {str(b[c]['accept_rate']):>8s}")
