"""ngram 投机的并发交叉点：同一引擎配置，只开关 ngram，测 1/4/8/16/32/64 并发"""
import json
import os
import time

import torch
from vllm import LLM, SamplingParams

GPTQ = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B-GPTQ-Int8/snapshots/master"
OUT = os.path.expanduser("~/apps/infer-engines/ngram_crossover.json")
PROMPT = "第 {i} 问：用两句话说明量化对显存和吞吐的影响。"
CONCS = [1, 4, 8, 16, 32, 64]
NG = {"method": "ngram", "num_speculative_tokens": 5,
      "prompt_lookup_max": 8, "prompt_lookup_min": 3}


def build(ngram):
    kw = dict(speculative_config=NG) if ngram else {}
    t0 = time.time()
    llm = LLM(model=GPTQ, dtype="bfloat16", max_model_len=1024,
              gpu_memory_utilization=0.78, disable_log_stats=True,
              kv_cache_dtype="fp8", **kw)
    print(f"  引擎就绪 {time.time()-t0:.0f}s", flush=True)
    return llm


def run(llm, n, max_tokens=128):
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
    prompts = [PROMPT.format(i=i) for i in range(n)]
    llm.generate(prompts[:1], sp)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    total = sum(len(o.outputs[0].token_ids) for o in outs)
    return dict(conc=n, total_tok=total, wall_s=round(dt, 3),
                tps=round(total / dt, 1), tp_per_seq=round(total / dt / n, 1))


res = {}
for tag, ng in (("no_ngram", False), ("ngram", True)):
    print(f"\n=== {tag} ===", flush=True)
    llm = build(ng)
    rows = []
    for c in CONCS:
        try:
            r = run(llm, c)
            rows.append(r)
            print(f"  {c:2d} 并发: {r['tps']:7.1f} tok/s 总 | {r['tp_per_seq']:6.1f} tok/s/条", flush=True)
        except Exception as e:
            print(f"  {c} 并发 ❌ {type(e).__name__}: {str(e)[:120]}", flush=True)
            rows.append(dict(conc=c, error=type(e).__name__))
    res[tag] = rows
    del llm
    torch.cuda.empty_cache()

json.dump(res, open(OUT, "w"), indent=2)

print("\n=== 交叉点 ===")
print(f"{'并发':>5s} {'无投机':>10s} {'+ngram':>10s} {'比值':>7s}")
a = {r["conc"]: r.get("tps") for r in res["no_ngram"]}
b = {r["conc"]: r.get("tps") for r in res["ngram"]}
for c in CONCS:
    if a.get(c) and b.get(c):
        print(f"{c:5d} {a[c]:10.1f} {b[c]:10.1f} {b[c]/a[c]:6.2f}x")
