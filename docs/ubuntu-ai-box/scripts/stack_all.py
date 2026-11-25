"""全优化叠加：GPTQ-Int8 × CUDA Graph × KV-FP8 × ngram 投机

矩阵（除 E 外都是 GPTQ-Int8）：
  A  eager               （无 CUDA Graph）
  B  graph               （vLLM 默认 CUDA Graph）
  C  graph + fp8 KV
  D  graph + fp8 KV + ngram 投机
  E  bf16 + graph        （隔离量化收益）
  F  bf16 + graph + fp8 KV + ngram   （bf16 全叠加）

两种负载分别测（ngram 只对"抄写/复用"型负载有效）：
  free  自由生成 256 token（ignore_eos）
  copy  复述提示里的 120 字文本
"""
import json
import os
import statistics
import time

import torch
from vllm import LLM, SamplingParams

GPTQ = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B-GPTQ-Int8/snapshots/master"
BF16 = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
OUT = os.path.expanduser("~/apps/infer-engines/stack_all.json")

FREE = "用五句话解释什么是显存带宽，以及它为什么决定了大模型的解码速度。"
COPY_TEXT = ("边缘计算把推理放在离用户最近的地方。它降低了网络延迟，减少了回传带宽，"
             "也避免了数据离开现场。代价是需要更小的模型、更激烈的量化。"
             "在十六 GB 显存的消费级显卡上，量化几乎是把大模型塞进去的唯一办法。")
COPY = f"请原样复述下面这段话，不要增删任何字：{COPY_TEXT}"
CONC = [f"第 {i} 问：用两句话说明量化对显存和吞吐的影响。" for i in range(32)]

NG = {"method": "ngram", "num_speculative_tokens": 5,
      "prompt_lookup_max": 8, "prompt_lookup_min": 3}


def build(model, **kw):
    t0 = time.time()
    llm = LLM(model=model, dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.78, disable_log_stats=True, **kw)
    return llm, time.time() - t0


def tps(llm, prompts, max_tokens, reps=1):
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
    llm.generate(prompts[:1], sp)          # 预热
    vals = []
    for _ in range(reps):
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        dt = time.time() - t0
        n = sum(len(o.outputs[0].token_ids) for o in outs)
        vals.append(n / dt)
    return round(statistics.median(vals), 1), vals


CONFIGS = [
    ("A_eager", GPTQ, dict(enforce_eager=True)),
    ("B_graph", GPTQ, dict()),
    ("C_graph_fp8kv", GPTQ, dict(kv_cache_dtype="fp8")),
    ("D_graph_fp8kv_ngram", GPTQ, dict(kv_cache_dtype="fp8", speculative_config=NG)),
    ("E_bf16_graph", BF16, dict()),
    ("F_bf16_graph_fp8kv_ngram", BF16, dict(kv_cache_dtype="fp8", speculative_config=NG)),
]

res = {}
for name, model, kw in CONFIGS:
    print(f"\n=== {name} ===", flush=True)
    try:
        llm, load_s = build(model, **kw)
        r = dict(load_s=round(load_s, 1))
        r["single"], _ = tps(llm, [FREE], 256, reps=3)
        r["conc32"], _ = tps(llm, CONC, 256, reps=2)
        r["copy"], _ = tps(llm, [COPY], 160, reps=3)
        res[name] = r
        print(f"  单流 {r['single']} | 32 并发 {r['conc32']} | 抄写 {r['copy']} tok/s", flush=True)
    except Exception as e:
        res[name] = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
        print(f"  ❌ {type(e).__name__}: {str(e)[:200]}", flush=True)
    finally:
        try:
            del llm
        except Exception:
            pass
        torch.cuda.empty_cache()
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)

print("\n=== 汇总 ===")
base = res.get("A_eager", {})
rows = []
for name, _, _ in CONFIGS:
    r = res.get(name, {})
    if "error" in r:
        rows.append((name, "FAIL", "", "", ""))
        continue
    rows.append((name, r["single"], r["conc32"], r["copy"],
                 f"{r['single']/base['single']:.2f}x" if base.get("single") else "-"))
print(f"{'配置':28s} {'单流':>8s} {'32并发':>8s} {'抄写':>8s} {'vs eager':>9s}")
for row in rows:
    print(f"{row[0]:28s} {str(row[1]):>8s} {str(row[2]):>8s} {str(row[3]):>8s} {str(row[4]):>9s}")
