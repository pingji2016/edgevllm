"""vLLM CUDA Graph vs eager：同一模型、同一负载的对照"""
import json
import os
import time

import torch
from vllm import LLM, SamplingParams

MD17 = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
PROMPT = "用三句话解释什么是显存带宽，以及它为什么决定了大模型的解码速度。"
OUT = os.path.expanduser("~/apps/infer-engines/cudagraph.json")


def bench(llm, prompts, reps=3, max_tokens=128):
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
    llm.generate(prompts[:1], sp)  # 预热
    best = 1e9
    for _ in range(reps):
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        best = min(best, time.time() - t0)
    n = sum(len(o.outputs[0].token_ids) for o in outs)
    return n / best, n


def main():
    res = {}
    for tag, eager in (("eager", True), ("cudagraph", False)):
        print(f"=== {tag} ===", flush=True)
        try:
            t0 = time.time()
            llm = LLM(model=MD17, dtype="bfloat16", max_model_len=2048,
                      gpu_memory_utilization=0.45, enforce_eager=eager,
                      disable_log_stats=True)
            load = time.time() - t0
            tps1, _ = bench(llm, [PROMPT])
            tps8, _ = bench(llm, [PROMPT] * 8)
            tps32, _ = bench(llm, [PROMPT] * 32)
            res[tag] = dict(load_s=round(load, 1), single=round(tps1, 1),
                            c8=round(tps8, 1), c32=round(tps32, 1))
            print(f"  加载 {load:.0f}s | 单流 {tps1:.1f} | 8 并发 {tps8:.1f} | 32 并发 {tps32:.1f}",
                  flush=True)
            del llm
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ❌ {type(e).__name__}: {str(e)[:160]}", flush=True)
            res[tag] = dict(error=type(e).__name__, msg=str(e)[:200])
            torch.cuda.empty_cache()

    if "eager" in res and "single" in res["eager"] and "cudagraph" in res and "single" in res["cudagraph"]:
        e, c = res["eager"], res["cudagraph"]
        print("\n=== 提升 ===")
        for k, label in (("single", "单流"), ("c8", "8 并发"), ("c32", "32 并发")):
            print(f"  {label:6s} {e[k]:7.1f} → {c[k]:7.1f}  ({c[k]/e[k]-1:+.1%})")
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
