"""诊断：草稿投机的接受率 + 不同草稿长度
"""
import json
import os
import time

from vllm import LLM, SamplingParams

TARGET = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
DRAFT = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master"
PROMPT = "第 {i} 问：用两句话说明量化对显存和吞吐的影响。"


def metrics_dump(llm, label):
    names = []
    try:
        for m in llm.get_metrics():
            if "spec" in m.name.lower():
                names.append((m.name, m.value))
    except Exception as e:
        print(f"  {label}: get_metrics 失败 {type(e).__name__}", flush=True)
    print(f"  {label}: {names[:8]}", flush=True)
    return names


sp = SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True)
prompts = [PROMPT.format(i=i) for i in range(8)]

for k in (3, 5):
    print(f"\n=== 草稿长度 {k} ===", flush=True)
    t0 = time.time()
    llm = LLM(model=TARGET, dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.78, disable_log_stats=False,
              speculative_config={"method": "draft_model", "model": DRAFT,
                                  "num_speculative_tokens": k})
    print(f"  引擎就绪 {time.time()-t0:.0f}s", flush=True)

    before = metrics_dump(llm, "运行前")
    llm.generate(prompts[:1], sp)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    total = sum(len(o.outputs[0].token_ids) for o in outs)
    after = metrics_dump(llm, "运行后")

    d = dict(after)
    b = dict(before)
    acc = next((v - b.get(nm, 0) for nm, v in d.items() if "accepted" in nm), None)
    drf = next((v - b.get(nm, 0) for nm, v in d.items() if "draft_tokens" in nm), None)
    print(f"  k={k}: 8 并发 {total/dt:.1f} tok/s | accepted {acc} / drafted {drf} "
          f"| 接受率 {acc/drf*100:.1f}%" if acc and drf else
          f"  k={k}: 8 并发 {total/dt:.1f} tok/s | 无指标", flush=True)
    del llm
    import torch
    torch.cuda.empty_cache()
