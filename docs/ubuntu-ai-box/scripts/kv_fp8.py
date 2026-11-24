"""单跑：vLLM KV cache fp8 vs auto（显存 + 吞吐）"""
import json
import os
import time

import torch
from vllm import LLM, SamplingParams

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
P = "用三句话解释什么是显存带宽，以及它为什么决定了大模型的解码速度。"


def one(kv):
    t0 = time.time()
    llm = LLM(model=MD, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.45, enforce_eager=True,
              disable_log_stats=True, kv_cache_dtype=kv)
    load = time.time() - t0
    sp = SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True)
    llm.generate([P] * 8, sp)
    t0 = time.time()
    outs = llm.generate([P] * 8, sp)
    dt = time.time() - t0
    n = sum(len(o.outputs[0].token_ids) for o in outs)
    res = dict(kv=kv, load_s=round(load, 1), tps_8=round(n / dt, 1))
    del llm
    torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    out = []
    for kv in ("auto", "fp8"):
        try:
            r = one(kv)
            print(f"RESULT {json.dumps(r)}", flush=True)
            out.append(r)
        except Exception as e:
            print(f"RESULT_FAIL {kv} {type(e).__name__}: {str(e)[:120]}", flush=True)
            out.append(dict(kv=kv, error=type(e).__name__))
    json.dump(out, open(os.path.expanduser("~/apps/infer-engines/kv_fp8.json"), "w"), indent=2)
