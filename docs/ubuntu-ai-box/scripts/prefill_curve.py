"""长上下文 prefill 曲线（vLLM，Qwen3-1.7B bf16）

测 512 / 2048 / 8192 / 16384 / 32768 token 的 prefill 吞吐
方法：max_tokens=1，耗时 ≈ prefill + 1 token decode
"""
import json
import os
import time

import torch
from vllm import LLM, SamplingParams

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
BASE = ("云计算平台在边缘侧部署推理服务时，需要同时考虑时延、带宽与能耗三个维度。"
        "工程师往往先测量显存带宽，再决定量化位宽与批大小，最后才调整算子实现。")
LENS = [512, 2048, 8192, 16384, 32768]


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MD)
    unit = tok(BASE, add_special_tokens=False).input_ids
    tok_per_unit = len(unit)

    t0 = time.time()
    llm = LLM(model=MD, dtype="bfloat16", max_model_len=32768,
              gpu_memory_utilization=0.55, enforce_eager=True, disable_log_stats=True,
              enable_prefix_caching=False)  # 必须关，否则第二次命中缓存，测到的是缓存不是 prefill
    print(f"引擎就绪 {time.time()-t0:.0f}s", flush=True)

    rows = []
    for L in LENS:
        reps = max(1, L // tok_per_unit)
        prompt = BASE * reps
        n = len(tok(prompt, add_special_tokens=False).input_ids)
        sp = SamplingParams(max_tokens=1, temperature=0.0)
        llm.generate([prompt], sp)  # 预热（含编译/缓存分配）
        torch.cuda.empty_cache()
        t0 = time.time()
        llm.generate([prompt], sp)
        dt = time.time() - t0
        rows.append(dict(tokens=n, seconds=round(dt, 3), prefill_tps=round(n / dt, 1)))
        print(f"  {n:6d} token | {dt*1000:8.1f} ms | {n/dt:8.1f} tok/s", flush=True)

    json.dump(rows, open(os.path.expanduser("~/apps/infer-engines/prefill_curve.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
