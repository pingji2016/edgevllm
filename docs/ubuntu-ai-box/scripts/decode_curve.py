"""长上下文 decode 曲线（用 T64 - T1 反推 decode 速度）"""
import json
import os
import time

import torch
from vllm import LLM, SamplingParams

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
BASE = ("云计算平台在边缘侧部署推理服务时，需要同时考虑时延、带宽与能耗三个维度。"
        "工程师往往先测量显存带宽，再决定量化位宽与批大小，最后才调整算子实现。")
LENS = [512, 2048, 8192, 16384, 32768]


def timeit(llm, prompt, n, reps=1):
    sp = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
    llm.generate([prompt], sp)  # 预热
    best = 1e9
    for _ in range(reps):
        t0 = time.time()
        llm.generate([prompt], sp)
        best = min(best, time.time() - t0)
    return best


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MD)
    unit = tok(BASE, add_special_tokens=False).input_ids

    llm = LLM(model=MD, dtype="bfloat16", max_model_len=32768,
              gpu_memory_utilization=0.55, enforce_eager=True, disable_log_stats=True,
              enable_prefix_caching=False)
    rows = []
    print("ctx | TTFT(prefill) | 64 token 总时长 | decode tok/s | 每 token 读取量", flush=True)
    for L in LENS:
        prompt = BASE * max(1, L // len(unit))
        n = len(tok(prompt, add_special_tokens=False).input_ids)
        t1 = timeit(llm, prompt, 1, reps=2)
        t256 = timeit(llm, prompt, 256, reps=2)
        dec = 255 / max(t256 - t1, 1e-6)
        # 每 token 需要读的字节：权重 3.4GB + KV(2*28*8*128*n*2)
        kv_gb = 2 * 28 * 8 * 128 * n * 2 / 1e9
        per_tok_gb = 3.4 + kv_gb
        rows.append(dict(tokens=n, ttft_ms=round(t1 * 1000, 1), decode_tps=round(dec, 1),
                         kv_gb=round(kv_gb, 2), per_token_gb=round(per_tok_gb, 2),
                         effective_bw=round(per_tok_gb * dec)))
        print(f"{n:6d} | {t1*1000:8.1f} ms | {t256*1000:9.1f} ms | {dec:7.1f} | "
              f"{per_tok_gb:5.2f} GB → 有效带宽 {per_tok_gb*dec:4.0f} GB/s", flush=True)

    json.dump(rows, open(os.path.expanduser("~/apps/infer-engines/decode_curve.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
