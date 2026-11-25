"""前缀缓存精确测量：按 token 数构造 prompt，命中/未命中各取 3 次最小值"""
import json
import os
import time

from vllm import LLM, SamplingParams

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master"
OUT = os.path.expanduser("~/apps/infer-research/prefix_curve.json")


def build_prompt(tok, n_tokens):
    unit = "设备在低功耗模式下会关闭无线模块。"
    ids = tok(unit, add_special_tokens=False).input_ids
    rep = (ids * (n_tokens // len(ids) + 1))[:n_tokens]
    return tok.decode(rep) + "\n问题：低功耗模式会关闭什么？"


def ttft_ms(llm, prompt, reps=3):
    sp = SamplingParams(max_tokens=8, temperature=0.0)
    best = 1e9
    for _ in range(reps):
        t0 = time.time()
        llm.generate([prompt], sp)
        best = min(best, time.time() - t0)
    return best * 1000


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MD)
    res = {"on": [], "off": []}

    for cache in (True, False):
        llm = LLM(model=MD, dtype="bfloat16", max_model_len=32768,
                  gpu_memory_utilization=0.45, enforce_eager=True, disable_log_stats=True,
                  enable_prefix_caching=cache)
        for n in (1024, 4096, 16384):
            p = build_prompt(tok, n)
            real = len(tok(p, add_special_tokens=False).input_ids)
            if cache:
                llm.generate([p], SamplingParams(max_tokens=8, temperature=0.0))  # 写缓存
            ms = ttft_ms(llm, p)
            res["on" if cache else "off"].append(dict(tokens=real, ms=round(ms, 1)))
            print(f"  {'命中' if cache else '无缓存'} {real:6d} token → {ms:7.1f} ms", flush=True)
        del llm
        import torch
        torch.cuda.empty_cache()

    print("\n前缀长度 | 命中 | 无缓存 | 节省")
    for a, b in zip(res["on"], res["off"]):
        print(f"{a['tokens']:7d} | {a['ms']:7.1f} | {b['ms']:7.1f} | {(1-a['ms']/b['ms'])*100:5.1f}%")

    json.dump(res, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
