"""vLLM + GPTQ-Int8 权重：吞吐 + ngram 投机（量化 + 投机组合）"""
import json
import os
import time

import torch
from vllm import LLM, SamplingParams

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B-GPTQ-Int8/snapshots/master"
OUT = os.path.expanduser("~/apps/infer-engines/gptq_test.json")
FREE = "用三句话解释什么是显存带宽，以及它为什么决定了大模型的解码速度。"
COPY_TEXT = ("边缘计算把推理放在离用户最近的地方。它降低了网络延迟，减少了回传带宽，"
             "也避免了数据离开现场。代价是需要更小的模型、更激烈的量化。")
COPY = f"请原样复述下面这段话，不要增删任何字：{COPY_TEXT}"


def build(**kw):
    t0 = time.time()
    llm = LLM(model=MD, dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.45, enforce_eager=True,
              disable_log_stats=True, **kw)
    print(f"  引擎就绪 {time.time()-t0:.0f}s", flush=True)
    return llm


def run(llm, prompts, max_tokens=128):
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
    llm.generate(prompts[:1], sp)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    n = sum(len(o.outputs[0].token_ids) for o in outs)
    return n / dt


def main():
    res = {}
    print("=== A. GPTQ-Int8 基线 ===", flush=True)
    try:
        llm = build()
        res["single"] = round(run(llm, [FREE]), 1)
        res["conc8"] = round(run(llm, [FREE] * 8), 1)
        res["copy_base"] = round(run(llm, [COPY], 160), 1)
        print(f"  单流 {res['single']} | 8 并发 {res['conc8']} | 抄写基线 {res['copy_base']}", flush=True)
        del llm
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ❌ {type(e).__name__}: {str(e)[:160]}", flush=True)
        res["error"] = type(e).__name__
        json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
        return

    print("\n=== B. GPTQ-Int8 + ngram 投机（抄写）===", flush=True)
    try:
        llm = build(speculative_config={"method": "ngram", "num_speculative_tokens": 5,
                                        "prompt_lookup_max": 8, "prompt_lookup_min": 3})
        t = run(llm, [COPY], 160)
        res["copy_ngram"] = round(t, 1)
        res["speedup"] = round(t / res["copy_base"], 2)
        print(f"  ngram: {t:.1f} tok/s | 相对量化模型基线 {res['speedup']}×", flush=True)
        del llm
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ❌ {type(e).__name__}: {str(e)[:160]}", flush=True)
        res["ngram_error"] = type(e).__name__

    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    print("\n结果:", json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
