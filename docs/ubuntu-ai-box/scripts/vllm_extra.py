"""新增：vLLM 并发曲线 + ngram 投机解码（引擎层验证）+ KV cache fp8

三个引擎实例分三次跑（每次加载约 2-3 分钟）：
  1) 基线：并发 1/2/4/8/16 的吞吐曲线
  2) ngram 投机：抄写类 prompt，看是否比基线快
  3) KV cache fp8：显存对比
"""
import json
import os
import time

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
OUT = os.path.expanduser("~/apps/infer-engines/vllm_extra.json")

FREE = "用三句话解释什么是显存带宽，以及它为什么决定了大模型的解码速度。"
COPY_TEXT = ("边缘计算把推理放在离用户最近的地方。它降低了网络延迟，减少了回传带宽，"
             "也避免了数据离开现场。代价是需要更小的模型、更激烈的量化。")
COPY = f"请原样复述下面这段话，不要增删任何字：{COPY_TEXT}"


def build(**kw):
    from vllm import LLM
    t0 = time.time()
    llm = LLM(model=MD, dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.45, enforce_eager=True,
              disable_log_stats=True, **kw)
    print(f"  引擎就绪 {time.time()-t0:.0f}s", flush=True)
    return llm


def tp(llm, prompts, max_tokens=128):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
    llm.generate(prompts[:1], sp)  # 预热
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    n = sum(len(o.outputs[0].token_ids) for o in outs)
    return n / dt, dt


def main():
    res = {}

    print("=== 1. 并发曲线（基线）===", flush=True)
    llm = build()
    curve = []
    for c in (1, 2, 4, 8, 16):
        tot, dt = tp(llm, [FREE] * c)
        curve.append(dict(concurrency=c, total_tps=round(tot, 1), per_stream=round(tot / c, 1)))
        print(f"  并发 {c:2d}: 合计 {tot:7.1f} tok/s | 单流 {tot/c:6.1f}", flush=True)
    res["curve"] = curve
    del llm
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()

    print("\n=== 2. ngram 投机（抄写类 prompt）===", flush=True)
    try:
        llm = build(speculative_config={"method": "ngram", "num_speculative_tokens": 5,
                                        "prompt_lookup_max": 8, "prompt_lookup_min": 3})
        spec_tps, _ = tp(llm, [COPY], max_tokens=160)
        print(f"  ngram 投机: {spec_tps:.1f} tok/s", flush=True)
        res["ngram"] = round(spec_tps, 1)
        del llm
        gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ❌ ngram 投机不可用: {type(e).__name__}: {str(e)[:150]}", flush=True)
        res["ngram"] = f"error: {type(e).__name__}"

    print("\n=== 3. 同 prompt 的基线（对照 ngram）===", flush=True)
    llm = build()
    base_tps, _ = tp(llm, [COPY], max_tokens=160)
    print(f"  基线: {base_tps:.1f} tok/s", flush=True)
    res["copy_baseline"] = round(base_tps, 1)
    del llm
    gc.collect(); torch.cuda.empty_cache()

    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    print("\n结果:", json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
