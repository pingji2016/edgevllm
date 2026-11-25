"""第二梯队 A：vLLM 多 LoRA 服务 + 前缀缓存曲线"""
import json
import os
import time

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master"
ADAPTER = os.path.expanduser("~/apps/infer-research/lora_llm")
BASE_PROMPT = "用一句话解释什么是显存带宽。"
LONG_BASE = ("以下是设备手册的一部分。" + "设备在低功耗模式下会关闭无线模块。" * 200)
OUT = os.path.expanduser("~/apps/infer-research/lora_vllm.json")


def timed(llm, prompts, sp, lora=None):
    llm.generate(prompts[:1], sp, lora_request=lora)  # 预热
    t0 = time.time()
    outs = llm.generate(prompts, sp, lora_request=lora)
    dt = time.time() - t0
    n = sum(len(o.outputs[0].token_ids) for o in outs)
    return n / dt, dt, outs


def main():
    res = {}
    sp = SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True)

    print("=== 1. LoRA 服务（enable_lora）===", flush=True)
    t0 = time.time()
    llm = LLM(model=MD, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.45, enforce_eager=True, disable_log_stats=True,
              enable_lora=True, max_lora_rank=16, max_loras=2)
    print(f"  引擎就绪 {time.time()-t0:.0f}s", flush=True)

    lr = LoRARequest("sft", 1, ADAPTER)
    tps_base, _, out_base = timed(llm, [BASE_PROMPT], sp)
    tps_lora, _, out_lora = timed(llm, [BASE_PROMPT], sp, lora=lr)
    print(f"  单请求 base  {tps_base:5.1f} tok/s")
    print(f"  单请求 lora  {tps_lora:5.1f} tok/s（{(tps_lora/tps_base-1)*100:+.1f}%）")
    res["lora"] = dict(base=round(tps_base, 1), lora=round(tps_lora, 1),
                       delta_pct=round((tps_lora / tps_base - 1) * 100, 1))

    # 8 并发下带 LoRA
    tps8_base, _, _ = timed(llm, [BASE_PROMPT] * 8, sp)
    tps8_lora, _, _ = timed(llm, [BASE_PROMPT] * 8, sp, lora=lr)
    print(f"  8 并发 base  {tps8_base:6.1f} tok/s")
    print(f"  8 并发 lora  {tps8_lora:6.1f} tok/s（{(tps8_lora/tps8_base-1)*100:+.1f}%）")
    res["lora_conc8"] = dict(base=round(tps8_base, 1), lora=round(tps8_lora, 1),
                             delta_pct=round((tps8_lora / tps8_base - 1) * 100, 1))

    print(f"\n  base 输出: {out_base[0].outputs[0].text.strip()[:60]}")
    print(f"  lora 输出: {out_lora[0].outputs[0].text.strip()[:60]}")
    res["samples"] = dict(base=out_base[0].outputs[0].text.strip()[:80],
                          lora=out_lora[0].outputs[0].text.strip()[:80])
    del llm
    torch.cuda.empty_cache()

    print("\n=== 2. 前缀缓存曲线 ===", flush=True)
    llm = LLM(model=MD, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.45, enforce_eager=True, disable_log_stats=True,
              enable_prefix_caching=True)
    sp2 = SamplingParams(max_tokens=16, temperature=0.0)
    curve = []
    for target in (512, 2048, 8192):
        prompt = LONG_BASE[: int(target * 3)] + "\n问题：低功耗模式会关闭什么？"
        llm.generate([prompt], sp2)  # 写入缓存
        t0 = time.time()
        llm.generate([prompt], sp2)  # 命中
        hit = time.time() - t0
        curve.append(dict(tokens_target=target, hit_ms=round(hit * 1000, 1)))
        print(f"  命中 ~{target:5d} token 前缀：{hit*1000:7.1f} ms", flush=True)
    res["prefix_curve"] = curve
    del llm
    torch.cuda.empty_cache()

    # 关闭缓存做对照
    llm = LLM(model=MD, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.45, enforce_eager=True, disable_log_stats=True,
              enable_prefix_caching=False)
    off = []
    for target in (512, 2048, 8192):
        prompt = LONG_BASE[: int(target * 3)] + "\n问题：低功耗模式会关闭什么？"
        llm.generate([prompt], sp2)
        t0 = time.time()
        llm.generate([prompt], sp2)
        miss = time.time() - t0
        off.append(dict(tokens_target=target, miss_ms=round(miss * 1000, 1)))
        print(f"  无缓存 ~{target:5d} token：{miss*1000:7.1f} ms", flush=True)
    res["prefix_off"] = off

    for a, b in zip(curve, off):
        print(f"  {a['tokens_target']:5d} token: {a['hit_ms']:7.1f} → {b['miss_ms']:7.1f} ms "
              f"（省 {(1-a['hit_ms']/b['miss_ms'])*100:.0f}%）")

    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    print("\n结果:", OUT)


if __name__ == "__main__":
    main()
