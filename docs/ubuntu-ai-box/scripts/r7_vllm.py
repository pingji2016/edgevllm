"""第七轮：vLLM 原生结构化输出 + 长上下文 decode 曲线"""
import json
import os
import time

import torch
from vllm import LLM, SamplingParams

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
BASE = ("云计算平台在边缘侧部署推理服务时，需要同时考虑时延、带宽与能耗三个维度。"
        "工程师往往先测量显存带宽，再决定量化位宽与批大小，最后才调整算子实现。")
SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "year": {"type": "integer"},
        "products": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["company", "year", "products", "confidence"],
}
JSON_PROMPT = ("从这句话抽取结构化信息：睿创微纳在 2024 年推出了红外热成像模组和车载夜视方案，"
               "置信度 0.9。只输出 JSON。")
LENS = [512, 2048, 8192, 16384]


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MD)
    unit = tok(BASE, add_special_tokens=False).input_ids
    res = {}

    llm = LLM(model=MD, dtype="bfloat16", max_model_len=32768,
              gpu_memory_utilization=0.55, enforce_eager=True, disable_log_stats=True,
              enable_prefix_caching=False)
    print(f"引擎就绪", flush=True)

    # ---------- 1. 结构化输出 ----------
    print("\n=== 1. 原生结构化输出（guided decoding）===", flush=True)
    for tag in ("free", "guided"):
        sp_kw = {}
        if tag == "guided":
            try:
                from vllm.sampling_params import StructuredOutputsParams
                sp_kw["structured_outputs"] = StructuredOutputsParams(json=SCHEMA)
            except Exception:
                sp_kw["guided_json"] = SCHEMA
        sp = SamplingParams(max_tokens=200, temperature=0.0, **sp_kw)
        llm.generate([JSON_PROMPT], sp)  # 预热
        t0 = time.time()
        outs = llm.generate([JSON_PROMPT] * 4, sp)
        dt = time.time() - t0
        n = sum(len(o.outputs[0].token_ids) for o in outs)
        ok = 0
        for o in outs:
            try:
                d = json.loads(o.outputs[0].text.strip())
                ok += all(k in d for k in ("company", "year", "products", "confidence"))
            except Exception:
                pass
        res[tag] = dict(valid=f"{ok}/4", tps=round(n / dt, 1), tokens=round(n / 4, 1))
        print(f"  {tag:7s} 合法 {ok}/4 | 平均 {n/4:.0f} token | {n/dt:.1f} tok/s", flush=True)

    # ---------- 2. 长上下文 decode ----------
    print("\n=== 2. 长上下文 decode（每次生成 64 token）===", flush=True)
    dec = []
    for L in LENS:
        prompt = BASE * max(1, L // len(unit))
        n = len(tok(prompt, add_special_tokens=False).input_ids)
        sp = SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True)
        llm.generate([prompt], sp)  # 预热
        out = llm.generate([prompt], sp)[0]
        m = getattr(out, "metrics", None)
        if m is not None and getattr(m, "first_token_time", None) and getattr(m, "finished_time", None):
            ttft = m.first_token_time - m.arrival_time
            dec_s = m.finished_time - m.first_token_time
            tps = max(1, len(out.outputs[0].token_ids) - 1) / max(dec_s, 1e-6)
        else:
            ttft, tps = float("nan"), float("nan")
        dec.append(dict(tokens=n, ttft_ms=round(ttft * 1000, 1), decode_tps=round(tps, 1)))
        print(f"  ctx {n:6d} | TTFT {ttft*1000:7.1f} ms | decode {tps:6.1f} tok/s", flush=True)
    res["decode_curve"] = dec

    json.dump(res, open(os.path.expanduser("~/apps/infer-engines/r7.json"), "w"), ensure_ascii=False, indent=2)
    print("\n结果:", json.dumps(res, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
