"""路线 B：KV cache 压缩与长上下文（Qwen3-1.7B + kvpress）

测什么：
  - 不同上下文长度下，全量 KV 与各种压缩策略的 **峰值显存**
  - 压缩后「大海捞针」检索是否还能答对
  - decode 速度是否几乎不变（验证：省容量 ≠ 省带宽）

用法：
  HF_ENDPOINT=https://hf-mirror.com ~/apps/bench/yolo-venv/bin/python routeB_kv.py
"""
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen3-1.7B")
CTX_LENS = [int(x) for x in os.environ.get("CTX_LENS", "2000,8000").split(",")]
GEN_TOKENS = 32
SECRET = "738291"

FILLER = (
    "云计算平台在边缘侧部署推理服务时，需要同时考虑时延、带宽与能耗三个维度。"
    "在实际工程中，工程师往往先测量显存带宽，再决定量化位宽与批大小，最后才调整算子实现。"
)


def build_prompt(tok, ctx_len):
    """构造 大海捞针：在约 ctx_len 个 token 的填充文本中间插入密码"""
    base = tok(FILLER, add_special_tokens=False).input_ids
    n_fill = max(16, (ctx_len - 64) // len(base))
    tokens = base * n_fill
    needle = tok(f"\n\n重要备忘：保险箱密码是 {SECRET}，请务必记住。\n\n",
                 add_special_tokens=False).input_ids
    pos = len(tokens) // 2
    tokens = tokens[:pos] + needle + tokens[pos:]
    question = tok("\n\n问：保险箱密码是多少？只回答数字。\n答：", add_special_tokens=False).input_ids
    ids = torch.tensor([tokens + question], dtype=torch.long)
    return ids


def run_one(model, tok, press, ctx_len):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ids = build_prompt(tok, ctx_len).to("cuda")
    real_len = ids.size(1)
    ctx = press(model) if press is not None else None
    if ctx is not None:
        ctx.__enter__()
    try:
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(ids, max_new_tokens=GEN_TOKENS, do_sample=False,
                                 min_new_tokens=GEN_TOKENS, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        dt = time.time() - t0
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    text = tok.decode(out[0, real_len:], skip_special_tokens=True)
    peak = torch.cuda.max_memory_allocated() / 2**20
    ok = SECRET in text
    tps = GEN_TOKENS / dt
    return dict(ctx_tokens=real_len, peak_mib=round(peak), hit=ok,
                answer=text.strip()[:40], total_s=round(dt, 2), decode_tps=round(tps, 1))


def make_press(name, ratio):
    import kvpress
    if name == "full":
        return None
    cls = {
        "snap": "SnapKVPress",
        "knorm": "KnormPress",
        "tova": "TOVAPress",
        "stream": "StreamingLLMPress",
        "random": "RandomPress",
    }[name]
    return getattr(kvpress, cls)(compression_ratio=ratio)


def main():
    import kvpress  # noqa: F401  确认可用
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    model.eval()
    print(f"模型 {MODEL} | GPU {torch.cuda.get_device_name(0)}")
    print(f"阈值密码={SECRET}（大海捞针）")

    variants = [("full", 0.0), ("snap", 0.5), ("snap", 0.75), ("knorm", 0.5),
                ("tova", 0.5), ("stream", 0.5), ("random", 0.5)]
    results = []
    for ctx_len in CTX_LENS:
        print(f"\n--- 目标上下文 ~{ctx_len} tokens ---")
        for name, ratio in variants:
            label = f"{name}" + ("" if name == "full" else f"@{ratio}")
            try:
                press = make_press(name, ratio)
                r = run_one(model, tok, press, ctx_len)
                r.update(variant=label, ratio=ratio)
                results.append(r)
                mark = "✅" if r["hit"] else "❌"
                print(f"  {label:12s} 实际 {r['ctx_tokens']:5d} tok | 峰值 {r['peak_mib']:5d} MiB | "
                      f"检索 {mark} | {r['total_s']:5.1f}s | {r['decode_tps']:.1f} tok/s | {r['answer']}",
                      flush=True)
            except Exception as e:
                print(f"  {label:12s} ❌ {type(e).__name__}: {str(e)[:120]}", flush=True)
                results.append(dict(variant=label, ctx_len_target=ctx_len, error=str(e)[:200]))
                torch.cuda.empty_cache()

    out = os.path.expanduser("~/apps/infer-research/routeB_results.json")
    json.dump(results, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\n结果已存: {out}")


if __name__ == "__main__":
    main()
