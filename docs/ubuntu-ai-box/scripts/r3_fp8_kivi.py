"""第三轮：FP8 量化（torchao）+ KIVI 式 KV cache 量化

FP8：Qwen3-0.6B，bf16 vs Float8WeightOnlyConfig（权重 8bit，算 bf16）
     vs Float8DynamicActivationFloat8WeightConfig（激活也 8bit）
KIVI：Qwen3-1.7B，先 prefill 长上下文，把 KV cache 量化到 8/4/2 bit，
      再继续生成问答题，看「大海捞针」是否还答得对 + 理论显存节省
用法：~/apps/bench/yolo-venv/bin/python r3_fp8_kivi.py
"""
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MD06 = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master"
MD17 = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
CORPUS = os.path.expanduser("~/models/infer-tmp/corpus.txt")
SECRET = "738291"


def load(md):
    m = AutoModelForCausalLM.from_pretrained(
        md, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa").eval()
    t = AutoTokenizer.from_pretrained(md)
    return m, t


def ppl(model, ids, chunk=256):
    losses = []
    with torch.inference_mode():
        for i in range(0, ids.size(1) - chunk, chunk):
            c = ids[:, i:i + chunk].to("cuda")
            losses.append(model(c, labels=c).loss.float())
    return torch.exp(torch.stack(losses).mean()).item()


def decode_tps(model, tok, prompt, n=64):
    enc = tok(prompt, return_tensors="pt").to("cuda")
    n_in = enc["input_ids"].size(1)
    with torch.inference_mode():
        model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        t0 = time.time()
        out = model.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        dt = time.time() - t0
    return (out.shape[1] - n_in) / dt


# ---------------- FP8 ----------------
def part_fp8():
    print("=== FP8（torchao）Qwen3-0.6B ===")
    from torchao.quantization import (quantize_, Float8WeightOnlyConfig,
                                      Float8DynamicActivationFloat8WeightConfig)
    tok = AutoTokenizer.from_pretrained(MD06)
    txt = open(CORPUS, encoding="utf-8").read()
    ids = tok(txt[:6000], return_tensors="pt").input_ids[:, :1024]
    rows = []

    for tag, cfg in (("bf16", None),
                     ("fp8-w", Float8WeightOnlyConfig()),
                     ("fp8-dyn", Float8DynamicActivationFloat8WeightConfig())):
        m, _ = load(MD06)
        try:
            if cfg is not None:
                quantize_(m, cfg)
            v = ppl(m, ids)
            tps = decode_tps(m, tok, "用一句话解释显存带宽。")
            vram = torch.cuda.max_memory_allocated() / 2**20
            rows.append(dict(kind=tag, ppl=round(v, 3), tps=round(tps, 1), vram=round(vram)))
            print(f"  {tag:8s} PPL {v:7.3f} | {tps:5.1f} tok/s | 峰值 {vram:.0f} MiB", flush=True)
        except Exception as e:
            print(f"  {tag:8s} ❌ {type(e).__name__}: {str(e)[:110]}", flush=True)
        del m
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return rows


# ---------------- KIVI ----------------
def quant_tensor(t, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    return ((t / scale).round().clamp(-qmax - 1, qmax) * scale).to(t.dtype)


def part_kivi():
    print("\n=== KIVI 式 KV 量化：Qwen3-1.7B，8K 上下文大海捞针 ===")
    m, tok = load(MD17)
    FILL = ("云计算平台在边缘侧部署推理服务时，需要同时考虑时延、带宽与能耗三个维度。"
            "在实际工程中，工程师往往先测量显存带宽，再决定量化位宽与批大小。")
    base = tok(FILL, add_special_tokens=False).input_ids
    needle = tok(f"\n\n重要备忘：保险箱密码是 {SECRET}。\n\n", add_special_tokens=False).input_ids
    question = tok("\n\n问：保险箱密码是多少？只回答数字。\n答：", add_special_tokens=False).input_ids
    ctx = (base * 700)[:7000]
    mid = len(ctx) // 2
    ctx = ctx[:mid] + needle + ctx[mid:]

    rows = []
    for bits in (16, 8, 4, 2):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ids = torch.tensor([ctx], dtype=torch.long).to("cuda")
        with torch.inference_mode():
            out = m(ids, use_cache=True)
            cache = out.past_key_values
            kv_bytes = 0
            for layer in cache.layers:
                kv_bytes += layer.keys.numel() * layer.keys.element_size()
                kv_bytes += layer.values.numel() * layer.values.element_size()
                if bits < 16:
                    layer.keys = quant_tensor(layer.keys, bits)
                    layer.values = quant_tensor(layer.values, bits)
            qbytes = sum(l.numel() * l.element_size() for l in [cache.layers[0].keys])
            # 用问题续写
            q = torch.tensor([question], dtype=torch.long).to("cuda")
            t0 = time.time()
            o = m(q, past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            nxt = o.logits[:, -1].argmax(-1, keepdim=True)
            toks = [nxt]
            for _ in range(15):
                o = m(nxt, past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                nxt = o.logits[:, -1].argmax(-1, keepdim=True)
                toks.append(nxt)
            torch.cuda.synchronize()
            dt = time.time() - t0
            gen = [torch.cat(toks, dim=1)]
        ans = tok.decode(gen[0][0], skip_special_tokens=True).strip()
        hit = SECRET in ans
        theoretical = kv_bytes * bits / 16 / 2**20
        rows.append(dict(bits=bits, hit=hit, answer=ans[:30], kv_mib=round(kv_bytes / 2**20, 1),
                         theoretical_quant_mib=round(theoretical, 1), sec=round(dt, 2)))
        print(f"  KV {bits:2d}bit | 原始 {kv_bytes/2**20:6.1f} MiB → 理论 {theoretical:6.1f} MiB | "
              f"检索 {'✅' if hit else '❌'} | {dt:4.1f}s | {ans[:20]}", flush=True)
        del cache, out
        torch.cuda.empty_cache()
    return rows


if __name__ == "__main__":
    import sys
    res = dict()
    which = os.environ.get("PART", "both")
    if which in ("both", "fp8"):
        res["fp8"] = part_fp8()
    if which in ("both", "kivi"):
        res["kivi"] = part_kivi()
    json.dump(res, open(os.path.expanduser("~/apps/infer-research/r3_fp8_kivi.json"), "w"),
              ensure_ascii=False, indent=2)
