"""路线 A：同一张卡上多种量化的横向实测（Qwen3-1.7B）

对比：BF16 / INT8(bnb) / NF4-4bit(bnb)
指标：加载后显存、PPL、decode 吞吐、加载耗时

用法：
  HF_ENDPOINT=https://hf-mirror.com ~/apps/bench/yolo-venv/bin/python routeA_quant.py
"""
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen3-1.7B")
CORPUS = os.path.expanduser("~/apps/minimind/dataset/pretrain_t2t_mini.jsonl")
PROMPT = "请用一句话解释什么是显存带宽。"
GEN_TOKENS = 128
PPL_TOKENS = 1024


def build(kind):
    kw = dict(device_map={"": 0}, attn_implementation="sdpa")
    if kind == "bf16":
        return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, **kw)
    if kind == "int8":
        return AutoModelForCausalLM.from_pretrained(
            MODEL, quantization_config=BitsAndBytesConfig(load_in_8bit=True), **kw)
    if kind == "nf4":
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=cfg, **kw)
    raise ValueError(kind)


def load_corpus_tokens(tok, n=PPL_TOKENS * 2):
    texts = []
    total = 0
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                t = d.get("text") or d.get("content") or ""
            except Exception:
                t = line
            if not t:
                continue
            texts.append(t)
            total += len(t)
            if total > n * 4:
                break
    ids = tok("".join(texts), return_tensors="pt").input_ids
    return ids[:, :n]


def measure_ppl(model, ids):
    model.eval()
    losses = []
    with torch.inference_mode():
        for i in range(0, ids.size(1) - PPL_TOKENS, PPL_TOKENS):
            chunk = ids[:, i:i + PPL_TOKENS].to("cuda")
            out = model(chunk, labels=chunk)
            losses.append(out.loss.float())
    return torch.exp(torch.stack(losses).mean()).item()


def measure_decode(model, tok):
    inputs = tok(PROMPT, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        # 预热（首次会触发 kernel 选择/内存池分配）
        model.generate(**inputs, max_new_tokens=8, do_sample=False, min_new_tokens=8)
        torch.cuda.synchronize()
        t0 = time.time()
        out = model.generate(**inputs, max_new_tokens=GEN_TOKENS, do_sample=False,
                             min_new_tokens=GEN_TOKENS)
        torch.cuda.synchronize()
        dt = time.time() - t0
    n_new = out.shape[1] - inputs.input_ids.shape[1]
    return n_new / dt, dt


def run(kind, tok, ids):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = build(kind)
    load_time = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 2**20
    ppl = measure_ppl(model, ids)
    tps, dt = measure_decode(model, tok)
    r = dict(kind=kind, vram_mib=round(vram), ppl=round(ppl, 3),
             decode_tps=round(tps, 1), load_s=round(load_time, 1))
    print(f"  {kind:5s} 显存 {r['vram_mib']:5d} MiB | PPL {r['ppl']:7.3f} | "
          f"decode {r['decode_tps']:6.1f} tok/s | 加载 {r['load_s']:.1f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    return r


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = load_corpus_tokens(tok)
    print(f"模型 {MODEL} | PPL 语料 {ids.size(1)} tokens | 生成长度 {GEN_TOKENS}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    results = []
    for kind in ("bf16", "int8", "nf4"):
        try:
            results.append(run(kind, tok, ids))
        except Exception as e:
            print(f"  {kind:5s} ❌ {type(e).__name__}: {str(e)[:160]}")
            results.append(dict(kind=kind, error=str(e)[:200]))
            torch.cuda.empty_cache()
    out = os.path.expanduser("~/apps/infer-research/routeA_results.json")
    json.dump(results, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\n结果已存: {out}")


if __name__ == "__main__":
    main()
