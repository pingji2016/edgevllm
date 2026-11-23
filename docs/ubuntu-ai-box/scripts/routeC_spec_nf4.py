"""路线 C 补充：把草稿模型换成 4bit（NF4），验证「草稿太贵」这个假设

机制假设：
  投机解码的收益 ≈ (目标每 token 带宽) / (草稿每步带宽 + 目标每步带宽/(1+αk))
  在带宽受限的卡上，草稿若用 bf16，每步要读 1.2 GB，把收益吃干；
  把草稿压到 4bit（~0.4 GB）后应该转正。

用法：DRAFT_4BIT=1 python routeC_spec_nf4.py
"""
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TGT = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
DRF = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master"
NEW_TOKENS = int(os.environ.get("NEW_TOKENS", "128"))
NUM_DRAFT = int(os.environ.get("NUM_DRAFT", "4"))

PROMPTS = [
    "用三句话解释什么是显存带宽，以及它为什么决定了大模型的解码速度。",
    "请写一个 Python 函数，判断一个字符串是否是回文，并说明时间复杂度。",
    "一个 GPU 有 448 GB/s 带宽，模型权重 4 GB，理论上 decode 速度上限是多少？请给出计算过程。",
]


def main():
    tok = AutoTokenizer.from_pretrained(TGT)
    target = AutoModelForCausalLM.from_pretrained(
        TGT, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa").eval()

    cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    draft = AutoModelForCausalLM.from_pretrained(
        DRF, quantization_config=cfg, device_map={"": 0}).eval()

    print(f"靶 Qwen3-1.7B (bf16, {3.4:.1f} GB) | 草稿 Qwen3-0.6B (NF4, ~0.45 GB) | k={NUM_DRAFT}")

    counter = {"n": 0}
    orig = target.forward

    def counting(*a, **kw):
        counter["n"] += 1
        return orig(*a, **kw)

    target.forward = counting

    def gen(**kw):
        with torch.inference_mode():
            return target.generate(**kw, max_new_tokens=NEW_TOKENS, do_sample=False,
                                   pad_token_id=tok.eos_token_id)

    rows = []
    for i, p in enumerate(PROMPTS, 1):
        enc = tok(p, return_tensors="pt").to("cuda")
        n_in = enc["input_ids"].size(1)
        gen(**enc)  # 预热
        torch.cuda.synchronize()

        t0 = time.time()
        base = gen(**enc)
        torch.cuda.synchronize()
        t_base = time.time() - t0
        n_base = base.shape[1] - n_in

        counter["n"] = 0
        torch.cuda.synchronize()
        t0 = time.time()
        spec = gen(**enc, assistant_model=draft, num_assistant_tokens=NUM_DRAFT)
        torch.cuda.synchronize()
        t_spec = time.time() - t0
        n_new = spec.shape[1] - n_in

        steps = counter["n"] or 1
        tps = n_new / steps
        alpha = max(0.0, (tps - 1) / NUM_DRAFT)
        same = base[0].tolist() == spec[0].tolist()
        rows.append(dict(prompt=i, base_tps=round(n_base / t_base, 1),
                         spec_tps=round(n_new / t_spec, 1),
                         speedup=round(t_base / t_spec, 2), steps=steps,
                         tokens_per_step=round(tps, 2), alpha_est=round(alpha, 2),
                         identical=same))
        print(f"[{i}] 基线 {rows[-1]['base_tps']:5.1f} tok/s | 投机 {rows[-1]['spec_tps']:5.1f} tok/s | "
              f"加速 {rows[-1]['speedup']:.2f}x | 每步 {tps:.2f} token (α≈{alpha:.2f}) | "
              f"{'一致 ✅' if same else '不一致'}")

    n = len(rows)
    print(f"\n平均加速比 {sum(r['speedup'] for r in rows)/n:.2f}x | "
          f"平均 α≈{sum(r['alpha_est'] for r in rows)/n:.2f}")
    json.dump(rows, open(os.path.expanduser("~/apps/infer-research/routeC_nf4_results.json"), "w"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
