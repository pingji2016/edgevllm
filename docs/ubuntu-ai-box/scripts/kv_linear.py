"""KV cache 线性增长验证（Qwen3-1.7B，transformers 实测）

理论：每 token KV 字节 = 2 × layers × kv_heads × head_dim × dtype_bytes
实测：不同上下文长度下 use_cache 前向的显存增量，求斜率
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
LENS = [1024, 2048, 4096, 8192, 16384]


def main():
    tok = AutoTokenizer.from_pretrained(MD)
    m = AutoModelForCausalLM.from_pretrained(
        MD, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa").eval()
    cfg = m.config
    layers = cfg.num_hidden_layers
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    per_tok = 2 * layers * kv_heads * head_dim * 2
    print(f"配置: layers={layers} kv_heads={kv_heads} head_dim={head_dim}")
    print(f"理论 KV: {per_tok} B/token = {per_tok/1024:.1f} KB/token\n")

    base = "云计算平台在边缘侧部署推理服务时，需要同时考虑时延带宽与能耗三个维度。"
    rows = []
    for L in LENS:
        text = base * max(1, L // len(tok(base, add_special_tokens=False).input_ids))
        ids = tok(text, return_tensors="pt").input_ids[:, :L].to("cuda")
        n = ids.size(1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        with torch.inference_mode():
            out = m.model(ids, use_cache=True)  # 用 base model，避免 logits 张量污染测量
        after = torch.cuda.memory_allocated()
        delta = after - before
        rows.append(dict(tokens=n, kv_mib=round(delta / 2**20, 1),
                         kb_per_token=round(delta / n / 1024, 1)))
        print(f"  ctx {n:6d} | KV {delta/2**20:8.1f} MiB | "
              f"{delta/n/1024:6.1f} KB/token | 理论比 {delta/n/per_tok:.3f}", flush=True)
        del out, ids
        torch.cuda.empty_cache()

    slope = (rows[-1]["kv_mib"] - rows[0]["kv_mib"]) / (rows[-1]["tokens"] - rows[0]["tokens"])
    print(f"\n实测斜率 {slope*1024:.1f} KB/token（理论 {per_tok/1024:.1f}）")
    json.dump(dict(theory_kb=round(per_tok / 1024, 1), rows=rows,
                   measured_kb_per_token=round(slope * 1024, 1)),
              open(os.path.expanduser("~/apps/infer-research/kv_linear.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
