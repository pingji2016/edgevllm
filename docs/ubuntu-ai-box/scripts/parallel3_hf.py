"""可行性实测：3 个 0.6B agent 并行

A. HF 三个独立实例：显存 + 各自不同 temperature/top_p/seed
B. vLLM 单实例 + 3 个不同 SamplingParams（真正的 continuous batching）
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master"
PROMPT = "用三句话解释什么是 KV cache。"

tok = AutoTokenizer.from_pretrained(MD)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

print("=== A. HF 三个独立实例 ===")
models = []
for i in range(3):
    m = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16,
                                             device_map={"": 0}).eval()
    models.append(m)
    print(f"  第 {i+1} 个加载后：已分配 {torch.cuda.memory_allocated()/2**30:.2f} GB | "
          f"保留 {torch.cuda.memory_reserved()/2**30:.2f} GB")

ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
cfgs = [(0.2, 0.9, 1), (0.8, 0.95, 2), (1.3, 0.8, 3)]
for i, (m, (T, p, seed)) in enumerate(zip(models, cfgs)):
    torch.manual_seed(seed)
    with torch.inference_mode():
        t0 = time.time()
        g = m.generate(ids, max_new_tokens=96, do_sample=True,
                       temperature=T, top_p=p, pad_token_id=tok.pad_token_id,
                       eos_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        dt = time.time() - t0
    n = g.size(1) - ids.size(1)
    print(f"  agent{i+1} T={T} top_p={p} seed={seed}: {n} tok / {dt:.2f}s "
          f"= {n/dt:.0f} tok/s")
print(f"  峰值显存 {torch.cuda.max_memory_allocated()/2**30:.2f} GB")
