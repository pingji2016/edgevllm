"""B. vLLM 单实例 + 3 个不同 SamplingParams（真并行批处理）"""
import time

from vllm import LLM, SamplingParams

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master"
PROMPT = "用三句话解释什么是 KV cache。"

llm = LLM(model=MD, dtype="bfloat16", max_model_len=2048,
          gpu_memory_utilization=0.35, disable_log_stats=True)
print("  vLLM 就绪")

sps = [SamplingParams(max_tokens=96, temperature=0.2, top_p=0.9, seed=1),
       SamplingParams(max_tokens=96, temperature=0.8, top_p=0.95, seed=2),
       SamplingParams(max_tokens=96, temperature=1.3, top_p=0.8, seed=3)]

llm.generate([PROMPT], sps[0])          # 预热
t0 = time.time()
outs = llm.generate([PROMPT] * 3, sps)  # 3 条请求，各自不同参数
torch_sync = time.time() - t0
n = sum(len(o.outputs[0].token_ids) for o in outs)
print(f"  3 并发：{n} tok / {torch_sync:.2f}s = {n/torch_sync:.0f} tok/s 总")
for i, o in enumerate(outs):
    print(f"    agent{i+1}: {len(o.outputs[0].token_ids)} tok | {o.outputs[0].text[:40]!r}")
