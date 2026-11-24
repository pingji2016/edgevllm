"""vLLM 在 sm_120 (RTX 5060 Ti) 上能否跑通 + 吞吐
对比对象：transformers 36.1 tok/s / llama.cpp 98.8 tok/s（同款 1.7B，bf16 vs F16）
"""
import os
import time

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
PROMPTS = ["用三句话解释什么是显存带宽，以及它为什么决定了大模型的解码速度。"] * 8

from vllm import LLM, SamplingParams

t0 = time.time()
llm = LLM(model=MD, dtype="bfloat16", max_model_len=2048,
          gpu_memory_utilization=0.45, enforce_eager=True, disable_log_stats=True)
print(f"模型加载 {time.time()-t0:.1f}s", flush=True)

sp = SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True)

# 单请求（与之前的口径一致）
llm.generate(PROMPTS[:1], sp)  # 预热
t0 = time.time()
out1 = llm.generate(PROMPTS[:1], sp)
dt1 = time.time() - t0
n1 = len(out1[0].outputs[0].token_ids)
print(f"单请求: {n1} token / {dt1:.2f}s = {n1/dt1:.1f} tok/s", flush=True)

# 8 并发（vLLM 的强项：连续批处理）
t0 = time.time()
outs = llm.generate(PROMPTS, sp)
dt8 = time.time() - t0
n8 = sum(len(o.outputs[0].token_ids) for o in outs)
print(f"8 并发: {n8} token / {dt8:.2f}s = {n8/dt8:.1f} tok/s "
      f"（单流等效 {n8/dt8/8:.1f}）", flush=True)
