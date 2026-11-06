import os
import sys
import time

sys.path.insert(0, "/tmp/nano-vllm")
import torch

if os.environ.get("THREADS"):
    torch.set_num_threads(int(os.environ["THREADS"]))

P = "/home/pingji/huggingface/Qwen3-0.6B"
NT = int(os.environ.get("NT", "48"))
BSLIST = [int(x) for x in os.environ.get("BSLIST", "1,2,4").split(",")]
MAXLEN = int(os.environ.get("MAXLEN", "1024"))

from nanovllm import LLM, SamplingParams

t0 = time.time()
llm = LLM(P, enforce_eager=True, max_model_len=MAXLEN, max_num_batched_tokens=512,
          num_kvcache_blocks=int(os.environ.get("KVB", "24")))
load_s = time.time() - t0
tok = llm.tokenizer
rss = int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) // 1024
print(f"[load] {load_s:.1f}s  RSS={rss}MB  threads={torch.get_num_threads()}", flush=True)

QS = ["用两句话介绍一下 RISC-V 架构。", "What is the capital of France?",
      "写一个 Python 函数判断回文数。", "Explain what a KV cache is in one sentence."]
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                   add_generation_prompt=True) for q in QS]


def run(bs, ntok):
    ps = prompts[:bs]
    sp = [SamplingParams(temperature=0.7, max_tokens=ntok, ignore_eos=True) for _ in ps]
    t = time.time()
    out = llm.generate(ps, sp, use_tqdm=False)
    return time.time() - t, out


run(1, 2)  # warmup

for bs in BSLIST:
    tp, _ = run(bs, 1)
    td, _ = run(bs, NT + 1)
    dec = td - tp
    n_dec = bs * NT
    print(f"[bs={bs}] prefill+1step={tp*1000:.0f} ms | {NT} tok x{bs} = {dec:.2f}s | "
          f"decode={n_dec/dec:.2f} tok/s | {dec/n_dec*1000:.0f} ms/token", flush=True)

if os.environ.get("PREFILL"):
    long_q = "请详细说明以下主题：" + "人工智能推理系统需要在算力、内存带宽和延迟之间取得平衡。" * 20
    lp = [tok.apply_chat_template([{"role": "user", "content": long_q}], tokenize=False,
                                  add_generation_prompt=True)]
    n_prompt = len(tok.encode(lp[0]))
    t = time.time()
    llm.generate(lp, [SamplingParams(temperature=0.7, max_tokens=1, ignore_eos=True)], use_tqdm=False)
    tp = time.time() - t
    print(f"[prefill] {n_prompt} tok -> {tp*1000:.0f} ms | {n_prompt/tp:.0f} tok/s", flush=True)

if os.environ.get("DEMO"):
    _, out = run(1, 32)
    print("Q:", QS[0], flush=True)
    print("A:", out[0]["text"][:200].replace("\n", " "), flush=True)

llm.exit()
