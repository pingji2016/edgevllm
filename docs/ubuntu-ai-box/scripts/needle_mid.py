"""长上下文 needle-in-a-haystack：Qwen3-1.7B，4K/8K/16K/32K

测三件事：
  1. 检索准确率（needle 放在 10%/50%/90% 深度）
  2. prefill 延迟与吞吐（batch=1，固定输出 8 token）
  3. KV 显存占用（从 vLLM 日志 + 理论值对照）
"""
import glob
import json
import os
import random
import re
import time

from vllm import LLM, SamplingParams

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
DOCS = os.path.expanduser("~/apps/edgevllm/docs/ubuntu-ai-box")
OUT = os.path.expanduser("~/apps/infer-engines/needle_mid.json")
NEEDLE = "【重要记录】机房代号“蓝鲸”的访问密码是 741928，请勿外传。"
QUERY = "机房代号“蓝鲸”的访问密码是多少？只回答数字。"
NUM = re.compile(r"741928")

LENS = [20480, 24576, 28672]
DEPTHS = [0.1, 0.5, 0.9]
SEEDS = [0, 1]


def load_pool(tok):
    """把我们的文档拼成一个长文本池，并切成段落"""
    text = []
    for f in sorted(glob.glob(os.path.join(DOCS, "*.md"))):
        text.append(open(f, encoding="utf-8", errors="ignore").read())
    blob = "\n\n".join(text)
    # 按段落切，过滤过短的
    paras = [p.strip() for p in blob.split("\n\n") if len(p.strip()) > 80]
    return paras


def build(tok, paras, target_len, depth, seed):
    """构造恰好 target_len 个 token 的 haystack，needle 插在 depth 处"""
    rng = random.Random(seed)
    # 先估算：中文约 1 token / 1.5 字 → 先取一大块再裁
    pieces, ntok = [], 0
    while ntok < target_len:
        pieces.append(paras[rng.randrange(len(paras))])
        ntok = len(tok("\n\n".join(pieces)).input_ids)
    hay = "\n\n".join(pieces)
    ids = tok(hay).input_ids
    # 留 96 token 给 needle + 问题 + 模板
    hay = tok.decode(ids[:max(1, target_len - 96)], skip_special_tokens=True)
    # 找插入点：depth 位置附近的换行
    cut = max(1, int(len(hay) * depth))
    pos = hay.find("\n", cut)
    if pos < 0:
        pos = cut
    hay = hay[:pos] + "\n\n" + NEEDLE + "\n\n" + hay[pos:]

    prompt = (f"<|im_start|>system\n你是一个精确的文档问答助手。只输出答案，不要解释。<|im_end|>\n"
              f"<|im_start|>user\n{hay}\n\n问题：{QUERY}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    return prompt


def main():
    t0 = time.time()
    llm = LLM(model=MD, dtype="bfloat16", max_model_len=33280,
              gpu_memory_utilization=0.80, disable_log_stats=True)
    print(f"  引擎就绪 {time.time()-t0:.0f}s", flush=True)
    tok = llm.get_tokenizer()
    paras = load_pool(tok)
    print(f"  语料段落 {len(paras)} 个", flush=True)

    sp = SamplingParams(max_tokens=12, temperature=0.0)
    rows = []
    for L in LENS:
        for d in DEPTHS:
            ok = 0
            times = []
            ntok = 0
            for s in SEEDS:
                prompt = build(tok, paras, L, d, s)
                if prompt is None:
                    print(f"  L={L} d={d} 构造失败", flush=True)
                    continue
                ntok = len(tok(prompt).input_ids)
                t0 = time.time()
                out = llm.generate([prompt], sp)[0]
                dt = time.time() - t0
                txt = out.outputs[0].text
                hit = bool(NUM.search(txt))
                ok += hit
                times.append(dt)
                if s == 0:
                    sample = txt.strip()[:30]
            acc = ok / len(SEEDS)
            if not times:
                print(f"  {L} tok | 深度 {int(d*100)}% | 全部构造失败，跳过", flush=True)
                continue
            med = sorted(times)[len(times) // 2]
            rows.append(dict(len=L, depth=d, ntok=ntok, acc=acc,
                             med_s=round(med, 3),
                             prefill_tps=round(ntok / med) if med else None,
                             sample=sample))
            print(f"  {L:6d} tok | 深度 {int(d*100):2d}% | 实际 {ntok:6d} | "
                  f"命中 {ok}/{len(SEEDS)} | {med:.2f}s | prefill {ntok/med:6.0f} tok/s | "
                  f"{sample!r}", flush=True)

    json.dump(rows, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"\n已存 {OUT}")


if __name__ == "__main__":
    main()
