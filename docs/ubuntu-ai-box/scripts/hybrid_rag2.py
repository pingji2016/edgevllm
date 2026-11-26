"""补完整消融：R@1/3/5 × 5 种组合
  dense      bge-m3 稠密
  dense_rr   稠密 + reranker
  bm25       关键词
  rrf        稠密+BM25 融合
  rrf_rr     融合 + reranker
"""
import json
import os

import numpy as np
import torch

RAG = os.path.expanduser("~/apps/infer-research/rag")
RR = open(os.path.expanduser("~/models/.rr_done")).read().strip()
BGE3 = os.path.expanduser("~/models/bge-m3")
OUT = os.path.join(RAG, "eval_hybrid2.json")

QAS = [
    ("CUDA Graph 相比 eager 在单流下提升多少？", "e2e-and-opt"),
    ("Kokoro 跑在 K3 上的实时率（RTF）是多少？", "tts-kokoro"),
    ("量化七档里 Q4_K_M 的 decode 速度是多少 tok/s？", "inference-acceleration"),
    ("minimind 预训练用了多长时间？", "minimind-training"),
    ("KV cache 每 token 占多少 KB？", "inference-acceleration"),
    ("FlexAttention 不编译时会比 SDPA 慢多少倍？", "inference-acceleration"),
    ("Wanda 剪枝 50% 时随机剪枝的 PPL 是多少？", "inference-acceleration"),
    ("DeepSeek-OCR 处理一页 A4 需要多长时间？", "deepseek-ocr"),
    ("MiniCPM-o 在 Linux 上编译报什么错？", "minicpm-o"),
    ("K3 上 ngram 投机解码的加速比是多少？", "inference-acceleration"),
    ("vLLM 32 并发能达到多少 tok/s？", "e2e-and-opt"),
    ("SDXL LoRA 训练的单步耗时是多少？", "roofline-and-lora"),
]


def main():
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    chunks = [json.loads(l) for l in open(os.path.join(RAG, "chunks.jsonl"), encoding="utf-8")]
    E = np.load(os.path.join(RAG, "emb_m3.npy"))
    bt = AutoTokenizer.from_pretrained(BGE3)
    bm = AutoModel.from_pretrained(BGE3, dtype=torch.bfloat16).cuda().eval()
    rt = AutoTokenizer.from_pretrained(RR)
    rm = AutoModelForSequenceClassification.from_pretrained(RR, dtype=torch.bfloat16).cuda().eval()

    import torch.nn.functional as F

    def qv(q):
        enc = bt(q, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        with torch.inference_mode():
            h = bm(**enc).last_hidden_state[:, 0]
        return F.normalize(h.float(), dim=-1).cpu().numpy()[0]

    def rerank(q, idx):
        enc = rt([[q, chunks[i]["text"]] for i in idx], padding=True, truncation=True,
                 max_length=512, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            s = rm(**enc).logits.view(-1).float().cpu().numpy()
        return [idx[o] for o in np.argsort(-s)]

    # 从 hybrid_rag.py 复用 BM25
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hybrid_rag import BM25, rrf

    bm25 = BM25([c["text"] for c in chunks])
    res = {k: {1: 0, 3: 0, 5: 0} for k in ("dense", "dense_rr", "bm25", "rrf", "rrf_rr")}
    n = len(QAS)

    for q, expect in QAS:
        ds = E @ qv(q)
        d_idx = np.argsort(-ds)[:50]
        bs = bm25.score(q)
        b_idx = np.argsort(-bs)[:50]
        rs = rrf([(d_idx, ds), (b_idx, bs)], len(chunks))
        r_idx = np.argsort(-rs)[:50]

        lists = {
            "dense": list(d_idx[:5]),
            "bm25": list(b_idx[:5]),
            "rrf": list(r_idx[:5]),
            "dense_rr": rerank(q, list(d_idx[:20]))[:5],
            "rrf_rr": rerank(q, list(r_idx[:20]))[:5],
        }
        for k, idx in lists.items():
            files = [chunks[i]["file"] for i in idx]
            pos = next((i for i, f in enumerate(files) if expect in f), None)
            if pos is not None:
                for cut in (1, 3, 5):
                    if pos < cut:
                        res[k][cut] += 1

    print(f"{'方案':10s} {'R@1':>7s} {'R@3':>7s} {'R@5':>7s}")
    for k in ("dense", "dense_rr", "bm25", "rrf", "rrf_rr"):
        r = res[k]
        print(f"{k:10s} {r[1]:4d}/{n:<2d} {r[3]:4d}/{n:<2d} {r[5]:4d}/{n:<2d}  "
              f"({r[1]/n*100:.0f}% / {r[3]/n*100:.0f}% / {r[5]/n*100:.0f}%)")
    print("\n基线（bge-small-zh）：dense 42%/67%/67% | dense+rr 58%/75%/75%")

    json.dump({k: {c: v / n for c, v in r.items()} for k, r in res.items()},
              open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
