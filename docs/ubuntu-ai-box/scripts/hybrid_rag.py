"""RAG 检索升级：bge-m3 稠密 + BM25 关键词 + RRF 融合 + reranker

对照基线（bge-small-zh，1726 切片）：
  纯向量 R@1 42% / R@3 67%
  向量+reranker R@1 58% / R@3 75%

本脚本输出四行消融：
  A bge-m3 稠密
  B BM25
  C RRF 融合（A+B）
  D 融合 + reranker
"""
import json
import math
import os
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

RAG = os.path.expanduser("~/apps/infer-research/rag")
RR = open(os.path.expanduser("~/models/.rr_done")).read().strip()
BGE3 = os.path.expanduser("~/models/bge-m3")
OUT = os.path.join(RAG, "eval_hybrid.json")

QAS = [
    ("CUDA Graph 相比 eager 在单流下提升多少？", "e2e-and-opt", ["83.8", "84"]),
    ("Kokoro 跑在 K3 上的实时率（RTF）是多少？", "tts-kokoro", ["2.08", "0.48"]),
    ("量化七档里 Q4_K_M 的 decode 速度是多少 tok/s？", "inference-acceleration", ["221.8", "222"]),
    ("minimind 预训练用了多长时间？", "minimind-training", ["2.8"]),
    ("KV cache 每 token 占多少 KB？", "inference-acceleration", ["116"]),
    ("FlexAttention 不编译时会比 SDPA 慢多少倍？", "inference-acceleration", ["20", "30"]),
    ("Wanda 剪枝 50% 时随机剪枝的 PPL 是多少？", "inference-acceleration", ["1780737", "1780"]),
    ("DeepSeek-OCR 处理一页 A4 需要多长时间？", "deepseek-ocr", ["7.26", "7.2"]),
    ("MiniCPM-o 在 Linux 上编译报什么错？", "minicpm-o", ["F_NOCACHE"]),
    ("K3 上 ngram 投机解码的加速比是多少？", "inference-acceleration", ["1.29", "2.5"]),
    ("vLLM 32 并发能达到多少 tok/s？", "e2e-and-opt", ["3020", "3022", "4409"]),
    ("SDXL LoRA 训练的单步耗时是多少？", "roofline-and-lora", ["676"]),
]


def toks(s):
    s = s.lower()
    latin = re.findall(r"[a-z0-9_]+", s)
    zh = re.findall(r"[\u4e00-\u9fff]", s)
    bigrams = [zh[i] + zh[i + 1] for i in range(len(zh) - 1)]
    return latin + bigrams + zh


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [toks(d) for d in docs]
        self.N = len(self.docs)
        self.avg = sum(len(d) for d in self.docs) / max(1, self.N)
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]
        self.len = [len(d) for d in self.docs]

    def score(self, q):
        qt = toks(q)
        out = np.zeros(self.N)
        for t in set(qt):
            idf = self.idf.get(t, 0.0)
            if idf == 0:
                continue
            for i in range(self.N):
                f = self.tf[i].get(t, 0)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


def rrf(lists, n, k=60):
    score = np.zeros(n)
    for idx, _ in lists:
        for rank, i in enumerate(idx):
            score[i] += 1.0 / (k + rank + 1)
    return score


def main():
    from modelscope import snapshot_download
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    if not os.path.exists(os.path.join(BGE3, "config.json")):
        print("  下载 bge-m3…", flush=True)
        os.makedirs(BGE3, exist_ok=True)
        snapshot_download("BAAI/bge-m3", local_dir=BGE3,
                          ignore_file_pattern=[r"onnx/.*", r"*.onnx"])
    print(f"  bge-m3: {BGE3}", flush=True)

    chunks = [json.loads(l) for l in open(os.path.join(RAG, "chunks.jsonl"), encoding="utf-8")]
    print(f"  切片 {len(chunks)}", flush=True)

    mtok = AutoTokenizer.from_pretrained(BGE3)
    mmodel = AutoModel.from_pretrained(BGE3, dtype=torch.bfloat16).cuda().eval()

    def embed(texts, batch=16):
        vs = []
        for i in range(0, len(texts), batch):
            enc = mtok(texts[i:i + batch], padding=True, truncation=True,
                       max_length=1024, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                h = mmodel(**enc).last_hidden_state[:, 0]
            vs.append(F.normalize(h.float(), dim=-1).cpu())
        return torch.cat(vs).numpy()

    cache = os.path.join(RAG, "emb_m3.npy")
    t0 = time.time()
    if os.path.exists(cache):
        E = np.load(cache)
        dt = 0.0
        print(f"  复用稠密索引 {E.shape}", flush=True)
    else:
        E = embed([c["text"] for c in chunks])
        np.save(cache, E)
    print(f"  稠密索引 {E.shape} 用时 {dt:.1f}s", flush=True)

    bm = BM25([c["text"] for c in chunks])
    rtok = AutoTokenizer.from_pretrained(RR)
    rmodel = AutoModelForSequenceClassification.from_pretrained(
        RR, dtype=torch.bfloat16).cuda().eval()

    ok = dict(dense=0, bm25=0, rrf=0, rrf_rr=0)
    lat = dict(dense=[], bm25=[], rrf=[], rrf_rr=[])
    rows = []

    for q, expect, kws in QAS:
        t0 = time.time()
        qv = embed([q])[0]
        ds = E @ qv
        d_idx = np.argsort(-ds)[:50]
        lat["dense"].append((time.time() - t0) * 1000)

        t0 = time.time()
        bs = bm.score(q)
        b_idx = np.argsort(-bs)[:50]
        lat["bm25"].append((time.time() - t0) * 1000)

        t0 = time.time()
        rs = rrf([(d_idx, ds), (b_idx, bs)], len(chunks))
        r_idx = np.argsort(-rs)[:50]
        lat["rrf"].append((time.time() - t0) * 1000)

        def files(idx):
            return [chunks[i]["file"] for i in idx[:5]]

        f_d = files(d_idx)
        f_b = files(b_idx)
        f_r = files(r_idx)
        hit_d = next((i for i, f in enumerate(f_d) if expect in f), None)
        hit_b = next((i for i, f in enumerate(f_b) if expect in f), None)
        hit_r = next((i for i, f in enumerate(f_r) if expect in f), None)
        ok["dense"] += hit_d == 0
        ok["bm25"] += hit_b == 0
        ok["rrf"] += hit_r == 0

        t0 = time.time()
        cand = list(r_idx[:20])
        enc = rtok([[q, chunks[i]["text"]] for i in cand], padding=True,
                   truncation=True, max_length=512, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            s = rmodel(**enc).logits.view(-1).float().cpu().numpy()
        rr_idx = [cand[o] for o in np.argsort(-s)]
        lat["rrf_rr"].append((time.time() - t0) * 1000)
        f_rr = files(rr_idx)
        hit_rr = next((i for i, f in enumerate(f_rr) if expect in f), None)
        ok["rrf_rr"] += hit_rr == 0

        rows.append(dict(q=q, dense=hit_d, bm25=hit_b, rrf=hit_r, rrf_rr=hit_rr,
                         top1_dense=f_d[0], top1_rrf=f_r[0], top1_final=f_rr[0]))
        print(f"  d{hit_d if hit_d is not None else '-':>2} "
              f"b{hit_b if hit_b is not None else '-':>2} "
              f"r{hit_r if hit_r is not None else '-':>2} "
              f"R{hit_rr if hit_rr is not None else '-':>2}  {q[:26]}", flush=True)

    n = len(QAS)
    print(f"\n=== Recall@1（{n} 题）===")
    for k in ("dense", "bm25", "rrf", "rrf_rr"):
        print(f"  {k:8s} {ok[k]}/{n} = {ok[k]/n*100:.0f}%  | "
              f"耗时 {np.mean(lat[k]):.0f} ms")
    print(f"\n  基线对照：bge-small 纯向量 42% | bge-small+reranker 58%")

    json.dump(dict(ok={k: ok[k] / n for k in ok},
                   lat={k: float(np.mean(v)) for k, v in lat.items()},
                   rows=rows), open(OUT, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
