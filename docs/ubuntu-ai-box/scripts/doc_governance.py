"""文档治理：用结构信息对抗"主题稀释"

四个配置（同一批 12 题，bge-m3 + reranker）：
  A 扁平切片            基线（75/83/92）
  B 切片前加文档标题     让切片带"出身"
  C 文档级路由           先选 top-3 文档，再在文档内检索切片
  D B + C               标题前缀 + 文档路由
"""
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

RAG = os.path.expanduser("~/apps/infer-research/rag")
DOCS = os.path.expanduser("~/apps/edgevllm/docs/ubuntu-ai-box")
BGE3 = os.path.expanduser("~/models/bge-m3")
RR = open(os.path.expanduser("~/models/.rr_done")).read().strip()
OUT = os.path.join(RAG, "eval_governance.json")

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


def doc_summary(path):
    lines = open(path, encoding="utf-8", errors="ignore").read().splitlines()
    title = next((l.lstrip("# ").strip() for l in lines if l.startswith("# ")), os.path.basename(path))
    heads = [l.lstrip("# ").strip() for l in lines if l.startswith("##")][:12]
    body = " ".join(l.strip() for l in lines if l.strip() and not l.startswith("#"))[:600]
    return f"[{title}] " + " / ".join(heads) + "。" + body


def main():
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    chunks = [json.loads(l) for l in open(os.path.join(RAG, "chunks.jsonl"), encoding="utf-8")]
    E = np.load(os.path.join(RAG, "emb_m3.npy"))
    files = sorted({c["file"] for c in chunks})
    print(f"  切片 {len(chunks)} | 文档 {len(files)}", flush=True)

    bt = AutoTokenizer.from_pretrained(BGE3)
    bm = AutoModel.from_pretrained(BGE3, dtype=torch.bfloat16).cuda().eval()
    rt = AutoTokenizer.from_pretrained(RR)
    rm = AutoModelForSequenceClassification.from_pretrained(
        RR, dtype=torch.bfloat16).cuda().eval()

    def embed(texts, batch=16):
        out = []
        for i in range(0, len(texts), batch):
            enc = bt(texts[i:i + batch], padding=True, truncation=True,
                     max_length=1024, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                h = bm(**enc).last_hidden_state[:, 0]
            out.append(F.normalize(h.float(), dim=-1).cpu())
        return torch.cat(out).numpy()

    # 文档级摘要向量
    abspath = {os.path.basename(f): os.path.join(os.path.expanduser("~/apps/edgevllm"), f) for f in files}
    summaries = [doc_summary(abspath[os.path.basename(f)]) for f in files]
    Ed = embed(summaries)
    print(f"  文档摘要向量 {Ed.shape}", flush=True)

    # 标题前缀的切片向量
    titled = [f"[{os.path.basename(c['file'])}] {c['head']}\n{c['text']}" for c in chunks]
    Et = embed(titled)
    print(f"  标题前缀向量 {Et.shape}", flush=True)

    # 文档 -> 该文档的切片下标
    doc2idx = {}
    for i, c in enumerate(chunks):
        doc2idx.setdefault(c["file"], []).append(i)

    def rerank(q, idx, k=20):
        if not idx:
            return []
        enc = rt([[q, chunks[i]["text"]] for i in idx[:k]], padding=True, truncation=True,
                 max_length=512, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            s = rm(**enc).logits.view(-1).float().cpu().numpy()
        return [idx[o] for o in np.argsort(-s)]

    def hit(expect, idx):
        fs = [chunks[i]["file"] for i in idx[:5]]
        return next((i for i, f in enumerate(fs) if expect in f), None)

    res = {k: {1: 0, 3: 0, 5: 0} for k in
           ("A_flat", "B_titled", "C_route", "D_both", "E_union", "F_union_titled")}
    rows = []
    n = len(QAS)

    for q, expect in QAS:
        qv = embed([q])[0]

        # A 扁平
        a_idx = rerank(q, list(np.argsort(-(E @ qv))[:50]))
        # B 标题前缀
        b_idx = rerank(q, list(np.argsort(-(Et @ qv))[:50]))
        # C 文档路由：先 top-3 文档，再在文档内取 top-50
        d_score = Ed @ qv
        top_docs = [files[o] for o in np.argsort(-d_score)[:3]]
        cand = [i for d in top_docs for i in doc2idx[d]]
        c_idx = rerank(q, sorted(cand, key=lambda i: -(E[i] @ qv))[:50])
        # D 两者结合：路由 + 标题前缀打分
        d_idx = rerank(q, sorted(cand, key=lambda i: -(Et[i] @ qv))[:50])

        # E: 扁平 top-20 ∪ 路由(top-5 文档) 的候选，再重排
        d_score5 = Ed @ qv
        top5 = [files[o] for o in np.argsort(-d_score5)[:5]]
        cand5 = [i for dd in top5 for i in doc2idx[dd]]
        flat20 = list(np.argsort(-(E @ qv))[:20])
        route20 = sorted(cand5, key=lambda i: -(E[i] @ qv))[:20]
        union = list(dict.fromkeys(flat20 + route20))
        e_idx = rerank(q, union, k=40)
        # F: 扁平 ∪ 标题前缀 top-20
        t20 = list(np.argsort(-(Et @ qv))[:20])
        union2 = list(dict.fromkeys(flat20 + t20))
        f_idx = rerank(q, union2, k=40)

        for tag, idx in (("A_flat", a_idx), ("B_titled", b_idx),
                         ("C_route", c_idx), ("D_both", d_idx),
                         ("E_union", e_idx), ("F_union_titled", f_idx)):
            p = hit(expect, idx)
            if p is not None:
                for cut in (1, 3, 5):
                    if p < cut:
                        res[tag][cut] += 1
        rows.append(dict(q=q, expect=expect,
                         A=hit(expect, a_idx), B=hit(expect, b_idx),
                         C=hit(expect, c_idx), D=hit(expect, d_idx),
                         E=hit(expect, e_idx), F=hit(expect, f_idx)))
        r_ = rows[-1]
        print(f"  A{r_['A']} B{r_['B']} C{r_['C']} D{r_['D']} E{r_['E']} F{r_['F']}  {q[:26]}",
              flush=True)

    print(f"\n{'配置':10s} {'R@1':>8s} {'R@3':>8s} {'R@5':>8s}")
    for k in ("A_flat", "B_titled", "C_route", "D_both", "E_union", "F_union_titled"):
        r = res[k]
        print(f"{k:10s} {r[1]/n*100:7.0f}% {r[3]/n*100:7.0f}% {r[5]/n*100:7.0f}%")
    print(f"\n  基线（bge-m3 + rerank，上一轮）: 75% / 83% / 92%")

    json.dump(dict(res={k: {c: v / n for c, v in vv.items()} for k, vv in res.items()},
                   rows=rows), open(OUT, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
