"""RAG 第一步：把我们自己的 77 篇文档切片 + 向量化

切片策略：按 markdown 标题切，过长的段落再按 700 字滑窗
"""
import glob
import json
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

CORPUS = os.path.expanduser("~/apps/edgevllm/docs")
BGE = open(os.path.expanduser("~/models/.bge_done")).read().strip()
OUT = os.path.expanduser("~/apps/infer-research/rag")
MAX_CHARS = 700
OVERLAP = 100


def split_by_headings(text):
    """按 ## / ### 切分，保留所属一级标题作为上下文"""
    lines = text.splitlines()
    h1 = ""
    chunks, buf, head = [], [], ""
    for ln in lines:
        m = re.match(r"^(#{1,3})\s+(.*)", ln)
        if m and len(buf) > 0:
            chunks.append((head, "\n".join(buf)))
            buf = []
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            if level == 1:
                h1 = title
            head = f"{h1} / {title}" if h1 and title != h1 else (title or h1)
        buf.append(ln)
    if buf:
        chunks.append((head, "\n".join(buf)))

    out = []
    for head, body in chunks:
        body = body.strip()
        if len(body) < 40:
            continue
        if len(body) <= MAX_CHARS:
            out.append((head, body))
        else:
            step = MAX_CHARS - OVERLAP
            for i in range(0, len(body), step):
                piece = body[i:i + MAX_CHARS]
                if len(piece) >= 120:
                    out.append((head, piece))
                if i + MAX_CHARS >= len(body):
                    break
    return out


def main():
    files = sorted(glob.glob(os.path.join(CORPUS, "**/*.md"), recursive=True))
    print(f"文档 {len(files)} 篇")
    rows = []
    for f in files:
        rel = os.path.relpath(f, os.path.expanduser("~/apps/edgevllm"))
        text = open(f, encoding="utf-8", errors="ignore").read()
        for head, body in split_by_headings(text):
            rows.append(dict(file=rel, head=head, text=body))
    print(f"切片 {len(rows)} 个 | 平均 {np.mean([len(r['text']) for r in rows]):.0f} 字")

    tok = AutoTokenizer.from_pretrained(BGE)
    model = AutoModel.from_pretrained(BGE, dtype=torch.bfloat16).cuda().eval()

    def embed(texts, batch=16, is_query=False):
        vecs = []
        for i in range(0, len(texts), batch):
            b = texts[i:i + batch]
            if is_query:
                b = ["为这个句子生成表示以用于检索相关文章：" + t for t in b]
            enc = tok(b, padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to("cuda")
            with torch.inference_mode():
                out = model(**enc).last_hidden_state[:, 0]  # CLS pooling
            vecs.append(F.normalize(out.float(), dim=-1).cpu())
        return torch.cat(vecs).numpy()

    t0 = time.time()
    E = embed([r["text"] for r in rows])
    dt = time.time() - t0
    print(f"向量化 {E.shape} 用时 {dt:.1f}s（{len(rows)/dt:.0f} 切片/s）")

    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "emb.npy"), E)
    with open(os.path.join(OUT, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(dict(n_docs=len(files), n_chunks=len(rows), dim=E.shape[1],
                   embed_s=round(dt, 1)), open(os.path.join(OUT, "stats.json"), "w"), indent=2)
    print(f"索引已存 {OUT}")


if __name__ == "__main__":
    main()
