#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K3 本地 RAG（embed → 余弦 → rerank → LLM 全离线，跑在板子上）

用法:
    ./rag.py                          # 用已建的 ~/k3-ai/rag/index.json
    ./rag.py --build 报告.md          # 先把 Markdown 切块 + 建索引，再问
    ./rag.py "问题一" "问题二"          # 提问

依赖: SpaceMiT AI Gateway (:18790)，且已下好并 load:
    embed  bge-small-zh-v1.5-q4_k_m
    rerank qwen3-reranker-0.6b-q4_0   (可选，很慢，见下)
    LLM    qwen3.5-2b                 (网关当前加载的任意模型)

★ 实测结论（报告 §12.3）：rerank 单次 ≈11s/篇，是纯向量检索的 ~8000 倍。
   板上 RAG 建议 --no-rerank；只有对精度极敏感时才开。

    # 准备模型
    curl -s -X POST localhost:18790/v1/embed/models/bge-small-zh-v1.5-q4_k_m/download -d '{}' -H 'Content-Type: application/json'
    curl -s -X POST localhost:18790/v1/embed/models/load  -H 'Content-Type: application/json' -d '{"model":"bge-small-zh-v1.5-q4_k_m"}'
"""
import json
import math
import os
import re
import sys
import time
import urllib.request

G = os.environ.get("GW", "http://127.0.0.1:18790")
EMB = "bge-small-zh-v1.5-q4_k_m"
RER = "qwen3-reranker-0.6b-q4_0"
RAG_DIR = os.path.expanduser("~/k3-ai/rag")
INDEX = os.path.join(RAG_DIR, "index.json")


def post(path, obj, timeout=600):
    req = urllib.request.Request(
        G + path, data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def chunk_markdown(path, limit=120, target=300):
    """按空行切段，合并到约 target 字；去掉代码块围栏与表格分隔行。"""
    lines, incode = [], False
    for ln in open(path, encoding="utf-8"):
        if ln.strip().startswith("```"):
            incode = not incode
            continue
        if incode or re.match(r"^\s*\|[\s\-:|]+\|\s*$", ln):
            continue
        lines.append(ln)
    segs = [re.sub(r"\s+", " ", s).strip()
            for s in re.split(r"\n\s*\n", "\n".join(lines)) if s.strip()]
    chunks, cur = [], ""
    for s in segs:
        if len(cur) + len(s) < target:
            cur = (cur + "\n" + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) > 40][:limit]


def embed(texts):
    out = []
    for i in range(0, len(texts), 8):
        r = post("/v1/embed/embeddings", {"model": EMB, "input": texts[i:i + 8]})
        out += [d["embedding"] for d in r["data"]]
    return out


def build(md_path):
    kb = chunk_markdown(md_path)
    t0 = time.time()
    vecs = embed(kb)
    dt = time.time() - t0
    os.makedirs(RAG_DIR, exist_ok=True)
    json.dump({"kb": kb, "vecs": vecs}, open(INDEX, "w"))
    print("索引 %d 块，嵌入 %.2fs（%.1f 块/s）" % (len(kb), dt, len(kb) / dt))
    return kb, vecs


def load_index():
    if not os.path.exists(INDEX):
        sys.exit("没有索引，先用 --build 报告.md 建一个")
    d = json.load(open(INDEX))
    return d["kb"], d["vecs"]


def cos(a, b):
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return s / (na * nb + 1e-9)


def rerank(q, idx, kb):
    try:
        r = post("/v1/rerank/rerank",
                 {"model": RER, "query": q, "documents": [kb[i] for i in idx]},
                 timeout=300)
    except Exception as e:                                   # noqa: BLE001
        return None, str(e)[:100]
    out = []
    for it in r.get("results", []):
        j = it.get("index")
        if j is not None and j < len(idx):
            out.append((it.get("relevance_score", it.get("score")), idx[j]))
    return out, None


def ask(q, kb, vecs, use_rerank=True, topk=6):
    t = {}
    a = time.time()
    qv = post("/v1/embed/embeddings", {"model": EMB, "input": [q]})["data"][0]["embedding"]
    t["embed"] = time.time() - a

    a = time.time()
    scored = sorted([(cos(qv, v), i) for i, v in enumerate(vecs)], reverse=True)[:topk]
    cand = [i for _, i in scored]
    t["cosine"] = time.time() - a

    note = ""
    if use_rerank:
        a = time.time()
        rr, err = rerank(q, cand, kb)
        t["rerank"] = time.time() - a
        if rr:
            cand = [i for _, i in sorted(rr, key=lambda x: -(x[0] or -1))]
        else:
            note = "（rerank 失败: %s）" % err

    ctx = "\n\n".join("[%d] %s" % (n + 1, kb[i]) for n, i in enumerate(cand[:3]))
    a = time.time()
    r = post("/v1/llm/chat/completions", {
        "messages": [
            {"role": "system", "content": "你是 K3 板子上的本地助手。只根据给定资料回答，用中文简洁分点，"
                                          "不要编造资料之外的内容。"},
            {"role": "user", "content": "资料：\n%s\n\n问题：%s" % (ctx, q)}],
        "temperature": 0.3, "max_tokens": 512})
    t["llm"] = time.time() - a
    tm = r.get("timings", {})
    return r["choices"][0]["message"]["content"], cand[:3], t, tm, note


def main(argv):
    build_path = None
    if "--build" in argv:
        i = argv.index("--build")
        build_path = argv[i + 1]
        del argv[i:i + 2]
    use_rerank = "--no-rerank" not in argv
    argv = [a for a in argv if a != "--no-rerank"]

    kb, vecs = build(build_path) if build_path else load_index()
    qs = argv or ["A100 核适合跑什么？为什么跑 llama.cpp 反而慢？"]
    for q in qs:
        print("\n=============== Q: %s" % q)
        ans, cand, t, tm, note = ask(q, kb, vecs, use_rerank=use_rerank)
        print("检索 top3 块号: %s %s" % (cand, note))
        print("耗时: embed %.2fs + cosine %.3fs + rerank %.2fs + llm %.2fs"
              % (t["embed"], t["cosine"], t.get("rerank", 0), t["llm"]))
        if tm:
            print("LLM: %s tok, %.2f tok/s"
                  % (tm.get("predicted_n"), tm.get("predicted_per_second") or 0))
        print("回答:\n%s" % ans)


if __name__ == "__main__":
    main(sys.argv[1:])
