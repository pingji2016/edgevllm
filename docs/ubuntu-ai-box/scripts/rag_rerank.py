"""RAG + Reranker：向量召回 top-20 → 交叉编码器重排 → top-3 生成

对比基线（纯向量 top-5）：R@1 42% / R@3 67% / R@5 67%
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

RAG = os.path.expanduser("~/apps/infer-research/rag")
BGE = open(os.path.expanduser("~/models/.bge_done")).read().strip()
RR = open(os.path.expanduser("~/models/.rr_done")).read().strip()
MD17 = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
OUT = os.path.join(RAG, "eval_rerank.json")
TOPK_RECALL = 20
TOPK_CTX = 3

QAS = [
    ("CUDA Graph 相比 eager 在单流下提升多少？", "e2e-and-opt", ["83.8", "84%"]),
    ("Kokoro 跑在 K3 上的实时率（RTF）是多少？", "tts-kokoro", ["2.08", "0.48"]),
    ("量化七档里 Q4_K_M 的 decode 速度是多少 tok/s？", "inference-acceleration", ["221.8", "222"]),
    ("minimind 预训练用了多长时间？", "minimind-training", ["2.8"]),
    ("KV cache 每 token 占多少 KB？", "inference-acceleration", ["116"]),
    ("FlexAttention 不编译时会比 SDPA 慢多少倍？", "inference-acceleration", ["20", "30"]),
    ("Wanda 剪枝 50% 时随机剪枝的 PPL 是多少？", "inference-acceleration", ["1780737", "1780"]),
    ("DeepSeek-OCR 处理一页 A4 需要多长时间？", "deepseek-ocr", ["7.26", "7.2"]),
    ("MiniCPM-o 在 Linux 上编译报什么错？", "minicpm-o", ["F_NOCACHE"]),
    ("K3 上 ngram 投机解码的加速比是多少？", "inference-acceleration", ["1.29", "2.5"]),
    ("vLLM 32 并发能达到多少 tok/s？", "e2e-and-opt", ["3020", "3,020"]),
    ("SDXL LoRA 训练的单步耗时是多少？", "roofline-and-lora", ["676"]),
]


def norm(s):
    return s.replace(",", "").replace("，", "")


def main():
    from transformers import (AutoModel, AutoModelForCausalLM,
                              AutoModelForSequenceClassification, AutoTokenizer)

    chunks = [json.loads(l) for l in open(os.path.join(RAG, "chunks.jsonl"), encoding="utf-8")]
    E = np.load(os.path.join(RAG, "emb.npy"))

    btok = AutoTokenizer.from_pretrained(BGE)
    bmodel = AutoModel.from_pretrained(BGE, dtype=torch.bfloat16).cuda().eval()

    print("加载 reranker…", flush=True)
    t0 = time.time()
    rtok = AutoTokenizer.from_pretrained(RR)
    rmodel = AutoModelForSequenceClassification.from_pretrained(
        RR, dtype=torch.bfloat16).cuda().eval()
    print(f"  reranker 就绪 {time.time()-t0:.1f}s | 参数 "
          f"{sum(p.numel() for p in rmodel.parameters())/1e6:.0f} M")

    gtok = AutoTokenizer.from_pretrained(MD17)
    if gtok.pad_token is None:
        gtok.pad_token = gtok.eos_token
    gmodel = AutoModelForCausalLM.from_pretrained(
        MD17, dtype=torch.bfloat16, device_map={"": 0}).eval()

    def qvec(q):
        enc = btok("为这个句子生成表示以用于检索相关文章：" + q, return_tensors="pt",
                   truncation=True, max_length=512).to("cuda")
        with torch.inference_mode():
            v = bmodel(**enc).last_hidden_state[:, 0]
        return F.normalize(v.float(), dim=-1).cpu().numpy()[0]

    def rerank(q, idxs):
        pairs = [[q, chunks[i]["text"]] for i in idxs]
        enc = rtok(pairs, padding=True, truncation=True, max_length=512,
                   return_tensors="pt").to("cuda")
        with torch.inference_mode():
            s = rmodel(**enc).logits.view(-1).float().cpu().numpy()
        order = np.argsort(-s)
        return [idxs[o] for o in order], s[order]

    rows, r1 = [], 0
    r3 = r5 = 0
    rr_ms, gen_ms = [], []
    for q, expect, kws in QAS:
        t0 = time.time()
        qv = qvec(q)
        cand = np.argsort(-(E @ qv))[:TOPK_RECALL]
        t_ret = (time.time() - t0) * 1000

        t0 = time.time()
        ranked, _ = rerank(q, cand)
        t_rr = (time.time() - t0) * 1000

        files = [chunks[i]["file"] for i in ranked[:5]]
        pos = next((i for i, f in enumerate(files) if expect in f), None)
        r1 += int(pos == 0)
        r3 += int(pos is not None and pos < 3)
        r5 += int(pos is not None)

        ctx = "\n\n".join(f"[{chunks[i]['file']}]\n{chunks[i]['text'][:600]}"
                          for i in ranked[:TOPK_CTX])
        msgs = [{"role": "system", "content": "根据资料回答问题，只答资料里的内容。"},
                {"role": "user", "content": f"【资料】\n{ctx}\n\n【问题】{q}"}]
        prompt = gtok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
        enc = gtok(prompt, return_tensors="pt", truncation=True, max_length=3072).to("cuda")
        n_in = enc["input_ids"].size(1)
        with torch.inference_mode():
            t0 = time.time()
            out = gmodel.generate(**enc, max_new_tokens=64, do_sample=False,
                                  pad_token_id=gtok.pad_token_id)
            torch.cuda.synchronize()
            t_gen = (time.time() - t0) * 1000
        ans = gtok.decode(out[0, n_in:], skip_special_tokens=True).strip()
        kw_ok = any(k in norm(ans) for k in kws)

        rr_ms.append(t_rr); gen_ms.append(t_gen)
        rows.append(dict(q=q, expect=expect, hit_pos=pos, ret_ms=round(t_ret, 1),
                         rerank_ms=round(t_rr, 1), gen_ms=round(t_gen, 1),
                         kw_ok=kw_ok, answer=ans[:160], top1=files[0]))
        print(f"[{pos if pos is not None else '-'}] {q[:30]:32s} 召回 {t_ret:4.0f}ms "
              f"重排 {t_rr:4.0f}ms 生成 {t_gen:4.0f}ms 关键词{'✅' if kw_ok else '❌'}", flush=True)

    n = len(QAS)
    print(f"\n=== 加 Reranker 后（{n} 题）===")
    print(f"  Recall@1 {r1}/{n} ({r1/n*100:.0f}%) | Recall@3 {r3}/{n} ({r3/n*100:.0f}%) | "
          f"Recall@5 {r5}/{n} ({r5/n*100:.0f}%)")
    print(f"  关键词命中 {sum(r['kw_ok'] for r in rows)}/{n}")
    print(f"  平均：召回 {np.mean([r['ret_ms'] for r in rows]):.0f} ms + "
          f"重排 {np.mean(rr_ms):.0f} ms + 生成 {np.mean(gen_ms):.0f} ms = "
          f"{np.mean([r['ret_ms'] for r in rows]) + np.mean(rr_ms) + np.mean(gen_ms):.0f} ms")
    print("\n=== 对比基线（纯向量 top-5）===")
    print("  R@1 42% | R@3 67% | R@5 67% | 关键词 4/12（归一化后）| 端到端 709 ms")

    json.dump(dict(recall_at_1=r1 / n, recall_at_3=r3 / n, recall_at_5=r5 / n,
                   kw_hit=sum(r["kw_ok"] for r in rows) / n,
                   ret_ms=float(np.mean([r["ret_ms"] for r in rows])),
                   rerank_ms=float(np.mean(rr_ms)), gen_ms=float(np.mean(gen_ms)),
                   rows=rows), open(OUT, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
