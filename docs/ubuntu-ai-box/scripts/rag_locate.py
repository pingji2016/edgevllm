"""RAG 定位式回答 v2：三条流水线对比

A  直接回答        模型自己写数字（基线）
B  单选定位        把 top-3 切成带编号的句子，模型只输出一个数字，
                   程序从**原文**取那句（模型抄错也不影响）
C  纯程序定位      reranker 对 (问题, 句子) 逐对打分，取最高分那句，
                   完全不需要 LLM 生成（零抄写误差）

判据：最终返回的原文句子是否含预期关键词
"""
import json
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

RAG = os.path.expanduser("~/apps/infer-research/rag")
BGE = open(os.path.expanduser("~/models/.bge_done")).read().strip()
RR = open(os.path.expanduser("~/models/.rr_done")).read().strip()
MD17 = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
OUT = os.path.join(RAG, "eval_locate.json")

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
SENT_SPLIT = re.compile(r"(?<=[。！？；\n])")
MAX_SENTS = 40


def norm(s):
    return s.replace(",", "").replace("，", "").replace(" ", "")


def split_sents(text):
    return [s.strip() for s in SENT_SPLIT.split(text) if len(s.strip()) >= 10]


def main():
    from transformers import (AutoModel, AutoModelForCausalLM,
                              AutoModelForSequenceClassification, AutoTokenizer)

    chunks = [json.loads(l) for l in open(os.path.join(RAG, "chunks.jsonl"), encoding="utf-8")]
    E = np.load(os.path.join(RAG, "emb.npy"))

    btok = AutoTokenizer.from_pretrained(BGE)
    bmodel = AutoModel.from_pretrained(BGE, dtype=torch.bfloat16).cuda().eval()
    rtok = AutoTokenizer.from_pretrained(RR)
    rmodel = AutoModelForSequenceClassification.from_pretrained(
        RR, dtype=torch.bfloat16).cuda().eval()
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

    def rerank(q, texts):
        pairs = [[q, t] for t in texts]
        enc = rtok(pairs, padding=True, truncation=True, max_length=512,
                   return_tensors="pt").to("cuda")
        with torch.inference_mode():
            s = rmodel(**enc).logits.view(-1).float().cpu().numpy()
        return s

    def gen(msgs, max_new=96):
        prompt = gtok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
        enc = gtok(prompt, return_tensors="pt", truncation=True, max_length=3072).to("cuda")
        n_in = enc["input_ids"].size(1)
        with torch.inference_mode():
            t0 = time.time()
            out = gmodel.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=gtok.pad_token_id)
            torch.cuda.synchronize()
            dt = (time.time() - t0) * 1000
        return gtok.decode(out[0, n_in:], skip_special_tokens=True).strip(), dt

    rows = []
    ok_a = ok_b = ok_c = 0
    t_a, t_b, t_c = [], [], []

    for q, expect, kws in QAS:
        kn = [norm(k) for k in kws]
        # ① 向量召回 20 → rerank → top3 块
        cand = np.argsort(-(E @ qvec(q)))[:20]
        s_chunk = rerank(q, [chunks[i]["text"] for i in cand])
        top3 = [chunks[cand[o]] for o in np.argsort(-s_chunk)[:3]]

        # ② 切成句子池
        pool = []
        for c in top3:
            for s in split_sents(c["text"]):
                pool.append((c["file"], s))
        pool = pool[:MAX_SENTS]

        # ---- A 直接回答 ----
        ctx = "\n\n".join(f"【{j+1}】[{c['file']}]\n{c['text'][:600]}"
                          for j, c in enumerate(top3))
        ans_a, dt_a = gen([{"role": "system", "content": "根据资料回答问题，只答资料里的内容。"},
                           {"role": "user", "content": f"【资料】\n{ctx}\n\n【问题】{q}"}])
        good_a = any(k in norm(ans_a) for k in kn)
        ok_a += good_a; t_a.append(dt_a)

        # ---- B 单选定位（模型只输出编号）----
        listing = "\n".join(f"{j+1}. {s}" for j, (_, s) in enumerate(pool))
        msgs = [{"role": "system",
                 "content": "下面每句有编号。请只输出包含问题答案的那一句的编号，"
                            "只输出一个阿拉伯数字，不要输出别的任何内容。"},
                {"role": "user", "content": f"{listing}\n\n问题：{q}\n\n编号："}]
        ans_b, dt_b = gen(msgs, 12)
        t_b.append(dt_b)
        m = re.search(r"\d+", ans_b)
        idx_b = int(m.group()) - 1 if m else 0
        idx_b = idx_b if 0 <= idx_b < len(pool) else 0
        sent_b = pool[idx_b][1] if pool else ""
        good_b = any(k in norm(sent_b) for k in kn)
        ok_b += good_b

        # ---- C 纯程序：reranker 句子级打分 ----
        t0 = time.time()
        if pool:
            s_sent = rerank(q, [s for _, s in pool])
            idx_c = int(np.argmax(s_sent))
            sent_c = pool[idx_c][1]
        else:
            sent_c, idx_c = "", -1
        dt_c = (time.time() - t0) * 1000
        good_c = any(k in norm(sent_c) for k in kn)
        ok_c += good_c; t_c.append(dt_c)

        rows.append(dict(q=q, expect=expect, n_pool=len(pool),
                         a_ok=good_a, b_ok=good_b, c_ok=good_c,
                         a_ans=ans_a[:100], b_out=ans_b[:40], b_idx=idx_b + 1,
                         b_sent=sent_b[:100], c_sent=sent_c[:100],
                         t_a=round(dt_a), t_b=round(dt_b), t_c=round(dt_c)))
        print(f"  A{'✅' if good_a else '❌'} B{'✅' if good_b else '❌'}"
              f" C{'✅' if good_c else '❌'}  {q[:26]:28s} B选{sent_b[:34]}", flush=True)

    n = len(QAS)
    print(f"\n=== 12 题 ===")
    print(f"  A 直接回答    {ok_a}/{n} ({ok_a/n*100:.0f}%)  {np.mean(t_a):.0f} ms")
    print(f"  B 单选定位    {ok_b}/{n} ({ok_b/n*100:.0f}%)  {np.mean(t_b):.0f} ms")
    print(f"  C 程序定位    {ok_c}/{n} ({ok_c/n*100:.0f}%)  {np.mean(t_c):.0f} ms")

    json.dump(dict(a=ok_a / n, b=ok_b / n, c=ok_c / n,
                   t_a=float(np.mean(t_a)), t_b=float(np.mean(t_b)),
                   t_c=float(np.mean(t_c)), rows=rows),
              open(OUT, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
