"""RAG 第二步：检索命中率 + 带上下文回答

12 个问题，每个有明确的「预期来源文件」和「预期关键词」
"""
import json
import os
import time

import numpy as np
import torch

RAG = os.path.expanduser("~/apps/infer-research/rag")
MD17 = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
BGE = open(os.path.expanduser("~/models/.bge_done")).read().strip()
OUT = os.path.join(RAG, "eval.json")

# (问题, 预期来源文件片段, 预期关键词)
QAS = [
    ("CUDA Graph 相比 eager 在单流下提升多少？", "e2e-and-opt", ["83.8", "84%"]),
    ("Kokoro 跑在 K3 上的实时率（RTF）是多少？", "tts-kokoro", ["2.08", "0.48"]),
    ("量化七档里 Q4_K_M 的 decode 速度是多少 tok/s？", "inference-acceleration", ["221.8", "222"]),
    ("minimind 预训练用了多长时间？", "minimind-training", ["2.8"]),
    ("KV cache 每 token 占多少 KB？", "inference-acceleration", ["116"]),
    ("FlexAttention 不编译时会比 SDPA 慢多少倍？", "inference-acceleration", ["20", "30"]),
    ("Wanda 剪枝 50% 时随机剪枝的 PPL 是多少？", "inference-acceleration", ["178"]),
    ("DeepSeek-OCR 处理一页 A4 需要多长时间？", "deepseek-ocr", ["7.26", "7.2"]),
    ("MiniCPM-o 在 Linux 上编译报什么错？", "minicpm-o", ["F_NOCACHE"]),
    ("K3 上 ngram 投机解码的加速比是多少？", "inference-acceleration", ["1.29", "2.5"]),
    ("vLLM 32 并发能达到多少 tok/s？", "e2e-and-opt", ["3020", "3,020"]),
    ("SDXL LoRA 训练的单步耗时是多少？", "roofline-and-lora", ["676"]),
]


def main():
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    import torch.nn.functional as F

    chunks = [json.loads(l) for l in open(os.path.join(RAG, "chunks.jsonl"), encoding="utf-8")]
    E = np.load(os.path.join(RAG, "emb.npy"))
    print(f"索引 {len(chunks)} 切片 / {E.shape}")

    btok = AutoTokenizer.from_pretrained(BGE)
    bmodel = AutoModel.from_pretrained(BGE, dtype=torch.bfloat16).cuda().eval()

    def embed_query(q):
        enc = btok("为这个句子生成表示以用于检索相关文章：" + q, return_tensors="pt",
                   truncation=True, max_length=512).to("cuda")
        with torch.inference_mode():
            v = bmodel(**enc).last_hidden_state[:, 0]
        return F.normalize(v.float(), dim=-1).cpu().numpy()[0]

    # ---- 生成模型 ----
    gtok = AutoTokenizer.from_pretrained(MD17)
    if gtok.pad_token is None:
        gtok.pad_token = gtok.eos_token
    gmodel = AutoModelForCausalLM.from_pretrained(MD17, dtype=torch.bfloat16,
                                                  device_map={"": 0}).eval()

    rows = []
    r1 = r3 = r5 = 0
    gen_ms_all = []
    for q, expect_file, kws in QAS:
        t0 = time.time()
        qv = embed_query(q)
        scores = E @ qv
        top = np.argsort(-scores)[:5]
        ret_ms = (time.time() - t0) * 1000
        files = [chunks[i]["file"] for i in top]
        hit_pos = next((i for i, f in enumerate(files) if expect_file in f), None)
        r1 += int(hit_pos == 0); r3 += int(hit_pos is not None and hit_pos < 3)
        r5 += int(hit_pos is not None)

        # 用 top-3 作上下文回答
        ctx = "\n\n".join(f"[{chunks[i]['file']}]\n{chunks[i]['text'][:600]}" for i in top[:3])
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
            gen_ms = (time.time() - t0) * 1000
        ans = gtok.decode(out[0, n_in:], skip_special_tokens=True).strip()
        gen_ms_all.append(gen_ms)
        hit_kw = any(k in ans for k in kws)

        rows.append(dict(q=q, expect=expect_file, hit_pos=hit_pos, ret_ms=round(ret_ms, 1),
                         gen_ms=round(gen_ms, 1), answer=ans[:150], kw_ok=hit_kw,
                         top1=files[0]))
        print(f"[{hit_pos if hit_pos is not None else '-'}] {q[:34]:36s} "
              f"检索 {ret_ms:5.0f} ms | 生成 {gen_ms:5.0f} ms | 关键词 {'✅' if hit_kw else '❌'}")

    n = len(QAS)
    print(f"\n=== 检索命中率（{n} 题）===")
    print(f"  Recall@1 {r1}/{n} ({r1/n*100:.0f}%) | Recall@3 {r3}/{n} ({r3/n*100:.0f}%) | "
          f"Recall@5 {r5}/{n} ({r5/n*100:.0f}%)")
    print(f"  平均检索 {np.mean([r['ret_ms'] for r in rows]):.0f} ms | "
          f"平均生成 {np.mean(gen_ms_all):.0f} ms | "
          f"端到端 {np.mean([r['ret_ms'] for r in rows]) + np.mean(gen_ms_all):.0f} ms")
    print(f"  关键词命中 {sum(r['kw_ok'] for r in rows)}/{n}")

    json.dump(dict(recall_at_1=r1 / n, recall_at_3=r3 / n, recall_at_5=r5 / n,
                   rows=rows), open(OUT, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
