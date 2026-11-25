"""定位式回答 v3：验证"表格行脱离表头就不可定位"的假设

对同一批句子池做三种上下文增强，比较 reranker 定位准确率：
  C0 原始句子（上一轮 5/12）
  C1 句子前加章节标题路径
  C2 句子前加章节标题 + 所在表格的表头行

另加一个路由方案：
  R  先看直接回答 A 里的数字是否能在「定位句」里逐字找到；
     找得到才采用 A（等于交叉验证），否则返回定位句原文
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
OUT = os.path.join(RAG, "eval_locate2.json")

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
SPLIT = re.compile(r"(?<=[。！？；\n])")
MAX_SENTS = 40


def norm(s):
    return s.replace(",", "").replace("，", "").replace(" ", "")


def extract_units(text):
    """返回 [(kind, sentence, heading, table_header)]"""
    units = []
    heading = ""
    header = ""
    prev_is_row = False
    for raw in SPLIT.split(text):
        s = raw.strip()
        if not s:
            prev_is_row = False
            continue
        if s.startswith("#"):
            heading = s.lstrip("#").strip()
            prev_is_row = False
            continue
        is_row = s.startswith("|")
        if is_row and not prev_is_row:
            header = s          # 表格第一行 = 表头
        if is_row and set(s.replace("|", "").strip()) <= set("-: "):
            prev_is_row = True
            continue
        if len(s) >= 10:
            units.append(("row" if is_row else "text", s, heading,
                          header if is_row else ""))
        prev_is_row = is_row
    return units


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
        enc = rtok([[q, t] for t in texts], padding=True, truncation=True,
                   max_length=512, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            return rmodel(**enc).logits.view(-1).float().cpu().numpy()

    def gen(msgs, max_new=96):
        prompt = gtok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
        enc = gtok(prompt, return_tensors="pt", truncation=True, max_length=3072).to("cuda")
        n_in = enc["input_ids"].size(1)
        with torch.inference_mode():
            out = gmodel.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=gtok.pad_token_id)
        return gtok.decode(out[0, n_in:], skip_special_tokens=True).strip()

    rows = []
    ok = dict(A=0, C0=0, C1=0, C2=0, R=0)
    lat = dict(C0=[], C1=[], C2=[])
    nums_re = re.compile(r"\d+(?:\.\d+)?")

    for q, expect, kws in QAS:
        kn = [norm(k) for k in kws]
        cand = np.argsort(-(E @ qvec(q)))[:20]
        s_chunk = rerank(q, [chunks[i]["text"] for i in cand])
        top3 = [chunks[cand[o]] for o in np.argsort(-s_chunk)[:3]]

        units = []
        for c in top3:
            hu = extract_units(c["text"])
            for kind, s, h, hdr in hu:
                units.append(dict(kind=kind, sent=s, head=h, header=hdr,
                                  file=c["file"]))
        units = units[:MAX_SENTS]

        variants = {
            "C0": [u["sent"] for u in units],
            "C1": [f"[{u['head']}] {u['sent']}" for u in units],
            "C2": [(f"[{u['head']}] {u['header']} || {u['sent']}" if u["kind"] == "row"
                    else f"[{u['head']}] {u['sent']}") for u in units],
        }
        picks = {}
        for tag, texts in variants.items():
            t0 = time.time()
            sc = rerank(q, texts)
            i = int(np.argmax(sc))
            picks[tag] = (i, texts[i])
            lat[tag].append((time.time() - t0) * 1000)
            ok[tag] += int(any(k in norm(texts[i]) for k in kn))

        ctx = "\n\n".join(f"【{j+1}】[{c['file']}]\n{c['text'][:600]}"
                          for j, c in enumerate(top3))
        ans_a = gen([{"role": "system", "content": "根据资料回答问题，只答资料里的内容。"},
                     {"role": "user", "content": f"【资料】\n{ctx}\n\n【问题】{q}"}])
        a_ok = any(k in norm(ans_a) for k in kn)
        ok["A"] += int(a_ok)

        # 路由：A 的数字必须能在 C2 定位句里逐字找到
        located = picks["C2"][1]
        a_nums = nums_re.findall(ans_a)
        loc_norm = norm(located)
        verify = any(n in loc_norm for n in a_nums) if a_nums else False
        if verify:
            final, src = ans_a, "A(已核对)"
        else:
            final, src = located, "C2(原文)"
        r_ok = any(k in norm(final) for k in kn)
        ok["R"] += int(r_ok)

        rows.append(dict(q=q, expect=expect, a_ok=a_ok,
                         c0_ok=any(k in norm(picks['C0'][1]) for k in kn),
                         c1_ok=any(k in norm(picks['C1'][1]) for k in kn),
                         c2_ok=any(k in norm(picks['C2'][1]) for k in kn),
                         r_ok=r_ok, verify=verify, src=src,
                         a_ans=ans_a[:100], c2_sent=located[:140],
                         final=final[:140]))
        print(f"  A{'✅' if a_ok else '❌'} C0{'✅' if rows[-1]['c0_ok'] else '❌'}"
              f" C1{'✅' if rows[-1]['c1_ok'] else '❌'}"
              f" C2{'✅' if rows[-1]['c2_ok'] else '❌'}"
              f" R{'✅' if r_ok else '❌'} 「{src}」 {q[:22]}", flush=True)

    n = len(QAS)
    print(f"\n=== 12 题 ===")
    for tag in ("A", "C0", "C1", "C2", "R"):
        print(f"  {tag:3s} {ok[tag]}/{n} ({ok[tag]/n*100:.0f}%)")
    for tag in ("C0", "C1", "C2"):
        print(f"  {tag} 定位耗时 {np.mean(lat[tag]):.0f} ms")

    json.dump(dict(ok={k: v / n for k, v in ok.items()},
                   lat={k: float(np.mean(v)) for k, v in lat.items()},
                   rows=rows), open(OUT, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
