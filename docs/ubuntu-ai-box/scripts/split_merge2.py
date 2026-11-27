"""分工拼图 v2：每段 3 个事实，共 9 个，逐题问

A 单 agent 24K（只能覆盖第 1 段 + 第 2 段一半）
B 3 个 agent 各 16K（全覆盖）
C 单 agent 30K（覆盖 1.875 段）

每题一问，答案逐字比对，避免"一次列 5 个"的格式问题
"""
import glob
import json
import os
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DOCS = os.path.expanduser("~/apps/edgevllm/docs/ubuntu-ai-box")
M17 = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
OUT = os.path.expanduser("~/apps/minimind/split_merge2.json")
L, PART, PER = 16000, 3, 3
DEPTHS = [0.2, 0.5, 0.8]
CODES = [["5839147", "2047638", "9174256"],
         ["3625184", "7419063", "2850371"],
         ["6504829", "1937645", "8274510"]]

tok = AutoTokenizer.from_pretrained(M17)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

pool = []
for f in sorted(glob.glob(os.path.join(DOCS, "*.md"))):
    blob = open(f, encoding="utf-8", errors="ignore").read()
    pool += [p.strip() for p in blob.split("\n\n") if len(p.strip()) > 80]
_blob = "".join(open(f, encoding="utf-8", errors="ignore").read()
                for f in sorted(glob.glob(os.path.join(DOCS, "*.md"))))
for row in CODES:
    for c in row:
        assert c not in _blob, c


def make(p):
    rng = random.Random(200 + p)
    pieces, n = [], 0
    while n < L:
        pieces.append(pool[rng.randrange(len(pool))])
        n = len(tok("\n\n".join(pieces)).input_ids)
    hay = tok.decode(tok("\n\n".join(pieces)).input_ids[:L - 400], skip_special_tokens=True)
    # 按深度插入 3 个 needle
    for j, d in enumerate(DEPTHS):
        needle = f"【重要记录】代号「蓝鲸{p+1}-{j+1}」的访问密码是 {CODES[p][j]}，请勿外传。"
        cut = max(1, int(len(hay) * d))
        pos = hay.find("\n", cut)
        if pos < 0:
            pos = cut
        hay = hay[:pos] + "\n\n" + needle + "\n\n" + hay[pos:]
    return hay


@torch.no_grad()
def ask(model, text, question, max_new=48):
    p = tok.apply_chat_template(
        [{"role": "system", "content": "你是精确的文档问答助手，只回答被问到的内容。"},
         {"role": "user", "content": f"{text}\n\n问题：{question}"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(p, return_tensors="pt", truncation=True, max_length=32000).input_ids.cuda()
    g = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(g[0, ids.size(1):], skip_special_tokens=True)


def q(p, j):
    return f"「蓝鲸{p+1}-{j+1}」的访问密码是多少？只回答数字。"


def main():
    parts = [make(p) for p in range(PART)]
    print(f"  {PART} 段 × {L:,} token，每段 {PER} 个 needle", flush=True)
    model = AutoModelForCausalLM.from_pretrained(M17, dtype=torch.bfloat16,
                                                 device_map={"": 0}).eval()
    joined = "\n\n".join(parts)
    all_ids = tok(joined).input_ids

    def eval_text(text, allowed_parts):
        got = 0
        detail = []
        for p in allowed_parts:
            for j in range(PER):
                out = ask(model, text, q(p, j))
                hit = CODES[p][j] in out
                got += hit
                detail.append((p + 1, j + 1, hit, out.strip()[:24]))
        return got, detail

    res = {}
    # A 24K：实际覆盖 p1 全部 + p2 一部分
    texts = tok.decode(all_ids[:24000], skip_special_tokens=True)
    got, det = eval_text(texts, range(PART))
    res["A 1agent/24K"] = dict(got=got, detail=det)
    print(f"  A 1x24K  {got}/9", flush=True)

    # B 3 agents
    got, det = eval_text_multi = (0, [])
    for p in range(PART):
        sub, d = eval_text(parts[p], [p])
        got += sub
        det += d
    res["B 3agent/16K"] = dict(got=got, detail=det)
    print(f"  B 3x16K  {got}/9", flush=True)

    # C 30K
    texts = tok.decode(all_ids[:30000], skip_special_tokens=True)
    got, det = eval_text(texts, range(PART))
    res["C 1agent/30K"] = dict(got=got, detail=det)
    print(f"  C 1x30K  {got}/9", flush=True)

    print(f"\n{'配置':16s} {'找回':>6s}")
    for k, v in res.items():
        print(f"  {k:14s} {v['got']}/9 = {v['got']/9*100:.0f}%")
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    print("已存 split_merge2.json")


if __name__ == "__main__":
    main()
