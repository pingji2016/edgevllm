"""分工拼图 v3：9 段 × 6K，每段只放 1 个事实

A 单 agent 24K（只覆盖约 4 段）
B 9 个 agent 各 6K（全覆盖，且每段只有 1 个事实 -> 无干扰）
C 单 agent 30K（覆盖约 5 段）

结论看两件事：覆盖（能答几段）与干扰（是否张冠李戴）
"""
import glob
import json
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DOCS = os.path.expanduser("~/apps/edgevllm/docs/ubuntu-ai-box")
M17 = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
OUT = os.path.expanduser("~/apps/minimind/split_merge3.json")
L, PARTS = 6000, 9
CODES = ["5839147", "2047638", "9174256", "3625184", "7419063",
         "2850371", "6504829", "1937645", "8274510"]

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
for c in CODES:
    assert c not in _blob, c


def make(i):
    rng = random.Random(300 + i)
    pieces, n = [], 0
    while n < L:
        pieces.append(pool[rng.randrange(len(pool))])
        n = len(tok("\n\n".join(pieces)).input_ids)
    hay = tok.decode(tok("\n\n".join(pieces)).input_ids[:L - 200], skip_special_tokens=True)
    needle = f"【重要记录】代号「蓝鲸{i+1}」的访问密码是 {CODES[i]}，请勿外传。"
    cut = len(hay) // 2
    pos = hay.find("\n", cut)
    return hay[:pos] + "\n\n" + needle + "\n\n" + hay[pos:]


@torch.no_grad()
def ask(model, text, q, max_new=48):
    p = tok.apply_chat_template(
        [{"role": "system", "content": "你是精确的文档问答助手，只回答被问到的内容。"},
         {"role": "user", "content": f"{text}\n\n问题：{q}"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(p, return_tensors="pt", truncation=True, max_length=32000).input_ids.cuda()
    g = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(g[0, ids.size(1):], skip_special_tokens=True)


def q(i):
    return f"「蓝鲸{i+1}」的访问密码是多少？只回答数字。"


def main():
    parts = [make(i) for i in range(PARTS)]
    print(f"  {PARTS} 段 × {L:,} token，每段 1 个事实", flush=True)
    model = AutoModelForCausalLM.from_pretrained(M17, dtype=torch.bfloat16,
                                                 device_map={"": 0}).eval()
    joined = tok.decode(tok("\n\n".join(parts)).input_ids, skip_special_tokens=True)
    ids = tok(joined).input_ids

    def run_all(text, label):
        got, det = 0, []
        for i in range(PARTS):
            out = ask(model, text, q(i))
            hit = CODES[i] in out
            got += hit
            det.append((i + 1, hit, out.strip()[:20]))
        print(f"  {label:16s} {got}/{PARTS}", flush=True)
        return dict(got=got, detail=det)

    res = {}
    res["A 1agent/24K"] = run_all(tok.decode(ids[:24000], skip_special_tokens=True), "A 1x24K")
    b = 0
    bd = []
    for i in range(PARTS):
        out = ask(model, parts[i], q(i))
        hit = CODES[i] in out
        b += hit
        bd.append((i + 1, hit, out.strip()[:20]))
    print(f"  {'B 9agent/6K':16s} {b}/{PARTS}", flush=True)
    res["B 9agent/6K"] = dict(got=b, detail=bd)
    res["C 1agent/30K"] = run_all(tok.decode(ids[:30000], skip_special_tokens=True), "C 1x30K")

    print(f"\n{'配置':16s} {'找回':>6s} {'比例':>6s}")
    for k, v in res.items():
        print(f"  {k:14s} {v['got']}/{PARTS} {v['got']/PARTS*100:5.0f}%")
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    print("已存 split_merge3.json")


if __name__ == "__main__":
    main()
