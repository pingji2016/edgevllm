"""多 agent 分工 v3：一篇文档一次调用（短上下文可靠）

对比"覆盖面"：
  A 单 agent 能看到的文档（累计 <=24K token，约 17% 语料）
  B 全部 39 篇（= 39 个 agent 分工，100% 覆盖）

同时报告 A 在"它看到的文档内"的召回（区分覆盖 vs 注意力）
"""
import glob
import json
import os
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DOCS = os.path.expanduser("~/apps/edgevllm/docs/ubuntu-ai-box")
M17 = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
OUT = os.path.expanduser("~/apps/minimind/mapreduce3.json")
NUM = re.compile(r"\d+(?:\.\d+)?")
KEY = re.compile(r"RTF|实时率", re.IGNORECASE)
BUDGET = 24000

tok = AutoTokenizer.from_pretrained(M17)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token


def norm_num(s):
    try:
        return f"{float(s):.4f}"
    except Exception:
        return None


def load():
    out = []
    for f in sorted(glob.glob(os.path.join(DOCS, "*.md"))):
        text = open(f, encoding="utf-8", errors="ignore").read()
        out.append((os.path.basename(f), text, len(tok(text).input_ids)))
    return out


def gt_of(name, text):
    s = set()
    for line in text.splitlines():
        if KEY.search(line):
            for m in NUM.findall(line):
                v = norm_num(m)
                if v:
                    s.add((name, v))
    return s


def prompt(name, text, budget):
    ids = tok(text).input_ids
    if len(ids) > budget:
        text = tok.decode(ids[:budget], skip_special_tokens=True)
    return ("<|im_start|>system\n只输出原文行，每行一条，不要解释、不要编号、不要代码块。\n<|im_end|>\n"
            f"<|im_start|>user\n下面文档里，把所有包含「RTF」或「实时率」的行原样抄出来：\n"
            f"{text}\n<|im_end|>\n<|im_start|>assistant\n")


@torch.no_grad()
def run(model, prompts, max_new=320):
    outs = []
    for p in prompts:
        ids = tok(p, return_tensors="pt", truncation=True,
                  max_length=30000).input_ids.cuda()
        g = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        outs.append(tok.decode(g[0, ids.size(1):], skip_special_tokens=True))
        del g
    return outs


def extract(name, out):
    got = set()
    for line in out.splitlines():
        if KEY.search(line):
            for m in NUM.findall(line):
                v = norm_num(m)
                if v:
                    got.add((name, v))
    return got


def main():
    files = load()
    total = sum(f[2] for f in files)
    print(f"  文档 {len(files)} | 总 token {total:,}", flush=True)

    # A 覆盖：按字母序取到 24K
    covered, used = [], 0
    for name, text, n in files:
        if used + n > BUDGET:
            continue
        covered.append(name)
        used += n
    gt_all, gt_cov = set(), set()
    for name, text, n in files:
        g = gt_of(name, text)
        gt_all |= g
        if name in covered:
            gt_cov |= g
    print(f"  A 覆盖 {len(covered)} 篇 / {used:,} token | GT 全量 {len(gt_all)} | "
          f"覆盖内 {len(gt_cov)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(M17, dtype=torch.bfloat16,
                                                 device_map={"": 0}).eval()
    t0 = time.time()
    preds = {}
    for name, text, n in files:
        out = run(model, [prompt(name, text, 100000)])[0]
        preds[name] = extract(name, out)
    print(f"  39 次调用完成 {time.time()-t0:.0f}s", flush=True)

    # A：只算覆盖内
    pred_a = set().union(*[preds[n] for n in covered]) if covered else set()
    inter_a = pred_a & gt_cov
    rec_a = len(inter_a) / len(gt_cov) if gt_cov else 0
    prec_a = len(inter_a) / len(pred_a) if pred_a else 0
    # 对全量 GT 的召回（覆盖惩罚）
    rec_a_all = len(pred_a & gt_all) / len(gt_all) if gt_all else 0

    # B：全部
    pred_b = set().union(*preds.values())
    inter_b = pred_b & gt_all
    rec_b = len(inter_b) / len(gt_all) if gt_all else 0
    prec_b = len(inter_b) / len(pred_b) if pred_b else 0

    print(f"\n{'配置':18s} {'召回(全量GT)':>13s} {'精确':>7s} {'覆盖内召回':>11s}")
    print(f"  A 单上下文({len(covered)}篇)   {rec_a_all*100:11.1f}% {prec_a*100:6.1f}% "
          f"{rec_a*100:9.1f}%")
    print(f"  B 全部(39个worker)     {rec_b*100:11.1f}% {prec_b*100:6.1f}% {'—':>9s}")

    json.dump(dict(total_tokens=total, covered=covered,
                   gt_all=len(gt_all), gt_cov=len(gt_cov),
                   A=dict(recall_all=rec_a_all, recall_cov=rec_a, precision=prec_a),
                   B=dict(recall_all=rec_b, precision=prec_b)),
              open(OUT, "w"), ensure_ascii=False, indent=2)
    print("已存 mapreduce3.json")


if __name__ == "__main__":
    main()
