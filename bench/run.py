#!/usr/bin/env python3
"""回归基准：验证文档里的关键结论是否还能复现

用法：
  python run.py                 # 跑全部"快"检查（纯数学 + 带宽 + 文档索引）
  python run.py --include-slow  # 加上需要加载模型的检查（RAG / JSON / 长上下文）
  python run.py --only rag_r1,json_compliance
  python run.py --list          # 列出所有检查

每个检查输出 PASS/FAIL；有失败时退出码为 1。
期望值在 manifest.json 里，改预期不用改代码。
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = json.load(open(os.path.join(HERE, "manifest.json"), encoding="utf-8"))

DOCS = os.path.expanduser("~/apps/edgevllm/docs/ubuntu-ai-box")
BGE3 = os.path.expanduser("~/models/bge-m3")
RR = open(os.path.expanduser("~/models/.rr_done")).read().strip()
RAGDIR = os.path.expanduser("~/apps/infer-research/rag")
MD17 = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"

CHECKS = {}
sys.path.insert(0, HERE)


def check(name, slow=False):
    def deco(fn):
        CHECKS[name] = dict(fn=fn, slow=slow)
        return fn
    return deco


# ---------------- 纯数学（不需要 GPU） ----------------

@check("kv_bytes_per_token")
def _kv():
    b = 2 * 28 * 8 * 128 * 2
    return b, f"2*28*8*128*2 = {b:,} B"


@check("decode_roofline_1p7b")
def _roof():
    return 448 / 3.4, "448 GB/s ÷ 3.4 GB"


@check("spec_break_even")
def _spec():
    a, c, k = 0.349, 0.35, 5
    v = (1 + k * a) / (1 + k * c)
    return v, f"a=34.9%, c=35%, k=5 → {v:.3f}"


@check("concurrency_32k")
def _conc():
    kv_budget = 16 * 0.8 - 3.4 - 1.5          # GB
    per_req = 32768 * 114688 / 2**30          # GB
    return kv_budget / per_req, f"{kv_budget:.1f} GB ÷ {per_req:.2f} GB/请求"


@check("docs_indexed")
def _docs():
    readme = open(os.path.join(DOCS, "README.md"), encoding="utf-8").read()
    files = [os.path.basename(f) for f in glob.glob(os.path.join(DOCS, "*.md"))
             if not f.endswith("README.md")]
    missing = [f for f in files if f not in readme]
    ratio = 1 - len(missing) / max(1, len(files))
    detail = f"{len(files)} 篇，缺 {missing[:3]}" if missing else f"{len(files)} 篇全部已索引"
    return ratio, detail


# ---------------- 需要 GPU ----------------

@check("gpu_bandwidth")
def _bw():
    import torch
    x = torch.empty(1 << 28, dtype=torch.uint8, device="cuda")
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        y.copy_(x)
    torch.cuda.synchronize()
    gb = 2 * x.numel() * 20 / 1e9
    return gb / (time.time() - t), "torch copy 20 次"


@check("rag_r1", slow=True)
def _rag():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

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
    chunks = [json.loads(l) for l in open(os.path.join(RAGDIR, "chunks.jsonl"), encoding="utf-8")]
    E = np.load(os.path.join(RAGDIR, "emb_m3.npy"))
    bt = AutoTokenizer.from_pretrained(BGE3)
    bm = AutoModel.from_pretrained(BGE3, dtype=torch.bfloat16).cuda().eval()
    rt = AutoTokenizer.from_pretrained(RR)
    rm = AutoModelForSequenceClassification.from_pretrained(
        RR, dtype=torch.bfloat16).cuda().eval()

    hit = 0
    for q, expect in QAS:
        enc = bt(q, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        with torch.inference_mode():
            h = bm(**enc).last_hidden_state[:, 0]
        qv = F.normalize(h.float(), dim=-1).cpu().numpy()[0]
        cand = list(np.argsort(-(E @ qv))[:20])
        enc2 = rt([[q, chunks[i]["text"]] for i in cand], padding=True, truncation=True,
                  max_length=512, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            s = rm(**enc2).logits.view(-1).float().cpu().numpy()
        top = [chunks[cand[o]]["file"] for o in np.argsort(-s)[:5]]
        hit += expect in top[0]          # 严格 R@1
    return hit / len(QAS), f"{hit}/{len(QAS)} 题 top-1 命中"


@check("json_compliance", slow=True)
def _json():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cases = [("张三", 28, "北京"), ("李四", 35, "上海"), ("王五", 19, "广州"),
             ("赵六", 42, "深圳"), ("钱七", 51, "成都"), ("孙八", 23, "杭州"),
             ("周九", 37, "武汉"), ("吴十", 64, "西安"), ("郑一", 26, "南京"),
             ("王二", 48, "重庆")]
    tok = AutoTokenizer.from_pretrained(MD17)
    m = AutoModelForCausalLM.from_pretrained(MD17, dtype=torch.bfloat16,
                                             device_map={"": 0}).eval()
    ok = 0
    for name, age, city in cases:
        p = tok.apply_chat_template(
            [{"role": "system", "content": "你是一个信息抽取器。只输出 JSON，不要解释、不要代码块。"},
             {"role": "user", "content": f"姓名 {name}，年龄 {age}，城市 {city}。转成 JSON。"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        with torch.inference_mode():
            g = m.generate(ids, max_new_tokens=80, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        t = re.sub(r"^```(?:json)?|```$", "", tok.decode(g[0, ids.size(1):],
                                                         skip_special_tokens=True).strip()).strip()
        try:
            d = json.loads(t)
            ok += isinstance(d, dict) and len(d) >= 3
        except Exception:
            pass
    return ok / len(cases), f"{ok}/{len(cases)} 合法 JSON"


@check("needle_8k", slow=True)
def _needle():
    import random
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    NEEDLE = "【重要记录】机房代号“蓝鲸”的访问密码是 741928，请勿外传。"
    tok = AutoTokenizer.from_pretrained(MD17)
    m = AutoModelForCausalLM.from_pretrained(MD17, dtype=torch.bfloat16,
                                             device_map={"": 0}).eval()
    pool = []
    for f in glob.glob(os.path.join(DOCS, "*.md")):
        blob = open(f, encoding="utf-8", errors="ignore").read()
        pool += [p.strip() for p in blob.split("\n\n") if len(p.strip()) > 80]

    ok = 0
    total = 0
    for d in (0.1, 0.9):
        for s in range(4):
            rng = random.Random(s)
            pieces, n = [], 0
            while n < 8192:
                pieces.append(pool[rng.randrange(len(pool))])
                n = len(tok("\n\n".join(pieces)).input_ids)
            hay = tok.decode(tok("\n\n".join(pieces)).input_ids[:8192 - 96],
                             skip_special_tokens=True)
            cut = max(1, int(len(hay) * d))
            pos = hay.find("\n", cut)
            hay = hay[:pos] + "\n\n" + NEEDLE + "\n\n" + hay[pos:]
            p = tok.apply_chat_template(
                [{"role": "system", "content": "只输出答案。"},
                 {"role": "user", "content": f"{hay}\n\n问题：代号“蓝鲸”的访问密码是多少？只回答数字。"}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(p, return_tensors="pt").input_ids.cuda()
            with torch.inference_mode():
                g = m.generate(ids, max_new_tokens=12, do_sample=False,
                               pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            ok += "741928" in tok.decode(g[0, ids.size(1):], skip_special_tokens=True)
            total += 1
    return ok / total, f"{ok}/{total} 命中"


@check("tool_selection", slow=True)
def _tool():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import toolcheck
    tok = AutoTokenizer.from_pretrained(MD17)
    m = AutoModelForCausalLM.from_pretrained(MD17, dtype=torch.bfloat16,
                                             device_map={"": 0}).eval()
    return toolcheck.run(m, tok)


# ---------------- 运行器 ----------------

def judge(name, value, spec):
    if "min" in spec:
        return value >= spec["min"]
    return abs(value - spec["expect"]) <= spec.get("tol", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--include-slow", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, v in CHECKS.items():
            print(f"  {k:24s} {'slow' if v['slow'] else 'fast'}")
        return 0

    want = {s.strip() for s in args.only.split(",") if s.strip()}
    names = [n for n, v in CHECKS.items()
             if (n in want if want else (args.include_slow or not v["slow"]))]

    if not names:
        print("❌ 没有匹配的检查（--only 或 slow 过滤后为空）")
        return 2

    print(f"{'检查':24s} {'实测':>14s} {'期望':>14s}  结果")
    print("-" * 70)
    fails = []
    for n in names:
        spec = MANIFEST[n]
        t0 = time.time()
        try:
            val, detail = CHECKS[n]["fn"]()
            ok = judge(n, val, spec)
        except Exception as e:
            val, detail, ok = None, f"{type(e).__name__}: {str(e)[:60]}", False
        exp = (f"≥{spec['min']}" if "min" in spec
               else f"{spec['expect']}±{spec.get('tol', 0)}")
        shown = f"{val:.3g}" if isinstance(val, (int, float)) else "ERR"
        print(f"{n:24s} {shown:>14s} {exp:>14s}  {'PASS' if ok else 'FAIL'}  "
              f"({detail}, {time.time()-t0:.1f}s)")
        if not ok:
            fails.append(n)

    print("-" * 70)
    if fails:
        print(f"❌ {len(fails)} 项未通过: {', '.join(fails)}")
        return 1
    print(f"✅ 全部 {len(names)} 项通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
