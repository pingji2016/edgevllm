"""NF4 质量对比（HF 引擎，与 bf16 配对）

同样的三个任务：算术 100 / JSON 10 / 8K needle 4
"""
import glob
import json
import os
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BF16 = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
DOCS = os.path.expanduser("~/apps/edgevllm/docs/ubuntu-ai-box")
OUT = os.path.expanduser("~/apps/minimind/quant_quality_hf.json")
NEEDLE = "【重要记录】机房代号“蓝鲸”的访问密码是 741928，请勿外传。"
NUM = re.compile(r"-?\d+")
SYS_MATH = ("你是一个计算器。只输出结果，格式：<answer>数字</answer>\n\n"
            "示例：\n3+4=<answer>7</answer>\n8+5=<answer>13</answer>\n6+2=<answer>8</answer>")
JSON_CASES = [("张三", 28, "北京"), ("李四", 35, "上海"), ("王五", 19, "广州"),
              ("赵六", 42, "深圳"), ("钱七", 51, "成都"), ("孙八", 23, "杭州"),
              ("周九", 37, "武汉"), ("吴十", 64, "西安"), ("郑一", 26, "南京"),
              ("王二", 48, "重庆")]

tok = AutoTokenizer.from_pretrained(BF16)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
rng = random.Random(777)
math_pairs = [(rng.randint(100, 999), rng.randint(10, 99)) for _ in range(100)]


def math_prompt(a, b):
    return tok.apply_chat_template([{"role": "system", "content": SYS_MATH},
                                    {"role": "user", "content": f"{a}*{b}="}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) + "<answer>"


def json_prompt(name, age, city):
    return tok.apply_chat_template(
        [{"role": "system", "content": "你是一个信息抽取器。只输出 JSON，不要解释、不要代码块。"},
         {"role": "user", "content": f"姓名 {name}，年龄 {age}，城市 {city}。转成 JSON。"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def needle_prompts(L=8192):
    pool = []
    for f in sorted(glob.glob(os.path.join(DOCS, "*.md"))):
        blob = open(f, encoding="utf-8", errors="ignore").read()
        pool += [p.strip() for p in blob.split("\n\n") if len(p.strip()) > 80]
    out = []
    for d in (0.1, 0.9):
        for s in (0, 1):
            r = random.Random(s)
            pieces, n = [], 0
            while n < L:
                pieces.append(pool[r.randrange(len(pool))])
                n = len(tok("\n\n".join(pieces)).input_ids)
            hay = tok.decode(tok("\n\n".join(pieces)).input_ids[:L - 96], skip_special_tokens=True)
            cut = max(1, int(len(hay) * d))
            pos = hay.find("\n", cut)
            hay = hay[:pos] + "\n\n" + NEEDLE + "\n\n" + hay[pos:]
            out.append(tok.apply_chat_template(
                [{"role": "system", "content": "只输出答案。"},
                 {"role": "user", "content": f"{hay}\n\n问题：代号“蓝鲸”的访问密码是多少？只回答数字。"}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False))
    return out


@torch.no_grad()
def gen(model, prompts, max_new, batch=8):
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for j in range(len(chunk)):
            outs.append(tok.decode(g[j, enc.input_ids.size(1):], skip_special_tokens=True))
        del g
    return outs


def run(name, model):
    a = gen(model, [math_prompt(x, y) for x, y in math_pairs], 16)
    right = 0
    for (x, y), t in zip(math_pairs, a):
        m = NUM.search(t)
        right += bool(m and int(m.group()) == x * y)
    jt = gen(model, [json_prompt(*c) for c in JSON_CASES], 80)
    jok = 0
    for t in jt:
        s = re.sub(r"^```(?:json)?|```$", "", t.strip()).strip()
        try:
            d = json.loads(s)
            jok += isinstance(d, dict) and len(d) >= 3
        except Exception:
            pass
    nt = gen(model, needle_prompts(), 12, batch=2)
    nok = sum("741928" in t for t in nt)
    vram = torch.cuda.max_memory_allocated() / 2**30
    r = dict(model=name, math_acc=round(right / 100, 3), json_ok=round(jok / 10, 3),
             needle=f"{nok}/4", peak_vram_gb=round(vram, 2))
    print(f"  {name}: 算术 {r['math_acc']*100:.0f}% | JSON {r['json_ok']*100:.0f}% | "
          f"needle {r['needle']} | 峰值显存 {r['peak_vram_gb']} GB", flush=True)
    return r


res = []
for name, kwargs in (("HF-bf16", dict(dtype=torch.bfloat16)),
                     ("HF-NF4", dict(quantization_config=BitsAndBytesConfig(
                         load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)))):
    torch.cuda.reset_peak_memory_stats()
    m = AutoModelForCausalLM.from_pretrained(BF16, device_map={"": 0}, **kwargs).eval()
    res.append(run(name, m))
    del m
    torch.cuda.empty_cache()

json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
print(f"已存 {OUT}")
