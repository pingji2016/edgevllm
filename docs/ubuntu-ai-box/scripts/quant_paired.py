"""配对检验：bf16 vs NF4 在 300 道算术题上的逐题对错

输出 2x2 列联表 + McNemar 统计量，判断 11pp 的差距是不是噪声
"""
import json
import math
import os
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BF16 = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
OUT = os.path.expanduser("~/apps/minimind/quant_paired.json")
NUM = re.compile(r"-?\d+")
SYS = ("你是一个计算器。只输出结果，格式：<answer>数字</answer>\n\n"
       "示例：\n3+4=<answer>7</answer>\n8+5=<answer>13</answer>\n6+2=<answer>8</answer>")

tok = AutoTokenizer.from_pretrained(BF16)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

rng = random.Random(2026)
pairs = [(rng.randint(100, 999), rng.randint(10, 99)) for _ in range(300)]
prompts = [tok.apply_chat_template([{"role": "system", "content": SYS},
                                    {"role": "user", "content": f"{a}*{b}="}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) + "<answer>"
           for a, b in pairs]


@torch.no_grad()
def run(model, batch=16):
    ok = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=16, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for j in range(len(enc.input_ids)):
            t = tok.decode(g[j, enc.input_ids.size(1):], skip_special_tokens=True)
            m = NUM.search(t)
            a, b = pairs[i + j]
            ok.append(bool(m and int(m.group()) == a * b))
        del g
    return ok


res = {}
for name, kw in (("bf16", dict(dtype=torch.bfloat16)),
                 ("NF4", dict(quantization_config=BitsAndBytesConfig(
                     load_in_4bit=True, bnb_4bit_quant_type="nf4",
                     bnb_4bit_compute_dtype=torch.bfloat16,
                     bnb_4bit_use_double_quant=True)))):
    m = AutoModelForCausalLM.from_pretrained(BF16, device_map={"": 0}, **kw).eval()
    res[name] = run(m)
    print(f"  {name}: {sum(res[name])}/{len(res[name])} = "
          f"{sum(res[name])/len(res[name])*100:.1f}%", flush=True)
    del m
    torch.cuda.empty_cache()

a, b = res["bf16"], res["NF4"]
only_bf16 = sum(1 for x, y in zip(a, b) if x and not y)
only_nf4 = sum(1 for x, y in zip(a, b) if y and not x)
both = sum(1 for x, y in zip(a, b) if x and y)
neither = sum(1 for x, y in zip(a, b) if not x and not y)
n = len(a)
z = (only_bf16 - only_nf4) / math.sqrt(only_bf16 + only_nf4) if only_bf16 + only_nf4 else 0.0

print(f"\n=== 配对结果（{n} 题）===")
print(f"  两者都对 {both} | 都对错 {neither}")
print(f"  只有 bf16 对 {only_bf16} | 只有 NF4 对 {only_nf4}")
print(f"  准确率差 {(only_bf16-only_nf4)/n*100:+.1f} pp | McNemar z = {z:.2f}")
print(f"  （|z| > 1.96 才是 95% 显著）")

json.dump(dict(n=n, bf16_acc=sum(a) / n, nf4_acc=sum(b) / n,
               both=both, neither=neither, only_bf16=only_bf16,
               only_nf4=only_nf4, mcnemar_z=z, per_item=dict(a=a, b=b)),
          open(OUT, "w"), indent=2)
print(f"已存 {OUT}")
