"""量化对任务准确率的影响：bf16 / GPTQ-Int8 / NF4（同一引擎 vLLM）

三个维度（都不是 PPL）：
  A. 算术      100 题 三位×两位（贪心，强制 <answer> 前缀）
  B. 指令      10 题 结构化输出（要求只输出 JSON，判 json.loads）
  C. 长上下文  8K needle × 2 深度 × 2 种子（复用明天的语料）
"""
import glob
import json
import os
import random
import re
import time

from vllm import LLM, SamplingParams

BF16 = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
GPTQ = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B-GPTQ-Int8/snapshots/master"
DOCS = os.path.expanduser("~/apps/edgevllm/docs/ubuntu-ai-box")
OUT = os.path.expanduser("~/apps/infer-engines/quant_quality.json")
NEEDLE = "【重要记录】机房代号“蓝鲸”的访问密码是 741928，请勿外传。"
NUM = re.compile(r"-?\d+")

SYS_MATH = ("你是一个计算器。只输出结果，格式：<answer>数字</answer>\n\n"
            "示例：\n3+4=<answer>7</answer>\n8+5=<answer>13</answer>\n6+2=<answer>8</answer>")

JSON_CASES = [
    ("张三", 28, "北京"), ("李四", 35, "上海"), ("王五", 19, "广州"),
    ("赵六", 42, "深圳"), ("钱七", 51, "成都"), ("孙八", 23, "杭州"),
    ("周九", 37, "武汉"), ("吴十", 64, "西安"), ("郑一", 26, "南京"),
    ("王二", 48, "重庆"),
]


def math_prompt(tok, a, b):
    return tok.apply_chat_template(
        [{"role": "system", "content": SYS_MATH},
         {"role": "user", "content": f"{a}*{b}="}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) + "<answer>"


def json_prompt(tok, name, age, city):
    return tok.apply_chat_template(
        [{"role": "system", "content": "你是一个信息抽取器。只输出 JSON，不要解释、不要代码块。"},
         {"role": "user", "content": f"姓名 {name}，年龄 {age}，城市 {city}。转成 JSON。"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def build_sets(seed=777):
    rng = random.Random(seed)
    math_pairs = [(rng.randint(100, 999), rng.randint(10, 99)) for _ in range(100)]
    return math_pairs


def needle_prompts(tok, hay_pool, L=8192, seeds=(0, 1)):
    out = []
    for d in (0.1, 0.9):
        for s in seeds:
            rng = random.Random(s)
            pieces, ntok = [], 0
            while ntok < L:
                pieces.append(hay_pool[rng.randrange(len(hay_pool))])
                ntok = len(tok("\n\n".join(pieces)).input_ids)
            hay = "\n\n".join(pieces)
            ids = tok(hay).input_ids
            hay = tok.decode(ids[:L - 96], skip_special_tokens=True)
            cut = max(1, int(len(hay) * d))
            pos = hay.find("\n", cut)
            hay = hay[:pos] + "\n\n" + NEEDLE + "\n\n" + hay[pos:]
            out.append((d, tok.apply_chat_template(
                [{"role": "system", "content": "只输出答案。"},
                 {"role": "user", "content": f"{hay}\n\n问题：代号“蓝鲸”的访问密码是多少？只回答数字。"}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)))
    return out


def run_config(name, model, extra_kw, tok, math_pairs, needle):
    t0 = time.time()
    llm = LLM(model=model, dtype="bfloat16", max_model_len=16384,
              gpu_memory_utilization=0.80, disable_log_stats=True, **extra_kw)
    load_s = time.time() - t0
    sp = SamplingParams(max_tokens=16, temperature=0.0)

    # A 算术
    outs = llm.generate([math_prompt(tok, a, b) for a, b in math_pairs], sp)
    right = 0
    for (a, b), o in zip(math_pairs, outs):
        m = NUM.search(o.outputs[0].text)
        right += bool(m and int(m.group()) == a * b)

    # B 指令（JSON）
    jp = [json_prompt(tok, *c) for c in JSON_CASES]
    jouts = llm.generate(jp, SamplingParams(max_tokens=80, temperature=0.0))
    jok = 0
    for o in jouts:
        t = o.outputs[0].text.strip()
        t = re.sub(r"^```(?:json)?|```$", "", t).strip()
        try:
            d = json.loads(t)
            jok += isinstance(d, dict) and len(d) >= 3
        except Exception:
            pass

    # C 长上下文
    nouts = llm.generate([p for _, p in needle], SamplingParams(max_tokens=12, temperature=0.0))
    nok = sum(1 for o in nouts if "741928" in o.outputs[0].text)

    del llm
    import torch
    torch.cuda.empty_cache()

    return dict(model=name, load_s=round(load_s, 1),
                math_acc=round(right / len(math_pairs), 3),
                json_ok=round(jok / len(JSON_CASES), 3),
                needle_ok=f"{nok}/{len(needle)}")


def main():
    tok_holder = {}
    math_pairs = build_sets()

    # 先用 bf16 的 tokenizer 构造长上下文 prompt（三个模型共用同一 tokenizer 家族）
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BF16)
    hay_pool = []
    for f in sorted(glob.glob(os.path.join(DOCS, "*.md"))):
        blob = open(f, encoding="utf-8", errors="ignore").read()
        hay_pool += [p.strip() for p in blob.split("\n\n") if len(p.strip()) > 80]
    needle = needle_prompts(tok, hay_pool)
    print(f"  长上下文 prompt {len(needle)} 条 | 语料段 {len(hay_pool)}", flush=True)

    results = []
    for name, model, kw in (
        ("bf16", BF16, {}),
        ("GPTQ-Int8", GPTQ, {}),
        ("NF4", BF16, dict(quantization="bitsandbytes", load_format="bitsandbytes")),
    ):
        try:
            print(f"\n=== {name} ===", flush=True)
            r = run_config(name, model, kw, tok, math_pairs, needle)
            print(f"  加载 {r['load_s']}s | 算术 {r['math_acc']*100:.1f}% | "
                  f"JSON {r['json_ok']*100:.0f}% | needle {r['needle_ok']}", flush=True)
            results.append(r)
        except Exception as e:
            print(f"  ❌ {type(e).__name__}: {str(e)[:200]}", flush=True)
            results.append(dict(model=name, error=f"{type(e).__name__}: {str(e)[:200]}"))

    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"\n已存 {OUT}")


if __name__ == "__main__":
    main()
