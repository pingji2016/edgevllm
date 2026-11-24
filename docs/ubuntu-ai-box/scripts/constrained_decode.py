"""约束解码实测 v2：自由生成 vs xgrammar（JSON Schema 强制）

对比：
  free —— 普通生成，事后尝试解析 JSON
  xgr  —— xgrammar 的 GrammarMatcher 通过 logits processor 强制合法
指标：JSON 合法率、生成 token 数、tok/s

用法：~/apps/bench/yolo-venv/bin/python constrained_decode.py
"""
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

MD = "/home/pingji/.cache/modezscope-placeholder"  # 占位，运行时替换
MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
MAX_NEW = 200

SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "year": {"type": "integer"},
        "products": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["company", "year", "products", "confidence"],
    "additionalProperties": False,
}

PROMPTS = [
    "从下面这段话抽取结构化信息：睿创微纳在 2024 年推出了红外热成像模组和车载夜视方案，我对这个判断的置信度是 0.9。",
    "从下面这段话抽取结构化信息：悟见智能 2023 年发布了端侧语音助手与视觉质检系统，把握大概八成。",
    "从下面这段话抽取结构化信息：某公司在 2025 年做了多模态大模型和 RISC-V 推理板，置信度 0.75。",
]

INSTRUCTION = (
    "\n\n只输出一个 JSON 对象，字段：company(字符串)、year(整数)、"
    "products(字符串数组)、confidence(0~1 的小数)。不要输出任何解释。\nJSON:"
)


def check(obj):
    if not isinstance(obj, dict):
        return False, "不是对象"
    for k in ("company", "year", "products", "confidence"):
        if k not in obj:
            return False, f"缺 {k}"
    if not isinstance(obj["company"], str):
        return False, "company 类型错"
    if not isinstance(obj["year"], int) or isinstance(obj["year"], bool):
        return False, "year 非整数"
    if not isinstance(obj["products"], list) or not all(isinstance(x, str) for x in obj["products"]):
        return False, "products 类型错"
    if not isinstance(obj["confidence"], (int, float)) or isinstance(obj["confidence"], bool):
        return False, "confidence 类型错"
    return True, "ok"


def extract_json(text):
    t = text.strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(t[i:j + 1])
    except Exception:
        return None


def make_processor(compiled, vocab_size):
    import xgrammar as xgr

    class C(LogitsProcessor):
        def __init__(self):
            self.m = xgr.GrammarMatcher(compiled)
            self.bitmask = xgr.allocate_token_bitmask(1, vocab_size)
            self.n = 0

        def __call__(self, input_ids, scores):
            cur = input_ids[0].tolist()
            if self.n == 0:
                self.n = len(cur)
            else:
                for t in cur[self.n:]:
                    self.m.accept_token(int(t))
                self.n = len(cur)
            if not self.m.is_terminated():
                self.m.fill_next_token_bitmask(self.bitmask)
                xgr.apply_token_bitmask_inplace(scores, self.bitmask.to(scores.device))
            return scores

    return C()


def main():
    import xgrammar as xgr

    tok = AutoTokenizer.from_pretrained(MD)
    model = AutoModelForCausalLM.from_pretrained(
        MD, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa").eval()

    tinfo = xgr.TokenizerInfo.from_huggingface(tok)
    compiler = xgr.GrammarCompiler(tinfo)
    compiled = compiler.compile_json_schema(json.dumps(SCHEMA))
    print(f"xgrammar 编译完成 | 词表 {tinfo.vocab_size}\n")

    rows = []
    for mode in ("free", "xgr"):
        print(f"--- {mode} ---")
        for i, p in enumerate(PROMPTS, 1):
            enc = tok(p + INSTRUCTION, return_tensors="pt").to("cuda")
            n_in = enc["input_ids"].size(1)
            kw = {}
            if mode == "xgr":
                kw["logits_processor"] = [make_processor(compiled, tinfo.vocab_size)]
            with torch.inference_mode():
                model.generate(**enc, max_new_tokens=8, do_sample=False,
                               pad_token_id=tok.eos_token_id)  # 预热
                torch.cuda.synchronize()
                t0 = time.time()
                out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.eos_token_id, **kw)
                torch.cuda.synchronize()
                dt = time.time() - t0
            n_new = out.shape[1] - n_in
            txt = tok.decode(out[0, n_in:], skip_special_tokens=True)
            obj = extract_json(txt)
            ok, why = check(obj) if obj is not None else (False, "无法解析")
            rows.append(dict(mode=mode, prompt=i, tokens=n_new, seconds=round(dt, 2),
                             tps=round(n_new / dt, 1), valid=ok, why=why,
                             sample=txt.strip()[:80]))
            print(f"  [{i}] {n_new:3d} token | {dt:5.2f}s | {n_new/dt:5.1f} tok/s | "
                  f"合法 {'✅' if ok else '❌ ' + why}")
            print(f"      {txt.strip()[:72]!r}")

    print("\n=== 汇总 ===")
    for mode in ("free", "xgr"):
        rs = [r for r in rows if r["mode"] == mode]
        valid = sum(r["valid"] for r in rs)
        tps = sum(r["tps"] for r in rs) / len(rs)
        toks = sum(r["tokens"] for r in rs) / len(rs)
        print(f"  {mode:4s} 合法率 {valid}/{len(rs)} | 平均 {tps:5.1f} tok/s | 平均 {toks:5.0f} token")

    json.dump(rows, open(os.path.expanduser("~/apps/infer-research/constrained_results.json"), "w"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
