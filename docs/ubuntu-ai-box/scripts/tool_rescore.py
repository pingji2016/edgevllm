"""重新评分：严格判据 vs 语义等价判据

等价规则（针对工具参数）：
  - 语言名/语言码：中文/zh/chinese
  - 单位别名：小时/hour/h；千克/kg；毫升/ml ...
  - 过滤表达式：去空格/引号，列名同义词（库存 stock/inventory）
  - 时间：日期词一致 + 数字一致（"明天早上8点" ≡ "明天 08:00"）
"""
import json
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/pingji/apps/minimind")
from tool_calling import CASES, build_prompt, extract_json, norm  # noqa

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"

ALIAS = {
    "中文": "zh", "汉语": "zh", "华语": "zh", "chinese": "zh", "zh": "zh", "zh-cn": "zh",
    "英文": "en", "英语": "en", "english": "en", "en": "en",
    "日语": "ja", "日文": "ja", "japanese": "ja", "ja": "ja",
    "法语": "fr", "french": "fr", "fr": "fr",
    "德语": "de", "german": "de", "de": "de",
    "西班牙语": "es", "spanish": "es", "es": "es",
    "小时": "h", "钟头": "h", "hour": "h", "hours": "h", "hr": "h", "h": "h",
    "分钟": "min", "minute": "min", "minutes": "min", "min": "min",
    "千克": "kg", "公斤": "kg", "kilogram": "kg", "kilograms": "kg", "kg": "kg",
    "磅": "lb", "pound": "lb", "pounds": "lb", "lb": "lb", "lbs": "lb",
    "毫升": "ml", "milliliter": "ml", "milliliters": "ml", "ml": "ml",
    "升": "l", "liter": "l", "liters": "l", "litre": "l", "l": "l",
    "公里": "km", "千米": "km", "kilometer": "km", "km": "km",
    "米": "m", "meter": "m", "meters": "m", "m": "m",
    "华氏度": "f", "fahrenheit": "f", "°f": "f",
    "摄氏度": "c", "celsius": "c", "°c": "c",
    "库存": "stock", "inventory": "stock", "stock": "stock",
    "错误": "error", "error": "error", "errors": "error",
}
DAY_WORDS = ["今天", "明天", "后天", "今晚", "明晚", "昨天",
             "周一", "周二", "周三", "周四", "周五", "周六", "周日",
             "每天", "一小时后", "下午", "上午", "晚上", "中午", "早上"]
CN_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
            "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def canon(s):
    t = str(s).strip().lower()
    t = t.replace("＝", "=").replace("：", ":")
    t = re.sub(r"[\s'\"`，。、]+", "", t)
    return ALIAS.get(t, t)


def nums(s):
    t = str(s)
    out = [int(x) for x in re.findall(r"\d+", t)]
    for ch, v in CN_DIGIT.items():
        if ch in t:
            out.append(v)
    return sorted(set(out))


def eq_time(gold, pred):
    gd = [w for w in DAY_WORDS if w in str(gold)]
    pd_ = [w for w in DAY_WORDS if w in str(pred)]
    if gd and pd_ and not set(gd) & set(pd_):
        return False
    gn, pn = nums(gold), nums(pred)
    if gn and pn:
        return set(gn) & set(pn) != set() or max(gn) == max(pn)
    return canon(gold) == canon(pred)


def eq_expr(gold, pred):
    g, p = canon(gold), canon(pred)
    if g == p or g in p or p in g:
        return True
    # 列名同义词：取 "=" 前部分比对
    gk = g.split("=")[0]
    pk = p.split("=")[0]
    return canon(gk) == canon(pk) and nums(g) == nums(p)


def equivalent(key, gold, pred):
    if isinstance(gold, (int, float)):
        try:
            return abs(float(pred) - float(gold)) <= 1e-6
        except Exception:
            return False
    if key == "time":
        return eq_time(gold, pred)
    if key in ("filter", "expression"):
        return eq_expr(gold, pred)
    return canon(gold) == canon(pred)


def judge_strict(pred_args, gold_args):
    for k, v in gold_args.items():
        if k not in pred_args:
            return False
        pv = pred_args[k]
        if isinstance(v, (int, float)):
            try:
                if abs(float(pv) - float(v)) > 1e-6:
                    return False
            except Exception:
                return False
        elif norm(v) not in norm(pv) and norm(pv) not in norm(v):
            return False
    return True


def judge_lenient(pred_args, gold_args):
    for k, v in gold_args.items():
        if k not in pred_args:
            return False
        if not equivalent(k, v, pred_args[k]):
            return False
    return True


tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForCausalLM.from_pretrained(
    MD, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa").eval()

rows = []
for q, gt, ga in CASES:
    enc = tok(build_prompt(tok, q, "free"), return_tensors="pt").to("cuda")
    n_in = enc["input_ids"].size(1)
    with torch.inference_mode():
        g = model.generate(**enc, max_new_tokens=96, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    txt = tok.decode(g[0, n_in:], skip_special_tokens=True).strip()
    obj = extract_json(txt) or {}
    rows.append(dict(q=q, gold_tool=gt, gold_args=ga,
                     pred_tool=obj.get("tool"), pred_args=obj.get("arguments") or {},
                     strict=obj.get("tool") == gt and judge_strict(obj.get("arguments") or {}, ga),
                     lenient=obj.get("tool") == gt and judge_lenient(obj.get("arguments") or {}, ga)))

n = len(rows)
s = sum(r["strict"] for r in rows)
l = sum(r["lenient"] for r in rows)
print(f"\n=== 重新评分（{n} 条）===")
print(f"  严格判据   {s}/{n} = {s/n*100:.1f}%")
print(f"  等价判据   {l}/{n} = {l/n*100:.1f}%")
print(f"  被严格判据误杀 {l-s} 条")

print("\n--- 仍然失败（等价判据下）---")
for r in rows:
    if not r["lenient"]:
        print(f"  {r['q']}")
        print(f"    金标 {r['gold_tool']} {json.dumps(r['gold_args'], ensure_ascii=False)}")
        print(f"    预测 {r['pred_tool']} {json.dumps(r['pred_args'], ensure_ascii=False)[:100]}")

json.dump(dict(strict=s / n, lenient=l / n, n=n, rows=rows),
          open("/home/pingji/apps/minimind/tool_rescore.json", "w"),
          ensure_ascii=False, indent=2)
print("\n已存 tool_rescore.json")
