"""工具调用失败分析：只跑 free 模式，打印错误案例并归类"""
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, "/home/pingji/apps/minimind")
from tool_calling import CASES, TOOLS, build_prompt, extract_json, args_ok, norm  # noqa

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForCausalLM.from_pretrained(
    MD, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa").eval()

bad = []
for q, gt, ga in CASES:
    enc = tok(build_prompt(tok, q, "free"), return_tensors="pt").to("cuda")
    n_in = enc["input_ids"].size(1)
    with torch.inference_mode():
        g = model.generate(**enc, max_new_tokens=96, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    txt = tok.decode(g[0, n_in:], skip_special_tokens=True).strip()
    obj = extract_json(txt) or {}
    gt_pred = obj.get("tool")
    ga_pred = obj.get("arguments") or {}
    if gt_pred != gt or not args_ok(ga_pred, ga):
        bad.append((q, gt, ga, gt_pred, ga_pred))

print(f"失败 {len(bad)}/{len(CASES)}\n")
cats = {"工具选错": 0, "参数缺失": 0, "参数值错": 0, "参数类型/格式错": 0}
for q, gt, ga, gp, ap in bad:
    if gp != gt:
        cats["工具选错"] += 1
        kind = "工具选错"
    else:
        miss = [k for k in ga if k not in ap]
        if miss:
            cats["参数缺失"] += 1
            kind = "参数缺失"
        else:
            wrong = []
            for k, v in ga.items():
                pv = ap.get(k)
                if isinstance(v, (int, float)):
                    try:
                        if abs(float(pv) - float(v)) > 1e-6:
                            wrong.append(k)
                    except Exception:
                        wrong.append(k)
                elif norm(v) not in norm(pv) and norm(pv) not in norm(v):
                    wrong.append(k)
            if wrong:
                cats["参数值错"] += 1
                kind = "参数值错"
            else:
                cats["参数类型/格式错"] += 1
                kind = "参数类型/格式错"
    print(f"[{kind}] {q}")
    print(f"    金标 {gt} {json.dumps(ga, ensure_ascii=False)}")
    print(f"    预测 {gp} {json.dumps(ap, ensure_ascii=False)[:110]}")

print("\n归类：", cats)
json.dump(dict(total=len(CASES), failures=len(bad), cats=cats,
               examples=[dict(q=q, gold=[g, a], pred=[gp, ap])
                         for q, g, a, gp, ap in bad]),
          open("/home/pingji/apps/minimind/tool_calling_fail.json", "w"),
          ensure_ascii=False, indent=2)
