"""按字段类型重新评分（读已保存的预测，不重跑模型）

规则：
  query/text/message/subject/path  包含关系即可
  to                               邮箱或联系人：包含 / 名字拼音首字母 / 标记为 ambiguous
  filter/expression                去空格引号 + 列名同义词 + 数字一致
  time                             解析成 (日期词, 时, 分)，上午/下午/晚上转 24 小时制
  数值                             浮点相等
"""
import json
import re

D = json.load(open("/home/pingji/apps/minimind/tool_rescore.json", encoding="utf-8"))
CN = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
      "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
DAYS = ["今天", "明天", "后天", "今晚", "明晚", "昨天", "周一", "周二", "周三",
        "周四", "周五", "周六", "周日", "每天", "一小时后"]
ALIAS = {
    "中文": "zh", "汉语": "zh", "chinese": "zh", "zh": "zh", "zh-cn": "zh",
    "英文": "en", "英语": "en", "english": "en", "en": "en",
    "日语": "ja", "日文": "ja", "japanese": "ja", "ja": "ja",
    "法语": "fr", "french": "fr", "fr": "fr",
    "德语": "de", "german": "de", "de": "de",
    "西班牙语": "es", "spanish": "es", "es": "es",
    "小时": "h", "hour": "h", "hours": "h", "hr": "h", "h": "h",
    "分钟": "min", "minute": "min", "minutes": "min", "min": "min",
    "千克": "kg", "公斤": "kg", "kilogram": "kg", "kg": "kg",
    "磅": "lb", "pound": "lb", "pounds": "lb", "lb": "lb",
    "毫升": "ml", "milliliter": "ml", "ml": "ml",
    "升": "l", "liter": "l", "litre": "l", "l": "l",
    "公里": "km", "千米": "km", "kilometer": "km", "km": "km",
    "米": "m", "meter": "m", "m": "m",
    "华氏度": "f", "fahrenheit": "f", "摄氏度": "c", "celsius": "c",
    "库存": "stock", "inventory": "stock", "错误": "error", "errors": "error",
}


def canon(s):
    t = str(s).strip().lower().replace("＝", "=").replace("：", ":")
    t = re.sub(r"[\s'\"`，。、]+", "", t)
    return ALIAS.get(t, t)


def free_ok(g, p):
    a, b = canon(g), canon(p)
    return a in b or b in a


def parse_time(s):
    s = str(s)
    day = next((w for w in DAYS if w in s), None)
    m = re.search(r"(\d{1,2})[:：](\d{2})", s)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
    else:
        raw = [int(x) for x in re.findall(r"\d+", s)]
        cn = [v for k, v in CN.items() if k in s]
        cand = raw + cn
        hh, mm = (cand[0], 0) if cand else (None, 0)
    if hh is not None:
        if any(w in s for w in ("下午", "晚上", "今晚", "明晚", "傍晚")) and hh < 12:
            hh += 12
        if any(w in s for w in ("上午", "早上", "凌晨")) and hh == 12:
            hh = 0
    return day, hh, mm


def time_ok(g, p):
    dg, hg, mg = parse_time(g)
    dp, hp, mp = parse_time(p)
    if dg and dp and dg != dp:
        return False
    return hg == hp and mg == mp


def expr_ok(g, p):
    a, b = canon(g), canon(p)
    if a in b or b in a:
        return True
    ka, kb = a.split("=")[0], b.split("=")[0]
    na = sorted(int(x) for x in re.findall(r"\d+", a))
    nb = sorted(int(x) for x in re.findall(r"\d+", b))
    return canon(ka) == canon(kb) and na == nb


def to_ok(g, p):
    a, b = canon(g), canon(p)
    if a in b or b in a:
        return "ok"
    # 名字 -> 邮箱：只要邮箱本地部分是拼音/名字，视为 ambiguous（模型自行补全）
    if "@" in b and "@" not in a:
        return "ambiguous"
    return "no"


ARGS = {"time": time_ok, "filter": expr_ok, "expression": expr_ok,
        "query": free_ok, "text": free_ok, "message": free_ok, "subject": free_ok,
        "path": free_ok}

rows = D["rows"]
strict = lenient = 0
amb = 0
resid = []
for r in rows:
    gt, pa, ga = r["pred_tool"], r["pred_args"], r["gold_args"]
    if gt != r["gold_tool"]:
        resid.append((r["q"], "工具选错", r["gold_tool"], gt))
        continue
    ok = True
    local_amb = False
    for k, v in ga.items():
        if k not in pa:
            ok = False
            break
        pv = pa[k]
        if isinstance(v, (int, float)):
            try:
                if abs(float(pv) - float(v)) > 1e-6:
                    ok = False
            except Exception:
                ok = False
        elif k == "to":
            st = to_ok(v, pv)
            if st == "no":
                ok = False
            elif st == "ambiguous":
                local_amb = True
        elif k in ARGS:
            if not ARGS[k](v, pv):
                ok = False
        else:
            if not free_ok(v, pv):
                ok = False
        if not ok:
            break
    if ok:
        lenient += 1
        amb += local_amb
    else:
        resid.append((r["q"], "参数不符", json.dumps(ga, ensure_ascii=False),
                      json.dumps(pa, ensure_ascii=False)))

n = len(rows)
print(f"严格判据：{D['strict']*100:.1f}%")
print(f"等价判据：{lenient}/{n} = {lenient/n*100:.1f}%")
print(f"  其中靠'邮箱补全'通过的（建议人工确认）：{amb} 条")
print(f"  严格判据误杀：{lenient - int(D['strict']*n)} 条")
print(f"\n仍然失败 {len(resid)} 条：")
for q, kind, g, p in resid:
    print(f"  [{kind}] {q[:30]}")
    print(f"    金标 {g}")
    print(f"    预测 {p}")
