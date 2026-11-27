"""判据工具库：别再用字符串精确匹配做评测

今天 5 个 bug 里有 4 个是"判据把对的判成错的"：
  千分位逗号、批量解码差异、工具参数的语言码/单位、CLIP 张量
这里把可复用的等价规则集中到一个文件。
"""
import re

ALIAS = {
    # 语言
    "中文": "zh", "汉语": "zh", "chinese": "zh", "zh": "zh", "zh-cn": "zh",
    "英文": "en", "英语": "en", "english": "en", "en": "en",
    "日语": "ja", "日文": "ja", "japanese": "ja", "ja": "ja",
    "法语": "fr", "french": "fr", "fr": "fr",
    "德语": "de", "german": "de", "de": "de",
    "西班牙语": "es", "spanish": "es", "es": "es",
    # 单位
    "小时": "h", "hour": "h", "hours": "h", "hr": "h", "h": "h",
    "分钟": "min", "minute": "min", "minutes": "min", "min": "min",
    "千克": "kg", "公斤": "kg", "kilogram": "kg", "kg": "kg",
    "磅": "lb", "pound": "lb", "pounds": "lb", "lb": "lb",
    "毫升": "ml", "milliliter": "ml", "ml": "ml",
    "升": "l", "liter": "l", "litre": "l", "l": "l",
    "公里": "km", "千米": "km", "kilometer": "km", "km": "km",
    "米": "m", "meter": "m", "m": "m",
    "华氏度": "f", "fahrenheit": "f", "摄氏度": "c", "celsius": "c",
    # 同义词
    "库存": "stock", "inventory": "stock", "错误": "error", "errors": "error",
}
CN = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
      "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
DAYS = ["今天", "明天", "后天", "今晚", "明晚", "昨天", "周一", "周二", "周三",
        "周四", "周五", "周六", "周日", "每天", "一小时后"]


def canon(s):
    t = str(s).strip().lower().replace("＝", "=").replace("：", ":")
    t = re.sub(r"[\s'\"`，。、]+", "", t)
    return ALIAS.get(t, t)


def num(s):
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group()) if m else None


def num_eq(a, b, tol=1e-6):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def time_key(s):
    """'明天早上8点' / '明天 08:00' / '下午三点' -> ('明天','08','00')"""
    s = str(s)
    day = next((w for w in DAYS if w in s), None)
    m = re.search(r"(\d{1,2})[:：](\d{2})", s)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
    else:
        cand = [int(x) for x in re.findall(r"\d+", s)] + [v for k, v in CN.items() if k in s]
        hh, mm = (cand[0], 0) if cand else (None, 0)
    if hh is not None:
        if any(w in s for w in ("下午", "晚上", "今晚", "明晚", "傍晚")) and hh < 12:
            hh += 12
        if any(w in s for w in ("上午", "早上", "凌晨")) and hh == 12:
            hh = 0
    return day, hh, mm


def equiv(key, gold, pred):
    """按字段类型判等价"""
    if isinstance(gold, (int, float)):
        return num_eq(pred, gold)
    if key == "time":
        dg, hg, mg = time_key(gold)
        dp, hp, mp = time_key(pred)
        return (not dg or not dp or dg == dp) and hg == hp and mg == mp
    if key in ("filter", "expression"):
        a, b = canon(gold), canon(pred)
        if a in b or b in a:
            return True
        ka, kb = a.split("=")[0], b.split("=")[0]
        return canon(ka) == canon(kb) and sorted(
            int(x) for x in re.findall(r"\d+", a)) == sorted(
            int(x) for x in re.findall(r"\d+", b))
    a, b = canon(gold), canon(pred)
    return a in b or b in a


def args_equiv(gold_args, pred_args):
    if not isinstance(pred_args, dict):
        return False
    for k, v in gold_args.items():
        if k not in pred_args or not equiv(k, v, pred_args[k]):
            return False
    return True


def mcnemar(only_a, only_b):
    """返回 z；|z|>1.96 显著"""
    import math
    n = only_a + only_b
    return (only_a - only_b) / math.sqrt(n) if n else 0.0
