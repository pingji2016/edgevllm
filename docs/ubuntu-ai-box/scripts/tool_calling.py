"""工具调用评测：Qwen3-1.7B，10 个工具 × 60 条指令

三种模式：
  free       自然语言要求输出 JSON（不约束）
  xgr        xgrammar 强制 JSON schema（tool 用 enum 锁死）
  name_only  只要求输出工具名（拆开"选择"与"格式"）

指标：JSON 合法率、工具选择准确率、参数准确率、完整准确率、延迟
"""
import json
import random
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"

TOOLS = {
    "get_weather": ("city: str, date: str", "查询某城市某天的天气"),
    "calculator": ("expression: str", "计算数学表达式"),
    "search_web": ("query: str", "搜索网页"),
    "create_calendar_event": ("title: str, start_time: str, duration_min: int", "创建日历事件"),
    "send_email": ("to: str, subject: str", "发送邮件"),
    "query_database": ("table: str, filter: str", "查询数据库表"),
    "translate": ("text: str, target_lang: str", "翻译文本"),
    "convert_unit": ("value: float, from_unit: str, to_unit: str", "单位换算"),
    "set_reminder": ("message: str, time: str", "设置提醒"),
    "read_file": ("path: str", "读取文件"),
}
NAMES = list(TOOLS)

# (指令, 工具, 关键参数 dict)
CASES = [
    ("明天北京天气怎么样？", "get_weather", {"city": "北京"}),
    ("帮我看下上海后天的天气", "get_weather", {"city": "上海"}),
    ("广州今天热不热", "get_weather", {"city": "广州"}),
    ("深圳周五会下雨吗", "get_weather", {"city": "深圳"}),
    ("查一下成都明天的气温", "get_weather", {"city": "成都"}),
    ("杭州现在多少度", "get_weather", {"city": "杭州"}),
    ("帮我算一下 37*18", "calculator", {"expression": "37*18"}),
    ("计算 (125+75)/4", "calculator", {"expression": "(125+75)/4"}),
    ("2 的 10 次方是多少", "calculator", {"expression": "2^10"}),
    ("算一下 3.14 * 8", "calculator", {"expression": "3.14*8"}),
    ("999 除以 27 等于多少", "calculator", {"expression": "999/27"}),
    ("求 18 的平方根", "calculator", {"expression": "sqrt(18)"}),
    ("搜索一下 vLLM 的 PagedAttention 原理", "search_web", {"query": "PagedAttention"}),
    ("帮我搜 transformer 论文原文", "search_web", {"query": "transformer"}),
    ("查一下今天的 AI 新闻", "search_web", {"query": "AI 新闻"}),
    ("搜索 llama.cpp 怎么编译", "search_web", {"query": "llama.cpp"}),
    ("找一下 bge-m3 的模型卡片", "search_web", {"query": "bge-m3"}),
    ("搜 Python asyncio 教程", "search_web", {"query": "asyncio"}),
    ("周五下午三点安排产品评审会，一小时", "create_calendar_event",
     {"title": "产品评审会", "duration_min": 60}),
    ("明天上午十点加一个周会，半小时", "create_calendar_event",
     {"title": "周会", "duration_min": 30}),
    ("下周一早九点排个面试，45 分钟", "create_calendar_event",
     {"title": "面试", "duration_min": 45}),
    ("今晚八点建一个健身提醒，60 分钟", "create_calendar_event",
     {"title": "健身", "duration_min": 60}),
    ("周三下午两点安排设计评审，两小时", "create_calendar_event",
     {"title": "设计评审", "duration_min": 120}),
    ("周六上午十点加一个读书会，90 分钟", "create_calendar_event",
     {"title": "读书会", "duration_min": 90}),
    ("给张三发邮件，主题是周报", "send_email", {"to": "张三", "subject": "周报"}),
    ("给 lisi@example.com 发一封会议纪要", "send_email",
     {"to": "lisi@example.com", "subject": "会议纪要"}),
    ("发邮件给王五，标题是请假申请", "send_email", {"to": "王五", "subject": "请假申请"}),
    ("给团队发一封项目进度邮件", "send_email", {"to": "团队", "subject": "项目进度"}),
    ("给老板发邮件说明延期原因", "send_email", {"to": "老板", "subject": "延期原因"}),
    ("发邮件给赵六，主题是合同确认", "send_email", {"to": "赵六", "subject": "合同确认"}),
    ("查一下 users 表里 city 等于北京的用户", "query_database",
     {"table": "users", "filter": "city=北京"}),
    ("查询 orders 表里金额大于 1000 的订单", "query_database",
     {"table": "orders", "filter": "amount>1000"}),
    ("看看 products 表里库存为 0 的商品", "query_database",
     {"table": "products", "filter": "stock=0"}),
    ("查 logs 表最近一小时的错误", "query_database", {"table": "logs", "filter": "错误"}),
    ("统计 sales 表里 region 等于华东的销售额", "query_database",
     {"table": "sales", "filter": "region=华东"}),
    ("查一下 members 表里 level 大于 5 的会员", "query_database",
     {"table": "members", "filter": "level>5"}),
    ("把 'hello world' 翻译成中文", "translate", {"text": "hello world", "target_lang": "中文"}),
    ("把 'good morning' 翻译成日语", "translate", {"text": "good morning", "target_lang": "日语"}),
    ("将 今天天气很好 翻译成英文", "translate", {"text": "今天天气很好", "target_lang": "英文"}),
    ("把 invoice 翻译成法语", "translate", {"text": "invoice", "target_lang": "法语"}),
    ("翻译 'see you tomorrow' 为西班牙语", "translate",
     {"text": "see you tomorrow", "target_lang": "西班牙语"}),
    ("把 机器学习 翻译成德语", "translate", {"text": "机器学习", "target_lang": "德语"}),
    ("把 5 公里换算成米", "convert_unit", {"value": 5, "from_unit": "公里", "to_unit": "米"}),
    ("100 华氏度等于多少摄氏度", "convert_unit",
     {"value": 100, "from_unit": "华氏度", "to_unit": "摄氏度"}),
    ("3 小时是多少分钟", "convert_unit", {"value": 3, "from_unit": "小时", "to_unit": "分钟"}),
    ("把 2.5 GB 换成 MB", "convert_unit", {"value": 2.5, "from_unit": "GB", "to_unit": "MB"}),
    ("10 千克等于多少磅", "convert_unit", {"value": 10, "from_unit": "千克", "to_unit": "磅"}),
    ("500 毫升是多少升", "convert_unit", {"value": 500, "from_unit": "毫升", "to_unit": "升"}),
    ("明天早上 8 点提醒我吃药", "set_reminder", {"message": "吃药", "time": "明天 08:00"}),
    ("今晚十点提醒我关电脑", "set_reminder", {"message": "关电脑", "time": "今晚 22:00"}),
    ("周五下午四点提醒我交报告", "set_reminder", {"message": "交报告", "time": "周五 16:00"}),
    ("一小时后提醒我喝水", "set_reminder", {"message": "喝水", "time": "一小时后"}),
    ("每天早上七点提醒我跑步", "set_reminder", {"message": "跑步", "time": "每天 07:00"}),
    ("下午三点提醒我开会", "set_reminder", {"message": "开会", "time": "下午 15:00"}),
    ("读一下 /etc/hosts", "read_file", {"path": "/etc/hosts"}),
    ("打开 ~/notes.txt 看看内容", "read_file", {"path": "~/notes.txt"}),
    ("读取 /var/log/syslog 最后几行", "read_file", {"path": "/var/log/syslog"}),
    ("把 config.yaml 读出来", "read_file", {"path": "config.yaml"}),
    ("查看 README.md 的内容", "read_file", {"path": "README.md"}),
    ("读一下 /tmp/data.csv", "read_file", {"path": "/tmp/data.csv"}),
]

SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": NAMES},
        "arguments": {"type": "object"},
    },
    "required": ["tool", "arguments"],
}


def tool_list_text():
    return "\n".join(f"- {n}({s}): {d}" for n, (s, d) in TOOLS.items())


def build_prompt(tok, query, mode):
    if mode == "name_only":
        sys = ("你可以调用以下工具。根据用户请求，**只输出工具名**，不要输出任何其他内容。\n"
               f"工具列表：\n{tool_list_text()}")
    else:
        sys = ("你可以调用以下工具。根据用户请求，只输出一个 JSON 对象，格式：\n"
               '{"tool": "<工具名>", "arguments": {...}}\n'
               f"工具列表：\n{tool_list_text()}\n"
               "不要解释，不要输出代码块。")
    return tok.apply_chat_template(
        [{"role": "system", "content": sys}, {"role": "user", "content": query}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def extract_json(text):
    t = text.strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(t[i:j + 1])
    except Exception:
        return None


def norm(v):
    s = str(v).lower().strip().strip("'\"")
    return re.sub(r"[\s，。,.!?：:]+", "", s)


def args_ok(pred, gold):
    if not isinstance(pred, dict):
        return False
    for k, v in gold.items():
        if k not in pred:
            return False
        pv = pred[k]
        if isinstance(v, (int, float)):
            try:
                if abs(float(pv) - float(v)) > 1e-6:
                    return False
            except Exception:
                return False
        else:
            if norm(v) not in norm(pv) and norm(pv) not in norm(v):
                return False
    return True


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
    compiled = xgr.GrammarCompiler(tinfo).compile_json_schema(json.dumps(SCHEMA))
    print(f"  xgrammar 就绪 | 词表 {tinfo.vocab_size} | 用例 {len(CASES)}", flush=True)

    out = {}
    for mode in ("free", "xgr", "name_only"):
        n = len(CASES)
        valid = tool_ok = args_ok_n = full = 0
        t_all = 0.0
        for q, gold_tool, gold_args in CASES:
            enc = tok(build_prompt(tok, q, mode), return_tensors="pt").to("cuda")
            n_in = enc["input_ids"].size(1)
            kw = {}
            if mode == "xgr":
                kw["logits_processor"] = [make_processor(compiled, tinfo.vocab_size)]
            max_new = 16 if mode == "name_only" else 96
            with torch.inference_mode():
                t0 = time.time()
                g = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                   pad_token_id=tok.eos_token_id, **kw)
                torch.cuda.synchronize()
                t_all += time.time() - t0
            txt = tok.decode(g[0, n_in:], skip_special_tokens=True).strip()

            if mode == "name_only":
                cand = re.sub(r"[^a-z_]", "", txt.lower().split()[0]) if txt.split() else ""
                got_tool = cand if cand in TOOLS else next(
                    (t for t in NAMES if t in txt.lower()), None)
                got_args = {}
                ok = got_tool == gold_tool          # 只看选择
                valid += 1
                tool_ok += got_tool == gold_tool
                full += ok
            else:
                obj = extract_json(txt)
                valid += obj is not None
                got_tool = obj.get("tool") if isinstance(obj, dict) else None
                got_args = obj.get("arguments") if isinstance(obj, dict) else {}
                t_ok = got_tool == gold_tool
                a_ok = t_ok and args_ok(got_args, gold_args)
                tool_ok += t_ok
                args_ok_n += a_ok
                full += a_ok

        r = dict(valid=valid / n, tool=tool_ok / n,
                 args=(args_ok_n / n if mode != "name_only" else None),
                 full=full / n, ms=round(t_all / n * 1000, 1))
        out[mode] = r
        args_str = "-" if r["args"] is None else f"{r['args']*100:5.1f}%"
        print(f"  {mode:9s} JSON {r['valid']*100:5.1f}% | 工具 {r['tool']*100:5.1f}% | "
              f"参数 {args_str} | 完整 {r['full']*100:5.1f}% | {r['ms']:.0f} ms", flush=True)

    json.dump(out, open("/home/pingji/apps/minimind/tool_calling.json", "w"),
              ensure_ascii=False, indent=2)
    print("已存 tool_calling.json")


if __name__ == "__main__":
    main()
