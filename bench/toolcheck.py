"""给 bench 增加 tool_selection 检查（用 judge.py 的等价判据）

20 条单轮指令，10 个工具；判据 = 工具名精确 + 关键参数等价
"""
import json
import re

CASES = [
    ("明天北京天气怎么样？", "get_weather", {"city": "北京"}),
    ("深圳周五会下雨吗", "get_weather", {"city": "深圳"}),
    ("帮我算一下 37*18", "calculator", {"expression": "37*18"}),
    ("2 的 10 次方是多少", "calculator", {"expression": "2^10"}),
    ("搜索一下 vLLM 的 PagedAttention 原理", "search_web", {"query": "PagedAttention"}),
    ("查一下今天的 AI 新闻", "search_web", {"query": "AI 新闻"}),
    ("周五下午三点安排产品评审会，一小时", "create_calendar_event",
     {"title": "产品评审会", "duration_min": 60}),
    ("明天上午十点加一个周会，半小时", "create_calendar_event",
     {"title": "周会", "duration_min": 30}),
    ("给张三发邮件，主题是周报", "send_email", {"subject": "周报"}),
    ("给老板发邮件说明延期原因", "send_email", {}),
    ("查一下 users 表里 city 等于北京的用户", "query_database",
     {"table": "users", "filter": "city=北京"}),
    ("查询 orders 表里金额大于 1000 的订单", "query_database",
     {"table": "orders", "filter": "amount>1000"}),
    ("把 'hello world' 翻译成中文", "translate",
     {"text": "hello world", "target_lang": "中文"}),
    ("把 invoice 翻译成法语", "translate", {"text": "invoice", "target_lang": "法语"}),
    ("把 5 公里换算成米", "convert_unit", {"value": 5, "from_unit": "公里", "to_unit": "米"}),
    ("10 千克等于多少磅", "convert_unit", {"value": 10, "from_unit": "千克", "to_unit": "磅"}),
    ("明天早上 8 点提醒我吃药", "set_reminder", {"message": "吃药", "time": "明天 08:00"}),
    ("下午三点提醒我开会", "set_reminder", {"message": "开会", "time": "下午 15:00"}),
    ("读一下 /etc/hosts", "read_file", {"path": "/etc/hosts"}),
    ("把 config.yaml 读出来", "read_file", {"path": "config.yaml"}),
]

TOOLS = {
    "get_weather": "city: str, date: str", "calculator": "expression: str",
    "search_web": "query: str",
    "create_calendar_event": "title: str, start_time: str, duration_min: int",
    "send_email": "to: str, subject: str", "query_database": "table: str, filter: str",
    "translate": "text: str, target_lang: str",
    "convert_unit": "value: float, from_unit: str, to_unit: str",
    "set_reminder": "message: str, time: str", "read_file": "path: str",
}


def build_prompt(tok, query):
    lines = "\n".join(f"- {n}({a})" for n, a in TOOLS.items())
    sys = ("你可以调用以下工具。只输出一个 JSON：\n"
           '{"tool": "<工具名>", "arguments": {...}}\n'
           f"工具列表：\n{lines}\n不要解释。")
    return tok.apply_chat_template(
        [{"role": "system", "content": sys}, {"role": "user", "content": query}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def extract_json(text):
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(text[i:j + 1])
    except Exception:
        return None


def run(model, tok):
    from judge import args_equiv
    hit = 0
    for q, gt, ga in CASES:
        p = build_prompt(tok, q)
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        import torch
        with torch.inference_mode():
            g = model.generate(ids, max_new_tokens=96, do_sample=False,
                               pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        obj = extract_json(tok.decode(g[0, ids.size(1):], skip_special_tokens=True)) or {}
        if obj.get("tool") == gt and args_equiv(ga, obj.get("arguments") or {}):
            hit += 1
    return hit / len(CASES), f"{hit}/{len(CASES)} 条（等价判据）"
