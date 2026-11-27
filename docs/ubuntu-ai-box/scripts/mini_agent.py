"""mini agent：多轮工具调用 + 真实执行

对照：
  no_tool  直接回答（不给工具）
  agent    最多 6 步，每步输出一个工具调用，观察回填后继续

20 个任务需要 2-3 步依赖：读文件 → 算 → 写文件 / 检索 → 回答
指标：成功率、平均步数、失败类型
"""
import ast
import json
import os
import re
import shutil
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
WS = "/tmp/agent_ws"
DOCS = os.path.expanduser("~/apps/edgevllm/docs/ubuntu-ai-box")
OUT = os.path.expanduser("~/apps/minimind/agent_result.json")
MAX_STEPS = 6


# ---------------- 工具 ----------------

def _p(path):
    p = os.path.normpath(os.path.join(WS, path.replace(WS, "").lstrip("/")))
    if not p.startswith(WS):
        raise ValueError("路径越界")
    return p


def t_read_file(path):
    with open(_p(path), encoding="utf-8") as f:
        return f.read()[:800]


def t_write_file(path, content):
    with open(_p(path), "w", encoding="utf-8") as f:
        f.write(str(content))
    return f"已写入 {len(str(content))} 字符"


def t_list_dir(path="."):
    return ", ".join(sorted(os.listdir(_p(path))))


def t_calc(expression):
    node = ast.parse(str(expression), mode="eval")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                return a / b
            if isinstance(n.op, ast.Pow):
                return a ** b
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        raise ValueError("不支持的表达式")

    return str(ev(node))


def t_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


class DocSearch:
    """BM25 检索（不加载模型，秒级）"""

    def __init__(self):
        import glob
        from collections import Counter
        self.docs, self.names = [], []
        for f in sorted(glob.glob(os.path.join(DOCS, "*.md"))):
            for line in open(f, encoding="utf-8", errors="ignore"):
                s = line.strip()
                if len(s) > 40 and not s.startswith("|"):
                    self.docs.append(s)
                    self.names.append(os.path.basename(f))
        self.tf = []
        df = Counter()
        for d in self.docs:
            t = self.token(d)
            c = Counter(t)
            self.tf.append(c)
            df.update(set(t))
        import math
        N = len(self.docs)
        self.idf = {k: math.log(1 + (N - v + 0.5) / (v + 0.5)) for k, v in df.items()}
        self.avg = sum(sum(c.values()) for c in self.tf) / N

    @staticmethod
    def token(s):
        s = s.lower()
        zh = re.findall(r"[\u4e00-\u9fff]", s)
        return re.findall(r"[a-z0-9_]+", s) + [zh[i] + zh[i + 1] for i in range(len(zh) - 1)]

    def search(self, query):
        q = set(self.token(query))
        scores = []
        for i, tf in enumerate(self.tf):
            n = sum(tf.values())
            s = sum(self.idf.get(t, 0) * tf.get(t, 0) * 2.5 /
                    (tf.get(t, 0) + 1.5 * (1 - 0.75 + 0.75 * n / self.avg)) for t in q)
            scores.append(s)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:3]
        return "\n".join(f"[{self.names[i]}] {self.docs[i][:160]}" for i in order)


TOOLS = {
    "read_file": ({"path": "str"}, "读取工作区文件"),
    "write_file": ({"path": "str", "content": "str"}, "写入工作区文件"),
    "list_dir": ({"path": "str"}, "列出目录"),
    "calc": ({"expression": "str"}, "算术计算"),
    "search_docs": ({"query": "str"}, "检索本地文档"),
    "now": ({}, "当前时间"),
}

# ---------------- 夹具 ----------------

def setup():
    shutil.rmtree(WS, ignore_errors=True)
    os.makedirs(WS)
    open(f"{WS}/sales.csv", "w").write("product,revenue\nA,120\nB,340\nC,90\n")
    open(f"{WS}/notes.txt", "w").write("第一行\n第二行\n第三行\n第四行\n第五行\n")
    open(f"{WS}/numbers.txt", "w").write("7 12 3 25 9\n")
    open(f"{WS}/a.txt", "w").write("17")
    open(f"{WS}/b.txt", "w").write("28")
    open(f"{WS}/config.json", "w").write('{"port": 8080, "host": "0.0.0.0"}')
    open(f"{WS}/prices.csv", "w").write("item,price\nX,10\nY,20\nZ,30\n")
    open(f"{WS}/log.txt", "w").write(
        "INFO start\nERROR db timeout\nINFO retry\nERROR disk full\nINFO done\nERROR net\n")
    open(f"{WS}/words.txt", "w").write("cat elephant dog butterfly\n")
    open(f"{WS}/data.csv", "w").write("a,b\n2,3\n4,5\n6,7\n")
    open(f"{WS}/stats.txt", "w").write("4 8 15 16 23 42\n")
    open(f"{WS}/dates.txt", "w").write("2024-05-03\n2023-11-20\n2025-01-15\n")


def norm(s):
    return re.sub(r"[\s,，。'\"%]+", "", str(s)).lower()


def num(s):
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group()) if m else None


# (任务, 检查函数, 单步基线答案)
def build_tasks():
    T = []

    def add(task, check, direct):
        T.append(dict(task=task, check=check, direct=direct))

    add("读 sales.csv，把 revenue 列所有值加起来，告诉我总和。",
        lambda a, o: num(a) == 550, "550")
    add("notes.txt 一共有几行？只回答数字。",
        lambda a, o: num(a) == 5, "5")
    add("工作区里有几个 .csv 文件？只回答数字。",
        lambda a, o: num(a) == 3, "3")
    add("numbers.txt 里的最大值是多少？",
        lambda a, o: num(a) == 25, "25")
    add("读 a.txt 和 b.txt，把两个数相加，回答结果。",
        lambda a, o: num(a) == 45, "45")
    add("计算 (12+8)*3，并把结果写入 out1.txt，然后回答结果。",
        lambda a, o: num(a) == 60 and os.path.exists(f"{WS}/out1.txt")
        and num(open(f"{WS}/out1.txt").read()) == 60, "60")
    add("config.json 里 port 的值是多少？",
        lambda a, o: num(a) == 8080, "8080")
    add("prices.csv 里 price 的平均值是多少？",
        lambda a, o: num(a) == 20, "20")
    add("在本地文档里搜索「KV cache 每 token 占多少 KB」，回答数字。",
        lambda a, o: num(a) == 116, "116")
    add("在本地文档里搜索「ngram 投机的并发交叉点」，回答并发数。",
        lambda a, o: num(a) == 8, "8")
    add("sales.csv 里 revenue 最高的产品名是什么？只回答产品名。",
        lambda a, o: "B" in str(a), "B")
    add("log.txt 里 ERROR 出现了几次？只回答数字。",
        lambda a, o: num(a) == 3, "3")
    add("把 hello 写入 greet.txt，然后读回来告诉我文件内容。",
        lambda a, o: "hello" in norm(a) and os.path.exists(f"{WS}/greet.txt"),
        "hello")
    add("words.txt 里最长的单词是哪个？",
        lambda a, o: "butterfly" in str(a).lower(), "butterfly")
    add("计算 37*18，把结果写入 math.txt 并回答结果。",
        lambda a, o: num(a) == 666 and os.path.exists(f"{WS}/math.txt"),
        "666")
    add("data.csv 里 a*b 的和是多少？",
        lambda a, o: num(a) == 110, "110")
    add("工作区里哪个文件名最长？",
        lambda a, o: "numbers.txt" in str(a), "numbers.txt")
    add("dates.txt 里最早的日期是哪天？",
        lambda a, o: "2023" in str(a), "2023-11-20")
    add("在本地文档里搜索「bge-m3 的 R@1」，回答百分比数字。",
        lambda a, o: num(a) in (75, 0.75), "75")
    add("stats.txt 里所有数的平均值是多少？",
        lambda a, o: abs((num(a) or -1) - 18.0) < 0.01, "18")
    return T


# ---------------- 提示词 ----------------

def tool_text():
    lines = []
    for n, (args, desc) in TOOLS.items():
        a = ", ".join(f"{k}: {v}" for k, v in args.items()) or "无参数"
        lines.append(f"- {n}({a}): {desc}")
    return "\n".join(lines)


SYS_AGENT = ("你是一个工具调用代理。每一步只输出一个 JSON 对象，不要输出别的：\n"
             '{"tool": "<工具名>", "arguments": {...}}\n'
             "任务完成时输出：\n"
             '{"tool": "finish", "arguments": {"answer": "<最终答案>"}}\n'
             f"可用工具：\n{tool_text()}\n"
             "规则：每次只调用一个工具；不要编造工具结果；"
             "工具结果会以 <observation> 返回给你；最多 6 步。")


def extract_json(text):
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(text[i:j + 1])
    except Exception:
        return None


@torch.no_grad()
def gen(model, tok, messages, max_new=128):
    p = tok.apply_chat_template(messages, tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    ids = tok(p, return_tensors="pt").input_ids.cuda()
    g = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(g[0, ids.size(1):], skip_special_tokens=True).strip()


def run_agent(model, tok, ds, task):
    msgs = [{"role": "system", "content": SYS_AGENT},
            {"role": "user", "content": f"任务：{task}"}]
    trace = []
    for step in range(MAX_STEPS):
        raw = gen(model, tok, msgs)
        obj = extract_json(raw)
        trace.append(("model", raw[:120]))
        if not obj or "tool" not in obj:
            return None, step + 1, trace, "格式错误"
        tool = obj["tool"]
        args = obj.get("arguments") or {}
        if tool == "finish":
            return str(args.get("answer", "")), step + 1, trace, "正常结束"
        try:
            if tool == "read_file":
                obs = t_read_file(**args)
            elif tool == "write_file":
                obs = t_write_file(**args)
            elif tool == "list_dir":
                obs = t_list_dir(**args)
            elif tool == "calc":
                obs = t_calc(**args)
            elif tool == "search_docs":
                obs = ds.search(**args)
            elif tool == "now":
                obs = t_now()
            else:
                obs = f"未知工具 {tool}"
        except Exception as e:
            obs = f"工具报错：{type(e).__name__}: {str(e)[:60]}"
        trace.append(("obs", obs[:120]))
        msgs.append({"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)})
        msgs.append({"role": "user", "content": f"<observation>{obs}</observation>"})
    return None, MAX_STEPS, trace, "超步数"


def main():
    setup()
    ds = DocSearch()
    tasks = build_tasks()
    tok = AutoTokenizer.from_pretrained(MD)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16,
                                                 device_map={"": 0}).eval()
    print(f"  夹具 {len(os.listdir(WS))} 个文件 | 任务 {len(tasks)} | 文档片段 {len(ds.docs)}",
          flush=True)

    res = {"no_tool": [], "agent": []}
    fails = []
    t0 = time.time()
    for i, t in enumerate(tasks, 1):
        # 无工具对照
        ans = gen(model, tok, [{"role": "system", "content": "只输出最终答案，不要解释。"},
                               {"role": "user", "content": t["task"]}], 96)
        ok_direct = t["check"](ans, None)
        # agent
        ans2, steps, trace, why = run_agent(model, tok, ds, t["task"])
        ok_agent = t["check"](ans2 or "", None)
        res["no_tool"].append(ok_direct)
        res["agent"].append(ok_agent)
        if not ok_agent:
            fails.append(dict(task=t["task"], answer=ans2, steps=steps,
                              why=why, trace=trace[-4:]))
        print(f"  [{i:2d}] 无工具 {'✅' if ok_direct else '❌'} | "
              f"agent {'✅' if ok_agent else '❌'} ({steps} 步) | {t['task'][:32]}", flush=True)

    n = len(tasks)
    print(f"\n=== 结果（{n} 任务，{time.time()-t0:.0f}s）===")
    print(f"  无工具直接答：{sum(res['no_tool'])}/{n} = {sum(res['no_tool'])/n*100:.0f}%")
    print(f"  mini agent  ：{sum(res['agent'])}/{n} = {sum(res['agent'])/n*100:.0f}%")
    print(f"  平均步数（仅成功）：{sum(1 for x in res['agent'] if x) and '见下'}")
    cats = {}
    for f in fails:
        cats[f["why"]] = cats.get(f["why"], 0) + 1
    print(f"  失败类型：{cats}")

    json.dump(dict(n=n, no_tool=res["no_tool"], agent=res["agent"],
                   fails=fails, cats=cats), open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"  已存 {OUT}")


if __name__ == "__main__":
    main()
