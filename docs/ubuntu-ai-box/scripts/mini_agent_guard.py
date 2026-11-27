"""Agent 护栏：验证 65% 是模型上限还是工程上限

四道护栏：
  1. 路径白名单：非法路径返回"文件不存在 + 可用文件列表"
  2. 强制先读后答：任务里点名的文件没读过，就拒绝 finish
  3. 每步校验：解析失败/多 JSON 时提示重试
  4. 步数预算 + 重复读缓存

对照：无护栏 13/20（今天的 mini_agent.py，修正判据后）
"""
import json
import os
import re
import sys

sys.path.insert(0, "/home/pingji/apps/minimind")
from mini_agent import (SYS_AGENT, WS, build_tasks, setup,  # noqa
                        DocSearch, gen, extract_json, TOOLS,
                        t_read_file, t_write_file, t_list_dir, t_calc, t_now)

MAX_STEPS = 6


def first_json(text):
    """只取第一个完整 JSON 对象（防一次吐两个）"""
    i = text.find("{")
    if i < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[i:])
        return obj
    except Exception:
        return None


def files_in_task(task):
    return re.findall(r"[\w\-.]+\.(?:txt|csv|json|md|log)", task)


def task_check(i, ans, tasks):
    """套用修正后的判据（修掉 KV/37*18/data.csv 三处）"""
    t = tasks[i]
    task = t["task"]
    ok = t["check"](ans, None)
    if "KV cache 每 token" in task:
        m = re.search(r"\d+", str(ans))
        return bool(m and int(m.group()) in (112, 116))
    if "计算 37*18" in task:
        nums = re.findall(r"\d+", str(ans))
        return bool(nums and int(nums[-1]) == 666 and os.path.exists(f"{WS}/math.txt"))
    if "data.csv 里 a*b" in task:
        m = re.search(r"\d+", str(ans))
        return bool(m and int(m.group()) == 68)
    return ok


GUARD = {"reject_finish": 0, "bad_json": 0, "not_found": 0, "repeat_read": 0}


def run_guarded(model, tok, ds, task):
    msgs = [{"role": "system", "content": SYS_AGENT},
            {"role": "user", "content": f"任务：{task}"}]
    need = files_in_task(task)
    read = {}
    for step in range(MAX_STEPS):
        raw = gen(model, tok, msgs)
        obj = first_json(raw)
        if not obj or "tool" not in obj:
            GUARD["bad_json"] += 1
            msgs.append({"role": "assistant", "content": raw[:200]})
            msgs.append({"role": "user",
                         "content": "<observation>输出不是合法 JSON。请只输出一个 JSON 对象，重试。</observation>"})
            continue
        tool, args = obj["tool"], obj.get("arguments") or {}

        if tool == "finish":
            missing = [f for f in need if f not in read]
            if missing:
                GUARD["reject_finish"] += 1
                msgs.append({"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)})
                msgs.append({"role": "user",
                             "content": f"<observation>你还没有读取 {missing}，不能直接回答。请先读取。</observation>"})
                continue
            return str(args.get("answer", "")), step + 1, "正常结束"

        try:
            if tool == "read_file":
                p = str(args.get("path", ""))
                if p in read:
                    GUARD["repeat_read"] += 1
                    obs = f"（已读过）{read[p]}"
                else:
                    obs = t_read_file(path=p)
                    read[os.path.basename(p)] = obs
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
        except FileNotFoundError:
            GUARD["not_found"] += 1
            obs = f"文件不存在。当前可用文件：{', '.join(sorted(os.listdir(WS)))}"
        except Exception as e:
            obs = f"工具报错：{type(e).__name__}: {str(e)[:60]}"

        msgs.append({"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)})
        msgs.append({"role": "user", "content": f"<observation>{obs}</observation>"})
    return None, MAX_STEPS, "超步数"


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    setup()
    ds = DocSearch()
    tasks = build_tasks()
    tok = AutoTokenizer.from_pretrained(
        "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master",
        dtype=__import__("torch").bfloat16, device_map={"": 0}).eval()

    ok = 0
    fails = []
    for i, t in enumerate(tasks):
        ans, steps, why = run_guarded(model, tok, ds, t["task"])
        good = ans is not None and task_check(i, ans, tasks)
        ok += good
        if not good:
            fails.append(dict(task=t["task"], ans=ans, steps=steps, why=why))
        print(f"  [{i+1:2d}] {'✅' if good else '❌'} ({steps} 步) {t['task'][:30]}", flush=True)

    n = len(tasks)
    print(f"\n=== 有护栏：{ok}/{n} = {ok/n*100:.0f}% ===")
    print(f"  对照（无护栏，修正判据）：13/{n} = 65%")
    print(f"  护栏触发次数：{GUARD}")
    json.dump(dict(ok=ok / n, fails=fails), open("/home/pingji/apps/minimind/agent_guard.json", "w"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
