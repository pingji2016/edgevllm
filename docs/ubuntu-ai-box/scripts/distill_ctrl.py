"""控制实验：RL 教师的数据截到与基座教师等量（1598 条）
  若仍得 46% -> 增益来自数据质量
  若降到 44% -> 增益只是数据数量
"""
import json
import os
import random
import re
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

T17 = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
T06 = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master"
LORA = "/home/pingji/apps/minimind/out_grpo_qwen"
NUM = re.compile(r"-?\d+")
SYS = ("你是一个计算器。只输出结果，格式：<answer>数字</answer>\n\n"
       "示例：\n3+4=<answer>7</answer>\n8+5=<answer>13</answer>\n6+2=<answer>8</answer>")
CAP = 1598


def make_tok(p):
    t = AutoTokenizer.from_pretrained(p)
    t.padding_side = "left"
    if t.pad_token_id is None:
        t.pad_token = t.eos_token
    return t


def prompt(a, b):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{a}*{b}=<|im_end|>\n<|im_start|>assistant\n<answer>")


@torch.no_grad()
def collect(model, tok, pairs, k=8, cap=CAP):
    data = []
    for i in range(0, len(pairs), 16):
        chunk = pairs[i:i + 16]
        enc = tok([prompt(a, b) for a, b in chunk], return_tensors="pt",
                  padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=12, do_sample=True, temperature=1.0,
                           top_p=0.95, num_return_sequences=k,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        pl = enc.input_ids.size(1)
        for j, (a, b) in enumerate(chunk):
            for s in range(k):
                t = tok.decode(g[j * k + s][pl:], skip_special_tokens=True)
                m = NUM.search(t)
                if m and int(m.group()) == a * b:
                    data.append((prompt(a, b), t))
                    break
        del g
        if len(data) >= cap:
            break
    return data[:cap]


@torch.no_grad()
def evaluate(model, tok, pairs):
    model.eval()
    right = near = 0
    for i in range(0, len(pairs), 16):
        chunk = pairs[i:i + 16]
        enc = tok([prompt(a, b) for a, b in chunk], return_tensors="pt",
                  padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=12, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        pl = enc.input_ids.size(1)
        for j, (a, b) in enumerate(chunk):
            t = tok.decode(g[j, pl:], skip_special_tokens=True)
            m = NUM.search(t)
            p = int(m.group()) if m else None
            right += p == a * b
            near += p is not None and abs(p - a * b) / (a * b) < 1e-2
        del g
    return dict(acc=right / len(pairs), near=near / len(pairs))


def main():
    tok17, tok06 = make_tok(T17), make_tok(T06)
    rng = random.Random(2027)
    pairs = [(rng.randint(100, 999), rng.randint(10, 99)) for _ in range(3000)]
    rng2 = random.Random(999)
    ev = [(rng2.randint(100, 999), rng2.randint(10, 99)) for _ in range(200)]

    teacher = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(T17, dtype=torch.bfloat16, device_map={"": 0}),
        LORA).eval()
    print(f"  采样 RL 教师（截到 {CAP} 条）…", flush=True)
    t0 = time.time()
    data = collect(teacher, tok17, pairs)
    print(f"  拿到 {len(data)} 条 | {time.time()-t0:.0f}s", flush=True)
    del teacher
    torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(T06, dtype=torch.bfloat16, device_map={"": 0})
    student = get_peft_model(base, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))

    proc = []
    for p, c in data:
        pid = tok06(p, add_special_tokens=False).input_ids
        cid = tok06(c, add_special_tokens=False).input_ids
        proc.append((pid + cid, [-100] * len(pid) + cid))
    opt = torch.optim.AdamW([x for x in student.parameters() if x.requires_grad],
                            lr=2e-5, weight_decay=0.0)
    rng3 = random.Random(0)
    t0 = time.time()
    step = 0
    for ep in range(3):
        rng3.shuffle(proc)
        for i in range(0, len(proc), 16):
            batch = proc[i:i + 16]
            L = max(len(x[0]) for x in batch)
            ids = torch.full((len(batch), L), tok06.pad_token_id, dtype=torch.long)
            lab = torch.full((len(batch), L), -100, dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for j, (x, y) in enumerate(batch):
                ids[j, L - len(x):] = torch.tensor(x)
                lab[j, L - len(x):] = torch.tensor(y)
                att[j, L - len(x):] = 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = student(input_ids=ids.cuda(), attention_mask=att.cuda(),
                               labels=lab.cuda()).loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [x for x in student.parameters() if x.requires_grad], 1.0)
            opt.step(); step += 1
    print(f"  训练 {step} 步，{time.time()-t0:.0f}s", flush=True)

    f = evaluate(student, tok06, ev)
    print(f"\n=== RL 教师（等量 {len(data)} 条）===")
    print(f"  准确率 {f['acc']*100:.1f}% | <1% {f['near']*100:.1f}%")
    print(f"  对照：基座教师 1598 条 -> 44.0% / 96.5%")
    print(f"        RL 教师 2059 条 -> 46.0% / 98.0%")
    json.dump(dict(n=len(data), **f),
              open(os.path.expanduser("~/apps/minimind/out_distill_ctrl.json"), "w"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
