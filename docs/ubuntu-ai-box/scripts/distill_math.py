"""蒸馏：1.7B 教师 → 0.6B 学生（三位×两位）

两个教师：
  A 基座 1.7B          47%
  B 基座 + GRPO LoRA   60%

问题：学生能继承多少？RL 的 +13pp 能传下去吗？

流程：教师对 3000 题各采样 8 次 → 每题保留 1 条正确样本（共 ~3000 条）
      → 0.6B LoRA SFT（只对答案算 loss）→ 同一 200 题评测
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
OUT = os.path.expanduser("~/apps/minimind/out_distill.json")
NUM = re.compile(r"-?\d+")
SYS = ("你是一个计算器。只输出结果，格式：<answer>数字</answer>\n\n"
       "示例：\n3+4=<answer>7</answer>\n8+5=<answer>13</answer>\n6+2=<answer>8</answer>")

N_TRAIN, N_SAMPLE = 3000, 8
EPOCHS, BS, LR = 3.0, 16, 2e-5
random.seed(0)


def make_tok(path):
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def prompt(a, b):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{a}*{b}=<|im_end|>\n<|im_start|>assistant\n<answer>")


@torch.no_grad()
def collect(model, tok, pairs, k=N_SAMPLE):
    """每题保留 1 条正确样本"""
    data, hits = [], 0
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
                    hits += 1
                    break
        del g
    return data, hits


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
    model.train()
    return dict(acc=right / len(pairs), near=near / len(pairs))


def sft(student, tok, data, tag):
    proc = []
    for p, c in data:
        pid = tok(p, add_special_tokens=False).input_ids
        cid = tok(c, add_special_tokens=False).input_ids
        proc.append((pid + cid, [-100] * len(pid) + cid))
    opt = torch.optim.AdamW([x for x in student.parameters() if x.requires_grad],
                            lr=LR, weight_decay=0.0)
    rng = random.Random(0)
    t0 = time.time()
    step = 0
    total = int(len(proc) / BS * EPOCHS)
    for ep in range(int(EPOCHS)):
        rng.shuffle(proc)
        for i in range(0, len(proc), BS):
            batch = proc[i:i + BS]
            L = max(len(x[0]) for x in batch)
            ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
            lab = torch.full((len(batch), L), -100, dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for j, (x, y) in enumerate(batch):
                ids[j, L - len(x):] = torch.tensor(x)
                lab[j, L - len(x):] = torch.tensor(y)
                att[j, L - len(x):] = 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = student(input_ids=ids.cuda(), attention_mask=att.cuda(),
                               labels=lab.cuda()).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [x for x in student.parameters() if x.requires_grad], 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                print(f"    [{tag}] step {step}/{total} loss {loss.item():.4f} "
                      f"{(time.time()-t0)/step:.2f}s/步", flush=True)
    return time.time() - t0


def new_student(tok06):
    base = AutoModelForCausalLM.from_pretrained(T06, dtype=torch.bfloat16,
                                                device_map={"": 0})
    return get_peft_model(base, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))


def main():
    tok17, tok06 = make_tok(T17), make_tok(T06)
    rng = random.Random(2027)
    train_pairs = [(rng.randint(100, 999), rng.randint(10, 99)) for _ in range(N_TRAIN)]
    ev = [(random.Random(999).randint(100, 999), 0) for _ in range(1)]  # 占位
    rng2 = random.Random(999)
    eval_pairs = [(rng2.randint(100, 999), rng2.randint(10, 99)) for _ in range(200)]

    # 教师：基座 + GRPO LoRA
    print("  加载教师 1.7B + GRPO LoRA…", flush=True)
    teacher = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(T17, dtype=torch.bfloat16, device_map={"": 0}),
        LORA).eval()
    teacher_tok = tok17

    print(f"\n  [教师 B] GRPO 教师采样 {N_TRAIN}×{N_SAMPLE}…", flush=True)
    t0 = time.time()
    data_rl, hit_rl = collect(teacher, teacher_tok, train_pairs)
    print(f"    有正确样本的题 {hit_rl}/{N_TRAIN} | 用时 {time.time()-t0:.0f}s", flush=True)

    print(f"\n  [教师 A] 基座教师采样…", flush=True)
    t0 = time.time()
    with teacher.disable_adapter():
        data_base, hit_base = collect(teacher, teacher_tok, train_pairs)
    print(f"    有正确样本的题 {hit_base}/{N_TRAIN} | 用时 {time.time()-t0:.0f}s", flush=True)

    del teacher
    torch.cuda.empty_cache()

    results = {}
    for tag, data in (("distill_base", data_base), ("distill_rl", data_rl)):
        print(f"\n  === 训练学生 {tag}（{len(data)} 条）===", flush=True)
        student = new_student(tok06)
        b = evaluate(student, tok06, eval_pairs)
        print(f"    0.6B 基线：{b['acc']*100:.1f}% | <1% {b['near']*100:.1f}%", flush=True)
        dur = sft(student, tok06, data, tag)
        f = evaluate(student, tok06, eval_pairs)
        print(f"    蒸馏后：{f['acc']*100:.1f}% | <1% {f['near']*100:.1f}% | 训练 {dur/60:.1f} min",
              flush=True)
        results[tag] = dict(base=b, after=f, n=len(data), train_min=round(dur / 60, 1))
        del student
        torch.cuda.empty_cache()

    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"\n已存 {OUT}")


if __name__ == "__main__":
    main()
