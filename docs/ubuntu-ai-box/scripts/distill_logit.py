"""logit（KL）蒸馏 vs hard-label 蒸馏

同一批教师正确样本（2059 条），同一学生初始化，只改损失：
  H  只做 CE（复现 46.0%）
  KL CE + T²·KL(teacher‖student)，T=2，权重 0.5

评测同一 200 题，保存逐题对错，做 McNemar
"""
import json
import math
import os
import random
import re
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

T17 = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
T06 = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master"
LORA = "/home/pingji/apps/minimind/out_grpo_qwen"
OUT = os.path.expanduser("~/apps/minimind/out_distill_logit.json")
NUM = re.compile(r"-?\d+")
SYS = ("你是一个计算器。只输出结果，格式：<answer>数字</answer>\n\n"
       "示例：\n3+4=<answer>7</answer>\n8+5=<answer>13</answer>\n6+2=<answer>8</answer>")

N_TRAIN, N_SAMPLE = 3000, 8
EPOCHS, BS, LR = 3.0, 8, 2e-5
TEMP, ALPHA = 2.0, 0.5
random.seed(0)
torch.manual_seed(0)


def prompt(a, b):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{a}*{b}=<|im_end|>\n<|im_start|>assistant\n<answer>")


def make_tok(p):
    t = AutoTokenizer.from_pretrained(p)
    t.padding_side = "left"
    if t.pad_token_id is None:
        t.pad_token = t.eos_token
    return t


@torch.no_grad()
def collect(model, tok, pairs, k=N_SAMPLE):
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
    return data


def encode(tok, data):
    out = []
    for p, c in data:
        pid = tok(p, add_special_tokens=False).input_ids
        cid = tok(c, add_special_tokens=False).input_ids
        out.append((pid + cid, [-100] * len(pid) + cid))
    return out


def collate(batch, pad_id):
    L = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for j, (x, y) in enumerate(batch):
        ids[j, :len(x)] = torch.tensor(x)
        lab[j, :len(y)] = torch.tensor(y)
        att[j, :len(x)] = 1
    return ids.cuda(), lab.cuda(), att.cuda()


def train(student, tok, data, use_kl, teacher=None, tag=""):
    proc = encode(tok, data)
    opt = torch.optim.AdamW([x for x in student.parameters() if x.requires_grad],
                            lr=LR, weight_decay=0.0)
    rng = random.Random(0)
    t0 = time.time()
    step = 0
    kl_hist = []
    for ep in range(int(EPOCHS)):
        rng.shuffle(proc)
        for i in range(0, len(proc), BS):
            ids, lab, att = collate(proc[i:i + BS], tok.pad_token_id)
            shift_lab = lab[:, 1:]
            mask = shift_lab != -100
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s_logits = student(input_ids=ids, attention_mask=att).logits[:, :-1]
            s_sel = s_logits[mask].float()
            ce = F.cross_entropy(s_sel, shift_lab[mask])
            loss = ce
            if use_kl:
                with torch.no_grad():
                    t_logits = teacher(input_ids=ids, attention_mask=att).logits[:, :-1]
                t_sel = t_logits[mask].float()
                sl = F.log_softmax(s_sel / TEMP, -1)
                tl = F.log_softmax(t_sel / TEMP, -1)
                kl = F.kl_div(sl, tl, log_target=True, reduction="batchmean")
                loss = (1 - ALPHA) * ce + ALPHA * TEMP * TEMP * kl
                kl_hist.append(kl.item())
                del t_logits, t_sel, sl, tl, kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [x for x in student.parameters() if x.requires_grad], 1.0)
            opt.step()
            step += 1
            del s_logits, s_sel, ce
            if step % 100 == 0:
                k = f" | KL {sum(kl_hist[-50:])/len(kl_hist[-50:]):.3f}" if kl_hist else ""
                print(f"    [{tag}] step {step} loss {loss.item():.4f}{k} "
                      f"{(time.time()-t0)/step:.2f}s/步", flush=True)
    return time.time() - t0


@torch.no_grad()
def evaluate(model, tok, pairs):
    model.eval()
    ok, near = [], []
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
            ok.append(p == a * b)
            near.append(p is not None and abs(p - a * b) / (a * b) < 1e-2)
        del g
    model.train()
    return ok, near


def new_student(tok):
    base = AutoModelForCausalLM.from_pretrained(T06, dtype=torch.bfloat16, device_map={"": 0})
    return get_peft_model(base, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))


def main():
    tok = make_tok(T06)
    rng = random.Random(2027)
    train_pairs = [(rng.randint(100, 999), rng.randint(10, 99)) for _ in range(N_TRAIN)]
    rng2 = random.Random(999)
    ev = [(rng2.randint(100, 999), rng2.randint(10, 99)) for _ in range(200)]

    print("  加载教师 1.7B + GRPO LoRA…", flush=True)
    teacher = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(T17, dtype=torch.bfloat16, device_map={"": 0}),
        LORA).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print(f"  采样教师（{N_TRAIN} 题 × {N_SAMPLE}）…", flush=True)
    t0 = time.time()
    data = collect(teacher, tok, train_pairs)
    print(f"  样本 {len(data)} 条 | {time.time()-t0:.0f}s", flush=True)

    res, items = {}, {}
    for tag, use_kl in (("H_ce", False), ("KL", True)):
        print(f"\n  === 学生 {tag} ===", flush=True)
        student = new_student(tok)
        dur = train(student, tok, data, use_kl, teacher if use_kl else None, tag)
        ok, near = evaluate(student, tok, ev)
        res[tag] = dict(acc=sum(ok) / len(ok), near=sum(near) / len(near),
                        train_min=round(dur / 60, 1))
        items[tag] = ok
        print(f"    {tag}: {sum(ok)/len(ok)*100:.1f}% | <1% {sum(near)/len(near)*100:.1f}% "
              f"| {dur/60:.1f} min", flush=True)
        del student
        torch.cuda.empty_cache()

    a, b = items["H_ce"], items["KL"]
    only_h = sum(1 for x, y in zip(a, b) if x and not y)
    only_kl = sum(1 for x, y in zip(a, b) if y and not x)
    z = (only_h - only_kl) / math.sqrt(only_h + only_kl) if only_h + only_kl else 0
    print(f"\n=== 配对（200 题）===")
    print(f"  只有 CE 对 {only_h} | 只有 KL 对 {only_kl} | McNemar z = {z:.2f}")

    json.dump(dict(results=res, only_ce=only_h, only_kl=only_kl, mcnemar_z=z,
                   n=len(ev), per_item=items), open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"已存 {OUT}")


if __name__ == "__main__":
    main()
