"""RFT（拒绝采样微调 / STaR）在 Qwen3-1.7B 上做三位×两位

流程：
  1. 基座在 2000 道训练题上每题采样 16 个答案
  2. 只保留答对的样本（拒绝采样）
  3. LoRA SFT（只在答案部分算 loss）
  4. 在固定的 200 题评测集（seed 999）上贪心评测

对照：同评测集上 基座 47.0% / GRPO 60.0%
"""
import argparse
import json
import os
import random
import re
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
OUT = os.path.expanduser("~/apps/minimind/out_rft_qwen")
NUM = re.compile(r"-?\d+")
SYS = ("你是一个计算器。只输出结果，格式：<answer>数字</answer>\n\n"
       "示例：\n3+4=<answer>7</answer>\n8+5=<answer>13</answer>\n6+2=<answer>8</answer>")

ap = argparse.ArgumentParser()
ap.add_argument("--train_n", type=int, default=2000)
ap.add_argument("--samples", type=int, default=16)
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--seed", type=int, default=1234)
args = ap.parse_args()

random.seed(0)
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MD)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token


def prompt(a, b):
    return tok.apply_chat_template([{"role": "system", "content": SYS},
                                    {"role": "user", "content": f"{a}*{b}="}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) + "<answer>"


def eval_pairs(n=200, seed=999):
    rng = random.Random(seed)
    return [(rng.randint(100, 999), rng.randint(10, 99)) for _ in range(n)]


@torch.no_grad()
def sample_correct(model, pairs, k):
    kept, stats = [], dict(problems=len(pairs), with_correct=0, samples=0)
    for i in range(0, len(pairs), 16):
        chunk = pairs[i:i + 16]
        enc = tok([prompt(a, b) for a, b in chunk], return_tensors="pt",
                  padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=12, do_sample=True, temperature=0.8,
                           top_p=0.95, num_return_sequences=k,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        pl = enc.input_ids.size(1)
        for j, (a, b) in enumerate(chunk):
            got = False
            for s in range(k):
                txt = tok.decode(g[j * k + s][pl:], skip_special_tokens=True)
                m = NUM.search(txt)
                if m and int(m.group()) == a * b:
                    kept.append((prompt(a, b), txt))
                    got = True
            stats["with_correct"] += got
        stats["samples"] = len(kept)
        del g
    return kept, stats


@torch.no_grad()
def evaluate(model, pairs):
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


def encode(entries):
    data = []
    for p, c in entries:
        pid = tok(p, add_special_tokens=False).input_ids
        cid = tok(c, add_special_tokens=False).input_ids
        data.append((pid + cid, [-100] * len(pid) + cid))
    return data


def main():
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16, device_map={"": 0})
    model = get_peft_model(base, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    print(f"  加载 {time.time()-t0:.0f}s", flush=True)

    ep = eval_pairs()
    b = evaluate(model, ep)
    print(f"  基座（200 题）：精确 {b['acc']*100:.1f}% | <1% {b['near']*100:.1f}%", flush=True)

    rng = random.Random(args.seed)
    train_pairs = [(rng.randint(100, 999), rng.randint(10, 99)) for _ in range(args.train_n)]
    print(f"\n  采样 {args.train_n} 题 × {args.samples} = "
          f"{args.train_n*args.samples} 次…", flush=True)
    t0 = time.time()
    kept, st = sample_correct(model, train_pairs, args.samples)
    print(f"  采样完成 {time.time()-t0:.0f}s | 有正确样本的题 {st['with_correct']}/{st['problems']} "
          f"({st['with_correct']/st['problems']*100:.1f}%) | 保留样本 {len(kept)} 条", flush=True)

    data = encode(kept)
    print(f"\n  开始 SFT：{len(data)} 条，{args.epochs} epoch，lr {args.lr}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    t0 = time.time()
    step = 0
    total_steps = int(len(data) / args.bs * args.epochs)
    for epoch in range(max(1, int(args.epochs))):
        rng.shuffle(data)
        for i in range(0, len(data), args.bs):
            batch = data[i:i + args.bs]
            L = max(len(x[0]) for x in batch)
            ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
            lab = torch.full((len(batch), L), -100, dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for j, (x, y) in enumerate(batch):
                ids[j, L - len(x):] = torch.tensor(x)
                lab[j, L - len(x):] = torch.tensor(y)
                att[j, L - len(x):] = 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids.cuda(), attention_mask=att.cuda(),
                             labels=lab.cuda()).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"    step {step}/{total_steps} | loss {loss.item():.4f} | "
                      f"{(time.time()-t0)/step:.2f}s/步", flush=True)

    f = evaluate(model, ep)
    print(f"\n=== 最终（同一 200 题）===")
    print(f"  基座 精确 {b['acc']*100:.1f}% | <1% {b['near']*100:.1f}%")
    print(f"  RFT  精确 {f['acc']*100:.1f}% | <1% {f['near']*100:.1f}%")
    print(f"  （对照：GRPO 300 步 = 60.0%）")

    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT)
    json.dump(dict(base=b, rft=f, stats=st, kept=len(kept),
                   train_n=args.train_n, samples=args.samples, epochs=args.epochs),
              open(os.path.join(OUT, "result.json"), "w"), ensure_ascii=False, indent=2)
    print(f"  LoRA 已存 {OUT}")


if __name__ == "__main__":
    main()
