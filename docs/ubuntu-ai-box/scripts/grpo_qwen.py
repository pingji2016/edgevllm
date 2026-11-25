"""GRPO on Qwen3-1.7B：三位数 × 两位数（基线 52.5%）

训练方式：LoRA（r=16，注意力+MLP 全部投影），基座冻结
KL 参考：同一模型的「关闭 adapter」版本（不额外占一份权重）
奖励：精确 +1.0；相对误差 <0.1% +0.5；<1% +0.2；<5% +0.05
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
OUT = os.path.expanduser("~/apps/minimind/out_grpo_qwen")
SYS = ("你是一个计算器。只输出结果，格式：<answer>数字</answer>\n\n"
       "示例：\n3+4=<answer>7</answer>\n8+5=<answer>13</answer>\n6+2=<answer>8</answer>")
NUM = re.compile(r"-?\d+")

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--group", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--kl", type=float, default=0.05)
ap.add_argument("--micro", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)


def build(a, b):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{a}*{b}=<|im_end|>\n"
            f"<|im_start|>assistant\n<answer>")


def reward(pred, gold):
    if pred is None:
        return 0.0
    if pred == gold:
        return 1.0
    rel = abs(pred - gold) / max(1, gold)
    return 0.5 if rel < 1e-3 else (0.2 if rel < 1e-2 else (0.05 if rel < 0.05 else 0.0))


def eval_set(n=200, seed=4242):
    rng = random.Random(seed)
    return [(rng.randint(100, 999), rng.randint(10, 99)) for _ in range(n)]


@torch.no_grad()
def evaluate(model, tok, pairs):
    model.eval()
    right = near = 0
    for a, b in pairs:
        ids = tok(build(a, b), return_tensors="pt").input_ids.cuda()
        g = model.generate(ids, max_new_tokens=12, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        t = tok.decode(g[0, ids.size(1):], skip_special_tokens=True)
        m = NUM.search(t)
        p = int(m.group()) if m else None
        right += p == a * b
        near += p is not None and abs(p - a * b) / (a * b) < 1e-2
    model.train()
    return dict(acc=right / len(pairs), near=near / len(pairs))


def main():
    tok = AutoTokenizer.from_pretrained(MD)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16,
                                                device_map={"": 0})
    model = get_peft_model(base, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    print(f"  加载 {time.time()-t0:.0f}s", flush=True)

    pairs_final = eval_set(200)
    pairs_eval = pairs_final[:100]
    base_m = evaluate(model, tok, pairs_final)
    base_eval = evaluate(model, tok, pairs_eval)
    print(f"  基线（200 题）：精确 {base_m['acc']*100:.1f}% | 误差<1% {base_m['near']*100:.1f}%", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    curve = [dict(step=0, eval=True, on200=base_m, on100=base_eval)]
    t0 = time.time()

    for step in range(1, args.steps + 1):
        pairs = [(random.randint(100, 999), random.randint(10, 99)) for _ in range(args.batch)]
        prompts = [build(a, b) for a, b in pairs]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")

        model.eval()
        with torch.no_grad():
            gen = model.generate(enc.input_ids, attention_mask=enc.attention_mask,
                                 max_new_tokens=12, do_sample=True, temperature=1.0,
                                 top_p=0.95, num_return_sequences=args.group,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        model.train()

        plen = enc.input_ids.size(1)
        comps, rs, hit, near = [], [], 0, 0
        for i, (a, b) in enumerate(pairs):
            for j in range(args.group):
                comp = gen[i * args.group + j][plen:]
                keep = (comp != tok.pad_token_id).sum().item()
                comp = comp[:keep]
                t = tok.decode(comp, skip_special_tokens=True)
                m = NUM.search(t)
                p = int(m.group()) if m else None
                rs.append(reward(p, a * b))
                hit += p == a * b
                near += p is not None and abs(p - a * b) / (a * b) < 1e-2
                comps.append(comp)
        del gen

        full = [prompts[i // args.group] + tok.decode(comps[i], skip_special_tokens=True)
                for i in range(len(comps))]
        enc2 = tok(full, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        ids, am = enc2.input_ids, enc2.attention_mask
        mask = torch.zeros((len(comps), ids.size(1) - 1), device="cuda")
        for i, c in enumerate(comps):
            if len(c):
                mask[i, -len(c):] = 1.0
        tok_per_seq = mask.sum(1).clamp(min=1)

        r = torch.tensor(rs, device="cuda")
        g = r.view(args.batch, args.group)
        adv = ((g - g.mean(1, keepdim=True)) /
               (g.std(1, keepdim=True, unbiased=False) + 1e-4)).reshape(-1)
        adv = torch.nan_to_num(adv)

        opt.zero_grad()
        N = len(comps)
        for s in range(0, N, args.micro):
            sl = slice(s, s + args.micro)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids[sl], attention_mask=am[sl]).logits
                logp = F.log_softmax(logits[:, :-1].float(), -1).gather(
                    2, ids[sl, 1:].unsqueeze(-1)).squeeze(-1)
            seq_logp = (logp * mask[sl]).sum(1)
            part = -(adv[sl].detach() * seq_logp / tok_per_seq[sl]).sum() / N
            if args.kl > 0:
                with torch.no_grad(), model.disable_adapter():
                    rl = model(ids[sl], attention_mask=am[sl]).logits
                    rlp = F.log_softmax(rl[:, :-1].float(), -1).gather(
                        2, ids[sl, 1:].unsqueeze(-1)).squeeze(-1)
                kl = ((logp - rlp) * mask[sl]).sum() / tok_per_seq[sl].sum()
                part = part + args.kl * kl
                del rl, rlp
            part.backward()
            del logits, logp, seq_logp, part

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        if step % 10 == 0:
            print(f"  step {step:4d} | reward {r.mean():+.3f} | 组内精确 {hit/len(rs)*100:5.1f}% | "
                  f"<1% {near/len(rs)*100:5.1f}% | {(time.time()-t0)/step:.1f}s/步", flush=True)
            curve.append(dict(step=step, reward=r.mean().item(),
                              sample_acc=hit / len(rs), sample_near=near / len(rs)))
        if step % 50 == 0:
            ev = evaluate(model, tok, pairs_eval)
            print(f"  --- eval@{step}: 精确 {ev['acc']*100:.1f}% | <1% {ev['near']*100:.1f}% ---", flush=True)
            curve.append(dict(step=step, eval=True, **ev))

    final = evaluate(model, tok, pairs_final)
    print(f"\n=== 最终（200 道新题）===")
    print(f"  基线 精确 {base_m['acc']*100:.1f}% | <1% {base_m['near']*100:.1f}%（同一组 200 题）")
    print(f"  训练后 精确 {final['acc']*100:.1f}% | 误差<1% {final['near']*100:.1f}%")

    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT)
    json.dump(dict(steps=args.steps, base=base_m, final=final, curve=curve),
              open(os.path.join(OUT, "curve.json"), "w"), ensure_ascii=False, indent=2)
    print(f"  LoRA 已存 {OUT}")


if __name__ == "__main__":
    main()
