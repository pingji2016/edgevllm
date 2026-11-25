"""RLVR：30M 模型能不能用 GRPO 学会一位数加法？

关键设计（因为基线完全不会）：
  1. 强制前缀：prompt 以 "<answer>" 结尾，模型只需生成数字（≤8 token）
  2. 距离塑形奖励：|pred-gold|=0 → +1.0，=1 → +0.5，≤3 → +0.2，否则 0
  3. 对照组（--control random）：奖励对着**与题目无关的随机数**算，
     用来证明"涨点不是训练本身/随机波动造成的"

判据：200 道留出题的贪心准确率（随机猜的期望 ≈ 1/19 ≈ 5.3%）
"""
import argparse
import json
import os
import random
import re
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.expanduser("~/apps/minimind")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SYS = ("你是一个计算器。只输出结果，格式：<answer>数字</answer>\n\n"
       "示例：\n3+4=<answer>7</answer>\n8+5=<answer>13</answer>\n6+2=<answer>8</answer>")
NUM = re.compile(r"-?\d+")

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--group", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--control", default="none", choices=["none", "random"])
ap.add_argument("--ckpt", default="out_dpo/dpo_512.pth")
ap.add_argument("--tag", default="")
ap.add_argument("--kl", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "model"))
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token


def build(a, b):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{a}+{b}=<|im_end|>\n"
            f"<|im_start|>assistant\n<answer>")


def reward(pred, gold):
    if pred is None:
        return 0.0
    d = abs(pred - gold)
    return 1.0 if d == 0 else (0.5 if d == 1 else (0.2 if d <= 3 else 0.0))


@torch.no_grad()
def evaluate(model, n=200):
    model.eval()
    right = within1 = 0
    preds = []
    for i in range(n):
        a, b = (i * 7) % 10, (i * 3 + 1) % 10
        ids = tok(build(a, b), return_tensors="pt").input_ids.cuda()
        g = model.generate(ids, max_new_tokens=8, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        t = tok.decode(g[0, ids.size(1):], skip_special_tokens=True)
        m = NUM.search(t)
        p = int(m.group()) if m else None
        preds.append(p)
        right += p == a + b
        within1 += p is not None and abs(p - (a + b)) <= 1
    model.train()
    return dict(acc=right / n, within1=within1 / n,
                mean_pred=(sum(p for p in preds if p is not None) /
                           max(1, sum(1 for p in preds if p is not None))))


def main():
    cfg = MiniMindConfig(hidden_size=512, num_hidden_layers=8, use_moe=False, vocab_size=6400)
    model = MiniMindForCausalLM(cfg)
    model.load_state_dict(torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu"),
                          strict=False)
    model = model.cuda()
    ref = None
    if args.kl > 0:
        ref = MiniMindForCausalLM(cfg)
        ref.load_state_dict(torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu"),
                            strict=False)
        ref = ref.cuda().eval()
        for p_ in ref.parameters():
            p_.requires_grad_(False)
        print(f"  参考模型已加载（KL 系数 {args.kl}）", flush=True)
    base = evaluate(model)
    print(f"  基线：答对 {base['acc']*100:.1f}% | 差1以内 {base['within1']*100:.1f}% | "
          f"平均预测 {base['mean_pred']:.1f}（随机期望 5.3%）", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    curve = [dict(step=0, eval=True, **base)]
    t0 = time.time()
    model.train()

    for step in range(1, args.steps + 1):
        pairs = [(random.randint(0, 9), random.randint(0, 9)) for _ in range(args.batch)]
        prompts = [build(a, b) for a, b in pairs]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")

        with torch.no_grad():
            gen = model.generate(enc.input_ids, attention_mask=enc.attention_mask,
                                 max_new_tokens=8, do_sample=True, temperature=1.0,
                                 top_p=0.95, num_return_sequences=args.group,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)

        plen = enc.input_ids.size(1)
        comps, txts, rs, hit, near = [], [], [], 0, 0
        for i, (a, b) in enumerate(pairs):
            gold = (a + b) if args.control == "none" else random.randint(0, 18)
            for j in range(args.group):
                row = gen[i * args.group + j]
                comp = row[plen:]
                keep = (comp != tok.pad_token_id).sum().item()
                comp = comp[:keep]
                txt = tok.decode(comp, skip_special_tokens=True)
                m = NUM.search(txt)
                p = int(m.group()) if m else None
                rs.append(reward(p, gold))
                hit += p == (a + b)
                near += p is not None and abs(p - (a + b)) <= 1
                comps.append(comp)
                txts.append(txt)

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

        # 微批 + bf16 autocast，避免 256 条序列 × 6400 词表 logits 撑爆显存
        opt.zero_grad()
        N, MB = len(comps), 32
        for s in range(0, N, MB):
            sl = slice(s, s + MB)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids[sl], attention_mask=am[sl]).logits
                logp = F.log_softmax(logits[:, :-1].float(), -1).gather(
                    2, ids[sl, 1:].unsqueeze(-1)).squeeze(-1)
            seq_logp = (logp * mask[sl]).sum(1)
            part = -(adv[sl].detach() * seq_logp / tok_per_seq[sl]).sum() / N
            if ref is not None:
                with torch.no_grad():
                    rl = ref(ids[sl], attention_mask=am[sl]).logits
                    rlp = F.log_softmax(rl[:, :-1].float(), -1).gather(
                        2, ids[sl, 1:].unsqueeze(-1)).squeeze(-1)
                kl = ((logp - rlp) * mask[sl]).sum() / tok_per_seq[sl].sum()
                part = part + args.kl * kl
                del rl, rlp
            part.backward()
            del logits, logp, seq_logp, part
        loss = torch.tensor(0.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 20 == 0:
            print(f"  step {step:4d} | reward {r.mean():+.3f} | 组内答对 {hit/len(rs)*100:5.1f}% | "
                  f"差1内 {near/len(rs)*100:5.1f}% | {(time.time()-t0)/step:.2f}s/步", flush=True)
            curve.append(dict(step=step, reward=r.mean().item(),
                              sample_acc=hit / len(rs), sample_within1=near / len(rs)))
        if step % 100 == 0:
            ev = evaluate(model)
            print(f"  --- eval@{step}: 答对 {ev['acc']*100:.1f}% | "
                  f"差1内 {ev['within1']*100:.1f}% | 平均预测 {ev['mean_pred']:.1f} ---", flush=True)
            curve.append(dict(step=step, eval=True, **ev))

    final = evaluate(model)
    print(f"\n=== 最终 ===")
    print(f"  基线    答对 {base['acc']*100:.1f}% | 差1内 {base['within1']*100:.1f}%")
    print(f"  训练后  答对 {final['acc']*100:.1f}% | 差1内 {final['within1']*100:.1f}%")

    out = os.path.join(ROOT, "out_grpo")
    os.makedirs(out, exist_ok=True)
    tag = args.tag or ("control" if args.control == "random" else "main")
    torch.save(model.state_dict(), os.path.join(out, f"grpo_{tag}.pth"))
    json.dump(dict(control=args.control, steps=args.steps, base=base, final=final,
                   curve=curve), open(os.path.join(out, f"curve_{tag}.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"  已存 out_grpo/grpo_{tag}.pth")


if __name__ == "__main__":
    main()
