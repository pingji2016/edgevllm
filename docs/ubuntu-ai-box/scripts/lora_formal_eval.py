"""LoRA 正式评估：留出集 eval loss + 生成答案长度分布 + 定性对比"""
import json
import os
import sys
import time

import torch

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master"
ADAPTER = os.path.expanduser("~/apps/infer-research/lora_llm")
DATA = os.path.expanduser("~/apps/minimind/dataset/sft_t2t_mini.jsonl")
OUT = os.path.expanduser("~/apps/infer-research/lora_formal_eval.json")
SKIP = 6000      # 跳过训练用过的前若干条
N_EVAL = 150
PROMPTS = [
    "你是谁？请用一句话介绍自己。",
    "用一句话解释什么是显存带宽。",
    "把这句话改得更礼貌：把文件发给我。",
    "用 Python 写一个判断回文串的函数。",
    "推荐三本适合入门机器学习的书。",
]


def load_eval(tok, n, skip):
    items, seen = [], 0
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                conv = d.get("conversations") or []
            except Exception:
                continue
            if len(conv) < 2 or conv[-1].get("role") != "assistant":
                continue
            ans = (conv[-1].get("content") or "").strip()
            if not (10 <= len(ans) <= 400):
                continue
            seen += 1
            if seen <= skip:
                continue
            msgs = [{"role": c["role"], "content": c.get("content") or ""} for c in conv[:-1]]
            try:
                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
            except Exception:
                continue
            items.append((prompt, ans))
            if len(items) >= n:
                break
    return items


def eval_loss(model, tok, items, bs=2, seq=512):
    total, count = 0.0, 0
    for i in range(0, len(items), bs):
        batch = items[i:i + bs]
        ids, labels, mask = [], [], []
        for prompt, ans in batch:
            p = tok(prompt, add_special_tokens=False).input_ids
            a = tok(ans + tok.eos_token, add_special_tokens=False).input_ids
            x = (p + a)[:seq]
            y = ([-100] * len(p) + a)[:seq]
            ids.append(x); labels.append(y); mask.append([1] * len(x))
        L = max(len(x) for x in ids)
        ids = [x + [tok.pad_token_id] * (L - len(x)) for x in ids]
        labels = [y + [-100] * (L - len(y)) for y in labels]
        mask = [m + [0] * (L - len(m)) for m in mask]
        out = model(input_ids=torch.tensor(ids).cuda(), attention_mask=torch.tensor(mask).cuda(),
                    labels=torch.tensor(labels).cuda())
        total += out.loss.item() * len(batch)
        count += len(batch)
    return total / max(count, 1)


def main():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MD)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    items = load_eval(tok, N_EVAL, SKIP)
    print(f"留出集 {len(items)} 条（跳过前 {SKIP} 条训练样本）")

    def run(use_lora, gen_prompts=False):
        m = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16, device_map={"": 0}).eval()
        if use_lora:
            m = PeftModel.from_pretrained(m, ADAPTER).eval()
        loss = eval_loss(m, tok, items)
        outs, lens, secs = [], [], []

        def chat(p):
            text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
            enc = tok(text, return_tensors="pt").to("cuda")
            n_in = enc["input_ids"].size(1)
            t0 = time.time()
            with torch.inference_mode():
                o = m.generate(**enc, max_new_tokens=96, do_sample=False,
                               pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize()
            return tok.decode(o[0, n_in:], skip_special_tokens=True).strip(), time.time() - t0

        if gen_prompts:
            for p in PROMPTS:
                t, dt = chat(p)
                outs.append(t); lens.append(len(t)); secs.append(dt)
        tokps = 96 * len(secs) / max(sum(secs), 1e-9) if secs else 0
        del m
        torch.cuda.empty_cache()
        return dict(loss=round(loss, 4), ppl=round(float(torch.tensor(loss).exp()), 2),
                    outputs=outs, avg_len=round(sum(lens) / len(lens), 1) if lens else 0,
                    tok_per_s=round(tokps, 1))

    base = run(False, True)
    lora = run(True, True)

    print(f"\n{'模型':8s} {'eval loss':>10s} {'PPL':>8s} {'平均答案长度':>12s}")
    print(f"{'base':8s} {base['loss']:10.4f} {base['ppl']:8.2f} {base['avg_len']:12.1f}")
    print(f"{'lora400':8s} {lora['loss']:10.4f} {lora['ppl']:8.2f} {lora['avg_len']:12.1f}")

    print("\n=== 定性对比 ===")
    for p, b, l in zip(PROMPTS, base["outputs"], lora["outputs"]):
        print(f"\n[问] {p}")
        print(f"[base] {b[:90]}")
        print(f"[lora] {l[:90]}")

    json.dump(dict(n_eval=len(items), skip=SKIP, base=base, lora=lora),
              open(OUT, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
