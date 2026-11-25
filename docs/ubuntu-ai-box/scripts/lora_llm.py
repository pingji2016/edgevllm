"""LLM LoRA 微调：Qwen3-0.6B + 本地 SFT 数据（只训最后一条 assistant 回复）

产出：ms/step、显存、loss 曲线、以及 base vs LoRA 的输出对比
"""
import json
import os
import random
import time

import torch
from torch.utils.data import DataLoader, Dataset

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master"
DATA = os.path.expanduser("~/apps/minimind/dataset/sft_t2t_mini.jsonl")
OUT = os.path.expanduser("~/apps/infer-research/lora_llm")
STEPS = int(os.environ.get("STEPS", "200"))
SEQ = int(os.environ.get("SEQ", "512"))
BS = int(os.environ.get("BS", "2"))
ACC = int(os.environ.get("ACC", "4"))
N_SAMPLES = int(os.environ.get("N", "4000"))

EVAL_PROMPTS = [
    "你是谁？请用一句话介绍自己。",
    "用一句话解释什么是显存带宽。",
    "把这句话改得更礼貌：把文件发给我。",
]


class ChatDS(Dataset):
    def __init__(self, tok, texts, seq):
        self.items = []
        for prompt_text, answer in texts:
            p = tok(prompt_text, add_special_tokens=False).input_ids
            a = tok(answer + tok.eos_token, add_special_tokens=False).input_ids
            ids = (p + a)[:seq]
            labels = ([-100] * len(p) + a)[:seq]
            if sum(1 for x in labels if x != -100) < 4:  # 目标太短就丢
                continue
            self.items.append((ids, labels))
        self.seq = seq

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id):
    maxlen = max(len(x) for x, _ in batch)
    ids, labels, mask = [], [], []
    for x, y in batch:
        n = maxlen - len(x)
        ids.append(x + [pad_id] * n)
        labels.append(y + [-100] * n)
        mask.append([1] * len(x) + [0] * n)
    return (torch.tensor(ids), torch.tensor(labels), torch.tensor(mask))


def build_data(tok):
    texts = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            if len(texts) >= N_SAMPLES:
                break
            try:
                d = json.loads(line)
                conv = d.get("conversations") or []
            except Exception:
                continue
            if len(conv) < 2 or conv[-1].get("role") != "assistant":
                continue
            answer = (conv[-1].get("content") or "").strip()
            if not answer:
                continue
            msgs = [{"role": c["role"], "content": c.get("content") or ""} for c in conv[:-1]]
            try:
                prompt_text = tok.apply_chat_template(msgs, tokenize=False,
                                                      add_generation_prompt=True)
            except Exception:
                continue
            texts.append((prompt_text, answer))
    return texts


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MD)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    t0 = time.time()
    texts = build_data(tok)
    print(f"数据: {len(texts)} 条（读取 {time.time()-t0:.1f}s）", flush=True)

    ds = ChatDS(tok, texts, SEQ)
    print(f"可用样本: {len(ds)}", flush=True)
    loader = DataLoader(ds, batch_size=BS, shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id),
                        drop_last=True, num_workers=0)

    model = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16, device_map={"": 0})
    model.requires_grad_(False)
    model.add_adapter(LoraConfig(r=8, lora_alpha=8, lora_dropout=0.05,
                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                 "gate_proj", "up_proj", "down_proj"]))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数 {n_train/1e6:.2f} M | 显存 {torch.cuda.memory_allocated()/2**20:.0f} MiB", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    losses, times = [], []
    step = 0
    it = iter(loader)
    while step < STEPS:
        opt.zero_grad(set_to_none=True)
        t0 = time.time()
        for _ in range(ACC):
            try:
                ids, labels, mask = next(it)
            except StopIteration:
                it = iter(loader)
                ids, labels, mask = next(it)
            ids, labels, mask = ids.cuda(), labels.cuda(), mask.cuda()
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            (out.loss / ACC).backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        losses.append(out.loss.item())
        step += 1
        if step % 20 == 0 or step == 1:
            print(f"  step {step:3d} | loss {out.loss.item():.4f} | "
                  f"{times[-1]*1000:.0f} ms/step | 显存 {torch.cuda.max_memory_allocated()/2**20:.0f} MiB",
                  flush=True)

    avg = sum(times[5:]) / max(len(times[5:]), 1)
    print(f"\n平均 {avg*1000:.0f} ms/step（含 {ACC} 次前向）| 3000 步约 {avg*3000/60:.1f} 分钟")
    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT)
    print(f"LoRA 已保存: {OUT}")

    # ---- A/B 对比 ----
    print("\n=== base vs LoRA ===")
    del model
    torch.cuda.empty_cache()

    def gen(use_lora):
        from peft import PeftModel
        m = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16, device_map={"": 0}).eval()
        if use_lora:
            m = PeftModel.from_pretrained(m, OUT).eval()
        outs = []
        for p in EVAL_PROMPTS:
            msgs = [{"role": "user", "content": p}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt").to("cuda")
            n_in = enc["input_ids"].size(1)
            with torch.inference_mode():
                o = m.generate(**enc, max_new_tokens=64, do_sample=False,
                               pad_token_id=tok.pad_token_id)
            outs.append(tok.decode(o[0, n_in:], skip_special_tokens=True).strip())
        del m
        torch.cuda.empty_cache()
        return outs

    base = gen(False)
    tuned = gen(True)
    for p, b, t in zip(EVAL_PROMPTS, base, tuned):
        print(f"\n[问] {p}")
        print(f"[base] {b[:120]}")
        print(f"[lora] {t[:120]}")

    json.dump(dict(steps=STEPS, seq=SEQ, bs=BS, acc=ACC, trainable_m=round(n_train / 1e6, 2),
                   ms_per_step=round(avg * 1000), loss_first=round(losses[0], 4),
                   loss_last=round(sum(losses[-10:]) / 10, 4),
                   peak_mib=round(torch.cuda.max_memory_allocated() / 2**20),
                   base=base, lora=tuned),
              open(os.path.join(OUT, "result.json"), "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
