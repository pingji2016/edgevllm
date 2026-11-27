"""跨能力 LoRA 合并：数学能力 + 数字转中文大写

任务 A：三位×两位乘法（GRPO LoRA，60%）
任务 B：数字 -> 中文大写（新训 LoRA）
三组评测：只有 A / 只有 B / 合并 0.5
假设：不同能力的合并能同时保留两者（对比同任务合并的纯插值）
"""
import json
import os
import random
import re

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
A_DIR = "/home/pingji/apps/minimind/out_grpo_qwen"
B_DIR = "/home/pingji/apps/minimind/out_cn_qwen"
NUM = re.compile(r"-?\d+")

D = "零壹贰叁肆伍陆柒捌玖"
U = ["", "拾", "佰", "仟"]
BIG = ["", "万", "亿"]


def cn(n):
    if n == 0:
        return "零"
    s = str(n)
    groups = []
    while s:
        groups.append(s[-4:])
        s = s[:-4]
    out = []
    for gi in range(len(groups) - 1, -1, -1):
        g = groups[gi]
        part = ""
        zero = False
        for i, ch in enumerate(g):
            d = int(ch)
            pos = len(g) - 1 - i
            if d == 0:
                zero = True
            else:
                if zero and part:
                    part += "零"
                zero = False
                part += D[d] + U[pos]
        if part:
            out.append(part + BIG[gi])
    return "".join(out)


SYS_MATH = ("你是一个计算器。只输出结果，格式：<answer>数字</answer>\n\n"
            "示例：\n3+4=<answer>7</answer>\n8+5=<answer>13</answer>\n6+2=<answer>8</answer>")


def p_math(a, b):
    return (f"<|im_start|>system\n{SYS_MATH}<|im_end|>\n"
            f"<|im_start|>user\n{a}*{b}=<|im_end|>\n<|im_start|>assistant\n<answer>")


def p_cn(n):
    return (f"<|im_start|>system\n把数字转成中文大写，只输出结果。\n<|im_end|>\n"
            f"<|im_start|>user\n{n}\n<|im_end|>\n<|im_start|>assistant\n")


def train_cn(tok, epochs=2, bs=16, lr=2e-5, n_train=3000):
    rng = random.Random(11)
    data = []
    for _ in range(n_train):
        x = rng.randint(0, 99999)
        p = p_cn(x)
        c = cn(x)
        pid = tok(p, add_special_tokens=False).input_ids
        cid = tok(c, add_special_tokens=False).input_ids
        data.append((pid + cid, [-100] * len(pid) + cid))
    base = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16, device_map={"": 0})
    m = get_peft_model(base, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    rng2 = random.Random(0)
    step = 0
    for ep in range(epochs):
        rng2.shuffle(data)
        for i in range(0, len(data), bs):
            batch = data[i:i + bs]
            L = max(len(x[0]) for x in batch)
            ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
            lab = torch.full((len(batch), L), -100, dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for j, (x, y) in enumerate(batch):
                ids[j, :len(x)] = torch.tensor(x)
                lab[j, :len(y)] = torch.tensor(y)
                att[j, :len(x)] = 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = m(input_ids=ids.cuda(), attention_mask=att.cuda(),
                         labels=lab.cuda()).loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
            opt.step(); step += 1
            if step % 100 == 0:
                print(f"    cn-sft step {step} loss {loss.item():.3f}", flush=True)
    m.save_pretrained(B_DIR)
    del m
    torch.cuda.empty_cache()


@torch.no_grad()
def gen(model, tok, prompts, max_new, bs=16):
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for j in range(len(enc.input_ids)):
            outs.append(tok.decode(g[j, enc.input_ids.size(1):],
                                   skip_special_tokens=True).strip())
        del g
    return outs


def main():
    tok = AutoTokenizer.from_pretrained(MD)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if not os.path.exists(os.path.join(B_DIR, "adapter_config.json")):
        print("  训练任务 B（数字->中文大写）…", flush=True)
        train_cn(tok)
    else:
        print("  复用已训练的 B adapter", flush=True)

    rng = random.Random(999)
    math = [(rng.randint(100, 999), rng.randint(10, 99)) for _ in range(200)]
    cnq = [random.Random(7).randint(0, 99999) for _ in range(200)]
    rng2 = random.Random(7)
    cnq = [rng2.randint(0, 99999) for _ in range(200)]

    def fresh():
        return AutoModelForCausalLM.from_pretrained(
            MD, dtype=torch.bfloat16, device_map={"": 0})

    def eval_model(model, tag):
        ma = gen(model, tok, [p_math(a, b) for a, b in math], 12)
        ok_a = 0
        for (a, b), t in zip(math, ma):
            m = NUM.search(t)
            ok_a += bool(m and int(m.group()) == a * b)
        cb = gen(model, tok, [p_cn(n) for n in cnq], 24)
        ok_b = sum(1 for n, t in zip(cnq, cb) if t.splitlines()[0].strip() == cn(n))
        print(f"  {tag:12s} 数学 {ok_a/200*100:5.1f}% | 中文大写 {ok_b/200*100:5.1f}%",
              flush=True)
        return ok_a / 200, ok_b / 200

    res = {}
    # 只有 A
    ma = PeftModel.from_pretrained(fresh(), A_DIR)
    res["only_A"] = eval_model(ma, "只有 A(数学)")
    sd_a = {k: v.clone() for k, v in ma.state_dict().items() if "lora_" in k}
    del ma; torch.cuda.empty_cache()

    # 只有 B
    mb = PeftModel.from_pretrained(fresh(), B_DIR)
    res["only_B"] = eval_model(mb, "只有 B(大写)")
    sd_b = {k: v.clone() for k, v in mb.state_dict().items() if "lora_" in k}
    del mb; torch.cuda.empty_cache()

    # 扫 alpha
    for alpha in (0.75, 0.5, 0.25):
        mm = PeftModel.from_pretrained(fresh(), A_DIR)
        sd = mm.state_dict()
        keys = [k for k in sd_a if k in sd_b]
        for k in keys:
            sd[k] = (alpha * sd_a[k].float() + (1 - alpha) * sd_b[k].float()).to(sd[k].dtype)
        mm.load_state_dict(sd, strict=False)
        res[f"merged_{alpha}"] = eval_model(mm, f"合并 {alpha}")
        del mm
        torch.cuda.empty_cache()

    json.dump(res, open("/home/pingji/apps/minimind/merge_cross.json", "w"), indent=2)
    print("已存 merge_cross.json")


if __name__ == "__main__":
    main()
