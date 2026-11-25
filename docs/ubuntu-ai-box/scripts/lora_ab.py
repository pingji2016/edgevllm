"""A/B 对比（公平版）：关掉 Qwen3 的 thinking，base vs LoRA"""
import json
import os

import torch

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master"
OUT = os.path.expanduser("~/apps/infer-research/lora_llm")
PROMPTS = [
    "你是谁？请用一句话介绍自己。",
    "用一句话解释什么是显存带宽。",
    "把这句话改得更礼貌：把文件发给我。",
    "用 Python 写一个判断回文串的函数。",
]


def main():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MD)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def run(use_lora):
        m = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16,
                                                 device_map={"": 0}).eval()
        if use_lora:
            m = PeftModel.from_pretrained(m, OUT).eval()
        outs = []
        for p in PROMPTS:
            msgs = [{"role": "user", "content": p}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            enc = tok(text, return_tensors="pt").to("cuda")
            n_in = enc["input_ids"].size(1)
            with torch.inference_mode():
                o = m.generate(**enc, max_new_tokens=96, do_sample=False,
                               pad_token_id=tok.pad_token_id)
            outs.append(tok.decode(o[0, n_in:], skip_special_tokens=True).strip())
        del m
        torch.cuda.empty_cache()
        return outs

    base = run(False)
    tuned = run(True)
    for p, b, t in zip(PROMPTS, base, tuned):
        print(f"\n[问] {p}")
        print(f"[base] {b[:110]!r}")
        print(f"[lora] {t[:110]!r}")

    json.dump(dict(prompts=PROMPTS, base=base, lora=tuned),
              open(os.path.join(OUT, "ab.json"), "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
