"""路线 C：投机解码（speculative decoding）实测

靶模型 Qwen3-1.7B + 草稿模型 Qwen3-0.6B，greedy 解码。
测：
  1. 基线（靶模型单独生成）tok/s
  2. 投机解码 tok/s 与加速比
  3. 每步被接受的草稿 token 数（由 forward 调用次数反推 α）
  4. 正确性：投机解码的输出必须与基线**逐 token 相同**（greedy 下无损）
用法：
  ~/apps/bench/yolo-venv/bin/python routeC_spec.py
"""
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TGT = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master"
DRF = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master"
NEW_TOKENS = int(os.environ.get("NEW_TOKENS", "128"))
NUM_DRAFT = int(os.environ.get("NUM_DRAFT", "4"))

PROMPTS = [
    "用三句话解释什么是显存带宽，以及它为什么决定了大模型的解码速度。",
    "请写一个 Python 函数，判断一个字符串是否是回文，并说明时间复杂度。",
    "一个 GPU 有 448 GB/s 带宽，模型权重 4 GB，理论上 decode 速度上限是多少？请给出计算过程。",
]


def load(path):
    m = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    return m.eval()


def gen(model, tok, enc, **kw):
    with torch.inference_mode():
        return model.generate(**enc, max_new_tokens=NEW_TOKENS, do_sample=False,
                              pad_token_id=tok.eos_token_id, **kw)


def main():
    tok = AutoTokenizer.from_pretrained(TGT)
    target = load(TGT)
    draft = load(DRF)
    print(f"靶 {os.path.basename(os.path.dirname(TGT))} / 草稿 Qwen3-0.6B | 每次生成 {NEW_TOKENS} token")
    print(f"草稿步长 k={NUM_DRAFT}")

    # 统计靶模型 forward 调用次数 → 推到每个 decode step 进程
    counter = {"n": 0}
    orig_forward = target.forward

    def counting_forward(*a, **kw):
        counter["n"] += 1
        return orig_forward(*a, **kw)

    target.forward = counting_forward

    rows = []
    for i, p in enumerate(PROMPTS, 1):
        enc = tok(p, return_tensors="pt").to("cuda")
        n_in = enc["input_ids"].size(1)
        # 预热
        gen(target, tok, enc)
        torch.cuda.synchronize()

        t0 = time.time()
        base_out = gen(target, tok, enc)
        torch.cuda.synchronize()
        t_base = time.time() - t0
        n_base = base_out.shape[1] - n_in
        base_txt = tok.decode(base_out[0, n_in:], skip_special_tokens=True)

        counter["n"] = 0
        torch.cuda.synchronize()
        t0 = time.time()
        spec_out = gen(target, tok, enc, assistant_model=draft,
                       num_assistant_tokens=NUM_DRAFT)
        torch.cuda.synchronize()
        t_spec = time.time() - t0
        spec_txt = tok.decode(spec_out[0, n_in:], skip_special_tokens=True)

        n_new = spec_out.shape[1] - n_in
        steps = counter["n"]
        tps = n_new / steps if steps else 0
        alpha = max(0.0, (tps - 1) / NUM_DRAFT)
        same = base_out[0].tolist() == spec_out[0].tolist()

        rows.append(dict(prompt=i, base_s=round(t_base, 2), spec_s=round(t_spec, 2),
                         base_tps=round(n_base / t_base, 1),
                         spec_tps=round(n_new / t_spec, 1),
                         speedup=round(t_base / t_spec, 2), target_steps=steps,
                         tokens_per_step=round(tps, 2), alpha_est=round(alpha, 2),
                         identical=same))
        print(f"\n[{i}] 基线 {rows[-1]['base_tps']:5.1f} tok/s ({t_base:.2f}s) | "
              f"投机 {rows[-1]['spec_tps']:5.1f} tok/s ({t_spec:.2f}s) | "
              f"加速 {rows[-1]['speedup']:.2f}x")
        print(f"    靶模型 forward {steps} 次 → 每步 {tps:.2f} token → 估计 α≈{alpha:.2f} | "
              f"输出{'完全一致 ✅' if same else '不一致 ❌'}")
        if not same:
            print(f"    基线: {base_txt[:60]!r}")
            print(f"    投机: {spec_txt[:60]!r}")

    import json
    out = os.path.expanduser("~/apps/infer-research/routeC_results.json")
    json.dump(rows, open(out, "w"), ensure_ascii=False, indent=2)
    n = len(rows)
    print(f"\n平均加速比 {sum(r['speedup'] for r in rows)/n:.2f}x | "
          f"平均每步 {sum(r['tokens_per_step'] for r in rows)/n:.2f} token | "
          f"结果存 {out}")


if __name__ == "__main__":
    main()
