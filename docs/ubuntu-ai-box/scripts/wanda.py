"""Wanda 剪枝实测：Qwen3-0.6B，按 |W| * ||X|| 打分剪枝

测：不同稀疏度下的 PPL（校准用 8 段语料）
关注：精度掉多少、以及"剪枝了但不会自动变快"

用法：~/apps/bench/yolo-venv/bin/python wanda.py
"""
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master"
CORPUS = os.path.expanduser("~/models/infer-tmp/corpus.txt")
SPARSITIES = [0.0, 0.2, 0.4, 0.5, 0.6]
PPL_TOKENS = 1024
CALIB_LEN = 512
CALIB_N = 8


def load_text(tok, n_tokens):
    txt = open(CORPUS, encoding="utf-8").read()
    ids = tok(txt[: n_tokens * 6], return_tensors="pt").input_ids
    return ids[:, :n_tokens]


def ppl(model, ids):
    model.eval()
    losses = []
    with torch.inference_mode():
        for i in range(0, ids.size(1) - 256, 256):
            chunk = ids[:, i:i + 256].to("cuda")
            losses.append(model(chunk, labels=chunk).loss.float())
    return torch.exp(torch.stack(losses).mean()).item()


def main():
    tok = AutoTokenizer.from_pretrained(MD)
    model = AutoModelForCausalLM.from_pretrained(
        MD, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa").eval()

    linears = [(n, m) for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and m.weight.shape[1] > 1]
    print(f"可剪枝 Linear 层: {len(linears)}")

    # ---------- 1. 校准：统计每层输入通道的 L2 范数 ----------
    norms = {}
    hooks = []

    def mk(n):
        def h(mod, inp, out):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1]).float()
            s = x.norm(dim=0)  # 每个输入通道
            norms[n] = norms.get(n, 0) + s
        return h

    for n, m in linears:
        hooks.append(m.register_forward_hook(mk(n)))

    calib = load_text(tok, CALIB_LEN).to("cuda")
    calib = calib.repeat(CALIB_N, 1) if CALIB_N > 1 else calib
    with torch.inference_mode():
        model(calib)
    for h in hooks:
        h.remove()
    for n in norms:
        norms[n] = (norms[n] / CALIB_N).clamp(min=1e-8)
    del calib
    torch.cuda.empty_cache()
    print("校准完成")

    orig = {n: m.weight.detach().clone() for n, m in linears}
    test_ids = load_text(tok, PPL_TOKENS)

    # ---------- 2. 逐档稀疏度：打分 → 按输出通道保留 top-k → 测 PPL ----------
    rows = []
    for p in SPARSITIES:
        for n, m in linears:
            w = orig[n].float()
            score = w.abs() * norms[n].to(w.device).unsqueeze(0)  # [out, in]
            if p == 0.0:
                mask = torch.ones_like(score)
            else:
                k = max(1, int(round(score.shape[1] * (1 - p))))
                thresh = score.topk(k, dim=1).values[:, -1:]
                mask = (score >= thresh).float()
            m.weight.data.copy_((w * mask).to(m.weight.dtype))

        t0 = time.time()
        v = ppl(model, test_ids)
        rows.append(dict(sparsity=p, ppl=round(v, 3), sec=round(time.time() - t0, 1)))
        print(f"  稀疏度 {p*100:4.0f}% | PPL {v:8.3f} | 用时 {time.time()-t0:.1f}s", flush=True)

    # 复原
    for n, m in linears:
        m.weight.data.copy_(orig[n])

    base = rows[0]["ppl"]
    print("\n| 稀疏度 | PPL | 相对基线 |")
    print("| ---: | ---: | ---: |")
    for r in rows:
        print(f"| {r['sparsity']*100:.0f}% | {r['ppl']:.3f} | {(r['ppl']/base-1)*100:+.1f}% |")

    # ---------- 3. 随机剪枝对照（同稀疏度 50%）----------
    p = 0.5
    for n, m in linears:
        w = orig[n].float()
        k = max(1, int(round(w.shape[1] * (1 - p))))
        idx = torch.rand_like(w).topk(k, dim=1).indices
        mask = torch.zeros_like(w).scatter_(1, idx, 1.0)
        m.weight.data.copy_((w * mask).to(m.weight.dtype))
    v_rand = ppl(model, test_ids)
    for n, m in linears:
        m.weight.data.copy_(orig[n])
    print(f"\n随机剪枝 50% PPL: {v_rand:.3f}（Wanda {[r['ppl'] for r in rows if r['sparsity']==0.5][0]:.3f}）")

    json.dump(dict(wanda=rows, random50=round(v_rand, 3)),
              open(os.path.expanduser("~/apps/infer-research/wanda_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
