"""QuaRot 验证（补充）：量化粒度对"旋转是否有用"的影响

假设：旋转把 outlier 从少数输入通道摊到全部通道，
      在「逐输出通道」粒度下会抬高该行动态范围 → 变差；
      但在「group-wise（每 128 个通道一组）」粒度下，旋转应该显著变好。
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = "/home/pingji/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master"
CORPUS = os.path.expanduser("~/models/infer-tmp/corpus.txt")
BITS = 4


def fold_norms_(model):
    for layer in model.model.layers:
        g = layer.input_layernorm.weight.data.float()
        for p in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
            p.weight.data = (p.weight.data.float() * g).to(p.weight.dtype)
        layer.input_layernorm.weight.data.fill_(1.0)
        g = layer.post_attention_layernorm.weight.data.float()
        for p in (layer.mlp.gate_proj, layer.mlp.up_proj):
            p.weight.data = (p.weight.data.float() * g).to(p.weight.dtype)
        layer.post_attention_layernorm.weight.data.fill_(1.0)
    g = model.model.norm.weight.data.float()
    model.lm_head.weight.data = (model.lm_head.weight.data.float() * g).to(model.lm_head.weight.dtype)
    model.model.norm.weight.data.fill_(1.0)


def rotate_(model, R):
    R = R.to(model.model.embed_tokens.weight.device)
    model.model.embed_tokens.weight.data = (
        model.model.embed_tokens.weight.data.float() @ R).to(model.model.embed_tokens.weight.dtype)
    for layer in model.model.layers:
        a = layer.self_attn
        for p in (a.q_proj, a.k_proj, a.v_proj):
            p.weight.data = (p.weight.data.float() @ R).to(p.weight.dtype)
        a.o_proj.weight.data = (R.T @ a.o_proj.weight.data.float()).to(a.o_proj.weight.dtype)
        for p in (layer.mlp.gate_proj, layer.mlp.up_proj):
            p.weight.data = (p.weight.data.float() @ R).to(p.weight.dtype)
        layer.mlp.down_proj.weight.data = (R.T @ layer.mlp.down_proj.weight.data.float()).to(
            layer.mlp.down_proj.weight.dtype)
    model.lm_head.weight.data = (model.lm_head.weight.data.float() @ R).to(model.lm_head.weight.dtype)


def quant_(model, group=None, bits=BITS):
    qmax = 2 ** (bits - 1) - 1
    for m in model.modules():
        if not isinstance(m, torch.nn.Linear):
            continue
        w = m.weight.data.float()
        if group is None:
            s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
            q = (w / s).round().clamp(-qmax - 1, qmax) * s
        else:
            out, inn = w.shape
            g = min(group, inn)
            pad = (g - inn % g) % g
            ww = torch.nn.functional.pad(w, (0, pad)) if pad else w
            wg = ww.reshape(out, -1, g)
            s = wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / qmax
            q = ((wg / s).round().clamp(-qmax - 1, qmax) * s).reshape(out, -1)[:, :inn]
        m.weight.data.copy_(q.to(m.weight.dtype))


def ppl(model, ids, chunk=256):
    model.eval()
    losses = []
    with torch.inference_mode():
        for i in range(0, ids.size(1) - chunk, chunk):
            c = ids[:, i:i + chunk].to("cuda")
            losses.append(model(c, labels=c).loss.float())
    return torch.exp(torch.stack(losses).mean()).item()


def hadamard(n):
    h = torch.tensor([[1.0]])
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / (n ** 0.5)


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MD)
    ids = tok(open(CORPUS, encoding="utf-8").read()[:6000], return_tensors="pt").input_ids[:, :1024]

    def fresh():
        return AutoModelForCausalLM.from_pretrained(
            MD, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa").eval()

    m = fresh()
    base = ppl(m, ids)
    print(f"bf16 基线 PPL {base:.3f}", flush=True)
    hidden = m.config.hidden_size
    R = hadamard(hidden)
    del m
    torch.cuda.empty_cache()

    rows = []
    for group in (None, 128):
        for rotated in (False, True):
            m = fresh()
            fold_norms_(m)
            if rotated:
                rotate_(m, R)
            quant_(m, group=group)
            v = ppl(m, ids)
            tag = f"{'rot' if rotated else 'RTN'}-4bit-{'row' if group is None else f'g{group}'}"
            rows.append(dict(tag=tag, ppl=round(v, 3)))
            print(f"  {tag:16s} PPL {v:8.3f}  ({(v/base-1)*100:+.1f}%)", flush=True)
            del m
            torch.cuda.empty_cache()

    json.dump(rows, open(os.path.expanduser("~/apps/infer-research/quarot2_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
