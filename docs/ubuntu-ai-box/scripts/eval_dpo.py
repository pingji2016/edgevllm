"""DPO 效果评估：留出集上的 margin（chosen 平均对数概率 − rejected）

对比：full_sft_512.pth（DPO 前） vs dpo_512.pth（DPO 后）
"""
import json
import os
import sys

import torch

sys.path.append("/home/pingji/apps/minimind")
from dataset.lm_dataset import DPODataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

BASE = "/home/pingji/apps/minimind"
EVAL = os.path.join(BASE, "dataset/dpo_eval.jsonl")
CKPTS = {
    "DPO 前 (full_sft)": os.path.join(BASE, "out/full_sft_512.pth"),
    "DPO 后 (dpo)": os.path.join(BASE, "out_dpo/dpo_512.pth"),
}


def logprob(model, x, y, mask):
    with torch.inference_mode():
        out = model(x, labels=None)
        logits = out.logits.float()          # [B, T, V]
        lp = torch.log_softmax(logits, dim=-1)
        gather = lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # [B, T]
        m = mask.float()
        return (gather * m).sum(-1) / m.sum(-1).clamp(min=1)


def main():
    tok = AutoTokenizer.from_pretrained(os.path.join(BASE, "model"))
    ds = DPODataset(EVAL, tok, max_length=512)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    print(f"评估样本 {len(ds)} 对")

    cfg = MiniMindConfig(hidden_size=512, num_hidden_layers=8)
    rows = {}
    for name, path in CKPTS.items():
        m = MiniMindForCausalLM(cfg)
        m.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        m.cuda().eval()

        margins, wins = [], 0
        for batch in loader:
            xc = batch["x_chosen"].cuda(); yc = batch["y_chosen"].cuda(); mc = batch["mask_chosen"].cuda()
            xr = batch["x_rejected"].cuda(); yr = batch["y_rejected"].cuda(); mr = batch["mask_rejected"].cuda()
            lc = logprob(m, xc, yc, mc)
            lr = logprob(m, xr, yr, mr)
            d = (lc - lr).cpu()
            margins.extend(d.tolist())
            wins += int((d > 0).sum())
        mean_margin = sum(margins) / len(margins)
        rows[name] = dict(mean_margin=round(mean_margin, 4),
                          win_rate=round(wins / len(margins) * 100, 1))
        print(f"  {name:22s} 平均 margin {mean_margin:+.4f} | chosen 更优占比 {wins/len(margins)*100:.1f}%")
        del m
        torch.cuda.empty_cache()

    json.dump(rows, open(os.path.expanduser("~/apps/infer-research/dpo_eval.json"), "w"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
