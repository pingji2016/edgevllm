#!/usr/bin/env python3
"""
一致性验证：板端 numpy 复刻 vs SDK(tonic libtorch) 的预处理。

对三件事逐位比对（用 von_ref.py 产出的 ref.json 当真值）：
  1. pack_sequence 拼出来的字符串
  2. tokenizers 直接编码的 ids（不经过 transformers）
  3. build_mask 造出来的 full/sliding mask 与 position_ids

用法（板上）：
    cd ~/von && HF_HUB_OFFLINE=1 ./venv/bin/python von_parity_check.py ref.json
"""
import json
import sys

import numpy as np
import torch

from tokenizers import Tokenizer
from von.models.option_marker import (
    build_independent_option_masks,
    build_option_invariant_position_ids,
)

from von_np import MASK, build_mask, pack_sequence

MODEL_DIR = "/home/pingji/models/von"
ref_path = sys.argv[1] if len(sys.argv) > 1 else "ref.json"
ref = json.load(open(ref_path))

cfg = json.load(open(f"{MODEL_DIR}/config.json"))
cal = json.load(open(f"{MODEL_DIR}/marker_calibration.json"))
digit_split = bool(cal.get("training", {}).get("digit_split", False))
print(f"digit_split={digit_split}  sliding_window={cfg.get('sliding_window')}")

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


# ---- 1) pack_sequence ----
STATE = "Ticket #48213: our invoice PDF downloads as a 0-byte file."
QUESTION = "Route this ticket to the right queue."
OPTIONS = ["billing", "bug_report", "feature_request", "account_access"]
packed = pack_sequence(STATE, QUESTION, OPTIONS, digit_split)
check("pack_sequence 字符串", packed == ref["packed"], "" if packed == ref["packed"] else
      f"\n     ours={packed}\n     torch={ref['packed']}")

# ---- 2) tokenizer（tokenizers 直编 vs transformers）----
tk = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
ids = tk.encode(packed).ids
mask_id = tk.token_to_id(MASK)
check("tokenize ids", ids == ref["ids"], f"n={len(ids)} mask_id={mask_id}")

# ---- 3) masks / position_ids ----
ids_row = np.array(ref["ids"], np.int64)
pos_list = [ref["pos"]]
ids_t = torch.tensor([ref["ids"]])
att_t = torch.ones_like(ids_t)
pids_t = build_option_invariant_position_ids(ids_t, att_t, pos_list)
m_t = build_independent_option_masks(ids_t, att_t, pos_list, pids_t, cfg.get("sliding_window"))

full_np, slid_np, pids_np, pos_np = build_mask(ids_row, mask_id, len(ids_row) - 1,
                                               cfg.get("sliding_window"))
check("mask_positions", pos_np.tolist() == ref["pos"], f"{pos_np.tolist()} vs {ref['pos']}")
check("position_ids", np.array_equal(pids_np, pids_t.numpy()),
      "" if np.array_equal(pids_np, pids_t.numpy()) else
      f"\n     ours={pids_np.tolist()}\n     torch={pids_t.tolist()}")
check("full_attention mask", np.array_equal(full_np, m_t["full_attention"].numpy()),
      f"shape={full_np.shape} 允许边数={int(full_np.sum())}")
check("sliding_attention mask", np.array_equal(slid_np, m_t["sliding_attention"].numpy()),
      f"允许边数={int(slid_np.sum())}")

print("\n结论：", "全部一致 ✓" if ok else "存在不一致 ✗")
sys.exit(0 if ok else 1)
