#!/usr/bin/env python3
"""
Von 的 torch 参考实现：生成 int8 ONNX 路径要比对的 logits / ids。

用法（板上）：
    cd ~/von && ./venv/bin/python von_ref.py ref.json

输出 JSON：{"packed","ids","pos","logits","torch_ms"}，交给 von_onnx_run.py --ref 比对。
"""
import json
import sys
import time

import torch

from von.models.option_marker import OptionMarkerModel

MODEL_DIR = "/home/pingji/models/von"
STATE = "Ticket #48213: our invoice PDF downloads as a 0-byte file."
QUESTION = "Route this ticket to the right queue."
OPTIONS = ["billing", "bug_report", "feature_request", "account_access"]

out = sys.argv[1] if len(sys.argv) > 1 else "ref.json"

torch.set_grad_enabled(False)
t = time.time()
m = OptionMarkerModel(base_model_id=MODEL_DIR)
sd = torch.load(f"{MODEL_DIR}/option_marker.pt", map_location="cpu", weights_only=True)
m.load_state_dict(sd, strict=True)
m.eval()
load = time.time() - t

tok = m.tokenizer
packed = m.pack_sequence(STATE, QUESTION, OPTIONS)
inp = tok(packed, return_tensors="pt")
ids = inp["input_ids"][0]
pos = (ids == m.mask_token_id).nonzero().flatten().tolist()

t = time.perf_counter()
logits = m(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
           mask_positions=[pos], independent_options=True)[0]
first = (time.perf_counter() - t) * 1000

t = time.perf_counter()
logits = m(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
           mask_positions=[pos], independent_options=True)[0]
second = (time.perf_counter() - t) * 1000

print(f"[load] {load:.1f}s   [forward] first={first:.0f}ms second={second:.0f}ms")
print(f"[packed] {packed}")
print(f"[logits] {logits.tolist()}")

json.dump({"packed": packed, "ids": ids.tolist(), "pos": pos,
           "logits": logits.tolist(), "torch_ms": second},
          open(out, "w"), indent=2)
print(f"[done] -> {out}")
