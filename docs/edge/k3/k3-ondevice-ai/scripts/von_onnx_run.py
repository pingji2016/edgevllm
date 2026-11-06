#!/usr/bin/env python3
"""
在 K3 上跑 Von 的 int8 ONNX（**不 import torch** —— 板上 torch 与 onnx 同进程必崩）。

链路：von_np.pack_sequence → tokenizers → von_np.build_mask → onnxruntime → logits

用法：
    python3 von_onnx_run.py                                   # 默认 von13_int8.onnx
    python3 von_onnx_run.py --ref ref.json --bench            # 与 torch 参考比对 + 延迟

  --ref  ：由 von_ref.py（torch 路径）生成的参考 logits
  --bench：跑延迟基准
"""
import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from von_np import MASK, build_mask, pack_sequence

MODEL_DIR = os.environ.get("VON_MODEL_DIR", "/home/pingji/models/von")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", nargs="?", default="/home/pingji/von/von13_int8.onnx")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--state", default="Ticket #48213: our invoice PDF downloads as a 0-byte file.")
    ap.add_argument("--question", default="Route this ticket to the right queue.")
    ap.add_argument("--options", default="billing,bug_report,feature_request,account_access")
    a = ap.parse_args()

    cal = json.load(open(f"{MODEL_DIR}/marker_calibration.json"))
    cfg = json.load(open(f"{MODEL_DIR}/config.json"))
    digit_split = bool(cal.get("training", {}).get("digit_split", False))
    tk = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
    mask_id = tk.token_to_id(MASK)
    options = [o.strip() for o in a.options.split(",")]

    packed = pack_sequence(a.state, a.question, options, digit_split)
    ids = tk.encode(packed).ids
    ids_row = np.array(ids, np.int64)
    full, sliding, position_ids, pos = build_mask(ids_row, mask_id, len(ids) - 1,
                                                  cfg.get("sliding_window"))

    feed = {
        "input_ids": ids_row[None],
        "full_mask": full,
        "sliding_mask": sliding,
        "position_ids": position_ids,
        "mask_positions": pos,
    }

    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    so.inter_op_num_threads = 1
    t = time.perf_counter()
    sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - t) * 1000

    logits = sess.run(None, feed)[0]
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()

    print(f"[in]  tokens={len(ids)}  options={len(options)}  "
          f"sliding_window={cfg.get('sliding_window')}  load={load_ms:.0f} ms")
    print(f"[out] logits={np.round(logits, 4).tolist()}")
    print(f"      probs={ {o: round(float(p), 4) for o, p in zip(options, probs)} }")

    if a.ref:
        ref = json.load(open(a.ref))
        same_ids = ref["ids"] == ids
        print(f"[check] tokenize {'一致 ✓' if same_ids else '不一致 ✗'}"
              f"（torch {len(ref['ids'])} tok vs onnx {len(ids)} tok）")
        rl = np.array(ref["logits"])
        cos = float(np.dot(rl, logits) / (np.linalg.norm(rl) * np.linalg.norm(logits) + 1e-9))
        print(f"[check] max|Δlogit|={np.abs(rl - logits).max():.4f}  cos={cos:.6f}  "
              f"argmax {'一致 ✓' if int(np.argmax(rl)) == int(np.argmax(logits)) else '不一致 ✗'}")

    if a.bench:
        for _ in range(2):
            sess.run(None, feed)
        ts = []
        for _ in range(12):
            t = time.perf_counter()
            sess.run(None, feed)
            ts.append((time.perf_counter() - t) * 1000)
        ts = np.array(ts)
        print(f"[bench] threads={a.threads}  p50={np.median(ts):.1f} ms  "
              f"p90={np.percentile(ts, 90):.1f} ms")


if __name__ == "__main__":
    main()
