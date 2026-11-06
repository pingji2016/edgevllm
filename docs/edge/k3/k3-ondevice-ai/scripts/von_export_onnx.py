#!/usr/bin/env python3
"""
Von 1.3 (ModernBERT-large + option-marker 打分头) -> ONNX -> int8。

必须在 x86 机器上跑（板上 torch 与 onnx 同进程会因 protobuf 描述符冲突崩溃）。

ONNX 只包 encoder + scorer：
    (input_ids, full_mask, sliding_mask, position_ids, mask_positions) -> logits[K]

"独立选项掩码" 的构建（build_independent_option_masks，Python 循环）留在图外，
板端用 numpy 复刻后喂进来 —— 否则导出时会被固化成常量，换个选项数就废了。

不下载 backbone 权重：模型按 config 随机初始化，随后用 option_marker.pt 全量覆盖
（该文件含 395.3M 参数的完整 state_dict）。

用法：
    /root/vonx/bin/python von_export_onnx.py --seq 128 --opts 4 --quant
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

MODEL_DIR = "/root/models/von"
CKPT = os.path.join(MODEL_DIR, "option_marker.pt")
WORK = "/root/vonwork"
FP32 = os.path.join(WORK, "von13_fp32.onnx")
INT8 = os.path.join(WORK, "von13_int8.onnx")


class Wrap(nn.Module):
    """导出单元：encoder 前向 + marker 打分。掩码/位置由外部喂入。"""

    def __init__(self, encoder, scorer):
        super().__init__()
        self.encoder = encoder
        self.scorer = scorer

    def forward(self, input_ids, full_mask, sliding_mask, position_ids, mask_positions):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask={"full_attention": full_mask, "sliding_attention": sliding_mask},
            position_ids=position_ids,
        )
        hidden = out.last_hidden_state[0]              # (S, H)
        reps = hidden.index_select(0, mask_positions)  # (K, H)
        return self.scorer(reps)                       # (K,)


def build_model():
    """按 config 建骨架 + 用 option_marker.pt 覆盖权重（不下载 backbone）。"""
    from transformers import AutoConfig, AutoModel
    from von.models.option_marker import OptionMarkerScorer

    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    cfg._attn_implementation = "eager"   # SDPA 在旧 exporter 下不可靠

    t = time.time()
    encoder = AutoModel.from_config(cfg)
    scorer = OptionMarkerScorer(cfg.hidden_size)
    model = Wrap(encoder, scorer)

    sd = torch.load(CKPT, map_location="cpu", weights_only=True, mmap=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] {time.time()-t:.1f}s  missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    model.eval()
    return model, cfg


def make_inputs(S, K, sliding_window=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(100, 30000, (1, S), generator=g)
    mask_id = 50284
    mp = torch.linspace(S // 8, S - 8, K).long()          # K 个互不相同的 MASK 位置
    ids[0, mp] = mask_id

    opt = torch.full((S,), -1, dtype=torch.long)
    for k, st in enumerate(mp.tolist()):
        en = mp[k + 1].item() if k + 1 < K else S - 1
        opt[st:en] = k
    oi, oj = opt[:, None], opt[None, :]
    qp, kp, same = oi == -1, oj == -1, oi == oj
    allowed = (qp & kp) | (~qp & (kp | same))
    allowed = allowed | torch.eye(S, dtype=torch.bool)
    full = allowed[None, None]
    if sliding_window:
        pos = torch.arange(S)
        local = (pos[:, None] - pos[None, :]).abs() <= sliding_window
        sliding = ((allowed & local) | torch.eye(S, dtype=torch.bool))[None, None]
    else:
        sliding = full
    position_ids = torch.arange(S)[None, :]
    return ids, full, sliding, position_ids, mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--opts", type=int, default=4)
    ap.add_argument("--quant", action="store_true")
    ap.add_argument("--reuse-fp32", action="store_true")
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)

    names = ["input_ids", "full_mask", "sliding_mask", "position_ids", "mask_positions"]
    ids, full, slid, pos, mp = make_inputs(a.seq, a.opts)
    feed = {n: v.numpy() for n, v in zip(names, (ids, full, slid, pos, mp))}

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2

    if a.reuse_fp32 and os.path.exists(FP32):
        print(f"[skip] 复用已有 {FP32}")
        ref = None
    else:
        model, cfg = build_model()
        with torch.no_grad():
            ref = model(ids, full, slid, pos, mp)
        print(f"[torch] logits={ref.tolist()}")

        dyn = {
            "input_ids": {1: "seq"},
            "full_mask": {2: "seq", 3: "seq"},
            "sliding_mask": {2: "seq", 3: "seq"},
            "position_ids": {1: "seq"},
            "mask_positions": {0: "k"},
            "logits": {0: "k"},
        }
        t = time.time()
        try:
            torch.onnx.export(
                model, (ids, full, slid, pos, mp), FP32,
                input_names=names, output_names=["logits"],
                opset_version=17, do_constant_folding=True,
                dynamo=False, dynamic_axes=dyn,
            )
        except Exception as e:                      # 新版本 torch 可能已删掉 legacy exporter
            print(f"[warn] legacy exporter 失败({type(e).__name__}), 改用 dynamo=True", flush=True)
            torch.onnx.export(
                model, (ids, full, slid, pos, mp), FP32,
                input_names=names, output_names=["logits"],
                opset_version=17, dynamo=True,
            )
        print(f"[export] {time.time()-t:.1f}s -> {FP32}  {os.path.getsize(FP32)/1e6:.0f} MB", flush=True)
        del model

    s = ort.InferenceSession(FP32, so, providers=["CPUExecutionProvider"])
    if ref is not None:
        got = s.run(None, feed)[0]
        cos = float(np.dot(ref.numpy(), got) / (np.linalg.norm(ref.numpy()) * np.linalg.norm(got) + 1e-9))
        print(f"[check fp32] max|diff|={np.abs(ref.numpy()-got).max():.6f}  cos={cos:.6f}")

    if not a.quant:
        return

    from onnxruntime.quantization import QuantType, quantize_dynamic
    t = time.time()
    quantize_dynamic(FP32, INT8, weight_type=QuantType.QInt8)
    print(f"[quant] {time.time()-t:.1f}s -> {INT8}  {os.path.getsize(INT8)/1e6:.0f} MB", flush=True)

    q = ort.InferenceSession(INT8, so, providers=["CPUExecutionProvider"])
    got_q = q.run(None, feed)[0]
    if ref is not None:
        r = ref.numpy()
        cosq = float(np.dot(r, got_q) / (np.linalg.norm(r) * np.linalg.norm(got_q) + 1e-9))
        print(f"[check int8] max|diff|={np.abs(r-got_q).max():.4f}  cos={cosq:.6f}  "
              f"argmax {'一致' if int(np.argmax(r)) == int(np.argmax(got_q)) else '不一致'}")

    for tag, sess in (("fp32", s), ("int8", q)):
        sess.run(None, feed)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            sess.run(None, feed)
            ts.append((time.perf_counter() - t0) * 1000)
        print(f"[{tag}] 111 x86 2vCPU: p50={np.median(ts):.1f} ms")


if __name__ == "__main__":
    main()
