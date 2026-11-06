#!/usr/bin/env python3
"""
K3 上「摄像头 → 预处理 → ViT 编码」的逐段成本 + 流式 vs snapshot 对照。

对应 docs/k3-world-model.md §3 的表。

用法（板上，需要 /opt/k3cam 的 camserver 在 :8090）：
    python3 vit-pipeline-bench.py            # 逐段拆解 + 两种取帧方式对照
    python3 vit-pipeline-bench.py --frames 150
"""
import argparse
import json
import time
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MODEL = "dino-vits16/onnx/model_quantized.onnx"
CAM = "http://127.0.0.1:8090"


def make_session(threads=8):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    s = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    return s, s.get_inputs()[0].name


def prep(bgr, size=224, interp=cv2.INTER_AREA):
    h, w = bgr.shape[:2]
    sc = size / min(h, w)
    bgr = cv2.resize(bgr, (int(round(w * sc)), int(round(h * sc))), interpolation=interp)
    h, w = bgr.shape[:2]
    bgr = bgr[(h - size) // 2:(h - size) // 2 + size, (w - size) // 2:(w - size) // 2 + size]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]


def median_ms(fn, n=20):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        r = fn()
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts)), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    sess, iname = make_session(a.threads)
    raw = urllib.request.urlopen(f"{CAM}/snapshot.jpg", timeout=5).read()

    # ---------- A. 逐段成本 ----------
    print("=== A. 逐段成本（各 20 次取中位）===")
    print(f"  cv2 线程数: {cv2.getNumThreads()}")
    d_full, full = median_ms(lambda: cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR))
    d_half, half = median_ms(lambda: cv2.imdecode(np.frombuffer(raw, np.uint8),
                                                  cv2.IMREAD_REDUCED_COLOR_2))
    d_quarter, _ = median_ms(lambda: cv2.imdecode(np.frombuffer(raw, np.uint8),
                                                  cv2.IMREAD_REDUCED_COLOR_4))
    print(f"  imdecode 1/1 1280x720     {d_full:6.1f} ms")
    print(f"  imdecode 1/2  640x360     {d_half:6.1f} ms")
    print(f"  imdecode 1/4  320x180     {d_quarter:6.1f} ms")

    p_area, _ = median_ms(lambda: prep(full, interp=cv2.INTER_AREA))
    p_lin, _ = median_ms(lambda: prep(full, interp=cv2.INTER_LINEAR))
    p_half_area, _ = median_ms(lambda: prep(half, interp=cv2.INTER_AREA))
    print(f"  预处理 全图+AREA          {p_area:6.1f} ms")
    print(f"  预处理 全图+LINEAR        {p_lin:6.1f} ms   ← 省 10 ms")
    print(f"  预处理 半图+AREA          {p_half_area:6.1f} ms")

    x = prep(half)
    enc, _ = median_ms(lambda: sess.run(None, {iname: x}))
    print(f"  编码 int8 ({a.threads} 线程)      {enc:6.1f} ms   ← 占整链 7 成")

    chain_full, z1 = median_ms(lambda: sess.run(None, {iname: prep(
        cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR))})[0][0, 0, :])
    chain_half, z2 = median_ms(lambda: sess.run(None, {iname: prep(
        cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_REDUCED_COLOR_2))})[0][0, 0, :])
    print(f"  全链 全图+AREA            {chain_full:6.1f} ms -> {1000/chain_full:5.1f} fps")
    print(f"  全链 半图+AREA            {chain_half:6.1f} ms -> {1000/chain_half:5.1f} fps")
    cos = float(np.dot(z1, z2) / (np.linalg.norm(z1) * np.linalg.norm(z2)))
    print(f"  半图 vs 全图表征余弦       {cos:.5f}")

    # ---------- B. 流式 vs snapshot ----------
    print("\n=== B. /stream.mjpg 持续取流 vs 每帧 snapshot ===")
    resp = urllib.request.urlopen(f"{CAM}/stream.mjpg", timeout=10)
    buf, n, lat, t0, W, H = b"", 0, [], time.time(), 0, 0
    while n < a.frames:
        chunk = resp.read(65536)
        if not chunk:
            break
        buf += chunk
        while n < a.frames:
            i = buf.find(b"\xff\xd8")
            j = buf.find(b"\xff\xd9", i + 2) if i >= 0 else -1
            if i < 0 or j < 0:
                break
            jpg, buf = buf[i:j + 2], buf[j + 2:]
            ts = time.perf_counter()
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            H, W = img.shape[:2]
            sess.run(None, {iname: prep(img)})
            lat.append((time.perf_counter() - ts) * 1000)
            n += 1
    elapsed = time.time() - t0
    resp.close()
    lat = np.array(lat)
    print(f"  取到 {n} 帧 / {elapsed:.1f}s（{W}x{H}）  {n/elapsed:.1f} fps")
    print(f"  每帧 p50={np.median(lat):.1f} ms  p90={np.percentile(lat, 90):.1f} ms")

    snap = []
    for _ in range(30):
        t = time.perf_counter()
        raw2 = urllib.request.urlopen(f"{CAM}/snapshot.jpg", timeout=5).read()
        sess.run(None, {iname: prep(cv2.imdecode(np.frombuffer(raw2, np.uint8), cv2.IMREAD_COLOR))})
        snap.append((time.perf_counter() - t) * 1000)
    snap = np.array(snap)
    print(f"  每帧 snapshot: p50={np.median(snap):.1f} ms -> {1000/np.median(snap):.1f} fps")
    print(f"  流式提速: {np.median(snap)/np.median(lat):.2f}x（≈ 没有）")

    json.dump({
        "decode_full_ms": d_full, "decode_half_ms": d_half, "decode_quarter_ms": d_quarter,
        "prep_full_area_ms": p_area, "prep_full_linear_ms": p_lin, "prep_half_area_ms": p_half_area,
        "encode_ms": enc, "chain_full_ms": chain_full, "chain_half_ms": chain_half,
        "cos_half_vs_full": cos, "stream_fps": round(n / elapsed, 2),
        "stream_frame_p50_ms": float(np.median(lat)),
        "snapshot_p50_ms": float(np.median(snap)),
        "stream_speedup": round(float(np.median(snap) / np.median(lat)), 2),
    }, open("pipeline.json", "w"), indent=2)
    print("\n-> pipeline.json")


if __name__ == "__main__":
    main()
