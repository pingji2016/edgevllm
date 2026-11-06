#!/usr/bin/env python3
# =============================================================================
#  vit-latent.py —— 世界模型的「编码器 + 潜空间动力学」基准（K3）
#
#  用法：
#    python3 vit-latent.py                 # 跑全部（需要板上摄像头 :8090）
#    python3 vit-latent.py --bench         # 只跑 ONNX 吞吐（不需要摄像头）
#
#  前置：
#    export HF_ENDPOINT=https://hf-mirror.com
#    python3 -c "from huggingface_hub import snapshot_download as d; \
#      d('Xenova/dino-vits16', local_dir='$HOME/wm/dino-vits16', \
#        allow_patterns=['onnx/model.onnx','onnx/model_quantized.onnx','*.json'])"
#
#  实测（2026-10-27，K3 / 8×X100）：
#    ViT-S/16 @224²  fp32 283ms → int8 80.4ms（12.4 fps，3.5×）
#    摄像头端到端 101.7ms（9.8 fps），其中抓帧 30.8ms
#    潜空间动力学：线性/PCA predictor 全部打不过「不变」基线
#      （受控平移 5 步 rollout：不变 0.935 vs 线性 0.876）
#    int8 与 fp32 的 CLS 表征余弦 0.9955
# =============================================================================
import json
import sys
import time
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort

ROOT = "dino-vits16/onnx"
CAM = "http://127.0.0.1:8090/snapshot.jpg"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def session(path, threads=8):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return s, s.get_inputs()[0].name


def prep(bgr, size=224):
    h, w = bgr.shape[:2]
    sc = size / min(h, w)
    bgr = cv2.resize(bgr, (int(round(w * sc)), int(round(h * sc))), interpolation=cv2.INTER_AREA)
    h, w = bgr.shape[:2]
    y, x = (h - size) // 2, (w - size) // 2
    bgr = bgr[y:y + size, x:x + size]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]


def timed(sess, iname, x, n=20):
    sess.run(None, {iname: x})
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        sess.run(None, {iname: x})
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts)), float(np.percentile(ts, 90))


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def bench_only():
    x = np.random.rand(1, 3, 224, 224).astype(np.float32)
    res = {}
    for tag, path in [("fp32", f"{ROOT}/model.onnx"), ("int8", f"{ROOT}/model_quantized.onnx")]:
        for nt in ([8, 4, 1] if tag == "int8" else [8]):
            s, n = session(path, nt)
            p50, p90 = timed(s, n, x)
            res[f"{tag}_t{nt}"] = {"p50_ms": round(p50, 1), "fps": round(1000 / p50, 1)}
            print(f"  {tag}_t{nt:2d}  p50={p50:7.1f}ms  {1000/p50:6.1f} fps")
    json.dump(res, open("vit-bench.json", "w"), indent=2)


def full():
    s, iname = session(f"{ROOT}/model_quantized.onnx", 8)
    sf, nf = session(f"{ROOT}/model.onnx", 8)
    out = {}

    # --- A. 摄像头端到端 ---
    Z, enc_ms, grab_ms = [], [], []
    print("=== 摄像头（120 帧）===")
    while len(Z) < 120:
        t = time.perf_counter()
        raw = urllib.request.urlopen(CAM, timeout=5).read()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        x = prep(img)
        grab_ms.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        Z.append(s.run(None, {iname: x})[0][0, 0, :])
        enc_ms.append((time.perf_counter() - t) * 1000)
        time.sleep(0.12)
    Z = np.stack(Z)
    enc_ms, grab_ms = np.array(enc_ms), np.array(grab_ms)
    e2e = grab_ms + enc_ms
    print(f"  编码 p50={np.median(enc_ms):.1f}ms  抓帧 p50={np.median(grab_ms):.1f}ms  "
          f"端到端 p50={np.median(e2e):.1f}ms ({1000/np.median(e2e):.1f} fps)")
    out["enc_p50_ms"] = float(np.median(enc_ms))
    out["e2e_p50_ms"] = float(np.median(e2e))
    out["cam_cos1"] = float(np.mean([cos(Z[i], Z[i + 1]) for i in range(len(Z) - 1)]))

    # --- B. 受控平移：动作条件下的潜空间动力学 ---
    print("=== 受控平移（6px/步）===")
    raw = urllib.request.urlopen(CAM, timeout=5).read()
    big = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    H, W = big.shape[:2]
    S, STEP, C = 320, 6, 150
    y0 = max(0, (H - S) // 2)
    zp = []
    for i in range(C):
        x0 = min(i * STEP, W - S)
        zp.append(s.run(None, {iname: prep(big[y0:y0 + S, x0:x0 + S])})[0][0, 0, :])
    zp = np.stack(zp)
    c1 = float(np.mean([cos(zp[i], zp[i + 1]) for i in range(C - 1)]))
    c5 = float(np.mean([cos(zp[i], zp[i + 5]) for i in range(C - 5)]))
    print(f"  不变基线: 1 步 {c1:.5f}  5 步 {c5:.5f}")
    out["pan_cos1"], out["pan_cos5"] = c1, c5

    # 线性增量 predictor（岭回归）—— 结论：打不过不变基线
    K = 100
    D = zp[1:] - zp[:-1]
    for lam in (100.0, 1000.0):
        X = np.hstack([zp[:-1], np.ones((C - 1, 1))])
        Wm = np.linalg.solve(X[:K].T @ X[:K] + lam * np.eye(X.shape[1]), X[:K].T @ D[:K])
        pred = zp[:-1][K:] + X[K:] @ Wm
        e1 = float(np.mean([cos(pred[i], zp[1:][K + i]) for i in range(len(pred))]))
        print(f"  线性增量 λ={lam:6.0f}: 1 步 cos={e1:.5f}")
        out[f"ridge_{int(lam)}_cos1"] = e1

    # --- C. 量化保真度 ---
    print("=== int8 vs fp32 保真度（20 帧）===")
    cs = []
    for _ in range(20):
        raw = urllib.request.urlopen(CAM, timeout=5).read()
        x = prep(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR))
        zf = sf.run(None, {nf: x})[0][0, 0, :]
        zq = s.run(None, {iname: x})[0][0, 0, :]
        cs.append(cos(zf, zq))
    print(f"  cos(int8, fp32) 均值 {np.mean(cs):.4f} 最小 {np.min(cs):.4f}")
    out["quant_cos_mean"] = float(np.mean(cs))

    json.dump(out, open("vit-latent.json", "w"), indent=2)
    print("[done] -> vit-latent.json")


if __name__ == "__main__":
    bench_only()
    if "--bench" not in sys.argv:
        full()
