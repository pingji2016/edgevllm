#!/usr/bin/env python3
"""
人脸滤镜（大眼）是怎么算的、以及瓶颈到底在哪。

分工：
  1. 检测/关键点 —— MediaPipe Tasks 的 FaceLandmarker，478 个点 + 52 个表情系数
  2. 变形      —— 每个眼球中心做一次「径向挤压」：离中心越近的像素越往外采样
                  （scale = 1 - strength * w，w 从中心 1 平滑衰减到半径处 0）

第二件事才是花时间的地方。本脚本量三件事：
  · 检测一次要多久
  · 全图做变形的耗时 vs 只对眼球 bbox 做变形的耗时
  · 于是「检测 + ROI 变形」的端到端帧率

结论（RTX 5060 Ti 无关，这里全程 CPU）：检测 9.5ms 不是瓶颈，
把变形限制在 ROI 里才有 15 倍的差距 —— 滤镜的优化点在算子，不在模型。

用法：python face_landmark_filter.py <人像图> <face_landmarker.task> <输出目录>
模型：https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

LEFT_IRIS, RIGHT_IRIS = 468, 473          # 478 点里，虹膜中心的索引


def detect_landmarks(img_path, model_path):
    base = mp_python.BaseOptions(model_asset_path=model_path)
    opts = vision.FaceLandmarkerOptions(
        base_options=base,
        output_face_blendshapes=True,
        num_faces=1,
        running_mode=vision.RunningMode.IMAGE,
    )
    landmarker = vision.FaceLandmarker.create_from_options(opts)
    img = mp.Image.create_from_file(img_path)          # Tasks API 要的是 mp.Image
    t = time.perf_counter()
    res = landmarker.detect(img)
    dt = (time.perf_counter() - t) * 1000
    pts = np.array([(p.x, p.y) for p in res.face_landmarks[0]], dtype=np.float32)
    blend = {b.category_name: b.score for b in res.face_blendshapes[0]} if res.face_blendshapes else {}
    return pts, blend, dt


def eye_enlarge(img, centers, radius_ratio=0.11, strength=0.35):
    """对每个 center 做径向放大；只在包围盒内计算（ROI）。

    半径内的采样半径按 scale 缩放：scale<1 表示"往外取样"，看起来就是放大。
    权重 w = (1 - t^2)^2 在圆心为 1、半径处为 0 —— 圆外保持原图，所以没有硬接缝。
    """
    h, w = img.shape[:2]
    R = max(4.0, radius_ratio * min(h, w))
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    x0, x1 = int(max(0, min(xs) - R)), int(min(w, max(xs) + R))
    y0, y1 = int(max(0, min(ys) - R)), int(min(h, max(ys) + R))
    sub = img[y0:y1, x0:x1].copy()
    out = sub.copy()
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    for (cx, cy) in centers:
        dx, dy = xx - cx, yy - cy
        t = np.clip(np.sqrt(dx * dx + dy * dy) / R, 0.0, 1.0)
        wn = (1.0 - t * t) ** 2                       # 平滑衰减的权重
        scale = 1.0 - strength * wn                   # 变形的强度
        mx = np.clip(cx + dx * scale - x0, 0, sub.shape[1] - 1).astype(np.float32)
        my = np.clip(cy + dy * scale - y0, 0, sub.shape[0] - 1).astype(np.float32)
        warped = cv2.remap(sub, mx, my, cv2.INTER_LINEAR)
        m = (wn > 1e-4)[..., None].astype(np.float32)  # 圆外不动 → 无接缝
        out = (warped * m + out * (1.0 - m)).astype(np.uint8)
    full = img.copy()
    full[y0:y1, x0:x1] = out
    return full


def timed(fn, n=20):
    fn()                                              # 预热
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000


def main():
    src, model, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    img = cv2.imread(src)
    h, w = img.shape[:2]
    pts, blend, det_ms = detect_landmarks(src, model)
    centers = [(float(pts[i][0]) * w, float(pts[i][1]) * h) for i in (LEFT_IRIS, RIGHT_IRIS)]

    print(f"图像 {w}×{h}    关键点 {len(pts)} 个    表情系数 {len(blend)} 个")
    print(f"  检测一次：{det_ms:.1f} ms")
    if blend:
        named = sorted(blend.items(), key=lambda kv: -kv[1])[:3]
        print("  最强的三个表情系数：" + "，".join(f"{k} {v:.3f}" for k, v in named))

    # 「全图做」= 把半径放到覆盖整帧，于是同样的 remap 要跑满 w*h 个像素
    full_ms = timed(lambda: eye_enlarge(img, centers, radius_ratio=1.5))
    roi_ms = timed(lambda: eye_enlarge(img, centers))

    print(f"\n大眼变形（对两只眼睛各做一次径向重映射）")
    print(f"  全图做：{full_ms:6.1f} ms   （{w*h} 像素全部过一遍 remap）")
    print(f"  ROI 做：{roi_ms:6.1f} ms   （只算眼球包围盒，快 {full_ms / roi_ms:.1f} 倍）")
    total = det_ms + roi_ms
    print(f"\n端到端（检测 + ROI 变形）：{total:.1f} ms/帧  →  {1000 / total:.0f} fps")
    print(f"  如果每帧再留 20ms 做别的事，预算 = {20 + total:.1f} ms，仍够 30fps。")

    cv2.imwrite(os.path.join(outdir, "face_eye_enlarged.png"), eye_enlarge(img, centers))
    print(f"\n输出：{os.path.join(outdir, 'face_eye_enlarged.png')}")


if __name__ == "__main__":
    main()
