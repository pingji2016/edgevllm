#!/usr/bin/env python3
"""
换脸（face swap）的完整流水线，每一步都单独计时。

两阶段是业界的通用结构：

  1. 对齐（alignment）
     检测 5 个关键点（双眼、鼻尖、嘴角）→ 用相似变换把它们搬到一张固定模板上。
     模板是 ArcFace 的 112×112 五点位（insightface / AdaFace / w600k 全用它），
     所以 ArcFace 的识别模型和换脸模型可以共用同一份对齐结果。
     这 5 个点是**虹膜中心**而不是眼角：虹膜中心不受眨眼/眼皮影响，抖得更少。

  2. 换脸网络（here: inswapper_128）
     - 目标脸先对齐成 112×112
     - 源脸过 ArcFace（w600k_r50）得到 512 维身份向量
     - 身份向量再乘一个 512×512 的矩阵（emap，存在 onnx 里当 initializer），
       归一化后当作 AdaIN 的风格输入 —— 这就是"把谁的脸"这件事的全部信息量
     - 输出 128×128 的脸，再按对齐的逆变换贴回原图

本脚本用**同一个人**做自换脸 smoke test（源 = 目标 = 同一张图）。
不是为了出效果，是为了验证流水线每一步都跑得通并且贴回去不糊边。

注意：inswapper / buffalo_l 这类权重只授权**非商业研究**用途，仓库代码是 MIT
但模型不是；拿真人（尤其公众人物）的脸做换脸还涉及肖像权。这里只做自换脸。

用法：python face_swap_pipeline.py <人像图> <模型目录> <输出目录>
模型目录里需要：face_landmarker.task、inswapper_128.onnx、w600k_r50.onnx
"""
import os
import sys
import time

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from onnx import numpy_helper

# ArcFace 官方 112×112 五点位模板（左眼、右眼、鼻尖、左嘴角、右嘴角）
ARCFACE_DST_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def timed(name, log):
    """上下文管理器：累计同名阶段的耗时。"""
    class _T:
        def __enter__(self):
            self.t = time.perf_counter()
            return self

        def __exit__(self, *a):
            dt = (time.perf_counter() - self.t) * 1000
            log.setdefault(name, []).append(dt)
    return _T()


def five_points(pts, w, h):
    """从 478 点里取对齐用的 5 个点。

    眼睛用虹膜中心（468/473），嘴角用 61/291。取完统一按 x 排序，
    这样不管 MediaPipe 的点位命名左右如何约定，顺序都和模板一致。
    """
    eyes = sorted([pts[468], pts[473]], key=lambda p: p[0])
    mouth = sorted([pts[61], pts[291]], key=lambda p: p[0])
    five = np.array([
        [eyes[0][0] * w, eyes[0][1] * h],
        [eyes[1][0] * w, eyes[1][1] * h],
        [pts[4][0] * w, pts[4][1] * h],
        [mouth[0][0] * w, mouth[0][1] * h],
        [mouth[1][0] * w, mouth[1][1] * h],
    ], dtype=np.float32)
    return five


def align(img, five, size):
    """相似变换（旋转+等比缩放+平移，不含剪切）把 5 个点搬到模板上。"""
    dst = ARCFACE_DST_112 * (size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(five, dst, method=cv2.LMEDS)
    return cv2.warpAffine(img, M, (size, size), borderValue=0.0), M


def arcface_embedding(face_bgr, sess, size=112):
    """ArcFace 输入规范：BGR→RGB、mean/scale = 127.5/127.5。"""
    blob = cv2.dnn.blobFromImage(face_bgr, 1 / 127.5, (size, size),
                                 (127.5, 127.5, 127.5), swapRB=True)
    out = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
    return out[0].astype(np.float32)


def extract_emap(onnx_path):
    """identity vector → AdaIN 风格向量的那个矩阵，藏在 initializer 里。

    inswapper 把 512×512 的 emap 当作一个没人用的 initializer（名为 buff2fs），
    runtime 不碰它，但权重文件里带着 —— 所以直接从 onnx 的 graph 里翻出来，
    条件是形状恰好 [512, 512]。用形状找比硬编码名字稳。
    """
    g = onnx.load(onnx_path).graph
    for t in g.initializer:
        if list(t.dims) == [512, 512]:
            return numpy_helper.to_array(t), t.name
    raise RuntimeError("没找到 512×512 的 emap initializer")


def paste_back(frame, swapped_128, M128):
    """把 128×128 的换脸结果贴回原图：椭圆掩码 + 高斯羽化 + Poisson 融合。"""
    inv = cv2.invertAffineTransform(M128)
    h, w = frame.shape[:2]
    warped = cv2.warpAffine(swapped_128, inv, (w, h), flags=cv2.INTER_LINEAR)

    mask = np.zeros((128, 128), np.uint8)
    cv2.ellipse(mask, (64, 64), (58, 58), 0, 0, 360, 255, -1)     # 脸内区域
    mask = cv2.GaussianBlur(mask, (0, 0), 8)                       # 羽化边缘
    wmask = cv2.warpAffine(mask, inv, (w, h))[..., None].astype(np.float32) / 255.0

    blend = (warped.astype(np.float32) * wmask
             + frame.astype(np.float32) * (1 - wmask)).astype(np.uint8)

    # Poisson 融合再修一次底 —— 需要 bbox，取掩码的非零范围
    ys, xs = np.where(wmask[..., 0] > 0.02)
    if len(xs) > 10:
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        center = ((x0 + x1) // 2, (y0 + y1) // 2)
        try:
            blended = cv2.seamlessClone(blend, frame, (wmask[..., 0] * 255).astype(np.uint8),
                                        center, cv2.NORMAL_CLONE)
            return blended, blend
        except cv2.error:
            return blend, blend
    return blend, blend


def main():
    src, modeldir, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    swap_sess = ort.InferenceSession(os.path.join(modeldir, "inswapper_128.onnx"),
                                     opts, providers=["CPUExecutionProvider"])
    arc_sess = ort.InferenceSession(os.path.join(modeldir, "w600k_r50.onnx"),
                                    opts, providers=["CPUExecutionProvider"])
    emap, emap_name = extract_emap(os.path.join(modeldir, "inswapper_128.onnx"))

    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=os.path.join(modeldir, "face_landmarker.task")),
        num_faces=1, running_mode=vision.RunningMode.IMAGE))

    frame = cv2.imread(src)
    h, w = frame.shape[:2]
    print(f"自换脸 smoke test：源 = 目标 = {os.path.basename(src)}（{w}×{h}）")
    print(f"换脸模型输入：{[i.name for i in swap_sess.get_inputs()]}")
    print(f"emap：initializer『{emap_name}』{emap.shape}，"
          f"min={emap.min():.3f} max={emap.max():.3f}\n")

    log = {}

    with timed("1 检测关键点", log):
        import mediapipe as mp
        res = lm.detect(mp.Image.create_from_file(src))
    pts = np.array([(p.x, p.y) for p in res.face_landmarks[0]], dtype=np.float32)
    five = five_points(pts, w, h)

    with timed("2 对齐（112 给识别 / 128 给换脸）", log):
        face112_src, _ = align(frame, five, 112)          # 源脸给 ArcFace（112）
        face128_dst, M128 = align(frame, five, 128)       # 目标脸给换脸网络（128）

    with timed("3 ArcFace 身份向量", log):
        emb = arcface_embedding(face112_src, arc_sess)
    print(f"身份向量 512 维：norm = {np.linalg.norm(emb):.1f}，"
          f"前 5 维 = {np.round(emb[:5], 2)}")

    with timed("4 emap 投影 + 归一化", log):
        # insightface 的规范：身份向量先归一化，再投影，再归一化。
        # 顺序不能省 —— 直接拿模长 20 的向量去投影，AdaIN 的风格强度会被放大 20 倍，
        # 贴回去就是一张花脸（这一步当时就是这么翻车的）。
        unit = emb / (np.linalg.norm(emb) + 1e-9)
        latent = unit @ emap.astype(np.float32)
        latent = latent / (np.linalg.norm(latent) + 1e-9)
    print(f"AdaIN 风格向量：norm = {np.linalg.norm(latent):.3f}（归一化到 1）")

    blob = cv2.dnn.blobFromImage(face128_dst, 1 / 255.0, (128, 128),
                                 (0, 0, 0), swapRB=True)
    with timed("5 换脸网络推理", log):
        swapped = swap_sess.run(None, {
            swap_sess.get_inputs()[0].name: blob.astype(np.float32),
            swap_sess.get_inputs()[1].name: latent.reshape(1, -1).astype(np.float32),
        })[0]
    out128 = np.clip(swapped[0].transpose(1, 2, 0)[:, :, ::-1] * 255, 0, 255).astype(np.uint8)
    print(f"换脸网络输出：{out128.shape}")

    with timed("6 贴回原图", log):
        blended, plain = paste_back(frame, out128, M128)

    print("\n各阶段耗时（单帧，CPU）：")
    total = 0.0
    for k in sorted(log):
        med = float(np.median(log[k]))
        total += med
        print(f"  {k:<18}{med:>8.1f} ms")
    print(f"  {'合计':<18}{total:>8.1f} ms  →  {1000 / total:.0f} fps")

    cv2.imwrite(os.path.join(outdir, "swap_aligned_128.png"), face128_dst)
    cv2.imwrite(os.path.join(outdir, "swap_out_128.png"), out128)
    cv2.imwrite(os.path.join(outdir, "swap_pasted_plain.png"), plain)
    cv2.imwrite(os.path.join(outdir, "swap_pasted_poisson.png"), blended)
    print(f"\n输出：{outdir}")


if __name__ == "__main__":
    main()
