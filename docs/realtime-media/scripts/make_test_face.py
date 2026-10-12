#!/usr/bin/env python3
"""
生成一张**程序化合成**的人脸测试图。

这个目录里所有实验都只用合成脸 —— 不用任何真人照片（尤其是公众人物）：
换脸技术本身是双刃的，仓库里不该出现拿真人脸跑出来的结果，哪怕是自换脸。
一张画出来的人脸足够让检测器/对齐/换脸网络都跑起来，也足够演示"贴回是否糊边"。

做法：皮肤底 + 椭圆脸部 + 眉毛/眼白/虹膜/鼻影/嘴，全部画完再叠一层噪点和高斯模糊
（检测器吃的是真实图像统计，纯色块容易被判成非人脸）。

用法：python make_test_face.py <输出.png> [宽] [高]
"""
import sys

import cv2
import numpy as np

SKIN = (168, 196, 226)          # BGR
SKIN_DARK = (140, 168, 198)
SKIN_LIGHT = (196, 216, 238)
LINE = (96, 84, 76)
IRIS = (96, 62, 40)


def make_face(w=760, h=1000, seed=7):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 236, np.uint8)                     # 浅灰背景
    cx, cy = w // 2, int(h * 0.46)                              # 脸的中心
    a, b = int(w * 0.30), int(h * 0.32)                         # 半长轴
    eye_y = cy - int(b * 0.20)
    eye_dx = int(a * 0.46)
    eye_r = int(a * 0.20)

    # 先画脖子和衣服（脸待会儿盖在上面，省得下巴被脖子糊掉）
    neck = cv2.ellipse(np.zeros((h, w), np.uint8), (cx, cy + b - 20),
                       (int(a * 0.30), int(b * 0.22)), 0, 0, 360, 255, -1)
    img[neck > 0] = (150, 176, 206)
    img[h - int(h * 0.10):] = (96, 92, 88)                      # 衣服

    # 脸：竖直椭圆，用渐变模拟侧光（左亮右暗）
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    inside = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1.0
    shade = np.clip(1.06 - 0.20 * (xx - cx) / a, 0.72, 1.18)
    skin = np.clip(np.array(SKIN, np.float32)[None, None, :] * shade[..., None], 0, 255)
    img[inside] = skin[inside].astype(np.uint8)

    # 眉毛
    for s in (-1, 1):
        p = np.array([[cx + s * eye_dx - s * eye_r - int(0.10 * a), eye_y - int(eye_r * 0.95)],
                      [cx + s * eye_dx, eye_y - int(eye_r * 1.45)],
                      [cx + s * eye_dx + s * eye_r + int(0.10 * a), eye_y - int(eye_r * 1.05)]])
        cv2.polylines(img, [p.astype(np.int32)], False, (58, 48, 42), max(3, eye_r // 4), cv2.LINE_AA)

    # 眼睛：眼白 + 虹膜 + 瞳孔 + 高光
    for s in (-1, 1):
        ec = (cx + s * eye_dx, eye_y)
        cv2.ellipse(img, ec, (eye_r, int(eye_r * 0.62)), 0, 0, 360, (248, 248, 246), -1, cv2.LINE_AA)
        cv2.circle(img, ec, int(eye_r * 0.50), IRIS, -1, cv2.LINE_AA)
        cv2.circle(img, ec, int(eye_r * 0.24), (32, 26, 24), -1, cv2.LINE_AA)
        cv2.circle(img, (ec[0] - int(eye_r * 0.18), ec[1] - int(eye_r * 0.20)),
                   max(1, int(eye_r * 0.12)), (255, 255, 255), -1, cv2.LINE_AA)

    # 鼻：两侧阴影 + 鼻头 + 鼻孔
    n = (cx, cy + int(b * 0.16))
    cv2.ellipse(img, (n[0] - int(a * 0.14), n[1]), (int(a * 0.07), int(b * 0.14)),
                0, 0, 360, SKIN_DARK, -1, cv2.LINE_AA)
    cv2.ellipse(img, (n[0] + int(a * 0.14), n[1]), (int(a * 0.07), int(b * 0.14)),
                0, 0, 360, SKIN_DARK, -1, cv2.LINE_AA)
    cv2.circle(img, n, int(a * 0.085), SKIN_LIGHT, -1, cv2.LINE_AA)
    for s in (-1, 1):
        cv2.circle(img, (n[0] + s * int(a * 0.05), n[1] + int(b * 0.02)),
                   max(2, int(a * 0.022)), (92, 76, 68), -1, cv2.LINE_AA)

    # 嘴：一条闭合的唇线 + 下唇高光
    m = (cx, cy + int(b * 0.52))
    pts = np.array([[m[0] - int(a * 0.30), m[1]],
                    [m[0] - int(a * 0.14), m[1] - int(b * 0.030)],
                    [m[0], m[1] + int(b * 0.012)],
                    [m[0] + int(a * 0.14), m[1] - int(b * 0.030)],
                    [m[0] + int(a * 0.30), m[1]]])
    cv2.polylines(img, [pts.astype(np.int32)], False, (86, 66, 72), max(3, b // 40), cv2.LINE_AA)
    cv2.ellipse(img, (m[0], m[1] + int(b * 0.055)), (int(a * 0.16), int(b * 0.035)),
                0, 0, 360, (150, 140, 176), -1, cv2.LINE_AA)

    # 真实图像统计：轻微模糊（去掉硬边）+ 噪点
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    img = np.clip(img.astype(np.float32) + rng.normal(0, 2.2, img.shape), 0, 255).astype(np.uint8)
    return img


def main():
    out = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 760
    h = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    img = make_face(w, h)
    cv2.imwrite(out, img)
    print(f"合成脸已写入 {out}（{w}×{h}）—— 本目录所有实验都用它，不含任何真人照片")


if __name__ == "__main__":
    main()
