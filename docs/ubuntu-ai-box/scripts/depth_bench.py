"""Depth-Anything-V2-Small 逐帧推理基准（端侧视觉链路相关）"""
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MD = "/home/pingji/.cache/modelscope/models/depth-anything--Depth-Anything-V2-Small-hf/snapshots/master"
IMG = os.path.expanduser("~/apps/camfilter/bg1.png")


def bench(fn, warmup=3, iters=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


def main():
    proc = AutoImageProcessor.from_pretrained(MD)
    model = AutoModelForDepthEstimation.from_pretrained(
        MD, dtype=torch.bfloat16, device_map={"": 0}).eval()
    n_param = sum(p.numel() for p in model.parameters())
    print(f"参数 {n_param/1e6:.1f} M | 权重 {sum(p.numel()*p.element_size() for p in model.parameters())/2**20:.0f} MiB\n")

    base = Image.open(IMG).convert("RGB")
    print(f"原图 {base.size}\n")

    rows = []
    for tag, size in (("518x518（模型原生）", (518, 518)),
                      ("640x480（端侧常见）", (640, 480)),
                      ("960x544（720p 附近）", (960, 544))):
        img = base.resize(size)
        inputs = proc(images=img, return_tensors="pt").to("cuda")
        torch.cuda.reset_peak_memory_stats()
        ms = bench(lambda: model(**inputs))
        vram = torch.cuda.max_memory_allocated() / 2**20
        rows.append(dict(size=tag, ms=round(ms, 2), fps=round(1000 / ms, 1), vram_mib=round(vram)))
        print(f"  {tag:22s} {ms:6.2f} ms/帧 | {1000/ms:5.1f} FPS | 峰值显存 {vram:.0f} MiB", flush=True)

    # 与相机链路对比：16.6 FPS 预算
    print("\n对照：camfilter 的 detect 模式 6.7 ms/帧、整链 16.6 FPS（60 ms 预算）")
    json.dump(dict(params_m=round(n_param / 1e6, 1), rows=rows),
              open(os.path.expanduser("~/apps/infer-research/depth_anything.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
