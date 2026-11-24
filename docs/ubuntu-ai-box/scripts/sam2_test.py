"""SAM2 逐帧分割：点提示 → mask，测单帧耗时与显存"""
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

MD = os.path.expanduser("~/.cache/modelscope/models/facebook--sam2.1-hiera-tiny/snapshots/master")
IMG = "/tmp/face.png"
OUT = os.path.expanduser("~/apps/infer-research/sam2_result.json")


def main():
    t0 = time.time()
    proc = Sam2Processor.from_pretrained(MD)
    model = Sam2Model.from_pretrained(MD, dtype=torch.bfloat16).cuda().eval()
    print(f"加载 {time.time()-t0:.1f}s | 参数 {sum(p.numel() for p in model.parameters())/1e6:.1f} M")

    img = Image.open(IMG).convert("RGB")
    W, H = img.size
    print(f"图像 {W}×{H}")

    # 点提示：画面中心
    pts = [[[[W // 2, H // 2]]]]  # [image][object][point][xy]
    inputs = proc(images=img, input_points=pts, return_tensors="pt").to("cuda")

    # 预热
    with torch.inference_mode():
        out = model(**inputs)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    lats = []
    for _ in range(10):
        with torch.inference_mode():
            t0 = time.time()
            out = model(**inputs)
            torch.cuda.synchronize()
        lats.append(time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**20

    # 绕开 v5 变化的 post_process_masks：按“最后两维=空间”统一处理
    import torch.nn.functional as F
    pm = out.pred_masks
    print(f"  pred_masks {tuple(pm.shape)} | iou_scores {tuple(out.iou_scores.shape)}")
    flat = pm.reshape(-1, *pm.shape[-2:])                      # [N, h, w]
    up = F.interpolate(flat.unsqueeze(1).float(), size=(H, W), mode="bilinear",
                       align_corners=False).squeeze(1)         # [N, H, W]
    iou = out.iou_scores.detach().reshape(-1).float().cpu().numpy()
    best = int(np.argmax(iou))
    m = (up[best] > 0).detach().cpu().numpy()
    area = m.mean() * 100

    mean_ms = np.mean(lats) * 1000
    print(f"单帧均值 {mean_ms:.0f} ms（{1000/mean_ms:.1f} FPS）| 显存 {peak:.0f} MiB")
    print(f"最佳 mask: IoU 预测 {iou[best]:.3f} | 覆盖 {area:.1f}% 画面 | 候选 {len(iou)}")

    # 保存可视化
    vis = np.array(img).copy()
    vis[m] = (vis[m] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
    Image.fromarray(vis).save("/tmp/sam2_mask.png")
    print("  可视化: /tmp/sam2_mask.png")

    import json
    json.dump(dict(params_m=round(sum(p.numel() for p in model.parameters())/1e6, 1),
                   ms=round(mean_ms, 1), fps=round(1000/mean_ms, 1),
                   peak_mib=round(peak), iou=round(float(iou[best]), 3),
                   area_pct=round(float(area), 1)), open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
