"""多模态 RAG：文搜图

24 张受控合成图（6 色 × 4 形状），12 条中英文查询
对比：
  openai/clip-vit-base-patch32（英文训练）  中文 vs 英文
  OFA-Sys/chinese-clip-vit-base-patch16     中文
"""
import json
import os
import time

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, CLIPModel

OUT = "/tmp/mm_rag"
COLORS = {"红": (220, 30, 30), "蓝": (30, 60, 220), "绿": (30, 170, 60),
          "黄": (240, 200, 20), "紫": (150, 40, 180), "橙": (240, 130, 20)}
EN = {"红": "red", "蓝": "blue", "绿": "green", "黄": "yellow",
      "紫": "purple", "橙": "orange",
      "圆形": "circle", "正方形": "square", "三角形": "triangle", "星星": "star"}


def make_images():
    os.makedirs(OUT, exist_ok=True)
    names = []
    for cname, rgb in COLORS.items():
        for shape in ("圆形", "正方形", "三角形", "星星"):
            img = Image.new("RGB", (224, 224), (250, 250, 250))
            d = ImageDraw.Draw(img)
            if shape == "圆形":
                d.ellipse([40, 40, 184, 184], fill=rgb)
            elif shape == "正方形":
                d.rectangle([40, 40, 184, 184], fill=rgb)
            elif shape == "三角形":
                d.polygon([(112, 30), (200, 190), (24, 190)], fill=rgb)
            else:  # 五角星
                import math
                pts = []
                for i in range(10):
                    r = 90 if i % 2 == 0 else 36
                    a = -math.pi / 2 + i * math.pi / 5
                    pts.append((112 + r * math.cos(a), 112 + r * math.sin(a)))
                d.polygon(pts, fill=rgb)
            p = f"{OUT}/{cname}{shape}.png"
            img.save(p)
            names.append((p, f"{cname}{shape}"))
    return names


def _feat(out, proj=None):
    """get_*_features 已经返回投影后的向量，不要再投影一次"""
    if isinstance(out, torch.Tensor):
        return out
    x = getattr(out, "pooler_output", None)
    return x if x is not None else out.last_hidden_state[:, 0]


def evaluate(model, proc, imgs, queries, name):
    t0 = time.time()
    with torch.no_grad():
        inp = proc(images=[Image.open(im).convert("RGB") for im, _ in imgs],
                   return_tensors="pt").to("cuda")
        feats = _feat(model.get_image_features(**inp))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        hit = 0
        for q, target in queries:
            tin = proc(text=[q], return_tensors="pt", padding=True).to("cuda")
            tf = _feat(model.get_text_features(**tin))
            tf = tf / tf.norm(dim=-1, keepdim=True)
            idx = int((feats @ tf.T).argmax())
            hit += imgs[idx][1] == target
    return hit, time.time() - t0


def main():
    imgs = make_images()
    print(f"  图片 {len(imgs)} 张 -> {OUT}", flush=True)

    zh = [("红色的圆形", "红圆形"), ("蓝色的正方形", "蓝正方形"), ("绿色的三角形", "绿三角形"),
          ("黄色的星星", "黄星星"), ("紫色的圆形", "紫圆形"), ("橙色的正方形", "橙正方形"),
          ("蓝色的星星", "蓝星星"), ("红色的三角形", "红三角形"), ("绿色的圆形", "绿圆形"),
          ("黄色的正方形", "黄正方形"), ("紫色的三角形", "紫三角形"), ("橙色的星星", "橙星星")]
    en = [(f"{EN[c]} {EN[s]}", f"{c}{s}") for c, s in
          [("红", "圆形"), ("蓝", "正方形"), ("绿", "三角形"), ("黄", "星星"),
           ("紫", "圆形"), ("橙", "正方形"), ("蓝", "星星"), ("红", "三角形"),
           ("绿", "圆形"), ("黄", "正方形"), ("紫", "三角形"), ("橙", "星星")]]

    res = {}
    # 原版 CLIP
    MD = "/home/pingji/models/clip-b32"
    m = CLIPModel.from_pretrained(MD).cuda().eval()
    proc = AutoProcessor.from_pretrained(MD)
    h, dt = evaluate(m, proc, imgs, zh, "clip-zh")
    res["clip_zh"] = h / len(zh)
    print(f"  原版 CLIP 中文：{h}/{len(zh)} = {h/len(zh)*100:.0f}%（{dt:.1f}s）", flush=True)
    h, dt = evaluate(m, proc, imgs, en, "clip-en")
    res["clip_en"] = h / len(en)
    print(f"  原版 CLIP 英文：{h}/{len(en)} = {h/len(en)*100:.0f}%（{dt:.1f}s）", flush=True)
    del m
    torch.cuda.empty_cache()

    # Chinese-CLIP
    try:
        from modelscope import snapshot_download
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        last = None
        for mid in ("AI-ModelScope/chinese-clip-vit-base-patch16",
                    "OFA-Sys/chinese-clip-vit-base-patch16",
                    "damo/multi-modal_chinese_clip_vit-base-patch16"):
            try:
                p = snapshot_download(mid)
                break
            except Exception as e:
                last = e
        else:
            raise last
        m = ChineseCLIPModel.from_pretrained(p).cuda().eval()
        proc = ChineseCLIPProcessor.from_pretrained(p)
        h, dt = evaluate(m, proc, imgs, zh, "chinese-clip")
        res["chinese_clip_zh"] = h / len(zh)
        print(f"  Chinese-CLIP 中文：{h}/{len(zh)} = {h/len(zh)*100:.0f}%（{dt:.1f}s）", flush=True)
    except Exception as e:
        res["chinese_clip_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        print(f"  Chinese-CLIP 失败：{type(e).__name__}: {str(e)[:120]}", flush=True)

    json.dump(res, open("/home/pingji/apps/minimind/mm_rag.json", "w"), indent=2)
    print("已存 mm_rag.json")


if __name__ == "__main__":
    main()
