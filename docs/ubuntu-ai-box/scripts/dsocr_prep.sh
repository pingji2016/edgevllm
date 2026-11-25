#!/bin/bash
set -u
echo "=== 后台下载 DeepSeek-OCR ==="
mkdir -p ~/models
if [ -f ~/models/.dsocr_done ]; then
  echo "  已下载"
else
  nohup ~/apps/bench/yolo-venv/bin/python - > /tmp/dsocr_dl.log 2>&1 <<'PY' &
from modelscope import snapshot_download
p = snapshot_download("deepseek-ai/DeepSeek-OCR")
print("DONE", p)
open("/home/pingji/models/.dsocr_done", "w").write(p)
PY
  echo "  下载已启动 PID=$!"
fi

echo
echo "=== 中文字体 ==="
fc-list :lang=zh 2>/dev/null | head -3 || echo "  无中文字体"

echo
echo "=== 准备测试文档图（中英混排 + 表格）==="
~/apps/bench/yolo-venv/bin/python - <<'PY'
import os
from PIL import Image, ImageDraw, ImageFont

cands = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]
font_path = next((c for c in cands if os.path.exists(c)), None)
print("  字体:", font_path or "未找到（用默认，中文可能显示为方块）")

W, H = 1240, 1754  # A4 @150dpi
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)
f_title = ImageFont.truetype(font_path, 42) if font_path else ImageFont.load_default()
f_body = ImageFont.truetype(font_path, 26) if font_path else ImageFont.load_default()
f_small = ImageFont.truetype(font_path, 20) if font_path else ImageFont.load_default()

y = 80
d.text((80, y), "端侧推理部署说明 v1.2", font=f_title, fill="black"); y += 80
lines = [
    "一、概述",
    "本设备支持在 16GB 显存上运行 30 亿参数的多模态模型。",
    "典型延迟为每帧 410 毫秒，吞吐约 2.4 帧每秒。",
    "",
    "二、关键指标",
    "  · 显存占用：6.2 GB",
    "  · 首 token 延迟：125 ms",
    "  · 解码速度：55.3 tokens/s（32K 上下文）",
    "",
    "三、注意事项",
    "1. 禁止在未预热的情况下测量首帧延迟。",
    "2. 前缀缓存必须开启，16K 前缀可节省 89% 的预填充时间。",
    "3. 量化模型的选用取决于是否有该架构的原生内核。",
]
for line in lines:
    d.text((80, y), line, font=f_body if line and not line.startswith("一") and not line.startswith("二") and not line.startswith("三") else f_title if line else f_body, fill="black")
    y += 46 if line else 24

y += 20
d.text((80, y), "Model | VRAM | Decode", font=f_body, fill="black"); y += 40
rows = [("Qwen3-1.7B", "3.4 GB", "62.3 tok/s"), ("Qwen3-8B Q4", "5.0 GB", "72.7 tok/s"), ("Qwen3-VL-8B", "6.2 GB", "0.41 s/frame")]
for r in rows:
    d.text((80, y), f"{r[0]:<16}{r[1]:<10}{r[2]}", font=f_small, fill="black"); y += 32

y += 30
d.text((80, y), 'const answer = "压缩比 10 倍时，OCR 精度仍有 97%。";', font=f_small, fill="black")

out = "/tmp/ocr_doc.png"
img.save(out)
print(f"  已生成 {out} ({W}×{H})")

# 参考文本（用于计算字符级准确率）
ref = ["端侧推理部署说明 v1.2", "一、概述", "本设备支持在16GB显存上运行30亿参数的多模态模型。",
       "典型延迟为每帧410毫秒，吞吐约2.4帧每秒。", "二、关键指标", "显存占用：6.2GB", "首token延迟：125ms",
       "解码速度：55.3tokens/s（32K上下文）", "三、注意事项", "禁止在未预热的情况下测量首帧延迟。",
       "前缀缓存必须开启，16K前缀可节省89%的预填充时间。", "量化模型的选用取决于是否有该架构的原生内核。",
       "Qwen3-1.7B", "Qwen3-8BQ4", "Qwen3-VL-8B", "压缩比10倍时，OCR精度仍有97%。"]
open("/tmp/ocr_ref.txt", "w", encoding="utf-8").write("\n".join(ref))
print(f"  参考文本 {len(ref)} 行 → /tmp/ocr_ref.txt")
PY

echo
echo "=== 下载进度 ==="
sleep 20
du -sh ~/.cache/modelscope/models/deepseek-ai--DeepSeek-OCR 2>/dev/null || echo "  刚开始"
