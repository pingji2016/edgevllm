import html, math

FONT = "font-family='-apple-system,BlinkMacSystemFont,\"PingFang SC\",\"Noto Sans CJK SC\",sans-serif'"

def text(x, y, s, size=13, fill='#374151', anchor='start', weight='400'):
    return (f"<text x='{x}' y='{y}' {FONT} font-size='{size}' fill='{fill}' "
            f"text-anchor='{anchor}' font-weight='{weight}'>{html.escape(str(s))}</text>")

items = [
    ("AGX Orin 64G · TensorRT INT8", 2.30, "#8b5cf6"),
    ("Orin NX 16G · TensorRT INT8", 3.49, "#8b5cf6"),
    ("Orin Nano Super · TensorRT INT8", 3.80, "#8b5cf6"),
    ("K3 · SpaceMIT EP int8（本机实测）", 14.8, "#16a34a"),
    ("树莓派 5 · NCNN（CPU）", 67.03, "#2563eb"),
    ("树莓派 5 · ONNX Runtime（CPU）", 125.99, "#2563eb"),
    ("K3 · ONNX Runtime CPU（本机实测）", 832, "#f59e0b"),
]

W, H = 940, 430
pad_l, pad_r, pad_t, pad_b = 300, 110, 84, 74
pw, ph = W - pad_l - pad_r, H - pad_t - pad_b
lo, hi = 1.0, 1500.0
def X(v):
    return pad_l + pw * (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))

out = []
out.append(text(24, 34, "同一张图、单帧推理，谁快谁慢（对数刻度）", 17, '#111827', weight='600'))
out.append(text(24, 56, "YOLO n 级 / 640 / 不含前后处理；K3 为本次实测，其余为 Ultralytics 官方 benchmark（YOLO26n）", 11.5, '#6b7280'))

# grid + ticks
for v in [1, 10, 100, 1000]:
    x = X(v)
    out.append(f"<line x1='{x:.1f}' y1='{pad_t-6}' x2='{x:.1f}' y2='{pad_t+ph}' stroke='#e5e7eb' stroke-width='1'/>")
    out.append(text(x, pad_t + ph + 18, f"{v} ms", 11, '#9ca3af', 'middle'))

n = len(items)
slot = ph / n
for i, (label, v, col) in enumerate(items):
    y = pad_t + i * slot + slot * 0.24
    bh = slot * 0.46
    bw = max(X(v) - pad_l, 2)
    out.append(f"<rect x='{pad_l}' y='{y:.1f}' width='{bw:.1f}' height='{bh:.1f}' rx='5' fill='{col}' opacity='0.9'/>")
    out.append(text(pad_l - 12, y + bh * 0.7, label, 12, '#374151', 'end'))
    txt = f"{v:.1f} ms" if v < 100 else f"{v:.0f} ms"
    out.append(text(pad_l + bw + 10, y + bh * 0.7, txt, 12.5, '#111827', 'start', '600'))

out.append(f"<line x1='{pad_l}' y1='{pad_t}' x2='{pad_l}' y2='{pad_t+ph}' stroke='#9ca3af' stroke-width='1'/>")
out.append(text(24, H - 34, "K3 的加速路径比树莓派 5 最快的 CPU 路径快 4.5×；但同一块板换回通用 ONNX，反而比树莓派 5 慢 6.6×。", 12, '#111827'))
out.append(text(24, H - 14, "差的不是算力，是算子库：RISC-V 上的 ONNX Runtime 还没人优化，IME 的 int8 内核只有厂方 EP 有。", 11.5, '#6b7280'))

svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>"
       f"<rect width='{W}' height='{H}' fill='#ffffff'/>" + "".join(out) + "</svg>")
open('/tmp/exc/device-cmp.svg', 'w').write(svg)
print('ok', len(svg))
