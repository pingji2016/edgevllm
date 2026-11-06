import html, math

FONT = "font-family='-apple-system,BlinkMacSystemFont,\"PingFang SC\",\"Noto Sans CJK SC\",sans-serif'"

def svg(w, h, parts):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"
            f"<rect width='{w}' height='{h}' fill='#ffffff'/>" + "".join(parts) + "</svg>")

def text(x, y, s, size=13, fill='#374151', anchor='start', weight='400'):
    return (f"<text x='{x}' y='{y}' {FONT} font-size='{size}' fill='{fill}' "
            f"text-anchor='{anchor}' font-weight='{weight}'>{html.escape(str(s))}</text>")

def vbars(title, items, unit, ref=None, note=None, w=880, h=380, fmt='{:.3f}', color='#2563eb', ref_label=''):
    pad_l, pad_r, pad_t, pad_b = 70, 30, 60, 78
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    vmax = max(v for _, v in items + ([('ref', ref)] if ref else [])) * 1.18
    out = [text(pad_l, 34, title, 17, '#111827', weight='600')]
    # grid
    for i in range(5):
        v = vmax * i / 4
        y = pad_t + ph - ph * (v / vmax)
        out.append(f"<line x1='{pad_l}' y1='{y:.1f}' x2='{pad_l+pw}' y2='{y:.1f}' stroke='#e5e7eb' stroke-width='1'/>")
        out.append(text(pad_l - 10, y + 4, f"{v:.2f}", 11, '#9ca3af', 'end'))
    out.append(text(pad_l + pw, pad_t - 12, unit, 11, '#9ca3af', 'end'))
    if ref:
        y = pad_t + ph - ph * (ref / vmax)
        out.append(f"<line x1='{pad_l}' y1='{y:.1f}' x2='{pad_l+pw}' y2='{y:.1f}' stroke='#ef4444' stroke-width='1.4' stroke-dasharray='6 4'/>")
        out.append(text(pad_l + pw - 2, y - 7, ref_label, 11, '#ef4444', 'end'))
    n = len(items)
    slot = pw / n
    bw = min(86, slot * 0.56)
    for i, (label, v) in enumerate(items):
        x = pad_l + slot * (i + 0.5) - bw / 2
        bh = ph * (v / vmax)
        y = pad_t + ph - bh
        out.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bw:.1f}' height='{bh:.1f}' rx='5' fill='{color}' opacity='0.88'/>")
        out.append(text(x + bw / 2, y - 8, fmt.format(v), 12, '#111827', 'middle', '600'))
        out.append(text(pad_l + slot * (i + 0.5), pad_t + ph + 20, label, 12, '#4b5563', 'middle'))
    out.append(f"<line x1='{pad_l}' y1='{pad_t+ph}' x2='{pad_l+pw}' y2='{pad_t+ph}' stroke='#9ca3af' stroke-width='1'/>")
    if note:
        out.append(text(pad_l, h - 18, note, 11, '#6b7280'))
    return svg(w, h, out)

def hbars(title, items, unit, note=None, w=880, h=300, fmt='{:.2f}', colors=None):
    pad_l, pad_r, pad_t, pad_b = 150, 90, 52, 46
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    vmax = max(v for _, v, _ in items) * 1.12
    out = [text(24, 32, title, 17, '#111827', weight='600')]
    n = len(items); slot = ph / n
    for i, (label, v, c) in enumerate(items):
        y = pad_t + i * slot + slot * 0.22
        bh = slot * 0.5
        bw = pw * v / vmax
        col = c or (colors[i] if colors else '#2563eb')
        out.append(f"<rect x='{pad_l}' y='{y:.1f}' width='{max(bw,2):.1f}' height='{bh:.1f}' rx='5' fill='{col}' opacity='0.9'/>")
        out.append(text(pad_l - 12, y + bh * 0.72, label, 12.5, '#374151', 'end'))
        out.append(text(pad_l + bw + 10, y + bh * 0.72, f"{fmt.format(v)} {unit}", 12.5, '#111827', 'start', '600'))
    out.append(text(24, h - 14, note, 11, '#6b7280'))
    return svg(w, h, out)

# 1) ASR 吞吐随长度劣化
open('/tmp/k3forum/asr-rtf.svg', 'w').write(vbars(
    'SenseVoice ASR：RTF 随音频长度近似二次增长（K3 板内实测）',
    [('5.1 s', 0.150), ('10.1 s', 0.216), ('20.1 s', 0.405), ('40.2 s', 0.807)],
    'RTF（越低越快）', ref=1.0, ref_label='RTF = 1.0（刚好实时）',
    note='数据：scripts/asr-length.sh  ·  RTF 0.15 → 0.81，40 秒音频几乎只能跑 1.2× 实时 → 长音频必须先 VAD 切段',
    color='#f59e0b', fmt='{:.3f}'))

# 2) 首音延迟
open('/tmp/k3forum/first-audio.svg', 'w').write(hbars(
    '同一个问题，用户听到第一声的时间（秒）',
    [('流式：LLM token → 按标点切句 → TTS append', 3.03, '#10b981'),
     ('非流式：等 LLM 全部生成 → 整段 TTS', 28.41, '#ef4444')],
    's', w=880, h=240, fmt='{:.2f}',
    note='LLM 首 token 1.59 s；流式方案 3.03 s 出声，快 9.4×（scripts/stream-voice.py）'))

# 3) 硬编 vs 软编
open('/tmp/k3forum/encode.svg', 'w').write(vbars(
    '1080p / 150 帧编码墙钟时间（越低越好，K3 板内实测）',
    [('VPU 硬编 HEVC', 0.792), ('VPU 硬编 H.264', 1.087), ('x264 veryfast', 5.219)],
    '秒（150 帧）', note='硬编 138 fps；CPU 时间 0.20 s vs 29.28 s（少 144×）· 输入同一份 NV12，排除解码干扰',
    color='#7c3aed', fmt='{:.2f}'))
print('ok')
