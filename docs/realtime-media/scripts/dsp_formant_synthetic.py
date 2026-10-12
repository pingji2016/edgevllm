#!/usr/bin/env python3
"""
变声里最容易被忽略的一件事：**音高和音色是两个可独立搬动的量**。

- 音高 F0：声带振动频率
- 音色（共振峰 F1/F2/F3）：声道形状决定的谱包络
- 只搬 F0 = 自然的变调；F0 和共振峰一起搬 = 花栗鼠/怪兽（chipmunk effect），
  因为真实的人不会一边把声带拉紧 1.5 倍、一边把声道缩短 1.5 倍

要证明这件事必须知道真值，所以这里**合成**一段元音：
脉冲串（F0=120Hz）→ 三个级联的二阶谐振器（F1/F2/F3 = 700/1200/2500）。

然后用两把尺子量：
  · LPC 求根 → 共振峰位置（在合成元音上先验证这把尺子准不准）
  · 倒谱低通 → 谱包络的整体搬移量 α（α≈1 表示共振峰没动，α≈音高比 表示一起搬了）

最后量"自己修包络"：拿变调后的信号，把原元音的包络重新扣回去，
看共振峰能不能被拉回原处 —— 这就是 formant-preserving 变调的底层操作。

用法：python dsp_formant_synthetic.py <输出目录>
需要 ffmpeg（librubberband）。
"""
import os
import subprocess
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter

SR = 16000
TRUTH = (700.0, 1200.0, 2500.0)
BANDWIDTHS = (80.0, 100.0, 120.0)
ALPHA_GRID = np.arange(0.60, 1.81, 0.01)


# ---------- 合成元音 ----------

def synth_vowel(f0=120.0, dur=1.0, sr=SR, formants=TRUTH, bw=BANDWIDTHS):
    """脉冲串激励 + 级联二阶谐振器 = 稳态元音。"""
    n = int(sr * dur)
    x = np.zeros(n)
    x[::max(1, int(round(sr / f0)))] = 1.0        # 脉冲串
    y = x
    for F, B in zip(formants, bw):
        r = np.exp(-np.pi * B / sr)               # 极点半径：决定带宽
        th = 2 * np.pi * F / sr                   # 极点角度：决定频率
        a1, a2 = -2 * r * np.cos(th), r * r
        b0 = 1 - 2 * r * np.cos(th) + r * r       # 归一化到直流增益 1
        y = lfilter([b0], [1.0, a1, a2], y)
    y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
    return y.astype(np.float64)


# ---------- 尺子 1：LPC 求共振峰 ----------

def levinson(r, order):
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    for i in range(1, order + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1:0:-1])
        k = -acc / e
        anew = a.copy()
        anew[i] = k
        for j in range(1, i):
            anew[j] = a[j] + k * a[i - j]
        a, e = anew, e * (1 - k * k)
    return a


def formants(x, sr=SR, order=20, n=3):
    """LPC 求根 → 共振峰频率（只保留上半平面、带宽合理的根）。"""
    seg = x[len(x) // 3: len(x) // 3 + int(sr * 0.03)].copy()
    seg *= np.hamming(len(seg))
    r = np.correlate(seg, seg, "full")[len(seg) - 1:]
    a = levinson(r, order)
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0]
    freqs = np.arctan2(np.imag(roots), np.real(roots)) * sr / (2 * np.pi)
    bw = -np.log(np.abs(roots)) * sr / np.pi
    keep = [(f, b) for f, b in zip(freqs, bw) if 50 < f < 0.45 * sr and b < 500]
    keep.sort()
    return [f for f, _ in keep[:n]]


# ---------- 尺子 2：谱包络 + 整体搬移量 α ----------

def global_envelope(x, sr=SR, nfft=1024, lifter=120):
    """帧平均的谱包络（倒谱低通 = 抹掉 F0 的谐波结构，只留声道形状）。"""
    win = np.hanning(nfft)
    thr = 0.15 * np.max(np.abs(x))
    acc, cnt = np.zeros(nfft // 2 + 1), 0
    for i in range(0, len(x) - nfft, nfft // 4):
        seg = x[i:i + nfft] * win
        if np.sqrt(np.mean(seg ** 2)) < thr:
            continue                              # 只看有声帧
        ceps = np.fft.irfft(np.log(np.abs(np.fft.rfft(seg)) + 1e-8))
        ceps[lifter:-lifter] = 0.0                # 倒谱低通
        acc += np.fft.rfft(ceps).real
        cnt += 1
    return acc / max(cnt, 1)


def envelope_shift_alpha(env_src, env_dst, sr=SR, nfft=1024, lo=150, hi=6000):
    """把 src 的包络在 log 频率上按 α 拉伸后去贴 dst，取最贴的 α。

    α = 1 表示共振峰没动；α = 音高比 表示共振峰跟着音高一起搬了。
    """
    f = np.arange(len(env_src)) * sr / nfft
    m = (f >= lo) & (f <= hi)
    lf = np.log2(f[m])
    g = np.linspace(lf[0], lf[-1], 256)
    a_src = np.interp(g, lf, env_src[m])
    a_dst = np.interp(g, lf, env_dst[m])
    a_src, a_dst = a_src - a_src.mean(), a_dst - a_dst.mean()
    best, best_err = 1.0, np.inf
    for alpha in ALPHA_GRID:
        warped = np.interp(g - np.log2(alpha), lf, env_src[m])
        warped -= warped.mean()
        err = np.mean((warped - a_dst) ** 2)
        if err < best_err:
            best_err, best = err, float(alpha)
    return best


# ---------- 自己修包络 ----------

def envelope_transplant(x, log_env_self, log_env_target, nfft=1024, hop=256, clip_db=20):
    """逐帧乘以增益，把 x 的谱包络换成 target 的，保留谐波与时间结构。"""
    gain = np.clip(log_env_target - log_env_self, -clip_db, clip_db)   # 对数域限幅
    g = np.exp(gain)
    win = np.hanning(nfft)
    out = np.zeros(len(x) + nfft)
    wsum = np.zeros_like(out)
    for i in range(0, len(x) - nfft + 1, hop):
        spec = np.fft.rfft(x[i:i + nfft] * win) * g
        out[i:i + nfft] += np.fft.irfft(spec, nfft) * win
        wsum[i:i + nfft] += win ** 2
    return out[:len(x)] / np.maximum(wsum[:len(x)], 1e-8)


# ---------- ffmpeg rubberband ----------

def rubberband(src, dst, pitch, formant, pitchq="speed"):
    if os.path.exists(dst):
        return dst
    af = f"rubberband=pitch={pitch}:formant={formant}:pitchq={pitchq}"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                    "-af", af, dst], check=True)
    return dst


def read(path):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float64) / (32768.0 if x.dtype == np.int16 else 1.0), sr


def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)

    x = synth_vowel()
    wavfile.write(os.path.join(outdir, "vowel_f0-120.wav"), SR,
                  (x * 32767).astype(np.int16))
    print(f"合成元音：F0=120Hz，真值共振峰 F1/F2/F3 = "
          f"{'/'.join(str(int(t)) for t in TRUTH)} Hz，{len(x) / SR:.1f}s\n")

    print("先验证尺子（LPC 求根）在合成元音上准不准：")
    f0 = formants(x)
    print(f"  LPC 量出来 = {'/'.join(f'{int(v)}' for v in f0)} Hz"
          f"    误差 = {'/'.join(f'{int(v - t):+d}' for v, t in zip(f0, TRUTH))}\n")

    src = os.path.join(outdir, "vowel_f0-120.wav")
    env_orig = global_envelope(x)

    print("ffmpeg rubberband 的两种模式（变调 ±，音色留不留）：")
    print(f"  {'变调':<8}{'formant=':<12}{'F1/F2/F3 (Hz)':<24}{'比值 vs 真值':<22}{'包络搬移 α':>10}")
    for pitch in (1.5, 0.7):
        for formant in ("shifted", "preserved"):
            path = rubberband(src, os.path.join(outdir, f"vowel_pitch{pitch}_{formant}.wav"),
                              pitch, formant)
            y, _ = read(path)
            fs = formants(y)
            ratio = "/".join(f"{v / t:.2f}" for v, t in zip(fs, TRUTH))
            alpha = envelope_shift_alpha(env_orig, global_envelope(y))
            print(f"  {pitch:<8}{formant:<12}"
                  f"{'/'.join(f'{int(v)}' for v in fs):<24}{ratio:<22}{alpha:>10.2f}")

    print("\n自己修包络（不动相位、只把原元音的包络扣回去）：")
    for pitch in (1.5, 0.7):
        path = os.path.join(outdir, f"vowel_pitch{pitch}_shifted.wav")
        y, _ = read(path)
        fixed = envelope_transplant(y, global_envelope(y), env_orig)
        wavfile.write(os.path.join(outdir, f"vowel_pitch{pitch}_自己修包络.wav"), SR,
                      (np.clip(fixed, -1, 1) * 32767).astype(np.int16))
        fs = formants(fixed)
        ratio = "/".join(f"{v / t:.2f}" for v, t in zip(fs, TRUTH))
        print(f"  变调 {pitch} 的自己修包络 = {'/'.join(f'{int(v)}' for v in fs):<22}"
              f"比值 {ratio}")

    print("\n结论：")
    print("  · formant=shifted   → F0 和共振峰一起搬，α ≈ 音高比（花栗鼠）")
    print("  · formant=preserved → 共振峰基本留在原位（α 更靠近 1），但 ffmpeg 的实现是近似的，")
    print("    高频共振峰（F3）仍会被带着走一些")
    print("  · 自己修包络=把变调后信号的谱包络按逐帧增益换回原包络，")
    print("    不需要知道 F0 就能把 F1/F2 拉回原处 —— 这就是 formant-preserving 的底层做法")
    print(f"\n输出目录：{outdir}")


if __name__ == "__main__":
    main()
