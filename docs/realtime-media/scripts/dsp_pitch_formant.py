#!/usr/bin/env python3
"""
在**真实语音**上量同一件事：变调之后，音高动了多少、音色（共振峰）动了多少。

合成元音（见 dsp_formant_synthetic.py）能拿到真值，但真实语音不是稳态的，
不能直接用 LPC 逐帧求根去比 —— 这里改用两个对非稳态更稳的量：

  · F0（音高）：自相关 + 抛物线插值，取有声帧的中位数
  · 频谱包络的整体搬移量 α：帧平均包络在 log 频率轴上按 α 拉伸后去贴合结果，
    取最贴的 α。α ≈ 1 表示共振峰没跟着走；α ≈ F0 比值 表示一起搬了。

用法：python dsp_pitch_formant.py <语音.wav> <输出目录>
需要 ffmpeg（librubberband）。
"""
import os
import subprocess
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

SR = 16000
ALPHA_GRID = np.arange(0.60, 1.81, 0.01)


def read(path, sr=SR):
    s, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64) / (32768.0 if x.dtype == np.int16 else 1.0)
    if s != sr:
        x = resample_poly(x, sr, s)
    return x


# ---------- F0：自相关 + 抛物线插值 ----------

def f0_track(x, sr=SR, fmin=70.0, fmax=400.0, frame=1024, hop=256):
    lo, hi = int(sr / fmax), int(sr / fmin)
    thr = 0.15 * np.max(np.abs(x))
    out = []
    for i in range(0, len(x) - frame, hop):
        seg = x[i:i + frame]
        if np.sqrt(np.mean(seg ** 2)) < thr:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[frame - 1:]
        ac /= (ac[0] + 1e-12)
        k = int(np.argmax(ac[lo:hi])) + lo
        if ac[k] < 0.3:                      # 不像周期信号（清音/静音）
            continue
        # 抛物线插值：用相邻两点把峰值位置细化到子样本精度
        y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        if abs(delta) < 1:
            out.append(sr / (k + delta))
    return np.array(out)


def f0_median(x):
    t = f0_track(x)
    return float(np.median(t)) if len(t) >= 10 else float("nan")


# ---------- 包络与 α ----------

def global_envelope(x, sr=SR, nfft=1024, lifter=120):
    """帧平均的谱包络。lifter 是倒谱低通的截断长度，**必须短于谐波间距**。

    谐波在倒谱上的位置 = sr / F0（这里 16000/246 ≈ 65 样本）。lifter 取 120
    时谐波峰仍在包络里，于是量出来的"包络"其实是谐波梳本身 —— 它会跟着音高走，
    让 shifted 和 preserved 量出同一个 α（这就是第一版踩的坑）。
    取 < 65 才把谐波抹掉、只留声道形状。
    """
    win = np.hanning(nfft)
    thr = 0.15 * np.max(np.abs(x))
    acc, cnt = np.zeros(nfft // 2 + 1), 0
    for i in range(0, len(x) - nfft, nfft // 4):
        seg = x[i:i + nfft] * win
        if np.sqrt(np.mean(seg ** 2)) < thr:
            continue
        ceps = np.fft.irfft(np.log(np.abs(np.fft.rfft(seg)) + 1e-8))
        ceps[lifter:-lifter] = 0.0
        acc += np.fft.rfft(ceps).real
        cnt += 1
    return acc / max(cnt, 1)


def envelope_shift_alpha(env_src, env_dst, sr=SR, nfft=1024, lo=200, hi=5000):
    f = np.arange(len(env_src)) * sr / nfft
    m = (f >= lo) & (f <= hi)
    lf = np.log2(f[m])
    g = np.linspace(lf[0], lf[-1], 256)
    a_dst = np.interp(g, lf, env_dst[m])
    a_dst -= a_dst.mean()
    best, best_err = 1.0, np.inf
    for alpha in ALPHA_GRID:
        warped = np.interp(g - np.log2(alpha), lf, env_src[m])
        warped -= warped.mean()
        err = np.mean((warped - a_dst) ** 2)
        if err < best_err:
            best_err, best = err, float(alpha)
    return best


def rubberband(src, dst, pitch, formant, pitchq="speed"):
    if os.path.exists(dst):
        return dst
    af = f"rubberband=pitch={pitch}:formant={formant}:pitchq={pitchq}"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-af", af, dst],
                   check=True)
    return dst


def main():
    src_path, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)

    x = read(src_path)
    f0_0 = f0_median(x)
    lifter = int(np.clip(0.85 * SR / f0_0, 30, 140))    # 短于谐波间距
    env_0 = global_envelope(x, lifter=lifter)
    print(f"输入 {os.path.basename(src_path)}：{len(x) / SR:.2f}s，"
          f"F0 中位数 = {f0_0:.1f} Hz（估计值，供对比用）")
    print(f"倒谱 lifter 自适应取 {lifter}（谐波在倒谱上的位置 = "
          f"{SR / f0_0:.0f} 样本，必须比它短）\n")

    norm = os.path.join(outdir, "speech_16k.wav")
    wavfile.write(norm, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))

    print("真实语音上，rubberband 的两种 formant 模式：")
    print(f"  {'变调':<6}{'formant=':<11}{'F0 (Hz)':>9}{'F0 比值':>10}"
          f"{'包络搬移 α':>12}{'α 更接近':>12}")
    for pitch in (1.5, 0.7):
        for formant in ("shifted", "preserved"):
            p = rubberband(norm, os.path.join(outdir, f"pitch{pitch}_{formant}.wav"),
                           pitch, formant)
            y = read(p)
            f0 = f0_median(y)
            alpha = envelope_shift_alpha(env_0, global_envelope(y, lifter=lifter))
            near = "音高比" if abs(alpha - pitch) < abs(alpha - 1.0) else "1（没动）"
            print(f"  {pitch:<6}{formant:<11}{f0:>9.1f}{f0 / f0_0:>10.2f}"
                  f"{alpha:>12.2f}{near:>12}")

    print("\n读法：")
    print("  F0 比值告诉你音高动了多少；α 告诉你共振峰跟着动了多少。")
    print("  formant=shifted 的 α 贴着 F0 比值 → 音色被一起搬走（花栗鼠/怪兽音）。")
    print("  formant=preserved 的 α 被拉向 1，但真实语音上做不到完全不动 ——")
    print("  这正是各家实现（WORLD / STRAIGHT / 神经声码器）拉开差距的地方。")
    print(f"\n输出目录：{outdir}")


if __name__ == "__main__":
    main()
