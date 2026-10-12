#!/usr/bin/env python3
"""
实时变声的两个工程问题，用可复现的数字量出来：

  A. 分块拼接的边界爆音
     逐块独立处理再硬拼回去 → 每块边界都有跳变（"咔"）。
     量法：块边界处的样本跳变 |x[i]-x[i-1]| 与块内部跳变的中位数对比。
     解法：块间重叠 + 余弦交叉淡入（SOLA 那套的最小版本），再量一次。

  B. 块长是延迟的大头
     延迟 = 块长 + 算法前视（这里用 STFT 窗长）；块越小延迟越低，
     但每块都有固定开销，CPU 占用反而上去。
     量法：不同块长下的"端到端延迟"与"实时率 RTF = 计算耗时 / 音频时长"。

处理器用了一个真实的相位声码器变调（不是空转），所以开销、边缘效应、相位不连续
都是真的 —— 边界那声"咔"的根因就是每块各算各的相位。

关于上下文：相位声码器会吃掉首尾约一个窗长的样本，所以每块要多喂 N_FFT 的上下文，
推理完再把上下文切掉 —— 这就是 RVC/VCClient 里 extra_time 那类参数在做的事。
偏移量用脉冲标定，不靠猜。

用法：python realtime_chunking.py <输入.wav> <输出目录>
"""
import os
import sys
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

N_FFT = 1024
HOP = N_FFT // 4
SR = 16000


# ---------- 相位声码器变调（时长不变）----------

def phase_vocoder_stretch(x, factor):
    """时间拉伸 factor 倍（factor>1 变长），音高不变。"""
    win = np.hanning(N_FFT)
    Hs = int(round(HOP * factor))
    n_frames = 1 + (len(x) - N_FFT) // HOP
    out = np.zeros(n_frames * Hs + N_FFT)
    wsum = np.zeros_like(out)
    prev_phase = np.zeros(N_FFT // 2 + 1)
    acc = np.zeros(N_FFT // 2 + 1)
    k = np.arange(N_FFT // 2 + 1)
    for i in range(n_frames):
        seg = x[i * HOP:i * HOP + N_FFT] * win
        spec = np.fft.rfft(seg)
        mag, phase = np.abs(spec), np.angle(spec)
        dphi = phase - prev_phase - 2 * np.pi * k * HOP / N_FFT
        dphi = np.mod(dphi + np.pi, 2 * np.pi) - np.pi        # 解卷绕
        acc += dphi + 2 * np.pi * k * HOP / N_FFT
        prev_phase = phase
        out[i * Hs:i * Hs + N_FFT] += np.fft.irfft(mag * np.exp(1j * acc), n=N_FFT) * win
        wsum[i * Hs:i * Hs + N_FFT] += win ** 2
    n = n_frames * Hs
    return out[:n] / (wsum[:n] + 1e-8)


def pitch_shift(x, ratio):
    """变调 ratio 倍、时长不变：先拉伸 ratio，再按 ratio 倍速重采样。"""
    return resample_poly(phase_vocoder_stretch(x, ratio),
                         int(round(100 * ratio)), 100)


def calibrate_offset(ratio):
    """脉冲标定：相位声码器输出相对输入的时移（样本）。"""
    n = 8 * N_FFT
    imp = np.zeros(n)
    imp[n // 2] = 1.0
    return int(np.argmax(np.abs(pitch_shift(imp, ratio)))) - n // 2


def chunk_pitch_shift(x, chunk_len, ratio, offset, context=N_FFT):
    """逐块变调：每块两侧各带 context 上下文，处理后切回本块对应的 chunk_len 个样本。"""
    pad = np.pad(x, (context, context + chunk_len))
    out = np.zeros(len(x))
    for i in range(0, len(x), chunk_len):
        seg = pad[i:i + chunk_len + 2 * context]
        y = pitch_shift(seg, ratio)
        start = context + offset
        piece = y[start:start + chunk_len]
        if len(piece) < chunk_len:                 # 末尾补齐（真实实现这里是补零/淡出）
            piece = np.pad(piece, (0, chunk_len - len(piece)))
        out[i:i + chunk_len] = piece[:chunk_len]
    return out


def chunk_pitch_shift_crossfade(x, chunk_len, ratio, offset, context=N_FFT, fade_ms=10):
    """同上，但块之间做余弦交叉淡入（SOLA 的最小版本）。"""
    fade = int(SR * fade_ms / 1000)
    step = chunk_len - fade
    pad = np.pad(x, (context, context + chunk_len + fade))
    out = np.zeros(len(x) + 2 * chunk_len)
    wsum = np.zeros_like(out)
    w = np.ones(chunk_len)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
    w[:fade], w[-fade:] = ramp, ramp[::-1]
    for i in range(0, len(x), step):
        seg = pad[i:i + chunk_len + 2 * context]
        y = pitch_shift(seg, ratio)
        piece = y[context + offset: context + offset + chunk_len]
        if len(piece) < chunk_len:
            piece = np.pad(piece, (0, chunk_len - len(piece)))
        out[i:i + chunk_len] += piece[:chunk_len] * w
        wsum[i:i + chunk_len] += w
    return (out[:len(x)] / np.maximum(wsum[:len(x)], 1e-6))


# ---------- 指标 ----------

def boundary_jump_ratio(sig, stride):
    """块边界处的样本跳变 / 块内部跳变中位数。"""
    d = np.abs(np.diff(sig))
    interior = np.median(d)
    idx = [i - 1 for i in range(stride, len(sig) - 1, stride) if i - 1 < len(d)]
    return float(np.max(d[idx]) / (interior + 1e-12))


def main():
    src, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    sr, x = wavfile.read(src)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64) / 32768.0
    if sr != SR:
        x = resample_poly(x, SR, sr)
        sr = SR
    print(f"输入 {len(x) / sr:.2f}s @ {sr}Hz     变调 1.3×，窗长 {N_FFT}，块长 100ms\n")

    ratio = 1.3
    offset = calibrate_offset(ratio)
    print(f"脉冲标定：相位声码器的时移 = {offset} 样本（{offset / sr * 1000:.1f} ms）\n")

    chunk_len = int(sr * 0.1)
    hard = chunk_pitch_shift(x, chunk_len, ratio, offset)
    soft = chunk_pitch_shift_crossfade(x, chunk_len, ratio, offset)

    print("A. 分块拼接的边界跳变（越接近 1 越好）")
    print(f"   硬拼回去       边界跳变 = 内部中位数的 {boundary_jump_ratio(hard, chunk_len):>6.1f} 倍")
    print(f"   交叉淡入       边界跳变 = 内部中位数的 {boundary_jump_ratio(soft, chunk_len):>6.1f} 倍")
    print("   → 硬拼的边界跳变远大于音频内部，听感就是每 100ms 一声『咔』；")
    print("     交叉淡入把它压回正常范围。这就是 SOLA/交叉淡入在实时变声里必做的原因。\n")
    wavfile.write(os.path.join(outdir, "chunk_hard_concat.wav"), sr,
                  (np.clip(hard, -1, 1) * 32767).astype(np.int16))
    wavfile.write(os.path.join(outdir, "chunk_crossfade.wav"), sr,
                  (np.clip(soft, -1, 1) * 32767).astype(np.int16))

    print("B. 块长 ↔ 延迟 / CPU 开销（变调 1.3×，纯 CPU）")
    print(f"   {'块长(ms)':>9} {'端到端延迟(ms)':>15} {'RTF(计算/音频)':>16}")
    dur = len(x) / sr
    for ms in (20, 50, 100, 250, 500):
        cl = int(sr * ms / 1000)
        t = time.perf_counter()
        chunk_pitch_shift(x, cl, ratio, offset)
        rtf = (time.perf_counter() - t) / dur
        latency = ms + N_FFT / sr * 1000          # 块长 + 窗长前视
        print(f"   {ms:>9} {latency:>15.1f} {rtf:>16.3f}")
    print("\n   RTF<1 表示比实时快。块越小延迟越低，但 RTF 越高（每块的固定开销摊不薄）。")
    print("   神经变声器同理，只是它的『每块开销』是模型推理，比这里重一到两个数量级。")


if __name__ == "__main__":
    main()
