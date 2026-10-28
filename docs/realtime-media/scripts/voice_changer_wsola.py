#!/usr/bin/env python3
"""变声器原理的最小实现：WSOLA 时间拉伸 + 线性重采样。

    pitch   : 时间拉伸 p 倍 -> 重采样回原长  => 音调 x p，语速不变（正确做法）
    tempo   : 只做时间拉伸 1/speed           => 语速变，音调不变
    resample: 只做重采样                      => 音调语速一起变（错误示范：这是快放/慢放）

用法:
    python3 voice_changer.py in.wav out.wav --mode pitch --semi 6
    python3 voice_changer.py in.wav out.wav --mode tempo --speed 1.25
    python3 voice_changer.py in.wav out.wav --mode resample --speed 1.25
"""
import argparse
import wave

import numpy as np

FRAME = 1024          # 分析帧长（约 21ms @48k / 64ms @16k）
SYN_HOP = 256         # 合成跳距 = 75% overlap
TOL = 300             # WSOLA 相似度搜索半径（样本）


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2, "只支持 16bit PCM"
        n = w.getnframes()
        sr = w.getframerate()
        ch = w.getnchannels()
        data = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float64) / 32768.0
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def write_wav(path, data, sr):
    pcm = np.clip(data, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def wsola(x, stretch):
    """把时长拉伸 stretch 倍，音调不变。"""
    if abs(stretch - 1.0) < 1e-3:
        return x.copy()
    win = np.hanning(FRAME)
    ana_hop = SYN_HOP / stretch
    out_len = int(len(x) * stretch) + FRAME
    out = np.zeros(out_len + FRAME)
    wsum = np.zeros(out_len + FRAME)

    # 上一帧"自然延续"的位置：用来做波形相似度对齐，避免相位跳变
    prev_tail = None
    n_out = int(len(x) * stretch / SYN_HOP)
    for i in range(n_out):
        center = int(i * ana_hop)
        if center + FRAME >= len(x):
            break
        if prev_tail is None:
            pos = center
        else:
            # 在 [center-TOL, center+TOL] 里找与 prev_tail 最相似的起点
            lo = max(0, center - TOL)
            hi = min(len(x) - FRAME, center + TOL)
            if hi <= lo:
                pos = center
            else:
                seg = x[lo:hi + len(prev_tail)]
                corr = np.correlate(seg, prev_tail, mode="valid")
                pos = lo + int(np.argmax(corr))
        frame = x[pos:pos + FRAME] * win
        o = i * SYN_HOP
        out[o:o + FRAME] += frame
        wsum[o:o + FRAME] += win
        prev_tail = x[pos + SYN_HOP:pos + SYN_HOP + (FRAME - SYN_HOP)]

    nz = wsum > 1e-6
    out[nz] /= wsum[nz]
    return out[:out_len]


def resample(x, ratio):
    """线性插值重采样：ratio>1 变短变高。"""
    n = int(len(x) / ratio)
    idx = np.arange(n) * ratio
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    frac = idx - i0
    return x[i0] * (1 - frac) + x[i1] * frac


def pitch_shift(x, semi):
    p = 2.0 ** (semi / 12.0)
    stretched = wsola(x, p)      # 时长 x p，音调不变
    return resample(stretched, p)  # 压回原时长 -> 音调 x p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--mode", default="pitch", choices=["pitch", "tempo", "resample"])
    ap.add_argument("--semi", type=float, default=6.0)
    ap.add_argument("--speed", type=float, default=1.25)
    args = ap.parse_args()

    x, sr = read_wav(args.inp)
    if args.mode == "pitch":
        y = pitch_shift(x, args.semi)
    elif args.mode == "tempo":
        y = wsola(x, 1.0 / args.speed)
    else:
        y = resample(x, args.speed)
    write_wav(args.out, y, sr)
    print(f"{args.mode}: {len(x)/sr:.2f}s -> {len(y)/sr:.2f}s  sr={sr}")


if __name__ == "__main__":
    main()
