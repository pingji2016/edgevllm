"""ASR 模型尺寸扫描：tiny / base / small / medium（同一段音频）"""
import difflib
import json
import os
import time

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

WAV = os.path.expanduser("~/apps/CosyVoice/asset/cross_lingual_prompt.wav")
ROOT = "~/.cache/modelscope/models/Systran--faster-whisper-{}/snapshots/master"
SIZES = ("tiny", "base", "small", "medium")
OUT = os.path.expanduser("~/apps/infer-research/asr_sizes.json")


def load16(path):
    d, s = sf.read(path, dtype="float32")
    if d.ndim > 1:
        d = d.mean(axis=1)
    if s != 16000:
        idx = np.linspace(0, len(d) - 1, int(len(d) * 16000 / s))
        d = np.interp(idx, np.arange(len(d)), d).astype("float32")
    return d


def main():
    audio = load16(WAV)
    dur = len(audio) / 16000
    print(f"音频 {dur:.2f}s\n")
    rows = []
    ref_text = None
    for size in SIZES:
        path = os.path.expanduser(ROOT.format(size))
        if not os.path.isdir(path):
            print(f"  {size}: 模型缺失，跳过")
            continue
        t0 = time.time()
        m = WhisperModel(path, device="cuda", compute_type="float16")
        load = time.time() - t0
        m.transcribe(audio)  # 预热
        t0 = time.time()
        segs, info = m.transcribe(audio, beam_size=5, language="zh")
        text = "".join(s.text for s in segs).strip()
        dt = time.time() - t0
        if ref_text is None:
            ref_text = text
        sim = difflib.SequenceMatcher(None, ref_text, text).ratio()
        vram = 0
        try:
            vram = torch.cuda.max_memory_allocated() / 2**20
        except Exception:
            pass
        rows.append(dict(size=size, load_s=round(load, 1), seconds=round(dt, 3),
                         rtf=round(dt / dur, 4), x_realtime=round(dur / dt, 1),
                         similarity=round(sim, 3), text=text[:120]))
        print(f"  {size:6s} 加载 {load:4.1f}s | 转写 {dt:5.2f}s | RTF {dt/dur:.3f} "
              f"({dur/dt:5.1f}×) | 与 tiny 一致 {sim*100:3.0f}%")
        print(f"         {text[:70]}")
        del m
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    json.dump(rows, open(OUT, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import torch  # noqa
    main()
