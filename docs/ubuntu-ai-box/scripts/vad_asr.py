"""VAD + ASR 串联：固定 2s 窗口 vs 按语音段切分，对比准确率与耗时"""
import difflib
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel

WAV = os.path.expanduser("~/apps/CosyVoice/asset/cross_lingual_prompt.wav")
MD = os.path.expanduser("~/.cache/modelscope/models/Systran--faster-whisper-small/snapshots/master")
OUT = os.path.expanduser("~/apps/infer-research/vad_asr.json")


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

    m = WhisperModel(MD, device="cuda", compute_type="float16")

    # 参考：离线整段
    t0 = time.time()
    segs, _ = m.transcribe(audio, beam_size=5, language="zh")
    ref = "".join(s.text for s in segs).strip()
    ref_t = time.time() - t0
    print(f"参考（离线整段）{ref_t:.2f}s：{ref}\n")

    res = {"duration": round(dur, 2), "reference": ref, "reference_s": round(ref_t, 2)}

    # A. 固定 2s 窗口
    def fixed(win):
        out, n = [], 0
        t0 = time.time()
        step = int(win * 16000)
        for i in range(0, len(audio), step):
            chunk = audio[i:i + step]
            if len(chunk) < 1600:  # 太短跳过
                continue
            s, _ = m.transcribe(chunk, beam_size=1, language="zh")
            out.append("".join(x.text for x in s).strip())
            n += 1
        dt = time.time() - t0
        text = "".join(out)
        return text, dt, n

    for win in (2.0,):
        text, dt, n = fixed(win)
        sim = difflib.SequenceMatcher(None, ref, text).ratio()
        print(f"A 固定 {win}s 窗口（{n} 次调用）：{dt:.2f}s | 相似度 {sim*100:.0f}%")
        print(f"   文本: {text[:70]}")
        res["fixed"] = dict(window=win, calls=n, seconds=round(dt, 2),
                            similarity=round(sim, 3), text=text[:200])
    print()

    # B. VAD 分段
    from silero_vad import get_speech_timestamps, load_silero_vad
    vad = load_silero_vad()
    t0 = time.time()
    stamps = get_speech_timestamps(torch.from_numpy(audio), vad, sampling_rate=16000,
                                   return_seconds=True, threshold=0.4)
    vad_t = time.time() - t0
    out = []
    t0 = time.time()
    for s in stamps:
        a, b = int(s["start"] * 16000), int(s["end"] * 16000)
        seg, _ = m.transcribe(audio[a:b], beam_size=1, language="zh")
        out.append("".join(x.text for x in seg).strip())
    asr_t = time.time() - t0
    text = "".join(out)
    sim = difflib.SequenceMatcher(None, ref, text).ratio()
    print(f"B VAD 分段（{len(stamps)} 段，VAD {vad_t*1000:.0f}ms + ASR {asr_t:.2f}s = {vad_t+asr_t:.2f}s）："
          f"相似度 {sim*100:.0f}%")
    print(f"   文本: {text[:70]}")
    res["vad"] = dict(segments=len(stamps), vad_ms=round(vad_t * 1000, 1),
                      asr_s=round(asr_t, 2), total_s=round(vad_t + asr_t, 2),
                      similarity=round(sim, 3), text=text[:200],
                      stamps=[(round(s["start"], 2), round(s["end"], 2)) for s in stamps])

    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
