"""ASR 闭环：faster-whisper-small 转写 TTS 音频，测 RTF"""
import json
import os
import time

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


def load_audio(path, sr=16000):
    """绕开 PyAV：soundfile 解码 + 线性重采样"""
    d, s = sf.read(path, dtype="float32")
    if d.ndim > 1:
        d = d.mean(axis=1)
    if s != sr:
        idx = np.linspace(0, len(d) - 1, int(len(d) * sr / s))
        d = np.interp(idx, np.arange(len(d)), d).astype("float32")
    return d

WAV = os.path.expanduser(os.environ.get("WAV", "~/apps/CosyVoice/asset/zero_shot_prompt.wav"))
MD = os.path.expanduser("~/.cache/modelscope/models/Systran--faster-whisper-small/snapshots/master")


def main():
    for dev, ct in (("cuda", "float16"), ("cpu", "int8")):
        try:
            t0 = time.time()
            m = WhisperModel(MD, device=dev, compute_type=ct)
            audio = load_audio(WAV)
            m.transcribe(audio)  # 预热（GPU 初始化错误会在这里暴露）
            t0 = time.time()
            segs, info = m.transcribe(audio, beam_size=5, language=None)
            text = "".join(s.text for s in segs)
            dt = time.time() - t0
            dur = info.duration
            print(f"  [{dev}/{ct}] 音频 {dur:.2f}s | 转写 {dt:.2f}s | "
                  f"RTF {dt/dur:.3f}（{dur/dt:.1f}× 实时）| 语种 {info.language}({info.language_probability:.2f})")
            print(f"    文本: {text.strip()}")
            json.dump(dict(device=dev, compute=ct, dur=round(dur, 2), sec=round(dt, 3),
                           rtf=round(dt / dur, 3), lang=info.language, text=text.strip()),
                      open(os.path.expanduser(f"~/apps/infer-research/asr_{dev}.json"), "w"),
                      ensure_ascii=False, indent=2)
            del m
        except Exception as e:
            print(f"  [{dev}/{ct}] 不可用: {type(e).__name__}: {str(e)[:100]}")


if __name__ == "__main__":
    main()
