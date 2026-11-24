"""ASR 回听 CosyVoice 输出，量化"内容对不对" """
import difflib
import sys

import numpy as np
import soundfile as sf

sys.path.append("/home/pingji/apps/CosyVoice")
from faster_whisper import WhisperModel

MD = "/home/pingji/.cache/modelscope/models/Systran--faster-whisper-small/snapshots/master"
PAIRS = [("你好，今天天气不错。", "/tmp/cv2_diag/fixed.wav")]


def load(path, sr=16000):
    d, s = sf.read(path, dtype="float32")
    if d.ndim > 1:
        d = d.mean(axis=1)
    if s != sr:
        idx = np.linspace(0, len(d) - 1, int(len(d) * sr / s))
        d = np.interp(idx, np.arange(len(d)), d).astype("float32")
    return d


m = WhisperModel(MD, device="cpu", compute_type="int8")
for want, path in PAIRS:
    d = load(path)
    segs, info = m.transcribe(d, beam_size=5)
    heard = "".join(s.text for s in segs).strip()
    print(f"音频 {len(d)/16000:.2f}s")
    print(f"目标: {want}")
    print(f"听到: {heard}")
    print(f"字符相似度: {difflib.SequenceMatcher(None, want, heard).ratio()*100:.0f}%")
