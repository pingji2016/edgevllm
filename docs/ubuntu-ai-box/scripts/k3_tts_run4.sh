#!/bin/bash
set -u
cd ~/kokoro || exit 1
python3 - <<'PY'
import json, sys, time, types, wave
import numpy as np
import torch

for name in ("misaki", "misaki.en", "misaki.espeak", "misaki.zh"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["misaki"].en = sys.modules["misaki.en"]
sys.modules["misaki"].espeak = sys.modules["misaki.espeak"]

ps = open("/home/pingji/kokoro/phonemes.txt", encoding="utf-8").read()
ref_s = torch.load("/home/pingji/kokoro/voice.pt", map_location="cpu")
cfg = json.load(open("/home/pingji/kokoro/config.json"))

from kokoro.model import KModel
t0 = time.time()
km = KModel(config=cfg, model="/home/pingji/kokoro/model.pth").eval()
n = sum(p.numel() for p in km.parameters())
print(f"  加载 {time.time()-t0:.1f}s | 参数 {n/1e6:.1f} M")

def gen():
    try:
        return km(ps, ref_s, 1.0, return_output=True)
    except TypeError:
        return km(ps, ref_s, 1.0)

with torch.inference_mode():
    gen()  # 预热
    t0 = time.time()
    out = gen()
    dt = time.time() - t0

a = out.audio if hasattr(out, "audio") else out
a = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
a = np.asarray(a).squeeze()
dur = len(a) / 24000
print(f"  K3: {dur:.2f}s 音频 / {dt*1000:.0f} ms | RTF {dt/dur:.3f}（{dur/dt:.2f}× 实时）")

pcm = (np.clip(a, -1, 1) * 32767).astype(np.int16)
w = wave.open("/home/pingji/kokoro/k3_out.wav", "wb")
w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
w.writeframes(pcm.tobytes()); w.close()
print("  产物: /home/pingji/kokoro/k3_out.wav")
json.dump(dict(audio_s=round(dur, 2), ms=round(dt*1000, 1), rtf=round(dt/dur, 3),
               x_realtime=round(dur/dt, 2), params_m=round(n/1e6, 1), kokoro="0.7.16"),
          open("/home/pingji/kokoro/result.json", "w"), indent=2)
PY
