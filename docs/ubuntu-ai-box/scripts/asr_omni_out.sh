#!/bin/bash
set -u
O=~/tools/omni/output/round_000/tts_wav
echo "=== 拼接 12 段输出语音 ==="
~/apps/bench/yolo-venv/bin/python - <<'PY'
import glob, os
import numpy as np
import soundfile as sf

files = sorted(glob.glob(os.path.expanduser("~/tools/omni/output/round_000/tts_wav/wav_*.wav")),
               key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit())))
print(f"  文件数 {len(files)}")
chunks, sr = [], None
for f in files:
    d, s = sf.read(f, dtype="float32")
    if d.ndim > 1:
        d = d.mean(1)
    chunks.append(d); sr = s
if chunks:
    out = np.concatenate(chunks)
    print(f"  采样率 {sr} | 总时长 {len(out)/sr:.2f}s")
    sf.write("/tmp/omni_tts_out.wav", out, sr)
PY

echo
echo "=== ASR 回听（GPU 需 cu12 库）==="
LIBS=$(ls -d ~/apps/minimind/.venv/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')
LD_LIBRARY_PATH=$LIBS ~/apps/bench/yolo-venv/bin/python - <<'PY'
import os
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

d, sr = sf.read("/tmp/omni_tts_out.wav", dtype="float32")
if d.ndim > 1:
    d = d.mean(1)
if sr != 16000:
    idx = np.linspace(0, len(d) - 1, int(len(d) * 16000 / sr))
    d = np.interp(idx, np.arange(len(d)), d).astype("float32")
m = WhisperModel(os.path.expanduser(
    "~/.cache/modelscope/models/Systran--faster-whisper-small/snapshots/master"),
    device="cuda", compute_type="float16")
segs, info = m.transcribe(d, beam_size=5, language="zh")
print("  ASR:", "".join(s.text for s in segs).strip())
PY
