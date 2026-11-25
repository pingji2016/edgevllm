"""第二梯队 D：视频理解（多帧送 Qwen3-VL）+ E：VAD 端点检测"""
import base64
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch

WAV = os.path.expanduser("~/apps/CosyVoice/asset/cross_lingual_prompt.wav")
PORT = 8092
OUT = os.path.expanduser("~/apps/infer-research/tier2_de.json")


# ---------- D. 视频理解 ----------
def video_understanding():
    import urllib.request
    frames = sorted(glob.glob("/tmp/vlframes/*.jpg"))[:4]
    if not frames:
        print("  没有帧，跳过")
        return None
    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    content = [{"type": "text", "text": "这是一段视频的连续几帧，请描述画面内容的变化。"}]
    for f in frames:
        b64 = base64.b64encode(open(f, "rb").read()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    body = {"model": "qwen3vl", "max_tokens": 64, "temperature": 0,
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    # 预热
    urllib.request.urlopen(req, timeout=300).read()
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.load(r)
    dt = time.time() - t0
    txt = resp["choices"][0]["message"]["content"].strip().replace("\n", " ")
    print(f"  多帧({len(frames)} 帧) 用时 {dt:.2f}s")
    print(f"  回答: {txt[:100]}")
    return dict(frames=len(frames), seconds=round(dt, 2), answer=txt[:200])


# ---------- E. VAD ----------
def load16(path):
    d, s = sf.read(path, dtype="float32")
    if d.ndim > 1:
        d = d.mean(axis=1)
    if s != 16000:
        idx = np.linspace(0, len(d) - 1, int(len(d) * 16000 / s))
        d = np.interp(idx, np.arange(len(d)), d).astype("float32")
    return d


def vad_test():
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
    except Exception as e:
        print(f"  silero-vad 不可用: {type(e).__name__}")
        return None
    model = load_silero_vad()
    audio = torch.from_numpy(load16(WAV))
    dur = audio.numel() / 16000
    # 预热
    get_speech_timestamps(audio, model, sampling_rate=16000)
    t0 = time.time()
    stamps = get_speech_timestamps(audio, model, sampling_rate=16000,
                                   return_seconds=True, threshold=0.4)
    dt = time.time() - t0
    speech = sum(s["end"] - s["start"] for s in stamps)
    print(f"  VAD: 音频 {dur:.2f}s | 检出 {len(stamps)} 段语音 | 语音占比 {speech/dur*100:.0f}%")
    print(f"       耗时 {dt*1000:.0f} ms | RTF {dt/dur:.4f}（{dur/max(dt,1e-9):.0f}× 实时）")
    print(f"       分段: {[(round(s['start'],2), round(s['end'],2)) for s in stamps][:5]}")
    return dict(duration=round(dur, 2), segments=len(stamps), speech_ratio=round(speech / dur, 3),
                ms=round(dt * 1000, 1), rtf=round(dt / dur, 4),
                stamps=[(round(s["start"], 2), round(s["end"], 2)) for s in stamps])


if __name__ == "__main__":
    print("=== D. 视频理解 ===")
    d = video_understanding()
    print("\n=== E. VAD 端点检测 ===")
    e = vad_test()
    json.dump(dict(video=d, vad=e), open(OUT, "w"), ensure_ascii=False, indent=2)
