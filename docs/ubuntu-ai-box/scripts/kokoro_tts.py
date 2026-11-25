"""Kokoro-82M 中文 TTS：生成 + ASR 验收 + RTF"""
import glob
import os
import time

import numpy as np

OUT = os.path.expanduser("~/apps/infer-research/kokoro")
TEXT = "端侧推理部署说明：解码速度五十五点三，每秒钟六十二个 token。"


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(os.path.expanduser("~/models/.kokoro_done")):
        from modelscope import snapshot_download
        p = snapshot_download("hexgrad/Kokoro-82M-v1.1-zh")
        open(os.path.expanduser("~/models/.kokoro_done"), "w").write(p)
    root = open(os.path.expanduser("~/models/.kokoro_done")).read().strip()
    print("模型:", root)
    voices = sorted(glob.glob(os.path.join(root, "voices", "*.pt")))
    print(f"音色数 {len(voices)}:", [os.path.basename(v) for v in voices[:8]])

    from kokoro import KPipeline

    t0 = time.time()
    pipe = KPipeline(lang_code="z", repo_id="hexgrad/Kokoro-82M-v1.1-zh",
                     model=os.path.join(root, "kokoro-v1.1-zh.pth"))
    print(f"管线加载 {time.time()-t0:.1f}s")

    import torch
    zf = [v for v in voices if os.path.basename(v).startswith("zf_")] or voices
    voice = torch.load(zf[0], map_location="cpu")  # 直接传音色张量，避免路径解析
    print("  使用音色:", os.path.basename(zf[0]))
    t0 = time.time()
    chunks, sr = [], 24000
    for gs, ps, audio in pipe(TEXT, voice=voice, speed=1.0):
        if audio is not None:
            chunks.append(audio.numpy() if hasattr(audio, "numpy") else audio)
    dt = time.time() - t0
    wav = np.concatenate(chunks)
    dur = len(wav) / sr
    p = os.path.join(OUT, "zh.wav")
    import soundfile as sf
    sf.write(p, wav, sr)
    print(f"生成 {dur:.2f}s 音频 | 耗时 {dt:.2f}s | RTF {dt/dur:.3f}（{dur/dt:.1f}× 实时）")
    print(f"产物: {p}")

    if os.environ.get("SKIP_ASR") == "1":
        print("（跳过 ASR）")
        return

    # ASR 验收
    from faster_whisper import WhisperModel
    d = wav.astype("float32")
    if sr != 16000:
        idx = np.linspace(0, len(d) - 1, int(len(d) * 16000 / sr))
        d = np.interp(idx, np.arange(len(d)), d).astype("float32")
    m = WhisperModel(os.path.expanduser(
        "~/.cache/modelscope/models/Systran--faster-whisper-small/snapshots/master"),
        device="cuda", compute_type="float16")
    segs, info = m.transcribe(d, beam_size=5, language="zh")
    heard = "".join(s.text for s in segs).strip()
    import difflib
    print(f"\n目标: {TEXT}")
    print(f"听到: {heard}")
    print(f"字符相似度: {difflib.SequenceMatcher(None, TEXT, heard).ratio()*100:.1f}%")


if __name__ == "__main__":
    main()
