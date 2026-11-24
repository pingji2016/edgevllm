"""流式 ASR：滚动窗口 + 增量出字（whisperlive 思路）

对比不同分块大小下的：
  · 每次更新延迟（mean / p95）
  · 端到端完成时间与 RTF
  · 最终文本 vs 离线转写的一致性
"""
import json
import os
import time

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

MD = os.path.expanduser("~/.cache/modelscope/models/Systran--faster-whisper-small/snapshots/master")
WAV = os.path.expanduser("~/apps/CosyVoice/asset/cross_lingual_prompt.wav")
OUT = os.path.expanduser("~/apps/infer-research/stream_asr.json")
WINDOWS = [0.5, 1.0, 2.0]
ROLLING = 6.0  # 滚动缓冲秒数


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

    # 离线基线
    m = WhisperModel(MD, device="cuda", compute_type="float16")
    t0 = time.time()
    segs, _ = m.transcribe(audio, beam_size=5, language="zh")
    offline = "".join(s.text for s in segs).strip()
    t_off = time.time() - t0
    print(f"离线: {t_off:.2f}s (RTF {t_off/dur:.3f}) -> {offline}\n")

    results = []
    for win in WINDOWS:
        lats, texts = [], []
        committed = ""
        n_chunks = int(np.ceil(dur / win))
        t_start = time.time()
        for i in range(1, n_chunks + 1):
            end = min(int(i * win * 16000), len(audio))
            start = max(0, end - int(ROLLING * 16000))
            buf = audio[start:end]
            t0 = time.time()
            segs, _ = m.transcribe(buf, beam_size=1, language="zh",
                                   condition_on_previous_text=False)
            txt = "".join(s.text for s in segs).strip()
            lats.append(time.time() - t0)
            texts.append(txt)
            # 增量合并：找 committed 后缀与 txt 前缀的最长重叠
            best = 0
            for L in range(min(len(committed), len(txt)), 0, -1):
                if committed[-L:] == txt[:L]:
                    best = L
                    break
            if best:
                committed = committed + txt[best:]
            elif txt.startswith(committed):
                committed = txt
            elif not committed.endswith(txt):
                committed = committed + txt
        total = time.time() - t_start
        final = committed
        lats_sorted = sorted(lats)
        p95 = lats_sorted[min(len(lats_sorted) - 1, int(0.95 * len(lats_sorted)))]
        # 与离线文本的字符级一致性（用最长公共子序列近似）
        import difflib
        match = difflib.SequenceMatcher(None, offline, final).ratio()
        results.append(dict(window=win, chunks=n_chunks, mean_lat=round(np.mean(lats), 3),
                            p95_lat=round(p95, 3), max_lat=round(max(lats), 3),
                            total_s=round(total, 2), rtf=round(total / dur, 3),
                            accuracy_vs_offline=round(match, 3), final=final))
        print(f"窗口 {win}s ({n_chunks} 块) | 更新延迟 mean {np.mean(lats)*1000:.0f} ms / "
              f"p95 {p95*1000:.0f} ms / max {max(lats)*1000:.0f} ms | 总计 {total:.2f}s "
              f"(RTF {total/dur:.2f}) | 与离线一致 {match*100:.0f}%")

    print(f"\n最新一次更新文本: {results[-1]['final'][:60]}")
    json.dump(dict(duration=round(dur, 2), offline=offline, offline_s=round(t_off, 2),
                   results=results), open(OUT, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
