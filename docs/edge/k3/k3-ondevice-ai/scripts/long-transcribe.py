#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""长音频转写：VAD 切段 -> (合并相邻段) -> 逐块 ASR -> SRT 字幕（跑在 K3 板子上）

用法:
    ./long-transcribe.py 音频.wav [输出前缀] [--compare]
        --compare   额外跑「逐段」与「整段」两种模式做对照
环境变量:
    MAX_CHUNK_MS (默认 12000)  合并后的单块上限
    GAP_MS       (默认 600)    相邻段间隔小于它就合并
    MIN_SEG_MS   (默认 250)    过滤掉比它还短的杂音段
输出: /tmp/k3tr/<前缀>.srt 与 <前缀>.txt

为什么这样设计（报告 §12.6）：
    SenseVoice 处理时间随音频长度近似二次增长 —— 整段 37s 要 28s（1.3× 实时），
    而切段后只要 3s（12× 实时）。但切太碎会丢上下文，所以合并到 ~12s 一块是平衡点。
"""
import json, os, subprocess, sys, time, wave

G = os.environ.get("GW", "http://127.0.0.1:18790")
MAX_CHUNK_MS = int(os.environ.get("MAX_CHUNK_MS", "12000"))
GAP_MS = int(os.environ.get("GAP_MS", "600"))
MIN_SEG_MS = int(os.environ.get("MIN_SEG_MS", "250"))


def curl_json(path, f=None, timeout=600):
    cmd = ["curl", "-s", "--max-time", str(timeout), "-X", "POST", G + path]
    if f:
        cmd += ["-F", "file=@%s" % f]
    out = subprocess.run(cmd, capture_output=True).stdout
    try:
        return json.loads(out)
    except Exception:
        return {"_raw": out.decode("utf-8", "replace")[:200]}


def slice_wav(src, out, a_ms, b_ms):
    w = wave.open(src); fr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
    w.setpos(int(a_ms * fr / 1000))
    data = w.readframes(int((b_ms - a_ms) * fr / 1000)); w.close()
    o = wave.open(out, "wb"); o.setnchannels(ch); o.setsampwidth(sw); o.setframerate(fr)
    o.writeframes(data); o.close()


def ts(ms):
    ms = int(ms); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


def merge(spans):
    out = []
    for s in spans:
        if out and (s["start_ms"] - out[-1]["end_ms"]) < GAP_MS \
                and (s["end_ms"] - out[-1]["start_ms"]) <= MAX_CHUNK_MS:
            out[-1]["end_ms"] = s["end_ms"]
        else:
            out.append({"start_ms": s["start_ms"], "end_ms": s["end_ms"]})
    return out


def run_spans(src, spans, tag, work):
    blocks, texts, t_sum = [], [], 0.0
    for i, s in enumerate(spans, 1):
        f = os.path.join(work, "%s%03d.wav" % (tag, i))
        slice_wav(src, f, s["start_ms"], s["end_ms"])
        a = time.time()
        txt = (curl_json("/v1/asr/recognize", f).get("text") or "").strip()
        t_sum += time.time() - a
        texts.append(txt)
        blocks.append("%d\n%s --> %s\n%s\n" % (i, ts(s["start_ms"]), ts(s["end_ms"]), txt))
    return blocks, texts, t_sum


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    compare = "--compare" in sys.argv
    if not args:
        sys.exit("用法: long-transcribe.py 音频.wav [输出前缀] [--compare]")
    src = args[0]
    pre = args[1] if len(args) > 1 else os.path.splitext(os.path.basename(src))[0]
    work = "/tmp/k3tr"; os.makedirs(work, exist_ok=True)
    w = wave.open(src); dur = w.getnframes() / float(w.getframerate()); w.close()

    t0 = time.time()
    raw = curl_json("/v1/vad/segments", src).get("segments", [])
    vad = time.time() - t0
    spans = [{"start_ms": s["start_ms"], "end_ms": s["end_ms"]}
             for s in raw if (s["end_ms"] - s["start_ms"]) >= MIN_SEG_MS]
    if not spans:
        sys.exit("VAD 没切出有效语音段")
    chunks = merge(spans)

    print("音频 %.1fs | VAD: %d 段(过滤后) %.2fs(%.0fx 实时)" % (dur, len(spans), vad, dur / vad))
    print("合并参数: gap<%dms 且 单块<=%ds  ->  %d 块\n" % (GAP_MS, MAX_CHUNK_MS // 1000, len(chunks)))
    print("%-14s %6s %8s %10s  %s" % ("模式", "块数", "ASR耗时", "相对实时", "样例文本"))
    print("-" * 78)

    blocks, texts, tm = run_spans(src, chunks, "mg", work)
    print("%-14s %6d %7.2fs %9.2fx  %s" % ("合并(推荐)", len(chunks), tm, dur / tm, texts[0][:28]))

    if compare:
        _, tseg, ts_ = run_spans(src, spans, "sg", work)
        print("%-14s %6d %7.2fs %9.2fx  %s" % ("逐段", len(spans), ts_, dur / ts_, tseg[0][:28]))
        a = time.time(); d = curl_json("/v1/asr/recognize", src); tw = time.time() - a
        print("%-14s %6d %7.2fs %9.2fx  %s" % ("整段", 1, tw, dur / tw, (d.get("text") or "")[:28]))

    srt = os.path.join(work, pre + ".srt")
    open(srt, "w").write("\n".join(blocks))
    open(os.path.join(work, pre + ".txt"), "w").write("\n".join(texts))
    print("\nSRT ->", srt, "| 纯文本 ->", os.path.join(work, pre + ".txt"))


if __name__ == "__main__":
    main()
