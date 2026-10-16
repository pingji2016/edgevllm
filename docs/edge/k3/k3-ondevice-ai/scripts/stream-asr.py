#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""流式 ASR 客户端（/v1/asr/stream WebSocket），跑在 K3 板子上。

用法:
    ./stream-asr.py 音频.wav [语速倍率]     默认 1.0（按真实时间喂帧）；0=不限速

协议（网关源码 domains/asr/stream.py）:
    1) POST /v1/asr/stream/session  -> session_id
    2) ws://.../v1/asr/stream?session_id=..&language=..&sample_rate=..&partial=true
       server -> {"type":"ready"} / partial / sentence_end / final / error
       client -> 二进制 PCM 帧  /  {"type":"end"}

★ 实测结论（报告 §13.2）：这个"流式"**是缓冲式的，不出增量 partial** ——
   37s 音频按真实时间喂完，唯一结果在音频结束 28.6s 后才到，RTF 与整段识别一样（0.76）。
   想要低延迟，只能客户端自己 VAD 切段（见 long-transcribe.py）。
"""
import asyncio, json, os, sys, time, urllib.request, wave

G = os.environ.get("GW", "http://127.0.0.1:18790")
WS = G.replace("http://", "ws://").replace("https://", "wss://")


async def run(path, speed):
    w = wave.open(path)
    fr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    if ch != 1 or sw != 2:
        sys.exit("需要 16k 单声道 16bit wav（用 ffmpeg -ar 16000 -ac 1 转）")

    req = urllib.request.Request(
        G + "/v1/asr/stream/session",
        data=json.dumps({"language": "zh", "sample_rate": fr}).encode(),
        headers={"Content-Type": "application/json"})
    sid = json.load(urllib.request.urlopen(req))["session_id"]
    print("会话 %s | 音频 %.1fs @%dHz" % (sid, n / fr, fr))

    url = "%s/v1/asr/stream?session_id=%s&language=zh&sample_rate=%d&partial=true" % (WS, sid, fr)
    state = {"t0": None, "final": None, "first": None, "npart": 0}

    async with asyncio.timeout(180):
        async with __import__("websockets").connect(url, ping_interval=None, max_size=None) as ws:
            async def reader():
                async for m in ws:
                    d = json.loads(m)
                    ty = d.get("type")
                    if ty == "ready":
                        print("  [ready] 已就绪")
                        continue
                    t = time.time() - state["t0"] if state["t0"] else 0.0
                    if ty == "partial":
                        state["npart"] += 1
                        if state["first"] is None:
                            state["first"] = t
                        print("  [%5.2fs] partial  %s" % (t, d.get("text", "")))
                    elif ty == "sentence_end":
                        print("  [%5.2fs] ★句子    %s" % (t, d.get("text", "")))
                    elif ty == "final":
                        state["final"] = t
                        print("  [%5.2fs] final    %s" % (t, d.get("text", "")))
                        print("           duration_ms=%s rtf=%s" % (d.get("duration_ms"), d.get("rtf")))
                        return
                    elif ty == "error":
                        print("  [err]", d); return

            rt = asyncio.create_task(reader())
            await asyncio.sleep(0.2)                       # 等 ready
            state["t0"] = time.time()

            CH = int(fr * 0.1)                             # 100ms 一帧
            i = 0
            while i < n:
                data = w.readframes(CH)
                if not data:
                    break
                await ws.send(data)
                i += CH
                if speed > 0:
                    await asyncio.sleep(0.1 / speed)
            await ws.send(json.dumps({"type": "end"}))
            await asyncio.wait_for(rt, timeout=60)

    dur = n / fr
    fin = state["final"] or 0.0
    print("\n结果: 音频 %.1fs | 首个 partial %.2fs | final 相对音频 %.2fs | partial 次数 %d"
          % (dur, state["first"] or -1, fin, state["npart"]))
    if fin:
        print("      => 音频结束后又等了 %.2fs 才出结果（说明是缓冲式，非增量）" % (fin - dur))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("用法: stream-asr.py 音频.wav [语速倍率]")
    asyncio.run(run(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1.0))
