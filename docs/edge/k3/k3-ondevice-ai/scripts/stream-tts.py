#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""流式 TTS 客户端（/v1/tts/stream WebSocket），跑在 K3 板子上。

用法:
    ./stream-tts.py "整段文本" [out.wav]      # 一次性 start：会一次性出整块
    ./stream-tts.py --append [out.wav]        # 按句 append：真流式，逐句出声

协议（网关源码 domains/tts/stream.py）:
    1) POST /v1/tts/stream/session -> session_id
    2) ws://.../v1/tts/stream?session_id=..&response_format=pcm
       client -> {"type":"start","text":..} / {"type":"append","text":..} / {"type":"end"}
       server -> {"type":"ready"} / <binary PCM> / {"type":"metadata",..} / {"type":"done",..}

★ 实测结论（报告 §13.3）：
    - 一次性 start 整段文本 → **只回一个整块**（等于批量）。
    - 按句 append → **逐句出音频**，每句约 0.5s 就出声 → 这才是"边说边播"。
"""
import asyncio, json, os, sys, time, urllib.request, wave

G = os.environ.get("GW", "http://127.0.0.1:18790")
WS = G.replace("http://", "ws://")
SR = int(os.environ.get("SR", "16000"))     # matcha_zh_en = 16000


async def _session():
    req = urllib.request.Request(G + "/v1/tts/stream/session",
                                 data=json.dumps({"response_format": "pcm"}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))["session_id"]


async def run_once(text, out):
    sid = await _session()
    print("会话 %s | 文本 %d 字（一次性 start）" % (sid, len(text)))
    url = "%s/v1/tts/stream?session_id=%s&response_format=pcm" % (WS, sid)
    pcm, st = bytearray(), {"t0": None, "first": None, "n": 0}

    async with asyncio.timeout(180):
        async with __import__("websockets").connect(url, ping_interval=None, max_size=None) as ws:
            async def reader():
                while True:
                    m = await ws.recv()
                    if isinstance(m, bytes):
                        st["n"] += 1
                        if st["first"] is None:
                            st["first"] = time.time() - st["t0"]
                            print("  [%5.2fs] 首个 PCM 块 (%d B)" % (st["first"], len(m)))
                        pcm.extend(m)
                    else:
                        d = json.loads(m)
                        if d.get("type") == "done":
                            print("  [%5.2fs] done duration_ms=%s rtf=%s"
                                  % (time.time() - st["t0"], d.get("duration_ms"), d.get("rtf")))
                            return

            rt = asyncio.create_task(reader())
            await asyncio.sleep(0.2); st["t0"] = time.time()
            await ws.send(json.dumps({"type": "start", "text": text}))
            await ws.send(json.dumps({"type": "end"}))
            await asyncio.wait_for(rt, timeout=120)

    _save(pcm, out)
    total = time.time() - st["t0"]; audio = len(pcm) / 2.0 / SR
    print("结果: 块数 %d | 音频 %.2fs | 首块 %.2fs | RTF %.3f" % (st["n"], audio, st["first"], total / audio))
    print("WAV ->", out)


async def run_append(out):
    sents = ["第一句话，你好。", "第二句话，世界。", "第三句话，再见。"]
    sid = await _session()
    print("会话 %s | 按句 append（%d 句）" % (sid, len(sents)))
    url = "%s/v1/tts/stream?session_id=%s&response_format=pcm" % (WS, sid)
    pcm, st = bytearray(), {"t0": None}

    async with asyncio.timeout(180):
        async with __import__("websockets").connect(url, ping_interval=None, max_size=None) as ws:
            async def reader():
                while True:
                    m = await ws.recv()
                    if isinstance(m, bytes):
                        print("  [%5.2fs] PCM 块 %d B (%.2fs 音频)"
                              % (time.time() - st["t0"], len(m), len(m) / 2 / SR))
                        pcm.extend(m)
                    else:
                        d = json.loads(m)
                        if d.get("type") == "done":
                            return

            rt = asyncio.create_task(reader())
            await asyncio.sleep(0.2); st["t0"] = time.time()
            for i, s in enumerate(sents, 1):
                print(">> [%5.2fs] 发送 %s" % (time.time() - st["t0"], "start" if i == 1 else "append"))
                await ws.send(json.dumps({"type": "start" if i == 1 else "append", "text": s}))
                await asyncio.sleep(1.5)
            await ws.send(json.dumps({"type": "end"}))
            await asyncio.wait_for(rt, timeout=120)

    _save(pcm, out)
    print("WAV ->", out)


def _save(pcm, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    w = wave.open(out, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(bytes(pcm)); w.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    mode_append = "--append" in args
    args = [a for a in args if a != "--append"]
    out = args[-1] if args and args[-1].endswith(".wav") else "/tmp/k3tr/stream_tts.wav"
    if mode_append:
        asyncio.run(run_append(out))
    else:
        if not args:
            sys.exit('用法: stream-tts.py "文本" [out.wav]  或  stream-tts.py --append [out.wav]')
        asyncio.run(run_once(args[0], out))
