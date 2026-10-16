#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真·流式语音助手：LLM 令牌流 -> 按句切分 -> TTS 逐句合成 -> 拼音频（全在 K3 板上）

用法:
    ./stream-voice.py "你的问题" [out.wav]

时序观测：
    t=0 发起 LLM 流式请求；首个 token、每句送入 TTS、每句首个音频块的时间都会打出来。
    关键指标 = **首个音频块延迟**（用户多快听到第一声）。

★ 实测结论（报告 §13.4）：
    流式链路首音频 **3.03s**；非流式（等 LLM 全生成 + TTS 整段）要 **28.41s** —— 快 9.4×。
    切句点包含逗号（，、），这样第一小句一出来就能开播。
"""
import asyncio, json, os, queue, sys, threading, time, urllib.request, wave

G = os.environ.get("GW", "http://127.0.0.1:18790")
WS = G.replace("http://", "ws://")
SR = 16000
PUNCT = "。！？；，、.!?\n"     # 也切逗号/顿号：让第一小句尽早出声
MIN_SENT = 8


def llm_thread(question, q, T):
    body = json.dumps({
        "messages": [{"role": "user",
                      "content": question + "（请用简洁的中文分点回答，控制在三到四点）"}],
        "stream": True, "max_tokens": 320, "temperature": 0.6}).encode()
    req = urllib.request.Request(G + "/v1/llm/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    buf = ""
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    d = json.loads(data)
                except Exception:
                    continue
                delta = (d.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                if not delta:
                    continue
                if T["first_token"] is None:
                    T["first_token"] = time.time() - T["t0"]
                    print("  [%5.2fs] LLM 首个 token" % T["first_token"])
                buf += delta
                while True:
                    idx = min([i for i in (buf.find(p) for p in PUNCT) if i >= 0] or [-1])
                    if idx < 0:
                        break
                    sent = buf[:idx + 1].strip()
                    buf = buf[idx + 1:]
                    if len(sent) >= MIN_SENT:
                        q.put(sent)
        if buf.strip():
            q.put(buf.strip())
    except Exception as e:                                   # noqa: BLE001
        print("  [LLM 错误]", e)
    q.put(None)


async def main(question, out):
    T = {"t0": None, "first_token": None, "first_audio": None}
    q = queue.Queue()

    req = urllib.request.Request(G + "/v1/tts/stream/session",
                                 data=json.dumps({"response_format": "pcm"}).encode(),
                                 headers={"Content-Type": "application/json"})
    sid = json.load(urllib.request.urlopen(req))["session_id"]
    url = "%s/v1/tts/stream?session_id=%s&response_format=pcm" % (WS, sid)

    pcm = bytearray()
    T["t0"] = time.time()
    threading.Thread(target=llm_thread, args=(question, q, T), daemon=True).start()

    async with asyncio.timeout(300):
        async with __import__("websockets").connect(url, ping_interval=None, max_size=None) as ws:
            state = {"idx": 0}

            async def reader():
                while True:
                    m = await ws.recv()
                    if isinstance(m, bytes):
                        if T["first_audio"] is None:
                            T["first_audio"] = time.time() - T["t0"]
                            print("  [%5.2fs] ★首个音频块（用户听到第一声）  %.2fs 音频"
                                  % (T["first_audio"], len(m) / 2 / SR))
                        pcm.extend(m)
                    else:
                        if json.loads(m).get("type") == "done":
                            return

            rt = asyncio.create_task(reader())
            await asyncio.sleep(0.2)
            T["t0"] = time.time()

            while True:
                sent = await asyncio.to_thread(q.get)
                if sent is None:
                    break
                state["idx"] += 1
                print("  [%5.2fs] 送 TTS 第 %d 句: %s" % (time.time() - T["t0"], state["idx"], sent[:30]))
                await ws.send(json.dumps({"type": "start" if state["idx"] == 1 else "append",
                                          "text": sent}))
            await ws.send(json.dumps({"type": "end"}))
            await asyncio.wait_for(rt, timeout=120)

    total = time.time() - T["t0"]
    audio = len(pcm) / 2.0 / SR
    os.makedirs(os.path.dirname(out), exist_ok=True)
    w = wave.open(out, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(bytes(pcm)); w.close()
    print("\n结果: 音频 %.2fs | LLM 首 token %.2fs | ★首音频 %.2fs | 总耗时 %.2fs"
          % (audio, T["first_token"] or -1, T["first_audio"] or -1, total))
    print("      => 用户等 %.2fs 就听到第一声（非流式要 ~28s）" % (T["first_audio"] or -1))
    print("WAV ->", out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('用法: stream-voice.py "问题" [out.wav]')
    asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "/tmp/k3tr/stream_voice.wav"))
