#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""live-voice.py —— VAD 门控的实时语音助手（全在 K3 板上离线跑）

链路：
    麦克风/文件 → 0.4s 窗口 → 网关 VAD 判语音
    → 攒成一句话（静音 0.8s 判定说话结束）→ ASR
    → LLM 流式生成 → 按标点切句 → TTS 逐句流式合成 → 逐句播出

两种音源：
    ./live-voice.py --file 说话.wav     # 文件当麦克风，按真实时间喂（无麦克风也能验证）
    ./live-voice.py --mic              # ALSA 麦克风（arecord 16k 单声道）

可选：
    --no-play      只跑链路不播音（无声卡/远程调试用）
    --no-llm       只做 VAD+ASR（验证门控与识别）
    --once         处理完一句话就退出（默认循环）

★ 板上没有麦克风（§11），不带 --mic 时用 --file 验证；插 USB 麦后用 --mic。
★ 声卡 ES8326 只支持 48k 立体声，所以播出前用 ffmpeg 转成 48k/2ch 再 aplay。
"""

import argparse
import asyncio
import collections
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import wave

G = os.environ.get("GW", "http://127.0.0.1:18790")
WS = G.replace("http://", "ws://")
SR = 16000
WIN_MS = 400           # VAD 窗口
SIL_END_MS = 800       # 连续静音多久判定说完
MIN_UTT_MS = 800       # 太短的忽略
MAX_UTT_MS = 20000     # 超长强制切断
AUDIO_DEV = os.environ.get("AUDIO_DEV", "hw:0,0")
PUNCT = "。！？；，、.!?\n"
MIN_SENT = 8


# ---------------------------------------------------------------- HTTP 小工具
def post_file(path, f, timeout=600):
    cmd = ["curl", "-s", "--max-time", str(timeout), "-X", "POST", G + path,
           "-F", "file=@%s" % f]
    out = subprocess.run(cmd, capture_output=True).stdout
    try:
        return json.loads(out)
    except Exception:
        return {"_raw": out.decode("utf-8", "replace")[:200]}


def write_wav(path, pcm, rate=SR):
    w = wave.open(path, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    w.writeframes(pcm)
    w.close()


def play_pcm(pcm, rate=SR):
    """16k 单声道 PCM → ffmpeg 转 48k/2ch → aplay（阻塞到播完）"""
    if not pcm:
        return
    p1 = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(rate), "-ac", "1", "-i", "-",
         "-ac", "2", "-ar", "48000", "-f", "wav", "-"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["aplay", "-D", AUDIO_DEV, "-q", "-"], stdin=p1.stdout)
    p1.stdout.close()
    try:
        p1.communicate(pcm, timeout=120)
    except Exception:
        pass
    p2.wait()


# ---------------------------------------------------------------- 音源
def mic_stream():
    """返回 (popen, read_fn)：从 arecord 持续读裸 PCM"""
    p = subprocess.Popen(
        ["arecord", "-q", "-f", "S16_LE", "-r", str(SR), "-c", "1", "-t", "raw", "-D", "default"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    return p


def main_loop(args):
    work = tempfile.mkdtemp(prefix="k3live-")
    win_bytes = int(SR * WIN_MS / 1000) * 2

    # --- 音源准备 ---
    src_file = None
    mic = None
    if args.mic:
        mic = mic_stream()
        print("[live] 麦克风已打开（arecord default, 16k mono）")
    else:
        if not args.file:
            sys.exit("要么 --mic，要么 --file 音频.wav")
        w = wave.open(args.file)
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != SR:
            sys.exit("--file 需要 16k 单声道 16bit wav（ffmpeg -ar 16000 -ac 1 转一下）")
        src_file = w
        print("[live] 用文件模拟麦克风：%s（%.1fs，按真实时间喂）"
              % (args.file, w.getnframes() / float(SR)))

    utt = bytearray()          # 当前这句话的 PCM
    silence_ms = 0             # 连续静音
    speech_ms = 0
    t_start = None
    n_window = 0

    print("[live] 开始，窗口 %dms / 静音 %dms 断句\n" % (WIN_MS, SIL_END_MS))

    while True:
        # 取一个窗口
        if mic is not None:
            data = mic.stdout.read(win_bytes)
            if not data:
                break
        else:
            t0 = time.time()
            data = src_file.readframes(int(SR * WIN_MS / 1000))
            if not data:
                break
            # 按真实时间节流
            dt = WIN_MS / 1000.0 - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

        n_window += 1
        wp = os.path.join(work, "w.wav")
        write_wav(wp, data)
        r = post_file("/v1/vad/analyze", wp)
        speech = bool(r.get("is_speech"))
        prob = r.get("probability")
        if prob is None:
            prob = r.get("speech_prob", 0.0)

        if speech:
            if not utt:
                t_start = time.time()
                print("  ▶ 检测到说话 …")
            utt.extend(data)
            silence_ms = 0
            speech_ms += WIN_MS
        else:
            if utt:
                utt.extend(data)          # 静音也留着，语音更自然
                silence_ms += WIN_MS
            else:
                silence_ms = 0

        if args.verbose:
            print("    [%5.1fs] speech=%-5s p=%.3f utt=%dms sil=%dms"
                  % (n_window * WIN_MS / 1000.0, speech, float(prob or 0),
                     len(utt) // 2 * 1000 // SR, silence_ms))

        # 断句
        done = False
        if utt and silence_ms >= SIL_END_MS:
            done = True
        if utt and len(utt) // 2 * 1000 // SR >= MAX_UTT_MS:
            done = True

        if not done:
            continue

        utt_ms = len(utt) // 2 * 1000 // SR
        if utt_ms < MIN_UTT_MS:
            utt = bytearray(); silence_ms = 0; speech_ms = 0
            continue

        print("  ■ 说完（%.1fs），开始识别" % (utt_ms / 1000.0))
        up = os.path.join(work, "utt.wav")
        write_wav(up, bytes(utt))
        t0 = time.time()
        a = post_file("/v1/asr/recognize", up)
        text = (a.get("text") or "").strip()
        t_asr = time.time() - t0
        utt = bytearray(); silence_ms = 0; speech_ms = 0

        if not text:
            print("    （识别为空，忽略）\n")
            continue
        print("  ASR %.2fs → 「%s」" % (t_asr, text))

        if args.no_llm:
            print()
            if args.once:
                break
            continue

        # --- LLM 流式 + TTS 逐句流式 ---
        try:
            answer, t_first_audio = turn(text, play=not args.no_play)
            print("  回复：「%s」" % answer[:80])
            if t_first_audio:
                print("  首个音频 %.2fs（从提问算起）" % t_first_audio)
        except Exception as e:                                  # noqa: BLE001
            print("  回合失败:", e)
        print()

        if args.once:
            break

    if mic is not None:
        mic.terminate()


# ---------------------------------------------------------------- 一个对话回合
def llm_sentences(question, q, T):
    body = json.dumps({
        "messages": [{"role": "user", "content": question + "（请用简洁中文回答，两句话以内）"}],
        "stream": True, "max_tokens": 200, "temperature": 0.6}).encode()
    req = urllib.request.Request(G + "/v1/llm/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    buf = ""
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    j = json.loads(d)
                except Exception:
                    continue
                delta = (j.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                if not delta:
                    continue
                if T["first_token"] is None:
                    T["first_token"] = time.time() - T["t0"]
                buf += delta
                while True:
                    idx = min([i for i in (buf.find(p) for p in PUNCT) if i >= 0] or [-1])
                    if idx < 0:
                        break
                    s = buf[:idx + 1].strip()
                    buf = buf[idx + 1:]
                    if len(s) >= MIN_SENT:
                        T["answer"] += s
                        q.put(s)
        if buf.strip():
            T["answer"] += buf.strip()
            q.put(buf.strip())
    except Exception as e:                                       # noqa: BLE001
        print("  [LLM 错误]", e)
    q.put(None)


def turn(question, play=True):
    import websockets

    T = {"t0": time.time(), "first_token": None, "answer": ""}
    q = queue.Queue()
    threading.Thread(target=llm_sentences, args=(question, q, T), daemon=True).start()

    req = urllib.request.Request(G + "/v1/tts/stream/session",
                                 data=json.dumps({"response_format": "pcm"}).encode(),
                                 headers={"Content-Type": "application/json"})
    sid = json.load(urllib.request.urlopen(req, timeout=60))["session_id"]
    url = "%s/v1/tts/stream?session_id=%s&response_format=pcm" % (WS, sid)

    async def run():
        first_audio = None
        async with websockets.connect(url, ping_interval=None, max_size=None) as ws:
            async def reader():
                nonlocal first_audio
                while True:
                    m = await ws.recv()
                    if isinstance(m, bytes):
                        if first_audio is None:
                            first_audio = time.time() - T["t0"]
                        if play:
                            await asyncio.to_thread(play_pcm, m)
                    else:
                        if json.loads(m).get("type") == "done":
                            return
            rt = asyncio.create_task(reader())
            await asyncio.sleep(0.15)
            idx = 0
            while True:
                s = await asyncio.to_thread(q.get)
                if s is None:
                    break
                idx += 1
                await ws.send(json.dumps({"type": "start" if idx == 1 else "append", "text": s}))
            await ws.send(json.dumps({"type": "end"}))
            await asyncio.wait_for(rt, timeout=180)
        return first_audio

    first = asyncio.run(run())
    return T["answer"], first


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic", action="store_true", help="用 ALSA 麦克风")
    ap.add_argument("--file", help="用 wav 文件模拟麦克风")
    ap.add_argument("--no-play", action="store_true", help="不播音")
    ap.add_argument("--no-llm", action="store_true", help="只跑 VAD+ASR")
    ap.add_argument("--once", action="store_true", help="处理一句话就退出")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if not a.mic and not a.file:
        a.file = os.path.expanduser("~/k3-ai/out/long2.wav")   # 方便直接试
        print("[live] 未指定音源，默认用 %s" % a.file)
    try:
        main_loop(a)
    except KeyboardInterrupt:
        print("\n[live] 中断")
