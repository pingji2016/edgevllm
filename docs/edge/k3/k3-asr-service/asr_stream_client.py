#!/usr/bin/env python3
"""流式 ASR 客户端示例：把 16k/单声道/16bit 的 wav 以流的方式发给 sherpa-onnx WebSocket 服务，
边发边收增量结果（partial），最后一行是 is_final=true 的整句结果。

用法:
  python3 asr_stream_client.py foo.wav [host] [port] [chunk_sec]
"""
import asyncio, sys, json, wave
import numpy as np
import websockets

async def run(wav, host="127.0.0.1", port=6006, chunk_sec=0.5):
    with wave.open(wav, "rb") as wf:
        sr = wf.getframerate()
        assert wf.getnchannels() == 1 and wf.getsampwidth() == 2, "需要 16kHz 单声道 16bit wav"
        x = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    n = max(1, int(sr * chunk_sec))

    async with websockets.connect(f"ws://{host}:{port}", max_size=None) as ws:
        async def recv():
            async for msg in ws:
                d = json.loads(msg)
                tag = "FINAL" if d.get("is_final") or d.get("is_eof") else "part "
                print(f"[{tag}] {d.get('text','')}", flush=True)
                if d.get("is_final") or d.get("is_eof"):
                    return
        rt = asyncio.create_task(recv())
        for i in range(0, len(x), n):
            await ws.send(x[i:i + n].tobytes())     # 二进制帧 = float32 样本
            await asyncio.sleep(chunk_sec)          # 模拟实时发送（mic 场景即音频到达速率）
        await ws.send("Done")                        # 结束
        await rt

if __name__ == "__main__":
    a = sys.argv[1:]
    wav = a[0]
    host = a[1] if len(a) > 1 else "127.0.0.1"
    port = int(a[2]) if len(a) > 2 else 6006
    cs = float(a[3]) if len(a) > 3 else 0.5
    asyncio.run(run(wav, host, port, cs))
