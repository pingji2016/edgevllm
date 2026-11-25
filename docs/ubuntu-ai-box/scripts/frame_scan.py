"""视频理解帧数扫描：2/4/8/16 帧的延迟与描述质量"""
import base64
import json
import os
import time
import urllib.request

PORT = 8092
FRAMES = "/tmp/vlframes16"
OUT = os.path.expanduser("~/apps/infer-research/frame_scan.json")
COUNT = (2, 4, 8, 16)


def main():
    import glob
    files = sorted(glob.glob(os.path.join(FRAMES, "*.jpg")))
    print(f"可用帧 {len(files)}")
    if not files:
        return

    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    rows = []
    for n in COUNT:
        use = files[:n]
        content = [{"type": "text",
                    "text": "这是一段视频的连续帧，请用两句话说明画面里发生了什么变化。"}]
        for f in use:
            b64 = base64.b64encode(open(f, "rb").read()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        body = {"model": "qwen3vl", "max_tokens": 80, "temperature": 0,
                "messages": [{"role": "user", "content": content}]}
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=600).read()  # 预热
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
        dt = time.time() - t0
        txt = resp["choices"][0]["message"]["content"].strip().replace("\n", " ")
        usage = resp.get("usage", {})
        rows.append(dict(frames=n, seconds=round(dt, 2), ptok=usage.get("prompt_tokens"),
                         ctok=usage.get("completion_tokens"), answer=txt[:150]))
        print(f"  {n:2d} 帧 | {dt:5.2f}s | prompt {usage.get('prompt_tokens')} tok | {txt[:60]}",
              flush=True)

    base = rows[0]["seconds"]
    print("\n帧数 | 耗时 | 相对 2 帧 | 每帧边际成本")
    for r in rows:
        marginal = (r["seconds"] - base) / max(r["frames"] - 2, 1)
        print(f"{r['frames']:4d} | {r['seconds']:5.2f}s | {r['seconds']/base:5.2f}× | {marginal*1000:6.0f} ms")

    json.dump(rows, open(OUT, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
