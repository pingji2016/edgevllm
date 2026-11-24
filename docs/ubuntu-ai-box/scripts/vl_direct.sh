#!/bin/bash
set -u
PORT=8092
echo "=== 服务健康检查 ==="
curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && echo "  ✅ vlm-server 在线" || { echo "  ❌ 不在线"; exit 1; }
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

~/apps/bench/yolo-venv/bin/python - "$PORT" /tmp/vlframes <<'PY'
import base64, glob, json, os, sys, time, urllib.request
port, d = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(d, "*.jpg")))
url = f"http://127.0.0.1:{port}/v1/chat/completions"
print(f"帧数 {len(files)}")
lats = []
for f in files[:5]:
    b64 = base64.b64encode(open(f, "rb").read()).decode()
    body = {"model": "qwen3vl", "max_tokens": 48, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "用一句话描述这张图。"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.load(r)
    dt = time.time() - t0
    lats.append(dt)
    txt = resp["choices"][0]["message"]["content"].strip().replace("\n", " ")[:46]
    n = resp.get("usage", {}).get("completion_tokens", 0)
    print(f"{os.path.basename(f):8s} {dt:6.2f}s ({n} tok)  {txt}")
print(f"\n平均 {sum(lats)/len(lats):.2f}s/帧 | 最快 {min(lats):.2f}s | 最慢 {max(lats):.2f}s")
json.dump(dict(lats=[round(x, 2) for x in lats], mean=round(sum(lats)/len(lats), 2)),
          open(os.path.expanduser("~/apps/infer-research/vl_stream.json"), "w"))
PY
