#!/bin/bash
# vision-concurrency.sh —— 视觉网关的并发放大曲线
#
# 用法：./vision-concurrency.sh [图片] [模型]
#   默认 ~/k3-ai/out/bus.jpg + yolov8n
#
# 结论（2026-10-20 实测，K3）：
#   并发 1 / 2 / 4 / 8 → 端到端 72 / 101 / 167 / 300 ms，吞吐 13.9 / 19.7 / 23.9 / 26.6 次/秒
#   单帧推理恒为 14.3–14.4 ms：IME 内部是并行/流水处理的，
#   吞吐到 26.6 次/秒就饱和 —— 瓶颈在网关的 HTTP 层与预处理（单进程），不在算力。

set -u
IMG=${1:-$HOME/k3-ai/out/bus.jpg}
MODEL=${2:-yolov8n}
API=http://127.0.0.1:18790/v1/vision/inference

curl -s -X POST http://127.0.0.1:18790/v1/vision/models/load \
     -H 'content-type: application/json' -d "{\"model_id\":\"$MODEL\"}" >/dev/null

for n in 1 2 4 8; do
    rm -f /tmp/vr_*.json
    t0=$(date +%s%N)
    for i in $(seq 1 "$n"); do
        curl -s -m 300 -X POST "$API" -F "file=@$IMG" -F "model_id=$MODEL" \
             -F 'tasks=["detect"]' -o "/tmp/vr_$i.json" &
    done
    wait
    t1=$(date +%s%N)
    python3 - "$n" "$t0" "$t1" <<'PY'
import glob, json, sys
n, t0, t1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
wall = (t1 - t0) / 1e6
ms = []
for f in glob.glob('/tmp/vr_*.json'):
    try:
        ms.append(json.load(open(f))['data']['timing']['model_infer_ms'])
    except Exception:
        pass
if ms:
    print(f"N={n} wall={wall:.0f}ms 吞吐={int(n)/(wall/1000):.1f}次/秒 "
          f"单帧={sum(ms)/len(ms):.1f}ms (max {max(ms):.1f})")
else:
    print(f"N={n} wall={wall:.0f}ms NO_DATA")
PY
    sleep 2
done
