#!/bin/sh
# K3 视觉模型全家桶 benchmark —— 跑在板子上（SpaceMiT AI Gateway :18790）
#
# 用法:
#   ./vision-zoo.sh [图片目录]        默认 $HOME/k3-ai/out，需要 bus.jpg / zidane.jpg
#   GW=http://127.0.0.1:18790 ./vision-zoo.sh
#
# 说明:
#   - 网关 /v1/vision/models/load 的字段是 model_id（不是 model）。
#   - /v1/vision/inference 是 multipart，tasks 必须是 JSON 数组字符串，
#     例如 -F 'tasks=["detect"]'；结果在 data.results.<任务名> 里。
#   - 每个模型首次 load 会从 archive.spacemit.com 现下（yolov8m 约 80s、yolov11m 约 116s）。
#   - ★ 板上别并发跑；一次一个模型，否则会把板子压到 load≈100（见报告 §12.8）。
set -u

G=${GW:-http://127.0.0.1:18790}
IMG_DIR=${1:-$HOME/k3-ai/out}
JSON=/tmp/vzoo.json

# 模型:任务:图片
SPECS="
yolov8n:detect:bus.jpg
yolov8s:detect:bus.jpg
yolov8m:detect:bus.jpg
yolov11n:detect:bus.jpg
yolov11s:detect:bus.jpg
yolov11m:detect:bus.jpg
yolov8n-pose:pose:zidane.jpg
yolov8s-pose:pose:zidane.jpg
yolov8n-seg:segment:bus.jpg
yolov8s-seg:segment:bus.jpg
yolov5-face:detect:zidane.jpg
yolov5-gesture:detect:bus.jpg
emotion:detect:zidane.jpg
resnet50:classify:bus.jpg
"

printf '%-16s %-9s %8s  %s\n' 模型 任务 推理ms 检出
printf -- '------------------------------------------------------------------------\n'

echo "$SPECS" | while IFS=: read -r m task img; do
    [ -n "$m" ] || continue
    f="$IMG_DIR/$img"
    [ -f "$f" ] || { printf '%-16s %-9s  (缺图 %s)\n' "$m" "$task" "$img"; continue; }

    curl -s --max-time 600 -X POST "$G/v1/vision/models/load" \
        -H 'Content-Type: application/json' -d "{\"model_id\":\"$m\"}" >/dev/null

    i=1
    while [ "$i" -le 3 ]; do
        curl -s --max-time 120 -X POST "$G/v1/vision/inference" \
            -F "file=@$f" -F "model_id=$m" -F "tasks=[\"$task\"]" -o "$JSON"
        i=$((i + 1))
    done

    python3 - "$JSON" "$m" "$task" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
t = d.get("data", {}).get("timing", {})
res = d.get("data", {}).get("results") or {}
key = next((k for k, v in res.items() if v), None)
items = res.get(key) or [] if key else []
lab = []
for r in items[:6]:
    n = r.get("label_name") or r.get("label")
    s = r.get("score")
    lab.append("%s %.2f" % (n, s) if isinstance(s, float) else str(n))
print("%-16s %-9s %8.1f  %s" % (sys.argv[2], sys.argv[3],
      t.get("model_infer_ms", 0), ", ".join(lab) or "(无检出)"))
PY
done
