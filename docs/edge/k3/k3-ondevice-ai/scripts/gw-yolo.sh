#!/usr/bin/env bash
# =============================================================================
#  gw-yolo.sh —— 用板载 SpaceMiT AI Gateway 跑目标检测（YOLO，int8）
#
#  用法：
#    ./gw-yolo.sh img.jpg                 # 默认 yolov8n
#    ./gw-yolo.sh img.jpg yolov11n        # 换模型
#    ./gw-yolo.sh --list                  # 列出可用视觉模型
#
#  实测：yolov8n 单帧 model_infer ≈14.8ms（native backend + SpaceMIT EP int8），
#        对比 onnxruntime CPU EP 的 832ms 快约 56 倍。
# =============================================================================
set -uo pipefail
GW="${GW:-http://127.0.0.1:18790}"

if [ "${1:-}" = "--list" ] || [ "${1:-}" = "" ]; then
  echo "可用视觉模型（部分）："
  curl -s --max-time 20 "$GW/v1/vision/models" \
    | python3 -c 'import sys,json
d=json.load(sys.stdin)["data"]["data"]
for m in d[:40]: print("  %-22s %-10s %s" % (m["model_id"], ",".join(m.get("capabilities") or []), m["status"]))' 2>/dev/null
  echo
  echo "用法: $0 <图片> [模型ID]"
  exit 0
fi

IMG="$1"; MODEL="${2:-yolov8n}"
[ -f "$IMG" ] || { echo "找不到图片 $IMG" >&2; exit 1; }
curl -s --max-time 5 "$GW/" >/dev/null || { echo "连不上网关 $GW" >&2; exit 1; }

# 没加载就先加载（模型已缓存时是毫秒级；首次会从 archive.spacemit.com 下载）
st=$(curl -s "$GW/v1/vision/models" | python3 -c "import sys,json
for m in json.load(sys.stdin)['data']['data']:
    if m['model_id']=='$MODEL': print(m['status'])" 2>/dev/null)
if [ "$st" != "ready" ] && [ "$st" != "loaded" ]; then
  echo "==> 加载模型 $MODEL ..."
  curl -s -X POST "$GW/v1/vision/models/load" -H 'Content-Type: application/json' \
    -d "{\"model_id\":\"$MODEL\"}" >/dev/null
fi

T0=$(date +%s%N)
curl -s --max-time 120 -X POST "$GW/v1/vision/inference" \
  -F "file=@$IMG" -F "model_id=$MODEL" -F 'tasks=["detect"]' -o /tmp/gw-yolo.json
T1=$(date +%s%N)

python3 - "$MODEL" "$(( (T1-T0)/1000000 ))" <<'PY'
import sys, json
model, wall = sys.argv[1], sys.argv[2]
d = json.load(open('/tmp/gw-yolo.json'))
data = d.get('data', {})
det = (data.get('results') or {}).get('detect') or []
t = data.get('timing') or {}
print("模型 %s   HTTP 端到端 %s ms" % (model, wall))
print("  preprocess %.2fms  model_infer %.2fms  postprocess %.2fms"
      % (t.get('preprocess_ms', 0), t.get('model_infer_ms', 0), t.get('postprocess_ms', 0)))
print("  检出 %d 个目标：" % len(det))
for x in det:
    print("    - %-14s %.3f" % (x.get('label', x.get('class_name', '?')), x.get('score', 0)))
PY
