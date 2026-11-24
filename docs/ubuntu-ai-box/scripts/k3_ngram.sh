#!/bin/bash
# K3：统一参数（-t 16 -c 1024）下，抄写任务 baseline vs ngram-simple
set -u
M=~/mage-vl/models/mage-vl-backbone-Q4_K_M.gguf
TEXT="边缘计算把推理放在离用户最近的地方，降低了网络延迟，减少了回传带宽，也避免了数据离开现场。"
COPY="请原样复述下面这段话，不要增删任何字：$TEXT"

run_case() {
  local tag="$1"; local prompt="$2"; shift 2
  echo "=== $tag ==="
  llama-server -m "$M" --port 8097 -t 16 -c 1024 "$@" > /tmp/srv3.log 2>&1 &
  local pid=$!
  local ok=0
  for i in $(seq 1 200); do
    sleep 2
    curl -sf -o /dev/null "http://127.0.0.1:8097/health" && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break
  done
  [ $ok -eq 1 ] || { echo "  ❌ 未就绪"; kill $pid 2>/dev/null; return; }
  curl -s "http://127.0.0.1:8097/completion" -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"$prompt\",\"n_predict\":64,\"temperature\":0}" \
    | python3 -c "
import json,sys
d=json.load(sys.stdin); t=d.get('timings',{})
print(f\"  {t.get('predicted_n')} token | {t.get('predicted_per_second',0):.2f} tok/s | prompt {t.get('prompt_ms',0):.0f} ms\")
" 2>/dev/null || echo "  解析失败"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 2
}

run_case "A 抄写 · 基线" "$COPY"
run_case "B 抄写 · ngram-simple" "$COPY" --spec-type ngram-simple
