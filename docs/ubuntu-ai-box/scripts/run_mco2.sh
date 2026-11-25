#!/bin/bash
set -u
B=~/apps/llama-omni/build/bin
D=~/.cache/modelscope/models/OpenBMB--MiniCPM-o-4_5-gguf/snapshots/master
export LD_LIBRARY_PATH="$B"

run() {
  local img="$1"; local prompt="$2"; local tag="$3"
  echo "=== $tag ==="
  local t0=$(date +%s)
  timeout 1200 "$B/llama-mtmd-cli" -m "$D/MiniCPM-o-4_5-Q4_K_M.gguf" \
    --mmproj "$D/vision/MiniCPM-o-4_5-vision-F16.gguf" --no-mmproj-offload \
    --image "$img" -p "$prompt" -n 220 -ngl 99 --no-warmup </dev/null 2>&1 \
    | grep -vE "^llama_|^load_|^print_info|^main:|^system_info|^sampling|^0\.[0-9]" \
    | tail -14
  local dt=$(( $(date +%s) - t0 ))
  echo "  [耗时 ${dt}s]"
  echo
}

run /tmp/face.png "用一句话描述这张图。" "合成人脸描述"
run /tmp/ocr_doc.png "这张文档图里提到的解码速度是多少？只回答数字和单位。" "文档图问答"
