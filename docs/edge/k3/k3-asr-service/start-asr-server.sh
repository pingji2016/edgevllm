#!/bin/sh
# 启动「流式语音转文字」WebSocket 服务（sherpa-onnx + streaming zipformer bilingual zh-en）
# 环境变量可覆盖：PORT THREADS BATCH
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
BIN=$DIR/sherpa-onnx-v1.13.8-linux-riscv64-spacemit-shared/bin/sherpa-onnx-online-websocket-server
PORT=${PORT:-6006}
THREADS=${THREADS:-4}
BATCH=${BATCH:-4}
exec "$BIN" \
  --tokens=$DIR/tokens.txt \
  --encoder=$DIR/encoder-epoch-99-avg-1.int8.onnx \
  --decoder=$DIR/decoder-epoch-99-avg-1.int8.onnx \
  --joiner=$DIR/joiner-epoch-99-avg-1.int8.onnx \
  --port="$PORT" \
  --num-work-threads="$THREADS" \
  --max-batch-size="$BATCH" \
  --loop-interval-ms=10 \
  --enable-endpoint=true \
  --log-file=$DIR/ws.log
