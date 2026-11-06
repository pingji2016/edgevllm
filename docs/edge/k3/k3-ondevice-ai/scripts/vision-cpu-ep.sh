#!/bin/bash
# vision-cpu-ep.sh —— 同一份 ONNX 模型，纯 CPU EP 跑不同线程数
#
# 用法：./vision-cpu-ep.sh [onnx] [线程...]
#   默认 ~/.cache/models/vision/yolov8/yolov8n_no_dfl.q.onnx，线程 1 2 4 8
#
# 结论（2026-10-20 实测，K3）：1 / 2 / 4 / 8 线程 = 860 / 820 / 840 / 943 ms
#   加线程不但没用，8 线程还更慢 —— RISC-V 上这条通用 ONNX 路径没并行起来。
#   （对照：同一个模型走网关的 SpaceMIT EP，单帧 14.4 ms）

set -u
M=${1:-$HOME/.cache/models/vision/yolov8/yolov8n_no_dfl.q.onnx}
shift 2>/dev/null || true
THREADS=${*:-"1 2 4 8"}

[ -f "$M" ] || { echo "找不到模型：$M" >&2; exit 1; }
for t in $THREADS; do
    echo "== threads=$t =="
    onnxruntime_perf_test -e cpu -m times -r 20 -t "$t" -I "$M" 2>&1 | grep -E '^P(95|99) Latency'
done
