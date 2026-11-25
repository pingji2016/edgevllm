#!/bin/bash
set -u
NCU=/usr/local/cuda/bin/ncu
PY=~/apps/bench/yolo-venv/bin/python
OUT=~/apps/infer-research/roofline

mkdir -p "$OUT"
cp /tmp/roof_kernels.py "$OUT/" 2>/dev/null || cp /tmp/rf16/roof_kernels.py "$OUT/"

run_case() {
  local case="$1"; local skip="$2"
  echo "================ $case ================"
  echo 123456 | sudo -S $NCU --target-processes all \
    --kernel-name-base demangled \
    --launch-skip "$skip" --launch-count 1 \
    --section SpeedOfLight \
    "$PY" "$OUT/roof_kernels.py" "$case" 2>&1 \
    | grep -E "Duration|Elapsed Cycles|Compute \(SM\)|Memory Throughput|DRAM Throughput|L1/TEX|L2 Cache|SM Frequency|Section:|Kernel Name" \
    | sed 's/^  */  /'
  echo
}

run_case gemm 3
run_case copy 3
run_case add 3
run_case attn 1
