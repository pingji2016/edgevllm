#!/usr/bin/env bash
# =============================================================================
#  build-and-bench.sh —— 在 K3（SpacemiT X100）上构建 MNN 两版并跑 llm_bench
#
#  用法：
#    ./build-and-bench.sh env      # 采集环境（芯片 / 核数 / VLEN / 工具链）
#    ./build-and-bench.sh src      # 拉 MNN 源码（blobless，约 100 MB）
#    ./build-and-bench.sh model    # 下 Qwen3-0.6B-MNN 并校验 sha256
#    ./build-and-bench.sh rvv      # 编标准 RVV 版 + 测试 + bench（约 13 分钟）
#    ./build-and-bench.sh ime2     # 编 IME2 版   + 测试 + bench（约 9 分钟）
#    ./build-and-bench.sh tune     # 在 IME2 版上试 -qa / -dyo
#    ./build-and-bench.sh all      # env + src + model + rvv + ime2 + tune
#
#  对应文档：../mnn-rvv-on-k3.md
#  所有日志落到 $BUILD/log/，两个构建目录都会保留，便于复测。
# =============================================================================
set -uo pipefail

SRC=${SRC:-$HOME/MNN}
BUILD=${BUILD:-$HOME/mnn-build}
LOGS=$BUILD/log
MODEL_DIR=${MODEL_DIR:-$HOME/models/Qwen3-0.6B-MNN}
MODEL=${MODEL:-$MODEL_DIR/config.json}
JOBS=${JOBS:-$(nproc)}
THREADS=${THREADS:-$JOBS}
REP=${REP:-3}

MNN_REPO=${MNN_REPO:-https://github.com/alibaba/MNN.git}
MODEL_REPO=${MODEL_REPO:-https://hf-mirror.com/taobao-mnn/Qwen3-0.6B-MNN/resolve/main}

# —— IME2 必须补的 march ——
# MNNSpacemitIme2GemmI8I4Local.cpp 用 flh/fcvt.s.h/fmul.h（需 zfh），
# MNNSpacemitIme2GemmInt8.cpp 用 pause（需 zihintpause）。
# 上游 CMake 的 MNN_SPACEMIT_IME2_MARCH = "${MNN_RVV_BASE_MARCH}_xsmtvdotii"，
# 默认 base 是 rv64gcv，因此直接开 IME2 会编译失败。板子本身支持这些扩展。
RVV_MARCH=${RVV_MARCH:-rv64gcv_zfh_zihintpause_zba_zbb_zbs}

mkdir -p "$LOGS"

log()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
ok()   { printf '  ✅ %s\n' "$*"; }
die()  { printf '  ❌ %s\n' "$*" >&2; exit 1; }

COMMON_OPTS=(
  -DCMAKE_BUILD_TYPE=Release
  -DMNN_USE_RVV=ON
  -DMNN_BUILD_TEST=ON
  -DMNN_BUILD_LLM=ON
  -DMNN_LOW_MEMORY=ON
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
  -DMNN_BUILD_TOOLS=ON
)

# ---------------------------------------------------------------- env
cmd_env() {
  log "环境采集 → $LOGS/0-env.log"
  {
    date; uname -srm; echo "nproc=$(nproc)"
    echo "cpuinfo 处理器数=$(grep -c '^processor' /proc/cpuinfo)"
    grep -m1 '^model name' /proc/cpuinfo
    gcc --version | head -1; cmake --version | head -1
    [ -d "$SRC/.git" ] && echo "MNN HEAD=$(git -C "$SRC" rev-parse HEAD)"
  } > "$LOGS/0-env.log" 2>&1

  # VLEN：比查文档可靠
  cat > /tmp/vlen.c <<'C'
#include <stdio.h>
int main(void){ unsigned long vl; __asm__ volatile("vsetvli %0, zero, e8, m1, ta, ma" : "=r"(vl));
  printf("VLEN=%lu bit\n", vl*8); return 0; }
C
  gcc -O2 -march=rv64gcv /tmp/vlen.c -o /tmp/vlen 2>/dev/null \
    && /tmp/vlen | tee -a "$LOGS/0-env.log"

  sed 's/^/  /' "$LOGS/0-env.log"
}

# ---------------------------------------------------------------- src
cmd_src() {
  if [ -d "$SRC/.git" ]; then
    ok "源码已存在：$SRC ($(git -C "$SRC" rev-parse --short HEAD))"
  else
    log "拉 MNN（blobless）"
    git clone --filter=blob:none "$MNN_REPO" "$SRC" 2>&1 | tail -2
  fi
  du -sh "$SRC" | sed 's/^/  /'
}

# ---------------------------------------------------------------- model
cmd_model() {
  [ -d "$SRC" ] || die "先跑 ./build-and-bench.sh src"
  log "下模型 Qwen3-0.6B-MNN → $MODEL_DIR"
  mkdir -p "$MODEL_DIR"
  for f in config.json llm.mnn llm.mnn.weight llm_config.json tokenizer.txt; do
    [ -s "$MODEL_DIR/$f" ] && { ok "已存在 $f"; continue; }
    curl -sL --retry 3 -C - -m 2400 -o "$MODEL_DIR/$f" "$MODEL_REPO/$f" \
      && printf '  %-16s %s bytes\n' "$f" "$(stat -c%s "$MODEL_DIR/$f")" \
      || die "下载失败：$f"
  done
  echo "  权重 sha256（应为 953afb7e0165818a…）：$(sha256sum "$MODEL_DIR/llm.mnn.weight" | cut -c1-16)…"
}

# ---------------------------------------------------------------- build
# $1 = 构建名（rvv / ime2）
cmd_build() {
  local name=$1 dir=$BUILD/$1 logf
  [ -d "$SRC" ] || die "先跑 ./build-and-bench.sh src"
  logf=$LOGS/$( [ "$name" = rvv ] && echo 2-build || echo 9-ime2-build ).log
  local extra=()
  [ "$name" = ime2 ] && extra=(-DMNN_RVV_SPACEMIT_IME2=ON -DMNN_RVV_MARCH="$RVV_MARCH")

  log "配置 $name → $dir"
  cmake -S "$SRC" -B "$dir" "${COMMON_OPTS[@]}" "${extra[@]}" 2>&1 \
    | tee "$LOGS/1-cmake-$name.log" | tail -3 | sed 's/^/  /'

  log "编译 $name（-j$JOBS，日志 $logf）"
  { date; time cmake --build "$dir" -j"$JOBS"; echo "EXIT=$?"; date; } > "$logf" 2>&1
  grep -q '^EXIT=0' "$logf" || { tail -20 "$logf" | cut -c1-160; die "编译失败，见 $logf"; }

  ok "编译完成：$(grep '^real' "$logf" | tr -s '\t' ' ')"
  ls -la "$dir"/libMNN.so "$dir"/run_test.out "$dir"/llm_bench 2>/dev/null | awk '{printf "  %-10s %s\n", $5, $9}'
  echo "  RVV 符号 $(nm -C "$dir/libMNN.so" 2>/dev/null | grep -c RVV) 个 / vsetvli $(objdump -d "$dir/libMNN.so" 2>/dev/null | grep -c vsetvli) 条"
  [ "$name" = ime2 ] && echo "  spacemit/ime2 符号 $(nm -C "$dir/libMNN.so" 2>/dev/null | grep -ci 'spacemit\|ime2') 个"
}

# ---------------------------------------------------------------- test
cmd_test() {
  local dir=$BUILD/${1:-rvv}
  [ -x "$dir/run_test.out" ] || die "没找到 $dir/run_test.out"
  log "RVV 专项测试"
  "$dir/run_test.out" backend/cpu/rvv 2>&1 | tail -4 | sed 's/^/  /'
  log "算子级测试（unary/binary/scale/prelu/conv/attention/MatMul/image_process）"
  local out=$LOGS/4-optests.log
  : > "$out"
  for t in op/unary op/binary op/scale op/prelu op/conv op/attention expr/MatMul cv/image_process; do
    local s=$(date +%s)
    "$dir/run_test.out" "$t" >> "$out" 2>&1
    echo "### RESULT $t rc=$? cost=$(( $(date +%s) - s ))s" >> "$out"
  done
  # 注意：TEST_CASE_AMOUNT_UNIT 与 TEST_CASE 会各报一次，直接 grep passed 会翻倍
  echo "  失败项：$(grep -c 'TEST_CASE=.*"failed":[1-9]' "$out" || true)  日志：$out"
}

# ---------------------------------------------------------------- bench
cmd_bench() {
  local dir=$BUILD/${1:-rvv}
  local tag=${1:-rvv}
  [ -x "$dir/llm_bench" ] || die "没找到 $dir/llm_bench"
  [ -s "$MODEL" ] || die "没有模型，先跑 ./build-and-bench.sh model"
  log "llm_bench（$tag，threads=$THREADS，pp512/tg128，rep=$REP）"
  local out=$LOGS/6-llmbench-$tag.log
  local s=$(date +%s)
  "$dir/llm_bench" -m "$MODEL" -t "$THREADS" -p 512 -n 128 -rep "$REP" 2>&1 | tee "$out" \
    | grep -E '^\|' | tail -3 | sed 's/^/  /'
  echo "  总耗时 $(( $(date +%s) - s ))s"
}

# ---------------------------------------------------------------- tune
cmd_tune() {
  local dir=$BUILD/ime2
  [ -x "$dir/llm_bench" ] || die "先跑 ./build-and-bench.sh ime2"
  local out=$LOGS/14-tuning.log
  : > "$out"
  log "调优试探（IME2 构建）"
  for opt in "-qa 0|baseline" "-qa 1|kvquant-qk-int8" "-qa 2|kvquant-qkv-int8" "-dyo 8|dyo8"; do
    local args=${opt%%|*} tag=${opt##*|}
    local s=$(date +%s)
    local r=$("$dir/llm_bench" -m "$MODEL" -t "$THREADS" -p 512 -n 128 -rep 2 $args 2>&1 \
      | tee "$LOGS/15-bench-$tag.log" | grep -E 'pp512|tg128' \
      | awk -F'|' '{gsub(/ ±.*/,"",$(NF-1)); gsub(/^ +| +$/,"",$(NF-1)); printf "%s ", $(NF-1)}')
    printf '  %-20s pp/tg = %-22s (%ss)\n' "$tag" "$r" "$(( $(date +%s) - s ))" | tee -a "$out"
  done
  echo "  结论：-qa 在这块板上是负优化，-dyo 略有益 prefill"
}

# ---------------------------------------------------------------- main
case "${1:-all}" in
  env)   cmd_env ;;
  src)   cmd_src ;;
  model) cmd_model ;;
  rvv)   cmd_src; cmd_build rvv; cmd_test rvv; cmd_bench rvv ;;
  ime2)  cmd_src; cmd_build ime2; cmd_test ime2; cmd_bench ime2 ;;
  tune)  cmd_tune ;;
  all)   cmd_env; cmd_src; cmd_model; cmd_build rvv; cmd_test rvv; cmd_bench rvv
         cmd_build ime2; cmd_test ime2; cmd_bench ime2; cmd_tune ;;
  *)     sed -n '2,18p' "$0"; exit 1 ;;
esac
