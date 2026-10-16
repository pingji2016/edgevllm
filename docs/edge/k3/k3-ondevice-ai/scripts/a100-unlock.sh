#!/usr/bin/env bash
# =============================================================================
#  a100-unlock.sh —— 在 K3 上启用 A100 核组（并给出正确用法提醒）
#
#  用法：
#    ./a100-unlock.sh                  # 打印当前核掩码与 A100 状态
#    ./a100-unlock.sh nproc            # 在 A100 上跑 nproc（应输出 8）
#    ./a100-unlock.sh llama-bench ...  # 在 A100 上跑任意程序
#
#  原理：内核 HMP（CONFIG_SPACEMIT_HMP=y）把 16 核分成两组——
#        X100(0-7, HMP_REGULAR) / A100(8-15, HMP_AI)。新线程默认落在 X100，
#        `taskset -c 8` 会 EINVAL（类别限制，不是故障）。写 /proc/set_ai_thread
#        把线程切成 HMP_AI 后，才允许绑到 8-15（内核会自动给 A100 上电）。
#
#  ★ 重要：A100 是向量/矩阵专用核，跑通用 LLM（llama.cpp Q4 路径）实测
#    比 X100 慢 5.8 倍（L2 1MB vs 4MB、主频 1.8G vs 2.15G）。
#    它适合 int8 CNN / IME 算子（即 SpaceMITExecutionProvider 那条路）。
# =============================================================================
set -uo pipefail

mask() { awk '/Cpus_allowed_list/{print $2}' /proc/self/status; }

if [ "${1:-}" = "" ] || [ "${1:-}" = "--status" ]; then
  echo "当前进程核掩码 : $(mask)   (nproc=$(nproc))"
  echo "A100 核组      : 8-15"
  if [ ! -e /proc/set_ai_thread ]; then
    echo "!! /proc/set_ai_thread 不存在 → 内核没开 CONFIG_SPACEMIT_HMP" >&2
    exit 1
  fi
  echo "开关           : /proc/set_ai_thread  ($(ls -l /proc/set_ai_thread | awk '{print $1}'))"
  echo
  echo "试试： $0 nproc        # 期望输出 8"
  echo "       $0 taskset -c 8-15 llama-bench -m model.gguf -t 8"
  exit 0
fi

# 优先用不带向量操作的 C 小工具（最贴近官方语义）；没有就退回 shell 写法
AIX="${AIX:-$HOME/.local/bin/k3aix}"
if [ -x "$AIX" ]; then
  exec "$AIX" "$@"
fi

if [ -x "$(dirname "$0")/k3aix" ]; then
  exec "$(dirname "$0")/k3aix" "$@"
fi

# 退路：shell 直接写（可能触发 "vector has been enabled already" 告警，通常仍可用）
echo 0 > /proc/set_ai_thread 2>/dev/null || {
  echo "写 /proc/set_ai_thread 失败" >&2; exit 1; }
exec taskset -c 8-15 "$@"
