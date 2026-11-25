#!/bin/bash
set -u
F=~/apps/llama-omni/src/llama-moe-slot.cpp
cp -n "$F" "$F.bak" 2>/dev/null || true

python3 - <<'PY'
import os
p = os.path.expanduser("~/apps/llama-omni/src/llama-moe-slot.cpp")
s = open(p, encoding="utf-8").read()

old1 = """        std::lock_guard<std::mutex> lock(fd_mutex);
        for (const auto & kv : fds) {
            if (fcntl(kv.second, F_NOCACHE, on ? 1 : 0) != 0) {
                moe_log("F_NOCACHE 切换失败: %s", std::strerror(errno));
            }
        }"""
new1 = """        std::lock_guard<std::mutex> lock(fd_mutex);
#ifdef __APPLE__
        for (const auto & kv : fds) {
            if (fcntl(kv.second, F_NOCACHE, on ? 1 : 0) != 0) {
                moe_log("F_NOCACHE 切换失败: %s", std::strerror(errno));
            }
        }
#endif"""
if old1 in s:
    s = s.replace(old1, new1, 1)
    print("  补丁1 ✅")
else:
    print("  补丁1 未匹配")

old2 = """        if (experts_nocache_desired() && fcntl(fd, F_NOCACHE, 1) != 0) {
            moe_log("F_NOCACHE 设置失败: %s", std::strerror(errno));
        }"""
new2 = """#ifdef __APPLE__
        if (experts_nocache_desired() && fcntl(fd, F_NOCACHE, 1) != 0) {
            moe_log("F_NOCACHE 设置失败: %s", std::strerror(errno));
        }
#endif"""
if old2 in s:
    s = s.replace(old2, new2, 1)
    print("  补丁2 ✅")
else:
    print("  补丁2 未匹配")

open(p, "w", encoding="utf-8").write(s)
PY

echo "=== 重新编译 ==="
cd ~/apps/llama-omni || exit 1
timeout 3000 cmake --build build -j 10 > /tmp/omni_build2.log 2>&1
rc=$?
echo "  返回码 $rc"
if [ $rc -eq 0 ]; then
  echo "BUILD_OK"
  ls build/bin/ | grep -E "llama-(cli|mtmd)" | head -5
else
  grep -nE "error:" /tmp/omni_build2.log | head -5 | cut -c1-180
fi
