#!/usr/bin/env python3
# =============================================================================
#  von-bench.py —— 在 K3 上量 System One 决策模型（Von）的延迟
#
#  用法：
#    HF_HUB_OFFLINE=1 VON_DEVICE=cpu ~/von/venv/bin/python von-bench.py
#
#  前置：
#    python3 -m venv --system-site-packages ~/von/venv
#    ~/von/venv/bin/pip install "von-sdk>=1.3.7"
#    权重放到 ~/models/von 后软链进 HF cache（见 docs/k3-system-one-von.md §2）
#
#  实测（2026-10-27，K3 / 8×X100）：19 token 输入 p50 = 1017 ms（2 选项）
#    → 8 选项 1082 ms（只 +6%，一次前向与选项数无关）
#    → 256 token 输入 7676 ms（约 40 ms/token，线性）
#    → 首次加载 5.4 s，峰值 RSS 3.46 GB
# =============================================================================
import json
import os
import resource
import statistics as st
import time

import von

SHORT = ("Ticket #48213: our invoice PDF downloads as a 0-byte file, "
         "and the customer needs it for month-end close today.")
LONG = ("Support thread (12 messages). The customer upgraded to the enterprise plan "
        "last quarter and reports that since the 3.4.1 release their nightly export job "
        "fails with a 502 from the billing gateway. ") * 8

ALL8 = ["billing", "bug_report", "feature_request", "account_access",
        "shipping", "security", "compliance", "other"]
INSTR = "Route this ticket to the right queue."

t = time.perf_counter()
von.decide(SHORT, ALL8[:2], INSTR)                     # 预热 = 加载权重
load = time.perf_counter() - t
print(f"[load] {load:.1f}s", flush=True)

out = {"_meta": {"load_s": round(load, 1), "device": os.environ.get("VON_DEVICE", "auto")}}

def bench(name, state, choices, instr=INSTR, n=12):
    von.decide(state, choices, instr)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = von.decide(state, choices, instr)
        ts.append((time.perf_counter() - t0) * 1000)
    p50, p90 = st.median(ts), sorted(ts)[int(0.9 * len(ts)) - 1]
    out[name] = {"p50_ms": round(p50, 1), "p90_ms": round(p90, 1),
                 "n_tok": len(state.split()), "answer": str(r)[:160]}
    print(f"  {name:16s} p50={p50:7.1f}ms p90={p90:7.1f}ms -> {str(r)[:80]}", flush=True)

for k in (2, 3, 4, 8):
    bench(f"choice_{k}_short", SHORT, ALL8[:k])
bench("choice_4_long", LONG, ALL8[:4])
bench("noul_short", SHORT, ALL8[:2], "Does this ticket require a refund?", n=10)

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
out["_meta"]["peak_rss_mb"] = round(rss, 0)
print(f"[meta] peak_rss={rss:.0f}MB")

json.dump(out, open("von-bench.json", "w"), indent=2, ensure_ascii=False)
print("[done] -> von-bench.json")
