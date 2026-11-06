import json
import os
import time

import numpy as np
import onnx
import spacemit_ort  # 必须先 import
import onnxruntime as ort
from onnx import TensorProto, helper

# Qwen3-0.6B 的真实线性层形状：(名字, K, N)
SHAPES = [("qkv_proj", 1024, 4096), ("o_proj", 2048, 1024),
          ("gate_up", 1024, 3072), ("down_proj", 3072, 1024)]
TOTAL = 28  # 层数


def build(path, M, K, N):
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [M, K])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [M, N])
    w = ((np.random.rand(K, N).astype(np.float32) - 0.5) / K)
    g = helper.make_graph([helper.make_node("MatMul", ["x", "wv"], ["y"])], "g", [X], [Y],
                          [helper.make_tensor("wv", TensorProto.FLOAT, [K, N], w.ravel())])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 9
    onnx.save(m, path)


def bench(path, M, K, N, iters):
    out = {}
    for tag, prov in (("EP", ["SpaceMITExecutionProvider", "CPUExecutionProvider"]),
                      ("CPU", ["CPUExecutionProvider"])):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 16
        so.enable_profiling = True
        s = ort.InferenceSession(path, so, providers=prov)
        x = np.random.rand(M, K).astype(np.float32)
        for _ in range(2):
            s.run(None, {"x": x})
        t = time.time()
        for _ in range(iters):
            s.run(None, {"x": x})
        dt = (time.time() - t) / iters
        used = set()
        for ev in json.load(open(s.end_profiling())):
            if ev.get("cat") == "Node":
                used.add(ev.get("args", {}).get("provider", "?"))
        out[tag] = (dt, used)
    return out


for M, iters in ((128, 6), (1, 12)):
    print(f"\n=== M={M} ({'prefill 形状' if M > 1 else 'decode/GEMV 形状'}) ===", flush=True)
    tot_ep = tot_cpu = 0.0
    for name, K, N in SHAPES:
        p = f"/tmp/sh_{name}_{M}.onnx"
        build(p, M, K, N)
        r = bench(p, M, K, N, iters)
        os.remove(p)
        fl = 2 * M * K * N
        ep, cpu = r["EP"][0], r["CPU"][0]
        # 单层 4 个 matmul 的耗时，×28 层 ≈ 每 token 的纯 matmul 成本
        tot_ep += ep * TOTAL
        tot_cpu += cpu * TOTAL
        print(f"  {name:<10} {M}x{K}x{N}  EP {ep*1000:6.2f} ms ({fl/ep/1e9:6.1f} GF) | "
              f"CPU {cpu*1000:6.2f} ms ({fl/cpu/1e9:6.1f} GF) | 加速 {cpu/ep:4.2f}x | "
              f"{sorted(r['EP'][1])}", flush=True)
    sp = tot_cpu / tot_ep
    print(f"  -> 28 层 matmul 合计: EP {tot_ep:.2f}s vs CPU {tot_cpu:.2f}s / token, 加速 {sp:.2f}x", flush=True)
    if M > 1:
        print(f"  -> 折算 prefill: CPU {M/tot_cpu:.1f} tok/s -> EP {M/tot_ep:.1f} tok/s", flush=True)
    else:
        print(f"  -> 折算 decode: CPU {1/tot_cpu:.2f} tok/s -> EP {1/tot_ep:.2f} tok/s", flush=True)
