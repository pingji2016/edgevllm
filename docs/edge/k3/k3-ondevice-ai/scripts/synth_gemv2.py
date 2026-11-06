"""冷权重 GEMV 链 v2（内存安全版）：模拟 Qwen3-0.6B decode 的真实权重流量。

与 v1 的区别：权重**直接流式写入 external data 文件**，ONNX proto 里只留
offset/length 元数据（几 KB），python 侧峰值内存 < 100 MB。
v1 把 2.25 GB 全拼在 proto 里，峰值 5–7 GB，把板子压挂了。

每层参数形状与 Qwen3-0.6B 一致（14.6M/层），28 层 ≈ 1.63 GB，
再加 lm_head（1024×151936，622 MB），合计 ≈ 2.25 GB/token。
每层权重互不相同 -> 一次前向必然把 2.25 GB 从 DRAM 读一遍，无法靠缓存作弊。

用法：
    python3 synth_gemv2.py            # 全部 28 层 + lm_head
    python3 synth_gemv2.py 3          # 只 3 层（先验证流程，约 240 MB）
    python3 synth_gemv2.py 28 --no-lm
"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort
import spacemit_ort  # 必须在 onnxruntime 之后 import，否则 session 创建时段错误
from onnx import TensorProto, helper
from onnx.external_data_helper import set_external_data

D, Q, KV, FF = 1024, 2048, 1024, 3072
L = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 28
LM = "--no-lm" not in sys.argv
LMV = int(os.environ.get("LMV", "151936"))

DIR = "/home/pingji/synth"
DATA = os.path.join(DIR, "w2.bin")
MODEL = os.path.join(DIR, "synth2.onnx")
CHUNK = 4 << 20  # 每次写 4 MB，避免大分配


def specs():
    s = []
    for i in range(L):
        s += [(f"w{i}_q", D, Q), (f"w{i}_k", D, KV), (f"w{i}_v", D, KV),
              (f"w{i}_o", Q, D), (f"w{i}_g", D, FF), (f"w{i}_u", D, FF),
              (f"w{i}_d", FF, D)]
    if LM:
        s.append(("w_lm", D, LMV))
    return s


def write_weights(sp):
    """流式写 external data，返回 {name: (offset, nbytes)}"""
    os.makedirs(DIR, exist_ok=True)
    total = sum(k * n * 4 for _, k, n in sp)
    if os.path.exists(DATA) and os.path.getsize(DATA) == total:
        off, offs = 0, {}
        for name, k, n in sp:
            nbytes = k * n * 4
            offs[name] = (off, nbytes)
            off += nbytes
        print(f"[data] 复用已存在 {total/1e9:.2f} GB", flush=True)
        return offs, total
    mm = np.memmap(DATA, dtype=np.uint8, mode="w+", shape=(total,))
    off, offs = 0, {}
    for name, k, n in sp:
        nbytes = k * n * 4
        row = np.full((n,), 0.01, dtype=np.float32).tobytes()
        rows_per_chunk = max(1, CHUNK // len(row))
        done = 0
        while done < k:
            r = min(rows_per_chunk, k - done)
            mm[off + done * len(row): off + (done + r) * len(row)] = \
                np.frombuffer(row * r, dtype=np.uint8)
            done += r
        offs[name] = (off, nbytes)
        off += nbytes
    mm.flush()
    del mm
    print(f"[data] {total/1e9:.2f} GB -> {DATA}", flush=True)
    return offs, total


def ext_tensor(name, k, n, offs):
    """手工构造 external-data initializer（无 raw_data，save/load 时按 offset 读）"""
    t = TensorProto()
    t.name = name
    t.data_type = TensorProto.FLOAT
    t.dims.extend([k, n])
    t.data_location = TensorProto.EXTERNAL
    o, ln = offs[name]
    for key, val in (("location", os.path.basename(DATA)),
                     ("offset", str(o)), ("length", str(ln))):
        e = t.external_data.add()
        e.key = key
        e.value = val
    return t


def build(sp, offs):
    nodes, inits = [], []
    cur = "x"
    for i in range(L):
        names = {}
        for tag, k, n in (("q", D, Q), ("k", D, KV), ("v", D, KV), ("o", Q, D),
                          ("g", D, FF), ("u", D, FF), ("d", FF, D)):
            nm = f"w{i}_{tag}"
            inits.append(ext_tensor(nm, k, n, offs))
            names[tag] = nm
        p = f"l{i}_"
        nodes += [
            helper.make_node("MatMul", [cur, names["q"]], [p + "q"]),
            helper.make_node("MatMul", [cur, names["k"]], [p + "k"]),
            helper.make_node("MatMul", [cur, names["v"]], [p + "v"]),
            helper.make_node("MatMul", [p + "q", names["o"]], [p + "o"]),
            helper.make_node("MatMul", [cur, names["g"]], [p + "g"]),
            helper.make_node("MatMul", [cur, names["u"]], [p + "u"]),
            helper.make_node("Mul", [p + "g", p + "u"], [p + "gu"]),
            helper.make_node("MatMul", [p + "gu", names["d"]], [p + "d"]),
            helper.make_node("Add", [p + "o", p + "d"], [p + "r1"]),
            helper.make_node("Add", [p + "r1", p + "k"], [p + "r2"]),
            helper.make_node("Add", [p + "r2", p + "v"], [p + "r3"]),
        ]
        cur = p + "r3"
    if LM:
        inits.append(ext_tensor("w_lm", D, LMV, offs))
        nodes.append(helper.make_node("MatMul", [cur, "w_lm"], ["logits"]))
        outs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, LMV])]
    else:
        nodes.append(helper.make_node("Identity", [cur], ["hidden"]))
        outs = [helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, D])]

    g = helper.make_graph(nodes, "synth2",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, D])],
                          outs, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 9
    onnx.save_model(m, MODEL)
    print(f"[graph] {len(nodes)} nodes -> {MODEL} "
          f"({os.path.getsize(MODEL)/1e6:.1f} MB)", flush=True)


def run(total):
    x = np.full((1, D), 0.01, dtype=np.float32)
    for tag, prov in (("EP", ["SpaceMITExecutionProvider", "CPUExecutionProvider"]),
                      ("CPU", ["CPUExecutionProvider"])):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 16
        so.enable_profiling = True
        try:
            t = time.time()
            s = ort.InferenceSession(MODEL, so, providers=prov)
            load = time.time() - t
        except Exception as e:
            print(f"[{tag}] 初始化失败: {str(e)[:200]}", flush=True)
            continue
        for _ in range(1):
            s.run(None, {"x": x})
        t = time.time()
        n = 5
        for _ in range(n):
            s.run(None, {"x": x})
        dt = (time.time() - t) / n
        c = Counter()
        for ev in json.load(open(s.end_profiling())):
            if ev.get("cat") == "Node":
                c[ev.get("args", {}).get("provider", "?")] += 1
        gbs = total / dt / 1e9
        print(f"[{tag}] load={load:.1f}s  {dt*1000:6.0f} ms  带宽 {gbs:5.1f} GB/s  "
              f"等效 decode {1/dt:5.2f} tok/s  nodes={dict(c)}", flush=True)


if __name__ == "__main__":
    sp = specs()
    offs, total = write_weights(sp)
    build(sp, offs)
    print(f"[cfg] {L} 层, lm_head={LM}, 每 token 权重 {total/1e9:.2f} GB", flush=True)
    run(total)
