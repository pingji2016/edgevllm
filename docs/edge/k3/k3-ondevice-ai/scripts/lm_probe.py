"""EP 对超大 N 的 MatMul 是否段错误；切片是否可绕过。

build lm  -> 单个 [1024,151936] MatMul（622 MB）
build lmw -> 同权重切成 19 个 8192 列的 MatMul + Concat
run   <path> [cpu|ep]
"""
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

D, V = 1024, 151936
DIR = "/home/pingji/synth"
BIN = f"{DIR}/lm_w.bin"
CH = 8192


def write_bin():
    mm = np.memmap(BIN, dtype=np.uint8, mode="w+", shape=(D * V * 4,))
    row = np.full((V,), 0.01, dtype=np.float32).tobytes()
    for i in range(D):
        mm[i * len(row):(i + 1) * len(row)] = np.frombuffer(row, dtype=np.uint8)
    mm.flush()
    del mm


def ext(name, k, n, off):
    t = TensorProto()
    t.name = name
    t.data_type = TensorProto.FLOAT
    t.dims.extend([k, n])
    t.data_location = TensorProto.EXTERNAL
    for key, val in (("location", os.path.basename(BIN)), ("offset", str(off)),
                     ("length", str(k * n * 4))):
        e = t.external_data.add()
        e.key = key
        e.value = val
    return t


def build(mode):
    if not os.path.exists(BIN) or os.path.getsize(BIN) != D * V * 4:
        write_bin()
    nodes, inits = [], []
    if mode == "lm":
        inits.append(ext("w_lm", D, V, 0))
        nodes.append(helper.make_node("MatMul", ["x", "w_lm"], ["y"]))
        out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, V])
    else:
        parts = []
        for i, off in enumerate(range(0, V, CH)):
            n = min(CH, V - off)
            nm = f"w{i}"
            inits.append(ext(nm, D, n, off * 4))
            nodes.append(helper.make_node("MatMul", ["x", nm], [f"y{i}"]))
            parts.append(f"y{i}")
        nodes.append(helper.make_node("Concat", parts, ["y"], axis=1))
        out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, V])
    g = helper.make_graph(nodes, "lm",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, D])],
                          [out], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 9
    p = f"{DIR}/lm_{mode}.onnx"
    onnx.save_model(m, p)
    print(f"BUILD_OK {p} nodes={len(nodes)} 列块={CH if mode=='lmw' else V}", flush=True)


def run(path, tag):
    import onnxruntime as ort
    if tag == "ep":
        import spacemit_ort  # noqa
        prov = ["SpaceMITExecutionProvider", "CPUExecutionProvider"]
    else:
        prov = ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.intra_op_num_threads = 16
    s = ort.InferenceSession(path, so, providers=prov)
    x = np.full((1, D), 0.01, dtype=np.float32)
    s.run(None, {"x": x})
    import time
    t = time.time()
    for _ in range(5):
        s.run(None, {"x": x})
    dt = (time.time() - t) / 5
    print(f"RUN_OK {tag} {dt*1000:.1f} ms  {D*V*4/dt/1e9:.1f} GB/s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2])
    else:
        run(sys.argv[2], sys.argv[3])
