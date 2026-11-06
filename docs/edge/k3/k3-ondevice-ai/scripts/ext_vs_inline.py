"""定位 SpaceMIT EP 的段错误：external data 还是模型规模。

build: python3 ext_vs_inline.py build <layers> <inline|ext>
run:   python3 ext_vs_inline.py run <path> <ep|cpu>
"""
import os
import sys
import time

import numpy as np
import onnx
from onnx import TensorProto, helper

D, Q, KV, FF = 1024, 2048, 1024, 3072
DIR = "/home/pingji/synth"
SHAPES = dict(q=(D, Q), k=(D, KV), v=(D, KV), o=(Q, D), g=(D, FF), u=(D, FF), d=(FF, D))


def build(L, mode, path, data):
    nodes, inits, offs = [], [], {}
    off = 0
    cur = "x"
    if mode == "ext":
        total = sum(k * n * 4 for k, n in SHAPES.values()) * L
        mm = np.memmap(data, dtype=np.uint8, mode="w+", shape=(total,))
    for i in range(L):
        for tag, (k, n) in SHAPES.items():
            nm = f"w{i}_{tag}"
            if mode == "ext":
                nbytes = k * n * 4
                row = np.full((n,), 0.01, dtype=np.float32).tobytes()
                mm[off:off + k * len(row)] = np.frombuffer(row * k, dtype=np.uint8)
                offs[nm] = (off, nbytes)
                off += nbytes
                t = TensorProto()
                t.name = nm
                t.data_type = TensorProto.FLOAT
                t.dims.extend([k, n])
                t.data_location = TensorProto.EXTERNAL
                for key, val in (("location", os.path.basename(data)),
                                 ("offset", str(offs[nm][0])), ("length", str(nbytes))):
                    e = t.external_data.add()
                    e.key = key
                    e.value = val
                inits.append(t)
            else:
                v = np.full((k, n), 0.01, dtype=np.float32).ravel()
                inits.append(helper.make_tensor(nm, TensorProto.FLOAT, [k, n], v))
                del v
        p = f"l{i}_"
        nodes += [
            helper.make_node("MatMul", [cur, f"w{i}_q"], [p + "q"]),
            helper.make_node("MatMul", [cur, f"w{i}_k"], [p + "k"]),
            helper.make_node("MatMul", [cur, f"w{i}_v"], [p + "v"]),
            helper.make_node("MatMul", [p + "q", f"w{i}_o"], [p + "o"]),
            helper.make_node("MatMul", [cur, f"w{i}_g"], [p + "g"]),
            helper.make_node("MatMul", [cur, f"w{i}_u"], [p + "u"]),
            helper.make_node("Mul", [p + "g", p + "u"], [p + "gu"]),
            helper.make_node("MatMul", [p + "gu", f"w{i}_d"], [p + "d"]),
            helper.make_node("Add", [p + "o", p + "d"], [p + "r1"]),
            helper.make_node("Add", [p + "r1", p + "k"], [p + "r2"]),
            helper.make_node("Add", [p + "r2", p + "v"], [p + "r3"]),
        ]
        cur = p + "r3"
    if mode == "ext":
        mm.flush()
        del mm
    nodes.append(helper.make_node("Identity", [cur], ["y"]))
    g = helper.make_graph(nodes, "t",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, D])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, D])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 9
    onnx.save_model(m, path)
    tot = 0 if mode == "ext" else sum(k * n * 4 for k, n in SHAPES.values()) * L
    print(f"BUILD_OK {path} layers={L} mode={mode} inline_weights={tot/1e6:.0f} MB "
          f"proto={os.path.getsize(path)/1e6:.1f} MB", flush=True)


def run(path, prov_tag):
    import onnxruntime as ort
    if prov_tag == "ep":
        import spacemit_ort  # noqa
        prov = ["SpaceMITExecutionProvider", "CPUExecutionProvider"]
    else:
        prov = ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.intra_op_num_threads = 16
    t = time.time()
    s = ort.InferenceSession(path, so, providers=prov)
    load = time.time() - t
    x = np.full((1, D), 0.01, dtype=np.float32)
    s.run(None, {"x": x})
    t = time.time()
    for _ in range(5):
        s.run(None, {"x": x})
    dt = (time.time() - t) / 5
    print(f"RUN_OK {prov_tag} load={load:.1f}s {dt*1000:.1f} ms", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        L = int(sys.argv[2])
        mode = sys.argv[3]
        p = f"{DIR}/t_{L}_{mode}.onnx"
        build(L, mode, p, f"{DIR}/t_{L}_{mode}.bin")
    else:
        run(sys.argv[2], sys.argv[3])
