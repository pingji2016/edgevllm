"""最小判别：EP 何时段错误。

A: x -> MatMul(1024x8192)  -> 输出           （1 个节点）
B: x -> MatMul(1024x1024) -> MatMul(1024x8192) -> 输出  （2 个节点串联）
C: x -> MatMul(1024x1024) -> MatMul(1024x1024) -> 输出  （2 个节点，头部 N=1024）
D: x -> Add(x,x) -> MatMul(1024x8192) -> 输出 （非 MatMul 输入）
用法: python3 ep_probe2.py <A|B|C|D>
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

D, BIG = 1024, 8192
V = sys.argv[1]
nodes, inits = [], []


def w(name, k, n):
    inits.append(helper.make_tensor(name, TensorProto.FLOAT, [k, n],
                                    np.full((k, n), 0.01, dtype=np.float32).ravel()))


if V == "A":
    w("w1", D, BIG)
    nodes = [helper.make_node("MatMul", ["x", "w1"], ["y"])]
    out_shape = [1, BIG]
elif V == "B":
    w("w0", D, D)
    w("w1", D, BIG)
    nodes = [helper.make_node("MatMul", ["x", "w0"], ["h"]),
             helper.make_node("MatMul", ["h", "w1"], ["y"])]
    out_shape = [1, BIG]
elif V == "C":
    w("w0", D, D)
    w("w1", D, D)
    nodes = [helper.make_node("MatMul", ["x", "w0"], ["h"]),
             helper.make_node("MatMul", ["h", "w1"], ["y"])]
    out_shape = [1, D]
elif V == "D":
    w("w1", D, BIG)
    nodes = [helper.make_node("Add", ["x", "x"], ["h"]),
             helper.make_node("MatMul", ["h", "w1"], ["y"])]
    out_shape = [1, BIG]

g = helper.make_graph(nodes, "p",
                      [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, D])],
                      [helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)], inits)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 9
p = f"/home/pingji/synth/probe_{V}.onnx"
onnx.save_model(m, p)

import spacemit_ort  # noqa
import onnxruntime as ort
s = ort.InferenceSession(p, providers=["SpaceMITExecutionProvider", "CPUExecutionProvider"])
x = np.full((1, D), 0.01, dtype=np.float32)
y = s.run(None, {"x": x})[0]
print(f"PROBE_{V}_OK out={y.shape}", flush=True)
