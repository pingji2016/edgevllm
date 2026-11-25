"""torch 侧实测：每个 op 的真实吞吐（TFLOPS / GB/s）"""
import json
import os

import torch


def bench(fn, reps=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps  # ms


def main():
    dev = "cuda"
    rows = []

    # 1) GEMM 4096 / 8192
    for n in (4096, 8192):
        a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
        b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
        ms = bench(lambda: a @ b, reps=10)
        tflops = 2 * n ** 3 / (ms / 1000) / 1e12
        rows.append(dict(op=f"GEMM {n}³ bf16", ms=round(ms, 2), metric=f"{tflops:.1f} TFLOPS",
                         kind="compute"))
        print(f"  GEMM {n}³      {ms:8.2f} ms | {tflops:6.1f} TFLOPS", flush=True)

    # 2) 带宽：add (读2写1)
    n = 256 * 1024 * 1024
    a = torch.randn(n, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, device=dev, dtype=torch.bfloat16)
    c = torch.empty_like(a)
    ms = bench(lambda: torch.add(a, b, out=c))
    traffic = n * 2 * 3 / 1e9  # GB
    gbs = traffic / (ms / 1000)
    rows.append(dict(op="add 1.5 GB 流量", ms=round(ms, 2), metric=f"{gbs:.0f} GB/s", kind="bandwidth"))
    print(f"  add           {ms:8.2f} ms | {gbs:6.0f} GB/s", flush=True)

    # 3) copy (读1写1)
    ms = bench(lambda: c.copy_(a))
    traffic = n * 2 * 2 / 1e9
    gbs = traffic / (ms / 1000)
    rows.append(dict(op="copy 1 GB 流量", ms=round(ms, 2), metric=f"{gbs:.0f} GB/s", kind="bandwidth"))
    print(f"  copy          {ms:8.2f} ms | {gbs:6.0f} GB/s", flush=True)

    # 4) attention 8K
    q = torch.randn(1, 16, 8192, 128, device=dev, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    ms = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True), reps=10)
    # 因果注意力 FLOPs ≈ 2 * 2 * B*H*(S²/2)*D
    flops = 2 * 2 * 1 * 16 * (8192 ** 2 / 2) * 128
    tflops = flops / (ms / 1000) / 1e12
    rows.append(dict(op="SDPA causal 8K", ms=round(ms, 2), metric=f"{tflops:.1f} TFLOPS", kind="attention"))
    print(f"  SDPA 8K       {ms:8.2f} ms | {tflops:6.1f} TFLOPS", flush=True)

    # 5) RMSNorm (读1写1)
    x = torch.randn(8192, 4096, device=dev, dtype=torch.bfloat16)
    w = torch.ones(4096, device=dev, dtype=torch.bfloat16)

    def rms():
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + 1e-6) * w

    ms = bench(rms)
    traffic = x.numel() * 2 * 3 / 1e9  # 近似：读 x 两次 + 写 y
    gbs = traffic / (ms / 1000)
    rows.append(dict(op="RMSNorm 8192×4096", ms=round(ms, 2), metric=f"{gbs:.0f} GB/s", kind="bandwidth"))
    print(f"  RMSNorm       {ms:8.2f} ms | {gbs:6.0f} GB/s", flush=True)

    # 6) 短序列 decode 形态的 GEMV（batch 1）——纯带宽
    W = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    x1 = torch.randn(1, 4096, device=dev, dtype=torch.bfloat16)
    ms = bench(lambda: x1 @ W.T, reps=50)
    gbs = W.numel() * 2 / (ms / 1000) / 1e9
    rows.append(dict(op="GEMV 4096² (decode 形态)", ms=round(ms, 3), metric=f"{gbs:.0f} GB/s", kind="bandwidth"))
    print(f"  GEMV 4096²    {ms:8.3f} ms | {gbs:6.0f} GB/s", flush=True)

    json.dump(rows, open(os.path.expanduser("~/apps/infer-research/roofline/torch_ops.json"), "w"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
