"""roofline 被测程序：三个代表性 kernel
  A. bf16 GEMM 4096³        （应跑在算力墙上）
  B. 1 GB 级 elementwise copy（应跑在带宽墙上）
  C. SDPA attention 8K      （混合）
用法: python roof_kernels.py <case>
"""
import sys

import torch

case = sys.argv[1]

if case == "gemm":
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        c = a @ b
elif case == "copy":
    n = 512 * 1024 * 1024  # 512M 个 bf16 = 1 GB
    a = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    b = torch.empty_like(a)
    for _ in range(5):
        b.copy_(a)
elif case == "add":
    n = 256 * 1024 * 1024  # 256M × 3 × 2B = 1.5 GB 流量
    a = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    c = torch.empty_like(a)
    for _ in range(5):
        torch.add(a, b, out=c)
elif case == "attn":
    q = torch.randn(1, 16, 8192, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 16, 8192, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 16, 8192, 128, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
elif case == "rmsnorm":
    x = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(4096, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        var = x.pow(2).mean(-1, keepdim=True)
        y = x * torch.rsqrt(var + 1e-6) * w

torch.cuda.synchronize()
print("done", case)
