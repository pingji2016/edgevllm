"""FlexAttention + torch.compile：能否在 sm_120 上生成真正的 Triton kernel"""
import time

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

dev = "cuda"
H, D = 16, 128
fc = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def bench(fn, reps=5):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1000


print("序列长度 | SDPA(causal) | Flex编译(全因果) | Flex编译(滑窗1024) | 滑窗 vs SDPA")
for S in (2048, 4096, 8192):
    try:
        q = torch.randn(1, H, S, D, device=dev, dtype=torch.bfloat16)
        k = torch.randn(1, H, S, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(1, H, S, D, device=dev, dtype=torch.bfloat16)
        bm_full = create_block_mask(lambda b, h, qq, kk: kk <= qq, B=None, H=None,
                                    Q_LEN=S, KV_LEN=S, device=dev)
        bm_sw = create_block_mask(lambda b, h, qq, kk: (kk <= qq) & (kk > qq - 1024),
                                  B=None, H=None, Q_LEN=S, KV_LEN=S, device=dev)
        t0 = time.time()
        fc(q, k, v, block_mask=bm_full)  # 首次触发编译
        torch.cuda.synchronize()
        compile_s = time.time() - t0

        t_sdpa = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True))
        t_full = bench(lambda: fc(q, k, v, block_mask=bm_full))
        t_sw = bench(lambda: fc(q, k, v, block_mask=bm_sw))
        print(f"S={S:5d} | {t_sdpa:9.2f} ms | {t_full:12.2f} ms | {t_sw:13.2f} ms | "
              f"{t_sdpa/max(t_sw,1e-9):6.2f}×  (首次编译 {compile_s:.0f}s)")
        del q, k, v, bm_full, bm_sw
        torch.cuda.empty_cache()
    except torch.OutOfMemoryError:
        print(f"S={S:5d} | ❌ OOM")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"S={S:5d} | ❌ {type(e).__name__}: {str(e)[:90]}")
        torch.cuda.empty_cache()
