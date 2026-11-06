"""把 nano-vllm 改成能在纯 CPU（RISC-V）上跑。

- attention.py 整体重写：triton/flash-attn -> torch eager SDPA（paged KV 手工 gather）
  第二轮优化：decode 一次 gather 所有序列 + enable_gqa（省掉 repeat_interleave）
- model_runner.py：nccl->gloo、cuda->cpu、去掉 cudagraph、KV cache 按内存预算分配
- 去掉所有 @torch.compile（CPU 上 inductor 编译极慢且无 triton）
"""
import io
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/nano-vllm"

ATTENTION = '''import os
import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.utils.context import get_context


def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    """把 key/value 写进 paged KV cache。k_cache: [num_blocks, block_size, nkv, hd]"""
    if slot_mapping is None or slot_mapping.numel() == 0:
        return
    nkv, hd = key.shape[1], key.shape[2]
    k = k_cache.view(-1, nkv, hd)
    v = v_cache.view(-1, nkv, hd)
    slots = slot_mapping.to(torch.long)
    valid = slots >= 0
    if bool(valid.all()):
        k.index_copy_(0, slots, key)
        v.index_copy_(0, slots, value)
    else:
        k.index_copy_(0, slots[valid], key[valid])
        v.index_copy_(0, slots[valid], value[valid])


def _attend(q, k, v, scale, causal=True, offset=0):
    """q:[lq,nh,d] k/v:[lk,nkv,d] -> [lq,nh,d]，K/V 头数不同时用 enable_gqa 免拷贝"""
    nh, nkv = q.size(1), k.size(1)
    qh = q.transpose(0, 1).unsqueeze(0)
    kh = k.transpose(0, 1).unsqueeze(0)
    vh = v.transpose(0, 1).unsqueeze(0)
    gqa = nh != nkv
    if causal and offset > 0:
        lq, lk = qh.size(2), kh.size(2)
        qpos = torch.arange(lq).view(-1, 1) + offset
        kpos = torch.arange(lk).view(1, -1)
        mask = (kpos <= qpos).unsqueeze(0).unsqueeze(0)
        o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask,
                                           scale=scale, enable_gqa=gqa)
    else:
        o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=causal,
                                           scale=scale, enable_gqa=gqa)
    return o.squeeze(0).transpose(0, 1)


class Attention(nn.Module):

    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q, k, v):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        nkv, hd = self.num_kv_heads, self.head_dim

        if context.is_prefill:
            outs = []
            cu_q = context.cu_seqlens_q.tolist()
            cu_k = context.cu_seqlens_k.tolist()
            for i in range(len(cu_q) - 1):
                qs, qe = cu_q[i], cu_q[i + 1]
                ks, ke = cu_k[i], cu_k[i + 1]
                if context.block_tables is not None:
                    bt = context.block_tables[i]
                    bt = bt[bt >= 0].to(torch.long)
                    kk = k_cache.index_select(0, bt).reshape(-1, nkv, hd)[:ke]
                    vv = v_cache.index_select(0, bt).reshape(-1, nkv, hd)[:ke]
                else:
                    kk, vv = k[ks:ke], v[ks:ke]
                outs.append(_attend(q[qs:qe], kk, vv, self.scale, True, offset=ke - (qe - qs)))
            return torch.cat(outs, 0) if outs else q.new_empty((0, q.size(1), q.size(2)))

        # decode：所有序列一次性 gather + 一次批量 SDPA（bs=1 走无 mask 快路径）
        bs = q.size(0)
        mode = os.environ.get("ATT_MODE", "full")
        if mode == "noattn":       # 拆解实验：跳过 attention，只量线性层
            return torch.zeros_like(q)
        clens = context.context_lens
        bt = context.block_tables
        L = int(clens.max())
        if mode == "short":        # 拆解实验：只读 1 个 block（256 token）
            L = min(L, 256)
        nb = (L + k_cache.size(1) - 1) // k_cache.size(1)
        bt = bt[:, :nb]                                  # 只取用到的块
        idx = torch.where(bt >= 0, bt, torch.zeros_like(bt)).to(torch.long)
        kk = k_cache.index_select(0, idx.reshape(-1)).reshape(bs, -1, nkv, hd)[:, :L]
        vv = v_cache.index_select(0, idx.reshape(-1)).reshape(bs, -1, nkv, hd)[:, :L]
        if bs == 1:
            return _attend(q, kk[0], vv[0], self.scale, False)
        keep = torch.arange(L, device=q.device).view(1, -1) < clens.view(-1, 1)
        qh = q.unsqueeze(2)                              # [bs,nh,1,hd]
        o = F.scaled_dot_product_attention(
            qh, kk.permute(0, 2, 1, 3), vv.permute(0, 2, 1, 3),
            attn_mask=keep.view(bs, 1, 1, L), scale=self.scale,
            enable_gqa=(q.size(1) != nkv))
        return o.squeeze(2)
'''


def patch_model_runner(path):
    s = open(path).read()
    reps = [
        ('dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)',
         'dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29533", world_size=self.world_size, rank=rank)'),
        ("        torch.cuda.set_device(rank)\n",
         "        if torch.cuda.is_available():\n            torch.cuda.set_device(rank)\n"),
        ('torch.set_default_device("cuda")', 'torch.set_default_device("cpu")'),
        ("torch.set_default_dtype(hf_config.dtype)", "torch.set_default_dtype(torch.float32)"),
        ("self.enforce_eager = config.enforce_eager", "self.enforce_eager = True  # CPU 版强制 eager"),
        ("seq_len = min(max_num_batched_tokens, max_model_len)",
         "seq_len = min(128, max_model_len)"),
        ("        free, total = torch.cuda.mem_get_info()\n"
         "        used = total - free\n"
         '        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]\n'
         '        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]\n',
         "        _mem = {}\n"
         "        for _l in open('/proc/meminfo'):\n"
         "            if ':' in _l:\n"
         "                _k, _v = _l.split(':')\n"
         "                _mem[_k] = int(_v.split()[0]) * 1024\n"
         "        total = _mem.get('MemTotal', 8 << 30)\n"
         "        avail = _mem.get('MemAvailable', 4 << 30)\n"
         "        used = peak = current = 0\n"
         "        budget = min(int(avail * config.gpu_memory_utilization), 2 << 30)\n"),
        ("config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes",
         "config.num_kvcache_blocks = config.num_kvcache_blocks if config.num_kvcache_blocks > 0 else budget // block_bytes"),
        ("        torch.cuda.synchronize()\n        dist.destroy_process_group()",
         "        dist.destroy_process_group()"),
        ("        torch.cuda.empty_cache()\n        torch.cuda.reset_peak_memory_stats()",
         "        pass"),
        # 未用到的 KV 位置是未初始化内存，NaN/inf 会污染 padded 的 SDPA
        ("self.kv_cache = torch.empty(2,", "self.kv_cache = torch.zeros(2,"),
    ]
    for old, new in reps:
        if old not in s:
            print(f"  !! 未找到: {old[:60]}", flush=True)
        s = s.replace(old, new)
    s = s.replace("torch.cuda.empty_cache()", "pass  # cpu")
    s = s.replace(", pin_memory=True).cuda(non_blocking=True)", ")")
    open(path, "w").write(s)


LAYERNORM = '''import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    # 注意：原实现用 in-place（x.float().add_()/mul_()）。当 activations 是
    # float32 时 x.float() 返回自身，residual 与归一化结果会指向同一块内存，
    # 结果直接算错（bf16 下才会产生副本，所以 GPU 上没暴露）。
    def rms_forward(self, x):
        orig = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        out = xf * torch.rsqrt(var + self.eps)
        return (out * self.weight).to(orig)

    def add_rms_forward(self, x, residual):
        orig = x.dtype
        xf = x.float() + residual.float()
        res = xf.to(orig)
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        out = xf * torch.rsqrt(var + self.eps)
        return (out * self.weight).to(orig), res

    def forward(self, x, residual: torch.Tensor | None = None):
        if residual is None:
            return self.rms_forward(x)
        return self.add_rms_forward(x, residual)
'''


def patch_layernorm(path):
    open(path, "w").write(LAYERNORM)


SAMPLER = '''import torch
from torch import nn


class Sampler(nn.Module):

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 不用 inplace：run() 在 inference_mode 外调用，inplace 会报错
        logits = logits.float() / temperatures.unsqueeze(dim=1)
        probs = torch.softmax(logits, dim=-1)
        noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        return (probs / noise).argmax(dim=-1)
'''


def patch_sampler(path):
    open(path, "w").write(SAMPLER)


XXHASH_FALLBACK = '''try:
    import xxhash
except ImportError:  # 板上没有 xxhash，用标准库替代（只用于 prefix cache 的块哈希）
    import hashlib

    class _Xxh64:
        def __init__(self):
            self._h = hashlib.blake2b(digest_size=8)

        def update(self, b):
            self._h.update(b)
            return self

        def intdigest(self):
            return int.from_bytes(self._h.digest(), "little")

    class _Xxhash:
        @staticmethod
        def xxh64():
            return _Xxh64()

    xxhash = _Xxhash()
'''


def patch_block_manager(path):
    s = open(path).read()
    s = s.replace("import xxhash\n", XXHASH_FALLBACK, 1)
    open(path, "w").write(s)


def main():
    att = os.path.join(ROOT, "nanovllm/layers/attention.py")
    open(att, "w").write(ATTENTION)
    print("  attention.py 重写完成（decode 批量 gather + enable_gqa）", flush=True)

    patch_model_runner(os.path.join(ROOT, "nanovllm/engine/model_runner.py"))
    print("  model_runner.py 补丁完成", flush=True)

    patch_block_manager(os.path.join(ROOT, "nanovllm/engine/block_manager.py"))
    print("  block_manager.py: xxhash -> hashlib 回退", flush=True)

    patch_sampler(os.path.join(ROOT, "nanovllm/layers/sampler.py"))
    print("  sampler.py: 去掉 inplace", flush=True)

    patch_layernorm(os.path.join(ROOT, "nanovllm/layers/layernorm.py"))
    print("  layernorm.py: 修 fp32 residual 别名 bug", flush=True)

    p = os.path.join(ROOT, "nanovllm/engine/llm_engine.py")
    s = open(p).read()
    s = s.replace('    def exit(self):\n        self.model_runner.call("exit")',
                  '    def exit(self):\n        if not hasattr(self, "model_runner"):\n            return\n        self.model_runner.call("exit")')
    open(p, "w").write(s)

    n = 0
    for dirpath, _, files in os.walk(os.path.join(ROOT, "nanovllm")):
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(dirpath, f)
            s = open(p).read()
            if "@torch.compile" in s:
                s = s.replace("    @torch.compile\n", "")
                open(p, "w").write(s)
                n += 1
    print(f"  去掉 @torch.compile: {n} 个文件", flush=True)

    import py_compile
    for dirpath, _, files in os.walk(os.path.join(ROOT, "nanovllm")):
        for f in files:
            if f.endswith(".py"):
                py_compile.compile(os.path.join(dirpath, f), doraise=True)
    print("  语法检查通过", flush=True)


main()
