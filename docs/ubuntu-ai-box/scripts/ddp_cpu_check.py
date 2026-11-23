"""CPU 双进程 DDP 正确性验证（gloo 后端，零成本）

在真实 MiniMind 模型（hidden_size=512, 8 层，与 pretrain_512 同构）上验证：
  A. 梯度一致性：DDP 同步后的梯度 == 单进程 4 样本批次的梯度（数值等价）
  B. 步进同步：3 次 optimizer step 后两进程参数完全一致
  C. 数据分片：两进程各自看到不同样本，loss 不同
  D. 反证：不用 DDP 时参数必然分叉（证明检测有效）

用法（不需要把脚本放进 minimind 仓库）：
  MINIMIND_HOME=~/apps/minimind \
  OMP_NUM_THREADS=4 torchrun --nproc_per_node 2 --standalone ddp_cpu_check.py

前置条件：minimind 的 train_pretrain.py 已按正文 6.3 节打过补丁
（或本地实验时手动把 init_process_group 的 backend 换成 gloo）。
"""
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

MINIMIND_HOME = os.environ.get("MINIMIND_HOME", os.path.expanduser("~/apps/minimind"))
sys.path.insert(0, MINIMIND_HOME)
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402

RANK = int(os.environ.get("RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))
MAIN = RANK == 0
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))

B, S, ACC = 2, 128, 2
VOCAB = 6400


def log(*a):
    if MAIN:
        print(*a, flush=True)


def build_cfg():
    return MiniMindConfig(hidden_size=512, num_hidden_layers=8)


def build_model(seed=1234):
    torch.manual_seed(seed)  # 各 rank 相同种子 → 起点完全一致
    return MiniMindForCausalLM(build_cfg())


def make_batch(seed, b=B, s=S):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (b, s), generator=g)
    return ids, ids.clone()


def grad_max_diff(module):
    """所有 rank 间对每个参数梯度做 AVG allreduce，返回 (自身 - 均值) 的最大绝对差"""
    worst = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        v = p.grad.detach().float().clone()
        dist.all_reduce(v, op=dist.ReduceOp.AVG)
        d = (p.grad.detach().float() - v).abs().max().item()
        if d > worst:
            worst = d
    return worst


def param_max_diff(module):
    worst = 0.0
    for p in module.parameters():
        v = p.detach().float().clone()
        dist.all_reduce(v, op=dist.ReduceOp.AVG)
        d = (p.detach().float() - v).abs().max().item()
        if d > worst:
            worst = d
    return worst


def loss_of(res):
    loss = res.loss
    if getattr(res, "aux_loss", None) is not None:
        loss = loss + res.aux_loss
    return loss


def main():
    dist.init_process_group(backend="gloo")
    world = dist.get_world_size()
    results = {}

    cfg = build_cfg()
    n_params = sum(p.numel() for p in build_model().parameters())
    results["params"] = n_params
    results["world"] = world

    # ---------------- 测试 A：梯度数值等价性 ----------------
    model = build_model()
    ddp = DDP(model)
    ddp.train()

    ids, labels = make_batch(100 + RANK * 100)
    t0 = time.time()
    loss_a = loss_of(ddp(ids, labels=labels))
    loss_a.backward()
    t_a = time.time() - t0

    d_sync = grad_max_diff(model)

    # 单进程参考：把 2 个 rank 的 4 条样本拼成一个 batch=4 的梯度
    d_equiv = float("nan")
    if MAIN:
        ref = build_model()
        ref.train()
        ids_all = torch.cat([make_batch(100)[0], make_batch(200)[0]], dim=0)
        ref_loss = loss_of(ref(ids_all, labels=ids_all.clone()))
        ref_loss.backward()
        d_equiv = max(
            (rp.grad.float() - dp.grad.float()).abs().max().item()
            for rp, dp in zip(ref.parameters(), model.parameters())
            if rp.grad is not None
        )
        results["ref_loss"] = ref_loss.item()
    results["grad_sync_max_diff"] = d_sync
    results["grad_vs_single_proc_max_diff"] = d_equiv
    results["loss_rank"] = loss_a.item()
    results["step_time_s"] = t_a
    del ddp, model

    # ---------------- 测试 B：3 步参数同步 ----------------
    model = build_model()
    ddp = DDP(model)
    ddp.train()
    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-3)

    step_diffs, step_losses, step_times = [], [], []
    for step in range(3):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        for micro in range(ACC):
            bi, bl = make_batch(900 + step * 100 + micro * 7 + RANK * 13)
            loss = loss_of(ddp(bi, labels=bl)) / ACC
            loss.backward()
        torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        opt.step()
        step_times.append(time.time() - t0)
        step_losses.append(loss.item() * ACC)
        step_diffs.append(param_max_diff(model))

    results["step_param_diffs"] = step_diffs
    results["step_losses_rank"] = step_losses
    results["step_time_s"] = step_times
    del ddp, model, opt

    # ---------------- 测试 C：反证（不包 DDP） ----------------
    model = build_model()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(2):
        bi, bl = make_batch(500 + step * 100 + RANK * 1000)
        opt.zero_grad(set_to_none=True)
        loss_of(model(bi, labels=bl)).backward()
        opt.step()
    results["no_ddp_param_diff"] = param_max_diff(model)

    # ---------------- 汇总 ----------------
    if MAIN:
        print("\n" + "=" * 62)
        print("CPU 双进程 DDP 验证结果")
        print("=" * 62)
        print(f"world_size              : {results['world']} 进程（gloo）")
        print(f"模型参数量              : {results['params'] / 1e6:.2f} M（hidden=512, 8 层）")
        print(f"每进程 batch            : {B} × seq {S}，梯度累积 {ACC}")
        print(f"等效 batch              : {B * results['world'] * ACC} 条样本/步")
        print("-" * 62)
        print(f"A1 梯度跨进程最大差     : {results['grad_sync_max_diff']:.3e}   (应≈0)")
        print(f"A2 DDP梯度 vs 单进程    : {results['grad_vs_single_proc_max_diff']:.3e}   (应<1e-5)")
        print(f"A3 loss(本进程)         : {results['loss_rank']:.4f}")
        if "ref_loss" in results:
            print(f"A4 loss(参考批次)       : {results['ref_loss']:.4f}")
        print("-" * 62)
        for i, (d, l, t) in enumerate(
            zip(results["step_param_diffs"], results["step_losses_rank"], results["step_time_s"]), 1
        ):
            print(f"B step{i} 参数最大差        : {d:.3e}    loss={l:.4f}  {t:.2f}s")
        print("-" * 62)
        print(f"C  不用DDP时参数最大差   : {results['no_ddp_param_diff']:.3e}   (应>>0，证明检测有效)")
        print("=" * 62)
        print(f"单步耗时(CPU, 2进程)    : {sum(results['step_time_s']) / len(results['step_time_s']):.2f}s")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
