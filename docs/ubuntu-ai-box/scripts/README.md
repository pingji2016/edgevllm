# 脚本与复现

本目录两个脚本对应正文 [minimind-training.md](../minimind-training.md) 第 6 节（分布式验证）。

它们**不属于 minimind 仓库**，通过 `MINIMIND_HOME` / `MINIMIND` 指向本地 minimind 目录即可，
不需要把脚本复制进去。

| 脚本 | 作用 | 需要 |
| --- | --- | --- |
| [`ddp_cpu_check.py`](ddp_cpu_check.py) | CPU 双进程验证 DDP 正确性（5 项检查） | 2 核以上 CPU，**不需要 GPU** |
| [`run_ddp_2gpu.sh`](run_ddp_2gpu.sh) | 多卡训练启动器（含 lr √N 缩放） | ≥2 张 GPU |

---

## 1. `ddp_cpu_check.py`

```bash
MINIMIND_HOME=~/apps/minimind \
OMP_NUM_THREADS=4 torchrun --nproc_per_node 2 --standalone ddp_cpu_check.py
```

本机实测输出（12 核 CPU，30.03 M 参数模型）：

```
world_size              : 2 进程（gloo）
模型参数量              : 30.03 M（hidden=512, 8 层）
每进程 batch            : 2 × seq 128，梯度累积 2
等效 batch              : 8 条样本/步
--------------------------------------------------------------
A1 梯度跨进程最大差     : 0.000e+00   (应≈0)
A2 DDP梯度 vs 单进程    : 1.118e-08   (应<1e-5)
A3 loss(本进程)         : 8.8881
A4 loss(参考批次)       : 8.8686
--------------------------------------------------------------
B step1 参数最大差        : 0.000e+00    loss=8.8446  1.22s
B step2 参数最大差        : 0.000e+00    loss=8.8834  1.16s
B step3 参数最大差        : 0.000e+00    loss=8.8534  1.07s
--------------------------------------------------------------
C  不用DDP时参数最大差   : 2.001e-03   (应>>0，证明检测有效)
==============================================================
单步耗时(CPU, 2进程)    : 1.15s
```

**怎么读这几行：**

- `A2` 是核心：DDP 的 AllReduce 求平均，数值上等于「把 2 张卡的 4 条样本拼成一个 batch=4 求梯度」。
  1e-8 的残差只是 float32 累加顺序不同。
- `B` 的三行全 0，说明经过 AdamW 更新 + 梯度裁剪 + 梯度累积之后，两个进程**没有悄悄跑偏**。
- `C` 是反证：故意不包 DDP，参数差立刻到 2e-3 —— 证明前面那些 0 不是「检测器坏了」。

---

## 2. `run_ddp_2gpu.sh`

```bash
MINIMIND=~/apps/minimind bash run_ddp_2gpu.sh            # 预训练，2 卡
MINIMIND=~/apps/minimind TASK=sft bash run_ddp_2gpu.sh   # SFT
MINIMIND=~/apps/minimind NPROC=4 BATCH=32 bash run_ddp_2gpu.sh
```

脚本会先打印等效 batch 和缩放后的 lr，再执行 `torchrun`：

| 变量 | 默认 | 说明 |
| --- | ---: | --- |
| `NPROC` | 2 | 卡数 |
| `BATCH` | 24 | **每卡** batch |
| `ACC` | 8 | 梯度累积步数 |
| `BASE_LR` | 1e-4 | 单卡 lr，脚本自动按 √NPROC 缩放 |
| `SEQ` | 512 | `--max_seq_len` |
| `TASK` | pretrain | `pretrain` / `sft` |
| `MINIMIND` | `~/apps/minimind` | minimind 仓库位置 |

---

## 3. 前置条件：训练脚本要能选 gloo

minimind 原始代码把分布式写死成 GPU，CPU 上会直接报错：

```python
dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)
args.device = f"cuda:{local_rank}"
```

改成按设备自动选择（**GPU 路径行为完全不变**）：

```python
def distributed_uses_cuda():
    """分布式是否走 GPU：无可用 GPU，或显式设置 DDP_CPU=1 时退回 CPU + gloo。"""
    return torch.cuda.is_available() and os.environ.get("DDP_CPU", "0") != "1"
```

用环境变量显式切换，而不是只靠 `torch.cuda.is_available()` —— 有显卡的机器上后者永远为真，
想跑 CPU 验证就永远切不过去。

改完用真实训练脚本验证（`pretrain_smoke200.jsonl` 是从真实语料切出来的 200 条小样本）：

```bash
head -200 ~/apps/minimind/dataset/pretrain_t2t_mini.jsonl \
  > ~/apps/minimind/dataset/pretrain_smoke200.jsonl
cd ~/apps/minimind/trainer
DDP_CPU=1 OMP_NUM_THREADS=4 ../.venv/bin/torchrun --nproc_per_node 2 --standalone \
  train_pretrain.py --device cpu --hidden_size 512 --batch_size 4 --max_seq_len 128 \
  --epochs 1 --data_path ../dataset/pretrain_smoke200.jsonl --num_workers 0 \
  --save_dir ../out_smoke
```

本机实测：200 条样本 / 25 步，loss 8.85 → 8.27，29.9 s（857 tok/s）。
同配置单卡 GPU（nccl）为 13.0 s（1,968 tok/s）—— 这组对比只说明两条路径都通，
seq 128 / batch 4 时启动开销占了大头，不代表真实速度比。
