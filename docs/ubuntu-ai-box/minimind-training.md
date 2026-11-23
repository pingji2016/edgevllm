# 从零训练一个 30M 小模型（minimind 全流程）

> 主机 `pingji-MS-7B89` · RTX 5060 Ti 16 GB（Blackwell sm_120）· Ubuntu 26.04 LTS
> Python 3.12.16 · PyTorch 2.11.0+cu128 · Transformers 4.57.6（独立 uv venv）
> 记录时间：2026-11-18（训练）· 2026-11-23（吞吐复测 + 分布式验证）

---

## 0. 结论速览

| 项 | 实测值 |
| --- | ---: |
| 模型规模 | **30.03 M** 参数（hidden 512 / 8 层 / vocab 6400） |
| 预训练 | 2 epochs ≈ **2.8 小时**，loss 2.7 → 2.02 |
| SFT | 1 epoch ≈ **1.8 小时**，loss 1.9 → 1.4 |
| 预训练吞吐（batch 24 × seq 512） | **65,234 tok/s**（834 步 / 157 s，端到端含启动） |
| 权重文件 | 各 **66.6 MB**（bf16 存储） |
| 显存占用 | 约 3 GB（batch 24 / seq 512），16 GB 余量极大 |
| 能力 | 会写代码、会自我介绍；**算不对 1+1** |

一句话：**30M 模型在这张卡上不是显存问题，是时间问题** —— 单卡 6 小时跑完「预训练 + SFT」全流程。

---

## 1. 为什么从零训一个 30M 模型

调 API 得到的结论往往是「它很强」，而自己从零训一个得到的结论是「**为什么弱**」：

- 见过一次 loss 从 8.76（≈ ln 6400 的随机水平）掉到 2.0 的完整曲线，对「训练到底在干什么」才有体感；
- 只有自己训过，才知道**数据质量、序列长度、学习率**三者的权重远大于「模型多大」；
- 30M 的规模能在单卡 6 小时内闭环，**一天之内可以迭代三次**，比租大卡跑一次更有价值。

---

## 2. 环境：独立 venv，不污染既有环境

| 项 | 值 |
| --- | --- |
| 位置 | `~/apps/minimind/.venv`（uv 创建） |
| Python | 3.12.16 |
| PyTorch | 2.11.0+cu128 |
| Transformers | 4.57.6 |

**为什么要独立 venv**：本机 `~/apps/bench/yolo-venv` 里是 torch 2.15 + transformers 5.x，
而 minimind 依赖 transformers 4.x 的 `MoeCausalLMOutputWithPast` 等接口。混装会直接 import 失败。

```bash
cd ~/apps/minimind
uv venv --python 3.12 .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install transformers==4.57.6 datasets accelerate
```

---

## 3. 预训练

**配置**：hidden 512 / 8 层 / vocab 6400 → 30.03 M 参数，问数语料约 1.2 GB。

```bash
cd ~/apps/minimind/trainer
../.venv/bin/torchrun --nproc_per_node 1 train_pretrain.py \
  --hidden_size 512 --num_hidden_layers 8 \
  --batch_size 24 --max_seq_len 512 --epochs 2 --learning_rate 1e-4 \
  --data_path ../dataset/pretrain_t2t_mini.jsonl
```

**实测**：2 个 epoch 约 **2.8 小时**，loss 2.7 → 2.02，产物 `out/pretrain_512.pth`（66.6 MB）。

### 吞吐复测（2026-11-23）

用 20,000 条样本（834 步、10.25 M token）单独跑一轮，端到端计时：

| 指标 | 值 |
| --- | ---: |
| 步数 | 834（batch 24 × seq 512） |
| 生成 token | 10,248,192 |
| 端到端耗时 | 157.1 s（含约 13 s 启动：导 torch / 载 tokenizer / 建数据集） |
| **吞吐** | **65,234 tok/s** |
| 单步耗时 | ≈ 0.19 s |

扣掉启动约 **71,000 tok/s**。本次测量时 `vlm-server` 常驻占 6.4 GB 显存，
早先空载时观测到过 **约 85,000 tok/s** —— 差 24%，说明**小模型训练对同卡的其他负载很敏感**。

另外一个有意思的观测：**从随机初始化开始，2.5 分钟（834 步）loss 就从 8.76 掉到 6.83**。
早期收敛极快，后面才是慢功夫。

---

## 4. SFT：把 10 小时压到 1.8 小时

仓库默认参数是 `lr=1e-5` + `2 epochs`，在本机要跑 **约 10 小时**。改成下面这组后 **1.8 小时**完成：

| 参数 | 默认 | 调整后 | 原因 |
| --- | ---: | ---: | --- |
| `--epochs` | 2 | **1** | 30M 模型 1 个 epoch 足够，2 轮只增加过拟合 |
| `--learning_rate` | 1e-5 | **1e-4** | 默认太保守，10 倍后 1 轮内收敛 |
| `--max_seq_len` | 340 | **512** | 与预训练对齐，避免位置编码分布不一致 |
| `--batch_size` | 32 | **24** | 显存/速度折中（也可用 32，余量够） |

```bash
../.venv/bin/torchrun --nproc_per_node 1 train_full_sft.py \
  --hidden_size 512 --num_hidden_layers 8 --from_weight pretrain \
  --epochs 1 --max_seq_len 512 --batch_size 24 --learning_rate 1e-4
```

**实测**：约 **1.8 小时**，loss 1.9 → 1.4，产物 `out/full_sft_512.pth`（66.6 MB）。

---

## 5. 训完能用吗？三个测试

加载 `full_sft_512.pth` 直接对话：

| 测试 | 结果 | 说明 |
| --- | --- | --- |
| 「你是谁？」 | ✅ 通过 | 能给出连贯的自我介绍，带人设 |
| 「用 Python 写冒泡排序」 | ✅ 通过 | 代码结构和缩进都对，能跑 |
| 「1+1 等于几？」 | ❌ 失败 | 开始解释 `sum()` 和 Python 的加法语义，**就是不说 2** |

**结论：30M 模型的「语言流畅度」已经够用，「事实/算术准确性」完全不行。**
这恰好说明了 SFT 的本质——它学的是**说话的方式**，不是**知识**。
要 1+1=2 这种能力，靠的是预训练语料里的模式密度，而不是指令微调。

---

## 6. 分布式：先在一台 CPU 上把 DDP 验完，再租卡

### 6.1 为什么值得先验

多卡训练最贵的坑不是写代码，而是**租了卡才发现逻辑错**：
跑到第 3 小时 loss 突然跳变，你无法区分是「学习率不对」还是「两张卡偷偷不同步」。

所以顺序应该是：**CPU（gloo，零成本）验证正确性 → GPU（nccl）验证速度**。

### 6.2 CPU 双进程验证（gloo）

模型用真实结构（hidden 512 / 8 层 / 30.03 M 参数），每进程独立数据，结果：

| 检测项 | 实测 | 判据 |
| --- | ---: | --- |
| 梯度跨进程最大差 | **0.000e+00** | 应为 0（逐位相同） |
| DDP 梯度 vs 单进程 4 样本批次 | **1.118e-08** | < 1e-5，证明梯度平均等价于大 batch |
| 3 次 optimizer step 后参数差 | **0 / 0 / 0** | 更新后仍严格同步 |
| 反证：不包 DDP 时参数差 | **2.001e-03** | >> 0，证明检测手段有效 |
| 数据分片 | loss 两进程不同 | `DistributedSampler` 生效 |

其中第 2 项是最关键的：**DDP 的 AllReduce 求平均，数值上等于「把 N 张卡的样本拼成一个大 batch 做梯度」**，
1e-8 的残差只是 float32 累加顺序不同。

### 6.3 需要给训练脚本打的小补丁

仓库原始代码把分布式写死成了 GPU：

```python
dist.init_process_group(backend="nccl")     # CPU 上直接报错
torch.cuda.set_device(local_rank)
args.device = f"cuda:{local_rank}"
```

改成按设备自动选择，**GPU 路径行为完全不变**：

```python
def distributed_uses_cuda():
    """分布式是否走 GPU：无可用 GPU，或显式设置 DDP_CPU=1 时退回 CPU + gloo。"""
    return torch.cuda.is_available() and os.environ.get("DDP_CPU", "0") != "1"
```

用环境变量显式切换，而不是只靠 `torch.cuda.is_available()` —— 因为本机有显卡，
后者永远为真，想跑 CPU 验证就永远切不过去。

### 6.4 真实训练脚本的端到端验证

| 配置 | 步数 | 耗时 | 吞吐 |
| --- | ---: | ---: | ---: |
| CPU 2 进程（gloo，batch 4 × seq 128） | 25 步/rank | 29.9 s | 857 tok/s |
| GPU 1 进程（nccl，batch 4 × seq 128） | 50 步 | 13.0 s | 1,968 tok/s |

这组对比**只说明两条路径都通**，不代表真实速度比：seq 128 / batch 4 时启动开销占了 GPU 那 13 秒里的大半。

### 6.5 上双卡前要知道的三件事

1. **lr 要按 √N 缩放**：总 batch 翻倍时 `1e-4 → 1.41e-4`。线性放大小模型容易发散。
2. **world_size 变了，`iters` 就变了**：epoch 步数减半，`SkipBatchSampler` 的 skip 偏移会错位，
   表现为 **loss 突然跳变**。跨 world_size 不要用旧 checkpoint 续训。
3. **4090/5060 Ti 没有 NVLink**，卡间走 PCIe；30M 小模型每步 AllReduce 约 120 MB（fp16 梯度 × 2），
   实测加速比预期 **1.8–1.9x** 而不是 2x（估算，待真实双卡验证）。

---

## 7. 踩坑清单

| # | 坑 | 现象 | 解法 |
| --- | --- | --- | --- |
| 1 | `--from_weight` 只接受前缀 | 传 `pretrain_512.pth` 会找不到权重 | 传 `pretrain`（代码内部拼 `.pth`） |
| 2 | HF 下载数据集漏 `repo_type` | 报 404，但文件明明存在 | `load_dataset(..., repo_type="dataset")`，配 `HF_ENDPOINT=https://hf-mirror.com` |
| 3 | 混用 bench 的 venv | `import transformers` 报 `MoeCausalLMOutputWithPast` 不存在 | 独立 venv 锁 4.57.6 |
| 4 | 产物放 `/tmp` | 重启/清理后解包内容和模型全丢 | 一律放 `~/apps/<项目>/` 持久目录 |
| 5 | 默认 SFT 参数 | 要跑 10 小时 | 见第 4 节：1 epoch + lr 1e-4 ≈ 1.8 小时 |
| 6 | 用 `torch.cuda.is_available()` 判断走不走 GPU | 有卡就永远进 GPU 分支，CPU 验证跑不了 | 加 `DDP_CPU=1` 显式开关 |

---

## 8. 复现命令（最短路径）

```bash
# 1. 环境
cd ~/apps/minimind && uv venv --python 3.12 .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install transformers==4.57.6 datasets accelerate

# 2. 预训练（约 2.8 小时 / 2 epochs）
cd trainer
../.venv/bin/torchrun --nproc_per_node 1 train_pretrain.py \
  --hidden_size 512 --num_hidden_layers 8 --batch_size 24 \
  --max_seq_len 512 --epochs 2 --learning_rate 1e-4

# 3. SFT（约 1.8 小时）
../.venv/bin/torchrun --nproc_per_node 1 train_full_sft.py \
  --hidden_size 512 --num_hidden_layers 8 --from_weight pretrain \
  --epochs 1 --max_seq_len 512 --batch_size 24 --learning_rate 1e-4

# 4. 先验证分布式正确性（零成本，可选）
DDP_CPU=1 OMP_NUM_THREADS=4 ../.venv/bin/torchrun --nproc_per_node 2 \
  --standalone train_pretrain.py --device cpu \
  --hidden_size 512 --batch_size 4 --max_seq_len 128 --epochs 1

# 5. 租到双卡后
bash run_ddp_2gpu.sh                       # 预训练，等效 batch 384
TASK=sft bash run_ddp_2gpu.sh              # SFT
```

### 产物对照

| 文件 | 大小 | 说明 |
| --- | ---: | --- |
| `out/pretrain_512.pth` | 66.6 MB | 预训练权重（2 epochs） |
| `out/full_sft_512.pth` | 66.6 MB | SFT 后权重，可直接对话 |

> 66.6 MB ≈ 30.03 M × 2 字节（bf16）+ 少量元数据。换算成 fp32 会翻倍到 120 MB。

---

## 9. 下一步可以做的

| 实验 | 预期耗时 | 目的 |
| --- | ---: | --- |
| 64M / 128M 规模对比 | 3–6 小时 | 看参数量对 loss 曲线的实际边际收益 |
| LoRA 人设微调 | 约 20 分钟 | 验证「小模型 + 轻量微调」能否快速定制 |
| DPO 对齐 | 约 1 小时 | 看 30M 模型能不能拉开「好回答/坏回答」的差距 |
| GRPO（`train_grpo.py`） | 数小时 | 小模型上 RL 是否真的有效 |

**唯一需要花钱的是真实双卡加速比**（约 ¥4/小时）。其余全部可以在本机零成本跑完。
