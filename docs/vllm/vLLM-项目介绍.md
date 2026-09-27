# vLLM 项目介绍

> 本文件是本地自建的说明文档，**不是上游 vllm-project/vllm 的内容**。
> 它出现在 `git status` 里，提交或建 PR 前请先排除，否则会触发上游的 no-busywork 政策。
> 内容基于本地 checkout（commit `e34685dfc0`）实际核对编写。

---

## 一、这是什么

vLLM 是一个**大语言模型推理与服务引擎**。它解决的核心问题是：把模型权重加载到 GPU 之后，如何用同一块卡同时、高效地服务大量并发请求。

- 起源于 UC Berkeley 的 Sky Computing Lab，2023 年随 PagedAttention 论文（SOSP 2023）开源
- 目前由 2000+ 贡献者、数十家公司和高校共同维护，是活跃度最高的开源 AI 项目之一
- 定位是**推理侧的基础设施**：不训练模型，只负责把已有模型跑得又快又省

一句话概括它的价值：**同样的硬件，更高的吞吐；同样的吞吐，更低的成本。**

---

## 二、为什么快：五个核心机制

### 1. PagedAttention — KV Cache 分页管理

这是 vLLM 的立身之本。传统实现里，每个请求的 KV cache 需要一整块**连续显存**，而序列长度事先不可知，只能按最大长度预留，导致大量显存被浪费（实测浪费可达 60%~80%）。

vLLM 借用了操作系统的虚拟内存思想：

- 把 KV cache 切成固定大小的 **block**（相当于内存页）
- 每个请求维护一张 **block table**（相当于页表），逻辑上连续、物理上可以不连续
- 显存按需分配、用完即回收，碎片率接近 0

副产品是**前缀共享**：多个请求如果开头相同（同一个 system prompt），它们可以指向同一批物理 block，显存直接省下 N 倍。这就是 prefix caching 的基础。

### 2. 连续批处理 Continuous Batching

传统静态批处理必须等一整批里最慢的序列生成完才能换下一批，GPU 有大量时间在空转。vLLM 在**每一步解码后**就重新组批：已经结束的序列立刻退出，排队的请求立刻补位。

结果是 GPU 利用率常年保持在饱和状态，吞吐相比静态批处理可以提升一个数量级。

### 3. Chunked Prefill 与 Prefix Caching

- **Chunked Prefill**：把超长的 prefill 切成小块，和多条 decode 混在同一个 batch 里执行。既避免长 prompt 阻塞其他请求（降低首 token 延迟抖动），又让计算密集型和访存密集型的负载互补，提升整体利用率。
- **Prefix Caching**：跨请求复用相同前缀的 KV cache。多轮对话、few-shot 提示词场景下收益极大。

### 4. CUDA/HIP Graph 捕获

逐 token 解码时，每个 kernel 的启动开销会显著放大。vLLM 把整段计算图**提前捕获**成一张 graph，之后一次下发、一次执行。同时支持 piecewise（分段）模式，以便在图中插入动态的注意力计算。

### 5. 高度优化的算子与量化

- **注意力内核**：FlashAttention、FlashInfer、TRTLLM-GEN、FlashMLA、Triton 多套后端按场景自动选择
- **GEMM/MoE 内核**：基于 CUTLASS、TRTLLM-GEN、CuTeDSL，覆盖各种精度
- **量化**：FP8、MXFP8/MXFP4、NVFP4、INT8、INT4、GPTQ/AWQ、GGUF、compressed-tensors、ModelOpt、TorchAO
- **推测解码**：n-gram、suffix、EAGLE、DFlash、Medusa 等多种草稿策略
- **图级优化**：torch.compile 驱动的自动内核生成与图变换

---

## 三、能力边界

| 维度 | 支持情况 |
| --- | --- |
| 模型架构 | 200+ 个 HF 架构：Decoder-only（Llama / Qwen / Gemma）、MoE（Mixtral / DeepSeek-V3 / Qwen-MoE / GPT-OSS）、混合注意力与状态空间（Mamba / Qwen3.5）、多模态（LLaVA / Qwen-VL / Pixtral）、Embedding 与 Rerank（E5-Mistral / GTE / ColBERT）、Reward 与分类模型 |
| 硬件 | NVIDIA GPU、AMD GPU、Intel GPU、x86/ARM/PowerPC CPU；插件形式支持 Google TPU、Intel Gaudi、IBM Spyre、华为昇腾、Rebellions NPU、Apple Silicon、沐曦等 |
| 并行策略 | Tensor / Pipeline / Data / Expert / Context Parallel |
| 服务接口 | OpenAI 兼容 API、Anthropic Messages API、gRPC |
| 进阶特性 | 多 LoRA、结构化输出（xgrammar / guidance）、工具调用与推理链解析、Prefix 与 Decode 分离部署、KV Connector 与 KV Offload |
| 系统要求 | **OS: Linux**（Windows 需 WSL2），Python 3.10–3.13，GPU 算力 ≥ 7.5 |

---

## 四、架构鸟瞰

### 分层结构

```text
              ┌─────────────────────────────────────────────┐
              │  entrypoints/   CLI / HTTP / gRPC / 离线 LLM │
              └──────────────────────┬──────────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────┐
              │  v1/engine/   LLMEngine · AsyncLLM · 主循环   │
              │  输入处理 → 调度 → 执行 → 采样 → 输出处理      │
              └──────────────────────┬──────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
┌───────▼────────┐          ┌────────▼────────┐          ┌────────▼────────┐
│ v1/core/sched/ │          │   v1/worker/    │          │ v1/attention/   │
│   调度器        │          │  GPU Worker     │          │  注意力后端      │
│ ─────────────  │          │  ModelRunner    │          │ ─────────────   │
│ v1/core/       │          │  block_table    │          │ FlashAttention  │
│ kv_cache_*     │          │  input_batch    │          │ FlashInfer      │
│   block_pool   │          │  CUDA Graph     │          │ Triton / MLA    │
└────────────────┘          └────────┬────────┘          └─────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
┌───────▼────────┐          ┌────────▼────────┐          ┌────────▼────────┐
│model_executor/ │          │   platforms/    │          │   csrc/ rust/   │
│ 层实现·加载·量化 │          │  cuda rocm xpu  │          │ C++/CUDA/RS 内核 │
│  models/ 各架构 │          │  cpu tpu zen    │          │                 │
└────────────────┘          └─────────────────┘          └─────────────────┘
```

### 关键目录导读

| 路径 | 职责 | 什么时候会碰它 |
| --- | --- | --- |
| `vllm/entrypoints/` | 对外入口：CLI、OpenAI/Anthropic/Cohere 兼容 API、gRPC、离线 `LLM` 类 | 改 API 行为、加接口、加参数 |
| `vllm/v1/engine/` | V1 引擎：`llm_engine.py`、`async_llm.py`、输入输出处理、detokenizer | 改进请求生命周期、加引擎级特性 |
| `vllm/v1/core/sched/` | 调度器：`scheduler.py`、`async_scheduler.py`、请求队列 | 改调度策略、批处理逻辑 |
| `vllm/v1/core/` | KV cache 管理：`kv_cache_manager.py`、`block_pool.py`、`kv_cache_coordinator.py` | 改显存分配、加缓存策略 |
| `vllm/v1/worker/` | GPU/CPU Worker 与 ModelRunner、block table、输入批、CUDA Graph | 改模型执行路径、加硬件后端 |
| `vllm/v1/attention/` | 注意力后端注册与实现，`selector.py` 负责选型 | 加新的 attention kernel |
| `vllm/v1/spec_decode/` | 推测解码：EAGLE、Medusa、DFlash、n-gram、suffix | 改或加推测解码方案 |
| `vllm/model_executor/` | 模型层实现、权重加载、量化、warmup | 加模型算子、改量化逻辑 |
| `vllm/models/` | 各模型架构的**具体**实现（按模型分子目录） | 适配新模型 |
| `vllm/platforms/` | 硬件平台抽象层：`interface.py` 定义契约，`cuda.py` / `rocm.py` / `xpu.py` / `cpu.py` 实现 | 适配新硬件、改平台行为 |
| `vllm/config/` | 配置解析与校验（对应 CLI 的每个参数） | 加一个新参数 |
| `vllm/multimodal/` | 多模态输入处理 | 加多模态模型 |
| `vllm/lora/` | LoRA 支持 | 改 LoRA 逻辑 |
| `vllm/distributed/` | 分布式通信原语（all-reduce、自定义 collective） | 改并行策略 |
| `csrc/` | C++/CUDA 内核源码，含 attention、moe、quantization、rocm | 写或改底层内核 |
| `rust/` | Rust 前端二进制 | 改前端性能 |
| `tests/` | pytest 测试套件 | 加测试 |
| `benchmarks/` | 性能基准，kernel 级压测放这里 | 做性能验证 |

### 一次请求的生命周期

1. **接入** — `vllm/entrypoints/openai/` 收到 HTTP 请求，转成内部请求对象
2. **输入处理** — `v1/engine/input_processor.py` 做 tokenize、多模态预处理、模板渲染
3. **调度** — `v1/core/sched/scheduler.py` 从队列取请求，决定这一轮要跑哪些序列、各自跑多少 token
4. **显存分配** — `v1/core/kv_cache_manager.py` 通过 `block_pool.py` 从 block 池里划物理块，更新 block table
5. **执行** — `v1/worker/gpu_model_runner.py` 组织输入批，通过 CUDA Graph 下发模型前向，`v1/attention/` 选定后端算注意力
6. **采样** — `v1/sample/` 按温度、top-p、惩罚项等参数采样出下一个 token
7. **输出处理** — `v1/engine/output_processor.py` 与 `detokenizer.py` 解码成文本，流式推回客户端
8. **回收** — 序列结束后归还 block，调度器在下一步立刻补入新请求

理解这条链路，基本就掌握了 vLLM 的主干。

---

## 五、在本机跑起来

### 前提：必须走 WSL2

vLLM **不支持 Windows 原生运行**（见 `docs/getting_started/installation/gpu.md`）。官方明确给出两条路：WSL2，或社区 fork。本机环境已核实可直接跑：

| 检查项 | 本机实测结果 |
| --- | --- |
| WSL 发行版 | Ubuntu-24.04（WSL2） |
| GPU 直通 | 可识别 `NVIDIA GeForce RTX 5060 Ti` |
| 算力 | 12.0（Blackwell sm_120），满足 ≥ 7.5 |
| CUDA 工具链 | `/usr/local/cuda-12.8/bin/nvcc` |
| Python | 3.12.3 |

**注意**：Blackwell 架构要求 CUDA ≥ 12.8，用 `--torch-backend=auto` 让 uv 自动匹配即可。

### 安装与运行

下面是**本机实测跑通**的命令，与上游文档有两处关键差异（国内网络 + WSL2）：

```bash
wsl -d Ubuntu-24.04        # 从 Windows 进入 WSL
cd ~                       # 别在仓库目录里跑，见下方说明

export VLLM_WSL2_ENABLE_PIN_MEMORY=1     # WSL2 必须
export VLLM_USE_FLASHINFER_SAMPLER=0     # 绕开 sm 检测误判
export HF_ENDPOINT=https://hf-mirror.com # 国内必须
export HF_HUB_DISABLE_XET=1              # 镜像不代理 Xet

uv pip install vllm --torch-backend=auto

~/vllm-runtime/.venv/bin/vllm serve Qwen/Qwen3-0.6B \
  --served-model-name qwen3-0.6b \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192
```

两个与上游文档不同的地方：

- **uv 不能按官方脚本装**。`curl -LsSf https://astral.sh/uv/install.sh | sh` 走 GitHub releases，国内会卡死在下载环节。改用清华 PyPI 源的 wheel 直接解包，见运行文档第二节。
- **四个环境变量缺一不可**。少了任何一个都会失败，且报错信息指向不了真正原因（例如 `UVA is not available` 实际是 pinned memory 被关闭）。

> **完整的实测记录见《vLLM-运行文档.md》**，含逐条 `现象 → 根因 → 解法` 的踩坑记录。

WSL2 会把端口转发到 Windows，浏览器直接开 `http://localhost:8000/docs` 即可。

### 跑本仓库源码（要改代码时）

```bash
cp -r /mnt/e/github/vllm ~/vllm && cd ~/vllm   # 挪到 ext4，跨文件系统 import 很慢
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

改 Python 立即生效。改 C++/CUDA 才需要走 `docs/contributing/incremental_build.md` 的增量编译流程。

**注意**：`wsl.exe` 会继承 Windows 侧的当前目录。在 `E:\github\vllm` 下执行 WSL 命令时，WSL 里的 cwd 就是 `/mnt/e/github/vllm`，Python 会优先 import 仓库里**未编译**的源码树，报 `No module named 'vllm._C_stable_libtorch'`。执行前先 `cd ~`。

### 本机显存约束（15.9 GB）

- `--gpu-memory-utilization` 建议 0.85
- **安全区是 8 GB 以内的 bf16 模型**，或 14 GB 以内的 4-bit 量化模型
- 本机实测：Qwen3-0.6B 跑通后 KV cache 还剩 11.25 GiB；Qwen3.5-9B（17.3 GB）超出显存且多模态编码器会卡住
- 拉模型慢的话先 `export HF_ENDPOINT=https://hf-mirror.com`

---

## 六、如何参与贡献

### 你能贡献什么

上游欢迎的不只是代码：

- 报 Bug、复现问题、补充日志
- 请求或实现**新模型支持**（对应 `new-model` 标签的 issue）
- 提新特性、改性能
- 改文档、写 how-to
- **回答问题、review 别人的 PR** —— 上游明确说这也算重要贡献

新手入口：仓库的 **Good first issues** 标签，以及 vllm-project 组织下的 onboarding 任务看板。

### 环境搭建

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm

uv venv --python 3.12
source .venv/bin/activate

# 只改 Python
VLLM_USE_PRECOMPILED=1 uv pip install -e .

# 装 lint
uv pip install pre-commit>=4.5.1
pre-commit install

# 装测试依赖
uv pip install -r requirements/common.txt -r requirements/dev.txt --torch-backend=auto
```

**硬性规定**：不要用系统 `python3` 或裸 `pip`，一律走 `uv` 和 `.venv/bin/python`。推荐 Python 3.12，与 CI 对齐。

只重编 Rust 前端：`./build_rust.sh`（release）或 `./build_rust.sh --debug`（快）。

### 代码规范

- Google Python / C++ 风格指南
- **行宽 88 字符**
- Docstring 用 Google 风格（`Args:` / `Returns:` / `Raises:`），**不要**用 Sphinx 的 `:param:`
- 少写注释，优先让代码自解释
- 改了用户可见行为，必须同步更新 `docs/`

```bash
pre-commit run            # 检查已暂存文件
pre-commit run --all-files
pre-commit run mypy-3.12 --all-files --hook-stage manual
```

### 测试

```bash
pytest -s -v tests/test_logger.py          # 单文件
pytest tests/                              # 全量
```

写测试前先回答四个问题：这个模块干什么、I/O 契约是什么、我要防的是什么故障、**能抓住它的最低成本层级是哪个**（单元测试 > 集成 > 端到端）。

- 优先扩展现有测试文件和 fixture，别急着新建文件
- 一个测试只验一个行为
- **kernel 性能测试不要放 `tests/`**，放 `benchmarks/kernels/`；正确性在既有 pytest 套件里证明
- 影响模型输出的改动，**主动**跑 `tests/evals/` 或用 `vllm bench` 并把结果贴进 PR，别等 reviewer 要

### 提 PR 的硬性要求

#### 1. DCO 签名

```bash
git commit -s    # 自动加 Signed-off-by
```

PyCharm 勾选 Sign-off commit；VSCode 打开 `git.alwaysSignOff`。

#### 2. 标题格式

`[标签][标签] 描述`，例如 `[Bugfix][Scheduler] Fix priority handling`。

- 类型标签：`[Bugfix]` `[Feature]` `[Perf]` `[Refactor]` `[CI]` `[Test]` `[Doc]` `[Misc]`
- 范围标签：`[Model]` `[Frontend]` `[Core]` `[Kernel]` `[Attention]` `[Multimodal]` `[Quantization]` `[MoE]` `[Spec Decode]` `[LoRA]` `[KV Connector]` `[ROCm]` `[XPU]` `[CPU]` 等

#### 3. 大改动先开 RFC

超过 500 行（不含 kernel/数据/配置/测试）的架构级改动，必须先开 issue 讨论设计，否则会被打上 `rfc-required` 并可能直接不审。

#### 4. AI 辅助的强制披露

这一条对本项目尤其重要，违反可能导致自动封禁：

- **禁止纯 agent PR**。提交人必须逐行读懂改动、端到端验证、亲自跑测试
- PR 描述里必须写明：为什么没有和已有 PR 重复、跑了什么测试命令、结果如何、用了 AI 辅助
- 影响输出的改动必须附模型评测结果
- commit 加 `Co-authored-by:` 署名

#### 5. 提 PR 前先查重

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<关键词>"
```

#### 6. 别做低价值 PR

单个错别字、孤立格式调整、一处 mutable default 之类的一次性小改动**不要单独开 PR**。机械清理只有搭在实质工作上一起提才可接受。

#### 7. 数量与升级

无写权限的贡献者**最多同时开 6 个 PR**。紧急工作被卡住可联系 committer 申请白名单。重要贡献需要加速 review，可发邮件到 `pr-review-request@vllm.ai`，用可验证的公司/学校邮箱，附上使用场景、遇到的问题、你的改动如何解决。

### Review 节奏

- PR 提交后会分配给 reviewer，按专长和空闲度领取
- reviewer 每 2–3 天给一次状态更新；**超过 7 天没人理，可以直接 ping**
- 需要改动的会打 `action-required` 标签，改完 ping reviewer 复审
- 新 commit **不会自动触发** CI。需要时用 `/ci run`、`/ci retry`、`/ci cancel`（AMD 用 `/amd-ci` 前缀）。PR 作者只有在 PR 被 approve 或打了 `ready` 标签后才能用这些命令，之前只有有写权限的 reviewer 和受信贡献者能用
- **`/ci run` 要求你的分支相对目标分支"零落后"**。落后了要先 rebase 到最新再跑。硬要在旧分支上跑得加 `--allow-stale`，且合并前必须在最新 commit 上重跑一次

### 红线

- 纯 agent 生成、提交人看不懂的代码 —— 不接受
- 刷量式小 PR —— 不接受
- 发现安全漏洞走 `SECURITY.md` 的流程，**不要**开公开 issue

---

## 七、参考资料

- 官方文档：<https://docs.vllm.ai>
- 贡献指南：<https://docs.vllm.ai/en/latest/contributing>
- PagedAttention 论文：<https://arxiv.org/abs/2309.06180>
- 用户论坛：<https://discuss.vllm.ai>
- 开发者 Slack：<https://slack.vllm.ai>
- 本地可读的上游文档：`docs/getting_started/`、`docs/contributing/`、`docs/design/`
