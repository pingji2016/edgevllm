# vLLM 运行文档（本机实测版）

> 本文件是本地自建文档，**不是上游 vllm-project/vllm 的内容**，会出现在 `git status` 里，提交前请排除。
> 所有命令和输出均为 2026-09-22 / 09-23 在本机实际跑通的结果，不是抄文档。

---

## 一、环境事实

| 项目 | 实测值 |
| --- | --- |
| 宿主系统 | Windows 11 Pro 26100 |
| GPU | NVIDIA GeForce RTX 5060 Ti，**15.9 GB 显存**，36 SM |
| 算力 | sm_120（Blackwell），满足 vLLM 要求的 ≥ 7.5 |
| 驱动 | 610.88（CUDA 13.3） |
| WSL 发行版 | Ubuntu-24.04（WSL2），内核 6.6.87.2 |
| WSL 内 Python | 3.12.3（系统自带，无 pip 无 ensurepip） |
| WSL 内存上限 | 15 GB（默认取宿主一半） |
| 磁盘 | ext4 根分区剩 928 GB |

**装出来的版本组合**（vLLM 对版本极其敏感，此组合已验证可用）：

```text
vllm    0.29.0
torch   2.13.0+cu132
uv      0.12.17
```

### 网络实测

这是本机装 vLLM 最大的障碍，先看清楚再动手：

| 源 | 实测速度 | 结论 |
| --- | --- | --- |
| `pypi.org` | **0.1 MB/s** | 不可用，5 GB 要下 14 小时 |
| `pypi.tuna.tsinghua.edu.cn` | **7.3 MB/s** | 主力源 |
| `download.pytorch.org` | 3.6 MB/s | torch 走这里 |
| `astral.sh`（uv 安装脚本） | 卡死 | 走 GitHub releases，不可用 |
| `huggingface.co` | **超时不通** | 必须用镜像 |
| `hf-mirror.com` | 3.9 MB/s | 模型源 |

---

## 二、一次性安装

装完约 7.5 GB，实测耗时约 20 分钟（清华源）。

### 1. 安装 uv（绕开官方脚本）

官方安装脚本从 GitHub 拉二进制，国内下不动。改用从清华源取 wheel 直接解包：

```bash
# 找到最新的 uv wheel 地址
curl -s https://pypi.tuna.tsinghua.edu.cn/simple/uv/ \
 | grep -o 'href="[^"]*"' | sed 's/href="//; s/"//' \
 | grep manylinux | grep x86_64 | grep -v aarch64 \
 | sed 's|^\.\./\.\./|https://pypi.tuna.tsinghua.edu.cn/|; s|#.*||' \
 | sed 's|.*/uv-\([0-9][0-9.]*\)-.*|\1 &|' \
 | sort -V | tail -1

# 下载并解包（python3 自带 zipfile，不需要 unzip）
curl -sL -o /tmp/uv.whl "<上面的地址>"
mkdir -p /tmp/uvx
python3 -c "import zipfile; zipfile.ZipFile('/tmp/uv.whl').extractall('/tmp/uvx')"

mkdir -p ~/.local/bin
cp /tmp/uvx/uv-*/.data/scripts/uv  ~/.local/bin/uv
cp /tmp/uvx/uv-*/.data/scripts/uvx ~/.local/bin/uvx
chmod +x ~/.local/bin/uv ~/.local/bin/uvx
uv --version
```

### 2. 建环境并安装 vLLM

```bash
export PATH="$HOME/.local/bin:$PATH"
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_HTTP_TIMEOUT=180

mkdir -p ~/vllm-runtime && cd ~/vllm-runtime
uv venv --python 3.12
uv pip install vllm --torch-backend=auto
```

`--torch-backend=auto` 会读驱动版本自动匹配 PyTorch 轮子。本机驱动 CUDA 13.3，最终选中 `torch 2.13.0+cu132`。

### 3. 验证

```bash
cd ~          # 注意：必须离开 vllm 仓库目录，见第六节坑 1
~/vllm-runtime/.venv/bin/python -c "
import torch, vllm
print('vllm      :', vllm.__version__)
print('torch     :', torch.__version__)
print('gpu       :', torch.cuda.get_device_name(0))
print('sm        :', torch.cuda.get_device_capability(0))
print('vram GB   :', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1))
"
```

期望输出：

```text
vllm      : 0.29.0
torch     : 2.13.0+cu132
gpu       : NVIDIA GeForce RTX 5060 Ti
sm        : (12, 0)
vram GB   : 15.9
```

---

## 三、启动服务

### 可直接复制的命令模板

```bash
wsl -d Ubuntu-24.04        # 从 Windows 进入 WSL

cd ~                       # 别在仓库目录里跑

export VLLM_WSL2_ENABLE_PIN_MEMORY=1     # WSL2 必须，见坑 3
export VLLM_USE_FLASHINFER_SAMPLER=0     # 绕开 sm 检测误判，见坑 4
export HF_ENDPOINT=https://hf-mirror.com # 国内必须
export HF_HUB_DISABLE_XET=1              # 镜像不代理 Xet，见坑 2

~/vllm-runtime/.venv/bin/vllm serve Qwen/Qwen3-0.6B \
  --served-model-name qwen3-0.6b \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192
```

### 本机实测启动日志（Qwen3-0.6B）

```text
Using FLASH_ATTN attention backend out of potential backends:
    ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']
Loading weights took 0.87 seconds
torch.compile took 32.93 s in total
Available KV cache memory: 11.25 GiB
GPU KV cache size: 105,328 tokens
    Maximum concurrency for 8,192 tokens per request: 12.86x
init engine (profile, create kv cache, warmup model) took 58.02 s
INFO:     Application startup complete.
```

**读法**：`Available KV cache memory` 和 `Maximum concurrency` 是判断配置是否合理的关键数字。这里 11.25 GiB 缓存、8192 上下文下能并发 12.86 个请求，说明 0.6B 模型对这张卡来说绰绰有余。

### 验证推理

```bash
curl -s http://localhost:8000/v1/models

curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-0.6b",
       "messages":[{"role":"user","content":"你好"}],
       "max_tokens":200}'
```

WSL2 会把端口转发到 Windows，**浏览器直接开 `http://localhost:8000/docs`** 即可（实测 Windows 侧 curl 同样可达）。

> **注意（Qwen3 系列的坑）**
> Qwen3 默认开启思考模式，短 `max_tokens` 会被 `<think>` 段吃光，表现为
> `finish_reason: "length"` 却看不到正式回答。要么把 `max_tokens` 调到 500+，
> 要么在请求里加 `"chat_template_kwargs": {"enable_thinking": false}`。
>
> **注意（Windows 侧 curl 传中文）**
> 在 **Git Bash / cmd 里**用 `-d '{"content":"你好"}'` 内联中文，服务端会返回
> `{"detail":"There was an error parsing the body"}` —— 这是 Windows shell 把
> UTF-8 传坏了，**不是 vLLM 的问题**。改用文件传参即可：
>
> ```bash
> printf '%s' '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"你好"}],"max_tokens":300}' > /tmp/body.json
> curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" --data-binary @/tmp/body.json
> ```
>
> 在 WSL 内执行、或用 Python / 浏览器 `/docs` 页面则不受影响。

---

## 四、显存规划：什么模型能跑

本机 15.9 GB 显存，判断公式很简单：

```text
能跑的条件 ≈ 权重体积 + KV cache + 约 1 GB 运行时开销  <  15.9 GB × gpu_memory_utilization
```

### 本机已有模型的实测结论

| 模型 | 格式 | 权重 | 结论 |
| --- | --- | --- | --- |
| Qwen3-0.6B | HF | 1.2 GB | ✅ **跑通，KV cache 还剩 11.25 GiB** |
| Hy-MT2-7B | GGUF Q4_K_M | 4.62 GB | 理论可跑，但插件不支持该架构（坑 6） |
| Hy-MT2-7B | HF | 16.06 GB | ⚠️ 需 `--cpu-offload-gb 5` 以上，且 15 GB 内存很紧张 |
| Qwen3.5-9B | HF | 17.3 GB | ❌ 超出显存，且多模态编码器会卡住（坑 5） |

**经验值**：16 GB 显存的安全区是 **8 GB 以内的模型**（bf16），或 14 GB 以内的 4-bit 量化模型。

### 显存不够时的三个选项

```bash
# 1. CPU 卸载（慢，但能跑）
--cpu-offload-gb 5

# 2. 缩短上下文，减少 KV cache 占用
--max-model-len 4096

# 3. 换量化模型（首选，速度损失最小）
#    Q4 量化约为 bf16 的四分之一
```

---

## 五、常用运维

```bash
# 看显存占用
/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

# 看服务日志
tail -f ~/vllm-serve.log

# 停服务
pkill -f "vllm.entrypoints"

# 查看本机 GPU 整体情况
/usr/lib/wsl/lib/nvidia-smi
```

从 Windows 侧彻底重启 WSL（改了 `.wslconfig` 后必须）：

```bash
wsl --shutdown
```

---

## 六、踩坑记录

按遇到顺序排列。每个坑都写清**现象 → 根因 → 解法**，因为报错信息本身往往指向不了根因。

### 坑 1：在仓库目录里跑 Python 会加载未编译源码

#### 现象

```text
ModuleNotFoundError: No module named 'vllm._C_stable_libtorch'
```

**根因**：`wsl.exe` 会继承 Windows 侧的当前目录。在 `E:\github\vllm` 下执行 `wsl -e ...`，WSL 里的工作目录就是 `/mnt/e/github/vllm`，Python 优先 import 仓库里的 `vllm/` 源码树，而不是 venv 里装好的包。源码树没有编译过的 C++ 扩展，自然找不到 `_C_stable_libtorch`。

**解法**：执行前 `cd ~`，或任何不在仓库内的目录。

### 坑 2：模型下载 401 Unauthorized

#### 现象

```text
RuntimeError: Task error: File reconstruction error: CAS Client Error:
HTTP status client error (401 Unauthorized),
domain: https://cas-server.xethub.hf.co/v2/reconstructions/...
```

**根因**：新版 `huggingface_hub` 默认启用 **Xet** 存储后端，它会绕过 `HF_ENDPOINT` 直连 `cas-server.xethub.hf.co`。国内这个域名不通，且镜像不代理它。

#### 解法

```bash
export HF_HUB_DISABLE_XET=1
```

强制走传统 HTTP 下载，镜像才生效。

### 坑 3：UVA is not available（WSL2 专属，必踩）

#### 现象

```text
File "vllm/v1/worker/gpu/buffer_utils.py", line 47, in __init__
    raise RuntimeError("UVA is not available")
```

**根因**：`is_uva_available()` 的实现就是 `is_pin_memory_available()`。而 vLLM 在 WSL2 上**默认关闭 pinned memory**，源码注释写得很直白：

```python
# On WSL2 with a compatible kernel (>= 4.19.121), pinned memory is
# supported but disabled by default due to a small performance regression.
# Set to 1 when pinned memory or UVA is required (e.g. CPU offloading
# or v2 model runner).
"VLLM_WSL2_ENABLE_PIN_MEMORY": ...
```

报错说 UVA，实际是 pinned memory 被关了 —— 这是最容易查错方向的一个坑。

#### 解法

```bash
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
```

本机内核 6.6.87.2 > 4.19.121，满足前提。

**附带警告**：开启后如果同时用 `--cpu-offload-gb`，vLLM 会提示 Windows（WDDM）对 pinned memory 有约**物理内存 50%** 的系统级上限，且过度占用页锁定内存可能导致宿主系统无响应、需要硬重启。本机 15 GB 内存，卸载量建议不超过 5 GB。

### 坑 4：FlashInfer 架构检测误判

#### 现象

```text
File "flashinfer/jit/core.py", line 109, in check_cuda_arch
    raise RuntimeError("FlashInfer requires GPUs with sm75 or higher")
```

**根因**：卡是 **sm120**，远高于 sm75，但 FlashInfer 的 `check_cuda_arch()` 遍历 `TARGET_CUDA_ARCHS` 判定失败（该列表在 WSL2 下为空或未被正确解析），于是把「检测不到架构」误报成了「架构太低」。

#### 解法

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

关掉 FlashInfer 采样器，改用原生采样。日志会确认：

```text
FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0.
```

采样之外的 FlashInfer 组件（autotune 等）仍可正常工作。

### 坑 5：多模态模型卡在编码器 profiling

**现象**：Qwen3.5-9B（`Qwen3_5ForConditionalGeneration`）启动后日志停在

```text
Encoder cache will be initialized with a budget of 16384 tokens,
and profiled with 1 image items of the maximum feature size.
```

之后 **8 分钟以上无任何日志推进，GPU 持续 100%**。

**根因**：Qwen3.5 是多模态模型，启动时要为**最大分辨率图像**做视觉编码器 profile。36 个 SM 的消费级卡做这件事极慢，叠加显存吃紧（15.98 GB / 15.9 GB），基本等于卡死。

**解法**：纯文本场景直接换非多模态模型。若必须用多模态模型，考虑限制输入分辨率或改用更大显存的卡。

**顺带**：`UvaBuffer` 的报错也是在这个模型上第一次出现的，说明坑 3 与模型无关，是 WSL2 的通用问题。

### 坑 6：GGUF 插件不支持自定义架构

#### 现象

```text
File "vllm_gguf_plugin/weights_adapter/default.py", line 150, in build_name_map
    raise RuntimeError(f"Unknown gguf model_type: {hunyuan_v1_dense}")
```

**根因**：GGUF 支持已从 vLLM 主仓迁移到独立插件 `vllm-gguf-plugin`。插件内置了一份架构名映射表，**不含 Hunyuan 系列**，遇到未知 `model_type` 直接抛错。

**解法**：GGUF 只对插件已覆盖的主流架构（Llama / Qwen / Mistral 等）可用。冷门架构请用 HF 原生格式。

### 坑 7：显存不够分给 KV cache

#### 现象

```text
ValueError: No available memory for the cache blocks.
Try increasing `gpu_memory_utilization` when initializing the engine
```

**根因**：Hy-MT2-7B 权重 16.06 GB，只卸载 3 GB 时，剩余显存连一个 cache block 都放不下。注意 `gpu_memory_utilization` 只控制总预算，**它不能变出显存**。

**解法**：加大 `--cpu-offload-gb`（本例需 5 GB 以上），或缩短 `--max-model-len`，或改用量化模型。

### 坑 8：Windows 侧 curl 传中文 body 报解析失败

#### 现象

```text
{"detail":"There was an error parsing the body"}
```

同样的请求，换成英文就正常，服务本身没报任何错。

**根因**：Windows 的 Git Bash / cmd 在 `-d '...'` 里内联中文时，会把 UTF-8 字节传坏。这是 **shell 的编码问题，与 vLLM 无关**（在 WSL 内执行同样的命令完全正常）。

**解法**：把 body 写进文件再传：

```bash
printf '%s' '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"你好"}],"max_tokens":300}' > /tmp/body.json
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" --data-binary @/tmp/body.json
```

或者干脆用浏览器打开 `http://localhost:8000/docs` 里的 Swagger 界面发请求，没有编码问题。

---

## 七、一页速查

```bash
# ---- 进环境 ----
wsl -d Ubuntu-24.04
cd ~

# ---- 必备环境变量（缺一个都可能失败）----
export PATH="$HOME/.local/bin:$PATH"
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

# ---- 起服务 ----
~/vllm-runtime/.venv/bin/vllm serve <模型> \
  --served-model-name <名字> \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192

# ---- 验证 ----
curl -s http://localhost:8000/v1/models

# ---- 浏览器 ----
# http://localhost:8000/docs
```

只要记住这几个环境变量，本机的 vLLM 就能稳定跑起来。
