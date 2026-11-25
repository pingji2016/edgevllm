# MiniCPM-o 4.5 实测：9B 全模态模型在本机跑通

> 机器 `pingji-MS-7B89` · RTX 5060 Ti 16 GB（sm_120）· Ubuntu 26.04
> 实测日期：2026-11-25 · 引擎：`OpenBMB/llama.cpp-omni`（Vulkan 后端）

---

## 1. 它是什么

MiniCPM-o 4.5 是端到端的**全模态**模型，9B 参数，由三块拼成：

```
视觉：SigLip2-400M
音频：Whisper-medium（理解）+ CosyVoice2（生成）
语言：Qwen3-8B（主干）
```

能力：
- 图像 / 视频 / 文本 / 音频四模态输入
- 输出文本 **和语音**（可声音克隆）
- **全双工**：能一边看/听实时流，一边生成语音和文本，互不阻塞（"边看、边听、边说"）

许可证 Apache-2.0。GGUF 量化档从 Q4_K_S（4.47 GB）到 Q8_0（8.11 GB）。

## 2. 本机部署：三个坑

### 坑 1：标准 llama.cpp 的 mtmd 不认这个视觉编码器

```
ggml_abort / clip_model_loader::load_tensors 崩溃
```

必须用官方的 **llama.cpp-omni** 分支。

### 坑 2：上游代码在 Linux 上编译不过（macOS 专有 API）

```
src/llama-moe-slot.cpp:1351: error: 'F_NOCACHE' was not declared in this scope
src/llama-moe-slot.cpp:1375: error: 'F_NOCACHE' was not declared in this scope
```

`F_NOCACHE` 是 macOS 的 `fcntl` 标志，用于让专家文件绕过 page cache。
上游只在 `_WIN32` 上做了排除，没排除 Linux。

**最小补丁**：两处调用用 `#ifdef __APPLE__` 包住（Linux 上没有等价物，先置空）：

```cpp
#ifndef _WIN32
        std::lock_guard<std::mutex> lock(fd_mutex);
#ifdef __APPLE__
        for (const auto & kv : fds) { fcntl(kv.second, F_NOCACHE, on ? 1 : 0); }
#endif
```

改完 `cmake --build build -j 10` 通过，产出 `llama-cli` / `llama-mtmd-cli`。

### 坑 3：视觉投影层上 GPU 会崩，必须留在 CPU

```
ggml_backend_buffer_set_usage → ggml_abort
```

加 **`--no-mmproj-offload`** 后正常（视觉编码在 CPU、主干在 GPU）。

## 3. 实测结果

模型：`MiniCPM-o-4_5-Q4_K_M.gguf`（4.7 GB）+ `vision/MiniCPM-o-4_5-vision-F16.gguf`
参数：`-ngl 99 --no-mmproj-offload -n 220`（此时 vlm-server 仍占 6.4 GB）

| 任务 | 结果 | 耗时 |
| --- | --- | ---: |
| 合成人脸描述 | "简约风格的卡通人脸，蓝色眼睛、中性表情，浅灰背景带深灰底条" ✅ | **20 s** |
| 文档图问答（解码速度是多少） | **"55.3 tokens/s（32K上下文）"** ✅ 完全正确 | **46 s** |

> 文档图问答答对了，说明它的 VLM 能力在中文文档场景可用——这也给了我们一个对照：
> DeepSeek-OCR 单页 7.26 s（专用 OCR），MiniCPM-o 46 s（通用 9B 全模态）。
> **专用小模型做 OCR 仍然更快**，通用模型的优势在"能对话、能推理"。

## 4. 两个值得注意的观察

1. **它默认开 thinking**：输出先是一大段 `<think>...</think>`，然后才是答案。
   评估时必须把 think 剥掉，且要给足 token（`-n 220` 才拿到完整回答）。
2. **启动会打印 `ffn up/down are swapped` 警告**：这是 MiniCPM 系模型的已知特征
   （FFN 的 up/down 命名与惯例相反），不影响结果。

## 5. 没做成/没做的部分

| 项目 | 状态 |
| --- | --- |
| **音频输入 / 语音输出（全双工）** | 未测：需要 `audio/`、`tts/`、`token2wav-gguf/` 三套额外权重（已下载但未接） |
| smolmaxxing | 面向 ARM（Cortex-A55 NEON），不适用于 x86/RISC-V；但"裁剪输出词表"的思路可迁移到 K3 |
| InferSSD | 需要 18 GB+ 的 MoE 模型 + 分页补丁，本轮未做 |
| TensorFold | 未找到可用的 release 二进制 |
| lookatmyquant | 仅支持 MLX（苹果），跳过 |

## 6. 部署命令（可复现）

```bash
# 1) 权重（ModelScope，7.8 GB：Q4_K_M + vision + audio + tts + token2wav）
python -c "from modelscope import snapshot_download; snapshot_download('OpenBMB/MiniCPM-o-4_5-gguf', allow_patterns=['MiniCPM-o-4_5-Q4_K_M.gguf','vision/*','audio/*','token2wav-gguf/*'])"

# 2) 引擎（用镜像克隆，直连经常失败）
git clone --depth 1 https://ghfast.top/https://github.com/OpenBMB/llama.cpp-omni.git ~/apps/llama-omni
# 打 F_NOCACHE 补丁（见坑 2），然后
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j 10

# 3) 推理
export LD_LIBRARY_PATH=~/apps/llama-omni/build/bin
~/apps/llama-omni/build/bin/llama-mtmd-cli \
  -m ~/.cache/modelscope/models/OpenBMB--MiniCPM-o-4_5-gguf/snapshots/master/MiniCPM-o-4_5-Q4_K_M.gguf \
  --mmproj ~/.cache/modelscope/models/OpenBMB--MiniCPM-o-4_5-gguf/snapshots/master/vision/MiniCPM-o-4_5-vision-F16.gguf \
  --no-mmproj-offload --image 图片.jpg -p "描述这张图" -n 220 -ngl 99 --no-warmup
```

> 镜像加速：`ghfast.top` 前缀对 GitHub clone 有效（直连 3 次全部失败，镜像一次成功）。
