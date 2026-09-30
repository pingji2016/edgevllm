# Mage-VL 在 K3 (RISC-V) 上的实跑记录

> 板子：pingji-k3 · Bianbu 4.0.1 · 16GB · riscv64 · 2026-09-29

## TL;DR

- ✅ **图片问答已跑通**：微软 4B 多模态模型 Mage-VL（自研 Mage-ViT + Qwen3-4B 解码器）在 K3 上能出正确结果。
- ❌ **视频暂时跑不了**：卡在预处理工具 `codec-video-prep`（PyPI 上没有）。
- ❌ **NPU 帮不上**：Mage-VL 走 llama.cpp/GGUF（CPU），NPU 只跑 ONNX 小视觉模型，且不支持 Mage 的自定义算子。
- 现实定位：K3 上只适合**单图问答**；视频建议换小模型（FastVLM 0.5B）或大幅降采样。

---

## 1. Mage-VL 是什么

- 微软的 **4B-scale「codec-native」流式多模态模型**：自研 **Mage-ViT** 视觉编码器 + **Qwen3-4B** 解码器，共 ~4.7B 参数，做图像 / 视频理解。
- 卖点：**直接在压缩域工作**（H.264/H.265 的运动矢量、残差），只编码稀疏的 I/P 帧，而不是把视频逐帧解成 RGB 再全帧跑 ViT → 更省 token/算力。
- HF：`microsoft/Mage-VL`（自定义代码 `mage_vl`，用 `trust_remote_code`）。
- **为什么 llama.cpp 原生跑不了**：llama.cpp 只认自己源码里实现过的架构；上游没有 `mage_vl`。好在社区给了补丁。

---

## 2. 在 K3 上的构建（图片运行时，全部实测）

依赖：`cmake`、`pkg-config`、`gcc`（`sudo` 口令 123456）。

```bash
mkdir -p ~/mage-vl && cd ~/mage-vl

# 1) 取补丁仓库 + llama.cpp 指定 commit（GitHub 慢，用 codeload，快）
curl -sL -o src.tgz      https://codeload.github.com/JohnTDI-cpu/mage-vl-gguf/tar.gz/refs/heads/main
curl -sL -o llama.cpp.tgz https://codeload.github.com/ggml-org/llama.cpp/tar.gz/a52077c4cabb4f3c0298329c9d2dd1324d5604cb
mkdir src && tar xzf src.tgz -C src --strip-components=1
tar xzf llama.cpp.tgz && mv llama.cpp-* llama.cpp && cd llama.cpp

# 2) 打补丁（CPU 配方，来自仓库 CI）
for p in llama.cpp-mage-runtime.patch llama.cpp-mage-prefill.patch llama.cpp-mage-hardening.patch; do
  git apply "../src/patches/$p"
done

# 3) 装 cmake，编译（GGML_NATIVE=ON 会自动启用 RVV 向量）
sudo apt-get install -y cmake ninja-build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build --target llama-server llama-mtmd-cli -j8

# 4) 下 GGUF（HF 直连不通，用 hf-mirror）
cd ~/mage-vl/models
B=https://hf-mirror.com/JohnTdi/Mage-VL-GGUF/resolve/main
for f in mage-vl-backbone-Q4_K_M.gguf mage-vit-mmproj-Q8_0.gguf \
         mage-streammind-epfe-Q8_0.gguf mage-streammind-cls-Q8_0.gguf \
         mage-dcvc-rt-intra-F16.gguf mage-dcvc-rt-inter-F16.gguf; do
  wget -c -t 0 -T 30 "$B/$f"
done
```

编译日志里可见 CPU backend 标志：`-march=rv64gcv_zfh_zvfh_zicbop_zihintpause`（RVV 已启用）。

---

## 3. 图片问答实测

```bash
cd ~/mage-vl
export LD_LIBRARY_PATH=$PWD/llama.cpp/build/bin
./llama.cpp/build/bin/llama-mtmd-cli \
  -m models/mage-vl-backbone-Q4_K_M.gguf \
  --mmproj models/mage-vit-mmproj-Q8_0.gguf \
  --image models/dog_small.png \
  -p "Describe this image in one sentence." -n 128 -t 16 -c 4096
```

结果：
- `Describe this image.` → **"A dog is sitting on the floor in front of a red and blue rug."**
- `What animal is in this image?` → **"dog"**

耗时：模型加载 ~8s；**ViT 编码 1 chunk ~4.5s**；整轮几十秒。

**坑（很重要）**：
- 默认大图（960×540）会被切成 3 个 chunk，**单个 chunk 编码 224 秒**；且默认上下文会 **OOM**（dmesg 显示峰值 RSS 15.7GB，被 oom-killer 杀）。
- 正解：**图片缩到 ≤336px + `-c 4096` + `-t 16`** → 稳定。

---

## 4. 视频：能做哪、卡在哪

视频需要**另一套运行时**（图片那套不够）：

1. `llama-streammind-e2e`（用 `llama.cpp-mage-native-streammind.patch` + `0006-...DCVC-RT.patch` 编译）——**已在 K3 编译成功**。
2. `codec-video-prep`（把 mp4 转成 `MAGECV1` 片段）——**卡点**：`pip install codec-video-prep` 报 **PyPI 上不存在**，造不出 `.mcv` 输入。
3. 上游的实时路径还要 DCVC 神经编解码 + StreamMind 门控 sidecar（GGUF 已下）。

即便打通，**CPU 上也极慢**：上游在 R9700(GPU) 上 480p 才 8fps；K3 单图编码就 4.5s/chunk，视频几十个 chunk → **分钟级每片段，不可能实时**。

> 结论：**视频在 K3 上暂时不可用**；即便可用也非实时。

运行时用法（供参考）：`llama-streammind-e2e backbone.gguf mmproj.gguf epfe.gguf classifier.gguf seg1.mcv ...`

---

## 5. NPU 能不能加速？（重点回答）

**不能「一键加速」**，原因：

1. **运行时不对口**：Mage-VL 跑在 llama.cpp 的 GGUF/CPU 路径上；K3 的 NPU 是通过 **ONNX Runtime 的 SpacemiT 执行器**（`libspacemit_ep.so`，吃 INT8/INT4 ONNX）访问的，两条路不搭。
2. **没有 ONNX 版本**：要用 NPU 得把 Mage-ViT（以及可能 Qwen3-4B）**导出成 ONNX → 量化 → 上 EP**；而 Mage 的**自定义算子**（temporal patching、2×2 spatial merger、3D RoPE）和自研 ViT **大概率不被 EP 支持**。
3. **EP 的定位**：SpacemiT NPU 面向**小视觉模型**（YOLO 这类），不是 4B LLM；他们的 LLM 本来就是 CPU+IME 跑的。
4. **瓶颈不在算力**：视频的成本 ≈ **视觉 token 数 × 注意力** + **共享内存带宽**（16GB 全共享）。NPU 消除不了 token 数和带宽墙。

→ 要用 NPU 加速 Mage-VL，是**研究级移植工程**（导出/量化/算子适配/混合调度），不是配置或开关问题。

---

## 6. 真要提速，能做什么

- **降视觉 token（最有效，近似线性）**：缩小分辨率、减少采样帧、缩小 canvas。
- **常驻服务**：用 `llama-server`（省掉每次 ~8s 的模型重载）。
- **多线程**：`-t 16`。
- **换小模型**：官方 **FastVLM 0.5B**（Q4_1，**58 t/s**）做帧采样问答，比 4B 实用得多。
- **短片段离线**：视频只取几秒、低分辨率。

---

## 7. 产物位置

```
~/mage-vl/
├── models/                              # 6 个 GGUF + dog.jpg + dog_small.png + soccer.mp4 (~4GB)
├── llama.cpp/build/bin/                 # 图片：llama-server, llama-mtmd-cli
├── llama.cpp-sm/build/bin/              # 视频：llama-streammind-e2e（已编译）
├── run.sh                               # 图片一键脚本
└── logs/
```

图片一键用法：

```bash
~/mage-vl/run.sh ~/mage-vl/models/dog_small.png "描述这张图片"
```

---

## 8. 结论

- K3 上 Mage-VL：**图片能跑（慢），视频暂时跑不了，NPU 帮不上**。
- 想跑视频：先用低分辨率小块试，或直接换 **FastVLM 0.5B**；Mage-VL 4B 在 CPU 上不适合视频/实时。
- 这套折腾本身是很好的 **RISC-V 移植 + 性能调优** 素材（给 llama.cpp 打社区补丁、RVV 编译、内存/分辨率调优、依赖排查）。

---

## 附：ONNX + NPU 可行性实测（真的动手试过了）

**结论：能导出，但导出的是「形状被写死」的图；要上 NPU 还得改模型代码 + 量化 + 重写运行时——是几周的移植工程，不是改配置。**

在 K3 上实际做的步骤：

```bash
sudo apt-get install -y python3-torch python3-numpy python3-onnx   # torch 2.9.1 / numpy 2.3.5 / onnx 1.20
pip3 install --user --break-system-packages transformers safetensors huggingface_hub   # 都有 riscv64 wheel
# 用模型 config 随机初始化 Mage-ViT，直接 eager 前向 + torch.onnx.export
```

结果：
- Mage-ViT（**330M 参数**）构建成功，**eager 前向 OK**（输出 shape `64×2560`）→ 自定义 RoPE / patch merger 在 PyTorch 下能跑。
- `torch.onnx.export`（TorchScript 路径）**trace 通过**，但报一串 TracerWarning，暴露真正的坑：

```
grid_thw[idx,0].item()                       # 动态 t/h/w → 被烤成常量
lengths = (cu_seqlens[1:]-cu_seqlens[:-1]).tolist()   # 变长注意力长度 → 常量
is_causal = q_length > 1 and attention_mask is None   # 数据相关分支
```

含义（就是"算子适配"要花的功夫）：
1. **动态形状被固化**：图是按"某一个 canvas 尺寸"定死的；视频里 canvas 尺寸/数量是变化的，要么每个尺寸导出/编译一份，要么把代码改成真动态形状。
2. **打包变长注意力（cu_seqlens / varlen）**：NPU 执行器基本不支持 flash-attention 那套，得改成 **dense + attention mask**，属于**改模型源码**。
3. 后面还有：INT8 量化 + EP 算子覆盖验证 + 用 onnxruntime+EP **重写整套多模态/codec/streammind 管线**。
4. 另外还踩了 onnx/protobuf 版本冲突（`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 可绕过）。

**最关键**：视频的真正成本是 **LLM 对几千个视觉 token 的 prefill**，那部分本来就在 CPU（SpacemiT 的 LLM 不走 NPU）。所以**即便把 ViT 搬上 NPU，视频也快不了几倍**。

→ 想"跑视频"，性价比最高的不是移植 Mage-VL 到 NPU，而是 **换小模型（FastVLM 0.5B, 58 t/s）+ 帧采样**。

---

## 附二：视频路线实测进展（不换模型，已跑到门控）

目标：在 K3 上把 Mage-VL 的**视频理解**跑起来。整条链路的依赖**全部 riscv64 从源码构建成功**：

| 步骤 | 状态 |
| --- | --- |
| 补丁版 FFmpeg（带 H.264/HEVC/VP9 bitcost） | ✅ 从源码编译（仓库自带快照） |
| `cv_reader_fast` C++ 扩展 | ✅ 编译成功 |
| `codec-video-prep` / `cv-preinfer` | ✅ 安装成功、可运行 |
| 视频 → codec-aware canvas（cv-preinfer） | ✅ 对 30s clip 产出 canvas |
| `pack_mage_codec.py` → MAGECV1 (`.mcv`) | ✅ |
| `llama-streammind-e2e`（视频运行时） | ✅ 加载 `.mcv`，**Mage-ViT 成功编码全部 canvas**，进入 StreamMind 门控 |
| StreamMind 门控输出 | ❌ 卡在 ggml 类型断言 |

### 关键修复 / 踩坑

1. **扩展链接错库**：`setup.py` 的 `-L` 排在系统路径后，`-lavcodec` 被系统的 FFmpeg 8(.62) 抢走 → 调补丁 API 直接段错误。
   改成**用绝对路径直连 bundle 的 `.59` 库**（`build_ffmpeg_install/lib/libav*.so`），`NEEDED` 才指向补丁版。
2. **`codec-video-prep` 的 readiness 路径取不到 bitcost**（Python 侧键不匹配）→ 改用**另一个 CLI `cv-preinfer`**，它走通了同一套 canvas 生成。
3. **canvas 尺寸必须能被 32 整除**（packer/ViT 要求）；用
   `--patch 16 --max_pixels 60000 --target_canvas 4 --images_per_group 4` → 得到 320×160 合法尺寸。
4. **门控上下文 `n_seq_max` 不能 > 256**：30s 视频会有 112 canvas / 587 时间戳，超限；用**短片段（3s → 16 canvas）**规避。

### 当前卡点（最后一个）

门控/分类器 decode 时：
```
ggml/src/ggml-cpu/binary-ops.cpp:135: binary_op: unsupported types: dst: f32, src0: f32, src1: f16
ggml_abort → ggml_compute_forward_mul
```
即：某个 **f16 张量直接与 f32 相乘**，llama.cpp CPU 后端不支持该类型组合 → abort。
修法（任一）：给 ggml 的 binary_op 增加 f16 支持；或把门控/分类器权重转成 f32；或改图避免 f16 直接相乘。这属于社区补丁在 riscv64 上的原生类型漏洞。

### 进度量化

- 视频预处理（30s clip）：~13s。
- 视觉编码：约 50s / 4 张 canvas（CPU），112 canvas 约 23 分钟；`.mcv` 里 canvas 越多越慢。
- 结论：**链路全通到门控**；差最后一步原生类型修复就能出 silent/speak 决策。

### 产物

```
~/mage-vl/
├── cvp/                                   # codec-video-prep 源码+构建（补丁FFmpeg + cv_reader_fast）
├── llama.cpp-sm/build/bin/llama-streammind-e2e   # 视频运行时
├── llama.cpp/build/bin/llama-mtmd-cli            # 图片运行时
├── vp4/  soccer3s.mcv                     # 3s 片段：16 canvas 的 MAGECV1
└── logs/
```

一键流程：
```bash
# 1) 视频 → canvas
cd ~/mage-vl/cvp && export LD_LIBRARY_PATH=$PWD/src/codec_video_prep/libs
cv-preinfer --video clip.mp4 --out_dir ~/mage-vl/vp \
  --patch 16 --max_pixels 60000 --target_canvas 4 --images_per_group 4
# 2) 打包
cd ~/mage-vl && python3 src/tools/pack_mage_codec.py vp clip.mcv
# 3) 视频运行时
export LD_LIBRARY_PATH=$PWD/llama.cpp-sm/build/bin
./llama.cpp-sm/build/bin/llama-streammind-e2e \
  models/mage-vl-backbone-Q4_K_M.gguf models/mage-vit-mmproj-Q8_0.gguf \
  models/mage-streammind-epfe-Q8_0.gguf models/mage-streammind-cls-Q8_0.gguf clip.mcv
```


