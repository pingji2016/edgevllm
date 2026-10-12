# edgevllm

面向边缘 / 端侧部署的 vLLM 实践。

## 文档

- [docs/kernel-opt/](docs/kernel-opt/) —— vLLM 算子与矩阵运算优化笔记

  Roofline 瓶颈判断、CUDA 基本功、GEMM、Attention、量化、算子融合、边缘设备实践、性能测量、
  Triton 与 CUDA 的选型，
  共 9 篇。从 [README](docs/kernel-opt/README.md) 开始。

- [docs/leetgpu/](docs/leetgpu/) —— LeetGPU 题库精读

  [01-vector-add.md](docs/leetgpu/01-vector-add.md)：题目解读、评测框架（`challenge.py` /
  ctypes 签名 / 三类测试）、模板逐行注释、正确实现、本地编译运行，以及一次完整的带宽算账
  （含 RTX 5060 Ti 实测数据）。

- [docs/vllm/](docs/vllm/) —— vLLM 学习笔记与本机实践

  从「数学公式 → 代码路径 → 硬件约束」拆解 vLLM 的算子实现：线性层的 TP 分片与
  为什么未量化 GEMM 不是自研战场，以及 RMSNorm 从数学一路到 CUDA kernel 逐行、
  再到 IR provider 派发机制。从 [README](docs/vllm/README.md) 开始。

  另有两份配套文档：[vLLM-项目介绍.md](docs/vllm/vLLM-项目介绍.md)（整体架构、
  目录地图、请求生命周期）和 [vLLM-运行文档.md](docs/vllm/vLLM-运行文档.md)
  （在 Windows + WSL2 + RTX 5060 Ti 上的实际安装、启动与踩坑记录）。

- [docs/edge/](docs/edge/) —— 边缘设备接入笔记

  [01-k3-riscv-board.md](docs/edge/01-k3-riscv-board.md)：从一根网线到一块
  **SpacemiT K3 RISC-V 板**的完整接入过程 —— ICS 绑定、ARP/TTL 识别设备、
  端口探测为什么会自相矛盾、`Permission denied` 的两种成因怎么分开、没有
  `sshpass` 时怎么用 `SSH_ASKPASS` 自动登录。以及一个反直觉的结论：
  插网线换来的不是连通性，是延迟。

  想优化已经上手的设备，看 [kernel-opt/07-edge.md](docs/kernel-opt/07-edge.md)。

- [docs/edge/k3/](docs/edge/k3/) —— K3 实机实测集（RISC-V）

  在 **SpacemiT K3** 上真跑出来的一批记录，7 个目录 + 汇总报告：内存带宽实测与各模型
  速度对比（[k3-report/](docs/edge/k3/k3-report/)，含 `report.html`）、**K3 没有 NPU**
  （[k3-npu/](docs/edge/k3/k3-npu/)）、ASR 选型 whisper vs sherpa-onnx 流式 zipformer
  （[k3-asr/](docs/edge/k3/k3-asr/)）及其封装成 WebSocket 服务
  （[k3-asr-service/](docs/edge/k3/k3-asr-service/)）、**绕开 ffmpeg 用 MPP C API 驱动
  VPU 硬解**（[k3-codec/](docs/edge/k3/k3-codec/)）、Mage-VL 4B 在板上图片问答到视频链路
  （[mage-vl-on-k3/](docs/edge/k3/mage-vl-on-k3/)、[k3-labs/](docs/edge/k3/k3-labs/)）。
  从 [README](docs/edge/k3/README.md) 开始。

- [docs/whisperlive/](docs/whisperlive/) —— WhisperLive 源码精读（流式语音转写）

  把一个**批处理**语音模型改造成**近实时**字幕服务的完整拆解。模型一行不改，只在服务端
  加一层「流式化包装」：滚动窗口、双游标、重复输出检测 —— 为什么「最后一段永远不算数」，
  以及静音时 Whisper 的幻觉怎么把时间轴拖垮。从 [README](docs/whisperlive/README.md) 开始，
  正文是 [QUICKSTART.zh-CN.md](docs/whisperlive/QUICKSTART.zh-CN.md)（跑通、参数表、FAQ）
  和 [DESIGN.zh-CN.md](docs/whisperlive/DESIGN.zh-CN.md)（13 条特殊处理逐条拆解、
  两个后端差异、代码走读顺序）。

- [docs/realtime-media/](docs/realtime-media/) —— 实时人脸与变声（综述 + 本机实测）

  把「手机摄像头前的人脸滤镜/换脸」和「实时变声」两件事拆开：
  [01](docs/realtime-media/01-face-filters.md) 人脸滤镜（478 点关键点、局部变形，
  实测检测 10.3 ms vs 全图变形 97.6 ms、ROI 后 7.4 ms）、
  [02](docs/realtime-media/02-face-swap.md) 换脸（五点位对齐 → ArcFace 身份向量 →
  emap/AdaIN → 贴回，一条 CPU 上跑通的流水线，逐阶段耗时）、
  [03](docs/realtime-media/03-voice-pitch-and-formant.md) 音高与共振峰（`formant=preserved`
  到底保住了什么，含合成元音真值校验）、
  [04](docs/realtime-media/04-voice-neural.md) 神经变声（RVC 一族、延迟与设备链路）、
  [05](docs/realtime-media/05-realtime-engineering.md) 实时工程（延迟预算、分块伪影、
  无锁缓冲）。所有实验器材都是**程序化合成**的人脸与 TTS 语音，不含真人素材；
  脚本与复现命令见 [scripts/](docs/realtime-media/scripts/)。

## 示例

- [examples/](examples/) —— 可直接编译运行的代码

  [examples/build.sh](examples/build.sh)（Windows 下用 [examples/build.cmd](examples/build.cmd)）
  一条命令编译并运行全部示例：

  ```bash
  cd examples && ./build.sh run
  ```

  ```bat
  cd examples && build.cmd run
  ```

  [examples/leetgpu/01_vector_add/](examples/leetgpu/01_vector_add/)：自带正确性校验与
  带宽实测的 vector add，以及打印设备属性的 `device_query`。编译/运行细节和
  Windows 编码坑见 [examples/README.md](examples/README.md)。

## 第三方依赖

以下内容**本地拉取，不纳入版本控制**。

### leetgpu-challenges

[AlphaGPU/leetgpu-challenges](https://github.com/AlphaGPU/leetgpu-challenges) 是
[LeetGPU](https://leetgpu.com) 的 GPU 编程题库。每道题包含问题描述、参考实现、测试用例，
以及 CUDA / Triton 等多种框架的起始模板，用来练手 kernel 编写。

```bash
git clone --depth 1 https://github.com/AlphaGPU/leetgpu-challenges.git \
    third_party/leetgpu-challenges
```

**刻意不进版本控制，有两个原因：**

1. **许可证不允许。** 上游采用 [CC BY-NC-ND 4.0](https://github.com/AlphaGPU/leetgpu-challenges/blob/main/LICENSE)，
   明确禁止商业使用、再分发和衍生作品。把它 vendor 进本仓库属于再分发，违反许可条款。
2. **它有自己的历史。** 上游更新频繁，本地拷贝没有长期价值，需要时重新拉取即可。

`third_party/` 整个目录已在 [.gitignore](.gitignore) 中，`git status` 不会看到它。

#### 为什么不用 git submodule

submodule 同样满足「内容不提交」，但它会把 `.gitmodules` 和一条指向特定 commit 的 gitlink
**提交进本仓库**。对本项目来说这没有价值——我们并不依赖它的某个固定版本——却会让每个
clone 本仓库的人都得额外处理 submodule 的初始化与拉取。

如果将来确实需要在 CI 里锁定某个版本，再切换成 submodule：

```bash
rm -rf third_party/leetgpu-challenges
git submodule add https://github.com/AlphaGPU/leetgpu-challenges.git \
    third_party/leetgpu-challenges
```

届时记得同步更新上面的 `.gitignore`（submodule 不能被忽略，否则不会生效）。
