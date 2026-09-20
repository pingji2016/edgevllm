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
