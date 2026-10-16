# 在 riscv64 Android 上跑 arm64 应用（NativeBridge / 二进制翻译）

**结论先说：`arm64 → riscv64` 这个方向，上游是空的。** 而它恰好决定了「K3 上能不能用国产 App」。

本文讲清机制、摆出三条路线、列出要动哪些系统组件，并给一条**第一步已经实测通过**的落地路径。

> 相关：[`k3-hardware.md`](../k3-hardware/k3-hardware.md)（板子能力 / 短板）、[`k3-android-cuttlefish.md`](../k3-android/k3-android.md)（Android on K3）

---

## 0. 结论速览

| 问题 | 答案 |
| --- | --- |
| 为什么不装 arm64 App？ | guest 是 `riscv64` ABI，App 里只有 `lib/arm64-v8a/*.so` → `INSTALL_FAILED_NO_MATCHING_ABIS` |
| 换个镜像行不行？ | ❌ 宿主 KVM 只能跑同架构 guest，riscv64 宿主跑 arm64 镜像**物理上不存在** |
| 那要做什么？ | 加一层 **arm64 → riscv64 的用户态二进制翻译**（Android 官方叫 NativeBridge） |
| 上游有人做吗？ | **没有。** 有开源的转译框架（Berberis），但官方只做 **riscv64 → x86_64**（反方向） |
| 需要什么「权限」？ | **不需要任何许可** —— AOSP 开源随便改。**但需要能自己编 AOSP 系统镜像** |
| 有没有零权限就能开始的？ | ✅ 有。转译器内核是纯用户态；**而且第一步已在 K3 上实测通过**（见 §6.1） |
| 现实难度 | 极高但个人可行（有人单独补出过 arm64→x86_64 后端）。**量级是「年」，不是「月」** |

---

## 1. 先分清方向：Rosetta 2 是「另一头」

很多人第一反应是「这不就是苹果的 Rosetta 2 吗」。**是同一技术家族，但方向相反：**

| | Rosetta 2（苹果） | 我们要的 |
| --- | --- | --- |
| 方向 | **x86_64 → arm64** | **arm64 → riscv64** |
| 形态 | 闭源，只有 macOS 有 | 上游空白 |
| 同类物 | Android x86 的 `libhoudini`（Intel）/ `libndk_translation` | 把「宿主」换成 riscv64 |

**动态二进制翻译（DBT）** 这个词才是共同点：在运行时把一种 ISA 的机器码翻成另一种。

---

## 2. Android 的官方机制：NativeBridge

**关键认知：这不是「改操作系统」，而是 Android 官方提供的一个「转译器插件」机制。**

- AOSP **定义了接口**（`libnativebridge` 是 host 侧胶水），**但不打包转译器**；
- 转译器是一个**独立 `.so`**，运行时加载；
- **每个 app 进程可以各自加载或各自不加载**（Zygote fork 之后按需）；
- 对 App **完全透明** —— `System.loadLibrary()` 和 JNI 调用照常工作。

**关键属性（决定你要改什么）：**

| 属性 | 作用 |
| --- | --- |
| `ro.dalvik.vm.native.bridge=<你的库>` | 选择用哪个转译器 |
| `ro.dalvik.vm.isa.<guest>=<host>` | 告诉 PackageManager：**哪个宿主 ISA 能翻译 guest**（如 `ro.dalvik.vm.isa.riscv64=x86_64`） |
| `ro.enable.native.bridge.exec=1` | 允许跑非 Android 的独立可执行文件 |

### 一个让问题变小的关键事实

**Android App 的代码大部分是 ART 字节码（`dex`），架构无关。** 只有 JNI 的 **native `.so`** 才是架构相关的。

所以问题不是「把整个 App 翻译一遍」，而是：
> **让 riscv64 的 ART 能加载并运行 arm64 的 `.so`。**

这正是 x86 设备上 `libndk_translation` 干的事。

---

## 3. 上游现状：有开源框架，但方向不对

| 项目 | 位置 | 方向 | 开源？ |
| --- | --- | --- | --- |
| **Berberis** | `platform/frameworks/libs/binary_translation`（AOSP） | **riscv64 → x86_64** | ✅ 开源 |
| `libndk_translation.so` | Google 的 x86_64 系统镜像内置（Android 14–17） | **arm64 → x86_64** | ❌ **后端未公开**（符号表显示它就是 Berberis + ARM64 后端） |
| `libhoudini` | Intel | arm / arm64 → x86 / x86_64 | ❌ 闭源 |
| **Digitalis** | 第三方，基于公开的 Berberis | **arm64 → x86_64** | ✅ 有人自己补的 |
| **nb-qemu** | `android_translation_layer`（ATL）项目 | arm / aarch64，**用 QEMU 当后端** | ✅ 开源 |
| ~~arm64 → riscv64~~ | — | **—— 空白 ——** | — |

**读出来的三件事：**

1. **框架是现成的、开源的** —— 你不用从零发明 NativeBridge；
2. **「补一个后端」这件事有先例** —— Digitalis 就是一个人基于公开框架补出 arm64→x86_64，还写了 102 页讲解；
3. **`arm64 → riscv64` 是真空** —— 这是你能填的位置。

### Berberis 的结构（你要加什么，全在这张图里）

```
frameworks/libs/binary_translation/
  decoder/          # guest ISA → IR      ← 需要新增 arm64/
  interpreter/      # 解释器（最慢那档）
  lite_translator/  # 单遍 JIT（第一档）
  heavy_optimizer/  # 优化 JIT（第二档，热区才上）
  translator/       # 三档派发
  backend/          # IR → 宿主机器码    ← 需要新增 riscv64/
  code_gen_lib/     # 分方向助手：riscv64_to_x86_64/、arm64_to_x86_64/ ← 需要新增 arm64_to_riscv64/
  intrinsics/       # 指令优化实现
  insn_tests/       # 逐 ISA 一致性测试
```

**Berberis 的架构就是为「加新方向」设计的** —— 三档 JIT + 分离的 decoder / backend，天然欢迎新组合。

---

## 4. 要动哪些系统组件（都是 AOSP 源码，公开可改）

| 类别 | 组件 | 说明 |
| --- | --- | --- |
| host 侧 | `libnativebridge` | host 胶水（通常直接用框架） |
| host 侧 | **proxy 库（约 21 个）** | `libc`/`libm`/`libEGL`/`libGLESv1_CM`/`libGLESv2`/`libGLESv3`/`libvulkan`/`libmediandk`/`libneuralnetworks`/`libbinder_ndk`/`libaaudio`/`libandroid`/`libandroid_runtime`/`libcamera2ndk`/`libjnigraphics`/`libnativehelper`/`libnativewindow`/`libOpenMAXAL`/`libOpenSLES`/`libamidi`/`libwebviewchromium_plat_support` |
| guest 侧 | **native_bridge_support** | 交叉编译成 **arm64** 的：`linker64`、`libc`、guest VDSO、`app_process64` … |
| guest 侧 | **modified guest libs** | 上面那 21 个库都要 guest 版本（装在 `/system/lib64/arm64/`），内部改过以经 VDSO 桥回 host |
| 配置 | product `.mk` | 设置 `ro.dalvik.vm.isa.arm64=riscv64`、`ro.dalvik.vm.native.bridge=…` |
| 可选 | `binfmt_misc` handler | 让非 Android 的 arm64 可执行文件也能跑 |
| 测试 | `insn_tests/` | 逐指令一致性测试（Berberis 自带这个机制，务必用） |

> ⚠️ 已知细节：Berberis 里 **`libRS.so` 被显式排除**（不从 host 回退），移植时注意。

---

## 5. 三条路线对比

| 路线 | 工作量 | 性能 | 说明 |
| --- | --- | --- | --- |
| **A. qemu-user（Linux 层）** | 极低 | 差 | **不需要改 Android**。用来做可行性 / 性能验证（§6.1 已实测）。**跑不了 Android App**，因为缺 Android 运行时 |
| **B. nb-qemu（把 QEMU 当转译后端）** | **中** | 中 | QEMU 已支持 aarch64 guest + riscv64 host。**最快能跑通 arm64 App 的路径** |
| **C. Berberis 加后端（正统）** | **极高** | **好** | 补 `decoder/arm64` + `backend/riscv64`。能提上游，长期正解 |

**建议顺序：A（验证）→ B（跑通）→ C（长期）。**

---

## 6. 最小验证路径（第 1 步已实测）

### 6.0 前置：一台编译机

**这是当前最大的缺口**：AOSP 构建宿主只支持 **Linux x86_64**，官方数字 **~150GB 同步 / 编完 ~320GB / 32GB+ 内存**。
（K3 当不了编译机；Apple Silicon Mac 也不行。）

### 6.1 ✅ 已在 K3 上实测通过：跑 arm64 Linux 二进制

**2026-10-15 在 `pingji-k3` 上实测：**

```bash
# 1) 装 qemu-user（Bianbu 源里就有；qemu-user-static 没有，qemu-user 有）
sudo apt-get install -y qemu-user        # 版本 1:10.2.1+ds-1ubuntu3（QEMU 10.2.1）
```

装完 **`qemu-user-binfmt` 会自动注册 `binfmt_misc`** —— 不用加前缀，直接执行 arm64 ELF：

```
/proc/sys/fs/binfmt_misc/ 里已注册约 30 个 guest：
  qemu-aarch64  qemu-arm  qemu-i386  qemu-loongarch64  qemu-ppc64le  …
```

```bash
# 2) 拿一个真 aarch64 静态二进制（改自 Aliyun ubuntu-ports，国内可达）
curl -o bbs.deb http://mirrors.aliyun.com/ubuntu-ports/pool/main/b/busybox/busybox-static_1.37.0-7ubuntu1_arm64.deb
dpkg-deb -x bbs.deb bb64          # → bb64/usr/bin/busybox
file bb64/usr/bin/busybox
#   ELF 64-bit LSB executable, ARM aarch64, statically linked
```

```bash
# 3) 直接运行（binfmt 自动走 qemu-aarch64）
cd bb64/usr/bin
ln -sf busybox uname && ./uname -m      # → aarch64   ✅
ln -sf busybox sh    && ./sh -c 'echo ok-arm64'   # → ok-arm64   ✅
```

**⚠️ 一个必须知道的坑**：busybox **必须用符号链接调用**（`ln -s busybox uname`）。
直接 `./busybox uname` 会报 `applet not found` —— 因为 busybox 靠 `readlinkat("/proc/self/exe")` 推断自己的名字，
在 binfmt 下 `/proc/self/exe` 指向解释器，推断失败。**这是 busybox 的特性，不是翻译坏了。**

**粗略性能开销（同机对比，300k 次 shell 循环）：**

| 运行方式 | 耗时 |
| --- | --- |
| 原生 riscv64（`bash`） | **3.24 s** |
| 转译 arm64（busybox `ash`） | **20.3 s** |

≈ **6 倍** 开销。

> ⚠️ **这不是严格基准**：两边是不同的 shell 实现（bash vs busybox ash），只能给个数量级。
> 要做准，得把**同一个程序**分别交叉编译成 riscv64 与 arm64 再比。

**这一步证明了什么、没证明什么：**

| ✅ 证明了 | ❌ 没证明 |
| --- | --- |
| K3 上**能**执行 arm64 机器码（翻译层可用） | **跑不了 Android App** —— qemu-user 没有 Android 运行时（ART / bionic / binder） |
| binfmt_misc 让执行变透明（无额外前缀） | 性能是否够用（6× 是 shell 循环，NEON 密集代码会更差） |
| 转译开销的量级 | — |

### 6.2 第二步：走 nb-qemu

`android_translation_layer` 的 **QEMU 版 native bridge** 已支持 arm / aarch64 目标。
它比从零写 JIT 少一个数量级的活，只要把 host 侧扩到 riscv64。

**这是「最快能跑通 arm64 App」的路径。**

### 6.3 长期：给 Berberis 补后端

补 `decoder/arm64/` + `backend/riscv64/` + `code_gen_lib/arm64_to_riscv64/`，
交叉编译 guest 库，出 AOSP 镜像，**提上游**。
这条路性能最好，也最「正统」——而且 Google 的 `android-riscv64` 明确说「所有工作都在 AOSP 里做」。

---

## 7. 为什么 arm64 → riscv64 可能比 arm64 → x86_64 **更容易**

这是个反直觉但站得住的判断：

| 维度 | ARM64 → x86_64（Digitalis 做的） | ARM64 → riscv64 |
| --- | --- | --- |
| 通用寄存器数 | 31 → **16（严重不够）** | 31 → **31（近乎 1:1）** |
| 指令风格 | RISC → CISC（变长编码、内存操作数） | **RISC → RISC** |
| 标志位 | 必须模拟 x86 的 FLAGS | RISC-V **没有标志寄存器** → 要模拟 ARM 的 **NZCV** |
| 内存序 | ARM 弱 → x86 **强（需插屏障）** | ARM 弱 → RISC-V **也是弱序（较兼容）** |

**结论：寄存器映射和指令风格这两块，arm64→riscv64 明显更轻松；难点集中在 NZCV 标志位模拟和向量。**
换句话说，**Digitalis 能把 arm64→x86_64 做出来，arm64→riscv64 在工程上只会更友好，不会更糟。**

**但要老实说的难点：**

- **NEON → RVV**：NEON 是固定 128-bit，RVV 是变长（A100 上 1024-bit），语义映射繁琐；
- **原子指令（LSE）** → RISC-V `A` 扩展；
- **加密扩展**两边完全不同；
- **proxy 库那 21 个接口**的 marshalling 是纯体力活，出错极难查。

---

## 8. 现实评估与风险

| 风险 | 说明 |
| --- | --- |
| **工作量** | 年量级。参考：Digitalis 一个人做出了 arm64→x86_64，但那是全职投入 + 102 页文档 |
| **跟版本** | AOSP 每个大版本都要跟，维护成本持续存在 |
| **上游可能重复实现** | Google 的 arm64 后端**已经有但没开源**（在 x86 镜像里）。你做的可能被官方覆盖 |
| **风控是另一层问题** | ⚠️ **转译只解决 ABI，不解决风控。** 微信/QQ 有模拟器/root 检测，装上≠能用 |
| **值不值得** | riscv64 Android 设备目前极少。做出来是「技术价值高、短期回报低」 |
| **前置缺口** | 没有编译机就一步都走不动 |

**一句话：这是个「上游空白、技术可行、个人可做、但短期回报不确定」的方向。**
适合当**长期兴趣 + 可展示的作品**去做，不适合当「三个月要出成果」的项目。

---

## 9. 参考

- **Berberis（AOSP 官方转译框架）**：`frameworks/libs/binary_translation`（riscv64 → x86_64）
- **native_bridge_support**：`frameworks/libs/native_bridge_support`（guest 侧库清单）
- **NativeBridge 机制说明**：AOSP Internals 第 19 章「Native Bridge」
- **Digitalis**：社区基于公开 Berberis 补出的 **arm64 → x86_64** 后端（含 102 页讲解）
- **nb-qemu**：`android_translation_layer/qemu-based-native-bridge`（QEMU 当后端）
- **Google 官方 RISC-V 进展**：`google/android-riscv64`（「所有工作直接在 AOSP 里做」）
- **RISC-V Summit 2024**：Google「Berberis: Dynamic Binary Translation from RISC-V to x86_64 on Android」
- 板子能力与缺口：[`k3-hardware.md`](../k3-hardware/k3-hardware.md)
