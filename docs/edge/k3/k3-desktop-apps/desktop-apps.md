# K3 桌面应用实测：Live2D 桌宠能跑，字体 9 秒首帧的说法存疑

> 【实测】2026-11-03，`pingji-k3` · Bianbu 4.0.1 · labwc 0.9.3 · Chromium 143.0.7499.169 ·
> 15 GiB · 无独立 GPU（PowerVR BXM-4-64）。
>
> 说明：板上 SDDM 停在**登录界面**，`pingji` 没有图形会话。为了测 GUI，
> 我用 labwc 的 headless 后端（`WLR_BACKENDS=headless WLR_RENDERER=pixman`）
> 起了一个**独立的合成器**，再用 `grim` 截图。这与真实桌面（Vulkan/GPU 渲染）有差异，
> 下文会标出来。

---

## 1. Live2D 桌宠（论坛 t/1621）

包：[`live2d-web-pet_0.3.18_all.deb`](https://gitee.com/achenjiayi/rv_ive2d_pet/releases/tag/v0.3.18)
（18 MB，`Architecture: all`）。

### 1.1 依赖是齐的

| deb 依赖 | 板上 |
| --- | --- |
| `python3-gi` / `python3-gi-cairo` / `python3-cairo` | ✅ |
| `python3-pil` / `python3-websockets` | ✅ |
| `gir1.2-gtk-3.0` / `gir1.2-gdkpixbuf-2.0` | ✅ |
| `libgtk-layer-shell0` | ✅ |
| `curl` | ✅ |

`dpkg -i` 即可，无需编译。

### 1.2 无 keyring 的坑

直接启动时 Chromium 弹 "Unlock Login Keyring" 对话框，页面卡在 `about:blank`：

```
[3c] present ≈0.0 fps
[cdp] attach about:blank
```

原因是 SSH 起的会话里没有解锁的 login keyring。给 Chromium 加一个参数即可：

```bash
# scripts/open_web_pet_3c.sh 的 COMMON 数组里
--password-store=basic
```

加完立刻正常（页面加载、CDP 挂上、模型出现）。

### 1.3 实测

| 指标 | 值 |
| --- | ---: |
| 渲染模式 | Wayland + `gtk-layer-shell` OVERLAY（贴角/透明/点击穿透均生效）|
| 分辨率 | 400 × 600 |
| 帧率 | **10.5 fps**（软件渲染 pixman，CPU 承担） |
| Chromium 主进程 RSS | 192 MB |
| Presenter（GTK/Python）RSS | 111 MB |
| 合计 | **≈ 300 MB** |

截图（headless labwc + grim）：角色 `haruto` 正常显示、有动作，右侧为桌宠图层。

> **注意**：10.5 fps 是**软件渲染**下的数字。真实桌面走 Vulkan/GPU 时会高不少；
> 但反过来说，这颗 GPU 对 WebGL/Vulkan 的支持也就是"能跑"，别期待丝滑。

---

## 2. 「723 个字体 → 首帧 9 秒」的验证（论坛 t/1729）

论坛结论：Bianbu 默认 **723 个字体**导致 Skia 的 fontconfig 字体族匹配极慢，
Flutter（LocalSend）首帧 **9.0 s**，打了 59 行补丁后 **1.6 s**；并称
"所有 GTK4 / libadwaita / Chromium 程序都在付这个开销"。

### 2.1 字体数量：对得上

| 项 | 数值 |
| --- | ---: |
| `fc-list \| wc -l` | **752** |
| `fc-list : family \| sort -u \| wc -l` | 390 |
| `fc-match -s sans-serif` 耗时 | **55–59 ms** |

752 vs 723 同一量级，说明**字体环境确实是同一个问题场景**。

### 2.2 Chromium：没有 9 秒

同一块板，headless Chromium 打开一页中英混排文字：

| 次数 | 首帧耗时 |
| --- | ---: |
| 冷启动（首次） | **2.21 s** |
| 第二次 | 0.70 s |
| 第三次 | 0.69 s |

**没有复现 9 秒**。如果"所有 Chromium 程序都付这个开销"成立，冷启动应该也是 9 s 量级；
实际只有 2.2 s（其中大部分还是 Chromium 自身启动）。

### 2.3 Flutter（LocalSend）：headless 下测不出来

按帖子装了 [`localsend-gui-riscv64_1.18.2`](https://github.com/fivif/localsend-riscv64)
（21 MB），补齐 5 个缺失库后能启动：

```bash
libayatana-appindicator3-1 libayatana-indicator3-7 libayatana-ido3-0.4-0
libdbusmenu-glib4 libdbusmenu-gtk4 libunwind8
```

但在 headless + pixman 下渲染不出来：

```
WARNING: ... selected Vulkan device (PowerVR B-Series BXM-4-64 MC1) doesn't
         support base Zink requirements: feats.features.fillModeNonSolid
MESA: error: zink: could not create swapchain
```

强制软件渲染（`LIBGL_ALWAYS_SOFTWARE=1` + `FLUTTER_ENGINE_SWITCH_1=enable-software-rendering`）
则直接 `Failed to create EGL context` 后 abort。90 帧截图全黑。

**结论**：Flutter 的首帧数字必须在**真实登录会话 + GPU** 下测，本次环境做不到。
所以对 t/1729 只能证实"字体数量同量级"，**不能证实也不能证伪 9.0 s**；
但 Chromium 的 2.2 s 说明这条开销**不是所有程序都付**。

### 2.4 建议 t/1729 作者补充

1. 9.0 s 是**哪个具体程序**、什么渲染后端（GPU / llvmpipe）、是否首次运行；
2. 补丁前后的同一程序对照（最好含 GTK4 原生程序，而不是只有 Flutter）；
3. 723 字体的 `fc-list` 完整列表与 `fontconfig` 版本，方便别人复现。

---

## 3. 复现命令

```bash
# 起一个独立的 headless 合成器（不影响 SDDM 的 seat）
export XDG_RUNTIME_DIR=/run/user/1000
WLR_BACKENDS=headless WLR_RENDERER=pixman WLR_LIBINPUT_NO_DEVICES=1 labwc -d &

# Live2D（解包版，不动系统）
export WAYLAND_DISPLAY=wayland-0
LIVE2D_PET_ROOT=~/live2d-app/usr/lib/live2d-web-pet \
  ~/live2d-app/usr/bin/live2d-web-pet window &

grim -o HEADLESS-1 /tmp/pet.png

# 字体与 Chromium 首帧
fc-list | wc -l
/usr/bin/chromium --headless=new --no-sandbox --disable-gpu \
  --screenshot=/tmp/f.png --window-size=900,600 \
  "data:text/html,<h1>你好 World</h1>"
```
