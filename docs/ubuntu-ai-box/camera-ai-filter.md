# K3 摄像头 AI 滤镜：使用说明

> 主机：K3 板子（RISC-V，`pingji-k3`）+ GPU 机器（RTX 5060 Ti 16 GB）
> 链路：摄像头 → K3 → 千兆直连网线 → GPU 逐帧推理 → MJPEG
> 写于 2026-11-12

---

## 1. 它是什么

给 K3 的摄像头（或手机摄像头）加**逐帧 AI 处理**，输出一条新的视频流，可以：
- 实时看效果（浏览器 / ffplay）
- 各模式带实时统计（FPS、推理耗时）

```
┌─────────────┐   MJPEG    ┌──────────────┐  HTTP   ┌───────────────┐
│ 摄像头源     │──────────▶│ K3 (10.99.0.1)│────────▶│ GPU (10.99.0.2)│
│ /dev/video1 │  camserve2 │  中继/采集     │ 千兆直连 │ filter_server  │
│ /dev/video3 │            │   :8090       │ 0.7ms   │   :8091        │
└─────────────┘            └──────────────┘         └───────┬───────┘
                                                             │ MJPEG
                                    ┌────────────────────────┴────────┐
                                    ▼                                 ▼
                           浏览器/ffplay                  111 准实时网页
                          (隧道 8091)                   (每 250ms 一帧)
```

---

## 2. 组件与位置

| 组件 | 位置 | 说明 |
| --- | --- | --- |
| 摄像头服务 | K3 `~/camserve2.py` | ffmpeg 抓 `/dev/videoX` → 多客户端 MJPEG，固定端口 **8090** |
| 摄像头切换 | K3 `~/cam_select.sh usb\|phone\|status` | **同一时刻只能开一路**（USB 带宽限制） |
| 滤镜服务 | GPU `~/apps/camfilter/filter_server.py` | 拉流 → 逐帧推理 → 输出 MJPEG **8091** |
| 启动脚本 | GPU `/tmp/start_filter.sh <mode> [args]` | 杀掉旧进程并按模式启动（建议复制到 `~/apps/camfilter/`） |
| 准实时网页 | 111 `k3-live-publish.service` | 每 250ms 把最新帧写到论坛静态目录 |
| 端口转发 | 111 `k3-cam-fwd.service` | 111:18091 → GPU:8091（⚠️ 云厂商安全组未放行，仅供本机） |
| 服务单元 | GPU `/etc/systemd/system/camfilter.service` | 已安装，**未启用**（避免每次开机都占显卡） |

---

## 3. 滤镜模式（实测）

| 模式 | 作用 | 推理/帧 | 输出 FPS | 显存 |
| --- | --- | ---: | ---: | ---: |
| `detect` | YOLO11n 检测框 + 计数 | **6.7 ms** | **16.7**（源上限） | 404 MiB |
| `replace` | 人像分割 → 换背景（图/视频/纯色） | **21.1 ms** | 16.75 | 442 MiB |
| `blur` | 人像分割 → 背景虚化 | 49.5 ms | 16.65 | 442 MiB |
| `anime` | AnimeGANv2 动漫风格化 | 93.9 ms | 10.0 | 890 MiB |
| `sdturbo` | SD-Turbo 扩散模型风格化（1 步） | 见下 | 见下 | ~3.5 GB |
| `raw` | 直通（用来测链路/源） | 0.6 ms | 16.7 | 400 MiB |

> 摄像头源本身是 **16.7 fps**（720p）或 **6–12 fps**（手机 640×480，USB 带宽所限），
> 所以 16.7 就是 `detect` 的上限——GPU 只用了 6.7 ms。

---

## 4. 怎么用

### 4.1 选摄像头（在 K3 上）

```bash
~/cam_select.sh phone    # 手机 UVC（/dev/video3, 640x480@30）
~/cam_select.sh usb      # 板载摄像头（/dev/video1, 1280x720@30）
~/cam_select.sh status   # 看当前
```

⚠️ **两个摄像头不能同时开**：K3 的 USB 带宽只够一路。
同时开会报 `VIDIOC_STREAMON: No space left on device`（ENOSPC）。
所以 `cam_select.sh` 会先停掉另一路。

### 4.2 起滤镜（在 GPU 机器上）

```bash
bash /tmp/start_filter.sh detect                        # 默认：K3 板载摄像头
bash /tmp/start_filter.sh replace --background ~/apps/camfilter/bg1.png
bash /tmp/start_filter.sh replace --background 背景视频.mp4    # 动态背景
bash /tmp/start_filter.sh anime --anime-model ~/apps/camfilter/models/animegan.onnx
bash /tmp/start_filter.sh sdturbo --sd-prompt "cinematic anime style" --sd-strength 0.55
bash /tmp/start_filter.sh raw                           # 直通
```

### 4.3 看画面

| 方式 | 地址 | 说明 |
| --- | --- | --- |
| **准实时网页** | <http://111.228.40.211/bbs/img/k3_live.html> | 每 250ms 刷新一帧，公网可看 |
| 真实时（隧道） | `ssh -L 8091:127.0.0.1:18091 root@111.228.40.211` → <http://127.0.0.1:8091/filter.mjpg> | 需要自己开隧道 |
| 本机播放 | `ffplay http://127.0.0.1:8091/filter.mjpg` | 在 GPU 机器上 |
| 统计 | `curl http://127.0.0.1:8091/stats` | JSON |

> `10.99.0.2:8091` 只能从 K3 或 GPU 机器自己访问——它是**直连网线的私有地址**，
> 你的 Mac 到不了。公网用上面的准实时页面。

---

## 5. 手机摄像头怎么接

| 步骤 | 说明 |
| --- | --- |
| 1 | 手机用 USB 线插到 K3（作为 UVC 摄像头），系统会多出 `/dev/video3`("Integrated Camera") |
| 2 | `v4l2-ctl -d /dev/video3 --list-formats-ext` 确认支持的格式（我这台 OPPO 只有 **640×480@30**） |
| 3 | `~/cam_select.sh phone` |
| 4 | 手机上确认「USB 网络共享/UVC」模式已允许，**镜头不要被挡住** |

排查：
- 打开成功但读不到帧 → 另一路摄像头占着带宽，先 `cam_select.sh usb/phone` 切换
- 画面全黑（亮度均值 < 10）→ 镜头被挡 / 手机 UVC 预览没开

### 另一条路：Wi-Fi 传流（不经 USB）

手机装 **IP Webcam**（Android）/ IP Camera（iOS），K3 上跑中继：

```bash
~/phone_relay.sh "http://手机IP:8080/video"    # 中继到 10.99.0.1:8092
bash /tmp/start_filter.sh detect --src phone    # 预设指向 8092
```

好处：不占 USB、可用手机更高分辨率；代价：走 Wi-Fi，延迟略高。

---

## 6. 常见问题

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| `/dev/video1` 打不开 | 板载 `camserver.py` 占着 | 由 `cam_select.sh` 接管（它会 kill 旧进程） |
| `VIDIOC_STREAMON: No space left` | USB 带宽不足（两路摄像头） | 只开一路 |
| cv2 用路径打不开 V4L2 | 必须用设备索引 | `VideoCapture(1, CAP_V4L2)` |
| `_C is None` / no CUDA toolkit | 缺 nvcc 或 PATH 没带 | 见「3D 重建」文档，`/usr/local/bin/nvcc` 软链 |
| AnimeGAN 很慢（800ms） | 用了 onnxruntime CPU | 改 `onnx2torch` 走 CUDA（93ms） |
| `negative strides` | `img[:, :, ::-1]` | `np.ascontiguousarray()` |
| 服务进程莫名消失 | ssh 会话结束带走 | `setsid ... < /dev/null`，或用 systemd |
| 公网 18091 打不开 | 云厂商安全组只放行 80/443 等 | 用准实时页面，或自己开 SSH 隧道 |

---

## 7. 已实现 / 可扩展

**已实现**：检测框、背景虚化、背景替换（图/视频/纯色）、动漫风格化、SD-Turbo 扩散滤镜、多客户端 MJPEG、摄像头切换、准实时网页、实时统计。

**可加**：
| 能力 | 难度 | 说明 |
| --- | --- | --- |
| 多路同时跑 | 低 | 每路一个 `filter_server --port 809x`，显存还有 15 GB |
| 超分（Real-ESRGAN） | 低 | 720p→1080p，约 20–30 ms |
| 人脸美颜/贴纸 | 中 | mediapipe 关键点 |
| 录像/回放 | 低 | 加 `cv2.VideoWriter` |
| 端侧推理（放回 K3） | 中 | 检测 14 ms 已实现；风格化很难 |
| 动态背景视频 | ✅ 已支持 | `--background xx.mp4` |
